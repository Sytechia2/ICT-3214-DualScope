"""Alert package for any scored split: the four files the dashboard and investigations read.

Same files and column layout as ``scripts/export_alert_handoff.py``:
``alerts.parquet``, ``events.parquet``, ``incidents.jsonl`` and ``manifest.json``.
Labels are optional. Without ``labels_dir`` every ``ground_truth_redteam`` is
False and the manifest says there is no answer key. Test labels are refused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from dualscope.handoff.alerts import (
    COUNT_COLUMNS,
    FUSION_METHOD,
    GRAPH_COLUMNS,
    PRIORITY_ABOVE_CUTOFF,
    PRIORITY_TIED_AT_CUTOFF,
    alert_events,
    build_incident_records,
    group_incidents,
    label_alerts,
    select_daily_alerts,
)
from dualscope.sequence.calibration import load_positive_user_hours
from dualscope.splits import SplitConfig

ALERT_COLUMNS = [
    "alert_id", "incident_id", "user", "day", "window_start", "window_end", "rank_in_day",
    "fusion", "day_cutoff_score", "tied_at_cutoff", "gru_max_event", "gru_percentile_in_day",
    *COUNT_COLUMNS, "is_machine_account", "ground_truth_redteam",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_alert_package(
    scores: pd.DataFrame,
    features_root: str | Path,
    output_dir: str | Path,
    *,
    budget: int,
    split_name: str,
    id_prefix: str,
    labels_dir: str | Path | None = None,
    splits_config: str | Path | None = None,
    graph_scores: str | Path | None = None,
    max_gap_seconds: int = 7_200,
    extra_manifest: dict[str, Any] | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Pick the top ``budget`` user-hours per day, attach their events, group incidents and write the package.

    ``scores`` is the output of ``score_days`` (its row order breaks ties).
    ``features_root`` supplies ``raw/events``. ``labels_dir`` and
    ``splits_config`` (both needed together) mark ``ground_truth_redteam`` for
    ``split_name`` (validation or train; "test" raises ``PermissionError``).
    ``graph_scores`` is an optional graph-detector score dataset used as context.
    ``extra_manifest`` is merged into the manifest. Returns the manifest.
    Raises if the events found for an alert differ from its scored ``n_events``.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass overwrite=True to replace it")
    if split_name == "test":
        raise PermissionError("test labels are reserved; build test packages with scripts/export_alert_handoff.py")
    if labels_dir is not None and splits_config is None:
        raise ValueError("splits_config is needed with labels_dir")
    alerts = select_daily_alerts(scores, budget)
    if alerts.empty:
        raise ValueError("no scored user-hours to build alerts from")
    if labels_dir is not None:
        positives = load_positive_user_hours(labels_dir, SplitConfig.from_file(splits_config), split_name)
        alerts["ground_truth_redteam"] = label_alerts(alerts, positives)
    else:
        positives = set()
        alerts["ground_truth_redteam"] = False
    alerts["incident_id"] = group_incidents(alerts, max_gap_seconds, id_prefix)

    dataset = ds.dataset(str(Path(features_root) / "raw" / "events"), format="parquet", partitioning="hive")
    events = alert_events(dataset, alerts, log)
    per_alert = events.groupby("alert_id").size().reindex(alerts["alert_id"], fill_value=0).to_numpy()
    if not np.array_equal(per_alert, alerts["n_events"].to_numpy()):
        raise ValueError("event counts per alert differ from the scored n_events")

    graph = None
    if graph_scores is not None:
        graph = ds.dataset(str(graph_scores), format="parquet", partitioning="hive").to_table(
            columns=GRAPH_COLUMNS,
            filter=ds.field("dataset_day").isin(sorted({int(d) for d in alerts["day"]}))
            & ds.field("user_id").isin(sorted(set(alerts["user"]))),
        ).to_pandas()
    incidents = build_incident_records(alerts, events, graph, split=split_name)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    alerts[ALERT_COLUMNS].rename(columns={"user": "user_id", "day": "dataset_day", "fusion": "fusion_score"}).to_parquet(
        output_dir / "alerts.parquet", index=False
    )
    events.to_parquet(output_dir / "events.parquet", index=False)
    with (output_dir / "incidents.jsonl").open("w", encoding="utf-8") as handle:
        for incident in incidents:
            handle.write(json.dumps(incident, sort_keys=True) + "\n")

    has_labels = labels_dir is not None
    manifest: dict[str, Any] = {
        "description": f"Alert queue for the {split_name} split ({budget} per day)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "dualscope.pipeline.alerts.build_alert_package",
        "split": split_name,
        "inputs": {
            "features_root": str(features_root),
            "graph_scores": str(graph_scores) if graph_scores is not None else None,
            "labels_dir": str(labels_dir) if has_labels else None,
        },
        "fusion_method": FUSION_METHOD,
        "budget_per_day": budget,
        "tie_break": "fixed random order, seed 0",
        "incident_max_gap_seconds": max_gap_seconds,
        "answer_key": has_labels,
        "counts": {
            "alerts": len(alerts),
            "incidents": len(incidents),
            "events": len(events),
            "users": int(alerts["user"].nunique()),
            "redteam_alerts": int(alerts["ground_truth_redteam"].sum()) if has_labels else None,
            "redteam_incidents": sum(i["ground_truth_redteam"] for i in incidents) if has_labels else None,
            "alerts_tied_at_cutoff": int(alerts["tied_at_cutoff"].sum()),
        },
        "notes": {
            "times": "Dataset-relative seconds; windows are half-open [start, end).",
            "priority": f"{PRIORITY_ABOVE_CUTOFF}: an hour scored above its day's cut-off. "
                        f"{PRIORITY_TIED_AT_CUTOFF}: every hour tied at the cut-off, picked by the fixed tie-break.",
            "rank_in_day": "Order within tied scores comes from the fixed tie-break and carries no meaning.",
            "fusion_score": "Gradient-boosting output; ranks user-hours, not a calibrated attack probability.",
            "detector_scores.sequence.max_score": "GRU score as a within-day percentile; the raw value is gru_max_event.",
            "graph_context": "Graph detector view of the same user-day; context only, not used by the final model."
                             if graph is not None else "No graph scores were supplied.",
            "ground_truth_redteam": f"Red-team answer key for the {split_name} split; evaluation only. Hide from analysts by default; never send to an LLM."
                                    if has_labels else "No answer key: every value is False and means unknown, not benign.",
        },
    }
    manifest.update(extra_manifest or {})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ("alerts.parquet", "events.parquet", "incidents.jsonl"):
        manifest.setdefault("files", {})[name] = _sha256(output_dir / name)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote {output_dir}: {manifest['counts']}")
    return manifest
