#!/usr/bin/env python3
"""Export the final model's days 17-30 alert queue for Tasks 6.x and 7.x.

Rebuilds the 532 alerts the one-time final test chose (38 per day) from the
stored test scores, attaches every contributing auth.txt event, groups alerts
into per-user incidents and adds the graph detector's same-day view as context.
Nothing is re-scored; the test is not rerun. It refuses to write unless the
rebuilt queue reproduces the test record's true-positive count.

Outputs (in --output-dir):
  alerts.parquet     one row per alert (user-hour)
  events.parquet     one row per contributing authentication event
  incidents.jsonl    one record per incident (dashboard and evidence-package input)
  manifest.json      inputs, hashes, counts and field notes

``ground_truth_redteam`` is the answer key for evaluation and demos. It must not
be shown to analysts by default or sent to an LLM.

Example:
  python scripts/export_alert_handoff.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.handoff.alerts import (  # noqa: E402
    COUNT_COLUMNS,
    EVENT_COLUMNS,  # noqa: F401
    GRAPH_COLUMNS,
    alert_events,
    FUSION_METHOD,
    HOUR_SECONDS,
    PRIORITY_ABOVE_CUTOFF,
    PRIORITY_TIED_AT_CUTOFF,
    build_incident_records,
    group_incidents,
    label_alerts,
    select_daily_alerts,
)
from dualscope.sequence.calibration import load_positive_user_hours  # noqa: E402
from dualscope.splits import SplitConfig  # noqa: E402

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=REPO_ROOT / "outputs/final_test/final_test_scores.parquet")
    parser.add_argument("--test-record", type=Path, default=REPO_ROOT / "outputs/final_test/final_test.json")
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "data/processed/lanl_features_v2_days_01_30")
    parser.add_argument("--graph-scores", type=Path, default=REPO_ROOT / "outputs/graph_scores_v1/scores")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--max-gap-seconds", type=int, default=7_200)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/handoff/final_test_alerts_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_dir} exists; pass --overwrite to replace it")

    record = json.loads(args.test_record.read_text(encoding="utf-8"))
    if record["budget_per_day"] != args.budget:
        raise SystemExit(f"test record budget is {record['budget_per_day']}, not {args.budget}")
    alerts = select_daily_alerts(pd.read_parquet(args.scores), args.budget)
    expected_alerts = record["scorers"]["fusion_frozen"]["user_hour"]["at_budget"]["alerts"]
    if len(alerts) != expected_alerts:
        raise SystemExit(f"rebuilt {len(alerts)} alerts, test record has {expected_alerts}")

    # The test is spent; labels only mark which queued hours were red-team activity.
    positives = load_positive_user_hours(
        args.labels_dir, SplitConfig.from_file(args.splits_config), "test", allow_test=True
    )
    alerts["ground_truth_redteam"] = label_alerts(alerts, positives)
    expected_tp = record["scorers"]["fusion_frozen"]["user_hour"]["at_budget"]["tp"]
    if int(alerts["ground_truth_redteam"].sum()) != expected_tp:
        raise SystemExit(f"rebuilt queue has {int(alerts['ground_truth_redteam'].sum())} red-team hours, test record has {expected_tp}")
    alerts["incident_id"] = group_incidents(alerts, args.max_gap_seconds)
    print(f"Rebuilt {len(alerts)} alerts ({expected_tp} red-team) in {alerts['incident_id'].nunique()} incidents")

    events = alert_events(ds.dataset(str(args.features_root / "raw" / "events"), format="parquet", partitioning="hive"), alerts)
    per_alert = events.groupby("alert_id").size().reindex(alerts["alert_id"], fill_value=0).to_numpy()
    if not np.array_equal(per_alert, alerts["n_events"].to_numpy()):
        raise SystemExit("event counts per alert differ from the scored n_events")

    graph = ds.dataset(str(args.graph_scores), format="parquet", partitioning="hive").to_table(
        columns=GRAPH_COLUMNS,
        filter=ds.field("dataset_day").isin(sorted({int(d) for d in alerts["day"]}))
        & ds.field("user_id").isin(sorted(set(alerts["user"]))),
    ).to_pandas()
    incidents = build_incident_records(alerts, events, graph)

    alert_columns = [
        "alert_id", "incident_id", "user", "day", "window_start", "window_end", "rank_in_day",
        "fusion", "day_cutoff_score", "tied_at_cutoff", "gru_max_event", "gru_percentile_in_day",
        *COUNT_COLUMNS, "is_machine_account", "ground_truth_redteam",
    ]
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    alerts[alert_columns].rename(columns={"user": "user_id", "day": "dataset_day", "fusion": "fusion_score"}).to_parquet(
        args.output_dir / "alerts.parquet", index=False
    )
    events.to_parquet(args.output_dir / "events.parquet", index=False)
    with (args.output_dir / "incidents.jsonl").open("w", encoding="utf-8") as handle:
        for incident in incidents:
            handle.write(json.dumps(incident, sort_keys=True) + "\n")

    manifest = {
        "description": "Final model alert queue for days 17-30 (38 per day), for Tasks 6.x and 7.x",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/export_alert_handoff.py",
        "inputs": {
            "scores": {"path": str(args.scores), "sha256": sha256(args.scores)},
            "test_record": {"path": str(args.test_record), "sha256": sha256(args.test_record)},
            "frozen_model_sha256": record["frozen_model"]["sha256"],
            "features_root": str(args.features_root),
            "graph_scores": str(args.graph_scores),
        },
        "fusion_method": FUSION_METHOD,
        "budget_per_day": args.budget,
        "tie_break": record["tie_break"],
        "incident_max_gap_seconds": args.max_gap_seconds,
        "counts": {
            "alerts": len(alerts),
            "incidents": len(incidents),
            "events": len(events),
            "users": int(alerts["user"].nunique()),
            "redteam_alerts": int(alerts["ground_truth_redteam"].sum()),
            "redteam_incidents": sum(incident["ground_truth_redteam"] for incident in incidents),
            "alerts_tied_at_cutoff": int(alerts["tied_at_cutoff"].sum()),
        },
        "notes": {
            "times": "Dataset-relative seconds; windows are half-open [start, end).",
            "priority": f"{PRIORITY_ABOVE_CUTOFF}: an hour scored above its day's cut-off. "
                        f"{PRIORITY_TIED_AT_CUTOFF}: every hour tied at the cut-off, picked by the fixed tie-break.",
            "rank_in_day": "Order within tied scores comes from the fixed tie-break and carries no meaning.",
            "fusion_score": "Gradient-boosting output; ranks user-hours, not a calibrated attack probability.",
            "detector_scores.sequence.max_score": "GRU score as a within-day percentile; the raw value is gru_max_event.",
            "graph_context": "Graph detector view of the same user-day. Not used by the final model; the graph team saw days 17-30 during development.",
            "ground_truth_redteam": "Answer key for evaluation and demos only. Hide from analysts by default; never send to an LLM.",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
