"""Task 3.4: export frozen sequence-detector scores and evidence per user-hour.

Loads a frozen detector (refusing to run if its checkpoint, record, feature
configuration or split policy changed), scores every user-hour of the
requested splits and writes day-partitioned Parquet in the sequence detector
output schema, plus ``summary.json``. Rows cover available, warm-up
(``insufficient_history``) and, for units named in ``--requested-units``,
``no_activity`` user-hours. A repeat-inference check re-scores one day and
compares the results within a stated tolerance. Red-team labels are not read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.sequence.detector import FrozenSequenceDetector  # noqa: E402
from dualscope.sequence.export import (  # noqa: E402
    REFERENCE_FORMAT,
    SCHEMA_VERSION,
    SEQUENCE_SCORE_SCHEMA,
    build_score_table,
    no_activity_rows,
    status_counts,
    to_detector_record,
    write_day_table,
)
from dualscope.sequence.pipeline import git_revision, load_inputs  # noqa: E402
from dualscope.splits import ScoringStatus, dataset_day  # noqa: E402

REPEAT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "data/processed/lanl_features_days_01_30")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features.json")
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Default: outputs/sequence_scores/<model_version>.")
    parser.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("--days", type=int, nargs="*", default=None)
    parser.add_argument("--requested-units", type=Path, default=None, help="CSV/Parquet with user_id, window_start.")
    parser.add_argument("--sample-output", type=Path, default=None, help="Write a small handoff sample here.")
    parser.add_argument("--repeat-check-day", type=int, default=None, help="Day to re-score (default: first exported validation day).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-pilot-features", action="store_true")
    return parser.parse_args()


def _read_units(path: Path) -> list[tuple[str, int]]:
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["user_id", "window_start"])
    else:
        from pyarrow import csv

        table = csv.read_csv(path).select(["user_id", "window_start"])
    return list(zip(table["user_id"].to_pylist(), (int(v) for v in table["window_start"].to_pylist())))


def _build(detector: FrozenSequenceDetector, day, hour_seconds: int) -> pa.Table:
    scores = detector.score_day(day)
    raw = scores.raw[detector.aggregation]
    return build_score_table(
        scores,
        detector.normalise(raw),
        raw,
        model_version=detector.model_version,
        run_id=detector.run_id,
        aggregation=detector.aggregation,
        alert_threshold=detector.alert_threshold,
        hour_seconds=hour_seconds,
        top_events=detector.config.top_events,
        top_features=detector.config.top_features,
    )


def _handoff_sample(tables: dict[int, pa.Table], no_activity: pa.Table | None, output: Path) -> dict:
    """Pick ordinary, high-error, alerting and insufficient rows for Members 4-6."""
    picked: list[dict] = []

    def take(table: pa.Table, mask, order_key: str | None, count: int, descending: bool, label: str):
        subset = table.filter(mask)
        if subset.num_rows == 0:
            return
        if order_key is not None:
            subset = subset.sort_by([(order_key, "descending" if descending else "ascending")])
        for row in subset.slice(0, count).to_pylist():
            row["sample_case"] = label
            picked.append(row)

    for day, table in sorted(tables.items()):
        available = pc.equal(table["status"], ScoringStatus.AVAILABLE.value)
        small = pc.less_equal(table["n_events"], 200)
        if pc.any(available).as_py():
            median = float(np.nanmedian(table.filter(available)["score"].to_numpy(zero_copy_only=False)))
            near = pc.and_(pc.and_(available, small), pc.less_equal(pc.abs(pc.subtract(table["score"], median)), 0.01))
            take(table, near, None, 3, False, "ordinary (score near the day median)")
            take(table, pc.and_(pc.and_(available, small), pc.equal(table["is_alert"], True)), "score", 3, True, "high-error alert")
            take(table, pc.and_(pc.and_(available, small), pc.greater(table["n_chunks"], 1)), "score", 1, True, "long hour split into several chunks")
        else:
            take(table, pc.equal(table["status"], ScoringStatus.INSUFFICIENT_HISTORY.value), None, 2, False, "warm-up: insufficient_history")
        if len(picked) >= 12:
            break
    if no_activity is not None:
        for row in no_activity.slice(0, 1).to_pylist():
            row["sample_case"] = "requested unit without events: no_activity"
            picked.append(row)

    output.mkdir(parents=True, exist_ok=True)
    cases = [row.pop("sample_case") for row in picked]
    pq.write_table(pa.Table.from_pylist(picked, schema=SEQUENCE_SCORE_SCHEMA), output / "sequence_scores_sample.parquet")
    with open(output / "sequence_scores_sample.jsonl", "w", encoding="utf-8") as handle:
        for case, row in zip(cases, picked):
            record = to_detector_record(row)
            record["sample_case"] = case
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"rows": len(picked), "cases": sorted(set(cases))}


def main() -> int:
    args = parse_args()
    inputs = load_inputs(
        args.features_root, args.splits_config, args.splits_manifest, args.feature_config,
        allow_pilot_features=args.allow_pilot_features,
    )
    detector = FrozenSequenceDetector.load(
        args.frozen_dir,
        feature_config_fingerprint=inputs.feature_cfg.fingerprint(),
        split_policy_fingerprint=inputs.split_cfg.fingerprint(),
    )
    if detector.record["feature"].get("preprocessing_sha256") != inputs.feature_info()["preprocessing_sha256"]:
        raise SystemExit("feature preprocessing artifact differs from the one the detector was frozen with")
    policy = detector.config.policy
    output = args.output or REPO_ROOT / "outputs/sequence_scores" / detector.model_version
    scores_dir = output / "scores"
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    days = args.days or sorted(d for name in args.splits for d in inputs.days_in(name))
    repeat_day = args.repeat_check_day or next((d for d in days if d in inputs.days_in("validation")), days[0])
    requested = _read_units(args.requested_units) if args.requested_units else []

    requested_by_day: dict[int, set[tuple[str, int]]] = {}
    for user, hour in requested:
        requested_by_day.setdefault(dataset_day(hour), set()).add((str(user), int(hour)))

    started = time.time()
    summary_splits: dict[str, dict] = {}
    sample_tables: dict[int, pa.Table] = {}
    missing: list[tuple[str, int]] = []
    repeat_check = None
    for day in inputs.iter_days(days, policy):
        day_started = time.time()
        table = _build(detector, day, policy.hour_seconds)
        write_day_table(table, scores_dir, day.dataset_day)
        wanted = requested_by_day.get(day.dataset_day)
        if wanted:
            present = set(zip(table["user_id"].to_pylist(), table["window_start"].to_pylist()))
            missing.extend(sorted(wanted - present))
        if day.dataset_day == repeat_day:
            again = _build(detector, day, policy.hour_seconds)
            a, b = table["raw_score"].to_numpy(zero_copy_only=False), again["raw_score"].to_numpy(zero_copy_only=False)
            both = ~np.isnan(a)
            repeat_check = {
                "dataset_day": day.dataset_day,
                "user_hours_compared": int(both.sum()),
                "max_abs_raw_difference": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0,
                "tolerance": REPEAT_TOLERANCE,
                "is_alert_identical": table["is_alert"].equals(again["is_alert"]),
            }
            repeat_check["passed"] = repeat_check["max_abs_raw_difference"] <= REPEAT_TOLERANCE and repeat_check["is_alert_identical"]
            if not repeat_check["passed"]:
                raise SystemExit(f"repeated inference differed beyond tolerance: {repeat_check}")
        if len(sample_tables) < 3 and (day.dataset_day == 1 or day.dataset_day == repeat_day):
            sample_tables[day.dataset_day] = table
        entry = summary_splits.setdefault(day.split, {"days": [], "rows": 0, "status_counts": {}, "alerts": 0, "events": 0})
        entry["days"].append(day.dataset_day)
        entry["rows"] += table.num_rows
        entry["events"] += day.n_events
        entry["alerts"] += int(pc.sum(pc.cast(pc.fill_null(table["is_alert"], False), pa.int64())).as_py() or 0)
        for status, n in status_counts(table).items():
            entry["status_counts"][status] = entry["status_counts"].get(status, 0) + n
        print(f"day {day.dataset_day:02d} [{day.split}] rows={table.num_rows:,} ({time.time() - day_started:.0f}s)")

    no_activity = None
    if requested:
        if missing:
            no_activity = no_activity_rows(
                missing, inputs.split_cfg, model_version=detector.model_version, run_id=detector.run_id,
                aggregation=detector.aggregation, alert_threshold=detector.alert_threshold, hour_seconds=policy.hour_seconds,
            )
            for day_value in sorted(set(no_activity["dataset_day"].to_pylist())):
                part = no_activity.filter(pc.equal(no_activity["dataset_day"], day_value))
                path = scores_dir / f"dataset_day={day_value:02d}" / "part-000001-no-activity.parquet"
                pq.write_table(part.drop_columns(["dataset_day"]), path, compression="zstd")
            entry = summary_splits.setdefault("requested_no_activity", {"rows": 0})
            entry["rows"] = no_activity.num_rows

    sample_info = _handoff_sample(sample_tables, no_activity, args.sample_output) if args.sample_output else None
    summary = {
        "task": "3.4 — Export sequence scores and evidence",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_revision": git_revision(),
        "schema_version": SCHEMA_VERSION,
        "schema": [{"name": f.name, "type": str(f.type)} for f in SEQUENCE_SCORE_SCHEMA],
        "reference_format": REFERENCE_FORMAT,
        "detector": detector.record["detector"],
        "model_version": detector.model_version,
        "frozen_record_sha256": detector.record["content_sha256"],
        "checkpoint_sha256": detector.record["checkpoint"]["sha256"],
        "aggregation": detector.aggregation,
        "max_sequence_length": policy.max_sequence_length,
        "alert_threshold": detector.alert_threshold,
        "score_availability": "score_available_at = window_end; a decision at time t may use only rows with score_available_at <= t",
        "missing_scores": "raw_score, score and is_alert are null unless status == 'available'; never treat them as zero",
        "splits": summary_splits,
        "repeat_inference_check": repeat_check,
        "handoff_sample": sample_info,
        "labels_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Scores written to {scores_dir}; summary {output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
