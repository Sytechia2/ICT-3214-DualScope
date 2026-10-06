"""Experiment features-v2: label-free counting rules as a reference for the sequence detector.

Each validation user-hour (acting user, hour) is scored by counting its events with:

* ``new_user_source``: ``is_new_user_source`` (first login by this user from this computer);
* ``new_host_connection``: ``is_new_host_connection`` (first ever source -> destination pair);
* ``either``: events with either flag.

Ties are broken by a fixed random order (seed 0), so the alert budget does not
depend on file or user order. Units, labels, budget and metrics are the same as
``evaluate_sequence_runs.py``. No model, no training and no labels are used to
build the scores; validation labels are used only to evaluate them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("evaluate_sequence_runs", REPO_ROOT / "scripts" / "evaluate_sequence_runs.py")
_evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate)

from dualscope.sequence.calibration import load_positive_user_hours, unit_labels  # noqa: E402
from dualscope.splits import SplitConfig  # noqa: E402


RULES = {
    "new_user_source": ["is_new_user_source"],
    "new_host_connection": ["is_new_host_connection"],
    "either": ["is_new_user_source", "is_new_host_connection"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/rules.json")
    args = parser.parse_args()

    split_cfg = SplitConfig.from_file(args.splits_config)
    validation = split_cfg.get_split("validation")
    positives = load_positive_user_hours(args.labels_dir, split_cfg, "validation")
    dataset = ds.dataset(str(args.features_root / "raw" / "events"), format="parquet", partitioning="hive")
    days = [d for d in range(1, 31) if validation.contains_day(d)]
    if max(days) > 16:
        raise SystemExit(f"unexpected validation days {days}")

    parts = []
    for day in days:
        started = time.time()
        table = dataset.to_table(
            columns=["acting_user", "timestamp", "is_new_user_source", "is_new_host_connection"],
            filter=ds.field("dataset_day") == day,
        ).to_pandas()
        table["hour"] = 1 + ((table["timestamp"] - 1) // 3600) * 3600
        table["either"] = table["is_new_user_source"] | table["is_new_host_connection"]
        grouped = table.groupby(["acting_user", "hour"], sort=False).agg(
            n_events=("timestamp", "size"),
            new_user_source=("is_new_user_source", "sum"),
            new_host_connection=("is_new_host_connection", "sum"),
            either=("either", "sum"),
        ).reset_index().rename(columns={"acting_user": "user"})
        grouped["day"] = day
        parts.append(grouped)
        print(f"day {day}: {len(grouped):,} user-hours ({time.time() - started:.0f}s)")
    frame = pd.concat(parts, ignore_index=True)
    frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    frame["machine"] = _evaluate.is_machine(frame["user"].to_numpy())
    print(f"{len(frame):,} validation user-hours, {int(frame['label'].sum())} positive of {len(positives)}")

    # Fixed random tie-break: shuffle rows once, then every stable sort keeps that order among ties.
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    results = {
        "experiment": "features-v2 counting rules (label-free)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features_root": str(args.features_root),
        "validation_days": days,
        "budget_per_day": args.budget,
        "tie_break": "fixed random order, seed 0",
        "test_labels_used": False,
        "rules": {},
    }
    print(f"\n{'rule':<20} {'AP':>8} {'ROC':>6} {'TP@budget':>10} {'machine/human':>14} {'day AP':>8} {'day TP@b':>9}")
    for rule in RULES:
        metrics = _evaluate.evaluate(frame.drop(columns=["score"], errors="ignore"), frame[rule].to_numpy(np.float64), positives, args.budget)
        results["rules"][rule] = {"columns": RULES[rule], **metrics}
        hour, day = metrics["user_hour"], metrics["user_day"]
        b = hour["at_budget"]
        print(
            f"{rule:<20} {hour['average_precision']:>8.5f} {hour['roc_auc']:>6.3f} {b['tp']:>4}/{len(positives):<5} "
            f"{b['alerts_machine_accounts']:>6}/{b['alerts_human_accounts']:<7} {day['average_precision']:>8.5f} "
            f"{day['at_budget']['tp']:>4}/{day['positive_units_scored']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
