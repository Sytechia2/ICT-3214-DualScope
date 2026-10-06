#!/usr/bin/env python3
"""Export the final model's Days 13-16 per-hour scores for Tasks 9.3 and 9.4.

Fits the final supervised fusion model (HistGradientBoosting on the GRU score and
hourly counts) on Days 08-12 with the unchanged code of
``scripts/supervised_fusion_validate.py`` and
writes one row per Days 13-16 user-hour: ``user_id``, ``window_start``, ``score``
and ``tie_order`` (that experiment's fixed seed-0 order, used to break ties).
It refuses to write unless it reproduces the experiment's recorded result.

``--experiment-root`` can point at another checkout; by default this one is
used. Days 17-30 are never read.

Example:
  python scripts/export_final_model_scores.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TP, EXPECTED_AP = 16, 0.04984


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-root", type=Path, default=REPO_ROOT,
                        help="Checkout containing scripts/supervised_fusion_validate.py")
    parser.add_argument("--units", type=Path, default=REPO_ROOT / "outputs/experiment_v2/units_A.parquet")
    parser.add_argument("--counts", type=Path, default=REPO_ROOT / "outputs/experiment_v2/fusion_counts.parquet")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/final_model_scores_days13_16.parquet")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    sys.path.insert(0, str(root / "src"))
    spec = importlib.util.spec_from_file_location("supervised_fusion_validate", root / "scripts" / "supervised_fusion_validate.py")
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)
    from dualscope.sequence.calibration import load_positive_user_hours, unit_labels
    from dualscope.splits import SplitConfig

    split_cfg = SplitConfig.from_file(root / "config/lanl_splits.json")
    positives = load_positive_user_hours(args.labels_dir, split_cfg, "validation")
    counts = pd.read_parquet(args.counts)
    gru = pd.read_parquet(args.units, columns=["user", "hour", "day", fusion.GRU_COLUMN])
    gru = gru.rename(columns={fusion.GRU_COLUMN: "gru_max_event"})
    frame = counts.merge(gru, on=["user", "hour", "day"], how="inner", validate="1:1")
    if not len(frame) == len(counts) == len(gru):
        raise SystemExit(f"unit mismatch: counts {len(counts):,}, GRU {len(gru):,}, joined {len(frame):,}")
    frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    frame["machine"] = fusion._evaluate.is_machine(frame["user"].to_numpy())
    # Same canonical order and seed-0 shuffle as supervised_fusion_validate.py.
    frame = frame.sort_values(["day", "user", "hour"], kind="stable").sample(frac=1.0, random_state=0).reset_index(drop=True)
    train = frame[frame["day"].isin(fusion.TRAIN_DAYS)].reset_index(drop=True)
    test = frame[frame["day"].isin(fusion.EVAL_DAYS)].reset_index(drop=True)

    model = fusion.MODELS["hist_gradient_boosting"]().fit(fusion.model_inputs(train), train["label"].to_numpy())
    test["score"] = model.predict_proba(fusion.model_inputs(test))[:, 1]

    test_positives = {p for p in positives if 1 + (p[1] - 1) // 86400 in fusion.EVAL_DAYS}
    metrics = fusion._evaluate.evaluate(
        test[["user", "day", "label", "machine"]], test["score"].to_numpy(np.float64), test_positives, 38
    )["user_hour"]
    tp, ap = metrics["at_budget"]["tp"], metrics["average_precision"]
    print(f"Days 13-16: {tp}/{len(test_positives)} attacks at 38 alerts/day, AP {ap:.5f}")
    if tp != EXPECTED_TP or round(ap, 5) != EXPECTED_AP:
        raise SystemExit(f"does not reproduce the recorded result ({EXPECTED_TP}, AP {EXPECTED_AP}); nothing written")

    test["tie_order"] = np.arange(len(test))
    out = test[["user", "hour", "score", "tie_order"]].rename(columns={"user": "user_id", "hour": "window_start"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
