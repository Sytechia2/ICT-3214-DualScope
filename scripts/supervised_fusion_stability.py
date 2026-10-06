"""Experiment features-v2, Option A: stability of the fusion result to GRU score noise.

Rebuilding the features and retraining the GRU (same settings, CPU instead of
GPU scoring) changed the GRU scores by a median 0.02% (rank correlation
0.9999998), and the gradient boosting result on days 13-16 moved from 16/136
to 13/136. This script measures that sensitivity on validation days only:

* the GRU score of every user-hour is multiplied by (1 + e), e ~ Normal(0, sigma),
  independently per seed (default sigma 3e-4, which gives the observed median
  relative change of about 2e-4);
* both fusion models are refitted on days 8-12 with their fixed settings and
  scored on days 13-16, exactly as in ``supervised_fusion_validate.py``.

The median and range over seeds are the reported validation result. Test days
(17-30) are never loaded and test labels are never read. The frozen model is
not changed.

Example:
  python scripts/supervised_fusion_stability.py --units outputs/final_test/units_A.parquet \
    --counts outputs/final_test/fusion_counts.parquet --output outputs/final_test/stability.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("supervised_fusion_validate", REPO_ROOT / "scripts" / "supervised_fusion_validate.py")
_fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fusion)

from dualscope.sequence.calibration import load_positive_user_hours, unit_labels  # noqa: E402
from dualscope.splits import SplitConfig  # noqa: E402


def load_frame(units: Path, counts: Path, positives: set) -> pd.DataFrame:
    """The experiment's unit table, label column and fixed seed-0 order."""
    gru = pd.read_parquet(units, columns=["user", "hour", "day", _fusion.GRU_COLUMN])
    gru = gru.rename(columns={_fusion.GRU_COLUMN: "gru_max_event"})
    table = pd.read_parquet(counts)
    frame = table.merge(gru, on=["user", "hour", "day"], how="inner", validate="1:1")
    if not len(frame) == len(table) == len(gru):
        raise SystemExit(f"unit mismatch: counts {len(table):,}, GRU {len(gru):,}, joined {len(frame):,}")
    frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    frame["machine"] = _fusion._evaluate.is_machine(frame["user"].to_numpy())
    return frame.sort_values(["day", "user", "hour"], kind="stable").sample(frac=1.0, random_state=0).reset_index(drop=True)


def fit_and_score(frame: pd.DataFrame, test_positives: set, budget: int) -> dict:
    train = frame[frame["day"].isin(_fusion.TRAIN_DAYS)].reset_index(drop=True)
    test = frame[frame["day"].isin(_fusion.EVAL_DAYS)].reset_index(drop=True)
    units = test[["user", "day", "label", "machine"]]
    out = {}
    for name, build in _fusion.MODELS.items():
        model = build().fit(_fusion.model_inputs(train), train["label"].to_numpy())
        scores = model.predict_proba(_fusion.model_inputs(test))[:, 1]
        hour = _fusion._evaluate.evaluate(units, scores, test_positives, budget)["user_hour"]
        out[name] = {"tp": hour["at_budget"]["tp"], "average_precision": hour["average_precision"]}
    return out


def summarise(runs: list[dict], name: str) -> dict:
    tp = np.array([r[name]["tp"] for r in runs])
    ap = np.array([r[name]["average_precision"] for r in runs])
    return {
        "tp_median": float(np.median(tp)), "tp_min": int(tp.min()), "tp_max": int(tp.max()),
        "ap_median": float(np.median(ap)), "ap_min": float(ap.min()), "ap_max": float(ap.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--sigma", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_cfg = SplitConfig.from_file(args.splits_config)
    positives = load_positive_user_hours(args.labels_dir, split_cfg, "validation")
    test_positives = {p for p in positives if 1 + (p[1] - 1) // 86400 in _fusion.EVAL_DAYS}
    frame = load_frame(args.units, args.counts, positives)
    clean = frame["gru_max_event"].to_numpy().copy()

    started = time.time()
    unperturbed = fit_and_score(frame, test_positives, args.budget)
    print(f"unperturbed: {unperturbed} ({time.time() - started:.0f}s)")
    runs = []
    for seed in range(1, args.seeds + 1):
        rng = np.random.default_rng(seed)
        frame["gru_max_event"] = clean * (1.0 + rng.normal(0.0, args.sigma, len(clean)))
        run = fit_and_score(frame, test_positives, args.budget)
        runs.append({"seed": seed, **run})
        print(f"seed {seed}: " + ", ".join(f"{m} {r['tp']}/{len(test_positives)} AP {r['average_precision']:.5f}" for m, r in run.items())
              + f" ({time.time() - started:.0f}s)")

    report = {
        "experiment": "features-v2 Option A: fusion stability to GRU score noise (validation days only)",
        "units": str(args.units),
        "counts": str(args.counts),
        "sigma_relative": args.sigma,
        "seeds": args.seeds,
        "evaluation_positives": len(test_positives),
        "budget_per_day": args.budget,
        "test_labels_used": False,
        "unperturbed": unperturbed,
        "runs": runs,
        "summary": {name: summarise(runs, name) for name in _fusion.MODELS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in report["summary"].items():
        print(f"{name}: TP median {s['tp_median']:.0f} (range {s['tp_min']}-{s['tp_max']}) of {len(test_positives)}; "
              f"AP median {s['ap_median']:.4f} (range {s['ap_min']:.4f}-{s['ap_max']:.4f})")
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
