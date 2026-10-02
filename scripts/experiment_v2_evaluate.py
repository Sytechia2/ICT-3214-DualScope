"""Experiment features-v2 (docs/experiment_features_v2.md, steps 4-5): compare trained sequence runs on validation days.

For each ``--run NAME=RUN_DIR`` the trained checkpoint scores every available
validation user-hour (days 8-16). With ``--scaled NAME`` the run is also scored
with each feature's loss divided by its mean loss on that run's training sample
(label-free), reported as ``NAME+scaled``.

Reported per variant and aggregation, using validation red-team labels only:

* average precision and ROC-AUC over user-hours (as in Task 3.3);
* recall at a fixed alert budget: the top ``--budget`` user-hours per day;
* the same at user-day level (max over the day's hours);
* alerts within the budget split into machine (``$``) and human accounts.

Test days are never loaded and test labels are never read.

Example:
  python scripts/experiment_v2_evaluate.py --allow-pilot-features --device cuda \
    --features-root data/processed/lanl_features_v2_days_01_16 \
    --run A=models/sequence/runs_experiment_v2/A --run B=models/sequence/runs_experiment_v2/B --scaled A --scaled B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.sequence.calibration import (  # noqa: E402
    average_precision,
    evaluation_summary,
    load_positive_user_hours,
    unit_labels,
)
from dualscope.sequence.config import SequenceDetectorConfig  # noqa: E402
from dualscope.sequence.pipeline import load_inputs, prepare_samples  # noqa: E402
from dualscope.sequence.scoring import score_day  # noqa: E402
from dualscope.sequence.training import load_checkpoint, mean_feature_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features_v2.json")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/processed/sequence_cache_experiment_v2")
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (repeatable)")
    parser.add_argument("--scaled", action="append", default=[], help="Also score this run with per-feature loss scaling")
    parser.add_argument("--budget", type=int, default=38, help="Alerts per validation day (v1 raised 37.8/day)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/results.json")
    parser.add_argument("--units-output", type=Path, help="Also write per-unit raw scores (parquet), one column per variant:aggregation")
    parser.add_argument("--allow-pilot-features", action="store_true")
    return parser.parse_args()


def is_machine(users: np.ndarray) -> np.ndarray:
    return np.fromiter((str(u).split("@", 1)[0].endswith("$") for u in users), dtype=bool, count=len(users))


def budget_metrics(frame: pd.DataFrame, score: str, budget: int, positives_total: int) -> dict:
    """Top ``budget`` units per day by score (stable ties)."""
    ranked = frame.sort_values(["day", score], ascending=[True, False], kind="stable")
    top = ranked.groupby("day", sort=False).head(budget)
    tp = int(top["label"].sum())
    return {
        "alerts": int(len(top)),
        "tp": tp,
        "precision": tp / len(top) if len(top) else None,
        "recall": tp / positives_total if positives_total else None,
        "alerts_machine_accounts": int(top["machine"].sum()),
        "alerts_human_accounts": int((~top["machine"]).sum()),
    }


def evaluate(units: pd.DataFrame, scores: np.ndarray, positives: set, budget: int) -> dict:
    frame = units.assign(score=scores)
    hour = evaluation_summary(frame["label"].to_numpy(), scores, len(positives))
    hour["at_budget"] = budget_metrics(frame, "score", budget, len(positives))

    days = frame.groupby(["user", "day"], sort=False).agg(score=("score", "max"), label=("label", "any"), machine=("machine", "first"))
    days = days.reset_index()
    positive_days = int(days["label"].sum())
    user_day = {
        "units_scored": int(len(days)),
        "positive_units_scored": positive_days,
        "average_precision": average_precision(days["label"].to_numpy(), days["score"].to_numpy()),
        "at_budget": budget_metrics(days, "score", budget, positive_days),
    }
    return {"user_hour": hour, "user_day": user_day}


def main() -> int:
    args = parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    unknown = set(args.scaled) - set(runs)
    if unknown:
        raise SystemExit(f"--scaled names not in --run: {sorted(unknown)}")
    inputs = load_inputs(
        args.features_root, args.splits_config, args.splits_manifest, args.feature_config,
        labels_dir=args.labels_dir, allow_pilot_features=args.allow_pilot_features,
    )
    validation_days = inputs.days_in("validation")
    if not validation_days or max(validation_days) > 16:
        raise SystemExit(f"unexpected validation days {validation_days}")
    positives = load_positive_user_hours(args.labels_dir, inputs.split_cfg, "validation")
    print(f"Validation days {validation_days}; positive user-hours: {len(positives)}")

    results: dict = {
        "experiment": "features-v2 (docs/experiment_features_v2.md)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features": inputs.feature_info(),
        "validation_days": validation_days,
        "budget_per_day": args.budget,
        "test_labels_used": False,
        "variants": {},
    }
    unit_scores: dict[str, pd.DataFrame] = {}
    for name, run_dir in runs.items():
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        config = SequenceDetectorConfig.from_dict(manifest["sequence_config"])
        if config.fingerprint() != manifest["sequence_config_fingerprint"]:
            raise SystemExit(f"{name}: run configuration fingerprint mismatch")
        if manifest["feature"]["preprocessing_sha256"] != inputs.feature_info()["preprocessing_sha256"]:
            raise SystemExit(f"{name}: run was trained on a different feature build")
        model, _ = load_checkpoint(run_dir / "checkpoint.pt", manifest["checkpoint"]["sha256"])
        model.to(args.device)

        variants = {name: None}
        if name in args.scaled:
            train, _, _ = prepare_samples(inputs, config, args.cache_dir)
            scale = mean_feature_errors(model, train)
            variants[f"{name}+scaled"] = scale
            print(f"{name}: per-feature training mean loss {np.round(scale, 4).tolist()}")

        started = time.time()
        units = {"user": [], "hour": [], "day": [], "n_events": []}
        raw = {variant: {} for variant in variants}
        for day in inputs.iter_days(validation_days, config.policy):
            available = day.scorable_mask()
            units["user"].append(day.users[available])
            units["hour"].append(day.hour_starts[available])
            units["day"].append(np.full(int(available.sum()), day.dataset_day))
            units["n_events"].append(day.counts[available])
            for variant, scale in variants.items():
                scores = score_day(model, day, config.policy.max_sequence_length, feature_scale=scale)
                for aggregation, values in scores.raw.items():
                    raw[variant].setdefault(aggregation, []).append(values[available])
            print(f"  {name}: day {day.dataset_day} scored ({time.time() - started:.0f}s)")
        frame = pd.DataFrame({key: np.concatenate(values) for key, values in units.items()})
        frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
        frame["machine"] = is_machine(frame["user"].to_numpy())

        if args.units_output:
            unit_scores[name] = frame[["user", "hour", "day"]].assign(
                **{f"{v}:{a}": np.concatenate(parts) for v in variants for a, parts in raw[v].items()}
            )
        for variant, scale in variants.items():
            entry = {
                "run": name,
                "run_dir": str(run_dir),
                "inputs": list(config.policy.binary_inputs),
                "feature_scale": None if scale is None else scale.tolist(),
                "aggregations": {},
            }
            for aggregation, parts in raw[variant].items():
                entry["aggregations"][aggregation] = evaluate(frame, np.concatenate(parts), positives, args.budget)
            results["variants"][variant] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.units_output:
        merged = None
        for part in unit_scores.values():
            merged = part if merged is None else merged.merge(part, on=["user", "hour", "day"], how="outer", validate="1:1")
        args.units_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(args.units_output, index=False)
        print(f"Per-unit scores written to {args.units_output} ({len(merged):,} rows)")

    print(f"\n{'variant':<12} {'agg':<15} {'AP':>8} {'ROC':>6} {'TP@budget':>10} {'machine/human':>14} {'day AP':>8} {'day TP@b':>9}")
    for variant, entry in results["variants"].items():
        for aggregation, metrics in entry["aggregations"].items():
            hour, day = metrics["user_hour"], metrics["user_day"]
            b = hour["at_budget"]
            print(
                f"{variant:<12} {aggregation:<15} {hour['average_precision']:>8.5f} {hour['roc_auc']:>6.3f} "
                f"{b['tp']:>4}/{len(positives):<5} {b['alerts_machine_accounts']:>6}/{b['alerts_human_accounts']:<7} "
                f"{day['average_precision']:>8.5f} {day['at_budget']['tp']:>4}/{day['positive_units_scored']}"
            )
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
