"""Experiment features-v2, Option A: one-shot final test on days 17-30 (docs/experiment_features_v2.md).

Scores every test user-hour with the frozen fusion model and the comparators,
saves the scores, and only then reads the test red-team labels. Before any test
label is read it checks that the days 1-30 feature build reproduces the
validation inputs the model was fitted on (preprocessing hash, day 16 GRU scores
and hourly counts). Refuses to run if its output already exists.

Example:
  python scripts/experiment_v2_final_test.py --device cuda \
    --features-root data/processed/lanl_features_v2_days_01_30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parent.parent
_load = importlib.util.spec_from_file_location
_spec = _load("experiment_v2_fusion", REPO_ROOT / "scripts" / "experiment_v2_fusion.py")
_fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fusion)
_spec = _load("experiment_v2_freeze_fusion", REPO_ROOT / "scripts" / "experiment_v2_freeze_fusion.py")
_freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_freeze)
_evaluate = _fusion._evaluate

from dualscope.sequence.calibration import load_positive_user_hours, unit_labels  # noqa: E402
from dualscope.sequence.config import SequenceDetectorConfig  # noqa: E402
from dualscope.sequence.pipeline import load_inputs  # noqa: E402
from dualscope.sequence.scoring import score_day  # noqa: E402
from dualscope.sequence.training import load_checkpoint  # noqa: E402

CHECK_DAY = 16


def gru_scores(model, inputs, config, days: list[int]) -> pd.DataFrame:
    parts = []
    started = time.time()
    for day in inputs.iter_days(days, config.policy):
        available = day.scorable_mask()
        scores = score_day(model, day, config.policy.max_sequence_length)
        parts.append(pd.DataFrame({
            "user": day.users[available], "hour": day.hour_starts[available], "day": day.dataset_day,
            "gru_max_event": scores.raw["max_event"][available],
        }))
        print(f"  GRU: day {day.dataset_day} scored, {int(available.sum()):,} of {len(available):,} user-hours available ({time.time() - started:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features_v2.json")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--frozen-dir", type=Path, default=REPO_ROOT / "models/fusion/experiment_v2/frozen")
    parser.add_argument("--validation-units", type=Path, default=REPO_ROOT / "outputs/experiment_v2/units_A.parquet")
    parser.add_argument("--validation-counts", type=Path, default=REPO_ROOT / "outputs/experiment_v2/fusion_counts.parquet")
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scores-output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/final_test_scores.parquet")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/final_test.json")
    args = parser.parse_args()
    for path in (args.output, args.scores_output):
        if path.exists():
            raise SystemExit(f"{path} exists: the final test runs once")

    # Frozen fusion model and its GRU input.
    manifest = json.loads((args.frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    if _freeze.sha256(args.frozen_dir / "model.joblib") != manifest["model_sha256"]:
        raise SystemExit("frozen model file changed")
    if manifest["inputs"] != _fusion.INPUTS:
        raise SystemExit("frozen model inputs differ from the fusion script")
    fusion_model = joblib.load(args.frozen_dir / "model.joblib")
    gru_run = Path(manifest["gru_score"]["run_dir"])
    run_manifest = json.loads((gru_run / "run_manifest.json").read_text(encoding="utf-8"))
    config = SequenceDetectorConfig.from_dict(run_manifest["sequence_config"])
    if config.fingerprint() != manifest["gru_score"]["sequence_config_fingerprint"]:
        raise SystemExit("GRU run configuration differs from the frozen manifest")
    gru_model, _ = load_checkpoint(gru_run / "checkpoint.pt", manifest["gru_score"]["checkpoint_sha256"])
    gru_model.to(args.device)

    inputs = load_inputs(args.features_root, args.splits_config, args.splits_manifest, args.feature_config)
    if inputs.feature_info()["preprocessing_sha256"] != manifest["gru_score"]["preprocessing_sha256"]:
        raise SystemExit("days 1-30 build has a different preprocessing file than the GRU was trained with")
    dataset = ds.dataset(str(args.features_root / "raw" / "events"), format="parquet", partitioning="hive")

    # Reproduction check on a validation day, before any test label is read.
    check = gru_scores(gru_model, inputs, config, [CHECK_DAY]).merge(_fusion.hourly_counts(dataset, CHECK_DAY), on=["user", "hour", "day"], validate="1:1")
    reference = pd.read_parquet(args.validation_units, columns=["user", "hour", "day", _fusion.GRU_COLUMN], filters=[("day", "==", CHECK_DAY)])
    reference = reference.merge(pd.read_parquet(args.validation_counts, filters=[("day", "==", CHECK_DAY)]), on=["user", "hour", "day"], validate="1:1")
    both = check.merge(reference, on=["user", "hour", "day"], how="outer", suffixes=("", "_ref"), validate="1:1", indicator=True)
    if not (both["_merge"] == "both").all():
        raise SystemExit(f"day {CHECK_DAY} units differ from the validation build")
    gru_diff = float(np.max(np.abs(both["gru_max_event"] - both[_fusion.GRU_COLUMN])))
    count_columns = [c for c in _fusion.INPUTS[1:]] + ["either"]
    counts_equal = all(np.array_equal(both[c].to_numpy(np.int64), both[f"{c}_ref"].to_numpy(np.int64)) for c in count_columns)
    print(f"day {CHECK_DAY} reproduction: {len(both):,} user-hours, max GRU score difference {gru_diff:.2e}, counts equal {counts_equal}")
    if gru_diff > 1e-4 or not counts_equal:
        raise SystemExit("days 1-30 build does not reproduce the validation inputs")

    test_days = inputs.days_in("test")
    if test_days != list(range(17, 31)):
        raise SystemExit(f"unexpected test days {test_days}")

    # Score every test user-hour; no labels yet.
    gru = gru_scores(gru_model, inputs, config, test_days)
    parts = []
    for day in test_days:
        started = time.time()
        parts.append(_fusion.hourly_counts(dataset, day))
        print(f"  counts: day {day} ({time.time() - started:.0f}s)")
    counts = pd.concat(parts, ignore_index=True)
    frame = gru.merge(counts, on=["user", "hour", "day"], how="inner", validate="1:1")
    if len(frame) != len(gru):
        raise SystemExit("some GRU-scored user-hours have no events")
    frame["machine"] = _evaluate.is_machine(frame["user"].to_numpy())
    frame = frame.sort_values(["day", "user", "hour"], kind="stable").sample(frac=1.0, random_state=0).reset_index(drop=True)
    frame["fusion"] = fusion_model.predict_proba(_fusion.model_inputs(frame))[:, 1]
    frame["gru_humans_only"] = np.where(frame["machine"], -1.0, frame["gru_max_event"])
    frame["either_humans_only"] = np.where(frame["machine"], -1.0, frame["either"])
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["machine"]).to_parquet(args.scores_output, index=False)
    print(f"Test scores written to {args.scores_output} ({len(frame):,} user-hours; {len(counts) - len(frame):,} user-hours with events not scored by the GRU)")

    # Only now: test labels.
    positives = load_positive_user_hours(args.labels_dir, inputs.split_cfg, "test", allow_test=True)
    frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    units = frame[["user", "day", "label", "machine"]]
    scorers = {
        "fusion_frozen": "fusion", "gru_max_event": "gru_max_event", "rule_either": "either",
        "gru_humans_only": "gru_humans_only", "rule_either_humans_only": "either_humans_only",
    }
    results: dict = {
        "experiment": "features-v2 Option A, final test (docs/experiment_features_v2.md)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_model": {"dir": str(args.frozen_dir), "sha256": manifest["model_sha256"], "git_revision": manifest["git_revision"]},
        "test_days": test_days,
        "budget_per_day": args.budget,
        "units": int(len(frame)),
        "user_hours_with_events_not_scored": int(len(counts) - len(frame)),
        "positives_total": len(positives),
        "positives_scored": int(frame["label"].sum()),
        "reproduction_check": {"day": CHECK_DAY, "max_gru_difference": gru_diff, "counts_equal": counts_equal},
        "tie_break": "fixed random order, seed 0",
        "test_labels_used": True,
        "scorers": {},
    }
    for name, column in scorers.items():
        results["scorers"][name] = _evaluate.evaluate(units, frame[column].to_numpy(np.float64), positives, args.budget)
        top = frame.sort_values(["day", column], ascending=[True, False], kind="stable").groupby("day", sort=False).head(args.budget)
        hits = top[top["label"]]
        results["scorers"][name]["hits_by_day"] = {int(k): int(v) for k, v in hits["day"].value_counts().sort_index().items()}
        results["scorers"][name]["hit_users"] = int(hits["user"].nunique())

    def hour(name: str) -> dict:
        return results["scorers"][name]["user_hour"]

    fusion, gru_hour, humans = hour("fusion_frozen"), hour("gru_max_event"), hour("gru_humans_only")
    needed = max(2 * gru_hour["at_budget"]["tp"], 2)
    results["success_rules"] = {
        "primary": {
            "rule": "fusion TP at budget >= 2x GRU TP (at least 2 if the GRU catches 0) and fusion AP > GRU AP",
            "tp_needed": needed,
            "met": bool(fusion["at_budget"]["tp"] >= needed and fusion["average_precision"] > gru_hour["average_precision"]),
        },
        "secondary": {
            "rule": "fusion TP at budget > GRU-humans-only TP and fusion AP > GRU-humans-only AP",
            "met": bool(fusion["at_budget"]["tp"] > humans["at_budget"]["tp"] and fusion["average_precision"] > humans["average_precision"]),
        },
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_pos = len(positives)
    print(f"\nTest days 17-30: {len(frame):,} user-hours, {n_pos} positive user-hours ({results['positives_scored']} scored)")
    print(f"{'scorer':<24} {'AP':>8} {'ROC':>6} {'TP@budget':>10} {'machine/human':>14} {'day AP':>8} {'day TP@b':>9}  hits by day / users")
    for name, metrics in results["scorers"].items():
        h, d = metrics["user_hour"], metrics["user_day"]
        b = h["at_budget"]
        print(
            f"{name:<24} {h['average_precision']:>8.5f} {h['roc_auc']:>6.3f} {b['tp']:>4}/{n_pos:<5} "
            f"{b['alerts_machine_accounts']:>6}/{b['alerts_human_accounts']:<7} {d['average_precision']:>8.5f} "
            f"{d['at_budget']['tp']:>4}/{d['positive_units_scored']}  {metrics['hits_by_day']} / {metrics['hit_users']}"
        )
    for rule, info in results["success_rules"].items():
        print(f"{rule} rule ({info['rule']}): {'MET' if info['met'] else 'NOT met'}")
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
