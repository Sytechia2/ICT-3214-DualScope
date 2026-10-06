"""Experiment features-v2, Option A: supervised fusion quick test (docs/experiment_features_v2.md).

One row per validation user-hour (acting user, hour), days 8-16. Inputs are
label-free and contain no user or computer names:

* run A's raw ``max_event`` score, from ``evaluate_sequence_runs.py --units-output``;
* hourly counts from the v2 raw events (events, failures, distinct sources and
  destinations, the three novelty flags, NTLM, Network logon type, LogOn);
* ``is_machine_account``.

Two models with settings fixed in advance (logistic regression and
HistGradientBoosting, both class-balanced) are fitted on days 8-12 with
validation red-team labels and compared on days 13-16 with the GRU alone and the
counting rule "either". Units, labels, budget and metrics are those of
``evaluate_sequence_runs.py``. Test days (17-30) are never loaded and test
labels are never read.

Example:
  python scripts/supervised_fusion_validate.py --features-root data/processed/lanl_features_v2_days_01_16 \
    --units outputs/experiment_v2/units_A.parquet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("evaluate_sequence_runs", REPO_ROOT / "scripts" / "evaluate_sequence_runs.py")
_evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate)

from dualscope.sequence.calibration import average_precision, load_positive_user_hours, unit_labels  # noqa: E402
from dualscope.splits import SplitConfig  # noqa: E402


TRAIN_DAYS = range(8, 13)
EVAL_DAYS = range(13, 17)
GRU_COLUMN = "A:max_event"
COUNTS = [
    "n_events", "n_failures", "n_sources", "n_destinations",
    "n_new_user_source", "n_new_host_connection", "n_new_user_destination",
    "n_ntlm", "n_network_logon", "n_logon",
]
INPUTS = ["gru_max_event", *COUNTS, "is_machine_account"]


def _codes(column) -> tuple[np.ndarray, np.ndarray]:
    encoded = column.combine_chunks().dictionary_encode()
    return encoded.indices.to_numpy(zero_copy_only=False), encoded.dictionary.to_numpy(zero_copy_only=False)


def hourly_counts(dataset: ds.Dataset, day: int) -> pd.DataFrame:
    table = dataset.to_table(
        columns=[
            "acting_user", "timestamp", "source_computer", "destination_computer", "authentication_type",
            "logon_type", "authentication_orientation", "authentication_result", "is_new_user_source",
            "is_new_host_connection", "is_new_user_destination", "is_machine_account",
        ],
        filter=ds.field("dataset_day") == day,
    )
    user, users = _codes(table["acting_user"])
    frame = pd.DataFrame({
        "u": user,
        "hour": 1 + ((table["timestamp"].to_numpy() - 1) // 3600) * 3600,
        "src": _codes(table["source_computer"])[0],
        "dst": _codes(table["destination_computer"])[0],
        "n_failures": pc.equal(table["authentication_result"], "Fail").to_numpy(),
        "n_new_user_source": table["is_new_user_source"].to_numpy(),
        "n_new_host_connection": table["is_new_host_connection"].to_numpy(),
        "n_new_user_destination": table["is_new_user_destination"].to_numpy(),
        "n_ntlm": pc.equal(table["authentication_type"], "NTLM").to_numpy(),
        "n_network_logon": pc.equal(table["logon_type"], "Network").to_numpy(),
        "n_logon": pc.equal(table["authentication_orientation"], "LogOn").to_numpy(),
        "is_machine_account": table["is_machine_account"].to_numpy(),
    })
    del table
    frame["either"] = frame["n_new_user_source"] | frame["n_new_host_connection"]
    flags = ["n_failures", "n_new_user_source", "n_new_host_connection", "n_new_user_destination", "n_ntlm", "n_network_logon", "n_logon", "either"]
    grouped = frame.groupby(["u", "hour"], sort=False)
    out = grouped[flags].sum()
    out["n_events"] = grouped.size()
    out["is_machine_account"] = grouped["is_machine_account"].max()
    for column, name in (("src", "n_sources"), ("dst", "n_destinations")):
        out[name] = frame[["u", "hour", column]].drop_duplicates().groupby(["u", "hour"], sort=False).size()
    out = out.reset_index()
    out.insert(0, "user", users[out.pop("u").to_numpy()])
    out["day"] = day
    return out


def model_inputs(frame: pd.DataFrame) -> np.ndarray:
    return frame[INPUTS].to_numpy(np.float64)


def _log_inputs(x: np.ndarray) -> np.ndarray:
    out = np.log1p(x)
    out[:, 0] = np.log(x[:, 0] + 1e-12)  # GRU score
    out[:, -1] = x[:, -1]  # is_machine_account stays 0/1
    return out


MODELS = {
    "logistic_regression": lambda: make_pipeline(
        FunctionTransformer(_log_inputs), StandardScaler(),
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000),
    ),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True, help="Per-unit GRU scores from evaluate_sequence_runs.py --units-output")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--counts-cache", type=Path, default=REPO_ROOT / "outputs/experiment_v2/fusion_counts.parquet")
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/experiment_v2/fusion.json")
    args = parser.parse_args()

    split_cfg = SplitConfig.from_file(args.splits_config)
    validation = split_cfg.get_split("validation")
    days = [d for d in range(1, 31) if validation.contains_day(d)]
    if days != [*TRAIN_DAYS, *EVAL_DAYS]:
        raise SystemExit(f"unexpected validation days {days}")
    positives = load_positive_user_hours(args.labels_dir, split_cfg, "validation")

    if args.counts_cache.exists():
        counts = pd.read_parquet(args.counts_cache)
        print(f"Hourly counts loaded from {args.counts_cache}")
    else:
        dataset = ds.dataset(str(args.features_root / "raw" / "events"), format="parquet", partitioning="hive")
        parts = []
        for day in days:
            started = time.time()
            parts.append(hourly_counts(dataset, day))
            print(f"day {day}: {len(parts[-1]):,} user-hours counted ({time.time() - started:.0f}s)")
        counts = pd.concat(parts, ignore_index=True)
        args.counts_cache.parent.mkdir(parents=True, exist_ok=True)
        counts.to_parquet(args.counts_cache, index=False)

    gru = pd.read_parquet(args.units, columns=["user", "hour", "day", GRU_COLUMN]).rename(columns={GRU_COLUMN: "gru_max_event"})
    frame = counts.merge(gru, on=["user", "hour", "day"], how="inner", validate="1:1")
    if not len(frame) == len(counts) == len(gru):
        raise SystemExit(f"unit mismatch: counts {len(counts):,}, GRU {len(gru):,}, joined {len(frame):,}")
    frame["label"] = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    frame["machine"] = _evaluate.is_machine(frame["user"].to_numpy())
    if not np.array_equal(frame["machine"].to_numpy(), frame["is_machine_account"].to_numpy(bool)):
        raise SystemExit("is_machine_account disagrees with the evaluation's machine split")
    if int(frame["label"].sum()) != len(positives):
        raise SystemExit(f"only {int(frame['label'].sum())} of {len(positives)} positives matched")
    # Fixed random tie-break, as in evaluate_counting_rules.py: canonical order, one shuffle, stable sorts afterwards.
    frame = frame.sort_values(["day", "user", "hour"], kind="stable").sample(frac=1.0, random_state=0).reset_index(drop=True)

    train = frame[frame["day"].isin(TRAIN_DAYS)].reset_index(drop=True)
    test = frame[frame["day"].isin(EVAL_DAYS)].reset_index(drop=True)
    test_positives = {p for p in positives if 1 + (p[1] - 1) // 86400 in EVAL_DAYS}
    print(f"train days 8-12: {len(train):,} user-hours, {int(train['label'].sum())} positive; "
          f"evaluation days 13-16: {len(test):,} user-hours, {int(test['label'].sum())} positive")

    results: dict = {
        "experiment": "features-v2 Option A: supervised fusion (docs/experiment_features_v2.md)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_days": list(TRAIN_DAYS),
        "evaluation_days": list(EVAL_DAYS),
        "budget_per_day": args.budget,
        "inputs": INPUTS,
        "train_units": int(len(train)),
        "train_positives": int(train["label"].sum()),
        "evaluation_units": int(len(test)),
        "evaluation_positives": int(test["label"].sum()),
        "tie_break": "fixed random order, seed 0",
        "test_labels_used": False,
        "scorers": {},
    }
    units = test[["user", "day", "label", "machine"]]
    scores = {"gru_max_event": test["gru_max_event"].to_numpy(np.float64), "rule_either": test["either"].to_numpy(np.float64)}
    x_train, x_test = model_inputs(train), model_inputs(test)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    for name, build in MODELS.items():
        started = time.time()
        model = build().fit(x_train, y_train)
        scores[name] = model.predict_proba(x_test)[:, 1]
        info: dict = {
            "fit_seconds": round(time.time() - started, 1),
            "train_average_precision": average_precision(y_train, model.predict_proba(x_train)[:, 1]),
        }
        if name == "logistic_regression":
            info["standardised_coefficients"] = dict(zip(INPUTS, model[-1].coef_[0].round(4).tolist()))
        else:
            info["n_iter"] = int(model.n_iter_)
        perm = permutation_importance(model, x_test, y_test, scoring="average_precision", n_repeats=5, random_state=0)
        info["permutation_importance_ap"] = {
            feature: {"mean": float(m), "std": float(s)}
            for feature, m, s in sorted(zip(INPUTS, perm.importances_mean, perm.importances_std), key=lambda t: -t[1])
        }
        results.setdefault("models", {})[name] = info
        print(f"{name}: fitted and scored ({time.time() - started:.0f}s)")

    for name, values in scores.items():
        results["scorers"][name] = _evaluate.evaluate(units, values, test_positives, args.budget)

    gru = results["scorers"]["gru_max_event"]["user_hour"]
    gru_tp = gru["at_budget"]["tp"]
    needed = max(2 * gru_tp, 2)
    results["success_rule"] = {
        "rule": "model TP at budget on days 13-16 >= 2x GRU TP (at least 2 if the GRU catches 0) and model AP > GRU AP",
        "gru_tp": gru_tp,
        "tp_needed": needed,
        "gru_average_precision": gru["average_precision"],
        "met": {
            name: bool(results["scorers"][name]["user_hour"]["at_budget"]["tp"] >= needed
                       and results["scorers"][name]["user_hour"]["average_precision"] > gru["average_precision"])
            for name in MODELS
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_pos = len(test_positives)
    print(f"\n{'scorer':<24} {'AP':>8} {'ROC':>6} {'TP@budget':>10} {'machine/human':>14} {'day AP':>8} {'day TP@b':>9}")
    for name, metrics in results["scorers"].items():
        hour, day = metrics["user_hour"], metrics["user_day"]
        b = hour["at_budget"]
        print(
            f"{name:<24} {hour['average_precision']:>8.5f} {hour['roc_auc']:>6.3f} {b['tp']:>4}/{n_pos:<5} "
            f"{b['alerts_machine_accounts']:>6}/{b['alerts_human_accounts']:<7} {day['average_precision']:>8.5f} "
            f"{day['at_budget']['tp']:>4}/{day['positive_units_scored']}"
        )
    for name, info in results["models"].items():
        top = list(info["permutation_importance_ap"].items())[:5]
        print(f"{name}: train AP {info['train_average_precision']:.4f}; top permutation importance "
              + ", ".join(f"{f} {v['mean']:+.4f}" for f, v in top))
    print(f"Success rule (TP >= {needed} and AP > {gru['average_precision']:.5f}): {results['success_rule']['met']}")
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
