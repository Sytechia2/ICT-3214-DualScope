"""Task 9.2: Train and evaluate the flat-feature Isolation Forest baseline.

Aggregates Task 2.4 transformed event features into 16-dimensional user-hour
vectors one dataset day at a time (a day always holds whole hours), fits an
Isolation Forest on a sample of training user-hours (Days 2-7; Day 1 is
warm-up), and scores the validation days (8-16).

* No labels are used for fitting. Calibration (0-1 rarity) uses the validation
  raw scores without labels.
* The alert threshold is the best-F1 threshold on Days 8-12 only, so Days 13-16
  stay unseen for the comparison matrix (``evaluate_comparison_matrix.py
  --baseline-scores``).
* ``--score-test`` scores Days 17-30 once: scores are saved first, and only then
  are test labels read. It refuses to run if the test scores already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.baseline.features import (
    EVENT_FEATURE_COLUMNS,
    aggregate_user_hours_table,
    build_user_hour_feature_matrix,
    fail_id_from_preprocessing,
)
from dualscope.baseline.isolation_forest import (
    IsolationForestBaseline,
)
from dualscope.sequence.calibration import average_precision, best_f1_threshold

TRAIN_DAYS = range(2, 8)
VALIDATION_DAYS = range(8, 17)
THRESHOLD_DAYS = range(8, 13)
EVALUATION_DAYS = range(13, 17)
TEST_DAYS = range(17, 31)
BUDGET_PER_DAY = 38
SCORE_COLUMNS = ["user_id", "window_start", "dataset_day", "raw_score", "score", "is_alert"]


def load_day_units(dataset: ds.Dataset, day: int, fail_id: int) -> tuple[pd.DataFrame, np.ndarray]:
    """User-hour keys and feature matrix for one whole dataset day."""
    table = dataset.to_table(columns=EVENT_FEATURE_COLUMNS, filter=ds.field("dataset_day") == day)
    keys, matrix = aggregate_user_hours_table(table, fail_id=fail_id)
    keys["dataset_day"] = day
    return keys, matrix


def load_redteam_positives(labels_dir: Path, start_ts: int, end_ts: int) -> set[tuple[str, int]]:
    """Load red-team positive user-hours within time window."""
    dataset = ds.dataset(str(labels_dir), format="parquet", partitioning="hive")
    window_filter = (ds.field("timestamp") >= start_ts) & (ds.field("timestamp") < end_ts)
    table = dataset.to_table(filter=window_filter)

    positives = set()
    for row in table.to_pylist():
        ts = row["timestamp"]
        hour_start = 1 + ((ts - 1) // 3600) * 3600
        user = row["user"]
        positives.add((user, hour_start))
    return positives


def day_range_ts(days: range) -> tuple[int, int]:
    return (days.start - 1) * 86400 + 1, (days.stop - 1) * 86400 + 1


def budget_hits(frame: pd.DataFrame, budget: int) -> int:
    """Labelled hours among the top ``budget`` scores of each day (ties in file order)."""
    top = frame.sort_values("score", ascending=False, kind="stable").groupby("dataset_day", sort=False).head(budget)
    return int(top["label"].sum())


def score_days(baseline: IsolationForestBaseline, dataset: ds.Dataset, days: range, fail_id: int) -> pd.DataFrame:
    parts = []
    for day in days:
        started = time.time()
        keys, matrix = load_day_units(dataset, day, fail_id)
        keys["raw_score"] = baseline.score_raw(matrix)
        parts.append(keys)
        print(f"  day {day}: {len(keys):,} user-hours scored ({time.time() - started:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def calibrated(baseline: IsolationForestBaseline, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["score"] = baseline.calibrator.transform(frame["raw_score"].to_numpy())
    frame["is_alert"] = frame["score"] >= baseline.alert_threshold
    return frame


def train_baseline(
    features_root: Path,
    labels_dir: Path,
    model_output: Path,
    scores_output: Path,
    n_estimators: int = 100,
    max_train_samples: int = 100_000,
    test_scores_output: Path | None = None,
) -> dict:
    """Fit on Days 2-7, score and evaluate Days 8-16, optionally score Days 17-30 once."""
    print("=" * 65)
    print("       TASK 9.2: FLAT-FEATURE ISOLATION FOREST BASELINE")
    print("=" * 65)
    if test_scores_output is not None and test_scores_output.exists():
        raise SystemExit(f"{test_scores_output} exists: the test days are scored once")
    preprocessing = json.loads((features_root / "preprocessing.json").read_text(encoding="utf-8"))
    fail_id = fail_id_from_preprocessing(preprocessing)
    dataset = ds.dataset(str(features_root / "transformed" / "events"), format="parquet", partitioning="hive")

    # 1. Training user-hours (Days 2-7), no labels.
    t0 = time.time()
    matrices = []
    for day in TRAIN_DAYS:
        keys, matrix = load_day_units(dataset, day, fail_id)
        matrices.append(matrix)
        print(f"  train day {day}: {len(keys):,} user-hours ({time.time() - t0:.0f}s)")
    X_train = np.vstack(matrices)
    n_train = len(X_train)
    if n_train > max_train_samples:
        rng = np.random.default_rng(42)
        X_train = X_train[rng.choice(n_train, size=max_train_samples, replace=False)]
    print(f"Fitting Isolation Forest ({n_estimators} trees) on {len(X_train):,} of {n_train:,} training user-hours...")
    baseline = IsolationForestBaseline(n_estimators=n_estimators, random_state=42)
    baseline.fit(X_train)

    # 2. Validation scores (Days 8-16); label-free calibration on their raw scores.
    print("Scoring validation days 8-16...")
    val = score_days(baseline, dataset, VALIDATION_DAYS, fail_id)
    baseline.calibrate(val["raw_score"].to_numpy())

    # 3. Labels: threshold from Days 8-12 only; metrics on Days 8-16 and 13-16.
    positives = load_redteam_positives(labels_dir, *day_range_ts(VALIDATION_DAYS))
    val["label"] = [(u, int(w)) in positives for u, w in zip(val["user_id"], val["window_start"])]
    val = calibrated(baseline, val)
    fit_part = val[val["dataset_day"].isin(THRESHOLD_DAYS)]
    threshold = best_f1_threshold(fit_part["label"].to_numpy(), fit_part["score"].to_numpy())
    baseline.alert_threshold = float(threshold["threshold"])
    val = calibrated(baseline, val)
    eval_part = val[val["dataset_day"].isin(EVALUATION_DAYS)]
    metrics = {
        "train_user_hours": n_train,
        "train_sample": len(X_train),
        "validation_user_hours": len(val),
        "validation_positives_scored": int(val["label"].sum()),
        "validation_average_precision_days_8_16": average_precision(val["label"].to_numpy(), val["score"].to_numpy()),
        "alert_threshold_best_f1_days_8_12": baseline.alert_threshold,
        "days_13_16": {
            "user_hours": len(eval_part),
            "positives_scored": int(eval_part["label"].sum()),
            "average_precision": average_precision(eval_part["label"].to_numpy(), eval_part["score"].to_numpy()),
            "tp_at_budget": budget_hits(eval_part, BUDGET_PER_DAY),
            "budget_per_day": BUDGET_PER_DAY,
        },
    }

    baseline.save(model_output)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    val[SCORE_COLUMNS].to_parquet(scores_output, index=False)
    print(f"Saved model to {model_output} and validation scores to {scores_output}")

    # 4. Optional one-time test scoring: scores are written before labels are read.
    if test_scores_output is not None:
        print("Scoring test days 17-30 (once)...")
        test = calibrated(baseline, score_days(baseline, dataset, TEST_DAYS, fail_id))
        test_scores_output.parent.mkdir(parents=True, exist_ok=True)
        test[SCORE_COLUMNS].to_parquet(test_scores_output, index=False)
        print(f"Test scores saved to {test_scores_output}; reading test labels now.")
        test_positives = load_redteam_positives(labels_dir, *day_range_ts(TEST_DAYS))
        test["label"] = [(u, int(w)) in test_positives for u, w in zip(test["user_id"], test["window_start"])]
        metrics["days_17_30"] = {
            "user_hours": len(test),
            "positives_total": len(test_positives),
            "positives_scored": int(test["label"].sum()),
            "average_precision": average_precision(test["label"].to_numpy(), test["score"].to_numpy()),
            "tp_at_budget": budget_hits(test, BUDGET_PER_DAY),
            "budget_per_day": BUDGET_PER_DAY,
        }

    print("\n" + "=" * 65)
    print(f"  Training user-hours (Days 2-7): {n_train:,} (fitted on {len(X_train):,})")
    print(f"  Validation AP (Days 8-16): {metrics['validation_average_precision_days_8_16']:.6f}")
    print(f"  Alert threshold (best F1 on Days 8-12): {baseline.alert_threshold:.6f}")
    d = metrics["days_13_16"]
    print(f"  Days 13-16: AP {d['average_precision']:.6f}, caught at {BUDGET_PER_DAY}/day {d['tp_at_budget']} / {d['positives_scored']}")
    if "days_17_30" in metrics:
        t = metrics["days_17_30"]
        print(f"  Days 17-30: AP {t['average_precision']:.6f}, caught at {BUDGET_PER_DAY}/day {t['tp_at_budget']} / {t['positives_total']}")
    print("=" * 65)
    metrics_path = scores_output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    return metrics


def generate_synthetic_demo_units() -> tuple[list, list, set[tuple[str, int]]]:
    """Generate synthetic train and validation units for demonstration and testing."""
    from dualscope.baseline.features import UserHourUnit
    rng = np.random.default_rng(42)

    # Train units: 500 normal user-hours
    train_units = []
    for i in range(500):
        vec = rng.normal(loc=0.0, scale=1.0, size=16).astype(np.float32)
        train_units.append(
            UserHourUnit(
                user_id=f"U_TRAIN_{i % 50}@DOM1",
                window_start=86401 + (i % 24) * 3600,
                window_end=86401 + (i % 24 + 1) * 3600,
                source_lines=[i * 10, i * 10 + 1],
                feature_vector=vec,
            )
        )

    # Validation units: 800 normal + 20 attack user-hours
    val_units = []
    val_positives = set()
    for i in range(800):
        vec = rng.normal(loc=0.0, scale=1.0, size=16).astype(np.float32)
        uid = f"U_NORM_{i % 80}@DOM1"
        w_start = 604801 + (i % 72) * 3600
        val_units.append(
            UserHourUnit(
                user_id=uid,
                window_start=w_start,
                window_end=w_start + 3600,
                source_lines=[10000 + i * 5],
                feature_vector=vec,
            )
        )

    for j in range(20):
        vec = rng.normal(loc=4.5, scale=1.2, size=16).astype(np.float32)
        uid = f"U_REDTEAM_{j}@DOM1"
        w_start = 604801 + (j * 3) * 3600
        val_units.append(
            UserHourUnit(
                user_id=uid,
                window_start=w_start,
                window_end=w_start + 3600,
                source_lines=[20000 + j * 5],
                feature_vector=vec,
            )
        )
        val_positives.add((uid, w_start))

    return train_units, val_units, val_positives


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Isolation Forest baseline.")
    parser.add_argument(
        "--features-root",
        type=Path,
        default=REPO_ROOT / "data/processed/lanl_features_v2_days_01_30",
        help="Task 2.4 feature build (with preprocessing.json and transformed/events)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels",
        help="Path to redteam labels directory",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=REPO_ROOT / "models/baseline/isolation_forest",
        help="Destination directory for trained model checkpoint",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=REPO_ROOT / "outputs/baseline_scores/val_baseline.parquet",
        help="Validation (Days 8-16) scores; metrics are written next to it as .json",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in Isolation Forest")
    parser.add_argument("--max-train-samples", type=int, default=100_000, help="Training user-hours sampled for fitting")
    parser.add_argument(
        "--score-test",
        type=Path,
        default=None,
        metavar="TEST_SCORES_OUTPUT",
        help="Also score Days 17-30 once and write their scores here (refuses if the file exists)",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run training and evaluation using synthetic demonstration data",
    )
    args = parser.parse_args()

    if args.synthetic_demo:
        print("=" * 65)
        print("   TASK 9.2: FLAT-FEATURE ISOLATION FOREST BASELINE (DEMO)")
        print("=" * 65)
        train_units, val_units, val_positives = generate_synthetic_demo_units()

        X_train = build_user_hour_feature_matrix(train_units)
        baseline = IsolationForestBaseline(n_estimators=args.n_estimators, random_state=42)
        baseline.fit(X_train)

        X_val = build_user_hour_feature_matrix(val_units)
        val_raw = baseline.score_raw(X_val)
        baseline.calibrate(val_raw)

        y_val = np.array([(u.user_id, u.window_start) in val_positives for u in val_units], dtype=bool)
        rows = baseline.predict_units(val_units)
        scores = np.array([r.score for r in rows])
        valid = np.isfinite(scores)

        ap = average_precision(y_val[valid], scores[valid]) or 0.0
        best_f1_info = best_f1_threshold(y_val[valid], scores[valid])
        baseline.alert_threshold = float(best_f1_info["threshold"])

        print("\n" + "=" * 55)
        print("      ISOLATION FOREST VALIDATION BENCHMARK RESULTS")
        print("=" * 55)
        print(f"  Optimal Alert Threshold : {best_f1_info['threshold']:.6f}")
        print(f"  Average Precision (PR)  : {ap:.6f}")
        print(f"  Best F1-Score           : {best_f1_info['f1']:.4f}")
        print(f"  Precision at Best F1    : {best_f1_info['precision']*100:.2f}%")
        print(f"  Recall at Best F1       : {best_f1_info['recall']*100:.2f}%")
        print(f"  False Positive Rate     : {best_f1_info['false_positive_rate']*100:.4f}%")
        print(f"  Attacks Caught (TP)     : {best_f1_info['tp']} / {int(y_val.sum())}")
        print(f"  False Alarms (FP)       : {best_f1_info['fp']}")
        print("=" * 55)

        baseline.save(args.model_output)
        print(f"\nSaved baseline model checkpoint to {args.model_output}")

        if args.scores_output:
            args.scores_output.parent.mkdir(parents=True, exist_ok=True)
            updated_rows = baseline.predict_units(val_units)
            table = baseline.to_arrow_table(updated_rows)
            pq.write_table(table, args.scores_output, compression="zstd")
            print(f"Exported validation baseline scores to {args.scores_output}")
    else:
        train_baseline(
            features_root=args.features_root,
            labels_dir=args.labels_dir,
            model_output=args.model_output,
            scores_output=args.scores_output,
            n_estimators=args.n_estimators,
            max_train_samples=args.max_train_samples,
            test_scores_output=args.score_test,
        )


if __name__ == "__main__":
    main()
