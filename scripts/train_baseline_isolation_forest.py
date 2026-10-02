"""Task 9.2: Train and evaluate flat-feature Isolation Forest baseline.

Aggregates Task 2.4 features into user-hour vectors, fits Isolation Forest on
training split (Days 2-7), calibrates scores and tunes alert threshold on validation
split (Days 8-16), and saves checkpoint for comparative evaluation (Task 9.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.baseline.features import (
    aggregate_events_to_user_hours,
    build_user_hour_feature_matrix,
)
from dualscope.baseline.isolation_forest import (
    IsolationForestBaseline,
)
from dualscope.sequence.calibration import average_precision, best_f1_threshold


def load_split_units(
    features_dir: Path,
    start_ts: int,
    end_ts: int,
    batch_size: int = 250_000,
) -> list:
    """Stream transformed events within timestamp interval and aggregate to user-hours."""
    dataset = ds.dataset(str(features_dir), format="parquet", partitioning="hive")
    window_filter = (ds.field("timestamp") >= start_ts) & (ds.field("timestamp") < end_ts)
    
    # Read relevant feature columns
    cols = [
        "timestamp",
        "acting_user",
        "source_line",
        "log1p_prior_auth_count_1h_scaled",
        "log1p_prior_failure_count_1h_scaled",
        "log1p_seconds_since_previous_auth_scaled",
        "log1p_prior_unique_destinations_24h_scaled",
        "log1p_prior_user_destination_count_24h_scaled",
        "has_user_history",
        "is_new_user_destination",
        "is_new_host_connection",
        "auth_type_id",
        "logon_type_id",
        "auth_orientation_id",
        "auth_result_id",
    ]

    print(f"Reading events for window [{start_ts}, {end_ts})...")
    batches = dataset.to_batches(filter=window_filter, columns=cols, batch_size=batch_size)
    
    all_units = []
    total_events = 0
    for batch in batches:
        table = pa.Table.from_batches([batch])
        total_events += table.num_rows
        units = aggregate_events_to_user_hours(table, hour_seconds=3600)
        all_units.extend(units)

    print(f"Aggregated {total_events} events into {len(all_units)} user-hour units.")
    return all_units


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


def train_baseline(
    features_dir: Path,
    labels_dir: Path,
    model_output: Path,
    scores_output: Path | None = None,
    n_estimators: int = 100,
    max_train_samples: int = 100_000,
) -> None:
    """Train Isolation Forest on training split and evaluate on validation split."""
    print("=" * 65)
    print("       TASK 9.2: FLAT-FEATURE ISOLATION FOREST BASELINE")
    print("=" * 65)

    # 1. Load Training Data (Days 2-7: 86401 to 604801)
    # Day 1 is warm-up, so Days 2-7 represent clean training period
    t0 = time.time()
    train_units = load_split_units(features_dir, start_ts=86401, end_ts=604801)
    if not train_units:
        raise ValueError("No training units extracted from features directory.")

    X_train = build_user_hour_feature_matrix(train_units)
    if len(X_train) > max_train_samples:
        print(f"Subsampling training matrix from {len(X_train)} to {max_train_samples} samples...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train = X_train[idx]

    # 2. Fit Isolation Forest
    print(f"Fitting Isolation Forest ({n_estimators} estimators, matrix shape {X_train.shape})...")
    baseline = IsolationForestBaseline(n_estimators=n_estimators, random_state=42)
    baseline.fit(X_train)
    print(f"Model fitted in {time.time() - t0:.2f}s.")

    # 3. Load Validation Data (Days 8-16: 604801 to 1382401)
    print("\nLoading Validation split for calibration and threshold tuning (Days 8-16)...")
    val_units = load_split_units(features_dir, start_ts=604801, end_ts=1382401)
    X_val = build_user_hour_feature_matrix(val_units)
    val_raw = baseline.score_raw(X_val)

    # 4. Calibrate Scores
    print("Fitting empirical quantile-tail calibrator on validation raw scores...")
    baseline.calibrate(val_raw)

    # 5. Evaluate against Validation Red-Team Labels
    print(f"Loading validation redteam labels from {labels_dir}...")
    val_positives = load_redteam_positives(labels_dir, start_ts=604801, end_ts=1382401)
    y_val = np.array(
        [(u.user_id, u.window_start) in val_positives for u in val_units],
        dtype=bool,
    )
    num_attacks = int(y_val.sum())
    print(f"Validation attack user-hours: {num_attacks}")

    # Predict calibrated scores
    rows = baseline.predict_units(val_units)
    scores = np.array([r.score for r in rows])
    valid = np.isfinite(scores)

    if num_attacks > 0 and valid.sum() > 0:
        ap = average_precision(y_val[valid], scores[valid]) or 0.0
        best_f1_info = best_f1_threshold(y_val[valid], scores[valid])
        baseline.alert_threshold = float(best_f1_info["threshold"])
    else:
        ap = 0.0
        best_f1_info = {
            "threshold": 0.999,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "tp": 0,
            "fp": int(np.sum(scores[valid] >= 0.999)) if valid.sum() > 0 else 0,
        }
        baseline.alert_threshold = 0.999

    print("\n" + "=" * 55)
    print("      ISOLATION FOREST VALIDATION BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  Optimal Alert Threshold : {best_f1_info['threshold']:.6f}")
    print(f"  Average Precision (PR)  : {ap:.6f}")
    print(f"  Best F1-Score           : {best_f1_info['f1']:.4f}")
    print(f"  Precision at Best F1    : {best_f1_info['precision']*100:.2f}%")
    print(f"  Recall at Best F1       : {best_f1_info['recall']*100:.2f}%")
    print(f"  False Positive Rate     : {best_f1_info['false_positive_rate']*100:.4f}%")
    print(f"  Attacks Caught (TP)     : {best_f1_info['tp']} / {num_attacks}")
    print(f"  False Alarms (FP)       : {best_f1_info['fp']}")
    print("=" * 55)

    # 6. Save Model Checkpoint
    baseline.save(model_output)
    print(f"\nSaved baseline model checkpoint to {model_output}")

    # 7. Export Scores if requested
    if scores_output:
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        # Update alert decisions with optimal threshold
        updated_rows = baseline.predict_units(val_units)
        table = baseline.to_arrow_table(updated_rows)
        pq.write_table(table, scores_output, compression="zstd")
        print(f"Exported validation baseline scores to {scores_output}")


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
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "data/samples/lanl_features_sample/transformed/events",
        help="Path to transformed events parquet directory",
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
        help="Destination parquet file for validation baseline scores",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in Isolation Forest")
    parser.add_argument("--train-start-ts", type=int, default=86401, help="Training window start timestamp")
    parser.add_argument("--train-end-ts", type=int, default=604801, help="Training window end timestamp")
    parser.add_argument("--val-start-ts", type=int, default=604801, help="Validation window start timestamp")
    parser.add_argument("--val-end-ts", type=int, default=1382401, help="Validation window end timestamp")
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
            features_dir=args.features_dir,
            labels_dir=args.labels_dir,
            model_output=args.model_output,
            scores_output=args.scores_output,
            n_estimators=args.n_estimators,
        )


if __name__ == "__main__":
    main()
