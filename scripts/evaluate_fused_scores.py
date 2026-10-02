"""Evaluate fused anomaly scores against ground-truth redteam labels.

Computes Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and confusion counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dualscope.sequence.calibration import detection_metrics, average_precision, best_f1_threshold


def evaluate_fused_scores(
    fused_path: Path,
    labels_dir: Path,
    threshold: float | None = None,
) -> dict:
    """Evaluate fused score parquet against red-team labels."""
    print(f"Loading fused scores from {fused_path}...")
    fused_ds = ds.dataset(str(fused_path), format="parquet")
    fused_table = fused_ds.to_table(columns=["user_id", "window_start", "fused_score", "is_fused_alert"])

    users = fused_table["user_id"].to_pylist()
    windows = fused_table["window_start"].to_pylist()
    scores = fused_table["fused_score"].to_numpy()

    # Filter only available scores
    valid_mask = np.isfinite(scores)
    users = [u for u, v in zip(users, valid_mask) if v]
    windows = [w for w, v in zip(windows, valid_mask) if v]
    scores = scores[valid_mask]

    print(f"Loaded {len(scores)} valid user-hours.")

    # Load redteam labels
    print(f"Loading redteam labels from {labels_dir}...")
    labels_ds = ds.dataset(str(labels_dir), format="parquet")
    labels_table = labels_ds.to_table()
    
    redteam_units = set()
    for row in labels_table.to_pylist():
        ts = row["timestamp"]
        hour_start = 1 + ((ts - 1) // 3600) * 3600
        user = row["user"]
        redteam_units.add((user, hour_start))

    # Match ground-truth labels to evaluated user-hours
    y_true = np.array([(u, w) in redteam_units for u, w in zip(users, windows)], dtype=bool)
    num_attacks = int(y_true.sum())
    print(f"Ground-truth attack user-hours in evaluated units: {num_attacks}")

    if num_attacks == 0:
        print("Warning: No red-team attacks present in this specific split/day.")
        return {}

    # Calculate metrics
    ap = average_precision(y_true, scores)
    
    if threshold is not None:
        metrics = detection_metrics(y_true, scores, threshold=threshold)
    else:
        metrics = best_f1_threshold(y_true, scores)

    print("\n" + "=" * 50)
    print("           FUSION EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Alert Threshold   : {metrics['threshold']:.6f}")
    print(f"  Precision         : {metrics['precision'] * 100:.2f}%" if metrics['precision'] else "  Precision: 0.0%")
    print(f"  Recall            : {metrics['recall'] * 100:.2f}%" if metrics['recall'] else "  Recall   : 0.0%")
    print(f"  F1-Score          : {metrics['f1']:.4f}")
    print(f"  Average Precision : {ap:.6f}" if ap is not None else "  Average Precision: N/A")
    print(f"  False Positive Rate: {metrics['false_positive_rate'] * 100:.4f}%" if metrics['false_positive_rate'] else "")
    print("-" * 50)
    print(f"  True Positives (Attacks Caught) : {metrics['tp']} / {num_attacks}")
    print(f"  False Positives (False Alarms)  : {metrics['fp']}")
    print(f"  False Negatives (Attacks Missed): {metrics['fn']}")
    print(f"  True Negatives (Normal Benign)  : {metrics['tn']}")
    print("=" * 50 + "\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fused scores against redteam labels.")
    parser.add_argument("--fused-scores", type=Path, required=True, help="Path to fused parquet file or directory")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/labels"), help="Path to redteam labels directory")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed alert threshold (if omitted, finds best F1 threshold)")
    args = parser.parse_args()

    evaluate_fused_scores(args.fused_scores, args.labels_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
