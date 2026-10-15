"""Supervised classifiers over graph-derived user-day features."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

FEATURE_COLUMNS = (
    "baseline_score", "burst_score", "n_edges", "new_edge_count",
    "prior_degree", "current_degree", "degree_growth",
    "peak_auth_count_60s", "peak_auth_count_300s",
    "peak_unique_destinations_300s", "source_count", "new_source_count",
    "new_source_auth_count", "max_auth_per_source", "min_source_popularity",
    "mean_source_popularity", "prior_source_degree",
    "new_host_connection_events", "destination_count",
)


def feature_matrix(frame) -> np.ndarray:
    """Log-scale count features while preserving the two calibrated scores."""
    matrix = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    matrix[:, 2:] = np.log1p(np.maximum(matrix[:, 2:], 0.0))
    if not np.isfinite(matrix).all():
        raise ValueError("supervised graph features must all be finite")
    return matrix


def best_f1_threshold(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-30)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def classification_report(labels, scores, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=float)
    alerts = scores >= threshold
    tp=int((alerts & labels).sum()); fp=int((alerts & ~labels).sum())
    fn=int((~alerts & labels).sum()); tn=int((~alerts & ~labels).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"threshold":float(threshold),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "average_precision":float(average_precision_score(labels,scores)),
            "roc_auc":float(roc_auc_score(labels,scores))}
