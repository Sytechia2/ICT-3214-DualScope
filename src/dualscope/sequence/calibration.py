"""Score normalisation, validation labels and threshold selection (Task 3.3).

Normalisation maps a raw user-hour reconstruction error to ``[0, 1)`` with a
frozen, strictly monotonic function fitted on label-free validation user-hour
scores:

* Inside the reference range, the score is the empirical quantile of
  ``log(raw)`` among reference user-hours (piecewise-linear between 1,000
  quantile knots), so 0.9 means "higher than about 90% of validation
  user-hours".
* Below the lowest knot the score is clipped to 0.
* Above the 99.9th-percentile knot the survival function continues as a
  generalized-Pareto tail (shape 1) in ``log(raw)``:
  ``S(x) = 0.001 / (1 + (x - x_top) / scale)``. Its initial slope matches an
  exponential decay fitted to the 99th-99.9th percentile spacing, but it
  decays polynomially, so scores keep increasing with raw error, never reach 1,
  and extreme hours remain rankable in float64.

The score is a relative-rarity scale, not an attack probability. Test data is
never used to fit it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score

from dualscope.splits import SplitConfig, aggregate_labels_to_user_hours, deduplicate_labels


_LOG_EPSILON = 1e-12


@dataclass(frozen=True)
class QuantileTailCalibrator:
    """Frozen raw-error to 0-1 score transformation."""

    knots_log_raw: tuple[float, ...]
    knots_probability: tuple[float, ...]
    tail_start_log_raw: float
    tail_start_probability: float
    tail_scale: float
    reference_count: int
    reference_description: str = ""

    @classmethod
    def fit(
        cls,
        raw_scores: np.ndarray,
        n_knots: int = 1000,
        tail_probability: float = 0.999,
        tail_anchor_probability: float = 0.99,
        reference_description: str = "",
        min_reference: int = 100,
    ) -> QuantileTailCalibrator:
        raw = np.asarray(raw_scores, dtype=np.float64)
        raw = raw[np.isfinite(raw)]
        if len(raw) < min_reference:
            raise ValueError(f"at least {min_reference} finite reference scores are required")
        x = np.log(np.maximum(raw, 0.0) + _LOG_EPSILON)
        probs = np.linspace(0.0, tail_probability, n_knots + 1)
        knots = np.quantile(x, probs)
        # Collapse repeated knot values, keeping the highest probability (right-continuous ECDF).
        unique_x, last_index = np.unique(knots[::-1], return_index=True)
        unique_p = probs[::-1][last_index]
        anchor = float(np.quantile(x, tail_anchor_probability))
        top = float(unique_x[-1])
        scale = (top - anchor) / math.log((1 - tail_anchor_probability) / (1 - tail_probability))
        if not scale > 0:
            scale = max(float(np.std(x)), 1e-6) / 10
        return cls(
            knots_log_raw=tuple(float(v) for v in unique_x),
            knots_probability=tuple(float(v) for v in unique_p),
            tail_start_log_raw=top,
            tail_start_probability=float(unique_p[-1]),
            tail_scale=float(scale),
            reference_count=int(len(raw)),
            reference_description=reference_description,
        )

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_scores, dtype=np.float64)
        out = np.full(raw.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(raw)
        x = np.log(np.maximum(raw[finite], 0.0) + _LOG_EPSILON)
        knots_x = np.asarray(self.knots_log_raw)
        knots_p = np.asarray(self.knots_probability)
        body = np.interp(x, knots_x, knots_p, left=0.0)
        above = x > self.tail_start_log_raw
        survival = (1.0 - self.tail_start_probability) / (
            1.0 + (x[above] - self.tail_start_log_raw) / self.tail_scale
        )
        body[above] = 1.0 - survival
        out[finite] = body
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "log_raw_empirical_quantile_with_generalized_pareto_tail",
            "tail_formula": "score = 1 - (1 - tail_start_probability) / (1 + (log(raw + log_epsilon) - tail_start_log_raw) / tail_scale)",
            "knots_log_raw": list(self.knots_log_raw),
            "knots_probability": list(self.knots_probability),
            "tail_start_log_raw": self.tail_start_log_raw,
            "tail_start_probability": self.tail_start_probability,
            "tail_scale": self.tail_scale,
            "reference_count": self.reference_count,
            "reference_description": self.reference_description,
            "log_epsilon": _LOG_EPSILON,
            "clipping": "scores below the lowest reference knot are clipped to 0; the tail approaches but never reaches 1",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantileTailCalibrator:
        return cls(
            knots_log_raw=tuple(float(v) for v in data["knots_log_raw"]),
            knots_probability=tuple(float(v) for v in data["knots_probability"]),
            tail_start_log_raw=float(data["tail_start_log_raw"]),
            tail_start_probability=float(data["tail_start_probability"]),
            tail_scale=float(data["tail_scale"]),
            reference_count=int(data["reference_count"]),
            reference_description=str(data.get("reference_description", "")),
        )


def load_positive_user_hours(
    labels_dir: str | Path,
    split_cfg: SplitConfig,
    split_name: str,
    hour_seconds: int = 3_600,
    allow_test: bool = False,
) -> set[tuple[str, int]]:
    """Red-team (source user, hour_start) positives for one split, for evaluation only.

    Test labels are refused unless ``allow_test`` is explicitly set, so model
    selection and calibration code cannot read them by accident.
    """
    if split_name == "test" and not allow_test:
        raise PermissionError("test labels are reserved for frozen final evaluation")
    split = split_cfg.get_split(split_name)
    dataset = ds.dataset(str(labels_dir), format="parquet", partitioning="hive")
    table = dataset.to_table(
        filter=(ds.field("timestamp") >= split.timestamp_start) & (ds.field("timestamp") < split.timestamp_end)
    )
    return aggregate_labels_to_user_hours(deduplicate_labels(table), hour_seconds=hour_seconds)


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion counts and rates for ``score >= threshold`` alerts."""
    labels = np.asarray(labels, dtype=bool)
    alerts = np.asarray(scores) >= threshold
    tp = int(np.sum(alerts & labels))
    fp = int(np.sum(alerts & ~labels))
    fn = int(np.sum(~alerts & labels))
    tn = int(np.sum(~alerts & ~labels))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Average precision (step-wise PR-AUC, scikit-learn definition)."""
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Threshold maximising F1 for ``score >= threshold``; ties keep the higher threshold."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        raise ValueError("threshold selection needs at least one positive")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    last_of_value = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp, fp, cut_scores = tp[last_of_value], fp[last_of_value], sorted_scores[last_of_value]
    positives = labels.sum()
    precision = tp / (tp + fp)
    recall = tp / positives
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return detection_metrics(labels, scores, float(cut_scores[best]))


def unit_labels(users: np.ndarray, hours: np.ndarray, positives: set[tuple[str, int]]) -> np.ndarray:
    """Boolean label per scored ``(user, hour_start)`` unit."""
    return np.fromiter(
        ((str(u), int(h)) in positives for u, h in zip(users, hours)), dtype=bool, count=len(users)
    )


def evaluation_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    positive_units_total: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    """AP and threshold metrics on scorable user-hours, plus positive coverage.

    ``positive_units_total`` counts every labelled unit in the split, including
    those without a scored user-hour, so coverage and full recall are explicit.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    covered = int(labels.sum())
    summary: dict[str, Any] = {
        "units_scored": int(len(scores)),
        "positive_units_total": int(positive_units_total),
        "positive_units_scored": covered,
        "positive_coverage": covered / positive_units_total if positive_units_total else None,
        "prevalence": covered / len(scores) if len(scores) else None,
        "average_precision": average_precision(labels, scores),
    }
    chosen = best_f1_threshold(labels, scores) if threshold is None else detection_metrics(labels, scores, threshold)
    summary["at_threshold"] = chosen
    if positive_units_total:
        summary["at_threshold"]["recall_including_unscored_positives"] = chosen["tp"] / positive_units_total
    return summary
