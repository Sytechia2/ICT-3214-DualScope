"""Baseline intrusion detection models (Work Package 9.2)."""

from dualscope.baseline.features import (
    BASELINE_FEATURE_NAMES,
    aggregate_events_to_user_hours,
    build_user_hour_feature_matrix,
)
from dualscope.baseline.isolation_forest import (
    IsolationForestBaseline,
    BaselineScoreRow,
    BASELINE_SCORE_SCHEMA,
)

__all__ = [
    "BASELINE_FEATURE_NAMES",
    "aggregate_events_to_user_hours",
    "build_user_hour_feature_matrix",
    "IsolationForestBaseline",
    "BaselineScoreRow",
    "BASELINE_SCORE_SCHEMA",
]
