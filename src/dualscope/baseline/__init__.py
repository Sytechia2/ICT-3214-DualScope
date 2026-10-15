"""Baseline intrusion detection models (Work Package 9.2)."""

from dualscope.baseline.features import (
    BASELINE_FEATURE_NAMES,
    DEFAULT_FAIL_ID,
    EVENT_FEATURE_COLUMNS,
    aggregate_events_to_user_hours,
    aggregate_user_hours_table,
    build_user_hour_feature_matrix,
    fail_id_from_preprocessing,
)
from dualscope.baseline.isolation_forest import (
    IsolationForestBaseline,
    BaselineScoreRow,
    BASELINE_SCORE_SCHEMA,
)

__all__ = [
    "BASELINE_FEATURE_NAMES",
    "DEFAULT_FAIL_ID",
    "EVENT_FEATURE_COLUMNS",
    "aggregate_events_to_user_hours",
    "aggregate_user_hours_table",
    "fail_id_from_preprocessing",
    "build_user_hour_feature_matrix",
    "IsolationForestBaseline",
    "BaselineScoreRow",
    "BASELINE_SCORE_SCHEMA",
]
