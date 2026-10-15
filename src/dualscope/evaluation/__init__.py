"""DualScope evaluation package (Work Package 9.0)."""

from dualscope.evaluation.timing import (
    AttackCampaignTiming,
    DetectorPrecedence,
    ErrorCaseRecord,
    TimingCategory,
    calculate_first_alert_offset,
    determine_detector_precedence,
    summarize_timing_distributions,
)

__all__ = [
    "AttackCampaignTiming",
    "DetectorPrecedence",
    "ErrorCaseRecord",
    "TimingCategory",
    "calculate_first_alert_offset",
    "determine_detector_precedence",
    "summarize_timing_distributions",
]
