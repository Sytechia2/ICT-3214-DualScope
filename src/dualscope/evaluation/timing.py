"""Task 9.4: Detection timing, lead-lag precedence, and forensic error analysis.

Provides causal timing alignment, first-alert offset calculation, early-warning
categorization, detector precedence tracking, and error case categorization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


class TimingCategory(str, Enum):
    """Categorisation of alert arrival relative to initial attack activity."""

    EARLY_WARNING = "early_warning"        # Alert available before first labelled event (offset < 0)
    IMMEDIATE = "immediate"                # Alert available within first hour of attack (0 <= offset <= 3600)
    DELAYED = "delayed"                    # Alert available in later hours of attack (offset > 3600)
    MISSED = "missed"                      # No alert generated within budget (offset is None)


class DetectorPrecedence(str, Enum):
    """Precedence relationship when multiple detectors alert on a compromise."""

    SEQUENCE_LEADS = "sequence_leads"      # Sequence detector alerted first
    GRAPH_LEADS = "graph_leads"            # Graph detector alerted first
    SIMULTANEOUS = "simultaneous"          # Both detectors alerted in same window
    SEQUENCE_ONLY = "sequence_only"        # Only sequence detector alerted
    GRAPH_ONLY = "graph_only"              # Only graph detector alerted
    NONE = "none"                          # Neither detector alerted


@dataclass(frozen=True)
class AttackCampaignTiming:
    """Detection timing record for a single compromised user on a given day."""

    user_id: str
    dataset_day: int
    first_attack_timestamp: int
    first_attack_hour_start: int
    total_attack_events: int
    attack_timestamps: list[int]

    # Alert availability timestamps (None if missed at budget)
    first_alert_available_ts: int | None
    alert_offset_seconds: int | None
    alert_offset_hours: float | None
    timing_category: TimingCategory

    # Detector precedence
    precedence: DetectorPrecedence
    lead_time_seconds: int | None

    # Scores at first alert
    fused_score: float | None
    seq_score: float | None
    graph_score: float | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timing_category"] = self.timing_category.value
        d["precedence"] = self.precedence.value
        return d


@dataclass(frozen=True)
class ErrorCaseRecord:
    """Forensic case study record for qualitative error inspection."""

    case_type: str                         # "true_positive" | "false_positive" | "false_negative"
    case_title: str
    user_id: str
    dataset_day: int
    window_start: int
    window_end: int
    fused_score: float
    seq_score: float
    graph_score: float
    temporal_boost: float
    is_ground_truth_attack: bool

    # Investigation detail
    root_cause_explanation: str
    operational_implication: str
    evidence_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_first_alert_offset(
    first_attack_timestamp: int,
    first_alert_window_start: int | None,
    hour_seconds: int = 3600,
) -> tuple[int | None, float | None, TimingCategory]:
    """Calculate the first-alert offset between attack event and score availability.

    Parameters
    ----------
    first_attack_timestamp : int
        Exact epoch second of the earliest red-team event for this user.
    first_alert_window_start : int | None
        Start timestamp of the earliest alerting user-hour window.
        Score availability timestamp is window_start + hour_seconds.
    hour_seconds : int
        Duration of scoring window in seconds (default: 3600).

    Returns
    -------
    offset_seconds : int | None
        Offset in seconds (avail_ts - attack_ts). Negative indicates early warning.
    offset_hours : float | None
        Offset in hours.
    category : TimingCategory
        Categorisation of the offset.
    """
    if first_alert_window_start is None:
        return None, None, TimingCategory.MISSED

    avail_ts = first_alert_window_start + hour_seconds
    offset_s = avail_ts - first_attack_timestamp
    offset_h = round(offset_s / hour_seconds, 2)

    if offset_s < 0:
        cat = TimingCategory.EARLY_WARNING
    elif offset_s <= hour_seconds:
        cat = TimingCategory.IMMEDIATE
    else:
        cat = TimingCategory.DELAYED

    return offset_s, offset_h, cat


def determine_detector_precedence(
    first_seq_alert_window: int | None,
    first_graph_alert_window: int | None,
) -> tuple[DetectorPrecedence, int | None]:
    """Determine which detector alerted first and compute lead time."""
    if first_seq_alert_window is None and first_graph_alert_window is None:
        return DetectorPrecedence.NONE, None
    if first_seq_alert_window is not None and first_graph_alert_window is None:
        return DetectorPrecedence.SEQUENCE_ONLY, None
    if first_seq_alert_window is None and first_graph_alert_window is not None:
        return DetectorPrecedence.GRAPH_ONLY, None

    # Both alerted
    seq_ts = first_seq_alert_window  # type: ignore[operator]
    graph_ts = first_graph_alert_window  # type: ignore[operator]

    if seq_ts < graph_ts:
        return DetectorPrecedence.SEQUENCE_LEADS, graph_ts - seq_ts
    if graph_ts < seq_ts:
        return DetectorPrecedence.GRAPH_LEADS, seq_ts - graph_ts
    return DetectorPrecedence.SIMULTANEOUS, 0


def summarize_timing_distributions(
    campaigns: Sequence[AttackCampaignTiming],
) -> dict[str, Any]:
    """Compute aggregate statistical summary of detection timing."""
    detected = [c for c in campaigns if c.timing_category != TimingCategory.MISSED]
    missed = [c for c in campaigns if c.timing_category == TimingCategory.MISSED]

    offsets = [c.alert_offset_hours for c in detected if c.alert_offset_hours is not None]

    category_counts = {
        TimingCategory.EARLY_WARNING.value: sum(1 for c in campaigns if c.timing_category == TimingCategory.EARLY_WARNING),
        TimingCategory.IMMEDIATE.value: sum(1 for c in campaigns if c.timing_category == TimingCategory.IMMEDIATE),
        TimingCategory.DELAYED.value: sum(1 for c in campaigns if c.timing_category == TimingCategory.DELAYED),
        TimingCategory.MISSED.value: len(missed),
    }

    precedence_counts = {
        DetectorPrecedence.SEQUENCE_LEADS.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.SEQUENCE_LEADS),
        DetectorPrecedence.GRAPH_LEADS.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.GRAPH_LEADS),
        DetectorPrecedence.SIMULTANEOUS.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.SIMULTANEOUS),
        DetectorPrecedence.SEQUENCE_ONLY.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.SEQUENCE_ONLY),
        DetectorPrecedence.GRAPH_ONLY.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.GRAPH_ONLY),
        DetectorPrecedence.NONE.value: sum(1 for c in campaigns if c.precedence == DetectorPrecedence.NONE),
    }

    stats: dict[str, Any] = {
        "total_campaigns": len(campaigns),
        "detected_campaigns": len(detected),
        "missed_campaigns": len(missed),
        "campaign_detection_rate_pct": round(len(detected) / max(1, len(campaigns)) * 100, 2),
        "timing_categories": category_counts,
        "precedence_distribution": precedence_counts,
    }

    if offsets:
        stats["offset_hours_stats"] = {
            "mean": round(float(np.mean(offsets)), 2),
            "median": round(float(np.median(offsets)), 2),
            "min": round(float(np.min(offsets)), 2),
            "max": round(float(np.max(offsets)), 2),
            "std": round(float(np.std(offsets)), 2),
        }
    else:
        stats["offset_hours_stats"] = None

    return stats
