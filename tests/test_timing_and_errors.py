"""Tests for Task 9.4: Timing, Precedence, and Error Analysis."""

import pytest

from dualscope.evaluation.timing import (
    AttackCampaignTiming,
    DetectorPrecedence,
    ErrorCaseRecord,
    TimingCategory,
    calculate_first_alert_offset,
    determine_detector_precedence,
    summarize_timing_distributions,
)


def test_calculate_first_alert_offset() -> None:
    """Test offset math and categorical assignment."""
    attack_ts = 1000000

    # 1. Early warning: alert window ends before attack event (e.g. window 995000 -> 998600 < 1000000)
    offset_s, offset_h, cat = calculate_first_alert_offset(attack_ts, 995000)
    assert offset_s == (995000 + 3600) - attack_ts  # 998600 - 1000000 = -1400
    assert cat == TimingCategory.EARLY_WARNING
    assert offset_h < 0

    # 2. Immediate: alert window covers attack event (e.g. window 998000 -> 1001600)
    offset_s, offset_h, cat = calculate_first_alert_offset(attack_ts, 998000)
    assert offset_s == (998000 + 3600) - attack_ts  # 1001600 - 1000000 = 1600 <= 3600
    assert cat == TimingCategory.IMMEDIATE

    # 3. Delayed: alert window occurs several hours later (e.g. window 1010000 -> 1013600)
    offset_s, offset_h, cat = calculate_first_alert_offset(attack_ts, 1010000)
    assert offset_s == (1010000 + 3600) - attack_ts  # 13600 > 3600
    assert cat == TimingCategory.DELAYED

    # 4. Missed: no alert triggered
    offset_s, offset_h, cat = calculate_first_alert_offset(attack_ts, None)
    assert offset_s is None
    assert offset_h is None
    assert cat == TimingCategory.MISSED


def test_determine_detector_precedence() -> None:
    """Test detector lead-lag precedence determination."""
    # Sequence leads
    prec, lead_s = determine_detector_precedence(10000, 20000)
    assert prec == DetectorPrecedence.SEQUENCE_LEADS
    assert lead_s == 10000

    # Graph leads
    prec, lead_s = determine_detector_precedence(25000, 10000)
    assert prec == DetectorPrecedence.GRAPH_LEADS
    assert lead_s == 15000

    # Simultaneous
    prec, lead_s = determine_detector_precedence(10000, 10000)
    assert prec == DetectorPrecedence.SIMULTANEOUS
    assert lead_s == 0

    # Single-detector alerts
    prec, lead_s = determine_detector_precedence(10000, None)
    assert prec == DetectorPrecedence.SEQUENCE_ONLY
    assert lead_s is None

    prec, lead_s = determine_detector_precedence(None, 10000)
    assert prec == DetectorPrecedence.GRAPH_ONLY
    assert lead_s is None

    # Neither
    prec, lead_s = determine_detector_precedence(None, None)
    assert prec == DetectorPrecedence.NONE
    assert lead_s is None


def test_summarize_timing_distributions() -> None:
    """Test summary statistics on campaign timing records."""
    c1 = AttackCampaignTiming(
        user_id="U1",
        dataset_day=13,
        first_attack_timestamp=1000000,
        first_attack_hour_start=999601,
        total_attack_events=5,
        attack_timestamps=[1000000, 1000500],
        first_alert_available_ts=1003201,
        alert_offset_seconds=3201,
        alert_offset_hours=0.89,
        timing_category=TimingCategory.IMMEDIATE,
        precedence=DetectorPrecedence.SEQUENCE_LEADS,
        lead_time_seconds=3600,
        fused_score=0.95,
        seq_score=0.98,
        graph_score=0.90,
    )
    c2 = AttackCampaignTiming(
        user_id="U2",
        dataset_day=13,
        first_attack_timestamp=1000000,
        first_attack_hour_start=999601,
        total_attack_events=1,
        attack_timestamps=[1000000],
        first_alert_available_ts=None,
        alert_offset_seconds=None,
        alert_offset_hours=None,
        timing_category=TimingCategory.MISSED,
        precedence=DetectorPrecedence.NONE,
        lead_time_seconds=None,
        fused_score=None,
        seq_score=None,
        graph_score=None,
    )

    stats = summarize_timing_distributions([c1, c2])
    assert stats["total_campaigns"] == 2
    assert stats["detected_campaigns"] == 1
    assert stats["missed_campaigns"] == 1
    assert stats["campaign_detection_rate_pct"] == 50.0
    assert stats["timing_categories"]["immediate"] == 1
    assert stats["timing_categories"]["missed"] == 1
    assert stats["offset_hours_stats"]["mean"] == 0.89


def test_error_case_record_serialization() -> None:
    """ErrorCaseRecord serializes properly to dictionary."""
    rec = ErrorCaseRecord(
        case_type="false_positive",
        case_title="Administrative fan-out",
        user_id="C395$@DOM1",
        dataset_day=13,
        window_start=1087201,
        window_end=1090801,
        fused_score=0.914,
        seq_score=0.999,
        graph_score=0.999,
        temporal_boost=0.0,
        is_ground_truth_attack=False,
        root_cause_explanation="Machine account performing domain sync",
        operational_implication="Filter machine accounts ($ suffix) from high-priority alert queues",
        evidence_references=["auth.txt:12345"],
    )
    d = rec.to_dict()
    assert d["case_type"] == "false_positive"
    assert d["user_id"] == "C395$@DOM1"
    assert len(d["evidence_references"]) == 1
