"""Tests for incident clustering and packaging (Task 5.4)."""

from __future__ import annotations

import json
import pytest

from dualscope.fusion.config import (
    DisagreementType,
    FusionMethod,
    IncidentConfig,
    IncidentPriority,
)
from dualscope.fusion.incident import (
    INCIDENT_RECORD_SCHEMA,
    IncidentClusterer,
    IncidentRecord,
)
from dualscope.fusion.temporal import TemporalFusedScoreRow


def _make_fused_row(
    user_id: str,
    window_start: int,
    fused_score: float = 0.9992,
    is_fused_alert: bool = True,
    disagreement: DisagreementType = DisagreementType.SEQ_ONLY_ALERT,
    temporal_boost: float = 0.0,
    source_lines: list[int] | None = None,
    evidence_nodes: list[str] | None = None,
) -> TemporalFusedScoreRow:
    return TemporalFusedScoreRow(
        user_id=user_id,
        window_start=window_start,
        window_end=window_start + 3600,
        dataset_day=2,
        split="validation",
        base_fused_score=fused_score,
        temporal_boost=temporal_boost,
        fused_score=fused_score,
        is_fused_alert=is_fused_alert,
        fusion_method=FusionMethod.TEMPORAL,
        disagreement_type=disagreement,
        lead_detector="sequence" if temporal_boost > 0 else None,
        lead_time_seconds=3600 if temporal_boost > 0 else None,
        seq_score=fused_score,
        is_seq_alert=is_fused_alert,
        graph_score=fused_score if disagreement == DisagreementType.CONCORDANT_ALERT else None,
        is_graph_alert=is_fused_alert if disagreement == DisagreementType.CONCORDANT_ALERT else False,
        alignment_status="both_available" if disagreement == DisagreementType.CONCORDANT_ALERT else "sequence_only",
        seq_source_lines=source_lines or [100, 101],
        seq_evidence_chunk=(0, 1),
        graph_evidence_nodes=evidence_nodes or ["C10"],
    )


@pytest.fixture
def clusterer() -> IncidentClusterer:
    cfg = IncidentConfig(
        max_merge_gap_seconds=7200,  # 2 hours
        critical_threshold=0.9995,
        high_threshold=0.9990,
        medium_threshold=0.9900,
    )
    return IncidentClusterer(cfg)


def test_single_hour_incident(clusterer: IncidentClusterer) -> None:
    """A single alerting hour creates an incident with stable ID and metadata."""
    row = _make_fused_row("U123@DOM1", 86401, fused_score=0.9992, source_lines=[50, 51])
    incidents = clusterer.cluster_incidents([row])

    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.incident_id == "INC-VALIDATION-D02-U123_DOM1-001"
    assert inc.user_id == "U123@DOM1"
    assert inc.start_time == 86401
    assert inc.end_time == 90001
    assert inc.duration_hours == 1
    assert inc.priority == IncidentPriority.HIGH
    assert inc.source_references == ["auth.txt:50", "auth.txt:51"]
    assert inc.evidence_chunk_references == ["auth.txt:50"]
    assert inc.evidence_count == 2


def test_consecutive_hours_clustered(clusterer: IncidentClusterer) -> None:
    """Consecutive alerting hours for the same user merge into one multi-hour envelope."""
    # Hour 1: [86401, 90001), Hour 2: [90001, 93601)
    r1 = _make_fused_row("U1@DOM1", 86401, fused_score=0.9991, source_lines=[10, 11])
    r2 = _make_fused_row("U1@DOM1", 90001, fused_score=0.9996, source_lines=[12, 13])

    incidents = clusterer.cluster_incidents([r1, r2])
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.start_time == 86401
    assert inc.end_time == 93601
    assert inc.duration_hours == 2
    # r2 has score 0.9996 >= critical_threshold (0.9995) -> priority is CRITICAL
    assert inc.priority == IncidentPriority.CRITICAL
    assert inc.max_fused_score == 0.9996
    # Source references are aggregated in order
    assert inc.source_references == ["auth.txt:10", "auth.txt:11", "auth.txt:12", "auth.txt:13"]
    assert inc.evidence_count == 4


def test_distinct_clusters_separated_by_gap(clusterer: IncidentClusterer) -> None:
    """Alerts separated by more than max_merge_gap_seconds (2h) form distinct incidents."""
    # Hour 1: [86401, 90001).
    # Hour 2 is 5 hours later: [108001, 111601) -> gap = 18000s > 7200s.
    r1 = _make_row = _make_fused_row("U1@DOM1", 86401)
    r2 = _make_row = _make_fused_row("U1@DOM1", 108001)

    incidents = clusterer.cluster_incidents([r1, r2])
    assert len(incidents) == 2
    assert incidents[0].incident_id.endswith("-001")
    assert incidents[1].incident_id.endswith("-002")
    assert incidents[0].start_time == 86401
    assert incidents[1].start_time == 108001


def test_concordant_alert_is_critical(clusterer: IncidentClusterer) -> None:
    """When both sequence and graph detectors alert, priority is automatically CRITICAL."""
    row = _make_fused_row(
        "U1@DOM1",
        86401,
        fused_score=0.991,  # below critical threshold, but concordant
        disagreement=DisagreementType.CONCORDANT_ALERT,
    )
    incidents = clusterer.cluster_incidents([row])
    assert incidents[0].priority == IncidentPriority.CRITICAL
    assert incidents[0].concordance == "concordant_alert"


def test_incident_serialization(clusterer: IncidentClusterer) -> None:
    """Verify JSON export and Arrow Table export schemas."""
    row = _make_fused_row("U1@DOM1", 86401)
    incidents = clusterer.cluster_incidents([row])

    # 1. JSON serialization
    json_str = incidents[0].to_json()
    parsed = json.loads(json_str)
    assert parsed["incident_id"] == incidents[0].incident_id
    assert parsed["priority"] == "HIGH"
    assert "detector_scores" in parsed

    # 2. PyArrow Table serialization
    table = clusterer.to_arrow_table(incidents)
    assert table.num_rows == 1
    assert table.schema == INCIDENT_RECORD_SCHEMA
    assert table["incident_id"][0].as_py() == incidents[0].incident_id
    assert table["priority"][0].as_py() == "HIGH"
