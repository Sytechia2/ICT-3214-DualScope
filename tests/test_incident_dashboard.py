"""Task 7.1 incident loading and queue filtering (the redesigned pages are tested in test_dashboard_app.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dualscope.dashboard.data import (
    DEFAULT_FIXTURE,
    IncidentDataError,
    filter_sort_incidents,
    format_dataset_second,
    load_incidents,
)



def test_fixture_has_full_incident_records_and_unavailable_detector() -> None:
    incidents = load_incidents(DEFAULT_FIXTURE)
    assert len(incidents) == 4
    assert {item.priority for item in incidents} == {"CRITICAL", "HIGH", "MEDIUM"}
    assert {item.user_id for item in incidents} == {"U1@DOM1", "U2@DOM1", "U3@DOM1"}
    assert incidents[1].graph.max_score is None
    assert incidents[1].graph.any_alert is False
    assert {"duration_seconds", "duration_hours", "dataset_day", "split", "concordance",
            "mean_fused_score", "fusion_method", "temporal_context", "source_references",
            "evidence_chunk_references", "graph_evidence_nodes", "evidence_count"} <= set(incidents[0].raw)
    assert format_dataset_second(86401) == "Day 2 00:00:00"
    assert format_dataset_second(172801) == "Day 3 00:00:00"


def test_loader_empty_and_malformed_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert load_incidents(empty) == []

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"incident_id":\n', encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 1: invalid JSON"):
        load_incidents(malformed)

    invalid = tmp_path / "invalid.jsonl"
    record = dict(load_incidents(DEFAULT_FIXTURE)[0].raw)
    record["detector_scores"] = {"sequence": {"max_score": None, "any_alert": "yes"}}
    invalid.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 1: detector_scores.sequence.any_alert"):
        load_incidents(invalid)

    with pytest.raises(IncidentDataError, match="cannot read"):
        load_incidents(tmp_path / "missing.jsonl")


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    record = load_incidents(DEFAULT_FIXTURE)[0].raw
    path = tmp_path / "duplicates.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 2: duplicate incident_id"):
        load_incidents(path)


def test_queue_filtering_and_sorting() -> None:
    incidents = load_incidents(DEFAULT_FIXTURE)
    assert [item.priority for item in filter_sort_incidents(incidents)] == [
        "CRITICAL", "HIGH", "HIGH", "MEDIUM"
    ]
    assert [item.start_time for item in filter_sort_incidents(incidents, sort_by="Newest first")] == [
        259201, 172801, 108001, 86401
    ]
    assert [item.user_id for item in filter_sort_incidents(incidents, user_query="u1@dom1")] == [
        "U1@DOM1", "U1@DOM1"
    ]
    assert [item.user_id for item in filter_sort_incidents(
        incidents, priorities=["MEDIUM"], detector_filter="Graph alert"
    )] == ["U3@DOM1"]
    assert [item.incident_id for item in filter_sort_incidents(
        incidents, start_time=93601, end_time=108002
    )] == ["INC-VALIDATION-D02-U2_DOM1-001"]
    assert filter_sort_incidents(incidents, user_query="missing") == []
