"""Task 7.2 evidence panels: event timeline, user-host relationships and graph context."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dualscope.dashboard.evidence import (
    EvidenceDataError,
    alert_hours_table,
    event_flags,
    host_pair_table,
    incident_events,
    load_events,
    timeline_table,
)

DAY2 = 86_401


def _event(line, timestamp, source="C1", destination="C2", **fields):
    return {
        "source_reference": f"auth.txt:{line}", "timestamp": timestamp, "source_user": "U1@DOM1",
        "destination_user": "U1@DOM1", "source_computer": source, "destination_computer": destination,
        "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
        "authentication_result": "Success", "is_new_user_destination": False, "is_new_host_connection": False,
        "is_new_user_source": False, **fields,
    }


EVENTS = [
    _event(12, DAY2 + 30, destination="C9", authentication_type="NTLM", is_new_user_destination=True, is_new_host_connection=True),
    _event(10, DAY2 + 10),
    _event(11, DAY2 + 20, authentication_result="Fail"),
    _event(13, DAY2 + 40, source="C9", destination="C9", authentication_orientation="LogOff", is_new_user_source=True),
    _event(99, DAY2 + 50, destination="C5"),  # another incident's event
]

INCIDENT = {
    "incident_id": "INC-TEST-D02-U1_DOM1-001", "user_id": "U1@DOM1", "start_time": DAY2, "end_time": DAY2 + 3600,
    "priority": "HIGH", "max_fused_score": 0.99, "fusion_method": "supervised_hist_gradient_boosting",
    "detector_scores": {"sequence": {"max_score": 0.97, "any_alert": None}, "graph": {"max_score": 0.5, "any_alert": False}},
    "alert_hours": [{
        "alert_id": "ALR-D02-01", "window_start": DAY2, "window_end": DAY2 + 3600, "rank_in_day": 1,
        "tied_at_cutoff": False, "fusion_score": 0.99, "gru_max_event": 2.5, "gru_percentile_in_day": 0.97,
        "counts": {"n_events": 4}, "ground_truth_redteam": True,
    }],
    "graph_context": {"used_by_final_model": False, "new_edge_count": 1, "degree_growth": 2, "top_edges": [{
        "dataset_day": 2, "destination_computer": "C9", "edge_raw_score": 0.6, "is_new_edge": True,
        "success_count": 1, "failure_count": 0, "source_references": ["auth.txt:12"], "references_truncated": False,
    }]},
    "source_references": ["auth.txt:12", "auth.txt:10", "auth.txt:11", "auth.txt:13", "auth.txt:404"],
    "evidence_count": 5, "ground_truth_redteam": True,
}


@pytest.fixture()
def package(tmp_path: Path) -> Path:
    pd.DataFrame(EVENTS).to_parquet(tmp_path / "events.parquet", index=False)
    path = tmp_path / "incidents.jsonl"
    path.write_text(json.dumps(INCIDENT) + "\n", encoding="utf-8")
    return path


def test_incident_events_are_time_ordered_and_report_missing(package: Path) -> None:
    rows, missing = incident_events(load_events(package.parent / "events.parquet"), INCIDENT)
    assert list(rows["source_reference"]) == ["auth.txt:10", "auth.txt:11", "auth.txt:12", "auth.txt:13"]
    assert missing == ["auth.txt:404"]
    table = timeline_table(rows)
    assert list(table["Event ID"]) == list(rows["source_reference"])
    assert table["Time"].iloc[0] == "Day 2 00:00:10"


def test_flags_ignore_log_off_novelty(package: Path) -> None:
    rows, _ = incident_events(load_events(package.parent / "events.parquet"), INCIDENT)
    assert list(event_flags(rows)) == ["", "failed", "new destination for user, new host pair, NTLM", ""]


def test_host_pairs_put_new_relationships_first(package: Path) -> None:
    rows, _ = incident_events(load_events(package.parent / "events.parquet"), INCIDENT)
    pairs = host_pair_table(rows)
    assert list(zip(pairs["Source"], pairs["Destination"])) == [("C1", "C9"), ("C1", "C2"), ("C9", "C9")]
    first = pairs.iloc[0]
    assert (first["Events"], first["New host pair"], first["First event"]) == (1, True, "auth.txt:12")
    assert pairs.iloc[1]["Failures"] == 1
    assert not pairs.iloc[2]["New source for user"]  # log-off only
    assert host_pair_table(rows.iloc[0:0]).empty


def test_alert_hours_hide_answer_key_unless_asked() -> None:
    assert "Red-team (answer key)" not in alert_hours_table(INCIDENT).columns
    table = alert_hours_table(INCIDENT, show_answer_key=True)
    assert table.iloc[0]["Red-team (answer key)"]
    assert table.iloc[0]["Fusion score"] == 0.99 and table.iloc[0]["Rank in day"] == 1


def test_load_events_rejects_bad_files(tmp_path: Path) -> None:
    pd.DataFrame([{"source_reference": "auth.txt:1"}]).to_parquet(tmp_path / "partial.parquet")
    with pytest.raises(EvidenceDataError, match="missing columns"):
        load_events(tmp_path / "partial.parquet")
    pd.DataFrame([EVENTS[0], EVENTS[0]]).to_parquet(tmp_path / "dupes.parquet")
    with pytest.raises(EvidenceDataError, match="repeated"):
        load_events(tmp_path / "dupes.parquet")
    with pytest.raises(EvidenceDataError, match="cannot read"):
        load_events(tmp_path / "missing.parquet")
