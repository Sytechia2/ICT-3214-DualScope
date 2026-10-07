"""Redesigned dashboard pages (Tasks 7.1, 7.2): queue, incident page, evidence lookup, sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dualscope.dashboard.data import load_incidents
from dualscope.dashboard.evidence import load_events
from dualscope.dashboard.queue import behaviour_totals, day_cutoff_counts, summarise_incidents
from dualscope.attack.retrieval import TechniqueRetriever

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "incident_dashboard.py"
DAY2 = 86_401


def _event(line, timestamp, user="U1@DOM1", destination="C2", **fields):
    return {
        "source_reference": f"auth.txt:{line}", "timestamp": timestamp, "source_user": user,
        "destination_user": user, "source_computer": "C1", "destination_computer": destination,
        "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
        "authentication_result": "Success", "is_new_user_destination": False, "is_new_host_connection": False,
        "is_new_user_source": False, **fields,
    }


def _incident(incident_id, user, start, *, priority, rank, tied, references, redteam=False):
    return {
        "incident_id": incident_id, "user_id": user, "start_time": start, "end_time": start + 3600,
        "dataset_day": 2, "priority": priority, "best_rank_in_day": rank, "max_fused_score": 0.99,
        "fusion_method": "supervised_hist_gradient_boosting",
        "detector_scores": {"sequence": {"max_score": 0.9, "any_alert": None}, "graph": {"max_score": 0.4, "any_alert": False}},
        "alert_hours": [{
            "alert_id": f"ALR-D02-{rank:02d}", "window_start": start, "window_end": start + 3600, "rank_in_day": rank,
            "tied_at_cutoff": tied, "fusion_score": 0.99, "gru_max_event": 2.0, "gru_percentile_in_day": 0.95,
            "counts": {"n_events": len(references)}, "ground_truth_redteam": redteam,
        }],
        "graph_context": {"used_by_final_model": False, "new_edge_count": 1, "degree_growth": 0, "top_edges": []},
        "source_references": references, "evidence_count": len(references), "ground_truth_redteam": redteam,
    }


INCIDENTS = [
    _incident("INC-TEST-D02-U1_DOM1-001", "U1@DOM1", DAY2 + 3600, priority="HIGH", rank=1, tied=False,
              references=["auth.txt:10", "auth.txt:11", "auth.txt:12"], redteam=True),
    _incident("INC-TEST-D02-U2_DOM1-001", "U2@DOM1", DAY2 + 7200, priority="MEDIUM", rank=2, tied=True,
              references=["auth.txt:20"]),
]
EVENTS = [
    _event(10, DAY2 + 3610),
    _event(11, DAY2 + 3620, destination="C9", authentication_type="NTLM", is_new_user_destination=True, is_new_host_connection=True),
    _event(12, DAY2 + 3630, authentication_result="Fail"),
    _event(20, DAY2 + 7210, user="U2@DOM1"),
]


@pytest.fixture(autouse=True)
def _restore_main_module():
    """AppTest installs the script as sys.modules["__main__"] and leaves it there; later
    multiprocessing tests would then re-run the dashboard script in their spawned workers."""
    main = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = main


@pytest.fixture()
def package(tmp_path: Path) -> Path:
    pd.DataFrame(EVENTS).to_parquet(tmp_path / "events.parquet", index=False)
    path = tmp_path / "incidents.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in INCIDENTS), encoding="utf-8")
    return path


def _main(path: Path | None = None, source: str = "Other JSONL export") -> AppTest:
    at = AppTest.from_file(str(SCRIPT), default_timeout=60)
    at.session_state["source"] = source
    if path is not None:
        at.session_state["source_path"] = str(path)
    return at.run()


def _page(page: str, path: str, selected: str | None = None, **state) -> None:
    import streamlit as st
    from dualscope.dashboard import app
    st.session_state["_source"] = ("Other JSONL export", app.Path(path))
    if selected:
        st.session_state["selected_incident_id"] = selected
    for key, value in state.items():
        st.session_state.setdefault(key, value)
    {"incident": app.incident_page, "evidence": app.evidence_page}[page]()


def _run_page(page: str, path: Path, **kwargs) -> AppTest:
    return AppTest.from_function(_page, args=(page, str(path)), kwargs=kwargs, default_timeout=60).run()


def _html(at: AppTest) -> str:
    return "\n".join(element.proto.body for element in at.get("html"))


def test_queue_facts_from_package(package: Path) -> None:
    summaries = summarise_incidents(load_incidents(package), load_events(package.parent / "events.parquet"),
                                    TechniqueRetriever.from_file())
    first, second = summaries
    assert first.headline == "NTLM logon to a new host"
    assert first.chips == ["NTLM to new host", "Failed logons", "New host"]
    assert first.top_candidate["technique_id"] == "T1550.002"
    assert second.headline == "Not enough evidence" and second.chips == []
    assert (first.above_cutoff, second.above_cutoff) == (True, False)
    counts = day_cutoff_counts(s.incident for s in summaries)
    assert counts.to_dict("records") == [{"day": 2, "above": 1, "tied": 1}]
    totals = dict(zip(*behaviour_totals(summaries).T.values))
    assert totals["NTLM logon to a new host"] == 1 and totals["Not enough evidence"] == 1


def test_queue_page_splits_at_cut_off_and_opens_side_panel(package: Path) -> None:
    at = _main(package)
    assert not at.exception
    assert [t.value for t in at.title] == ["Alert queue"]
    html = _html(at)
    assert "Day cut-off" in html and "not a live feed" in html
    above, tied = (frame.value for frame in at.dataframe)
    assert list(above["User"]) == ["U1@DOM1"] and list(above["Why flagged"]) == ["NTLM to new host · Failed logons +1"]
    assert list(tied["User"]) == ["U2@DOM1"] and list(tied["Why flagged"]) == ["Not enough evidence"]
    assert "U1@DOM1 — NTLM logon to a new host" in html  # side panel defaults to the first incident
    assert "Red-team" not in html  # answer key hidden by default

    at.text_input(key="f_search").input("C9").run()  # a host in U1's events
    assert [frame.value["User"].tolist() for frame in at.dataframe] == [["U1@DOM1"]]
    at.text_input(key="f_search").input("nobody").run()
    assert any("No incidents on this day match" in item.value for item in at.info)


def test_answer_key_switch_shows_labels(package: Path) -> None:
    at = _main(package)
    at.sidebar.toggle(key="show_answer_key").set_value(True).run()
    assert "Red-team activity" in _html(at)
    assert at.dataframe[0].value["Red-team (answer key)"].tolist() == [True]


def test_fixture_and_missing_sources(tmp_path: Path) -> None:
    at = _main(source="Synthetic fixture")
    assert not at.exception
    assert "Synthetic fixture" in _html(at)
    assert "Event details are not available" in _html(at)

    at = _main(tmp_path / "missing.jsonl")
    assert any("Could not load incidents" in item.value for item in at.error)


def test_incident_page_shows_evidence(package: Path) -> None:
    at = _run_page("incident", package, selected="INC-TEST-D02-U1_DOM1-001")
    assert not at.exception
    assert [t.value for t in at.title] == ["U1@DOM1 — NTLM logon to a new host"]
    assert [tab.label for tab in at.tabs] == ["Overview", "Timeline (3)", "Hosts (2)", "Evidence", "Investigation"]
    html = _html(at)
    assert "Rank 1 of 38" in html and "Above cut-off" in html
    assert "auth.txt:11" in html and "C1 → C9" in html  # why-flagged example event
    assert "not yet verified" in html and "T1550.002" in html


def test_incident_page_deep_link_and_tab(package: Path) -> None:
    at = AppTest.from_file(str(SCRIPT), default_timeout=60)
    at.session_state["source"] = "Other JSONL export"
    at.session_state["source_path"] = str(package)
    at.query_params["incident"] = "INC-TEST-D02-U2_DOM1-001"
    at.run()
    assert at.session_state["selected_incident_id"] == "INC-TEST-D02-U2_DOM1-001"
    assert "U2@DOM1 — Not enough evidence" in _html(at)  # queue side panel follows the link


def test_evidence_page_looks_up_an_event(package: Path) -> None:
    at = _run_page("evidence", package, evidence_lookup="auth.txt:11")
    assert not at.exception
    assert "Source record" in _html(at)
    record = at.dataframe[0].value
    assert dict(zip(record["Field"], record["Value"]))["destination_computer"] == "C9"

    at = _run_page("evidence", package, evidence_lookup="auth.txt:999")
    assert any("not cited by any incident" in item.value for item in at.warning)
