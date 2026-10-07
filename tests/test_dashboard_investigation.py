"""Investigation tab (Task 7.3): loader states and the rendered tab."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dualscope.dashboard import investigation, ui
from test_dashboard_app import (  # noqa: F401  (fixtures and helpers)
    _html, _restore_main_module, _run_page, package,
)

INC = "INC-TEST-D02-U1_DOM1-001"
OTHER = "INC-TEST-D02-U2_DOM1-001"


def _check(status="supported", reasons=()):
    return {"status": status, "reasons": list(reasons), "broken_references": []}


def _technique(technique_id, name, status, reasons=("cited events show a thing",)):
    return {"technique_id": technique_id, "name": name, "rationale": f"why {technique_id}", "evidence": ["auth.txt:11"],
            "confidence": "low", "status": status,
            "verification": {"status": status, "reasons": list(reasons), "cited_events": ["auth.txt:11"]}}


def _verified(outcome="supported_mapping"):
    return {
        "summary": "U1 logged on to a new host.", "summary_check": _check(), "uncertainty": ["No process data."],
        "model_no_supported_mapping": False,
        "observations": [
            {"text": "Observation A", "evidence": ["auth.txt:10"], "check": _check()},
            {"text": "Observation B", "evidence": ["auth.txt:11"], "check": _check("partly_supported", ["C467 not cited"])},
        ],
        "interpretations": [{"text": "Interpretation A", "evidence": ["package:counts"], "confidence": "medium", "check": _check()}],
        "removed_observations": [{"text": "Dropped claim", "evidence": [], "check": _check("unsupported", ["no valid citation"])}],
        "removed_interpretations": [],
        "techniques": [_technique("T1550.002", "Pass the Hash", "Supported"), _technique("T1021", "Remote Services", "Uncertain")]
        if outcome != "no_supported_mapping" else [],
        "removed_techniques": [_technique("T1078", "Valid Accounts", "Rejected", ("no matching cited event",))],
        "outcome": outcome,
    }


def _write(path: Path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _record(incident_id, mode, verified=None, status="ok", errors=()):
    return {"incident_id": incident_id, "mode": mode, "status": status, "errors": list(errors), "model": "m",
            "model_version": "m-1", "created_utc": "2026-10-07T11:00:00Z", "retrieved_candidates": ["T1550.002"],
            "verified": verified}


@pytest.fixture()
def run(tmp_path: Path, monkeypatch) -> Path:
    folder = tmp_path / "run"
    folder.mkdir()
    _write(folder / "rag_verified.jsonl", [
        _record(INC, "rag_verified", _verified()),
        _record(OTHER, "rag_verified", status="failed", errors=["schema invalid"]),
    ])
    _write(folder / "generated_rag.jsonl", [_record(INC, "rag"), _record(OTHER, "rag", status="failed", errors=["schema invalid"])])
    _write(folder / "generated_direct.jsonl", [_record(INC, "direct")])
    _write(folder / "verification.jsonl", [
        {"incident_id": INC, "mode": "rag", "checks": {}, "verified": _verified()},
        {"incident_id": INC, "mode": "direct", "checks": {}, "verified": _verified("no_supported_mapping")},
    ])
    monkeypatch.setattr(investigation, "DEFAULT_RUN", folder)
    return folder


def test_loader_states(run: Path) -> None:
    views = investigation.load_run(run)
    ok = investigation.lookup(views, investigation.VERIFIED_MODE, INC)
    assert ok.state == investigation.OK and ok.model_version == "m-1" and ok.retrieved_candidates == ("T1550.002",)
    failed = investigation.lookup(views, investigation.VERIFIED_MODE, OTHER)
    assert failed.state == investigation.FAILED and failed.errors == ("schema invalid",) and failed.verified is None
    assert investigation.lookup(views, investigation.VERIFIED_MODE, "INC-NONE").state == investigation.PENDING
    assert investigation.lookup(views, "Direct", OTHER).state == investigation.PENDING
    assert investigation.lookup(views, "RAG", OTHER).state == investigation.FAILED
    assert investigation.lookup(views, "Direct", INC).verified["outcome"] == "no_supported_mapping"
    assert investigation.lookup(investigation.load_run(run / "missing"), "RAG", INC).state == investigation.NOT_GENERATED


def test_loader_tolerates_bad_files(tmp_path: Path) -> None:
    (tmp_path / "rag_verified.jsonl").write_text("not json\n[1]\n{\"incident_id\": \"X\"}\n", encoding="utf-8")
    views = investigation.load_run(tmp_path)
    assert investigation.lookup(views, investigation.VERIFIED_MODE, "X").state == investigation.PENDING
    assert investigation.read_jsonl(tmp_path / "nope.jsonl") == []
    assert investigation.technique_url("T1550.002") == "https://attack.mitre.org/techniques/T1550/002/"


def test_tab_shows_verified_investigation(package: Path, run: Path) -> None:
    at = _run_page("incident", package, selected=INC)
    assert not at.exception
    html = _html(at)
    assert "U1 logged on to a new host." in [m.value for m in at.markdown]
    assert "Observation A" in html and "Interpretation A" in html
    assert "medium confidence" in html and "auth.txt:10" in html and "package:counts" in html
    assert "partly supported" in html and "C467 not cited" in html
    assert "https://attack.mitre.org/techniques/T1550/002/" in html and "Pass the Hash" in html
    assert ">Supported<" in html and ">Uncertain<" in html
    assert "No process data." in html and "Retrieved candidates: T1550.002" in html
    assert "m-1 · 2026-10-07T11:00:00Z" in html
    assert "Removed by verification (2)" in [e.label for e in at.expander]
    assert "Valid Accounts" in html and "Dropped claim" in html and "no matching cited event" in html
    assert "Unverified until" not in html


def test_mode_switch_and_no_supported_mapping(package: Path, run: Path) -> None:
    at = _run_page("incident", package, selected=INC)
    at.radio(key="investigation_mode").set_value("Direct").run()
    assert not at.exception
    html = _html(at)
    assert "No supported mapping" in html and "Pass the Hash</span></a>" not in html and "ds-badge supported" not in html
    at.radio(key="investigation_mode").set_value("RAG").run()
    assert "ds-badge supported" in _html(at)


def test_pending_failed_and_missing_folder(package: Path, run: Path, tmp_path: Path, monkeypatch) -> None:
    at = _run_page("incident", package, selected=OTHER)
    assert not at.exception and "Generation failed" in _html(at) and "schema invalid" in _html(at)
    at.radio(key="investigation_mode").set_value("Direct").run()
    assert "Pending" in _html(at)
    monkeypatch.setattr(investigation, "DEFAULT_RUN", tmp_path / "nowhere")
    at = _run_page("incident", package, selected=INC)
    assert not at.exception and "Not generated" in _html(at)
    assert [tab.label for tab in at.tabs][-1] == "Investigation"  # other tabs still render
    assert "Possible techniques (unverified)" in _html(at)


def test_glossary_has_verification_terms() -> None:
    for term in ("Supported", "Uncertain", "Rejected", "Partly supported", "No supported mapping", "Verification"):
        assert term in ui.GLOSSARY
    assert "ds-badge supported" in ui.status_badge("Supported")
