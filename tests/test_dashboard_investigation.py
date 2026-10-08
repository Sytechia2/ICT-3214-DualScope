"""Investigation tab (Task 7.3): loader states and the rendered tab."""

from __future__ import annotations

import json
from html import escape
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
    assert investigation.lookup(views, investigation.DIRECT_MODE, OTHER).state == investigation.PENDING
    assert investigation.lookup(views, investigation.RAG_MODE, OTHER).state == investigation.FAILED
    assert investigation.lookup(views, investigation.DIRECT_MODE, INC).verified["outcome"] == "no_supported_mapping"
    assert investigation.lookup(investigation.load_run(run / "missing"), investigation.RAG_MODE, INC).state == investigation.NOT_GENERATED


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
    assert "U1 logged on to a new host." in html
    assert "Observation A" in html and "Interpretation A" in html
    assert "title='It fits, with other explanations still possible.'>medium</span>" in html
    assert 'aria-label="What is Confidence?"' in html and "The AI sets this level itself" in html and '<span class="ds-pop-k">Low:</span>' in html  # heading popover from the glossary
    assert ">01:00:10<" in html and "title='auth.txt:10 · C1 → C2 · Kerberos'" in html  # chip shows the time, title the ref
    assert ">counts<" in html and ">package:counts<" not in html
    assert "◐" in html and "C467 not cited" in html
    assert "https://attack.mitre.org/techniques/T1550/002/" in html and "Pass the Hash" in html
    assert ">Supported<" in html and ">Uncertain<" in html
    assert "Techniques offered to the AI: T1550.002" in html
    assert "What the logs can't show (1)" in [e.label for e in at.expander]
    assert "m-1 · 7 Oct 11:00" in html
    assert "Removed by verification (2)" in [e.label for e in at.expander]
    assert "Valid Accounts" in html and "Dropped claim" in html and "no matching cited event" in html
    assert "Unverified until" not in html


def test_verdict_strip_and_mode_counts(package: Path, run: Path) -> None:
    at = _run_page("incident", package, selected=INC)
    html = _html(at)
    verdict = html[html.index("ds-verdict'>"):]
    assert verdict.index(">Supported<") < verdict.index(">Uncertain<")
    assert "2 removed by verification" in html
    assert "Techniques: AI only 0 · AI + ATT&amp;CK 2 · checked 2" in html
    assert 'aria-label="What is AI modes?"' in html
    at.radio(key="investigation_mode").set_value(investigation.DIRECT_MODE).run()
    html = _html(at)
    assert "ds-verdict-none'>No supported mapping" in html and "Techniques offered to the AI" not in html
    assert "All removed by verification." in html  # the fixture's removed technique is still listed
    verified = _verified("no_supported_mapping")
    assert verified["removed_techniques"]
    verified["removed_techniques"] = []
    _write(run / "verification.jsonl", [{"incident_id": INC, "mode": "direct", "checks": {}, "verified": verified}])
    at = _run_page("incident", package, selected=INC, investigation_mode=investigation.DIRECT_MODE)
    assert "No techniques mapped." in _html(at) and "All removed" not in _html(at)


def test_selecting_an_observation_shows_its_events(package: Path, run: Path) -> None:
    at = _run_page("incident", package, selected=INC)
    assert "ds-obs-events" not in _html(at)
    at.button(key="inv_pick_2").click().run()
    assert not at.exception
    html = _html(at)
    assert "ds-obs-events" in html and "C1 → C9" in html and "new destination for user" in html
    at.button(key="inv_pick_2").click().run()  # again: clear
    assert "ds-obs-events" not in _html(at)


def test_mode_switch_and_no_supported_mapping(package: Path, run: Path) -> None:
    at = _run_page("incident", package, selected=INC)
    at.radio(key="investigation_mode").set_value(investigation.DIRECT_MODE).run()
    assert not at.exception
    html = _html(at)
    assert "No supported mapping" in html and "Pass the Hash</span></a>" not in html and "ds-badge supported" not in html
    at.radio(key="investigation_mode").set_value(investigation.RAG_MODE).run()
    assert "ds-badge supported" in _html(at)


def test_pending_failed_and_missing_folder(package: Path, run: Path, tmp_path: Path, monkeypatch) -> None:
    at = _run_page("incident", package, selected=OTHER)
    assert not at.exception and "The AI summary failed" in _html(at) and "schema invalid" in _html(at)
    at.radio(key="investigation_mode").set_value(investigation.DIRECT_MODE).run()
    assert "No AI summary yet" in _html(at)
    monkeypatch.setattr(investigation, "DEFAULT_RUN", tmp_path / "nowhere")
    at = _run_page("incident", package, selected=INC)
    assert not at.exception and "No AI summary for this data" in _html(at)
    assert [tab.label for tab in at.tabs][-1] == "Investigation"  # other tabs still render
    assert "Possible attacker techniques" in _html(at)


def test_glossary_has_verification_terms() -> None:
    for term in ("Supported", "Uncertain", "Rejected", "Partly supported", "No supported mapping", "Verification"):
        assert term in ui.GLOSSARY
    assert "ds-badge supported" in ui.status_badge("Supported")


# ─── pure logic ───

import pandas as pd

BASE = 90_001  # 01:00:00


def _events() -> pd.DataFrame:
    rows = [("auth.txt:1", BASE), ("auth.txt:2", BASE + 5), ("auth.txt:3", BASE + 50), ("auth.txt:4", BASE + 60),
            ("auth.txt:5", BASE + 70)]
    frame = pd.DataFrame(rows, columns=["source_reference", "timestamp"])
    frame["source_computer"], frame["destination_computer"], frame["authentication_type"] = "C1", "C2", "NTLM"
    return frame.set_index("source_reference", drop=False)


def _obs(text, *refs):
    return {"text": text, "evidence": list(refs), "check": _check()}


def test_event_time_and_clock() -> None:
    events = _events()
    assert investigation.event_time(events, "auth.txt:2") == BASE + 5
    assert investigation.event_time(events, "auth.txt:99") is None
    assert investigation.event_time(events, "package:counts") is None
    assert investigation.event_time(None, "auth.txt:1") is None
    assert investigation.clock(BASE + 5) == "01:00:05"
    assert investigation.clock(1) == "00:00:00"


def test_order_observations_is_chronological_and_splits_context() -> None:
    items = [_obs("late", "auth.txt:4"), _obs("context", "package:counts"), _obs("early", "auth.txt:5", "auth.txt:1"),
             _obs("unknown time", "auth.txt:99"), _obs("none")]
    numbered, context = investigation.order_observations(items, _events())
    assert [(o.number, o.text) for o in numbered] == [(1, "early"), (2, "late"), (3, "unknown time")]
    assert not hasattr(investigation, "number_glyph")
    assert numbered[0].time == BASE and numbered[2].time is None
    assert [c["text"] for c in context] == ["context", "none"]


def test_merge_dots_and_technique_links() -> None:
    items = [_obs("a", "auth.txt:1"), _obs("b", "auth.txt:2"), _obs("c", "auth.txt:3"), _obs("d", "auth.txt:4"),
             _obs("e", "auth.txt:5")]
    numbered, _ = investigation.order_observations(items, _events())
    dots = investigation.merge_dots(numbered)
    assert [d.label for d in dots] == ["1–2", "3–5"]
    assert [d.time for d in dots] == [BASE, BASE + 50]
    assert investigation.merge_dots(numbered, seconds=0)[0].label == "1"
    assert investigation.merge_dots(numbered, seconds=5)[0].numbers == (1, 2)
    techniques = [{"technique_id": "T1", "evidence": ["auth.txt:2"]},
                  {"technique_id": "T2", "evidence": ["auth.txt:9"], "verification": {"cited_events": ["auth.txt:4", "package:x"]}}]
    linked = investigation.link_techniques(dots, techniques)
    assert [d.techniques for d in linked] == [("T1",), ("T2",)]
    assert investigation.dot_for(linked, 5).numbers == (3, 4, 5) and investigation.dot_for(linked, None) is None
    assert len(investigation.truncate("x" * 200)) == 120


def test_evidence_chips() -> None:
    events = _events()
    chips = investigation.evidence_chips(["auth.txt:3", "package:graph_context", "auth.txt:1", "weird", "auth.txt:99"], events)
    assert [(c.label, c.kind) for c in chips] == [("01:00:00", "event"), ("01:00:50", "event"), ("auth.txt:99", "unknown"), ("+2", "more")]
    assert chips[0].title == "auth.txt:1 · C1 → C2 · NTLM" and chips[3].title.startswith("auth.txt:3") is False and "graph_context" in chips[3].title
    short = investigation.evidence_chips(["package:graph_context", "package:counts", "package:detector", "package:incident", "x"], None, limit=9)
    assert [c.label for c in short] == ["graph", "counts", "detector", "incident", "x"]
    assert investigation.evidence_chips([], events) == []


def test_mode_counts_and_technique_order(run: Path) -> None:
    views = investigation.load_run(run)
    counts = investigation.technique_counts(views, INC)
    assert counts == {investigation.VERIFIED_MODE: 2, investigation.RAG_MODE: 2, investigation.DIRECT_MODE: 0}
    assert investigation.technique_counts(views, OTHER)[investigation.VERIFIED_MODE] is None
    assert investigation.counts_line(counts) == "AI only 0 · AI + ATT&CK 2 · checked 2"
    assert "—" in investigation.counts_line({investigation.VERIFIED_MODE: None})
    ordered = investigation.sort_techniques([{"technique_id": "A", "status": "Uncertain"}, {"technique_id": "B", "status": "Supported"}])
    assert [t["technique_id"] for t in ordered] == ["B", "A"]


def test_fit_domain_merge_threshold_and_created() -> None:
    start, end = BASE - 600, BASE + 3000
    assert investigation.fit_domain([BASE, BASE + 1000], start, end) == (BASE - 100, BASE + 1100)  # 10% padding
    assert investigation.fit_domain([BASE, BASE + 100], start, end) == (BASE - 60, BASE + 160)  # at least 60 s
    assert investigation.fit_domain([BASE, BASE + 1000], BASE - 50, BASE + 1050) == (BASE - 50, BASE + 1050)  # clamped to window
    assert investigation.fit_domain([BASE - 900, BASE], BASE, end)[0] == BASE - 900  # a time before the window widens it
    assert investigation.fit_domain([BASE], start, end) == (BASE - 300, BASE + 300)
    assert investigation.merge_seconds(0, 100) == 2 and investigation.merge_seconds(0, 3600) == 45
    assert investigation.format_created("2026-10-07T12:17:41Z") == "7 Oct 12:17"
    assert investigation.format_created("not a date") == "not a date"


def test_pixel_aware_merge() -> None:
    items = [_obs("a", "auth.txt:1"), _obs("b", "auth.txt:2"), _obs("c", "auth.txt:3"), _obs("d", "auth.txt:4"),
             _obs("e", "auth.txt:5")]
    numbered, _ = investigation.order_observations(items, _events())  # BASE, +5, +50, +60, +70
    assert investigation.marker_width("1") == 20 and investigation.marker_width("2–3") == 27.5
    low, high = BASE - 60, BASE + 130  # 190 s over 600 px: 3.2 px per second
    dots = investigation.merge_dots_px(numbered, low, high, 600)
    assert dots[0].numbers == (1, 2) and sorted(n for d in dots for n in d.numbers) == [1, 2, 3, 4, 5]
    # No two markers overlap: their pixel boxes are at least 4 px apart.
    px = [((d.time - low) * 600 / (high - low), investigation.marker_width(d.label)) for d in dots]
    assert all(b[0] - a[0] >= (a[1] + b[1]) / 2 + 4 for a, b in zip(px, px[1:]))
    wide = investigation.merge_dots_px(numbered, BASE - 10_000, BASE + 10_000, 600)
    assert [d.numbers for d in wide] == [(1, 2, 3, 4, 5)]  # a very long range packs everything into one marker
    assert [d.numbers for d in investigation.merge_dots_px(numbered, BASE, BASE + 70, 6000)] == [(1,), (2,), (3,), (4,), (5,)]
