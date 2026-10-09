"""Interface and failure-handling tests (Task 8.2).

Each test records the expected outcome of one failure case from
``docs/interface_failure_tests.md``: malformed records, schema problems,
missing detector scores, missing graph context, empty retrieval, model
failures, timestamp and window conversions, evidence links, stage isolation
in the runner and the test-day guard. Everything here is synthetic or uses
the tracked pipeline subset, and the model is always a fake. Cases that
earlier tasks already test are listed in the document instead of repeated.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import requests

from dualscope.attack.queries import observed_behaviours, retrieve_candidates
from dualscope.attack.retrieval import DEFAULT_CATALOG, TechniqueRetriever
from dualscope.dashboard.data import IncidentDataError, format_dataset_second, load_incidents
from dualscope.dashboard.queue import NO_EVIDENCE, behaviour_totals, summarise_incidents
from dualscope.fusion.hourly import COUNTS, hourly_counts
from dualscope.handoff.alerts import (
    COUNT_COLUMNS,
    GRAPH_COLUMNS,
    alert_events,
    build_incident_records,
    group_incidents,
    select_daily_alerts,
)
from dualscope.ingestion.lanl import IngestionConfig, dataset_day, ingest_authentication, ingest_authentication_parallel
from dualscope.investigation import gemini
from dualscope.investigation.gemini import GeminiClient, GeminiConfig, Generation, GenerationError
from dualscope.investigation.generate import investigate
from dualscope.investigation.package import AnswerKeyLeak, assert_no_answer_key, build_package, dataset_time
from dualscope.investigation.prompts import SYSTEM_PROMPT, user_prompt
from dualscope.investigation.verify import REJECTED, apply_verification, verify_response
from dualscope.pipeline import runner, scoring
from dualscope.pipeline.alerts import build_alert_package
from dualscope.pipeline.runner import (
    ConfigError, Context, Options, Outcome, PeakMemory, PipelineState, Stage, StageFailed,
    final_report, run_command, run_pipeline,
)
from dualscope.pipeline.scoring import ScoringError, score_days
from dualscope.sequence.builder import hour_start_array
from dualscope.sequence.calibration import load_positive_user_hours
from dualscope.splits import aggregate_labels_to_user_hours, dataset_hour_start
from scripts import run_investigations as ri
from test_pipeline_scoring import _scores, _write_events

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET = REPO_ROOT / "data" / "samples" / "lanl_pipeline_subset"
SAMPLE_RUN = REPO_ROOT / "outputs" / "pipeline" / "sample"
DAY = 86_400
GOOD = "{t},U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success"


def _source(tmp_path: Path, lines: list[str], name: str = "auth.txt") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rejections(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "rejections.jsonl").read_text(encoding="utf-8").splitlines()]


# ─── Case 1: malformed raw records ─────────────────────────────────────


def test_malformed_raw_records_are_rejected_with_line_numbers_and_counts_reconcile(tmp_path):
    source = _source(tmp_path, [
        GOOD.format(t=5),                                         # 1 ok
        "6,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn",         # 2 eight fields
        "abc,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",   # 3 text timestamp
        "0,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",     # 4 zero timestamp
        "-7,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",    # 5 negative timestamp
        "8,U1@DOM1,,C1,C2,Kerberos,Network,LogOn,Success",            # 6 empty field
        "1.5,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",   # 7 fractional timestamp
        ",U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",      # 8 empty timestamp
        "2,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn",             # 9 malformed and earlier than line 1
        GOOD.format(t=10),                                        # 10 ok
    ])
    summary = ingest_authentication(source, tmp_path / "out", IngestionConfig(chunk_rows=3))
    rejected = _rejections(tmp_path / "out")
    assert [(r["source_line"], r["reason"]) for r in rejected] == [
        (2, "field_count"), (3, "invalid_timestamp"), (4, "invalid_timestamp"), (5, "invalid_timestamp"),
        (6, "empty_field"), (7, "invalid_timestamp"), (8, "invalid_timestamp"), (9, "field_count"),
    ]
    assert all(r["raw_record"] for r in rejected) and rejected[0]["raw_record"].startswith("6,U1")
    counts = summary["counts"]
    assert (counts["input_rows_scanned"], counts["accepted_rows"], counts["rejected_rows"]) == (10, 2, 8)
    assert counts["reconciled"] and summary["rejection_reasons"] == {"empty_field": 1, "field_count": 2, "invalid_timestamp": 5}
    rows = pq.read_table(tmp_path / "out" / "events").to_pylist()
    assert sorted(row["source_line"] for row in rows) == [1, 10]


def test_timestamp_going_backwards_stops_the_run_and_leaves_no_summary(tmp_path):
    source = _source(tmp_path, [GOOD.format(t=5), GOOD.format(t=9), GOOD.format(t=7), GOOD.format(t=11)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"order inversion at auth\.txt:3: 7 < 9"):
        ingest_authentication(source, out, IngestionConfig(chunk_rows=2))
    # summary.json is written last, so a stopped run cannot be mistaken for a finished one.
    assert not (out / "summary.json").exists()
    with pytest.raises(FileExistsError):
        ingest_authentication(source, out, IngestionConfig())
    source.write_text("\n".join(GOOD.format(t=t) for t in (5, 7, 9, 11)) + "\n", encoding="utf-8")
    assert ingest_authentication(source, out, IngestionConfig(overwrite=True))["counts"]["accepted_rows"] == 4


def test_parallel_ingestion_rejects_malformed_rows_with_line_numbers_too(tmp_path):
    day_one = [GOOD.format(t=5) + "\n", "6,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn\n",
               "7,U1@DOM1,,C1,C2,Kerberos,Network,LogOn,Success\n",
               "x,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n"]
    day_two = [GOOD.format(t=DAY + 1) + "\n"]
    source = tmp_path / "auth.txt"
    source.write_text("".join(day_one + day_two), encoding="utf-8", newline="")
    inspection = tmp_path / "inspection.json"
    inspection.write_text(json.dumps({"authentication_stream_inspection": {
        "row_counts_per_dataset_day": {"1": 4, "2": 1},
        "byte_counts_per_dataset_day": {"1": len("".join(day_one).encode()), "2": len("".join(day_two).encode())}}}), encoding="utf-8")
    summary = ingest_authentication_parallel(source, tmp_path / "out", IngestionConfig(day_end=2), inspection, workers=2)
    counts = summary["counts"]
    assert (counts["input_rows_scanned"], counts["accepted_rows"], counts["rejected_rows"]) == (5, 2, 3) and counts["reconciled"]
    found = [json.loads(line) for part in (tmp_path / "out").rglob("*.jsonl") for line in part.read_text().splitlines()]
    assert {(r["source_line"], r["reason"]) for r in found} == {(2, "field_count"), (3, "empty_field"), (4, "invalid_timestamp")}


@pytest.mark.parametrize("content, message", [
    ("1\nx\n", "non-integer"),
    ("1\n1\n", "strictly increasing"),
    ("0\n5\n", "strictly increasing positive"),
])
def test_bad_line_maps_are_refused_before_any_output(tmp_path, content, message):
    source = _source(tmp_path, [GOOD.format(t=5), GOOD.format(t=6)])
    mapping = tmp_path / "map.txt"
    mapping.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        ingest_authentication(source, tmp_path / "out", IngestionConfig(line_map=mapping))
    assert not (tmp_path / "out" / "summary.json").exists()


# ─── Case 2: schema problems ───────────────────────────────────────────


def _package(tmp_path: Path, name: str = "pkg", scores: pd.DataFrame | None = None, **kwargs) -> Path:
    scores = _scores() if scores is None else scores
    features = tmp_path / "features"
    if not features.exists():
        _write_events(features, scores)
    out = tmp_path / name
    build_alert_package(scores, features, out, budget=10, split_name="train", id_prefix="INC-LLM", log=lambda _: None, **kwargs)
    return out


def test_bad_incident_lines_name_the_line_and_field(tmp_path):
    package = _package(tmp_path)
    lines = (package / "incidents.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    del record["priority"]
    broken = tmp_path / "broken.jsonl"
    broken.write_text("\n".join([lines[0], json.dumps(record)]) + "\n", encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 2: priority must be one of"):
        load_incidents(broken)
    del record["incident_id"]
    broken.write_text("\n".join([lines[0], json.dumps(record)]) + "\n", encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 2: incident_id must be a non-empty string"):
        load_incidents(broken)
    broken.write_text(lines[0] + "\n{not json\n", encoding="utf-8")
    with pytest.raises(IncidentDataError, match="line 2: invalid JSON"):
        load_incidents(broken)


def test_investigation_inputs_with_bad_files_fail_with_the_file_named(tmp_path):
    package = _package(tmp_path)
    bad = tmp_path / "bad_json"
    bad.mkdir()
    for name in ("alerts.parquet", "events.parquet"):
        (bad / name).write_bytes((package / name).read_bytes())
    (bad / "incidents.jsonl").write_text(
        (package / "incidents.jsonl").read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"incidents\.jsonl.*line 5"):
        ri.load_inputs(bad, None, None)

    missing = tmp_path / "missing_column"
    missing.mkdir()
    for name in ("alerts.parquet", "incidents.jsonl"):
        (missing / name).write_bytes((package / name).read_bytes())
    pd.read_parquet(package / "events.parquet").drop(columns=["authentication_result"]).to_parquet(missing / "events.parquet")
    with pytest.raises(SystemExit, match=r"events\.parquet.*authentication_result"):
        ri.load_inputs(missing, None, None)

    (missing / "incidents.jsonl").write_text('{"user_id": "U1"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match=r"incidents\.jsonl.*line 1.*incident_id"):
        ri.load_inputs(missing, None, None)


class _FakeBuild:
    def __init__(self, sha: str):
        self.preprocessing_path = Path("features_other/preprocessing.json")
        self._sha = sha

    def feature_info(self):
        return {"preprocessing_sha256": self._sha}


class _FakeRelease:
    preprocessing_sha256 = "a" * 64

    def __init__(self):
        self.gru_model = type("Model", (), {"to": lambda self, device: None})()
        self.fusion_model = type("Fusion", (), {"predict_proba": lambda self, x: np.column_stack([1 - x[:, 0] / 10, x[:, 0] / 10])})()


def test_a_build_with_a_different_preprocessing_file_is_refused_by_name(monkeypatch, tmp_path):
    monkeypatch.setattr(scoring, "load_inputs", lambda *a, **k: _FakeBuild("b" * 64))
    with pytest.raises(ScoringError, match=r"preprocessing\.json differs from the release"):
        scoring.load_build(tmp_path, _FakeRelease())
    called = []
    monkeypatch.setattr(scoring, "gru_scores", lambda *a, **k: called.append("gru"))
    with pytest.raises(ScoringError, match="differs from the release"):
        score_days(tmp_path, [9], _FakeRelease(), log=lambda _: None)
    assert called == []  # nothing was scored with the wrong inputs


def test_alert_package_refusals_leave_no_half_written_folder(tmp_path):
    scores = _scores()
    _write_events(tmp_path / "features", scores)
    out = tmp_path / "pkg"
    with pytest.raises(ValueError, match="splits_config is needed"):
        build_alert_package(scores, tmp_path / "features", out, budget=3, split_name="train", id_prefix="X", labels_dir=tmp_path)
    graph = tmp_path / "graph" / "dataset_day=1"
    graph.mkdir(parents=True)
    pq.write_table(pa.table({"user_id": ["U1@DOM1"], "score": [0.5]}), graph / "part.parquet")
    with pytest.raises(Exception, match="status|dataset_day|top_edges|column"):
        build_alert_package(scores, tmp_path / "features", out, budget=3, split_name="train", id_prefix="X",
                            graph_scores=tmp_path / "graph", log=lambda _: None)
    assert not out.exists()  # the graph file is read before anything is written


# ─── Case 3: missing detector scores ───────────────────────────────────


def _unit_frames(unscored_hours: int):
    users = ["U1@DOM1", "U2@DOM1"]
    keys = pd.DataFrame({"user": users * 3, "hour": np.repeat([1 + 7 * DAY, 3601 + 7 * DAY, 7201 + 7 * DAY], 2), "day": 8})
    gru = keys.assign(gru_max_event=np.linspace(1.0, 6.0, len(keys)))
    extra = pd.DataFrame({"user": ["U1@DOM1"] * unscored_hours, "hour": 1 + 7 * DAY + 3600 * (10 + np.arange(unscored_hours)), "day": 8})
    counts = pd.concat([keys, extra], ignore_index=True)
    for column in COUNTS[:1]:
        counts[column] = 2
    for column in COUNTS[1:]:
        counts[column] = 1
    counts["either"] = 0
    counts["is_machine_account"] = False
    return gru, counts


def test_user_hours_with_events_but_no_gru_score_are_counted_not_hidden(monkeypatch, tmp_path):
    gru, counts = _unit_frames(unscored_hours=3)
    monkeypatch.setattr(scoring, "load_build", lambda *a, **k: object())
    monkeypatch.setattr(scoring, "gru_scores", lambda *a, **k: gru)
    monkeypatch.setattr(scoring, "count_days", lambda *a, **k: counts)
    lines: list[str] = []
    frame = score_days(tmp_path, [8], _FakeRelease(), log=lines.append)
    assert len(frame) == len(gru) == 6  # unscorable hours are not in the queue
    assert frame.attrs["unscored_user_hours"] == {"8": 3}
    assert any("3 user-hours with events have no GRU score" in line for line in lines)


def test_a_gru_score_without_events_stops_scoring(monkeypatch, tmp_path):
    gru, counts = _unit_frames(unscored_hours=0)
    monkeypatch.setattr(scoring, "load_build", lambda *a, **k: object())
    monkeypatch.setattr(scoring, "gru_scores", lambda *a, **k: gru)
    monkeypatch.setattr(scoring, "count_days", lambda *a, **k: counts.iloc[:-1])
    with pytest.raises(ScoringError, match="no events"):
        score_days(tmp_path, [8], _FakeRelease(), log=lambda _: None)


def test_score_frames_with_missing_columns_or_scores_are_refused_by_name(tmp_path):
    scores = _scores()
    _write_events(tmp_path / "features", scores)
    out = tmp_path / "pkg"
    with pytest.raises(ValueError, match="missing columns: gru_max_event"):
        build_alert_package(scores.drop(columns=["gru_max_event"]), tmp_path / "features", out, budget=3,
                            split_name="train", id_prefix="X", log=lambda _: None)
    holes = scores.copy()
    holes.loc[2, "fusion"] = np.nan
    with pytest.raises(ValueError, match="1 user-hours have no fusion score"):
        build_alert_package(holes, tmp_path / "features", out, budget=3, split_name="train", id_prefix="X", log=lambda _: None)
    assert not out.exists()


def test_events_that_do_not_match_the_scored_count_refuse_the_package(tmp_path):
    scores = _scores()
    _write_events(tmp_path / "features", scores)
    path = next((tmp_path / "features" / "raw" / "events" / "dataset_day=1").glob("*.parquet"))
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)  # one stored event is gone
    out = tmp_path / "pkg"
    with pytest.raises(ValueError, match="n_events"):
        build_alert_package(scores, tmp_path / "features", out, budget=10, split_name="train", id_prefix="X", log=lambda _: None)
    assert not out.exists()


# ─── Case 4: stale or missing graph context ────────────────────────────


def _graph_rows(day: int, user: str = "U1@DOM1") -> pd.DataFrame:
    edge = {"destination_computer": "C9", "edge_raw_score": 0.5, "is_new_edge": True, "success_count": 1,
            "failure_count": 0, "source_lines": [101], "references_truncated": False}
    return pd.DataFrame({
        "user_id": [user], "dataset_day": [day], "status": ["available"], "score": [0.8], "is_alert": [True],
        "alert_threshold": [0.7], "n_edges": [3], "new_edge_count": [2], "prior_degree": [1], "current_degree": [3],
        "degree_growth": [2], "evidence_nodes": [["C9"]], "top_edges": [[edge]],
    })[GRAPH_COLUMNS]


def _records(graph):
    scores = _scores()
    alerts = select_daily_alerts(scores, budget=10)
    alerts["incident_id"] = group_incidents(alerts)
    alerts["ground_truth_redteam"] = False
    events = pd.DataFrame({"alert_id": alerts["alert_id"], "timestamp": alerts["window_start"] + 5,
                           "source_reference": [f"auth.txt:{i}" for i in range(len(alerts))]})
    return {r["incident_id"]: r for r in build_incident_records(alerts, events, graph, split="validation")}


def test_graph_context_is_empty_when_there_is_no_graph_or_it_is_for_another_day():
    with_graph = _records(_graph_rows(day=1))
    assert with_graph["INC-TEST-D01-U1_DOM1-001"]["graph_context"]["new_edge_count"] == 2  # control: a matching row is used
    for graph in (None, _graph_rows(day=5), _graph_rows(day=1, user="U9@DOM1")):
        records = _records(graph)
        assert len(records) == 4  # no incident is lost
        for record in records.values():
            assert record["graph_context"]["new_edge_count"] is None and record["graph_context"]["top_edges"] == []
            assert record["detector_scores"]["graph"] == {"max_score": None, "any_alert": None}
            assert record["graph_evidence_nodes"] == [] and record["source_references"]
    # A day-1 snapshot is not carried over to the day-2 incident of the same user.
    assert with_graph["INC-TEST-D02-U1_DOM1-002"]["graph_context"]["new_edge_count"] is None


def test_packages_without_graph_rows_still_build_and_load_in_the_dashboard(tmp_path):
    graph = tmp_path / "graph" / "dataset_day=5"
    graph.mkdir(parents=True)
    _graph_rows(day=5).drop(columns=["dataset_day"]).to_parquet(graph / "part.parquet")
    package = _package(tmp_path, graph_scores=tmp_path / "graph")
    incidents = [json.loads(line) for line in (package / "incidents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(incidents) == 4 and all(i["graph_context"]["top_edges"] == [] for i in incidents)
    check = runner.check_package(package / "incidents.jsonl", None)
    assert check["incidents"] == 4 and check["unresolved_references"] == 0
    package_json = build_package(incidents[0], pd.read_parquet(package / "events.parquet").head(2))
    assert package_json["graph_context"]["top_edges"] == []  # the model sees a graph section with no edges


# ─── Case 5: empty retrieval ───────────────────────────────────────────


def _plain_event(line, offset=0):
    return {
        "source_reference": f"auth.txt:{line}", "source_line": line, "timestamp": 100 + offset, "acting_user": "U1@DOM1",
        "source_user": "U1@DOM1", "destination_user": "U1@DOM1", "source_computer": "C1", "destination_computer": "C2",
        "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
        "authentication_result": "Success", "is_new_user_source": False, "is_new_host_connection": False,
        "is_new_user_destination": False, "prior_auth_count_1h": 0, "prior_failure_count_1h": 0,
        "prior_unique_destinations_24h": 1,
    }


def _plain_incident():
    return {
        "incident_id": "INC-PLAIN-D01-U1_DOM1-001", "user_id": "U1@DOM1", "dataset_day": 1, "start_time": 1, "end_time": 3601,
        "duration_hours": 1, "priority": "HIGH", "alert_hours": [{
            "alert_id": "ALR-D01-01", "window_start": 1, "window_end": 3601, "fusion_score": 0.9, "rank_in_day": 1,
            "tied_at_cutoff": False, "gru_percentile_in_day": 0.99}],
        "detector_scores": {"graph": {"max_score": None, "any_alert": None}},
        "graph_context": {"new_edge_count": None, "degree_growth": None, "top_edges": []},
    }


def test_evidence_with_no_matching_rule_gives_no_candidates_and_no_invented_technique():
    events = pd.DataFrame([_plain_event(i, i) for i in range(1, 5)])
    retriever = TechniqueRetriever.from_file(DEFAULT_CATALOG)
    assert observed_behaviours(events) == []
    retrieval = retrieve_candidates(events, retriever)
    assert retrieval["insufficient_evidence"] and retrieval["candidates"] == []
    package = build_package(_plain_incident(), events)
    assert package["behaviours"] == []
    prompt = user_prompt("rag", package, retrieval)
    assert "No ATT&CK candidates were retrieved" in prompt and "<attack_candidates>" not in prompt

    # A reply that maps techniques anyway is rejected technique by technique.
    reply = {
        "summary": "U1@DOM1 logged on to C2.",
        "observations": [{"text": "Kerberos logon from C1 to C2.", "evidence": ["auth.txt:1"]}],
        "interpretations": [],
        "techniques": [{"technique_id": tid, "name": "x", "rationale": "x", "evidence": ["auth.txt:1"], "confidence": "high"}
                       for tid in ("T1550.002", "T1110.001", "T9999.999")],
        "no_supported_mapping": False, "uncertainty": [],
    }
    checks = verify_response(reply, package, retriever.catalog, [])
    assert [t["status"] for t in checks["techniques"]] == [REJECTED] * 3
    verified = apply_verification({"response": reply}, checks)
    assert verified["techniques"] == [] and len(verified["removed_techniques"]) == 3
    assert verified["outcome"] == "no_supported_mapping"
    # An honest "no supported mapping" reply passes through unchanged.
    honest = {**reply, "techniques": [], "no_supported_mapping": True}
    kept = apply_verification({"response": honest}, verify_response(honest, package, retriever.catalog, []))
    assert kept["outcome"] == "no_supported_mapping" and kept["model_no_supported_mapping"]


def test_dashboard_shows_incidents_with_no_matching_rule(tmp_path):
    scores = _scores()
    package = _package(tmp_path, scores=scores)  # the synthetic events carry no novelty flags and no failures
    incidents = load_incidents(package / "incidents.jsonl")
    events = pd.read_parquet(package / "events.parquet")
    events.index = pd.Index(events["source_reference"].to_numpy())
    summaries = summarise_incidents(incidents, events, TechniqueRetriever.from_file(DEFAULT_CATALOG))
    assert all(s.headline == NO_EVIDENCE and s.candidates == [] and s.event_count > 0 for s in summaries)
    totals = behaviour_totals(summaries).set_index("behaviour")["incidents"]
    assert totals[NO_EVIDENCE] == len(summaries)
    check = runner.check_package(package / "incidents.jsonl", None)
    assert check["incidents_with_attack_candidates"] == 0 and check["incidents"] == len(summaries)


# ─── Case 6: model failures ────────────────────────────────────────────


def _reply():
    return {
        "summary": "U1@DOM1 logged on.", "observations": [{"text": "A logon.", "evidence": ["package:counts"]}],
        "interpretations": [], "techniques": [], "no_supported_mapping": True, "uncertainty": ["Little data."],
    }


class FakeClient:
    """Stands in for GeminiClient: timeouts, rate limits and bad JSON are chosen per incident ID in the prompt."""

    plan: dict[str, str] = {}
    calls: list[str] = []

    def __init__(self, config):
        self.config = config

    def generate(self, system, user, schema):
        incident = next(i for i in self.plan if i in user) if any(i in user for i in self.plan) else None
        FakeClient.calls.append(incident or "other")
        behaviour = self.plan.get(incident, "ok")
        if behaviour == "timeout":
            raise GenerationError("ReadTimeout: HTTPSConnectionPool read timed out (attempts exhausted)")
        if behaviour == "429":
            raise GenerationError("HTTP 429: quota exceeded")
        if behaviour == "500":
            raise GenerationError("HTTP 500: internal error")
        text = "<html>not json</html>" if behaviour == "bad_json" else json.dumps(_reply())
        return Generation(text, "STOP", "fake-1", {"totalTokenCount": 5}, 0.1, 1)


def _run_script(monkeypatch, handoff: Path, output: Path, plan: dict[str, str], *extra: str):
    FakeClient.plan, FakeClient.calls = plan, []
    monkeypatch.setattr(ri, "GeminiClient", FakeClient)
    monkeypatch.setattr(sys, "argv", ["run_investigations.py", "--handoff-dir", str(handoff), "--output-dir", str(output),
                                       "--workers", "2", *extra])
    ri.main()


def _states(output: Path, mode: str) -> dict[str, str]:
    records = [json.loads(line) for line in (output / f"generated_{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["incident_id"]: r["status"] for r in records}


def test_model_failures_are_isolated_per_incident_and_retry_failed_recovers_them(tmp_path, monkeypatch):
    package = _package(tmp_path)
    output = tmp_path / "investigations"
    plan = {"INC-LLM-D01-U1_DOM1-001": "timeout", "INC-LLM-D01-U2_DOM1-001": "429",
            "INC-LLM-D01-C1_DOM1-001": "bad_json", "INC-LLM-D02-U1_DOM1-002": "ok"}
    before = {name: (package / name).read_bytes() for name in ("alerts.parquet", "events.parquet", "incidents.jsonl", "manifest.json")}
    _run_script(monkeypatch, package, output, plan)
    expected = {"INC-LLM-D01-U1_DOM1-001": "failed", "INC-LLM-D01-U2_DOM1-001": "failed",
                "INC-LLM-D01-C1_DOM1-001": "invalid", "INC-LLM-D02-U1_DOM1-002": "ok"}
    for mode in ("rag", "direct"):
        assert _states(output, mode) == expected
    failed = [json.loads(line) for line in (output / "generated_rag.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {r["incident_id"]: r for r in failed}
    assert "ReadTimeout" in by_id["INC-LLM-D01-U1_DOM1-001"]["errors"][0] and by_id["INC-LLM-D01-U1_DOM1-001"]["response"] is None
    assert by_id["INC-LLM-D01-C1_DOM1-001"]["raw_text"] == "<html>not json</html>"  # the reply is kept for diagnosis
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["incidents"] == 4 and summary["modes"]["rag"]["status"] == {"failed": 2, "invalid": 1, "ok": 1}
    verified = {json.loads(line)["incident_id"]: json.loads(line) for line in (output / "rag_verified.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(verified) == 4 and verified["INC-LLM-D01-U1_DOM1-001"]["verified"] is None
    assert verified["INC-LLM-D02-U1_DOM1-002"]["verified"]["outcome"] == "no_supported_mapping"
    # The detector results are exactly as before.
    assert {name: (package / name).read_bytes() for name in before} == before
    assert runner.check_package(package / "incidents.jsonl", output)["investigation_states"]["AI + ATT&CK, checked"] == {
        "failed": 3, "ok": 1}

    # A plain rerun resumes without calling the model again; the failures stay recorded.
    _run_script(monkeypatch, package, output, {})
    assert FakeClient.calls == [] and _states(output, "rag") == expected
    # --retry-failed asks again for the three that failed, in both modes, and nothing else.
    _run_script(monkeypatch, package, output, {}, "--retry-failed")
    assert len(FakeClient.calls) == 6
    for mode in ("rag", "direct"):
        assert set(_states(output, mode).values()) == {"ok"} and len(_states(output, mode)) == 4
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["modes"]["rag"]["status"] == {"ok": 4}
    assert runner.check_package(package / "incidents.jsonl", output)["investigation_states"]["AI + ATT&CK, checked"] == {"ok": 4}


class _Session:
    def __init__(self, outcomes):
        self.outcomes, self.posts = list(outcomes), 0

    def post(self, url, json=None, timeout=None, headers=None):
        self.posts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Response:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


def _client(monkeypatch, outcomes, attempts=3):
    monkeypatch.setattr(gemini, "_gcloud", lambda *args: "token")
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    session = _Session(outcomes)
    return GeminiClient(GeminiConfig(project="p", max_attempts=attempts), session=session), session


def test_client_turns_every_transport_failure_into_a_generation_error(monkeypatch):
    for outcome, message in (
        (requests.Timeout("read timed out"), "Timeout: read timed out"),
        (requests.ConnectionError("no route"), "ConnectionError: no route"),
        (_Response(500, text="internal"), "HTTP 500"),
        (_Response(503, text="busy"), "HTTP 503"),
        (_Response(200, body=None, text="<html>proxy</html>"), "unreadable"),
        (_Response(200, body=["not", "a", "dict"]), "unreadable"),
    ):
        client, session = _client(monkeypatch, [outcome] * 3)
        with pytest.raises(GenerationError, match=message):
            client.generate("s", "u", {})
        assert session.posts == 3  # transient failures are retried before giving up


def test_investigate_records_a_timeout_and_an_unreadable_reply_as_failed(monkeypatch):
    events = pd.DataFrame([_plain_event(1)])
    package = build_package(_plain_incident(), events)
    for outcome in (requests.Timeout("read timed out"), _Response(200, body=None)):
        client, _ = _client(monkeypatch, [outcome] * 2, attempts=2)
        record, _ = investigate("direct", package, None, client, {"model": "m"})
        assert record["status"] == "failed" and record["response"] is None and record["errors"]


def test_missing_gcloud_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(gemini.shutil, "which", lambda name: None)
    with pytest.raises(GenerationError, match="gcloud is not installed"):
        GeminiClient(GeminiConfig())


def _alerts_stage_from(package: Path) -> Stage:
    def run(ctx: Context) -> Outcome:
        out = ctx.run_dir / "alerts"
        out.mkdir(parents=True, exist_ok=True)
        for name in package.iterdir():
            (out / name.name).write_bytes(name.read_bytes())
        return Outcome(runner.DONE, {"package": out})

    return Stage("alerts", run)


def test_runner_with_gcloud_missing_marks_investigate_failed_and_keeps_the_detector_results(tmp_path, monkeypatch):
    package = _package(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("PATH", "")  # no gcloud, and the child process is started by absolute path
    config = {"profile": "test",
              "investigate": {"handoff": "alerts.package", "output": "{run}/investigations", "workers": 2, "timeout_seconds": 120},
              "dashboard": {"packages": [{"name": "t", "incidents": "alerts.package", "investigations": "investigate.investigations"}],
                            "output": "{run}/dashboard"}}
    real = {s.name: s for s in runner.build_stages({"stages": [
        {"name": "investigate", "requires": ["alerts"]}, {"name": "dashboard", "requires": ["alerts"], "optional": ["investigate"]}]})}
    stages = [_alerts_stage_from(package), real["investigate"], real["dashboard"]]
    run_dir = tmp_path / "run"
    result = run_pipeline(config, run_dir, Options(llm=True), stages, echo=lambda _: None)
    status = {r.name: r.status for r in result.results}
    assert status == {"alerts": "done", "investigate": "failed", "dashboard": "done"} and result.exit_code == 1
    assert "gcloud is not installed" in result.failed[0].reason and "exited with code" in result.failed[0].reason
    for name in ("alerts.parquet", "events.parquet", "incidents.jsonl", "manifest.json"):
        assert (run_dir / "alerts" / name).read_bytes() == (package / name).read_bytes()
    check = json.loads((run_dir / "dashboard" / "check.json").read_text(encoding="utf-8"))["packages"]["t"]
    assert check["incidents"] == 4 and check["unresolved_references"] == 0
    assert check["investigation_states"]["AI + ATT&CK, checked"] == {"not_generated": 4}
    report = final_report(result, "sample", llm=True)
    assert "--only investigate --llm" in report and "untouched" in report


# ─── Case 7: timestamps and windows ────────────────────────────────────

BOUNDARIES = [  # timestamp, dataset day, hour window start
    (1, 1, 1), (3600, 1, 1), (3601, 1, 3601), (7200, 1, 3601), (7201, 1, 7201),
    (86_400, 1, 82_801), (86_401, 2, 86_401), (172_800, 2, 169_201), (172_801, 3, 172_801),
]


@pytest.mark.parametrize("timestamp, day, hour", BOUNDARIES)
def test_day_and_hour_formulas_agree_across_components(timestamp, day, hour):
    assert dataset_day(timestamp) == day == ((timestamp - 1) // 86_400) + 1
    assert 1 + ((timestamp - 1) // 3600) * 3600 == hour
    assert int(hour_start_array(np.array([timestamp]), 3600, 1)[0]) == hour
    assert dataset_hour_start(timestamp) == hour
    # Every display of the same second names the same day and clock time.
    assert dataset_time(timestamp) == format_dataset_second(timestamp)
    assert dataset_time(timestamp).startswith(f"Day {day} ")


def test_dataset_day_refuses_zero_and_negative_seconds():
    for timestamp in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            dataset_day(timestamp)


def _events_dataset(tmp_path: Path):
    """Ingest events at the boundary seconds, then lay them out the way a feature build does (one folder per day)."""
    users = ["U1@DOM1"]
    stamps = sorted({t for t, _, _ in BOUNDARIES} | {3599, 7199, 82_800})
    source = _source(tmp_path, [GOOD.format(t=t).replace("U1@DOM1", users[0]) for t in stamps])
    ingest_authentication(source, tmp_path / "ing", IngestionConfig(day_end=3))
    table = ds.dataset(tmp_path / "ing" / "events", format="parquet", partitioning="hive").to_table().to_pandas()
    table["is_machine_account"] = False
    for flag in ("is_new_user_source", "is_new_host_connection", "is_new_user_destination"):
        table[flag] = False
    for count in ("prior_auth_count_1h", "prior_failure_count_1h", "prior_unique_destinations_24h"):
        table[count] = 0
    root = tmp_path / "raw_events"
    for day, rows in table.groupby("dataset_day"):
        part = root / f"dataset_day={day}"
        part.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(rows.drop(columns=["dataset_day"]), preserve_index=False), part / "part.parquet")
    return table, ds.dataset(root, format="parquet", partitioning="hive")


def test_hourly_counts_alert_windows_and_attached_events_use_the_same_hours(tmp_path):
    table, dataset = _events_dataset(tmp_path)
    expected = table.assign(hour=1 + ((table["timestamp"] - 1) // 3600) * 3600).groupby(["dataset_day", "hour"]).size()
    assert (table["dataset_day"] == ((table["timestamp"] - 1) // 86_400 + 1)).all()
    counted = pd.concat([hourly_counts(dataset, day) for day in (1, 2, 3)], ignore_index=True)
    assert dict(zip(zip(counted["day"], counted["hour"]), counted["n_events"])) == {k: int(v) for k, v in expected.items()}
    # One alert per user-hour: each event lands in exactly the alert whose half-open window contains it.
    scores = counted.assign(gru_max_event=0.5, fusion=0.9, is_machine_account=False)
    alerts = select_daily_alerts(scores, budget=50)
    events = alert_events(dataset, alerts, log=lambda _: None)
    assert len(events) == len(table) and events["source_reference"].is_unique
    joined = events.merge(alerts[["alert_id", "window_start", "window_end", "n_events"]], on="alert_id")
    assert ((joined["timestamp"] >= joined["window_start"]) & (joined["timestamp"] < joined["window_end"])).all()
    assert (joined.groupby("alert_id").size().reindex(alerts["alert_id"]).to_numpy() == alerts["n_events"].to_numpy()).all()
    by_ref = dict(zip(joined["source_reference"], joined["window_start"]))
    stamp = dict(zip(table["source_reference"], table["timestamp"]))
    refs = {t: ref for ref, t in stamp.items()}
    assert by_ref[refs[3600]] == 1 and by_ref[refs[3601]] == 3601  # the end second belongs to the next window
    assert by_ref[refs[86_400]] == 82_801 and by_ref[refs[86_401]] == 86_401  # the day change starts a new hour


def _window_alerts(starts):
    return pd.DataFrame({"user": "U1@DOM1", "day": 1, "window_start": starts, "window_end": [s + 3600 for s in starts]})


@pytest.mark.parametrize("gap, merged", [(0, True), (7200, True), (7201, False), (10_800, False)])
def test_incident_grouping_gap_is_inclusive_at_7200_seconds(gap, merged):
    second_start = 1 + 3600 + gap
    ids = group_incidents(_window_alerts([1, second_start]), max_gap_seconds=7200)
    assert (ids.iloc[0] == ids.iloc[1]) is merged


def test_hour_aligned_alerts_three_hours_apart_form_one_incident_and_four_hours_apart_two():
    alerts = select_daily_alerts(pd.concat([
        pd.DataFrame({"user": "U1@DOM1", "day": 1, "hour": [1, 1 + 3 * 3600], "fusion": 0.9, "gru_max_event": 0.1}),
        pd.DataFrame({"user": "U2@DOM1", "day": 1, "hour": [1, 1 + 4 * 3600], "fusion": 0.8, "gru_max_event": 0.1}),
    ]), budget=10)
    ids = group_incidents(alerts)
    by_user = {user: set(ids[alerts["user"] == user]) for user in ("U1@DOM1", "U2@DOM1")}
    assert len(by_user["U1@DOM1"]) == 1 and len(by_user["U2@DOM1"]) == 2


def test_red_team_labels_use_the_same_hour_as_the_scores(tmp_path):
    labels = pa.table({"timestamp": pa.array([3600, 3601, 86_400, 86_401], pa.int64()),
                       "user": ["U1@DOM1"] * 4, "source_computer": ["C1"] * 4, "destination_computer": ["C2"] * 4})
    positives = aggregate_labels_to_user_hours(labels, hour_seconds=3600)
    assert positives == {("U1@DOM1", 1), ("U1@DOM1", 3601), ("U1@DOM1", 82_801), ("U1@DOM1", 86_401)}


# ─── Case 8: evidence links ────────────────────────────────────────────


def _check_package_links(package: Path) -> tuple[int, int]:
    """Every incident reference resolves to exactly one event, inside an alert window of that incident."""
    incidents = [json.loads(line) for line in (package / "incidents.jsonl").read_text(encoding="utf-8").splitlines()]
    events = pd.read_parquet(package / "events.parquet")
    alerts = pd.read_parquet(package / "alerts.parquet")
    assert events["source_reference"].is_unique and set(events["alert_id"]) <= set(alerts["alert_id"])
    assert (events["source_reference"] == "auth.txt:" + events["source_line"].astype(str)).all()
    owner = events.merge(alerts[["alert_id", "incident_id", "window_start", "window_end", "user_id"]], on="alert_id")
    assert ((owner["timestamp"] >= owner["window_start"]) & (owner["timestamp"] < owner["window_end"])).all()
    assert (owner["acting_user"] == owner["user_id"]).all()
    by_ref = owner.set_index("source_reference")
    references = 0
    for incident in incidents:
        refs = incident["source_references"]
        assert len(refs) == len(set(refs)) == incident["evidence_count"]
        assert by_ref.index.isin(refs).sum() == len(refs)
        assert set(by_ref.loc[refs, "incident_id"]) <= {incident["incident_id"]}
        references += len(refs)
    assert references == len(events)  # no event is cited by two incidents or by none
    return len(incidents), references


def test_synthetic_package_has_a_one_to_one_link_from_every_reference_to_an_event(tmp_path):
    assert _check_package_links(_package(tmp_path)) == (4, 9)


@pytest.mark.skipif(not (SAMPLE_RUN / "alerts" / "incidents.jsonl").is_file(), reason="run the sample profile first")
def test_sample_run_references_resolve_and_map_back_to_the_subset_records():
    package = SAMPLE_RUN / "alerts"
    incidents, references = _check_package_links(package)
    assert incidents > 0 and references > 0
    events = pd.read_parquet(package / "events.parquet")
    with gzip.open(SUBSET / "line_map.txt.gz", "rt", encoding="utf-8") as handle:
        line_map = {int(line): number for number, line in enumerate(handle, start=1) if line.strip()}
    wanted = {line_map[int(line)] for line in events["source_line"]}  # position of each event in the subset file
    kept = {}
    with gzip.open(SUBSET / "auth_subset.txt.gz", "rt", encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, start=1):
            if number in wanted:
                kept[number] = line.rstrip("\r\n").split(",")
    fields = ["timestamp", "source_user", "destination_user", "source_computer", "destination_computer",
              "authentication_type", "logon_type", "authentication_orientation", "authentication_result"]
    for row in events[["source_line", *fields]].itertuples(index=False):
        raw = kept[line_map[int(row.source_line)]]
        assert [str(value) for value in row[1:]] == raw, row.source_line
    # The answer key stays out of everything built for the model.
    retriever = TechniqueRetriever.from_file(DEFAULT_CATALOG)
    alerts = pd.read_parquet(package / "alerts.parquet", columns=["alert_id", "incident_id"])
    merged = events.merge(alerts, on="alert_id")
    incident_records = {json.loads(line)["incident_id"]: json.loads(line)
                        for line in (package / "incidents.jsonl").read_text(encoding="utf-8").splitlines()}
    for incident_id, rows in list(merged.groupby("incident_id"))[:40]:
        built = build_package(incident_records[incident_id], rows.drop(columns=["incident_id"]))
        retrieval = retrieve_candidates(rows, retriever)
        assert_no_answer_key(retrieval, "retrieval")
        for mode in ("direct", "rag"):
            assert_no_answer_key(user_prompt(mode, built, retrieval), "prompt")
    assert_no_answer_key(SYSTEM_PROMPT, "system prompt")
    dry = SAMPLE_RUN / "investigations" / "dry_run"
    for path in dry.glob("*.jsonl"):
        text = path.read_text(encoding="utf-8").lower()
        assert "ground_truth" not in text and "red-team" not in text and "redteam" not in text, path.name


def test_subset_ingestion_keeps_each_event_tied_to_its_subset_line(tmp_path):
    summary = ingest_authentication(
        SUBSET / "auth_subset.txt.gz", tmp_path / "ing", IngestionConfig(day_end=9, max_source_rows=3000, line_map=SUBSET / "line_map.txt.gz"))
    assert summary["counts"]["reconciled"] and summary["counts"]["accepted_rows"] > 2900
    rows = ds.dataset(tmp_path / "ing" / "events", format="parquet", partitioning="hive").to_table().to_pandas()
    with gzip.open(SUBSET / "line_map.txt.gz", "rt", encoding="utf-8") as handle:
        original = [int(line) for line in handle if line.strip()][:3000]
    with gzip.open(SUBSET / "auth_subset.txt.gz", "rt", encoding="utf-8", newline="") as handle:
        raw = [line.rstrip("\r\n") for _, line in zip(range(3000), handle)]
    by_original = dict(zip(original, raw))
    assert len(rows) == len(rows["source_reference"].unique())
    assert all(by_original[line] == record for line, record in zip(rows["source_line"], rows["raw_record"]))
    assert (rows["source_reference"] == "auth.txt:" + rows["source_line"].astype(str)).all()


def test_the_answer_key_is_refused_wherever_it_could_enter_a_prompt():
    package = {"incident_id": "I", "events": [], "note": "marked ground_truth_redteam"}
    with pytest.raises(AnswerKeyLeak):
        assert_no_answer_key(package)
    incident = {**_plain_incident(), "ground_truth_redteam": True, "ground_truth_redteam_hours": 1}
    incident["alert_hours"][0]["ground_truth_redteam"] = True
    built = build_package(incident, pd.DataFrame([_plain_event(1)]))
    for mode in ("direct", "rag"):
        text = user_prompt(mode, built, {"candidates": [], "snapshot": {"attack_version": "x"}})
        assert "ground_truth" not in text and "redteam" not in text.lower()


# ─── Case 9: stage isolation in the runner ─────────────────────────────


def _profile_chain(tmp_path: Path, failing: set[str]):
    config = runner.load_profile("sample")
    calls: list[str] = []

    def make(entry):
        name = entry["name"]

        def run(ctx: Context) -> Outcome:
            calls.append(name)
            if name in failing:
                raise StageFailed(f"{name} broke")
            out = ctx.run_dir / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.txt").write_text(name, encoding="utf-8")
            return Outcome(runner.DONE, {"result": out / "result.txt"})

        return Stage(name, run, list(entry.get("requires", [])), list(entry.get("optional", [])))

    return config, [make(entry) for entry in config["stages"]], calls


BLOCKED = {  # the stages that need the failed one, from config/pipeline_sample.json
    "ingest": ["features", "score", "alerts", "investigate", "dashboard"],
    "features": ["score", "alerts", "investigate", "dashboard"],
    "score": ["alerts", "investigate", "dashboard"],
    "alerts": ["investigate", "dashboard"],
    "investigate": [],  # the dashboard only lists it as optional
    "dashboard": [],
}


@pytest.mark.parametrize("failed", list(BLOCKED))
def test_a_failed_stage_blocks_its_dependents_keeps_earlier_outputs_and_rerun_resumes(tmp_path, failed):
    config, stages, calls = _profile_chain(tmp_path, {failed})
    run_dir = tmp_path / "run"
    result = run_pipeline(config, run_dir, Options(), stages, echo=lambda _: None)
    status = {r.name: r.status for r in result.results}
    names = [s.name for s in stages]
    assert status[failed] == "failed" and result.exit_code == 1
    assert [n for n in names if status[n] == "skipped"] == BLOCKED[failed]
    assert [r.reason.startswith("blocked") for r in result.results if r.name in BLOCKED[failed]] == [True] * len(BLOCKED[failed])
    earlier = names[:names.index(failed)]
    assert all(status[n] == "done" for n in earlier)
    outputs = {n: (run_dir / n / "result.txt").read_bytes() for n in earlier}
    report = final_report(result, "sample", llm=False)
    hint = "--only investigate --llm" if failed == "investigate" else f"--profile sample --from {failed}"
    assert hint in report and f"FAILED: {failed}" in report
    state = json.loads((run_dir / "pipeline_state.json").read_text(encoding="utf-8"))["stages"]
    assert state[failed]["error"] == f"{failed} broke" and state[failed]["completed"] is None
    assert all(state[n]["completed"]["status"] == "done" for n in earlier)

    calls.clear()  # fix the cause and rerun: finished stages are skipped, the rest run
    config, stages, calls = _profile_chain(tmp_path, set())
    again = run_pipeline(config, run_dir, Options(), stages, echo=lambda _: None)
    assert again.exit_code == 0 and calls == names[names.index(failed):]
    assert {n: (run_dir / n / "result.txt").read_bytes() for n in earlier} == outputs


def test_a_killed_script_fails_its_stage_with_the_exit_code_and_last_output(tmp_path):
    ctx = Context({}, tmp_path, Options(), PipelineState(tmp_path / "s.json", "t", tmp_path), "x", PeakMemory(), None, lambda _: None)
    with pytest.raises(StageFailed, match="exited with code 137: last words"):
        run_command(ctx, [sys.executable, "-c", "import os; print('last words', flush=True); os._exit(137)"])


def test_a_damaged_state_file_means_every_stage_runs_again(tmp_path):
    state = tmp_path / "pipeline_state.json"
    for damaged in ('{"profile": "test", "stages": {"a": ', "[]", '{"profile": "test", "stages": null}', ""):
        state.write_text(damaged, encoding="utf-8")
        assert PipelineState(state, "test", tmp_path).stages == {}


# ─── Case 10: the test-day guard ───────────────────────────────────────


def test_test_labels_and_test_alert_stage_are_refused_before_reading_anything(tmp_path):
    with pytest.raises(PermissionError, match="reserved"):
        load_positive_user_hours(tmp_path / "does_not_exist", None, "test")
    ctx = Context({"alerts": {"split_name": "test"}}, tmp_path, Options(), PipelineState(tmp_path / "s.json", "t", tmp_path),
                  "alerts", PeakMemory(), None, lambda _: None)
    with pytest.raises(StageFailed, match="stored one-time test"):
        runner.stage_alerts(ctx)


def test_a_profile_with_test_days_stops_the_command_before_any_stage(tmp_path, monkeypatch, capsys):
    profile = json.loads((REPO_ROOT / "config" / "pipeline_sample.json").read_text(encoding="utf-8"))
    profile["score"]["days"] = [8, 17]
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline_sample.json").write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(runner, "CONFIG_DIR", config_dir)
    run_dir = tmp_path / "run"
    assert runner.main(["--profile", "sample", "--run-dir", str(run_dir)]) == 2
    assert "test days" in capsys.readouterr().err and not run_dir.exists()
