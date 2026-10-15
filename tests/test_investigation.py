import json

import pandas as pd
import pytest

from dualscope.attack.retrieval import DEFAULT_CATALOG, TechniqueRetriever
from dualscope.attack.queries import retrieve_candidates
from dualscope.investigation import gemini
from dualscope.investigation.gemini import GeminiClient, GeminiConfig, Generation, GenerationError
from dualscope.investigation.generate import investigate, validate_response
from dualscope.investigation.package import (
    MAX_PACKAGE_EVENTS,
    AnswerKeyLeak,
    assert_no_answer_key,
    build_package,
    package_references,
)
from dualscope.investigation.prompts import SYSTEM_PROMPT, render_evidence, user_prompt
from dualscope.investigation.verify import (
    REJECTED,
    SUPPORTED,
    UNCERTAIN,
    apply_verification,
    verify_claim,
    verify_response,
    verify_technique,
)

DAY = 86_400
START = 27 * DAY + 14 * 3_600 + 1  # day 28, 14:00


def _event(line, offset=0, **overrides):
    event = {
        "source_reference": f"auth.txt:{line}", "source_line": line, "timestamp": START + offset,
        "acting_user": "U7@DOM1", "source_user": "U7@DOM1", "destination_user": "U7@DOM1",
        "source_computer": "C1", "destination_computer": "C2", "authentication_type": "Kerberos",
        "logon_type": "Network", "authentication_orientation": "LogOn", "authentication_result": "Success",
        "is_new_user_source": False, "is_new_host_connection": False, "is_new_user_destination": False,
        "prior_auth_count_1h": 0, "prior_failure_count_1h": 0, "prior_unique_destinations_24h": 1,
        "alert_id": "ALR-D28-07",
    }
    event.update(overrides)
    return event


def _incident():
    return {
        "incident_id": "INC-TEST-D28-U7_DOM1-001", "user_id": "U7@DOM1", "dataset_day": 28,
        "start_time": START, "end_time": START + 3_600, "duration_hours": 1, "priority": "HIGH",
        "ground_truth_redteam": True, "ground_truth_redteam_hours": 1,
        "alert_hours": [{"alert_id": "ALR-D28-07", "window_start": START, "window_end": START + 3_600,
                         "fusion_score": 0.99, "rank_in_day": 7, "tied_at_cutoff": False,
                         "gru_percentile_in_day": 0.999, "ground_truth_redteam": True}],
        "detector_scores": {"graph": {"max_score": 0.9, "any_alert": False}},
        "graph_context": {"degree_growth": 2, "new_edge_count": 1, "used_by_final_model": False,
                          "top_edges": [{"destination_computer": "C9", "is_new_edge": True, "success_count": 1,
                                         "failure_count": 0, "source_references": ["auth.txt:1"]}]},
    }


@pytest.fixture()
def events():
    rows = [_event(100 + i, i) for i in range(10)]
    rows.append(_event(200, 20, authentication_type="NTLM", destination_computer="C5", is_new_user_destination=True))
    rows.append(_event(201, 21, authentication_result="Fail"))
    rows.append(_event(202, 22, authentication_orientation="LogOff", is_new_user_source=True, destination_computer="C1"))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def retriever():
    return TechniqueRetriever.from_file(DEFAULT_CATALOG)


@pytest.fixture()
def package(events):
    return build_package(_incident(), events)


# ─── 6.2 packages and prompts ──────────────────────────────────────────


def test_package_never_contains_the_answer_key(package):
    text = json.dumps(package)
    assert "ground_truth" not in text and "redteam" not in text
    assert_no_answer_key(package)


def test_answer_key_check_finds_keys_and_text():
    with pytest.raises(AnswerKeyLeak):
        assert_no_answer_key({"a": {"ground_truth_redteam": False}})
    with pytest.raises(AnswerKeyLeak):
        assert_no_answer_key(["this was red-team activity"])


def test_package_counts_cover_all_events_and_skip_logoff_novelty(package):
    assert package["counts"]["events"] == 13
    assert package["counts"]["failed"] == 1
    assert package["counts"]["first_events_from_source"] == 0  # only a log-off had the flag
    logoff = next(e for e in package["events"] if e["ref"] == "auth.txt:202")
    assert logoff["flags"] == []


def test_large_incidents_keep_rule_matched_events_first():
    rows = [_event(i, i) for i in range(500)]
    rows += [_event(1000 + i, 600 + i, authentication_type="NTLM", is_new_host_connection=True) for i in range(5)]
    package = build_package(_incident(), pd.DataFrame(rows))
    refs = {e["ref"] for e in package["events"]}
    assert len(package["events"]) == MAX_PACKAGE_EVENTS
    assert {f"auth.txt:{1000 + i}" for i in range(5)} <= refs
    assert package["selection"]["truncated"] and package["selection"]["total_events"] == 505
    assert [e["timestamp"] for e in package["events"]] == sorted(e["timestamp"] for e in package["events"])


def test_modes_share_the_evidence_and_only_rag_gets_candidates(package, events, retriever):
    retrieval = retrieve_candidates(events, retriever)
    direct, rag = user_prompt("direct", package, retrieval), user_prompt("rag", package, retrieval)
    evidence = render_evidence(package)
    assert evidence in direct and evidence in rag
    assert "<attack_candidates>" in rag and "<attack_candidates>" not in direct
    assert "T1550.002" in rag


def test_rag_without_candidates_asks_for_no_mapping(package):
    prompt = user_prompt("rag", package, {"candidates": [], "snapshot": {"attack_version": "19.2"}})
    assert "No ATT&CK candidates were retrieved" in prompt and "<attack_candidates>" not in prompt


def test_package_references_include_events_and_sections(package):
    references = package_references(package)
    assert "auth.txt:200" in references and "package:counts" in references and "package:incident" in references


# ─── 6.2 generation and parsing ────────────────────────────────────────


def _reply(**overrides):
    reply = {
        "summary": "U7@DOM1 made an NTLM logon to C5.",
        "observations": [{"text": "NTLM network logon from C1 to C5.", "evidence": ["auth.txt:200"]}],
        "interpretations": [{"text": "Could be lateral movement.", "evidence": ["auth.txt:200"], "confidence": "low"}],
        "techniques": [{"technique_id": "T1550.002", "name": "Pass the Hash", "rationale": "NTLM to new host",
                        "evidence": ["auth.txt:200"], "confidence": "low"}],
        "no_supported_mapping": False,
        "uncertainty": ["Hash use is not visible."],
    }
    reply.update(overrides)
    return reply


class FakeGenerator:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, []

    def generate(self, system, user, schema):
        self.calls.append((system, user))
        if self.error:
            raise GenerationError(self.error)
        return Generation(self.text, "STOP", "fake-1", {"totalTokenCount": 10}, 0.1, 1)


def test_validate_response_accepts_a_good_reply_and_flags_problems():
    assert validate_response(_reply()) == []
    assert validate_response([]) == ["reply is not a JSON object"]
    bad_id = _reply(techniques=[{**_reply()["techniques"][0], "technique_id": "TA0008"}])
    assert any("not an ATT&CK ID" in p for p in validate_response(bad_id))
    contradiction = _reply(no_supported_mapping=True)
    assert any("no_supported_mapping" in p for p in validate_response(contradiction))
    assert validate_response(_reply(observations=[{"text": "x"}]))


def test_investigate_records_ok_invalid_and_failed(package, events, retriever):
    retrieval = retrieve_candidates(events, retriever)
    settings = {"model": "fake"}
    ok, prompt = investigate("rag", package, retrieval, FakeGenerator(json.dumps(_reply())), settings)
    assert ok["status"] == "ok" and ok["response"]["techniques"][0]["technique_id"] == "T1550.002"
    assert ok["retrieved_candidates"] and "<evidence>" in prompt
    invalid, _ = investigate("direct", package, None, FakeGenerator("not json"), settings)
    assert invalid["status"] == "invalid" and invalid["raw_text"] == "not json"
    failed, _ = investigate("direct", package, None, FakeGenerator(error="HTTP 503"), settings)
    assert failed["status"] == "failed" and failed["errors"] == ["HTTP 503"] and failed["response"] is None


def test_system_prompt_is_the_same_for_both_modes(package, events, retriever):
    retrieval = retrieve_candidates(events, retriever)
    generator = FakeGenerator(json.dumps(_reply()))
    investigate("direct", package, retrieval, generator, {})
    investigate("rag", package, retrieval, generator, {})
    assert generator.calls[0][0] == generator.calls[1][0] == SYSTEM_PROMPT


# ─── 6.3 verification ──────────────────────────────────────────────────


def _technique(technique_id, *refs):
    return {"technique_id": technique_id, "name": "x", "rationale": "x", "evidence": list(refs), "confidence": "low"}


def test_supported_pattern(package, retriever):
    result = verify_technique(_technique("T1550.002", "auth.txt:200"), package, retriever.catalog)
    assert result["status"] == SUPPORTED and result["matching_events"] == ["auth.txt:200"]


def test_invented_technique_id_is_rejected(package, retriever):
    result = verify_technique(_technique("T1999.999", "auth.txt:200"), package, retriever.catalog)
    assert result["status"] == REJECTED and "not an active technique" in result["reasons"][0]


def test_fabricated_citations_are_rejected(package, retriever):
    result = verify_technique(_technique("T1550.002", "auth.txt:999999"), package, retriever.catalog)
    assert result["status"] == REJECTED and result["broken_references"] == ["auth.txt:999999"]


def test_mixed_citations_cap_at_uncertain(package, retriever):
    result = verify_technique(_technique("T1550.002", "auth.txt:200", "auth.txt:999999"), package, retriever.catalog)
    assert result["status"] == UNCERTAIN


def test_unsupported_mapping_is_rejected(package, retriever):
    # Brute force needs failed logons; the cited event succeeded.
    result = verify_technique(_technique("T1110.001", "auth.txt:100"), package, retriever.catalog)
    assert result["status"] == REJECTED


def test_too_few_failures_is_uncertain(package, retriever):
    result = verify_technique(_technique("T1110.001", "auth.txt:201"), package, retriever.catalog)
    assert result["status"] == UNCERTAIN


def test_known_false_candidate_is_rejected(package, retriever):
    result = verify_technique(_technique("T1556.004", "auth.txt:200"), package, retriever.catalog)
    assert result["status"] == REJECTED and "network devices" in result["reasons"][0]


def test_subtechnique_without_rule_is_capped_by_parent(package, retriever):
    result = verify_technique(_technique("T1021.002", "auth.txt:200"), package, retriever.catalog)
    assert result["status"] == UNCERTAIN and "T1021 rule" in result["reasons"][0]


def test_weaker_pattern_is_uncertain(package, retriever):
    # NTLM logon, but to a known destination.
    rows = pd.DataFrame([_event(300, 0, authentication_type="NTLM")])
    pkg = build_package(_incident(), rows)
    assert verify_technique(_technique("T1550.002", "auth.txt:300"), pkg, retriever.catalog)["status"] == UNCERTAIN


def test_technique_outside_retrieved_candidates_is_flagged(package, retriever):
    result = verify_technique(_technique("T1078", "auth.txt:200"), package, retriever.catalog, ["T1550.002"])
    assert result["in_retrieved_candidates"] is False


def test_claims_must_name_only_cited_entities(package):
    assert verify_claim({"text": "NTLM logon from C1 to C5.", "evidence": ["auth.txt:200"]}, package)["status"] == "supported"
    partly = verify_claim({"text": "Logon to C9.", "evidence": ["auth.txt:200"]}, package)
    assert partly["status"] == "partly_supported"  # C9 is only in the graph context
    invented = verify_claim({"text": "Logon to C4242.", "evidence": ["auth.txt:200"]}, package)
    assert invented["status"] == "unsupported"
    uncited = verify_claim({"text": "Logon to C5.", "evidence": ["auth.txt:1"]}, package)
    assert uncited["status"] == "unsupported"
    wrong_protocol = verify_claim({"text": "A Kerberos logon.", "evidence": ["auth.txt:200"]}, package)
    assert wrong_protocol["status"] == "partly_supported"


def test_verified_view_removes_rejected_techniques_and_unsupported_claims(package, retriever):
    reply = _reply(
        observations=[{"text": "Logon to C5.", "evidence": ["auth.txt:200"]},
                      {"text": "Logon to C4242.", "evidence": ["auth.txt:200"]}],
        techniques=[_technique("T1550.002", "auth.txt:200"), _technique("T1556.004", "auth.txt:200")],
    )
    record = {"response": reply}
    verified = apply_verification(record, verify_response(reply, package, retriever.catalog))
    assert [t["technique_id"] for t in verified["techniques"]] == ["T1550.002"]
    assert [t["technique_id"] for t in verified["removed_techniques"]] == ["T1556.004"]
    assert len(verified["observations"]) == 1 and len(verified["removed_observations"]) == 1
    assert verified["outcome"] == "supported_mapping"


def test_no_mapping_outcome(package, retriever):
    reply = _reply(techniques=[], no_supported_mapping=True)
    verified = apply_verification({"response": reply}, verify_response(reply, package, retriever.catalog))
    assert verified["outcome"] == "no_supported_mapping" and verified["model_no_supported_mapping"]


# ─── Gemini client ─────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses, self.bodies = list(responses), []

    def post(self, url, json=None, timeout=None, headers=None):
        self.bodies.append(json)
        return self.responses.pop(0)


def _client(monkeypatch, responses, **config):
    monkeypatch.setattr(gemini, "_gcloud", lambda *args: "token")
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: None)
    session = FakeSession(responses)
    return GeminiClient(GeminiConfig(project="p", **config), session=session), session


def test_client_sends_schema_settings_and_skips_thought_parts(monkeypatch):
    ok = FakeResponse(200, {"modelVersion": "gemini-x", "usageMetadata": {"totalTokenCount": 5},
                            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                                {"text": "thinking", "thought": True}, {"text": "{\"a\": 1}"}]}}]})
    client, session = _client(monkeypatch, [ok])
    result = client.generate("sys", "user", {"type": "object"})
    assert result.text == "{\"a\": 1}" and result.model_version == "gemini-x"
    config = session.bodies[0]["generationConfig"]
    assert config["responseJsonSchema"] == {"type": "object"} and config["temperature"] == 1.0
    assert config["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}
    assert "project" not in client.config.settings()


def test_client_retries_rate_limits_then_gives_up(monkeypatch):
    busy = FakeResponse(429, {"error": "quota"})
    client, session = _client(monkeypatch, [busy] * 3, max_attempts=3)
    with pytest.raises(GenerationError, match="429"):
        client.generate("s", "u", {})
    assert len(session.bodies) == 3


def test_client_does_not_retry_bad_requests(monkeypatch):
    client, session = _client(monkeypatch, [FakeResponse(400, {"error": "bad"})])
    with pytest.raises(GenerationError, match="400"):
        client.generate("s", "u", {})
    assert len(session.bodies) == 1
