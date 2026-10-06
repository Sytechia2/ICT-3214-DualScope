import pandas as pd
import pytest

from dualscope.attack.catalog import TECHNIQUE_ID, TechniqueCatalog, clean_description, extract_techniques
from dualscope.attack.queries import MAX_EVIDENCE_REFERENCES, observed_behaviours, retrieve_candidates
from dualscope.attack.retrieval import DEFAULT_CATALOG, TechniqueRetriever


@pytest.fixture(scope="module")
def retriever():
    return TechniqueRetriever.from_file(DEFAULT_CATALOG)


def _pattern(stix_id, technique_id, name, **extra):
    return {
        "type": "attack-pattern", "id": stix_id, "name": name, "description": extra.pop("description", ""),
        "external_references": [{"source_name": "mitre-attack", "external_id": technique_id,
                                 "url": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}"}],
        **extra,
    }


def test_extract_keeps_active_techniques_and_links_detection_data():
    bundle = {"objects": [
        _pattern("ap--1", "T1550.002", "Pass the Hash", x_mitre_is_subtechnique=True,
                 description="Uses hashes (Citation: Someone 2020) via [NTLM](https://example.org).",
                 kill_chain_phases=[{"kill_chain_name": "mitre-attack", "phase_name": "lateral-movement"}]),
        _pattern("ap--2", "T9999", "Old", revoked=True),
        _pattern("ap--3", "T9998", "Deprecated", x_mitre_deprecated=True),
        {"type": "x-mitre-detection-strategy", "id": "ds--1", "x_mitre_analytic_refs": ["an--1"]},
        {"type": "x-mitre-analytic", "id": "an--1", "x_mitre_log_source_references": [
            {"x_mitre_data_component_ref": "dc--1", "name": "WinEventLog:Security", "channel": "EventCode=4624"}]},
        {"type": "x-mitre-data-component", "id": "dc--1", "name": "Logon Session Creation"},
        {"type": "relationship", "id": "rel--1", "relationship_type": "detects", "source_ref": "ds--1", "target_ref": "ap--1"},
    ]}
    techniques = extract_techniques(bundle)
    assert [t["technique_id"] for t in techniques] == ["T1550.002"]
    technique = techniques[0]
    assert technique["description"] == "Uses hashes via NTLM."
    assert technique["parent_id"] == "T1550"
    assert technique["tactics"] == ["lateral-movement"]
    assert technique["data_components"] == ["Logon Session Creation"]
    assert technique["log_sources"] == ["WinEventLog:Security EventCode=4624"]


def test_clean_description_strips_citations_and_links():
    assert clean_description("A [B](http://x) c (Citation: D E) f.") == "A B c f."


def test_committed_catalogue_is_a_recorded_snapshot():
    catalog = TechniqueCatalog.load(DEFAULT_CATALOG)
    assert catalog.snapshot["attack_version"] == "19.2"
    assert len(catalog.snapshot["bundle_sha256"]) == 64
    assert len(catalog) == catalog.snapshot["techniques"]
    assert all(TECHNIQUE_ID.fullmatch(t.technique_id) and t.url.startswith("https://attack.mitre.org/") for t in catalog.techniques)
    assert "T1550.002" in catalog and "T0000" not in catalog
    assert catalog.get("T1550.002").name == "Pass the Hash"


@pytest.mark.parametrize("query, expected", [
    ("pass the hash NTLM authentication network logon using a stolen password hash", "T1550.002"),
    ("brute force password guessing password spraying many failed logon attempts against accounts", "T1110.001"),
    ("steal or forge kerberos tickets many kerberos service ticket requests", "T1558"),
])
def test_search_returns_expected_top_technique(retriever, query, expected):
    results = retriever.search(query, k=3, auth_observable_only=True)
    assert results[0].technique_id == expected
    assert results[0].url and results[0].description
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_weak_or_unrelated_queries_return_nothing(retriever):
    assert retriever.search("user logged on during the working day", auth_observable_only=True) == []
    assert retriever.search("quarterly marketing budget spreadsheet") == []
    assert retriever.search("   ") == []


def test_auth_observable_filter_drops_techniques_logon_records_cannot_show(retriever):
    query = "lateral movement remote services logon to remote hosts not accessed before network logon"
    unfiltered = {r.technique_id for r in retriever.search(query, k=10, min_score=0.0)}
    filtered = retriever.search(query, k=10, min_score=0.0, auth_observable_only=True)
    assert "T1037.003" in unfiltered  # Network Logon Script: no authentication data component
    assert all(r.auth_observable for r in filtered)


def _event(line, **fields):
    base = {
        "source_reference": f"auth.txt:{line}", "timestamp": line, "authentication_type": "Kerberos",
        "logon_type": "Network", "authentication_orientation": "LogOn", "authentication_result": "Success",
        "is_new_user_destination": False, "is_new_host_connection": False, "is_new_user_source": False,
    }
    return {**base, **fields}


def test_behaviours_fire_on_single_events_with_their_evidence():
    events = pd.DataFrame([
        _event(3, authentication_type="NTLM", is_new_user_destination=True),
        _event(1, authentication_result="Fail"),
        _event(2, is_new_host_connection=True),
        _event(4, is_new_user_source=True, authentication_orientation="LogOff"),  # log-offs are ignored
        _event(5, authentication_type="NTLM"),  # NTLM to a known destination is not a behaviour
    ])
    found = {b["behaviour"]: b for b in observed_behaviours(events)}
    assert set(found) == {"failed_logons", "ntlm_logon_to_new_destination", "network_logon_to_new_destination"}
    assert found["ntlm_logon_to_new_destination"]["evidence_references"] == ["auth.txt:3"]
    assert found["network_logon_to_new_destination"]["evidence_references"] == ["auth.txt:2", "auth.txt:3"]
    assert found["failed_logons"]["event_count"] == 1


def test_evidence_references_are_capped():
    events = pd.DataFrame([_event(i, authentication_result="Fail") for i in range(1, MAX_EVIDENCE_REFERENCES + 6)])
    (behaviour,) = observed_behaviours(events)
    assert behaviour["event_count"] == MAX_EVIDENCE_REFERENCES + 5
    assert len(behaviour["evidence_references"]) == MAX_EVIDENCE_REFERENCES
    assert behaviour["evidence_references_truncated"] is True


def test_candidates_record_behaviours_and_insufficient_evidence(retriever):
    result = retrieve_candidates(pd.DataFrame([_event(1, authentication_type="NTLM", is_new_user_destination=True)]), retriever)
    assert result["insufficient_evidence"] is False
    top = result["candidates"][0]
    assert top["technique_id"] == "T1550.002"
    assert top["retrieved_by"] == ["ntlm_logon_to_new_destination"]
    assert result["snapshot"]["attack_version"] == "19.2"

    quiet = retrieve_candidates(pd.DataFrame([_event(1), _event(2, authentication_orientation="LogOff", is_new_user_source=True)]), retriever)
    assert quiet == {**quiet, "behaviours": [], "candidates": [], "insufficient_evidence": True}
