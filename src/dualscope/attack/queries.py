"""Evidence-based query templates: suspicious authentication events -> ATT&CK candidates.

Each template is a rule on single events (the rows of the alert handoff's
``events.parquet``) plus a fixed query in ATT&CK's own wording. A behaviour
fires for an incident when at least one of its events matches the rule, and it
carries the number of matching events and their ``auth.txt`` references, so
later steps can cite the exact evidence. An incident with no matching event
gets no candidates: ``insufficient_evidence``.

The novelty flags are first appearances since day 1 of the replay
(docs/lanl_features.md): ``is_new_user_destination`` is the user's first logon
to that destination, ``is_new_host_connection`` the first connection between
the two computers, and ``is_new_user_source`` the user's first event from that
source computer. Log-off events are ignored by the novelty rules.

Candidates are retrieval output, not conclusions. A technique is retrieved
because its text matches the behaviour; whether the evidence supports it is
decided later (Task 6.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from dualscope.attack.retrieval import TechniqueRetriever

MAX_EVIDENCE_REFERENCES = 20


def _new_destination(events: pd.DataFrame) -> pd.Series:
    return events["is_new_user_destination"] | events["is_new_host_connection"]


def _network(events: pd.DataFrame) -> pd.Series:
    return events["logon_type"] == "Network"


def _not_logoff(events: pd.DataFrame) -> pd.Series:
    # A log-off is recorded on the machine being left (often source = destination),
    # so its novelty flags do not show a new logon path.
    return events["authentication_orientation"] != "LogOff"


@dataclass(frozen=True)
class BehaviourTemplate:
    behaviour: str
    description: str
    query: str
    matches: Callable[[pd.DataFrame], pd.Series]


TEMPLATES: tuple[BehaviourTemplate, ...] = (
    BehaviourTemplate(
        "failed_logons",
        "Failed authentication",
        "brute force password guessing password spraying many failed logon attempts against accounts",
        lambda e: e["authentication_result"] == "Fail",
    ),
    BehaviourTemplate(
        "ntlm_logon_to_new_destination",
        "NTLM network logon to a destination the user (or the source computer) had never reached before",
        "pass the hash NTLM authentication network logon using a stolen password hash",
        lambda e: (e["authentication_type"] == "NTLM") & _network(e) & _new_destination(e) & _not_logoff(e),
    ),
    BehaviourTemplate(
        "network_logon_to_new_destination",
        "Network logon to a destination the user (or the source computer) had never reached before",
        "lateral movement using remote services and SMB admin shares with network logons to remote hosts",
        lambda e: _network(e) & _new_destination(e) & _not_logoff(e),
    ),
    BehaviourTemplate(
        "logon_from_new_source",
        "The user authenticated from a source computer it had never used before",
        "valid accounts: stolen credentials of an existing account used to log on from an unusual source computer",
        lambda e: e["is_new_user_source"].astype(bool) & _not_logoff(e),
    ),
)


def observed_behaviours(events: pd.DataFrame) -> list[dict]:
    """Templates that at least one of ``events`` matches, with their evidence.

    ``events`` are one incident's authentication events. References are listed
    in time order, at most ``MAX_EVIDENCE_REFERENCES`` per behaviour.
    """
    ordered = events.sort_values(["timestamp", "source_reference"], kind="stable")
    found = []
    for template in TEMPLATES:
        matched = ordered[template.matches(ordered).fillna(False).astype(bool)]
        if matched.empty:
            continue
        references = matched["source_reference"].tolist()
        found.append({
            "behaviour": template.behaviour,
            "description": template.description,
            "query": template.query,
            "event_count": len(references),
            "evidence_references": references[:MAX_EVIDENCE_REFERENCES],
            "evidence_references_truncated": len(references) > MAX_EVIDENCE_REFERENCES,
        })
    return found


def retrieve_candidates(events: pd.DataFrame, retriever: TechniqueRetriever, k: int = 5) -> dict:
    """Candidate techniques for one incident, with the behaviours that retrieved them.

    Only techniques an authentication log can show are searched. A technique
    retrieved by several behaviours keeps its best score.
    """
    behaviours = observed_behaviours(events)
    candidates: dict[str, dict] = {}
    for behaviour in behaviours:
        for result in retriever.search(behaviour["query"], k=k, auth_observable_only=True):
            entry = candidates.setdefault(result.technique_id, {**result.to_dict(), "retrieved_by": []})
            entry["score"] = max(entry["score"], round(result.score, 4))
            entry["retrieved_by"].append(behaviour["behaviour"])
    ranked = sorted(candidates.values(), key=lambda entry: (-entry["score"], entry["technique_id"]))
    return {
        "behaviours": behaviours,
        "insufficient_evidence": not ranked,
        "candidates": ranked,
        "snapshot": {
            "attack_version": retriever.catalog.snapshot["attack_version"],
            "bundle_sha256": retriever.catalog.snapshot["bundle_sha256"],
        },
    }
