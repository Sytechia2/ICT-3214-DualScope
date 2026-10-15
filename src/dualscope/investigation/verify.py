"""Deterministic evidence verification of investigation summaries (Task 6.3).

Every check is a fixed rule on the evidence package; no model is involved.

Techniques get one of three statuses:

- **Rejected**: the ID is not an active ATT&CK technique in the pinned
  snapshot; no cited reference is an event in the package; ATT&CK lists no
  authentication-log data source for it; it is a documented false candidate
  (docs/attack_retrieval.md); or its support rule finds no matching cited event.
- **Uncertain**: the cited events match only partly (fewer than the rule needs,
  or a weaker form of the pattern), the technique's specific variant cannot be
  seen in logon records (most sub-techniques), some citations are broken, or
  there is no rule for it.
- **Supported**: the cited events contain the pattern the technique
  describes. This means *consistent with* the technique, not proof of intent.

Observations and interpretations are checked for their citations and for the
entities they name: every computer (C123), user (U123@DOM1), authentication
type and clock time in the text must appear in the cited evidence.
``supported`` = all found in the cited evidence; ``partly_supported`` = some
are only elsewhere in the package; ``unsupported`` = no valid citation, or an
entity that is nowhere in the package. Counts and other numbers are not
checked.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from dualscope.attack.catalog import TechniqueCatalog
from dualscope.attack.retrieval import AUTH_DATA_COMPONENTS
from dualscope.investigation.package import PACKAGE_REFERENCES, package_references

SUPPORTED, UNCERTAIN, REJECTED = "Supported", "Uncertain", "Rejected"
_ORDER = {REJECTED: 0, UNCERTAIN: 1, SUPPORTED: 2}

COMPUTER = re.compile(r"\bC\d+\b")
USER = re.compile(r"\bU\d+@DOM\d+\b")
CLOCK = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
AUTH_TYPES = ("NTLM", "Kerberos", "Negotiate", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0")

Event = Mapping[str, Any]


def _logon(e: Event) -> bool:
    return e["orientation"] != "LogOff"


def _new_destination(e: Event) -> bool:
    return _logon(e) and (e["is_new_user_destination"] or e["is_new_host_connection"])


def _network_new_destination(e: Event) -> bool:
    return e["logon_type"] == "Network" and _new_destination(e)


@dataclass(frozen=True)
class SupportRule:
    requirement: str
    matches: Callable[[Event], bool]
    best: str = SUPPORTED          # the strongest status the pattern can give
    min_events: int = 1            # fewer matching cited events -> Uncertain
    weaker: Callable[[Event], bool] | None = None  # a partial pattern -> Uncertain


SUPPORT_RULES: dict[str, SupportRule] = {
    "T1110": SupportRule("repeated failed logons", lambda e: e["result"] == "Fail", min_events=3),
    "T1110.001": SupportRule("repeated failed logons", lambda e: e["result"] == "Fail", min_events=3),
    "T1110.003": SupportRule(
        "failed logons (spraying needs failures across many accounts, which one user's incident cannot show)",
        lambda e: e["result"] == "Fail", best=UNCERTAIN),
    "T1110.004": SupportRule(
        "failed logons (credential stuffing needs many accounts, which one user's incident cannot show)",
        lambda e: e["result"] == "Fail", best=UNCERTAIN),
    "T1550.002": SupportRule(
        "an NTLM network logon to a destination the user or source computer never reached before",
        lambda e: e["auth_type"] == "NTLM" and _network_new_destination(e),
        weaker=lambda e: e["auth_type"] == "NTLM" and _logon(e)),
    "T1550.003": SupportRule(
        "a Kerberos network logon to a new destination (ticket reuse itself is not visible)",
        lambda e: e["auth_type"] == "Kerberos" and _network_new_destination(e), best=UNCERTAIN),
    "T1550": SupportRule(
        "an NTLM or Kerberos network logon to a new destination",
        lambda e: e["auth_type"] in ("NTLM", "Kerberos") and _network_new_destination(e), best=UNCERTAIN),
    "T1021": SupportRule("a network logon to a never-reached destination", _network_new_destination,
                         weaker=lambda e: _logon(e) and e["logon_type"] == "Network"),
    "T1021.001": SupportRule("a RemoteInteractive (RDP) logon to a new destination",
                             lambda e: e["logon_type"] == "RemoteInteractive" and _new_destination(e),
                             weaker=lambda e: e["logon_type"] == "RemoteInteractive" and _logon(e)),
    "T1078": SupportRule(
        "a successful logon from a new source or to a new destination",
        lambda e: e["result"] == "Success" and _logon(e) and (e["is_new_user_source"] or _new_destination(e)),
        weaker=lambda e: e["result"] == "Success" and _logon(e)),
    "T1558": SupportRule("Kerberos ticket requests (TGT/TGS); forged or stolen tickets are not visible",
                         lambda e: e["orientation"] in ("TGS", "TGT"), best=UNCERTAIN),
}

# Sub-techniques without their own rule use the parent's pattern, capped at
# Uncertain: logon records cannot show the specific variant (which remote
# service, which kind of account, ...).
SUBTECHNIQUE_CAP = UNCERTAIN

# Retrieved by 6.1 but documented as wrong for logon evidence (docs/attack_retrieval.md).
KNOWN_FALSE = {
    "T1556.004": "Network Device Authentication is about modifying network devices; logon records cannot show it",
    "T1110.002": "Password Cracking happens offline and leaves no logon record",
}


def _cap(status: str, cap: str) -> str:
    return status if _ORDER[status] <= _ORDER[cap] else cap


def _split_references(references: Iterable[str], valid: set[str]) -> tuple[list[str], list[str]]:
    good, broken = [], []
    for reference in dict.fromkeys(r.strip() for r in references):
        (good if reference in valid else broken).append(reference)
    return good, broken


def verify_technique(
    technique: Mapping[str, Any],
    package: Mapping[str, Any],
    catalog: TechniqueCatalog,
    retrieved: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Status, reasons and checked citations for one mapped technique."""
    technique_id = technique["technique_id"]
    events = {event["ref"]: event for event in package["events"]}
    cited, broken = _split_references(technique.get("evidence") or [], set(events) | set(PACKAGE_REFERENCES))
    cited_events = [events[ref] for ref in cited if ref in events]
    result: dict[str, Any] = {
        "technique_id": technique_id,
        "name": technique.get("name"),
        "catalog_name": None,
        "status": REJECTED,
        "reasons": [],
        "cited_events": [e["ref"] for e in cited_events],
        "broken_references": broken,
        "matching_events": [],
        "in_retrieved_candidates": None if retrieved is None else technique_id in retrieved,
    }
    reasons = result["reasons"]
    entry = catalog.get(technique_id)
    if entry is None:
        reasons.append(f"{technique_id} is not an active technique in ATT&CK {catalog.snapshot.get('attack_version')}")
        return result
    result["catalog_name"] = entry.name
    if not cited_events:
        reasons.append("no cited reference is an event in the evidence package")
        return result
    if technique_id in KNOWN_FALSE:
        reasons.append(KNOWN_FALSE[technique_id])
        return result
    if not AUTH_DATA_COMPONENTS.intersection(entry.data_components):
        reasons.append("ATT&CK lists no authentication-log data source for this technique")
        return result

    rule = SUPPORT_RULES.get(technique_id)
    cap = SUPPORTED
    if rule is None and entry.parent_id and entry.parent_id in SUPPORT_RULES:
        rule, cap = SUPPORT_RULES[entry.parent_id], SUBTECHNIQUE_CAP
        reasons.append(f"checked with the {entry.parent_id} rule; the specific sub-technique is not visible in logon records")
    if rule is None:
        status = UNCERTAIN
        reasons.append("no verification rule for this technique; cited events exist but the pattern is unchecked")
    else:
        matching = [e["ref"] for e in cited_events if rule.matches(e)]
        result["matching_events"] = matching
        if len(matching) >= rule.min_events:
            status = rule.best
            reasons.append(f"cited events show {rule.requirement} ({len(matching)} event(s))")
            if rule.best != SUPPORTED:
                reasons.append("this pattern cannot confirm the technique, only fit it")
        elif matching:
            status = UNCERTAIN
            reasons.append(f"only {len(matching)} cited event(s) show {rule.requirement}; the rule needs {rule.min_events}")
        elif rule.weaker and any(rule.weaker(e) for e in cited_events):
            status = UNCERTAIN
            reasons.append(f"cited events show a weaker form of the pattern, not {rule.requirement}")
        else:
            status = REJECTED
            reasons.append(f"no cited event shows {rule.requirement}")
    status = _cap(status, cap)
    if broken and status != REJECTED:
        status = _cap(status, UNCERTAIN)
        reasons.append(f"{len(broken)} cited reference(s) are not in the evidence package")
    result["status"] = status
    return result


def _facts(package: Mapping[str, Any], references: Iterable[str]) -> set[str]:
    """Computers, users, auth types and clock times that the given references show."""
    events = {event["ref"]: event for event in package["events"]}
    facts: set[str] = {package["user_id"]}
    for reference in references:
        if reference in events:
            e = events[reference]
            facts.update([e["source_user"], e["destination_user"], e["source_computer"],
                          e["destination_computer"], e["auth_type"], e["time"][-8:]])
        elif reference == "package:counts":
            counts = package["counts"]
            facts.update(counts["by_auth_type"])
            facts.update(item["computer"] for item in counts["top_destinations"])
        elif reference == "package:incident":
            facts.update(CLOCK.findall(package["start"] + " " + package["end"]))
        elif reference == "package:detector":
            for hour in package["detector"]["alert_hours"]:
                facts.update(CLOCK.findall(hour["window"]))
            facts.update(CLOCK.findall(package["start"] + " " + package["end"]))
        elif reference == "package:graph_context" and package.get("graph_context"):
            facts.update(edge["destination_computer"] for edge in package["graph_context"]["top_edges"])
    return facts


def mentioned(text: str) -> set[str]:
    found = set(COMPUTER.findall(text)) | set(USER.findall(text)) | set(CLOCK.findall(text))
    found.update(kind for kind in AUTH_TYPES if re.search(rf"\b{kind}\b", text, re.IGNORECASE))
    return found


def _normalise(values: Iterable[str]) -> set[str]:
    return {value.lower() for value in values}


def verify_claim(claim: Mapping[str, Any], package: Mapping[str, Any]) -> dict[str, Any]:
    """Citation and entity check for one observation or interpretation."""
    valid = package_references(package)
    cited, broken = _split_references(claim.get("evidence") or [], valid)
    names = mentioned(claim.get("text", ""))
    in_cited = _normalise(_facts(package, cited))
    in_package = _normalise(_facts(package, valid))
    not_cited = sorted(name for name in names if name.lower() not in in_cited)
    not_in_package = sorted(name for name in not_cited if name.lower() not in in_package)
    reasons = []
    if not cited:
        status = "unsupported"
        reasons.append("no valid citation")
    elif not_in_package:
        status = "unsupported"
        reasons.append(f"names {', '.join(not_in_package)}, which the evidence package does not contain")
    elif not_cited or broken:
        status = "partly_supported"
        if not_cited:
            reasons.append(f"{', '.join(not_cited)} not in the cited evidence (but elsewhere in the package)")
        if broken:
            reasons.append(f"{len(broken)} cited reference(s) are not in the package")
    else:
        status = "supported"
    return {"status": status, "reasons": reasons, "broken_references": broken}


def verify_response(
    response: Mapping[str, Any],
    package: Mapping[str, Any],
    catalog: TechniqueCatalog,
    retrieved: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verifier decisions for every part of one generated summary."""
    summary_names = mentioned(response.get("summary", ""))
    unknown = sorted(n for n in summary_names if n.lower() not in _normalise(_facts(package, package_references(package))))
    return {
        "summary": {"status": "unsupported" if unknown else "supported",
                    "reasons": [f"names {', '.join(unknown)}, which the evidence package does not contain"] if unknown else []},
        "observations": [verify_claim(item, package) for item in response["observations"]],
        "interpretations": [verify_claim(item, package) for item in response["interpretations"]],
        "techniques": [verify_technique(item, package, catalog, retrieved) for item in response["techniques"]],
    }


def mapping_outcome(statuses: Iterable[str]) -> str:
    statuses = set(statuses)
    if SUPPORTED in statuses:
        return "supported_mapping"
    if UNCERTAIN in statuses:
        return "uncertain_mapping"
    return "no_supported_mapping"


def apply_verification(record: Mapping[str, Any], checks: Mapping[str, Any]) -> dict[str, Any]:
    """The verified view of a generated record: rejected techniques and unsupported claims removed.

    Removed items are kept under ``removed_*`` with the reasons, so the
    evaluation can count what verification took out.
    """
    response = record["response"]
    verified: dict[str, Any] = {
        "summary": response["summary"],
        "summary_check": checks["summary"],
        "uncertainty": response["uncertainty"],
        "model_no_supported_mapping": response["no_supported_mapping"],
    }
    for section in ("observations", "interpretations"):
        kept, removed = [], []
        for item, check in zip(response[section], checks[section]):
            (removed if check["status"] == "unsupported" else kept).append({**item, "check": check})
        verified[section] = kept
        verified[f"removed_{section}"] = removed
    kept, removed = [], []
    for item, check in zip(response["techniques"], checks["techniques"]):
        entry = {**item, "status": check["status"], "verification": check}
        (removed if check["status"] == REJECTED else kept).append(entry)
    verified["techniques"] = sorted(kept, key=lambda t: -_ORDER[t["status"]])
    verified["removed_techniques"] = removed
    verified["outcome"] = mapping_outcome(t["status"] for t in kept)
    return verified
