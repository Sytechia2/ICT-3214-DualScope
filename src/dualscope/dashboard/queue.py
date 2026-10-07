"""Queue-level facts for the dashboard: why each incident was flagged, and the day cut-off.

Behaviours and ATT&CK candidates come from the Task 6.1 templates
(``dualscope.attack.queries``) applied to each incident's events, so the
dashboard says "why flagged" in the same terms the investigation step uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from dualscope.attack.queries import retrieve_candidates
from dualscope.attack.retrieval import TechniqueRetriever
from dualscope.dashboard.data import IncidentView
from dualscope.dashboard.evidence import incident_events

# Short chip labels and the order a title picks its headline behaviour from.
CHIP_LABELS = {
    "ntlm_logon_to_new_destination": "NTLM to new host",
    "failed_logons": "Failed logons",
    "network_logon_to_new_destination": "New host",
    "logon_from_new_source": "New source",
}
HEADLINES = {
    "ntlm_logon_to_new_destination": "NTLM logon to a new host",
    "failed_logons": "Failed logons",
    "network_logon_to_new_destination": "Network logon to a new host",
    "logon_from_new_source": "Logon from a new source computer",
}
NO_EVIDENCE = "Not enough evidence"
NO_EVENTS = "Event details unavailable"


@dataclass(frozen=True)
class IncidentSummary:
    incident: IncidentView
    behaviours: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    events_available: bool = True

    @property
    def incident_id(self) -> str:
        return self.incident.incident_id

    @property
    def day(self) -> int:
        return int(self.incident.raw.get("dataset_day") or 1 + (self.incident.start_time - 1) // 86_400)

    @property
    def rank(self) -> int | None:
        rank = self.incident.raw.get("best_rank_in_day")
        return int(rank) if rank is not None else None

    @property
    def above_cutoff(self) -> bool:
        """HIGH incidents have an hour above the day's cut-off; MEDIUM ones were all tied."""
        return self.incident.priority != "MEDIUM"

    @property
    def event_count(self) -> int:
        return int(self.incident.raw.get("evidence_count") or len(self.incident.raw.get("source_references") or []))

    @property
    def chips(self) -> list[str]:
        if not self.events_available:
            return []
        return [CHIP_LABELS[b["behaviour"]] for b in self.ordered_behaviours]

    @property
    def ordered_behaviours(self) -> list[dict[str, Any]]:
        order = list(HEADLINES)
        return sorted(self.behaviours, key=lambda b: order.index(b["behaviour"]))

    @property
    def headline(self) -> str:
        if not self.events_available:
            return NO_EVENTS
        behaviours = self.ordered_behaviours
        return HEADLINES[behaviours[0]["behaviour"]] if behaviours else NO_EVIDENCE

    @property
    def top_candidate(self) -> dict[str, Any] | None:
        return self.candidates[0] if self.candidates else None

    @property
    def redteam(self) -> bool | None:
        value = self.incident.raw.get("ground_truth_redteam")
        return bool(value) if value is not None else None


def summarise_incidents(
    incidents: Iterable[IncidentView],
    events: pd.DataFrame | None,
    retriever: TechniqueRetriever,
) -> list[IncidentSummary]:
    """Behaviours and ATT&CK candidates per incident; without events, both stay empty."""
    summaries = []
    for incident in incidents:
        if events is None:
            summaries.append(IncidentSummary(incident, events_available=False))
            continue
        rows, _ = incident_events(events, incident.raw)
        result = retrieve_candidates(rows, retriever) if not rows.empty else {"behaviours": [], "candidates": []}
        summaries.append(IncidentSummary(incident, result["behaviours"], result["candidates"]))
    return summaries


def day_cutoff_counts(incidents: Iterable[IncidentView]) -> pd.DataFrame:
    """Per day, how many queued alert hours scored above the cut-off and how many were tied."""
    rows = [
        {"day": 1 + (hour["window_start"] - 1) // 86_400, "tied": bool(hour["tied_at_cutoff"])}
        for incident in incidents
        for hour in incident.raw.get("alert_hours") or []
    ]
    if not rows:
        return pd.DataFrame(columns=["day", "above", "tied"])
    frame = pd.DataFrame(rows)
    counts = frame.groupby("day")["tied"].agg(tied="sum", total="size")
    counts["above"] = counts["total"] - counts["tied"]
    return counts.reset_index()[["day", "above", "tied"]].astype(int)


def behaviour_totals(summaries: Iterable[IncidentSummary]) -> pd.DataFrame:
    """How many incidents show each behaviour, plus those with nothing to cite."""
    summaries = [s for s in summaries if s.events_available]
    counts = {headline: 0 for headline in HEADLINES.values()}
    none = 0
    for summary in summaries:
        if not summary.behaviours:
            none += 1
        for behaviour in summary.behaviours:
            counts[HEADLINES[behaviour["behaviour"]]] += 1
    ordered = sorted(counts.items(), key=lambda item: -item[1]) + [(NO_EVIDENCE, none)]
    return pd.DataFrame(ordered, columns=["behaviour", "incidents"])


def matches_search(summary: IncidentSummary, query: str, events: pd.DataFrame | None) -> bool:
    """User ID, incident ID, a host in the incident's events, or one of its event IDs."""
    query = query.strip().casefold()
    if not query:
        return True
    if query in summary.incident.user_id.casefold() or query in summary.incident_id.casefold():
        return True
    references = summary.incident.raw.get("source_references") or []
    if any(query == reference.casefold() for reference in references):
        return True
    if events is None:
        return False
    rows, _ = incident_events(events, summary.incident.raw)
    hosts = pd.concat([rows["source_computer"], rows["destination_computer"]]).str.casefold()
    return bool((hosts == query).any())
