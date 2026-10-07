"""Evidence packages: what the LLM is allowed to see about one incident (Task 6.2).

A package is built from the alert handoff (docs/alert_handoff.md), never from
the raw dataset. It holds the incident's detector scores, the hourly counts,
the graph detector's same-day view (marked as context), the behaviours the
Task 6.1 rules found, counts over all of the incident's events, and a
selection of individual events with their ``auth.txt`` references.

Event selection, so large incidents fit the context window: events matched by
a behaviour rule come first, taken in turn from each behaviour in time order
(at most ``MAX_FLAGGED_EVENTS``), then an evenly spaced time sample of the
other events fills the package up to ``MAX_PACKAGE_EVENTS``. The counts always
cover every event, so the model can see how much was left out.

Fields are copied from an explicit list. The answer key
(``ground_truth_redteam``) is never copied, and ``assert_no_answer_key``
checks every package and prompt before it is sent.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dualscope.attack.queries import TEMPLATES, observed_behaviours

MAX_PACKAGE_EVENTS = 60
MAX_FLAGGED_EVENTS = 40
TOP_DESTINATIONS = 5
TOP_GRAPH_EDGES = 5

# Citations that point at package sections rather than single events.
PACKAGE_REFERENCES = ("package:incident", "package:counts", "package:detector", "package:graph_context")
FORBIDDEN_MARKERS = ("ground_truth", "redteam", "red_team", "red-team")


class AnswerKeyLeak(ValueError):
    """An evidence package or prompt contains the evaluation answer key."""


def dataset_time(second: int) -> str:
    """Dataset-relative seconds as ``Day N HH:MM:SS`` (no calendar date exists)."""
    day, into_day = divmod(int(second) - 1, 86_400)
    hours, rest = divmod(into_day, 3_600)
    minutes, seconds = divmod(rest, 60)
    return f"Day {day + 1} {hours:02d}:{minutes:02d}:{seconds:02d}"


def _not_logoff(rows: pd.DataFrame) -> pd.Series:
    return rows["authentication_orientation"] != "LogOff"


def event_flags(row: Mapping[str, Any]) -> list[str]:
    """Plain-language flags for one event; log-offs never count as new paths."""
    flags = []
    if row["authentication_orientation"] != "LogOff":
        if row["is_new_user_destination"]:
            flags.append("first logon by this user to this destination")
        if row["is_new_host_connection"]:
            flags.append("first connection between these two computers")
        if row["is_new_user_source"]:
            flags.append("first event by this user from this source")
    if row["authentication_result"] == "Fail":
        flags.append("failed")
    return flags


def _event_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ref": row["source_reference"],
        "time": dataset_time(row["timestamp"]),
        "timestamp": int(row["timestamp"]),
        "source_user": row["source_user"],
        "destination_user": row["destination_user"],
        "source_computer": row["source_computer"],
        "destination_computer": row["destination_computer"],
        "auth_type": row["authentication_type"],
        "logon_type": row["logon_type"],
        "orientation": row["authentication_orientation"],
        "result": row["authentication_result"],
        "is_new_user_destination": bool(row["is_new_user_destination"]),
        "is_new_host_connection": bool(row["is_new_host_connection"]),
        "is_new_user_source": bool(row["is_new_user_source"]),
        "flags": event_flags(row),
    }


def select_events(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The events to show the model, in time order, and how they were chosen."""
    ordered = rows.sort_values(["timestamp", "source_reference"], kind="stable").reset_index(drop=True)
    queues = []
    for template in TEMPLATES:
        matched = template.matches(ordered).fillna(False).astype(bool)
        queues.append(list(ordered.index[matched.to_numpy()]))
    flagged: list[int] = []
    taken: set[int] = set()
    position = 0
    while len(flagged) < MAX_FLAGGED_EVENTS and any(position < len(q) for q in queues):
        for queue in queues:
            if position < len(queue) and queue[position] not in taken and len(flagged) < MAX_FLAGGED_EVENTS:
                flagged.append(queue[position])
                taken.add(queue[position])
        position += 1
    all_flagged = set().union(*queues) if queues else set()
    others = [index for index in ordered.index if index not in all_flagged]
    room = MAX_PACKAGE_EVENTS - len(flagged)
    if len(others) <= room:
        sampled = others
    else:
        sampled = [others[i] for i in np.unique(np.linspace(0, len(others) - 1, room).round().astype(int))]
    chosen = sorted(set(flagged) | set(sampled))
    selection = {
        "total_events": len(ordered),
        "rule_matched_events": len(all_flagged),
        "shown_rule_matched": len(flagged),
        "shown_other_sampled": len(sampled),
        "shown_events": len(chosen),
        "truncated": len(chosen) < len(ordered),
        "rule": (f"Up to {MAX_FLAGGED_EVENTS} events matched by a behaviour rule (taken in turn from each "
                 f"behaviour, in time order), then an evenly spaced time sample of the other events, "
                 f"{MAX_PACKAGE_EVENTS} events at most."),
    }
    return ordered.loc[chosen], selection


def event_counts(rows: pd.DataFrame) -> dict[str, Any]:
    """Counts over all of an incident's events (not just the shown ones)."""
    logons = rows[_not_logoff(rows)]
    destinations = Counter(logons["destination_computer"])
    return {
        "events": len(rows),
        "by_auth_type": dict(Counter(rows["authentication_type"]).most_common()),
        "by_logon_type": dict(Counter(rows["logon_type"]).most_common()),
        "by_orientation": dict(Counter(rows["authentication_orientation"]).most_common()),
        "failed": int((rows["authentication_result"] == "Fail").sum()),
        "source_computers": int(rows["source_computer"].nunique()),
        "destination_computers": int(rows["destination_computer"].nunique()),
        "first_logons_to_destination": int(logons["is_new_user_destination"].sum()),
        "first_host_connections": int(logons["is_new_host_connection"].sum()),
        "first_events_from_source": int(logons["is_new_user_source"].sum()),
        "top_destinations": [{"computer": c, "events": n} for c, n in destinations.most_common(TOP_DESTINATIONS)],
        "note": "'?' means LANL did not record the value. Log-off events are excluded from the 'first' counts.",
    }


def detector_section(incident: Mapping[str, Any]) -> dict[str, Any]:
    hours = []
    for hour in incident.get("alert_hours") or []:
        hours.append({
            "alert_id": hour["alert_id"],
            "window": f"{dataset_time(hour['window_start'])} to {dataset_time(hour['window_end'])}",
            "fusion_score": round(float(hour["fusion_score"]), 4),
            "rank_in_day": hour.get("rank_in_day"),
            "tied_at_cutoff": bool(hour.get("tied_at_cutoff")),
            "gru_percentile_in_day": round(float(hour["gru_percentile_in_day"]), 4),
        })
    return {
        "priority": incident.get("priority"),
        "alert_hours": hours,
        "notes": ("The final model flags the 38 highest-scoring user-hours per day. fusion_score ranks "
                  "user-hours; it is not an attack probability. HIGH = scored above the day's cut-off; "
                  "MEDIUM = tied at the cut-off and queued by a fixed tie-break. gru_percentile_in_day is "
                  "how unusual the event sequence was compared with other users that day."),
    }


def graph_section(incident: Mapping[str, Any]) -> dict[str, Any] | None:
    context = incident.get("graph_context")
    scores = (incident.get("detector_scores") or {}).get("graph") or {}
    if not context:
        return None
    edges = [{
        "destination_computer": edge["destination_computer"],
        "new_edge": bool(edge["is_new_edge"]),
        "successes": int(edge["success_count"]),
        "failures": int(edge["failure_count"]),
    } for edge in (context.get("top_edges") or [])[:TOP_GRAPH_EDGES]]
    return {
        "user_day_score_percentile": scores.get("max_score"),
        "graph_alert_same_day": scores.get("any_alert"),
        "new_edges_that_day": context.get("new_edge_count"),
        "degree_growth": context.get("degree_growth"),
        "top_edges": edges,
        "notes": ("Context only: the graph detector's view of the whole user-day. It is not part of the "
                  "final model, and its edges are not limited to the alerted hours."),
    }


def build_package(incident: Mapping[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    """The evidence package for one incident; ``rows`` are all of its events."""
    shown, selection = select_events(rows)
    behaviours = [{
        "behaviour": b["behaviour"],
        "description": b["description"],
        "event_count": b["event_count"],
        "example_refs": b["evidence_references"][:5],
    } for b in observed_behaviours(rows)]
    package = {
        "incident_id": incident["incident_id"],
        "user_id": incident["user_id"],
        "dataset_day": incident.get("dataset_day"),
        "start": dataset_time(incident["start_time"]),
        "end": dataset_time(incident["end_time"]),
        "duration_hours": incident.get("duration_hours"),
        "detector": detector_section(incident),
        "graph_context": graph_section(incident),
        "counts": event_counts(rows),
        "behaviours": behaviours,
        "selection": selection,
        "events": [_event_record(row) for row in shown.to_dict("records")],
    }
    assert_no_answer_key(package)
    return package


def package_references(package: Mapping[str, Any]) -> set[str]:
    """Every citation the model may use for this package."""
    references = {event["ref"] for event in package["events"]}
    references.update(ref for ref in PACKAGE_REFERENCES
                      if ref == "package:incident" or package.get(ref.split(":", 1)[1]))
    return references


def assert_no_answer_key(value: Any, where: str = "package") -> None:
    """Raise if any key or text in ``value`` mentions the evaluation answer key."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_answer_key(key, f"{where}.{key}")
            assert_no_answer_key(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_answer_key(item, f"{where}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in FORBIDDEN_MARKERS:
            if marker in lowered:
                raise AnswerKeyLeak(f"{where} mentions {marker!r}")
