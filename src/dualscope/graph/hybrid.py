"""Side-by-side GAE novelty and confirmed-relationship alert packaging."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from dualscope.graph.threat_history import ThreatHistory


def collect_relationship_matches(
    events: Iterable[Mapping[str, object]], history: ThreatHistory
) -> dict[str, list[dict[str, object]]]:
    """Collect exact confirmed relationships, retaining event evidence."""
    matches: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if not history.matches(event):
            continue
        user = str(event["acting_user"])
        matches[user].append({
            "timestamp": int(event["timestamp"]),
            "source_computer": str(event["source_computer"]),
            "destination_computer": str(event["destination_computer"]),
            "source_reference": str(event.get("source_reference") or ""),
        })
    return dict(matches)


def combine_user_day(
    graph_record: Mapping[str, object], relationship_matches: list[dict[str, object]]
) -> dict[str, object]:
    """Package separate model signals without changing their semantics."""
    relationship_alert = bool(relationship_matches)
    gae_alert = bool(graph_record.get("is_alert"))
    if relationship_alert:
        priority = "confirmed_relationship"
    elif gae_alert:
        priority = "graph_anomaly_review"
    else:
        priority = "none"
    return {
        "dataset_day": int(graph_record["dataset_day"]),
        "user_id": str(graph_record["user_id"]),
        "window_start": int(graph_record["window_start"]),
        "window_end": int(graph_record["window_end"]),
        "score_available_at": int(graph_record["score_available_at"]),
        "gae_model_version": str(graph_record["model_version"]),
        "gae_score": graph_record.get("score"),
        "gae_is_alert": gae_alert,
        "base_gae_score": graph_record.get("base_gae_score"),
        "base_gae_is_alert": graph_record.get("base_gae_is_alert"),
        "peak_unique_destinations_300s": graph_record.get(
            "peak_unique_destinations_300s"
        ),
        "confirmed_relationship_alert": relationship_alert,
        "relationship_match_count": len(relationship_matches),
        "relationship_matches": relationship_matches,
        "any_graph_signal": gae_alert or relationship_alert,
        "priority_reason": priority,
        "primary_high_confidence_alert": relationship_alert,
    }
