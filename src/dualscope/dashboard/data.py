"""Load and prepare Task 5.4 incident exports without running detector code."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_FIXTURE = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "incidents_mock.jsonl"
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
SORT_OPTIONS = (
    "Priority (highest first)",
    "Newest first",
    "Oldest first",
    "Fused score (highest first)",
)
DETECTOR_FILTERS = ("Any", "Sequence alert", "Graph alert", "Both alert", "Neither alerts")


class IncidentDataError(ValueError):
    """A selected incident file cannot be displayed safely."""


@dataclass(frozen=True)
class DetectorSummary:
    max_score: float | None
    any_alert: bool | None


@dataclass(frozen=True)
class IncidentView:
    incident_id: str
    user_id: str
    start_time: int
    end_time: int
    priority: str
    max_fused_score: float
    sequence: DetectorSummary
    graph: DetectorSummary
    raw: Mapping[str, Any]


def _nonempty_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise IncidentDataError(f"{field} must be a non-empty string")
    return value


def _score(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IncidentDataError(f"{field} must be a number between 0 and 1")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise IncidentDataError(f"{field} must be a finite number between 0 and 1")
    return score


def _detector(record: Mapping[str, Any], name: str) -> DetectorSummary:
    scores = record.get("detector_scores")
    if not isinstance(scores, dict) or not isinstance(scores.get(name), dict):
        raise IncidentDataError(f"detector_scores.{name} must be an object")
    detector = scores[name]
    if "max_score" not in detector or "any_alert" not in detector:
        raise IncidentDataError(f"detector_scores.{name} requires max_score and any_alert")
    alert = detector["any_alert"]
    if alert is not None and not isinstance(alert, bool):
        raise IncidentDataError(f"detector_scores.{name}.any_alert must be true, false, or null")
    return DetectorSummary(
        max_score=_score(detector["max_score"], f"detector_scores.{name}.max_score", nullable=True),
        any_alert=alert,
    )


def parse_incident(record: Any) -> IncidentView:
    """Validate the fields needed by navigation and retain the full export record."""
    if not isinstance(record, dict):
        raise IncidentDataError("incident must be a JSON object")
    incident_id = _nonempty_string(record, "incident_id")
    user_id = _nonempty_string(record, "user_id")
    start = record.get("start_time")
    end = record.get("end_time")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end <= start
    ):
        raise IncidentDataError("start_time and end_time must form a positive half-open range")
    priority = record.get("priority")
    if not isinstance(priority, str) or priority not in PRIORITY_ORDER:
        raise IncidentDataError(f"priority must be one of {', '.join(PRIORITY_ORDER)}")
    max_fused_score = _score(record.get("max_fused_score"), "max_fused_score")
    return IncidentView(
        incident_id=incident_id,
        user_id=user_id,
        start_time=start,
        end_time=end,
        priority=priority,
        max_fused_score=max_fused_score,
        sequence=_detector(record, "sequence"),
        graph=_detector(record, "graph"),
        raw=record,
    )


def load_incidents(path: Path) -> list[IncidentView]:
    """Read a full Task 5.4 JSONL export, preserving file order and source data."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".jsonl":
        raise IncidentDataError("incident source must be a .jsonl file")
    incidents: list[IncidentView] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IncidentDataError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
                try:
                    incident = parse_incident(raw)
                except IncidentDataError as exc:
                    raise IncidentDataError(f"line {line_number}: {exc}") from exc
                if incident.incident_id in seen_ids:
                    raise IncidentDataError(f"line {line_number}: duplicate incident_id {incident.incident_id}")
                seen_ids.add(incident.incident_id)
                incidents.append(incident)
    except (OSError, UnicodeError) as exc:
        raise IncidentDataError(f"cannot read {path}: {exc}") from exc
    return incidents


def filter_sort_incidents(
    incidents: Iterable[IncidentView],
    *,
    user_query: str = "",
    priorities: Iterable[str] = PRIORITY_ORDER,
    start_time: int | None = None,
    end_time: int | None = None,
    detector_filter: str = "Any",
    sort_by: str = SORT_OPTIONS[0],
) -> list[IncidentView]:
    """Filter by overlapping half-open time ranges and sort deterministically."""
    if detector_filter not in DETECTOR_FILTERS:
        raise ValueError(f"unknown detector filter: {detector_filter}")
    if sort_by not in SORT_OPTIONS:
        raise ValueError(f"unknown sort order: {sort_by}")
    wanted_priorities = set(priorities)
    query = user_query.casefold().strip()
    result = []
    for incident in incidents:
        sequence_alert = incident.sequence.any_alert is True
        graph_alert = incident.graph.any_alert is True
        if query not in incident.user_id.casefold() or incident.priority not in wanted_priorities:
            continue
        if start_time is not None and incident.end_time <= start_time:
            continue
        if end_time is not None and incident.start_time >= end_time:
            continue
        if detector_filter == "Sequence alert" and not sequence_alert:
            continue
        if detector_filter == "Graph alert" and not graph_alert:
            continue
        if detector_filter == "Both alert" and not (sequence_alert and graph_alert):
            continue
        if detector_filter == "Neither alerts" and (sequence_alert or graph_alert):
            continue
        result.append(incident)
    if sort_by == "Newest first":
        return sorted(result, key=lambda item: (-item.start_time, item.incident_id))
    if sort_by == "Oldest first":
        return sorted(result, key=lambda item: (item.start_time, item.incident_id))
    if sort_by == "Fused score (highest first)":
        return sorted(result, key=lambda item: (-item.max_fused_score, -item.start_time, item.incident_id))
    return sorted(result, key=lambda item: (PRIORITY_ORDER[item.priority], -item.start_time, item.incident_id))


def format_dataset_second(second: int) -> str:
    """Render dataset-relative seconds without implying a calendar date or timezone."""
    day, seconds_into_day = divmod(second - 1, 86_400)
    hours, remainder = divmod(seconds_into_day, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"Day {day + 1} {hours:02d}:{minutes:02d}:{seconds:02d}"


def format_score(score: float | None) -> str:
    return "Unavailable" if score is None else f"{score:.4f}"


def format_alert(alert: bool | None) -> str:
    return "Unavailable" if alert is None else "Alert" if alert else "No alert"
