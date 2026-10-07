"""Prepare an incident's event evidence for the Task 7.2 panels.

Events come from the ``events.parquet`` written next to ``incidents.jsonl`` by
``scripts/export_alert_handoff.py`` (docs/alert_handoff.md). An incident's
events are the rows whose ``source_reference`` is in its
``source_references``. Exports without an events file still show their
references, just not the event details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dualscope.dashboard.data import format_dataset_second

EVENTS_FILE = "events.parquet"
UNKNOWN = "unknown"  # LANL writes "?" for a missing authentication or logon type
EVENT_COLUMNS = (
    "source_reference", "timestamp", "source_user", "destination_user", "source_computer",
    "destination_computer", "authentication_type", "logon_type", "authentication_orientation",
    "authentication_result", "is_new_user_destination", "is_new_host_connection", "is_new_user_source",
)


class EvidenceDataError(ValueError):
    """An events file exists but cannot be used."""


def events_path_for(incidents_path: Path) -> Path:
    return Path(incidents_path).expanduser().parent / EVENTS_FILE


def load_events(path: Path) -> pd.DataFrame:
    """Read an events file, indexed by ``source_reference``."""
    try:
        events = pd.read_parquet(path)
    except Exception as exc:  # pyarrow raises several types for unreadable files
        raise EvidenceDataError(f"cannot read {path}: {exc}") from exc
    missing = [column for column in EVENT_COLUMNS if column not in events.columns]
    if missing:
        raise EvidenceDataError(f"{path} is missing columns: {', '.join(missing)}")
    if events["source_reference"].duplicated().any():
        raise EvidenceDataError(f"{path} has repeated source_reference values")
    events.index = pd.Index(events["source_reference"].to_numpy())
    return events


def incident_events(events: pd.DataFrame, incident: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """The incident's events in time order, and any references the file lacks."""
    references = list(dict.fromkeys(incident.get("source_references") or []))
    present = [reference for reference in references if reference in events.index]
    missing = [reference for reference in references if reference not in events.index]
    rows = events.loc[present].sort_values(["timestamp", "source_reference"], kind="stable")
    return rows.reset_index(drop=True), missing


def _novel(rows: pd.DataFrame, column: str) -> pd.Series:
    # Log-offs are recorded on the machine being left, so their novelty flags
    # do not show a new logon path (same rule as docs/attack_retrieval.md).
    return rows[column].astype(bool) & (rows["authentication_orientation"] != "LogOff")


def event_flags(rows: pd.DataFrame) -> pd.Series:
    labels = [
        (_novel(rows, "is_new_user_destination"), "new destination for user"),
        (_novel(rows, "is_new_host_connection"), "new host pair"),
        (_novel(rows, "is_new_user_source"), "new source for user"),
        (rows["authentication_result"] == "Fail", "failed"),
        (rows["authentication_type"] == "NTLM", "NTLM"),
    ]
    return pd.Series(
        [", ".join(label for mask, label in labels if mask.iloc[i]) for i in range(len(rows))],
        index=rows.index, dtype=object,
    )


def timeline_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One display row per event, in time order."""
    return pd.DataFrame({
        "Time": [format_dataset_second(int(t)) for t in rows["timestamp"]],
        "Event ID": rows["source_reference"],
        "Source": rows["source_computer"],
        "Destination": rows["destination_computer"],
        "Auth type": rows["authentication_type"].replace("?", UNKNOWN),
        "Logon type": rows["logon_type"].replace("?", UNKNOWN),
        "Orientation": rows["authentication_orientation"],
        "Result": rows["authentication_result"],
        "Flags": event_flags(rows),
    })


def host_pair_table(rows: pd.DataFrame) -> pd.DataFrame:
    """User-host relationships: one row per source -> destination pair, new pairs first."""
    if rows.empty:
        return pd.DataFrame(columns=[
            "Source", "Destination", "Events", "Failures", "Auth types",
            "New destination for user", "New host pair", "New source for user", "First event", "First seen",
        ])
    frame = rows.assign(
        failed=rows["authentication_result"] == "Fail",
        new_destination=_novel(rows, "is_new_user_destination"),
        new_pair=_novel(rows, "is_new_host_connection"),
        new_source=_novel(rows, "is_new_user_source"),
    )
    grouped = frame.groupby(["source_computer", "destination_computer"], sort=False)
    table = pd.DataFrame({
        "Events": grouped.size(),
        "Failures": grouped["failed"].sum().astype(int),
        "Auth types": grouped["authentication_type"].agg(lambda s: ", ".join(sorted({UNKNOWN if v == "?" else v for v in s}))),
        "New destination for user": grouped["new_destination"].any(),
        "New host pair": grouped["new_pair"].any(),
        "New source for user": grouped["new_source"].any(),
        "First event": grouped["source_reference"].first(),
        "first": grouped["timestamp"].min(),
    }).reset_index().rename(columns={"source_computer": "Source", "destination_computer": "Destination"})
    novelty = table[["New destination for user", "New host pair", "New source for user"]].any(axis=1)
    table = table.assign(_novel=novelty).sort_values(["_novel", "first"], ascending=[False, True], kind="stable")
    table["First seen"] = [format_dataset_second(int(t)) for t in table["first"]]
    return table.drop(columns=["_novel", "first"]).reset_index(drop=True)


def alert_hours_table(incident: Mapping[str, Any], show_answer_key: bool = False) -> pd.DataFrame:
    """The incident's alerted hours with the scores the final model used."""
    rows = []
    for hour in incident.get("alert_hours") or []:
        row = {
            "Alert ID": hour["alert_id"],
            "Hour [start, end)": f"[{format_dataset_second(hour['window_start'])}, {format_dataset_second(hour['window_end'])})",
            "Rank in day": hour["rank_in_day"],
            "Tied at cut-off": hour["tied_at_cutoff"],
            "Fusion score": round(hour["fusion_score"], 6),
            "GRU score (raw)": round(hour["gru_max_event"], 4),
            "GRU percentile in day": round(hour["gru_percentile_in_day"], 4),
            "Events": hour["counts"]["n_events"],
        }
        if show_answer_key:
            row["Red-team (answer key)"] = hour["ground_truth_redteam"]
        rows.append(row)
    return pd.DataFrame(rows)


def graph_edges_table(incident: Mapping[str, Any]) -> pd.DataFrame:
    edges = (incident.get("graph_context") or {}).get("top_edges") or []
    return pd.DataFrame([
        {
            "Day": edge["dataset_day"],
            "Destination": edge["destination_computer"],
            "Edge score (raw)": round(edge["edge_raw_score"], 4),
            "New edge": edge["is_new_edge"],
            "Successes": edge["success_count"],
            "Failures": edge["failure_count"],
            "Evidence events": len(edge["source_references"]),
            "References truncated": edge["references_truncated"],
        }
        for edge in edges
    ])
