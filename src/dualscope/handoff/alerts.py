"""Daily alert queue and incident records from the final model's test scores.

The final model (docs/supervised_fusion.md) was tested once on days 17-30 at a
budget of 38 alerts per day. These helpers rebuild exactly those 532 alerts
from ``final_test_scores.parquet`` and group them into per-user incidents for
the investigation (6.x) and dashboard (7.x) work. They only read stored scores;
nothing is re-scored.

Row order matters: the score file is stored in the test's fixed seed-0 order,
and a stable sort on it breaks score ties exactly as the test did.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HOUR_SECONDS = 3_600
COUNT_COLUMNS = [
    "n_events", "n_failures", "n_new_user_source", "n_new_host_connection", "n_new_user_destination",
    "n_ntlm", "n_network_logon", "n_logon", "n_sources", "n_destinations",
]
# Hours scored above the day's cut-off are in the queue whatever the tie-break;
# hours tied at the cut-off were picked by the fixed seed-0 order.
PRIORITY_ABOVE_CUTOFF = "HIGH"
PRIORITY_TIED_AT_CUTOFF = "MEDIUM"
FUSION_METHOD = "supervised_hist_gradient_boosting"
EVENT_COLUMNS = [
    "source_reference", "source_line", "timestamp", "acting_user", "source_user", "destination_user",
    "source_computer", "destination_computer", "authentication_type", "logon_type",
    "authentication_orientation", "authentication_result", "is_new_user_source",
    "is_new_host_connection", "is_new_user_destination", "prior_auth_count_1h",
    "prior_failure_count_1h", "prior_unique_destinations_24h",
]
GRAPH_COLUMNS = [
    "user_id", "dataset_day", "status", "score", "is_alert", "alert_threshold", "n_edges",
    "new_edge_count", "prior_degree", "current_degree", "degree_growth", "evidence_nodes", "top_edges",
]


def clean_user(user: str) -> str:
    return user.replace("@", "_").replace("$", "")


def select_daily_alerts(scores: pd.DataFrame, budget: int = 38) -> pd.DataFrame:
    """Top ``budget`` user-hours per day by ``fusion``, as the final test chose them.

    ``scores`` must be in its stored order. Adds ``rank_in_day`` (1 = highest),
    ``day_cutoff_score`` (the lowest score in that day's queue),
    ``tied_at_cutoff`` and ``gru_percentile_in_day`` (share of that day's
    user-hours with a GRU score at or below this one).
    """
    frame = scores.copy()
    frame["gru_percentile_in_day"] = frame.groupby("day")["gru_max_event"].rank(method="max", pct=True)
    ordered = frame.sort_values(["day", "fusion"], ascending=[True, False], kind="stable")
    top = ordered.groupby("day", sort=False).head(budget).copy()
    top["rank_in_day"] = top.groupby("day", sort=False).cumcount() + 1
    top["day_cutoff_score"] = top.groupby("day", sort=False)["fusion"].transform("min")
    top["tied_at_cutoff"] = top["fusion"] == top["day_cutoff_score"]
    top["window_start"] = top["hour"].astype(np.int64)
    top["window_end"] = top["window_start"] + HOUR_SECONDS
    top["alert_id"] = [
        f"ALR-D{day:02d}-{rank:02d}" for day, rank in zip(top["day"], top["rank_in_day"])
    ]
    return top.reset_index(drop=True)


def group_incidents(alerts: pd.DataFrame, max_gap_seconds: int = 7_200, id_prefix: str = "INC-TEST") -> pd.Series:
    """Incident ID for each alert: one user's alert hours with gaps up to ``max_gap_seconds``.

    The gap rule matches ``IncidentConfig.max_merge_gap_seconds`` (Task 5.4).
    IDs follow the Task 5.4 pattern ``<id_prefix>-D<first day>-<user>-<seq>``.
    """
    incident_ids = pd.Series(index=alerts.index, dtype=object)
    for user, rows in alerts.sort_values("window_start").groupby("user", sort=True):
        sequence = 0
        previous_end: int | None = None
        current_id = ""
        for index, row in rows.iterrows():
            if previous_end is None or row["window_start"] - previous_end > max_gap_seconds:
                sequence += 1
                current_id = f"{id_prefix}-D{int(row['day']):02d}-{clean_user(user)}-{sequence:03d}"
            incident_ids[index] = current_id
            previous_end = int(row["window_end"])
    return incident_ids


def alert_events(dataset: ds.Dataset, alerts: pd.DataFrame, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Every authentication event behind each alert, tagged with its ``alert_id`` (sorted by alert, time, line)."""
    parts = []
    for day, rows in alerts.groupby("day", sort=True):
        started = time.time()
        table = dataset.to_table(
            columns=EVENT_COLUMNS,
            filter=(ds.field("dataset_day") == int(day)) & ds.field("acting_user").isin(sorted(set(rows["user"]))),
        )
        events = table.to_pandas()
        events["window_start"] = 1 + ((events["timestamp"] - 1) // HOUR_SECONDS) * HOUR_SECONDS
        events = events.merge(
            rows[["user", "window_start", "alert_id"]].rename(columns={"user": "acting_user"}),
            on=["acting_user", "window_start"], how="inner", validate="m:1",
        )
        parts.append(events.drop(columns=["window_start"]))
        log(f"  events: day {day}, {len(events):,} rows ({time.time() - started:.0f}s)")
    events = pd.concat(parts, ignore_index=True)
    return events.sort_values(["alert_id", "timestamp", "source_line"], kind="stable").reset_index(drop=True)


def _graph_summary(graph_rows: pd.DataFrame) -> dict[str, Any]:
    scored = graph_rows[graph_rows["status"] == "available"] if "status" in graph_rows else graph_rows
    scores = scored["score"].dropna()
    alerts = scored["is_alert"].dropna()
    return {
        "max_score": float(scores.max()) if len(scores) else None,
        "any_alert": bool(alerts.any()) if len(alerts) else None,
    }


def _graph_edges(graph_rows: pd.DataFrame) -> list[dict[str, Any]]:
    edges = []
    for _, row in graph_rows.iterrows():
        for edge in row["top_edges"] if row["top_edges"] is not None else []:
            edges.append({
                "dataset_day": int(row["dataset_day"]),
                "destination_computer": edge["destination_computer"],
                "edge_raw_score": float(edge["edge_raw_score"]),
                "is_new_edge": bool(edge["is_new_edge"]),
                "success_count": int(edge["success_count"]),
                "failure_count": int(edge["failure_count"]),
                "source_references": [f"auth.txt:{line}" for line in edge["source_lines"]],
                "references_truncated": bool(edge["references_truncated"]),
            })
    return edges


def build_incident_records(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    graph: pd.DataFrame | None = None,
    split: str = "test",
) -> list[dict[str, Any]]:
    """One JSON-ready record per incident, in the Task 5.4 / dashboard field layout.

    ``alerts`` needs ``incident_id`` and ``ground_truth_redteam`` columns;
    ``events`` has one row per contributing event with ``alert_id`` and
    ``source_reference``; ``graph`` holds graph-detector rows keyed by
    ``user_id`` and ``dataset_day`` (context only: the final model does not use it).
    ``split`` is written to each record's ``split`` field.
    """
    references = (
        events.sort_values(["timestamp", "source_reference"], kind="stable")
        .groupby("alert_id", sort=False)["source_reference"].agg(list)
    )
    graph_by_user_day = (
        {key: rows for key, rows in graph.groupby(["user_id", "dataset_day"], sort=False)}
        if graph is not None else {}
    )
    records = []
    for incident_id, rows in alerts.groupby("incident_id", sort=False):
        rows = rows.sort_values("window_start")
        user = rows["user"].iloc[0]
        days = sorted({int(day) for day in rows["day"]})
        graph_rows = [graph_by_user_day[(user, day)] for day in days if (user, day) in graph_by_user_day]
        graph_rows = pd.concat(graph_rows) if graph_rows else pd.DataFrame(columns=["score", "is_alert", "top_edges"])
        sources = [ref for alert_id in rows["alert_id"] for ref in references.get(alert_id, [])]
        start, end = int(rows["window_start"].min()), int(rows["window_end"].max())
        hours = [
            {
                "alert_id": row["alert_id"],
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "rank_in_day": int(row["rank_in_day"]),
                "tied_at_cutoff": bool(row["tied_at_cutoff"]),
                "fusion_score": float(row["fusion"]),
                "gru_max_event": float(row["gru_max_event"]),
                "gru_percentile_in_day": float(row["gru_percentile_in_day"]),
                "counts": {column: int(row[column]) for column in COUNT_COLUMNS},
                "ground_truth_redteam": bool(row["ground_truth_redteam"]),
            }
            for _, row in rows.iterrows()
        ]
        records.append({
            "incident_id": incident_id,
            "user_id": user,
            "start_time": start,
            "end_time": end,
            "duration_seconds": end - start,
            "duration_hours": int(np.ceil((end - start) / HOUR_SECONDS)),
            "dataset_day": days[0],
            "split": split,
            "priority": PRIORITY_ABOVE_CUTOFF if (~rows["tied_at_cutoff"]).any() else PRIORITY_TIED_AT_CUTOFF,
            "best_rank_in_day": int(rows["rank_in_day"].min()),
            "max_fused_score": float(rows["fusion"].max()),
            "mean_fused_score": float(rows["fusion"].mean()),
            "fusion_method": FUSION_METHOD,
            "alert_hours": hours,
            "detector_scores": {
                # The final model uses the raw GRU score; the percentile keeps it on a 0-1 scale.
                "sequence": {"max_score": float(rows["gru_percentile_in_day"].max()), "any_alert": None},
                "graph": _graph_summary(graph_rows),
            },
            "graph_context": {
                "used_by_final_model": False,
                "new_edge_count": int(graph_rows["new_edge_count"].sum()) if len(graph_rows) else None,
                "degree_growth": int(graph_rows["degree_growth"].max()) if len(graph_rows) else None,
                "top_edges": _graph_edges(graph_rows),
            },
            "source_references": sources,
            "evidence_count": len(sources),
            "graph_evidence_nodes": sorted({
                node for nodes in graph_rows.get("evidence_nodes", pd.Series(dtype=object))
                if nodes is not None for node in nodes
            }),
            "ground_truth_redteam": bool(rows["ground_truth_redteam"].any()),
            "ground_truth_redteam_hours": int(rows["ground_truth_redteam"].sum()),
        })
    records.sort(key=lambda record: (record["start_time"], record["user_id"]))
    return records


def label_alerts(alerts: pd.DataFrame, positives: Iterable[tuple[str, int]]) -> np.ndarray:
    positive_set = set(positives)
    return np.array(
        [(user, int(hour)) in positive_set for user, hour in zip(alerts["user"], alerts["window_start"])],
        dtype=bool,
    )
