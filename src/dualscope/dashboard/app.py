"""Streamlit views for incident navigation (Task 7.1) and detector evidence (Task 7.2)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dualscope.dashboard.data import (
    DEFAULT_FIXTURE,
    DETECTOR_FILTERS,
    PRIORITY_ORDER,
    SORT_OPTIONS,
    IncidentDataError,
    IncidentView,
    filter_sort_incidents,
    format_alert,
    format_dataset_second,
    format_score,
    load_incidents,
)
from dualscope.dashboard.evidence import (
    EvidenceDataError,
    alert_hours_table,
    events_path_for,
    graph_edges_table,
    host_pair_table,
    incident_events,
    load_events,
    timeline_table,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDOFF = REPOSITORY_ROOT / "outputs" / "handoff" / "final_test_alerts_v1" / "incidents.jsonl"
HANDOFF_SOURCE = "Alert handoff package"


def _source_controls() -> tuple[str, Path | None]:
    source = st.sidebar.radio("Incident source", ("Synthetic fixture", "Local JSONL export", HANDOFF_SOURCE))
    if source == "Synthetic fixture":
        st.info("Synthetic fixture — demonstration incidents, not detector results or a live feed.")
        return source, DEFAULT_FIXTURE
    if source == HANDOFF_SOURCE:
        st.info("Alert handoff package — the final model's saved days 17–30 alerts (38 per day), not a live feed.")
        entered_path = st.sidebar.text_input("Package incidents.jsonl", value=str(DEFAULT_HANDOFF))
        if not entered_path.strip():
            st.error("Enter the path of the package's incidents.jsonl.")
            return source, None
        return source, Path(entered_path.strip()).expanduser()
    st.info("Local JSONL export — a saved pipeline file, not a live feed. File provenance is not verified.")
    entered_path = st.sidebar.text_input("Incident JSONL path", placeholder="outputs/incidents.jsonl")
    if not entered_path.strip():
        st.error("Enter a local .jsonl incident file path to load incidents.")
        return source, None
    return source, Path(entered_path.strip()).expanduser()


@st.cache_data(show_spinner=False)
def _cached_events(path: str, modified: float):
    return load_events(Path(path))


def _render_detail(incident: IncidentView, incidents_path: Path, show_answer_key: bool) -> None:
    if st.button("← Back to queue"):
        st.session_state["selected_incident_id"] = None
        st.session_state["queue_nonce"] = st.session_state.get("queue_nonce", 0) + 1
        st.rerun()
    st.subheader(f"Incident {incident.incident_id}")
    st.write(f"**User ID:** {incident.user_id}")
    st.write(f"**Time range:** [{format_dataset_second(incident.start_time)}, "
             f"{format_dataset_second(incident.end_time)}) "
             f"(dataset seconds [{incident.start_time}, {incident.end_time}))")
    st.write(f"**Fusion priority:** {incident.priority}")
    st.write(f"**Maximum fused score:** {format_score(incident.max_fused_score)}")
    st.write(f"**Fusion method:** {incident.raw.get('fusion_method', 'Unavailable')}")
    st.write(f"**Detector concordance:** {incident.raw.get('concordance', 'Unavailable')}")
    if show_answer_key and "ground_truth_redteam" in incident.raw:
        verdict = "red-team activity" if incident.raw["ground_truth_redteam"] else "not red-team activity"
        st.warning(f"Answer key (evaluation only): {verdict}.")
    st.caption("Scores are anomaly rankings, not probabilities of an attack.")
    st.subheader("Detector breakdown")
    st.table(
        [
            {"Detector": "Sequence", "Maximum score": format_score(incident.sequence.max_score),
             "Flag": format_alert(incident.sequence.any_alert)},
            {"Detector": "Graph", "Maximum score": format_score(incident.graph.max_score),
             "Flag": format_alert(incident.graph.any_alert)},
        ]
    )
    if incident.raw.get("alert_hours"):
        st.caption("For final-model packages the sequence score is the GRU score's percentile within its day; "
                   "the graph detector is shown as context and was not used by the final model.")
        st.subheader("Alerted hours")
        st.caption("Rank order among hours tied at the day's cut-off comes from a fixed tie-break and carries no meaning.")
        st.dataframe(alert_hours_table(incident.raw, show_answer_key), hide_index=True, width="stretch")
    _render_events(incident, incidents_path)
    _render_graph_context(incident)


def _render_events(incident: IncidentView, incidents_path: Path) -> None:
    references = incident.raw.get("source_references") or []
    st.subheader("Authentication events")
    if not references:
        st.info("This incident has no event references.")
        return
    events_path = events_path_for(incidents_path)
    if not events_path.is_file():
        st.info(f"Event details are unavailable: no {events_path.name} next to the incident file. "
                f"The incident cites {len(references)} event references.")
        st.dataframe({"Event ID": references}, hide_index=True, width="stretch")
        return
    try:
        events = _cached_events(str(events_path), events_path.stat().st_mtime)
    except EvidenceDataError as exc:
        st.error(f"Could not load event details: {exc}")
        return
    rows, missing = incident_events(events, incident.raw)
    if missing:
        st.warning(f"{len(missing)} of {len(references)} cited events are missing from {events_path.name}.")

    timeline, hosts = st.tabs([f"Timeline ({len(rows)} events)", "User–host relationships"])
    with timeline:
        st.caption("Novelty flags mean a first appearance since day 1; log-off events are not counted as new.")
        selection = st.dataframe(
            timeline_table(rows), hide_index=True, width="stretch", on_select="rerun",
            selection_mode="single-row", key=f"timeline_{incident.incident_id}",
        )
        if selection.selection.rows:
            record = rows.iloc[selection.selection.rows[0]]
            st.markdown(f"**Source record {record['source_reference']}**")
            st.json({key: (value.item() if hasattr(value, "item") else value) for key, value in record.items()})
        else:
            st.caption("Select an event to see its full source record.")
    with hosts:
        st.caption("One row per source → destination pair in this incident; pairs with something new come first.")
        st.dataframe(host_pair_table(rows), hide_index=True, width="stretch")


def _render_graph_context(incident: IncidentView) -> None:
    context = incident.raw.get("graph_context")
    if not context:
        return
    st.subheader("Graph detector context")
    st.caption("The graph detector's view of this user's day. The final model does not use it, and the graph "
               "team inspected days 17–30 during development, so it is context, not an independent result.")
    st.write(f"**New edges that day:** {context.get('new_edge_count', 'Unavailable')} · "
             f"**Degree growth:** {context.get('degree_growth', 'Unavailable')}")
    edges = graph_edges_table(incident.raw)
    if edges.empty:
        st.info("No graph edges were recorded for this user-day.")
    else:
        st.dataframe(edges, hide_index=True, width="stretch")


def _render_queue(incidents: list[IncidentView]) -> None:
    st.subheader("Incident queue")
    if not incidents:
        st.warning("This incident source contains no incidents.")
        return

    st.caption("Times are dataset-relative. Ranges are [start, end); the end is excluded.")
    user_query = st.text_input("Filter by user ID", key="filter_user")
    priorities = st.multiselect("Fusion priority", tuple(PRIORITY_ORDER),
                                default=tuple(PRIORITY_ORDER), key="filter_priorities")
    start_col, end_col = st.columns(2)
    with start_col:
        start_time = st.number_input("From dataset second (inclusive)", min_value=1,
                                     value=min(item.start_time for item in incidents), step=3600,
                                     key="filter_start")
    with end_col:
        end_time = st.number_input("Before dataset second (exclusive)", min_value=1,
                                   value=max(item.end_time for item in incidents), step=3600,
                                   key="filter_end")
    detector_filter = st.selectbox("Detector flags", DETECTOR_FILTERS, key="filter_detector")
    sort_by = st.selectbox("Sort incidents", SORT_OPTIONS, key="filter_sort")

    if start_time >= end_time:
        st.error("The end of the time filter must be after its start.")
        return
    visible = filter_sort_incidents(
        incidents,
        user_query=user_query,
        priorities=priorities,
        start_time=start_time,
        end_time=end_time,
        detector_filter=detector_filter,
        sort_by=sort_by,
    )
    if not visible:
        st.info("No incidents match the current filters.")
        return
    st.caption(f"Showing {len(visible)} of {len(incidents)} incidents. Select a row to view details.")
    rows = [
        {
            "Incident ID": item.incident_id,
            "User ID": item.user_id,
            "Time range [start, end)": f"[{format_dataset_second(item.start_time)}, "
                                       f"{format_dataset_second(item.end_time)})",
            "Priority": item.priority,
            "Fused score": format_score(item.max_fused_score),
            "Sequence score": format_score(item.sequence.max_score),
            "Sequence flag": format_alert(item.sequence.any_alert),
            "Graph score": format_score(item.graph.max_score),
            "Graph flag": format_alert(item.graph.any_alert),
        }
        for item in visible
    ]
    visible_ids = tuple(item.incident_id for item in visible)
    if st.session_state.get("visible_ids") != visible_ids:
        st.session_state["visible_ids"] = visible_ids
        st.session_state["queue_nonce"] = st.session_state.get("queue_nonce", 0) + 1
    selection = st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"incident_queue_{st.session_state['queue_nonce']}",
    )
    if selection.selection.rows:
        st.session_state["selected_incident_id"] = visible[selection.selection.rows[0]].incident_id
        st.rerun()


def run() -> None:
    st.set_page_config(page_title="DualScope incidents", layout="wide")
    st.title("DualScope incident navigation")
    source, path = _source_controls()
    show_answer_key = st.sidebar.toggle(
        "Show answer key (evaluation only)", value=False,
        help="Shows which alerts were red-team activity. Keep off for analyst review.",
    )
    source_identity = (source, str(path) if path is not None else "")
    if st.session_state.get("source_identity") != source_identity:
        st.session_state["source_identity"] = source_identity
        st.session_state["selected_incident_id"] = None
        st.session_state["queue_nonce"] = st.session_state.get("queue_nonce", 0) + 1
        for key in ("filter_user", "filter_priorities", "filter_start", "filter_end",
                    "filter_detector", "filter_sort"):
            st.session_state.pop(key, None)
    if path is None:
        return
    try:
        with st.spinner("Loading incidents..."):
            incidents = load_incidents(path)
    except IncidentDataError as exc:
        st.error(f"Could not load incidents: {exc}")
        return

    selected_id = st.session_state.get("selected_incident_id")
    if selected_id is not None:
        selected = next((item for item in incidents if item.incident_id == selected_id), None)
        if selected is None:
            st.warning("The selected incident is no longer in this source.")
            if st.button("← Back to queue"):
                st.session_state["selected_incident_id"] = None
                st.rerun()
            return
        _render_detail(selected, path, show_answer_key)
    else:
        _render_queue(incidents)
