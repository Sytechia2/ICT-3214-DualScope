"""Streamlit views for Task 7.1 incident navigation."""

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


def _source_controls() -> tuple[str, Path | None]:
    source = st.sidebar.radio("Incident source", ("Synthetic fixture", "Local JSONL export"))
    if source == "Synthetic fixture":
        st.info("Synthetic fixture — demonstration incidents, not detector results or a live feed.")
        return source, DEFAULT_FIXTURE
    st.info("Local JSONL export — a saved pipeline file, not a live feed. File provenance is not verified.")
    entered_path = st.sidebar.text_input("Incident JSONL path", placeholder="outputs/incidents.jsonl")
    if not entered_path.strip():
        st.error("Enter a local .jsonl incident file path to load incidents.")
        return source, None
    return source, Path(entered_path.strip()).expanduser()


def _render_detail(incident: IncidentView) -> None:
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
    st.subheader("Detector breakdown")
    st.table(
        [
            {"Detector": "Sequence", "Maximum score": format_score(incident.sequence.max_score),
             "Flag": format_alert(incident.sequence.any_alert)},
            {"Detector": "Graph", "Maximum score": format_score(incident.graph.max_score),
             "Flag": format_alert(incident.graph.any_alert)},
        ]
    )


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
        _render_detail(selected)
    else:
        _render_queue(incidents)
