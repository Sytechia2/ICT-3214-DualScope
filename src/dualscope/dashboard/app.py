"""DualScope analyst dashboard: alert queue (Task 7.1) and incident evidence (Task 7.2).

Four pages share one data source: the queue for a day with an incident side
panel, the incident page (overview, timeline, hosts, evidence, investigation),
an event lookup, and a plain-language page about the model. Layout borrows from
established SOC tools (Defender-style queue and incident page, Elastic-style
summary strip and field table); colours and fonts are in .streamlit/config.toml.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dualscope.attack.retrieval import TechniqueRetriever
from dualscope.dashboard import ui
from dualscope.dashboard.data import (
    DEFAULT_FIXTURE,
    IncidentDataError,
    format_dataset_second,
    format_score,
    load_incidents,
)
from dualscope.dashboard.evidence import (
    EvidenceDataError,
    alert_hours_table,
    event_flags,
    events_path_for,
    graph_edges_table,
    host_pair_table,
    incident_events,
    load_events,
    timeline_table,
)
from dualscope.dashboard.queue import (
    CHIP_LABELS,
    HEADLINES,
    NO_EVIDENCE,
    IncidentSummary,
    behaviour_totals,
    day_cutoff_counts,
    matches_search,
    summarise_incidents,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDOFF = REPOSITORY_ROOT / "outputs" / "handoff" / "final_test_alerts_v1" / "incidents.jsonl"
HANDOFF_SOURCE = "Alert package (days 17–30)"
FIXTURE_SOURCE = "Synthetic fixture"
OTHER_SOURCE = "Other JSONL export"
QUEUE_SIZE = 38
TAB_LABELS = ("Overview", "Timeline", "Hosts", "Evidence", "Investigation")
_BASE = datetime(2000, 1, 1)  # charts need a date; only the time of day is shown


# ─── Data ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Workspace:
    source: str
    path: Path
    summaries: list[IncidentSummary]
    events: pd.DataFrame | None

    @property
    def by_id(self) -> dict[str, IncidentSummary]:
        return {summary.incident_id: summary for summary in self.summaries}

    @property
    def is_package(self) -> bool:
        return self.events is not None and any(s.incident.raw.get("alert_hours") for s in self.summaries)


@st.cache_resource(show_spinner=False)
def _retriever() -> TechniqueRetriever:
    return TechniqueRetriever.from_file()


@st.cache_resource(show_spinner=False)
def _load(path: str, modified: float, events_modified: float | None):
    incidents = load_incidents(Path(path))
    events_file = events_path_for(Path(path))
    events = load_events(events_file) if events_modified is not None else None
    return summarise_incidents(incidents, events, _retriever()), events


def _source_controls() -> tuple[str, Path | None]:
    default = HANDOFF_SOURCE if DEFAULT_HANDOFF.is_file() else FIXTURE_SOURCE
    options = (HANDOFF_SOURCE, FIXTURE_SOURCE, OTHER_SOURCE)
    with st.sidebar.expander("Data source", icon=":material/database:"):
        source = st.radio("Incident source", options, index=options.index(default), key="source")
        if source == HANDOFF_SOURCE:
            return source, DEFAULT_HANDOFF
        if source == FIXTURE_SOURCE:
            return source, DEFAULT_FIXTURE
        entered = st.text_input("Incident JSONL path", placeholder="outputs/incidents.jsonl", key="source_path")
    return source, Path(entered.strip()).expanduser() if entered.strip() else None


def _workspace() -> Workspace | None:
    source, path = st.session_state["_source"]
    if path is None:
        st.info("Enter the path of a saved incidents .jsonl file under **Data source** in the sidebar.")
        return None
    events_file = events_path_for(path)
    try:
        with st.spinner("Loading incidents…"):
            summaries, events = _load(
                str(path), path.stat().st_mtime if path.exists() else 0.0,
                events_file.stat().st_mtime if events_file.is_file() else None,
            )
    except (IncidentDataError, EvidenceDataError, OSError) as exc:
        st.error(f"Could not load incidents from {path}: {exc}")
        return None
    return Workspace(source, path, summaries, events)


def _provenance(ws: Workspace) -> str:
    if ws.source == FIXTURE_SOURCE:
        return "Synthetic fixture — demonstration incidents, not detector results"
    if ws.is_package:
        return f"Final model · test days 17–30 · {QUEUE_SIZE} alerts per day · saved package, not a live feed"
    return f"Saved incident export {ws.path.name} · not a live feed"


def _events_of(ws: Workspace, summary: IncidentSummary) -> pd.DataFrame:
    if ws.events is None:
        return pd.DataFrame()
    rows, _ = incident_events(ws.events, summary.incident.raw)
    return rows


def _hour_range(summary: IncidentSummary) -> str:
    start, end = format_dataset_second(summary.incident.start_time), format_dataset_second(summary.incident.end_time)
    if start.split()[:2] == end.split()[:2]:
        return f"{start[:-3]}–{end.split()[-1][:-3]}"
    return f"{start[:-3]} – {end[:-3]}"


def _clock(seconds: int) -> str:
    return format_dataset_second(seconds).split()[-1][:-3]


def _select(incident_id: str) -> None:
    st.session_state["selected_incident_id"] = incident_id
    st.session_state["queue_nonce"] = st.session_state.get("queue_nonce", 0) + 1


# ─── Shared pieces ─────────────────────────────────────────────────────


def _badges(summary: IncidentSummary, *, extra: list[str] | None = None) -> str:
    parts = [ui.priority_badge(summary.incident.priority)]
    if summary.rank is not None:
        parts.append(ui.badge(f"Rank {summary.rank} of {QUEUE_SIZE}"))
        parts.append(ui.badge("Above cut-off" if summary.above_cutoff else "Tied at cut-off"))
    parts.extend(extra or [])
    if st.session_state.get("show_answer_key") and summary.redteam is not None:
        parts.append(ui.badge("Red-team activity" if summary.redteam else "Not red-team", "answer"))
    return "".join(parts)


def _first_event_line(ws: Workspace, reference: str) -> str:
    if ws.events is None or reference not in ws.events.index:
        return ui.mono(reference)
    event = ws.events.loc[reference]
    return (f"{ui.mono(reference)} <span class='ds-muted'>· <span style='white-space:nowrap'>{escape(event['source_computer'])} → "
            f"{escape(event['destination_computer'])}</span> · {_clock(int(event['timestamp']))}</span>")


def _example_events(summary: IncidentSummary) -> list[str]:
    """One example event per behaviour, preferring one an earlier behaviour has not already shown."""
    shown: list[str] = []
    for behaviour in summary.ordered_behaviours:
        references = behaviour["evidence_references"]
        shown.append(next((r for r in references if r not in shown), references[0]))
    return shown


def _reasons_html(ws: Workspace, summary: IncidentSummary) -> str:
    if not summary.events_available:
        return "<div class='ds-reason ds-muted'>Event details are not available for this source.</div>"
    if not summary.behaviours:
        return ("<div class='ds-reason'><b>No single event matched a behaviour rule.</b><br>"
                "<span class='ds-muted'>The model ranked this hour on its overall pattern; there is no event to cite.</span></div>")
    items = []
    for behaviour, example in zip(summary.ordered_behaviours, _example_events(summary)):
        count = behaviour["event_count"]
        items.append(
            f"<div class='ds-reason'><b>{escape(HEADLINES[behaviour['behaviour']])}</b>"
            f" · {count} event{'s' if count != 1 else ''}<br>{_first_event_line(ws, example)}</div>"
        )
    return "".join(items)


def _candidates_html(summary: IncidentSummary, limit: int = 3) -> str:
    if not summary.candidates:
        return "<div class='ds-muted'>None — no behaviour to search with.</div>"
    return "".join(
        f"<div class='ds-candidate'><span>{ui.mono(c['technique_id'])} {escape(c['name'])}</span>"
        f"{ui.badge('Unverified')}</div>"
        for c in summary.candidates[:limit]
    )


def _host_counts(rows: pd.DataFrame) -> tuple[int, int]:
    if rows.empty:
        return 0, 0
    pairs = host_pair_table(rows)
    new = pairs[["New destination for user", "New host pair", "New source for user"]].any(axis=1)
    return int(rows["destination_computer"].nunique()), int(new.sum())


# ─── Queue page ────────────────────────────────────────────────────────


def _queue_by_day_chart(counts: pd.DataFrame, day: int) -> alt.Chart:
    long = counts.melt(id_vars="day", value_vars=["above", "tied"], var_name="status", value_name="alerts")
    long["status"] = long["status"].map({"above": "Above cut-off", "tied": "Tied"})
    long["Day"] = long["day"].astype(str)
    days = counts.assign(Day=counts["day"].astype(str), label=counts["above"].astype(str) + " / " + counts["tied"].astype(str),
                         total=counts["above"] + counts["tied"])
    order = list(days["Day"])
    x = alt.X("Day:N", sort=order, scale=alt.Scale(domain=order), title=None,
              axis=alt.Axis(labelAngle=0, labelFontSize=11, labelColor=ui.INK, labelOverlap=False, labelPadding=6, titleFontSize=13,
                            titleColor=ui.MUTED, titleFontWeight="normal"))
    band = (alt.Chart(days[days["day"] == day]).mark_bar(width={"band": 1.0}, color=ui.ACCENT_TINT,
                                                         cornerRadius=3)
            .encode(x=x, y=alt.datum(QUEUE_SIZE + 14), y2=alt.datum(0)))
    bars = alt.Chart(long).mark_bar(width={"band": 0.7}).encode(
        x=x,
        y=alt.Y("alerts:Q", title=None, stack=True, scale=alt.Scale(domain=[0, QUEUE_SIZE + 14]),
                axis=alt.Axis(values=[0, 19, 38], grid=True, gridColor=ui.LINE, labelFontSize=11, labelColor=ui.MUTED)),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Above cut-off", "Tied"], range=[ui.ACCENT, ui.TIED]),
                        legend=alt.Legend(orient="top", direction="horizontal", title=None, labelFontSize=12, labelColor=ui.INK, symbolType="square")),
        order=alt.Order("status:N", sort="ascending"),
        tooltip=[alt.Tooltip("Day:N", title="Test day"), "status:N", "alerts:Q"],
    )
    labels = (alt.Chart(days[days["day"] == day]).mark_text(dy=-7, fontSize=12, fontWeight=600, color=ui.INK)
              .encode(x=x, y="total:Q", text="label:N"))
    return (band + bars + labels).properties(height=158).configure_view(stroke=None)


def _why_flagged_chart(totals: pd.DataFrame) -> alt.Chart:
    short = {headline: CHIP_LABELS[key] for key, headline in HEADLINES.items()}
    totals = totals.assign(kind=["none" if b == NO_EVIDENCE else "rule" for b in totals["behaviour"]],
                           behaviour=[short.get(b, b) for b in totals["behaviour"]])
    base = alt.Chart(totals).encode(
        y=alt.Y("behaviour:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=160, labelFontSize=12, labelColor=ui.INK, ticks=False, domain=False)),
        x=alt.X("incidents:Q", title=None, axis=None, scale=alt.Scale(domain=[0, max(1, int(totals["incidents"].max())) * 1.55])),
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=15).encode(
        color=alt.Color("kind:N", scale=alt.Scale(domain=["rule", "none"], range=[ui.SHELL, "#B8BEC6"]), legend=None),
        tooltip=["behaviour:N", "incidents:Q"],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=12, color=ui.INK).encode(text="incidents:Q")
    return (bars + labels).properties(height=150).configure_view(stroke=None)


def _summary_strip(ws: Workspace, day: int, day_summaries: list[IncidentSummary]) -> None:
    counts = day_cutoff_counts(s.incident for s in ws.summaries)
    left, middle, right = st.columns([1.45, 1.1, 0.55])
    with left.container(border=True, height=212, key="card_queue_by_day"):
        st.html("<div class='ds-section'>Queue by test day</div>")
        if counts.empty:
            st.caption("The cut-off needs a final-model package; this source has no alert hours.")
        else:
            st.altair_chart(_queue_by_day_chart(counts, day), width="stretch")
    with middle.container(border=True, height=212, key="card_why_flagged"):
        evidenced = [s for s in ws.summaries if s.events_available]
        st.html(f"<div class='ds-section'>Why flagged <span class='ds-muted' style='font-weight:400'>· "
                f"{len(evidenced)} incidents</span></div>")
        if evidenced:
            st.altair_chart(_why_flagged_chart(behaviour_totals(evidenced)), width="stretch")
        else:
            st.caption("Needs event details (events.parquet next to the incident file).")
    with right.container(border=True, height=212, key="card_day_glance"):
        st.html(f"<div class='ds-section'>Day {day}</div>")
        row = counts[counts["day"] == day]
        hours = int(row[["above", "tied"]].sum(axis=1).iloc[0]) if len(row) else None
        tied = int(row["tied"].iloc[0]) if len(row) else None
        st.html(
            "<div class='ds-stats'>"
            f"<div><b>{len(day_summaries)}</b><span>incidents</span></div>"
            f"<div><b>{hours if hours is not None else '—'}</b><span>alert hours</span></div>"
            f"<div><b>{tied if tied is not None else '—'}</b><span>tied at cut-off</span></div></div>"
        )


def _queue_frame(summaries: list[IncidentSummary], selected: str | None, answer_key: bool):
    rows = []
    for s in summaries:
        candidate = s.top_candidate
        hours = len(s.incident.raw.get("alert_hours") or []) or 1
        row = {
            "Priority": s.incident.priority,
            "Rank": s.rank,
            "Incident": s.incident_id,
            "User": s.incident.user_id,
            "Hour": _clock(s.incident.start_time) + (f" +{hours - 1}h" if hours > 1 else ""),
            "Why flagged": _why_text(s),
            "Events": s.event_count,
            "Top ATT&CK candidate": f"{candidate['technique_id']} {candidate['name']}" if candidate else "—",
        }
        if answer_key:
            row["Red-team (answer key)"] = s.redteam
        rows.append(row)
    frame = pd.DataFrame(rows)

    def style_row(row: pd.Series) -> list[str]:
        tint = f"background-color: {ui.ACCENT_TINT};" if row["Incident"] == selected else ""
        return [tint + _priority_cell(row[c]) if c == "Priority" else tint for c in row.index]

    return frame.style.apply(style_row, axis=1) if not frame.empty else frame


def _why_text(summary: IncidentSummary) -> str:
    labels = summary.chips or [NO_EVIDENCE if summary.events_available else "—"]
    extra = f" +{len(labels) - 2}" if len(labels) > 2 else ""
    return " · ".join(labels[:2]) + extra


def _priority_cell(priority: str) -> str:
    if priority in ("HIGH", "CRITICAL"):
        return f"color: {ui.HIGH}; font-weight: 700;"
    if priority == "MEDIUM":
        return "color: #8A5A00; font-weight: 700;"
    return ""


QUEUE_COLUMNS = {
    "Priority": st.column_config.TextColumn(width=60),
    "Rank": st.column_config.NumberColumn(width=40, format="%d"),
    "Incident": None,  # shown in the side panel; kept in the frame to tint the selected row
    "User": st.column_config.TextColumn(width=104),
    "Hour": st.column_config.TextColumn(width=50),
    "Why flagged": st.column_config.TextColumn(width=206, help="Behaviours behind the alert; +n means more in the side panel"),
    "Events": st.column_config.NumberColumn(width=50, format="%d"),
    "Top ATT&CK candidate": st.column_config.TextColumn(width=236),
}
ROW_HEIGHT = 32
HEADER_HEIGHT = 35  # Streamlit dataframe header plus borders


def _whole_rows(count: int, limit: int) -> int:
    """Table height showing whole rows only (header plus up to ``limit`` rows)."""
    return min(count, limit) * ROW_HEIGHT + HEADER_HEIGHT


def _queue_table(summaries: list[IncidentSummary], key: str, selected: str | None, limit: int) -> None:
    if not summaries:
        return
    data = _queue_frame(summaries, selected, bool(st.session_state.get("show_answer_key")))
    event = st.dataframe(
        data, hide_index=True, column_config=QUEUE_COLUMNS, on_select="rerun", selection_mode="single-row",
        key=f"{key}_{st.session_state.get('queue_nonce', 0)}", row_height=ROW_HEIGHT,
        height=_whole_rows(len(summaries), limit),
    )
    if event.selection.rows:
        _select(summaries[event.selection.rows[0]].incident_id)
        st.rerun()


def _side_panel(ws: Workspace, summary: IncidentSummary, visible: list[IncidentSummary]) -> None:
    ids = [s.incident_id for s in visible]
    position = ids.index(summary.incident_id) if summary.incident_id in ids else 0
    nav = st.container(horizontal=True, vertical_alignment="center")
    if nav.button("", icon=":material/keyboard_arrow_up:", key="panel_up", help="Previous incident", disabled=position == 0):
        _select(ids[position - 1])
        st.rerun()
    if nav.button("", icon=":material/keyboard_arrow_down:", key="panel_down", help="Next incident",
                  disabled=position >= len(ids) - 1):
        _select(ids[position + 1])
        st.rerun()
    nav.caption(f"{position + 1} of {len(ids)} on this day")
    rows = _events_of(ws, summary)
    hosts, new_pairs = _host_counts(rows)
    st.html(
        f"<h3 style='margin:0.1rem 0 0.1rem'>{escape(summary.incident.user_id)} — {escape(summary.headline)}</h3>"
        f"<div style='margin-bottom:0.55rem'>{ui.mono(summary.incident_id)}</div>{_badges(summary)}"
        f"<div class='ds-section' style='margin-top:0.6rem'>Why flagged</div>{_reasons_html(ws, summary)}"
        f"<div class='ds-section' style='margin-top:0.55rem'>Facts</div>"
        + ui.facts([
            ("Hour", _hour_range(summary)),
            ("Events", str(summary.event_count)),
            ("Hosts reached", str(hosts) if ws.events is not None else "—"),
            ("New host pairs", str(new_pairs) if ws.events is not None else "—"),
        ])
        + f"<div class='ds-muted' style='margin-top:0.45rem'>Fusion score {format_score(summary.incident.max_fused_score)}"
          " — a ranking, not a probability</div>"
        + f"<div class='ds-section' style='margin-top:0.55rem'>ATT&amp;CK candidates "
          f"<span class='ds-muted' style='font-weight:400'>(not yet verified)</span></div>{_candidates_html(summary, 2)}"
    )
    if st.button("Open incident", type="primary", icon=":material/open_in_new:", width="stretch", key="open_incident"):
        st.switch_page(_pages()["incident"])


def queue_page() -> None:
    ws = _workspace()
    if ws is None:
        return
    if not ws.summaries:
        st.title("Alert queue")
        st.warning("This incident source contains no incidents.")
        return
    days = sorted({s.day for s in ws.summaries})
    selected_id = st.session_state.get("selected_incident_id")
    if st.session_state.get("day") not in days:
        selected = ws.by_id.get(selected_id)
        st.session_state["day"] = selected.day if selected else days[0]

    # The incident panel runs full height on the right, beside the header and strip.
    main, panel = st.columns([0.72, 0.28], gap="medium")
    with main:
        head, pick, _, search = st.columns([0.36, 0.16, 0.08, 0.4], vertical_alignment="center")
        head.title("Alert queue")
        day = pick.selectbox("Day", days, format_func=lambda d: f"Day {d}", key="day", label_visibility="collapsed")
        query = search.text_input("Search", placeholder="Search user, host or event ID", key="f_search",
                                  label_visibility="collapsed", icon=":material/search:")
        st.html(f"<div class='ds-provenance'>{escape(_provenance(ws))}</div>")

        day_summaries = sorted((s for s in ws.summaries if s.day == day),
                               key=lambda s: (s.rank if s.rank is not None else 10_000, s.incident.start_time))
        _summary_strip(ws, day, day_summaries)

        behaviours = [*CHIP_LABELS.values(), NO_EVIDENCE]
        filters = st.container(horizontal=True, gap="medium")
        priorities = filters.pills("Priority", ["HIGH", "MEDIUM"], selection_mode="multi", default=["HIGH", "MEDIUM"], key="f_priority", width="content")
        wanted = filters.pills("Why flagged", behaviours, selection_mode="multi", key="f_behaviour", width="content")

        visible = [
            s for s in day_summaries
            if s.incident.priority in (priorities or ["HIGH", "MEDIUM"])
            and (not wanted or any(label in wanted for label in (s.chips or [NO_EVIDENCE])))
            and matches_search(s, query, ws.events)
        ]
        if visible and st.session_state.get("selected_incident_id") not in {s.incident_id for s in visible}:
            st.session_state["selected_incident_id"] = visible[0].incident_id
        selected_id = st.session_state.get("selected_incident_id")

        if not visible:
            st.info("No incidents on this day match the filters. Clear a filter or pick another day.")
        above = [s for s in visible if s.above_cutoff]
        tied = [s for s in visible if not s.above_cutoff]
        _queue_table(above, "queue_above", selected_id, limit=12)
        if tied and ws.is_package:
            st.html(ui.cutoff_rule())
        _queue_table(tied, "queue_tied", selected_id, limit=6)
        if ws.is_package:
            st.caption("Rank order among tied rows comes from a fixed tie-break and carries no meaning.")
    with panel:
        if visible:
            with st.container(border=True, key="card_panel"):
                _side_panel(ws, ws.by_id[selected_id], visible)


# ─── Incident page ─────────────────────────────────────────────────────


def _lane(orientation: str) -> str:
    return {"LogOn": "Logon", "LogOff": "Log-off", "TGS": "Ticket request", "TGT": "Ticket request"}.get(orientation, "Other")


def _activity_chart(rows: pd.DataFrame, cited: set[str], start: int, end: int) -> alt.Chart:
    frame = pd.DataFrame({
        "time": [_BASE + timedelta(seconds=int(t - 1) % 86_400) for t in rows["timestamp"]],
        "lane": [_lane(o) for o in rows["authentication_orientation"]],
        "event": rows["source_reference"],
        "route": rows["source_computer"] + " → " + rows["destination_computer"],
        "auth": rows["authentication_type"],
        "cited": rows["source_reference"].isin(cited),
    })
    span = [_BASE + timedelta(seconds=(start - 1) % 86_400), _BASE + timedelta(seconds=(start - 1) % 86_400 + (end - start))]
    lanes = [lane for lane in ("Logon", "Log-off", "Ticket request", "Other") if lane in set(frame["lane"])]
    base = alt.Chart(frame).encode(
        x=alt.X("time:T", title=None, scale=alt.Scale(domain=span),
                axis=alt.Axis(format="%H:%M", labelFontSize=12, labelColor=ui.MUTED, grid=True, gridColor=ui.LINE, tickCount=6)),
        y=alt.Y("lane:N", sort=lanes, title=None, scale=alt.Scale(domain=lanes),
                axis=alt.Axis(labelFontSize=12, labelColor=ui.INK, ticks=False, domain=False)),
        tooltip=[alt.Tooltip("time:T", format="%H:%M:%S"), "event:N", "route:N", "auth:N"],
    )
    ticks = base.transform_filter("!datum.cited").mark_tick(thickness=2, size=18, color=ui.ACCENT, opacity=0.75)
    marks = base.transform_filter("datum.cited").mark_tick(thickness=3, size=24, color=ui.HIGH)
    return (ticks + marks).properties(height=170).configure_view(stroke=None)


def _first_time_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    pairs = host_pair_table(rows)
    new = pairs[pairs[["New destination for user", "New host pair", "New source for user"]].any(axis=1)]
    return pd.DataFrame({
        "Source → destination": new["Source"] + " → " + new["Destination"],
        "Auth type": new["Auth types"],
        "Events": new["Events"],
        "First seen": [value.split()[-1][:-3] for value in new["First seen"]],
    })


def _overview_tab(ws: Workspace, summary: IncidentSummary, rows: pd.DataFrame) -> None:
    left, middle, right = st.columns([1.05, 1.75, 1.0], gap="medium")
    with left:
        st.html("<div class='ds-section'>Why flagged</div>")
        examples = _example_events(summary)
        for index, (behaviour, example) in enumerate(zip(summary.ordered_behaviours, examples)):
            with st.container(border=True, key=f"card_reason_{index}"):
                count = behaviour["event_count"]
                st.html(f"<div class='ds-reason' style='padding:0'><b>{escape(HEADLINES[behaviour['behaviour']])}</b>"
                        f" · {count} event{'s' if count != 1 else ''}<small>{escape(behaviour['description'])}</small>"
                        f"{_first_event_line(ws, example)}</div>")
                if st.button("Show in timeline", key=f"show_{index}", type="tertiary", icon=":material/arrow_forward:"):
                    st.session_state["incident_tab"] = "Timeline"
                    st.session_state["timeline_focus"] = example
                    st.rerun()
        if not summary.ordered_behaviours:
            with st.container(border=True, key="card_reason_none"):
                st.html(_reasons_html(ws, summary))
        st.caption("New = first time since Day 1. Log-offs are not counted as new.")
    with middle:
        st.html("<div class='ds-section'>Activity in this incident</div>")
        if rows.empty:
            st.info("Event details are unavailable for this source.")
        else:
            cited = {ref for b in summary.behaviours for ref in b["evidence_references"]}
            with st.container(border=True, key="card_activity"):
                st.altair_chart(_activity_chart(rows, cited, summary.incident.start_time, summary.incident.end_time),
                                width="stretch")
                st.html(f"<div class='ds-legend'><span class='ds-swatch' style='background:{ui.ACCENT}'></span>event"
                        f"<span class='ds-swatch' style='background:{ui.HIGH}'></span>event behind “Why flagged”</div>")
            first = _first_time_pairs(rows)
            st.html("<div class='ds-section' style='margin-top:0.6rem'>Hosts reached for the first time</div>")
            if first.empty:
                st.caption("None: every source → destination pair here had been seen before.")
            else:
                st.dataframe(first, hide_index=True, row_height=ROW_HEIGHT)
    with right:
        st.html("<div class='ds-section'>Investigation</div>")
        with st.container(border=True, key="card_summary"):
            st.html("<div class='ds-section'>Summary</div><div class='ds-pending'><b>AI investigation summary not generated yet</b>"
                    "Detector evidence is available in the tabs.</div>")
        with st.container(border=True, key="card_candidates"):
            st.html("<div class='ds-section'>ATT&amp;CK candidates <span class='ds-muted' style='font-weight:400'>"
                    f"— not yet verified</span></div>{_candidates_html(summary)}")
    _scores_strip(summary)


def _scores_strip(summary: IncidentSummary) -> None:
    hours = summary.incident.raw.get("alert_hours") or []
    gru = max((h["gru_percentile_in_day"] for h in hours), default=None)
    graph = summary.incident.graph.max_score
    with st.container(border=True, key="card_scores"):
        st.html("<div class='ds-section'>Scores</div>" + ui.facts([
            ("Fusion score — a ranking, not a probability", format_score(summary.incident.max_fused_score)),
            ("GRU score, percentile within its day", f"{gru:.2%}" if gru is not None else format_score(summary.incident.sequence.max_score)),
            ("Graph detector — context only, not used by the model", format_score(graph)),
            ("Alert hours in this incident", str(len(hours)) if hours else "—"),
        ]))


def _timeline_tab(ws: Workspace, summary: IncidentSummary, rows: pd.DataFrame) -> None:
    if rows.empty:
        st.info("Event details are unavailable for this source.")
        st.dataframe({"Event ID": summary.incident.raw.get("source_references") or []}, hide_index=True)
        return
    cited = list(dict.fromkeys(r for b in summary.ordered_behaviours for r in b["evidence_references"]))
    focus = st.session_state.get("timeline_focus")
    options = [f"Events behind “Why flagged” ({len(cited)})", f"All events ({len(rows)})"] if cited else [f"All events ({len(rows)})"]
    choice = st.segmented_control("Show", options, default=options[0],
                                  key=f"timeline_view_{summary.incident_id}_{focus or ''}")
    shown = rows[rows["source_reference"].isin(cited)] if choice == options[0] and cited else rows
    shown = shown.reset_index(drop=True)

    present = list(shown["source_reference"])
    target = focus if focus in present else next((r for r in present if r in set(cited)), present[0])
    visible_rows = min(len(shown), max(12, present.index(target) + 2), 24)
    table = timeline_table(shown)
    styled = table.style.apply(
        lambda row: [f"background-color: {ui.ACCENT_TINT}; font-weight: 600;" if row["Event ID"] == target else ""] * len(row), axis=1)
    st.caption("Select an event to see its full source record. Novelty flags mean a first appearance since Day 1; "
               "log-offs are not counted as new.")
    event = st.dataframe(styled, hide_index=True, on_select="rerun", selection_mode="single-row",
                         key=f"timeline_{summary.incident_id}_{choice}", row_height=ROW_HEIGHT,
                         height=visible_rows * ROW_HEIGHT + HEADER_HEIGHT)
    if event.selection.rows:
        record = shown.iloc[event.selection.rows[0]]
    else:
        record = shown[shown["source_reference"] == target].iloc[0] if target in set(shown["source_reference"]) else shown.iloc[0]
    _record_table(record)


def _record_table(record: pd.Series) -> None:
    st.html(f"<div class='ds-section' style='margin-top:0.6rem'>Source record {ui.mono(record['source_reference'])}</div>")
    fields = pd.DataFrame({
        "Field": list(record.index),
        "Value": [format_dataset_second(int(v)) if k == "timestamp" else str(v.item() if hasattr(v, "item") else v)
                  for k, v in record.items()],
    })
    query = st.text_input("Filter fields", placeholder="Filter fields", key="record_filter", label_visibility="collapsed")
    if query.strip():
        mask = fields["Field"].str.contains(query, case=False, regex=False) | fields["Value"].str.contains(query, case=False, regex=False)
        fields = fields[mask]
    st.dataframe(fields, hide_index=True, row_height=30, height=len(fields) * 30 + HEADER_HEIGHT)


def _host_graph(pairs: pd.DataFrame, limit: int = 16) -> tuple[str, int]:
    """Radial diagram around the busiest source; self-pairs (local log-on/off records) are left out."""
    remote = pairs[pairs["Source"] != pairs["Destination"]]
    new = remote[["New destination for user", "New host pair"]].any(axis=1)
    shown = pd.concat([remote[new], remote[~new].sort_values("Events", ascending=False).head(max(0, limit - int(new.sum())))])
    centre = shown.groupby("Source")["Events"].sum().idxmax() if len(shown) else ""
    lines = [
        "digraph G {",
        f"  graph [layout=twopi, root=\"{centre}\", ranksep=0.7, overlap=false, bgcolor=transparent, fontname=\"public-sans\"];",
        f"  node [shape=box, style=\"rounded,filled\", fillcolor=\"#FFFFFF\", color=\"#A3A9B1\", fontname=\"public-sans\","
        f" fontsize=15, fontcolor=\"{ui.INK}\", margin=\"0.12,0.06\"];",
        f"  \"{centre}\" [fillcolor=\"{ui.SHELL}\", fontcolor=\"#FFFFFF\", color=\"{ui.SHELL}\"];",
        "  edge [color=\"#A3A9B1\", arrowsize=0.7, penwidth=1.3];",
    ]
    is_new = shown[["New destination for user", "New host pair"]].any(axis=1)
    for host in sorted(set(shown.loc[is_new, "Destination"]) - {centre}):
        lines.append(f"  \"{host}\" [color=\"{ui.HIGH}\", penwidth=2, label=<{host}<BR/>"
                     f"<FONT COLOR=\"{ui.HIGH}\" POINT-SIZE=\"13\"><B>new</B></FONT>>];")
    for (_, row), new in zip(shown.iterrows(), is_new):
        attrs = f"color=\"{ui.HIGH}\", penwidth=2.4" if new else ""
        lines.append(f"  \"{row['Source']}\" -> \"{row['Destination']}\" [{attrs}];")
    lines.append("}")
    return "\n".join(lines), len(remote) - len(shown)


def _hosts_tab(rows: pd.DataFrame) -> None:
    if rows.empty:
        st.info("Event details are unavailable for this source.")
        return
    pairs = host_pair_table(rows)
    st.html("<div class='ds-section'>Who connected to what</div>")
    dot, hidden = _host_graph(pairs)
    with st.container(border=True, key="card_host_graph"):
        st.graphviz_chart(dot, width="stretch", height=470)
    st.caption("Red arrows and boxes marked new: the first time this user, or this pair of computers, made the connection. "
               "Records where a computer logs on or off itself are left out."
               + (f" {hidden} known pairs with fewer events are also left out; the table lists every pair." if hidden else ""))
    st.html("<div class='ds-section' style='margin-top:0.6rem'>Source → destination pairs</div>")
    st.dataframe(pairs, hide_index=True, row_height=ROW_HEIGHT, height=min(len(pairs), 15) * ROW_HEIGHT + HEADER_HEIGHT, column_config={
        "Events": st.column_config.NumberColumn(width=70),
        "Failures": st.column_config.NumberColumn(width=80),
        "New destination for user": st.column_config.CheckboxColumn(width=180),
        "New host pair": st.column_config.CheckboxColumn(width=120),
        "New source for user": st.column_config.CheckboxColumn(width=160),
    })


def _evidence_tab(summary: IncidentSummary) -> None:
    raw = summary.incident.raw
    if raw.get("alert_hours"):
        st.html("<div class='ds-section'>Alerted hours</div>")
        st.dataframe(alert_hours_table(raw, bool(st.session_state.get("show_answer_key"))), hide_index=True)
    references = raw.get("source_references") or []
    behind = len({r for b in summary.behaviours for r in b["evidence_references"]})
    st.html(f"<div class='ds-section' style='margin-top:0.6rem'>Events in this incident</div>"
            f"<div class='ds-muted'>{len(references)} auth.txt lines make up this incident; {behind} of them are behind "
            "“Why flagged”. The Timeline tab lists them all.</div>")
    context = raw.get("graph_context")
    if context:
        st.html("<div class='ds-section' style='margin-top:0.6rem'>Graph detector context</div>")
        st.caption("The graph detector's view of this user's day. The final model does not use it, and the graph team "
                   "inspected days 17–30 during development, so it is context, not an independent result.")
        st.html(ui.facts([("New edges that day", str(context.get("new_edge_count", "—"))),
                          ("Degree growth", str(context.get("degree_growth", "—")))]))
        edges = graph_edges_table(raw)
        if edges.empty:
            st.info("No graph edges were recorded for this user-day.")
        else:
            st.dataframe(edges, hide_index=True)


def _investigation_tab(summary: IncidentSummary) -> None:
    left, right = st.columns([1, 1.4], gap="large")
    with left, st.container(border=True, key="card_inv_summary"):
        st.html("<div class='ds-section'>Summary</div><div class='ds-pending'><b>AI investigation summary not generated yet</b>"
                "When Task 6.2 runs, its summary appears here with every claim linked to an event. "
                "Detector evidence stays available in the other tabs either way.</div>")
    with right:
        st.html("<div class='ds-section'>ATT&amp;CK candidates — not yet verified</div>")
        st.caption("Retrieved from MITRE ATT&CK 19.2 by matching each behaviour's description. A candidate is a "
                   "lead to check, not a finding; verification (Task 6.3) will mark each supported, uncertain or rejected.")
        if not summary.candidates:
            st.info("No candidates: no event matched a behaviour rule.")
        for candidate in summary.candidates:
            with st.expander(f"{candidate['technique_id']} · {candidate['name']}"):
                st.html(f"<div>{ui.badge('Unverified')}{ui.badge('Retrieved by: ' + ', '.join(CHIP_LABELS[b] for b in candidate['retrieved_by']))}</div>")
                st.write(candidate["description"][:700] + ("…" if len(candidate["description"]) > 700 else ""))
                st.markdown(f"[{candidate['url']}]({candidate['url']})")


def _incident_picker(ws: Workspace) -> IncidentSummary | None:
    st.title("Incidents")
    st.html(f"<div class='ds-provenance'>{escape(_provenance(ws))}</div>")
    if not ws.summaries:
        st.warning("This incident source contains no incidents.")
        return None
    options = [s.incident_id for s in sorted(ws.summaries, key=lambda s: (s.day, s.rank or 10_000))]
    labels = {s.incident_id: f"Day {s.day} · {s.incident.user_id} · {s.headline}" for s in ws.summaries}
    choice = st.selectbox("Open an incident", options, index=None, format_func=labels.get, placeholder="Choose an incident")
    if choice:
        _select(choice)
        st.rerun()
    st.caption("Or pick one from the Queue page.")
    return None


def incident_page() -> None:
    ws = _workspace()
    if ws is None:
        return
    summary = ws.by_id.get(st.session_state.get("selected_incident_id") or "")
    if summary is None:
        _incident_picker(ws)
        return
    ordered = sorted(ws.summaries, key=lambda s: (s.day, s.rank if s.rank is not None else 10_000, s.incident.start_time))
    ids = [s.incident_id for s in ordered]
    position = ids.index(summary.incident_id)

    st.html(f"<div class='ds-crumb'>Queue › Day {summary.day} › {ui.mono(summary.incident_id)}</div>")
    title, buttons = st.columns([4, 1.3], vertical_alignment="center")
    title.title(f"{summary.incident.user_id} — {summary.headline}")
    with buttons.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Previous incident", disabled=position == 0, key="prev_incident"):
            _select(ids[position - 1])
            st.rerun()
        if st.button("Next incident", disabled=position == len(ids) - 1, key="next_incident"):
            _select(ids[position + 1])
            st.rerun()
    rows = _events_of(ws, summary)
    hosts, _ = _host_counts(rows)
    facts = [ui.badge(_hour_range(summary)), ui.badge(f"{summary.event_count} events")]
    if ws.events is not None:
        facts.append(ui.badge(f"{hosts} hosts"))
    st.html(f"<div>{_badges(summary, extra=facts)}</div>")

    pairs = len(host_pair_table(rows)) if not rows.empty else 0
    labels = {"Overview": "Overview", "Timeline": f"Timeline ({len(rows)})", "Hosts": f"Hosts ({pairs})",
              "Evidence": "Evidence", "Investigation": "Investigation"}
    wanted = st.session_state.pop("incident_tab", None)
    tabs = st.tabs(list(labels.values()), default=labels.get(wanted) if wanted else None,
                   key=f"incident_tabs_{summary.incident_id}_{wanted or ''}")
    with tabs[0]:
        _overview_tab(ws, summary, rows)
    with tabs[1]:
        _timeline_tab(ws, summary, rows)
    with tabs[2]:
        _hosts_tab(rows)
    with tabs[3]:
        _evidence_tab(summary)
    with tabs[4]:
        _investigation_tab(summary)


# ─── Evidence page ─────────────────────────────────────────────────────


def evidence_page() -> None:
    ws = _workspace()
    if ws is None:
        return
    st.title("Evidence")
    st.html(f"<div class='ds-provenance'>Look up any cited authentication event by its source line.</div>")
    if ws.events is None:
        st.info("Event details are unavailable for this source: there is no events.parquet next to the incident file.")
        return
    reference = st.text_input("Event ID", placeholder="auth.txt:463603187", key="evidence_lookup").strip()
    if not reference:
        st.caption(f"{len(ws.events):,} events from {len(ws.summaries)} incidents are available.")
        return
    if reference not in ws.events.index:
        st.warning(f"{reference} is not cited by any incident in this package. Event IDs look like auth.txt:463603187.")
        return
    citing = [s for s in ws.summaries if reference in set(s.incident.raw.get("source_references") or [])]
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        _record_table(ws.events.loc[reference])
    with right:
        st.html("<div class='ds-section'>Cited by</div>")
        for s in citing:
            with st.container(border=True, key=f"card_cite_{citing.index(s)}"):
                st.html(f"{ui.priority_badge(s.incident.priority)} <b>{escape(s.incident.user_id)}</b> — {escape(s.headline)}"
                        f"<br>{ui.mono(s.incident_id)}")
                if st.button("Open incident", key=f"cite_{s.incident_id}", icon=":material/open_in_new:"):
                    _select(s.incident_id)
                    st.switch_page(_pages()["incident"])


# ─── About page ────────────────────────────────────────────────────────


def about_page() -> None:
    st.title("About the model")
    st.html("<div class='ds-provenance'>How to read this dashboard, in plain words.</div>")
    left, right, _ = st.columns([1, 1, 0.45], gap="large")
    with left:
        st.markdown(
            "**What DualScope does.** It reads the LANL enterprise logon records and, for every user and hour, "
            "scores how unusual the activity looks. The final model is a gradient-boosted classifier over a GRU "
            "sequence score and hourly counts (new hosts, NTLM, failures and similar). Each day, the "
            f"{QUEUE_SIZE} highest-scoring user-hours become alerts, and alerts for the same user a few hours apart "
            "form one incident.\n\n"
            "**The day cut-off.** The model gives many hours exactly the same score. Hours scored above the "
            "lowest queued score are in the queue whatever happens (HIGH). Hours tied at that score fill the last "
            "places, picked by a fixed tie-break (MEDIUM); their rank among each other means nothing.\n\n"
            "**Scores.** The fusion score ranks user-hours; it is not the probability of an attack."
        )
    with right:
        st.markdown(
            "**Why flagged.** Each incident's events are checked against four rules: failed logons, an NTLM "
            "logon to a host the user never reached before, any network logon to a new host, and a logon from a "
            "new source computer. *New* means the first time since Day 1; log-offs are not counted.\n\n"
            "**ATT&CK candidates.** Each behaviour is matched against MITRE ATT&CK 19.2 to suggest techniques. "
            "They are leads, not findings, until verification marks them supported or rejected.\n\n"
            "**How well it did.** On the held-out test days 17–30 the queue held 1 of the 39 red-team user-hours. "
            "Its average precision was about six times the sequence detector's alone (0.00124 against 0.00020), but that "
            "did not turn into more catches at 38 alerts a day. The labels are incomplete and come from one red team."
        )
    st.caption("Switch on **Answer key** in the sidebar to see which alerts were red-team activity. Keep it off for analyst review.")


# ─── App shell ─────────────────────────────────────────────────────────

def _pages() -> dict[str, st.Page]:
    """Fresh page objects for this run; st.Page holds per-session state, so pages are not shared across viewers."""
    return {
        "queue": st.Page(queue_page, title="Queue", icon=":material/format_list_bulleted:", url_path="queue", default=True),
        "incident": st.Page(incident_page, title="Incidents", icon=":material/report:", url_path="incident"),
        "evidence": st.Page(evidence_page, title="Evidence", icon=":material/description:", url_path="evidence"),
        "about": st.Page(about_page, title="About the model", icon=":material/info:", url_path="about"),
    }


def run() -> None:
    st.set_page_config(page_title="DualScope", page_icon=":material/radar:", layout="wide", initial_sidebar_state="auto")
    st.html(ui.CSS)
    page = st.navigation(list(_pages().values()))
    # Deep links (?incident=<id>&tab=Hosts) select an incident once per link.
    link = (st.query_params.get("incident"), st.query_params.get("tab"))
    if link[0] and st.session_state.get("_deep_link") != link:
        st.session_state["_deep_link"] = link
        _select(link[0])
        if link[1] in TAB_LABELS:
            st.session_state["incident_tab"] = link[1]
    st.session_state["_source"] = _source_controls()
    st.sidebar.toggle("Answer key (evaluation only)", key="show_answer_key",
                      help="Shows which alerts were red-team activity. Keep off for analyst review.")
    page.run()
