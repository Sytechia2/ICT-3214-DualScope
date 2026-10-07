"""DualScope analyst dashboard: alert queue (Task 7.1) and incident evidence (Task 7.2).

Four pages share one data source: the queue for a day with an incident side
panel, the incident page (overview, timeline, hosts, evidence, investigation),
an event lookup, and a plain-language page about the model. Layout borrows from
established SOC tools (Defender-style queue and incident page, Elastic-style
summary strip and field table); colours and fonts are in .streamlit/config.toml.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dualscope.attack.retrieval import TechniqueRetriever
from dualscope.dashboard import investigation, ui
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
    matches_rule,
    matches_search,
    summarise_incidents,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDOFF = REPOSITORY_ROOT / "outputs" / "handoff" / "final_test_alerts_v1" / "incidents.jsonl"
HANDOFF_SOURCE = "Alert package (days 17–30)"
FIXTURE_SOURCE = "Synthetic fixture"
OTHER_SOURCE = "Other JSONL export"
QUEUE_SIZE = 38
TAB_LABELS = ("Overview", "Timeline", "Connections", "Model details", "Investigation")
TAB_ALIASES = {"Evidence": "Model details", "Hosts": "Connections"}  # older deep links
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
        st.info("Set an incidents .jsonl path under **Data source**.")
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
        return "Synthetic fixture · demo data"
    if ws.is_package:
        return f"Final model · Test days 17–30 · {QUEUE_SIZE} alerts/day · Offline package"
    return f"{ws.path.name} · Offline export"


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


def _request_tab(tab: str) -> None:
    """Open ``tab`` on the incident page at its next render (a fresh tabs widget picks up the default)."""
    st.session_state["incident_tab"] = tab
    st.session_state["tab_request"] = st.session_state.get("tab_request", 0) + 1


def _select(incident_id: str) -> None:
    st.session_state["selected_incident_id"] = incident_id
    st.session_state["queue_nonce"] = st.session_state.get("queue_nonce", 0) + 1


# ─── Shared pieces ─────────────────────────────────────────────────────


def _badges(summary: IncidentSummary) -> str:
    """Priority badge, plus the answer-key badge when that is switched on."""
    parts = [ui.priority_badge(summary.incident.priority)]
    if st.session_state.get("show_answer_key") and summary.redteam is not None:
        parts.append(ui.badge("Red-team activity" if summary.redteam else "Not red-team", "answer"))
    return "".join(parts)


def _rank_text(summary: IncidentSummary) -> str | None:
    """Rank only means something for HIGH; MEDIUM incidents were tied at the cut-off."""
    if summary.rank is None:
        return None
    if not summary.above_cutoff:
        return "Tied at cut-off"
    return f"Rank {summary.rank} of {QUEUE_SIZE} · above cut-off"


def _first_time_counts(rows: pd.DataFrame) -> tuple[int, int]:
    """First-time connections (source → destination pairs) and how many of them used NTLM."""
    if rows.empty:
        return 0, 0
    pairs = host_pair_table(rows)
    new = pairs[pairs[["New destination for user", "New host pair", "New source for user"]].any(axis=1)]
    return len(new), int(new["Auth types"].str.contains("NTLM").sum())


def _fact_line(summary: IncidentSummary, rows: pd.DataFrame, *, full: bool) -> str:
    """One line of counts taken from the data. ``full`` adds rank and computers (incident page)."""
    first, ntlm = _first_time_counts(rows)
    connections = f"{first} first-time connection{'s' if first != 1 else ''}"
    parts = []
    if full:
        if summary.rank is not None:
            parts.append(f"Rank {summary.rank} of {QUEUE_SIZE}" if summary.above_cutoff else "tied at cut-off")
        parts += [_hour_range(summary), f"{summary.event_count} log lines"]
        if not rows.empty:
            parts.append(f"{_host_counts(rows)[0]} computers")
        if first:
            parts.append(connections + (f" ({ntlm} over NTLM)" if ntlm else ""))
    else:
        if first:
            parts.append(connections)
            if ntlm:
                parts.append(f"{ntlm} over NTLM")
        parts += [f"{summary.event_count} log lines", _hour_range(summary)]
    return " · ".join(p for p in parts if p)


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
        return "<div class='ds-reason ds-muted'>No event data for this source.</div>"
    if not summary.behaviours:
        return ("<div class='ds-reason'><b>No rule matched</b><br>"
                "<span class='ds-muted'>Ranked on overall activity</span></div>")
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
        return "<div class='ds-muted'>None</div>"
    return "".join(
        f"<div class='ds-candidate'><span>{ui.mono(c['technique_id'])} {escape(c['name'])}</span>"
        "</div>"
        for c in summary.candidates[:limit]
    )


def _host_counts(rows: pd.DataFrame) -> tuple[int, int]:
    if rows.empty:
        return 0, 0
    pairs = host_pair_table(rows)
    new = pairs[["New destination for user", "New host pair", "New source for user"]].any(axis=1)
    return int(rows["destination_computer"].nunique()), int(new.sum())


# ─── Queue page ────────────────────────────────────────────────────────


def _pick(key: str, field: str) -> str | None:
    """The mark clicked in the chart under ``key``, if any."""
    state = st.session_state.get(key)
    picked = state.selection.get("pick") if state is not None else None
    return str(picked[0][field]) if picked else None


def _pick_rule(key: str) -> None:
    """Chart click: filter the queue to that rule; clicking the active rule clears it."""
    rule = _pick(key, "behaviour")
    if rule is None:
        return
    st.session_state["f_rule"] = None if rule == st.session_state.get("f_rule") else rule
    st.session_state["rule_nonce"] = st.session_state.get("rule_nonce", 0) + 1  # a fresh chart can be clicked again


def _clear_rule() -> None:
    st.session_state["f_rule"] = None
    st.session_state["rule_nonce"] = st.session_state.get("rule_nonce", 0) + 1


def _pick_day(key: str) -> None:
    day = _pick(key, "Day")
    if day is not None:
        st.session_state["day"] = int(day)


def _queue_by_day_chart(counts: pd.DataFrame, day: int) -> alt.Chart:
    """HIGH hours rise above a zero line in red, MEDIUM hours hang below it in amber."""
    days = counts.assign(Day=counts["day"].astype(str)).rename(columns={"above": "HIGH", "tied": "MEDIUM"})
    days["below"] = -days["MEDIUM"]
    order = list(days["Day"])
    high, low = int(days["HIGH"].max()) + 9, -int(days["MEDIUM"].max()) - 9
    scale = alt.Scale(domain=[low, high])
    x = alt.X("Day:N", sort=order, scale=alt.Scale(domain=order), title=None,
              axis=alt.Axis(labelAngle=0, labelFontSize=11, labelColor=ui.INK, labelOverlap=False, labelPadding=4, ticks=False,
                            domain=False))
    band = (alt.Chart(days[days["day"] == day]).mark_bar(width={"band": 1.0}, color=ui.CHART_BAND, cornerRadius=3)
            .encode(x=x, y=alt.Y("hi:Q", scale=scale, axis=None), y2="lo:Q")
            .transform_calculate(hi=str(high), lo=str(low)))
    up = alt.Chart(days).mark_bar(width={"band": 0.62}, color=ui.HIGH, cursor="pointer").encode(
        x=x, y=alt.Y("HIGH:Q", scale=scale, axis=None), y2=alt.datum(0))
    down = alt.Chart(days).mark_bar(width={"band": 0.62}, color=ui.MEDIUM, cursor="pointer").encode(
        x=x, y=alt.Y("below:Q", scale=scale, axis=None), y2=alt.datum(0))
    up_text = alt.Chart(days).mark_text(dy=-7, fontSize=11, fontWeight=600, color=ui.INK).encode(
        x=x, y=alt.Y("HIGH:Q", scale=scale, axis=None), text="HIGH:Q")
    down_text = alt.Chart(days).mark_text(dy=8, fontSize=11, fontWeight=600, color=ui.INK).encode(
        x=x, y=alt.Y("below:Q", scale=scale, axis=None), text="MEDIUM:Q")
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#A3A9B1").encode(y=alt.Y("zero:Q", scale=scale, axis=None))
    # A full-height invisible bar per day, so a click anywhere in the column picks that day.
    pick = alt.selection_point(name="pick", fields=["Day"])
    hit = (alt.Chart(days).mark_bar(width={"band": 1.0}, opacity=0, cursor="pointer").add_params(pick)
           .encode(x=x, y=alt.Y("hi:Q", scale=scale, axis=None), y2="lo:Q",
                   tooltip=[alt.Tooltip("Day:N", title="Test day"), "HIGH:Q", "MEDIUM:Q"])
           .transform_calculate(hi=str(high), lo=str(low)))
    return (band + up + down + zero + up_text + down_text + hit).properties(height=190).configure_view(stroke=None)


def _why_flagged_chart(totals: pd.DataFrame, total: int, rule: str | None) -> alt.Chart:
    """Bars for one day's incidents; the x-scale runs to the day's incident total."""
    short = {headline: CHIP_LABELS[key] for key, headline in HEADLINES.items()}
    totals = totals.assign(kind=["none" if b == NO_EVIDENCE else "rule" for b in totals["behaviour"]],
                           behaviour=[short.get(b, b) for b in totals["behaviour"]])
    totals["label"] = totals["incidents"].astype(str) + f" of {total}"
    totals["alpha"] = [0.35 if rule and b != rule else 1.0 for b in totals["behaviour"]]
    pick = alt.selection_point(name="pick", fields=["behaviour"])
    scale = alt.Scale(domain=[0, max(1, total)])
    base = alt.Chart(totals).encode(
        y=alt.Y("behaviour:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=160, labelFontSize=12, labelColor=ui.INK, ticks=False, domain=False)),
        x=alt.X("incidents:Q", title=None, axis=None, scale=scale),
    )
    # The whole row is clickable, not just the (sometimes tiny) bar.
    totals["zero"], totals["full"] = 0, max(1, total)
    hit = (alt.Chart(totals).mark_bar(opacity=0, height=26, cursor="pointer").add_params(pick)
           .encode(y="behaviour:N", x=alt.X("zero:Q", scale=scale), x2="full:Q",
                   tooltip=["behaviour:N", alt.Tooltip("label:N", title="incidents")]))
    bars = base.mark_bar(cornerRadiusEnd=3, height=18, cursor="pointer").encode(
        color=alt.Color("kind:N", scale=alt.Scale(domain=["rule", "none"], range=[ui.CHART, "#B8BEC6"]), legend=None),
        opacity=alt.Opacity("alpha:Q", scale=None, legend=None),
    )
    labels = base.mark_text(align="left", dx=4, fontSize=12, color=ui.INK).encode(text="label:N")
    return (bars + labels + hit).properties(height=150, padding={"left": 0, "right": 52, "top": 4, "bottom": 4}).configure_view(stroke=None)


def _glance_html(day_summaries: list[IncidentSummary]) -> str:
    high = sum(s.incident.priority in ("HIGH", "CRITICAL") for s in day_summaries)
    medium = sum(s.incident.priority == "MEDIUM" for s in day_summaries)
    users = len({s.incident.user_id for s in day_summaries})
    return (
        "<div class='ds-glance'>"
        f"<div class='big'><b>{len(day_summaries)}</b><span>incidents{ui.tip_html('Incident')}</span></div>"
        f"<div class='row high'><span>HIGH{ui.tip_html('HIGH')}</span><b>{high}</b></div>"
        f"<div class='row medium'><span>MEDIUM{ui.tip_html('MEDIUM')}</span><b>{medium}</b></div>"
        + (f"<div class='row'><span>distinct users{ui.tip_html('Distinct users')}</span><b>{users}</b></div>"
           if users < len(day_summaries) else "")
        + "</div>"
    )


def _summary_strip(ws: Workspace, day: int, day_summaries: list[IncidentSummary]) -> None:
    counts = day_cutoff_counts(s.incident for s in ws.summaries)
    glance, why, days = st.columns([0.62, 1.15, 1.35])
    with glance.container(border=True, height=240, key="card_day_glance"):
        st.html(f"<div class='ds-section'>Day {day} at a glance</div>")
        st.html(_glance_html(day_summaries))
    with why.container(border=True, height=240, key="card_why_flagged"):
        st.html(f"<div class='ds-section'>Why flagged · Day {day}{ui.tip_html('Why flagged chart')}</div>")
        evidenced = [s for s in day_summaries if s.events_available]
        if evidenced:
            key = f"why_chart_{st.session_state.get('rule_nonce', 0)}"
            st.altair_chart(_why_flagged_chart(behaviour_totals(evidenced), len(evidenced), st.session_state.get("f_rule")),
                            width="stretch", key=key, on_select=functools.partial(_pick_rule, key), selection_mode="pick")
            st.caption("An incident can match several rules")
        else:
            st.caption("No event data (events.parquet missing).")
    with days.container(border=True, height=240, key="card_queue_by_day"):
        st.html(f"<div class='ds-section'>HIGH ↑ / MEDIUM ↓ · days 17–30{ui.tip_html('HIGH vs MEDIUM')}</div>")
        if counts.empty:
            st.caption("No cut-off data for this source.")
        else:
            key = f"day_chart_{day}"  # a new key per day drops the old click
            st.altair_chart(_queue_by_day_chart(counts, day), width="stretch", key=key,
                            on_select=functools.partial(_pick_day, key), selection_mode="pick")


def _queue_frame(summaries: list[IncidentSummary], selected: str | None, answer_key: bool, columns: tuple[str, ...]):
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
            "Log lines": s.event_count,
            "Likely technique": f"{candidate['technique_id']} {candidate['name']}" if candidate else "—",
        }
        if answer_key:
            row["Red-team (answer key)"] = s.redteam
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[[c for c in frame.columns if c in columns or c in ("Incident", "Red-team (answer key)")]]

    def style_row(row: pd.Series) -> list[str]:
        tint = ""
        if answer_key and row.get("Red-team (answer key)"):
            tint = f"background-color: {ui.HIGH_TINT};"
        if row["Incident"] == selected:
            tint = f"background-color: {ui.ACCENT_TINT}; font-weight: 600;"
        return [_priority_cell(row[c]) if c == "Priority" else tint for c in row.index]

    return frame.style.apply(style_row, axis=1) if not frame.empty else frame


def _why_text(summary: IncidentSummary) -> str:
    labels = summary.chips or [NO_EVIDENCE if summary.events_available else "—"]
    extra = f" +{len(labels) - 2}" if len(labels) > 2 else ""
    return " · ".join(labels[:2]) + extra


def _priority_cell(priority: str) -> str:
    if priority in ("HIGH", "CRITICAL"):
        return f"background-color: {ui.HIGH}; color: #FFFFFF; font-weight: 700;"
    if priority == "MEDIUM":
        return f"background-color: {ui.MEDIUM}; color: {ui.MEDIUM_INK}; font-weight: 700;"
    return ""


QUEUE_COLUMNS = {
    "Priority": st.column_config.TextColumn(width=60),
    "Rank": st.column_config.NumberColumn(width=40, format="%d", help=ui.tip("Rank")),
    "Incident": None,  # shown in the side panel; kept in the frame to tint the selected row
    "User": st.column_config.TextColumn(width=104, help=ui.tip("Incident")),
    "Hour": st.column_config.TextColumn(width=50, help=ui.tip("Queue")),
    "Why flagged": st.column_config.TextColumn(width=206, help=ui.tip("Why-flagged rules")),
    "Log lines": st.column_config.NumberColumn(width=62, format="%d", help=ui.tip("Event")),
    "Likely technique": st.column_config.TextColumn(width=236, help=ui.tip("ATT&CK candidate")),
}
HIGH_COLUMNS = ("Rank", "User", "Hour", "Why flagged", "Log lines", "Likely technique")
MEDIUM_COLUMNS = tuple(c for c in HIGH_COLUMNS if c != "Rank")  # tie-break order means nothing
ALL_COLUMNS = ("Priority", *HIGH_COLUMNS)
ROW_HEIGHT = 32
HEADER_HEIGHT = 35  # Streamlit dataframe header plus borders


def _whole_rows(count: int, limit: int) -> int:
    """Table height showing whole rows only (header plus up to ``limit`` rows)."""
    return min(count, limit) * ROW_HEIGHT + HEADER_HEIGHT


def _queue_table(summaries: list[IncidentSummary], key: str, selected: str | None, limit: int,
                 columns: tuple[str, ...]) -> None:
    if not summaries:
        return
    data = _queue_frame(summaries, selected, bool(st.session_state.get("show_answer_key")), columns)
    event = st.dataframe(
        data, hide_index=True, column_config=QUEUE_COLUMNS, on_select="rerun", selection_mode="single-row",
        key=f"{key}_{st.session_state.get('queue_nonce', 0)}", row_height=ROW_HEIGHT,
        height=_whole_rows(len(summaries), limit),
    )
    if event.selection.rows:
        _select(summaries[event.selection.rows[0]].incident_id)
        st.rerun()


def _queue_group(summaries: list[IncidentSummary], key: str, selected: str | None, limit: int,
                 title: str, kind: str, columns: tuple[str, ...], term: str | None = None) -> None:
    """One queue group as a card: a header bar (name and incident count) above its table."""
    if not summaries:
        return
    with st.container(border=True, key=f"card_{key}"):
        st.html(f"<div class='ds-group {escape(kind)}'><span class='ds-group-title'>{escape(title)}"
                f"{ui.tip_html(term) if term else ''}</span><span class='ds-count'>{len(summaries)}</span></div>")
        _queue_table(summaries, key, selected, limit, columns)


def _side_panel(ws: Workspace, summary: IncidentSummary, visible: list[IncidentSummary], filtered: bool) -> None:
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
    nav.caption(f"{position + 1} of {len(ids)} {'shown' if filtered else 'on this day'}")
    rows = _events_of(ws, summary)
    unusual = []
    if (rank := _rank_text(summary)):
        unusual.append(f"<div class='ds-fact'>{escape(rank)}</div>")
    percentile = _gru_percentile(summary)
    if percentile != "—":
        unusual.append(f"<div class='ds-fact'>Sequence more unusual than {escape(percentile)} of user-hours that day</div>")
    st.html(
        f"<h3 style='margin:0.1rem 0 0.1rem'>{escape(summary.incident.user_id)} — {escape(summary.headline)}</h3>"
        f"<div style='margin-bottom:0.55rem'>{ui.mono(summary.incident_id)}</div>{_badges(summary)}"
        f"<div class='ds-muted' style='margin:0.1rem 0 0.4rem'>{escape(_fact_line(summary, rows, full=False))}</div>"
        f"<div class='ds-section' style='margin-top:0.6rem'>Why flagged{ui.tip_html('Why-flagged rules')}</div>{_reasons_html(ws, summary)}"
        f"<div class='ds-section' style='margin-top:0.55rem'>How unusual{ui.tip_html('Sequence percentile')}</div>{''.join(unusual)}"
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

        rule = st.session_state.get("f_rule")
        if rule:
            active = st.container(horizontal=True, vertical_alignment="center")
            active.html(f"<div>{ui.badge('Why flagged: ' + rule)}</div>")
            active.button("Clear filter", icon=":material/close:", type="tertiary", key="clear_rule", on_click=_clear_rule)

        visible = [s for s in day_summaries if matches_rule(s, rule) and matches_search(s, query, ws.events)]
        if visible and st.session_state.get("selected_incident_id") not in {s.incident_id for s in visible}:
            st.session_state["selected_incident_id"] = visible[0].incident_id
        selected_id = st.session_state.get("selected_incident_id")

        if not visible:
            st.info("No incidents on this day match the filters. Clear a filter or pick another day.")
        above = [s for s in visible if s.above_cutoff]
        tied = sorted((s for s in visible if not s.above_cutoff), key=lambda s: s.incident.start_time)
        if not ws.is_package:
            _queue_group(visible, "queue_all", selected_id, 12, "Incidents", "neutral", ALL_COLUMNS)
        else:
            _queue_group(above, "queue_above", selected_id, 12, "HIGH · above cut-off", "high", HIGH_COLUMNS, "HIGH")
            _queue_group(tied, "queue_tied", selected_id, 8, "MEDIUM · tied at cut-off", "medium", MEDIUM_COLUMNS, "MEDIUM")
    with panel:
        if visible:
            prio = {"HIGH": "high", "CRITICAL": "high", "MEDIUM": "medium"}.get(ws.by_id[selected_id].incident.priority, "none")
            with st.container(border=True, key=f"card_panel_{prio}"):
                _side_panel(ws, ws.by_id[selected_id], visible, bool(rule or query.strip()))


# ─── Incident page ─────────────────────────────────────────────────────


BIN_MINUTES = (1, 2, 5, 10, 15, 30)


def _bin_minutes(span_seconds: int) -> int:
    """Smallest standard bin that gives about 30 bins across the incident's span."""
    wanted = span_seconds / 60 / 30
    return next((m for m in BIN_MINUTES if m >= wanted), BIN_MINUTES[-1])


def _activity_chart(summary: IncidentSummary, rows: pd.DataFrame, minutes: int) -> alt.Chart:
    """Log lines per time bin as bars, with a red rule and marker at each flagged event."""
    start, end = summary.incident.start_time, summary.incident.end_time
    origin = _BASE + timedelta(seconds=(start - 1) % 86_400)
    at = lambda seconds: origin + timedelta(seconds=int(seconds))  # noqa: E731
    width = minutes * 60
    count = max(1, -(-(end - start) // width))
    slot = ((rows["timestamp"] - start) // width).clip(0, count - 1)
    per_slot = slot.value_counts().reindex(range(count), fill_value=0)
    bins = pd.DataFrame({
        "slot_start": [at(i * width) for i in range(count)], "slot_end": [at((i + 1) * width) for i in range(count)],
        "log lines": per_slot.to_numpy(),
    })
    bins["slot"] = bins["slot_start"].dt.strftime("%H:%M") + "–" + bins["slot_end"].dt.strftime("%H:%M")

    rules_by_ref: dict[str, list[str]] = {}
    for behaviour in summary.ordered_behaviours:
        for ref in behaviour["evidence_references"]:
            rules_by_ref.setdefault(ref, []).append(CHIP_LABELS[behaviour["behaviour"]])
    flagged = rows[rows["source_reference"].isin(rules_by_ref)]
    marks = pd.DataFrame({
        "time": [at(t - start) for t in flagged["timestamp"]], "event": flagged["source_reference"].to_numpy(),
        "route": (flagged["source_computer"] + " → " + flagged["destination_computer"]).to_numpy(),
        "rule": [", ".join(rules_by_ref[r]) for r in flagged["source_reference"]],
    })
    domain = [origin, at(end - start)]
    x = alt.X("time:T", title=None, scale=alt.Scale(domain=domain),
              axis=alt.Axis(format="%H:%M", labelFontSize=12, labelColor=ui.MUTED, grid=False, tickCount=6))
    bars = alt.Chart(bins[bins["log lines"] > 0]).mark_bar(color=ui.CHART, stroke="#FFFFFF", strokeWidth=1).encode(
        x=alt.X("slot_start:T", title=None, scale=alt.Scale(domain=domain),
                axis=alt.Axis(format="%H:%M", labelFontSize=12, labelColor=ui.MUTED, grid=False, tickCount=6)),
        x2="slot_end:T", y2=alt.datum(0),
        y=alt.Y("log lines:Q", title=None, scale=alt.Scale(domain=[0, max(1, int(bins["log lines"].max()))], nice=True),
                axis=alt.Axis(tickCount=4, grid=True, gridColor=ui.LINE, labelFontSize=11, labelColor=ui.MUTED,
                              ticks=False, domain=False)),
        tooltip=[alt.Tooltip("slot:N", title="Time slot"), alt.Tooltip("log lines:Q", title="Log lines")],
    )
    layers = [bars]
    if not marks.empty:
        tooltip = [alt.Tooltip("time:T", format="%H:%M:%S"), alt.Tooltip("event:N", title="Event"),
                   alt.Tooltip("route:N", title="Route"), alt.Tooltip("rule:N", title="Rule")]
        layers.append(alt.Chart(marks).mark_rule(color=ui.HIGH, strokeWidth=1.5).encode(x=x, tooltip=tooltip))
        layers.append(alt.Chart(marks).mark_point(shape="triangle-down", filled=True, color=ui.HIGH, size=70, opacity=1)
                      .encode(x=x, y=alt.value(4), tooltip=tooltip))
    return alt.layer(*layers).properties(height=220).configure_view(stroke=None)


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
                        f" · {count} event{'s' if count != 1 else ''}<br>"
                        f"{_first_event_line(ws, example)}</div>")
                if st.button("Show in timeline", key=f"show_{index}", type="tertiary", icon=":material/arrow_forward:"):
                    _request_tab("Timeline")
                    st.session_state["timeline_focus"] = example
                    st.rerun()
        if not summary.ordered_behaviours:
            with st.container(border=True, key="card_reason_none"):
                st.html(_reasons_html(ws, summary))
    with middle:
        minutes = _bin_minutes(summary.incident.end_time - summary.incident.start_time)
        st.html(f"<div class='ds-section'>When it happened · log lines per {minutes} min</div>")
        if rows.empty:
            st.info("Event details are unavailable for this source.")
        else:
            with st.container(border=True, key="card_activity"):
                st.altair_chart(_activity_chart(summary, rows, minutes), width="stretch")
                st.html(f"<div class='ds-legend'><span class='ds-swatch' style='background:{ui.CHART}'></span>log lines"
                        f"<span class='ds-swatch' style='background:{ui.HIGH};width:0.2rem;height:0.9rem'></span>flagged event</div>")
            first = _first_time_pairs(rows)
            st.html(f"<div class='ds-section' style='margin-top:0.6rem'>First-time connections{ui.tip_html('New')}</div>")
            if first.empty:
                st.caption("None")
            else:
                st.dataframe(first.style.map(lambda _: f"color: {ui.HIGH}; font-weight: 600;", subset=["Source → destination"]),
                             hide_index=True, row_height=ROW_HEIGHT)
    with right:
        _unusual_card(summary)
        with st.container(border=True, key="card_candidates"):
            st.html(f"<div class='ds-section'>Possible techniques (unverified){ui.tip_html('ATT&CK candidate')}</div>"
                    f"{_candidates_html(summary)}")


def _unusual_card(summary: IncidentSummary) -> None:
    hours = summary.incident.raw.get("alert_hours") or []
    items = []
    percentile = _gru_percentile(summary)
    if percentile != "—":
        items.append(f"<div><b>{escape(percentile)}</b><span>of user-hours that day are less unusual</span></div>")
    if len(hours) > 1:
        items.append(f"<div><b>{len(hours)}</b><span>alert hours</span></div>")
    with st.container(border=True, key="card_unusual"):
        st.html(f"<div class='ds-section'>How unusual{ui.tip_html('Sequence percentile')}</div>"
                f"<div class='ds-stats'>{''.join(items)}</div>")


def _gru_percentile(summary: IncidentSummary) -> str:
    hours = summary.incident.raw.get("alert_hours") or []
    gru = max((h["gru_percentile_in_day"] for h in hours), default=None)
    return f"{gru:.2%}" if gru is not None else "—"


def _timeline_tab(ws: Workspace, summary: IncidentSummary, rows: pd.DataFrame) -> None:
    if rows.empty:
        st.info("No event data for this source.")
        st.dataframe({"Event ID": summary.incident.raw.get("source_references") or []}, hide_index=True)
        return
    cited = list(dict.fromkeys(r for b in summary.ordered_behaviours for r in b["evidence_references"]))
    focus = st.session_state.get("timeline_focus")
    options = [f"Flagged events ({len(cited)})", f"All events ({len(rows)})"] if cited else [f"All events ({len(rows)})"]
    choice = st.segmented_control("Show", options, default=options[0],
                                  key=f"timeline_view_{summary.incident_id}_{focus or ''}")
    shown = rows[rows["source_reference"].isin(cited)] if choice == options[0] and cited else rows
    shown = shown.reset_index(drop=True)

    present = list(shown["source_reference"])
    target = focus if focus in present else next((r for r in present if r in set(cited)), present[0])
    visible_rows = min(len(shown), max(12, present.index(target) + 2), 24)
    table = timeline_table(shown)
    flagged = set(cited)

    def style_event(row: pd.Series) -> list[str]:
        if row["Event ID"] == target:
            css = f"background-color: {ui.ACCENT_TINT}; font-weight: 600;"
        elif row["Event ID"] in flagged:
            css = f"background-color: {ui.HIGH_TINT};"
        else:
            css = ""
        return [css + (f"color: {ui.HIGH}; font-weight: 600;" if c == "Flags" and "new" in str(row[c]) else "") for c in row.index]

    styled = table.style.apply(style_event, axis=1)
    event = st.dataframe(styled, hide_index=True, on_select="rerun", selection_mode="single-row",
                         column_config={"Time": st.column_config.TextColumn(width=130),
                                        "Event ID": st.column_config.TextColumn(width=135, help=ui.tip("Event")),
                                        "Source": st.column_config.TextColumn(width=75),
                                        "Destination": st.column_config.TextColumn(width=85),
                                        "Auth type": st.column_config.TextColumn(width=75),
                                        "Logon type": st.column_config.TextColumn(width=80),
                                        "Orientation": st.column_config.TextColumn(width=85),
                                        "Result": st.column_config.TextColumn(width=70),
                                        "Flags": st.column_config.TextColumn(width=380, help=ui.tip("New"))},
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
    dot, _ = _host_graph(pairs)
    with st.container(border=True, key="card_host_graph"):
        st.graphviz_chart(dot, width="stretch", height=470)
    st.caption("Red = first-time connection · Dark box = user's main computer")
    st.html("<div class='ds-section' style='margin-top:0.6rem'>Source → destination pairs</div>")
    st.dataframe(pairs, hide_index=True, row_height=ROW_HEIGHT, height=min(len(pairs), 15) * ROW_HEIGHT + HEADER_HEIGHT, column_config={
        "Events": st.column_config.NumberColumn(width=70),
        "Failures": st.column_config.NumberColumn(width=80),
        "New destination for user": st.column_config.CheckboxColumn(width=180, help=ui.tip("New")),
        "New host pair": st.column_config.CheckboxColumn(width=120, help=ui.tip("New")),
        "New source for user": st.column_config.CheckboxColumn(width=160, help=ui.tip("New")),
    })


def _model_details_tab(summary: IncidentSummary) -> None:
    raw = summary.incident.raw
    if raw.get("alert_hours"):
        st.html("<div class='ds-section'>Alerted hours</div>")
        st.dataframe(alert_hours_table(raw, bool(st.session_state.get("show_answer_key"))), hide_index=True,
                     column_config={"Tied at cut-off": st.column_config.CheckboxColumn(help=ui.tip("Tied at cut-off"))})
    st.html("<div class='ds-section' style='margin-top:0.6rem'>Model scores</div>")
    fusion, graph, _ = st.columns(3)
    fusion.metric("Fusion score", format_score(summary.incident.max_fused_score), help=ui.tip("Fusion score"), border=True)
    graph.metric("Graph score", format_score(summary.incident.graph.max_score), help=ui.tip("Graph score"), border=True)
    context = raw.get("graph_context")
    if context:
        st.html(f"<div class='ds-section' style='margin-top:0.6rem'>Graph detector context{ui.tip_html('Graph context')}</div>")
        edges_new, growth, _ = st.columns(3)
        edges_new.metric("New edges", str(context.get("new_edge_count", "—")), help=ui.tip("New edges"), border=True)
        growth.metric("Degree growth", str(context.get("degree_growth", "—")), help=ui.tip("Degree growth"), border=True)
        edges = graph_edges_table(raw)
        if edges.empty:
            st.info("No graph edges for this user-day.")
        else:
            st.dataframe(edges, hide_index=True,
                         column_config={"Edge score (raw)": st.column_config.NumberColumn(help=ui.tip("Edge score"))})


@st.cache_resource(show_spinner=False)
def _investigation_views(folder: str, modified: tuple[tuple[str, float], ...]):
    return investigation.load_run(Path(folder), modified)


def _claims_html(items: list[dict], *, confidence: bool = False) -> str:
    rows = []
    for item in items:
        check = item.get("check") or {}
        mark = ""
        if check.get("status") == "partly_supported":
            reasons = "; ".join(check.get("reasons") or []) or "Partly supported"
            mark = f'<span class="ds-partly" title="{escape(reasons, quote=True)}">◐ partly supported</span>'
        level = f"<span class='ds-muted'> · {escape(str(item['confidence']))} confidence</span>" if confidence and item.get("confidence") else ""
        rows.append(f"<div class='ds-claim'>{escape(str(item.get('text', '')))}{level} {mark}"
                    f"<div>{ui.refs(item.get('evidence') or [])}</div></div>")
    return "".join(rows)


def _inv_list_html(reasons) -> str:
    items = "".join(f"<li>{escape(str(r))}</li>" for r in reasons or [])
    return f"<ul class='ds-reasons'>{items}</ul>" if items else ""


def _technique_html(item: dict) -> str:
    technique_id = str(item.get("technique_id", ""))
    verification = item.get("verification") or {}
    status = str(item.get("status") or verification.get("status") or "Uncertain")
    confidence = f"<span class='ds-muted'> · {escape(str(item['confidence']))} confidence</span>" if item.get("confidence") else ""
    return (f"<div class='ds-claim'>{ui.status_badge(status)}"
            f"<a href='{escape(investigation.technique_url(technique_id), quote=True)}' target='_blank' rel='noopener'>"
            f"<b class='ds-mono'>{escape(technique_id)}</b> {escape(str(item.get('name', '')))}</a>{confidence}"
            f"<div>{escape(str(item.get('rationale', '')))}</div>"
            f"<div>{ui.refs(item.get('evidence') or [])}</div>"
            f"{_inv_list_html(verification.get('reasons'))}</div>")


def _removed_html(kind: str, item: dict) -> str:
    if kind == "Technique":
        return _technique_html(item)
    check = item.get("check") or {}
    return (f"<div class='ds-claim ds-removed'><span class='ds-muted'>{escape(kind)}</span> {escape(str(item.get('text', '')))}"
            f"<div>{ui.refs(item.get('evidence') or [])}</div>{_inv_list_html(check.get('reasons'))}</div>")


def _investigation_state(result: investigation.Investigation) -> None:
    if result.state == investigation.NOT_GENERATED:
        title, note = "Not generated", "No investigation outputs for this data source."
    elif result.state == investigation.FAILED:
        title, note = "Generation failed", "; ".join(result.errors)[:200] or "The model reply could not be used."
    else:
        title, note = "Pending", "Not generated for this incident yet."
    st.html(f"<div class='ds-pending'><b>{escape(title)}</b>{escape(note)}</div>")


def _investigation_tab(summary: IncidentSummary, ws: Workspace | None = None) -> None:
    folder = investigation.DEFAULT_RUN if ws is not None and ws.is_package else None
    views = _investigation_views(str(folder), tuple(investigation.run_files(folder).items())) if folder else {}
    mode = st.radio("Investigation view", investigation.MODES, horizontal=True, label_visibility="collapsed",
                    key="investigation_mode")
    result = investigation.lookup(views, mode, summary.incident_id)
    if result.state != investigation.OK or result.verified is None:
        with st.container(border=True, key="card_inv_state"):
            _investigation_state(result)
        return
    verified = result.verified
    provenance = " · ".join(p for p in (result.model_version or result.model, result.created_utc) if p)
    if provenance:
        st.html(f"<div class='ds-provenance'>{escape(provenance)}</div>")
    left, right = st.columns([1, 1.2], gap="large")
    with left, st.container(border=True, key="card_inv_summary"):
        st.html("<div class='ds-section'>Summary</div>")
        st.write(verified.get("summary", ""))
        if verified.get("observations"):
            st.html("<div class='ds-section' style='margin-top:0.6rem'>Observations</div>"
                    + _claims_html(verified["observations"]))
        if verified.get("interpretations"):
            st.html("<div class='ds-section' style='margin-top:0.6rem'>Interpretations</div>"
                    + _claims_html(verified["interpretations"], confidence=True))
        if verified.get("uncertainty"):
            st.html("<div class='ds-section' style='margin-top:0.6rem'>Uncertainty</div>"
                    + _inv_list_html(verified["uncertainty"]))
    with right, st.container(border=True, key="card_inv_attack"):
        st.html(f"<div class='ds-section'>ATT&amp;CK techniques{ui.tip_html('Verification')}</div>")
        techniques = verified.get("techniques") or []
        if techniques:
            st.html("".join(_technique_html(t) for t in techniques))
        elif investigation.outcome_of(verified) == "no_supported_mapping":
            st.html(f"<div class='ds-pending'><b>No supported mapping{ui.tip_html('No supported mapping')}</b>"
                    "No ATT&amp;CK technique is supported by the cited events.</div>")
        removed = investigation.removed_items(verified)
        if removed:
            with st.expander(f"Removed by verification ({len(removed)})"):
                st.html("".join(_removed_html(kind, item) for kind, item in removed))
        if result.retrieved_candidates:
            shown = ", ".join(result.retrieved_candidates[:6]) + ("…" if len(result.retrieved_candidates) > 6 else "")
            st.html(f"<div class='ds-muted' style='margin-top:0.5rem'>Retrieved candidates: {escape(shown)}</div>")


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

    st.html(ui.priority_bar(summary.incident.priority)
            + f"<div class='ds-crumb'>Queue › Day {summary.day} › {ui.mono(summary.incident_id)}</div>")
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
    st.html(f"<div>{_badges(summary)}<span class='ds-factline'>{escape(_fact_line(summary, rows, full=True))}</span></div>")

    pairs = len(host_pair_table(rows)) if not rows.empty else 0
    labels = {"Overview": "Overview", "Timeline": f"Timeline ({len(rows)})", "Connections": f"Connections ({pairs})",
              "Model details": "Model details", "Investigation": "Investigation"}
    # The key changes only when a tab is requested, so clicks inside a tab keep it open.
    wanted = st.session_state.get("incident_tab")
    tabs = st.tabs(list(labels.values()), default=labels.get(wanted) if wanted else None,
                   key=f"incident_tabs_{summary.incident_id}_{st.session_state.get('tab_request', 0)}")
    with tabs[0]:
        _overview_tab(ws, summary, rows)
    with tabs[1]:
        _timeline_tab(ws, summary, rows)
    with tabs[2]:
        _hosts_tab(rows)
    with tabs[3]:
        _model_details_tab(summary)
    with tabs[4]:
        _investigation_tab(summary, ws)


# ─── Evidence page ─────────────────────────────────────────────────────


def evidence_page() -> None:
    ws = _workspace()
    if ws is None:
        return
    st.title("Evidence")
    st.html("<div class='ds-provenance'>Look up an event by ID</div>")
    if ws.events is None:
        st.info("No event data (events.parquet missing).")
        return
    reference = st.text_input("Event ID", placeholder="auth.txt:463603187", key="evidence_lookup").strip()
    if not reference:
        st.caption(f"{len(ws.events):,} events · {len(ws.summaries)} incidents")
        return
    if reference not in ws.events.index:
        st.warning(f"{reference}: not cited by any incident")
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
    st.html("<div class='ds-provenance'>Model, scores and labels</div>")
    table, notes, _ = st.columns([1.5, 1, 0.2], gap="large")
    with table, st.container(border=True, key="card_glossary"):
        st.html(ui.glossary_table())
    with notes:
        st.markdown(
            "**Model**\n"
            "- Data: LANL enterprise authentication logs, scored per user per hour\n"
            "- Final model: gradient boosting over a GRU sequence score and hourly counts\n"
            f"- Queue: top {QUEUE_SIZE} user-hours per day; nearby hours for one user form an incident\n\n"
            "**Priority**\n"
            "- HIGH: score above the day's cut-off\n"
            "- MEDIUM: tied at the cut-off score; picked by a fixed tie-break, so rank order is arbitrary\n\n"
            "**Test result (days 17–30)**\n"
            "- Red-team user-hours caught: 1 of 39\n"
            "- Average precision: 0.00124 (GRU alone: 0.00020)\n"
            "- Labels come from one red team and are incomplete"
        )


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
    # Deep links (?incident=<id>&tab=Connections) select an incident once per link.
    link = (st.query_params.get("incident"), st.query_params.get("tab"))
    if link[0] and st.session_state.get("_deep_link") != link:
        st.session_state["_deep_link"] = link
        _select(link[0])
        tab = TAB_ALIASES.get(link[1], link[1])
        if tab in TAB_LABELS:
            _request_tab(tab)
    st.session_state["_source"] = _source_controls()
    st.sidebar.toggle("Answer key", key="show_answer_key",
                      help="Show red-team labels (evaluation only)")
    page.run()
