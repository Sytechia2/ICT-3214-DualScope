"""Shared look for the dashboard: palette, page CSS and small HTML pieces.

Colours and helpers shared by every page (.streamlit/config.toml holds the
Streamlit theme). Everything here is presentation; data comes from the
other dashboard modules, and every value is HTML-escaped before rendering.
"""

from __future__ import annotations

from html import escape
from typing import Iterable

SHELL = "#1F2328"  # graphite side rail
ACCENT = "#3B4250"  # slate: selection, primary actions, above-cut-off bars
ACCENT_TINT = "#E4E7EC"
INK = "#1F2328"
MUTED = "#5F6670"
LINE = "#E1E4E8"
PANEL = "#F6F7F8"
GROUND = "#F3F4F6"  # neutral page ground behind white cards
CHART = "#2563EB"  # charts carry the only accent colour so they stand out on the neutral page
CHART_LIGHT = "#9DBBF5"  # tied hours
CHART_BAND = "#E6EEFD"  # selected day
HIGH = "#C4391F"
HIGH_TINT = "#FCE9E5"  # rows and cells that need attention
MEDIUM = "#D99A1E"
MEDIUM_INK = "#3B2900"

# One definition per term: hover tooltips (tip, tip_html) and the About page all read from here.
GLOSSARY = {
    "Event": "One authentication log line from LANL auth.txt.",
    "Incident": "One user's flagged hour, or several flagged hours close together. Each row in the queue is one incident.",
    "Queue": "The 38 user-hours the model scored highest that day.",
    "Cut-off": "The score of the 38th place in the day's queue.",
    "HIGH": "Scored above the cut-off. These would be in the queue whatever the tie-break.",
    "MEDIUM": ("Scored exactly the cut-off. More hours tied than places were left, so a fixed tie-break picked these. "
               "Their order means nothing."),
    "Rank": "Position in the day's queue, 1 to 38. Only meaningful for HIGH.",
    "New": "First time since Day 1 of the dataset. Log-offs never count as new.",
    "Why-flagged rules": ("Four checks run on each incident's events: failed logons, NTLM logon to a new host, "
                          "network logon to a new host, logon from a new source computer. "
                          "They explain the alert; the model itself scores the whole hour."),
    "No rule matched": "The model ranked the hour highly, but no single event matched one of the four rules.",
    "Sequence percentile": ("How unusual the user's logon sequence was compared with every other user-hour that day "
                            "(GRU detector). 99.5% = more unusual than 99.5% of them."),
    "Fusion score": "The final model's output. Used only to rank hours; it is not an attack probability.",
    "Graph score": "The graph detector's score for this user's day. Context only; the final model does not use it.",
    "ATT&CK candidate": ("A MITRE ATT&CK technique whose description matches the behaviour. "
                         "A lead to check, not a confirmed finding."),
    "Answer key": "The dataset's red-team labels. For evaluation only; keep it off when triaging.",
    "New edges": "Computer connections this user made that day for the first time, as counted by the graph detector.",
    "Degree growth": ("Change in how many computers this user connected to, compared with their usual days. "
                      "Negative = fewer than usual."),
    "Edge score": "The graph detector's unusualness score for one user → computer connection. Higher = more unusual.",
    "Verification": ("Fixed rules check each AI claim against the cited events. No model is involved. "
                     "Claims that fail are removed and listed separately."),
    "Supported": "The cited events contain the pattern the technique describes. Consistent with it, not proof of intent.",
    "Uncertain": ("The cited events match only partly, the specific variant cannot be seen in logon records, "
                  "or there is no rule for it."),
    "Rejected": ("Not an active technique, no valid cited event, no authentication-log data source, "
                 "or the cited events do not match the pattern."),
    "Partly supported": "Some names or times in the claim are not in the cited events, only elsewhere in the evidence package.",
    "No supported mapping": "No ATT&CK technique passed verification. A valid result, not an error.",
    "Tied at cut-off": "Whether this alert hour scored exactly the cut-off (MEDIUM).",
}

# Hover text for page elements that are not glossary terms (kept out of the About table).
TIPS = {
    "Distinct users": "Some users have more than one incident today.",
    "HIGH vs MEDIUM": ("Each day's 38 incidents: HIGH (above cut-off) up, MEDIUM (tie-break picks) down. "
                       "Click a day to open it."),
    "Why flagged chart": GLOSSARY["Why-flagged rules"] + " Click a bar to filter the queue.",
    "Graph context": ("The graph detector's most unusual connections for this user that day; "
                      "they may differ from the flagged events."),
}

CSS = f"""
<style>
:root {{ --ds-accent: {ACCENT}; --ds-line: {LINE}; --ds-muted: {MUTED}; }}
::selection {{ background: #DCE0E5; color: {INK}; }}
html {{ caret-color: {ACCENT}; scrollbar-color: #B8BEC6 transparent; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 2rem; padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }}
h1 {{ font-size: 1.75rem !important; padding: 0.2rem 0 0.3rem !important; }}
h3 {{ font-size: 1.1rem !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {GROUND}; }}
[class*="st-key-card"] {{ background: #FFFFFF; }}
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="select"] > div {{ background: #FFFFFF !important; }}
[data-testid="stSidebar"][aria-expanded="true"] {{ width: 200px !important; min-width: 200px !important; }}
[data-testid="stSidebarCollapseButton"] {{ display: flex !important; visibility: visible !important; opacity: 1 !important; }}
[data-testid="stTab"] p {{ font-size: 1rem !important; font-weight: 550; }}
[data-testid="stTab"] {{ padding-bottom: 0.25rem; margin-right: 0.5rem; }}
.ds-reason small {{ display: block; color: {MUTED}; font-size: 0.9rem; margin: 0.1rem 0 0.15rem; }}
h1 {{ letter-spacing: -0.015em; }}
[data-testid="stSidebarNav"] a span {{ font-size: 0.95rem; }}
[data-testid="stSidebarNav"]::before {{ content: "DualScope"; display: block; color: #FFFFFF; font-weight: 700;
  font-size: 1.25rem; letter-spacing: -0.01em; padding: 0.1rem 1rem 0.8rem; }}
.ds-provenance {{ color: {MUTED}; margin: -0.3rem 0 0.5rem; font-size: 0.93rem; }}
.ds-badge {{ display: inline-block; padding: 0.12rem 0.62rem; border-radius: 999px; font-weight: 650;
  font-size: 0.86rem; line-height: 1.5; margin: 0 0.35rem 0.35rem 0; white-space: nowrap; }}
.ds-badge.high {{ background: {HIGH}; color: #FFFFFF; }}
.ds-badge.medium {{ background: {MEDIUM}; color: {MEDIUM_INK}; }}
.ds-badge.neutral {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE}; font-weight: 500; }}
.ds-badge.answer {{ background: #F7E3DF; color: #7A1F0E; border: 1px solid #E9B8AD; }}
.ds-chip {{ display: inline-block; padding: 0.08rem 0.6rem; border-radius: 999px; border: 1px solid #CDD1D6;
  background: #FFFFFF; font-size: 0.86rem; margin: 0 0.3rem 0.3rem 0; white-space: nowrap; }}
.ds-mono {{ font-family: "jetbrains-mono", monospace; font-size: 0.92em; }}
.ds-muted {{ color: {MUTED}; }}
.ds-group {{ display: flex; align-items: center; gap: 0.6rem; padding: 0 0 0 0.6rem; border-left: 3px solid {LINE};
  min-width: 0; }}
.ds-group.high {{ border-left-color: {HIGH}; }}
.ds-group.medium {{ border-left-color: {MEDIUM}; }}
.ds-group.high .ds-count {{ background: {HIGH}; border-color: {HIGH}; color: #FFFFFF; }}
.ds-group.medium .ds-count {{ background: {MEDIUM}; border-color: {MEDIUM}; color: {MEDIUM_INK}; }}
.ds-group-title {{ font-weight: 650; font-size: 1rem; white-space: nowrap; }}
.ds-count {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 999px; padding: 0 0.5rem; font-weight: 650;
  font-size: 0.86rem; font-variant-numeric: tabular-nums; }}
.ds-group-note {{ color: {MUTED}; font-size: 0.88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }}
[class*="st-key-card_queue_"] {{ padding: 0.7rem 0.8rem 0.8rem; gap: 0.55rem; }}
.ds-section {{ font-weight: 650; font-size: 1rem; margin: 0 0 0.35rem; }}
.ds-reason {{ padding: 0.4rem 0 0.45rem; border-bottom: 1px solid {LINE}; }}
.ds-reason:last-child {{ border-bottom: 0; }}
.ds-reason b {{ font-weight: 600; }}
.ds-facts {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid {LINE}; }}
.ds-facts div {{ padding: 0.35rem 0.6rem 0.4rem 0; border-bottom: 1px solid {LINE}; }}
.ds-facts span {{ display: block; color: {MUTED}; font-size: 0.86rem; }}
.ds-facts b {{ font-weight: 600; }}
.ds-facts b.alert {{ color: {HIGH}; font-weight: 700; }}
.ds-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; }}
.ds-kpis div {{ border-left: 3px solid {LINE}; padding: 0.1rem 0 0.1rem 0.7rem; }}
.ds-kpis div:first-child {{ border-left-color: {CHART}; }}
.ds-kpis b {{ display: block; font-size: 1.7rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums; }}
.ds-kpis span {{ color: {MUTED}; font-size: 0.86rem; }}
.ds-prio-bar {{ height: 4px; border-radius: 2px; background: {LINE}; margin: -0.4rem 0 0.2rem; }}
.ds-prio-bar.high {{ background: {HIGH}; }}
.ds-prio-bar.medium {{ background: {MEDIUM}; }}
[class*="st-key-card_panel_high"] {{ border-top: 4px solid {HIGH} !important; }}
[class*="st-key-card_panel_medium"] {{ border-top: 4px solid {MEDIUM} !important; }}
[class*="st-key-card_panel"] h3 {{ font-size: 1.2rem !important; }}
[class*="st-key-card_panel"] .ds-reason {{ border-left: 3px solid {HIGH}; padding-left: 0.6rem; margin-bottom: 0.35rem;
  border-bottom: 0; background: #FFFFFF; }}
[class*="st-key-card_reason_"] {{ border-left: 4px solid {HIGH} !important; }}
[class*="st-key-card_reason_"] .ds-reason b {{ font-size: 1.05rem; }}
.ds-stats {{ display: grid; gap: 0.55rem; padding: 0.2rem 0 0; }}
.ds-stats b {{ display: block; font-size: clamp(1.3rem, 1.5vw, 1.7rem); font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }}
.ds-stats b.medium {{ color: #A86E00; }}
.ds-stats span {{ display: block; color: {MUTED}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ds-crumb {{ color: {MUTED}; margin-bottom: 0.2rem; }}
.ds-callout {{ background: {ACCENT_TINT}; border-radius: 6px; padding: 0.7rem 0.9rem; }}
.ds-pending {{ text-align: center; color: {MUTED}; padding: 1.1rem 0.6rem; }}
.ds-pending b {{ display: block; color: {INK}; margin-bottom: 0.3rem; }}
.ds-badge.supported {{ background: {ACCENT}; color: #FFFFFF; }}
.ds-badge.uncertain {{ background: #FFFFFF; color: {INK}; border: 1px solid {MUTED}; font-weight: 600; }}
.ds-badge.rejected {{ background: {PANEL}; color: {MUTED}; border: 1px solid {LINE}; text-decoration: line-through; font-weight: 500; }}
.ds-claim {{ padding: 0.4rem 0 0.45rem; border-bottom: 1px solid {LINE}; }}
.ds-claim:last-child {{ border-bottom: 0; }}
.ds-claim a {{ color: {INK}; text-decoration: none; border-bottom: 1px dotted {MUTED}; }}
.ds-ref {{ display: inline-block; font-family: "jetbrains-mono", monospace; font-size: 0.8rem; color: {MUTED};
  background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; padding: 0 0.35rem; margin: 0.15rem 0.25rem 0 0; }}
.ds-partly {{ color: {MUTED}; font-size: 0.86rem; cursor: help; white-space: nowrap; }}
.ds-reasons {{ margin: 0.2rem 0 0; padding-left: 1.1rem; color: {MUTED}; font-size: 0.9rem; }}
.ds-removed {{ color: {MUTED}; }}
.ds-candidate {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: baseline;
  padding: 0.45rem 0; border-bottom: 1px solid {LINE}; }}
.ds-candidate:last-child {{ border-bottom: 0; }}
/* Streamlit shrinks chart SVG text to 0.6em; the hosts diagram keeps its Graphviz sizes. */
.st-key-card_host_graph svg text {{ font-size: 12.5px !important; }}
.st-key-card_host_graph svg text[fill="#c4391f" i] {{ font-size: 11px !important; font-weight: 700; }}
.ds-legend {{ color: {MUTED}; font-size: 0.86rem; }}
.ds-swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; vertical-align: -0.1rem;
  margin: 0 0.3rem 0 0.8rem; }}
.ds-tip {{ color: {MUTED}; cursor: help; font-weight: 400; margin-left: 0.25rem; }}
.ds-glance {{ display: grid; gap: 0.35rem; }}
.ds-glance .big b {{ font-size: 2.1rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }}
.ds-glance .big span, .ds-glance .row span {{ color: {MUTED}; }}
.ds-glance .big span {{ margin-left: 0.4rem; }}
.ds-glance .row {{ display: flex; justify-content: space-between; align-items: baseline; border-top: 1px solid {LINE};
  padding-top: 0.3rem; }}
.ds-glance .row b {{ font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.ds-glance .row.high b, .ds-glance .row.high span {{ color: {HIGH}; font-weight: 650; }}
.ds-glance .row.medium b, .ds-glance .row.medium span {{ color: #A86E00; font-weight: 650; }}
.ds-fact {{ padding: 0.3rem 0; }}
.ds-glance .ds-tip {{ color: {MUTED}; font-weight: 400; }}
.ds-factline {{ color: {INK}; margin-left: 0.2rem; }}
.ds-glossary {{ width: 100%; border-collapse: collapse; }}
.ds-glossary th {{ text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.86rem; padding: 0.4rem 0.8rem 0.4rem 0;
  border-bottom: 1px solid {LINE}; }}
.ds-glossary td {{ padding: 0.45rem 0.8rem 0.45rem 0; border-bottom: 1px solid {LINE}; vertical-align: top; }}
.ds-glossary td:first-child {{ font-weight: 600; white-space: nowrap; width: 11rem; }}
</style>
"""


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="ds-badge {escape(kind)}">{escape(text)}</span>'


def priority_badge(priority: str) -> str:
    kind = {"HIGH": "high", "MEDIUM": "medium", "CRITICAL": "high"}.get(priority, "neutral")
    return badge(priority, kind)


def chips(labels: Iterable[str]) -> str:
    return "".join(f'<span class="ds-chip">{escape(label)}</span>' for label in labels)


def status_badge(status: str) -> str:
    """Verification status (Supported / Uncertain / Rejected) with its glossary tooltip."""
    kind = status.lower() if status.lower() in ("supported", "uncertain", "rejected") else "neutral"
    hint = escape(GLOSSARY.get(status, ""), quote=True)
    return f'<span class="ds-badge {kind}" title="{hint}">{escape(status)}</span>'


def refs(references: Iterable[str]) -> str:
    """Evidence references as small mono chips."""
    return "".join(f'<span class="ds-ref">{escape(str(ref))}</span>' for ref in references)


def mono(text: str) -> str:
    return f'<span class="ds-mono">{escape(text)}</span>'


def facts(items: Iterable[tuple[str, str]], alert: Iterable[str] = ()) -> str:
    """Label/value grid; values whose label is in ``alert`` are shown in the alert colour."""
    flagged = set(alert)
    cells = "".join(f"<div><span>{escape(k)}</span><b class='{'alert' if k in flagged else ''}'>{escape(v)}</b></div>"
                    for k, v in items)
    return f'<div class="ds-facts">{cells}</div>'


def kpis(items: Iterable[tuple[str, str]]) -> str:
    """Large-number tiles for the scores that matter most."""
    cells = "".join(f"<div><b>{escape(v)}</b><span>{escape(k)}</span></div>" for k, v in items)
    return f'<div class="ds-kpis">{cells}</div>'


def priority_bar(priority: str) -> str:
    kind = {"HIGH": "high", "CRITICAL": "high", "MEDIUM": "medium"}.get(priority, "neutral")
    return f'<div class="ds-prio-bar {kind}"></div>'


def tip(term: str) -> str:
    """Glossary definition for a ``help=`` tooltip."""
    return GLOSSARY[term]


def tip_html(term: str) -> str:
    """Small ⓘ marker whose hover text is the glossary definition, for st.html headings."""
    return f'<span class="ds-tip" title="{escape(GLOSSARY.get(term) or TIPS[term], quote=True)}">ⓘ</span>'


def glossary_table() -> str:
    rows = "".join(f"<tr><td>{escape(term)}</td><td>{escape(meaning)}</td></tr>" for term, meaning in GLOSSARY.items())
    return f'<table class="ds-glossary"><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>'
