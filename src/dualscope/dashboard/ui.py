"""Shared look for the dashboard: palette, page CSS and small HTML pieces.

Colours and helpers shared by every page (.streamlit/config.toml holds the
Streamlit theme). Everything here is presentation; data comes from the
other dashboard modules, and every value is HTML-escaped before rendering.
"""

from __future__ import annotations

import itertools
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
    "Event": "One line from the login log, such as a sign-in, a sign-out or a ticket request.",
    "Incident": ("A stretch of unusual activity by one user. It can be a single flagged hour or a few flagged hours "
                 "close together, and each row in the queue is one incident."),
    "Queue": "The 38 hours the system found most unusual that day, listed for review.",
    "Cut-off": "The lowest score that still made it into the day's top 38.",
    "HIGH": "These scored above the cut-off, so they made the list on their score alone.",
    "MEDIUM": ("They met the cut-off, but there weren't enough open spots for everyone with that score. "
               "A tie-breaker decided which ones made the list, and they're in no particular order."),
    "Rank": ("Where the incident sits in the day's list, with 1 being the most unusual. "
             "It's only meaningful for HIGH incidents."),
    "New": "This happened for the first time since the data began on Day 1. Sign-outs are left out.",
    "Why-flagged rules": ("Four simple checks that point to what stands out: failed logins, an NTLM login to a new computer "
                          "(NTLM is an older login method attackers like to abuse), a network login to a new computer, "
                          "and a login from a computer the user hasn't used before. The model ranks the hour on its own, "
                          "and these checks help explain why."),
    "No rule matched": "The model found the hour unusual overall, and none of the four checks fired.",
    "Sequence percentile": ("How unusual this user's run of logins was next to everyone else's that day. "
                            "At 99.5%, it was more unusual than 99.5% of all other hours."),
    "Fusion score": "The final model's score, used to sort hours from most to least unusual. Treat it as a ranking only.",
    "Graph score": ("How unusual this user's whole day of computer connections looked. "
                    "It's shown for background and has no effect on the ranking."),
    "ATT&CK candidate": ("A known attacker technique from the MITRE ATT&CK list that matches what happened. "
                         "Treat it as a lead worth checking."),
    "Answer key": ("The dataset's real attack labels, kept for testing the system. "
                   "Leave it switched off while reviewing incidents."),
    "New edges": "Computers this user connected to for the first time that day.",
    "Degree growth": ("How many more computers this user connected to than on a normal day. "
                      "A negative number means fewer."),
    "Edge score": "How unusual a single user-to-computer connection looks. Higher means more unusual.",
    "Verification": ("An automatic check that compares the AI's answer with the log lines it points to, using fixed rules "
                     "with no AI involved. Anything the logs don't back up is removed and listed separately."),
    "Supported": ("The log lines show the behaviour this technique describes. "
                  "Whether it was an attack is still for the analyst to decide."),
    "Uncertain": "The log lines only partly match, or login records don't carry enough detail to confirm it.",
    "Rejected": ("Removed because the technique doesn't exist, the log lines it points to don't exist, "
                 "or those lines don't show the behaviour."),
    "Confidence": ("How sure the AI is about its own explanation.\nLow: one of several possible explanations.\n"
                   "Medium: it fits, with other explanations still possible.\nHigh: the logs strongly back it up.\n"
                   "The AI sets this level itself, and the automatic check leaves it alone."),
    "Partly supported": ("Some computer names or times in this sentence appear elsewhere in the incident "
                         "and are missing from the log lines it points to."),
    "No supported mapping": ("None of the known attacker techniques fit this incident. "
                             "This is a normal outcome and happens often."),
    "Tied at cut-off": "Ticked when the hour landed exactly on the cut-off score, which makes it MEDIUM.",
}

CONFIDENCE_MEANING = {"low": "One of several possible explanations.", "medium": "It fits, with other explanations still possible.",
                      "high": "The logs strongly back it up."}

# Hover text for page elements that are not glossary terms (kept out of the About table).
TIPS = {
    "Distinct users": "Some users have more than one incident today.",
    "HIGH vs MEDIUM": "Each day's 38 incidents, with HIGH shown above the line and MEDIUM below. Click a day to open it.",
    "Why flagged chart": "How many incidents each of the four checks flagged. Click a bar to show only those incidents.",
    "Observation timeline": "Each dot is one finding, placed at the time it happened. Click a dot to see its log lines.",
    "Graph context": "The user's most unusual computer connections that day, which may differ from the flagged events.",
    "AI modes": ("AI only: the AI reads the incident's log lines and names techniques from what it already knows.\n"
                 "AI + ATT&CK: it also gets a short list of matching techniques from the MITRE ATT&CK list to choose from.\n"
                 "AI + ATT&CK, checked: the same answer after the automatic check, "
                 "with anything the log lines don't back up removed."),
}

CSS = f"""
<style>
:root {{ --ds-info-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E");
  --ds-accent: {ACCENT}; --ds-line: {LINE}; --ds-muted: {MUTED}; }}
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
.ds-tip {{ display: inline-flex; vertical-align: -0.2em; margin-left: 0.3rem; font-weight: 400; text-transform: none; }}
.ds-tipbtn {{ background: transparent; border: 0; padding: 1px; margin: 0; color: {MUTED}; cursor: pointer; line-height: 0;
  border-radius: 50%; display: inline-flex; }}
.ds-tipbtn::before {{ content: ""; width: 15px; height: 15px; background: currentColor;
  -webkit-mask: var(--ds-info-icon) center / contain no-repeat; mask: var(--ds-info-icon) center / contain no-repeat; }}
.ds-tipbtn:hover, .ds-tipbtn:focus-visible, .ds-tip:has(:popover-open) .ds-tipbtn {{ color: {INK}; }}
.ds-tipbtn:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 1px; }}
.ds-pop {{ position: fixed; inset: auto; margin: 0; max-width: 280px; width: max-content; background: #FFFFFF; color: {INK};
  border: 1px solid {LINE}; border-radius: 6px; box-shadow: 0 4px 14px rgba(31, 35, 40, 0.14); padding: 0.5rem 0.65rem;
  font-size: 13px; line-height: 1.4; font-weight: 400; text-align: left; white-space: normal; letter-spacing: 0; }}
.ds-pop-k {{ font-weight: 650; }}
.ds-pop b {{ display: block; font-weight: 650; margin-bottom: 0.15rem; }}
.ds-metric {{ padding: 0.1rem 0.2rem; }}
.ds-metric span.label {{ display: block; color: {MUTED}; font-size: 0.93rem; }}
.ds-metric b {{ display: block; font-size: 1.8rem; font-weight: 500; line-height: 1.25; font-variant-numeric: tabular-nums; }}
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
.ds-factline {{ color: {INK}; margin-left: 0.2rem; }}
.ds-glossary {{ width: 100%; border-collapse: collapse; }}
.ds-glossary th {{ text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.86rem; padding: 0.4rem 0.8rem 0.4rem 0;
  border-bottom: 1px solid {LINE}; }}
.ds-glossary td {{ padding: 0.45rem 0.8rem 0.45rem 0; border-bottom: 1px solid {LINE}; vertical-align: top; }}
.ds-glossary td:first-child {{ font-weight: 600; white-space: nowrap; width: 11rem; }}
.ds-small {{ font-size: 0.93rem; }}
.ds-verdict {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.1rem 1.1rem; }}
.ds-verdict-label {{ font-weight: 650; }}
.ds-verdict-item {{ white-space: nowrap; }}
.ds-verdict-item .ds-badge {{ margin-bottom: 0; }}
.ds-verdict-none {{ font-weight: 600; }}
.ds-verdict a {{ color: {INK}; text-decoration: none; border-bottom: 1px dotted {MUTED}; }}
.ds-verdict-note {{ margin-left: auto; font-size: 0.93rem; }}
[class*="st-key-card_inv_verdict"] {{ padding: 0.55rem 0.9rem; }}
.ds-inv-summary {{ font-size: 1.07rem; line-height: 1.5; }}
.ds-context {{ margin-top: 0.6rem; font-size: 0.93rem; }}
.ds-context > div {{ padding: 0.1rem 0; }}
.ds-obs {{ line-height: 1.45; padding: 0.15rem 0; }}
.ds-time {{ font-family: "jetbrains-mono", monospace; font-size: 0.9em; color: {MUTED}; margin-right: 0.6rem; }}
.ds-chips {{ margin-left: 0.4rem; white-space: nowrap; }}
.ds-ref.pkg {{ background: none; border-color: transparent; font-family: inherit; font-size: 0.86rem; padding: 0 0.2rem; }}
.ds-ref.more {{ background: none; border-style: dashed; }}
.ds-conf {{ display: inline-block; min-width: 3.6rem; text-align: center; margin-right: 0.5rem; padding: 0 0.45rem;
  border-radius: 999px; background: {PANEL}; border: 1px solid {LINE}; color: {MUTED}; font-size: 0.8rem; }}
[class*="st-key-inv_row_"] {{ gap: 0.2rem; padding: 0.05rem 0; border-left: 3px solid transparent; }}
[class*="st-key-inv_row_sel_"] {{ border-left-color: {ACCENT}; background: #F3F4F7; border-radius: 0 6px 6px 0; }}
[class*="st-key-inv_pick_"] button {{ min-height: 0; min-width: 1.4rem; width: auto; height: 1.4rem; padding: 0 0.35rem; border-radius: 999px;
  margin-top: 0.1rem; background: #FFFFFF; border: 1.5px solid {ACCENT}; display: flex; align-items: center; justify-content: center; }}
[class*="st-key-inv_pick_"] button:hover {{ background: {ACCENT_TINT}; }}
[class*="st-key-inv_row_sel_"] [class*="st-key-inv_pick_"] button {{ background: {ACCENT}; }}
[class*="st-key-inv_row_sel_"] [class*="st-key-inv_pick_"] button p {{ color: #FFFFFF !important; }}
[class*="st-key-inv_pick_"] button p {{ font-size: 0.78rem !important; font-weight: 700; line-height: 1; margin: 0 !important; font-variant-numeric: tabular-nums; color: {ACCENT}; }}
[class*="st-key-inv_pick_"] button div, [class*="st-key-inv_pick_"] button span {{ margin: 0 !important; padding: 0 !important;
  display: flex; align-items: center; justify-content: center; height: 100%; width: 100%; min-height: 0; }}
[class*="st-key-inv_row_"] [data-testid="stColumn"]:first-child {{ min-width: 0; }}
.ds-obs-events {{ padding: 0 0.6rem 0.5rem 3.2rem; overflow-x: auto; }}
.ds-evt {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
.ds-evt th {{ text-align: left; color: {MUTED}; font-weight: 600; padding: 0.15rem 0.7rem 0.15rem 0; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
.ds-evt td {{ padding: 0.15rem 0.7rem 0.15rem 0; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
.ds-evt td:nth-child(2), .ds-evt td:last-child {{ white-space: normal; }}
.ds-evt td:last-child {{ min-width: 7rem; }}
.ds-tech {{ border-bottom: 1px solid {LINE}; padding: 0.4rem 0; }}
.ds-tech:last-child {{ border-bottom: 0; }}
.ds-tech summary {{ cursor: pointer; list-style: none; padding-left: 0.1rem; }}
.ds-tech summary::-webkit-details-marker {{ display: none; }}
.ds-tech summary::before {{ content: "▸"; color: {MUTED}; margin-right: 0.35rem; }}
.ds-tech[open] summary::before {{ content: "▾"; }}
.ds-tech summary a {{ color: {INK}; text-decoration: none; border-bottom: 1px dotted {MUTED}; }}
.ds-reason-line {{ color: {MUTED}; font-size: 0.9rem; margin: 0.1rem 0 0 1.2rem; }}
.ds-tech-body {{ margin: 0.4rem 0 0.2rem 1.2rem; overflow-x: auto; }}
.ds-lbl {{ color: {MUTED}; font-size: 0.86rem; font-weight: 600; margin: 0.45rem 0 0.1rem; }}
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


_pop_ids = itertools.count(1)


_LEAD_WORDS = ("AI + ATT&CK, checked", "AI + ATT&CK", "AI only", "Low", "Medium", "High")


def _card_text(text: str) -> str:
    """Escaped definition with line breaks kept; a leading "Low:" or "AI only:" is bolded."""
    lines = []
    for line in text.split("\n"):
        lead = next((w for w in _LEAD_WORDS if line.startswith(w + ":")), None)
        if lead:
            lines.append(f'<span class="ds-pop-k">{escape(lead)}:</span>{escape(line[len(lead) + 1:])}')
        else:
            lines.append(escape(line))
    return "<br>".join(lines)


def tip_html(term: str) -> str:
    """Info icon that toggles a small definition card (HTML Popover API), for st.html headings.

    The card is positioned and closed on scroll by POPOVER_SCRIPT, which ``app.run`` loads once per page.
    """
    text = GLOSSARY.get(term) or TIPS[term]
    pop_id = f"dspop-{next(_pop_ids)}"
    return (f'<span class="ds-tip"><button type="button" class="ds-tipbtn" popovertarget="{pop_id}" '
            f'aria-label="What is {escape(term, quote=True)}?"></button>'
            f'<div popover id="{pop_id}" class="ds-pop"><b>{escape(term)}</b>{_card_text(text)}</div></span>')


# Installed once per page load (the flag survives Streamlit reruns): place a card beside its icon when it opens,
# and close open cards when anything scrolls. Light-dismiss, a second click and Esc come from the Popover API.
POPOVER_SCRIPT = """<script>
(function () {
  if (window.__dsPopovers) return;
  window.__dsPopovers = true;
  document.addEventListener('toggle', function (e) {
    var pop = e.target;
    if (e.newState !== 'open' || !pop.classList || !pop.classList.contains('ds-pop')) return;
    var btn = document.querySelector('[popovertarget="' + pop.id + '"]');
    if (!btn) return;
    var r = btn.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight, m = 8;
    var left = Math.max(m, Math.min(r.left - 4, window.innerWidth - w - m));
    var top = r.bottom + 6;
    if (top + h > window.innerHeight - m) top = Math.max(m, r.top - h - 6);
    pop.style.left = left + 'px';
    pop.style.top = top + 'px';
  }, true);
  document.addEventListener('scroll', function (e) {
    if (e.target && e.target.closest && e.target.closest('.ds-pop')) return;
    document.querySelectorAll('.ds-pop:popover-open').forEach(function (p) { p.hidePopover(); });
  }, true);
})();
</script>"""


def metric_html(label: str, value: str, term: str) -> str:
    """A metric tile (label with info icon over a value) for places where st.metric's native help icon would be used."""
    return f"<div class='ds-metric'><span class='label'>{escape(label)}{tip_html(term)}</span><b>{escape(value)}</b></div>"


def glossary_table() -> str:
    rows = "".join(f"<tr><td>{escape(term)}</td><td>{escape(meaning).replace(chr(10), "<br>")}</td></tr>" for term, meaning in GLOSSARY.items())
    return f'<table class="ds-glossary"><thead><tr><th>Term</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>'
