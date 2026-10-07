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
TIED = "#C5CAD1"
INK = "#1F2328"
MUTED = "#5F6670"
LINE = "#E1E4E8"
PANEL = "#F6F7F8"
GROUND = "#F3F4F6"  # neutral page ground behind white cards
HIGH = "#C4391F"
MEDIUM = "#D99A1E"
MEDIUM_INK = "#3B2900"

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
.ds-cutoff {{ display: flex; align-items: center; gap: 0.75rem; color: {MUTED}; font-size: 0.9rem; margin: 0.2rem 0 0; }}
.ds-cutoff::before, .ds-cutoff::after {{ content: ""; flex: 1; border-top: 2px dashed #A3A9B1; }}
.ds-cutoff::before {{ flex: 0 0 1.5rem; }}
.ds-section {{ font-weight: 650; font-size: 1rem; margin: 0 0 0.35rem; }}
.ds-reason {{ padding: 0.4rem 0 0.45rem; border-bottom: 1px solid {LINE}; }}
.ds-reason:last-child {{ border-bottom: 0; }}
.ds-reason b {{ font-weight: 600; }}
.ds-facts {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid {LINE}; }}
.ds-facts div {{ padding: 0.35rem 0.6rem 0.4rem 0; border-bottom: 1px solid {LINE}; }}
.ds-facts span {{ display: block; color: {MUTED}; font-size: 0.86rem; }}
.ds-stats {{ display: grid; gap: 0.55rem; padding: 0.2rem 0 0; }}
.ds-stats b {{ display: block; font-size: clamp(1.3rem, 1.5vw, 1.7rem); font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }}
.ds-stats span {{ display: block; color: {MUTED}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ds-crumb {{ color: {MUTED}; margin-bottom: 0.2rem; }}
.ds-callout {{ background: {ACCENT_TINT}; border-radius: 6px; padding: 0.7rem 0.9rem; }}
.ds-pending {{ text-align: center; color: {MUTED}; padding: 1.1rem 0.6rem; }}
.ds-pending b {{ display: block; color: {INK}; margin-bottom: 0.3rem; }}
.ds-candidate {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: baseline;
  padding: 0.45rem 0; border-bottom: 1px solid {LINE}; }}
.ds-candidate:last-child {{ border-bottom: 0; }}
/* Streamlit shrinks chart SVG text to 0.6em; the hosts diagram keeps its Graphviz sizes. */
.st-key-card_host_graph svg text {{ font-size: 12.5px !important; }}
.st-key-card_host_graph svg text[fill="#c4391f" i] {{ font-size: 11px !important; font-weight: 700; }}
.ds-legend {{ color: {MUTED}; font-size: 0.86rem; }}
.ds-swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; vertical-align: -0.1rem;
  margin: 0 0.3rem 0 0.8rem; }}
</style>
"""


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="ds-badge {escape(kind)}">{escape(text)}</span>'


def priority_badge(priority: str) -> str:
    kind = {"HIGH": "high", "MEDIUM": "medium", "CRITICAL": "high"}.get(priority, "neutral")
    return badge(priority, kind)


def chips(labels: Iterable[str]) -> str:
    return "".join(f'<span class="ds-chip">{escape(label)}</span>' for label in labels)


def mono(text: str) -> str:
    return f'<span class="ds-mono">{escape(text)}</span>'


def cutoff_rule(label: str = "Day cut-off — rows below were tied and picked by a fixed tie-break") -> str:
    return f'<div class="ds-cutoff" role="separator">{escape(label)}</div>'


def facts(items: Iterable[tuple[str, str]]) -> str:
    cells = "".join(f"<div><span>{escape(k)}</span>{escape(v)}</div>" for k, v in items)
    return f'<div class="ds-facts">{cells}</div>'
