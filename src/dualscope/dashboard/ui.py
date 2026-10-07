"""Shared look for the dashboard: palette, page CSS and small HTML pieces.

Colours and helpers shared by every page (.streamlit/config.toml holds the
Streamlit theme). Everything here is presentation; data comes from the
other dashboard modules, and every value is HTML-escaped before rendering.
"""

from __future__ import annotations

from html import escape
from typing import Iterable

SHELL = "#0E3B43"
TEAL = "#13707A"
TEAL_TINT = "#E3F1F2"
TIED = "#A9CFD2"
INK = "#17262B"
MUTED = "#55666B"
LINE = "#DDE3E5"
PANEL = "#F4F6F7"
GROUND = "#F1F6F6"  # cool page ground behind white cards
HIGH = "#C4391F"
MEDIUM = "#D99A1E"
MEDIUM_INK = "#3B2900"

CSS = f"""
<style>
:root {{ --ds-teal: {TEAL}; --ds-line: {LINE}; --ds-muted: {MUTED}; }}
::selection {{ background: #CDE7E8; color: {INK}; }}
html {{ caret-color: {TEAL}; scrollbar-color: #B9C6C9 transparent; }}
.block-container {{ padding-top: 3.4rem; padding-left: 2.2rem; padding-right: 2.2rem; max-width: 100%; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: {GROUND}; }}
[class*="st-key-card"] {{ background: #FFFFFF; }}
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="select"] > div {{ background: #FFFFFF !important; }}
[data-testid="stSidebar"] {{ width: 232px !important; min-width: 232px !important; }}
[data-testid="stTab"] p {{ font-size: 1.2rem !important; font-weight: 550; }}
[data-testid="stTab"] {{ padding-bottom: 0.4rem; margin-right: 0.8rem; }}
.ds-reason small {{ display: block; color: {MUTED}; font-size: 0.9rem; margin: 0.1rem 0 0.15rem; }}
h1 {{ letter-spacing: -0.02em; }}
[data-testid="stSidebarNav"] a span {{ font-size: 1rem; }}
[data-testid="stSidebarNav"]::before {{ content: "DualScope"; display: block; color: #FFFFFF; font-weight: 700;
  font-size: 1.45rem; letter-spacing: -0.01em; padding: 0.2rem 1rem 1.1rem; }}
.ds-provenance {{ color: {MUTED}; margin: -0.6rem 0 1.1rem; }}
.ds-badge {{ display: inline-block; padding: 0.12rem 0.62rem; border-radius: 999px; font-weight: 650;
  font-size: 0.86rem; line-height: 1.5; margin: 0 0.35rem 0.35rem 0; white-space: nowrap; }}
.ds-badge.high {{ background: {HIGH}; color: #FFFFFF; }}
.ds-badge.medium {{ background: {MEDIUM}; color: {MEDIUM_INK}; }}
.ds-badge.neutral {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE}; font-weight: 500; }}
.ds-badge.answer {{ background: #F7E3DF; color: #7A1F0E; border: 1px solid #E9B8AD; }}
.ds-chip {{ display: inline-block; padding: 0.08rem 0.6rem; border-radius: 999px; border: 1px solid #C5D1D4;
  background: #FFFFFF; font-size: 0.86rem; margin: 0 0.3rem 0.3rem 0; white-space: nowrap; }}
.ds-mono {{ font-family: "jetbrains-mono", monospace; font-size: 0.92em; }}
.ds-muted {{ color: {MUTED}; }}
.ds-cutoff {{ display: flex; align-items: center; gap: 0.75rem; color: {MUTED}; margin: 0.9rem 0 0.6rem; }}
.ds-cutoff::before, .ds-cutoff::after {{ content: ""; flex: 1; border-top: 2px dashed #9FB3B7; }}
.ds-cutoff::before {{ flex: 0 0 1.5rem; }}
.ds-section {{ font-weight: 650; font-size: 1.12rem; margin: 0 0 0.55rem; }}
.ds-reason {{ padding: 0.55rem 0 0.65rem; border-bottom: 1px solid {LINE}; }}
.ds-reason:last-child {{ border-bottom: 0; }}
.ds-reason b {{ font-weight: 600; }}
.ds-facts {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid {LINE}; }}
.ds-facts div {{ padding: 0.5rem 0.6rem 0.55rem 0; border-bottom: 1px solid {LINE}; }}
.ds-facts span {{ display: block; color: {MUTED}; font-size: 0.86rem; }}
.ds-stats {{ display: grid; gap: 0.75rem; padding: 0.3rem 0 0.2rem; }}
.ds-stats b {{ display: block; font-size: clamp(1.6rem, 1.9vw, 2.2rem); font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }}
.ds-stats span {{ display: block; color: {MUTED}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.ds-crumb {{ color: {MUTED}; margin-bottom: 0.2rem; }}
.ds-callout {{ background: {TEAL_TINT}; border-radius: 6px; padding: 0.7rem 0.9rem; }}
.ds-pending {{ text-align: center; color: {MUTED}; padding: 1.1rem 0.6rem; }}
.ds-pending b {{ display: block; color: {INK}; margin-bottom: 0.3rem; }}
.ds-candidate {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: baseline;
  padding: 0.45rem 0; border-bottom: 1px solid {LINE}; }}
.ds-candidate:last-child {{ border-bottom: 0; }}
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
