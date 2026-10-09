"""Load LLM investigation outputs (Tasks 6.2-6.4) for the incident page.

Reads ``outputs/investigations/<run>/`` files only; nothing here calls a model.
Every function returns a value for bad or missing files instead of raising, so
the Investigation tab never blocks the detector evidence tabs.
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPOSITORY_ROOT / "outputs" / "investigations" / "gemini_v1"

VERIFIED_MODE = "AI + ATT&CK, checked"
RAG_MODE, DIRECT_MODE = "AI + ATT&CK", "AI only"
MODES = (VERIFIED_MODE, RAG_MODE, DIRECT_MODE)
_MODE_KEY = {RAG_MODE: "rag", DIRECT_MODE: "direct"}

OK, PENDING, FAILED, NOT_GENERATED = "ok", "pending", "failed", "not_generated"


@dataclass(frozen=True)
class Investigation:
    state: str  # ok | pending | failed | not_generated
    verified: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()
    model: str = ""
    model_version: str = ""
    created_utc: str = ""
    retrieved_candidates: tuple[str, ...] = ()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Dict records from a JSONL file; unreadable files and bad lines are skipped."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("incident_id"):
                    records.append(record)
    except OSError:
        return []
    return records


def run_files(folder: Path) -> dict[str, float]:
    """Modification time of each file the views read (0.0 if absent); a cache key."""
    names = ("rag_verified.jsonl", "verification.jsonl", "generated_rag.jsonl", "generated_direct.jsonl")
    times = {}
    for name in names:
        try:
            times[name] = (folder / name).stat().st_mtime
        except OSError:
            times[name] = 0.0
    return times


def _errors(record: dict[str, Any]) -> tuple[str, ...]:
    errors = record.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    return tuple(str(e) for e in errors if e)


def _candidates(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(c) for c in (record.get("retrieved_candidates") or []))


def _from_record(record: dict[str, Any], verified: Any = "same") -> Investigation:
    verified = record.get("verified") if verified == "same" else verified
    status = record.get("status")
    if isinstance(verified, dict) and status in (None, "ok"):
        state = OK
    elif status in ("failed", "invalid"):
        state = FAILED
    elif status == "ok" or (not status and not _errors(record)):
        state = PENDING  # generated but not verified yet
    else:
        state = FAILED
    return Investigation(
        state, verified if state == OK else None, _errors(record),
        str(record.get("model") or ""), str(record.get("model_version") or ""),
        str(record.get("created_utc") or ""), _candidates(record),
    )


def load_run(folder: Path, modified: tuple[tuple[str, float], ...] = ()) -> dict[str, dict[str, Investigation]]:
    """``{mode label: {incident_id: Investigation}}`` for a run folder; empty if the folder is missing.

    ``modified`` is unused; callers pass it so a cache keyed on file times refreshes.
    """
    if not folder.is_dir() or not any(t for t in run_files(folder).values()):
        return {}  # no generated or verified files, for example after a dry run
    views: dict[str, dict[str, Investigation]] = {
        VERIFIED_MODE: {r["incident_id"]: _from_record(r) for r in read_jsonl(folder / "rag_verified.jsonl")}
    }
    verification = read_jsonl(folder / "verification.jsonl")
    for label, key in _MODE_KEY.items():
        generated = {r["incident_id"]: r for r in read_jsonl(folder / f"generated_{key}.jsonl")}
        view: dict[str, Investigation] = {}
        for incident_id, record in generated.items():
            view[incident_id] = _from_record(record, verified=None)
        for record in verification:
            if record.get("mode") != key:
                continue
            incident_id = record["incident_id"]
            base = generated.get(incident_id, {})
            verified = record.get("verified")
            if isinstance(verified, dict):
                view[incident_id] = _from_record({**base, "status": "ok"}, verified=verified)
        views[label] = view
    return views


def lookup(views: dict[str, dict[str, Investigation]], mode: str, incident_id: str) -> Investigation:
    """The investigation for one incident and mode, or a not-generated / pending placeholder."""
    if not views:
        return Investigation(NOT_GENERATED)
    return views.get(mode, {}).get(incident_id) or Investigation(PENDING)


def technique_url(technique_id: str) -> str:
    return "https://attack.mitre.org/techniques/" + technique_id.replace(".", "/") + "/"


def outcome_of(verified: dict[str, Any]) -> str:
    return str(verified.get("outcome") or "")


def removed_items(verified: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Everything verification dropped, as (kind, item) pairs."""
    kinds = (("Technique", "removed_techniques"), ("Observation", "removed_observations"),
             ("Interpretation", "removed_interpretations"))
    return [(kind, item) for kind, key in kinds for item in (verified.get(key) or []) if isinstance(item, dict)]


# ─── Evidence times, ordering and timeline dots (pure logic; no Streamlit) ───

PACKAGE_LABELS = {"counts": "counts", "detector": "detector", "graph_context": "graph", "incident": "incident",
                  "behaviours_found_by_rules": "rules"}
MERGE_SECONDS = 20  # dots closer than this share one marker
MAX_CHIPS = 3


def is_event_ref(ref: Any) -> bool:
    return str(ref).startswith("auth.txt:")


def event_time(events, ref: str) -> int | None:
    """Dataset second of a cited event, or None when it is not a log line or not in the table."""
    if events is None or not is_event_ref(ref) or ref not in events.index:
        return None
    value = events.loc[ref, "timestamp"]
    value = value.iloc[0] if hasattr(value, "iloc") else value
    return int(value)


def clock(seconds: int) -> str:
    """HH:MM:SS of a dataset second (no date)."""
    seconds = (int(seconds) - 1) % 86_400
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def claim_time(events, refs) -> int | None:
    """Earliest time among a claim's cited log events."""
    times = [t for t in (event_time(events, r) for r in refs or []) if t is not None]
    return min(times) if times else None


@dataclass(frozen=True)
class Observation:
    number: int  # 1-based position in the chronological list
    item: dict[str, Any]
    time: int | None  # None: cites log lines that are not in the event table

    @property
    def refs(self) -> list[str]:
        return [str(r) for r in self.item.get("evidence") or []]

    @property
    def text(self) -> str:
        return str(self.item.get("text", ""))


def order_observations(observations, events) -> tuple[list[Observation], list[dict[str, Any]]]:
    """(numbered observations in time order, context observations that cite only package data).

    An observation is context when it cites no log line. Numbered ones with an unknown time go last.
    """
    timed: list[tuple[int | None, int, dict[str, Any]]] = []
    context: list[dict[str, Any]] = []
    for index, item in enumerate(o for o in observations or [] if isinstance(o, dict)):
        refs = item.get("evidence") or []
        if any(is_event_ref(r) for r in refs):
            timed.append((claim_time(events, refs), index, item))
        else:
            context.append(item)
    timed.sort(key=lambda row: (row[0] is None, row[0] or 0, row[1]))
    return [Observation(n, item, t) for n, (t, _, item) in enumerate(timed, 1)], context


@dataclass(frozen=True)
class Dot:
    numbers: tuple[int, ...]
    time: int
    text: str
    refs: tuple[str, ...]
    techniques: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """"4", "4–5" for consecutive numbers, else "4,6"."""
        first, last = self.numbers[0], self.numbers[-1]
        if len(self.numbers) >= 2 and last - first == len(self.numbers) - 1:
            return f"{first}–{last}"
        return ",".join(str(n) for n in self.numbers)


def truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def merge_dots(observations: list[Observation], seconds: int = MERGE_SECONDS) -> list[Dot]:
    """One dot per observation with a time; dots within ``seconds`` of the group's first merge."""
    groups: list[list[Observation]] = []
    for obs in (o for o in observations if o.time is not None):
        if groups and obs.time - groups[-1][0].time <= seconds:
            groups[-1].append(obs)
        else:
            groups.append([obs])
    dots = []
    for group in groups:
        refs = tuple(dict.fromkeys(r for o in group for r in o.refs))
        text = group[0].text if len(group) == 1 else " / ".join(f"{o.number}. {o.text}" for o in group)
        dots.append(Dot(tuple(o.number for o in group), group[0].time, truncate(text), refs))
    return dots


def marker_width(label: str) -> float:
    """Pixel width of a timeline marker: a 20px circle for one character, a pill of 8px + 6.5px per character beyond."""
    return 20.0 if len(label) <= 1 else 8 + 6.5 * len(label)


def merge_dots_px(observations: list[Observation], low: int, high: int, width_px: float = 600, gap_px: float = 4) -> list[Dot]:
    """Like ``merge_dots``, but a marker joins the previous one when their pixel boxes would touch (gap < ``gap_px``).

    ``low``/``high`` is the fitted time range drawn across ``width_px`` pixels. A merged marker sits at its first time
    and is wider, so each later marker is compared with the group's current box.
    """
    scale = width_px / max(1, high - low)
    groups: list[list[Observation]] = []
    for obs in (o for o in observations if o.time is not None):
        if groups:
            first = groups[-1][0]
            box = marker_width(Dot(tuple(o.number for o in groups[-1]), 0, "", ()).label)
            own = marker_width(str(obs.number))
            if (obs.time - first.time) * scale < (box + own) / 2 + gap_px:
                groups[-1].append(obs)
                continue
        groups.append([obs])
    dots = []
    for group in groups:
        refs = tuple(dict.fromkeys(r for o in group for r in o.refs))
        text = group[0].text if len(group) == 1 else " / ".join(f"{o.number}. {o.text}" for o in group)
        dots.append(Dot(tuple(o.number for o in group), group[0].time, truncate(text), refs))
    return dots


def technique_refs(technique: dict[str, Any]) -> set[str]:
    verification = technique.get("verification") or {}
    return {str(r) for r in (verification.get("cited_events") or technique.get("evidence") or []) if is_event_ref(r)}


def link_techniques(dots: list[Dot], techniques) -> list[Dot]:
    """Dots tagged with the ID of every kept technique whose cited events overlap the dot's events."""
    linked = []
    for dot in dots:
        refs = {r for r in dot.refs if is_event_ref(r)}
        ids = tuple(dict.fromkeys(str(t.get("technique_id", "")) for t in techniques or []
                                  if isinstance(t, dict) and refs & technique_refs(t)))
        linked.append(Dot(dot.numbers, dot.time, dot.text, dot.refs, ids))
    return linked


def fit_domain(times: list[int], window_start: int, window_end: int) -> tuple[int, int]:
    """Timeline x-range: the span of the plotted times, padded by max(60 s, 10%), kept inside the incident window.

    One time alone is centred with +/- 5 minutes. A time outside the window widens the range to include it.
    """
    low, high = min(times), max(times)
    if low == high:
        return low - 300, high + 300
    pad = max(60, (high - low) // 10)
    return max(low - pad, min(window_start, low)), min(high + pad, max(window_end, high))


def merge_seconds(low: int, high: int) -> int:
    """Dots closer than 1/80 of the plotted range (at least 2 s) would overlap on screen."""
    return max(2, (high - low) // 80)


def format_created(created: str) -> str:
    """"2026-10-07T12:17:41Z" -> "7 Oct 12:17" (UTC); text that is not a timestamp is returned unchanged."""
    try:
        moment = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return created
    return f"{moment.day} {moment:%b %H:%M}"


def dot_for(dots: list[Dot], number: int | None) -> Dot | None:
    return next((d for d in dots if number in d.numbers), None) if number is not None else None


@dataclass(frozen=True)
class Chip:
    label: str
    title: str
    kind: str  # event | package | unknown | more


def evidence_chips(refs, events, limit: int = MAX_CHIPS) -> list[Chip]:
    """Chips for cited references: event time for log lines, short names for package data.

    Log events come first, in time order (events in the same second share a chip); beyond ``limit`` a single "+n" chip lists the rest.
    """
    chips: list[tuple[int, Chip]] = []
    for ref in dict.fromkeys(str(r) for r in refs or []):
        if is_event_ref(ref):
            when = event_time(events, ref)
            if when is None:
                chips.append((2**40, Chip(ref, ref, "unknown")))
                continue
            row = events.loc[ref]
            row = row.iloc[0] if hasattr(row, "iloc") and getattr(row, "ndim", 1) == 2 else row
            title = f"{ref} · {row['source_computer']} → {row['destination_computer']} · {row['authentication_type']}"
            chips.append((when, Chip(clock(when), title, "event")))
        elif ref.startswith("package:"):
            name = ref.split(":", 1)[1]
            chips.append((2**41, Chip(PACKAGE_LABELS.get(name, name.replace("_", " ")), ref, "package")))
        else:
            chips.append((2**42, Chip(ref, ref, "unknown")))
    ordered: list[Chip] = []
    for _, chip in sorted(chips, key=lambda pair: pair[0]):
        same = next((i for i, c in enumerate(ordered) if c.kind == "event" and chip.kind == "event" and c.label == chip.label), None)
        if same is None:
            ordered.append(chip)
        else:  # several events in the same second show as one chip
            ordered[same] = Chip(chip.label, ordered[same].title + "; " + chip.title, "event")
    if len(ordered) <= limit:
        return ordered
    rest = ordered[limit:]
    return ordered[:limit] + [Chip(f"+{len(rest)}", "; ".join(c.title for c in rest), "more")]


def technique_counts(views, incident_id: str) -> dict[str, int | None]:
    """Kept-technique count per mode for one incident (None when that mode has no usable result)."""
    counts: dict[str, int | None] = {}
    for mode in MODES:
        result = lookup(views, mode, incident_id)
        counts[mode] = len(result.verified.get("techniques") or []) if result.state == OK and result.verified else None
    return counts


def counts_line(counts: dict[str, int | None]) -> str:
    """e.g. "AI only 1 · AI + ATT&CK 3 · checked 3"."""
    names = {VERIFIED_MODE: "checked"}
    return " · ".join(f"{names.get(mode, mode)} {'—' if counts.get(mode) is None else counts[mode]}" for mode in reversed(MODES))


def sort_techniques(techniques) -> list[dict[str, Any]]:
    """Supported first, then the rest, each in the model's order."""
    items = [t for t in techniques or [] if isinstance(t, dict)]
    return sorted(items, key=lambda t: str(t.get("status") or (t.get("verification") or {}).get("status")) != "Supported")
