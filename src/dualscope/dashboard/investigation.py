"""Load LLM investigation outputs (Tasks 6.2-6.4) for the incident page.

Reads ``outputs/investigations/<run>/`` files only; nothing here calls a model.
Every function returns a value for bad or missing files instead of raising, so
the Investigation tab never blocks the detector evidence tabs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = REPOSITORY_ROOT / "outputs" / "investigations" / "gemini_v1"

VERIFIED_MODE = "RAG + verification"
MODES = (VERIFIED_MODE, "RAG", "Direct")
_MODE_KEY = {"RAG": "rag", "Direct": "direct"}

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
    if not folder.is_dir():
        return {}
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
