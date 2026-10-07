"""Generate one investigation summary and check its shape (Task 6.2).

``investigate`` never raises for a model or parsing problem: the record gets
``status`` ``failed`` (the call failed) or ``invalid`` (the reply did not
match the schema), keeps the raw text and the error, and the detector incident
stays usable without a summary. Evidence checking is Task 6.3
(``dualscope.investigation.verify``).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Protocol

from dualscope.attack.catalog import TECHNIQUE_ID
from dualscope.investigation.gemini import Generation, GenerationError
from dualscope.investigation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, user_prompt

CONFIDENCE = {"low", "medium", "high"}


class Generator(Protocol):
    def generate(self, system: str, user: str, schema: Mapping[str, Any]) -> Generation: ...


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_response(data: Any) -> list[str]:
    """Schema problems in a parsed reply; an empty list means it is usable.

    Technique IDs are only checked for format here; whether they exist in
    ATT&CK, and whether the evidence supports them, is verification's job.
    """
    if not isinstance(data, dict):
        return ["reply is not a JSON object"]
    problems = []
    for key, kind in (("summary", str), ("observations", list), ("interpretations", list),
                      ("techniques", list), ("no_supported_mapping", bool), ("uncertainty", list)):
        if not isinstance(data.get(key), kind):
            problems.append(f"{key} missing or not a {kind.__name__}")
    if problems:
        return problems
    for section in ("observations", "interpretations"):
        for index, item in enumerate(data[section]):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not _strings(item.get("evidence")):
                problems.append(f"{section}[{index}] needs text and an evidence list")
            elif section == "interpretations" and item.get("confidence") not in CONFIDENCE:
                problems.append(f"{section}[{index}] has no valid confidence")
    for index, item in enumerate(data["techniques"]):
        if not isinstance(item, dict) or not _strings(item.get("evidence")):
            problems.append(f"techniques[{index}] needs an evidence list")
            continue
        for key in ("technique_id", "name", "rationale"):
            if not isinstance(item.get(key), str):
                problems.append(f"techniques[{index}].{key} missing")
        if isinstance(item.get("technique_id"), str) and not TECHNIQUE_ID.fullmatch(item["technique_id"].strip()):
            problems.append(f"techniques[{index}].technique_id {item['technique_id']!r} is not an ATT&CK ID")
        if item.get("confidence") not in CONFIDENCE:
            problems.append(f"techniques[{index}] has no valid confidence")
    if not _strings(data["uncertainty"]):
        problems.append("uncertainty must be a list of strings")
    if data["no_supported_mapping"] and data["techniques"]:
        problems.append("no_supported_mapping is true but techniques are listed")
    return problems


def prompt_sha256(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n\n" + user).encode("utf-8")).hexdigest()


def investigate(
    mode: str,
    package: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
    generator: Generator,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Run one incident in one mode; returns the output record and the user prompt sent."""
    prompt = user_prompt(mode, package, retrieval)
    record: dict[str, Any] = {
        "incident_id": package["incident_id"],
        "mode": mode,
        "model": settings.get("model"),
        "settings": dict(settings),
        "prompt_sha256": prompt_sha256(SYSTEM_PROMPT, prompt),
        "retrieved_candidates": [c["technique_id"] for c in retrieval["candidates"]] if mode == "rag" and retrieval else None,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "response": None,
        "errors": [],
        "raw_text": None,
    }
    try:
        generation = generator.generate(SYSTEM_PROMPT, prompt, RESPONSE_SCHEMA)
    except GenerationError as exc:
        record.update(status="failed", errors=[str(exc)])
        return record, prompt
    record.update(
        raw_text=generation.text,
        model_version=generation.model_version,
        finish_reason=generation.finish_reason,
        usage=generation.usage,
        latency_seconds=generation.latency_seconds,
        attempts=generation.attempts,
    )
    try:
        data = json.loads(generation.text)
    except json.JSONDecodeError as exc:
        record.update(status="invalid", errors=[f"reply is not JSON ({generation.finish_reason}): {exc}"])
        return record, prompt
    problems = validate_response(data)
    if problems:
        record.update(status="invalid", errors=problems)
        return record, prompt
    for technique in data["techniques"]:
        technique["technique_id"] = technique["technique_id"].strip()
    record.update(response=data, raw_text=None)
    return record, prompt
