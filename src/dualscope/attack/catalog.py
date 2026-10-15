"""Compact Enterprise ATT&CK technique catalogue built from a pinned STIX bundle.

``extract_techniques`` keeps the active (not revoked, not deprecated)
techniques and sub-techniques with the fields investigation needs: ID, name,
description, tactics, platforms, source URL, and the data components and log
sources ATT&CK's detection strategies list for them. ``TechniqueCatalog``
loads the extract and answers ID checks for retrieval and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

TECHNIQUE_ID = re.compile(r"T[0-9]{4}(\.[0-9]{3})?\Z")
_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def clean_description(text: str) -> str:
    """Drop ATT&CK citation markers and keep only the text of Markdown links."""
    text = _CITATION.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _attack_reference(stix_object: Mapping[str, Any]) -> Mapping[str, Any]:
    for reference in stix_object.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            return reference
    raise ValueError(f"{stix_object.get('id')} has no mitre-attack external reference")


def _active(stix_object: Mapping[str, Any]) -> bool:
    return not stix_object.get("revoked", False) and not stix_object.get("x_mitre_deprecated", False)


def extract_techniques(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Active techniques from an Enterprise ATT&CK STIX 2.1 bundle, sorted by ID."""
    objects = bundle["objects"]
    by_id = {item["id"]: item for item in objects}

    # Detection strategy --detects--> technique; strategy -> analytics -> log sources.
    detection: dict[str, dict[str, set[str]]] = {}
    for relation in objects:
        if relation.get("type") != "relationship" or relation.get("relationship_type") != "detects":
            continue
        strategy = by_id.get(relation["source_ref"])
        if strategy is None or not _active(strategy) or relation.get("revoked"):
            continue
        found = detection.setdefault(relation["target_ref"], {"data_components": set(), "log_sources": set()})
        for analytic_ref in strategy.get("x_mitre_analytic_refs", []):
            analytic = by_id.get(analytic_ref)
            if analytic is None or not _active(analytic):
                continue
            for source in analytic.get("x_mitre_log_source_references", []):
                component = by_id.get(source.get("x_mitre_data_component_ref", ""))
                if component is not None:
                    found["data_components"].add(component["name"])
                found["log_sources"].add(f"{source['name']} {source.get('channel', '')}".strip())

    techniques = []
    for item in objects:
        if item.get("type") != "attack-pattern" or not _active(item):
            continue
        reference = _attack_reference(item)
        technique_id = reference["external_id"]
        if not TECHNIQUE_ID.fullmatch(technique_id):
            raise ValueError(f"unexpected technique ID {technique_id!r}")
        found = detection.get(item["id"], {"data_components": set(), "log_sources": set()})
        techniques.append({
            "technique_id": technique_id,
            "name": item["name"],
            "description": clean_description(item.get("description", "")),
            "tactics": sorted(phase["phase_name"] for phase in item.get("kill_chain_phases", [])
                              if phase.get("kill_chain_name") == "mitre-attack"),
            "platforms": sorted(item.get("x_mitre_platforms", [])),
            "is_subtechnique": bool(item.get("x_mitre_is_subtechnique", False)),
            "parent_id": technique_id.split(".")[0] if "." in technique_id else None,
            "url": reference["url"],
            "version": item.get("x_mitre_version"),
            "modified": item.get("modified"),
            "data_components": sorted(found["data_components"]),
            "log_sources": sorted(found["log_sources"]),
        })
    techniques.sort(key=lambda technique: technique["technique_id"])
    ids = [technique["technique_id"] for technique in techniques]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate active technique IDs in bundle")
    return techniques


def collection_info(bundle: Mapping[str, Any]) -> dict[str, Any]:
    collections = [item for item in bundle["objects"] if item.get("type") == "x-mitre-collection"]
    if len(collections) != 1:
        raise ValueError("bundle must contain exactly one x-mitre-collection")
    return {
        "name": collections[0]["name"],
        "version": collections[0].get("x_mitre_version"),
        "modified": collections[0].get("modified"),
    }


@dataclass(frozen=True)
class Technique:
    technique_id: str
    name: str
    description: str
    tactics: tuple[str, ...]
    platforms: tuple[str, ...]
    is_subtechnique: bool
    parent_id: str | None
    url: str
    data_components: tuple[str, ...]
    log_sources: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Technique:
        return cls(
            technique_id=data["technique_id"],
            name=data["name"],
            description=data["description"],
            tactics=tuple(data["tactics"]),
            platforms=tuple(data["platforms"]),
            is_subtechnique=bool(data["is_subtechnique"]),
            parent_id=data["parent_id"],
            url=data["url"],
            data_components=tuple(data["data_components"]),
            log_sources=tuple(data["log_sources"]),
        )


class TechniqueCatalog:
    """Active techniques of one ATT&CK snapshot, keyed by technique ID."""

    def __init__(self, techniques: Iterable[Technique], snapshot: Mapping[str, Any]):
        self.techniques = list(techniques)
        self.snapshot = dict(snapshot)
        self._by_id = {technique.technique_id: technique for technique in self.techniques}

    @classmethod
    def load(cls, path: str | Path) -> TechniqueCatalog:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls((Technique.from_dict(item) for item in data["techniques"]), data["snapshot"])

    def __len__(self) -> int:
        return len(self.techniques)

    def __contains__(self, technique_id: object) -> bool:
        return technique_id in self._by_id

    def get(self, technique_id: str) -> Technique | None:
        return self._by_id.get(technique_id)
