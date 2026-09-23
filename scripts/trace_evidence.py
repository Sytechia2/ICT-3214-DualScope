"""Trace a sample incident's detector evidence back to LANL source rows.

Run from the repository root with ``python scripts/trace_evidence.py``. The
default data root is the checked-in, small LANL ingestion sample, so the demo
does not require a full LANL dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "data" / "samples" / "lanl_ingestion_sample" / "authentication"

# The first three rows are U1@DOM1 activity within the first hour. Rows 1 and
# 2 are exact duplicate raw records but retain distinct source references.
# Together with the U1 -> C1 edge, both detector outputs describe the same user.
SAMPLE_SEQUENCE_REFERENCES = ("auth.txt:1", "auth.txt:2", "auth.txt:3")
SAMPLE_EDGE_REFERENCES = ("auth.txt:1", "auth.txt:2")


def _load_api() -> tuple[Any, Any, Any, Any]:
    """Load project code when invoked directly from the scripts directory."""
    source_root = str(REPOSITORY_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from dualscope.evidence.carriers import (  # noqa: PLC0415
        attach_edge_references,
        attach_sequence_references,
        paginate_references,
    )
    from dualscope.evidence.lookup import AuthenticationEvidenceLookup  # noqa: PLC0415

    return (
        AuthenticationEvidenceLookup,
        attach_sequence_references,
        attach_edge_references,
        paginate_references,
    )


def build_trace(dataset_root: Path, page_size: int = 1) -> dict[str, object]:
    """Build an incident-to-source trace and two stable pages of edge evidence."""
    (
        AuthenticationEvidenceLookup,
        attach_sequence_references,
        attach_edge_references,
        paginate_references,
    ) = _load_api()

    lookup = AuthenticationEvidenceLookup(dataset_root)
    sequence_events_by_ref = lookup.lookup_many(SAMPLE_SEQUENCE_REFERENCES)
    edge_events_by_ref = lookup.lookup_many(SAMPLE_EDGE_REFERENCES)
    missing = [
        reference
        for reference, event in {**sequence_events_by_ref, **edge_events_by_ref}.items()
        if event is None
    ]
    if missing:
        raise ValueError(
            "The selected authentication dataset is missing sample evidence rows: "
            + ", ".join(missing)
        )

    sequence_events = [sequence_events_by_ref[ref] for ref in SAMPLE_SEQUENCE_REFERENCES]
    edge_events = [edge_events_by_ref[ref] for ref in SAMPLE_EDGE_REFERENCES]

    sequence_users = {event["acting_user"] for event in sequence_events}
    sequence_hours = {int(event["timestamp"]) // 3600 for event in sequence_events}
    if sequence_users != {"U1@DOM1"} or len(sequence_hours) != 1:
        raise ValueError("sample sequence must contain one acting user in one hour")
    if (
        sequence_events[0]["raw_record"] != sequence_events[1]["raw_record"]
        or sequence_events[0]["source_reference"] == sequence_events[1]["source_reference"]
        or sequence_events[0]["exact_duplicate_ordinal"] == sequence_events[1]["exact_duplicate_ordinal"]
    ):
        raise ValueError("sample duplicate rows must retain distinct source references")

    sequence = attach_sequence_references(
        {
            "sequence_id": "sequence-incident-001",
            "user": "U1@DOM1",
            "window": f"dataset-relative hour {next(iter(sequence_hours))}",
        },
        sequence_events,
    )
    edge = attach_edge_references(
        {"source_user": "U1@DOM1", "destination_computer": "C1", "relation": "logon"},
        edge_events,
    )

    # The incident packages both detector outputs. The evidence IDs are the
    # join keys into the normalized authentication dataset.
    incident = {
        "incident_id": "sample-incident-001",
        "summary": "U1@DOM1 activity sequence and repeated logon edge to C1",
        "sequence": sequence,
        "graph_edges": [edge],
    }

    page_1 = paginate_references(edge["source_references"], offset=0, limit=page_size)
    page_2 = paginate_references(
        edge["source_references"],
        offset=page_1["next_offset"] if page_1["next_offset"] is not None else len(edge["source_references"]),
        limit=page_size,
    )
    resolved_pages = [
        lookup.resolve_page(
            edge["source_references"], offset=page["offset"], limit=page["limit"]
        )
        for page in (page_1, page_2)
        if page["source_references"]
    ]

    sequence_sources = [
        {"source_reference": ref, "event": sequence_events_by_ref[ref]}
        for ref in sequence["source_references"]
    ]
    return {
        "incident": incident,
        "sequence_source_records": sequence_sources,
        "graph_edge_evidence_pages": resolved_pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="authentication ingestion output (defaults to the checked-in sample)",
    )
    parser.add_argument(
        "--page-size", type=int, default=1, help="source references per graph evidence page"
    )
    args = parser.parse_args()
    if args.page_size < 1:
        parser.error("--page-size must be at least 1")
    try:
        result = build_trace(args.dataset_root, args.page_size)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
