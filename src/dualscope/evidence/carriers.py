"""Carry source event references into detector evidence records.

The helpers accept row mappings (including mappings yielded from Parquet batches)
and operate on only the iterable supplied by the caller. They preserve input
order and repeated references so downstream evidence can represent every
contributing event, including exact duplicate events.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Mapping


def _collect_references(
    events: Iterable[Mapping[str, Any]],
    reference_field: str,
    split_field: str | None = None,
) -> tuple[list[str], set[Any]]:
    references: list[str] = []
    splits: set[Any] = set()
    for index, event in enumerate(events):
        if reference_field not in event:
            raise ValueError(
                f"Event at position {index} is missing reference field '{reference_field}'"
            )
        reference = event[reference_field]
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                f"Event at position {index} has an invalid '{reference_field}' value: "
                f"{reference!r}"
            )
        references.append(reference)
        if split_field is not None and split_field in event and event[split_field] is not None:
            splits.add(event[split_field])
    return references, splits


def attach_sequence_references(
    sequence: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    reference_field: str = "source_reference",
    split_field: str = "split",
) -> dict[str, Any]:
    """Copy a user-hour sequence and attach its ordered event references.

    Event order is retained exactly as supplied by the sequence builder, and
    duplicate source references are retained. If split metadata is present on
    the sequence or its events, the helper rejects events assigned to another
    split (and mixed split values when the sequence has no split field).
    """
    references, event_splits = _collect_references(events, reference_field, split_field)
    sequence_split = sequence.get(split_field)
    if sequence_split is not None:
        mismatched = event_splits - {sequence_split}
        if mismatched:
            raise ValueError(
                f"Sequence split {sequence_split!r} contains events from other splits: "
                f"{sorted(mismatched, key=repr)!r}"
            )
    elif len(event_splits) > 1:
        raise ValueError(
            "Sequence events cross split boundaries: "
            f"{sorted(event_splits, key=repr)!r}"
        )

    result = dict(sequence)
    result["source_references"] = references
    return result


def attach_edge_references(
    edge: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    reference_field: str = "source_reference",
) -> dict[str, Any]:
    """Copy an aggregated graph edge and attach its contributing references.

    The event iterable must be in the desired source order. Repeated references
    are intentional and remain in the returned list.
    """
    references, _ = _collect_references(events, reference_field)
    result = dict(edge)
    result["source_references"] = references
    result["evidence_count"] = len(references)
    return result


def paginate_references(
    references: Iterable[str], *, offset: int = 0, limit: int = 100
) -> dict[str, Any]:
    """Return one stable page from an ordered reference iterable.

    Uses a one-reference lookahead, so it does not materialize a potentially
    large iterable. To request later pages from a one-shot iterator, pass a new
    iterator over the same ordered references for each call.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if isinstance(references, (str, bytes)):
        raise TypeError("references must be an iterable of IDs, not a single string")

    window = list(islice(iter(references), offset, offset + limit + 1))
    has_more = len(window) > limit
    page = window[:limit]
    return {
        "source_references": page,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": offset + len(page) if has_more else None,
    }
