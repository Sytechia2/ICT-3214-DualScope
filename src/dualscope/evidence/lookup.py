"""Bounded lookup of LANL authentication events by source reference.

The index is built from Parquet row-group statistics, so opening a lookup does
not read event rows. A request reads only the row groups whose ``source_line``
statistics contain the requested line numbers.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


_REFERENCE = re.compile(r"auth\.txt:([1-9][0-9]*)\Z")
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 10_000


@dataclass(frozen=True)
class _RowGroup:
    path: Path
    number: int
    minimum: int
    maximum: int
    dataset_day: int


class AuthenticationEvidenceLookup:
    """Resolve ``auth.txt:<source_line>`` IDs to normalized event records.

    Args:
        dataset_root: The authentication output directory containing
            ``summary.json`` and ``events/``, or the ``events`` directory
            itself. The summary's part list is used as the authoritative file
            manifest.

    The returned record contains every stored normalized event field,
    including ``raw_record``, plus ``source_location`` with the original
    source filename and line number. Valid references absent from the selected
    dataset (including rejected or out-of-scope source lines) resolve to
    ``None``.
    """

    def __init__(self, dataset_root: str | Path):
        supplied_root = Path(dataset_root).expanduser().resolve()
        if supplied_root.name == "events":
            self.events_root = supplied_root
            self.dataset_root = supplied_root.parent
        else:
            self.dataset_root = supplied_root
            self.events_root = supplied_root / "events"
        summary_path = self.dataset_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"authentication summary not found: {summary_path}")
        if not self.events_root.is_dir():
            raise FileNotFoundError(f"authentication events directory not found: {self.events_root}")

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read authentication summary: {summary_path}") from exc
        if summary.get("source_reference_format") != "auth.txt:<source_line>":
            raise ValueError("summary does not describe auth.txt source references")
        parts = summary.get("parts")
        if not isinstance(parts, list):
            raise ValueError("authentication summary is missing its parts manifest")

        self._groups = self._read_row_group_index(parts)
        self._starts = [group.minimum for group in self._groups]
        prefix_maximum: list[int] = []
        running_maximum = -1
        for group in self._groups:
            running_maximum = max(running_maximum, group.maximum)
            prefix_maximum.append(running_maximum)
        self._prefix_maximum = prefix_maximum

    def _read_row_group_index(self, parts: list[object]) -> list[_RowGroup]:
        groups: list[_RowGroup] = []
        root = self.events_root.resolve()
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("path"), str):
                raise ValueError("invalid part entry in authentication summary")
            relative_path = Path(part["path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("unsafe part path in authentication summary")
            path = (root / relative_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("part path resolves outside the authentication events directory")
            if not path.is_file():
                raise FileNotFoundError(f"authentication Parquet part not found: {path}")
            try:
                dataset_day = int(part["dataset_day"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("part entry is missing a valid dataset_day") from exc
            parquet = pq.ParquetFile(path)
            source_line_column = None
            for column_index in range(parquet.metadata.num_columns):
                if parquet.schema_arrow.names[column_index] == "source_line":
                    source_line_column = column_index
                    break
            if source_line_column is None:
                raise ValueError(f"Parquet part has no source_line column: {path}")
            for row_group_number in range(parquet.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_number)
                stats = row_group.column(source_line_column).statistics
                if stats is None or not stats.has_min_max:
                    raise ValueError(
                        "source_line min/max statistics are required for bounded evidence lookup; "
                        f"missing in {path}, row group {row_group_number}"
                    )
                minimum, maximum = int(stats.min), int(stats.max)
                if minimum < 1 or maximum < minimum:
                    raise ValueError(f"invalid source_line statistics in {path}, row group {row_group_number}")
                groups.append(_RowGroup(path, row_group_number, minimum, maximum, dataset_day))

        # Metadata ranges need not be disjoint. Sorting plus prefix maxima lets
        # us find every possible containing group without inspecting event rows.
        groups.sort(
            key=lambda group: (group.minimum, group.maximum, str(group.path), group.number)
        )
        return groups

    @staticmethod
    def _parse_reference(source_reference: str) -> int:
        if not isinstance(source_reference, str):
            raise ValueError("source_reference must be a string")
        match = _REFERENCE.fullmatch(source_reference)
        if match is None:
            raise ValueError("source_reference must have the form 'auth.txt:<positive source line>'")
        return int(match.group(1))

    def _candidate_groups(self, source_line: int) -> list[_RowGroup]:
        upper = bisect_right(self._starts, source_line)
        candidates: list[_RowGroup] = []
        index = upper - 1
        while index >= 0 and self._prefix_maximum[index] >= source_line:
            group = self._groups[index]
            if group.maximum >= source_line:
                candidates.append(group)
            index -= 1
        return candidates

    @staticmethod
    def _event_from_row(row: dict[str, object], source_line: int) -> dict[str, object]:
        event = dict(row)
        event["source_location"] = {"file": "auth.txt", "line": source_line}
        return event

    def lookup(self, source_reference: str) -> dict[str, object] | None:
        """Return one matching event, or ``None`` when it is absent."""

        return self.lookup_many([source_reference])[source_reference]

    def lookup_many(
        self, source_references: Iterable[str]
    ) -> dict[str, dict[str, object] | None]:
        """Resolve many IDs while reading each candidate row group at most once.

        The returned dictionary preserves first-seen input order. Repeated IDs
        are represented once; different source lines remain distinct even if
        their raw records have identical text.
        """

        references = list(dict.fromkeys(source_references))
        lines = {reference: self._parse_reference(reference) for reference in references}
        results: dict[str, dict[str, object] | None] = {reference: None for reference in references}
        groups_to_references: dict[_RowGroup, list[str]] = {}
        for reference, source_line in lines.items():
            for group in self._candidate_groups(source_line):
                groups_to_references.setdefault(group, []).append(reference)

        for group, group_references in groups_to_references.items():
            wanted = set(group_references)
            parquet = pq.ParquetFile(group.path)
            # Metadata min/max only narrows the read; source_reference equality
            # is the final check and protects against ranges with gaps. Read
            # batches so a large Parquet row group does not become a second,
            # fully materialized copy in Python memory.
            for batch in parquet.iter_batches(row_groups=[group.number], batch_size=8192):
                for row in batch.to_pylist():
                    reference = row.get("source_reference")
                    if reference in wanted:
                        source_line = lines[reference]
                        if row.get("source_line") != source_line:
                            raise ValueError(
                                f"source reference and source_line disagree in {group.path}: "
                                f"{reference!r} has source_line {row.get('source_line')!r}"
                            )
                        if results[reference] is not None:
                            raise ValueError(f"source reference occurs in multiple rows: {reference}")
                        event = self._event_from_row(row, source_line)
                        event["dataset_day"] = group.dataset_day
                        results[reference] = event
        return results

    def resolve_page(
        self,
        source_references: Iterable[str],
        *,
        offset: int = 0,
        limit: int = _DEFAULT_PAGE_SIZE,
    ) -> dict[str, object]:
        """Resolve one page of a detector's evidence IDs.

        Pagination is applied before lookup, so large evidence lists can be
        served in bounded requests. The ``total`` counts input evidence IDs,
        including repeats, and pagination preserves their order and
        multiplicity. Only the requested page is retained in memory. An
        iterable is consumed once to calculate the exact total, but rows are
        read only for IDs on the page.
        """

        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if isinstance(source_references, (str, bytes)):
            raise TypeError("source_references must be an iterable of IDs, not a single string")
        page_references: list[str] = []
        total = 0
        page_end = offset + limit
        for index, reference in enumerate(source_references):
            if offset <= index < page_end:
                page_references.append(reference)
            total = index + 1
        resolved = self.lookup_many(page_references)
        return {
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": [
                {"source_reference": reference, "event": resolved[reference]}
                for reference in page_references
            ],
        }
