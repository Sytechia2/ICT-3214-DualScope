"""Streaming normalisation for the LANL authentication dataset.

The parser deliberately does not attach red-team labels to authentication
events. Events and labels are written to separate Parquet trees so detector
inputs cannot accidentally include evaluation labels.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO

import pyarrow as pa
import pyarrow.parquet as pq


SECONDS_PER_DAY = 86_400
AUTH_FIELD_NAMES = (
    "timestamp",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "authentication_result",
)
LABEL_FIELD_NAMES = ("timestamp", "user", "source_computer", "destination_computer")

EVENT_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("source_user", pa.string()),
        ("destination_user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("authentication_type", pa.string()),
        ("logon_type", pa.string()),
        ("authentication_orientation", pa.string()),
        ("authentication_result", pa.string()),
        ("acting_user", pa.string()),
        ("exact_duplicate_ordinal", pa.int32()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("raw_record", pa.string()),
    ]
)

LABEL_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("raw_record", pa.string()),
    ]
)


@dataclass(frozen=True)
class IngestionConfig:
    """Settings that define one deterministic ingestion run."""

    day_start: int = 1
    day_end: int = 30
    chunk_rows: int = 500_000
    compression: str = "zstd"
    max_source_rows: int | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.day_start < 1 or self.day_end < self.day_start:
            raise ValueError("day range must be positive and inclusive")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        if self.max_source_rows is not None and self.max_source_rows < 1:
            raise ValueError("max_source_rows must be positive when provided")


@dataclass(frozen=True)
class DayRange:
    """An exact, line-aligned source range from the Task 2.1 inspection."""

    day: int
    byte_start: int
    byte_end: int
    source_line_start: int
    expected_rows: int


def dataset_day(timestamp: int) -> int:
    """Convert a positive dataset-relative second to its one-based day."""

    if timestamp < 1:
        raise ValueError("timestamp must be a positive dataset-relative second")
    return ((timestamp - 1) // SECONDS_PER_DAY) + 1


def _prepare_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("refusing to use a filesystem root as output")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_progress(path: Path, value: dict[str, object]) -> None:
    """Atomically publish progress so readers never observe partial JSON."""

    temporary = path.with_suffix(".json.tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _dataset_schema(schema: pa.Schema) -> list[dict[str, str]]:
    """Describe physical columns plus the Hive-derived partition column."""

    return (
        [{"name": "dataset_day", "type": "int32", "storage": "hive_partition"}]
        + [{"name": field.name, "type": str(field.type), "storage": "parquet"} for field in schema]
    )


def _write_rejection(handle: TextIO, *, line: int, raw: str, reason: str) -> None:
    handle.write(json.dumps({"source_line": line, "reason": reason, "raw_record": raw}) + "\n")


def _write_partitions(
    rows: list[dict[str, object]],
    root: Path,
    schema: pa.Schema,
    counters: defaultdict[int, int],
    compression: str,
) -> list[dict[str, object]]:
    by_day: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_day[int(row["dataset_day"])].append(row)

    written: list[dict[str, object]] = []
    for day in sorted(by_day):
        day_dir = root / f"dataset_day={day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        part_number = counters[day]
        part_path = day_dir / f"part-{part_number:06d}.parquet"
        table = pa.Table.from_pylist(by_day[day], schema=schema)
        pq.write_table(
            table,
            part_path,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        counters[day] += 1
        written.append(
            {
                "dataset_day": day,
                "part": part_number,
                "path": part_path.relative_to(root).as_posix(),
                "rows": table.num_rows,
                "minimum_timestamp": min(table.column("timestamp").to_pylist()),
                "maximum_timestamp": max(table.column("timestamp").to_pylist()),
            }
        )
    return written


def _chunks(handle: TextIO, size: int) -> Iterable[list[tuple[int, str]]]:
    chunk: list[tuple[int, str]] = []
    for line_number, line in enumerate(handle, start=1):
        chunk.append((line_number, line.rstrip("\r\n")))
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def ingest_authentication(source: Path, output: Path, config: IngestionConfig) -> dict[str, object]:
    """Stream LANL auth rows into day-partitioned, ordered Parquet files."""

    _prepare_output(output, config.overwrite)
    events_root = output / "events"
    events_root.mkdir()
    rejections_path = output / "rejections.jsonl"
    parts: list[dict[str, object]] = []
    part_counters: defaultdict[int, int] = defaultdict(int)
    category_counts = {name: Counter() for name in AUTH_FIELD_NAMES[5:]}
    day_counts: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    duplicate_instances = 0
    duplicate_groups = 0
    input_rows = accepted_rows = out_of_scope_rows = 0
    previous_timestamp: int | None = None
    current_timestamp: int | None = None
    current_timestamp_records: Counter[tuple[str, ...]] = Counter()
    stopped_at_day_end = False

    with source.open("r", encoding="utf-8", newline="") as handle, rejections_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejected:
        stop = False
        for chunk in _chunks(handle, config.chunk_rows):
            normalized: list[dict[str, object]] = []
            for line_number, raw in chunk:
                if config.max_source_rows is not None and input_rows >= config.max_source_rows:
                    stop = True
                    break
                input_rows += 1
                fields = next(csv.reader([raw]))
                if len(fields) != len(AUTH_FIELD_NAMES):
                    rejection_counts["field_count"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="field_count")
                    continue
                try:
                    timestamp = int(fields[0])
                    day = dataset_day(timestamp)
                except ValueError:
                    rejection_counts["invalid_timestamp"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="invalid_timestamp")
                    continue
                if any(value == "" for value in fields[1:]):
                    rejection_counts["empty_field"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="empty_field")
                    continue
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(
                        f"timestamp order inversion at {source.name}:{line_number}: "
                        f"{timestamp} < {previous_timestamp}"
                    )
                previous_timestamp = timestamp
                if day < config.day_start:
                    out_of_scope_rows += 1
                    continue
                if day > config.day_end:
                    out_of_scope_rows += 1
                    stopped_at_day_end = True
                    stop = True
                    break

                exact_key = tuple(fields)
                if timestamp != current_timestamp:
                    current_timestamp = timestamp
                    current_timestamp_records.clear()
                current_timestamp_records[exact_key] += 1
                duplicate_ordinal = current_timestamp_records[exact_key]
                if duplicate_ordinal == 2:
                    duplicate_groups += 1
                if duplicate_ordinal > 1:
                    duplicate_instances += 1

                row = dict(zip(AUTH_FIELD_NAMES, fields, strict=True))
                row["timestamp"] = timestamp
                row["dataset_day"] = day
                row["acting_user"] = row["source_user"]
                row["exact_duplicate_ordinal"] = duplicate_ordinal
                row["source_line"] = line_number
                row["source_reference"] = f"auth.txt:{line_number}"
                row["raw_record"] = raw
                normalized.append(row)
                accepted_rows += 1
                day_counts[day] += 1
                for name in category_counts:
                    category_counts[name][str(row[name])] += 1
            if normalized:
                parts.extend(
                    _write_partitions(
                        normalized, events_root, EVENT_SCHEMA, part_counters, config.compression
                    )
                )
            if stop:
                break

    rejected_rows = sum(rejection_counts.values())
    summary: dict[str, object] = {
        "format_version": 1,
        "dataset": "LANL authentication",
        "source_file": str(source),
        "source_reference_format": "auth.txt:<source_line>",
        "acting_user_definition": "source_user",
        "timestamp_definition": "positive dataset-relative seconds",
        "dataset_day_formula": "((timestamp - 1) // 86400) + 1",
        "selection": {"day_start_inclusive": config.day_start, "day_end_inclusive": config.day_end},
        "storage": {
            "format": "parquet",
            "compression": config.compression,
            "partitioning": "dataset_day",
            "part_order": "six-digit sequence in source order",
        },
        "chunk_rows": config.chunk_rows,
        "max_source_rows": config.max_source_rows,
        "counts": {
            "input_rows_scanned": input_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "out_of_scope_rows": out_of_scope_rows,
            "reconciled": input_rows == accepted_rows + rejected_rows + out_of_scope_rows,
            "exact_duplicate_instances_beyond_first": duplicate_instances,
            "distinct_exact_duplicate_groups": duplicate_groups,
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "rows_per_day": {str(k): v for k, v in sorted(day_counts.items())},
        "category_counts": {
            name: dict(sorted(counts.items())) for name, counts in category_counts.items()
        },
        "ordering": {
            "validated_non_decreasing_across_chunks": True,
            "stopped_after_first_row_beyond_day_end": stopped_at_day_end,
        },
        "deduplication": "none; every accepted source row is retained",
        "labels_in_detector_events": False,
        "parts": parts,
        "schema": _dataset_schema(EVENT_SCHEMA),
    }
    _write_json(output / "summary.json", summary)
    return summary


def _load_day_ranges(inspection: Path, day_start: int, day_end: int) -> list[DayRange]:
    payload = json.loads(inspection.read_text(encoding="utf-8"))
    stream = payload["authentication_stream_inspection"]
    row_counts = {int(day): int(count) for day, count in stream["row_counts_per_dataset_day"].items()}
    byte_counts = {
        int(day): int(count) for day, count in stream["byte_counts_per_dataset_day"].items()
    }
    if set(row_counts) != set(byte_counts):
        raise ValueError("inspection row and byte day keys do not match")

    byte_offset = 0
    line_offset = 0
    selected: list[DayRange] = []
    for day in sorted(row_counts):
        byte_end = byte_offset + byte_counts[day]
        if day_start <= day <= day_end:
            selected.append(
                DayRange(
                    day=day,
                    byte_start=byte_offset,
                    byte_end=byte_end,
                    source_line_start=line_offset + 1,
                    expected_rows=row_counts[day],
                )
            )
        byte_offset = byte_end
        line_offset += row_counts[day]
    if [item.day for item in selected] != list(range(day_start, day_end + 1)):
        raise ValueError("inspection does not contain every requested dataset day")
    return selected


def _ingest_authentication_day(
    source_text: str,
    events_root_text: str,
    rejections_root_text: str,
    day_range: DayRange,
    chunk_rows: int,
    compression: str,
) -> dict[str, object]:
    """Worker entry point: parse and write one independent dataset day."""

    source = Path(source_text)
    events_root = Path(events_root_text)
    rejection_path = Path(rejections_root_text) / f"dataset_day={day_range.day:02d}.jsonl"
    part_counters: defaultdict[int, int] = defaultdict(int)
    parts: list[dict[str, object]] = []
    category_counts = {name: Counter() for name in AUTH_FIELD_NAMES[5:]}
    rejection_counts: Counter[str] = Counter()
    accepted_rows = input_rows = 0
    duplicate_instances = duplicate_groups = 0
    previous_timestamp: int | None = None
    current_timestamp: int | None = None
    current_timestamp_records: Counter[tuple[str, ...]] = Counter()
    line_number = day_range.source_line_start

    with source.open("rb") as handle, rejection_path.open("w", encoding="utf-8", newline="\n") as rejected:
        handle.seek(day_range.byte_start)
        while handle.tell() < day_range.byte_end:
            normalized: list[dict[str, object]] = []
            for _ in range(chunk_rows):
                if handle.tell() >= day_range.byte_end:
                    break
                raw_bytes = handle.readline()
                if not raw_bytes or handle.tell() > day_range.byte_end:
                    raise ValueError(f"inspection byte boundary is not line-aligned for day {day_range.day}")
                raw = raw_bytes.rstrip(b"\r\n").decode("utf-8")
                input_rows += 1
                fields = raw.split(",")
                if len(fields) != len(AUTH_FIELD_NAMES):
                    rejection_counts["field_count"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="field_count")
                    line_number += 1
                    continue
                try:
                    timestamp = int(fields[0])
                    day = dataset_day(timestamp)
                except ValueError:
                    rejection_counts["invalid_timestamp"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="invalid_timestamp")
                    line_number += 1
                    continue
                if any(value == "" for value in fields[1:]):
                    rejection_counts["empty_field"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="empty_field")
                    line_number += 1
                    continue
                if day != day_range.day:
                    raise ValueError(
                        f"inspection boundary mismatch: expected day {day_range.day}, got day {day} "
                        f"at auth.txt:{line_number}"
                    )
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(f"timestamp order inversion at auth.txt:{line_number}")
                previous_timestamp = timestamp

                exact_key = tuple(fields)
                if timestamp != current_timestamp:
                    current_timestamp = timestamp
                    current_timestamp_records.clear()
                current_timestamp_records[exact_key] += 1
                duplicate_ordinal = current_timestamp_records[exact_key]
                if duplicate_ordinal == 2:
                    duplicate_groups += 1
                if duplicate_ordinal > 1:
                    duplicate_instances += 1

                row = dict(zip(AUTH_FIELD_NAMES, fields, strict=True))
                row["timestamp"] = timestamp
                row["dataset_day"] = day
                row["acting_user"] = row["source_user"]
                row["exact_duplicate_ordinal"] = duplicate_ordinal
                row["source_line"] = line_number
                row["source_reference"] = f"auth.txt:{line_number}"
                row["raw_record"] = raw
                normalized.append(row)
                accepted_rows += 1
                for name in category_counts:
                    category_counts[name][str(row[name])] += 1
                line_number += 1
            if normalized:
                parts.extend(
                    _write_partitions(
                        normalized, events_root, EVENT_SCHEMA, part_counters, compression
                    )
                )

    if input_rows != day_range.expected_rows:
        raise ValueError(
            f"day {day_range.day} row mismatch: read {input_rows}, expected {day_range.expected_rows}"
        )
    return {
        "day": day_range.day,
        "input_rows": input_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": sum(rejection_counts.values()),
        "duplicate_instances": duplicate_instances,
        "duplicate_groups": duplicate_groups,
        "rejection_reasons": dict(rejection_counts),
        "category_counts": {name: dict(counts) for name, counts in category_counts.items()},
        "parts": parts,
    }


def ingest_authentication_parallel(
    source: Path,
    output: Path,
    config: IngestionConfig,
    inspection: Path,
    workers: int,
) -> dict[str, object]:
    """Normalise independent day ranges concurrently in worker processes."""

    if workers < 2:
        raise ValueError("parallel ingestion requires at least two workers")
    if config.max_source_rows is not None:
        raise ValueError("max_source_rows is supported only by serial sample ingestion")
    day_ranges = _load_day_ranges(inspection, config.day_start, config.day_end)
    if source.stat().st_size < max(item.byte_end for item in day_ranges):
        raise ValueError("authentication source is shorter than the inspected byte ranges")

    _prepare_output(output, config.overwrite)
    events_root = output / "events"
    rejections_root = output / "rejections"
    events_root.mkdir()
    rejections_root.mkdir()
    progress_path = output / "progress.json"
    expected_rows = sum(item.expected_rows for item in day_ranges)
    completed: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)

    def publish(status: str, error: str | None = None) -> None:
        scanned = sum(int(item["input_rows"]) for item in completed)
        progress: dict[str, object] = {
            "status": status,
            "started_at_utc": started.isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "workers": workers,
            "completed_days": sorted(int(item["day"]) for item in completed),
            "days_total": len(day_ranges),
            "rows_completed": scanned,
            "rows_expected": expected_rows,
            "percent_complete": round((100 * scanned / expected_rows), 4),
            "parts_written": sum(len(item["parts"]) for item in completed),
        }
        if error is not None:
            progress["error"] = error
        _write_progress(progress_path, progress)

    publish("running")
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _ingest_authentication_day,
                    str(source),
                    str(events_root),
                    str(rejections_root),
                    day_range,
                    config.chunk_rows,
                    config.compression,
                ): day_range.day
                for day_range in day_ranges
            }
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                publish("running")
                scanned = sum(int(item["input_rows"]) for item in completed)
                print(
                    f"[progress] day {result['day']:02d} complete; "
                    f"{scanned:,}/{expected_rows:,} rows ({100 * scanned / expected_rows:.2f}%)",
                    flush=True,
                )
    except Exception as exc:
        publish("failed", str(exc))
        raise

    completed.sort(key=lambda item: int(item["day"]))
    rejection_counts: Counter[str] = Counter()
    category_counts = {name: Counter() for name in AUTH_FIELD_NAMES[5:]}
    for result in completed:
        rejection_counts.update(result["rejection_reasons"])
        for name, counts in result["category_counts"].items():
            category_counts[name].update(counts)

    rejections_path = output / "rejections.jsonl"
    with rejections_path.open("w", encoding="utf-8", newline="\n") as combined:
        for day_range in day_ranges:
            daily = rejections_root / f"dataset_day={day_range.day:02d}.jsonl"
            combined.write(daily.read_text(encoding="utf-8"))

    input_rows = sum(int(item["input_rows"]) for item in completed)
    accepted_rows = sum(int(item["accepted_rows"]) for item in completed)
    rejected_rows = sum(int(item["rejected_rows"]) for item in completed)
    parts = [part for result in completed for part in result["parts"]]
    parts.sort(key=lambda part: (int(part["dataset_day"]), int(part["part"])))
    summary: dict[str, object] = {
        "format_version": 1,
        "dataset": "LANL authentication",
        "source_file": str(source),
        "source_reference_format": "auth.txt:<source_line>",
        "acting_user_definition": "source_user",
        "timestamp_definition": "positive dataset-relative seconds",
        "dataset_day_formula": "((timestamp - 1) // 86400) + 1",
        "selection": {"day_start_inclusive": config.day_start, "day_end_inclusive": config.day_end},
        "storage": {
            "format": "parquet",
            "compression": config.compression,
            "partitioning": "dataset_day",
            "part_order": "six-digit sequence in source order within each day",
        },
        "chunk_rows": config.chunk_rows,
        "parallel_workers": workers,
        "boundary_source": str(inspection),
        "counts": {
            "input_rows_scanned": input_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "out_of_scope_rows": 0,
            "reconciled": input_rows == accepted_rows + rejected_rows == expected_rows,
            "exact_duplicate_instances_beyond_first": sum(
                int(item["duplicate_instances"]) for item in completed
            ),
            "distinct_exact_duplicate_groups": sum(
                int(item["duplicate_groups"]) for item in completed
            ),
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "rows_per_day": {str(item["day"]): int(item["accepted_rows"]) for item in completed},
        "category_counts": {
            name: dict(sorted(counts.items())) for name, counts in category_counts.items()
        },
        "ordering": {
            "validated_non_decreasing_within_every_day": True,
            "day_ranges_are_non_overlapping_and_ordered": True,
        },
        "deduplication": "none; every accepted source row is retained",
        "labels_in_detector_events": False,
        "parts": parts,
        "schema": _dataset_schema(EVENT_SCHEMA),
    }
    _write_json(output / "summary.json", summary)
    publish("complete")
    print(f"[complete] authentication ingestion wrote {accepted_rows:,} rows", flush=True)
    return summary


def ingest_redteam_labels(source: Path, output: Path, config: IngestionConfig) -> dict[str, object]:
    """Normalise red-team labels into a physically separate Parquet tree."""

    _prepare_output(output, config.overwrite)
    labels_root = output / "labels"
    labels_root.mkdir()
    rejections_path = output / "rejections.jsonl"
    parts: list[dict[str, object]] = []
    part_counters: defaultdict[int, int] = defaultdict(int)
    day_counts: Counter[int] = Counter()
    rejection_counts: Counter[str] = Counter()
    input_rows = accepted_rows = out_of_scope_rows = 0
    previous_timestamp: int | None = None

    with source.open("r", encoding="utf-8", newline="") as handle, rejections_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejected:
        stop = False
        for chunk in _chunks(handle, config.chunk_rows):
            normalized: list[dict[str, object]] = []
            for line_number, raw in chunk:
                if config.max_source_rows is not None and input_rows >= config.max_source_rows:
                    stop = True
                    break
                input_rows += 1
                fields = next(csv.reader([raw]))
                if len(fields) != len(LABEL_FIELD_NAMES):
                    rejection_counts["field_count"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="field_count")
                    continue
                try:
                    timestamp = int(fields[0])
                    day = dataset_day(timestamp)
                except ValueError:
                    rejection_counts["invalid_timestamp"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="invalid_timestamp")
                    continue
                if any(value == "" for value in fields[1:]):
                    rejection_counts["empty_field"] += 1
                    _write_rejection(rejected, line=line_number, raw=raw, reason="empty_field")
                    continue
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(f"timestamp order inversion at {source.name}:{line_number}")
                previous_timestamp = timestamp
                if not config.day_start <= day <= config.day_end:
                    out_of_scope_rows += 1
                    continue
                row = dict(zip(LABEL_FIELD_NAMES, fields, strict=True))
                row["timestamp"] = timestamp
                row["dataset_day"] = day
                row["source_line"] = line_number
                row["source_reference"] = f"redteam.txt:{line_number}"
                row["raw_record"] = raw
                normalized.append(row)
                accepted_rows += 1
                day_counts[day] += 1
            if normalized:
                parts.extend(
                    _write_partitions(
                        normalized, labels_root, LABEL_SCHEMA, part_counters, config.compression
                    )
                )
            if stop:
                break

    rejected_rows = sum(rejection_counts.values())
    summary: dict[str, object] = {
        "format_version": 1,
        "dataset": "LANL red-team labels",
        "source_file": str(source),
        "source_reference_format": "redteam.txt:<source_line>",
        "separate_from_detector_inputs": True,
        "storage": {
            "format": "parquet",
            "compression": config.compression,
            "partitioning": "dataset_day",
        },
        "counts": {
            "input_rows_scanned": input_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "out_of_scope_rows": out_of_scope_rows,
            "reconciled": input_rows == accepted_rows + rejected_rows + out_of_scope_rows,
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "rows_per_day": {str(k): v for k, v in sorted(day_counts.items())},
        "parts": parts,
        "schema": _dataset_schema(LABEL_SCHEMA),
    }
    _write_json(output / "summary.json", summary)
    return summary
