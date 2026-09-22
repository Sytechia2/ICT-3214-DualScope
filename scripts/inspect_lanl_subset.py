"""Count records and identities in a bounded LANL authentication selection.

This Task 2.1 utility reads independent byte ranges in parallel. It only
reports aggregate metadata; it does not write, normalize, or label events.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int


def _ranges(byte_limit: int, workers: int) -> list[ByteRange]:
    return [
        ByteRange(byte_limit * index // workers, byte_limit * (index + 1) // workers)
        for index in range(workers)
    ]


def _inspect_range(
    path_text: str,
    byte_range: ByteRange,
    start_timestamp: int,
    end_timestamp_exclusive: int,
) -> dict[str, object]:
    source_users: set[bytes] = set()
    destination_users: set[bytes] = set()
    source_computers: set[bytes] = set()
    destination_computers: set[bytes] = set()
    selected_rows = 0
    malformed_rows = 0
    outside_time_range_rows = 0

    with Path(path_text).open("rb") as handle:
        if byte_range.start:
            handle.seek(byte_range.start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)

        while handle.tell() < byte_range.end:
            line = handle.readline()
            if not line:
                break
            fields = line.rstrip(b"\r\n").split(b",")
            if len(fields) != 9:
                malformed_rows += 1
                continue
            try:
                timestamp = int(fields[0])
            except ValueError:
                malformed_rows += 1
                continue
            if not start_timestamp <= timestamp < end_timestamp_exclusive:
                outside_time_range_rows += 1
                continue

            selected_rows += 1
            source_users.add(fields[1])
            destination_users.add(fields[2])
            source_computers.add(fields[3])
            destination_computers.add(fields[4])

    return {
        "selected_rows": selected_rows,
        "malformed_rows": malformed_rows,
        "outside_time_range_rows": outside_time_range_rows,
        "source_users": source_users,
        "destination_users": destination_users,
        "source_computers": source_computers,
        "destination_computers": destination_computers,
    }


def inspect(
    path: Path,
    byte_limit: int,
    start_timestamp: int,
    end_timestamp_exclusive: int,
    workers: int,
) -> dict[str, object]:
    started = perf_counter()
    byte_ranges = _ranges(byte_limit, workers)
    arguments = [
        (str(path), item, start_timestamp, end_timestamp_exclusive)
        for item in byte_ranges
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_inspect_range_from_tuple, arguments))

    identity_fields = (
        "source_users",
        "destination_users",
        "source_computers",
        "destination_computers",
    )
    identities = {
        field: set().union(*(result[field] for result in results))
        for field in identity_fields
    }
    return {
        "source_file_name": path.name,
        "source_file_size_bytes": path.stat().st_size,
        "inspected_byte_limit_exclusive": byte_limit,
        "selection": {
            "start_timestamp_inclusive": start_timestamp,
            "end_timestamp_exclusive": end_timestamp_exclusive,
        },
        "selected_rows": sum(int(result["selected_rows"]) for result in results),
        "malformed_rows": sum(int(result["malformed_rows"]) for result in results),
        "outside_time_range_rows": sum(
            int(result["outside_time_range_rows"]) for result in results
        ),
        "distinct_source_users": len(identities["source_users"]),
        "distinct_destination_users": len(identities["destination_users"]),
        "distinct_source_computers": len(identities["source_computers"]),
        "distinct_destination_computers": len(identities["destination_computers"]),
        "workers": workers,
        "runtime_seconds": round(perf_counter() - started, 3),
    }


def _inspect_range_from_tuple(arguments: tuple[str, ByteRange, int, int]) -> dict[str, object]:
    return _inspect_range(*arguments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--byte-limit", type=int, required=True)
    parser.add_argument("--start-timestamp", type=int, default=1)
    parser.add_argument("--end-timestamp-exclusive", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.auth.is_file():
        raise SystemExit(f"Authentication source not found: {args.auth}")
    if not 0 < args.byte_limit <= args.auth.stat().st_size:
        raise SystemExit("--byte-limit must be within the authentication file")
    if args.start_timestamp >= args.end_timestamp_exclusive:
        raise SystemExit("timestamp range must be non-empty")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    result = inspect(
        path=args.auth,
        byte_limit=args.byte_limit,
        start_timestamp=args.start_timestamp,
        end_timestamp_exclusive=args.end_timestamp_exclusive,
        workers=args.workers,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
