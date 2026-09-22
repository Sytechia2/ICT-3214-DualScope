"""Command-line entry point for LANL Task 2.2 ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dualscope.ingestion import (
    IngestionConfig,
    ingest_authentication,
    ingest_authentication_parallel,
    ingest_redteam_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", type=Path, help="Path to extracted auth.txt")
    parser.add_argument("--redteam", type=Path, help="Path to extracted redteam.txt")
    parser.add_argument("--output", type=Path, required=True, help="Root output directory")
    parser.add_argument("--day-start", type=int, default=1)
    parser.add_argument("--day-end", type=int, default=30)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Authentication worker processes")
    parser.add_argument(
        "--inspection",
        type=Path,
        default=Path("outputs/lanl_auth_inspection.json"),
        help="Task 2.1 artifact containing exact day byte/row boundaries",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.auth is None and args.redteam is None:
        parser.error("at least one of --auth or --redteam is required")
    return args


def main() -> int:
    args = parse_args()
    config = IngestionConfig(
        day_start=args.day_start,
        day_end=args.day_end,
        chunk_rows=args.chunk_rows,
        max_source_rows=args.max_source_rows,
        overwrite=args.overwrite,
    )
    summaries: dict[str, object] = {}
    if args.auth is not None:
        if args.workers > 1:
            summaries["authentication"] = ingest_authentication_parallel(
                args.auth,
                args.output / "authentication",
                config,
                args.inspection,
                args.workers,
            )
        else:
            summaries["authentication"] = ingest_authentication(
                args.auth, args.output / "authentication", config
            )
    if args.redteam is not None:
        summaries["redteam_labels"] = ingest_redteam_labels(
            args.redteam, args.output / "redteam_labels", config
        )
    print(json.dumps({name: summary["counts"] for name, summary in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
