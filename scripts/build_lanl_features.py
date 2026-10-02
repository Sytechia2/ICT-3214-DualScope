"""CLI script to build LANL historical features using an explicit two-pass pipeline.

Pass 1: Streams normalized events, generates raw causal features, and accumulates
        training-only statistics exclusively on eligible training rows.
Freeze: Finalizes standardization parameters and categorical vocabularies.
Pass 2: Transforms saved raw features into model-ready arrays with frozen statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Add src to sys.path if not present
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dualscope.features.config import FeatureConfig
from dualscope.features.engine import HistoricalFeatureEngine
from dualscope.features.preprocessing import FeaturePreprocessor
from dualscope.features.schemas import (
    RAW_FEATURE_SCHEMA,
    TRANSFORMED_FEATURE_SCHEMA,
)
from dualscope.splits import (
    SplitConfig,
    is_fitting_eligible,
)


class PartitionedParquetWriter:
    """Writes RecordBatches to day-partitioned directories: dataset_day=NN/part-XXXXXX.parquet."""

    def __init__(self, base_dir: Path, schema: pa.Schema, compression: str = "zstd") -> None:
        self.base_dir = base_dir
        self.schema = schema
        self.compression = compression
        self.day_writers: dict[int, pq.ParquetWriter] = {}
        self.day_part_counts: dict[int, int] = {}
        self.total_rows_written = 0

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if len(batch) == 0:
            return

        # Find unique days in batch
        day_col = batch["dataset_day"]
        unique_days = pc.unique(day_col).to_pylist()

        for day in unique_days:
            mask = pc.equal(day_col, day)
            day_batch = pc.filter(batch, mask)
            day_int = int(day)

            writer = self.day_writers.get(day_int)
            if writer is None:
                day_dir = self.base_dir / f"dataset_day={day_int:02d}"
                day_dir.mkdir(parents=True, exist_ok=True)
                part_idx = self.day_part_counts.get(day_int, 0)
                out_path = day_dir / f"part-{part_idx:06d}.parquet"
                writer = pq.ParquetWriter(out_path, self.schema, compression=self.compression)
                self.day_writers[day_int] = writer
                self.day_part_counts[day_int] = part_idx + 1

            writer.write_batch(day_batch)
            self.total_rows_written += len(day_batch)

    def close(self) -> None:
        for writer in self.day_writers.values():
            writer.close()
        self.day_writers.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DualScope Task 2.4: Build shared historical features with two-pass training preprocessing."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/authentication/events",
        help="Path to day-partitioned input authentication events.",
    )
    parser.add_argument(
        "--splits-config",
        type=Path,
        default=REPO_ROOT / "config/lanl_splits.json",
        help="Path to split policy configuration JSON.",
    )
    parser.add_argument(
        "--splits-manifest",
        type=Path,
        default=REPO_ROOT / "data/manifests/lanl_splits_v1.json",
        help="Path to splits manifest JSON (used to load excluded training users).",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=REPO_ROOT / "config/lanl_features.json",
        help="Path to feature configuration JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory to write day-partitioned Parquet files.",
    )
    parser.add_argument(
        "--preprocessing-path",
        type=Path,
        default=None,
        help="Path to save or load fitted FeaturePreprocessor JSON. Defaults to <output>/preprocessing.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="Batch size for event streaming and processing.",
    )
    parser.add_argument(
        "--pilot-rows",
        type=int,
        default=None,
        help="Optional row limit for pilot runs.",
    )
    parser.add_argument(
        "--pilot-days",
        type=int,
        default=None,
        help="Optional max dataset day for pilot runs.",
    )
    parser.add_argument(
        "--pilot-mode",
        action="store_true",
        help="Flag indicating this is a pilot or sample run (marks preprocessing as non-production).",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Execute Pass 1 only (compute raw features and fit preprocessing).",
    )
    parser.add_argument(
        "--transform-only",
        action="store_true",
        help="Execute Pass 2 only (transform raw features using existing preprocessing artifact).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel build with this many user shards (opt-in; 1 = the sequential build). "
             "Rows within a day are then grouped by shard rather than in input order.",
    )
    parser.add_argument(
        "--assembly-workers",
        type=int,
        default=8,
        help="Days assembled and transformed concurrently in a parallel build.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    return parser.parse_args()


def load_excluded_users(
    manifest_path: Path | None,
    splits_cfg: SplitConfig,
    is_production: bool = True,
) -> list[str]:
    """Load excluded training users from split manifest, strictly verifying split policy fingerprint.

    Raises:
        FileNotFoundError: If manifest does not exist in production mode, or if manifest_path is explicitly given but missing.
        ValueError: If manifest config fingerprint does not match the active SplitConfig fingerprint,
                    or if training_exclusions is missing from the manifest.
    """
    if manifest_path is None or not manifest_path.exists():
        if manifest_path is not None and not manifest_path.exists():
            raise FileNotFoundError(f"Specified splits manifest does not exist: {manifest_path}")
        if is_production:
            raise ValueError(
                "Production feature building requires a valid splits manifest matching the active split policy. "
                "Specify --splits-manifest <path_to_manifest.json>."
            )
        return []

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Fingerprint verification
    manifest_fp = manifest_data.get("config_fingerprint")
    expected_fp = splits_cfg.fingerprint()
    if manifest_fp != expected_fp:
        raise ValueError(
            f"Splits manifest config_fingerprint mismatch: "
            f"manifest has '{manifest_fp}', but active splits config has '{expected_fp}'."
        )

    # 2. Extract excluded users
    if "training_exclusions" not in manifest_data:
        raise ValueError(f"Splits manifest '{manifest_path}' is missing 'training_exclusions' section.")

    exclusions_section = manifest_data["training_exclusions"]
    if "excluded_users" not in exclusions_section:
        raise ValueError(f"Splits manifest 'training_exclusions' section is missing 'excluded_users' field.")

    excluded = exclusions_section["excluded_users"]
    if not isinstance(excluded, list):
        raise ValueError(f"'excluded_users' in splits manifest must be a list, got {type(excluded).__name__}.")

    return sorted(str(u) for u in excluded)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    start_wall_time = time.time()

    # 1. Validate inputs and directories
    if not args.events.exists():
        raise FileNotFoundError(f"Input events directory does not exist: {args.events}")

    out_dir = args.output
    if out_dir.exists() and not args.transform_only:
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"Output directory '{out_dir}' already exists. Use --overwrite to replace it."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_file = args.preprocessing_path or (out_dir / "preprocessing.json")

    # 2. Load configurations
    feature_cfg = FeatureConfig.from_file(args.feature_config)
    splits_cfg = SplitConfig.from_file(args.splits_config)
    train_split = splits_cfg.get_split("train")
    is_production = not args.pilot_mode and args.pilot_rows is None and args.pilot_days is None
    excluded_users = load_excluded_users(args.splits_manifest, splits_cfg, is_production=is_production)

    raw_out_dir = out_dir / "raw" / "events"
    transformed_out_dir = out_dir / "transformed" / "events"

    preprocessor: FeaturePreprocessor | None = None
    pass1_duration = 0.0
    pass2_duration = 0.0
    pass1_events = 0
    pass2_events = 0
    mem_stats: dict[str, Any] = {}
    parallel_info: dict[str, Any] | None = None

    workers = getattr(args, "workers", 1)
    if workers > 1:
        if args.raw_only or args.transform_only or args.pilot_rows is not None:
            raise ValueError("--workers > 1 supports full builds only (no --raw-only, --transform-only or --pilot-rows)")
        from dualscope.features.parallel import build_features_parallel

        result = build_features_parallel(
            events_dir=args.events,
            out_dir=out_dir,
            feature_config_path=args.feature_config,
            splits_config_path=args.splits_config,
            excluded_users=excluded_users,
            preprocessing_path=preprocessing_file,
            workers=workers,
            max_day=args.pilot_days,
            batch_size=args.batch_size,
            assembly_workers=getattr(args, "assembly_workers", 8),
            mode="pilot" if args.pilot_mode else "production",
            is_production=not args.pilot_mode,
        )
        preprocessor = result["preprocessor"]
        pass1_events, pass2_events = result["pass1_events"], result["pass2_events"]
        pass1_duration, pass2_duration = result["pass1_seconds"], result["pass2_seconds"]
        mem_stats = result["memory"]
        parallel_info = result["parallel"]

    # =========================================================================
    # PASS 1: Generate Raw Features and Accumulate Training Statistics
    # =========================================================================
    if not args.transform_only and parallel_info is None:
        p1_start = time.time()
        print(f"[Pass 1] Generating raw features and accumulating training statistics from {args.events}...")
        raw_out_dir.mkdir(parents=True, exist_ok=True)

        engine = HistoricalFeatureEngine(feature_cfg)
        preprocessor = FeaturePreprocessor(
            config=feature_cfg,
            split_policy_fingerprint=splits_cfg.fingerprint(),
            mode="pilot" if args.pilot_mode else "production",
            is_production=not args.pilot_mode,
        )

        dataset = ds.dataset(str(args.events), format="parquet", partitioning="hive")

        # Build scan filter
        filter_expr = ds.field("timestamp") >= 1
        if args.pilot_days is not None:
            filter_expr = filter_expr & (ds.field("dataset_day") <= args.pilot_days)

        scanner = dataset.scanner(filter=filter_expr, batch_size=args.batch_size)
        raw_writer = PartitionedParquetWriter(raw_out_dir, RAW_FEATURE_SCHEMA)

        try:
            for batch in scanner.to_batches():
                if len(batch) == 0:
                    continue

                if args.pilot_rows is not None and pass1_events + len(batch) > args.pilot_rows:
                    batch = batch.slice(0, args.pilot_rows - pass1_events)
                    if len(batch) == 0:
                        break

                n_b = len(batch)
                pass1_events += n_b
                preprocessor.total_scanned_rows += n_b

                # Compute causal raw features
                raw_batch = engine.process_batch(batch)

                # Determine split membership and fitting eligibility
                ts_col = raw_batch["timestamp"]
                in_train = pc.and_(
                    pc.greater_equal(ts_col, train_split.timestamp_start),
                    pc.less(ts_col, train_split.timestamp_end),
                )
                in_val = pc.and_(
                    pc.greater_equal(ts_col, splits_cfg.get_split("validation").timestamp_start),
                    pc.less(ts_col, splits_cfg.get_split("validation").timestamp_end),
                )
                in_test = pc.and_(
                    pc.greater_equal(ts_col, splits_cfg.get_split("test").timestamp_start),
                    pc.less(ts_col, splits_cfg.get_split("test").timestamp_end),
                )

                train_rows = pc.sum(pc.cast(in_train, pa.int64())).as_py() or 0
                val_rows = pc.sum(pc.cast(in_val, pa.int64())).as_py() or 0
                test_rows = pc.sum(pc.cast(in_test, pa.int64())).as_py() or 0

                preprocessor.training_scanned_rows += train_rows
                preprocessor.validation_scanned_rows += val_rows
                preprocessor.test_scanned_rows += test_rows

                # Training eligibility filter: in_train AND neither source nor destination user is excluded
                if train_rows > 0:
                    user_eligible = is_fitting_eligible(raw_batch, excluded_users)
                    train_eligible = pc.and_(in_train, user_eligible)
                    eligible_cnt = pc.sum(pc.cast(train_eligible, pa.int64())).as_py() or 0
                    preprocessor.training_excluded_skipped_rows += (train_rows - eligible_cnt)

                    # Accumulate training statistics
                    preprocessor.accumulate_training_batch(raw_batch, train_eligible)

                # Write raw feature batch to partitioned Parquet
                raw_writer.write_batch(raw_batch)

                if pass1_events % 200_000 == 0:
                    print(f"  Processed {pass1_events:,} events...")

                if args.pilot_rows is not None and pass1_events >= args.pilot_rows:
                    break
        finally:
            raw_writer.close()

        # Freeze preprocessor after Pass 1
        preprocessor.freeze()
        preprocessor.save(preprocessing_file)
        mem_stats = engine.get_memory_breakdown()
        pass1_duration = time.time() - p1_start
        print(f"[Pass 1] Completed {pass1_events:,} raw events in {pass1_duration:.2f}s "
              f"({pass1_events / max(pass1_duration, 0.001):.0f} events/s). Preprocessing frozen and saved.")

    # =========================================================================
    # PASS 2: Frozen Transformation
    # =========================================================================
    if not args.raw_only and parallel_info is None:
        p2_start = time.time()
        print(f"[Pass 2] Transforming raw features using frozen preprocessor from {preprocessing_file}...")
        transformed_out_dir.mkdir(parents=True, exist_ok=True)

        if preprocessor is None:
            preprocessor = FeaturePreprocessor.load(
                preprocessing_file,
                config=feature_cfg,
                splits_cfg=splits_cfg,
            )
        else:
            preprocessor.verify_compatibility(feature_cfg=feature_cfg, splits_cfg=splits_cfg)

        raw_dataset = ds.dataset(str(raw_out_dir), format="parquet", partitioning="hive")
        transformed_writer = PartitionedParquetWriter(transformed_out_dir, TRANSFORMED_FEATURE_SCHEMA)

        try:
            for raw_batch in raw_dataset.to_batches():
                if len(raw_batch) == 0:
                    continue

                transformed_batch = preprocessor.transform_batch(raw_batch)
                transformed_writer.write_batch(transformed_batch)
                pass2_events += len(transformed_batch)

                if pass2_events % 200_000 == 0:
                    print(f"  Transformed {pass2_events:,} events...")
        finally:
            transformed_writer.close()

        pass2_duration = time.time() - p2_start
        print(f"[Pass 2] Completed {pass2_events:,} transformed events in {pass2_duration:.2f}s "
              f"({pass2_events / max(pass2_duration, 0.001):.0f} events/s).")

    total_wall_time = time.time() - start_wall_time

    # 4. Count Reconciliation
    reconciled = (pass1_events == pass2_events) if (not args.raw_only and not args.transform_only) else True

    summary: dict[str, Any] = {
        "task": "2.4 — Shared historical features",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_version": feature_cfg.feature_version,
        "feature_config_fingerprint": feature_cfg.fingerprint(),
        "split_policy_fingerprint": splits_cfg.fingerprint(),
        "mode": "pilot" if args.pilot_mode else "production",
        "is_production": not args.pilot_mode,
        "inputs": {
            "events_path": str(args.events),
            "splits_config_path": str(args.splits_config),
            "splits_manifest_path": str(args.splits_manifest),
            "feature_config_path": str(args.feature_config),
            "excluded_users": excluded_users,
        },
        "outputs": {
            "output_directory": str(out_dir),
            "raw_events_directory": str(raw_out_dir) if not args.transform_only else None,
            "transformed_events_directory": str(transformed_out_dir) if not args.raw_only else None,
            "preprocessing_path": str(preprocessing_file),
        },
        "counts": {
            "raw_events_written": pass1_events,
            "transformed_events_written": pass2_events,
            "reconciled": reconciled,
            "fitting_row_counts": preprocessor.to_dict()["fitting_row_counts"] if preprocessor else {},
        },
        "performance": {
            "total_wall_time_seconds": round(total_wall_time, 2),
            "pass1_seconds": round(pass1_duration, 2),
            "pass2_seconds": round(pass2_duration, 2),
            "pass1_throughput_events_per_sec": round(pass1_events / max(pass1_duration, 0.001), 1),
            "pass2_throughput_events_per_sec": round(pass2_events / max(pass2_duration, 0.001), 1),
            "memory_breakdown": mem_stats,
        },
    }
    if parallel_info is not None:
        summary["parallel_build"] = parallel_info
    if args.pilot_days is not None:
        summary["inputs"]["max_dataset_day"] = args.pilot_days

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pipeline completed successfully. Summary saved to {summary_file}")
    return summary


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"Error during feature pipeline execution: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
