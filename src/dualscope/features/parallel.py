"""Parallel two-pass Task 2.4 feature build (opt-in alternative to the sequential build).

Every per-user feature depends only on that acting user's own earlier events,
so users are split into shards by a stable hash and each shard runs its own
``HistoricalFeatureEngine`` over its users' events in the original
``(timestamp, source_line)`` order. The one cross-user feature,
``is_new_host_connection`` (an ever-seen (source, destination) computer pair),
is computed separately over all events in order and replaces the shard value.

Steps:

1. Pass 1 (in parallel): one task per user shard writes raw features and
   returns training statistics; one extra task computes host-connection
   novelty for every event.
2. Training statistics are merged (Chan merge for numeric moments, union of
   categorical candidates) and the preprocessor is frozen and saved.
3. Per day (in parallel): each shard's rows get their host-connection
   novelty by source line and are written to ``raw/events`` and transformed to
   ``transformed/events``, one part file per shard.

The output contains the same rows and values as the sequential build, but
within a day the rows are grouped by shard (each group in input order), so
readers must sort by ``(timestamp, source_line)`` when order matters (the
sequence builder already does). Numeric scaling statistics can differ in the
last floating-point digits because moments are merged in a different order.
"""

from __future__ import annotations

import multiprocessing
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dualscope.features.config import FeatureConfig
from dualscope.features.engine import HistoricalFeatureEngine
from dualscope.features.preprocessing import CATEGORICAL_COLUMNS, FeaturePreprocessor
from dualscope.features.schemas import RAW_FEATURE_SCHEMA, TRANSFORMED_FEATURE_SCHEMA
from dualscope.splits import SplitConfig, is_fitting_eligible


INPUT_COLUMNS = (
    "timestamp", "source_user", "destination_user", "source_computer", "destination_computer",
    "authentication_type", "logon_type", "authentication_orientation", "authentication_result",
    "acting_user", "exact_duplicate_ordinal", "source_line", "source_reference", "dataset_day",
)


def user_shards(users: pa.Array | pa.ChunkedArray, n_shards: int) -> np.ndarray:
    """Stable shard number per row: CRC-32 of the acting user, modulo ``n_shards``."""
    encoded = pc.dictionary_encode(users)
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    dictionary = encoded.dictionary.to_pylist()
    shard_of_value = np.fromiter(
        (zlib.crc32(value.encode("utf-8")) % n_shards for value in dictionary), dtype=np.int64, count=len(dictionary)
    )
    return shard_of_value[encoded.indices.to_numpy(zero_copy_only=False)]


def _scanner(events_dir: str, max_day: int | None, batch_size: int, columns: list[str]) -> ds.Scanner:
    dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
    filter_expr = ds.field("timestamp") >= 1
    if max_day is not None:
        filter_expr = filter_expr & (ds.field("dataset_day") <= max_day)
    return dataset.scanner(filter=filter_expr, batch_size=batch_size, columns=columns)


class _DayWriter:
    """One Parquet file per dataset day, opened lazily."""

    def __init__(self, base_dir: Path, schema: pa.Schema, name: str = "part-000000.parquet") -> None:
        self.base_dir = base_dir
        self.schema = schema
        self.name = name
        self.writers: dict[int, pq.ParquetWriter] = {}

    def write(self, batch: pa.RecordBatch | pa.Table) -> None:
        days = batch["dataset_day"]
        for day in pc.unique(days).to_pylist():
            part = batch.filter(pc.equal(days, day))
            writer = self.writers.get(int(day))
            if writer is None:
                directory = self.base_dir / f"dataset_day={int(day):02d}"
                directory.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(directory / self.name, self.schema, compression="zstd")
                self.writers[int(day)] = writer
            writer.write(part)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def account_pass1_batch(
    preprocessor: FeaturePreprocessor,
    raw_batch: pa.RecordBatch,
    splits_cfg: SplitConfig,
    excluded_users: list[str],
) -> None:
    """Split counts and training-only statistics for one raw batch (same rules as the sequential build)."""
    ts_col = raw_batch["timestamp"]
    counts = {}
    for name in ("train", "validation", "test"):
        split = splits_cfg.get_split(name)
        inside = pc.and_(pc.greater_equal(ts_col, split.timestamp_start), pc.less(ts_col, split.timestamp_end))
        counts[name] = (inside, pc.sum(pc.cast(inside, pa.int64())).as_py() or 0)
    preprocessor.total_scanned_rows += len(raw_batch)
    preprocessor.training_scanned_rows += counts["train"][1]
    preprocessor.validation_scanned_rows += counts["validation"][1]
    preprocessor.test_scanned_rows += counts["test"][1]
    in_train, train_rows = counts["train"]
    if train_rows > 0:
        train_eligible = pc.and_(in_train, is_fitting_eligible(raw_batch, excluded_users))
        eligible = pc.sum(pc.cast(train_eligible, pa.int64())).as_py() or 0
        preprocessor.training_excluded_skipped_rows += train_rows - eligible
        preprocessor.accumulate_training_batch(raw_batch, train_eligible)


def _pass1_shard(task: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    feature_cfg = FeatureConfig.from_file(task["feature_config"])
    splits_cfg = SplitConfig.from_file(task["splits_config"])
    engine = HistoricalFeatureEngine(feature_cfg)
    preprocessor = FeaturePreprocessor(
        config=feature_cfg, split_policy_fingerprint=splits_cfg.fingerprint(),
        mode=task["mode"], is_production=task["is_production"],
    )
    writer = _DayWriter(Path(task["shard_dir"]), RAW_FEATURE_SCHEMA)
    events = 0
    try:
        for batch in _scanner(task["events_dir"], task["max_day"], task["batch_size"], list(INPUT_COLUMNS)).to_batches():
            if len(batch) == 0:
                continue
            mine = user_shards(batch["acting_user"], task["n_shards"]) == task["shard"]
            if not mine.any():
                continue
            batch = batch.filter(pa.array(mine))
            raw_batch = engine.process_batch(batch)
            account_pass1_batch(preprocessor, raw_batch, splits_cfg, task["excluded_users"])
            writer.write(raw_batch)
            events += len(raw_batch)
    finally:
        writer.close()
    return {
        "shard": task["shard"],
        "events": events,
        "seconds": round(time.time() - started, 2),
        "numeric": {k: (s.count, s.mean, s.m2) for k, s in preprocessor.numeric_stats.items()},
        "categories": {k: sorted(v) for k, v in preprocessor._cat_candidates.items()},
        "counts": {
            name: getattr(preprocessor, name)
            for name in (
                "total_scanned_rows", "training_scanned_rows", "training_eligible_fitted_rows",
                "training_excluded_skipped_rows", "validation_scanned_rows", "test_scanned_rows",
            )
        },
        "memory": engine.get_memory_breakdown(),
    }


def _host_connection_novelty(task: dict[str, Any]) -> dict[str, Any]:
    """Ever-seen (source_computer, destination_computer) flags for every event, in input order."""
    started = time.time()
    out_dir = Path(task["out_dir"])
    computer_ids: dict[str, int] = {}
    seen = np.zeros(0, dtype=np.int64)
    last = (-1, -1)
    events = 0
    columns = ["timestamp", "source_line", "source_computer", "destination_computer", "dataset_day"]
    writer = _DayWriter(out_dir, pa.schema([("source_line", pa.int64()), ("is_new_host_connection", pa.bool_()), ("dataset_day", pa.int32())]))
    try:
        for batch in _scanner(task["events_dir"], task["max_day"], task["batch_size"], columns).to_batches():
            if len(batch) == 0:
                continue
            ts = batch["timestamp"].to_numpy()
            lines = batch["source_line"].to_numpy()
            # The engine's ordering contract, plus increasing source lines, which
            # lets day assembly look flags up by source line.
            prev_ts = np.r_[last[0], ts[:-1]]
            prev_lines = np.r_[last[1], lines[:-1]]
            if np.any(ts < prev_ts) or np.any(lines <= prev_lines):
                raise ValueError("input events are not in timestamp and increasing source_line order")
            last = (int(ts[-1]), int(lines[-1]))

            ids = []
            for column in ("source_computer", "destination_computer"):
                encoded = pc.dictionary_encode(batch[column])
                mapped = np.fromiter(
                    (computer_ids.setdefault(value, len(computer_ids)) for value in encoded.dictionary.to_pylist()),
                    dtype=np.int64,
                )
                ids.append(mapped[encoded.indices.to_numpy(zero_copy_only=False)])
            keys = (ids[0] << 32) | ids[1]
            unique, first = np.unique(keys, return_index=True)
            position = np.searchsorted(seen, unique)
            known = np.zeros(len(unique), dtype=bool)
            inside = position < len(seen)
            known[inside] = seen[position[inside]] == unique[inside]
            flags = np.zeros(len(keys), dtype=bool)
            flags[first[~known]] = True
            seen = np.insert(seen, position[~known], unique[~known])
            writer.write(pa.RecordBatch.from_arrays(
                [batch["source_line"], pa.array(flags), batch["dataset_day"]],
                names=["source_line", "is_new_host_connection", "dataset_day"],
            ))
            events += len(batch)
    finally:
        writer.close()
    return {"events": events, "distinct_host_connections": int(len(seen)), "seconds": round(time.time() - started, 2)}


def _assemble_day(task: dict[str, Any]) -> dict[str, Any]:
    """Join host-connection novelty onto one day's shard rows; write raw and transformed part files."""
    started = time.time()
    day = task["day"]
    day_dir = f"dataset_day={day:02d}"
    novelty = pq.ParquetFile(Path(task["host_dir"]) / day_dir / "part-000000.parquet").read()
    novelty_lines = novelty["source_line"].to_numpy()
    novelty_flags = novelty["is_new_host_connection"].to_numpy(zero_copy_only=False)
    column = RAW_FEATURE_SCHEMA.get_field_index("is_new_host_connection")

    feature_cfg = FeatureConfig.from_file(task["feature_config"])
    preprocessor = FeaturePreprocessor.load(task["preprocessing_path"], config=feature_cfg)
    raw_dir = Path(task["raw_dir"]) / day_dir
    transformed_dir = Path(task["transformed_dir"]) / day_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    transformed_dir.mkdir(parents=True, exist_ok=True)

    events = 0
    for shard, shard_dir in enumerate(task["shard_dirs"]):
        path = Path(shard_dir) / day_dir / "part-000000.parquet"
        if not path.exists():
            continue
        table = pq.ParquetFile(path).read()
        if not table.schema.equals(RAW_FEATURE_SCHEMA):
            raise ValueError(f"unexpected raw schema in {path}")
        lines = table["source_line"].to_numpy()
        position = np.searchsorted(novelty_lines, lines)
        if np.any(position >= len(novelty_lines)) or np.any(novelty_lines[np.minimum(position, len(novelty_lines) - 1)] != lines):
            raise ValueError(f"day {day}, shard {shard}: source lines missing from the host-connection pass")
        table = table.set_column(column, RAW_FEATURE_SCHEMA.field(column), pa.array(novelty_flags[position]))
        name = f"part-{shard:06d}.parquet"
        pq.write_table(table, raw_dir / name, compression="zstd", row_group_size=task["batch_size"])
        with pq.ParquetWriter(transformed_dir / name, TRANSFORMED_FEATURE_SCHEMA, compression="zstd") as writer:
            for batch in table.to_batches(max_chunksize=task["batch_size"]):
                writer.write_batch(preprocessor.transform_batch(batch))
        events += table.num_rows
    if events != len(novelty_lines):
        raise ValueError(f"day {day}: shards hold {events} events but the input has {len(novelty_lines)}")
    return {"day": day, "events": events, "seconds": round(time.time() - started, 2)}


def merge_shard_statistics(preprocessor: FeaturePreprocessor, results: list[dict[str, Any]]) -> None:
    """Combine shard statistics in shard order (Chan merge) and union category candidates."""
    for result in sorted(results, key=lambda r: r["shard"]):
        for name, (count, mean, m2) in result["numeric"].items():
            preprocessor.numeric_stats[name].combine_batch(count, mean, m2)
        for column in CATEGORICAL_COLUMNS:
            preprocessor._cat_candidates[column].update(result["categories"][column])
        for name, value in result["counts"].items():
            setattr(preprocessor, name, getattr(preprocessor, name) + value)


def build_features_parallel(
    events_dir: Path,
    out_dir: Path,
    feature_config_path: Path,
    splits_config_path: Path,
    excluded_users: list[str],
    preprocessing_path: Path,
    workers: int,
    max_day: int | None = None,
    batch_size: int = 200_000,
    assembly_workers: int = 8,
    mode: str = "production",
    is_production: bool = True,
    log=print,
) -> dict[str, Any]:
    """Run both passes in parallel; return counts, timings and memory for the build summary."""
    feature_cfg = FeatureConfig.from_file(feature_config_path)
    splits_cfg = SplitConfig.from_file(splits_config_path)
    work_dir = out_dir / "_parallel_work"
    shard_dirs = [work_dir / f"shard={i:02d}" for i in range(workers)]
    host_dir = work_dir / "host_connections"
    common = {
        "events_dir": str(events_dir), "max_day": max_day, "batch_size": batch_size,
        "feature_config": str(feature_config_path), "splits_config": str(splits_config_path),
    }
    context = multiprocessing.get_context("spawn")

    p1_start = time.time()
    log(f"[Pass 1] {workers} user shards + host-connection novelty from {events_dir}...")
    with ProcessPoolExecutor(max_workers=workers + 1, mp_context=context) as pool:
        host_future = pool.submit(_host_connection_novelty, {**common, "out_dir": str(host_dir)})
        shard_futures = [
            pool.submit(_pass1_shard, {
                **common, "shard": i, "n_shards": workers, "shard_dir": str(shard_dirs[i]),
                "excluded_users": excluded_users, "mode": mode, "is_production": is_production,
            })
            for i in range(workers)
        ]
        shard_results = [f.result() for f in shard_futures]
        host_result = host_future.result()
    pass1_events = sum(r["events"] for r in shard_results)
    if pass1_events != host_result["events"]:
        raise ValueError(f"shards processed {pass1_events} events but the input has {host_result['events']}")

    preprocessor = FeaturePreprocessor(
        config=feature_cfg, split_policy_fingerprint=splits_cfg.fingerprint(), mode=mode, is_production=is_production,
    )
    merge_shard_statistics(preprocessor, shard_results)
    preprocessor.freeze()
    preprocessor.save(preprocessing_path)
    pass1_seconds = time.time() - p1_start
    log(f"[Pass 1] {pass1_events:,} events in {pass1_seconds:.1f}s; preprocessing frozen and saved.")

    p2_start = time.time()
    days = sorted({int(p.name.split("=")[1]) for p in host_dir.glob("dataset_day=*")})
    log(f"[Pass 2] Assembling and transforming {len(days)} days with {assembly_workers} workers...")
    with ProcessPoolExecutor(max_workers=assembly_workers, mp_context=context) as pool:
        day_results = list(pool.map(_assemble_day, [
            {
                "day": day, "shard_dirs": [str(d) for d in shard_dirs], "host_dir": str(host_dir),
                "raw_dir": str(out_dir / "raw" / "events"), "transformed_dir": str(out_dir / "transformed" / "events"),
                "feature_config": str(feature_config_path), "preprocessing_path": str(preprocessing_path),
                "batch_size": batch_size,
            }
            for day in days
        ]))
    pass2_events = sum(r["events"] for r in day_results)
    pass2_seconds = time.time() - p2_start
    log(f"[Pass 2] {pass2_events:,} events in {pass2_seconds:.1f}s.")
    shutil.rmtree(work_dir)

    return {
        "preprocessor": preprocessor,
        "pass1_events": pass1_events,
        "pass2_events": pass2_events,
        "pass1_seconds": pass1_seconds,
        "pass2_seconds": pass2_seconds,
        "parallel": {
            "workers": workers,
            "assembly_workers": assembly_workers,
            "shard_events": [r["events"] for r in sorted(shard_results, key=lambda r: r["shard"])],
            "shard_seconds": [r["seconds"] for r in sorted(shard_results, key=lambda r: r["shard"])],
            "host_connection_pass": host_result,
            "days": day_results,
        },
        "memory": {f"shard_{r['shard']:02d}": r["memory"] for r in shard_results},
    }
