"""The parallel Task 2.4 build produces the same features as the sequential build."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from dualscope.features.parallel import user_shards


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_script():
    spec = importlib.util.spec_from_file_location("build_lanl_features", REPO_ROOT / "scripts" / "build_lanl_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(events: Path, output: Path, workers: int, low_memory_read: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        events=events,
        splits_config=REPO_ROOT / "data/fixtures/fixture_splits.json",
        splits_manifest=REPO_ROOT / "data/manifests/fixture_splits_v1.json",
        feature_config=REPO_ROOT / "config/lanl_features_v2.json",
        output=output,
        preprocessing_path=None,
        batch_size=97,  # small, so the parallel build sees many batches per shard
        pilot_rows=None,
        pilot_days=3,
        pilot_mode=True,
        raw_only=False,
        transform_only=False,
        workers=workers,
        assembly_workers=2,
        low_memory_read=low_memory_read,
        overwrite=False,
    )


def _sorted_table(path: Path) -> pa.Table:
    table = ds.dataset(str(path), format="parquet", partitioning="hive").to_table()
    return table.sort_by([("timestamp", "ascending"), ("source_line", "ascending")])


def test_user_shards_are_stable_and_cover_every_shard() -> None:
    users = pa.array([f"U{i}@DOM1" for i in range(200)] + ["U1@DOM1"])
    shards = user_shards(users, 5)
    assert shards[1] == shards[-1]
    assert set(shards.tolist()) == {0, 1, 2, 3, 4}
    np.testing.assert_array_equal(shards, user_shards(pa.chunked_array([users[:50], users[50:]]), 5))


def test_parallel_build_matches_sequential(synthetic_sequences, tmp_path) -> None:
    script = _build_script()
    events = synthetic_sequences.auth_root / "events"
    sequential = script.run_pipeline(_args(events, tmp_path / "sequential", workers=1))
    parallel = script.run_pipeline(_args(events, tmp_path / "parallel", workers=3))

    assert sequential["counts"]["raw_events_written"] == parallel["counts"]["raw_events_written"] > 0
    assert parallel["counts"]["reconciled"]
    assert sequential["counts"]["fitting_row_counts"] == parallel["counts"]["fitting_row_counts"]

    for kind in ("raw", "transformed"):
        expected = _sorted_table(tmp_path / "sequential" / kind / "events")
        actual = _sorted_table(tmp_path / "parallel" / kind / "events")
        assert actual.schema == expected.schema
        assert actual.num_rows == expected.num_rows
        for name in expected.column_names:
            left, right = expected[name].to_numpy(zero_copy_only=False), actual[name].to_numpy(zero_copy_only=False)
            if np.issubdtype(left.dtype, np.floating):
                np.testing.assert_allclose(right, left, rtol=1e-6, atol=1e-6, err_msg=f"{kind}.{name}")
            else:
                np.testing.assert_array_equal(right, left, err_msg=f"{kind}.{name}")

    # The cross-user feature must use all users' history, not only the shard's.
    raw = _sorted_table(tmp_path / "parallel" / "raw" / "events")
    pairs = list(zip(raw["source_computer"].to_pylist(), raw["destination_computer"].to_pylist()))
    flagged = [pair for pair, new in zip(pairs, raw["is_new_host_connection"].to_pylist()) if new]
    assert sorted(flagged) == sorted(set(pairs))  # each pair is new exactly once, across all shards

    seq_pre = json.loads((tmp_path / "sequential" / "preprocessing.json").read_text(encoding="utf-8"))
    par_pre = json.loads((tmp_path / "parallel" / "preprocessing.json").read_text(encoding="utf-8"))
    assert seq_pre["categorical_vocabularies"] == par_pre["categorical_vocabularies"]
    for name, stat in seq_pre["numeric_stats"].items():
        assert par_pre["numeric_stats"][name]["count"] == stat["count"]
        for field in ("mean", "std", "scale"):
            assert par_pre["numeric_stats"][name][field] == pytest.approx(stat[field], rel=1e-9, abs=1e-12)
    assert not (tmp_path / "parallel" / "_parallel_work").exists()


def test_low_memory_read_gives_identical_output(synthetic_sequences, tmp_path) -> None:
    script = _build_script()
    events = synthetic_sequences.auth_root / "events"
    default = script.run_pipeline(_args(events, tmp_path / "default", workers=2))
    low = script.run_pipeline(_args(events, tmp_path / "low", workers=2, low_memory_read=True))

    assert default["counts"]["raw_events_written"] == low["counts"]["raw_events_written"] > 0
    for kind in ("raw", "transformed"):
        assert _sorted_table(tmp_path / "low" / kind / "events").equals(_sorted_table(tmp_path / "default" / kind / "events"))
    assert (tmp_path / "low" / "preprocessing.json").read_text(encoding="utf-8") == (
        tmp_path / "default" / "preprocessing.json"
    ).read_text(encoding="utf-8")
