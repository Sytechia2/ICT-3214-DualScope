"""Optional extra binary sequence inputs (v2 features) without changing v1 behaviour."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pytest

from dualscope.features.config import FeatureConfig
from dualscope.sequence.builder import load_day_sequences
from dualscope.sequence.config import SequenceDetectorConfig, SequencePolicy, input_feature_names
from dualscope.sequence.pipeline import SequenceInputs, prepare_samples


EXTRA = ("is_new_user_source", "is_machine_account")
V2_CONFIG = Path(__file__).resolve().parents[1] / "config" / "sequence_detector_v2.json"


def test_default_config_has_no_extra_inputs_key_and_round_trips() -> None:
    config = SequenceDetectorConfig()
    assert "extra_binary_inputs" not in config.to_dict()["policy"]
    assert SequenceDetectorConfig.from_dict(config.to_dict()).fingerprint() == config.fingerprint()


def test_extra_inputs_change_fingerprint_and_round_trip() -> None:
    base = SequenceDetectorConfig()
    extended = replace(base, policy=replace(base.policy, extra_binary_inputs=EXTRA))
    assert extended.fingerprint() != base.fingerprint()
    data = json.loads(json.dumps(extended.to_dict()))
    assert data["policy"]["extra_binary_inputs"] == list(EXTRA)
    assert SequenceDetectorConfig.from_dict(data).fingerprint() == extended.fingerprint()
    assert input_feature_names(extended.policy)[5:10] == list(extended.policy.binary_inputs)


def test_v2_config_matches_v1_except_inputs() -> None:
    v1 = SequenceDetectorConfig.from_file(V2_CONFIG.with_name("sequence_detector.json"))
    v2 = SequenceDetectorConfig.from_file(V2_CONFIG)
    assert v2.policy.extra_binary_inputs == EXTRA
    assert replace(v2, policy=replace(v2.policy, extra_binary_inputs=())).fingerprint() == v1.fingerprint()


def test_duplicate_extra_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        SequenceDetectorConfig(policy=SequencePolicy(extra_binary_inputs=("is_new_host_connection",)))


def test_extra_inputs_are_appended_after_default_binary_columns(synthetic_sequences) -> None:
    data = synthetic_sequences
    v1 = load_day_sequences(data.features_dir, 2, data.split_cfg, SequencePolicy(max_sequence_length=8))
    v2 = load_day_sequences(
        data.features_dir, 2, data.split_cfg, SequencePolicy(max_sequence_length=8, extra_binary_inputs=EXTRA)
    )
    assert v1.numeric.shape[1] == 8 and v2.numeric.shape[1] == 10
    np.testing.assert_array_equal(v2.numeric[:, :8], v1.numeric)
    np.testing.assert_array_equal(v2.source_lines, v1.source_lines)

    table = ds.dataset(str(data.features_dir), format="parquet", partitioning="hive").to_table(
        filter=ds.field("dataset_day") == 2, columns=["source_line", *EXTRA]
    )
    by_line = {line: (a, b) for line, a, b in zip(*(table[c].to_pylist() for c in ("source_line", *EXTRA)))}
    expected = np.array([by_line[line] for line in v2.source_lines], dtype=np.float32)
    np.testing.assert_array_equal(v2.numeric[:, 8:], expected)


def test_cached_samples_are_not_shared_between_input_lists(synthetic_sequences, tmp_path) -> None:
    data = synthetic_sequences
    inputs = SequenceInputs(
        features_dir=data.features_dir,
        preprocessing=data.preprocessing,
        preprocessing_path=data.root / "features" / "preprocessing.json",
        feature_cfg=FeatureConfig(),
        split_cfg=data.split_cfg,
        excluded_users=data.excluded_users,
        labels_dir=None,
    )
    base = SequenceDetectorConfig().with_trial(8, "small")
    base = replace(base, training=replace(base.training, train_sequences=40, monitor_sequences=10))
    extended = replace(base, policy=replace(base.policy, extra_binary_inputs=EXTRA))

    quiet = lambda message: None  # noqa: E731
    train_v1, _, _ = prepare_samples(inputs, base, tmp_path, log=quiet)
    train_v2, _, _ = prepare_samples(inputs, extended, tmp_path, log=quiet)
    cached_v1, _, _ = prepare_samples(inputs, base, tmp_path, log=quiet)
    assert train_v1.dense.shape[1] == 8
    assert train_v2.dense.shape[1] == 10
    np.testing.assert_array_equal(cached_v1.dense, train_v1.dense)
    assert len(list(tmp_path.glob("samples_*.npz"))) == 2
