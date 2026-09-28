"""Task 3.4: frozen detector, score export schema, statuses and evidence retrieval."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from dualscope.evidence.lookup import AuthenticationEvidenceLookup
from dualscope.sequence.builder import load_day_sequences, source_reference
from dualscope.sequence.calibration import QuantileTailCalibrator
from dualscope.sequence.config import SequenceDetectorConfig, input_feature_names
from dualscope.sequence.detector import FROZEN_FILENAME, FrozenSequenceDetector, freeze_detector
from dualscope.sequence.export import (
    SEQUENCE_SCORE_SCHEMA,
    STATUS_NO_ACTIVITY,
    SequenceScoreStore,
    build_score_table,
    no_activity_rows,
    to_detector_record,
    write_day_table,
)
from dualscope.sequence.model import ModelSpec
from dualscope.sequence.training import categorical_cardinalities, sample_chunks, save_checkpoint, train_autoencoder
from dualscope.splits import ScoringStatus


CONFIG = SequenceDetectorConfig().with_trial(8, "small")


@pytest.fixture(scope="module")
def frozen(synthetic_sequences, tmp_path_factory):
    data = synthetic_sequences
    policy = CONFIG.policy
    days = {d: load_day_sequences(data.features_dir, d, data.split_cfg, policy, data.excluded_users) for d in range(1, 5)}
    train, _ = sample_chunks([days[2]], 8, 10_000, 2, np.random.default_rng(0), lambda d: d.fitting_eligible)
    spec = ModelSpec.from_settings(CONFIG.model, 5, 3, categorical_cardinalities(data.preprocessing))
    model, _ = train_autoencoder(spec, train, train, replace(CONFIG.training, max_epochs=2, batch_size=16))
    root = tmp_path_factory.mktemp("frozen")
    checkpoint = root / "run" / "checkpoint.pt"
    save_checkpoint(checkpoint, model, {})

    from dualscope.sequence.scoring import score_day

    reference = score_day(model, days[3], 8).raw[CONFIG.aggregation]
    calibrator = QuantileTailCalibrator.fit(reference[~np.isnan(reference)], min_reference=10)
    freeze_detector(
        root / "frozen",
        checkpoint,
        CONFIG,
        calibrator,
        alert_threshold=0.9,
        run_id="test-run",
        feature_info={"feature_config_fingerprint": "fp-features"},
        split_policy_fingerprint=data.split_cfg.fingerprint(),
        selection={"rule": "test"},
    )
    detector = FrozenSequenceDetector.load(root / "frozen", "fp-features", data.split_cfg.fingerprint())
    return detector, days, root / "frozen"


def _table(detector, day):
    scores = detector.score_day(day)
    raw = scores.raw[detector.aggregation]
    return build_score_table(
        scores,
        detector.normalise(raw),
        raw,
        model_version=detector.model_version,
        run_id=detector.run_id,
        aggregation=detector.aggregation,
        alert_threshold=detector.alert_threshold,
        hour_seconds=3600,
        top_events=5,
        top_features=3,
    )


def test_available_rows_conform_to_schema_and_carry_retrievable_evidence(frozen, synthetic_sequences) -> None:
    detector, days, _ = frozen
    table = _table(detector, days[3])
    assert table.schema.equals(SEQUENCE_SCORE_SCHEMA)
    assert table.num_rows == days[3].n_user_hours
    lookup = AuthenticationEvidenceLookup(synthetic_sequences.auth_root)
    feature_names = set(input_feature_names())

    for row in table.to_pylist():
        assert row["status"] == ScoringStatus.AVAILABLE.value
        assert 0.0 <= row["score"] < 1.0 and np.isfinite(row["raw_score"])
        assert row["is_alert"] == (row["score"] >= row["alert_threshold"])
        assert row["window_end"] - row["window_start"] == 3600 == row["score_available_at"] - row["window_start"]
        assert row["split"] == "validation"
        assert len(row["source_lines"]) == row["n_events"]
        assert 0 <= row["evidence_chunk_offset"] < row["n_events"]
        assert row["evidence_chunk_offset"] + row["evidence_chunk_length"] <= row["n_events"]
        errors = [event["event_error"] for event in row["top_events"]]
        assert errors == sorted(errors, reverse=True) and 1 <= len(errors) <= 5
        assert {event["top_feature"] for event in row["top_events"]} <= feature_names
        shares = [item["share"] for item in row["top_feature_contributions"]]
        assert len(shares) == 3 and shares == sorted(shares, reverse=True) and sum(shares) <= 1.0 + 1e-6

        record = to_detector_record(row)
        assert {e["source_reference"] for e in row["top_events"]} <= set(record["source_references"])
        resolved = lookup.lookup_many(record["source_references"])
        assert all(event is not None for event in resolved.values())
        assert {event["acting_user"] for event in resolved.values()} == {row["user_id"]}
        assert all(row["window_start"] <= event["timestamp"] < row["window_end"] for event in resolved.values())
        assert len(record["evidence_source_references"]) == row["evidence_chunk_length"]


def test_evidence_chunk_is_the_one_that_set_the_score(frozen) -> None:
    detector, days, _ = frozen
    scores = detector.score_day(days[2])
    day = scores.day
    for u in np.flatnonzero(day.scorable_mask()):
        span = day.user_hour_events(int(u))
        mean_chunk = scores.evidence_chunks("max_chunk_mean")[u]
        assert scores.chunk_mean[mean_chunk] == pytest.approx(scores.raw["max_chunk_mean"][u])
        event_chunk = scores.evidence_chunks("max_event")[u]
        worst = span.start + int(np.argmax(scores.event_error[span]))
        start = scores.chunks.start[event_chunk]
        assert start <= worst < start + scores.chunks.length[event_chunk]
        assert scores.chunks.user_hour[mean_chunk] == scores.chunks.user_hour[event_chunk] == u
        hour_errors = scores.event_error[span]
        assert scores.raw["hour_mean"][u] == pytest.approx(float(hour_errors.mean()), rel=1e-6)
        assert scores.raw["max_event"][u] == pytest.approx(float(hour_errors.max()))


def test_warm_up_and_no_activity_rows_have_explicit_status_and_no_score(frozen, synthetic_sequences) -> None:
    detector, days, _ = frozen
    for row in _table(detector, days[1]).to_pylist():
        assert row["status"] == ScoringStatus.INSUFFICIENT_HISTORY.value
        assert row["raw_score"] is None and row["score"] is None and row["is_alert"] is None
        assert row["top_events"] == [] and len(row["source_lines"]) == row["n_events"] > 0

    empty = no_activity_rows(
        [("U404@DOM1", 172_801)],
        synthetic_sequences.split_cfg,
        model_version=detector.model_version,
        run_id=detector.run_id,
        aggregation=detector.aggregation,
        alert_threshold=detector.alert_threshold,
        hour_seconds=3600,
    )
    assert empty.schema.equals(SEQUENCE_SCORE_SCHEMA)
    row = empty.to_pylist()[0]
    assert row["status"] == STATUS_NO_ACTIVITY and row["score"] is None and row["n_events"] == 0
    assert row["split"] == "validation"


def test_repeated_inference_is_consistent_and_store_round_trips(frozen, tmp_path) -> None:
    detector, days, _ = frozen
    first = _table(detector, days[4])
    second = _table(detector, days[4])
    np.testing.assert_allclose(
        first["raw_score"].to_numpy(), second["raw_score"].to_numpy(), rtol=0, atol=1e-9
    )
    assert first["is_alert"].to_pylist() == second["is_alert"].to_pylist()

    write_day_table(first, tmp_path / "scores", 4)
    store = SequenceScoreStore(tmp_path / "scores")
    sample = first.slice(0, 1).to_pylist()[0]
    record = store.get(sample["user_id"], sample["window_start"])
    assert record["score"] == pytest.approx(sample["score"])
    assert record["source_references"] == [source_reference(line) for line in sample["source_lines"]]
    assert store.get("nobody", sample["window_start"]) is None


def test_modified_frozen_record_is_rejected(frozen, synthetic_sequences) -> None:
    _, _, frozen_dir = frozen
    path = frozen_dir / FROZEN_FILENAME
    original = path.read_text(encoding="utf-8")
    try:
        record = json.loads(original)
        record["alert_threshold"] = 0.1
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="modified"):
            FrozenSequenceDetector.load(frozen_dir)
    finally:
        path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="split policy"):
        FrozenSequenceDetector.load(frozen_dir, split_policy_fingerprint="different")
