"""Unit tests for Task 9.2: Flat-feature Isolation Forest baseline."""

import numpy as np
import pyarrow as pa
import pytest

from dualscope.baseline.features import (
    BASELINE_FEATURE_NAMES,
    UserHourUnit,
    aggregate_events_to_user_hours,
    aggregate_user_hours_table,
    build_user_hour_feature_matrix,
)
from dualscope.baseline.isolation_forest import (
    BASELINE_SCORE_SCHEMA,
    IsolationForestBaseline,
)


def _mock_events_table() -> pa.Table:
    """Create a small mock events table simulating Task 2.4 transformed features."""
    return pa.Table.from_pydict(
        {
            "timestamp": [100, 200, 3700, 3800],
            "acting_user": ["U1@DOM1", "U1@DOM1", "U2@DOM1", "U2@DOM1"],
            "source_line": [10, 11, 20, 21],
            "log1p_prior_auth_count_1h_scaled": [0.5, 0.6, 1.2, 1.3],
            "log1p_prior_failure_count_1h_scaled": [0.0, 0.0, 0.5, 0.8],
            "log1p_seconds_since_previous_auth_scaled": [2.0, 1.5, 0.2, 0.1],
            "log1p_prior_unique_destinations_24h_scaled": [0.3, 0.3, 1.5, 1.5],
            "log1p_prior_user_destination_count_24h_scaled": [0.4, 0.4, 2.0, 2.0],
            "has_user_history": [1.0, 1.0, 1.0, 1.0],
            "is_new_user_destination": [0.0, 0.0, 1.0, 0.0],
            "is_new_host_connection": [0.0, 0.0, 1.0, 0.0],
            "auth_type_id": [1, 1, 2, 2],
            "logon_type_id": [2, 2, 3, 3],
            "auth_orientation_id": [1, 1, 2, 2],
            "auth_result_id": [1, 1, 1, 2],  # U2 has a failure
        }
    )


def test_feature_aggregation_creates_correct_dimensions():
    events = _mock_events_table()
    units = aggregate_events_to_user_hours(events, hour_seconds=3600)

    # 4 events split into 2 user-hours: U1 in hour 1, U2 in hour 3601
    assert len(units) == 2
    assert units[0].user_id == "U1@DOM1"
    assert units[0].window_start == 1
    assert units[0].window_end == 3601
    assert len(units[0].source_lines) == 2
    assert len(units[0].feature_vector) == len(BASELINE_FEATURE_NAMES)
    assert len(units[0].feature_vector) == 16

    assert units[1].user_id == "U2@DOM1"
    assert units[1].window_start == 3601
    assert len(units[1].source_lines) == 2
    # Check failure rate for U2 is 0.5 (1 success, 1 failure)
    assert units[1].feature_vector[BASELINE_FEATURE_NAMES.index("failure_rate_1h")] == 0.5


def test_feature_matrix_stacking():
    events = _mock_events_table()
    units = aggregate_events_to_user_hours(events, hour_seconds=3600)
    matrix = build_user_hour_feature_matrix(units)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 16)
    assert matrix.dtype == np.float32


def test_isolation_forest_fit_predict_score_orientation():
    # Synthetic normal data clustered around 0, anomalies clustered around 10
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 16))
    X_test_normal = rng.normal(loc=0.0, scale=1.0, size=(10, 16))
    X_test_anomaly = rng.normal(loc=10.0, scale=1.0, size=(10, 16))

    baseline = IsolationForestBaseline(n_estimators=50, random_state=42)
    baseline.fit(X_train)

    raw_norm = baseline.score_raw(X_test_normal)
    raw_anom = baseline.score_raw(X_test_anomaly)

    # Inverted score: larger raw score indicates higher anomaly
    assert np.mean(raw_anom) > np.mean(raw_norm)


def test_quantile_calibration_monotonicity():
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 16))
    X_val = rng.normal(loc=0.0, scale=2.0, size=(500, 16))

    baseline = IsolationForestBaseline(n_estimators=50, random_state=42)
    baseline.fit(X_train)

    val_raw = baseline.score_raw(X_val)
    baseline.calibrate(val_raw, threshold=0.99)

    units = [
        UserHourUnit(
            user_id=f"U{i}",
            window_start=1 + i * 3600,
            window_end=3601 + i * 3600,
            source_lines=[i],
            feature_vector=X_val[i].astype(np.float32),
        )
        for i in range(10)
    ]

    rows = baseline.predict_units(units)
    assert len(rows) == 10
    for r in rows:
        assert 0.0 <= r.score < 1.0
        assert r.detector == "isolation_forest_baseline"
        assert r.is_alert is not None


def test_arrow_table_schema_conformance():
    events = _mock_events_table()
    units = aggregate_events_to_user_hours(events, hour_seconds=3600)
    baseline = IsolationForestBaseline(n_estimators=20, random_state=42)
    X = build_user_hour_feature_matrix(units)
    baseline.fit(X)

    rows = baseline.predict_units(units)
    table = baseline.to_arrow_table(rows)

    assert table.num_rows == len(rows)
    assert table.schema == BASELINE_SCORE_SCHEMA
    assert table["detector"][0].as_py() == "isolation_forest_baseline"


def test_model_checkpoint_save_and_reload(tmp_path):
    events = _mock_events_table()
    units = aggregate_events_to_user_hours(events, hour_seconds=3600)
    X = build_user_hour_feature_matrix(units)

    baseline = IsolationForestBaseline(n_estimators=20, random_state=42)
    baseline.fit(X)

    save_dir = tmp_path / "baseline_model"
    baseline.save(save_dir)

    loaded = IsolationForestBaseline.load(save_dir)
    assert loaded.n_estimators == 20
    assert loaded.model_version == baseline.model_version

    preds_orig = baseline.score_raw(X)
    preds_loaded = loaded.score_raw(X)
    np.testing.assert_allclose(preds_orig, preds_loaded)


def test_synthetic_demo_units_and_calibrator_persistence(tmp_path):
    from scripts.train_baseline_isolation_forest import generate_synthetic_demo_units

    train_units, val_units, val_positives = generate_synthetic_demo_units()
    assert len(train_units) == 500
    assert len(val_units) == 820
    assert len(val_positives) == 20

    X_train = build_user_hour_feature_matrix(train_units)
    baseline = IsolationForestBaseline(n_estimators=15, random_state=42)
    baseline.fit(X_train)

    X_val = build_user_hour_feature_matrix(val_units)
    val_raw = baseline.score_raw(X_val)
    baseline.calibrate(val_raw, threshold=0.95)

    save_dir = tmp_path / "baseline_calibrated"
    baseline.save(save_dir)

    reloaded = IsolationForestBaseline.load(save_dir)
    assert reloaded.alert_threshold == baseline.alert_threshold
    assert reloaded.calibrator is not None

    rows_orig = baseline.predict_units(val_units[:10])
    rows_reloaded = reloaded.predict_units(val_units[:10])

    for r1, r2 in zip(rows_orig, rows_reloaded):
        assert r1.user_id == r2.user_id
        assert r1.score == pytest.approx(r2.score)
        assert r1.is_alert == r2.is_alert



def _random_events_table(n: int = 2_000, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(1, 4 * 3600, n))
    return pa.Table.from_pydict(
        {
            "timestamp": ts,
            "acting_user": rng.choice(["U1@DOM1", "U2@DOM1", "C3$@DOM1", "U4@C5"], n).tolist(),
            "source_line": np.arange(n),
            "log1p_prior_auth_count_1h_scaled": rng.normal(size=n).astype(np.float32),
            "log1p_prior_failure_count_1h_scaled": rng.normal(size=n).astype(np.float32),
            "log1p_seconds_since_previous_auth_scaled": rng.normal(size=n).astype(np.float32),
            "log1p_prior_unique_destinations_24h_scaled": rng.normal(size=n).astype(np.float32),
            "log1p_prior_user_destination_count_24h_scaled": rng.normal(size=n).astype(np.float32),
            "has_user_history": rng.integers(0, 2, n).astype(np.float32),
            "is_new_user_destination": rng.integers(0, 2, n).astype(np.float32),
            "is_new_host_connection": rng.integers(0, 2, n).astype(np.float32),
            # Few categories, so the mode often has ties.
            "auth_type_id": rng.integers(2, 4, n).astype(np.int32),
            "logon_type_id": rng.integers(2, 5, n).astype(np.int32),
            "auth_orientation_id": rng.integers(2, 4, n).astype(np.int32),
            "auth_result_id": rng.choice([2, 3], n, p=[0.1, 0.9]).astype(np.int32),
        }
    )


def test_vectorised_aggregation_matches_per_unit_aggregation():
    events = _random_events_table()
    units = aggregate_events_to_user_hours(events, hour_seconds=3600)
    keys, matrix = aggregate_user_hours_table(events, hour_seconds=3600)

    assert [(u.user_id, u.window_start) for u in units] == list(zip(keys["user_id"], keys["window_start"]))
    np.testing.assert_allclose(matrix, build_user_hour_feature_matrix(units), rtol=1e-6, atol=1e-6)


def test_failure_rate_counts_the_fail_id_only():
    events = _random_events_table(n=500, seed=1)
    keys, matrix = aggregate_user_hours_table(events, hour_seconds=3600, fail_id=2)
    rate = matrix[:, BASELINE_FEATURE_NAMES.index("failure_rate_1h")]
    frame = events.to_pandas()
    frame["window_start"] = 1 + ((frame["timestamp"] - 1) // 3600) * 3600
    expected = frame.assign(f=frame["auth_result_id"] == 2).groupby(["acting_user", "window_start"])["f"].mean()
    np.testing.assert_allclose(rate, expected.to_numpy(), rtol=1e-6)
    assert 0.0 < rate.mean() < 0.5  # not the old constant 1.0
