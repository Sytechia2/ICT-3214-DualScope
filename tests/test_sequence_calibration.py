"""Task 3.3: score normalisation, threshold selection and label guards."""

from __future__ import annotations

import numpy as np
import pytest

from dualscope.sequence.calibration import (
    QuantileTailCalibrator,
    best_f1_threshold,
    detection_metrics,
    evaluation_summary,
    load_positive_user_hours,
    unit_labels,
)


def test_calibrator_is_monotonic_bounded_and_keeps_extremes_rankable() -> None:
    rng = np.random.default_rng(0)
    reference = rng.lognormal(mean=-1.0, sigma=0.6, size=50_000)
    calibrator = QuantileTailCalibrator.fit(reference)

    grid = np.exp(np.linspace(-8, 6, 5_000))
    scores = calibrator.transform(grid)
    assert np.all(np.diff(scores) >= 0)
    assert scores.min() == 0.0 and scores.max() < 1.0
    extremes = calibrator.transform(np.array([reference.max() * 2, reference.max() * 4, reference.max() * 8]))
    assert np.all(np.diff(extremes) > 0), "tail scores must stay strictly increasing"
    # Inside the reference range the score is the empirical quantile.
    median_score = calibrator.transform(np.array([np.median(reference)]))[0]
    assert median_score == pytest.approx(0.5, abs=0.002)
    assert np.isnan(calibrator.transform(np.array([np.nan]))[0])

    restored = QuantileTailCalibrator.from_dict(calibrator.to_dict())
    np.testing.assert_array_equal(restored.transform(grid), scores)


def test_calibrator_handles_ties_in_reference_scores() -> None:
    reference = np.r_[np.full(500, 0.2), np.linspace(0.3, 2.0, 500)]
    calibrator = QuantileTailCalibrator.fit(reference)
    scores = calibrator.transform(np.array([0.1, 0.2, 0.25, 2.0, 5.0]))
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(0.5, abs=0.002)
    assert scores[1] <= scores[2] <= scores[3] < scores[4] < 1.0


def test_threshold_selection_matches_hand_calculation() -> None:
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    scores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    # Cut at 0.8: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3 (the maximum).
    best = best_f1_threshold(labels, scores)
    assert best["threshold"] == 0.8
    assert (best["tp"], best["fp"], best["fn"], best["tn"]) == (2, 1, 1, 4)
    assert best["f1"] == pytest.approx(2 / 3)
    assert best["false_positive_rate"] == pytest.approx(1 / 5)
    assert detection_metrics(labels, scores, 0.99)["alerts"] == 0


def test_evaluation_summary_reports_coverage_of_unscored_positives() -> None:
    users = np.array(["A", "B", "C", "D"], dtype=object)
    hours = np.array([1, 1, 1, 1])
    scores = np.array([0.9, 0.2, 0.7, 0.1])
    positives = {("A", 1), ("Z", 1)}  # Z has no scored user-hour
    labels = unit_labels(users, hours, positives)
    assert labels.tolist() == [True, False, False, False]
    summary = evaluation_summary(labels, scores, len(positives))
    assert summary["positive_units_total"] == 2
    assert summary["positive_units_scored"] == 1
    assert summary["positive_coverage"] == 0.5
    assert summary["average_precision"] == pytest.approx(1.0)
    assert summary["at_threshold"]["recall_including_unscored_positives"] == 0.5


def test_test_labels_are_refused_during_selection(synthetic_sequences) -> None:
    with pytest.raises(PermissionError):
        load_positive_user_hours("unused", synthetic_sequences.split_cfg, "test")
