import pytest

from dualscope.graph.evaluation import evaluate_records, label_snapshot_cutoff
from dualscope.graph.threat_history import ThreatHistory, alerted_user_days, classification_metrics
from dualscope.graph.overfitting import temporal_overfitting_report

def test_labels_align_to_first_strictly_later_daily_cutoff_and_metrics_include_auc():
    assert label_snapshot_cutoff(1) == 86401
    assert label_snapshot_cutoff(86401) == 172801
    rows=[
        {"user_id":"U1","window_end":86401,"status":"available","score":.9},
        {"user_id":"U2","window_end":86401,"status":"available","score":.1},
        {"user_id":"U3","window_end":86401,"status":"insufficient_history","score":None},
    ]
    result=evaluate_records(rows,[{"user":"U1","timestamp":2}],threshold=.5)
    assert result["average_precision"] == 1.0 and result["roc_auc"] == 1.0
    assert result["at_threshold"]["precision"] == result["at_threshold"]["recall"] == 1.0


def test_threat_history_is_causal_and_matches_exact_relationships():
    labels = [
        {"timestamp": 100, "user": "U1", "source_computer": "S1", "destination_computer": "D1"},
        {"timestamp": 300, "user": "FUTURE", "source_computer": "S2", "destination_computer": "D2"},
    ]
    history = ThreatHistory.from_labels(labels, frozen_before=200)
    matching = {"timestamp": 250, "acting_user": "U1", "source_computer": "S1", "destination_computer": "D1"}
    assert history.matches(matching)
    assert not history.matches(dict(matching, destination_computer="D9"))
    assert ("FUTURE", "S2", "D2") not in history.triples


def test_threat_history_user_day_metrics():
    history = ThreatHistory.from_labels(
        [{"timestamp": 10, "user": "U1", "source_computer": "S1", "destination_computer": "D1"}],
        frozen_before=100,
    )
    events = [
        {"timestamp": 100, "dataset_day": 2, "acting_user": "U1", "source_computer": "S1", "destination_computer": "D1"},
        {"timestamp": 101, "dataset_day": 2, "acting_user": "U2", "source_computer": "S2", "destination_computer": "D2"},
    ]
    alerts = alerted_user_days(events, history)
    metrics = classification_metrics({(2, "U1"), (2, "U2")}, {(2, "U1")}, alerts)
    assert metrics == {"tp": 1, "fp": 0, "fn": 0, "tn": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_temporal_overfitting_check_flags_holdout_collapse():
    development = {"f1": 0.90, "precision": 0.90, "recall": 0.90, "tp": 90, "fn": 10}
    holdout = {"f1": 0.30, "precision": 0.30, "recall": 0.30, "tp": 30, "fn": 70}
    report = temporal_overfitting_report(development, holdout)
    assert report["status"] == "FAIL"
    assert report["overfitting_detected"] is True
    assert report["absolute_f1_drop"] == pytest.approx(0.6)


def test_temporal_overfitting_check_passes_stable_holdout():
    development = {"f1": 0.82, "precision": 0.84, "recall": 0.80, "tp": 80, "fn": 20}
    holdout = {"f1": 0.79, "precision": 0.81, "recall": 0.77, "tp": 77, "fn": 23}
    report = temporal_overfitting_report(development, holdout)
    assert report["status"] == "PASS"
    assert report["overfitting_detected"] is False
