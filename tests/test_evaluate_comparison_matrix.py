"""Tests for the Task 9.3 comparison matrix helpers."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.evaluate_comparison_matrix import (
    add_temporal_features,
    evaluate_incident_triage,
    load_baseline_scores,
    load_final_model_scores,
    top_k,
)


def _day(day: int = 13) -> dict:
    start = (day - 1) * 86400 + 1
    seq = np.array([0.99, 0.50, 0.99, 0.10])
    graph = np.array([0.99, 0.99, 0.10, 0.10])
    return {
        "day": day,
        "users": ["U1", "U2", "U3", "U4"],
        "windows": np.array([start, start + 3600, start + 7200, start + 10800]),
        "seq_scores": seq,
        "graph_scores": graph,
        "base_fused": 0.5 * seq + 0.5 * graph,
    }


def test_temporal_features_use_given_cutoffs_only() -> None:
    d = _day()
    add_temporal_features(d, seq_cutoff=0.9, graph_cutoff=0.9)

    assert d["seq_alerts"].tolist() == [True, False, True, False]
    assert d["graph_alerts"].tolist() == [True, True, False, False]
    # Only unit 0 has both detectors alerting; it starts at the day boundary.
    assert d["temporal_boost"].tolist() == pytest.approx([0.15, 0.0, 0.0, 0.0])
    assert d["lead_time_seconds"].tolist() == [0.0, 0.0, 0.0, 0.0]

    d = _day()
    add_temporal_features(d, seq_cutoff=0.4, graph_cutoff=0.9)
    # Unit 1 now co-alerts one hour after the day boundary.
    assert d["lead_time_seconds"][1] == 3600
    assert d["temporal_boost"][1] == pytest.approx(0.15 * np.exp(-3600 / 21600))
    assert d["log_lead_time"][1] == pytest.approx(np.log1p(3600))
    assert d["temp_scores"][1] == pytest.approx(d["base_fused"][1] + d["temporal_boost"][1])


def test_top_k_breaks_ties_by_given_order() -> None:
    scores = np.array([0.5, 0.9, 0.5, 0.5])
    assert top_k(scores, 2).tolist() == [1, 0]
    assert top_k(scores, 2, tie_order=np.array([3, 0, 1, 2])).tolist() == [1, 2]


def _incident(user: str, start: int, hours: int) -> dict:
    return {"user_id": user, "start_time": start, "end_time": start + hours * 3600, "duration_hours": hours}


def test_incident_precision_divides_by_incidents_reviewed() -> None:
    incidents = [_incident("U1", 1, 2), _incident("U2", 1, 1), _incident("U3", 1, 1), _incident("U4", 1, 1)]
    redteam_hours = {("U1", 3601)}

    result = evaluate_incident_triage(incidents, redteam_hours, budget=152, total_attacks=10)

    assert result["triaged_incidents"] == 4
    assert result["malicious_incidents_caught"] == 1
    assert result["incident_precision_pct"] == 25.0
    assert result["total_hours_reviewed"] == 5
    assert result["actual_attack_hours_caught"] == 1


def test_final_model_scores_must_cover_every_test_unit(tmp_path) -> None:
    d = _day()
    path = tmp_path / "scores.parquet"
    pq.write_table(
        pa.table({
            "user_id": d["users"][:3],
            "window_start": d["windows"][:3],
            "score": [0.1, 0.2, 0.3],
            "tie_order": [2, 1, 0],
        }),
        path,
    )
    with pytest.raises(SystemExit, match="missing for 1"):
        load_final_model_scores(path, [d])

    pq.write_table(
        pa.table({
            "user_id": d["users"],
            "window_start": d["windows"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "tie_order": [3, 2, 1, 0],
        }),
        path,
    )
    load_final_model_scores(path, [d])
    assert d["final_scores"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert d["final_tie_order"].tolist() == [3, 2, 1, 0]


def test_baseline_scores_must_cover_every_test_unit(tmp_path) -> None:
    d = _day()
    path = tmp_path / "baseline.parquet"
    pq.write_table(pa.table({"user_id": d["users"][1:], "window_start": d["windows"][1:], "score": [0.2, 0.3, 0.4]}), path)
    with pytest.raises(SystemExit, match="missing for 1"):
        load_baseline_scores(path, [d])

    pq.write_table(pa.table({"user_id": d["users"][::-1], "window_start": d["windows"][::-1], "score": [0.4, 0.3, 0.2, 0.1]}), path)
    load_baseline_scores(path, [d])
    assert d["baseline_scores"].tolist() == [0.1, 0.2, 0.3, 0.4]
