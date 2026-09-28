"""Tests for detector output alignment (Task 5.1)."""

from __future__ import annotations

import pytest

from dualscope.fusion.alignment import (
    ALIGNED_SCORE_SCHEMA,
    AlignedScoreRow,
    ScoreAlignmentEngine,
)
from dualscope.fusion.config import AlignmentConfig, AlignmentStatus


@pytest.fixture
def alignment_engine() -> ScoreAlignmentEngine:
    return ScoreAlignmentEngine(
        AlignmentConfig(
            hour_seconds=3600,
            hour_origin=1,
            max_graph_staleness_seconds=86400,
            warmup_seconds=86400,
        )
    )


def test_align_single_both_available(alignment_engine: ScoreAlignmentEngine) -> None:
    """When both detectors have valid scores for Day 2+, status is both_available."""
    # Window: Day 2 hour 1: [86401, 90001)
    w_start = 86401
    w_end = 90001
    seq_rec = {
        "user_id": "U123@DOM1",
        "window_start": w_start,
        "window_end": w_end,
        "score_available_at": w_end,
        "status": "available",
        "raw_score": 0.052,
        "score": 0.985,
        "alert_threshold": 0.990,
        "is_alert": False,
        "source_lines": [101, 102, 103],
        "evidence_chunk_offset": 0,
        "evidence_chunk_length": 2,
    }
    graph_rec = {
        "user_id": "U123@DOM1",
        "window_start": w_start,
        "window_end": w_end,
        "score_available_at": w_end,
        "status": "available",
        "raw_score": 1.42,
        "score": 0.995,
        "alert_threshold": 0.990,
        "is_alert": True,
        "evidence_nodes": ["C100", "C200"],
    }

    row = alignment_engine.align_single("U123@DOM1", w_start, seq_rec, graph_rec, split="validation")
    assert row.alignment_status == AlignmentStatus.BOTH_AVAILABLE
    assert row.seq_score == 0.985
    assert row.graph_score == 0.995
    assert row.is_seq_alert is False
    assert row.is_graph_alert is True
    assert row.graph_age_seconds == 0
    assert row.seq_source_lines == [101, 102, 103]
    assert row.seq_evidence_chunk == (0, 2)
    assert row.graph_evidence_nodes == ["C100", "C200"]
    assert row.dataset_day == 2


def test_temporal_leakage_rejected(alignment_engine: ScoreAlignmentEngine) -> None:
    """Future scores (available after window_end) must be rejected to prevent temporal leakage."""
    w_start = 86401
    w_end = 90001

    # Future sequence score
    future_seq = {
        "user_id": "U123@DOM1",
        "score_available_at": w_end + 3600,  # 1 hour in future
        "score": 0.95,
    }
    with pytest.raises(ValueError, match="Temporal leakage violation"):
        alignment_engine.align_single("U123@DOM1", w_start, future_seq, None)

    # Future graph score
    future_graph = {
        "user_id": "U123@DOM1",
        "score_available_at": w_end + 1,  # 1 second in future
        "score": 0.95,
    }
    with pytest.raises(ValueError, match="Temporal leakage violation"):
        alignment_engine.align_single("U123@DOM1", w_start, None, future_graph)


def test_stale_graph_score_handling(alignment_engine: ScoreAlignmentEngine) -> None:
    """Graph scores older than max_graph_staleness_seconds (24h) are flagged stale."""
    w_start = 200001
    w_end = w_start + 3600

    seq_rec = {
        "user_id": "U123@DOM1",
        "status": "available",
        "score": 0.88,
        "alert_threshold": 0.99,
        "score_available_at": w_end,
    }
    # Graph score from 25 hours ago
    stale_graph = {
        "user_id": "U123@DOM1",
        "status": "available",
        "score": 0.92,
        "alert_threshold": 0.99,
        "score_available_at": w_end - 90000,  # 25 hours prior
    }

    row = alignment_engine.align_single("U123@DOM1", w_start, seq_rec, stale_graph)
    assert row.alignment_status == AlignmentStatus.STALE_GRAPH
    assert row.graph_status == AlignmentStatus.STALE_GRAPH.value
    # Graph score is decoupled when stale
    assert row.graph_score is None
    assert row.graph_age_seconds == 90000
    assert row.seq_score == 0.88


def test_missing_scores_not_zero_filled(alignment_engine: ScoreAlignmentEngine) -> None:
    """Missing detector scores must be None/null, never 0.0."""
    w_start = 100001
    w_end = w_start + 3600

    # Sequence only
    seq_rec = {
        "user_id": "U123@DOM1",
        "status": "available",
        "score": 0.75,
        "alert_threshold": 0.99,
        "score_available_at": w_end,
    }
    row_seq = alignment_engine.align_single("U123@DOM1", w_start, seq_rec, None)
    assert row_seq.alignment_status == AlignmentStatus.SEQUENCE_ONLY
    assert row_seq.seq_score == 0.75
    assert row_seq.graph_score is None  # MUST NOT BE 0.0

    # Graph only (e.g. no activity in sequence)
    graph_rec = {
        "user_id": "U123@DOM1",
        "status": "available",
        "score": 0.82,
        "alert_threshold": 0.99,
        "score_available_at": w_end,
    }
    row_graph = alignment_engine.align_single("U123@DOM1", w_start, None, graph_rec)
    assert row_graph.alignment_status == AlignmentStatus.GRAPH_ONLY
    assert row_graph.graph_score == 0.82
    assert row_graph.seq_score is None  # MUST NOT BE 0.0

    # Neither
    row_none = alignment_engine.align_single("U123@DOM1", w_start, None, None)
    assert row_none.alignment_status == AlignmentStatus.NO_ACTIVITY
    assert row_none.seq_score is None
    assert row_none.graph_score is None


def test_warmup_period_insufficient_history(alignment_engine: ScoreAlignmentEngine) -> None:
    """Windows within the Day 1 warm-up period (first 24h) have status insufficient_history."""
    # Window in Day 1: start at 3601 (< 86401)
    w_start = 3601
    seq_rec = {
        "user_id": "U123@DOM1",
        "status": "insufficient_history",
        "score": None,
        "score_available_at": 7201,
    }
    row = alignment_engine.align_single("U123@DOM1", w_start, seq_rec, None)
    assert row.alignment_status == AlignmentStatus.INSUFFICIENT_HISTORY
    assert row.seq_score is None
    assert row.graph_score is None
    assert row.dataset_day == 1


def test_align_records_causal_graph_selection(alignment_engine: ScoreAlignmentEngine) -> None:
    """align_records selects the latest graph score with score_available_at <= window_end."""
    seq_records = [
        {"user_id": "U1", "window_start": 90001, "score_available_at": 93601, "score": 0.8, "status": "available"},
        {"user_id": "U1", "window_start": 93601, "score_available_at": 97201, "score": 0.85, "status": "available"},
    ]
    graph_records = [
        {"user_id": "U1", "window_start": 86401, "score_available_at": 90001, "score": 0.5, "status": "available"},
        {"user_id": "U1", "window_start": 90001, "score_available_at": 93601, "score": 0.7, "status": "available"},
        # Future graph score available at 100000 (> 97201)
        {"user_id": "U1", "window_start": 93601, "score_available_at": 100000, "score": 0.99, "status": "available"},
    ]

    rows = alignment_engine.align_records(seq_records, graph_records)
    assert len(rows) >= 2

    row_1 = [r for r in rows if r.window_start == 90001][0]
    # For window [90001, 93601), graph score available at 93601 is eligible
    assert row_1.graph_score == 0.7
    assert row_1.graph_age_seconds == 0

    row_2 = [r for r in rows if r.window_start == 93601][0]
    # For window [93601, 97201), the score available at 100000 cannot be used.
    # The latest eligible score is the one available at 93601 (score 0.7)
    assert row_2.graph_score == 0.7
    assert row_2.graph_age_seconds == 3600


def test_to_arrow_table(alignment_engine: ScoreAlignmentEngine) -> None:
    """PyArrow table conversion preserves column types and schema."""
    row = alignment_engine.align_single(
        "U1",
        86401,
        {
            "user_id": "U1",
            "score": 0.9,
            "status": "available",
            "source_lines": [10, 20],
            "evidence_chunk_offset": 0,
            "evidence_chunk_length": 2,
        },
        {"user_id": "U1", "score": 0.8, "status": "available", "evidence_nodes": ["C1"]},
    )
    table = alignment_engine.to_arrow_table([row])
    assert table.num_rows == 1
    assert table.schema == ALIGNED_SCORE_SCHEMA
    assert table["user_id"][0].as_py() == "U1"
    assert table["seq_score"][0].as_py() == 0.9
    assert table["graph_score"][0].as_py() == 0.8
    assert table["seq_source_lines"][0].as_py() == [10, 20]
    assert table["graph_evidence_nodes"][0].as_py() == ["C1"]
