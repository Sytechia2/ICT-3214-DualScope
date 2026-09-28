"""Tests for temporal proximity fusion (Task 5.3)."""

from __future__ import annotations

import math
import pytest

from dualscope.fusion.alignment import AlignedScoreRow
from dualscope.fusion.config import (
    AlignmentStatus,
    BasicFusionConfig,
    FusionMethod,
    TemporalConfig,
)
from dualscope.fusion.temporal import (
    TEMPORAL_FUSED_SCORE_SCHEMA,
    TemporalFusionEngine,
)


def _make_row(
    user_id: str,
    window_start: int,
    seq_score: float | None = 0.5,
    is_seq_alert: bool = False,
    graph_score: float | None = 0.5,
    is_graph_alert: bool = False,
) -> AlignedScoreRow:
    return AlignedScoreRow(
        user_id=user_id,
        window_start=window_start,
        window_end=window_start + 3600,
        dataset_day=2,
        split="validation",
        alignment_status=AlignmentStatus.BOTH_AVAILABLE,
        seq_status="available" if seq_score is not None else None,
        seq_raw_score=seq_score,
        seq_score=seq_score,
        seq_threshold=0.90,
        is_seq_alert=is_seq_alert,
        graph_status="available" if graph_score is not None else None,
        graph_raw_score=graph_score,
        graph_score=graph_score,
        graph_threshold=0.90,
        is_graph_alert=is_graph_alert,
        graph_score_available_at=window_start + 3600,
        graph_age_seconds=0,
    )


@pytest.fixture
def temporal_engine() -> TemporalFusionEngine:
    t_cfg = TemporalConfig(
        lookback_seconds=86400,  # 24 hours
        half_life_seconds=21600.0,  # 6 hours
        boost_weight=0.10,  # gamma = 0.10
        max_fused_score_cap=0.9999,
        fused_alert_threshold=0.90,
    )
    b_cfg = BasicFusionConfig(method=FusionMethod.WEIGHTED, sequence_weight=0.5)
    return TemporalFusionEngine(temporal_config=t_cfg, base_config=b_cfg)


def test_cooccurrence_boost_applied(temporal_engine: TemporalFusionEngine) -> None:
    """When both detectors alert close in time, temporal co-occurrence boost is applied."""
    # User U1: Sequence alerts at hour 1 (end 90001). Graph alerts at hour 3 (end 97201).
    # Delta t = 7200 seconds (2 hours)
    r1 = _make_row("U1", 86401, seq_score=0.95, is_seq_alert=True, graph_score=0.50, is_graph_alert=False)
    r2 = _make_row("U1", 90001, seq_score=0.50, is_seq_alert=False, graph_score=0.50, is_graph_alert=False)
    r3 = _make_row("U1", 93601, seq_score=0.50, is_seq_alert=False, graph_score=0.95, is_graph_alert=True)

    stream = temporal_engine.fuse_stream([r1, r2, r3])
    assert len(stream) == 3

    # Window 1: only sequence alerted so far -> no co-occurrence yet
    assert stream[0].temporal_boost == 0.0
    assert stream[0].lead_detector is None

    # Window 3: Graph alerts at t=97201. Sequence alert was at t=90001 (diff = 7200s).
    expected_boost = 0.10 * math.exp(-7200 / 21600)
    assert stream[2].temporal_boost == pytest.approx(expected_boost, rel=1e-4)
    # Sequence alerted first
    assert stream[2].lead_detector == "sequence"
    assert stream[2].lead_time_seconds == 7200
    # Base score: 0.5 * 0.50 + 0.5 * 0.95 = 0.725
    assert stream[2].base_fused_score == pytest.approx(0.725)
    assert stream[2].fused_score == pytest.approx(0.725 + expected_boost, rel=1e-4)


def test_outside_window_receives_no_boost(temporal_engine: TemporalFusionEngine) -> None:
    """Alerts outside lookback window (> 24 hours) produce zero temporal boost."""
    # Sequence alerts at day 2 hour 1 (end 90001).
    # Graph alerts 26 hours later at day 3 hour 3 (end 183601 > 90001 + 86400).
    r1 = _make_row("U1", 86401, seq_score=0.95, is_seq_alert=True, graph_score=0.5, is_graph_alert=False)
    r2 = _make_row("U1", 180001, seq_score=0.5, is_seq_alert=False, graph_score=0.95, is_graph_alert=True)

    stream = temporal_engine.fuse_stream([r1, r2])
    # Outside 24h window -> no boost
    assert stream[1].temporal_boost == 0.0
    assert stream[1].lead_detector is None


def test_future_alerts_do_not_retroactively_modify_past(
    temporal_engine: TemporalFusionEngine,
) -> None:
    """Causal check: a future alert at t2 must NEVER increase the score at t1 < t2."""
    r1 = _make_row("U1", 86401, seq_score=0.85, is_seq_alert=False, graph_score=0.5, is_graph_alert=False)
    r2_alert = _make_row("U1", 90001, seq_score=0.95, is_seq_alert=True, graph_score=0.95, is_graph_alert=True)

    # Score r1 alone
    stream_alone = temporal_engine.fuse_stream([r1])
    score_alone = stream_alone[0].fused_score
    boost_alone = stream_alone[0].temporal_boost

    # Score r1 followed by future alert r2
    stream_with_future = temporal_engine.fuse_stream([r1, r2_alert])
    score_with_future = stream_with_future[0].fused_score
    boost_with_future = stream_with_future[0].temporal_boost

    # Past score must be strictly unchanged
    assert score_alone == score_with_future
    assert boost_alone == boost_with_future == 0.0


def test_simultaneous_alerts(temporal_engine: TemporalFusionEngine) -> None:
    """When both detectors alert in the exact same hour, delta t = 0 -> boost = gamma."""
    r = _make_row("U1", 86401, seq_score=0.95, is_seq_alert=True, graph_score=0.95, is_graph_alert=True)
    stream = temporal_engine.fuse_stream([r])

    assert stream[0].temporal_boost == pytest.approx(0.10)  # max boost (gamma)
    assert stream[0].lead_detector == "simultaneous"
    assert stream[0].lead_time_seconds == 0


def test_score_capping(temporal_engine: TemporalFusionEngine) -> None:
    """Fused score must never exceed max_fused_score_cap (0.9999)."""
    # Base score 0.999 + boost 0.10 = 1.099 -> capped to 0.9999
    r = _make_row("U1", 86401, seq_score=0.999, is_seq_alert=True, graph_score=0.999, is_graph_alert=True)
    stream = temporal_engine.fuse_stream([r])

    assert stream[0].fused_score == 0.9999


def test_temporal_to_arrow_table(temporal_engine: TemporalFusionEngine) -> None:
    """PyArrow table conversion matches TEMPORAL_FUSED_SCORE_SCHEMA."""
    r = _make_row("U1", 86401, seq_score=0.9, is_seq_alert=True, graph_score=0.9, is_graph_alert=True)
    stream = temporal_engine.fuse_stream([r])
    table = temporal_engine.to_arrow_table(stream)

    assert table.num_rows == 1
    assert table.schema == TEMPORAL_FUSED_SCORE_SCHEMA
    assert table["lead_detector"][0].as_py() == "simultaneous"
    assert table["temporal_boost"][0].as_py() == pytest.approx(0.10)
