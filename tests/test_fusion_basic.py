"""Tests for basic fusion methods (Task 5.2)."""

from __future__ import annotations

import pytest

from dualscope.fusion.alignment import AlignedScoreRow
from dualscope.fusion.basic import (
    FUSED_SCORE_SCHEMA,
    BasicFusionEngine,
    classify_disagreement,
)
from dualscope.fusion.config import (
    AlignmentStatus,
    BasicFusionConfig,
    DisagreementType,
    FusionMethod,
)


def _make_aligned_row(
    user_id: str = "U1",
    window_start: int = 86401,
    seq_score: float | None = 0.8,
    is_seq_alert: bool | None = False,
    graph_score: float | None = 0.6,
    is_graph_alert: bool | None = False,
    alignment_status: AlignmentStatus = AlignmentStatus.BOTH_AVAILABLE,
) -> AlignedScoreRow:
    return AlignedScoreRow(
        user_id=user_id,
        window_start=window_start,
        window_end=window_start + 3600,
        dataset_day=2,
        split="validation",
        alignment_status=alignment_status,
        seq_status="available" if seq_score is not None else None,
        seq_raw_score=seq_score,
        seq_score=seq_score,
        seq_threshold=0.95,
        is_seq_alert=is_seq_alert,
        graph_status="available" if graph_score is not None else None,
        graph_raw_score=graph_score,
        graph_score=graph_score,
        graph_threshold=0.90,
        is_graph_alert=is_graph_alert,
        graph_score_available_at=window_start + 3600,
        graph_age_seconds=0,
    )


def test_max_fusion_hand_calculated() -> None:
    """Validate Maximum score fusion against hand-calculated examples."""
    engine = BasicFusionEngine(BasicFusionConfig(method=FusionMethod.MAX, fused_alert_threshold=0.9))

    # Case 1: Sequence dominates
    r1 = engine.fuse_single(_make_aligned_row(seq_score=0.85, graph_score=0.40))
    assert r1.fused_score == 0.85
    assert r1.is_fused_alert is False

    # Case 2: Graph dominates
    r2 = engine.fuse_single(_make_aligned_row(seq_score=0.30, graph_score=0.92))
    assert r2.fused_score == 0.92
    assert r2.is_fused_alert is True

    # Case 3: Sequence only
    r3 = engine.fuse_single(
        _make_aligned_row(seq_score=0.75, graph_score=None, alignment_status=AlignmentStatus.SEQUENCE_ONLY)
    )
    assert r3.fused_score == 0.75

    # Case 4: Neither available (missing score preserved)
    r4 = engine.fuse_single(
        _make_aligned_row(seq_score=None, graph_score=None, alignment_status=AlignmentStatus.NO_ACTIVITY)
    )
    assert r4.fused_score is None
    assert r4.is_fused_alert is None


def test_simple_average_fusion_hand_calculated() -> None:
    """Validate Simple Average score fusion against hand-calculated examples."""
    engine = BasicFusionEngine(
        BasicFusionConfig(method=FusionMethod.AVERAGE, renormalize_single_detector=True)
    )

    # Case 1: Equal weighting: (0.70 + 0.30) / 2 = 0.50
    r1 = engine.fuse_single(_make_aligned_row(seq_score=0.70, graph_score=0.30))
    assert r1.fused_score == pytest.approx(0.50)

    # Case 2: (0.85 + 0.95) / 2 = 0.90
    r2 = engine.fuse_single(_make_aligned_row(seq_score=0.85, graph_score=0.95))
    assert r2.fused_score == pytest.approx(0.90)

    # Case 3: Sequence only with renormalisation -> returns 0.80 directly
    r3 = engine.fuse_single(
        _make_aligned_row(seq_score=0.80, graph_score=None, alignment_status=AlignmentStatus.SEQUENCE_ONLY)
    )
    assert r3.fused_score == pytest.approx(0.80)

    # Case 4: Graph only with renormalisation -> returns 0.60 directly
    r4 = engine.fuse_single(
        _make_aligned_row(seq_score=None, graph_score=0.60, alignment_status=AlignmentStatus.GRAPH_ONLY)
    )
    assert r4.fused_score == pytest.approx(0.60)


def test_weighted_fusion_hand_calculated() -> None:
    """Validate Validation-Tuned Weighted score fusion against hand-calculated examples."""
    # w_seq = 0.6, w_graph = 0.4
    cfg_renorm = BasicFusionConfig(
        method=FusionMethod.WEIGHTED,
        sequence_weight=0.6,
        renormalize_single_detector=True,
    )
    engine_renorm = BasicFusionEngine(cfg_renorm)

    # Case 1: Both present -> 0.6 * 0.80 + 0.4 * 0.50 = 0.48 + 0.20 = 0.68
    r1 = engine_renorm.fuse_single(_make_aligned_row(seq_score=0.80, graph_score=0.50))
    assert r1.fused_score == pytest.approx(0.68)
    assert r1.weights_applied == (0.6, 0.4)

    # Case 2: Both present -> 0.6 * 0.20 + 0.4 * 0.90 = 0.12 + 0.36 = 0.48
    r2 = engine_renorm.fuse_single(_make_aligned_row(seq_score=0.20, graph_score=0.90))
    assert r2.fused_score == pytest.approx(0.48)

    # Case 3: Sequence only with renormalisation -> 0.80 (unscaled)
    r3 = engine_renorm.fuse_single(
        _make_aligned_row(seq_score=0.80, graph_score=None, alignment_status=AlignmentStatus.SEQUENCE_ONLY)
    )
    assert r3.fused_score == pytest.approx(0.80)
    assert r3.weights_applied == (1.0, 0.0)

    # Case 4: Sequence only WITHOUT renormalisation -> 0.6 * 0.80 = 0.48
    cfg_no_renorm = BasicFusionConfig(
        method=FusionMethod.WEIGHTED,
        sequence_weight=0.6,
        renormalize_single_detector=False,
    )
    engine_no_renorm = BasicFusionEngine(cfg_no_renorm)
    r4 = engine_no_renorm.fuse_single(
        _make_aligned_row(seq_score=0.80, graph_score=None, alignment_status=AlignmentStatus.SEQUENCE_ONLY)
    )
    assert r4.fused_score == pytest.approx(0.48)
    assert r4.weights_applied == (0.6, 0.0)


def test_disagreement_classification() -> None:
    """Ensure detector consensus and disagreement are categorised accurately."""
    assert classify_disagreement(True, True) == DisagreementType.CONCORDANT_ALERT
    assert classify_disagreement(True, False) == DisagreementType.SEQ_ONLY_ALERT
    assert classify_disagreement(True, None) == DisagreementType.SEQ_ONLY_ALERT
    assert classify_disagreement(False, True) == DisagreementType.GRAPH_ONLY_ALERT
    assert classify_disagreement(None, True) == DisagreementType.GRAPH_ONLY_ALERT
    assert classify_disagreement(False, False) == DisagreementType.CONCORDANT_NORMAL
    assert classify_disagreement(None, None) == DisagreementType.UNAVAILABLE


def test_fused_to_arrow_table() -> None:
    """Serialise fused outputs to PyArrow Table and verify schema."""
    engine = BasicFusionEngine()
    row = engine.fuse_single(_make_aligned_row(seq_score=0.9, graph_score=0.85))
    table = engine.to_arrow_table([row])

    assert table.num_rows == 1
    assert table.schema == FUSED_SCORE_SCHEMA
    assert table["user_id"][0].as_py() == "U1"
    assert table["fused_score"][0].as_py() == pytest.approx(0.875)  # 0.5 * 0.9 + 0.5 * 0.85
    assert table["fusion_method"][0].as_py() == "weighted"


def test_invalid_weights_rejected() -> None:
    """Weight must be strictly bounded in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="sequence_weight must be between 0.0 and 1.0"):
        BasicFusionConfig(sequence_weight=1.5)

    with pytest.raises(ValueError, match="sequence_weight must be between 0.0 and 1.0"):
        BasicFusionConfig(sequence_weight=-0.1)


def test_pipeline_integration(tmp_path: pytest.TempPathFactory) -> None:
    """End-to-end pipeline check using align_and_fuse_scores."""
    import json
    from scripts.align_and_fuse_scores import run_pipeline

    seq_file = tmp_path / "seq.jsonl"
    graph_file = tmp_path / "graph.jsonl"
    out_file = tmp_path / "fused.parquet"

    seq_data = [
        {"user_id": "U1", "window_start": 86401, "score_available_at": 90001, "score": 0.95, "status": "available"},
        {"user_id": "U2", "window_start": 86401, "score_available_at": 90001, "score": 0.40, "status": "available"},
    ]
    graph_data = [
        {"user_id": "U1", "window_start": 86401, "score_available_at": 90001, "score": 0.90, "status": "available"},
    ]

    with open(seq_file, "w") as f:
        for r in seq_data:
            f.write(json.dumps(r) + "\n")

    with open(graph_file, "w") as f:
        for r in graph_data:
            f.write(json.dumps(r) + "\n")

    result = run_pipeline(
        seq_path=seq_file,
        graph_path=graph_file,
        output_path=out_file,
        fusion_method=FusionMethod.WEIGHTED,
        seq_weight=0.7,
        threshold=0.9,
        max_staleness=86400,
    )

    assert result["num_units"] == 2
    assert result["total_alerts"] == 1  # U1: 0.7*0.95 + 0.3*0.90 = 0.665 + 0.270 = 0.935 >= 0.9
    assert out_file.exists()

