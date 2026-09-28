"""Basic fusion methods for combining sequence and graph detector scores (Task 5.2).

Implements Maximum, Simple Average, and Validation-Tuned Weighted score fusion.
Handles missing/single-detector scores cleanly via explicit weight renormalisation.
Categorises consensus and disagreement between detectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence
import numpy as np
import pyarrow as pa

from dualscope.fusion.alignment import AlignedScoreRow
from dualscope.fusion.config import BasicFusionConfig, DisagreementType, FusionMethod


@dataclass(frozen=True)
class FusedScoreRow:
    """The result of applying a fusion method to an aligned scoring unit."""

    user_id: str
    window_start: int
    window_end: int
    dataset_day: int
    split: str | None

    # Fused results
    fused_score: float | None
    is_fused_alert: bool | None
    fusion_method: FusionMethod
    disagreement_type: DisagreementType
    weights_applied: tuple[float, float] | None  # (seq_weight, graph_weight)

    # Underlying detector outputs
    seq_score: float | None
    is_seq_alert: bool | None
    graph_score: float | None
    is_graph_alert: bool | None
    alignment_status: str

    # Evidence preservation
    seq_source_lines: list[int] | None = None
    seq_evidence_chunk: tuple[int, int] | None = None
    graph_evidence_nodes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fusion_method"] = self.fusion_method.value
        data["disagreement_type"] = self.disagreement_type.value
        return data


FUSED_SCORE_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("window_start", pa.int64(), nullable=False),
        pa.field("window_end", pa.int64(), nullable=False),
        pa.field("dataset_day", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("fused_score", pa.float64(), nullable=True),
        pa.field("is_fused_alert", pa.bool_(), nullable=True),
        pa.field("fusion_method", pa.string(), nullable=False),
        pa.field("disagreement_type", pa.string(), nullable=False),
        pa.field("seq_weight_applied", pa.float64(), nullable=True),
        pa.field("graph_weight_applied", pa.float64(), nullable=True),
        pa.field("seq_score", pa.float64(), nullable=True),
        pa.field("is_seq_alert", pa.bool_(), nullable=True),
        pa.field("graph_score", pa.float64(), nullable=True),
        pa.field("is_graph_alert", pa.bool_(), nullable=True),
        pa.field("alignment_status", pa.string(), nullable=False),
        pa.field("seq_source_lines", pa.list_(pa.int64()), nullable=True),
        pa.field("seq_evidence_chunk_offset", pa.int32(), nullable=True),
        pa.field("seq_evidence_chunk_length", pa.int32(), nullable=True),
        pa.field("graph_evidence_nodes", pa.list_(pa.string()), nullable=True),
    ]
)


def classify_disagreement(
    is_seq_alert: bool | None, is_graph_alert: bool | None
) -> DisagreementType:
    """Classify consensus vs disagreement between sequence and graph detectors."""
    if is_seq_alert is None and is_graph_alert is None:
        return DisagreementType.UNAVAILABLE
    if is_seq_alert is True and is_graph_alert is True:
        return DisagreementType.CONCORDANT_ALERT
    if is_seq_alert is True and not is_graph_alert:
        return DisagreementType.SEQ_ONLY_ALERT
    if is_graph_alert is True and not is_seq_alert:
        return DisagreementType.GRAPH_ONLY_ALERT
    return DisagreementType.CONCORDANT_NORMAL


class BasicFusionEngine:
    """Combines aligned multi-timescale scores using standard fusion policies."""

    def __init__(self, config: BasicFusionConfig | None = None) -> None:
        self.config = config or BasicFusionConfig()

    def fuse_single(self, aligned: AlignedScoreRow) -> FusedScoreRow:
        """Apply the configured fusion method to one aligned user-hour row."""
        s_seq = aligned.seq_score
        s_graph = aligned.graph_score
        is_seq_alert = aligned.is_seq_alert
        is_graph_alert = aligned.is_graph_alert

        # 1. Determine disagreement type
        disagreement = classify_disagreement(is_seq_alert, is_graph_alert)

        # 2. Compute fused score based on availability
        fused_score: float | None = None
        weights: tuple[float, float] | None = None

        if s_seq is not None and s_graph is not None:
            # Both detectors present
            if self.config.method == FusionMethod.MAX:
                fused_score = max(s_seq, s_graph)
                weights = (1.0 if s_seq >= s_graph else 0.0, 1.0 if s_graph > s_seq else 0.0)
            elif self.config.method == FusionMethod.AVERAGE:
                fused_score = 0.5 * s_seq + 0.5 * s_graph
                weights = (0.5, 0.5)
            elif self.config.method == FusionMethod.WEIGHTED:
                w_seq = self.config.sequence_weight
                w_graph = self.config.graph_weight
                fused_score = w_seq * s_seq + w_graph * s_graph
                weights = (w_seq, w_graph)
            else:
                raise ValueError(f"Unsupported basic fusion method: {self.config.method}")

        elif s_seq is not None:
            # Sequence only
            if self.config.renormalize_single_detector:
                fused_score = s_seq
                weights = (1.0, 0.0)
            else:
                # Without renormalisation, weight is unscaled
                fused_score = self.config.sequence_weight * s_seq
                weights = (self.config.sequence_weight, 0.0)

        elif s_graph is not None:
            # Graph only
            if self.config.renormalize_single_detector:
                fused_score = s_graph
                weights = (0.0, 1.0)
            else:
                fused_score = self.config.graph_weight * s_graph
                weights = (0.0, self.config.graph_weight)

        else:
            # Neither detector has score (missingness is preserved, never zero-filled)
            fused_score = None
            weights = None

        # 3. Determine if fused score alerts
        is_fused_alert: bool | None = None
        if fused_score is not None:
            is_fused_alert = bool(fused_score >= self.config.fused_alert_threshold)

        return FusedScoreRow(
            user_id=aligned.user_id,
            window_start=aligned.window_start,
            window_end=aligned.window_end,
            dataset_day=aligned.dataset_day,
            split=aligned.split,
            fused_score=fused_score,
            is_fused_alert=is_fused_alert,
            fusion_method=self.config.method,
            disagreement_type=disagreement,
            weights_applied=weights,
            seq_score=s_seq,
            is_seq_alert=is_seq_alert,
            graph_score=s_graph,
            is_graph_alert=is_graph_alert,
            alignment_status=aligned.alignment_status.value,
            seq_source_lines=aligned.seq_source_lines,
            seq_evidence_chunk=aligned.seq_evidence_chunk,
            graph_evidence_nodes=aligned.graph_evidence_nodes,
        )

    def fuse_rows(self, aligned_rows: Iterable[AlignedScoreRow]) -> list[FusedScoreRow]:
        """Apply fusion across an iterable of aligned score rows."""
        return [self.fuse_single(row) for row in aligned_rows]

    def to_arrow_table(self, rows: Sequence[FusedScoreRow]) -> pa.Table:
        """Serialise fused rows to a PyArrow Table."""
        pylist = []
        for r in rows:
            d = r.to_dict()
            w = r.weights_applied
            d["seq_weight_applied"] = w[0] if w else None
            d["graph_weight_applied"] = w[1] if w else None
            d.pop("weights_applied", None)

            chunk = r.seq_evidence_chunk
            d["seq_evidence_chunk_offset"] = chunk[0] if chunk else None
            d["seq_evidence_chunk_length"] = chunk[1] if chunk else None
            d.pop("seq_evidence_chunk", None)

            pylist.append(d)
        return pa.Table.from_pylist(pylist, schema=FUSED_SCORE_SCHEMA)
