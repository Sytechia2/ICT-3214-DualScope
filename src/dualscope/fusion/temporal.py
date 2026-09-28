"""Temporal fusion engine for multi-timescale intrusion detection (Task 5.3).

Combines base detector scores with a time-decayed co-occurrence bonus when
short-term sequence anomalies and long-term graph structural shifts co-occur
within a temporal lookback window.
Guarantees strict causality: future alerts never retroactively modify earlier scores.
Tracks early-warning lead-lag precedence between detectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence
import pyarrow as pa

from dualscope.fusion.alignment import AlignedScoreRow
from dualscope.fusion.basic import BasicFusionEngine, FusedScoreRow, classify_disagreement
from dualscope.fusion.config import (
    BasicFusionConfig,
    DisagreementType,
    FusionMethod,
    TemporalConfig,
)


@dataclass(frozen=True)
class TemporalFusedScoreRow:
    """The result of applying temporal proximity fusion to an aligned scoring unit."""

    user_id: str
    window_start: int
    window_end: int
    dataset_day: int
    split: str | None

    # Scores
    base_fused_score: float | None
    temporal_boost: float
    fused_score: float | None
    is_fused_alert: bool | None
    fusion_method: FusionMethod
    disagreement_type: DisagreementType

    # Lead-lag diagnostics
    lead_detector: str | None  # "sequence" | "graph" | "simultaneous" | None
    lead_time_seconds: int | None

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


TEMPORAL_FUSED_SCORE_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("window_start", pa.int64(), nullable=False),
        pa.field("window_end", pa.int64(), nullable=False),
        pa.field("dataset_day", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("base_fused_score", pa.float64(), nullable=True),
        pa.field("temporal_boost", pa.float64(), nullable=False),
        pa.field("fused_score", pa.float64(), nullable=True),
        pa.field("is_fused_alert", pa.bool_(), nullable=True),
        pa.field("fusion_method", pa.string(), nullable=False),
        pa.field("disagreement_type", pa.string(), nullable=False),
        pa.field("lead_detector", pa.string(), nullable=True),
        pa.field("lead_time_seconds", pa.int64(), nullable=True),
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


class TemporalFusionEngine:
    """Multi-timescale temporal fusion engine incorporating temporal correlation."""

    def __init__(
        self,
        temporal_config: TemporalConfig | None = None,
        base_config: BasicFusionConfig | None = None,
    ) -> None:
        self.temporal_config = temporal_config or TemporalConfig()
        self.base_config = base_config or BasicFusionConfig(method=FusionMethod.WEIGHTED)
        self.base_engine = BasicFusionEngine(self.base_config)

    def compute_temporal_boost(
        self,
        t_curr: int,
        is_seq_active: bool,
        is_graph_active: bool,
        seq_alert_history: Sequence[int],
        graph_alert_history: Sequence[int],
    ) -> tuple[float, str | None, int | None]:
        """Compute the temporal co-occurrence boost and lead-lag diagnostics.

        Parameters
        ----------
        t_curr : int
            Current window end timestamp.
        is_seq_active : bool
            Whether sequence detector has an alert in the current window.
        is_graph_active : bool
            Whether graph detector has an alert in the current window.
        seq_alert_history : Sequence[int]
            Timestamps of strictly prior sequence alerts (t <= t_curr).
        graph_alert_history : Sequence[int]
            Timestamps of strictly prior graph alerts (t <= t_curr).

        Returns
        -------
        boost : float
            Value in [0, boost_weight].
        lead_detector : str | None
            "sequence", "graph", "simultaneous", or None.
        lead_time_seconds : int | None
            Time difference between first and second alerting detectors.
        """
        w_max = self.temporal_config.lookback_seconds
        tau = self.temporal_config.half_life_seconds
        gamma = self.temporal_config.boost_weight

        # Filter history to lookback window: [t_curr - w_max, t_curr]
        recent_seq = [t for t in seq_alert_history if (t_curr - w_max) <= t <= t_curr]
        recent_graph = [t for t in graph_alert_history if (t_curr - w_max) <= t <= t_curr]

        if is_seq_active and t_curr not in recent_seq:
            recent_seq.append(t_curr)
        if is_graph_active and t_curr not in recent_graph:
            recent_graph.append(t_curr)

        # A temporal co-occurrence requires evidence from BOTH detectors within the lookback window
        if not recent_seq or not recent_graph:
            return 0.0, None, None

        # Calculate minimum delta t between any sequence and graph alert in window
        min_delta = min(abs(s - g) for s in recent_seq for g in recent_graph)
        decay = math.exp(-min_delta / tau)
        boost = gamma * decay

        # Determine lead-lag precedence
        earliest_seq = min(recent_seq)
        earliest_graph = min(recent_graph)

        if earliest_seq < earliest_graph:
            lead_det = "sequence"
            lead_time = earliest_graph - earliest_seq
        elif earliest_graph < earliest_seq:
            lead_det = "graph"
            lead_time = earliest_seq - earliest_graph
        else:
            lead_det = "simultaneous"
            lead_time = 0

        return boost, lead_det, lead_time

    def fuse_stream(
        self, aligned_rows: Iterable[AlignedScoreRow]
    ) -> list[TemporalFusedScoreRow]:
        """Process aligned rows causally in chronological order per user."""
        # Group rows by user while tracking order
        rows_by_user: dict[str, list[AlignedScoreRow]] = {}
        for row in aligned_rows:
            rows_by_user.setdefault(row.user_id, []).append(row)

        all_results: list[TemporalFusedScoreRow] = []

        for user_id, user_rows in rows_by_user.items():
            # Sort chronologically by window_end
            user_rows.sort(key=lambda r: (r.window_end, r.window_start))

            seq_history: list[int] = []
            graph_history: list[int] = []

            for row in user_rows:
                # 1. Compute basic fusion score
                base_fused = self.base_engine.fuse_single(row)
                s_base = base_fused.fused_score

                is_seq_alert = bool(row.is_seq_alert)
                is_graph_alert = bool(row.is_graph_alert)
                t_avail = row.window_end

                # 2. Compute temporal boost using historical and current alerts
                boost, lead_det, lead_time = self.compute_temporal_boost(
                    t_curr=t_avail,
                    is_seq_active=is_seq_alert,
                    is_graph_active=is_graph_alert,
                    seq_alert_history=seq_history,
                    graph_alert_history=graph_history,
                )

                # 3. Apply boost to base score if score is available
                fused_score: float | None = None
                is_alert: bool | None = None

                if s_base is not None:
                    unbounded_score = s_base + boost
                    fused_score = min(unbounded_score, self.temporal_config.max_fused_score_cap)
                    is_alert = bool(fused_score >= self.temporal_config.fused_alert_threshold)

                # 4. Update causal history for future windows
                if is_seq_alert:
                    seq_history.append(t_avail)
                if is_graph_alert:
                    graph_history.append(t_avail)

                all_results.append(
                    TemporalFusedScoreRow(
                        user_id=row.user_id,
                        window_start=row.window_start,
                        window_end=row.window_end,
                        dataset_day=row.dataset_day,
                        split=row.split,
                        base_fused_score=s_base,
                        temporal_boost=boost,
                        fused_score=fused_score,
                        is_fused_alert=is_alert,
                        fusion_method=FusionMethod.TEMPORAL,
                        disagreement_type=base_fused.disagreement_type,
                        lead_detector=lead_det,
                        lead_time_seconds=lead_time,
                        seq_score=row.seq_score,
                        is_seq_alert=row.is_seq_alert,
                        graph_score=row.graph_score,
                        is_graph_alert=row.is_graph_alert,
                        alignment_status=row.alignment_status.value,
                        seq_source_lines=row.seq_source_lines,
                        seq_evidence_chunk=row.seq_evidence_chunk,
                        graph_evidence_nodes=row.graph_evidence_nodes,
                    )
                )

        # Restore original chronological order
        all_results.sort(key=lambda r: (r.window_start, r.user_id))
        return all_results

    def to_arrow_table(self, rows: Sequence[TemporalFusedScoreRow]) -> pa.Table:
        """Serialise temporal fused rows to a PyArrow Table."""
        pylist = []
        for r in rows:
            d = r.to_dict()
            chunk = r.seq_evidence_chunk
            d["seq_evidence_chunk_offset"] = chunk[0] if chunk else None
            d["seq_evidence_chunk_length"] = chunk[1] if chunk else None
            d.pop("seq_evidence_chunk", None)
            pylist.append(d)
        return pa.Table.from_pylist(pylist, schema=TEMPORAL_FUSED_SCORE_SCHEMA)
