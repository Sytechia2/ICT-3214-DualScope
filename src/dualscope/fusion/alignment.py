"""Output alignment between short-term sequence and long-term graph detectors (Task 5.1).

Enforces causal score availability: at any evaluation timestamp T, a detector
score may be used if and only if score_available_at <= T.
Missing scores are recorded with explicit statuses (never zero-filled).
Graph scores exceeding the staleness limit are flagged and decoupled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pyarrow as pa

from dualscope.fusion.config import AlignmentConfig, AlignmentStatus


@dataclass(frozen=True)
class AlignedScoreRow:
    """A single temporally and causally aligned user-hour scoring unit."""

    user_id: str
    window_start: int
    window_end: int
    dataset_day: int
    split: str | None
    alignment_status: AlignmentStatus

    # Sequence detector view
    seq_status: str | None
    seq_raw_score: float | None
    seq_score: float | None
    seq_threshold: float | None
    is_seq_alert: bool | None

    # Graph detector view
    graph_status: str | None
    graph_raw_score: float | None
    graph_score: float | None
    graph_threshold: float | None
    is_graph_alert: bool | None
    graph_score_available_at: int | None
    graph_age_seconds: int | None

    # Evidence references
    seq_source_lines: list[int] | None = None
    seq_evidence_chunk: tuple[int, int] | None = None
    graph_evidence_nodes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alignment_status"] = self.alignment_status.value
        return data


ALIGNED_SCORE_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("window_start", pa.int64(), nullable=False),
        pa.field("window_end", pa.int64(), nullable=False),
        pa.field("dataset_day", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("alignment_status", pa.string(), nullable=False),
        # Sequence view
        pa.field("seq_status", pa.string(), nullable=True),
        pa.field("seq_raw_score", pa.float64(), nullable=True),
        pa.field("seq_score", pa.float64(), nullable=True),
        pa.field("seq_threshold", pa.float64(), nullable=True),
        pa.field("is_seq_alert", pa.bool_(), nullable=True),
        pa.field("seq_source_lines", pa.list_(pa.int64()), nullable=True),
        pa.field("seq_evidence_chunk_offset", pa.int32(), nullable=True),
        pa.field("seq_evidence_chunk_length", pa.int32(), nullable=True),
        # Graph view
        pa.field("graph_status", pa.string(), nullable=True),
        pa.field("graph_raw_score", pa.float64(), nullable=True),
        pa.field("graph_score", pa.float64(), nullable=True),
        pa.field("graph_threshold", pa.float64(), nullable=True),
        pa.field("is_graph_alert", pa.bool_(), nullable=True),
        pa.field("graph_score_available_at", pa.int64(), nullable=True),
        pa.field("graph_age_seconds", pa.int64(), nullable=True),
        pa.field("graph_evidence_nodes", pa.list_(pa.string()), nullable=True),
    ]
)


class ScoreAlignmentEngine:
    """Aligns sequence and graph detector outputs into a joint scoring matrix."""

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()

    def align_single(
        self,
        user_id: str,
        window_start: int,
        seq_record: Mapping[str, Any] | None,
        graph_record: Mapping[str, Any] | None,
        *,
        split: str | None = None,
        dataset_day: int | None = None,
    ) -> AlignedScoreRow:
        """Align sequence and graph records for one user-hour window [window_start, window_end).

        Causal constraint:
          Evaluation decision time is T = window_end = window_start + hour_seconds.
          A detector score may participate only if score_available_at <= T.
        """
        w_start = int(window_start)
        w_end = w_start + self.config.hour_seconds
        day = dataset_day if dataset_day is not None else int(1 + (w_start - self.config.hour_origin) // 86400)

        # 1. Warm-up check (Day 1)
        is_warmup = w_start < (self.config.hour_origin + self.config.warmup_seconds)

        # 2. Extract sequence fields
        seq_status: str | None = None
        seq_raw: float | None = None
        seq_score: float | None = None
        seq_thresh: float | None = None
        is_seq_alert: bool | None = None
        seq_source_lines: list[int] | None = None
        seq_evidence_chunk: tuple[int, int] | None = None

        if seq_record is not None:
            seq_avail = int(seq_record.get("score_available_at", w_end))
            if seq_avail > w_end:
                raise ValueError(
                    f"Temporal leakage violation: sequence score_available_at ({seq_avail}) > window_end ({w_end})"
                )
            seq_status = str(seq_record.get("status", "available"))
            if seq_status == "available":
                seq_raw = float(seq_record["raw_score"]) if seq_record.get("raw_score") is not None else None
                seq_score = float(seq_record["score"]) if seq_record.get("score") is not None else None
                seq_thresh = (
                    float(seq_record["alert_threshold"])
                    if seq_record.get("alert_threshold") is not None
                    else None
                )
                if seq_record.get("is_alert") is not None:
                    is_seq_alert = bool(seq_record["is_alert"])
                elif seq_score is not None and seq_thresh is not None:
                    is_seq_alert = bool(seq_score >= seq_thresh)

                lines = seq_record.get("source_lines")
                if lines is not None:
                    seq_source_lines = [int(x) for x in lines]
                offset = seq_record.get("evidence_chunk_offset")
                length = seq_record.get("evidence_chunk_length")
                if offset is not None and length is not None:
                    seq_evidence_chunk = (int(offset), int(length))
            elif seq_status == "insufficient_history":
                is_warmup = True

        # 3. Extract graph fields with causal filter and staleness check
        graph_status: str | None = None
        graph_raw: float | None = None
        graph_score: float | None = None
        graph_thresh: float | None = None
        is_graph_alert: bool | None = None
        graph_avail_at: int | None = None
        graph_age: int | None = None
        graph_nodes: list[str] | None = None

        if graph_record is not None:
            avail_at = int(graph_record.get("score_available_at", graph_record.get("window_end", w_end)))
            if avail_at > w_end:
                raise ValueError(
                    f"Temporal leakage violation: graph score_available_at ({avail_at}) > window_end ({w_end})"
                )
            graph_avail_at = avail_at
            graph_age = w_end - avail_at
            base_status = str(graph_record.get("status", "available"))

            if graph_age > self.config.max_graph_staleness_seconds:
                graph_status = AlignmentStatus.STALE_GRAPH.value
            elif base_status == "available":
                graph_status = "available"
                graph_raw = float(graph_record["raw_score"]) if graph_record.get("raw_score") is not None else None
                graph_score = float(graph_record["score"]) if graph_record.get("score") is not None else None
                graph_thresh = (
                    float(graph_record["alert_threshold"])
                    if graph_record.get("alert_threshold") is not None
                    else None
                )
                if graph_record.get("is_alert") is not None:
                    is_graph_alert = bool(graph_record["is_alert"])
                elif graph_score is not None and graph_thresh is not None:
                    is_graph_alert = bool(graph_score >= graph_thresh)

                nodes = graph_record.get("evidence_nodes")
                if nodes is not None:
                    graph_nodes = [str(n) for n in nodes]
            else:
                graph_status = base_status

        # 4. Resolve overall alignment status
        if is_warmup or seq_status == "insufficient_history" or graph_status == "insufficient_history":
            status = AlignmentStatus.INSUFFICIENT_HISTORY
        elif seq_score is not None and graph_score is not None and graph_status == "available":
            status = AlignmentStatus.BOTH_AVAILABLE
        elif seq_score is not None:
            if graph_status == AlignmentStatus.STALE_GRAPH.value:
                status = AlignmentStatus.STALE_GRAPH
            else:
                status = AlignmentStatus.SEQUENCE_ONLY
        elif graph_score is not None and graph_status == "available":
            status = AlignmentStatus.GRAPH_ONLY
        else:
            status = AlignmentStatus.NO_ACTIVITY

        return AlignedScoreRow(
            user_id=user_id,
            window_start=w_start,
            window_end=w_end,
            dataset_day=day,
            split=split,
            alignment_status=status,
            seq_status=seq_status,
            seq_raw_score=seq_raw,
            seq_score=seq_score,
            seq_threshold=seq_thresh,
            is_seq_alert=is_seq_alert,
            seq_source_lines=seq_source_lines,
            seq_evidence_chunk=seq_evidence_chunk,
            graph_status=graph_status,
            graph_raw_score=graph_raw,
            graph_score=graph_score,
            graph_threshold=graph_thresh,
            is_graph_alert=is_graph_alert,
            graph_score_available_at=graph_avail_at,
            graph_age_seconds=graph_age,
            graph_evidence_nodes=graph_nodes,
        )

    def align_records(
        self,
        sequence_records: Iterable[Mapping[str, Any]],
        graph_records: Iterable[Mapping[str, Any]],
        *,
        requested_units: Iterable[tuple[str, int]] | None = None,
        split: str | None = None,
    ) -> list[AlignedScoreRow]:
        """Align sets of sequence and graph records.

        Graph scores are matched causally: for a user-hour window [w_s, w_e), the graph
        record selected is the latest one with score_available_at <= w_e.
        """
        # Index sequence records by (user_id, window_start)
        seq_map: dict[tuple[str, int], Mapping[str, Any]] = {}
        all_units: set[tuple[str, int]] = set()

        for sr in sequence_records:
            u = str(sr["user_id"])
            ws = int(sr["window_start"])
            seq_map[(u, ws)] = sr
            all_units.add((u, ws))

        # Index graph records by user_id sorted chronologically by availability
        graph_by_user: dict[str, list[Mapping[str, Any]]] = {}
        for gr in graph_records:
            u = str(gr["user_id"])
            graph_by_user.setdefault(u, []).append(gr)

        for u in graph_by_user:
            graph_by_user[u].sort(
                key=lambda g: int(g.get("score_available_at", g.get("window_end", 0)))
            )

        # Include explicit requested units if provided
        if requested_units is not None:
            for u, ws in requested_units:
                all_units.add((str(u), int(ws)))

        # Also add units where graph has scores that end at a window boundary
        for u, g_list in graph_by_user.items():
            for gr in g_list:
                # If graph record aligns with an hour window
                ws = gr.get("window_start")
                if ws is not None:
                    all_units.add((u, int(ws)))

        aligned_rows: list[AlignedScoreRow] = []
        for u, ws in sorted(all_units, key=lambda x: (x[1], x[0])):
            we = ws + self.config.hour_seconds
            sr = seq_map.get((u, ws))

            # Find latest graph record with avail_at <= we
            best_gr: Mapping[str, Any] | None = None
            if u in graph_by_user:
                for gr in graph_by_user[u]:
                    avail_at = int(gr.get("score_available_at", gr.get("window_end", we)))
                    if avail_at <= we:
                        best_gr = gr
                    else:
                        break  # since sorted chronologically

            row = self.align_single(u, ws, sr, best_gr, split=split)
            aligned_rows.append(row)

        return aligned_rows

    def to_arrow_table(self, rows: Sequence[AlignedScoreRow]) -> pa.Table:
        """Convert a sequence of AlignedScoreRow instances to a PyArrow Table."""
        pylist = []
        for r in rows:
            d = r.to_dict()
            chunk = r.seq_evidence_chunk
            d["seq_evidence_chunk_offset"] = chunk[0] if chunk else None
            d["seq_evidence_chunk_length"] = chunk[1] if chunk else None
            d.pop("seq_evidence_chunk", None)
            pylist.append(d)
        return pa.Table.from_pylist(pylist, schema=ALIGNED_SCORE_SCHEMA)
