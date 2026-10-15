"""Sequence detector output schema, table builder and reader (Task 3.4).

One row describes one ``(user_id, window)`` user-hour. Every user-hour that
has authentication events gets a row, whatever its status, and callers can
request explicit ``no_activity`` rows for units without events. A score is
present only when ``status == "available"``; missing scores are null, never
zero.

Evidence columns:

* ``source_lines`` — every event of the hour in sequence order. The Task 2.2
  evidence reference of line ``n`` is ``auth.txt:<n>`` and resolves through
  ``dualscope.evidence.AuthenticationEvidenceLookup``.
* ``evidence_chunk_*`` — the slice of ``source_lines`` that best explains the
  score: the highest-mean-error chunk, or for ``max_event`` scoring the chunk
  holding the worst event.
* ``top_events`` — the highest-error events, their error and the input
  feature that contributed most to each.
* ``top_feature_contributions`` — each input feature's share of the evidence
  chunk's reconstruction error.

These are model-deviation evidence, not proof that an event was malicious.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dualscope.sequence.builder import DaySequences, source_reference
from dualscope.sequence.config import DETECTOR_NAME, input_feature_names
from dualscope.sequence.scoring import DayScores
from dualscope.splits import ScoringStatus, SplitConfig, dataset_day


SCHEMA_VERSION = "1.0.0"
STATUS_NO_ACTIVITY = "no_activity"
REFERENCE_FORMAT = "auth.txt:<source_line>"

TOP_EVENT_TYPE = pa.struct(
    [
        ("source_reference", pa.string()),
        ("timestamp", pa.int64()),
        ("position", pa.int32()),
        ("event_error", pa.float32()),
        ("top_feature", pa.string()),
    ]
)
FEATURE_SHARE_TYPE = pa.struct([("feature", pa.string()), ("share", pa.float32())])

SEQUENCE_SCORE_SCHEMA = pa.schema(
    [
        ("detector", pa.string()),
        ("model_version", pa.string()),
        ("run_id", pa.string()),
        ("user_id", pa.string()),
        ("window_start", pa.int64()),
        ("window_end", pa.int64()),
        ("score_available_at", pa.int64()),
        ("dataset_day", pa.int32()),
        ("split", pa.string()),
        ("status", pa.string()),
        ("status_detail", pa.string()),
        ("raw_score", pa.float64()),
        ("score", pa.float64()),
        ("alert_threshold", pa.float64()),
        ("is_alert", pa.bool_()),
        ("aggregation", pa.string()),
        ("n_events", pa.int32()),
        ("n_chunks", pa.int32()),
        ("source_lines", pa.list_(pa.int64())),
        ("evidence_chunk_index", pa.int32()),
        ("evidence_chunk_offset", pa.int32()),
        ("evidence_chunk_length", pa.int32()),
        ("evidence_chunk_error", pa.float64()),
        ("top_events", pa.list_(TOP_EVENT_TYPE)),
        ("top_feature_contributions", pa.list_(FEATURE_SHARE_TYPE)),
    ],
    metadata={
        "schema_version": SCHEMA_VERSION,
        "reference_format": REFERENCE_FORMAT,
        "score_semantics": "0-1 rarity relative to validation user-hours; not an attack probability",
    },
)


def _masked(values: np.ndarray, mask: np.ndarray, kind: pa.DataType) -> pa.Array:
    return pa.array(values, type=kind, mask=~mask)


def _top_events_array(scores: DayScores, available: np.ndarray, k: int, feature_names: list[str]) -> pa.Array:
    """Vectorised top-k events per available user-hour, highest error first."""
    day = scores.day
    counts = day.counts
    group = np.repeat(np.arange(day.n_user_hours), counts)
    position = np.arange(day.n_events) - np.repeat(day.offsets, counts)
    error = scores.event_error
    usable = np.flatnonzero(~np.isnan(error) & available[group])
    order = usable[np.lexsort((position[usable], -error[usable], group[usable]))]
    sorted_group = group[order]
    first = np.r_[True, sorted_group[1:] != sorted_group[:-1]] if len(order) else np.zeros(0, bool)
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0)) if len(order) else order
    keep = order[(np.arange(len(order)) - group_start) < k]
    per_group = np.bincount(group[keep], minlength=day.n_user_hours)
    offsets = np.r_[0, np.cumsum(per_group)].astype(np.int32)
    names = np.asarray(feature_names, dtype=object)
    values = pa.StructArray.from_arrays(
        [
            pa.array([source_reference(line) for line in day.source_lines[keep]], type=pa.string()),
            pa.array(day.timestamps[keep], type=pa.int64()),
            pa.array(position[keep], type=pa.int32()),
            pa.array(error[keep], type=pa.float32()),
            pa.array(names[scores.event_top_feature[keep]], type=pa.string()),
        ],
        fields=list(TOP_EVENT_TYPE),
    )
    return pa.ListArray.from_arrays(pa.array(offsets), values)


def _feature_share_array(
    scores: DayScores, evidence: np.ndarray, available: np.ndarray, k: int, feature_names: list[str]
) -> pa.Array:
    rows = np.flatnonzero(available)
    per_row = np.zeros(scores.day.n_user_hours, dtype=np.int64)
    if len(rows) == 0:
        empty = pa.StructArray.from_arrays([pa.array([], pa.string()), pa.array([], pa.float32())], fields=list(FEATURE_SHARE_TYPE))
        return pa.ListArray.from_arrays(pa.array(np.zeros(len(per_row) + 1, dtype=np.int32)), empty)
    feature = scores.chunk_feature_mean[evidence[rows]].astype(np.float64)
    share = feature / np.maximum(feature.sum(axis=1, keepdims=True), 1e-12)
    k = min(k, share.shape[1])
    top = np.argsort(-share, axis=1, kind="stable")[:, :k]
    per_row[rows] = k
    offsets = np.r_[0, np.cumsum(per_row)].astype(np.int32)
    names = np.asarray(feature_names, dtype=object)
    values = pa.StructArray.from_arrays(
        [
            pa.array(names[top.ravel()], type=pa.string()),
            pa.array(np.take_along_axis(share, top, axis=1).ravel(), type=pa.float32()),
        ],
        fields=list(FEATURE_SHARE_TYPE),
    )
    return pa.ListArray.from_arrays(pa.array(offsets), values)


def build_score_table(
    scores: DayScores,
    normalised: np.ndarray,
    raw: np.ndarray,
    *,
    model_version: str,
    run_id: str,
    aggregation: str,
    alert_threshold: float,
    hour_seconds: int,
    top_events: int,
    top_features: int,
) -> pa.Table:
    """Assemble the detector-output rows for one scored day."""
    day = scores.day
    n = day.n_user_hours
    available = day.status == ScoringStatus.AVAILABLE.value
    if np.any(np.isnan(raw[available])) or np.any(~np.isnan(raw[~available])):
        raise ValueError("raw scores must be present exactly for available user-hours")
    names = input_feature_names()
    if len(names) != scores.chunk_feature_mean.shape[1]:
        raise ValueError("export supports only the default sequence inputs; feature names do not match the model")
    evidence = scores.evidence_chunks(aggregation)
    has_chunk = available & (evidence >= 0)
    if np.any(available & ~has_chunk):
        raise ValueError("an available user-hour has no evidence chunk")
    chunk_rows = np.where(has_chunk, evidence, 0)
    chunk_index = scores.chunks.chunk_index[chunk_rows] if len(scores.chunks) else np.zeros(n, np.int32)
    chunk_offset = (scores.chunks.start[chunk_rows] - day.offsets) if len(scores.chunks) else np.zeros(n, np.int64)
    chunk_length = scores.chunks.length[chunk_rows] if len(scores.chunks) else np.zeros(n, np.int32)
    chunk_error = scores.chunk_mean[chunk_rows] if len(scores.chunks) else np.zeros(n)
    line_offsets = np.r_[day.offsets, day.n_events].astype(np.int32)

    window_start = day.hour_starts.astype(np.int64)
    window_end = window_start + hour_seconds
    columns = {
        "detector": pa.array([DETECTOR_NAME] * n, pa.string()),
        "model_version": pa.array([model_version] * n, pa.string()),
        "run_id": pa.array([run_id] * n, pa.string()),
        "user_id": pa.array(day.users, pa.string()),
        "window_start": pa.array(window_start, pa.int64()),
        "window_end": pa.array(window_end, pa.int64()),
        "score_available_at": pa.array(window_end, pa.int64()),
        "dataset_day": pa.array(np.full(n, day.dataset_day), pa.int32()),
        "split": pa.array([day.split] * n, pa.string()),
        "status": pa.array(day.status, pa.string()),
        "status_detail": pa.array(day.status_detail, pa.string()),
        "raw_score": _masked(raw, available, pa.float64()),
        "score": _masked(normalised, available, pa.float64()),
        "alert_threshold": pa.array(np.full(n, alert_threshold), pa.float64()),
        "is_alert": _masked(np.nan_to_num(normalised, nan=-1.0) >= alert_threshold, available, pa.bool_()),
        "aggregation": pa.array([aggregation] * n, pa.string()),
        "n_events": pa.array(day.counts, pa.int32()),
        "n_chunks": pa.array(scores.n_chunks, pa.int32()),
        "source_lines": pa.ListArray.from_arrays(pa.array(line_offsets), pa.array(day.source_lines, pa.int64())),
        "evidence_chunk_index": _masked(chunk_index, has_chunk, pa.int32()),
        "evidence_chunk_offset": _masked(chunk_offset, has_chunk, pa.int32()),
        "evidence_chunk_length": _masked(chunk_length, has_chunk, pa.int32()),
        "evidence_chunk_error": _masked(chunk_error, has_chunk, pa.float64()),
        "top_events": _top_events_array(scores, available, top_events, names),
        "top_feature_contributions": _feature_share_array(scores, evidence, available, top_features, names),
    }
    return pa.Table.from_arrays([columns[f.name] for f in SEQUENCE_SCORE_SCHEMA], schema=SEQUENCE_SCORE_SCHEMA)


def no_activity_rows(
    units: Iterable[tuple[str, int]],
    split_cfg: SplitConfig,
    *,
    model_version: str,
    run_id: str,
    aggregation: str,
    alert_threshold: float,
    hour_seconds: int,
) -> pa.Table:
    """Explicit rows for requested user-hours that had no authentication events."""
    units = sorted(set((str(u), int(h)) for u, h in units), key=lambda x: (x[1], x[0]))
    rows = []
    for user, hour in units:
        split = split_cfg.find_split_for_timestamp(hour)
        rows.append(
            {
                "detector": DETECTOR_NAME,
                "model_version": model_version,
                "run_id": run_id,
                "user_id": user,
                "window_start": hour,
                "window_end": hour + hour_seconds,
                "score_available_at": hour + hour_seconds,
                "dataset_day": dataset_day(hour),
                "split": split.name if split else None,
                "status": STATUS_NO_ACTIVITY,
                "status_detail": "no authentication events by this acting user in the window",
                "raw_score": None,
                "score": None,
                "alert_threshold": alert_threshold,
                "is_alert": None,
                "aggregation": aggregation,
                "n_events": 0,
                "n_chunks": 0,
                "source_lines": [],
                "evidence_chunk_index": None,
                "evidence_chunk_offset": None,
                "evidence_chunk_length": None,
                "evidence_chunk_error": None,
                "top_events": [],
                "top_feature_contributions": [],
            }
        )
    return pa.Table.from_pylist(rows, schema=SEQUENCE_SCORE_SCHEMA)


def write_day_table(table: pa.Table, output_dir: str | Path, day: int) -> Path:
    """Write one day partition (``dataset_day=NN/part-000000.parquet``)."""
    path = Path(output_dir) / f"dataset_day={day:02d}" / "part-000000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.drop_columns(["dataset_day"]), path, compression="zstd")
    return path


def to_detector_record(row: dict[str, Any]) -> dict[str, Any]:
    """JSON-ready record with explicit ``source_references`` for downstream consumers."""
    record = dict(row)
    lines = record.pop("source_lines") or []
    record["source_references"] = [source_reference(line) for line in lines]
    if record.get("evidence_chunk_offset") is not None:
        start = record["evidence_chunk_offset"]
        record["evidence_source_references"] = record["source_references"][start : start + record["evidence_chunk_length"]]
    else:
        record["evidence_source_references"] = []
    record["reference_format"] = REFERENCE_FORMAT
    return record


class SequenceScoreStore:
    """Read exported user-hour scores and resolve their evidence references."""

    def __init__(self, scores_dir: str | Path) -> None:
        self.scores_dir = Path(scores_dir)
        self.dataset = ds.dataset(str(self.scores_dir), format="parquet", partitioning="hive")

    def table(self, columns: list[str] | None = None, filter: ds.Expression | None = None) -> pa.Table:
        return self.dataset.to_table(columns=columns, filter=filter)

    def get(self, user_id: str, window_start: int) -> dict[str, Any] | None:
        """Return one user-hour record, or ``None`` if no row exists for it."""
        table = self.table(
            filter=(ds.field("dataset_day") == dataset_day(window_start))
            & (ds.field("window_start") == window_start)
            & (ds.field("user_id") == user_id)
        )
        if table.num_rows == 0:
            return None
        if table.num_rows > 1:
            raise ValueError(f"duplicate score rows for {user_id} at {window_start}")
        return to_detector_record(table.to_pylist()[0])

    def alerts(self, columns: list[str] | None = None) -> pa.Table:
        return self.table(columns=columns, filter=ds.field("is_alert") == True)  # noqa: E712


def write_json_records(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_detector_record(row), sort_keys=True) + "\n")


def status_counts(table: pa.Table) -> dict[str, int]:
    values = pc.value_counts(table["status"]).to_pylist()
    return {item["values"]: int(item["counts"]) for item in values}
