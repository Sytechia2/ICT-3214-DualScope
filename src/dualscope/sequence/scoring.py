"""Frozen-model inference over user-hour sequences (Tasks 3.2-3.4).

Scoring never updates weights. Each chunk receives per-event, per-feature
reconstruction losses; these are reduced to:

* a chunk score (mean event error over the chunk);
* three candidate user-hour raw scores, chosen between on validation data:
  ``max_chunk_mean`` (most anomalous chunk), ``max_event`` (single worst
  event) and ``hour_mean`` (mean over every event in the hour);
* evidence: the highest-error chunk, the top events by error, and each
  input feature's share of the evidence chunk's error.

Evidence describes where the model's reconstruction deviated; it is not proof
that an event was malicious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dualscope.sequence.builder import ChunkIndex, DaySequences, build_chunks, gather_padded
from dualscope.sequence.config import AGGREGATIONS
from dualscope.sequence.model import GRUSequenceAutoencoder, per_feature_errors


@dataclass
class DayScores:
    """Per-user-hour detector results for one day; NaN where not scorable."""

    day: DaySequences
    chunks: ChunkIndex
    chunk_mean: np.ndarray
    chunk_feature_mean: np.ndarray
    event_error: np.ndarray
    event_top_feature: np.ndarray
    raw: dict[str, np.ndarray]
    n_chunks: np.ndarray
    max_mean_chunk: np.ndarray
    max_event_chunk: np.ndarray

    def evidence_chunks(self, aggregation: str) -> np.ndarray:
        """Chunk row per user-hour that best explains the score (-1 when not scored).

        ``max_event`` points at the chunk holding the worst event; the other
        aggregations point at the chunk with the highest mean error.
        """
        return self.max_event_chunk if aggregation == "max_event" else self.max_mean_chunk


@torch.no_grad()
def score_chunks(
    model: GRUSequenceAutoencoder,
    dense: np.ndarray,
    categorical: np.ndarray,
    chunks: ChunkIndex,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score chunks; return chunk mean errors, chunk feature means, event errors, event top features.

    Event-level outputs are indexed like ``dense``; events outside any chunk stay NaN / -1.
    """
    model.eval()
    n_features = model.spec.n_features
    chunk_mean = np.full(len(chunks), np.nan, dtype=np.float64)
    chunk_feature = np.full((len(chunks), n_features), np.nan, dtype=np.float32)
    event_error = np.full(len(dense), np.nan, dtype=np.float32)
    event_top = np.full(len(dense), -1, dtype=np.int8)
    order = np.argsort(chunks.length, kind="stable")
    for begin in range(0, len(order), batch_size):
        idx = order[begin : begin + batch_size]
        starts, lengths = chunks.start[idx], chunks.length[idx]
        d, c, mask = gather_padded(dense, categorical, starts, lengths)
        errors = per_feature_errors(
            model, torch.from_numpy(d), torch.from_numpy(c), torch.from_numpy(lengths.astype(np.int64))
        ).numpy()
        per_event = errors.mean(axis=-1)
        valid_len = lengths.astype(np.float64)
        chunk_mean[idx] = per_event.sum(axis=1, dtype=np.float64) / valid_len
        chunk_feature[idx] = (errors.sum(axis=1, dtype=np.float64) / valid_len[:, None]).astype(np.float32)
        rows, cols = np.nonzero(mask)
        positions = starts[rows] + cols
        event_error[positions] = per_event[rows, cols]
        event_top[positions] = errors[rows, cols].argmax(axis=-1)
    if not np.all(np.isfinite(chunk_mean)):
        raise FloatingPointError("non-finite sequence reconstruction error")
    return chunk_mean, chunk_feature, event_error, event_top


def score_day(
    model: GRUSequenceAutoencoder,
    day: DaySequences,
    max_length: int,
    batch_size: int = 2048,
) -> DayScores:
    """Score every available user-hour in a day."""
    scorable = np.flatnonzero(day.scorable_mask())
    chunks = build_chunks(day.offsets, day.counts, max_length, scorable)
    chunk_mean, chunk_feature, event_error, event_top = score_chunks(
        model, day.numeric, day.categorical, chunks, batch_size
    )

    n = day.n_user_hours
    raw = {name: np.full(n, np.nan, dtype=np.float64) for name in AGGREGATIONS}
    n_chunks = np.zeros(n, dtype=np.int32)
    max_mean_chunk = np.full(n, -1, dtype=np.int64)
    max_event_chunk = np.full(n, -1, dtype=np.int64)
    if len(chunks):
        # Chunks of one user-hour are contiguous, in chunk order, and their
        # starts increase through the day's event array.
        first = np.flatnonzero(np.r_[True, chunks.user_hour[1:] != chunks.user_hour[:-1]])
        owners = chunks.user_hour[first]
        n_chunks[owners] = np.diff(np.r_[first, len(chunks)])
        raw["max_chunk_mean"][owners] = np.maximum.reduceat(chunk_mean, first)
        best = np.lexsort((chunks.chunk_index, -chunk_mean, chunks.user_hour))
        max_mean_chunk[owners] = best[first]

        filled = np.where(np.isnan(event_error), 0.0, event_error).astype(np.float64)
        highest = np.where(np.isnan(event_error), -np.inf, event_error).astype(np.float64)
        hour_mean = np.add.reduceat(filled, day.offsets) / day.counts
        hour_max = np.maximum.reduceat(highest, day.offsets)
        raw["hour_mean"][scorable] = hour_mean[scorable]
        raw["max_event"][scorable] = hour_max[scorable]

        group = np.repeat(np.arange(n), day.counts)
        candidates = np.flatnonzero((highest == hour_max[group]) & np.isfinite(highest))
        owner, first_hit = np.unique(group[candidates], return_index=True)
        worst_event = candidates[first_hit]  # earliest event with the hour's maximum error
        max_event_chunk[owner] = np.searchsorted(chunks.start, worst_event, side="right") - 1
    return DayScores(
        day=day,
        chunks=chunks,
        chunk_mean=chunk_mean,
        chunk_feature_mean=chunk_feature,
        event_error=event_error,
        event_top_feature=event_top,
        raw=raw,
        n_chunks=n_chunks,
        max_mean_chunk=max_mean_chunk,
        max_event_chunk=max_event_chunk,
    )
