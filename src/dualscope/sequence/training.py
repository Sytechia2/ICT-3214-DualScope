"""Training-sample construction, GRU autoencoder fitting and checkpoints (Task 3.2).

Training chunks come only from fitting-eligible training user-hours (after the
24-hour warm-up, and with no event touching a training-excluded user). Up to
``max_chunks_per_user_hour`` chunks are taken from any one hour so a few very
busy accounts cannot dominate, then a uniform random sample of the remaining
candidates is kept. A label-free validation sample monitors reconstruction
loss for checkpoint selection; red-team labels are never read here.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from dualscope.sequence.builder import DaySequences, build_chunks, gather_padded
from dualscope.sequence.config import (
    CATEGORICAL_VOCABULARY_COLUMNS,
    SEQUENCE_BINARY_INPUTS,
    SEQUENCE_CATEGORICAL_INPUTS,
    SEQUENCE_NUMERIC_INPUTS,
    TrainingSettings,
)
from dualscope.sequence.model import (
    GRUSequenceAutoencoder,
    ModelSpec,
    masked_sequence_loss,
    per_feature_errors,
)


CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class SequenceSample:
    """A compact set of chunks copied out of one or more days."""

    dense: np.ndarray
    categorical: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    keys: np.ndarray

    def __len__(self) -> int:
        return int(len(self.starts))

    @property
    def n_events(self) -> int:
        return int(self.lengths.sum())


def _empty_sample() -> SequenceSample:
    n_dense = len(SEQUENCE_NUMERIC_INPUTS) + len(SEQUENCE_BINARY_INPUTS)
    return SequenceSample(
        dense=np.zeros((0, n_dense), dtype=np.float32),
        categorical=np.zeros((0, len(SEQUENCE_CATEGORICAL_INPUTS)), dtype=np.int16),
        starts=np.zeros(0, dtype=np.int64),
        lengths=np.zeros(0, dtype=np.int64),
        keys=np.zeros(0, dtype=np.float64),
    )


def _extract(day: DaySequences, starts: np.ndarray, lengths: np.ndarray, keys: np.ndarray) -> SequenceSample:
    lengths = lengths.astype(np.int64)
    total = int(lengths.sum())
    new_starts = np.cumsum(lengths) - lengths
    index = np.repeat(starts.astype(np.int64) - new_starts, lengths) + np.arange(total, dtype=np.int64)
    return SequenceSample(
        dense=day.numeric[index],
        categorical=day.categorical[index],
        starts=new_starts,
        lengths=lengths,
        keys=keys,
    )


def _concat(samples: Sequence[SequenceSample]) -> SequenceSample:
    samples = [s for s in samples if len(s)]
    if not samples:
        return _empty_sample()
    shift = np.cumsum([0] + [s.n_events for s in samples[:-1]])
    return SequenceSample(
        dense=np.concatenate([s.dense for s in samples]),
        categorical=np.concatenate([s.categorical for s in samples]),
        starts=np.concatenate([s.starts + off for s, off in zip(samples, shift)]),
        lengths=np.concatenate([s.lengths for s in samples]),
        keys=np.concatenate([s.keys for s in samples]),
    )


def _keep_smallest_keys(sample: SequenceSample, limit: int) -> SequenceSample:
    if len(sample) <= limit:
        return sample
    keep = np.sort(np.argpartition(sample.keys, limit - 1)[:limit])
    starts, lengths = sample.starts[keep], sample.lengths[keep]
    total = int(lengths.sum())
    new_starts = np.cumsum(lengths) - lengths
    index = np.repeat(starts - new_starts, lengths) + np.arange(total, dtype=np.int64)
    return SequenceSample(
        dense=sample.dense[index],
        categorical=sample.categorical[index],
        starts=new_starts,
        lengths=lengths,
        keys=sample.keys[keep],
    )


def sample_chunks(
    days: Iterable[DaySequences],
    max_length: int,
    limit: int,
    max_chunks_per_user_hour: int,
    rng: np.random.Generator,
    select: Callable[[DaySequences], np.ndarray],
) -> tuple[SequenceSample, dict[str, int]]:
    """Uniformly sample ``limit`` chunks across days (bottom-k of random keys).

    ``select`` returns the boolean mask of user-hours that may contribute.
    Returns the sample and candidate accounting.
    """
    kept = _empty_sample()
    stats = {"days": 0, "user_hours_selected": 0, "chunks_total": 0, "chunk_candidates": 0}
    for day in days:
        selected = np.flatnonzero(select(day))
        stats["days"] += 1
        stats["user_hours_selected"] += int(len(selected))
        if len(selected) == 0:
            continue
        chunks = build_chunks(day.offsets, day.counts, max_length, selected)
        stats["chunks_total"] += len(chunks)
        # Randomly rank chunks inside each hour; keep at most N per hour.
        within_rank = rng.random(len(chunks))
        order = np.lexsort((within_rank, chunks.user_hour))
        group_first = np.r_[True, chunks.user_hour[order][1:] != chunks.user_hour[order][:-1]]
        group_start = np.maximum.accumulate(np.where(group_first, np.arange(len(order)), 0))
        rank_in_group = np.arange(len(order)) - group_start
        candidates = np.sort(order[rank_in_group < max_chunks_per_user_hour])
        stats["chunk_candidates"] += int(len(candidates))
        keys = rng.random(len(candidates))
        if len(candidates) > limit:
            best = np.argpartition(keys, limit - 1)[:limit]
            candidates, keys = candidates[best], keys[best]
        day_sample = _extract(day, chunks.start[candidates], chunks.length[candidates], keys)
        kept = _keep_smallest_keys(_concat([kept, day_sample]), limit)
    stats["chunks_sampled"] = len(kept)
    stats["events_sampled"] = kept.n_events
    return kept, stats


def categorical_cardinalities(preprocessing: dict[str, Any]) -> tuple[int, ...]:
    """Embedding table sizes from a frozen Task 2.4 preprocessing artifact."""
    vocabularies = preprocessing["categorical_vocabularies"]
    sizes = []
    for column in SEQUENCE_CATEGORICAL_INPUTS:
        vocab = vocabularies[CATEGORICAL_VOCABULARY_COLUMNS[column]]
        ids = list(vocab["category_to_id"].values()) + [vocab["unseen_id"], vocab["missing_id"]]
        sizes.append(int(max(ids)) + 1)
    return tuple(sizes)


def _to_tensors(dense: np.ndarray, categorical: np.ndarray, mask: np.ndarray, device: torch.device | str = "cpu"):
    return (
        torch.from_numpy(dense).to(device),
        torch.from_numpy(categorical).to(device),
        torch.from_numpy(mask.sum(axis=1).astype(np.int64)).to(device),
    )


def _length_bucketed_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle, sort within blocks of 50 batches by length, then shuffle batch order."""
    order = rng.permutation(len(lengths))
    block = batch_size * 50
    batches = []
    for begin in range(0, len(order), block):
        part = order[begin : begin + block]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def mean_reconstruction_loss(model: GRUSequenceAutoencoder, sample: SequenceSample, batch_size: int = 2048) -> float:
    """Event-weighted mean reconstruction error over a sample (no weight updates)."""
    model.eval()
    device = next(model.parameters()).device
    order = np.argsort(sample.lengths, kind="stable")
    total, events = 0.0, 0
    for begin in range(0, len(order), batch_size):
        idx = order[begin : begin + batch_size]
        dense, cat, mask = gather_padded(sample.dense, sample.categorical, sample.starts[idx], sample.lengths[idx])
        d, c, lengths = _to_tensors(dense, cat, mask, device)
        errors = per_feature_errors(model, d, c, lengths)
        total += float(errors.mean(dim=-1).sum())
        events += int(lengths.sum())
    return total / max(events, 1)


@torch.no_grad()
def mean_feature_errors(model: GRUSequenceAutoencoder, sample: SequenceSample, batch_size: int = 2048) -> np.ndarray:
    """Mean per-feature reconstruction loss over a sample's events (no weight updates)."""
    model.eval()
    device = next(model.parameters()).device
    order = np.argsort(sample.lengths, kind="stable")
    total = np.zeros(model.spec.n_features, dtype=np.float64)
    events = 0
    for begin in range(0, len(order), batch_size):
        idx = order[begin : begin + batch_size]
        dense, cat, mask = gather_padded(sample.dense, sample.categorical, sample.starts[idx], sample.lengths[idx])
        d, c, lengths = _to_tensors(dense, cat, mask, device)
        total += per_feature_errors(model, d, c, lengths).sum(dim=(0, 1)).double().cpu().numpy()
        events += int(lengths.sum())
    return total / max(events, 1)


def train_autoencoder(
    spec: ModelSpec,
    train: SequenceSample,
    monitor: SequenceSample,
    settings: TrainingSettings,
    log: Callable[[dict[str, Any]], None] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[GRUSequenceAutoencoder, list[dict[str, Any]]]:
    """Fit the autoencoder and return the epoch with the lowest monitor loss.

    ``device`` only selects where training runs (``"cuda"`` is faster but not
    bit-identical to CPU); the returned model is always on the CPU.
    """
    if len(train) == 0:
        raise ValueError("no training sequences")
    if settings.torch_threads > 0:
        torch.set_num_threads(settings.torch_threads)
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    model = GRUSequenceAutoencoder(spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    history: list[dict[str, Any]] = []
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    initial_monitor = mean_reconstruction_loss(model, monitor) if len(monitor) else float("nan")
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        started = time.time()
        running, events = 0.0, 0
        for batch in _length_bucketed_batches(train.lengths, settings.batch_size, rng):
            dense, cat, mask = gather_padded(train.dense, train.categorical, train.starts[batch], train.lengths[batch])
            d, c, lengths = _to_tensors(dense, cat, mask, device)
            loss = masked_sequence_loss(per_feature_errors(model, d, c, lengths), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if settings.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            n = int(lengths.sum())
            running += loss.item() * n
            events += n
        train_loss = running / max(events, 1)
        monitor_loss = mean_reconstruction_loss(model, monitor) if len(monitor) else train_loss
        if not (math.isfinite(train_loss) and math.isfinite(monitor_loss)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "monitor_loss": monitor_loss,
            "initial_monitor_loss": initial_monitor,
            "epoch_seconds": round(time.time() - started, 2),
        }
        history.append(record)
        if log is not None:
            log(record)
        if monitor_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = monitor_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.early_stopping_patience:
                break
    assert best_state is not None
    model = model.to("cpu")
    model.load_state_dict(best_state)
    model.eval()
    for record in history:
        record["selected"] = record["epoch"] == best_epoch
    return model, history


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: GRUSequenceAutoencoder, metadata: dict[str, Any]) -> str:
    """Save weights, architecture and JSON-compatible metadata; return the file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_spec": model.spec.to_dict(),
            "model_state": model.state_dict(),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        },
        path,
    )
    return file_sha256(path)


def load_checkpoint(path: str | Path, expected_sha256: str | None = None) -> tuple[GRUSequenceAutoencoder, dict[str, Any]]:
    """Rebuild a model in evaluation mode from a checkpoint (weights-only load)."""
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"checkpoint hash mismatch: expected {expected_sha256}, found {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')}")
    model = GRUSequenceAutoencoder(ModelSpec.from_dict(payload["model_spec"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, json.loads(payload["metadata_json"])


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
    }
