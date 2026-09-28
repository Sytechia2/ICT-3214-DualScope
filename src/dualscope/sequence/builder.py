"""User-hour sequence construction for the short-term detector (Task 3.1).

Events from the Task 2.4 transformed feature table are grouped by the agreed
acting-user identity and dataset hour, then ordered by ``(timestamp,
source_line)``. Hours longer than the maximum sequence length are divided into
balanced consecutive chunks, so no event is discarded. Every user-hour keeps
its source lines, and every chunk is a contiguous slice of its hour, so model
arrays always map back to ``auth.txt:<source_line>`` evidence references.

Arrays are built one dataset day at a time. Dataset hours never cross a day,
and the split policy uses whole days, so a user-hour never crosses a split
boundary; this is verified for every event rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from dualscope.sequence.config import (
    SEQUENCE_BINARY_INPUTS,
    SEQUENCE_CATEGORICAL_INPUTS,
    SEQUENCE_NUMERIC_INPUTS,
    SequencePolicy,
)
from dualscope.splits import (
    LABEL_COLUMN_NAMES,
    ScoringStatus,
    SequenceCandidateRejectionTracker,
    SplitConfig,
)


STATUS_DETAIL_WARMUP = "warmup_incomplete_24h_history"
STATUS_DETAIL_TOO_FEW = "fewer_than_min_events"

_BASE_COLUMNS = ("timestamp", "source_line", "source_reference", "acting_user", "history_complete_24h")
_ELIGIBILITY_COLUMNS = ("source_user", "destination_user")


def source_reference(source_line: int) -> str:
    """Return the Task 2.2 evidence reference for a source line."""
    return f"auth.txt:{int(source_line)}"


def hour_start_array(timestamps: np.ndarray, hour_seconds: int, origin: int = 1) -> np.ndarray:
    """Vectorised ``dualscope.splits.dataset_hour_start`` for an origin-1 dataset."""
    return origin + ((timestamps - origin) // hour_seconds) * hour_seconds


@dataclass
class DaySequences:
    """All user-hour sequences of one dataset day in model-ready form.

    Event arrays are in sequence order: acting user (alphabetical), then
    ``timestamp``, then ``source_line``. User-hour ``u`` owns events
    ``offsets[u] : offsets[u] + counts[u]``.
    """

    dataset_day: int
    timestamps: np.ndarray
    source_lines: np.ndarray
    numeric: np.ndarray
    categorical: np.ndarray
    users: np.ndarray
    hour_starts: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    split: str
    status: np.ndarray
    status_detail: np.ndarray
    fitting_eligible: np.ndarray
    ineligible_events: np.ndarray

    @property
    def n_events(self) -> int:
        return int(len(self.timestamps))

    @property
    def n_user_hours(self) -> int:
        return int(len(self.users))

    def scorable_mask(self) -> np.ndarray:
        return self.status == ScoringStatus.AVAILABLE.value

    def user_hour_events(self, index: int) -> slice:
        start = int(self.offsets[index])
        return slice(start, start + int(self.counts[index]))

    def source_references_for(self, index: int) -> list[str]:
        return [source_reference(line) for line in self.source_lines[self.user_hour_events(index)]]


@dataclass
class ChunkIndex:
    """Model sequences for one day: contiguous, bounded slices of user-hours."""

    user_hour: np.ndarray
    chunk_index: np.ndarray
    start: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        return int(len(self.start))

    def take(self, positions: np.ndarray) -> ChunkIndex:
        return ChunkIndex(
            user_hour=self.user_hour[positions],
            chunk_index=self.chunk_index[positions],
            start=self.start[positions],
            length=self.length[positions],
        )


def build_chunks(
    offsets: np.ndarray,
    counts: np.ndarray,
    max_length: int,
    user_hours: np.ndarray | None = None,
) -> ChunkIndex:
    """Split each selected user-hour into ``ceil(n / L)`` balanced consecutive chunks.

    Balanced chunks differ in length by at most one event, which avoids a very
    short trailing sequence without context. Chunks never cross their hour.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if user_hours is None:
        user_hours = np.arange(len(offsets), dtype=np.int64)
    user_hours = np.asarray(user_hours, dtype=np.int64)
    n = counts[user_hours].astype(np.int64)
    if np.any(n < 1):
        raise ValueError("cannot build chunks for an empty user-hour")
    k = -(-n // max_length)
    base = n // k
    remainder = n % k
    total = int(k.sum())
    group_start = np.cumsum(k) - k
    rep_uh = np.repeat(user_hours, k)
    within = np.arange(total, dtype=np.int64) - np.repeat(group_start, k)
    rep_base = np.repeat(base, k)
    rep_rem = np.repeat(remainder, k)
    length = rep_base + (within < rep_rem)
    start = np.repeat(offsets[user_hours].astype(np.int64), k) + within * rep_base + np.minimum(within, rep_rem)
    return ChunkIndex(
        user_hour=rep_uh,
        chunk_index=within.astype(np.int32),
        start=start,
        length=length.astype(np.int32),
    )


def gather_padded(
    numeric: np.ndarray,
    categorical: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    pad_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise right-padded ``[B, T, F]`` arrays and a ``[B, T]`` validity mask.

    Padded positions hold zeros (numeric) and ID 0 (categorical) and are
    ``False`` in the mask; the model and loss must ignore them.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    width = int(lengths.max()) if pad_to is None else int(pad_to)
    if len(lengths) and width < int(lengths.max()):
        raise ValueError("pad_to is shorter than the longest sequence")
    positions = np.arange(width, dtype=np.int64)
    mask = positions[None, :] < lengths[:, None]
    index = np.where(mask, np.asarray(starts, dtype=np.int64)[:, None] + positions[None, :], 0)
    num = numeric[index]
    num[~mask] = 0.0
    cat = categorical[index].astype(np.int64)
    cat[~mask] = 0
    return num, cat, mask


def available_days(features_dir: str | Path) -> list[int]:
    """Dataset days present as ``dataset_day=NN`` partitions."""
    days = []
    for child in Path(features_dir).iterdir():
        if child.is_dir() and child.name.startswith("dataset_day="):
            days.append(int(child.name.split("=", 1)[1]))
    return sorted(days)


def _split_for_day(split_cfg: SplitConfig, day: int):
    for split in split_cfg.splits.values():
        if split.contains_day(day):
            return split
    raise ValueError(f"dataset day {day} is outside every configured split")


def load_day_sequences(
    features_dir: str | Path,
    day: int,
    split_cfg: SplitConfig,
    policy: SequencePolicy,
    excluded_users: Sequence[str] = (),
    tracker: SequenceCandidateRejectionTracker | None = None,
    dataset: ds.Dataset | None = None,
) -> DaySequences:
    """Build all user-hour sequences for one day of transformed features."""
    if policy.hour_seconds != split_cfg.sequence_hour_seconds:
        raise ValueError(
            f"sequence hour ({policy.hour_seconds}s) differs from the split policy "
            f"({split_cfg.sequence_hour_seconds}s)"
        )
    if dataset is None:
        dataset = ds.dataset(str(features_dir), format="parquet", partitioning="hive")
    contaminated = set(dataset.schema.names) & LABEL_COLUMN_NAMES
    if contaminated:
        raise ValueError(f"label columns found in detector feature input: {sorted(contaminated)}")

    split = _split_for_day(split_cfg, day)
    is_train = split.name == "train"
    dense_columns = list(SEQUENCE_NUMERIC_INPUTS) + list(SEQUENCE_BINARY_INPUTS)
    model_columns = dense_columns + list(SEQUENCE_CATEGORICAL_INPUTS)
    columns = list(_BASE_COLUMNS) + model_columns + (list(_ELIGIBILITY_COLUMNS) if is_train else [])
    table = dataset.to_table(filter=ds.field("dataset_day") == day, columns=columns)
    if table.num_rows == 0:
        raise ValueError(f"no transformed feature rows for dataset day {day}")

    timestamps = table["timestamp"].to_numpy()
    source_lines = table["source_line"].to_numpy()
    _check_reference_sample(table["source_reference"], source_lines)

    # Encode one contiguous array so every row shares a single dictionary.
    encoded = pc.dictionary_encode(table["acting_user"].combine_chunks())
    dictionary = np.asarray(encoded.dictionary.to_pylist(), dtype=object)
    alphabetical = np.argsort(dictionary, kind="stable")
    rank = np.empty(len(dictionary), dtype=np.int64)
    rank[alphabetical] = np.arange(len(dictionary))
    user_codes = rank[encoded.indices.to_numpy()]
    sorted_users = dictionary[alphabetical]

    order = np.lexsort((source_lines, timestamps, user_codes))
    timestamps = timestamps[order]
    source_lines = source_lines[order]
    user_codes = user_codes[order]
    hours = hour_start_array(timestamps, policy.hour_seconds, split_cfg.timestamp_start_inclusive)

    boundary = np.flatnonzero((user_codes[1:] != user_codes[:-1]) | (hours[1:] != hours[:-1])) + 1
    offsets = np.concatenate(([0], boundary)).astype(np.int64)
    counts = np.diff(np.concatenate((offsets, [len(timestamps)]))).astype(np.int64)
    group_of_event = np.repeat(np.arange(len(offsets)), counts)

    # Every event must lie inside its user-hour's split and hour.
    hour_starts = hours[offsets]
    if not (np.all(timestamps >= split.timestamp_start) and np.all(timestamps < split.timestamp_end)):
        raise ValueError(f"dataset day {day} contains events outside split '{split.name}'")
    if np.any(hours != hour_starts[group_of_event]) or np.any(
        user_codes != user_codes[offsets][group_of_event]
    ):
        raise ValueError("an event was assigned to the wrong user-hour")

    numeric = np.column_stack(
        [table[name].to_numpy().astype(np.float32, copy=False)[order] for name in dense_columns]
    )
    categorical = np.column_stack(
        [table[name].to_numpy()[order] for name in SEQUENCE_CATEGORICAL_INPUTS]
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"non-finite model input on dataset day {day}")
    if categorical.min() < 0 or categorical.max() > np.iinfo(np.int16).max:
        raise ValueError(f"categorical IDs out of range on dataset day {day}")
    categorical = categorical.astype(np.int16)

    if policy.require_complete_24h_history:
        history_ok = table["history_complete_24h"].to_numpy()[order].astype(bool)
        warm = np.logical_and.reduceat(history_ok, offsets)
    else:
        warm = np.ones(len(offsets), dtype=bool)

    if is_train and excluded_users:
        excluded = pa.array(sorted(set(excluded_users)), type=pa.string())
        bad = pc.or_(
            pc.is_in(table["source_user"], value_set=excluded),
            pc.is_in(table["destination_user"], value_set=excluded),
        ).to_numpy()[order]
        ineligible = np.add.reduceat(bad.astype(np.int64), offsets)
    else:
        ineligible = np.zeros(len(offsets), dtype=np.int64)

    status, detail = _assign_status(timestamps, offsets, counts, hour_starts, warm, split, policy, tracker)
    available = status == ScoringStatus.AVAILABLE.value
    fitting = available & is_train & (ineligible == 0)

    return DaySequences(
        dataset_day=day,
        timestamps=timestamps,
        source_lines=source_lines,
        numeric=numeric,
        categorical=categorical,
        users=sorted_users[user_codes[offsets]],
        hour_starts=hour_starts,
        offsets=offsets,
        counts=counts,
        split=split.name,
        status=status,
        status_detail=detail,
        fitting_eligible=fitting,
        ineligible_events=ineligible,
    )


def _check_reference_sample(references: pa.ChunkedArray, source_lines: np.ndarray, stride: int = 997) -> None:
    """Confirm the ``auth.txt:<source_line>`` identity on a strided sample of rows."""
    positions = np.arange(0, len(source_lines), stride)
    sample = references.take(pa.array(positions)).to_pylist()
    for position, reference in zip(positions, sample):
        if reference != source_reference(source_lines[position]):
            raise ValueError(
                f"source_reference {reference!r} does not match source_line {source_lines[position]}"
            )


def _assign_status(
    timestamps: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    hour_starts: np.ndarray,
    warm: np.ndarray,
    split,
    policy: SequencePolicy,
    tracker: SequenceCandidateRejectionTracker | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Record every user-hour candidate and return its scoring status and reason."""
    tracker = tracker if tracker is not None else SequenceCandidateRejectionTracker(hour_seconds=policy.hour_seconds)
    n = len(offsets)
    status = np.empty(n, dtype=object)
    detail = np.full(n, "", dtype=object)
    last_ts = timestamps[offsets + counts - 1]
    for i in range(n):
        if not warm[i]:
            tracker.record_status(ScoringStatus.INSUFFICIENT_HISTORY)
            status[i] = ScoringStatus.INSUFFICIENT_HISTORY.value
            detail[i] = STATUS_DETAIL_WARMUP
            continue
        ok, reason = tracker.record_candidate(
            int(last_ts[i]),
            split,
            expected_hour_start=int(hour_starts[i]),
            min_events_required=policy.min_events,
            event_count=int(counts[i]),
            hour_seconds=policy.hour_seconds,
        )
        status[i] = reason
        if not ok:
            if reason != ScoringStatus.INSUFFICIENT_HISTORY.value:
                raise ValueError(f"user-hour candidate rejected by boundary check: {reason}")
            detail[i] = STATUS_DETAIL_TOO_FEW
    return status, detail


def iter_day_sequences(
    features_dir: str | Path,
    days: Iterable[int],
    split_cfg: SplitConfig,
    policy: SequencePolicy,
    excluded_users: Sequence[str] = (),
    tracker: SequenceCandidateRejectionTracker | None = None,
):
    """Yield ``DaySequences`` for each requested day, sharing one dataset handle."""
    dataset = ds.dataset(str(features_dir), format="parquet", partitioning="hive")
    for day in days:
        yield load_day_sequences(
            features_dir, day, split_cfg, policy, excluded_users, tracker=tracker, dataset=dataset
        )
