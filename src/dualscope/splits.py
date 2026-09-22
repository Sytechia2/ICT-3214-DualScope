"""Chronological data split policy, boundaries, exclusions, and replay helpers.

This module implements Task 2.3 of the DualScope data preparation pipeline:
- Defines exact half-open chronological intervals for training, validation, and test.
- Derives training exclusions based exclusively on red-team compromise events in the
  training interval, without using validation/test labels.
- Provides reusable filtering helpers for model fitting, vocabulary construction, and scaling.
- Implements deterministic chronological replay streaming with ordering verification and
  strict isolation between detector events and evaluation labels.
- Provides graph warm-up and sequence boundary verification helpers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600

DEFAULT_GRAPH_LOOKBACK = 86_400
DEFAULT_SEQUENCE_HOUR = 3_600

SUPPORTED_EXCLUSION_POLICIES = frozenset({"train_redteam_users_source_or_destination"})
LABEL_COLUMN_NAMES = frozenset({"user", "label", "is_redteam", "is_compromised", "redteam_label"})


class ScoringStatus(str, Enum):
    """Explicit scoring eligibility and candidate rejection status codes."""
    AVAILABLE = "available"
    INSUFFICIENT_HISTORY = "insufficient_history"
    SPLIT_BOUNDARY_VIOLATION = "split_boundary_violation"
    HOUR_BOUNDARY_VIOLATION = "hour_boundary_violation"
    OUT_OF_BOUNDS = "out_of_bounds"


def dataset_day(timestamp: int) -> int:
    """Calculate the 1-indexed dataset day for a positive integer timestamp.

    Formula: ((timestamp - 1) // 86400) + 1.
    """
    if timestamp < 1:
        raise ValueError(f"Timestamp must be a positive integer >= 1, got {timestamp}")
    return ((timestamp - 1) // SECONDS_PER_DAY) + 1


def dataset_hour_start(timestamp: int, hour_seconds: int = SECONDS_PER_HOUR) -> int:
    """Calculate the 1-indexed hour start timestamp for an event timestamp.

    Origin is 1: hour_start = 1 + ((timestamp - 1) // hour_seconds) * hour_seconds.
    For timestamp 1, returns 1. For timestamp 3600, returns 1. For timestamp 3601, returns 3601.
    """
    if timestamp < 1:
        raise ValueError(f"Timestamp must be a positive integer >= 1, got {timestamp}")
    if hour_seconds < 1:
        raise ValueError(f"hour_seconds must be positive, got {hour_seconds}")
    return 1 + ((timestamp - 1) // hour_seconds) * hour_seconds


def dataset_hour_index(timestamp: int, hour_seconds: int = SECONDS_PER_HOUR) -> int:
    """Calculate the 1-indexed hour index across the dataset.

    Formula: ((timestamp - 1) // hour_seconds) + 1.
    """
    if timestamp < 1:
        raise ValueError(f"Timestamp must be a positive integer >= 1, got {timestamp}")
    if hour_seconds < 1:
        raise ValueError(f"hour_seconds must be positive, got {hour_seconds}")
    return ((timestamp - 1) // hour_seconds) + 1


@dataclass(frozen=True)
class SplitInterval:
    """Defines a half-open dataset time interval [timestamp_start, timestamp_end)."""

    name: str
    dataset_day_start: int
    dataset_day_end: int
    timestamp_start: int
    timestamp_end: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.dataset_day_start < 1:
            raise ValueError(f"dataset_day_start must be >= 1, got {self.dataset_day_start}")
        if self.dataset_day_end < self.dataset_day_start:
            raise ValueError(
                f"dataset_day_end ({self.dataset_day_end}) cannot be less than "
                f"dataset_day_start ({self.dataset_day_start})"
            )
        if self.timestamp_start < 1:
            raise ValueError(f"timestamp_start must be >= 1, got {self.timestamp_start}")
        if self.timestamp_end <= self.timestamp_start:
            raise ValueError(
                f"timestamp_end ({self.timestamp_end}) must be strictly greater than "
                f"timestamp_start ({self.timestamp_start})"
            )

    @property
    def duration_seconds(self) -> int:
        return self.timestamp_end - self.timestamp_start

    def contains_timestamp(self, timestamp: int) -> bool:
        """Check if timestamp falls within the half-open interval [start, end)."""
        return self.timestamp_start <= timestamp < self.timestamp_end

    def contains_day(self, day: int) -> bool:
        """Check if day falls within [day_start, day_end] inclusive."""
        return self.dataset_day_start <= day <= self.dataset_day_end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitInterval:
        return cls(
            name=str(data["name"]),
            dataset_day_start=int(data["dataset_day_start"]),
            dataset_day_end=int(data["dataset_day_end"]),
            timestamp_start=int(data["timestamp_start"]),
            timestamp_end=int(data["timestamp_end"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class SplitConfig:
    """Overall configuration for chronological dataset partitioning."""

    policy_version: str
    policy_name: str
    dataset_day_start_inclusive: int
    dataset_day_end_inclusive: int
    timestamp_start_inclusive: int
    timestamp_end_exclusive: int
    graph_lookback_seconds: int
    sequence_hour_seconds: int
    exclusion_policy: str
    exclusion_window: str
    splits: dict[str, SplitInterval]

    def validate(self) -> None:
        """Validate configuration sanity, ordering, continuity, and supported policies."""
        if self.dataset_day_start_inclusive < 1:
            raise ValueError("dataset_day_start_inclusive must be >= 1")
        if self.dataset_day_end_inclusive < self.dataset_day_start_inclusive:
            raise ValueError("dataset_day_end_inclusive cannot be less than start")
        if self.timestamp_start_inclusive < 1:
            raise ValueError("timestamp_start_inclusive must be >= 1")
        if self.timestamp_end_exclusive <= self.timestamp_start_inclusive:
            raise ValueError("timestamp_end_exclusive must be greater than start")
        if self.graph_lookback_seconds <= 0:
            raise ValueError("graph_lookback_seconds must be positive")
        if self.sequence_hour_seconds <= 0:
            raise ValueError("sequence_hour_seconds must be positive")

        if self.exclusion_policy not in SUPPORTED_EXCLUSION_POLICIES:
            raise ValueError(
                f"Unsupported exclusion policy: '{self.exclusion_policy}'. "
                f"Supported policies are: {sorted(SUPPORTED_EXCLUSION_POLICIES)}"
            )

        if not self.splits:
            raise ValueError("At least one split interval must be defined")

        # Enforce whole-day policy on overall dataset boundaries
        expected_ds_start = ((self.dataset_day_start_inclusive - 1) * SECONDS_PER_DAY) + 1
        expected_ds_end = (self.dataset_day_end_inclusive * SECONDS_PER_DAY) + 1
        if self.timestamp_start_inclusive != expected_ds_start:
            raise ValueError(
                f"Dataset timestamp_start_inclusive ({self.timestamp_start_inclusive}) "
                f"does not match whole-day formula for day {self.dataset_day_start_inclusive} ({expected_ds_start})"
            )
        if self.timestamp_end_exclusive != expected_ds_end:
            raise ValueError(
                f"Dataset timestamp_end_exclusive ({self.timestamp_end_exclusive}) "
                f"does not match whole-day formula for day {self.dataset_day_end_inclusive} ({expected_ds_end})"
            )

        # Require train, validation, and test splits
        required_splits = {"train", "validation", "test"}
        missing_splits = required_splits - set(self.splits.keys())
        if missing_splits:
            raise ValueError(
                f"Missing required splits: {sorted(missing_splits)}. "
                f"SplitConfig must contain 'train', 'validation', and 'test'."
            )

        # Require each dictionary key to match SplitInterval.name
        for key, split in self.splits.items():
            if key != split.name:
                raise ValueError(
                    f"Split dictionary key '{key}' does not match SplitInterval.name '{split.name}'"
                )

        # Verify ordering, whole-day boundaries, and gap-free continuity
        split_list = sorted(self.splits.values(), key=lambda s: s.timestamp_start)
        expected_start = self.timestamp_start_inclusive
        expected_day = self.dataset_day_start_inclusive

        for split in split_list:
            expected_split_start = ((split.dataset_day_start - 1) * SECONDS_PER_DAY) + 1
            expected_split_end = (split.dataset_day_end * SECONDS_PER_DAY) + 1
            if split.timestamp_start != expected_split_start:
                raise ValueError(
                    f"Split '{split.name}' timestamp_start ({split.timestamp_start}) "
                    f"does not match whole-day formula for day {split.dataset_day_start} ({expected_split_start})"
                )
            if split.timestamp_end != expected_split_end:
                raise ValueError(
                    f"Split '{split.name}' timestamp_end ({split.timestamp_end}) "
                    f"does not match whole-day formula for day {split.dataset_day_end} ({expected_split_end})"
                )

            if split.timestamp_start != expected_start:
                raise ValueError(
                    f"Gap or overlap detected at split '{split.name}': "
                    f"expected start {expected_start}, found {split.timestamp_start}"
                )
            if split.dataset_day_start != expected_day:
                raise ValueError(
                    f"Day gap or overlap detected at split '{split.name}': "
                    f"expected day start {expected_day}, found {split.dataset_day_start}"
                )
            expected_start = split.timestamp_end
            expected_day = split.dataset_day_end + 1

        if expected_start != self.timestamp_end_exclusive:
            raise ValueError(
                f"Final split end {expected_start} does not match total interval end "
                f"{self.timestamp_end_exclusive}"
            )
        if expected_day - 1 != self.dataset_day_end_inclusive:
            raise ValueError(
                f"Final split day {expected_day - 1} does not match total dataset day end "
                f"{self.dataset_day_end_inclusive}"
            )

    def get_split(self, name: str) -> SplitInterval:
        if name not in self.splits:
            raise KeyError(f"Split '{name}' not found. Available splits: {list(self.splits.keys())}")
        return self.splits[name]

    def find_split_for_timestamp(self, timestamp: int) -> SplitInterval | None:
        for split in self.splits.values():
            if split.contains_timestamp(timestamp):
                return split
        return None

    def fingerprint(self) -> str:
        """Deterministic SHA-256 hash of canonical configuration content."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "policy_name": self.policy_name,
            "dataset_day_start_inclusive": self.dataset_day_start_inclusive,
            "dataset_day_end_inclusive": self.dataset_day_end_inclusive,
            "timestamp_start_inclusive": self.timestamp_start_inclusive,
            "timestamp_end_exclusive": self.timestamp_end_exclusive,
            "graph_lookback_seconds": self.graph_lookback_seconds,
            "sequence_hour_seconds": self.sequence_hour_seconds,
            "exclusion_policy": self.exclusion_policy,
            "exclusion_window": self.exclusion_window,
            "splits": {name: s.to_dict() for name, s in self.splits.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitConfig:
        raw_splits = data.get("splits", {})
        splits = {name: SplitInterval.from_dict(s) for name, s in raw_splits.items()}
        config = cls(
            policy_version=str(data.get("policy_version", "1.0.0")),
            policy_name=str(data.get("policy_name", "lanl_chronological_splits_v1")),
            dataset_day_start_inclusive=int(data.get("dataset_day_start_inclusive", 1)),
            dataset_day_end_inclusive=int(data.get("dataset_day_end_inclusive", 30)),
            timestamp_start_inclusive=int(data.get("timestamp_start_inclusive", 1)),
            timestamp_end_exclusive=int(data.get("timestamp_end_exclusive", 2592001)),
            graph_lookback_seconds=int(data.get("graph_lookback_seconds", DEFAULT_GRAPH_LOOKBACK)),
            sequence_hour_seconds=int(data.get("sequence_hour_seconds", DEFAULT_SEQUENCE_HOUR)),
            exclusion_policy=str(
                data.get("exclusion_policy", "train_redteam_users_source_or_destination")
            ),
            exclusion_window=str(data.get("exclusion_window", "entire_training_interval")),
            splits=splits,
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> SplitConfig:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    @classmethod
    def default(cls) -> SplitConfig:
        splits = {
            "train": SplitInterval(
                name="train",
                dataset_day_start=1,
                dataset_day_end=7,
                timestamp_start=1,
                timestamp_end=604801,
                description="Initial baseline and early activity period for normal model fitting and vocabulary/scaling learning with red-team user exclusions.",
            ),
            "validation": SplitInterval(
                name="validation",
                dataset_day_start=8,
                dataset_day_end=16,
                timestamp_start=604801,
                timestamp_end=1382401,
                description="Intermediate period containing major red-team lateral movement campaigns for hyperparameter tuning, score calibration, and alert threshold selection.",
            ),
            "test": SplitInterval(
                name="test",
                dataset_day_start=17,
                dataset_day_end=30,
                timestamp_start=1382401,
                timestamp_end=2592001,
                description="Held-out final evaluation period evaluated using frozen model weights and calibration parameters.",
            ),
        }
        cfg = cls(
            policy_version="1.0.0",
            policy_name="lanl_chronological_splits_v1",
            dataset_day_start_inclusive=1,
            dataset_day_end_inclusive=30,
            timestamp_start_inclusive=1,
            timestamp_end_exclusive=2592001,
            graph_lookback_seconds=DEFAULT_GRAPH_LOOKBACK,
            sequence_hour_seconds=DEFAULT_SEQUENCE_HOUR,
            exclusion_policy="train_redteam_users_source_or_destination",
            exclusion_window="entire_training_interval",
            splits=splits,
        )
        cfg.validate()
        return cfg


# -----------------------------------------------------------------------------
# Graph and Sequence Warm-Up Helpers
# -----------------------------------------------------------------------------

def graph_history_bounds(score_time: int, lookback_seconds: int = DEFAULT_GRAPH_LOOKBACK) -> tuple[int, int]:
    """Calculate half-open graph historical window [start, end) for a given score time t.

    Window: [t - lookback_seconds, t).
    """
    if score_time < 1:
        raise ValueError(f"score_time must be >= 1, got {score_time}")
    if lookback_seconds <= 0:
        raise ValueError(f"lookback_seconds must be positive, got {lookback_seconds}")
    return score_time - lookback_seconds, score_time


def check_graph_history_status(
    score_time: int,
    dataset_start: int = 1,
    lookback_seconds: int = DEFAULT_GRAPH_LOOKBACK,
    dataset_end_exclusive: int = 2592001,
) -> tuple[bool, str]:
    """Check if a graph history window [score_time - lookback, score_time) is available.

    Returns:
      (True, "available") if score_time - lookback >= dataset_start and score_time <= dataset_end_exclusive.
      (False, "insufficient_history") if score_time - lookback < dataset_start.
      (False, "out_of_bounds") if score_time > dataset_end_exclusive.
    """
    if score_time < 1:
        raise ValueError(f"score_time must be >= 1, got {score_time}")
    history_start = score_time - lookback_seconds
    if history_start < dataset_start:
        return False, ScoringStatus.INSUFFICIENT_HISTORY.value
    if score_time > dataset_end_exclusive:
        return False, ScoringStatus.OUT_OF_BOUNDS.value
    return True, ScoringStatus.AVAILABLE.value


def check_sequence_boundary(
    timestamp: int,
    split: SplitInterval,
    expected_hour_start: int | None = None,
    hour_seconds: int = DEFAULT_SEQUENCE_HOUR,
) -> tuple[bool, str]:
    """Verify that a sequence target event satisfies split and hour boundaries.

    Returns:
      (True, "available") if event is within the split and matches the expected hour.
      (False, "split_boundary_violation") if event is outside split [start, end).
      (False, "hour_boundary_violation") if event hour does not match expected_hour_start.
    """
    if not split.contains_timestamp(timestamp):
        return False, ScoringStatus.SPLIT_BOUNDARY_VIOLATION.value
    if expected_hour_start is not None:
        actual_hour = dataset_hour_start(timestamp, hour_seconds=hour_seconds)
        if actual_hour != expected_hour_start:
            return False, ScoringStatus.HOUR_BOUNDARY_VIOLATION.value
    return True, ScoringStatus.AVAILABLE.value


@dataclass
class SequenceCandidateRejectionTracker:
    """Contract and tracker for Task 3.1 sequence candidate scoring status.

    Task 3.1 must evaluate candidate sequences against split and hour boundaries
    and explicitly record any unscorable/rejected instances rather than silently
    discarding them.
    """
    dataset_start: int = 1
    dataset_end_exclusive: int = 2592001
    hour_seconds: int = DEFAULT_SEQUENCE_HOUR
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = {
                ScoringStatus.AVAILABLE.value: 0,
                ScoringStatus.INSUFFICIENT_HISTORY.value: 0,
                ScoringStatus.SPLIT_BOUNDARY_VIOLATION.value: 0,
                ScoringStatus.HOUR_BOUNDARY_VIOLATION.value: 0,
                ScoringStatus.OUT_OF_BOUNDS.value: 0,
            }

    def record_candidate(
        self,
        timestamp: int,
        split: SplitInterval,
        expected_hour_start: int | None = None,
        min_events_required: int = 1,
        event_count: int = 1,
        hour_seconds: int | None = None,
    ) -> tuple[bool, str]:
        """Evaluate a candidate event/window and record its status."""
        # 1. Dataset-level boundaries
        if timestamp < self.dataset_start or timestamp >= self.dataset_end_exclusive:
            status = ScoringStatus.OUT_OF_BOUNDS.value
            self.counts[status] = self.counts.get(status, 0) + 1
            return False, status

        # 2. Split-level boundaries
        if timestamp < split.timestamp_start or timestamp >= split.timestamp_end:
            status = ScoringStatus.SPLIT_BOUNDARY_VIOLATION.value
            self.counts[status] = self.counts.get(status, 0) + 1
            return False, status

        # 3. Hour boundaries
        eff_hour_seconds = hour_seconds if hour_seconds is not None else self.hour_seconds
        if expected_hour_start is not None:
            actual_hour = dataset_hour_start(timestamp, hour_seconds=eff_hour_seconds)
            if actual_hour != expected_hour_start:
                status = ScoringStatus.HOUR_BOUNDARY_VIOLATION.value
                self.counts[status] = self.counts.get(status, 0) + 1
                return False, status

        # 4. Insufficient events
        if event_count < min_events_required:
            status = ScoringStatus.INSUFFICIENT_HISTORY.value
            self.counts[status] = self.counts.get(status, 0) + 1
            return False, status

        status = ScoringStatus.AVAILABLE.value
        self.counts[status] = self.counts.get(status, 0) + 1
        return True, status

    def record_status(self, status: ScoringStatus | str) -> None:
        val = status.value if isinstance(status, ScoringStatus) else str(status)
        self.counts[val] = self.counts.get(val, 0) + 1

    def total_candidates(self) -> int:
        return sum(self.counts.values())

    def total_rejected(self) -> int:
        return sum(c for k, c in self.counts.items() if k != ScoringStatus.AVAILABLE.value)

    def summary(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates(),
            "scorable_candidates": self.counts.get(ScoringStatus.AVAILABLE.value, 0),
            "rejected_candidates": self.total_rejected(),
            "status_counts": dict(self.counts),
        }


# -----------------------------------------------------------------------------
# Red-Team Labels and Excluded Users Helpers
# -----------------------------------------------------------------------------

def derive_excluded_users(
    labels_source: pa.Table | ds.Dataset,
    train_start: int = 1,
    train_end_exclusive: int = 604801,
) -> list[str]:
    """Derive excluded user identities using only red-team labels in the training interval.

    Validation and test labels MUST NOT be used to choose excluded training users.
    """
    if isinstance(labels_source, ds.Dataset):
        train_filter = (ds.field("timestamp") >= train_start) & (
            ds.field("timestamp") < train_end_exclusive
        )
        labels_table = labels_source.to_table(filter=train_filter, columns=["user"])
    else:
        # pa.Table
        mask = pc.and_(
            pc.greater_equal(labels_source["timestamp"], train_start),
            pc.less(labels_source["timestamp"], train_end_exclusive),
        )
        labels_table = labels_source.filter(mask).select(["user"])

    if len(labels_table) == 0:
        return []

    unique_users = pc.unique(labels_table["user"]).to_pylist()
    return sorted(str(u) for u in unique_users if u is not None)


def build_fitting_filter_expression(excluded_users: Sequence[str]) -> ds.Expression:
    """Build a PyArrow dataset filter expression that accepts only fitting-eligible events.

    An event is eligible if neither source_user nor destination_user is in excluded_users.
    """
    if not excluded_users:
        return ds.field("timestamp") > 0  # Always true for valid events

    user_arr = pa.array(sorted(set(excluded_users)), type=pa.string())
    src_in = pc.is_in(ds.field("source_user"), value_set=user_arr)
    dst_in = pc.is_in(ds.field("destination_user"), value_set=user_arr)
    return ~src_in & ~dst_in


def is_fitting_eligible(
    data: pa.Table | pa.RecordBatch,
    excluded_users: Sequence[str],
) -> pa.BooleanArray:
    """Return a BooleanArray mask indicating which rows are eligible for model fitting.

    True indicates the row is eligible (neither source nor destination user is excluded).
    False indicates the row must be excluded from model fitting and learned transformations.
    """
    if not excluded_users:
        return pa.array([True] * len(data), type=pa.bool_())

    user_arr = pa.array(sorted(set(excluded_users)), type=pa.string())
    src_mask = pc.is_in(data["source_user"], value_set=user_arr)
    dst_mask = pc.is_in(data["destination_user"], value_set=user_arr)
    excluded = pc.or_(src_mask, dst_mask)
    return pc.invert(excluded)


def filter_fitting_eligible(
    data: pa.Table | pa.RecordBatch,
    excluded_users: Sequence[str],
) -> pa.Table | pa.RecordBatch:
    """Filter a Table or RecordBatch to retain only rows eligible for model fitting.

    Preserves excluded events in the underlying dataset/replay stream; this function
    is only for model fitting, vocabulary construction, and normalisation scalers.
    """
    if not excluded_users or len(data) == 0:
        return data

    mask = is_fitting_eligible(data, excluded_users)
    return pc.filter(data, mask)


def deduplicate_labels(labels: pa.Table | ds.Dataset) -> pa.Table:
    """Deduplicate red-team labels by (timestamp, user, source_computer, destination_computer).

    Retains the first occurrence of each unique compromise record.
    """
    if isinstance(labels, ds.Dataset):
        labels_table = labels.to_table()
    else:
        labels_table = labels

    if len(labels_table) == 0:
        return labels_table

    ts_col = labels_table["timestamp"].to_pylist()
    u_col = labels_table["user"].to_pylist()
    sc_col = labels_table["source_computer"].to_pylist()
    dc_col = labels_table["destination_computer"].to_pylist()

    seen = set()
    keep_indices = []
    for i in range(len(labels_table)):
        key = (ts_col[i], u_col[i], sc_col[i], dc_col[i])
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)

    return labels_table.take(pa.array(keep_indices, type=pa.int64()))


def aggregate_labels_to_user_hours(
    labels: pa.Table | ds.Dataset,
    hour_seconds: int = SECONDS_PER_HOUR,
) -> set[tuple[str, int]]:
    """Aggregate red-team labels to (user, hour_start) evaluation units.

    Multiple labelled events within the same user-hour evaluate as a single positive unit.
    """
    if isinstance(labels, ds.Dataset):
        labels_table = labels.to_table(columns=["timestamp", "user"])
    else:
        labels_table = labels.select(["timestamp", "user"])

    if len(labels_table) == 0:
        return set()

    ts_list = labels_table["timestamp"].to_pylist()
    user_list = labels_table["user"].to_pylist()

    user_hours = set()
    for ts, user in zip(ts_list, user_list):
        h_start = dataset_hour_start(ts, hour_seconds=hour_seconds)
        user_hours.add((str(user), h_start))

    return user_hours


# -----------------------------------------------------------------------------
# Chronological Replay Streaming Helper
# -----------------------------------------------------------------------------

def stream_split_events(
    events_source: str | Path | ds.Dataset,
    target_split: SplitInterval,
    include_prior_history: bool = True,
    history_start_timestamp: int | None = None,
    columns: Sequence[str] | None = None,
    batch_size: int = 65_536,
) -> Iterator[pa.RecordBatch]:
    """Stream authentication events in strictly non-decreasing (timestamp, source_line) order.

    Features:
    - Replay keeps all observed events (including training exclusions); validation/test labels
      must never selectively clean replay history.
    - If include_prior_history is True (default), starts replay at history_start_timestamp
      (or dataset start if None), giving downstream cumulative features (such as destination
      novelty) complete prior context.
    - Replay strictly excludes events at or beyond target_split.timestamp_end.
    - Ordering check is verified across batches and file boundaries, even when callers
      project fewer columns.
    - Rejects event sources or projections contaminated with red-team label columns.
    """
    # 1. Validate requested projection
    if columns is not None:
        contaminated = set(columns) & LABEL_COLUMN_NAMES
        if contaminated:
            raise ValueError(
                f"Label contamination detected: requested columns contain evaluation label fields: {contaminated}"
            )

    # 2. Open dataset
    if isinstance(events_source, (str, Path)):
        dataset = ds.dataset(events_source, format="parquet", partitioning="hive")
    else:
        dataset = events_source

    schema_contaminated = set(dataset.schema.names) & LABEL_COLUMN_NAMES
    if schema_contaminated:
        raise ValueError(
            f"Label contamination detected: event dataset schema contains evaluation label fields: {schema_contaminated}"
        )

    # 3. Determine time boundaries for the stream
    if include_prior_history:
        start_ts = 1 if history_start_timestamp is None else history_start_timestamp
    else:
        start_ts = target_split.timestamp_start

    end_ts = target_split.timestamp_end
    if start_ts >= end_ts:
        return

    # Day partition bounds
    day_start = dataset_day(start_ts)
    day_end = dataset_day(end_ts - 1)

    time_filter = (
        (ds.field("timestamp") >= start_ts)
        & (ds.field("timestamp") < end_ts)
        & (ds.field("dataset_day") >= day_start)
        & (ds.field("dataset_day") <= day_end)
    )

    # 4. Determine columns to scan (always include timestamp and source_line for ordering)
    needs_slice = False
    if columns is not None:
        scan_cols = list(columns)
        if "timestamp" not in scan_cols:
            scan_cols.append("timestamp")
            needs_slice = True
        if "source_line" not in scan_cols:
            scan_cols.append("source_line")
            needs_slice = True
    else:
        scan_cols = None

    scanner = dataset.scanner(
        filter=time_filter,
        columns=scan_cols,
        batch_size=batch_size,
    )

    last_timestamp = -1
    last_source_line = -1

    for batch in scanner.to_batches():
        if len(batch) == 0:
            continue

        ts_col = batch["timestamp"]
        sl_col = batch["source_line"]

        # Vectorized check within the batch
        # Verify non-decreasing order: (timestamp_i, source_line_i) >= (timestamp_{i-1}, source_line_{i-1})
        # Check boundary from previous batch
        first_ts = ts_col[0].as_py()
        first_sl = sl_col[0].as_py()

        if first_ts < last_timestamp or (first_ts == last_timestamp and first_sl < last_source_line):
            raise ValueError(
                f"Deterministic ordering violation across batch boundary: "
                f"(timestamp={first_ts}, source_line={first_sl}) < "
                f"previous (timestamp={last_timestamp}, source_line={last_source_line})"
            )

        # Check inside batch if batch length > 1
        if len(batch) > 1:
            diff_ts = pc.subtract(ts_col.slice(1), ts_col.slice(0, len(batch) - 1))
            diff_sl = pc.subtract(sl_col.slice(1), sl_col.slice(0, len(batch) - 1))

            # Disallowed condition: diff_ts < 0 or (diff_ts == 0 and diff_sl < 0)
            inv_ts = pc.less(diff_ts, 0)
            inv_sl = pc.and_(pc.equal(diff_ts, 0), pc.less(diff_sl, 0))
            inversion_mask = pc.or_(inv_ts, inv_sl)

            if pc.any(inversion_mask).as_py():
                # Locate exact row of first inversion for clear error message
                for i in range(len(batch) - 1):
                    t_curr, sl_curr = ts_col[i].as_py(), sl_col[i].as_py()
                    t_next, sl_next = ts_col[i + 1].as_py(), sl_col[i + 1].as_py()
                    if t_next < t_curr or (t_next == t_curr and sl_next < sl_curr):
                        raise ValueError(
                            f"Deterministic ordering violation inside batch at index {i + 1}: "
                            f"(timestamp={t_next}, source_line={sl_next}) < "
                            f"(timestamp={t_curr}, source_line={sl_curr})"
                        )

        last_timestamp = ts_col[-1].as_py()
        last_source_line = sl_col[-1].as_py()

        # Check upper boundary
        if last_timestamp >= end_ts:
            raise ValueError(
                f"Replay stream included event at or beyond split end timestamp {end_ts}: "
                f"found timestamp {last_timestamp}"
            )

        if needs_slice and columns is not None:
            yield batch.select(columns)
        else:
            yield batch
