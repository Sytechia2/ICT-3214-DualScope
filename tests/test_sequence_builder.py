"""Task 3.1: user-hour sequence construction, chunking, padding and boundaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import EXCLUDED_USER, LONG_HOUR, LONG_HOUR_EVENTS
from dualscope.sequence.builder import (
    STATUS_DETAIL_TOO_FEW,
    STATUS_DETAIL_WARMUP,
    build_chunks,
    gather_padded,
    load_day_sequences,
    source_reference,
)
from dualscope.sequence.config import SequencePolicy
from dualscope.splits import ScoringStatus, SequenceCandidateRejectionTracker


def _load(data, day, policy=None, tracker=None):
    return load_day_sequences(
        data.features_dir, day, data.split_cfg, policy or SequencePolicy(), data.excluded_users, tracker=tracker
    )


def test_every_event_is_assigned_once_to_its_ordered_user_hour(synthetic_sequences) -> None:
    data = synthetic_sequences
    expected: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for event in data.events:
        hour = 1 + ((event["timestamp"] - 1) // 3600) * 3600
        expected[(event["acting_user"], hour)].append((event["timestamp"], event["source_line"]))

    seen: dict[tuple[str, int], list[tuple[int, int]]] = {}
    total = 0
    for day in range(1, 5):
        sequences = _load(data, day)
        total += sequences.n_events
        for u in range(sequences.n_user_hours):
            span = sequences.user_hour_events(u)
            key = (str(sequences.users[u]), int(sequences.hour_starts[u]))
            rows = list(zip(sequences.timestamps[span].tolist(), sequences.source_lines[span].tolist()))
            assert rows == sorted(rows), "events must be ordered by (timestamp, source_line)"
            assert all(1 + (t - 1) // 3600 * 3600 == key[1] for t, _ in rows)
            assert key not in seen
            seen[key] = rows
    assert total == len(data.events)
    assert seen == {key: sorted(rows) for key, rows in expected.items()}


def test_user_hours_stay_in_one_split_and_warm_up_is_explicit(synthetic_sequences) -> None:
    data = synthetic_sequences
    tracker = SequenceCandidateRejectionTracker()
    by_day = {day: _load(data, day, tracker=tracker) for day in range(1, 5)}

    for day, sequences in by_day.items():
        split = data.split_cfg.find_split_for_timestamp(int(sequences.timestamps[0]))
        assert sequences.split == split.name
        assert np.all((sequences.timestamps >= split.timestamp_start) & (sequences.timestamps < split.timestamp_end))
        ends = sequences.hour_starts + 3600
        assert np.all(ends <= split.timestamp_end)

    day_one = by_day[1]
    assert set(day_one.status) == {ScoringStatus.INSUFFICIENT_HISTORY.value}
    assert set(day_one.status_detail) == {STATUS_DETAIL_WARMUP}
    for day in (2, 3, 4):
        assert set(by_day[day].status) == {ScoringStatus.AVAILABLE.value}

    summary = tracker.summary()
    assert summary["status_counts"]["split_boundary_violation"] == 0
    assert summary["status_counts"]["hour_boundary_violation"] == 0
    assert summary["total_candidates"] == sum(s.n_user_hours for s in by_day.values())
    assert summary["rejected_candidates"] == day_one.n_user_hours


def test_fitting_eligibility_drops_hours_touching_excluded_users(synthetic_sequences) -> None:
    data = synthetic_sequences
    day_two = _load(data, 2)
    ineligible = {
        (str(user), int(hour)) for user, hour, ok in zip(day_two.users, day_two.hour_starts, day_two.fitting_eligible) if not ok
    }
    excluded_hour = 86_400 + 7 * 3_600 + 1
    assert (EXCLUDED_USER, excluded_hour) in ineligible
    assert ("U2@DOM1", excluded_hour) in ineligible  # excluded user only as destination
    assert day_two.fitting_eligible.sum() == day_two.n_user_hours - 2
    for day in (1, 3, 4):
        assert not _load(data, day).fitting_eligible.any()


def test_min_events_policy_marks_short_hours_insufficient(synthetic_sequences) -> None:
    sequences = _load(synthetic_sequences, 2, SequencePolicy(min_events=5))
    short = sequences.counts < 5
    assert short.any()
    assert set(sequences.status[short]) == {ScoringStatus.INSUFFICIENT_HISTORY.value}
    assert set(sequences.status_detail[short]) == {STATUS_DETAIL_TOO_FEW}
    assert set(sequences.status[~short]) == {ScoringStatus.AVAILABLE.value}


def test_long_hours_split_into_balanced_contiguous_chunks(synthetic_sequences) -> None:
    sequences = _load(synthetic_sequences, 2)
    index = int(np.flatnonzero((sequences.users == LONG_HOUR[0]) & (sequences.hour_starts == LONG_HOUR[1]))[0])
    assert sequences.counts[index] == LONG_HOUR_EVENTS

    chunks = build_chunks(sequences.offsets, sequences.counts, 32, np.array([index]))
    assert chunks.length.tolist() == [24, 23, 23]
    assert chunks.chunk_index.tolist() == [0, 1, 2]
    span = sequences.user_hour_events(index)
    covered = np.concatenate([np.arange(s, s + n) for s, n in zip(chunks.start, chunks.length)])
    assert covered.tolist() == list(range(span.start, span.stop))

    all_chunks = build_chunks(sequences.offsets, sequences.counts, 16)
    assert all_chunks.length.max() <= 16
    assert all_chunks.length.sum() == sequences.n_events
    owner_start = sequences.offsets[all_chunks.user_hour]
    owner_end = owner_start + sequences.counts[all_chunks.user_hour]
    assert np.all((all_chunks.start >= owner_start) & (all_chunks.start + all_chunks.length <= owner_end))


def test_padding_is_masked_and_references_follow_model_rows(synthetic_sequences) -> None:
    sequences = _load(synthetic_sequences, 3)
    chunks = build_chunks(sequences.offsets, sequences.counts, 8)
    pick = np.array([0, 1, 2, 3])
    dense, cat, mask = gather_padded(sequences.numeric, sequences.categorical, chunks.start[pick], chunks.length[pick])
    assert dense.shape[:2] == cat.shape[:2] == mask.shape
    assert mask.sum(axis=1).tolist() == chunks.length[pick].tolist()
    assert np.all(dense[~mask] == 0.0) and np.all(cat[~mask] == 0)
    for row, (start, length) in enumerate(zip(chunks.start[pick], chunks.length[pick])):
        np.testing.assert_array_equal(dense[row, :length], sequences.numeric[start : start + length])
    refs = sequences.source_references_for(0)
    assert refs == [source_reference(line) for line in sequences.source_lines[sequences.user_hour_events(0)]]
    events_by_ref = {e["source_reference"]: e for e in synthetic_sequences.events}
    assert all(events_by_ref[r]["acting_user"] == sequences.users[0] for r in refs)


def test_label_columns_in_feature_input_are_rejected(synthetic_sequences, tmp_path) -> None:
    day_dir = synthetic_sequences.features_dir / "dataset_day=03"
    table = pa.concat_tables(pq.ParquetFile(path).read() for path in sorted(day_dir.glob("*.parquet")))
    contaminated = table.append_column("is_redteam", pa.array([False] * table.num_rows))
    target = tmp_path / "events" / "dataset_day=03" / "part-000000.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(contaminated, target)
    with pytest.raises(ValueError, match="label columns"):
        load_day_sequences(tmp_path / "events", 3, synthetic_sequences.split_cfg, SequencePolicy())
