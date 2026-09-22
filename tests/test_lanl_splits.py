from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from dualscope.splits import (
    DEFAULT_GRAPH_LOOKBACK,
    DEFAULT_SEQUENCE_HOUR,
    ScoringStatus,
    SequenceCandidateRejectionTracker,
    SplitConfig,
    SplitInterval,
    aggregate_labels_to_user_hours,
    build_fitting_filter_expression,
    check_graph_history_status,
    check_sequence_boundary,
    dataset_day,
    dataset_hour_index,
    dataset_hour_start,
    deduplicate_labels,
    derive_excluded_users,
    filter_fitting_eligible,
    graph_history_bounds,
    is_fitting_eligible,
    stream_split_events,
)
from dualscope.ingestion.lanl import (
    IngestionConfig,
    ingest_authentication,
    ingest_redteam_labels,
)
from scripts.create_lanl_splits import (
    count_training_exclusions,
    generate_splits_manifest,
)


# -----------------------------------------------------------------------------
# 1. Configuration Validation and Fingerprinting
# -----------------------------------------------------------------------------

def test_split_config_validation_and_fingerprint() -> None:
    cfg = SplitConfig.default()
    cfg.validate()

    assert cfg.policy_version == "1.0.0"
    assert len(cfg.splits) == 3
    assert set(cfg.splits.keys()) == {"train", "validation", "test"}
    assert cfg.fingerprint() == "c5e3d5d89c0324b24ca79f391abfec197f47f89fd9ddb3f7652f4fe5a1284af9"

    # Reject unsupported exclusion policy
    invalid_policy_dict = cfg.to_dict()
    invalid_policy_dict["exclusion_policy"] = "all_hosts"
    with pytest.raises(ValueError, match="Unsupported exclusion policy"):
        SplitConfig.from_dict(invalid_policy_dict)

    # Reject split timestamp_start mismatch with whole-day formula
    mismatch_start_dict = cfg.to_dict()
    mismatch_start_dict["splits"]["train"]["timestamp_start"] = 2
    with pytest.raises(ValueError, match="does not match whole-day formula"):
        SplitConfig.from_dict(mismatch_start_dict)

    # Reject split timestamp_end mismatch with whole-day formula
    mismatch_end_dict = cfg.to_dict()
    mismatch_end_dict["splits"]["train"]["timestamp_end"] = 600000
    with pytest.raises(ValueError, match="does not match whole-day formula"):
        SplitConfig.from_dict(mismatch_end_dict)

    # Reject overall dataset timestamp start mismatch
    mismatch_ds_start = cfg.to_dict()
    mismatch_ds_start["timestamp_start_inclusive"] = 100
    with pytest.raises(ValueError, match="does not match whole-day formula"):
        SplitConfig.from_dict(mismatch_ds_start)

    # Reject overall dataset timestamp end mismatch
    mismatch_ds_end = cfg.to_dict()
    mismatch_ds_end["timestamp_end_exclusive"] = 2600000
    with pytest.raises(ValueError, match="does not match whole-day formula"):
        SplitConfig.from_dict(mismatch_ds_end)

    # Reject missing required splits
    missing_test_dict = cfg.to_dict()
    del missing_test_dict["splits"]["test"]
    with pytest.raises(ValueError, match="Missing required splits.*test"):
        SplitConfig.from_dict(missing_test_dict)

    # Reject dictionary key mismatch with SplitInterval.name
    key_mismatch_dict = cfg.to_dict()
    key_mismatch_dict["splits"]["train"]["name"] = "training"
    with pytest.raises(ValueError, match="does not match SplitInterval.name"):
        SplitConfig.from_dict(key_mismatch_dict)

    # Reject split gap (valid whole days with gap)
    gap_dict = cfg.to_dict()
    gap_dict["splits"]["train"]["dataset_day_end"] = 6
    gap_dict["splits"]["train"]["timestamp_end"] = 518401
    with pytest.raises(ValueError, match="Gap or overlap"):
        SplitConfig.from_dict(gap_dict)

    # Reject split overlap (valid whole days with overlap)
    overlap_dict = cfg.to_dict()
    overlap_dict["splits"]["train"]["dataset_day_end"] = 9
    overlap_dict["splits"]["train"]["timestamp_end"] = 777601
    with pytest.raises(ValueError, match="Gap or overlap"):
        SplitConfig.from_dict(overlap_dict)



# -----------------------------------------------------------------------------
# 2. Exact Boundaries on Both Sides of Each Split
# -----------------------------------------------------------------------------

def test_exact_boundaries_on_both_sides_of_each_split() -> None:
    cfg = SplitConfig.default()
    train = cfg.get_split("train")
    val = cfg.get_split("validation")
    test = cfg.get_split("test")

    # Training: [1, 604801) -> Days 1 to 7
    assert train.contains_timestamp(1)
    assert train.contains_timestamp(604800)
    assert not train.contains_timestamp(604801)
    assert dataset_day(1) == 1
    assert dataset_day(604800) == 7

    # Validation: [604801, 1382401) -> Days 8 to 16
    assert not val.contains_timestamp(604800)
    assert val.contains_timestamp(604801)
    assert val.contains_timestamp(1382400)
    assert not val.contains_timestamp(1382401)
    assert dataset_day(604801) == 8
    assert dataset_day(1382400) == 16

    # Test: [1382401, 2592001) -> Days 17 to 30
    assert not test.contains_timestamp(1382400)
    assert test.contains_timestamp(1382401)
    assert test.contains_timestamp(2592000)
    assert not test.contains_timestamp(2592001)
    assert dataset_day(1382401) == 17
    assert dataset_day(2592000) == 30


# -----------------------------------------------------------------------------
# 3. No Gaps or Overlap Across All Splits
# -----------------------------------------------------------------------------

def test_no_gaps_or_overlap_across_all_splits() -> None:
    cfg = SplitConfig.default()
    boundary_points = [
        0,  # Before start
        1,  # Train start
        300000,  # Train mid
        604800,  # Train last second
        604801,  # Val start
        1000000,  # Val mid
        1382400,  # Val last second
        1382401,  # Test start
        2000000,  # Test mid
        2592000,  # Test last second
        2592001,  # Out of bounds
    ]

    for ts in boundary_points:
        matched_splits = [s.name for s in cfg.splits.values() if s.contains_timestamp(ts)]
        if 1 <= ts < 604801:
            assert matched_splits == ["train"]
        elif 604801 <= ts < 1382401:
            assert matched_splits == ["validation"]
        elif 1382401 <= ts < 2592001:
            assert matched_splits == ["test"]
        else:
            assert matched_splits == []


# -----------------------------------------------------------------------------
# 4. Source and Destination User Exclusions
# -----------------------------------------------------------------------------

def test_source_and_destination_user_exclusions() -> None:
    excluded_users = ["U_BAD_1", "U_BAD_2"]

    table = pa.table(
        {
            "timestamp": [10, 20, 30, 40, 50],
            "source_user": ["U_BAD_1", "U_NORMAL", "U_BAD_2", "U_NORMAL", "U_CLEAN"],
            "destination_user": ["U_NORMAL", "U_BAD_1", "U_BAD_2", "U_NORMAL", "U_CLEAN"],
        }
    )

    mask = is_fitting_eligible(table, excluded_users)
    assert mask.to_pylist() == [False, False, False, True, True]

    filtered = filter_fitting_eligible(table, excluded_users)
    assert len(filtered) == 2
    assert filtered["source_user"].to_pylist() == ["U_NORMAL", "U_CLEAN"]

    # Test PyArrow dataset expression equivalent
    expr = build_fitting_filter_expression(excluded_users)
    dataset = ds.dataset([table])
    ds_filtered = dataset.to_table(filter=expr)
    assert len(ds_filtered) == 2
    assert ds_filtered["source_user"].to_pylist() == ["U_NORMAL", "U_CLEAN"]


# -----------------------------------------------------------------------------
# 5. Future Labels Do Not Change Training Eligibility
# -----------------------------------------------------------------------------

def test_future_labels_do_not_change_training_eligibility() -> None:
    labels = pa.table(
        {
            "timestamp": [100, 200, 700000, 1500000],
            "user": ["U_TRAIN_A", "U_TRAIN_B", "U_VAL_ONLY", "U_TEST_ONLY"],
            "source_computer": ["C1", "C2", "C3", "C4"],
            "destination_computer": ["C10", "C20", "C30", "C40"],
        }
    )

    excluded = derive_excluded_users(labels, train_start=1, train_end_exclusive=604801)
    assert excluded == ["U_TRAIN_A", "U_TRAIN_B"]

    # Even if future labels are modified, training exclusions remain unchanged
    labels_modified_future = pa.table(
        {
            "timestamp": [100, 200, 800000, 900000, 1800000],
            "user": ["U_TRAIN_A", "U_TRAIN_B", "U_NEW_VAL", "U_ANOTHER_VAL", "U_NEW_TEST"],
            "source_computer": ["C1", "C2", "C3", "C4", "C5"],
            "destination_computer": ["C10", "C20", "C30", "C40", "C50"],
        }
    )
    excluded_mod = derive_excluded_users(
        labels_modified_future, train_start=1, train_end_exclusive=604801
    )
    assert excluded_mod == ["U_TRAIN_A", "U_TRAIN_B"]


# -----------------------------------------------------------------------------
# 6. Excluded Events Remain Available for Replay
# -----------------------------------------------------------------------------

def test_excluded_events_remain_available_for_replay(tmp_path: Path) -> None:
    auth_lines = [
        "1,U_EXCLUDED,U_EXCLUDED,C1,C1,NTLM,Network,LogOn,Success\n",
        "2,U_NORMAL,U_NORMAL,C2,C2,NTLM,Network,LogOn,Success\n",
        "3,U_NORMAL,U_EXCLUDED,C2,C1,Kerberos,Network,LogOn,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(auth_lines), encoding="utf-8")
    out_dir = tmp_path / "auth_out"
    ingest_authentication(source, out_dir, IngestionConfig(day_end=1, chunk_rows=10))

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    batches = list(
        stream_split_events(out_dir / "events", target_split=train_split, include_prior_history=False)
    )
    total_events = sum(len(b) for b in batches)
    assert total_events == 3

    # Confirm the excluded users are present in the replay stream
    all_src_users = [u for b in batches for u in b["source_user"].to_pylist()]
    assert "U_EXCLUDED" in all_src_users
    assert "U_NORMAL" in all_src_users


# -----------------------------------------------------------------------------
# 7. Default Full History and Optional Bounded History
# -----------------------------------------------------------------------------

def test_default_full_history_and_optional_bounded_history(tmp_path: Path) -> None:
    # Events spanning Day 1, Day 2, and Day 8 (Validation start is Day 8: 604801)
    auth_lines = [
        "100,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n",
        "100000,U2,U2,C2,C2,NTLM,Network,LogOn,Success\n",
        "604805,U3,U3,C3,C3,NTLM,Network,LogOn,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(auth_lines), encoding="utf-8")
    out_dir = tmp_path / "auth_out"
    ingest_authentication(source, out_dir, IngestionConfig(day_end=8, chunk_rows=10))

    cfg = SplitConfig.default()
    val_split = cfg.get_split("validation")

    # Case A: Default full history (replays from beginning of dataset)
    stream_full = list(stream_split_events(out_dir / "events", target_split=val_split, include_prior_history=True))
    ts_full = [t for b in stream_full for t in b["timestamp"].to_pylist()]
    assert ts_full == [100, 100000, 604805]

    # Case B: Optional bounded history (start at timestamp 50000)
    stream_bounded = list(
        stream_split_events(
            out_dir / "events",
            target_split=val_split,
            include_prior_history=True,
            history_start_timestamp=50000,
        )
    )
    ts_bounded = [t for b in stream_bounded for t in b["timestamp"].to_pylist()]
    assert ts_bounded == [100000, 604805]

    # Case C: No prior history (targets only, starting at validation start 604801)
    stream_no_hist = list(
        stream_split_events(out_dir / "events", target_split=val_split, include_prior_history=False)
    )
    ts_no_hist = [t for b in stream_no_hist for t in b["timestamp"].to_pylist()]
    assert ts_no_hist == [604805]


# -----------------------------------------------------------------------------
# 8. No Future Events Beyond Replay End
# -----------------------------------------------------------------------------

def test_no_future_events_beyond_replay_end(tmp_path: Path) -> None:
    # Split train ends at 604801
    auth_lines = [
        "604799,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n",
        "604800,U2,U2,C2,C2,NTLM,Network,LogOn,Success\n",
        "604801,U3,U3,C3,C3,NTLM,Network,LogOn,Success\n",
        "604802,U4,U4,C4,C4,NTLM,Network,LogOn,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(auth_lines), encoding="utf-8")
    out_dir = tmp_path / "auth_out"
    ingest_authentication(source, out_dir, IngestionConfig(day_end=8, chunk_rows=10))

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    batches = list(
        stream_split_events(out_dir / "events", target_split=train_split, include_prior_history=True)
    )
    ts_all = [t for b in batches for t in b["timestamp"].to_pylist()]
    assert ts_all == [604799, 604800]
    assert all(t < train_split.timestamp_end for t in ts_all)


# -----------------------------------------------------------------------------
# 9. Deterministic Ordering Across Batches and Equal Timestamps
# -----------------------------------------------------------------------------

def test_deterministic_ordering_across_batches_and_equal_timestamps(tmp_path: Path) -> None:
    # Multiple events with equal timestamps, ordered by source_line
    auth_lines = [
        "100,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n",
        "100,U2,U2,C2,C2,NTLM,Network,LogOn,Success\n",
        "100,U3,U3,C3,C3,NTLM,Network,LogOn,Success\n",
        "200,U4,U4,C4,C4,NTLM,Network,LogOn,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(auth_lines), encoding="utf-8")
    out_dir = tmp_path / "auth_out"
    ingest_authentication(source, out_dir, IngestionConfig(day_end=1, chunk_rows=2))

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    batches = list(
        stream_split_events(out_dir / "events", target_split=train_split, batch_size=2)
    )
    lines_order = [
        (t, sl)
        for b in batches
        for t, sl in zip(b["timestamp"].to_pylist(), b["source_line"].to_pylist())
    ]
    assert lines_order == [(100, 1), (100, 2), (100, 3), (200, 4)]


# -----------------------------------------------------------------------------
# 10. Projection Does Not Bypass Ordering Checks
# -----------------------------------------------------------------------------

def test_projection_does_not_bypass_ordering_checks(tmp_path: Path) -> None:
    auth_lines = [
        "10,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n",
        "20,U2,U2,C2,C2,NTLM,Network,LogOn,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(auth_lines), encoding="utf-8")
    out_dir = tmp_path / "auth_out"
    ingest_authentication(source, out_dir, IngestionConfig(day_end=1, chunk_rows=10))

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # Project only acting_user and authentication_type
    batches = list(
        stream_split_events(
            out_dir / "events",
            target_split=train_split,
            columns=["acting_user", "authentication_type"],
        )
    )
    assert len(batches) == 1
    assert batches[0].schema.names == ["acting_user", "authentication_type"]
    assert batches[0]["acting_user"].to_pylist() == ["U1", "U2"]

    # Now create corrupt order manually and verify projection still detects order violation
    corrupt_table = pa.table(
        {
            "timestamp": [20, 10],  # Out of order
            "source_line": [1, 2],
            "acting_user": ["U2", "U1"],
        }
    )
    corrupt_dir = tmp_path / "corrupt_events" / "dataset_day=1"
    corrupt_dir.mkdir(parents=True)
    pq.write_table(corrupt_table, corrupt_dir / "part-000000.parquet")

    with pytest.raises(ValueError, match="ordering violation"):
        list(
            stream_split_events(
                tmp_path / "corrupt_events",
                target_split=train_split,
                columns=["acting_user"],
            )
        )


# -----------------------------------------------------------------------------
# 11. Label-Contaminated Event Inputs Are Rejected
# -----------------------------------------------------------------------------

def test_label_contaminated_event_inputs_are_rejected(tmp_path: Path) -> None:
    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # A: Caller requests label column projection
    with pytest.raises(ValueError, match="Label contamination detected"):
        list(
            stream_split_events(
                tmp_path,
                target_split=train_split,
                columns=["timestamp", "source_line", "user"],
            )
        )

    # B: Event dataset itself contains label column
    contaminated_table = pa.table(
        {
            "timestamp": [10],
            "source_line": [1],
            "source_user": ["U1"],
            "user": ["U1"],  # Label column name
        }
    )
    c_dir = tmp_path / "c_events" / "dataset_day=1"
    c_dir.mkdir(parents=True)
    pq.write_table(contaminated_table, c_dir / "part-000000.parquet")

    with pytest.raises(ValueError, match="Label contamination detected"):
        list(stream_split_events(tmp_path / "c_events", target_split=train_split))


# -----------------------------------------------------------------------------
# 12. Duplicate Label Counting and User-Hour Aggregation
# -----------------------------------------------------------------------------

def test_duplicate_label_counting_and_aggregation() -> None:
    labels = pa.table(
        {
            "timestamp": [100, 100, 200, 3600, 3601],
            "user": ["U1", "U1", "U1", "U1", "U1"],
            "source_computer": ["C1", "C1", "C1", "C1", "C1"],
            "destination_computer": ["C2", "C2", "C2", "C2", "C2"],
        }
    )

    # First row and second row are exact duplicates
    dedup = deduplicate_labels(labels)
    assert len(dedup) == 4

    # User-hour aggregation:
    # Timestamps 100, 200, 3600 are all in Hour 1 (hour_start = 1)
    # Timestamp 3601 is in Hour 2 (hour_start = 3601)
    user_hours = aggregate_labels_to_user_hours(dedup)
    assert len(user_hours) == 2
    assert user_hours == {("U1", 1), ("U1", 3601)}


# -----------------------------------------------------------------------------
# 13. Warm-Up Boundaries and Final Window End
# -----------------------------------------------------------------------------

def test_warm_up_boundaries_and_final_window_end() -> None:
    # Graph lookback: 86400 seconds
    bounds = graph_history_bounds(86401, lookback_seconds=86400)
    assert bounds == (1, 86401)

    # Status before 86401
    status_early, msg_early = check_graph_history_status(86400, dataset_start=1, lookback_seconds=86400)
    assert not status_early
    assert msg_early == "insufficient_history"

    # Status at exactly 86401
    status_ok, msg_ok = check_graph_history_status(86401, dataset_start=1, lookback_seconds=86400)
    assert status_ok
    assert msg_ok == "available"

    # Status at exclusive dataset end
    status_end, msg_end = check_graph_history_status(2592001, dataset_start=1, lookback_seconds=86400, dataset_end_exclusive=2592001)
    assert status_end
    assert msg_end == "available"

    # Status beyond dataset end
    status_oob, msg_oob = check_graph_history_status(2592002, dataset_start=1, lookback_seconds=86400, dataset_end_exclusive=2592001)
    assert not status_oob
    assert msg_oob == "out_of_bounds"


# -----------------------------------------------------------------------------
# 14. Sequence Split and Hour Boundaries
# -----------------------------------------------------------------------------

def test_sequence_split_and_hour_boundaries() -> None:
    assert dataset_hour_start(1) == 1
    assert dataset_hour_start(3600) == 1
    assert dataset_hour_start(3601) == 3601
    assert dataset_hour_index(1) == 1
    assert dataset_hour_index(3601) == 2

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # Valid event within split and matching expected hour 1
    ok, msg = check_sequence_boundary(100, split=train_split, expected_hour_start=1)
    assert ok
    assert msg == "available"

    # Hour mismatch
    ok_hr, msg_hr = check_sequence_boundary(3601, split=train_split, expected_hour_start=1)
    assert not ok_hr
    assert msg_hr == "hour_boundary_violation"

    # Split boundary violation (event is in validation period)
    ok_sp, msg_sp = check_sequence_boundary(604801, split=train_split, expected_hour_start=604801)
    assert not ok_sp
    assert msg_sp == "split_boundary_violation"


# -----------------------------------------------------------------------------
# 15. CLI Manifest Generation on Synthetic Fixture
# -----------------------------------------------------------------------------

def test_cli_manifest_generation_on_synthetic_fixture(tmp_path: Path) -> None:
    auth_src = Path("data/fixtures/lanl_auth_sample.txt")
    rt_src = Path("data/fixtures/lanl_redteam_sample.txt")

    events_out = tmp_path / "proc" / "authentication"
    labels_out = tmp_path / "proc" / "redteam_labels"

    ingest_authentication(auth_src, events_out, IngestionConfig(day_end=3, chunk_rows=3))
    ingest_redteam_labels(rt_src, labels_out, IngestionConfig(day_end=3, chunk_rows=1))

    # Create config for 3-split fixture (train Day 1, validation Day 2, test Day 3)
    fixture_config = SplitConfig(
        policy_version="1.0.0-fixture",
        policy_name="fixture_splits",
        dataset_day_start_inclusive=1,
        dataset_day_end_inclusive=3,
        timestamp_start_inclusive=1,
        timestamp_end_exclusive=259201,
        graph_lookback_seconds=86400,
        sequence_hour_seconds=3600,
        exclusion_policy="train_redteam_users_source_or_destination",
        exclusion_window="entire_training_interval",
        splits={
            "train": SplitInterval(
                name="train",
                dataset_day_start=1,
                dataset_day_end=1,
                timestamp_start=1,
                timestamp_end=86401,
            ),
            "validation": SplitInterval(
                name="validation",
                dataset_day_start=2,
                dataset_day_end=2,
                timestamp_start=86401,
                timestamp_end=172801,
            ),
            "test": SplitInterval(
                name="test",
                dataset_day_start=3,
                dataset_day_end=3,
                timestamp_start=172801,
                timestamp_end=259201,
            ),
        },
    )
    cfg_file = tmp_path / "fixture_config.json"
    cfg_file.write_text(json.dumps(fixture_config.to_dict()), encoding="utf-8")

    out_manifest = tmp_path / "fixture_manifest.json"

    manifest = generate_splits_manifest(
        config_path=cfg_file,
        events_dir=events_out / "events",
        auth_summary_path=events_out / "summary.json",
        labels_dir=labels_out / "labels",
        labels_summary_path=labels_out / "summary.json",
        selection_manifest_path=None,  # Not required for synthetic fixture
        output_manifest_path=out_manifest,
    )

    assert out_manifest.exists()
    assert manifest["reconciliation"]["all_reconciled"] is True
    assert manifest["dataset_scope"]["total_authentication_events"] == 7
    assert manifest["dataset_scope"]["total_red_team_rows"] == 2
    assert manifest["splits"]["train"]["authentication_events"] == 5
    assert manifest["splits"]["validation"]["authentication_events"] == 2
    assert manifest["splits"]["test"]["authentication_events"] == 0
    assert manifest["splits"]["train"]["graph_scoring_eligibility"]["insufficient_history"] == 5
    assert manifest["splits"]["validation"]["graph_scoring_eligibility"]["available"] == 2
    assert manifest["graph_warm_up_policy"]["scoring_status_counts"]["insufficient_history"] == 5
    assert manifest["graph_warm_up_policy"]["scoring_status_counts"]["available"] == 2
    assert (
        manifest["sequence_policy"]["unscorable_candidate_tracking"]["status"]
        == "contract_defined_not_materialized"
    )



# -----------------------------------------------------------------------------
# 16. Inversion Detection on Equal Timestamps (Source Line Tie-Breaker Failure)
# -----------------------------------------------------------------------------

def test_ordering_violation_on_equal_timestamp_inverted_source_line(tmp_path: Path) -> None:
    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # Same timestamp, but source_line goes backwards (2 -> 1)
    inverted_table = pa.table(
        {
            "timestamp": [100, 100],
            "source_line": [2, 1],
            "acting_user": ["U2", "U1"],
        }
    )
    p_dir = tmp_path / "tie_breaker_events" / "dataset_day=1"
    p_dir.mkdir(parents=True)
    pq.write_table(inverted_table, p_dir / "part-000000.parquet")

    with pytest.raises(ValueError, match="ordering violation inside batch"):
        list(stream_split_events(tmp_path / "tie_breaker_events", target_split=train_split))


# -----------------------------------------------------------------------------
# 17. Inversion Detection Across Batch Boundaries
# -----------------------------------------------------------------------------

def test_ordering_violation_across_batch_boundaries(tmp_path: Path) -> None:
    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # Create two parquet parts in day 1 where part 2 has an earlier timestamp than part 1
    part1 = pa.table(
        {
            "timestamp": [100, 200],
            "source_line": [1, 2],
            "acting_user": ["U1", "U2"],
        }
    )
    part2 = pa.table(
        {
            "timestamp": [150, 250],  # 150 is < 200 from previous part!
            "source_line": [3, 4],
            "acting_user": ["U3", "U4"],
        }
    )
    p_dir = tmp_path / "cross_batch_events" / "dataset_day=1"
    p_dir.mkdir(parents=True)
    pq.write_table(part1, p_dir / "part-000000.parquet")
    pq.write_table(part2, p_dir / "part-000001.parquet")

    with pytest.raises(ValueError, match="ordering violation across batch boundary"):
        list(
            stream_split_events(
                tmp_path / "cross_batch_events",
                target_split=train_split,
                batch_size=2,
            )
        )


# -----------------------------------------------------------------------------
# 18. Excluded User Derivation Directly from PyArrow Dataset
# -----------------------------------------------------------------------------

def test_derive_excluded_users_from_pyarrow_dataset(tmp_path: Path) -> None:
    labels_day1 = pa.table(
        {
            "timestamp": [100, 200],
            "user": ["U_TRAIN_1", "U_TRAIN_2"],
            "source_computer": ["C1", "C2"],
            "destination_computer": ["C3", "C4"],
        }
    )
    labels_day8 = pa.table(
        {
            "timestamp": [700000],
            "user": ["U_VAL_ONLY"],
            "source_computer": ["C5"],
            "destination_computer": ["C6"],
        }
    )
    lbl_dir1 = tmp_path / "labels_ds" / "dataset_day=1"
    lbl_dir2 = tmp_path / "labels_ds" / "dataset_day=8"
    lbl_dir1.mkdir(parents=True)
    lbl_dir2.mkdir(parents=True)
    pq.write_table(labels_day1, lbl_dir1 / "part-000000.parquet")
    pq.write_table(labels_day8, lbl_dir2 / "part-000000.parquet")

    dataset = ds.dataset(tmp_path / "labels_ds", format="parquet", partitioning="hive")
    excluded = derive_excluded_users(dataset, train_start=1, train_end_exclusive=604801)
    assert excluded == ["U_TRAIN_1", "U_TRAIN_2"]
    assert "U_VAL_ONLY" not in excluded


# -----------------------------------------------------------------------------
# 19. Input Validation and Defensive Boundary Guards
# -----------------------------------------------------------------------------

def test_input_validation_and_defensive_boundary_guards() -> None:
    # Non-positive timestamps in helper functions
    with pytest.raises(ValueError, match="Timestamp must be a positive integer"):
        dataset_hour_start(0)
    with pytest.raises(ValueError, match="Timestamp must be a positive integer"):
        dataset_hour_index(0)

    # Graph helpers
    with pytest.raises(ValueError, match="score_time must be >= 1"):
        graph_history_bounds(0)
    with pytest.raises(ValueError, match="lookback_seconds must be positive"):
        graph_history_bounds(100, lookback_seconds=0)
    with pytest.raises(ValueError, match="score_time must be >= 1"):
        check_graph_history_status(0)

    # Missing split name lookup
    cfg = SplitConfig.default()
    with pytest.raises(KeyError, match="Split 'nonexistent' not found"):
        cfg.get_split("nonexistent")


# -----------------------------------------------------------------------------
# 20. No-Excluded-Users Case Returns All Training Events as Eligible
# -----------------------------------------------------------------------------

def test_no_excluded_users_reconciliation_and_counting(tmp_path: Path) -> None:
    # Create synthetic events in day 1
    events_day1 = pa.table(
        {
            "timestamp": [10, 20, 30],
            "source_line": [1, 2, 3],
            "source_user": ["U1", "U2", "U3"],
            "destination_user": ["U1", "U2", "U3"],
        }
    )
    p_dir = tmp_path / "events" / "dataset_day=1"
    p_dir.mkdir(parents=True)
    pq.write_table(events_day1, p_dir / "part-000000.parquet")

    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # A: Call count_training_exclusions with empty excluded_users and day_counts
    ex_cnt, el_cnt = count_training_exclusions(
        tmp_path / "events",
        train_split,
        excluded_users=[],
        day_counts={1: 3},
    )
    assert ex_cnt == 0
    assert el_cnt == 3

    # B: Call count_training_exclusions without day_counts (reading Parquet metadata directly)
    ex_cnt2, el_cnt2 = count_training_exclusions(
        tmp_path / "events",
        train_split,
        excluded_users=[],
        day_counts=None,
    )
    assert ex_cnt2 == 0
    assert el_cnt2 == 3


# -----------------------------------------------------------------------------
# 21. Sequence Candidate Rejection Tracker Contract for Task 3.1
# -----------------------------------------------------------------------------

def test_sequence_candidate_rejection_tracker_contract() -> None:
    tracker = SequenceCandidateRejectionTracker(
        dataset_start=1,
        dataset_end_exclusive=2592001,
        hour_seconds=3600,
    )
    cfg = SplitConfig.default()
    train_split = cfg.get_split("train")

    # 1. Valid scorable candidate at timestamp 100 in hour 1 (hour start 1)
    ok1, status1 = tracker.record_candidate(
        timestamp=100,
        split=train_split,
        expected_hour_start=1,
        min_events_required=5,
        event_count=10,
    )
    assert ok1 is True
    assert status1 == ScoringStatus.AVAILABLE.value

    # 2. Rejection: split boundary violation (event is in validation period: 604801, but within dataset)
    ok2, status2 = tracker.record_candidate(
        timestamp=604801,
        split=train_split,
        expected_hour_start=604801,
    )
    assert ok2 is False
    assert status2 == ScoringStatus.SPLIT_BOUNDARY_VIOLATION.value

    # 3. Automatic classification of out_of_bounds beyond dataset end
    ok_oob_high, status_oob_high = tracker.record_candidate(
        timestamp=2592002,
        split=train_split,
    )
    assert ok_oob_high is False
    assert status_oob_high == ScoringStatus.OUT_OF_BOUNDS.value

    # 4. Automatic classification of out_of_bounds before dataset start
    ok_oob_low, status_oob_low = tracker.record_candidate(
        timestamp=0,
        split=train_split,
    )
    assert ok_oob_low is False
    assert status_oob_low == ScoringStatus.OUT_OF_BOUNDS.value

    # 5. Rejection: hour boundary violation (event timestamp 3601 is in hour 2, but expected hour 1)
    ok3, status3 = tracker.record_candidate(
        timestamp=3601,
        split=train_split,
        expected_hour_start=1,
    )
    assert ok3 is False
    assert status3 == ScoringStatus.HOUR_BOUNDARY_VIOLATION.value

    # 6. Rejection: insufficient event history / count
    ok4, status4 = tracker.record_candidate(
        timestamp=200,
        split=train_split,
        expected_hour_start=1,
        min_events_required=10,
        event_count=3,
    )
    assert ok4 is False
    assert status4 == ScoringStatus.INSUFFICIENT_HISTORY.value

    # 7. Configurable hour_seconds (e.g. 1800s half-hour intervals)
    tracker_custom_hr = SequenceCandidateRejectionTracker(hour_seconds=1800)
    # At t=1801, actual hour with 1800s is 1801 (Hour 2)
    ok_hr_cust, status_hr_cust = tracker_custom_hr.record_candidate(
        timestamp=1801,
        split=train_split,
        expected_hour_start=1,
    )
    assert ok_hr_cust is False
    assert status_hr_cust == ScoringStatus.HOUR_BOUNDARY_VIOLATION.value

    # 8. Verify summary counts
    summary = tracker.summary()
    assert summary["total_candidates"] == 6
    assert summary["scorable_candidates"] == 1
    assert summary["rejected_candidates"] == 5
    assert summary["status_counts"][ScoringStatus.AVAILABLE.value] == 1
    assert summary["status_counts"][ScoringStatus.SPLIT_BOUNDARY_VIOLATION.value] == 1
    assert summary["status_counts"][ScoringStatus.OUT_OF_BOUNDS.value] == 2
    assert summary["status_counts"][ScoringStatus.HOUR_BOUNDARY_VIOLATION.value] == 1
    assert summary["status_counts"][ScoringStatus.INSUFFICIENT_HISTORY.value] == 1


# -----------------------------------------------------------------------------
# 22. Dynamic Label Matching Aggregation from Selection Manifest Daily Coverage
# -----------------------------------------------------------------------------

def test_dynamic_label_matching_calculation_across_splits(tmp_path: Path) -> None:
    # Create fake daily_coverage spanning Days 1 to 3
    daily_coverage = [
        {"day": 1, "authentication_records": 100, "label_rows": 2, "unique_labels": 2, "matched_unique_labels": 2, "unmatched_unique_labels": 0},
        {"day": 2, "authentication_records": 200, "label_rows": 5, "unique_labels": 5, "matched_unique_labels": 3, "unmatched_unique_labels": 2},
        {"day": 3, "authentication_records": 150, "label_rows": 1, "unique_labels": 1, "matched_unique_labels": 1, "unmatched_unique_labels": 0},
    ]
    selection_manifest = {
        "manifest_version": 1,
        "daily_coverage": daily_coverage,
        "red_team_authentication_matching": {
            "matched_unique_labels": 6,
            "unmatched_unique_labels": 2,
            "matched_authentication_rows": 6,
        },
    }
    sel_path = tmp_path / "selection_manifest.json"
    sel_path.write_text(json.dumps(selection_manifest), encoding="utf-8")

    # Ingest synthetic data for 3 days
    auth_src = Path("data/fixtures/lanl_auth_sample.txt")
    rt_src = Path("data/fixtures/lanl_redteam_sample.txt")
    events_out = tmp_path / "proc" / "authentication"
    labels_out = tmp_path / "proc" / "redteam_labels"
    ingest_authentication(auth_src, events_out, IngestionConfig(day_end=3, chunk_rows=3))
    ingest_redteam_labels(rt_src, labels_out, IngestionConfig(day_end=3, chunk_rows=1))

    fixture_config = SplitConfig(
        policy_version="1.0.0-fixture",
        policy_name="fixture_splits",
        dataset_day_start_inclusive=1,
        dataset_day_end_inclusive=3,
        timestamp_start_inclusive=1,
        timestamp_end_exclusive=259201,
        graph_lookback_seconds=86400,
        sequence_hour_seconds=3600,
        exclusion_policy="train_redteam_users_source_or_destination",
        exclusion_window="entire_training_interval",
        splits={
            "train": SplitInterval(name="train", dataset_day_start=1, dataset_day_end=1, timestamp_start=1, timestamp_end=86401),
            "validation": SplitInterval(name="validation", dataset_day_start=2, dataset_day_end=2, timestamp_start=86401, timestamp_end=172801),
            "test": SplitInterval(name="test", dataset_day_start=3, dataset_day_end=3, timestamp_start=172801, timestamp_end=259201),
        },
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(fixture_config.to_dict()), encoding="utf-8")
    out_manifest = tmp_path / "manifest.json"

    manifest = generate_splits_manifest(
        config_path=cfg_path,
        events_dir=events_out / "events",
        auth_summary_path=events_out / "summary.json",
        labels_dir=labels_out / "labels",
        labels_summary_path=labels_out / "summary.json",
        selection_manifest_path=sel_path,
        output_manifest_path=out_manifest,
    )

    prov = manifest["label_matching_provenance"]
    breakdown = prov["split_matching_breakdown"]

    # Train (Day 1): 2 matched, 0 unmatched -> 100%
    assert breakdown["train"]["matched_unique_labels"] == 2
    assert breakdown["train"]["unmatched_unique_labels"] == 0
    assert breakdown["train"]["match_coverage_percentage"] == 100.0

    # Validation (Day 2): 3 matched, 2 unmatched -> 60%
    assert breakdown["validation"]["matched_unique_labels"] == 3
    assert breakdown["validation"]["unmatched_unique_labels"] == 2
    assert breakdown["validation"]["match_coverage_percentage"] == 60.0

    # Test (Day 3): 1 matched, 0 unmatched -> 100%
    assert breakdown["test"]["matched_unique_labels"] == 1
    assert breakdown["test"]["unmatched_unique_labels"] == 0
    assert breakdown["test"]["match_coverage_percentage"] == 100.0

    # Totals
    assert prov["matched_unique_labels_total"] == 6
    assert prov["unmatched_unique_labels_total"] == 2
    assert prov["unmatched_labels_by_day"] == {"2": 2}
    assert prov["unmatched_labels_split_assignment"] == ["validation"]

    # Dynamic evaluation protocol note
    assert "2 unmatched labels" in manifest["evaluation_protocol_note"]


# -----------------------------------------------------------------------------
# 23. Incomplete and Inconsistent Provenance Manifests Are Rejected
# -----------------------------------------------------------------------------

def test_incomplete_and_inconsistent_provenance_manifests_are_rejected(tmp_path: Path) -> None:
    auth_src = Path("data/fixtures/lanl_auth_sample.txt")
    rt_src = Path("data/fixtures/lanl_redteam_sample.txt")
    events_out = tmp_path / "proc" / "authentication"
    labels_out = tmp_path / "proc" / "redteam_labels"
    ingest_authentication(auth_src, events_out, IngestionConfig(day_end=3, chunk_rows=3))
    ingest_redteam_labels(rt_src, labels_out, IngestionConfig(day_end=3, chunk_rows=1))

    fixture_config = SplitConfig(
        policy_version="1.0.0-fixture",
        policy_name="fixture_splits",
        dataset_day_start_inclusive=1,
        dataset_day_end_inclusive=3,
        timestamp_start_inclusive=1,
        timestamp_end_exclusive=259201,
        graph_lookback_seconds=86400,
        sequence_hour_seconds=3600,
        exclusion_policy="train_redteam_users_source_or_destination",
        exclusion_window="entire_training_interval",
        splits={
            "train": SplitInterval(name="train", dataset_day_start=1, dataset_day_end=1, timestamp_start=1, timestamp_end=86401),
            "validation": SplitInterval(name="validation", dataset_day_start=2, dataset_day_end=2, timestamp_start=86401, timestamp_end=172801),
            "test": SplitInterval(name="test", dataset_day_start=3, dataset_day_end=3, timestamp_start=172801, timestamp_end=259201),
        },
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(fixture_config.to_dict()), encoding="utf-8")

    # Case A: Missing days in daily_coverage (only Days 1 and 2 present, Day 3 missing)
    bad_manifest_missing_days = {
        "daily_coverage": [
            {"day": 1, "matched_unique_labels": 2, "unmatched_unique_labels": 0},
            {"day": 2, "matched_unique_labels": 3, "unmatched_unique_labels": 2},
        ],
        "red_team_authentication_matching": {"matched_unique_labels": 5, "unmatched_unique_labels": 2},
    }
    p_missing_days = tmp_path / "missing_days.json"
    p_missing_days.write_text(json.dumps(bad_manifest_missing_days), encoding="utf-8")

    with pytest.raises(ValueError, match="missing expected dataset days"):
        generate_splits_manifest(
            config_path=cfg_path,
            events_dir=events_out / "events",
            auth_summary_path=events_out / "summary.json",
            labels_dir=labels_out / "labels",
            labels_summary_path=labels_out / "summary.json",
            selection_manifest_path=p_missing_days,
            output_manifest_path=tmp_path / "out1.json",
        )

    # Case B: Missing required field in daily_coverage row
    bad_manifest_missing_field = {
        "daily_coverage": [
            {"day": 1, "matched_unique_labels": 2},  # missing unmatched_unique_labels
            {"day": 2, "matched_unique_labels": 3, "unmatched_unique_labels": 2},
            {"day": 3, "matched_unique_labels": 1, "unmatched_unique_labels": 0},
        ],
    }
    p_missing_field = tmp_path / "missing_field.json"
    p_missing_field.write_text(json.dumps(bad_manifest_missing_field), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        generate_splits_manifest(
            config_path=cfg_path,
            events_dir=events_out / "events",
            auth_summary_path=events_out / "summary.json",
            labels_dir=labels_out / "labels",
            labels_summary_path=labels_out / "summary.json",
            selection_manifest_path=p_missing_field,
            output_manifest_path=tmp_path / "out2.json",
        )

    # Case C: Daily totals do not reconcile against red_team_authentication_matching
    bad_manifest_mismatch = {
        "daily_coverage": [
            {"day": 1, "matched_unique_labels": 2, "unmatched_unique_labels": 0},
            {"day": 2, "matched_unique_labels": 3, "unmatched_unique_labels": 2},
            {"day": 3, "matched_unique_labels": 1, "unmatched_unique_labels": 0},
        ],
        "red_team_authentication_matching": {"matched_unique_labels": 999, "unmatched_unique_labels": 2},
    }
    p_mismatch = tmp_path / "mismatch.json"
    p_mismatch.write_text(json.dumps(bad_manifest_mismatch), encoding="utf-8")

    with pytest.raises(ValueError, match="Reconciliation error"):
        generate_splits_manifest(
            config_path=cfg_path,
            events_dir=events_out / "events",
            auth_summary_path=events_out / "summary.json",
            labels_dir=labels_out / "labels",
            labels_summary_path=labels_out / "summary.json",
            selection_manifest_path=p_mismatch,
            output_manifest_path=tmp_path / "out3.json",
        )


# -----------------------------------------------------------------------------
# 24. Partial-Day Graph Lookback Exact Counting
# -----------------------------------------------------------------------------

def test_partial_day_graph_lookback_scoring_eligibility(tmp_path: Path) -> None:
    # Events in day 1 with timestamps 10, 50,000, and 70,000
    # If lookback is 43,200 seconds (12 hours):
    # first_scorable_ts = 1 + 43200 = 43201
    # Events before 43201: timestamp 10 (1 event)
    # Events at or after 43201: timestamps 50000, 70000 (2 events)
    events_day1 = pa.table(
        {
            "timestamp": [10, 50_000, 70_000],
            "source_line": [1, 2, 3],
            "source_user": ["U1", "U2", "U3"],
            "destination_user": ["U1", "U2", "U3"],
        }
    )
    p_dir = tmp_path / "events_part" / "dataset_day=1"
    p_dir.mkdir(parents=True)
    pq.write_table(events_day1, p_dir / "part-000000.parquet")

    # Day 2 with 1 event
    events_day2 = pa.table(
        {
            "timestamp": [90_000],
            "source_line": [4],
            "source_user": ["U4"],
            "destination_user": ["U4"],
        }
    )
    p_dir2 = tmp_path / "events_part" / "dataset_day=2"
    p_dir2.mkdir(parents=True)
    pq.write_table(events_day2, p_dir2 / "part-000000.parquet")

    # Day 3 with 1 event
    events_day3 = pa.table(
        {
            "timestamp": [180_000],
            "source_line": [5],
            "source_user": ["U5"],
            "destination_user": ["U5"],
        }
    )
    p_dir3 = tmp_path / "events_part" / "dataset_day=3"
    p_dir3.mkdir(parents=True)
    pq.write_table(events_day3, p_dir3 / "part-000000.parquet")

    # Labels dir with empty label table
    empty_labels = pa.table(
        {
            "timestamp": pa.array([], type=pa.int64()),
            "user": pa.array([], type=pa.string()),
            "source_computer": pa.array([], type=pa.string()),
            "destination_computer": pa.array([], type=pa.string()),
        }
    )
    l_dir = tmp_path / "labels_part" / "dataset_day=1"
    l_dir.mkdir(parents=True)
    pq.write_table(empty_labels, l_dir / "part-000000.parquet")

    cfg_partial = SplitConfig(
        policy_version="1.0.0-fixture",
        policy_name="partial_lookback_fixture",
        dataset_day_start_inclusive=1,
        dataset_day_end_inclusive=3,
        timestamp_start_inclusive=1,
        timestamp_end_exclusive=259201,
        graph_lookback_seconds=43200,  # 12-hour partial-day lookback
        sequence_hour_seconds=3600,
        exclusion_policy="train_redteam_users_source_or_destination",
        exclusion_window="entire_training_interval",
        splits={
            "train": SplitInterval(name="train", dataset_day_start=1, dataset_day_end=1, timestamp_start=1, timestamp_end=86401),
            "validation": SplitInterval(name="validation", dataset_day_start=2, dataset_day_end=2, timestamp_start=86401, timestamp_end=172801),
            "test": SplitInterval(name="test", dataset_day_start=3, dataset_day_end=3, timestamp_start=172801, timestamp_end=259201),
        },
    )
    cfg_file = tmp_path / "cfg_part.json"
    cfg_file.write_text(json.dumps(cfg_partial.to_dict()), encoding="utf-8")
    out_man = tmp_path / "part_manifest.json"

    manifest = generate_splits_manifest(
        config_path=cfg_file,
        events_dir=tmp_path / "events_part",
        auth_summary_path=None,
        labels_dir=tmp_path / "labels_part",
        labels_summary_path=None,
        selection_manifest_path=None,
        output_manifest_path=out_man,
    )

    train_graph = manifest["splits"]["train"]["graph_scoring_eligibility"]
    # Day 1 has 3 events: 1 event before 43201, 2 events at/after 43201
    assert train_graph["insufficient_history"] == 1
    assert train_graph["available"] == 2
    assert manifest["splits"]["validation"]["graph_scoring_eligibility"]["available"] == 1
    assert manifest["splits"]["test"]["graph_scoring_eligibility"]["available"] == 1
