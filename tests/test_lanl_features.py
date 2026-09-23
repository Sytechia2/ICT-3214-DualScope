"""Comprehensive tests for DualScope Task 2.4: Shared Historical Features.

Covers:
- Hand-calculated synthetic fixture verification
- Rolling window boundaries and expiry
- Same-timestamp multi-line events and legitimate 0-second gaps
- Feature G (user-dest) vs Feature H (host-conn) distinction
- Active inactive-user memory purging
- Preprocessing Chan batch updates, constant features, empty inputs
- Unseen and missing category code separation
- Post-scaling 0.0 placeholder insertion for missing gaps
- Essential leakage tests A, B, C, D, E
- Reproducibility across batch sizes and preprocessor save/reload
- Defensive rejection of unordered input
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from dualscope.features.config import FeatureConfig
from dualscope.features.engine import HistoricalFeatureEngine
from dualscope.features.preprocessing import FeaturePreprocessor, NumericStat
from dualscope.features.schemas import (
    RAW_FEATURE_SCHEMA,
    TRANSFORMED_FEATURE_SCHEMA,
)
from dualscope.splits import SplitConfig, SplitInterval


# -----------------------------------------------------------------------------
# Fixtures and Helpers
# -----------------------------------------------------------------------------

def make_sample_raw_batch(rows: list[dict]) -> pa.RecordBatch:
    """Helper to create a normalized input RecordBatch from a list of dicts."""
    schema = pa.schema([
        ("timestamp", pa.int64()),
        ("source_user", pa.string()),
        ("destination_user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("authentication_type", pa.string()),
        ("logon_type", pa.string()),
        ("authentication_orientation", pa.string()),
        ("authentication_result", pa.string()),
        ("acting_user", pa.string()),
        ("exact_duplicate_ordinal", pa.int32()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("dataset_day", pa.int32()),
    ])

    cols: dict[str, list] = {name: [] for name in schema.names}
    for r in rows:
        for name in schema.names:
            cols[name].append(r.get(name))

    arrays = [pa.array(cols[name], type=schema.field(name).type) for name in schema.names]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def make_7_event_hand_calculated_rows() -> list[dict]:
    """The exact 7-event sequence analyzed and hand-calculated during design."""
    return [
        {
            "timestamp": 1,
            "source_line": 1,
            "source_user": "U1",
            "destination_user": "U1",
            "acting_user": "U1",
            "source_computer": "C1",
            "destination_computer": "C1",
            "authentication_type": "Negotiate",
            "logon_type": "Interactive",
            "authentication_orientation": "LogOn",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:1",
            "dataset_day": 1,
        },
        {
            "timestamp": 1,
            "source_line": 2,
            "source_user": "U1",
            "destination_user": "U1",
            "acting_user": "U1",
            "source_computer": "C1",
            "destination_computer": "C1",
            "authentication_type": "Negotiate",
            "logon_type": "Interactive",
            "authentication_orientation": "LogOn",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 2,
            "source_reference": "auth.txt:2",
            "dataset_day": 1,
        },
        {
            "timestamp": 2,
            "source_line": 3,
            "source_user": "U1",
            "destination_user": "U2",
            "acting_user": "U1",
            "source_computer": "C1",
            "destination_computer": "C2",
            "authentication_type": "?",
            "logon_type": "Network",
            "authentication_orientation": "TGS",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:3",
            "dataset_day": 1,
        },
        {
            "timestamp": 3,
            "source_line": 4,
            "source_user": "U2",
            "destination_user": "U3",
            "acting_user": "U2",
            "source_computer": "C2",
            "destination_computer": "C3",
            "authentication_type": "NTLM",
            "logon_type": "Network",
            "authentication_orientation": "LogOn",
            "authentication_result": "Fail",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:4",
            "dataset_day": 1,
        },
        {
            "timestamp": 3602,
            "source_line": 5,
            "source_user": "U1",
            "destination_user": "U1",
            "acting_user": "U1",
            "source_computer": "C3",
            "destination_computer": "C2",
            "authentication_type": "Kerberos",
            "logon_type": "Network",
            "authentication_orientation": "LogOn",
            "authentication_result": "Fail",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:5",
            "dataset_day": 1,
        },
        {
            "timestamp": 3603,
            "source_line": 6,
            "source_user": "U1",
            "destination_user": "U2",
            "acting_user": "U1",
            "source_computer": "C1",
            "destination_computer": "C2",
            "authentication_type": "Kerberos",
            "logon_type": "Network",
            "authentication_orientation": "LogOn",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:6",
            "dataset_day": 1,
        },
        {
            "timestamp": 86401,
            "source_line": 7,
            "source_user": "U1",
            "destination_user": "U4",
            "acting_user": "U1",
            "source_computer": "C1",
            "destination_computer": "C4",
            "authentication_type": "Kerberos",
            "logon_type": "Network",
            "authentication_orientation": "LogOn",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 1,
            "source_reference": "auth.txt:7",
            "dataset_day": 2,
        },
    ]


# -----------------------------------------------------------------------------
# 1. Hand-Calculated Fixture Verification
# -----------------------------------------------------------------------------

def test_hand_calculated_synthetic_fixture() -> None:
    """Verify all 8 causal features and 2 completeness flags against hand-calculated values."""
    engine = HistoricalFeatureEngine(FeatureConfig.default())
    input_batch = make_sample_raw_batch(make_7_event_hand_calculated_rows())
    raw_batch = engine.process_batch(input_batch)

    assert len(raw_batch) == 7

    # Event 1 (U1, t=1, line=1, C1->C1)
    assert raw_batch["prior_auth_count_1h"][0].as_py() == 0
    assert raw_batch["prior_failure_count_1h"][0].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][0].as_py() is None
    assert raw_batch["has_user_history"][0].as_py() is False
    assert raw_batch["prior_unique_destinations_24h"][0].as_py() == 0
    assert raw_batch["prior_user_destination_count_24h"][0].as_py() == 0
    assert raw_batch["is_new_user_destination"][0].as_py() is True
    assert raw_batch["is_new_host_connection"][0].as_py() is True
    assert raw_batch["history_complete_1h"][0].as_py() is False
    assert raw_batch["history_complete_24h"][0].as_py() is False

    # Event 2 (U1, t=1, line=2, C1->C1, same timestamp, later line)
    assert raw_batch["prior_auth_count_1h"][1].as_py() == 1
    assert raw_batch["prior_failure_count_1h"][1].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][1].as_py() == 0.0  # Legitimate 0-second gap
    assert raw_batch["has_user_history"][1].as_py() is True
    assert raw_batch["prior_unique_destinations_24h"][1].as_py() == 1
    assert raw_batch["prior_user_destination_count_24h"][1].as_py() == 1
    assert raw_batch["is_new_user_destination"][1].as_py() is False  # Already seen in line 1
    assert raw_batch["is_new_host_connection"][1].as_py() is False

    # Event 3 (U1, t=2, line=3, C1->C2, TGS)
    assert raw_batch["prior_auth_count_1h"][2].as_py() == 2
    assert raw_batch["prior_failure_count_1h"][2].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][2].as_py() == 1.0
    assert raw_batch["has_user_history"][2].as_py() is True
    assert raw_batch["prior_unique_destinations_24h"][2].as_py() == 1  # Only C1 seen
    assert raw_batch["prior_user_destination_count_24h"][2].as_py() == 0  # C2 not yet seen in 24h
    assert raw_batch["is_new_user_destination"][2].as_py() is True  # (U1, C2) is new
    assert raw_batch["is_new_host_connection"][2].as_py() is True  # (C1, C2) is new

    # Event 4 (U2, t=3, line=4, C2->C3, Fail)
    assert raw_batch["prior_auth_count_1h"][3].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][3].as_py() is None
    assert raw_batch["has_user_history"][3].as_py() is False
    assert raw_batch["is_new_user_destination"][3].as_py() is True
    assert raw_batch["is_new_host_connection"][3].as_py() is True

    # Event 5 (U1, t=3602, line=5, C3->C2, Fail)
    # Window 1h lower bound: 3602 - 3600 = 2. Events at t=1 (lines 1, 2) expired! Line 3 (t=2) is kept!
    assert raw_batch["prior_auth_count_1h"][4].as_py() == 1  # Line 3 only
    assert raw_batch["prior_failure_count_1h"][4].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][4].as_py() == 3600.0  # 3602 - 2
    assert raw_batch["has_user_history"][4].as_py() is True
    assert raw_batch["prior_unique_destinations_24h"][4].as_py() == 2  # C1 and C2
    assert raw_batch["prior_user_destination_count_24h"][4].as_py() == 1  # C2 seen in line 3
    assert raw_batch["is_new_user_destination"][4].as_py() is False  # (U1, C2) was seen in line 3
    assert raw_batch["is_new_host_connection"][4].as_py() is True  # (C3, C2) has NEVER appeared!
    assert raw_batch["history_complete_1h"][4].as_py() is True
    assert raw_batch["history_complete_24h"][4].as_py() is False

    # Event 6 (U1, t=3603, line=6, C1->C2, Success)
    # Window 1h lower bound: 3603 - 3600 = 3. Line 3 (t=2) has now expired! Line 5 (t=3602, Fail) is in window!
    assert raw_batch["prior_auth_count_1h"][5].as_py() == 1
    assert raw_batch["prior_failure_count_1h"][5].as_py() == 1  # Line 5 failed!
    assert raw_batch["seconds_since_previous_auth"][5].as_py() == 1.0  # 3603 - 3602
    assert raw_batch["has_user_history"][5].as_py() is True
    assert raw_batch["prior_unique_destinations_24h"][5].as_py() == 2  # C1, C2
    assert raw_batch["prior_user_destination_count_24h"][5].as_py() == 2  # Line 3 and Line 5
    assert raw_batch["is_new_user_destination"][5].as_py() is False
    assert raw_batch["is_new_host_connection"][5].as_py() is False  # (C1, C2) was seen in line 3

    # Event 7 (U1, t=86401, line=7, C1->C4, Success)
    # 1h lower bound = 86401 - 3600 = 82801 (all 1h expired)
    # 24h lower bound = 86401 - 86400 = 1 (lines at t=1, 2, 3602, 3603 all >= 1)
    assert raw_batch["prior_auth_count_1h"][6].as_py() == 0
    assert raw_batch["prior_failure_count_1h"][6].as_py() == 0
    assert raw_batch["seconds_since_previous_auth"][6].as_py() == 82798.0  # 86401 - 3603
    assert raw_batch["has_user_history"][6].as_py() is True
    assert raw_batch["prior_unique_destinations_24h"][6].as_py() == 2  # C1, C2
    assert raw_batch["prior_user_destination_count_24h"][6].as_py() == 0  # C4 not in 24h
    assert raw_batch["is_new_user_destination"][6].as_py() is True  # (U1, C4) is new
    assert raw_batch["is_new_host_connection"][6].as_py() is True  # (C1, C4) is new
    assert raw_batch["history_complete_1h"][6].as_py() is True
    assert raw_batch["history_complete_24h"][6].as_py() is True


# -----------------------------------------------------------------------------
# 2. Window Expiry & Boundary Semantics
# -----------------------------------------------------------------------------

def test_exact_window_lower_bound_and_expiry() -> None:
    """Verify that t - W is strictly included and t - W - 1 is strictly expired."""
    engine = HistoricalFeatureEngine(FeatureConfig(lookback_1h_seconds=100))

    # Event at t=100
    r1 = make_sample_raw_batch([{
        "timestamp": 100, "source_line": 1, "source_user": "U1", "destination_user": "U1",
        "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:1", "dataset_day": 1,
    }])
    b1 = engine.process_batch(r1)
    assert b1["prior_auth_count_1h"][0].as_py() == 0

    # Event at t=200: lower bound is 200 - 100 = 100.
    # Since r1 is at timestamp 100 >= 100, it MUST be included!
    r2 = make_sample_raw_batch([{
        "timestamp": 200, "source_line": 2, "source_user": "U1", "destination_user": "U1",
        "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:2", "dataset_day": 1,
    }])
    b2 = engine.process_batch(r2)
    assert b2["prior_auth_count_1h"][0].as_py() == 1  # r1 is included

    # Event at t=201: lower bound is 201 - 100 = 101.
    # r1 (at 100 < 101) MUST expire! r2 (at 200 >= 101) is included.
    r3 = make_sample_raw_batch([{
        "timestamp": 201, "source_line": 3, "source_user": "U1", "destination_user": "U1",
        "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:3", "dataset_day": 1,
    }])
    b3 = engine.process_batch(r3)
    assert b3["prior_auth_count_1h"][0].as_py() == 1  # Only r2 is included


# -----------------------------------------------------------------------------
# 3. Novelty Divergence: Feature G vs Feature H
# -----------------------------------------------------------------------------

def test_feature_g_vs_feature_h_divergence() -> None:
    """Verify that user-dest novelty (G) and host-host connection novelty (H) diverge correctly."""
    engine = HistoricalFeatureEngine(FeatureConfig.default())

    # Step 1: Alice logs on to C1 from C1
    r1 = make_sample_raw_batch([{
        "timestamp": 1, "source_line": 1, "source_user": "Alice", "destination_user": "Alice",
        "acting_user": "Alice", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:1", "dataset_day": 1,
    }])
    b1 = engine.process_batch(r1)
    assert b1["is_new_user_destination"][0].as_py() is True  # (Alice, C1) is new
    assert b1["is_new_host_connection"][0].as_py() is True  # (C1, C1) is new

    # Step 2: Bob connects from C1 to C1:
    # (Bob, C1) is NEW (Feature G = True)
    # (C1, C1) was already seen (Feature H = False)
    r2 = make_sample_raw_batch([{
        "timestamp": 2, "source_line": 2, "source_user": "Bob", "destination_user": "Bob",
        "acting_user": "Bob", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:2", "dataset_day": 1,
    }])
    b2 = engine.process_batch(r2)
    assert b2["is_new_user_destination"][0].as_py() is True   # Bob has never accessed C1
    assert b2["is_new_host_connection"][0].as_py() is False  # C1->C1 host edge was seen in r1!

    # Step 3: Alice connects from C2 to C1:
    # (Alice, C1) was already seen in r1 (Feature G = False)
    # (C2, C1) host connection has NEVER occurred (Feature H = True)
    r3 = make_sample_raw_batch([{
        "timestamp": 3, "source_line": 3, "source_user": "Alice", "destination_user": "Alice",
        "acting_user": "Alice", "source_computer": "C2", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:3", "dataset_day": 1,
    }])
    b3 = engine.process_batch(r3)
    assert b3["is_new_user_destination"][0].as_py() is False  # Alice already accessed C1
    assert b3["is_new_host_connection"][0].as_py() is True   # C2->C1 directed host edge is NEW!


# -----------------------------------------------------------------------------
# 4. Inactive User Purging and Memory Tracking
# -----------------------------------------------------------------------------

def test_inactive_user_purging_and_memory_instrumentation() -> None:
    """Verify that users inactive past window expiry are pruned from rolling state."""
    engine = HistoricalFeatureEngine(FeatureConfig(lookback_1h_seconds=100, lookback_24h_seconds=1000))

    # User U_Inactive acts at t=10
    r1 = make_sample_raw_batch([{
        "timestamp": 10, "source_line": 1, "source_user": "U_Inactive", "destination_user": "U_Inactive",
        "acting_user": "U_Inactive", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:1", "dataset_day": 1,
    }])
    engine.process_batch(r1)

    mem_before = engine.get_memory_breakdown()
    assert mem_before["active_rolling_users"] == 1
    assert mem_before["rolling_1h_events"] == 1
    assert mem_before["rolling_24h_events"] == 1

    # Time advances past 24h lookback (t=2000 > 10 + 1000) for another user U_Active
    r2 = make_sample_raw_batch([{
        "timestamp": 2000, "source_line": 2, "source_user": "U_Active", "destination_user": "U_Active",
        "acting_user": "U_Active", "source_computer": "C2", "destination_computer": "C2",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:2", "dataset_day": 1,
    }])
    engine.process_batch(r2)

    mem_after = engine.get_memory_breakdown()
    # U_Inactive rolling history was pruned! Only U_Active remains in rolling state.
    assert mem_after["active_rolling_users"] == 1
    assert mem_after["distinct_users_tracked"] == 2  # Total users tracked in intern table is 2

    # If U_Inactive returns much later at t=5000, gap is accurately computed and user history is True
    r3 = make_sample_raw_batch([{
        "timestamp": 5000, "source_line": 3, "source_user": "U_Inactive", "destination_user": "U_Inactive",
        "acting_user": "U_Inactive", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:3", "dataset_day": 1,
    }])
    b3 = engine.process_batch(r3)
    assert b3["seconds_since_previous_auth"][0].as_py() == 4990.0  # 5000 - 10
    assert b3["has_user_history"][0].as_py() is True
    assert b3["prior_auth_count_1h"][0].as_py() == 0  # 1h window was cleanly 0


# -----------------------------------------------------------------------------
# 5. Preprocessing: Chan Batch Accumulation, Category IDs, and Missing Gaps
# -----------------------------------------------------------------------------

def test_preprocessing_fitting_freezing_and_transform() -> None:
    """Verify Chan updates, deterministic vocabularies, and post-scaling 0.0 placeholder insertion."""
    preprocessor = FeaturePreprocessor(
        FeatureConfig.default(),
        split_policy_fingerprint="test_fp",
        mode="test",
        is_production=False,
    )

    # Batch 1 of eligible training raw rows
    rows_b1 = [
        {"prior_auth_count_1h": 0, "prior_failure_count_1h": 0, "seconds_since_previous_auth": None,
         "has_user_history": False, "prior_unique_destinations_24h": 0, "prior_user_destination_count_24h": 0,
         "is_new_user_destination": True, "is_new_host_connection": True, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 1, "source_line": 1, "source_reference": "ref:1",
         "source_user": "U1", "destination_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U1", "exact_duplicate_ordinal": 1, "dataset_day": 1,
         "authentication_type": "Negotiate", "logon_type": "Interactive", "authentication_orientation": "LogOn",
         "authentication_result": "Success"},
        {"prior_auth_count_1h": 2, "prior_failure_count_1h": 1, "seconds_since_previous_auth": 10.0,
         "has_user_history": True, "prior_unique_destinations_24h": 1, "prior_user_destination_count_24h": 1,
         "is_new_user_destination": False, "is_new_host_connection": False, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 11, "source_line": 2, "source_reference": "ref:2",
         "source_user": "U1", "destination_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U1", "exact_duplicate_ordinal": 1, "dataset_day": 1,
         "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "TGS",
         "authentication_result": "Fail"},
    ]
    raw_b1 = pa.RecordBatch.from_pylist(rows_b1, schema=RAW_FEATURE_SCHEMA)
    mask1 = pa.array([True, True], type=pa.bool_())
    preprocessor.accumulate_training_batch(raw_b1, mask1)

    # Freeze preprocessor
    preprocessor.freeze()
    assert preprocessor.is_frozen

    # Check that null gap was excluded from gap statistics fitting:
    # Only the second row (gap=10.0) was included
    gap_stat = preprocessor.numeric_stats["seconds_since_previous_auth"]
    assert gap_stat.count == 1
    assert pytest.approx(gap_stat.mean) == math.log1p(10.0)
    assert gap_stat.is_constant  # Count <= 1 => scale=1.0

    # Check categorical vocabularies
    auth_vocab = preprocessor.vocabularies["authentication_type"]
    assert auth_vocab.unseen_id == 0
    assert auth_vocab.missing_id == 1
    # Sorted training categories: "Kerberos" -> 2, "Negotiate" -> 3
    assert auth_vocab.category_to_id["Kerberos"] == 2
    assert auth_vocab.category_to_id["Negotiate"] == 3

    # Transform batch containing null gap, unseen category, and missing category
    test_rows = [
        # Row with null gap and unseen category "NTLM"
        {"prior_auth_count_1h": 0, "prior_failure_count_1h": 0, "seconds_since_previous_auth": None,
         "has_user_history": False, "prior_unique_destinations_24h": 0, "prior_user_destination_count_24h": 0,
         "is_new_user_destination": True, "is_new_host_connection": True, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 1, "source_line": 1, "source_reference": "ref:1",
         "source_user": "U1", "destination_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U1", "exact_duplicate_ordinal": 1, "dataset_day": 1,
         "authentication_type": "NTLM", "logon_type": "Interactive", "authentication_orientation": "LogOn",
         "authentication_result": "Success"},
        # Row with empty category ""
        {"prior_auth_count_1h": 2, "prior_failure_count_1h": 1, "seconds_since_previous_auth": 10.0,
         "has_user_history": True, "prior_unique_destinations_24h": 1, "prior_user_destination_count_24h": 1,
         "is_new_user_destination": False, "is_new_host_connection": False, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 11, "source_line": 2, "source_reference": "ref:2",
         "source_user": "U1", "destination_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U1", "exact_duplicate_ordinal": 1, "dataset_day": 1,
         "authentication_type": "", "logon_type": "Network", "authentication_orientation": "TGS",
         "authentication_result": "Fail"},
    ]
    raw_test = pa.RecordBatch.from_pylist(test_rows, schema=RAW_FEATURE_SCHEMA)
    transformed = preprocessor.transform_batch(raw_test)

    # 1. Null gap is transformed to 0.0 post-scaling placeholder
    assert transformed["log1p_seconds_since_previous_auth_scaled"][0].as_py() == 0.0
    assert transformed["has_user_history"][0].as_py() == 0.0

    # 2. Observed gap (10.0) is standardized: (log1p(10.0) - mean) / scale = 0.0 since mean=log1p(10.0)
    assert pytest.approx(transformed["log1p_seconds_since_previous_auth_scaled"][1].as_py()) == 0.0
    assert transformed["has_user_history"][1].as_py() == 1.0  # Distinguishes from missing gap!

    # 3. Unseen category "NTLM" maps to 0 (<UNSEEN>)
    assert transformed["auth_type_id"][0].as_py() == 0
    # 4. Empty category "" maps to 1 (<MISSING>)
    assert transformed["auth_type_id"][1].as_py() == 1


def test_constant_feature_and_empty_training_input() -> None:
    """Verify fallback behavior when a feature is constant or has zero training observations."""
    preprocessor = FeaturePreprocessor(FeatureConfig.default())

    # Empty batch accumulation
    empty_raw = pa.RecordBatch.from_pylist([], schema=RAW_FEATURE_SCHEMA)
    preprocessor.accumulate_training_batch(empty_raw, pa.array([], type=pa.bool_()))
    preprocessor.freeze()

    # Fallback: scale=1.0, mean=0.0
    for stat in preprocessor.numeric_stats.values():
        assert stat.scale == 1.0
        assert stat.mean == 0.0


# -----------------------------------------------------------------------------
# 6. Essential Leakage Tests (A, B, C, D, E)
# -----------------------------------------------------------------------------

def test_leakage_a_append_later_events_does_not_change_earlier_raw_features() -> None:
    """Test A: Causal integrity — earlier raw feature rows must not change when future events arrive."""
    rows = make_7_event_hand_calculated_rows()
    b_part1 = make_sample_raw_batch(rows[:4])
    b_part2 = make_sample_raw_batch(rows[4:])

    # Run 1: process only part 1
    engine1 = HistoricalFeatureEngine(FeatureConfig.default())
    raw1 = engine1.process_batch(b_part1)

    # Run 2: process part 1, then append part 2
    engine2 = HistoricalFeatureEngine(FeatureConfig.default())
    raw2_part1 = engine2.process_batch(b_part1)
    raw2_part2 = engine2.process_batch(b_part2)

    # Earlier raw feature rows must be bit-for-bit identical
    for col in RAW_FEATURE_SCHEMA.names:
        assert raw1[col].equals(raw2_part1[col]), f"Mismatch in col {col} after future events arrived"


def test_leakage_b_validation_test_events_do_not_alter_training_preprocessor() -> None:
    """Test B: Training-only preprocessor is strictly unaltered by validation/test events."""
    preprocessor = FeaturePreprocessor(FeatureConfig.default())

    train_rows = [
        {"prior_auth_count_1h": 1, "prior_failure_count_1h": 0, "seconds_since_previous_auth": 5.0,
         "has_user_history": True, "prior_unique_destinations_24h": 1, "prior_user_destination_count_24h": 1,
         "is_new_user_destination": False, "is_new_host_connection": False, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 100, "source_line": 1, "source_reference": "ref:1",
         "source_user": "U1", "destination_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U1", "exact_duplicate_ordinal": 1, "dataset_day": 1,
         "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
         "authentication_result": "Success"},
    ]
    raw_train = pa.RecordBatch.from_pylist(train_rows, schema=RAW_FEATURE_SCHEMA)
    preprocessor.accumulate_training_batch(raw_train, pa.array([True], type=pa.bool_()))

    mean_before = preprocessor.numeric_stats["prior_auth_count_1h"].mean
    cats_before = set(preprocessor._cat_candidates["authentication_type"])

    # Simulate validation/test batch with extreme numbers and novel category
    val_rows = [
        {"prior_auth_count_1h": 999999, "prior_failure_count_1h": 999999, "seconds_since_previous_auth": 999999.0,
         "has_user_history": True, "prior_unique_destinations_24h": 999999, "prior_user_destination_count_24h": 999999,
         "is_new_user_destination": False, "is_new_host_connection": False, "history_complete_1h": False,
         "history_complete_24h": False, "timestamp": 1_000_000, "source_line": 2, "source_reference": "ref:2",
         "source_user": "U_Val", "destination_user": "U_Val", "source_computer": "C1", "destination_computer": "C1",
         "acting_user": "U_Val", "exact_duplicate_ordinal": 1, "dataset_day": 10,
         "authentication_type": "NovelValidationAuthType", "logon_type": "Network", "authentication_orientation": "LogOn",
         "authentication_result": "Success"},
    ]
    raw_val = pa.RecordBatch.from_pylist(val_rows, schema=RAW_FEATURE_SCHEMA)
    # Eligibility mask for validation rows is strictly False
    val_eligibility_mask = pa.array([False], type=pa.bool_())
    preprocessor.accumulate_training_batch(raw_val, val_eligibility_mask)

    # Statistics and category sets must not change!
    assert preprocessor.numeric_stats["prior_auth_count_1h"].mean == mean_before
    assert preprocessor._cat_candidates["authentication_type"] == cats_before


def test_leakage_c_frozen_preprocessor_transforms_earlier_events_identically() -> None:
    """Test C: With the same frozen preprocessor, transforming earlier batches is deterministic."""
    preprocessor = FeaturePreprocessor(FeatureConfig.default())
    rows = make_7_event_hand_calculated_rows()
    b_part1 = make_sample_raw_batch(rows[:4])
    b_part2 = make_sample_raw_batch(rows[4:])

    engine = HistoricalFeatureEngine(FeatureConfig.default())
    raw_part1 = engine.process_batch(b_part1)
    raw_part2 = engine.process_batch(b_part2)

    preprocessor.accumulate_training_batch(raw_part1, pa.array([True] * len(raw_part1), type=pa.bool_()))
    preprocessor.freeze()

    trans1 = preprocessor.transform_batch(raw_part1)
    _ = preprocessor.transform_batch(raw_part2)
    trans1_again = preprocessor.transform_batch(raw_part1)

    for col in TRANSFORMED_FEATURE_SCHEMA.names:
        assert trans1[col].equals(trans1_again[col])


def test_leakage_d_excluded_training_rows_update_replay_but_not_statistics() -> None:
    """Test D: Excluded training rows update replay state but contribute zero statistics to fitting."""
    engine = HistoricalFeatureEngine(FeatureConfig.default())
    preprocessor = FeaturePreprocessor(FeatureConfig.default())

    excluded_user = "U_EXCLUDED@DOM1"
    rows = [
        # Normal event at t=1
        {"timestamp": 1, "source_line": 1, "source_user": "U_Clean", "destination_user": "U_Clean",
         "acting_user": "U_Clean", "source_computer": "C1", "destination_computer": "C1",
         "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:1",
         "dataset_day": 1},
        # Excluded event at t=2
        {"timestamp": 2, "source_line": 2, "source_user": excluded_user, "destination_user": "C2",
         "acting_user": excluded_user, "source_computer": "C1", "destination_computer": "C2",
         "authentication_type": "CompromiseAuthType", "logon_type": "Network", "authentication_orientation": "LogOn",
         "authentication_result": "Fail", "exact_duplicate_ordinal": 1, "source_reference": "ref:2",
         "dataset_day": 1},
        # Later excluded user event at t=3
        {"timestamp": 3, "source_line": 3, "source_user": excluded_user, "destination_user": "C2",
         "acting_user": excluded_user, "source_computer": "C1", "destination_computer": "C2",
         "authentication_type": "CompromiseAuthType", "logon_type": "Network", "authentication_orientation": "LogOn",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:3",
         "dataset_day": 1},
    ]
    raw_batch = engine.process_batch(make_sample_raw_batch(rows))

    # Excluded event at t=3 MUST see prior replay history from t=2
    assert raw_batch["prior_auth_count_1h"][2].as_py() == 1
    assert raw_batch["prior_failure_count_1h"][2].as_py() == 1
    assert raw_batch["seconds_since_previous_auth"][2].as_py() == 1.0  # 3 - 2

    # But when fitting preprocessor, excluded rows are masked out!
    eligibility_mask = pa.array([True, False, False], type=pa.bool_())
    preprocessor.accumulate_training_batch(raw_batch, eligibility_mask)
    preprocessor.freeze()

    # Preprocessor fitted count must be exactly 1
    assert preprocessor.training_eligible_fitted_rows == 1
    # Category "CompromiseAuthType" must NOT be in vocabulary!
    auth_vocab = preprocessor.vocabularies["authentication_type"]
    assert "CompromiseAuthType" not in auth_vocab.category_to_id
    assert "Kerberos" in auth_vocab.category_to_id


# -----------------------------------------------------------------------------
# 7. Reproducibility & Defensive Guards
# -----------------------------------------------------------------------------

def test_reproducibility_across_different_batch_sizes() -> None:
    """Verify that batch chunk boundaries do not affect computed raw feature values."""
    rows = make_7_event_hand_calculated_rows()

    # Run A: single batch of 7 rows
    engine_a = HistoricalFeatureEngine(FeatureConfig.default())
    raw_a = engine_a.process_batch(make_sample_raw_batch(rows))

    # Run B: batches of size 1
    engine_b = HistoricalFeatureEngine(FeatureConfig.default())
    batches_b = [engine_b.process_batch(make_sample_raw_batch([r])) for r in rows]
    combined_b = pa.Table.from_batches(batches_b).combine_chunks()

    # Run C: batches of sizes [2, 3, 2]
    engine_c = HistoricalFeatureEngine(FeatureConfig.default())
    batches_c = [
        engine_c.process_batch(make_sample_raw_batch(rows[:2])),
        engine_c.process_batch(make_sample_raw_batch(rows[2:5])),
        engine_c.process_batch(make_sample_raw_batch(rows[5:])),
    ]
    combined_c = pa.Table.from_batches(batches_c).combine_chunks()

    for col in RAW_FEATURE_SCHEMA.names:
        assert raw_a[col].equals(combined_b[col].chunk(0))
        assert raw_a[col].equals(combined_c[col].chunk(0))


def test_preprocessor_save_and_reload(tmp_path: Path) -> None:
    """Verify that saving and loading FeaturePreprocessor JSON produces identical transformations."""
    preprocessor = FeaturePreprocessor(FeatureConfig.default(), split_policy_fingerprint="fp123")
    rows = make_7_event_hand_calculated_rows()
    raw_batch = HistoricalFeatureEngine(FeatureConfig.default()).process_batch(make_sample_raw_batch(rows))

    preprocessor.accumulate_training_batch(raw_batch, pa.array([True] * len(raw_batch), type=pa.bool_()))
    preprocessor.freeze()

    save_path = tmp_path / "preprocessor.json"
    preprocessor.save(save_path)

    reloaded = FeaturePreprocessor.load(save_path)
    assert reloaded.is_frozen
    assert reloaded.split_policy_fingerprint == "fp123"

    t1 = preprocessor.transform_batch(raw_batch)
    t2 = reloaded.transform_batch(raw_batch)

    for col in TRANSFORMED_FEATURE_SCHEMA.names:
        assert t1[col].equals(t2[col])


def test_defensive_rejection_of_unordered_events() -> None:
    """Verify that timestamp or source_line inversions are rejected with clear errors."""
    engine = HistoricalFeatureEngine(FeatureConfig.default())

    # Timestamp inversion: t=10 followed by t=9
    inverted_ts_rows = [
        {"timestamp": 10, "source_line": 1, "source_user": "U1", "destination_user": "U1",
         "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:1",
         "dataset_day": 1},
        {"timestamp": 9, "source_line": 2, "source_user": "U1", "destination_user": "U1",
         "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:2",
         "dataset_day": 1},
    ]
    with pytest.raises(ValueError, match="Deterministic ordering violation"):
        engine.process_batch(make_sample_raw_batch(inverted_ts_rows))

    # Same timestamp, inverted source_line: t=10 line=5 followed by t=10 line=4
    engine2 = HistoricalFeatureEngine(FeatureConfig.default())
    inverted_line_rows = [
        {"timestamp": 10, "source_line": 5, "source_user": "U1", "destination_user": "U1",
         "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:1",
         "dataset_day": 1},
        {"timestamp": 10, "source_line": 4, "source_user": "U1", "destination_user": "U1",
         "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
         "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
         "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:2",
         "dataset_day": 1},
    ]
    with pytest.raises(ValueError, match="Deterministic ordering violation"):
        engine2.process_batch(make_sample_raw_batch(inverted_line_rows))


# -----------------------------------------------------------------------------
# 8. Additional Boundary, Split Continuity, and CLI Tests
# -----------------------------------------------------------------------------

def test_leakage_e_validation_and_test_rows_cannot_fit_preprocessing() -> None:
    """Test E: Strict enforcement that only timestamps within [train_start, train_end) can fit."""
    preprocessor = FeaturePreprocessor(FeatureConfig.default())

    # Training split is [1, 604801). Row at timestamp 604801 is in Validation!
    val_row = {
        "timestamp": 604801, "source_line": 1, "source_user": "U1", "destination_user": "U1",
        "acting_user": "U1", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "Kerberos", "logon_type": "Network", "authentication_orientation": "LogOn",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1, "source_reference": "ref:1",
        "dataset_day": 8, "prior_auth_count_1h": 1, "prior_failure_count_1h": 0,
        "seconds_since_previous_auth": 1.0, "has_user_history": True,
        "prior_unique_destinations_24h": 1, "prior_user_destination_count_24h": 1,
        "is_new_user_destination": False, "is_new_host_connection": False,
        "history_complete_1h": True, "history_complete_24h": True,
    }
    raw_batch = pa.RecordBatch.from_pylist([val_row], schema=RAW_FEATURE_SCHEMA)

    # In pipeline, eligibility mask is: in_train & in_exclusion_filter
    in_train = (val_row["timestamp"] < 604801)
    eligibility_mask = pa.array([in_train], type=pa.bool_())
    preprocessor.accumulate_training_batch(raw_batch, eligibility_mask)
    preprocessor.freeze()

    # Zero rows fitted
    assert preprocessor.training_eligible_fitted_rows == 0
    assert preprocessor.numeric_stats["prior_auth_count_1h"].count == 0


def test_state_continuity_across_splits_and_days() -> None:
    """Verify that engine maintains state across days and splits without resetting."""
    engine = HistoricalFeatureEngine(FeatureConfig.default())

    # Event on Day 1 (Train): t=1
    r1 = make_sample_raw_batch([{
        "timestamp": 1, "source_line": 1, "source_user": "Alice", "destination_user": "Alice",
        "acting_user": "Alice", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:1", "dataset_day": 1,
    }])
    b1 = engine.process_batch(r1)
    assert b1["prior_auth_count_1h"][0].as_py() == 0

    # Event crossing into Day 8 (Validation split boundary is 604801)
    # Event at t=604801: Alice logs on to C1 again!
    r2 = make_sample_raw_batch([{
        "timestamp": 604801, "source_line": 2, "source_user": "Alice", "destination_user": "Alice",
        "acting_user": "Alice", "source_computer": "C1", "destination_computer": "C1",
        "authentication_type": "K", "logon_type": "N", "authentication_orientation": "L",
        "authentication_result": "Success", "exact_duplicate_ordinal": 1,
        "source_reference": "ref:2", "dataset_day": 8,
    }])
    b2 = engine.process_batch(r2)

    # History is NOT reset: Alice is known, relationship (Alice, C1) is NOT new
    assert b2["has_user_history"][0].as_py() is True
    assert b2["seconds_since_previous_auth"][0].as_py() == 604800.0  # 604801 - 1
    assert b2["is_new_user_destination"][0].as_py() is False
    assert b2["is_new_host_connection"][0].as_py() is False


def test_raw_record_is_excluded_from_schemas() -> None:
    """Verify that raw_record is strictly absent from RAW and TRANSFORMED schemas."""
    assert "raw_record" not in RAW_FEATURE_SCHEMA.names
    assert "raw_record" not in TRANSFORMED_FEATURE_SCHEMA.names
    assert "source_reference" in RAW_FEATURE_SCHEMA.names
    assert "source_reference" in TRANSFORMED_FEATURE_SCHEMA.names


def test_history_complete_flags_are_metadata_not_model_inputs() -> None:
    """Verify that history_complete_1h and 24h are metadata flags, not in model inputs."""
    cfg = FeatureConfig.default()
    all_inputs = cfg.all_model_input_names()
    assert "history_complete_1h" not in all_inputs
    assert "history_complete_24h" not in all_inputs
    assert "has_user_history" in all_inputs
    assert "history_complete_1h" in cfg.metadata_columns
    assert "history_complete_24h" in cfg.metadata_columns


def test_cli_two_pass_on_sample(tmp_path: Path) -> None:
    """Test full CLI two-pass execution on data/samples/lanl_ingestion_sample."""
    from scripts.build_lanl_features import parse_args, run_pipeline
    import pyarrow.dataset as ds

    sample_events = Path("data/samples/lanl_ingestion_sample/authentication/events")
    if not sample_events.exists():
        pytest.skip("data/samples/lanl_ingestion_sample not present")

    out_dir = tmp_path / "lanl_features_sample_test"

    args = parse_args.__wrapped__ if hasattr(parse_args, "__wrapped__") else None
    # Construct namespace directly
    import argparse
    cli_args = argparse.Namespace(
        events=sample_events,
        splits_config=Path("data/fixtures/fixture_splits.json"),
        splits_manifest=Path("data/manifests/fixture_splits_v1.json"),
        feature_config=Path("data/fixtures/fixture_features.json"),
        output=out_dir,
        preprocessing_path=None,
        batch_size=3,
        pilot_rows=None,
        pilot_days=None,
        pilot_mode=True,
        raw_only=False,
        transform_only=False,
        overwrite=True,
    )

    summary = run_pipeline(cli_args)
    assert summary["counts"]["reconciled"] is True
    assert summary["counts"]["raw_events_written"] == 7
    assert summary["counts"]["transformed_events_written"] == 7

    # Check that raw and transformed directories exist and can be loaded
    raw_ds = ds.dataset(str(out_dir / "raw" / "events"), format="parquet", partitioning="hive")
    assert raw_ds.count_rows() == 7
    trans_ds = ds.dataset(str(out_dir / "transformed" / "events"), format="parquet", partitioning="hive")
    assert trans_ds.count_rows() == 7

    # Check preprocessor JSON metadata and exclusion reconciliation
    prep_data = json.loads((out_dir / "preprocessing.json").read_text(encoding="utf-8"))
    assert prep_data["is_production"] is False
    assert prep_data["mode"] == "pilot"
    counts = prep_data["fitting_row_counts"]
    assert counts["total_scanned_rows"] == 7
    assert counts["training_eligible_fitted_rows"] == 5
    assert counts["training_excluded_skipped_rows"] == 2


def test_timeline_queue_cleanup_under_continuous_events() -> None:
    """Verify that under continuous 1-second events, periodic sweeper bounds timeline queue entries."""
    # Configure small lookback windows and sweep interval: w1=10s, w24=20s, sweep_interval=5s
    config = FeatureConfig(lookback_1h_seconds=10, lookback_24h_seconds=20)
    engine = HistoricalFeatureEngine(config, sweep_interval_seconds=5)

    # Generate 100 consecutive 1-second timestamps: t = 1..100
    rows = [
        {
            "timestamp": t,
            "source_line": t,
            "source_user": f"U{t}",
            "destination_user": f"U{t}",
            "acting_user": f"U{t}",
            "source_computer": f"C{t}",
            "destination_computer": f"C{t}",
            "authentication_type": "Kerberos",
            "logon_type": "Interactive",
            "authentication_orientation": "LogOn",
            "authentication_result": "Success",
            "exact_duplicate_ordinal": 1,
            "source_reference": f"ref:{t}",
            "dataset_day": 1,
        }
        for t in range(1, 101)
    ]

    engine.process_batch(make_sample_raw_batch(rows))
    engine.purge_inactive_users(100)

    mem = engine.get_memory_breakdown()
    # Under old logic without elapsed time sweeper, timeline_1h and timeline_24h accumulated 100 entries.
    # Under new periodic sweeper:
    # At t=100:
    # 10s lookback (w1=10) retains at most entries from t >= 100 - 10 = 90 -> 11 entries
    # 20s lookback (w24=20) retains at most entries from t >= 100 - 20 = 80 -> 21 entries
    assert mem["timeline_1h_entries"] == 11
    assert mem["timeline_24h_entries"] == 21
    assert mem["timeline_memory_mb"] >= 0.0


def test_splits_manifest_validation_and_exclusion_loading(tmp_path: Path) -> None:
    """Verify that load_excluded_users validates fingerprints, accepts empty lists, and rejects mismatches."""
    from scripts.build_lanl_features import load_excluded_users

    splits_cfg = SplitConfig.from_file(Path("data/fixtures/fixture_splits.json"))
    expected_fp = splits_cfg.fingerprint()

    # 1. Valid fixture manifest
    fixture_manifest = Path("data/manifests/fixture_splits_v1.json")
    excluded = load_excluded_users(fixture_manifest, splits_cfg, is_production=True)
    assert excluded == ["U2@DOM1"]

    # 2. Valid empty exclusions list
    empty_manifest = tmp_path / "empty_splits_manifest.json"
    empty_manifest.write_text(json.dumps({
        "config_fingerprint": expected_fp,
        "training_exclusions": {"excluded_users": []}
    }), encoding="utf-8")
    excluded_empty = load_excluded_users(empty_manifest, splits_cfg, is_production=True)
    assert excluded_empty == []

    # 3. Mismatched config fingerprint
    mismatched_manifest = tmp_path / "mismatched_splits_manifest.json"
    mismatched_manifest.write_text(json.dumps({
        "config_fingerprint": "wrong_fingerprint",
        "training_exclusions": {"excluded_users": ["U1"]}
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Splits manifest config_fingerprint mismatch"):
        load_excluded_users(mismatched_manifest, splits_cfg, is_production=True)

    # 4. Missing training_exclusions section
    missing_section_manifest = tmp_path / "missing_section_manifest.json"
    missing_section_manifest.write_text(json.dumps({
        "config_fingerprint": expected_fp,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'training_exclusions' section"):
        load_excluded_users(missing_section_manifest, splits_cfg, is_production=True)

    # 5. Missing manifest in production mode
    with pytest.raises(ValueError, match="Production feature building requires a valid splits manifest"):
        load_excluded_users(None, splits_cfg, is_production=True)


def test_preprocessor_verify_compatibility(tmp_path: Path) -> None:
    """Verify that preprocessor.verify_compatibility and load enforce config matching."""
    feature_cfg = FeatureConfig.default()
    splits_cfg = SplitConfig.from_file(Path("config/lanl_splits.json"))

    prep = FeaturePreprocessor(feature_cfg, split_policy_fingerprint=splits_cfg.fingerprint())
    prep.freeze()

    # Matching verification succeeds
    prep.verify_compatibility(feature_cfg=feature_cfg, splits_cfg=splits_cfg)

    # Mismatched FeatureConfig
    diff_feature_cfg = FeatureConfig(lookback_1h_seconds=999)
    with pytest.raises(ValueError, match="FeatureConfig fingerprint mismatch"):
        prep.verify_compatibility(feature_cfg=diff_feature_cfg)

    # Mismatched SplitConfig
    diff_splits_cfg = SplitConfig.from_file(Path("data/fixtures/fixture_splits.json"))
    with pytest.raises(ValueError, match="SplitConfig fingerprint mismatch"):
        prep.verify_compatibility(splits_cfg=diff_splits_cfg)

    # Save and load with compatibility check
    save_file = tmp_path / "prep.json"
    prep.save(save_file)

    # Loading with matching configs succeeds
    loaded = FeaturePreprocessor.load(save_file, config=feature_cfg, splits_cfg=splits_cfg)
    assert loaded.is_frozen

    # Loading with mismatched splits_cfg raises ValueError
    with pytest.raises(ValueError, match="SplitConfig fingerprint mismatch"):
        FeaturePreprocessor.load(save_file, config=feature_cfg, splits_cfg=diff_splits_cfg)

    # Missing feature_config_fingerprint in artifact
    no_feature_fp_data = json.loads(save_file.read_text(encoding="utf-8"))
    del no_feature_fp_data["feature_config_fingerprint"]
    no_feature_fp_file = tmp_path / "prep_no_feature_fp.json"
    no_feature_fp_file.write_text(json.dumps(no_feature_fp_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing 'feature_config_fingerprint'"):
        FeaturePreprocessor.load(no_feature_fp_file, config=feature_cfg)

    unverified_feature = FeaturePreprocessor.load(no_feature_fp_file)
    with pytest.raises(ValueError, match="Missing 'feature_config_fingerprint'"):
        unverified_feature.verify_compatibility(feature_cfg=feature_cfg)

    # Missing split_policy_fingerprint in artifact
    no_split_fp_data = json.loads(save_file.read_text(encoding="utf-8"))
    del no_split_fp_data["split_policy_fingerprint"]
    no_split_fp_file = tmp_path / "prep_no_split_fp.json"
    no_split_fp_file.write_text(json.dumps(no_split_fp_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing 'split_policy_fingerprint'"):
        FeaturePreprocessor.load(no_split_fp_file, splits_cfg=splits_cfg)

    unverified_split = FeaturePreprocessor.load(no_split_fp_file)
    with pytest.raises(ValueError, match="Missing 'split_policy_fingerprint'"):
        unverified_split.verify_compatibility(splits_cfg=splits_cfg)
