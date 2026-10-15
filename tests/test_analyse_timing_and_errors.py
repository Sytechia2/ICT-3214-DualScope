"""Tests for the Task 9.4 timing and error analysis helpers."""

from __future__ import annotations

import numpy as np

from scripts.analyse_timing_and_errors import campaigns, daily_ranks, hour_start


def test_daily_ranks_follow_top_k_order() -> None:
    scores = np.array([0.2, 0.9, 0.2, 0.5])
    assert daily_ranks(scores, None).tolist() == [3, 1, 4, 2]
    # A tie order reverses the two tied units.
    assert daily_ranks(scores, np.array([1, 0, 0, 0])).tolist() == [4, 1, 3, 2]


def test_campaigns_group_test_day_labels_by_user_and_day() -> None:
    day13 = 12 * 86400 + 1
    rows = [
        {"timestamp": day13 + 7200, "user": "U1@DOM1"},
        {"timestamp": day13 + 60, "user": "U1@DOM1"},
        {"timestamp": day13 + 86400 + 5, "user": "U1@DOM1"},  # day 14
        {"timestamp": 8 * 86400 + 1, "user": "U2@DOM1"},  # day 9: not a test day
        {"timestamp": 16 * 86400 + 1, "user": "U3@DOM1"},  # day 17: not a test day
    ]
    assert campaigns(rows) == {
        ("U1@DOM1", 13): [day13 + 60, day13 + 7200],
        ("U1@DOM1", 14): [day13 + 86400 + 5],
    }


def test_hour_start_matches_window_convention() -> None:
    assert hour_start(1) == 1
    assert hour_start(3600) == 1
    assert hour_start(3601) == 3601
