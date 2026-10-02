"""Task 9.2: Aggregate per-event historical features to user-hour tabular vectors.

Extracts fixed-length 16-dimensional summary representations from the 12 Task 2.4
event features for standard tabular anomaly baselines (e.g. Isolation Forest).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pyarrow as pa


BASELINE_FEATURE_NAMES: list[str] = [
    "log1p_event_count_1h",
    "mean_prior_auth_count_1h",
    "max_prior_auth_count_1h",
    "mean_prior_failure_count_1h",
    "max_prior_failure_count_1h",
    "min_seconds_since_prev",
    "mean_seconds_since_prev",
    "max_unique_destinations_24h",
    "max_user_dest_count_24h",
    "new_user_dest_rate",
    "new_host_conn_rate",
    "has_user_history_rate",
    "auth_type_mode",
    "logon_type_mode",
    "auth_orientation_mode",
    "failure_rate_1h",
]


@dataclass(frozen=True)
class UserHourUnit:
    """Represents an active user-hour and its contributing events."""

    user_id: str
    window_start: int
    window_end: int
    source_lines: list[int]
    feature_vector: np.ndarray


def aggregate_events_to_user_hours(
    events_table: pa.Table,
    hour_seconds: int = 3600,
) -> list[UserHourUnit]:
    """Group transformed event rows by (acting_user, hour_start) and aggregate features."""
    if events_table.num_rows == 0:
        return []

    # Extract columns as python / numpy arrays for grouping
    timestamps = events_table["timestamp"].to_numpy()
    acting_users = events_table["acting_user"].to_pylist()
    source_lines = events_table["source_line"].to_numpy()

    # Numeric & binary features
    f_auth_1h = events_table["log1p_prior_auth_count_1h_scaled"].to_numpy()
    f_fail_1h = events_table["log1p_prior_failure_count_1h_scaled"].to_numpy()
    f_secs_prev = events_table["log1p_seconds_since_previous_auth_scaled"].to_numpy()
    f_uniq_dest = events_table["log1p_prior_unique_destinations_24h_scaled"].to_numpy()
    f_user_dest = events_table["log1p_prior_user_destination_count_24h_scaled"].to_numpy()
    f_hist = events_table["has_user_history"].to_numpy()
    f_new_dest = events_table["is_new_user_destination"].to_numpy()
    f_new_host = events_table["is_new_host_connection"].to_numpy()

    # Categorical IDs
    f_auth_type = events_table["auth_type_id"].to_numpy()
    f_logon_type = events_table["logon_type_id"].to_numpy()
    f_orient = events_table["auth_orientation_id"].to_numpy()
    f_result = events_table["auth_result_id"].to_numpy()

    # Group by (user, hour_start)
    groups: dict[tuple[str, int], list[int]] = {}
    for idx, (ts, user) in enumerate(zip(timestamps, acting_users)):
        h_start = 1 + ((ts - 1) // hour_seconds) * hour_seconds
        key = (user, h_start)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    units: list[UserHourUnit] = []

    for (user, h_start), indices in sorted(groups.items()):
        h_end = h_start + hour_seconds
        lines = [int(source_lines[i]) for i in indices]
        n_events = len(indices)

        # Compute 16 aggregated features
        log1p_count = np.log1p(n_events)
        mean_auth = float(np.mean(f_auth_1h[indices]))
        max_auth = float(np.max(f_auth_1h[indices]))
        mean_fail = float(np.mean(f_fail_1h[indices]))
        max_fail = float(np.max(f_fail_1h[indices]))
        min_secs = float(np.min(f_secs_prev[indices]))
        mean_secs = float(np.mean(f_secs_prev[indices]))
        max_uniq = float(np.max(f_uniq_dest[indices]))
        max_dest = float(np.max(f_user_dest[indices]))
        rate_new_dest = float(np.mean(f_new_dest[indices]))
        rate_new_host = float(np.mean(f_new_host[indices]))
        rate_hist = float(np.mean(f_hist[indices]))

        # Mode of categorical IDs
        mode_auth = Counter(f_auth_type[indices]).most_common(1)[0][0]
        mode_logon = Counter(f_logon_type[indices]).most_common(1)[0][0]
        mode_orient = Counter(f_orient[indices]).most_common(1)[0][0]

        # Failure rate (auth_result_id != 1, where 1 is usually Success)
        fail_rate = float(np.mean(f_result[indices] != 1))

        vec = np.array(
            [
                log1p_count,
                mean_auth,
                max_auth,
                mean_fail,
                max_fail,
                min_secs,
                mean_secs,
                max_uniq,
                max_dest,
                rate_new_dest,
                rate_new_host,
                rate_hist,
                float(mode_auth),
                float(mode_logon),
                float(mode_orient),
                fail_rate,
            ],
            dtype=np.float32,
        )

        units.append(
            UserHourUnit(
                user_id=user,
                window_start=h_start,
                window_end=h_end,
                source_lines=lines,
                feature_vector=vec,
            )
        )

    return units


def build_user_hour_feature_matrix(units: Sequence[UserHourUnit]) -> np.ndarray:
    """Stack user-hour feature vectors into a 2D numpy matrix."""
    if not units:
        return np.empty((0, len(BASELINE_FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([u.feature_vector for u in units])
