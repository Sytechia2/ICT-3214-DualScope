"""Task 9.2: Aggregate per-event historical features to user-hour tabular vectors.

Extracts fixed-length 16-dimensional summary representations from the 12 Task 2.4
event features for standard tabular anomaly baselines (e.g. Isolation Forest).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa


# Encoded id of "Fail" in the frozen authentication_result vocabulary
# (0 = unseen, 1 = missing, 2 = Fail, 3 = Success). Read it from
# preprocessing.json when available; see ``fail_id_from_preprocessing``.
DEFAULT_FAIL_ID = 2

EVENT_FEATURE_COLUMNS: list[str] = [
    "timestamp",
    "acting_user",
    "log1p_prior_auth_count_1h_scaled",
    "log1p_prior_failure_count_1h_scaled",
    "log1p_seconds_since_previous_auth_scaled",
    "log1p_prior_unique_destinations_24h_scaled",
    "log1p_prior_user_destination_count_24h_scaled",
    "has_user_history",
    "is_new_user_destination",
    "is_new_host_connection",
    "auth_type_id",
    "logon_type_id",
    "auth_orientation_id",
    "auth_result_id",
]

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


def fail_id_from_preprocessing(preprocessing: dict[str, Any]) -> int:
    """Encoded id of the "Fail" authentication result in a frozen preprocessing artifact."""
    return int(preprocessing["categorical_vocabularies"]["authentication_result"]["category_to_id"]["Fail"])


def aggregate_user_hours_table(
    events_table: pa.Table,
    hour_seconds: int = 3600,
    fail_id: int = DEFAULT_FAIL_ID,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Vectorised user-hour aggregation: the same 16 features as ``aggregate_events_to_user_hours``.

    Returns the unit keys (``user_id``, ``window_start``, sorted by user then
    hour) and a float32 feature matrix in ``BASELINE_FEATURE_NAMES`` order.
    Every event of a user-hour must be in ``events_table`` (pass whole days),
    otherwise that hour's features are computed from a part of it. Category
    modes break ties by first occurrence, as ``Counter.most_common`` does.
    """
    keys_out = pd.DataFrame({"user_id": pd.Series([], dtype=object), "window_start": pd.Series([], dtype=np.int64)})
    if events_table.num_rows == 0:
        return keys_out, np.zeros((0, len(BASELINE_FEATURE_NAMES)), dtype=np.float32)

    ts = events_table["timestamp"].to_numpy()
    frame = pd.DataFrame({
        "user_id": events_table["acting_user"].to_pandas(),
        "window_start": 1 + ((ts - 1) // hour_seconds) * hour_seconds,
        "auth": events_table["log1p_prior_auth_count_1h_scaled"].to_numpy(),
        "fail": events_table["log1p_prior_failure_count_1h_scaled"].to_numpy(),
        "secs": events_table["log1p_seconds_since_previous_auth_scaled"].to_numpy(),
        "uniq": events_table["log1p_prior_unique_destinations_24h_scaled"].to_numpy(),
        "dest": events_table["log1p_prior_user_destination_count_24h_scaled"].to_numpy(),
        "hist": events_table["has_user_history"].to_numpy(),
        "new_dest": events_table["is_new_user_destination"].to_numpy(),
        "new_host": events_table["is_new_host_connection"].to_numpy(),
        "auth_type": events_table["auth_type_id"].to_numpy(),
        "logon_type": events_table["logon_type_id"].to_numpy(),
        "orient": events_table["auth_orientation_id"].to_numpy(),
        "failed": (events_table["auth_result_id"].to_numpy() == fail_id).astype(np.float64),
        "position": np.arange(events_table.num_rows),
    })
    keys = ["user_id", "window_start"]
    agg = frame.groupby(keys, sort=True).agg(
        n=("auth", "size"),
        mean_auth=("auth", "mean"), max_auth=("auth", "max"),
        mean_fail=("fail", "mean"), max_fail=("fail", "max"),
        min_secs=("secs", "min"), mean_secs=("secs", "mean"),
        max_uniq=("uniq", "max"), max_dest=("dest", "max"),
        rate_new_dest=("new_dest", "mean"), rate_new_host=("new_host", "mean"),
        rate_hist=("hist", "mean"), fail_rate=("failed", "mean"),
    )
    for column in ("auth_type", "logon_type", "orient"):
        counts = frame.groupby([*keys, column], sort=False).agg(count=("position", "size"), first=("position", "min"))
        counts = counts.reset_index().sort_values(["count", "first"], ascending=[False, True], kind="stable")
        agg[f"mode_{column}"] = counts.drop_duplicates(keys).set_index(keys)[column]

    matrix = np.column_stack([
        np.log1p(agg["n"].to_numpy(np.float64)),
        agg["mean_auth"], agg["max_auth"], agg["mean_fail"], agg["max_fail"],
        agg["min_secs"], agg["mean_secs"], agg["max_uniq"], agg["max_dest"],
        agg["rate_new_dest"], agg["rate_new_host"], agg["rate_hist"],
        agg["mode_auth_type"], agg["mode_logon_type"], agg["mode_orient"],
        agg["fail_rate"],
    ]).astype(np.float32)
    return agg.index.to_frame(index=False), matrix


def aggregate_events_to_user_hours(
    events_table: pa.Table,
    hour_seconds: int = 3600,
    fail_id: int = DEFAULT_FAIL_ID,
) -> list[UserHourUnit]:
    """Group transformed event rows by (acting_user, hour_start) and aggregate features.

    Keeps each unit's source lines, so it is meant for small tables (incident
    evidence, tests). For whole days use ``aggregate_user_hours_table``.
    """
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

        # Failure rate: share of events whose result is "Fail"
        fail_rate = float(np.mean(f_result[indices] == fail_id))

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
