"""Streaming causal feature engine for LANL authentication events.

Maintains exact rolling 1h and 24h history, cumulative relationship novelty,
and strict non-decreasing (timestamp, source_line) event processing.
Features are calculated strictly BEFORE state is updated with the current event.
"""

from __future__ import annotations

import collections
import sys
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import pyarrow as pa

from dualscope.features.config import FeatureConfig
from dualscope.features.schemas import RAW_FEATURE_SCHEMA


try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class UserRollingState:
    """Memory-efficient rolling history state for one active user."""

    history_1h: collections.deque[tuple[int, bool]]
    failure_count_1h: int
    history_24h: collections.deque[tuple[int, int]]
    dest_counts_24h: dict[int, int]

    @classmethod
    def empty(cls) -> UserRollingState:
        return cls(
            history_1h=collections.deque(),
            failure_count_1h=0,
            history_24h=collections.deque(),
            dest_counts_24h={},
        )

    def is_empty(self) -> bool:
        return len(self.history_1h) == 0 and len(self.history_24h) == 0


class HistoricalFeatureEngine:
    """Stateful streaming engine for computing causal historical behavioral features.

    Maintains:
    - User 1h sliding window (prior auth count, prior failure count)
    - User 24h sliding window (prior unique destinations, prior user-destination count)
    - User previous authentication timestamp (seconds since previous auth, has history)
    - Ever-seen sets of (user, destination), (user, source_host) and (source_host, destination_host)
    - Active timeline sweeper to purge memory of inactive users
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        sweep_interval_seconds: int = 60,
    ) -> None:
        self.config = config or FeatureConfig.default()
        self.w1 = self.config.lookback_1h_seconds
        self.w24 = self.config.lookback_24h_seconds
        self.ds_start = self.config.dataset_start_timestamp
        self._sweep_interval_seconds = sweep_interval_seconds

        # Entity string intern tables for compact integer representations
        self._user_to_id: dict[str, int] = {}
        self._comp_to_id: dict[str, int] = {}

        # Ever-seen relationship sets (packed 64-bit integers: (id1 << 32) | id2)
        self.seen_user_destinations: set[int] = set()
        self.seen_host_connections: set[int] = set()
        self.seen_user_sources: set[int] = set()
        # Per-user flag: account name (before '@') ends with '$' (a machine account)
        self._user_is_machine: dict[int, bool] = {}

        # Last observed timestamp per user (stored permanently for gap calculation)
        self.user_last_auth: dict[int, int] = {}

        # Rolling active history per user
        self.user_rolling: dict[int, UserRollingState] = {}

        # Activity timeline queues for global inactive user purging: (timestamp, user_id)
        self._timeline_1h: collections.deque[tuple[int, int]] = collections.deque()
        self._timeline_24h: collections.deque[tuple[int, int]] = collections.deque()
        self._user_last_timeline_ts: dict[int, int] = {}

        # State ordering guards and periodic sweeper tracking
        self._last_timestamp: int = -1
        self._last_source_line: int = -1
        self._last_sweep_timestamp: int = -1
        self._total_events_processed: int = 0

    def _get_user_id(self, user: str) -> int:
        uid = self._user_to_id.get(user)
        if uid is None:
            uid = len(self._user_to_id)
            self._user_to_id[user] = uid
        return uid

    def _get_comp_id(self, comp: str) -> int:
        cid = self._comp_to_id.get(comp)
        if cid is None:
            cid = len(self._comp_to_id)
            self._comp_to_id[comp] = cid
        return cid

    def purge_inactive_users(self, current_timestamp: int | None = None) -> None:
        """Purge rolling history of users that have been inactive past window boundaries."""
        t = current_timestamp if current_timestamp is not None else self._last_timestamp
        if t >= 0:
            self._purge_inactive_users(t)
            self._last_sweep_timestamp = t

    def _purge_inactive_users(self, current_timestamp: int) -> None:
        """Purge rolling history of users that have been inactive past window boundaries."""
        # 1. Purge 1h history for users inactive > w1
        t_expire_1h = current_timestamp - self.w1
        while self._timeline_1h and self._timeline_1h[0][0] < t_expire_1h:
            ts, uid = self._timeline_1h.popleft()
            if self.user_last_auth.get(uid, 0) < t_expire_1h:
                state = self.user_rolling.get(uid)
                if state is not None:
                    state.history_1h.clear()
                    state.failure_count_1h = 0
                    if state.is_empty():
                        del self.user_rolling[uid]

        # 2. Purge 24h history for users inactive > w24
        t_expire_24h = current_timestamp - self.w24
        while self._timeline_24h and self._timeline_24h[0][0] < t_expire_24h:
            ts, uid = self._timeline_24h.popleft()
            if self.user_last_auth.get(uid, 0) < t_expire_24h:
                state = self.user_rolling.get(uid)
                if state is not None:
                    state.history_24h.clear()
                    state.dest_counts_24h.clear()
                    if state.is_empty():
                        del self.user_rolling[uid]
                if uid in self._user_last_timeline_ts and self._user_last_timeline_ts[uid] < t_expire_24h:
                    del self._user_last_timeline_ts[uid]

    def process_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Process a batch of input events, compute raw features, and update internal state."""
        n_rows = len(batch)
        if n_rows == 0:
            return pa.RecordBatch.from_pylist([], schema=RAW_FEATURE_SCHEMA)

        # Extract required columns from incoming batch
        ts_arr = batch["timestamp"]
        line_arr = batch["source_line"]
        act_user_arr = batch["acting_user"]
        src_comp_arr = batch["source_computer"]
        dst_comp_arr = batch["destination_computer"]
        result_arr = batch["authentication_result"]

        # Python lists for fast sequential iteration
        ts_list = ts_arr.to_pylist()
        line_list = line_arr.to_pylist()
        act_user_list = act_user_arr.to_pylist()
        src_comp_list = src_comp_arr.to_pylist()
        dst_comp_list = dst_comp_arr.to_pylist()
        result_list = result_arr.to_pylist()

        # Output feature lists
        out_prior_auth_count_1h = [0] * n_rows
        out_prior_failure_count_1h = [0] * n_rows
        out_seconds_since_prev = [None] * n_rows
        out_has_user_history = [False] * n_rows
        out_prior_unique_dest_24h = [0] * n_rows
        out_prior_user_dest_count_24h = [0] * n_rows
        out_is_new_user_dest = [False] * n_rows
        out_is_new_host_conn = [False] * n_rows
        out_is_new_user_src = [False] * n_rows
        out_is_machine = [False] * n_rows
        out_history_complete_1h = [False] * n_rows
        out_history_complete_24h = [False] * n_rows

        w1 = self.w1
        w24 = self.w24
        ds_start = self.ds_start

        # Process each event sequentially in deterministic order
        for i in range(n_rows):
            t = ts_list[i]
            line = line_list[i]
            user = act_user_list[i]
            src_c = src_comp_list[i]
            dst_c = dst_comp_list[i]
            res = result_list[i]

            # 0. Strict non-decreasing ordering verification
            if t < self._last_timestamp or (t == self._last_timestamp and line < self._last_source_line):
                raise ValueError(
                    f"Deterministic ordering violation at event index {i} (line {line}): "
                    f"(timestamp={t}, source_line={line}) < "
                    f"previous (timestamp={self._last_timestamp}, source_line={self._last_source_line})"
                )

            # Periodic inactive user sweeper (runs when elapsed dataset time since last sweep >= sweep_interval)
            if self._last_sweep_timestamp < 0 or (t - self._last_sweep_timestamp >= self._sweep_interval_seconds):
                self._purge_inactive_users(t)
                self._last_sweep_timestamp = t

            self._last_timestamp = t
            self._last_source_line = line
            self._total_events_processed += 1

            uid = self._get_user_id(user)
            sc_id = self._get_comp_id(src_c)
            dc_id = self._get_comp_id(dst_c)

            # Get user state
            ustate = self.user_rolling.get(uid)
            if ustate is None:
                ustate = UserRollingState.empty()
                self.user_rolling[uid] = ustate

            # -------------------------------------------------------------
            # Step 1: Expire records outside sliding lookbacks
            # -------------------------------------------------------------
            # Expiry condition: timestamp < t - W
            t_min_1h = t - w1
            h1 = ustate.history_1h
            while h1 and h1[0][0] < t_min_1h:
                _, expired_fail = h1.popleft()
                if expired_fail:
                    ustate.failure_count_1h -= 1

            t_min_24h = t - w24
            h24 = ustate.history_24h
            dcounts = ustate.dest_counts_24h
            while h24 and h24[0][0] < t_min_24h:
                _, expired_dc = h24.popleft()
                dcounts[expired_dc] -= 1
                if dcounts[expired_dc] == 0:
                    del dcounts[expired_dc]

            # -------------------------------------------------------------
            # Step 2: Compute features from prior state
            # -------------------------------------------------------------
            out_prior_auth_count_1h[i] = len(h1)
            out_prior_failure_count_1h[i] = ustate.failure_count_1h

            last_t = self.user_last_auth.get(uid)
            if last_t is not None:
                out_seconds_since_prev[i] = float(t - last_t)
                out_has_user_history[i] = True
            else:
                out_seconds_since_prev[i] = None
                out_has_user_history[i] = False

            out_prior_unique_dest_24h[i] = len(dcounts)
            out_prior_user_dest_count_24h[i] = dcounts.get(dc_id, 0)

            # Relationship novelty (packed 64-bit int keys)
            user_dest_key = (uid << 32) | dc_id
            host_conn_key = (sc_id << 32) | dc_id
            user_src_key = (uid << 32) | sc_id

            out_is_new_user_dest[i] = (user_dest_key not in self.seen_user_destinations)
            out_is_new_host_conn[i] = (host_conn_key not in self.seen_host_connections)
            out_is_new_user_src[i] = (user_src_key not in self.seen_user_sources)

            is_machine = self._user_is_machine.get(uid)
            if is_machine is None:
                is_machine = user.split("@", 1)[0].endswith("$")
                self._user_is_machine[uid] = is_machine
            out_is_machine[i] = is_machine

            out_history_complete_1h[i] = (t >= ds_start + w1)
            out_history_complete_24h[i] = (t >= ds_start + w24)

            # -------------------------------------------------------------
            # Step 4: Update state with current event
            # -------------------------------------------------------------
            is_fail = (res == "Fail")
            h1.append((t, is_fail))
            if is_fail:
                ustate.failure_count_1h += 1

            h24.append((t, dc_id))
            dcounts[dc_id] = dcounts.get(dc_id, 0) + 1

            self.user_last_auth[uid] = t
            self.seen_user_destinations.add(user_dest_key)
            self.seen_host_connections.add(host_conn_key)
            self.seen_user_sources.add(user_src_key)

            if self._user_last_timeline_ts.get(uid) != t:
                self._user_last_timeline_ts[uid] = t
                self._timeline_1h.append((t, uid))
                self._timeline_24h.append((t, uid))

        # Assemble PyArrow RecordBatch conforming strictly to RAW_FEATURE_SCHEMA
        raw_cols: dict[str, pa.Array] = {
            "timestamp": ts_arr,
            "source_user": batch["source_user"],
            "destination_user": batch["destination_user"],
            "source_computer": batch["source_computer"],
            "destination_computer": batch["destination_computer"],
            "authentication_type": batch["authentication_type"],
            "logon_type": batch["logon_type"],
            "authentication_orientation": batch["authentication_orientation"],
            "authentication_result": batch["authentication_result"],
            "acting_user": act_user_arr,
            "exact_duplicate_ordinal": batch["exact_duplicate_ordinal"],
            "source_line": line_arr,
            "source_reference": batch["source_reference"],
            "prior_auth_count_1h": pa.array(out_prior_auth_count_1h, type=pa.int64()),
            "prior_failure_count_1h": pa.array(out_prior_failure_count_1h, type=pa.int64()),
            "seconds_since_previous_auth": pa.array(out_seconds_since_prev, type=pa.float64()),
            "has_user_history": pa.array(out_has_user_history, type=pa.bool_()),
            "prior_unique_destinations_24h": pa.array(out_prior_unique_dest_24h, type=pa.int64()),
            "prior_user_destination_count_24h": pa.array(out_prior_user_dest_count_24h, type=pa.int64()),
            "is_new_user_destination": pa.array(out_is_new_user_dest, type=pa.bool_()),
            "is_new_host_connection": pa.array(out_is_new_host_conn, type=pa.bool_()),
            "is_new_user_source": pa.array(out_is_new_user_src, type=pa.bool_()),
            "is_machine_account": pa.array(out_is_machine, type=pa.bool_()),
            "history_complete_1h": pa.array(out_history_complete_1h, type=pa.bool_()),
            "history_complete_24h": pa.array(out_history_complete_24h, type=pa.bool_()),
            "dataset_day": batch["dataset_day"],
        }

        # Build in schema order
        arrays = [raw_cols[field.name] for field in RAW_FEATURE_SCHEMA]
        return pa.RecordBatch.from_arrays(arrays, schema=RAW_FEATURE_SCHEMA)

    def get_memory_breakdown(self) -> dict[str, Any]:
        """Return explicit memory instrumentation for rolling history, timelines, and relationship sets."""
        # 1. Rolling history structures
        rolling_users_count = len(self.user_rolling)
        rolling_mem = sys.getsizeof(self.user_rolling)
        rolling_1h_items = 0
        rolling_24h_items = 0

        for state in self.user_rolling.values():
            rolling_mem += sys.getsizeof(state.history_1h) + sys.getsizeof(state.history_24h)
            rolling_mem += sys.getsizeof(state.dest_counts_24h)
            rolling_1h_items += len(state.history_1h)
            rolling_24h_items += len(state.history_24h)

        timeline_mem = (
            sys.getsizeof(self._timeline_1h)
            + sys.getsizeof(self._timeline_24h)
            + sys.getsizeof(self._user_last_timeline_ts)
        )
        rolling_mem += timeline_mem

        # 2. Ever-seen relationship sets
        seen_user_dst_mem = sys.getsizeof(self.seen_user_destinations)
        seen_host_conn_mem = sys.getsizeof(self.seen_host_connections)
        seen_user_src_mem = sys.getsizeof(self.seen_user_sources)
        ever_seen_mem = seen_user_dst_mem + seen_host_conn_mem + seen_user_src_mem

        # 3. Entity tables and user last auth
        entity_mem = sys.getsizeof(self._user_to_id) + sys.getsizeof(self._comp_to_id)
        entity_mem += sys.getsizeof(self.user_last_auth) + sys.getsizeof(self._user_is_machine)

        total_internal_bytes = rolling_mem + ever_seen_mem + entity_mem

        rss_mb = None
        if psutil is not None:
            try:
                rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        return {
            "total_events_processed": self._total_events_processed,
            "distinct_users_tracked": len(self._user_to_id),
            "distinct_computers_tracked": len(self._comp_to_id),
            "active_rolling_users": rolling_users_count,
            "rolling_1h_events": rolling_1h_items,
            "rolling_24h_events": rolling_24h_items,
            "timeline_1h_entries": len(self._timeline_1h),
            "timeline_24h_entries": len(self._timeline_24h),
            "timeline_memory_mb": round(timeline_mem / (1024 * 1024), 3),
            "rolling_history_memory_mb": round(rolling_mem / (1024 * 1024), 3),
            "distinct_user_destinations_ever_seen": len(self.seen_user_destinations),
            "distinct_host_connections_ever_seen": len(self.seen_host_connections),
            "distinct_user_sources_ever_seen": len(self.seen_user_sources),
            "ever_seen_sets_memory_mb": round(ever_seen_mem / (1024 * 1024), 3),
            "entity_tables_memory_mb": round(entity_mem / (1024 * 1024), 3),
            "total_internal_state_memory_mb": round(total_internal_bytes / (1024 * 1024), 3),
            "process_rss_mb": round(rss_mb, 2) if rss_mb is not None else None,
        }
