"""Causal rolling user-computer graph construction (Task 4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Iterable, Mapping

import numpy as np

from dualscope.graph.config import GraphPolicy


@dataclass(frozen=True)
class GraphSnapshot:
    window_start: int
    window_end: int
    score_available_at: int
    users: tuple[str, ...]
    computers: tuple[str, ...]
    edge_users: np.ndarray
    edge_computers: np.ndarray
    success_counts: np.ndarray
    failure_counts: np.ndarray
    edge_weights: np.ndarray
    source_lines: tuple[tuple[int, ...], ...]
    source_reference_counts: np.ndarray
    new_edges: np.ndarray
    prior_user_degrees: np.ndarray
    current_user_degrees: np.ndarray
    peak_user_auth_count_60s: np.ndarray
    peak_user_auth_count_300s: np.ndarray
    peak_user_unique_destinations_300s: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.users) + len(self.computers)

    @property
    def n_edges(self) -> int:
        return len(self.edge_users)

    def node_features(self, input_size: int = 9) -> np.ndarray:
        """Structural and burst features, usable for unseen node identities.

        The first six columns preserve checkpoint compatibility with the original
        model. Columns 7-9 describe exact rolling authentication bursts.
        """
        n_u, n_c = len(self.users), len(self.computers)
        degree = np.zeros(n_u + n_c, dtype=np.float32)
        strength = np.zeros_like(degree)
        failures = np.zeros_like(degree)
        for u, c, w, f in zip(self.edge_users, self.edge_computers, self.edge_weights, self.failure_counts):
            ci = n_u + int(c)
            degree[int(u)] += 1; degree[ci] += 1
            strength[int(u)] += w; strength[ci] += w
            failures[int(u)] += f; failures[ci] += f
        type_user = np.r_[np.ones(n_u), np.zeros(n_c)].astype(np.float32)
        type_computer = 1.0 - type_user
        new_degree = np.r_[self.current_user_degrees - self.prior_user_degrees, np.zeros(n_c)]
        features = np.column_stack([
            type_user, type_computer, np.log1p(degree), np.log1p(strength),
            np.log1p(failures), np.log1p(np.maximum(new_degree, 0)),
            np.r_[np.log1p(self.peak_user_auth_count_60s), np.zeros(n_c)],
            np.r_[np.log1p(self.peak_user_auth_count_300s), np.zeros(n_c)],
            np.r_[np.log1p(self.peak_user_unique_destinations_300s), np.zeros(n_c)],
        ]).astype(np.float32)
        if input_size not in (6, 9):
            raise ValueError(f"graph input_size must be 6 or 9, got {input_size}")
        return features[:, :input_size]


def build_snapshot(
    events: Iterable[Mapping[str, Any]], cutoff: int, policy: GraphPolicy | None = None,
    *, seen_edges: set[tuple[str, str]] | None = None,
    prior_neighbours: Mapping[str, set[str]] | None = None,
) -> GraphSnapshot:
    """Aggregate events in ``[cutoff-window, cutoff)`` without reading future data.

    ``seen_edges`` and ``prior_neighbours`` must contain only events before the
    window start. They are read-only inputs used for observed-change evidence.
    """
    policy = policy or GraphPolicy()
    start = cutoff - policy.window_seconds
    if cutoff <= start:
        raise ValueError("invalid snapshot cutoff")
    aggregates: dict[tuple[str, str], list[Any]] = {}
    # Exact rolling burst state. Input rows are required to be chronological.
    recent_60: dict[str, deque[int]] = {}
    recent_300: dict[str, deque[tuple[int, str]]] = {}
    destination_counts_300: dict[str, dict[str, int]] = {}
    peak_60: dict[str, int] = {}
    peak_300: dict[str, int] = {}
    peak_dest_300: dict[str, int] = {}
    last_timestamp = -1
    for row in events:
        t = int(row["timestamp"])
        if not start <= t < cutoff:
            continue
        if t < last_timestamp:
            raise ValueError("graph events must be ordered by non-decreasing timestamp")
        last_timestamp = t
        user, computer = str(row["acting_user"]), str(row["destination_computer"])
        q60 = recent_60.setdefault(user, deque())
        while q60 and q60[0] < t - 59: q60.popleft()
        q60.append(t); peak_60[user] = max(peak_60.get(user, 0), len(q60))
        q300 = recent_300.setdefault(user, deque()); counts = destination_counts_300.setdefault(user, {})
        while q300 and q300[0][0] < t - 299:
            _, old = q300.popleft(); counts[old] -= 1
            if counts[old] == 0: del counts[old]
        q300.append((t, computer)); counts[computer] = counts.get(computer, 0) + 1
        peak_300[user] = max(peak_300.get(user, 0), len(q300))
        peak_dest_300[user] = max(peak_dest_300.get(user, 0), len(counts))
        key = (user, computer)
        value = aggregates.setdefault(key, [0, 0, [], 0])
        if str(row.get("authentication_result", "Success")) == "Success": value[0] += 1
        else: value[1] += 1
        value[3] += 1
        if len(value[2]) < policy.evidence_references_per_edge:
            line = row.get("source_line")
            if line is None and row.get("source_reference"):
                line = int(str(row["source_reference"]).rsplit(":", 1)[1])
            if line is not None: value[2].append(int(line))
    users = tuple(sorted({u for u, _ in aggregates}))
    computers = tuple(sorted({c for _, c in aggregates}))
    uid, cid = {v: i for i, v in enumerate(users)}, {v: i for i, v in enumerate(computers)}
    keys = sorted(aggregates)
    success = np.asarray([aggregates[k][0] for k in keys], dtype=np.int64)
    failure = np.asarray([aggregates[k][1] for k in keys], dtype=np.int64)
    seen = seen_edges or set()
    prior = prior_neighbours or {}
    current: dict[str, set[str]] = {u: set() for u in users}
    for u, c in keys: current[u].add(c)
    return GraphSnapshot(
        window_start=start, window_end=cutoff, score_available_at=cutoff,
        users=users, computers=computers,
        edge_users=np.asarray([uid[u] for u, _ in keys], dtype=np.int64),
        edge_computers=np.asarray([cid[c] for _, c in keys], dtype=np.int64),
        success_counts=success, failure_counts=failure,
        edge_weights=success * policy.success_weight + failure * policy.failure_weight,
        source_lines=tuple(tuple(aggregates[k][2]) for k in keys),
        source_reference_counts=np.asarray([aggregates[k][3] for k in keys], dtype=np.int64),
        new_edges=np.asarray([k not in seen for k in keys], dtype=bool),
        prior_user_degrees=np.asarray([len(prior.get(u, set())) for u in users], dtype=np.int64),
        current_user_degrees=np.asarray([len(current[u]) for u in users], dtype=np.int64),
        peak_user_auth_count_60s=np.asarray([peak_60[u] for u in users], dtype=np.int64),
        peak_user_auth_count_300s=np.asarray([peak_300[u] for u in users], dtype=np.int64),
        peak_user_unique_destinations_300s=np.asarray([peak_dest_300[u] for u in users], dtype=np.int64),
    )
