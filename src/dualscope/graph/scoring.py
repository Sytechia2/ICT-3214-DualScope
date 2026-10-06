"""Edge-to-user aggregation and structural evidence for graph snapshots."""

from __future__ import annotations

import numpy as np

from dualscope.graph.model import GraphAutoencoder, edge_anomaly_scores
from dualscope.graph.snapshot import GraphSnapshot


def aggregate_user_scores(snapshot: GraphSnapshot, edge_scores: np.ndarray, method: str) -> np.ndarray:
    out = np.full(len(snapshot.users), np.nan, dtype=np.float64)
    for user in range(len(snapshot.users)):
        values = edge_scores[snapshot.edge_users == user]
        if not len(values): continue
        if method == "max": out[user] = values.max()
        elif method.startswith("top") and method.endswith("_mean"):
            k = int(method[3:-5]); out[user] = np.sort(values)[-k:].mean()
        else: raise ValueError(f"unsupported graph aggregation: {method}")
    return out


def score_snapshot(model: GraphAutoencoder, snapshot: GraphSnapshot, aggregation: str) -> tuple[np.ndarray, np.ndarray]:
    edge = edge_anomaly_scores(model, snapshot)
    return edge, aggregate_user_scores(snapshot, edge, aggregation)
