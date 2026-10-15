"""Feature-based graph convolutional autoencoder for bipartite link reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import torch
from torch import nn

from dualscope.graph.config import GraphModelSettings
from dualscope.graph.snapshot import GraphSnapshot


class GraphAutoencoder(nn.Module):
    """Two-layer GCN encoder and dot-product decoder, without node-ID embeddings."""
    def __init__(self, settings: GraphModelSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or GraphModelSettings()
        self.input = nn.Linear(self.settings.input_size, self.settings.hidden_size)
        self.output = nn.Linear(self.settings.hidden_size, self.settings.latent_size)
        self.dropout = nn.Dropout(self.settings.dropout)

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input(torch.sparse.mm(adjacency, features)))
        hidden = self.dropout(hidden)
        return self.output(torch.sparse.mm(adjacency, hidden))

    @staticmethod
    def decode(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        return (z[pairs[0]] * z[pairs[1]]).sum(dim=1) / math.sqrt(z.shape[1])


def normalized_adjacency(snapshot: GraphSnapshot) -> torch.Tensor:
    n_u, n = len(snapshot.users), snapshot.n_nodes
    u = snapshot.edge_users
    c = snapshot.edge_computers + n_u
    rows = np.r_[u, c, np.arange(n)]
    cols = np.r_[c, u, np.arange(n)]
    values = np.r_[snapshot.edge_weights, snapshot.edge_weights, np.ones(n)].astype(np.float32)
    degree = np.bincount(rows, weights=values, minlength=n)
    values *= (degree[rows] * degree[cols]) ** -0.5
    indices = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(values), (n, n), check_invariants=False).coalesce()


def observed_pairs(snapshot: GraphSnapshot) -> torch.Tensor:
    return torch.from_numpy(np.vstack([snapshot.edge_users, snapshot.edge_computers + len(snapshot.users)]))


def sample_bipartite_negatives(snapshot: GraphSnapshot, count: int, rng: np.random.Generator) -> torch.Tensor:
    """Sample only valid user-computer non-edges; never self/type-invalid edges."""
    if not snapshot.users or not snapshot.computers or count <= 0:
        return torch.empty((2, 0), dtype=torch.int64)
    existing = set(zip(snapshot.edge_users.tolist(), snapshot.edge_computers.tolist()))
    capacity = len(snapshot.users) * len(snapshot.computers) - len(existing)
    target = min(count, capacity)
    result: set[tuple[int, int]] = set()
    while len(result) < target:
        batch = max(64, 2 * (target - len(result)))
        us = rng.integers(0, len(snapshot.users), size=batch)
        cs = rng.integers(0, len(snapshot.computers), size=batch)
        result.update((int(u), int(c)) for u, c in zip(us, cs) if (int(u), int(c)) not in existing)
    pairs = sorted(result)[:target]
    return torch.tensor([[u for u, _ in pairs], [len(snapshot.users) + c for _, c in pairs]], dtype=torch.int64)


@torch.no_grad()
def edge_anomaly_scores(model: GraphAutoencoder, snapshot: GraphSnapshot) -> np.ndarray:
    model.eval()
    if snapshot.n_edges == 0: return np.empty(0, dtype=np.float64)
    x = torch.from_numpy(snapshot.node_features(model.settings.input_size))
    z = model.encode(x, normalized_adjacency(snapshot))
    return torch.sigmoid(-model.decode(z, observed_pairs(snapshot))).cpu().numpy().astype(np.float64)
