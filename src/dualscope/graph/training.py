"""Training, checkpointing and reload for the graph autoencoder (Task 4.2)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from dualscope.graph.config import GraphModelSettings, GraphTrainingSettings
from dualscope.graph.model import GraphAutoencoder, normalized_adjacency, observed_pairs, sample_bipartite_negatives
from dualscope.graph.snapshot import GraphSnapshot


def train_autoencoder(snapshots: Sequence[GraphSnapshot], model_settings: GraphModelSettings | None = None,
                      settings: GraphTrainingSettings | None = None,
                      log: Callable[[dict[str, Any]], None] | None = None) -> tuple[GraphAutoencoder, list[dict[str, Any]]]:
    if not snapshots or not any(s.n_edges for s in snapshots): raise ValueError("no training graph edges")
    settings = settings or GraphTrainingSettings(); torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed); model = GraphAutoencoder(model_settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, settings.max_epochs + 1):
        started=time.time(); model.train(); total=0.0; batches=0
        for snapshot in snapshots:
            if not snapshot.n_edges: continue
            pos = observed_pairs(snapshot)
            if pos.shape[1] > settings.max_positive_edges_per_snapshot:
                keep=rng.choice(pos.shape[1], settings.max_positive_edges_per_snapshot, replace=False); pos=pos[:, keep]
            neg=sample_bipartite_negatives(snapshot, int(pos.shape[1]*settings.negative_ratio), rng)
            z=model.encode(torch.from_numpy(snapshot.node_features(model.settings.input_size)), normalized_adjacency(snapshot))
            logits=torch.cat([model.decode(z,pos), model.decode(z,neg)])
            labels=torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item(); batches += 1
        mean=total/max(batches,1); record={"epoch":epoch,"train_loss":mean,"epoch_seconds":round(time.time()-started,3)}
        history.append(record)
        if log is not None: log(record)
        if mean < best-1e-6: best=mean; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= settings.early_stopping_patience: break
    assert best_state is not None; model.load_state_dict(best_state)
    for h in history: h["selected"] = h["train_loss"] == best
    return model, history


def save_checkpoint(path: str | Path, model: GraphAutoencoder, metadata: dict[str, Any]) -> str:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"format_version":1,"model_settings":vars(model.settings),"model_state":model.state_dict(),"metadata":metadata},path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: str | Path, expected_sha256: str | None=None) -> tuple[GraphAutoencoder,dict[str,Any]]:
    path=Path(path); actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual != expected_sha256: raise ValueError("graph checkpoint hash mismatch")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    model=GraphAutoencoder(GraphModelSettings(**payload["model_settings"])); model.load_state_dict(payload["model_state"]); model.eval()
    return model, payload.get("metadata",{})
