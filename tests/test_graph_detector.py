"""Tasks 4.1-4.4: causal snapshots, GAE, calibration and score evidence."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import torch

from dualscope.graph.config import GraphDetectorConfig, GraphPolicy
from dualscope.graph.export import GRAPH_SCORE_SCHEMA, build_score_table
from dualscope.graph.io import load_snapshot, save_snapshot
from dualscope.graph.model import edge_anomaly_scores, sample_bipartite_negatives
from dualscope.graph.scoring import aggregate_user_scores
from dualscope.graph.snapshot import build_snapshot
from dualscope.graph.training import load_checkpoint, save_checkpoint, train_autoencoder


def _events():
    return [
        {"timestamp": 90_000, "acting_user": "U1", "destination_computer": "C1", "authentication_result": "Success", "source_line": 1},
        {"timestamp": 90_100, "acting_user": "U1", "destination_computer": "C1", "authentication_result": "Fail", "source_line": 2},
        {"timestamp": 91_000, "acting_user": "U1", "destination_computer": "C2", "authentication_result": "Success", "source_line": 3},
        {"timestamp": 92_000, "acting_user": "U2", "destination_computer": "C2", "authentication_result": "Success", "source_line": 4},
        {"timestamp": 200_000, "acting_user": "FUTURE", "destination_computer": "C9", "authentication_result": "Success", "source_line": 5},
    ]


def _snapshot():
    return build_snapshot(_events(), 172_801, GraphPolicy(), seen_edges={("U1", "C1")}, prior_neighbours={"U1": {"C1"}})


def test_snapshot_has_expected_weight_boundaries_and_observed_changes():
    s=_snapshot(); assert s.users == ("U1","U2") and s.computers == ("C1","C2")
    assert s.n_edges == 3 and s.score_available_at == s.window_end == 172_801
    edges={(s.users[u],s.computers[c]):i for i,(u,c) in enumerate(zip(s.edge_users,s.edge_computers))}
    i=edges[("U1","C1")]; assert s.success_counts[i] == s.failure_counts[i] == 1
    assert s.edge_weights[i] == 1.25 and list(s.source_lines[i]) == [1,2] and not s.new_edges[i]
    assert s.new_edges[edges[("U1","C2")]] and "FUTURE" not in s.users
    u=s.users.index("U1"); assert s.prior_user_degrees[u] == 1 and s.current_user_degrees[u] == 2
    assert s.peak_user_auth_count_60s[u] == 1
    assert s.peak_user_auth_count_300s[u] == 2
    assert s.peak_user_unique_destinations_300s[u] == 1

def test_snapshot_round_trip(tmp_path):
    expected=_snapshot(); path=save_snapshot(expected,tmp_path/"snapshot.npz"); actual=load_snapshot(path)
    assert actual.users == expected.users and actual.source_lines == expected.source_lines
    np.testing.assert_array_equal(actual.edge_weights,expected.edge_weights)


def test_negative_sampling_stays_bipartite_and_excludes_observed_edges():
    s=_snapshot(); neg=sample_bipartite_negatives(s,20,np.random.default_rng(2)).numpy()
    observed={(int(u),int(c)+len(s.users)) for u,c in zip(s.edge_users,s.edge_computers)}
    assert all(0<=u<len(s.users)<=c<s.n_nodes for u,c in neg.T)
    assert not (set(map(tuple,neg.T)) & observed)


def test_training_reload_and_export_produce_finite_scores(tmp_path):
    s=_snapshot(); cfg=GraphDetectorConfig(); training=replace(cfg.training,max_epochs=4,early_stopping_patience=4)
    model,history=train_autoencoder([s],cfg.model,training); assert history and np.isfinite(history[-1]["train_loss"])
    path=tmp_path/"checkpoint.pt"; sha=save_checkpoint(path,model,{"purpose":"test"}); loaded,meta=load_checkpoint(path,sha)
    assert meta["purpose"] == "test"
    edge=edge_anomaly_scores(loaded,s); assert edge.shape == (s.n_edges,) and np.isfinite(edge).all()
    user=aggregate_user_scores(s,edge,"top3_mean"); normal=(user-user.min())/(np.ptp(user)+1e-9)
    table=build_score_table(s,edge,user,normal,model_version="test",run_id="test",split="validation",aggregation="top3_mean",alert_threshold=.5)
    assert table.schema.equals(GRAPH_SCORE_SCHEMA) and table.num_rows == len(s.users)
    row=table.to_pylist()[0]; assert row["score_available_at"] == s.window_end and row["top_edges"]
    assert row["new_edge_count"] >= 1 and row["degree_growth"] == 1


def test_aggregation_rules_limit_volume_effects():
    s=_snapshot(); edge=np.array([.1,.8,.7]);
    assert aggregate_user_scores(s,edge,"max")[0] == .8
    assert aggregate_user_scores(s,edge,"top3_mean")[0] == .45
