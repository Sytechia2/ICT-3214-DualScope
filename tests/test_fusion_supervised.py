"""Tests for SupervisedFusionModel (Task 5.2 / 5.3 / 9.3)."""

import numpy as np
import pytest

from dualscope.fusion.supervised import (
    SUPERVISED_FEATURE_NAMES,
    SupervisedFusionModel,
    extract_fusion_features,
)


def test_extract_fusion_features() -> None:
    """Feature extraction extracts 4 features and handles None/NaN."""
    rows = [
        {"seq_score": 0.9, "graph_score": 0.8, "temporal_boost": 0.15, "lead_time_seconds": 3600},
        {"seq_score": None, "graph_score": 0.7, "temporal_boost": 0.0, "lead_time_seconds": None},
        {"seq_score": 0.5, "graph_score": None, "temporal_boost": 0.0, "lead_time_seconds": 0},
    ]
    X = extract_fusion_features(rows)
    assert X.shape == (3, 4)
    assert np.all(np.isfinite(X))
    assert X[0, 0] == 0.9
    assert X[0, 1] == 0.8
    assert X[0, 2] == 0.15
    assert X[0, 3] > 0.0  # log1p(3600)

    assert X[1, 0] == 0.0  # None seq imputed to 0
    assert X[1, 1] == 0.7  # graph score present

    assert X[2, 0] == 0.5
    assert X[2, 1] == 0.0  # None graph imputed to 0


def test_train_and_predict_supervised_fusion(tmp_path) -> None:
    """Supervised fusion model trains, predicts probabilities, and persists."""
    np.random.seed(42)
    N = 100
    X = np.random.rand(N, 4)
    # Target correlated with features
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)

    model = SupervisedFusionModel.train(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == N
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test save and load
    save_path = tmp_path / "supervised_fusion"
    model.save(save_path)
    loaded = SupervisedFusionModel.load(save_path)

    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(probs, loaded_probs)
