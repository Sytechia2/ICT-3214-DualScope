"""Supervised fusion model for multi-timescale anomaly detection (Task 5.2 / 5.3).

Combines short-term sequence scores, long-term graph scores, and engineered
temporal features (temporal co-occurrence boost and detector lead time) into a
calibrated compromise probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from dualscope.fusion.temporal import TemporalFusedScoreRow
from dualscope.sequence.calibration import QuantileTailCalibrator


SUPERVISED_FEATURE_NAMES = (
    "seq_score",
    "graph_score",
    "temporal_boost",
    "log_lead_time",
)


def extract_fusion_features(rows: Sequence[TemporalFusedScoreRow | dict[str, Any]]) -> np.ndarray:
    """Extract tabular feature matrix from temporal fused score rows.

    Parameters
    ----------
    rows : Sequence[TemporalFusedScoreRow | dict]
        Fused score rows containing detector scores and temporal context.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (N, 4):
        [seq_score, graph_score, temporal_boost, log_lead_time].
    """
    n = len(rows)
    X = np.zeros((n, 4), dtype=np.float64)

    for i, row in enumerate(rows):
        if isinstance(row, dict):
            s_seq = row.get("seq_score")
            s_graph = row.get("graph_score")
            boost = row.get("temporal_boost", 0.0)
            lead_time = row.get("lead_time_seconds")
        else:
            s_seq = row.seq_score
            s_graph = row.graph_score
            boost = row.temporal_boost
            lead_time = row.lead_time_seconds

        # Feature 0: seq_score
        X[i, 0] = float(s_seq) if s_seq is not None and np.isfinite(s_seq) else 0.0

        # Feature 1: graph_score
        X[i, 1] = float(s_graph) if s_graph is not None and np.isfinite(s_graph) else 0.0

        # Feature 2: temporal_boost
        X[i, 2] = float(boost) if boost is not None and np.isfinite(boost) else 0.0

        # Feature 3: log_lead_time
        lt = float(lead_time) if lead_time is not None and np.isfinite(lead_time) else 0.0
        X[i, 3] = np.log1p(max(0.0, lt))

    return X


@dataclass
class SupervisedFusionModel:
    """Logistic regression classifier over multi-timescale fused features."""

    model: LogisticRegression
    alert_threshold: float = 0.5
    calibrator: QuantileTailCalibrator | None = None
    feature_names: tuple[str, ...] = SUPERVISED_FEATURE_NAMES

    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        alert_threshold: float = 0.5,
        random_state: int = 42,
    ) -> SupervisedFusionModel:
        """Fit a balanced logistic regression model on multi-timescale features."""
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            random_state=random_state,
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        return cls(model=clf, alert_threshold=alert_threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous probability of attack."""
        probs = self.model.predict_proba(X)[:, 1]
        return probs.astype(np.float64)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated anomaly scores in [0, 1]."""
        raw_probs = self.predict_proba(X)
        if self.calibrator is not None:
            return self.calibrator.transform(raw_probs)
        return raw_probs

    def calibrate(self, X_val: np.ndarray, threshold: float | None = None) -> None:
        """Fit empirical quantile-tail calibrator on validation probabilities."""
        val_probs = self.predict_proba(X_val)
        self.calibrator = QuantileTailCalibrator.fit(
            val_probs,
            reference_description="Validation Supervised Fusion probabilities",
        )
        if threshold is not None:
            self.alert_threshold = float(threshold)

    def save(self, output_path: Path) -> None:
        """Save model checkpoint and metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path.with_suffix(".joblib"))
        meta = {
            "alert_threshold": self.alert_threshold,
            "feature_names": list(self.feature_names),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
        }
        output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path) -> SupervisedFusionModel:
        """Load trained model checkpoint."""
        joblib_path = model_path.with_suffix(".joblib")
        json_path = model_path.with_suffix(".json")
        model = joblib.load(joblib_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        calibrator = (
            QuantileTailCalibrator.from_dict(meta["calibrator"])
            if meta.get("calibrator")
            else None
        )
        return cls(
            model=model,
            alert_threshold=float(meta.get("alert_threshold", 0.5)),
            calibrator=calibrator,
            feature_names=tuple(meta.get("feature_names", SUPERVISED_FEATURE_NAMES)),
        )
