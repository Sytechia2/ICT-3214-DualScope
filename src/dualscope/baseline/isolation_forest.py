"""Task 9.2: Isolation Forest baseline model for user-hour tabular anomaly detection.

Implements fitting, score orientation (larger = more anomalous), quantile calibration,
and PyArrow dataset export matching the DualScope evaluation schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pyarrow as pa
from sklearn.ensemble import IsolationForest

from dualscope.baseline.features import UserHourUnit
from dualscope.sequence.calibration import QuantileTailCalibrator


BASELINE_SCORE_SCHEMA = pa.schema(
    [
        ("detector", pa.string()),
        ("model_version", pa.string()),
        ("user_id", pa.string()),
        ("window_start", pa.int64()),
        ("window_end", pa.int64()),
        ("score_available_at", pa.int64()),
        ("status", pa.string()),
        ("raw_score", pa.float64()),
        ("score", pa.float64()),
        ("alert_threshold", pa.float64()),
        ("is_alert", pa.bool_()),
        ("source_lines", pa.list_(pa.int64())),
    ]
)


@dataclass(frozen=True)
class BaselineScoreRow:
    """Individual scored user-hour evaluation row."""

    detector: str
    model_version: str
    user_id: str
    window_start: int
    window_end: int
    score_available_at: int
    status: str
    raw_score: float | None
    score: float | None
    alert_threshold: float
    is_alert: bool | None
    source_lines: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "score_available_at": self.score_available_at,
            "status": self.status,
            "raw_score": self.raw_score,
            "score": self.score,
            "alert_threshold": self.alert_threshold,
            "is_alert": self.is_alert,
            "source_lines": self.source_lines,
        }


class IsolationForestBaseline:
    """Isolation Forest anomaly detection baseline for tabular user-hour features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str | int = "auto",
        contamination: str | float = "auto",
        random_state: int = 42,
        model_version: str = "iforest-v1-n100",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.model_version = model_version

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.calibrator: QuantileTailCalibrator | None = None
        self.alert_threshold: float = 0.999

    def fit(self, X: np.ndarray) -> IsolationForestBaseline:
        """Fit the isolation forest on clean training feature vectors."""
        self.model.fit(X)
        return self

    def score_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute raw anomaly score. Invert decision function so higher = more anomalous."""
        # decision_function yields negative values for anomalies and positive for inliers.
        # We negate it so larger scores represent higher anomaly rarity.
        raw = -self.model.decision_function(X)
        return raw.astype(np.float64)

    def calibrate(self, val_raw_scores: np.ndarray, threshold: float | None = None) -> None:
        """Fit empirical quantile-tail calibrator on validation raw scores."""
        self.calibrator = QuantileTailCalibrator.fit(
            val_raw_scores,
            reference_description="Validation Days 8-16 Isolation Forest scores",
        )
        if threshold is not None:
            self.alert_threshold = float(threshold)

    def predict_units(self, units: Sequence[UserHourUnit]) -> list[BaselineScoreRow]:
        """Predict calibrated scores and alert decisions for user-hour units."""
        if not units:
            return []

        X = np.vstack([u.feature_vector for u in units])
        raw_scores = self.score_raw(X)

        if self.calibrator is not None:
            cal_scores = self.calibrator.transform(raw_scores)
        else:
            # Min-max fallback if uncalibrated
            r_min, r_max = raw_scores.min(), raw_scores.max()
            cal_scores = (raw_scores - r_min) / (r_max - r_min + 1e-9)

        rows: list[BaselineScoreRow] = []
        for unit, raw, score in zip(units, raw_scores, cal_scores):
            is_alert = bool(score >= self.alert_threshold) if np.isfinite(score) else None
            rows.append(
                BaselineScoreRow(
                    detector="isolation_forest_baseline",
                    model_version=self.model_version,
                    user_id=unit.user_id,
                    window_start=unit.window_start,
                    window_end=unit.window_end,
                    score_available_at=unit.window_end,
                    status="available",
                    raw_score=float(raw) if np.isfinite(raw) else None,
                    score=float(score) if np.isfinite(score) else None,
                    alert_threshold=self.alert_threshold,
                    is_alert=is_alert,
                    source_lines=unit.source_lines,
                )
            )
        return rows

    def to_arrow_table(self, rows: Sequence[BaselineScoreRow]) -> pa.Table:
        """Serialize baseline score rows to a PyArrow Table."""
        pylist = [r.to_dict() for r in rows]
        return pa.Table.from_pylist(pylist, schema=BASELINE_SCORE_SCHEMA)

    def save(self, output_dir: Path) -> None:
        """Save model checkpoint, parameters, and calibrator to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_dir / "model.joblib")

        meta = {
            "model_version": self.model_version,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "alert_threshold": self.alert_threshold,
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
        }
        (output_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> IsolationForestBaseline:
        """Load trained model checkpoint from directory."""
        meta = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        instance = cls(
            n_estimators=meta["n_estimators"],
            max_samples=meta["max_samples"],
            contamination=meta["contamination"],
            random_state=meta["random_state"],
            model_version=meta["model_version"],
        )
        instance.model = joblib.load(model_dir / "model.joblib")
        instance.alert_threshold = float(meta["alert_threshold"])
        if meta.get("calibrator"):
            instance.calibrator = QuantileTailCalibrator.from_dict(meta["calibrator"])
        return instance
