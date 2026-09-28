"""Frozen sequence detector: checkpoint + scoring settings + calibration (Tasks 3.3-3.4).

``freeze_detector`` writes one ``frozen_detector.json`` next to a copy of the
selected checkpoint. The record carries hashes of the checkpoint and of its
own content, and the feature/split fingerprints it was fitted against.
``FrozenSequenceDetector.load`` refuses to score if any of them changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dualscope.sequence.builder import DaySequences
from dualscope.sequence.calibration import QuantileTailCalibrator
from dualscope.sequence.config import DETECTOR_NAME, SequenceDetectorConfig
from dualscope.sequence.model import GRUSequenceAutoencoder
from dualscope.sequence.scoring import DayScores, score_day
from dualscope.sequence.training import file_sha256, load_checkpoint


FROZEN_FILENAME = "frozen_detector.json"
CHECKPOINT_FILENAME = "checkpoint.pt"


def _content_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def model_version_for(config: SequenceDetectorConfig, checkpoint_sha256: str) -> str:
    return f"seq-gru-ae-v1-L{config.policy.max_sequence_length}-h{config.model.hidden_size}-{checkpoint_sha256[:10]}"


def freeze_detector(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    config: SequenceDetectorConfig,
    calibrator: QuantileTailCalibrator,
    alert_threshold: float,
    run_id: str,
    feature_info: dict[str, Any],
    split_policy_fingerprint: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Copy the selected checkpoint and write the immutable scoring record."""
    output_dir = Path(output_dir)
    if (output_dir / FROZEN_FILENAME).exists():
        raise FileExistsError(f"a frozen detector already exists in {output_dir}; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / CHECKPOINT_FILENAME
    shutil.copyfile(checkpoint_path, target)
    checkpoint_sha = file_sha256(target)
    record: dict[str, Any] = {
        "detector": DETECTOR_NAME,
        "model_version": model_version_for(config, checkpoint_sha),
        "run_id": run_id,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {"path": CHECKPOINT_FILENAME, "sha256": checkpoint_sha},
        "sequence_config": config.to_dict(),
        "sequence_config_fingerprint": config.fingerprint(),
        "aggregation": config.aggregation,
        "max_sequence_length": config.policy.max_sequence_length,
        "calibration": calibrator.to_dict(),
        "alert_threshold": float(alert_threshold),
        "alert_rule": "is_alert = score >= alert_threshold (score in [0, 1))",
        "feature": feature_info,
        "split_policy_fingerprint": split_policy_fingerprint,
        "selection": selection,
        "test_data_used": False,
    }
    record["content_sha256"] = _content_hash(record)
    (output_dir / FROZEN_FILENAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


@dataclass
class FrozenSequenceDetector:
    """A loaded, verified detector ready for inference."""

    model: GRUSequenceAutoencoder
    config: SequenceDetectorConfig
    calibrator: QuantileTailCalibrator
    alert_threshold: float
    record: dict[str, Any]

    @property
    def model_version(self) -> str:
        return str(self.record["model_version"])

    @property
    def run_id(self) -> str:
        return str(self.record["run_id"])

    @property
    def aggregation(self) -> str:
        return self.config.aggregation

    @classmethod
    def load(
        cls,
        frozen_dir: str | Path,
        feature_config_fingerprint: str | None = None,
        split_policy_fingerprint: str | None = None,
    ) -> FrozenSequenceDetector:
        frozen_dir = Path(frozen_dir)
        record = json.loads((frozen_dir / FROZEN_FILENAME).read_text(encoding="utf-8"))
        if record.get("content_sha256") != _content_hash(record):
            raise ValueError("frozen detector record was modified after freezing")
        if feature_config_fingerprint is not None and record["feature"].get("feature_config_fingerprint") != feature_config_fingerprint:
            raise ValueError("feature configuration differs from the one the detector was frozen with")
        if split_policy_fingerprint is not None and record["split_policy_fingerprint"] != split_policy_fingerprint:
            raise ValueError("split policy differs from the one the detector was frozen with")
        model, _ = load_checkpoint(frozen_dir / record["checkpoint"]["path"], record["checkpoint"]["sha256"])
        config = SequenceDetectorConfig.from_dict(record["sequence_config"])
        if config.fingerprint() != record["sequence_config_fingerprint"]:
            raise ValueError("sequence configuration fingerprint mismatch")
        return cls(
            model=model,
            config=config,
            calibrator=QuantileTailCalibrator.from_dict(record["calibration"]),
            alert_threshold=float(record["alert_threshold"]),
            record=record,
        )

    def score_day(self, day: DaySequences, batch_size: int = 2048) -> DayScores:
        return score_day(self.model, day, self.config.policy.max_sequence_length, batch_size)

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(raw)
