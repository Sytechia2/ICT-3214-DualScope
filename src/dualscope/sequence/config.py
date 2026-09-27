"""Configuration for the short-term sequence detector (Tasks 3.1-3.4).

One JSON file (``config/sequence_detector.json``) holds the sequence policy,
model settings, training settings and the bounded validation search space.
Every run records the fingerprint of the exact settings it used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from dualscope.features.config import (
    DEFAULT_MODEL_INPUTS_BINARY,
    DEFAULT_MODEL_INPUTS_CATEGORICAL,
    DEFAULT_MODEL_INPUTS_NUMERIC,
)


DETECTOR_NAME = "sequence_gru_autoencoder"
SEQUENCE_POLICY_VERSION = "1.0.0"

# Per-event model inputs. The numeric and binary inputs are the transformed
# Task 2.4 columns; categorical inputs are Task 2.4 vocabulary IDs.
SEQUENCE_NUMERIC_INPUTS = tuple(DEFAULT_MODEL_INPUTS_NUMERIC)
SEQUENCE_BINARY_INPUTS = tuple(DEFAULT_MODEL_INPUTS_BINARY)
SEQUENCE_CATEGORICAL_INPUTS = tuple(DEFAULT_MODEL_INPUTS_CATEGORICAL)

# Source vocabulary for each categorical input in the Task 2.4 preprocessing artifact.
CATEGORICAL_VOCABULARY_COLUMNS = {
    "auth_type_id": "authentication_type",
    "logon_type_id": "logon_type",
    "auth_orientation_id": "authentication_orientation",
    "auth_result_id": "authentication_result",
}

AGGREGATIONS = ("max_chunk_mean", "max_event", "hour_mean")


@dataclass(frozen=True)
class SequencePolicy:
    """How events become user-hour sequences (Task 3.1)."""

    hour_seconds: int = 3_600
    max_sequence_length: int = 64
    min_events: int = 1
    require_complete_24h_history: bool = True

    def validate(self) -> None:
        if self.hour_seconds <= 0:
            raise ValueError("hour_seconds must be positive")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if self.min_events < 1:
            raise ValueError("min_events must be at least 1")


@dataclass(frozen=True)
class ModelSettings:
    """GRU autoencoder architecture and reconstruction objective (Task 3.2)."""

    hidden_size: int = 64
    latent_size: int = 32
    num_layers: int = 1
    embedding_dim: int = 4
    dropout: float = 0.0
    numeric_weight: float = 1.0
    binary_weight: float = 1.0
    categorical_weight: float = 1.0

    def validate(self) -> None:
        if min(self.hidden_size, self.latent_size, self.num_layers, self.embedding_dim) < 1:
            raise ValueError("model sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.numeric_weight, self.binary_weight, self.categorical_weight) < 0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class TrainingSettings:
    """Sampling and optimisation settings for fitting on eligible training user-hours."""

    seed: int = 42
    train_sequences: int = 300_000
    max_chunks_per_user_hour: int = 2
    monitor_sequences: int = 30_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    max_epochs: int = 8
    early_stopping_patience: int = 2
    gradient_clip_norm: float = 1.0
    torch_threads: int = 0

    def validate(self) -> None:
        if self.train_sequences < 1 or self.monitor_sequences < 1:
            raise ValueError("sample sizes must be positive")
        if self.max_chunks_per_user_hour < 1:
            raise ValueError("max_chunks_per_user_hour must be at least 1")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class SearchSpace:
    """Bounded validation search (Task 3.3)."""

    max_sequence_lengths: list[int] = field(default_factory=lambda: [32, 64, 128])
    model_sizes: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "small": {"hidden_size": 32, "latent_size": 16, "embedding_dim": 4},
            "medium": {"hidden_size": 64, "latent_size": 32, "embedding_dim": 8},
        }
    )
    aggregations: list[str] = field(default_factory=lambda: list(AGGREGATIONS))
    selection_metric: str = "average_precision"
    threshold_objective: str = "max_f1"

    def validate(self) -> None:
        if not self.max_sequence_lengths or not self.model_sizes or not self.aggregations:
            raise ValueError("search space must not be empty")
        unknown = set(self.aggregations) - set(AGGREGATIONS)
        if unknown:
            raise ValueError(f"unknown aggregations: {sorted(unknown)}")
        if self.selection_metric != "average_precision":
            raise ValueError("only average_precision model selection is implemented")
        if self.threshold_objective != "max_f1":
            raise ValueError("only max_f1 threshold selection is implemented")


@dataclass(frozen=True)
class SequenceDetectorConfig:
    """Complete sequence detector configuration."""

    policy: SequencePolicy = field(default_factory=SequencePolicy)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    search: SearchSpace = field(default_factory=SearchSpace)
    aggregation: str = "max_chunk_mean"
    top_events: int = 5
    top_features: int = 3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.policy.validate()
        self.model.validate()
        self.training.validate()
        self.search.validate()
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation: {self.aggregation}")
        if self.top_events < 1 or self.top_features < 1:
            raise ValueError("top_events and top_features must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_trial(self, max_sequence_length: int, size_name: str) -> SequenceDetectorConfig:
        """Return the configuration for one search-space trial."""
        size = self.search.model_sizes[size_name]
        return replace(
            self,
            policy=replace(self.policy, max_sequence_length=int(max_sequence_length)),
            model=replace(self.model, **{k: int(v) for k, v in size.items()}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceDetectorConfig:
        def build(kind: type, values: dict[str, Any] | None):
            values = dict(values or {})
            known = {f.name for f in fields(kind)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"unknown {kind.__name__} fields: {sorted(unknown)}")
            return kind(**values)

        known_top = {"policy", "model", "training", "search", "aggregation", "top_events", "top_features"}
        unknown_top = set(data) - known_top - {"description"}
        if unknown_top:
            raise ValueError(f"unknown sequence config fields: {sorted(unknown_top)}")
        return cls(
            policy=build(SequencePolicy, data.get("policy")),
            model=build(ModelSettings, data.get("model")),
            training=build(TrainingSettings, data.get("training")),
            search=build(SearchSpace, data.get("search")),
            aggregation=str(data.get("aggregation", "max_chunk_mean")),
            top_events=int(data.get("top_events", 5)),
            top_features=int(data.get("top_features", 3)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SequenceDetectorConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def input_feature_names() -> list[str]:
    """Per-event input names in the order used by model arrays and evidence."""
    return list(SEQUENCE_NUMERIC_INPUTS) + list(SEQUENCE_BINARY_INPUTS) + list(SEQUENCE_CATEGORICAL_INPUTS)
