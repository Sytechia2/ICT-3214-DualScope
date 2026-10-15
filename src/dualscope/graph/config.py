"""Configuration for the long-term graph detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GraphPolicy:
    window_seconds: int = 86_400
    stride_seconds: int = 86_400
    warmup_seconds: int = 86_400
    success_weight: float = 1.0
    failure_weight: float = 0.25
    evidence_references_per_edge: int = 100

    def __post_init__(self) -> None:
        if min(self.window_seconds, self.stride_seconds, self.warmup_seconds) <= 0:
            raise ValueError("graph window, stride and warm-up must be positive")
        if self.window_seconds <= 3_600:
            raise ValueError("graph window must be longer than the one-hour sequence window")
        if self.success_weight <= 0 or self.failure_weight < 0:
            raise ValueError("authentication weights must be non-negative and success must be positive")


@dataclass(frozen=True)
class GraphModelSettings:
    input_size: int = 9
    hidden_size: int = 32
    latent_size: int = 16
    dropout: float = 0.0


@dataclass(frozen=True)
class GraphTrainingSettings:
    seed: int = 42
    negative_ratio: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    max_positive_edges_per_snapshot: int = 200_000


@dataclass(frozen=True)
class GraphDetectorConfig:
    policy: GraphPolicy = field(default_factory=GraphPolicy)
    model: GraphModelSettings = field(default_factory=GraphModelSettings)
    training: GraphTrainingSettings = field(default_factory=GraphTrainingSettings)
    aggregation: str = "top3_mean"
    maximum_score_age_seconds: int = 86_400
    top_evidence_edges: int = 5

    @classmethod
    def from_file(cls, path: str | Path) -> "GraphDetectorConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            policy=GraphPolicy(**data.get("policy", {})),
            model=GraphModelSettings(**data.get("model", {})),
            training=GraphTrainingSettings(**data.get("training", {})),
            aggregation=str(data.get("aggregation", "top3_mean")),
            maximum_score_age_seconds=int(data.get("maximum_score_age_seconds", 86_400)),
            top_evidence_edges=int(data.get("top_evidence_edges", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()
