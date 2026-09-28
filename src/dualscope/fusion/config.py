"""Configuration definitions and enums for output alignment and fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class FusionMethod(str, Enum):
    """Supported fusion algorithms."""

    MAX = "max"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    TEMPORAL = "temporal"


class AlignmentStatus(str, Enum):
    """Alignment status for a joint scoring unit."""

    BOTH_AVAILABLE = "both_available"
    SEQUENCE_ONLY = "sequence_only"
    GRAPH_ONLY = "graph_only"
    STALE_GRAPH = "stale_graph"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NO_ACTIVITY = "no_activity"


class DisagreementType(str, Enum):
    """Categorisation of consensus/disagreement between detectors."""

    CONCORDANT_ALERT = "concordant_alert"
    SEQ_ONLY_ALERT = "seq_only_alert"
    GRAPH_ONLY_ALERT = "graph_only_alert"
    CONCORDANT_NORMAL = "concordant_normal"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AlignmentConfig:
    """Settings governing temporal alignment across sequence and graph outputs."""

    hour_seconds: int = 3600
    hour_origin: int = 1
    max_graph_staleness_seconds: int = 86400  # 24 hours
    warmup_seconds: int = 86400  # Day 1 warm-up

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlignmentConfig:
        return cls(
            hour_seconds=int(data.get("hour_seconds", 3600)),
            hour_origin=int(data.get("hour_origin", 1)),
            max_graph_staleness_seconds=int(data.get("max_graph_staleness_seconds", 86400)),
            warmup_seconds=int(data.get("warmup_seconds", 86400)),
        )


@dataclass(frozen=True)
class BasicFusionConfig:
    """Configuration for Task 5.2 basic fusion methods."""

    method: FusionMethod = FusionMethod.WEIGHTED
    sequence_weight: float = 0.5
    renormalize_single_detector: bool = True
    fused_alert_threshold: float = 0.999

    def __post_init__(self) -> None:
        if not (0.0 <= self.sequence_weight <= 1.0):
            raise ValueError(f"sequence_weight must be between 0.0 and 1.0, got {self.sequence_weight}")
        if not (0.0 <= self.fused_alert_threshold <= 1.0):
            raise ValueError(
                f"fused_alert_threshold must be between 0.0 and 1.0, got {self.fused_alert_threshold}"
            )

    @property
    def graph_weight(self) -> float:
        return 1.0 - self.sequence_weight

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["method"] = self.method.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BasicFusionConfig:
        method_raw = data.get("method", FusionMethod.WEIGHTED.value)
        return cls(
            method=FusionMethod(method_raw),
            sequence_weight=float(data.get("sequence_weight", 0.5)),
            renormalize_single_detector=bool(data.get("renormalize_single_detector", True)),
            fused_alert_threshold=float(data.get("fused_alert_threshold", 0.999)),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> BasicFusionConfig:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class TemporalConfig:
    """Settings governing temporal proximity fusion (Task 5.3)."""

    lookback_seconds: int = 86400  # 24-hour maximum lookback window W_max
    half_life_seconds: float = 21600.0  # 6-hour exponential decay parameter tau
    boost_weight: float = 0.15  # Gain multiplier gamma
    max_fused_score_cap: float = 0.9999  # Cap to prevent saturation at 1.0
    fused_alert_threshold: float = 0.999  # Threshold to trigger fused alert

    def __post_init__(self) -> None:
        if self.lookback_seconds <= 0:
            raise ValueError(f"lookback_seconds must be positive, got {self.lookback_seconds}")
        if self.half_life_seconds <= 0:
            raise ValueError(f"half_life_seconds must be positive, got {self.half_life_seconds}")
        if not (0.0 <= self.boost_weight <= 1.0):
            raise ValueError(f"boost_weight must be between 0.0 and 1.0, got {self.boost_weight}")
        if not (0.0 < self.max_fused_score_cap <= 1.0):
            raise ValueError(f"max_fused_score_cap must be in (0, 1], got {self.max_fused_score_cap}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemporalConfig:
        return cls(
            lookback_seconds=int(data.get("lookback_seconds", 86400)),
            half_life_seconds=float(data.get("half_life_seconds", 21600.0)),
            boost_weight=float(data.get("boost_weight", 0.15)),
            max_fused_score_cap=float(data.get("max_fused_score_cap", 0.9999)),
            fused_alert_threshold=float(data.get("fused_alert_threshold", 0.999)),
        )


class IncidentPriority(str, Enum):
    """Operational triage priority tier for incident records."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class IncidentConfig:
    """Settings governing incident clustering, prioritization and packaging (Task 5.4)."""

    max_merge_gap_seconds: int = 7200  # Merge anomalous hours within 2 hours
    critical_threshold: float = 0.9995  # Automatic CRITICAL priority threshold
    high_threshold: float = 0.9990  # HIGH priority threshold
    medium_threshold: float = 0.9900  # MEDIUM priority threshold
    include_sub_threshold_investigations: bool = False  # Whether to export LOW/informational items

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IncidentConfig:
        return cls(
            max_merge_gap_seconds=int(data.get("max_merge_gap_seconds", 7200)),
            critical_threshold=float(data.get("critical_threshold", 0.9995)),
            high_threshold=float(data.get("high_threshold", 0.9990)),
            medium_threshold=float(data.get("medium_threshold", 0.9900)),
            include_sub_threshold_investigations=bool(
                data.get("include_sub_threshold_investigations", False)
            ),
        )

