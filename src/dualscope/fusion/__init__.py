"""Fusion and incident generation package for DualScope."""

from dualscope.fusion.alignment import (
    ALIGNED_SCORE_SCHEMA,
    AlignedScoreRow,
    ScoreAlignmentEngine,
)
from dualscope.fusion.basic import (
    FUSED_SCORE_SCHEMA,
    BasicFusionEngine,
    FusedScoreRow,
    classify_disagreement,
)
from dualscope.fusion.config import (
    AlignmentConfig,
    AlignmentStatus,
    BasicFusionConfig,
    DisagreementType,
    FusionMethod,
    IncidentConfig,
    IncidentPriority,
    TemporalConfig,
)
from dualscope.fusion.incident import (
    INCIDENT_RECORD_SCHEMA,
    IncidentClusterer,
    IncidentRecord,
)
from dualscope.fusion.temporal import (
    TEMPORAL_FUSED_SCORE_SCHEMA,
    TemporalFusionEngine,
    TemporalFusedScoreRow,
)

__all__ = [
    "ALIGNED_SCORE_SCHEMA",
    "AlignedScoreRow",
    "AlignmentConfig",
    "AlignmentStatus",
    "BasicFusionConfig",
    "BasicFusionEngine",
    "DisagreementType",
    "FUSED_SCORE_SCHEMA",
    "FusedScoreRow",
    "FusionMethod",
    "INCIDENT_RECORD_SCHEMA",
    "IncidentClusterer",
    "IncidentConfig",
    "IncidentPriority",
    "IncidentRecord",
    "ScoreAlignmentEngine",
    "TEMPORAL_FUSED_SCORE_SCHEMA",
    "TemporalConfig",
    "TemporalFusedScoreRow",
    "TemporalFusionEngine",
    "classify_disagreement",
]
