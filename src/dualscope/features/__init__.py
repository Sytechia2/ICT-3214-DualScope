"""DualScope shared historical features package.

Provides a streaming causal feature engine, training-only fitted preprocessing,
and versioned schemas for converting authentication records into behavioral features.
"""

from dualscope.features.config import FeatureConfig
from dualscope.features.engine import HistoricalFeatureEngine
from dualscope.features.preprocessing import FeaturePreprocessor
from dualscope.features.schemas import (
    RAW_FEATURE_SCHEMA,
    TRANSFORMED_FEATURE_SCHEMA,
)

__all__ = [
    "FeatureConfig",
    "HistoricalFeatureEngine",
    "FeaturePreprocessor",
    "RAW_FEATURE_SCHEMA",
    "TRANSFORMED_FEATURE_SCHEMA",
]
