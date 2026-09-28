"""Short-term sequence detector: user-hour sequences and a GRU autoencoder (Tasks 3.1-3.4).

The package imports PyTorch lazily through its model, training, scoring and
detector modules; sequence construction and configuration need only NumPy
and PyArrow.
"""

from dualscope.sequence.builder import (
    ChunkIndex,
    DaySequences,
    build_chunks,
    gather_padded,
    load_day_sequences,
)
from dualscope.sequence.config import SequenceDetectorConfig, input_feature_names

__all__ = [
    "ChunkIndex",
    "DaySequences",
    "SequenceDetectorConfig",
    "build_chunks",
    "gather_padded",
    "input_feature_names",
    "load_day_sequences",
]
