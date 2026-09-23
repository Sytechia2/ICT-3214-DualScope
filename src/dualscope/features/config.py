"""Feature pipeline configuration and validation.

Defines feature versions, lookback windows, category tokens, and explicit
model-input column groupings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LOOKBACK_1H = 3_600
DEFAULT_LOOKBACK_24H = 86_400

DEFAULT_MODEL_INPUTS_NUMERIC = (
    "log1p_prior_auth_count_1h_scaled",
    "log1p_prior_failure_count_1h_scaled",
    "log1p_seconds_since_previous_auth_scaled",
    "log1p_prior_unique_destinations_24h_scaled",
    "log1p_prior_user_destination_count_24h_scaled",
)

DEFAULT_MODEL_INPUTS_BINARY = (
    "has_user_history",
    "is_new_user_destination",
    "is_new_host_connection",
)

DEFAULT_MODEL_INPUTS_CATEGORICAL = (
    "auth_type_id",
    "logon_type_id",
    "auth_orientation_id",
    "auth_result_id",
)

DEFAULT_METADATA_COLUMNS = (
    "source_reference",
    "source_line",
    "timestamp",
    "dataset_day",
    "acting_user",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "authentication_result",
    "prior_auth_count_1h",
    "prior_failure_count_1h",
    "seconds_since_previous_auth",
    "prior_unique_destinations_24h",
    "prior_user_destination_count_24h",
    "history_complete_1h",
    "history_complete_24h",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that govern causal feature extraction and training-only preprocessing."""

    feature_version: str = "1.0.0"
    feature_name: str = "lanl_historical_features_v1"
    dataset_start_timestamp: int = 1
    lookback_1h_seconds: int = DEFAULT_LOOKBACK_1H
    lookback_24h_seconds: int = DEFAULT_LOOKBACK_24H
    missing_gap_placeholder: float = 0.0
    unseen_category_token: str = "<UNSEEN>"
    unseen_category_id: int = 0
    missing_category_token: str = "<MISSING>"
    missing_category_id: int = 1
    model_inputs_numeric: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_INPUTS_NUMERIC))
    model_inputs_binary: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_INPUTS_BINARY))
    model_inputs_categorical: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_INPUTS_CATEGORICAL))
    metadata_columns: list[str] = field(default_factory=lambda: list(DEFAULT_METADATA_COLUMNS))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.dataset_start_timestamp < 1:
            raise ValueError(f"dataset_start_timestamp must be >= 1, got {self.dataset_start_timestamp}")
        if self.lookback_1h_seconds <= 0:
            raise ValueError(f"lookback_1h_seconds must be positive, got {self.lookback_1h_seconds}")
        if self.lookback_24h_seconds <= self.lookback_1h_seconds:
            raise ValueError("lookback_24h_seconds must be strictly greater than lookback_1h_seconds")
        if self.unseen_category_id == self.missing_category_id:
            raise ValueError("unseen_category_id and missing_category_id must be distinct")

        # Disallow timestamp or identity columns in model inputs
        prohibited = {"timestamp", "source_line", "dataset_day", "raw_record", "source_reference",
                      "history_complete_1h", "history_complete_24h"}
        all_model_inputs = (
            set(self.model_inputs_numeric)
            | set(self.model_inputs_binary)
            | set(self.model_inputs_categorical)
        )
        contaminated = all_model_inputs & prohibited
        if contaminated:
            raise ValueError(f"Prohibited administrative/time fields in model inputs: {sorted(contaminated)}")

    def all_model_input_names(self) -> list[str]:
        """Return deterministic ordered list of all model input column names."""
        return list(self.model_inputs_numeric) + list(self.model_inputs_binary) + list(self.model_inputs_categorical)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 hash of configuration."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureConfig:
        return cls(
            feature_version=str(data.get("feature_version", "1.0.0")),
            feature_name=str(data.get("feature_name", "lanl_historical_features_v1")),
            dataset_start_timestamp=int(data.get("dataset_start_timestamp", 1)),
            lookback_1h_seconds=int(data.get("lookback_1h_seconds", DEFAULT_LOOKBACK_1H)),
            lookback_24h_seconds=int(data.get("lookback_24h_seconds", DEFAULT_LOOKBACK_24H)),
            missing_gap_placeholder=float(data.get("missing_gap_placeholder", 0.0)),
            unseen_category_token=str(data.get("unseen_category_token", "<UNSEEN>")),
            unseen_category_id=int(data.get("unseen_category_id", 0)),
            missing_category_token=str(data.get("missing_category_token", "<MISSING>")),
            missing_category_id=int(data.get("missing_category_id", 1)),
            model_inputs_numeric=list(data.get("model_inputs_numeric", DEFAULT_MODEL_INPUTS_NUMERIC)),
            model_inputs_binary=list(data.get("model_inputs_binary", DEFAULT_MODEL_INPUTS_BINARY)),
            model_inputs_categorical=list(data.get("model_inputs_categorical", DEFAULT_MODEL_INPUTS_CATEGORICAL)),
            metadata_columns=list(data.get("metadata_columns", DEFAULT_METADATA_COLUMNS)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> FeatureConfig:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    @classmethod
    def default(cls) -> FeatureConfig:
        return cls()
