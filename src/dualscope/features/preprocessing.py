"""Training-only fitted preprocessing for LANL historical features.

Implements numerically stable batch-combined statistics (Chan et al.),
deterministic categorical encoding with separate UNSEEN vs MISSING codes,
and post-scaling finite placeholder insertion for missing time gaps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from dualscope.features.config import FeatureConfig
from dualscope.features.schemas import TRANSFORMED_FEATURE_SCHEMA
from dualscope.splits import SplitConfig


NUMERIC_FEATURES_TO_LOG1P = (
    "prior_auth_count_1h",
    "prior_failure_count_1h",
    "seconds_since_previous_auth",
    "prior_unique_destinations_24h",
    "prior_user_destination_count_24h",
)

CATEGORICAL_COLUMNS = (
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "authentication_result",
)


@dataclass
class NumericStat:
    """Fitted standardization parameters for a single numeric feature."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    std: float = 0.0
    scale: float = 1.0
    is_constant: bool = False

    def combine_batch(self, count_b: int, mean_b: float, m2_b: float) -> None:
        """Combine batch statistics using Chan's parallel variance algorithm."""
        if count_b == 0:
            return
        if self.count == 0:
            self.count = count_b
            self.mean = mean_b
            self.m2 = m2_b
            return

        n_a = self.count
        n_b = count_b
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + (delta ** 2) * (n_a * n_b / n)
        self.count = n

    def finalize(self) -> None:
        """Compute std and final scale factor with constant feature fallback."""
        if self.count <= 1:
            self.std = 0.0
            self.scale = 1.0
            self.is_constant = True
        else:
            var = self.m2 / self.count
            self.std = math.sqrt(max(0.0, var))
            if self.std < 1e-12:
                self.scale = 1.0
                self.is_constant = True
            else:
                self.scale = self.std
                self.is_constant = False


@dataclass
class CategoricalVocab:
    """Categorical vocabulary with distinct UNSEEN and MISSING code assignments."""

    column_name: str
    unseen_token: str = "<UNSEEN>"
    unseen_id: int = 0
    missing_token: str = "<MISSING>"
    missing_id: int = 1
    category_to_id: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.category_to_id is None:
            self.category_to_id = {}


class FeaturePreprocessor:
    """Manages training-only statistical fitting and frozen feature transformation."""

    def __init__(
        self,
        config: FeatureConfig | None = None,
        split_policy_fingerprint: str = "",
        mode: str = "production",
        is_production: bool = True,
        feature_config_fingerprint: str | None = None,
    ) -> None:
        self.config = config or FeatureConfig.default()
        self.feature_config_fingerprint = (
            feature_config_fingerprint
            if feature_config_fingerprint is not None
            else self.config.fingerprint()
        )
        self.split_policy_fingerprint = split_policy_fingerprint
        self.mode = mode
        self.is_production = is_production
        self.is_frozen = False

        # Accumulators
        self.numeric_stats: dict[str, NumericStat] = {
            f: NumericStat() for f in NUMERIC_FEATURES_TO_LOG1P
        }
        self._cat_candidates: dict[str, set[str]] = {
            c: set() for c in CATEGORICAL_COLUMNS
        }
        self.vocabularies: dict[str, CategoricalVocab] = {
            c: CategoricalVocab(
                column_name=c,
                unseen_token=self.config.unseen_category_token,
                unseen_id=self.config.unseen_category_id,
                missing_token=self.config.missing_category_token,
                missing_id=self.config.missing_category_id,
            )
            for c in CATEGORICAL_COLUMNS
        }

        # Fitting row counts
        self.total_scanned_rows: int = 0
        self.training_scanned_rows: int = 0
        self.training_eligible_fitted_rows: int = 0
        self.training_excluded_skipped_rows: int = 0
        self.validation_scanned_rows: int = 0
        self.test_scanned_rows: int = 0

    def accumulate_training_batch(
        self,
        raw_batch: pa.RecordBatch,
        is_training_eligible_mask: pa.BooleanArray,
    ) -> None:
        """Accumulate vector statistics and category candidates for eligible training rows."""
        if self.is_frozen:
            raise RuntimeError("Cannot accumulate statistics on a frozen preprocessor.")

        n_rows = len(raw_batch)
        if n_rows == 0:
            return

        eligible_count = pc.sum(pc.cast(is_training_eligible_mask, pa.int64())).as_py() or 0
        if eligible_count == 0:
            return

        self.training_eligible_fitted_rows += eligible_count

        # Filter batch to eligible rows only for fitting
        eligible_batch = pc.filter(raw_batch, is_training_eligible_mask)

        # 1. Numeric statistics (vectorized PyArrow compute)
        for col in NUMERIC_FEATURES_TO_LOG1P:
            raw_arr = eligible_batch[col]

            if col == "seconds_since_previous_auth":
                # Exclude null gaps from fitting
                non_null_arr = pc.drop_null(raw_arr)
                if len(non_null_arr) == 0:
                    continue
                # log1p(gap)
                val_arr = pc.ln(pc.add(non_null_arr, 1.0))
            else:
                # Counts: log1p(count)
                val_arr = pc.ln(pc.add(pc.cast(raw_arr, pa.float64()), 1.0))

            k = len(val_arr)
            if k == 0:
                continue

            mean_val = pc.mean(val_arr).as_py()
            # M2 = sum((x - mean)^2)
            diff = pc.subtract(val_arr, mean_val)
            m2_val = pc.sum(pc.power(diff, 2)).as_py() or 0.0

            self.numeric_stats[col].combine_batch(k, mean_val, m2_val)

        # 2. Categorical vocabularies candidates
        for cat_col in CATEGORICAL_COLUMNS:
            unique_arr = pc.unique(eligible_batch[cat_col])
            for u in unique_arr.to_pylist():
                if u is not None and str(u) != "":
                    self._cat_candidates[cat_col].add(str(u))

    def freeze(self) -> None:
        """Finalize standard scalers and build deterministic categorical vocabularies."""
        if self.is_frozen:
            return

        # Finalize numeric stats
        for stat in self.numeric_stats.values():
            stat.finalize()

        # Build categorical mappings
        for col, candidates in self._cat_candidates.items():
            sorted_cats = sorted(candidates)
            vocab = self.vocabularies[col]
            # Reserved: 0 -> UNSEEN, 1 -> MISSING
            # Training categories start at 2
            vocab.category_to_id = {
                cat: idx + 2 for idx, cat in enumerate(sorted_cats)
            }

        self.is_frozen = True

    def transform_batch(self, raw_batch: pa.RecordBatch) -> pa.RecordBatch:
        """Transform a raw feature batch into the model-ready schema using frozen preprocessing."""
        if not self.is_frozen:
            raise RuntimeError("Preprocessor must be frozen before transforming batches.")

        n_rows = len(raw_batch)
        if n_rows == 0:
            return pa.RecordBatch.from_pylist([], schema=TRANSFORMED_FEATURE_SCHEMA)

        transformed_cols: dict[str, pa.Array] = {}

        # 1. Retain metadata & identifier columns
        for name in [
            "timestamp", "source_user", "destination_user", "source_computer",
            "destination_computer", "authentication_type", "logon_type",
            "authentication_orientation", "authentication_result", "acting_user",
            "exact_duplicate_ordinal", "source_line", "source_reference",
            "prior_auth_count_1h", "prior_failure_count_1h", "seconds_since_previous_auth",
            "prior_unique_destinations_24h", "prior_user_destination_count_24h",
            "history_complete_1h", "history_complete_24h", "dataset_day",
        ]:
            transformed_cols[name] = raw_batch[name]

        # 2. Transformed numeric features (log1p + standard scaling)
        for col in NUMERIC_FEATURES_TO_LOG1P:
            stat = self.numeric_stats[col]
            out_col_name = f"log1p_{col}_scaled"

            if col == "seconds_since_previous_auth":
                # Special missing gap rule:
                # If observed: (log1p(gap) - mean) / scale
                # If missing (null): placeholder 0.0 inserted AFTER scaling
                raw_gap_list = raw_batch["seconds_since_previous_auth"].to_pylist()
                transformed_gaps = [0.0] * n_rows
                mean = stat.mean
                scale = stat.scale
                for i, gap in enumerate(raw_gap_list):
                    if gap is not None:
                        transformed_gaps[i] = (math.log1p(gap) - mean) / scale
                    else:
                        transformed_gaps[i] = self.config.missing_gap_placeholder
                transformed_cols[out_col_name] = pa.array(transformed_gaps, type=pa.float32())
            else:
                # Standard count features
                val_arr = pc.ln(pc.add(pc.cast(raw_batch[col], pa.float64()), 1.0))
                scaled = pc.divide(pc.subtract(val_arr, stat.mean), stat.scale)
                transformed_cols[out_col_name] = pc.cast(scaled, pa.float32())

        # 3. Transformed binary model inputs (cast to float32 0.0/1.0)
        transformed_cols["has_user_history"] = pc.cast(raw_batch["has_user_history"], pa.float32())
        transformed_cols["is_new_user_destination"] = pc.cast(raw_batch["is_new_user_destination"], pa.float32())
        transformed_cols["is_new_host_connection"] = pc.cast(raw_batch["is_new_host_connection"], pa.float32())
        transformed_cols["is_new_user_source"] = pc.cast(raw_batch["is_new_user_source"], pa.float32())
        transformed_cols["is_machine_account"] = pc.cast(raw_batch["is_machine_account"], pa.float32())

        # 4. Transformed categorical IDs
        # Column mappings:
        # authentication_type -> auth_type_id
        # logon_type -> logon_type_id
        # authentication_orientation -> auth_orientation_id
        # authentication_result -> auth_result_id
        cat_out_map = {
            "authentication_type": "auth_type_id",
            "logon_type": "logon_type_id",
            "authentication_orientation": "auth_orientation_id",
            "authentication_result": "auth_result_id",
        }

        for raw_col, out_col in cat_out_map.items():
            vocab = self.vocabularies[raw_col]
            mapping = vocab.category_to_id or {}
            unseen_id = vocab.unseen_id
            missing_id = vocab.missing_id

            raw_vals = raw_batch[raw_col].to_pylist()
            encoded_ids = [0] * n_rows
            for i, val in enumerate(raw_vals):
                if val is None or val == "":
                    encoded_ids[i] = missing_id
                elif str(val) in mapping:
                    encoded_ids[i] = mapping[str(val)]
                else:
                    encoded_ids[i] = unseen_id

            transformed_cols[out_col] = pa.array(encoded_ids, type=pa.int32())

        # Build in schema order
        arrays = [transformed_cols[field.name] for field in TRANSFORMED_FEATURE_SCHEMA]
        return pa.RecordBatch.from_arrays(arrays, schema=TRANSFORMED_FEATURE_SCHEMA)

    def to_dict(self) -> dict[str, Any]:
        """Convert preprocessor configuration, fitted statistics, and vocabularies to a serializable dict."""
        return {
            "feature_version": self.config.feature_version,
            "feature_name": self.config.feature_name,
            "feature_config_fingerprint": self.feature_config_fingerprint or self.config.fingerprint(),
            "split_policy_fingerprint": self.split_policy_fingerprint,
            "mode": self.mode,
            "is_production": self.is_production,
            "is_frozen": self.is_frozen,
            "missing_gap_placeholder": self.config.missing_gap_placeholder,
            "fitting_row_counts": {
                "total_scanned_rows": self.total_scanned_rows,
                "training_scanned_rows": self.training_scanned_rows,
                "training_eligible_fitted_rows": self.training_eligible_fitted_rows,
                "training_excluded_skipped_rows": self.training_excluded_skipped_rows,
                "validation_scanned_rows": self.validation_scanned_rows,
                "test_scanned_rows": self.test_scanned_rows,
            },
            "numeric_stats": {
                k: {
                    "count": v.count,
                    "mean": v.mean,
                    "std": v.std,
                    "scale": v.scale,
                    "is_constant": v.is_constant,
                }
                for k, v in self.numeric_stats.items()
            },
            "categorical_vocabularies": {
                k: {
                    "column_name": v.column_name,
                    "unseen_token": v.unseen_token,
                    "unseen_id": v.unseen_id,
                    "missing_token": v.missing_token,
                    "missing_id": v.missing_id,
                    "training_distinct_categories": len(v.category_to_id or {}),
                    "category_to_id": dict(v.category_to_id or {}),
                }
                for k, v in self.vocabularies.items()
            },
            "rules": {
                "counts_transformation": "log1p(count) followed by standard scaling (x - mean) / scale",
                "seconds_since_previous_auth_fitting": "nulls excluded from training statistics fitting",
                "seconds_since_previous_auth_transform": "log1p(gap) standardized; missing gap assigned missing_gap_placeholder (0.0) post-scaling",
                "categorical_encoding": "0 = UNSEEN, 1 = MISSING, 2+ = sorted training categories",
                "constant_feature_handling": "scale = 1.0 when std < 1e-12",
            },
        }

    def verify_compatibility(
        self,
        feature_cfg: FeatureConfig | None = None,
        splits_cfg: SplitConfig | None = None,
    ) -> None:
        """Verify that this preprocessor is compatible with given feature and split configurations.

        Raises:
            ValueError: If feature configuration fingerprint or split policy fingerprint is missing or mismatches.
        """
        if feature_cfg is not None:
            expected_feature_fp = feature_cfg.fingerprint()
            if not self.feature_config_fingerprint:
                raise ValueError(
                    "Missing 'feature_config_fingerprint' in preprocessor artifact; cannot verify compatibility."
                )
            if self.feature_config_fingerprint != expected_feature_fp:
                raise ValueError(
                    f"FeatureConfig fingerprint mismatch: preprocessor artifact specifies '{self.feature_config_fingerprint}', "
                    f"but active feature config has '{expected_feature_fp}'."
                )
        if splits_cfg is not None:
            expected_splits_fp = splits_cfg.fingerprint()
            if not self.split_policy_fingerprint:
                raise ValueError(
                    "Missing 'split_policy_fingerprint' in preprocessor artifact; cannot verify compatibility."
                )
            if self.split_policy_fingerprint != expected_splits_fp:
                raise ValueError(
                    f"SplitConfig fingerprint mismatch: preprocessor was fitted with split fingerprint "
                    f"'{self.split_policy_fingerprint}', but active split config has '{expected_splits_fp}'."
                )

    def save(self, path: str | Path) -> None:
        """Save preprocessor artifact to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: FeatureConfig | None = None) -> FeaturePreprocessor:
        cfg = config or FeatureConfig.default()
        saved_feature_fp = data.get("feature_config_fingerprint")

        if config is not None:
            if not saved_feature_fp:
                raise ValueError(
                    "Missing 'feature_config_fingerprint' in preprocessor artifact; cannot verify compatibility."
                )
            if config.fingerprint() != saved_feature_fp:
                raise ValueError(
                    f"FeatureConfig fingerprint mismatch on load: artifact specifies '{saved_feature_fp}', "
                    f"provided config has '{config.fingerprint()}'."
                )

        preprocessor = cls(
            config=cfg,
            feature_config_fingerprint=str(saved_feature_fp or ""),
            split_policy_fingerprint=str(data.get("split_policy_fingerprint", "")),
            mode=str(data.get("mode", "production")),
            is_production=bool(data.get("is_production", True)),
        )

        # Restore fitting row counts
        counts = data.get("fitting_row_counts", {})
        preprocessor.total_scanned_rows = int(counts.get("total_scanned_rows", 0))
        preprocessor.training_scanned_rows = int(counts.get("training_scanned_rows", 0))
        preprocessor.training_eligible_fitted_rows = int(counts.get("training_eligible_fitted_rows", 0))
        preprocessor.training_excluded_skipped_rows = int(counts.get("training_excluded_skipped_rows", 0))
        preprocessor.validation_scanned_rows = int(counts.get("validation_scanned_rows", 0))
        preprocessor.test_scanned_rows = int(counts.get("test_scanned_rows", 0))

        # Restore numeric stats
        for k, v in data.get("numeric_stats", {}).items():
            preprocessor.numeric_stats[k] = NumericStat(
                count=int(v["count"]),
                mean=float(v["mean"]),
                std=float(v["std"]),
                scale=float(v["scale"]),
                is_constant=bool(v.get("is_constant", False)),
            )

        # Restore vocabularies
        for k, v in data.get("categorical_vocabularies", {}).items():
            preprocessor.vocabularies[k] = CategoricalVocab(
                column_name=str(v["column_name"]),
                unseen_token=str(v.get("unseen_token", "<UNSEEN>")),
                unseen_id=int(v.get("unseen_id", 0)),
                missing_token=str(v.get("missing_token", "<MISSING>")),
                missing_id=int(v.get("missing_id", 1)),
                category_to_id={cat: int(cid) for cat, cid in v.get("category_to_id", {}).items()},
            )

        preprocessor.is_frozen = bool(data.get("is_frozen", True))
        return preprocessor

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: FeatureConfig | None = None,
        splits_cfg: SplitConfig | None = None,
    ) -> FeaturePreprocessor:
        """Load frozen preprocessor artifact from a JSON file and optionally verify compatibility."""
        text = Path(path).read_text(encoding="utf-8")
        preprocessor = cls.from_dict(json.loads(text), config=config)
        if splits_cfg is not None:
            preprocessor.verify_compatibility(splits_cfg=splits_cfg)
        return preprocessor
