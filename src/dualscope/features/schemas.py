"""PyArrow schemas for raw and transformed historical feature tables.

Explicitly separates model inputs from metadata, and excludes large source
blobs such as `raw_record` to maintain disk efficiency.
"""

from __future__ import annotations

import pyarrow as pa


RAW_FEATURE_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("source_user", pa.string()),
        ("destination_user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("authentication_type", pa.string()),
        ("logon_type", pa.string()),
        ("authentication_orientation", pa.string()),
        ("authentication_result", pa.string()),
        ("acting_user", pa.string()),
        ("exact_duplicate_ordinal", pa.int32()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("prior_auth_count_1h", pa.int64()),
        ("prior_failure_count_1h", pa.int64()),
        ("seconds_since_previous_auth", pa.float64()),
        ("has_user_history", pa.bool_()),
        ("prior_unique_destinations_24h", pa.int64()),
        ("prior_user_destination_count_24h", pa.int64()),
        ("is_new_user_destination", pa.bool_()),
        ("is_new_host_connection", pa.bool_()),
        ("history_complete_1h", pa.bool_()),
        ("history_complete_24h", pa.bool_()),
        ("dataset_day", pa.int32()),
    ]
)

TRANSFORMED_FEATURE_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("source_user", pa.string()),
        ("destination_user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("authentication_type", pa.string()),
        ("logon_type", pa.string()),
        ("authentication_orientation", pa.string()),
        ("authentication_result", pa.string()),
        ("acting_user", pa.string()),
        ("exact_duplicate_ordinal", pa.int32()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("prior_auth_count_1h", pa.int64()),
        ("prior_failure_count_1h", pa.int64()),
        ("seconds_since_previous_auth", pa.float64()),
        ("prior_unique_destinations_24h", pa.int64()),
        ("prior_user_destination_count_24h", pa.int64()),
        ("history_complete_1h", pa.bool_()),
        ("history_complete_24h", pa.bool_()),
        ("dataset_day", pa.int32()),
        # Transformed numeric model inputs
        ("log1p_prior_auth_count_1h_scaled", pa.float32()),
        ("log1p_prior_failure_count_1h_scaled", pa.float32()),
        ("log1p_seconds_since_previous_auth_scaled", pa.float32()),
        ("log1p_prior_unique_destinations_24h_scaled", pa.float32()),
        ("log1p_prior_user_destination_count_24h_scaled", pa.float32()),
        # Transformed binary model inputs
        ("has_user_history", pa.float32()),
        ("is_new_user_destination", pa.float32()),
        ("is_new_host_connection", pa.float32()),
        # Transformed categorical ID model inputs
        ("auth_type_id", pa.int32()),
        ("logon_type_id", pa.int32()),
        ("auth_orientation_id", pa.int32()),
        ("auth_result_id", pa.int32()),
    ]
)
