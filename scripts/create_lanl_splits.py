#!/usr/bin/env python3
"""CLI script to generate and verify DualScope Task 2.3 chronological splits manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Allow running directly from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dualscope.splits import (
    SECONDS_PER_DAY,
    SplitConfig,
    SplitInterval,
    aggregate_labels_to_user_hours,
    dataset_day,
    deduplicate_labels,
    derive_excluded_users,
    is_fitting_eligible,
)


def verify_parquet_metadata_and_counts(
    events_dir: Path,
    auth_summary_path: Path | None,
) -> dict[int, int]:
    """Verify Parquet file metadata and return row counts aggregated by dataset_day."""
    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory does not exist: {events_dir}")

    parquet_files = sorted(events_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {events_dir}")

    day_counts: dict[int, int] = {}
    total_metadata_rows = 0

    for pf in parquet_files:
        # Expected structure: dataset_day=NN/part-NNNNNN.parquet
        day_part = pf.parent.name
        if not day_part.startswith("dataset_day="):
            raise ValueError(f"Unexpected partition directory name: {day_part}")
        day = int(day_part.split("=")[1])

        meta = pq.read_metadata(pf)
        num_rows = meta.num_rows
        total_metadata_rows += num_rows
        day_counts[day] = day_counts.get(day, 0) + num_rows

    if auth_summary_path and auth_summary_path.exists():
        auth_summary = json.loads(auth_summary_path.read_text(encoding="utf-8"))
        summary_counts = auth_summary.get("counts", {})
        if not summary_counts.get("reconciled", False):
            raise ValueError(f"Authentication summary at {auth_summary_path} is not reconciled")

        accepted_rows = summary_counts.get("accepted_rows")
        if accepted_rows != total_metadata_rows:
            raise ValueError(
                f"Parquet metadata total rows ({total_metadata_rows}) does not match "
                f"summary accepted_rows ({accepted_rows})"
            )

        summary_parts = auth_summary.get("parts", [])
        if summary_parts:
            summary_part_names = {p["path"] for p in summary_parts}
            actual_part_names = {
                f"{pf.parent.name}/{pf.name}" for pf in parquet_files
            }
            if summary_part_names != actual_part_names:
                missing = summary_part_names - actual_part_names
                unexpected = actual_part_names - summary_part_names
                raise ValueError(
                    f"Parquet part mismatch! Missing: {missing}, Unexpected: {unexpected}"
                )

    return day_counts


def inspect_labels_by_split(
    labels_dir: Path,
    config: SplitConfig,
    labels_summary_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Inspect red-team labels and compute raw rows, unique labels, duplicates, and user-hours."""
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    labels_dataset = ds.dataset(labels_dir, format="parquet", partitioning="hive")
    total_label_table = labels_dataset.to_table()
    total_raw_rows = len(total_label_table)

    if labels_summary_path and labels_summary_path.exists():
        labels_summary = json.loads(labels_summary_path.read_text(encoding="utf-8"))
        summary_counts = labels_summary.get("counts", {})
        if not summary_counts.get("reconciled", False):
            raise ValueError(f"Labels summary at {labels_summary_path} is not reconciled")
        if summary_counts.get("accepted_rows") != total_raw_rows:
            raise ValueError(
                f"Labels metadata total rows ({total_raw_rows}) does not match summary ({summary_counts.get('accepted_rows')})"
            )

    split_label_stats: dict[str, dict[str, Any]] = {}
    for name, split in config.splits.items():
        mask = pc.and_(
            pc.greater_equal(total_label_table["timestamp"], split.timestamp_start),
            pc.less(total_label_table["timestamp"], split.timestamp_end),
        )
        split_table = total_label_table.filter(mask)
        raw_count = len(split_table)

        dedup_table = deduplicate_labels(split_table)
        unique_count = len(dedup_table)
        duplicate_count = raw_count - unique_count

        user_hours = aggregate_labels_to_user_hours(
            dedup_table, hour_seconds=config.sequence_hour_seconds
        )
        user_hour_count = len(user_hours)

        # Unique entities in this split
        distinct_users = len(set(dedup_table["user"].to_pylist())) if raw_count > 0 else 0
        distinct_src_comps = len(set(dedup_table["source_computer"].to_pylist())) if raw_count > 0 else 0
        distinct_dst_comps = len(set(dedup_table["destination_computer"].to_pylist())) if raw_count > 0 else 0

        # Days represented
        days_rep = sorted(set(dataset_day(ts) for ts in dedup_table["timestamp"].to_pylist())) if raw_count > 0 else []

        split_label_stats[name] = {
            "label_rows": raw_count,
            "unique_labels": unique_count,
            "duplicate_rows": duplicate_count,
            "user_hour_units": user_hour_count,
            "distinct_users": distinct_users,
            "distinct_source_computers": distinct_src_comps,
            "distinct_destination_computers": distinct_dst_comps,
            "labelled_days": days_rep,
        }

    return split_label_stats


def count_training_exclusions(
    events_dir: Path,
    train_split: SplitInterval,
    excluded_users: list[str],
    day_counts: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Scan training events projecting only source/destination users to count exclusions.

    When excluded_users is empty, returns (0, total_train_events).
    """
    if not excluded_users:
        if day_counts is not None:
            total_train = sum(
                day_counts.get(d, 0)
                for d in range(train_split.dataset_day_start, train_split.dataset_day_end + 1)
            )
        else:
            total_train = 0
            for d in range(train_split.dataset_day_start, train_split.dataset_day_end + 1):
                day_dir = events_dir / f"dataset_day={d}"
                if day_dir.exists():
                    for pf in day_dir.glob("*.parquet"):
                        total_train += pq.read_metadata(pf).num_rows
        return 0, total_train

    dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
    train_filter = (
        (ds.field("timestamp") >= train_split.timestamp_start)
        & (ds.field("timestamp") < train_split.timestamp_end)
        & (ds.field("dataset_day") >= train_split.dataset_day_start)
        & (ds.field("dataset_day") <= train_split.dataset_day_end)
    )

    scanner = dataset.scanner(
        filter=train_filter,
        columns=["source_user", "destination_user"],
        batch_size=262_144,
    )

    total_scanned = 0
    excluded_count = 0
    user_arr = pa.array(excluded_users, type=pa.string())

    for batch in scanner.to_batches():
        total_scanned += len(batch)
        src_mask = pc.is_in(batch["source_user"], value_set=user_arr)
        dst_mask = pc.is_in(batch["destination_user"], value_set=user_arr)
        ex_mask = pc.or_(src_mask, dst_mask)
        excluded_count += pc.sum(ex_mask).as_py()

    eligible_count = total_scanned - excluded_count
    return excluded_count, eligible_count



def generate_splits_manifest(
    config_path: Path,
    events_dir: Path,
    auth_summary_path: Path | None,
    labels_dir: Path,
    labels_summary_path: Path | None,
    selection_manifest_path: Path | None,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Generate and write the complete Task 2.3 split manifest."""
    t0 = time.time()
    config = SplitConfig.from_file(config_path)
    config.validate()

    print(f"Loaded and validated split configuration from {config_path}")
    print(f"Policy version: {config.policy_version}, Fingerprint: {config.fingerprint()}")

    # 1. Parquet metadata and event counts
    print(f"Reading event Parquet metadata from {events_dir}...")
    day_counts = verify_parquet_metadata_and_counts(events_dir, auth_summary_path)
    total_events = sum(day_counts.values())
    print(f"Total authentication events in scope: {total_events:,} across {len(day_counts)} days")

    # 2. Inspect labels by split
    print(f"Reading red-team labels from {labels_dir}...")
    labels_dataset = ds.dataset(labels_dir, format="parquet", partitioning="hive")
    split_label_stats = inspect_labels_by_split(labels_dir, config, labels_summary_path)

    total_label_rows = sum(s["label_rows"] for s in split_label_stats.values())
    total_unique_labels = sum(s["unique_labels"] for s in split_label_stats.values())
    print(f"Total red-team labels: {total_label_rows} rows, {total_unique_labels} unique records")

    # 3. Derive training exclusions
    train_split = config.get_split("train")
    excluded_users = derive_excluded_users(
        labels_dataset,
        train_start=train_split.timestamp_start,
        train_end_exclusive=train_split.timestamp_end,
    )
    print(f"Derived {len(excluded_users)} excluded users from training labels: {excluded_users}")

    # 4. Count training exclusions
    print("Scanning training events for source/destination user exclusions...")
    excluded_events, eligible_events = count_training_exclusions(
        events_dir, train_split, excluded_users, day_counts=day_counts
    )
    print(f"Training events: {excluded_events:,} excluded, {eligible_events:,} eligible for fitting")

    # Calculate graph scoring eligibility per split
    first_scorable_ts = config.timestamp_start_inclusive + config.graph_lookback_seconds
    scorable_day_threshold = dataset_day(first_scorable_ts)
    is_day_aligned = ((first_scorable_ts - 1) % SECONDS_PER_DAY) == 0

    # If partial-day lookback, scan boundary day partition to get exact event count < first_scorable_ts
    partial_day_insufficient = 0
    if not is_day_aligned and scorable_day_threshold <= config.dataset_day_end_inclusive:
        b_filter = (
            (ds.field("dataset_day") == scorable_day_threshold)
            & (ds.field("timestamp") < first_scorable_ts)
        )
        b_dataset = ds.dataset(events_dir, format="parquet", partitioning="hive")
        partial_day_insufficient = b_dataset.to_table(filter=b_filter, columns=["timestamp"]).num_rows

    split_graph_stats: dict[str, dict[str, int]] = {}
    total_graph_available = 0
    total_graph_insufficient = 0

    for name, split in config.splits.items():
        split_ev = sum(
            day_counts.get(d, 0)
            for d in range(split.dataset_day_start, split.dataset_day_end + 1)
        )
        if split.dataset_day_end < scorable_day_threshold:
            insufficient = split_ev
            available = 0
        elif split.dataset_day_start > scorable_day_threshold:
            insufficient = 0
            available = split_ev
        else:
            # Split contains the scorable_day_threshold
            full_days_insufficient = sum(
                day_counts.get(d, 0)
                for d in range(split.dataset_day_start, scorable_day_threshold)
            )
            if is_day_aligned:
                insufficient = full_days_insufficient
            else:
                insufficient = full_days_insufficient + partial_day_insufficient
            available = split_ev - insufficient

        split_graph_stats[name] = {
            "available": available,
            "insufficient_history": insufficient,
            "out_of_bounds": 0,
        }
        total_graph_available += available
        total_graph_insufficient += insufficient

    # 5. Split summaries
    splits_manifest_data: dict[str, Any] = {}
    for name, split in config.splits.items():
        split_events = sum(
            day_counts.get(d, 0)
            for d in range(split.dataset_day_start, split.dataset_day_end + 1)
        )
        pct_events = (split_events / total_events) * 100 if total_events > 0 else 0
        lbl_stats = split_label_stats[name]
        pct_labels = (lbl_stats["label_rows"] / total_label_rows) * 100 if total_label_rows > 0 else 0

        split_info: dict[str, Any] = {
            "name": name,
            "dataset_day_start": split.dataset_day_start,
            "dataset_day_end": split.dataset_day_end,
            "timestamp_start": split.timestamp_start,
            "timestamp_end": split.timestamp_end,
            "duration_seconds": split.duration_seconds,
            "description": split.description,
            "authentication_events": split_events,
            "percentage_of_total_events": round(pct_events, 4),
            "label_rows": lbl_stats["label_rows"],
            "percentage_of_label_rows": round(pct_labels, 4),
            "unique_labels": lbl_stats["unique_labels"],
            "duplicate_label_rows": lbl_stats["duplicate_rows"],
            "user_hour_positive_units": lbl_stats["user_hour_units"],
            "distinct_labelled_users": lbl_stats["distinct_users"],
            "distinct_labelled_source_computers": lbl_stats["distinct_source_computers"],
            "distinct_labelled_destination_computers": lbl_stats["distinct_destination_computers"],
            "labelled_days": lbl_stats["labelled_days"],
            "graph_scoring_eligibility": split_graph_stats[name],
        }

        if name == "train":
            split_info["excluded_training_events"] = excluded_events
            split_info["eligible_training_events"] = eligible_events
            split_info["training_eligibility_percentage"] = round(
                (eligible_events / split_events) * 100 if split_events > 0 else 0, 4
            )

        splits_manifest_data[name] = split_info

    # 6. Import and dynamically calculate selection matching provenance if available
    matching_provenance: dict[str, Any] = {}
    unmatched_labels_total = 0
    if selection_manifest_path and selection_manifest_path.exists():
        sel = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        daily_cov = sel.get("daily_coverage", [])
        rt_match = sel.get("red_team_authentication_matching", {})

        if daily_cov:
            # 1. Validate required fields in each daily record
            required_fields = {"day", "matched_unique_labels", "unmatched_unique_labels"}
            for idx, row in enumerate(daily_cov):
                missing_fields = required_fields - set(row.keys())
                if missing_fields:
                    raise ValueError(
                        f"Selection manifest daily_coverage row {idx} is missing required fields: {sorted(missing_fields)}"
                    )
                if not isinstance(row["day"], int) or row["day"] < 1:
                    raise ValueError(
                        f"Selection manifest daily_coverage row {idx} has invalid day: {row.get('day')}"
                    )
                if not isinstance(row["matched_unique_labels"], int) or row["matched_unique_labels"] < 0:
                    raise ValueError(
                        f"Selection manifest daily_coverage row {idx} has invalid matched_unique_labels: {row.get('matched_unique_labels')}"
                    )
                if not isinstance(row["unmatched_unique_labels"], int) or row["unmatched_unique_labels"] < 0:
                    raise ValueError(
                        f"Selection manifest daily_coverage row {idx} has invalid unmatched_unique_labels: {row.get('unmatched_unique_labels')}"
                    )

            # 2. Validate that all expected days in dataset scope are present
            expected_days = set(range(config.dataset_day_start_inclusive, config.dataset_day_end_inclusive + 1))
            present_days = {row["day"] for row in daily_cov}
            missing_days = expected_days - present_days
            if missing_days:
                raise ValueError(
                    f"Selection manifest daily_coverage is missing expected dataset days: {sorted(missing_days)}"
                )

            # 3. Calculate per-split breakdowns
            split_matching: dict[str, dict[str, Any]] = {}
            unmatched_by_day: dict[str, int] = {}
            unmatched_splits_set: set[str] = set()

            for name, split in config.splits.items():
                split_matched = sum(
                    row["matched_unique_labels"]
                    for row in daily_cov
                    if split.dataset_day_start <= row["day"] <= split.dataset_day_end
                )
                split_unmatched = sum(
                    row["unmatched_unique_labels"]
                    for row in daily_cov
                    if split.dataset_day_start <= row["day"] <= split.dataset_day_end
                )
                split_total_unique = split_matched + split_unmatched
                split_coverage = (
                    round((split_matched / split_total_unique) * 100, 4)
                    if split_total_unique > 0
                    else 100.0
                )
                split_matching[name] = {
                    "matched_unique_labels": split_matched,
                    "unmatched_unique_labels": split_unmatched,
                    "match_coverage_percentage": split_coverage,
                }

            for row in daily_cov:
                u_count = row["unmatched_unique_labels"]
                if u_count > 0:
                    d = row["day"]
                    unmatched_by_day[str(d)] = u_count
                    matched_split = config.find_split_for_timestamp(((d - 1) * SECONDS_PER_DAY) + 1)
                    if matched_split:
                        unmatched_splits_set.add(matched_split.name)

            total_matched = sum(s["matched_unique_labels"] for s in split_matching.values())
            total_unmatched = sum(s["unmatched_unique_labels"] for s in split_matching.values())
            unmatched_labels_total = total_unmatched

            # 4. Reconcile against red_team_authentication_matching
            if rt_match:
                expected_matched = rt_match.get("matched_unique_labels")
                expected_unmatched = rt_match.get("unmatched_unique_labels")
                if expected_matched is not None and total_matched != expected_matched:
                    raise ValueError(
                        f"Reconciliation error: sum of daily matched unique labels ({total_matched}) "
                        f"does not match selection manifest red_team_authentication_matching ({expected_matched})"
                    )
                if expected_unmatched is not None and total_unmatched != expected_unmatched:
                    raise ValueError(
                        f"Reconciliation error: sum of daily unmatched unique labels ({total_unmatched}) "
                        f"does not match selection manifest red_team_authentication_matching ({expected_unmatched})"
                    )

            matched_auth_rows = rt_match.get("matched_authentication_rows")

            matching_provenance = {
                "source_manifest": str(selection_manifest_path.as_posix()),
                "provenance_note": (
                    "Dynamically aggregated per split from Task 2.1 selection manifest daily coverage."
                ),
                "matched_unique_labels_total": total_matched,
                "unmatched_unique_labels_total": total_unmatched,
                "matched_authentication_rows_total": matched_auth_rows,
                "unmatched_labels_by_day": unmatched_by_day,
                "unmatched_labels_split_assignment": sorted(unmatched_splits_set),
                "split_matching_breakdown": split_matching,
            }

        else:
            matching_provenance = {
                "source_manifest": str(selection_manifest_path.as_posix()),
                "status": "omitted_insufficient_provenance",
                "explanation": (
                    "Selection manifest at source path lacks 'daily_coverage'; "
                    "matching breakdown omitted to avoid magic numbers."
                ),
            }
    else:
        matching_provenance = {
            "status": "omitted_no_selection_manifest",
            "explanation": "No selection manifest provided; label-matching breakdown omitted.",
        }

    # 7. Dynamic evaluation note
    if unmatched_labels_total > 0:
        eval_note = (
            f"Treatment of the {unmatched_labels_total} unmatched labels is deferred to the Task 9.1 evaluation protocol. "
            "They are retained in the label source and must not be silently discarded."
        )
    else:
        eval_note = (
            "Treatment of unmatched labels is deferred to the Task 9.1 evaluation protocol. "
            "All labels are retained in the label source and must not be silently discarded."
        )

    # 8. Reconciliation checks
    total_split_events = sum(
        s["authentication_events"] for s in splits_manifest_data.values()
    )
    event_reconciled = total_split_events == total_events
    label_reconciled = (
        sum(s["label_rows"] for s in splits_manifest_data.values()) == total_label_rows
    )
    train_ev = splits_manifest_data.get("train", {}).get("authentication_events", 0)
    eligibility_reconciled = (
        (excluded_events + eligible_events) == train_ev if "train" in splits_manifest_data else True
    )

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "task": "2.3 — Create chronological data splits",
        "split_policy_id": config.policy_name,
        "created_date": time.strftime("%Y-%m-%d"),
        "config_fingerprint": config.fingerprint(),
        "config_reference": str(config_path.as_posix()),
        "source_dataset_reference": str(events_dir.parent.as_posix()),
        "selection_manifest_reference": (
            str(selection_manifest_path.as_posix()) if selection_manifest_path else None
        ),
        "dataset_scope": {
            "dataset_day_start_inclusive": config.dataset_day_start_inclusive,
            "dataset_day_end_inclusive": config.dataset_day_end_inclusive,
            "timestamp_start_inclusive": config.timestamp_start_inclusive,
            "timestamp_end_exclusive": config.timestamp_end_exclusive,
            "total_authentication_events": total_events,
            "total_red_team_rows": total_label_rows,
            "total_unique_red_team_labels": total_unique_labels,
        },
        "splits": splits_manifest_data,
        "training_exclusions": {
            "exclusion_policy": config.exclusion_policy,
            "exclusion_window": config.exclusion_window,
            "excluded_users_count": len(excluded_users),
            "excluded_users": excluded_users,
            "training_total_events": train_ev,
            "excluded_events": excluded_events,
            "eligible_events": eligible_events,
            "eligibility_percentage": round(
                (eligible_events / train_ev) * 100 if train_ev > 0 else 0, 4
            ),
            "eligibility_rule": (
                "An event is eligible for model fitting, learned vocabulary construction, "
                "and numeric scaling if and only if neither source_user nor destination_user "
                "is in excluded_users."
            ),
            "benign_disclaimer": (
                "Eligible training data is not guaranteed benign. Unlabelled authentication activity "
                "is not proven benign and may contain undetected lateral movement, routine administrative "
                "activity, or unlabelled anomalous behaviour."
            ),
            "implementation_note": (
                "Future sequence and graph builders must not reintroduce excluded events into "
                "model fitting samples."
            ),
        },
        "graph_warm_up_policy": {
            "lookback_seconds": config.graph_lookback_seconds,
            "first_scorable_timestamp": first_scorable_ts,
            "insufficient_history_status": "insufficient_history",
            "scoring_status_counts": {
                "available": total_graph_available,
                "insufficient_history": total_graph_insufficient,
                "out_of_bounds": 0,
            },
            "by_split": split_graph_stats,
            "activity_disclaimer": (
                "Time coverage does not guarantee sufficient user or host activity. "
                "Downstream graph models must handle unscorable or low-degree entities explicitly."
            ),
        },
        "sequence_policy": {
            "hour_seconds": config.sequence_hour_seconds,
            "hour_origin": 1,
            "hour_start_formula": "1 + ((timestamp - 1) // 3600) * 3600",
            "boundary_constraint": (
                "Sequence target windows must stay strictly within one assigned split "
                "and one dataset hour."
            ),
            "unscorable_candidate_tracking": {
                "status": "contract_defined_not_materialized",
                "implementation_task": "3.1",
                "contract_class": "dualscope.splits.SequenceCandidateRejectionTracker",
                "rejection_reasons": [
                    "split_boundary_violation",
                    "hour_boundary_violation",
                    "insufficient_history",
                    "out_of_bounds",
                ],
                "policy_note": (
                    "Sequence instances are not materialized until Task 3.1. Downstream sequence "
                    "builders must use SequenceCandidateRejectionTracker to record rejected candidate "
                    "instances rather than silently dropping them."
                ),
            },
        },
        "replay_policy": {
            "ordering": "Ascending (timestamp, source_line)",
            "replay_scope": (
                "Default replay retains all observed events up to the split exclusive end, "
                "including training exclusions, so cumulative features initialize correctly."
            ),
            "frozen_inference": (
                "Model weights, learned preprocessing, and thresholds remain strictly frozen "
                "during validation and test inference."
            ),
        },
        "label_matching_provenance": matching_provenance,
        "evaluation_protocol_note": eval_note,
        "reconciliation": {
            "all_reconciled": bool(
                event_reconciled and label_reconciled and eligibility_reconciled
            ),
            "event_counts_reconciled": event_reconciled,
            "label_counts_reconciled": label_reconciled,
            "training_eligibility_reconciled": eligibility_reconciled,
        },
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dt = time.time() - t0
    print(f"Manifest written to {output_manifest_path} in {dt:.2f}s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DualScope Task 2.3 splits manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/lanl_splits.json"),
        help="Path to splits configuration JSON",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/processed/lanl_auth_days_01_30/authentication/events"),
        help="Path to processed authentication events directory",
    )
    parser.add_argument(
        "--auth-summary",
        type=Path,
        default=Path("data/processed/lanl_auth_days_01_30/authentication/summary.json"),
        help="Path to authentication ingestion summary JSON",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/labels"),
        help="Path to processed redteam labels directory",
    )
    parser.add_argument(
        "--labels-summary",
        type=Path,
        default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/summary.json"),
        help="Path to redteam labels ingestion summary JSON",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("data/manifests/lanl_auth_days_01_30.json"),
        help="Path to Task 2.1 selection manifest (optional)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/lanl_splits_v1.json"),
        help="Path for generated splits manifest JSON",
    )

    args = parser.parse_args()

    auth_summary = args.auth_summary if args.auth_summary.exists() else None
    labels_summary = args.labels_summary if args.labels_summary.exists() else None
    sel_manifest = args.selection_manifest if args.selection_manifest.exists() else None

    manifest = generate_splits_manifest(
        config_path=args.config,
        events_dir=args.events_dir,
        auth_summary_path=auth_summary,
        labels_dir=args.labels_dir,
        labels_summary_path=labels_summary,
        selection_manifest_path=sel_manifest,
        output_manifest_path=args.output_manifest,
    )

    if not manifest["reconciliation"]["all_reconciled"]:
        print("ERROR: Split counts did not reconcile!", file=sys.stderr)
        sys.exit(1)

    print("Task 2.3 manifest generation and reconciliation successful.")


if __name__ == "__main__":
    main()
