#!/usr/bin/env python3
"""Export counter-enhanced GAE and relationship recurrence side by side."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dualscope.graph.hybrid import collect_relationship_matches, combine_user_day
from dualscope.graph.threat_history import ThreatHistory, classification_metrics
from dualscope.sequence.calibration import QuantileTailCalibrator

DEFAULT_NOVELTY_THRESHOLD = 0.9981996450132613


def _counter_lookup(dataset: ds.Dataset) -> dict[tuple[int, str], int]:
    rows = dataset.to_table(
        columns=["dataset_day", "user_id", "peak_unique_destinations_300s"]
    ).to_pylist()
    return {
        (int(row["dataset_day"]), str(row["user_id"])): int(
            row["peak_unique_destinations_300s"]
        )
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="data/raw/lanl_features_days_01_30/raw/events")
    parser.add_argument("--graph-scores", default="outputs/graph_evaluation/scores")
    parser.add_argument("--burst-scores", default="outputs/graph_evaluation/burst_scores")
    parser.add_argument("--labels", default="outputs/graph_evaluation/redteam_labels")
    parser.add_argument("--history-start-day", type=int, default=8)
    parser.add_argument("--history-end-day", type=int, default=16)
    parser.add_argument("--start-day", type=int, default=17)
    parser.add_argument("--end-day", type=int, default=30)
    parser.add_argument(
        "--novelty-threshold", type=float, default=DEFAULT_NOVELTY_THRESHOLD
    )
    parser.add_argument("--output", default="outputs/graph_evaluation/hybrid_alerts")
    args = parser.parse_args()

    labels = ds.dataset(
        args.labels, format="parquet", partitioning="hive"
    ).to_table().to_pylist()
    history_rows = [
        row for row in labels
        if args.history_start_day <= int(row["dataset_day"]) <= args.history_end_day
    ]
    freeze_timestamp = (args.start_day - 1) * 86400 + 1
    history = ThreatHistory.from_labels(history_rows, frozen_before=freeze_timestamp)
    source_computers = pa.array(sorted(history.source_computers))

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    graph = ds.dataset(args.graph_scores, format="parquet", partitioning="hive")
    counters = _counter_lookup(
        ds.dataset(args.burst_scores, format="parquet", partitioning="hive")
    )

    # Fit the counter transform on label-free validation data and freeze it for test.
    validation_rows = graph.to_table(
        columns=["dataset_day", "user_id"],
        filter=(ds.field("dataset_day") >= args.history_start_day)
        & (ds.field("dataset_day") <= args.history_end_day),
    ).to_pylist()
    validation_counters = np.asarray(
        [counters[(int(row["dataset_day"]), str(row["user_id"]))]
         for row in validation_rows],
        dtype=np.float64,
    )
    counter_calibrator = QuantileTailCalibrator.fit(
        validation_counters,
        reference_description=(
            f"label-free Days {args.history_start_day}-{args.history_end_day} "
            "peak_unique_destinations_300s"
        ),
    )
    (output_root / "counter_calibrator.json").write_text(
        json.dumps(counter_calibrator.to_dict(), indent=2) + "\n"
    )

    all_rows: list[dict[str, object]] = []
    for day in range(args.start_day, args.end_day + 1):
        event_path = (
            Path(args.events) / f"dataset_day={day:02d}" / "part-000000.parquet"
        )
        events = pq.ParquetFile(event_path).read(
            columns=["timestamp", "acting_user", "source_computer",
                     "destination_computer", "source_reference"]
        )
        if len(source_computers):
            events = events.filter(
                pc.is_in(events["source_computer"], value_set=source_computers)
            )
        matches = collect_relationship_matches(events.to_pylist(), history)

        graph_rows = graph.to_table(
            filter=ds.field("dataset_day") == day
        ).to_pylist()
        raw_counters = np.asarray(
            [counters[(day, str(row["user_id"]))] for row in graph_rows],
            dtype=np.float64,
        )
        scaled_counters = counter_calibrator.transform(raw_counters)
        rows = []
        for graph_row, raw_counter, scaled_counter in zip(
            graph_rows, raw_counters, scaled_counters
        ):
            graph_row["base_gae_score"] = graph_row["score"]
            graph_row["base_gae_is_alert"] = graph_row["is_alert"]
            graph_row["peak_unique_destinations_300s"] = int(raw_counter)
            graph_row["score"] = (
                0.25 * float(graph_row["score"]) + 0.75 * float(scaled_counter)
            )
            graph_row["is_alert"] = graph_row["score"] >= args.novelty_threshold
            graph_row["model_version"] = f"{graph_row['model_version']}+counter-v1"
            rows.append(
                combine_user_day(
                    graph_row, matches.get(str(graph_row["user_id"]), [])
                )
            )

        all_rows.extend(rows)
        day_root = output_root / f"dataset_day={day:02d}"
        day_root.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([
                {key: value for key, value in row.items() if key != "dataset_day"}
                for row in rows
            ]),
            day_root / "part-000000.parquet",
            compression="zstd",
        )
        print(
            f"day={day:02d} users={len(rows):,} "
            f"confirmed={sum(bool(row['confirmed_relationship_alert']) for row in rows):,}",
            flush=True,
        )

    universe = {(int(row["dataset_day"]), str(row["user_id"])) for row in all_rows}
    positives = {
        (int(row["dataset_day"]), str(row["user"])) for row in labels
        if args.start_day <= int(row["dataset_day"]) <= args.end_day
    }
    channels = {
        "confirmed_relationship": {
            (int(row["dataset_day"]), str(row["user_id"])) for row in all_rows
            if row["confirmed_relationship_alert"]
        },
        "counter_enhanced_gae_alert": {
            (int(row["dataset_day"]), str(row["user_id"])) for row in all_rows
            if row["gae_is_alert"]
        },
        "any_graph_signal": {
            (int(row["dataset_day"]), str(row["user_id"])) for row in all_rows
            if row["any_graph_signal"]
        },
    }
    report = {
        "name": "hybrid_long_term_graph_detector",
        "semantics": {
            "counter_enhanced_gae_alert": "unsupervised novelty review",
            "confirmed_relationship": (
                "supervised exact recurrence, primary high-confidence alert"
            ),
            "any_graph_signal": (
                "union for analyst visibility; not the source of the 84.06% result"
            ),
        },
        "novelty_formula": (
            "0.25 * baseline_graph_score + 0.75 * "
            "validation_quantile_scaled_peak_unique_destinations_300s"
        ),
        "novelty_threshold": args.novelty_threshold,
        "history_days": [args.history_start_day, args.history_end_day],
        "evaluation_days": [args.start_day, args.end_day],
        "freeze_timestamp": freeze_timestamp,
        "confirmed_relationships": len(history.triples),
        "metrics": {
            name: classification_metrics(universe, positives, alerts)
            for name, alerts in channels.items()
        },
        "warning": (
            "Confirmed-relationship performance is not GAE performance. This "
            "backtest is retrospective and needs a fresh holdout."
        ),
    }
    (output_root / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
