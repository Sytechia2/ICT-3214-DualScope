#!/usr/bin/env python3
"""Evaluate the causal confirmed-relationship layer on user-day labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualscope.graph.threat_history import ThreatHistory, classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="data/raw/lanl_features_days_01_30/raw/events")
    parser.add_argument("--labels", default="outputs/graph_evaluation/redteam_labels")
    parser.add_argument("--scores", default="outputs/graph_evaluation/scores")
    parser.add_argument("--history-start-day", type=int, default=8)
    parser.add_argument("--history-end-day", type=int, default=16)
    parser.add_argument("--evaluation-start-day", type=int, default=17)
    parser.add_argument("--evaluation-end-day", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()

    label_rows = ds.dataset(args.labels, format="parquet", partitioning="hive").to_table(
        columns=["timestamp", "user", "source_computer", "destination_computer", "dataset_day"]
    ).to_pylist()
    history_labels = [
        row
        for row in label_rows
        if args.history_start_day <= int(row["dataset_day"]) <= args.history_end_day
    ]
    freeze_timestamp = (args.evaluation_start_day - 1) * 86_400 + 1
    history = ThreatHistory.from_labels(history_labels, frozen_before=freeze_timestamp)

    alerts: set[tuple[int, str]] = set()
    source_values = pa.array(sorted(history.source_computers))
    for day in range(args.evaluation_start_day, args.evaluation_end_day + 1):
        path = Path(args.events) / f"dataset_day={day:02d}" / "part-000000.parquet"
        table = pq.ParquetFile(path).read(
            columns=["timestamp", "acting_user", "source_computer", "destination_computer"]
        )
        if len(source_values):
            table = table.filter(pc.is_in(table["source_computer"], value_set=source_values))
        for row in table.to_pylist():
            if history.matches(row):
                alerts.add((day, str(row["acting_user"])))

    score_rows = ds.dataset(args.scores, format="parquet", partitioning="hive").to_table(
        columns=["dataset_day", "user_id"]
    ).to_pylist()
    universe = {
        (int(row["dataset_day"]), str(row["user_id"]))
        for row in score_rows
        if args.evaluation_start_day <= int(row["dataset_day"]) <= args.evaluation_end_day
    }
    positives = {
        (int(row["dataset_day"]), str(row["user"]))
        for row in label_rows
        if args.evaluation_start_day <= int(row["dataset_day"]) <= args.evaluation_end_day
    }
    result = {
        "detector": "confirmed_relationship_history",
        "semantics": "signature match; separate from the unsupervised graph anomaly score",
        "history_days": [args.history_start_day, args.history_end_day],
        "evaluation_days": [args.evaluation_start_day, args.evaluation_end_day],
        "freeze_timestamp": freeze_timestamp,
        "confirmed_relationships": len(history.triples),
        "metrics": classification_metrics(universe, positives, alerts),
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
