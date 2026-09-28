"""CLI script for Task 5.1 (Alignment) and Task 5.2 (Basic Fusion).

Demonstrates and executes score alignment and basic fusion across sequence
and graph detector outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from dualscope.fusion.alignment import ScoreAlignmentEngine
from dualscope.fusion.basic import BasicFusionEngine
from dualscope.fusion.temporal import TemporalFusionEngine
from dualscope.fusion.config import (
    AlignmentConfig,
    BasicFusionConfig,
    FusionMethod,
    TemporalConfig,
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from a Parquet or JSONL file/dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file() and path.suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # Use PyArrow dataset to support single parquet or partitioned directory
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table()
    return table.to_pylist()


def run_pipeline(
    seq_path: Path,
    graph_path: Path | None,
    output_path: Path | None,
    fusion_method: FusionMethod,
    seq_weight: float,
    threshold: float,
    max_staleness: int,
    temporal_decay: float = 21600.0,
    temporal_boost: float = 0.15,
    temporal_lookback: int = 86400,
    incidents_output: Path | None = None,
    incident_merge_gap: int = 7200,
) -> dict[str, Any]:
    """Execute alignment, fusion, and optional incident generation end-to-end."""
    print(f"Loading sequence records from {seq_path}...")
    seq_records = _load_records(seq_path)
    print(f"Loaded {len(seq_records)} sequence records.")

    graph_records: list[dict[str, Any]] = []
    if graph_path and graph_path.exists():
        print(f"Loading graph records from {graph_path}...")
        graph_records = _load_records(graph_path)
        print(f"Loaded {len(graph_records)} graph records.")
    else:
        print("No graph records provided; aligning with sequence-only inputs.")

    # 1. Align outputs (Task 5.1)
    align_config = AlignmentConfig(max_graph_staleness_seconds=max_staleness)
    align_engine = ScoreAlignmentEngine(align_config)
    aligned_rows = align_engine.align_records(seq_records, graph_records)
    print(f"Aligned {len(aligned_rows)} scoring units.")

    status_counts = Counter(r.alignment_status.value for r in aligned_rows)
    print(f"Alignment status distribution: {dict(status_counts)}")

    # 2. Fusion (Tasks 5.2 or 5.3)
    basic_cfg = BasicFusionConfig(
        method=fusion_method if fusion_method != FusionMethod.TEMPORAL else FusionMethod.WEIGHTED,
        sequence_weight=seq_weight,
        fused_alert_threshold=threshold,
    )

    if fusion_method == FusionMethod.TEMPORAL:
        temporal_cfg = TemporalConfig(
            lookback_seconds=temporal_lookback,
            half_life_seconds=temporal_decay,
            boost_weight=temporal_boost,
            fused_alert_threshold=threshold,
        )
        temporal_engine = TemporalFusionEngine(temporal_config=temporal_cfg, base_config=basic_cfg)
        fused_rows = temporal_engine.fuse_stream(aligned_rows)
        table = temporal_engine.to_arrow_table(fused_rows)
    else:
        basic_engine = BasicFusionEngine(basic_cfg)
        fused_rows = basic_engine.fuse_rows(aligned_rows)
        table = basic_engine.to_arrow_table(fused_rows)

    disagreement_counts = Counter(r.disagreement_type.value for r in fused_rows)
    alert_count = sum(1 for r in fused_rows if r.is_fused_alert is True)
    print(f"Disagreement distribution: {dict(disagreement_counts)}")
    print(f"Total fused alerts at threshold {threshold}: {alert_count}")

    # 3. Export scores if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(table, output_path, compression="zstd")
        print(f"Wrote {table.num_rows} fused rows to {output_path}")

    # 4. Incident Generation (Task 5.4)
    incidents_count = 0
    if incidents_output:
        from dualscope.fusion.incident import IncidentClusterer
        from dualscope.fusion.config import IncidentConfig

        # If basic fusion was run, promote to temporal stream structure for clustering
        if fusion_method != FusionMethod.TEMPORAL:
            temporal_cfg = TemporalConfig(fused_alert_threshold=threshold)
            temporal_engine = TemporalFusionEngine(temporal_config=temporal_cfg, base_config=basic_cfg)
            temporal_fused_rows = temporal_engine.fuse_stream(aligned_rows)
        else:
            temporal_fused_rows = fused_rows

        inc_cfg = IncidentConfig(max_merge_gap_seconds=incident_merge_gap)
        clusterer = IncidentClusterer(inc_cfg)
        incidents = clusterer.cluster_incidents(temporal_fused_rows)
        incidents_count = len(incidents)
        print(f"Clustered {len(incidents)} multi-hour incidents.")

        incidents_output.parent.mkdir(parents=True, exist_ok=True)
        if incidents_output.suffix == ".jsonl":
            with open(incidents_output, "w", encoding="utf-8") as f:
                for inc in incidents:
                    f.write(inc.to_json() + "\n")
        else:
            inc_table = clusterer.to_arrow_table(incidents)
            import pyarrow.parquet as pq

            pq.write_table(inc_table, incidents_output, compression="zstd")
        print(f"Wrote incidents to {incidents_output}")

    return {
        "num_units": len(fused_rows),
        "alignment_statuses": dict(status_counts),
        "disagreement_types": dict(disagreement_counts),
        "total_alerts": alert_count,
        "total_incidents": incidents_count,
        "fusion_method": fusion_method.value,
        "sequence_weight": seq_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Align and fuse multi-timescale detector outputs.")
    parser.add_argument("--sequence-scores", type=Path, required=True, help="Path to sequence scores (parquet or jsonl)")
    parser.add_argument("--graph-scores", type=Path, default=None, help="Path to graph scores (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Destination parquet file for fused scores")
    parser.add_argument(
        "--method",
        type=str,
        choices=["max", "average", "weighted", "temporal"],
        default="temporal",
        help="Fusion method (default: temporal)",
    )
    parser.add_argument("--sequence-weight", type=float, default=0.5, help="Sequence weight w for weighted fusion")
    parser.add_argument("--alert-threshold", type=float, default=0.999, help="Threshold for fused alert")
    parser.add_argument(
        "--max-staleness",
        type=int,
        default=86400,
        help="Maximum allowed graph score age in seconds (default: 86400)",
    )
    parser.add_argument(
        "--temporal-decay",
        type=float,
        default=21600.0,
        help="Temporal half-life decay parameter tau in seconds (default: 21600 = 6h)",
    )
    parser.add_argument(
        "--temporal-boost",
        type=float,
        default=0.15,
        help="Maximum temporal co-occurrence boost gamma (default: 0.15)",
    )
    parser.add_argument(
        "--temporal-lookback",
        type=int,
        default=86400,
        help="Lookback window W_max in seconds (default: 86400 = 24h)",
    )
    parser.add_argument(
        "--incidents-output",
        type=Path,
        default=None,
        help="Destination JSONL or Parquet file for clustered incident records (Task 5.4)",
    )
    parser.add_argument(
        "--incident-merge-gap",
        type=int,
        default=7200,
        help="Maximum gap in seconds between alert hours to merge into an incident (default: 7200)",
    )

    args = parser.parse_args()
    run_pipeline(
        seq_path=args.sequence_scores,
        graph_path=args.graph_scores,
        output_path=args.output,
        fusion_method=FusionMethod(args.method),
        seq_weight=args.sequence_weight,
        threshold=args.alert_threshold,
        max_staleness=args.max_staleness,
        temporal_decay=args.temporal_decay,
        temporal_boost=args.temporal_boost,
        temporal_lookback=args.temporal_lookback,
        incidents_output=args.incidents_output,
        incident_merge_gap=args.incident_merge_gap,
    )


if __name__ == "__main__":
    main()
