"""Task 5.2/5.3: Grid search and tuning of fusion parameters on validation data.

Compares candidate fusion policies (Maximum, Simple Average, Weighted, Temporal)
against ground-truth validation red-team attack labels. Ranks them by Average Precision
(PR-AUC) and Best F1-Score, and freezes the optimal configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.fusion.alignment import ScoreAlignmentEngine
from dualscope.fusion.basic import BasicFusionEngine
from dualscope.fusion.temporal import TemporalFusionEngine
from dualscope.fusion.config import (
    AlignmentConfig,
    BasicFusionConfig,
    FusionMethod,
    TemporalConfig,
)
from dualscope.sequence.calibration import average_precision, best_f1_threshold


def _load_records(path: Path) -> list[dict]:
    """Load records from parquet file or directory."""
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table()
    return table.to_pylist()


def _load_redteam_positives(labels_dir: Path) -> set[tuple[str, int]]:
    """Load red-team positive user-hours (user, hour_start)."""
    dataset = ds.dataset(str(labels_dir), format="parquet")
    table = dataset.to_table()
    positives = set()
    for row in table.to_pylist():
        ts = row["timestamp"]
        hour_start = 1 + ((ts - 1) // 3600) * 3600
        user = row["user"]
        positives.add((user, hour_start))
    return positives


def tune_fusion_parameters(
    seq_path: Path,
    graph_path: Path | None,
    labels_dir: Path,
    output_config: Path | None = None,
) -> list[dict]:
    """Evaluate candidate fusion settings and print ranked leaderboard."""
    print("=" * 65)
    print("        DUALSCOPE FUSION PARAMETER GRID SEARCH & TUNING")
    print("=" * 65)

    # 1. Load inputs and align
    t0 = time.time()
    print(f"Loading sequence records from {seq_path}...")
    seq_records = _load_records(seq_path)
    print(f"Loaded {len(seq_records)} sequence records.")

    graph_records = []
    if graph_path and graph_path.exists():
        print(f"Loading graph records from {graph_path}...")
        graph_records = _load_records(graph_path)
        print(f"Loaded {len(graph_records)} graph records.")
    else:
        print("Note: No graph records provided. Evaluating single-detector baseline.")

    align_engine = ScoreAlignmentEngine(AlignmentConfig())
    aligned_rows = align_engine.align_records(seq_records, graph_records)
    print(f"Aligned {len(aligned_rows)} scoring units in {time.time() - t0:.2f}s.")

    # 2. Load ground-truth labels
    print(f"Loading redteam attack labels from {labels_dir}...")
    redteam_positives = _load_redteam_positives(labels_dir)
    
    # Ground-truth boolean label array
    y_true = np.array(
        [(r.user_id, r.window_start) in redteam_positives for r in aligned_rows],
        dtype=bool,
    )
    num_attacks = int(y_true.sum())
    print(f"Ground-truth attack user-hours in evaluation: {num_attacks}")

    if num_attacks == 0:
        print("Warning: No attack user-hours found in evaluation data.")
        return []

    # 3. Define candidate configurations
    candidates: list[dict] = [
        {"name": "Maximum Fusion", "method": FusionMethod.MAX, "w_seq": 0.5, "temporal": False, "tau": None, "boost": None},
        {"name": "Simple Average", "method": FusionMethod.AVERAGE, "w_seq": 0.5, "temporal": False, "tau": None, "boost": None},
        {"name": "Weighted (w=0.3)", "method": FusionMethod.WEIGHTED, "w_seq": 0.3, "temporal": False, "tau": None, "boost": None},
        {"name": "Weighted (w=0.5)", "method": FusionMethod.WEIGHTED, "w_seq": 0.5, "temporal": False, "tau": None, "boost": None},
        {"name": "Weighted (w=0.7)", "method": FusionMethod.WEIGHTED, "w_seq": 0.7, "temporal": False, "tau": None, "boost": None},
        {"name": "Temporal (tau=6h, boost=0.15)", "method": FusionMethod.WEIGHTED, "w_seq": 0.5, "temporal": True, "tau": 21600.0, "boost": 0.15},
        {"name": "Temporal (tau=12h, boost=0.15)", "method": FusionMethod.WEIGHTED, "w_seq": 0.5, "temporal": True, "tau": 43200.0, "boost": 0.15},
        {"name": "Temporal (tau=24h, boost=0.20)", "method": FusionMethod.WEIGHTED, "w_seq": 0.5, "temporal": True, "tau": 86400.0, "boost": 0.20},
    ]

    results = []

    print(f"\nEvaluating {len(candidates)} candidate fusion configurations...")
    for idx, cand in enumerate(candidates, 1):
        basic_cfg = BasicFusionConfig(
            method=cand["method"],
            sequence_weight=cand["w_seq"],
        )

        if cand["temporal"]:
            temp_cfg = TemporalConfig(
                half_life_seconds=cand["tau"],
                boost_weight=cand["boost"],
            )
            engine = TemporalFusionEngine(temporal_config=temp_cfg, base_config=basic_cfg)
            fused_rows = engine.fuse_stream(aligned_rows)
            scores = np.array([r.fused_score if r.fused_score is not None else np.nan for r in fused_rows])
        else:
            basic_engine = BasicFusionEngine(basic_cfg)
            fused_rows = basic_engine.fuse_rows(aligned_rows)
            scores = np.array([r.fused_score if r.fused_score is not None else np.nan for r in fused_rows])

        # Filter finite scores for metric calculation
        valid = np.isfinite(scores)
        y_valid = y_true[valid]
        scores_valid = scores[valid]

        ap = average_precision(y_valid, scores_valid) or 0.0
        best_f1_info = best_f1_threshold(y_valid, scores_valid)

        res = {
            "name": cand["name"],
            "method": cand["method"].value,
            "w_seq": cand["w_seq"],
            "temporal": cand["temporal"],
            "tau_hours": (cand["tau"] / 3600.0) if cand["tau"] else None,
            "boost": cand["boost"],
            "average_precision": ap,
            "best_f1": best_f1_info["f1"],
            "precision": best_f1_info["precision"] or 0.0,
            "recall": best_f1_info["recall"] or 0.0,
            "best_threshold": best_f1_info["threshold"],
            "tp": best_f1_info["tp"],
            "fp": best_f1_info["fp"],
        }
        results.append(res)
        print(f"  [{idx}/{len(candidates)}] {cand['name']:<32}: PR-AUC={ap:.6f}, Best F1={res['best_f1']:.4f} (P={res['precision']*100:.1f}%, R={res['recall']*100:.1f}%)")

    # Sort results by Average Precision (PR-AUC) descending, then F1 descending
    results.sort(key=lambda r: (r["average_precision"], r["best_f1"]), reverse=True)

    print("\n" + "=" * 80)
    print("                    FINAL FUSION CANDIDATE LEADERBOARD")
    print("=" * 80)
    print(f"{'Rank':<5} {'Configuration':<32} {'PR-AUC':<10} {'Best F1':<9} {'Prec':<8} {'Recall':<8} {'Threshold':<10}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['name']:<32} {r['average_precision']:<10.6f} {r['best_f1']:<9.4f} {r['precision']*100:<7.1f}% {r['recall']*100:<7.1f}% {r['best_threshold']:<10.6f}")
    print("=" * 80)

    winner = results[0]
    print(f"\nWinning Policy: {winner['name']}")
    print(f"  Optimal Threshold : {winner['best_threshold']:.6f}")
    print(f"  Average Precision : {winner['average_precision']:.6f}")
    print(f"  Best F1-Score     : {winner['best_f1']:.4f} (Recall: {winner['recall']*100:.1f}%, Precision: {winner['precision']*100:.1f}%)")

    if output_config:
        output_config.parent.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "selected_method": winner["method"],
            "sequence_weight": winner["w_seq"],
            "graph_weight": 1.0 - winner["w_seq"],
            "is_temporal": winner["temporal"],
            "temporal_half_life_seconds": (winner["tau_hours"] * 3600.0) if winner["tau_hours"] else None,
            "temporal_boost_weight": winner["boost"],
            "fused_alert_threshold": winner["best_threshold"],
            "metrics": {
                "average_precision": winner["average_precision"],
                "best_f1": winner["best_f1"],
                "precision": winner["precision"],
                "recall": winner["recall"],
            }
        }
        output_config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        print(f"Wrote winning configuration to {output_config}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune fusion parameters across candidate policies.")
    parser.add_argument("--sequence-scores", type=Path, required=True, help="Path to sequence scores parquet")
    parser.add_argument("--graph-scores", type=Path, default=None, help="Path to graph scores parquet (optional)")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/labels"), help="Path to redteam labels directory")
    parser.add_argument("--output-config", type=Path, default=Path("config/fusion.json"), help="Destination JSON to freeze optimal parameters")
    args = parser.parse_args()

    tune_fusion_parameters(args.sequence_scores, args.graph_scores, args.labels_dir, output_config=args.output_config)


if __name__ == "__main__":
    main()
