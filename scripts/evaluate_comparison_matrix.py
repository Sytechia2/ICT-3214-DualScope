#!/usr/bin/env python3
"""Task 9.3: Unified Master Comparative Evaluation Matrix.

Evaluates all models on the exact same unit (user-hour), under a rigorous
train-test split:
  - Training Set: Days 08-12 (fitting SupervisedFusionModel and calibration)
  - Test Set: Days 13-16 (unseen validation holdout, 1,776,074 user-hours, 136 attacks)
  - Final Test Set: Days 17-30 (strictly frozen, untouched)

Evaluates at Member 2's empirical GRU alert budget (38 alerts/day, 152 alerts total across
the 4-day test period), alongside budget sensitivity curves (10, 25, 38, 50, 100 alerts/day)
and incident-level triage metrics with strict ground-truth temporal overlap.

Models compared:
  1. GRU Alone (Sequence Autoencoder)
  2. Graph GAE Alone (Graph Novelty)
  3. Simple Baseline: Maximum Fusion
  4. Simple Baseline: Average Fusion (50/50)
  5. DualScope Temporal Fusion (calibrated tau=6h, boost=0.15)
  6. DualScope Supervised Fusion (seq + graph + temporal boost + lead time)
  7. Known-Attack Lookup (Non-ML ThreatHistory recurrence baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.fusion.config import DisagreementType, FusionMethod, IncidentConfig
from dualscope.fusion.incident import IncidentClusterer
from dualscope.fusion.supervised import SupervisedFusionModel
from dualscope.fusion.temporal import TemporalFusedScoreRow
from dualscope.graph.threat_history import ThreatHistory


def load_redteam_labels(labels_path: Path) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    """Load ground-truth red team labels as user-hour tuples."""
    labels_ds = ds.dataset(str(labels_path), format="parquet", partitioning="hive")
    rows = labels_ds.to_table().to_pylist()
    redteam_hours = set()
    for r in rows:
        ts = int(r["timestamp"])
        hour_start = 1 + ((ts - 1) // 3600) * 3600
        redteam_hours.add((str(r["user"]), hour_start))
    return redteam_hours, rows


def load_day_scoring_units(
    day: int,
    seq_root: Path,
    graph_root: Path,
    redteam_hours: set[tuple[str, int]],
    temporal_decay: float = 21600.0,
    temporal_boost_weight: float = 0.15,
) -> dict[str, Any]:
    """Load and causally align sequence and graph scores for one dataset day."""
    seq_path = seq_root / f"dataset_day={day:02d}"
    seq_files = list(seq_path.glob("*.parquet"))
    if not seq_files:
        raise FileNotFoundError(f"Missing sequence scores for day {day} at {seq_path}")

    st = pq.read_table(seq_files[0], columns=["user_id", "window_start", "score", "is_alert"])
    users = st["user_id"].to_pylist()
    windows = st["window_start"].to_numpy()
    seq_scores = np.nan_to_num(st["score"].to_numpy(), nan=0.0)
    seq_alerts = np.nan_to_num(st["is_alert"].to_numpy(), nan=False).astype(bool)

    # Causally available graph scores are from day - 1
    graph_path = graph_root / f"dataset_day={day - 1:02d}"
    graph_files = list(graph_path.glob("*.parquet"))
    graph_map: dict[str, float] = {}
    graph_alert_map: dict[str, bool] = {}

    if graph_files:
        gt = pq.read_table(graph_files[0], columns=["user_id", "score", "is_alert"])
        g_users = gt["user_id"].to_pylist()
        g_scores = gt["score"].to_numpy()
        g_alerts = gt["is_alert"].to_numpy()
        for u, s, a in zip(g_users, g_scores, g_alerts):
            graph_map[u] = float(s) if s is not None and not np.isnan(s) else 0.0
            graph_alert_map[u] = bool(a) if a is not None else False

    graph_scores = np.array([graph_map.get(u, 0.0) for u in users], dtype=np.float64)
    graph_alerts = np.array([graph_alert_map.get(u, False) for u in users], dtype=bool)

    # Base simple average fusion
    base_fused = 0.5 * seq_scores + 0.5 * graph_scores

    # Temporal co-occurrence boost and lead time
    day_start = (day - 1) * 86400 + 1
    boosts = np.zeros(len(users), dtype=np.float64)
    lead_times = np.zeros(len(users), dtype=np.float64)

    for i in range(len(users)):
        if graph_alerts[i] and seq_alerts[i]:
            delta_t = max(0, int(windows[i]) - day_start)
            boost = temporal_boost_weight * np.exp(-delta_t / temporal_decay)
            boosts[i] = boost
            lead_times[i] = delta_t

    temp_scores = np.clip(base_fused + boosts, 0.0, 1.0)
    log_lead_time = np.log1p(lead_times)

    y_true = np.array([(u, w) in redteam_hours for u, w in zip(users, windows)], dtype=bool)

    return {
        "day": day,
        "users": users,
        "windows": windows,
        "seq_scores": seq_scores,
        "seq_alerts": seq_alerts,
        "graph_scores": graph_scores,
        "graph_alerts": graph_alerts,
        "base_fused": base_fused,
        "temporal_boost": boosts,
        "temp_scores": temp_scores,
        "log_lead_time": log_lead_time,
        "y": y_true,
    }


def compute_best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute best F1-score and corresponding threshold via precision-recall curve."""
    valid = np.isfinite(scores)
    y_v = y_true[valid]
    s_v = scores[valid]
    if len(s_v) == 0 or y_v.sum() == 0:
        return 0.0, 0.0
    p, r, t = precision_recall_curve(y_v, s_v)
    f1 = 2 * p * r / (p + r + 1e-12)
    best_idx = np.argmax(f1)
    thresh = float(t[best_idx]) if best_idx < len(t) else 1.0
    return float(f1[best_idx]), thresh


def evaluate_incident_triage(
    incidents: list[Any],
    redteam_hours: set[tuple[str, int]],
    budget: int,
    total_attacks: int,
) -> dict[str, Any]:
    """Evaluate incident triage with strict temporal overlap and workload tracking.

    An incident is counted as a hit IF AND ONLY IF at least one hour inside its
    span [start_time, end_time) is a confirmed red-team attack hour for that user.
    Total hours reviewed is tracked as the sum of incident duration hours.
    """
    top_incidents = incidents[:budget]
    hit_incidents = []
    attack_hours_caught: set[tuple[str, int]] = set()
    total_hours_reviewed = 0

    for inc in top_incidents:
        if isinstance(inc, dict):
            u = inc["user_id"]
            st = inc["start_time"]
            et = inc["end_time"]
            dur_h = inc.get("duration_hours", max(1, (et - st) // 3600))
        else:
            u = inc.user_id
            st = inc.start_time
            et = inc.end_time
            dur_h = inc.duration_hours

        total_hours_reviewed += dur_h
        inc_hours = range(st, et, 3600)
        attacks_in_inc = {h for h in inc_hours if (u, h) in redteam_hours}
        if attacks_in_inc:
            hit_incidents.append(inc)
            attack_hours_caught.update((u, h) for h in attacks_in_inc)

    caught_users = {
        (inc["user_id"] if isinstance(inc, dict) else inc.user_id) for inc in hit_incidents
    }
    num_hits = len(hit_incidents)
    prec = num_hits / budget * 100 if budget > 0 else 0.0
    rec = len(attack_hours_caught) / total_attacks * 100 if total_attacks > 0 else 0.0

    return {
        "triaged_incidents": len(top_incidents),
        "total_hours_reviewed": total_hours_reviewed,
        "avg_hours_per_incident": round(total_hours_reviewed / max(1, len(top_incidents)), 2),
        "malicious_incidents_caught": num_hits,
        "incident_precision_pct": round(prec, 2),
        "actual_attack_hours_caught": len(attack_hours_caught),
        "attack_hours_recall_pct": round(rec, 2),
        "distinct_attack_accounts_caught": len(caught_users),
        "caught_accounts": sorted(list(caught_users)),
    }


def run_proper_train_test_matrix(
    seq_root: Path,
    graph_root: Path,
    labels_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute complete out-of-sample evaluation: Train Days 8-12, Test Days 13-16."""
    t0 = time.time()
    print("=" * 95)
    print("DUALSCOPE MASTER COMPARATIVE EVALUATION MATRIX (OUT-OF-SAMPLE)")
    print("Protocol: Train on Days 08-12 | Test on Days 13-16 (Unseen Validation Days)")
    print("Days 17-30 remain strictly frozen.")
    print("=" * 95)

    # 1. Load labels
    print(f"Loading ground-truth redteam labels from {labels_path}...")
    redteam_hours, raw_label_rows = load_redteam_labels(labels_path)

    # 2. Load Train Days 8-12
    print("Loading and aligning training days (Days 08-12)...")
    train_days = [
        load_day_scoring_units(d, seq_root, graph_root, redteam_hours) for d in range(8, 13)
    ]
    X_train_4 = np.column_stack(
        [
            np.concatenate([d["seq_scores"] for d in train_days]),
            np.concatenate([d["graph_scores"] for d in train_days]),
            np.concatenate([d["temporal_boost"] for d in train_days]),
            np.concatenate([d["log_lead_time"] for d in train_days]),
        ]
    )
    y_train = np.concatenate([d["y"] for d in train_days])
    n_train_units = len(y_train)
    n_train_attacks = int(y_train.sum())
    print(f"Training set: {n_train_units:,} user-hours | Ground-truth attacks: {n_train_attacks}")

    # 3. Fit Supervised Fusion Model
    print("Fitting Supervised Fusion Model (Logistic Regression) on Days 08-12...")
    clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
    clf.fit(X_train_4, y_train)
    print(f"Model coefficients: {clf.coef_}, intercept: {clf.intercept_}")

    # 4. Load Test Days 13-16
    print("Loading and aligning test days (Days 13-16)...")
    test_days = [
        load_day_scoring_units(d, seq_root, graph_root, redteam_hours) for d in range(13, 17)
    ]
    y_test_pool = np.concatenate([d["y"] for d in test_days])
    n_test_units = len(y_test_pool)
    n_test_attacks = int(y_test_pool.sum())
    print(f"Test set: {n_test_units:,} user-hours | Ground-truth attacks: {n_test_attacks}")

    # Compute model predictions on test days
    for d in test_days:
        X_test_4 = np.column_stack(
            [d["seq_scores"], d["graph_scores"], d["temporal_boost"], d["log_lead_time"]]
        )
        d["sup_scores"] = clf.predict_proba(X_test_4)[:, 1]

        # Threat history signature baseline (frozen before test days start: Day 13 ts)
        freeze_ts = (13 - 1) * 86400 + 1
        th = ThreatHistory.from_labels(raw_label_rows, frozen_before=freeze_ts)
        known_users = {u for u, s, dst in th.triples}
        d["sig_scores"] = np.array(
            [1.0 if u in known_users else 0.0 for u in d["users"]], dtype=np.float64
        )

    # 5. Model score definitions across test days
    models: dict[str, list[np.ndarray]] = {
        "GRU Alone (Sequence)": [d["seq_scores"] for d in test_days],
        "Graph GAE Alone": [d["graph_scores"] for d in test_days],
        "Baseline: Maximum Fusion": [
            np.fmax(d["seq_scores"], d["graph_scores"]) for d in test_days
        ],
        "Baseline: Average Fusion": [d["base_fused"] for d in test_days],
        "DualScope: Temporal Fusion": [d["temp_scores"] for d in test_days],
        "DualScope: Supervised Fusion": [d["sup_scores"] for d in test_days],
        "Known-Attack Lookup": [d["sig_scores"] for d in test_days],
    }

    # 6. Evaluation at Member 2's empirical GRU alert budget (38 alerts/day x 4 days = 152 alerts)
    budget_daily = 38
    tot_budget = budget_daily * len(test_days)

    primary_results = []
    print("\n" + "=" * 95)
    print(f"        DUALSCOPE MASTER COMPARATIVE EVALUATION MATRIX (HOURLY BENCHMARK)")
    print(f"  Fixed Budget: {budget_daily} alerts/day ({tot_budget} alerts over 4 days) | Attacks: {n_test_attacks}")
    print("=" * 95)
    header = (
        f"| {'Model / Detection Architecture':<32} | {'Hits @ 152':<10} | {'Precision':<10} "
        f"| {'Recall':<10} | {'PR-AUC (AP)':<12} | {'Best F1':<8} |"
    )
    separator = f"|{'-'*34}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*10}|"
    print(header)
    print(separator)

    for name, daily_scores in models.items():
        daily_hits = []
        for d, s in zip(test_days, daily_scores):
            topK = np.argsort(-s, kind="stable")[:budget_daily]
            daily_hits.append(int(d["y"][topK].sum()))

        tot_hits = sum(daily_hits)
        prec = tot_hits / tot_budget * 100
        rec = tot_hits / n_test_attacks * 100

        pooled_s = np.concatenate(daily_scores)
        ap = average_precision_score(y_test_pool, pooled_s)
        f1, thresh = compute_best_f1(y_test_pool, pooled_s)

        res_entry = {
            "model_name": name,
            "hits": tot_hits,
            "budget": tot_budget,
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "average_precision": round(float(ap), 6),
            "best_f1": round(float(f1), 4),
            "best_threshold": round(float(thresh), 6),
            "daily_hits": daily_hits,
        }
        primary_results.append(res_entry)

        hits_str = f"{tot_hits:>2} / {tot_budget}"
        prec_str = f"{prec:>6.2f}%"
        rec_str = f"{rec:>6.2f}%"
        ap_str = f"{ap:>10.6f}"
        f1_str = f"{f1:>6.4f}"
        print(f"| {name:<32} | {hits_str:<10} | {prec_str:<10} | {rec_str:<10} | {ap_str:<12} | {f1_str:<8} |")

    # 7. Budget sensitivity curves (10, 25, 38, 50, 100 alerts/day)
    budget_curve_results: dict[int, list[dict[str, Any]]] = {}
    budgets = [10, 25, 38, 50, 100]

    print("\n" + "=" * 95)
    print("        BUDGET SENSITIVITY CURVES ACROSS TEST DAYS 13-16")
    print("=" * 95)
    b_header = (
        f"| {'Alert Budget / Day':<20} | {'Total Alerts (4d)':<18} | {'GRU Alone (Hits)':<18} "
        f"| {'Supervised (Hits)':<18} | {'Supervised Prec':<16} | {'Supervised Rec':<15} |"
    )
    b_sep = f"|{'-'*22}|{'-'*20}|{'-'*20}|{'-'*20}|{'-'*18}|{'-'*17}|"
    print(b_header)
    print(b_sep)

    for b in budgets:
        b_tot = b * len(test_days)
        b_entries = []
        for name in ["GRU Alone (Sequence)", "DualScope: Supervised Fusion"]:
            daily_scores = models[name]
            h = sum(int(d["y"][np.argsort(-s, kind="stable")[:b]].sum()) for d, s in zip(test_days, daily_scores))
            b_entries.append({"model": name, "hits": h, "prec": h / b_tot * 100, "rec": h / n_test_attacks * 100})
        budget_curve_results[b] = b_entries

        gru_entry = b_entries[0]
        sup_entry = b_entries[1]
        print(
            f"| {b:>2} alerts/day        | {b_tot:>3} alerts         "
            f"| {gru_entry['hits']:>2}/{b_tot} ({gru_entry['rec']:>5.2f}%)   "
            f"| {sup_entry['hits']:>2}/{b_tot} ({sup_entry['rec']:>5.2f}%)   "
            f"| {sup_entry['prec']:>6.2f}%         | {sup_entry['rec']:>6.2f}%         |"
        )

    # 8. Incident-Level Triage (Task 5.4 Operational Workload & Clustering)
    print("\n" + "=" * 95)
    print("        TASK 5.4 INCIDENT-LEVEL TRIAGE & WORKLOAD REDUCTION (TEST DAYS 13-16)")
    print("=" * 95)

    inc_cfg = IncidentConfig(max_merge_gap_seconds=7200)
    clusterer = IncidentClusterer(inc_cfg)

    all_test_incidents = []
    total_raw_fused_alerts = 0

    for d in test_days:
        day = d["day"]
        alert_mask = d["temp_scores"] >= 0.998188
        total_raw_fused_alerts += int(alert_mask.sum())

        temp_rows = []
        for i in np.where(alert_mask)[0]:
            u = d["users"][i]
            ws = int(d["windows"][i])
            temp_rows.append(
                TemporalFusedScoreRow(
                    user_id=u,
                    window_start=ws,
                    window_end=ws + 3600,
                    dataset_day=day,
                    split="validation",
                    base_fused_score=float(d["base_fused"][i]),
                    temporal_boost=float(d["temporal_boost"][i]),
                    fused_score=float(d["temp_scores"][i]),
                    is_fused_alert=True,
                    fusion_method=FusionMethod.TEMPORAL,
                    disagreement_type=DisagreementType.CONCORDANT_NORMAL,
                    lead_detector="sequence" if d["seq_alerts"][i] else "graph",
                    lead_time_seconds=int(d["log_lead_time"][i]),
                    seq_score=float(d["seq_scores"][i]),
                    is_seq_alert=bool(d["seq_alerts"][i]),
                    graph_score=float(d["graph_scores"][i]),
                    is_graph_alert=bool(d["graph_alerts"][i]),
                    alignment_status="both_available",
                )
            )
        day_incidents = clusterer.cluster_incidents(temp_rows)
        # Score each incident by max supervised probability
        sup_map = {(u, w): s for u, w, s in zip(d["users"], d["windows"], d["sup_scores"])}
        for inc in day_incidents:
            probs = [sup_map.get((inc.user_id, h), 0.0) for h in range(inc.start_time, inc.end_time, 3600)]
            inc_score = max(probs) if probs else inc.max_fused_score
            all_test_incidents.append((inc_score, inc))

    total_incidents_created = len(all_test_incidents)
    clustering_reduction_pct = (
        (total_raw_fused_alerts - total_incidents_created) / total_raw_fused_alerts * 100
        if total_raw_fused_alerts > 0
        else 0.0
    )

    # Sort incidents globally by supervised score
    all_test_incidents.sort(key=lambda x: -x[0])
    sorted_incidents = [inc for _, inc in all_test_incidents]

    inc_eval_res = evaluate_incident_triage(
        incidents=sorted_incidents,
        redteam_hours=redteam_hours,
        budget=tot_budget,
        total_attacks=n_test_attacks,
    )

    print(f"Alert Volume Compression:")
    print(f"  Raw Fused Hourly Alerts: {total_raw_fused_alerts} alerts across 4 test days")
    print(f"  Clustered Incident Envelopes: {total_incidents_created} incidents ({clustering_reduction_pct:.1f}% ticket reduction)")
    print(f"Incident Triage Performance (Triaging all {inc_eval_res['triaged_incidents']} available incidents):")
    print(f"  Total Analyst Hours Reviewed: {inc_eval_res['total_hours_reviewed']} hours (workload explicitly reported)")
    print(f"  Malicious Incidents Caught: {inc_eval_res['malicious_incidents_caught']} / {inc_eval_res['triaged_incidents']} ({inc_eval_res['incident_precision_pct']}%)")
    print(f"  Actual Ground-Truth Attack Hours Caught: {inc_eval_res['actual_attack_hours_caught']} / {n_test_attacks} ({inc_eval_res['attack_hours_recall_pct']}% recall)")
    print(f"  Distinct Attacker Accounts Caught: {inc_eval_res['distinct_attack_accounts_caught']} accounts: {inc_eval_res['caught_accounts']}")
    print("=" * 95 + "\n")

    matrix_report = {
        "evaluation_protocol": "Train: Days 08-12 | Test: Days 13-16 (out-of-sample)",
        "train_units": n_train_units,
        "train_attacks": n_train_attacks,
        "test_units": n_test_units,
        "test_attacks": n_test_attacks,
        "fixed_budget_daily": budget_daily,
        "fixed_budget_total": tot_budget,
        "hourly_models": primary_results,
        "budget_sensitivity": budget_curve_results,
        "incident_triage": {
            "raw_fused_alerts": total_raw_fused_alerts,
            "clustered_incidents": total_incidents_created,
            "clustering_reduction_pct": round(clustering_reduction_pct, 2),
            **inc_eval_res,
        },
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_out = output_dir / "comparison_matrix_outofsample.json"
        json_out.write_text(json.dumps(matrix_report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation report to {json_out}")

    return matrix_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Master Comparative Evaluation Matrix (Task 9.3).")
    parser.add_argument(
        "--seq-scores-dir",
        type=Path,
        default=Path("seq-gru-ae-v1-L32-h32/sequence_scores_seq_validation/outputs/sequence_scores/seq-gru-ae-v1/scores"),
        help="Root directory containing sequence scores dataset_day=XX",
    )
    parser.add_argument(
        "--graph-scores-dir",
        type=Path,
        default=Path("Long Term Graphs/scores"),
        help="Root directory containing graph scores dataset_day=XX",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/labels"),
        help="Path to ground truth redteam labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for matrix reports",
    )

    args = parser.parse_args()
    run_proper_train_test_matrix(
        seq_root=args.seq_scores_dir,
        graph_root=args.graph_scores_dir,
        labels_path=args.labels_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
