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
  6. Supervised Fusion, logistic regression (seq + graph + temporal boost + lead time)
  7. Known-Attack Lookup (Non-ML ThreatHistory recurrence baseline)
  8. DualScope final model, gradient boosting (GRU + hourly counts), when
     ``--final-model-scores`` is given

Every threshold that uses labels (detector alert cut-offs, temporal fusion alert
threshold) is fitted on Days 08-12 by best F1. Test-day labels are used only to
score the results.
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

LR_NAME = "Supervised Fusion: LogReg (seq+graph+temporal)"
FINAL_NAME = "DualScope Final: GradBoost (GRU+hourly counts)"
IFOREST_NAME = "Baseline: Isolation Forest (flat features)"


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
) -> dict[str, Any]:
    """Load and causally align sequence and graph scores for one dataset day.

    The exported ``is_alert`` flags are kept only for reference: their thresholds
    were chosen on all validation days 8-16, so the temporal features are built
    later by ``add_temporal_features`` from cut-offs fitted on the training days.
    """
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

    y_true = np.array([(u, w) in redteam_hours for u, w in zip(users, windows)], dtype=bool)

    return {
        "day": day,
        "users": users,
        "windows": windows,
        "seq_scores": seq_scores,
        "exported_seq_alerts": seq_alerts,
        "graph_scores": graph_scores,
        "exported_graph_alerts": graph_alerts,
        "base_fused": 0.5 * seq_scores + 0.5 * graph_scores,
        "y": y_true,
    }


def add_temporal_features(
    d: dict[str, Any],
    seq_cutoff: float,
    graph_cutoff: float,
    temporal_decay: float = 21600.0,
    temporal_boost_weight: float = 0.15,
) -> None:
    """Add detector alerts, temporal boost and lead time to one loaded day.

    ``lead_time`` is the time from the start of the day (when the previous day's
    graph score becomes available) to the start of the co-alerting hour. It is
    non-zero only when both detectors alert.
    """
    seq_alerts = d["seq_scores"] >= seq_cutoff
    graph_alerts = d["graph_scores"] >= graph_cutoff
    both = seq_alerts & graph_alerts
    day_start = (d["day"] - 1) * 86400 + 1
    delta_t = np.maximum(0, d["windows"].astype(np.int64) - day_start)
    lead_times = np.where(both, delta_t, 0).astype(np.float64)
    boosts = np.where(both, temporal_boost_weight * np.exp(-delta_t / temporal_decay), 0.0)

    d["seq_alerts"] = seq_alerts
    d["graph_alerts"] = graph_alerts
    d["temporal_boost"] = boosts
    d["lead_time_seconds"] = lead_times
    d["log_lead_time"] = np.log1p(lead_times)
    d["temp_scores"] = np.clip(d["base_fused"] + boosts, 0.0, 1.0)


def load_final_model_scores(path: Path, test_days: list[dict[str, Any]]) -> None:
    """Attach the final model's per-hour scores and tie order to each test day.

    The file holds one row per test user-hour (``user_id``, ``window_start``,
    ``score``, ``tie_order``). Every test unit must have a score.
    """
    table = pq.read_table(str(path), columns=["user_id", "window_start", "score", "tie_order"])
    by_unit = {
        (u, int(w)): (float(s), int(t))
        for u, w, s, t in zip(
            table["user_id"].to_pylist(),
            table["window_start"].to_numpy(),
            table["score"].to_numpy(),
            table["tie_order"].to_numpy(),
        )
    }
    for d in test_days:
        pairs = [by_unit.get((u, int(w))) for u, w in zip(d["users"], d["windows"])]
        missing = sum(p is None for p in pairs)
        if missing:
            raise SystemExit(f"final model scores missing for {missing:,} day-{d['day']} units in {path}")
        d["final_scores"] = np.array([p[0] for p in pairs], dtype=np.float64)
        d["final_tie_order"] = np.array([p[1] for p in pairs], dtype=np.int64)


def load_baseline_scores(path: Path, test_days: list[dict[str, Any]]) -> None:
    """Attach Isolation Forest scores (``user_id``, ``window_start``, ``score``) to each test day.

    The baseline scores every user-hour with events; a test unit without a
    baseline score fails the run, so every model is compared on the same units.
    """
    table = pq.read_table(str(path), columns=["user_id", "window_start", "score"])
    by_unit = {
        (u, int(w)): float(s)
        for u, w, s in zip(table["user_id"].to_pylist(), table["window_start"].to_numpy(), table["score"].to_numpy())
    }
    for d in test_days:
        scores = [by_unit.get((u, int(w))) for u, w in zip(d["users"], d["windows"])]
        missing = sum(x is None for x in scores)
        if missing:
            raise SystemExit(f"baseline scores missing for {missing:,} day-{d['day']} units in {path}")
        d["baseline_scores"] = np.array(scores, dtype=np.float64)


def top_k(scores: np.ndarray, k: int, tie_order: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` highest scores; ties by ``tie_order``, else file order."""
    if tie_order is None:
        return np.argsort(-scores, kind="stable")[:k]
    return np.lexsort((tie_order, -scores))[:k]


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
    prec = num_hits / len(top_incidents) * 100 if top_incidents else 0.0
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
    final_scores_path: Path | None = None,
    baseline_scores_path: Path | None = None,
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
    y_train = np.concatenate([d["y"] for d in train_days])

    # Detector alert cut-offs: best F1 on the training days only, so no label from
    # the test days 13-16 reaches the temporal features.
    _, seq_cutoff = compute_best_f1(y_train, np.concatenate([d["seq_scores"] for d in train_days]))
    _, graph_cutoff = compute_best_f1(y_train, np.concatenate([d["graph_scores"] for d in train_days]))
    print(f"Alert cut-offs fitted on Days 08-12 (best F1): sequence >= {seq_cutoff:.6f}, graph >= {graph_cutoff:.6f}")
    for d in train_days:
        add_temporal_features(d, seq_cutoff, graph_cutoff)
    # The temporal fusion alert threshold (incident input) uses the same rule.
    _, temporal_cutoff = compute_best_f1(y_train, np.concatenate([d["temp_scores"] for d in train_days]))
    print(f"Temporal fusion alert threshold fitted on Days 08-12 (best F1): >= {temporal_cutoff:.6f}")

    X_train_4 = np.column_stack(
        [
            np.concatenate([d["seq_scores"] for d in train_days]),
            np.concatenate([d["graph_scores"] for d in train_days]),
            np.concatenate([d["temporal_boost"] for d in train_days]),
            np.concatenate([d["log_lead_time"] for d in train_days]),
        ]
    )
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
    for d in test_days:
        add_temporal_features(d, seq_cutoff, graph_cutoff)
    if final_scores_path is not None:
        load_final_model_scores(final_scores_path, test_days)
    if baseline_scores_path is not None:
        load_baseline_scores(baseline_scores_path, test_days)
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
        LR_NAME: [d["sup_scores"] for d in test_days],
        "Known-Attack Lookup": [d["sig_scores"] for d in test_days],
    }
    # Ties: file order, except the final model, which keeps its own experiment's
    # fixed random tie order (seed 0) so its result matches the recorded one.
    tie_orders: dict[str, list[np.ndarray] | None] = {name: None for name in models}
    if baseline_scores_path is not None:
        models[IFOREST_NAME] = [d["baseline_scores"] for d in test_days]
        tie_orders[IFOREST_NAME] = None
    if final_scores_path is not None:
        models[FINAL_NAME] = [d["final_scores"] for d in test_days]
        tie_orders[FINAL_NAME] = [d["final_tie_order"] for d in test_days]

    # 6. Evaluation at Member 2's empirical GRU alert budget (38 alerts/day x 4 days = 152 alerts)
    budget_daily = 38
    tot_budget = budget_daily * len(test_days)

    primary_results = []
    print("\n" + "=" * 95)
    print(f"        DUALSCOPE MASTER COMPARATIVE EVALUATION MATRIX (HOURLY BENCHMARK)")
    print(f"  Fixed Budget: {budget_daily} alerts/day ({tot_budget} alerts over 4 days) | Attacks: {n_test_attacks}")
    print("=" * 95)
    header = (
        f"| {'Model / Detection Architecture':<48} | {'Hits @ 152':<10} | {'Precision':<10} "
        f"| {'Recall':<10} | {'PR-AUC (AP)':<12} | {'Best F1':<8} |"
    )
    separator = f"|{'-'*50}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*10}|"
    print(header)
    print(separator)

    def daily_top_hits(name: str, k: int) -> list[int]:
        orders = tie_orders[name] or [None] * len(test_days)
        return [
            int(d["y"][top_k(s, k, o)].sum())
            for d, s, o in zip(test_days, models[name], orders)
        ]

    for name, daily_scores in models.items():
        daily_hits = daily_top_hits(name, budget_daily)

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
        print(f"| {name:<48} | {hits_str:<10} | {prec_str:<10} | {rec_str:<10} | {ap_str:<12} | {f1_str:<8} |")

    # 7. Budget sensitivity curves (10, 25, 38, 50, 100 alerts/day)
    budget_curve_results: dict[int, list[dict[str, Any]]] = {}
    budgets = [10, 25, 38, 50, 100]
    curve_models = [name for name in ("GRU Alone (Sequence)", IFOREST_NAME, LR_NAME, FINAL_NAME) if name in models]

    print("\n" + "=" * 95)
    print("        BUDGET SENSITIVITY ACROSS TEST DAYS 13-16 (hits / precision / recall)")
    print("=" * 95)
    print(f"| {'Alerts/day':<10} | {'Total':>5} | " + " | ".join(f"{name:<48}" for name in curve_models) + " |")
    print(f"|{'-'*12}|{'-'*7}|" + "|".join("-" * 50 for _ in curve_models) + "|")

    for b in budgets:
        b_tot = b * len(test_days)
        b_entries = []
        for name in curve_models:
            h = sum(daily_top_hits(name, b))
            b_entries.append({"model": name, "hits": h, "prec": h / b_tot * 100, "rec": h / n_test_attacks * 100})
        budget_curve_results[b] = b_entries
        cells = [f"{e['hits']:>3} / {e['prec']:>5.2f}% / {e['rec']:>5.2f}%" for e in b_entries]
        print(f"| {b:<10} | {b_tot:>5} | " + " | ".join(f"{c:<48}" for c in cells) + " |")

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
        alert_mask = d["temp_scores"] >= temporal_cutoff
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
                    lead_time_seconds=int(d["lead_time_seconds"][i]),
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
        "cutoffs_fitted_on_train_days": {
            "rule": "best F1 on Days 08-12",
            "sequence_alert": seq_cutoff,
            "graph_alert": graph_cutoff,
            "temporal_fusion_alert": temporal_cutoff,
        },
        "final_model_scores": str(final_scores_path) if final_scores_path else None,
        "baseline_scores": str(baseline_scores_path) if baseline_scores_path else None,
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
        default=Path("outputs/sequence_scores/seq-gru-ae-v1-L32-h32-91e4b11d34/scores"),
        help="Root directory containing sequence scores dataset_day=XX",
    )
    parser.add_argument(
        "--graph-scores-dir",
        type=Path,
        default=Path("outputs/graph_scores_v1/scores"),
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
    parser.add_argument(
        "--final-model-scores",
        type=Path,
        default=None,
        help="Per-hour Days 13-16 scores of the final gradient boosting model (fitted on Days 08-12)",
    )
    parser.add_argument(
        "--baseline-scores",
        type=Path,
        default=None,
        help="Isolation Forest validation scores from train_baseline_isolation_forest.py",
    )

    args = parser.parse_args()
    run_proper_train_test_matrix(
        seq_root=args.seq_scores_dir,
        graph_root=args.graph_scores_dir,
        labels_path=args.labels_dir,
        output_dir=args.output_dir,
        final_scores_path=args.final_model_scores,
        baseline_scores_path=args.baseline_scores,
    )


if __name__ == "__main__":
    main()
