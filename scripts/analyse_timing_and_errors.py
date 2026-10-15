#!/usr/bin/env python3
"""Task 9.4: detection timing, detector precedence and error analysis (Days 13-16).

Everything is computed from the data; nothing is typed in by hand.

* Alerts: the top 38 user-hours per day of the final model (gradient boosting,
  fitted on Days 08-12), with the GRU alone and the logistic regression fusion
  for comparison. Detector alert cut-offs are fitted on Days 08-12 exactly as in
  ``evaluate_comparison_matrix.py``.
* Timing: one campaign per (user, day) with labelled red-team events. The first
  alert for that user and day is compared with the first labelled event. Alerts
  on hours that are not attack hours are counted separately.
* Errors: false alarms split by account type and by whether the user was
  attacked that day; missed attack hours compared with caught ones by activity.
* Case studies, chosen by fixed rules from the final model's ranking:
  - caught attack: the highest-ranked attack hour inside the budget;
  - false alarm: the highest-ranked hour without an attack label;
  - missed attack: the attack hour outside the budget with the median daily rank
    among all missed attack hours.
  Each case lists the hour's authentication profile, the labelled red-team
  events with their matching ``auth.txt`` lines, and the evidence exported by
  the GRU and the graph detector.

Days 17-30 are never read.

Example:
  python scripts/analyse_timing_and_errors.py \
    --final-model-scores outputs/experiment_v2/final_model_scores_days13_16.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dualscope.evaluation.timing import (  # noqa: E402
    AttackCampaignTiming,
    TimingCategory,
    calculate_first_alert_offset,
    determine_detector_precedence,
    summarize_timing_distributions,
)
from scripts.evaluate_comparison_matrix import (  # noqa: E402
    add_temporal_features,
    compute_best_f1,
    load_day_scoring_units,
    load_final_model_scores,
    load_redteam_labels,
    top_k,
)

TRAIN_DAYS = range(8, 13)
TEST_DAYS = range(13, 17)
HOUR = 3600
FINAL = "final_gradient_boosting"
COMPARATORS = ("gru_alone", "logreg_fusion", FINAL)


def hour_start(ts: int) -> int:
    return 1 + ((ts - 1) // HOUR) * HOUR


def daily_ranks(scores: np.ndarray, tie_order: np.ndarray | None) -> np.ndarray:
    """1-based rank of every unit within its day (1 = highest score)."""
    order = top_k(scores, len(scores), tie_order)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def load_days(seq_root: Path, graph_root: Path, labels_path: Path, final_scores: Path) -> dict[str, Any]:
    redteam_hours, label_rows = load_redteam_labels(labels_path)
    train = [load_day_scoring_units(d, seq_root, graph_root, redteam_hours) for d in TRAIN_DAYS]
    y_train = np.concatenate([d["y"] for d in train])
    _, seq_cutoff = compute_best_f1(y_train, np.concatenate([d["seq_scores"] for d in train]))
    _, graph_cutoff = compute_best_f1(y_train, np.concatenate([d["graph_scores"] for d in train]))
    for d in train:
        add_temporal_features(d, seq_cutoff, graph_cutoff)
    features = ["seq_scores", "graph_scores", "temporal_boost", "log_lead_time"]
    clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
    clf.fit(np.column_stack([np.concatenate([d[f] for d in train]) for f in features]), y_train)

    test = [load_day_scoring_units(d, seq_root, graph_root, redteam_hours) for d in TEST_DAYS]
    for d in test:
        add_temporal_features(d, seq_cutoff, graph_cutoff)
        d["logreg_scores"] = clf.predict_proba(np.column_stack([d[f] for f in features]))[:, 1]
    load_final_model_scores(final_scores, test)
    return {
        "test": test,
        "label_rows": label_rows,
        "redteam_hours": redteam_hours,
        "seq_cutoff": seq_cutoff,
        "graph_cutoff": graph_cutoff,
    }


def model_view(d: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray | None]:
    if name == "gru_alone":
        return d["seq_scores"], None
    if name == "logreg_fusion":
        return d["logreg_scores"], None
    return d["final_scores"], d["final_tie_order"]


def campaigns(label_rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[int]]:
    """Labelled red-team event timestamps per (user, day) on the test days."""
    out: dict[tuple[str, int], list[int]] = {}
    for r in label_rows:
        ts = int(r["timestamp"])
        day = (ts - 1) // 86400 + 1
        if day in TEST_DAYS:
            out.setdefault((str(r["user"]), day), []).append(ts)
    return {k: sorted(v) for k, v in sorted(out.items())}


def timing_analysis(days: dict[str, Any], budget: int) -> dict[str, Any]:
    by_day = {d["day"]: d for d in days["test"]}
    camps = campaigns(days["label_rows"])
    redteam_hours = days["redteam_hours"]
    attacked_users = {user for user, _ in camps}
    units_by_user_day: dict[tuple[str, int], dict[int, int]] = {}
    for d in days["test"]:
        for i, (u, w) in enumerate(zip(d["users"], d["windows"])):
            if u in attacked_users:
                units_by_user_day.setdefault((u, d["day"]), {})[int(w)] = i
    results: dict[str, Any] = {}

    for name in COMPARATORS:
        alerts: dict[tuple[str, int], list[int]] = {}
        for d in days["test"]:
            scores, tie = model_view(d, name)
            for i in top_k(scores, budget, tie):
                alerts.setdefault((d["users"][i], d["day"]), []).append(int(d["windows"][i]))

        records = []
        first_alert_on_attack_hour = 0
        campaigns_with_attack_hour_caught = 0
        for (user, day), stamps in camps.items():
            d = by_day[day]
            idx = units_by_user_day.get((user, day), {})
            wins = sorted(alerts.get((user, day), []))
            first_win = wins[0] if wins else None
            offset_s, offset_h, category = calculate_first_alert_offset(stamps[0], first_win)
            if first_win is not None and (user, first_win) in redteam_hours:
                first_alert_on_attack_hour += 1
            if any((user, w) in redteam_hours for w in wins):
                campaigns_with_attack_hour_caught += 1

            # Detector precedence with the Days 08-12 cut-offs: the graph score is
            # available at the start of the day, a GRU score at the end of its hour.
            seq_avail = [w + HOUR for w, i in idx.items() if d["seq_alerts"][i]]
            graph_alert = any(d["graph_alerts"][i] for i in idx.values())
            precedence, lead = determine_detector_precedence(
                min(seq_avail) if seq_avail else None,
                (day - 1) * 86400 + 1 if graph_alert else None,
            )
            i_first = idx.get(first_win) if first_win is not None else None
            records.append(
                AttackCampaignTiming(
                    user_id=user,
                    dataset_day=day,
                    first_attack_timestamp=stamps[0],
                    first_attack_hour_start=hour_start(stamps[0]),
                    total_attack_events=len(stamps),
                    attack_timestamps=stamps,
                    first_alert_available_ts=first_win + HOUR if first_win is not None else None,
                    alert_offset_seconds=offset_s,
                    alert_offset_hours=offset_h,
                    timing_category=category,
                    precedence=precedence,
                    lead_time_seconds=lead,
                    fused_score=float(model_view(d, name)[0][i_first]) if i_first is not None else None,
                    seq_score=float(d["seq_scores"][i_first]) if i_first is not None else None,
                    graph_score=float(d["graph_scores"][i_first]) if i_first is not None else None,
                )
            )
        summary = summarize_timing_distributions(records)
        summary["first_alert_on_attack_hour"] = first_alert_on_attack_hour
        summary["campaigns_with_an_attack_hour_alerted"] = campaigns_with_attack_hour_caught
        results[name] = {"summary": summary, "campaigns": [r.to_dict() for r in records]}
    return results


def error_analysis(days: dict[str, Any], counts_path: Path, budget: int) -> dict[str, Any]:
    """False alarms and misses of the final model at the budget."""
    attacked_user_days = {(u, (w - 1) // 86400 + 1) for u, w in days["redteam_hours"]}
    n_events: dict[tuple[str, int], int] = {}
    if counts_path.exists():
        counts = pq.read_table(
            str(counts_path), columns=["user", "hour", "day", "n_events"], filters=[("day", ">=", TEST_DAYS[0])]
        ).to_pandas()
        attack_keys = set(days["redteam_hours"])
        counts = counts[[k in attack_keys for k in zip(counts["user"], counts["hour"])]]
        n_events = dict(zip(zip(counts["user"], counts["hour"]), counts["n_events"]))

    fp_machine = fp_human = fp_attacked_user_day = 0
    caught_events, missed_events = [], []
    for d in days["test"]:
        top = set(top_k(d["final_scores"], budget, d["final_tie_order"]).tolist())
        for i in top:
            if d["y"][i]:
                continue
            user = d["users"][i]
            if user.split("@")[0].endswith("$"):
                fp_machine += 1
            else:
                fp_human += 1
            if (user, d["day"]) in attacked_user_days:
                fp_attacked_user_day += 1
        for i in np.where(d["y"])[0]:
            key = (d["users"][i], int(d["windows"][i]))
            (caught_events if i in top else missed_events).append(n_events.get(key))

    def describe(values: list[int | None]) -> dict[str, Any]:
        v = np.array([x for x in values if x is not None])
        if not len(v):
            return {"hours": len(values)}
        return {
            "hours": len(values),
            "median_events": float(np.median(v)),
            "hours_with_1_or_2_events": int((v <= 2).sum()),
            "hours_with_over_100_events": int((v > 100).sum()),
        }

    return {
        "false_alarms": {
            "total": fp_machine + fp_human,
            "machine_accounts": fp_machine,
            "human_accounts": fp_human,
            "on_a_user_attacked_the_same_day": fp_attacked_user_day,
        },
        "attack_hours_caught": describe(caught_events),
        "attack_hours_missed": describe(missed_events),
    }


def hour_profile(events: ds.Dataset, user: str, window_start: int, day: int) -> dict[str, Any]:
    t = events.to_table(
        filter=(ds.field("dataset_day") == day)
        & (ds.field("acting_user") == user)
        & (ds.field("timestamp") >= window_start)
        & (ds.field("timestamp") < window_start + HOUR)
    )
    rows = t.to_pylist()
    count = lambda key: dict(Counter(r[key] for r in rows).most_common())  # noqa: E731
    return {
        "events": len(rows),
        "failures": sum(r["authentication_result"] == "Fail" for r in rows),
        "authentication_type": count("authentication_type"),
        "logon_type": count("logon_type"),
        "orientation": count("authentication_orientation"),
        "distinct_sources": len({r["source_computer"] for r in rows}),
        "distinct_destinations": len({r["destination_computer"] for r in rows}),
        "first_time_user_source": sum(bool(r["is_new_user_source"]) for r in rows),
        "first_time_host_connection": sum(bool(r["is_new_host_connection"]) for r in rows),
        "first_time_user_destination": sum(bool(r["is_new_user_destination"]) for r in rows),
        "machine_account": any(bool(r["is_machine_account"]) for r in rows),
    }


def attack_events(events: ds.Dataset, label_rows: list[dict[str, Any]], user: str, window_start: int, day: int) -> list[dict[str, Any]]:
    """Labelled red-team events in the hour, with the auth.txt lines that match them."""
    labels = [
        r for r in label_rows
        if str(r["user"]) == user and window_start <= int(r["timestamp"]) < window_start + HOUR
    ]
    out = []
    for r in sorted(labels, key=lambda x: int(x["timestamp"])):
        t = events.to_table(
            columns=["source_reference", "authentication_type", "logon_type", "authentication_orientation", "authentication_result"],
            filter=(ds.field("dataset_day") == day)
            & (ds.field("timestamp") == int(r["timestamp"]))
            & (ds.field("source_user") == user)
            & (ds.field("source_computer") == r["source_computer"])
            & (ds.field("destination_computer") == r["destination_computer"]),
        ).to_pylist()
        out.append({
            "redteam_reference": r["source_reference"],
            "timestamp": int(r["timestamp"]),
            "source_computer": r["source_computer"],
            "destination_computer": r["destination_computer"],
            "matching_auth_events": t,
        })
    return out


def detector_evidence(seq_root: Path, graph_root: Path, user: str, window_start: int, day: int) -> dict[str, Any]:
    seq = pq.read_table(
        str(seq_root / f"dataset_day={day:02d}" / "part-000000.parquet"),
        columns=["user_id", "window_start", "n_events", "top_events", "top_feature_contributions"],
        filters=[("user_id", "=", user), ("window_start", "=", window_start)],
    ).to_pylist()
    graph = pq.read_table(
        str(graph_root / f"dataset_day={day - 1:02d}" / "part-000000.parquet"),
        columns=["user_id", "window_start", "window_end", "n_edges", "new_edge_count", "top_edges"],
        filters=[("user_id", "=", user)],
    ).to_pylist()
    out: dict[str, Any] = {"gru": None, "graph": None}
    if seq:
        s = seq[0]
        out["gru"] = {
            "events_scored": s["n_events"],
            "top_events": [
                {k: e[k] for k in ("source_reference", "timestamp", "event_error", "top_feature")}
                for e in (s["top_events"] or [])[:3]
            ],
            "top_feature_contributions": (s["top_feature_contributions"] or [])[:3],
        }
    if graph:
        g = graph[0]
        out["graph"] = {
            "snapshot": [g["window_start"], g["window_end"]],
            "edges": g["n_edges"],
            "new_edges": g["new_edge_count"],
            "top_edges": [
                {
                    "destination_computer": e["destination_computer"],
                    "edge_anomaly": e["edge_raw_score"],
                    "is_new_edge": e["is_new_edge"],
                    "auth_references": [f"auth.txt:{line}" for line in (e["source_lines"] or [])[:3]],
                }
                for e in (g["top_edges"] or [])[:3]
            ],
        }
    return out


def pick_cases(days: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    candidates = {"tp": [], "fp": [], "fn": []}
    for d in days["test"]:
        ranks = daily_ranks(d["final_scores"], d["final_tie_order"])
        seq_ranks = daily_ranks(d["seq_scores"], None)
        for i, (rank, attack) in enumerate(zip(ranks, d["y"])):
            entry = (int(rank), d, i, int(seq_ranks[i]))
            if rank <= budget:
                candidates["tp" if attack else "fp"].append(entry)
            elif attack:
                candidates["fn"].append(entry)

    def best(entries):  # highest score across days; ties by daily rank then day
        return max(entries, key=lambda e: (e[1]["final_scores"][e[2]], -e[0], -e[1]["day"]))

    missed = sorted(candidates["fn"], key=lambda e: (e[0], e[1]["day"]))
    chosen = [
        ("true_positive", "Highest-ranked caught attack hour", best(candidates["tp"])),
        ("false_positive", "Highest-ranked false alarm", best(candidates["fp"])),
        ("false_negative", "Typical missed attack hour (median rank among misses)", missed[len(missed) // 2]),
    ]
    cases = []
    for case_type, rule, (rank, d, i, seq_rank) in chosen:
        cases.append({
            "case_type": case_type,
            "selection_rule": rule,
            "user_id": d["users"][i],
            "dataset_day": d["day"],
            "window": [int(d["windows"][i]), int(d["windows"][i]) + HOUR],
            "is_labelled_attack_hour": bool(d["y"][i]),
            "final_model": {"score": float(d["final_scores"][i]), "daily_rank": rank, "units_that_day": len(d["users"])},
            "gru": {"score": float(d["seq_scores"][i]), "daily_rank": seq_rank, "alert_at_train_cutoff": bool(d["seq_alerts"][i])},
            "graph": {"score_previous_day": float(d["graph_scores"][i]), "alert_at_train_cutoff": bool(d["graph_alerts"][i])},
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-scores-dir", type=Path, default=Path("outputs/sequence_scores/seq-gru-ae-v1-L32-h32-91e4b11d34/scores"))
    parser.add_argument("--graph-scores-dir", type=Path, default=Path("outputs/graph_scores_v1/scores"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed/lanl_auth_days_01_30/redteam_labels/labels"))
    parser.add_argument("--events-dir", type=Path, default=Path("data/processed/lanl_features_v2_days_01_16/raw/events"),
                        help="Feature-build events (auth fields, first-time flags and auth.txt references)")
    parser.add_argument("--hourly-counts", type=Path, default=Path("outputs/experiment_v2/fusion_counts.parquet"))
    parser.add_argument("--final-model-scores", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=38)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/timing_and_errors.json"))
    args = parser.parse_args()

    days = load_days(args.seq_scores_dir, args.graph_scores_dir, args.labels_dir, args.final_model_scores)
    timing = timing_analysis(days, args.budget)
    errors = error_analysis(days, args.hourly_counts, args.budget)

    events = ds.dataset(str(args.events_dir), format="parquet", partitioning="hive")
    cases = pick_cases(days, args.budget)
    for case in cases:
        user, day, start = case["user_id"], case["dataset_day"], case["window"][0]
        case["hour_profile"] = hour_profile(events, user, start, day)
        case["labelled_attack_events"] = attack_events(events, days["label_rows"], user, start, day)
        case["detector_evidence"] = detector_evidence(args.seq_scores_dir, args.graph_scores_dir, user, start, day)
        if case["is_labelled_attack_hour"] and not case["labelled_attack_events"]:
            raise SystemExit(f"attack hour {user} {start} has no labelled events")

    report = {
        "protocol": "final model and LogReg fitted on Days 08-12; analysed on Days 13-16; Days 17-30 not read",
        "budget_per_day": args.budget,
        "cutoffs_fitted_on_train_days": {"sequence_alert": days["seq_cutoff"], "graph_alert": days["graph_cutoff"]},
        "timing": {name: r["summary"] for name, r in timing.items()},
        "final_model_errors": errors,
        "case_studies": cases,
        "campaigns_final_model": timing[FINAL]["campaigns"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Cut-offs (Days 08-12, best F1): sequence >= {days['seq_cutoff']:.6f}, graph >= {days['graph_cutoff']:.6f}")
    print(f"\n{'model':<26} {'campaigns':>9} {'any alert':>9} {'attack hr':>9} {'early':>6} {'immed':>6} {'delay':>6} {'median h':>9}")
    for name, r in timing.items():
        s = r["summary"]
        c = s["timing_categories"]
        med = (s["offset_hours_stats"] or {}).get("median")
        print(f"{name:<26} {s['total_campaigns']:>9} {s['detected_campaigns']:>9} {s['campaigns_with_an_attack_hour_alerted']:>9} "
              f"{c[TimingCategory.EARLY_WARNING.value]:>6} {c[TimingCategory.IMMEDIATE.value]:>6} {c[TimingCategory.DELAYED.value]:>6} {str(med):>9}")
    print("\nDetector precedence (all campaigns, Days 08-12 cut-offs):", timing[FINAL]["summary"]["precedence_distribution"])
    print("\nFinal model errors:", json.dumps(errors, indent=2))
    for c in cases:
        p = c["hour_profile"]
        print(f"\n[{c['case_type']}] {c['selection_rule']}: {c['user_id']} day {c['dataset_day']} window {c['window']}")
        print(f"  final score {c['final_model']['score']:.4f} rank {c['final_model']['daily_rank']} | "
              f"GRU {c['gru']['score']:.4f} rank {c['gru']['daily_rank']} | graph(prev day) {c['graph']['score_previous_day']:.4f}")
        print(f"  hour: {p['events']} events, {p['failures']} failures, types {p['authentication_type']}, logon {p['logon_type']}, "
              f"{p['distinct_sources']} sources, {p['distinct_destinations']} destinations, first-time user->source {p['first_time_user_source']}, "
              f"host->host {p['first_time_host_connection']}, machine {p['machine_account']}")
        for e in c["labelled_attack_events"][:5]:
            refs = [m["source_reference"] for m in e["matching_auth_events"]]
            print(f"  attack {e['redteam_reference']} {e['source_computer']}->{e['destination_computer']} at {e['timestamp']}: {refs}")
    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
