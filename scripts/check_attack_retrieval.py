#!/usr/bin/env python3
"""Run the documented ATT&CK retrieval samples and record their results (Task 6.1).

Part 1 runs the fixed sample queries in docs/attack_retrieval.md and fails if a
query no longer returns its expected top technique (or, for the
insufficient-evidence samples, returns anything). Part 2, when the alert
handoff package exists, applies the behaviour templates to a few real days
17-30 incidents: the red-team incident, the first incident of each behaviour
combination and an incident with no matching behaviour.

Example:
  python scripts/check_attack_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.attack.queries import TEMPLATES, retrieve_candidates  # noqa: E402
from dualscope.attack.retrieval import DEFAULT_CATALOG, TechniqueRetriever  # noqa: E402

# (name, query, expected top technique or None for "nothing returned")
SAMPLE_QUERIES = [
    *[(t.behaviour, t.query, top) for t, top in zip(TEMPLATES, ["T1110.001", "T1550.002", "T1021.002", "T1078"])],
    ("kerberos_tickets", "steal or forge kerberos tickets many kerberos service ticket requests", "T1558"),
    ("insufficient_routine_logon", "user logged on during the working day", None),
    ("insufficient_machine_account", "computer account authenticating to many hosts", None),
    ("insufficient_unrelated", "quarterly marketing budget spreadsheet", None),
]


def incident_samples(handoff_dir: Path, retriever: TechniqueRetriever) -> list[dict]:
    events = pd.read_parquet(handoff_dir / "events.parquet")
    alerts = pd.read_parquet(handoff_dir / "alerts.parquet", columns=["alert_id", "incident_id", "ground_truth_redteam"])
    events = events.merge(alerts[["alert_id", "incident_id"]], on="alert_id", validate="m:1")
    redteam = set(alerts.loc[alerts["ground_truth_redteam"], "incident_id"])
    chosen: dict[str, str] = {}
    results = {}
    for incident_id, rows in events.groupby("incident_id", sort=True):
        result = retrieve_candidates(rows, retriever)
        combination = "+".join(b["behaviour"] for b in result["behaviours"]) or "none"
        if incident_id in redteam:
            chosen[incident_id] = "red-team incident"
        elif combination not in chosen.values():
            chosen[incident_id] = combination
        results[incident_id] = result
    samples = []
    for incident_id, reason in chosen.items():
        result = results[incident_id]
        samples.append({
            "incident_id": incident_id,
            "selected_as": reason,
            "behaviours": result["behaviours"],
            "insufficient_evidence": result["insufficient_evidence"],
            "candidates": [
                {key: c[key] for key in ("technique_id", "name", "score", "url", "retrieved_by")}
                for c in result["candidates"]
            ],
        })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--handoff-dir", type=Path, default=REPO_ROOT / "outputs/handoff/final_test_alerts_v1")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/attack_retrieval/sample_checks.json")
    args = parser.parse_args()

    retriever = TechniqueRetriever.from_file(args.catalog)
    failures = []
    queries = []
    for name, query, expected in SAMPLE_QUERIES:
        results = retriever.search(query, k=5, auth_observable_only=True)
        top = results[0].technique_id if results else None
        if top != expected:
            failures.append(f"{name}: expected {expected}, got {top}")
        queries.append({
            "name": name, "query": query, "expected_top": expected,
            "results": [{key: r.to_dict()[key] for key in ("technique_id", "name", "score", "url")} for r in results],
        })
        print(f"{name:<34} {' | '.join(f'{r.technique_id} {r.name} ({r.score:.3f})' for r in results) or '(no technique above the cut-off)'}")

    record = {"snapshot": retriever.catalog.snapshot, "sample_queries": queries}
    if (args.handoff_dir / "events.parquet").exists():
        record["incident_samples"] = incident_samples(args.handoff_dir, retriever)
        for sample in record["incident_samples"]:
            print(f"\n{sample['incident_id']} ({sample['selected_as']})")
            for behaviour in sample["behaviours"]:
                print(f"  {behaviour['behaviour']}: {behaviour['event_count']} events, e.g. {behaviour['evidence_references'][0]}")
            print("  candidates: " + (", ".join(f"{c['technique_id']} ({c['score']:.3f})" for c in sample["candidates"]) or "none (insufficient evidence)"))
    else:
        print(f"\n{args.handoff_dir} not found; incident samples skipped")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    if failures:
        print("Sample query changes:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
