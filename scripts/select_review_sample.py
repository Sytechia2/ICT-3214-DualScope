#!/usr/bin/env python3
"""Pick the incident sample that reviewers label for Task 9.5.

Reviewers write reference ATT&CK labels for these incidents before they see
any LLM output; the three investigation modes are later scored against them.
The sample is chosen from detector and evidence fields only (priority, graph
detector agreement, the Task 6.1 behaviour rules, event count, dataset day).
No LLM output is used. One red-team incident is force-included so the sample
contains a true positive (this is not written to the sample file or the review pack, so reviewers
are not primed to look for it); the answer key is read for that inclusion only and
is never written to the sample, the strata or the log.

Method: force-include, then greedily fill the minimum counts in
``DEFAULT_TARGETS`` (rarest unmet target first, random pick with a fixed
seed), then fill at random up to the sample size. No dataset day may hold more
than ``--max-per-day`` incidents. The script exits non-zero if a target
cannot be met.

Output: data/reference/evaluation/investigation_review_sample_v1.json
        {"version", "created_utc", "seed", "method", "incidents": [{incident_id, strata}]}
        sorted by incident_id.

Example:
  python scripts/select_review_sample.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.attack.queries import observed_behaviours  # noqa: E402

DEFAULT_HANDOFF = REPO_ROOT / "outputs" / "handoff" / "final_test_alerts_v1"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reference" / "evaluation" / "investigation_review_sample_v1.json"
VERSION = "investigation_review_sample_v1"
SAMPLE_SIZE = 25
MAX_PER_DAY = 3
BEHAVIOURS = ("failed_logons", "ntlm_logon_to_new_destination", "network_logon_to_new_destination",
              "logon_from_new_source")
LARGE_EVENTS = 1000
SMALL_EVENTS = 60

# target name -> minimum number of sampled incidents with that strata value
DEFAULT_TARGETS = {
    "priority_medium": 7,
    "graph_agrees": 4,
    "no_rule_match": 5,
    "failed_logons": 4,
    "ntlm_logon_to_new_destination": 6,
    "network_logon_to_new_destination": 6,
    "logon_from_new_source": 5,
    "large_over_1000_events": 3,
    "small_under_60_events": 5,
}


def incident_strata(incident: dict, events: pd.DataFrame) -> dict:
    """Strata from detector/evidence fields only."""
    behaviours = [b["behaviour"] for b in observed_behaviours(events)]
    n = len(events)
    graph = (incident.get("detector_scores") or {}).get("graph") or {}
    return {
        "priority": incident.get("priority"),
        "graph_agrees": bool(graph.get("any_alert")),
        "behaviours": behaviours,
        "no_rule_match": not behaviours,
        "events": n,
        "size": "large_over_1000_events" if n > LARGE_EVENTS else "small_under_60_events" if n < SMALL_EVENTS else "medium",
        "dataset_day": int(incident["dataset_day"]),
    }


def has_target(strata: dict, target: str) -> bool:
    if target == "priority_medium":
        return strata["priority"] == "MEDIUM"
    if target in ("graph_agrees", "no_rule_match"):
        return bool(strata[target])
    if target in BEHAVIOURS:
        return target in strata["behaviours"]
    return strata["size"] == target


def counts(chosen: list[str], strata: dict[str, dict], targets: dict[str, int]) -> dict[str, int]:
    return {t: sum(has_target(strata[i], t) for i in chosen) for t in targets}


def select(strata: dict[str, dict], forced: list[str], size: int, targets: dict[str, int],
           max_per_day: int, seed: int = 0) -> list[str]:
    """Deterministic selection; raises ValueError when a target cannot be met."""
    rng = random.Random(seed)
    chosen = list(dict.fromkeys(forced))
    days = Counter(strata[i]["dataset_day"] for i in chosen)

    def available(extra_filter=None) -> list[str]:
        return sorted(i for i in strata if i not in chosen and days[strata[i]["dataset_day"]] < max_per_day
                      and (extra_filter is None or extra_filter(strata[i])))

    def add(incident_id: str) -> None:
        chosen.append(incident_id)
        days[strata[incident_id]["dataset_day"]] += 1

    while len(chosen) < size:
        have = counts(chosen, strata, targets)
        unmet = [t for t in targets if have[t] < targets[t]]
        options = {t: available(lambda s, t=t: has_target(s, t)) for t in unmet}
        unmet = [t for t in unmet if options[t]]
        if not unmet:
            break
        rarest = min(unmet, key=lambda t: (len(options[t]), t))
        add(rng.choice(options[rarest]))
    while len(chosen) < size:
        pool = available()
        if not pool:
            break
        add(rng.choice(pool))
    final = counts(chosen, strata, targets)
    missing = {t: (final[t], targets[t]) for t in targets if final[t] < targets[t]}
    if missing or len(chosen) != size:
        raise ValueError(f"targets not met (have, need): {missing}; sample size {len(chosen)}/{size}")
    return sorted(chosen)


def load_incidents_and_strata(handoff: Path) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    incidents = {}
    for line in (handoff / "incidents.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            incidents[record["incident_id"]] = record
    alerts = pd.read_parquet(handoff / "alerts.parquet", columns=["alert_id", "incident_id"])
    events = pd.read_parquet(handoff / "events.parquet").merge(alerts, on="alert_id", validate="m:1")
    by_incident = dict(tuple(events.groupby("incident_id", sort=True)))
    strata = {i: incident_strata(rec, by_incident[i]) for i, rec in incidents.items() if i in by_incident}
    forced = sorted(i for i in strata if incidents[i].get("ground_truth_redteam"))  # the only use of the key
    return incidents, strata, forced


def summary_table(chosen: list[str], strata: dict[str, dict], targets: dict[str, int]) -> str:
    have = counts(chosen, strata, targets)
    lines = [f"{'stratum':38s}{'sampled':>8s}{'minimum':>8s}"]
    lines += [f"{t:38s}{have[t]:8d}{targets[t]:8d}" for t in targets]
    lines.append(f"{'priority HIGH':38s}{sum(strata[i]['priority'] == 'HIGH' for i in chosen):8d}")
    days = Counter(strata[i]["dataset_day"] for i in chosen)
    lines.append("per day: " + ", ".join(f"{d}:{n}" for d, n in sorted(days.items())))
    return "\n".join(lines)


METHOD = ("Incidents are chosen from detector and evidence fields only (priority, graph detector agreement, "
          "Task 6.1 behaviour rules, event count, dataset day); no LLM output is used. Incidents are "
          "filled at random (seed {seed}) until each minimum count is met, then up to {size} incidents, with at most {cap} per dataset day.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--max-per-day", type=int, default=MAX_PER_DAY)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, strata, forced = load_incidents_and_strata(args.handoff)
    try:
        chosen = select(strata, forced, args.size, DEFAULT_TARGETS, args.max_per_day, args.seed)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    document = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "method": METHOD.format(seed=args.seed, size=args.size, cap=args.max_per_day),
        "incidents": [{"incident_id": i, "strata": strata[i]} for i in chosen],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(summary_table(chosen, strata, DEFAULT_TARGETS))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
