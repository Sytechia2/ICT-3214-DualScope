#!/usr/bin/env python3
"""Generate and verify investigation summaries in three modes (Tasks 6.2-6.4).

For every incident in the alert handoff it builds the evidence package
(dualscope.investigation.package), retrieves ATT&CK candidates (Task 6.1) and
asks Gemini on Vertex AI for a structured summary in two generation modes:

  direct        evidence package only; techniques from the model's own knowledge
  rag           evidence package + retrieved ATT&CK candidates
  rag_verified  the saved rag outputs after deterministic verification (6.3);
                no new generation, so verification is the only difference

Direct outputs are verified too, for the evaluation, but only the rag ones form
a mode. All modes use the same incidents, model, prompts and settings; the run
refuses to continue in an output folder made with different ones.

Generation resumes: incidents that already have a reply are skipped, so an
interrupted run can be restarted. ``--retry-failed`` retries the failed and
invalid ones. ``--verify-only`` reruns verification on saved replies.

Outputs (in --output-dir):
  config.json             model, settings, prompt hashes, input hashes
  packages.jsonl          evidence package + retrieval result per incident
  prompts_<mode>.jsonl    the user prompt sent per incident
  generated_<mode>.jsonl  model replies (status ok / invalid / failed)
  verification.jsonl      verifier decisions for direct and rag replies
  rag_verified.jsonl      the verified rag summaries (the dashboard's input)
  summary.json            counts per mode for the evaluation
  system_prompt.txt, response_schema.json

The answer key (ground_truth_redteam) never enters a package or a prompt.

Examples:
  python scripts/run_investigations.py --dry-run
  python scripts/run_investigations.py --incidents INC-TEST-D28-U737_DOM1-003
  python scripts/run_investigations.py --workers 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.attack.queries import retrieve_candidates  # noqa: E402
from dualscope.attack.retrieval import TechniqueRetriever  # noqa: E402
from dualscope.investigation.gemini import GeminiClient, GeminiConfig  # noqa: E402
from dualscope.investigation.generate import investigate  # noqa: E402
from dualscope.investigation.package import (  # noqa: E402
    MAX_FLAGGED_EVENTS,
    MAX_PACKAGE_EVENTS,
    assert_no_answer_key,
    build_package,
)
from dualscope.investigation.prompts import (  # noqa: E402
    MODES,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    prompt_fingerprint,
    user_prompt,
)
from dualscope.investigation.verify import apply_verification, verify_response  # noqa: E402

DEFAULT_HANDOFF = REPO_ROOT / "outputs" / "handoff" / "final_test_alerts_v1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "investigations" / "gemini_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    tmp.replace(path)


def latest_by_incident(records: list[dict]) -> dict[str, dict]:
    """The last record per incident (a retry replaces an earlier failure)."""
    return {record["incident_id"]: record for record in records}


def load_inputs(handoff: Path, wanted: list[str] | None, limit: int | None):
    incidents = {record["incident_id"]: record for record in read_jsonl(handoff / "incidents.jsonl")}
    ids = sorted(incidents)
    if wanted:
        unknown = sorted(set(wanted) - set(incidents))
        if unknown:
            raise SystemExit(f"unknown incident IDs: {', '.join(unknown)}")
        ids = [i for i in ids if i in set(wanted)]
    if limit:
        ids = ids[:limit]
    alerts = pd.read_parquet(handoff / "alerts.parquet", columns=["alert_id", "incident_id"])
    events = pd.read_parquet(handoff / "events.parquet").merge(alerts, on="alert_id", validate="m:1")
    events = events[events["incident_id"].isin(ids)]
    return incidents, ids, dict(tuple(events.groupby("incident_id", sort=True)))


def build_packages(incidents, ids, events_by_incident, retriever) -> dict[str, dict]:
    built = {}
    for incident_id in ids:
        rows = events_by_incident[incident_id].drop(columns=["incident_id"])
        package = build_package(incidents[incident_id], rows)
        retrieval = retrieve_candidates(rows, retriever)
        assert_no_answer_key(retrieval, "retrieval")
        built[incident_id] = {"incident_id": incident_id, "package": package, "retrieval": retrieval}
    return built


def run_config(client_settings: dict, handoff: Path, retriever: TechniqueRetriever) -> dict:
    return {
        "model": client_settings["model"],
        "settings": client_settings,
        "prompts": prompt_fingerprint(),
        "package_rule": {"max_events": MAX_PACKAGE_EVENTS, "max_rule_matched_events": MAX_FLAGGED_EVENTS},
        "handoff_manifest_sha256": sha256(handoff / "manifest.json"),
        "incidents_sha256": sha256(handoff / "incidents.jsonl"),
        "attack_snapshot": {"attack_version": retriever.catalog.snapshot["attack_version"],
                            "bundle_sha256": retriever.catalog.snapshot["bundle_sha256"]},
    }


def check_config(output: Path, config: dict) -> None:
    path = output / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        differs = sorted(key for key in config if saved.get(key) != config[key])
        if differs:
            raise SystemExit(f"{output} was made with a different {', '.join(differs)}; "
                             "use another --output-dir so runs are not mixed")
    else:
        path.write_text(json.dumps({**config, "created_utc": _now()}, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_mode(mode, packages, output, client, settings, workers, retry_failed) -> None:
    out_path = output / f"generated_{mode}.jsonl"
    prompt_path = output / f"prompts_{mode}.jsonl"
    existing = latest_by_incident(read_jsonl(out_path))
    todo = [i for i in packages if i not in existing or (retry_failed and existing[i]["status"] != "ok")]
    print(f"[{mode}] {len(todo)} to generate, {len(packages) - len(todo)} already done", flush=True)
    if not todo:
        return
    lock = threading.Lock()
    started = time.time()
    done = 0
    with out_path.open("a", encoding="utf-8") as out, prompt_path.open("a", encoding="utf-8") as prompts, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(investigate, mode, packages[i]["package"], packages[i]["retrieval"], client, settings): i
                   for i in todo}
        for future in as_completed(futures):
            record, prompt = future.result()
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
                prompts.write(json.dumps({"incident_id": record["incident_id"], "prompt_sha256": record["prompt_sha256"],
                                          "user_prompt": prompt}) + "\n")
                prompts.flush()
                done += 1
                if record["status"] != "ok" or done % 10 == 0 or done == len(todo):
                    note = "" if record["status"] == "ok" else f" {record['status']}: {record['errors'][:1]}"
                    print(f"[{mode}] {done}/{len(todo)} ({time.time() - started:.0f}s) {record['incident_id']}{note}",
                          flush=True)
    # Keep one record per incident, in incident order.
    latest = latest_by_incident(read_jsonl(out_path))
    write_jsonl(out_path, [latest[i] for i in sorted(latest)])
    prompts_latest = latest_by_incident(read_jsonl(prompt_path))
    write_jsonl(prompt_path, [prompts_latest[i] for i in sorted(prompts_latest)])


def verify_all(packages, output, catalog) -> dict:
    verification, rag_verified = [], []
    generated = {mode: latest_by_incident(read_jsonl(output / f"generated_{mode}.jsonl")) for mode in MODES}
    for mode in MODES:
        for incident_id, record in sorted(generated[mode].items()):
            if incident_id not in packages or record["status"] != "ok":
                continue
            checks = verify_response(record["response"], packages[incident_id]["package"], catalog,
                                     record["retrieved_candidates"])
            verification.append({"incident_id": incident_id, "mode": mode, "checks": checks,
                                 "verified": apply_verification(record, checks)})
    by_rag = {v["incident_id"]: v for v in verification if v["mode"] == "rag"}
    for incident_id, record in sorted(generated["rag"].items()):
        if incident_id not in packages:
            continue
        rag_verified.append({
            "incident_id": incident_id,
            "mode": "rag_verified",
            "source_mode": "rag",
            "status": record["status"],
            "errors": record.get("errors", []),
            "model": record.get("model"),
            "model_version": record.get("model_version"),
            "created_utc": record.get("created_utc"),
            "retrieved_candidates": record.get("retrieved_candidates"),
            "verified": by_rag[incident_id]["verified"] if incident_id in by_rag else None,
        })
    write_jsonl(output / "verification.jsonl", verification)
    write_jsonl(output / "rag_verified.jsonl", rag_verified)
    return summarise(generated, verification, packages)


def summarise(generated, verification, packages) -> dict:
    summary = {"incidents": len(packages), "modes": {}}
    for mode in MODES:
        records = [r for i, r in generated[mode].items() if i in packages]
        ok = [r for r in records if r["status"] == "ok"]
        checks = [v for v in verification if v["mode"] == mode]
        technique_status = Counter(t["status"] for v in checks for t in v["checks"]["techniques"])
        claims = Counter(c["status"] for v in checks for section in ("observations", "interpretations")
                         for c in v["checks"][section])
        summary["modes"][mode] = {
            "replies": len(records),
            "status": dict(Counter(r["status"] for r in records)),
            "with_techniques": sum(bool(r["response"]["techniques"]) for r in ok),
            "model_no_supported_mapping": sum(r["response"]["no_supported_mapping"] for r in ok),
            "techniques_mapped": sum(len(r["response"]["techniques"]) for r in ok),
            "technique_ids": dict(Counter(t["technique_id"] for r in ok for t in r["response"]["techniques"]).most_common()),
            "verifier_status": dict(technique_status),
            "claim_status": dict(claims),
            "tokens": {key: sum(r.get("usage", {}).get(key, 0) for r in records)
                       for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")},
            "mean_latency_seconds": round(sum(r.get("latency_seconds", 0) for r in ok) / len(ok), 1) if ok else None,
        }
    rag = [v["verified"] for v in verification if v["mode"] == "rag"]
    summary["modes"]["rag_verified"] = {
        "summaries": len(rag),
        "outcome": dict(Counter(v["outcome"] for v in rag)),
        "techniques_kept": dict(Counter(t["status"] for v in rag for t in v["techniques"])),
        "techniques_removed": sum(len(v["removed_techniques"]) for v in rag),
        "removed_technique_ids": dict(Counter(t["technique_id"] for v in rag for t in v["removed_techniques"]).most_common()),
        "claims_removed": sum(len(v["removed_observations"]) + len(v["removed_interpretations"]) for v in rag),
        "emptied_by_verification": sum(bool(v["removed_techniques"]) and not v["techniques"] for v in rag),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--incidents", nargs="+", help="only these incident IDs")
    parser.add_argument("--incidents-file", type=Path, help="text file with one incident ID per line")
    parser.add_argument("--limit", type=int, help="only the first N incidents (by ID)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default=GeminiConfig.model)
    parser.add_argument("--thinking-level", default=GeminiConfig.thinking_level,
                        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--dry-run", action="store_true", help="build packages and prompts only; no model calls")
    parser.add_argument("--verify-only", action="store_true", help="rerun verification on saved replies")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed and invalid replies")
    args = parser.parse_args()

    wanted = list(args.incidents or [])
    if args.incidents_file:
        wanted += [line.strip() for line in args.incidents_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    retriever = TechniqueRetriever.from_file()
    incidents, ids, events_by_incident = load_inputs(args.handoff_dir, wanted or None, args.limit)
    print(f"{len(ids)} incidents; building evidence packages", flush=True)
    packages = build_packages(incidents, ids, events_by_incident, retriever)

    if args.dry_run:
        dry = output / "dry_run"
        dry.mkdir(exist_ok=True)
        write_jsonl(dry / "packages.jsonl", [packages[i] for i in ids])
        for mode in args.modes:
            write_jsonl(dry / f"prompts_{mode}.jsonl", [
                {"incident_id": i, "user_prompt": user_prompt(mode, packages[i]["package"], packages[i]["retrieval"])}
                for i in ids])
        sizes = [len(user_prompt("rag", packages[i]["package"], packages[i]["retrieval"])) for i in ids]
        print(f"dry run written to {dry}; RAG prompt characters: median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")
        return

    config = GeminiConfig(model=args.model, thinking_level=args.thinking_level)
    check_config(output, run_config(config.settings(), args.handoff_dir, retriever))
    (output / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (output / "response_schema.json").write_text(json.dumps(RESPONSE_SCHEMA, indent=2), encoding="utf-8")
    saved = latest_by_incident(read_jsonl(output / "packages.jsonl"))
    saved.update(packages)
    write_jsonl(output / "packages.jsonl", [saved[i] for i in sorted(saved)])

    if not args.verify_only:
        client = GeminiClient(config)
        for mode in args.modes:
            generate_mode(mode, packages, output, client, config.settings(), args.workers, args.retry_failed)

    summary = verify_all(saved, output, retriever.catalog)
    summary["updated_utc"] = _now()
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["modes"], indent=1))


if __name__ == "__main__":
    main()
