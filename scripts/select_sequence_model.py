"""Task 3.3: select sequence-model settings on validation data, calibrate and freeze.

For every (maximum sequence length, model size) in the bounded search space,
train a model (Task 3.2) or reuse a completed run with the same configuration
fingerprint, then score every available validation user-hour. Each
(trial, aggregation) is evaluated against validation red-team user-hours.

Selection rule (fixed before any test data is scored):
  1. highest validation average precision over scorable user-hours;
  2. ties -> fewer parameters, then shorter maximum length, then the
     configured aggregation order.

The selected model's label-free validation raw scores fit the 0-1 score
calibration; the alert threshold maximises validation F1 on calibrated scores.
Everything is written to a frozen detector directory and a tracked manifest.
Test labels are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.sequence.calibration import (  # noqa: E402
    QuantileTailCalibrator,
    evaluation_summary,
    load_positive_user_hours,
    unit_labels,
)
from dualscope.sequence.config import SequenceDetectorConfig  # noqa: E402
from dualscope.sequence.detector import FROZEN_FILENAME, freeze_detector  # noqa: E402
from dualscope.sequence.pipeline import git_revision, load_inputs, prepare_samples, score_split, train_trial  # noqa: E402
from dualscope.sequence.training import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "data/processed/lanl_features_days_01_30")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--sequence-config", type=Path, default=REPO_ROOT / "config/sequence_detector.json")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "models/sequence/runs")
    parser.add_argument("--selection-dir", type=Path, default=REPO_ROOT / "outputs/sequence/selection")
    parser.add_argument("--frozen-root", type=Path, default=REPO_ROOT / "models/sequence/frozen")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/manifests/sequence_detector_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/processed/sequence_cache")
    parser.add_argument("--train-sequences", type=int, default=None, help="Override (development only).")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override (development only).")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--no-reuse", action="store_true", help="Retrain even when a matching run exists.")
    parser.add_argument("--allow-pilot-features", action="store_true")
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    base = SequenceDetectorConfig.from_file(args.sequence_config)
    overrides = {
        key: value
        for key, value in {
            "train_sequences": args.train_sequences,
            "max_epochs": args.max_epochs,
            "torch_threads": args.torch_threads,
        }.items()
        if value is not None
    }
    if overrides:
        base = replace(base, training=replace(base.training, **overrides))
    inputs = load_inputs(
        args.features_root, args.splits_config, args.splits_manifest, args.feature_config,
        labels_dir=args.labels_dir, allow_pilot_features=args.allow_pilot_features,
    )
    positives = load_positive_user_hours(args.labels_dir, inputs.split_cfg, "validation", base.policy.hour_seconds)
    print(f"Validation positive user-hours: {len(positives)}")
    args.selection_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.selection_dir / "trials.jsonl"
    trials_path.write_text("", encoding="utf-8")

    trials: list[dict] = []
    validation_scores: dict[str, dict] = {}
    started_all = time.time()
    size_names = list(base.search.model_sizes)
    for length in base.search.max_sequence_lengths:
        length_cfg = base.with_trial(length, size_names[0])
        train, monitor, stats = prepare_samples(inputs, length_cfg, args.cache_dir)
        models, run_ids, manifests = {}, {}, {}
        for size in size_names:
            config = base.with_trial(length, size)
            run_id = f"seq-L{length}-{size}"
            run_dir = args.runs_dir / run_id
            manifest_path = run_dir / "run_manifest.json"
            reuse = False
            if manifest_path.is_file() and not args.no_reuse:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                reuse = (
                    existing.get("sequence_config_fingerprint") == config.fingerprint()
                    and existing.get("feature", {}).get("preprocessing_sha256") == inputs.feature_info()["preprocessing_sha256"]
                )
            if reuse:
                print(f"Reusing completed run {run_id}")
                manifest = existing
            else:
                print(f"Training {run_id}")
                try:
                    manifest = train_trial(inputs, config, train, monitor, stats, run_dir, run_id)
                except Exception as exc:  # record unsuccessful runs instead of hiding them
                    failure = {"run_id": run_id, "max_sequence_length": length, "model_size": size, "status": "failed", "error": repr(exc)}
                    trials.append(failure)
                    with open(trials_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure) + "\n")
                    print(f"  FAILED: {exc!r}")
                    continue
            model, _ = load_checkpoint(run_dir / "checkpoint.pt", manifest["checkpoint"]["sha256"])
            models[size], run_ids[size], manifests[size] = model, run_id, manifest
        if not models:
            continue

        print(f"Scoring validation split with L={length}: {sorted(models)}")
        scored = score_split(inputs, length_cfg.policy, models, "validation")
        labels = unit_labels(scored["users"], scored["hours"], positives)
        for size, model in models.items():
            run_id = run_ids[size]
            raw_by_agg = scored["raw"][size]
            table = {
                "user_id": pa.array(scored["users"], pa.string()),
                "window_start": pa.array(scored["hours"], pa.int64()),
                "dataset_day": pa.array(scored["days"], pa.int32()),
                "n_events": pa.array(scored["n_events"], pa.int32()),
            }
            table.update({f"raw_{agg}": pa.array(values, pa.float64()) for agg, values in raw_by_agg.items()})
            pq.write_table(pa.table(table), args.selection_dir / f"{run_id}_validation_raw.parquet", compression="zstd")
            validation_scores[run_id] = raw_by_agg
            for aggregation in base.search.aggregations:
                summary = evaluation_summary(labels, raw_by_agg[aggregation], len(positives))
                trial = {
                    "run_id": run_id,
                    "status": "completed",
                    "max_sequence_length": length,
                    "model_size": size,
                    "model_settings": base.search.model_sizes[size],
                    "aggregation": aggregation,
                    "parameters": manifests[size]["parameters"],
                    "sequence_config_fingerprint": manifests[size]["sequence_config_fingerprint"],
                    "checkpoint_sha256": manifests[size]["checkpoint"]["sha256"],
                    "training_seconds": manifests[size]["training_seconds"],
                    "best_monitor_loss": min(h["monitor_loss"] for h in manifests[size]["loss_history"]),
                    "validation": summary,
                }
                trials.append(trial)
                with open(trials_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_jsonable(trial)) + "\n")
                print(
                    f"  {run_id} {aggregation}: AP={summary['average_precision']:.5f} "
                    f"bestF1={summary['at_threshold']['f1']:.4f} coverage={summary['positive_coverage']:.3f}"
                )

    completed = [t for t in trials if t["status"] == "completed" and t["validation"]["average_precision"] is not None]
    if not completed:
        raise SystemExit("no completed trial to select")
    aggregation_order = {name: i for i, name in enumerate(base.search.aggregations)}
    selected = min(
        completed,
        key=lambda t: (
            -t["validation"]["average_precision"],
            t["parameters"],
            t["max_sequence_length"],
            aggregation_order[t["aggregation"]],
        ),
    )
    print(f"Selected {selected['run_id']} with {selected['aggregation']} (AP={selected['validation']['average_precision']:.5f})")

    final_cfg = replace(
        base.with_trial(selected["max_sequence_length"], selected["model_size"]), aggregation=selected["aggregation"]
    )
    reference = validation_scores[selected["run_id"]][selected["aggregation"]]
    calibrator = QuantileTailCalibrator.fit(
        reference,
        reference_description=(
            f"all {len(reference):,} available validation user-hours (days {inputs.days_in('validation')[0]}-"
            f"{inputs.days_in('validation')[-1]}), label-free, {selected['aggregation']} raw scores"
        ),
    )
    normalised = calibrator.transform(reference)
    scored_users = pq.read_table(args.selection_dir / f"{selected['run_id']}_validation_raw.parquet", columns=["user_id", "window_start", "dataset_day"])
    labels = unit_labels(
        scored_users["user_id"].to_numpy(zero_copy_only=False), scored_users["window_start"].to_numpy(), positives
    )
    calibrated = evaluation_summary(labels, normalised, len(positives))
    threshold = calibrated["at_threshold"]["threshold"]
    days_in_validation = len(inputs.days_in("validation"))
    alerts = calibrated["at_threshold"]["alerts"]
    print(f"Alert threshold {threshold:.6f}: {alerts} validation alerts ({alerts / days_in_validation:.1f}/day)")

    selection = {
        "rule": [
            "maximise validation average precision (scikit-learn step-wise PR-AUC) over scorable user-hours",
            "ties: fewer parameters, then shorter maximum sequence length, then aggregation order "
            + str(base.search.aggregations),
        ],
        "threshold_rule": "maximise validation F1 of score >= threshold on calibrated scores; ties keep the higher threshold",
        "evaluation_unit": "(acting_user, hour_start) user-hour; positive if any deduplicated validation red-team label (source user) falls in it",
        "selected_run_id": selected["run_id"],
        "selected_trial": _jsonable(selected),
        "validation_at_threshold": _jsonable(calibrated),
        "validation_alerts_per_day": alerts / days_in_validation,
        "trials_file": "trials.jsonl",
        "test_labels_used": False,
    }
    frozen_dir = args.frozen_root / selected["run_id"]
    record = freeze_detector(
        frozen_dir,
        args.runs_dir / selected["run_id"] / "checkpoint.pt",
        final_cfg,
        calibrator,
        threshold,
        selected["run_id"],
        inputs.feature_info(),
        inputs.split_cfg.fingerprint(),
        selection,
    )

    manifest = {
        "manifest_version": 1,
        "task": "3.3 — Select model settings and calibrate scores",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_revision": git_revision(),
        "search_space": base.search.__dict__,
        "training_settings": base.training.__dict__,
        "trials": [_jsonable(t) for t in trials],
        "selection": selection,
        "frozen_detector": {
            "directory": frozen_dir.relative_to(REPO_ROOT).as_posix() if frozen_dir.is_relative_to(REPO_ROOT) else str(frozen_dir),
            "record_file": FROZEN_FILENAME,
            "model_version": record["model_version"],
            "content_sha256": record["content_sha256"],
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "frozen_at_utc": record["frozen_at_utc"],
            "alert_threshold": threshold,
            "aggregation": final_cfg.aggregation,
            "max_sequence_length": final_cfg.policy.max_sequence_length,
            "calibration": record["calibration"],
        },
        "runtime_seconds": round(time.time() - started_all, 1),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen detector: {frozen_dir}; manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
