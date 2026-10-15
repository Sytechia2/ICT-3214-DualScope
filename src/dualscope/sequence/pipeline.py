"""Orchestration shared by the sequence-detector scripts (Tasks 3.1-3.4).

Keeps the command-line scripts thin: input verification, sample preparation
with an on-disk cache, one training trial, validation scoring and run records.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from dualscope.features.config import FeatureConfig
from dualscope.sequence.builder import DaySequences, available_days, iter_day_sequences
from dualscope.sequence.config import SequenceDetectorConfig, SequencePolicy
from dualscope.splits import SequenceCandidateRejectionTracker, SplitConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SequenceInputs:
    """Verified locations and versions of the detector's upstream inputs."""

    features_dir: Path
    preprocessing: dict[str, Any]
    preprocessing_path: Path
    feature_cfg: FeatureConfig
    split_cfg: SplitConfig
    excluded_users: list[str]
    labels_dir: Path | None

    def feature_info(self) -> dict[str, Any]:
        from dualscope.sequence.training import file_sha256

        return {
            "feature_version": self.preprocessing.get("feature_version"),
            "feature_name": self.preprocessing.get("feature_name"),
            "feature_config_fingerprint": self.preprocessing.get("feature_config_fingerprint"),
            "preprocessing_sha256": file_sha256(self.preprocessing_path),
            "preprocessing_mode": self.preprocessing.get("mode"),
            "features_dir": _display_path(self.features_dir),
        }

    def days_in(self, split_name: str) -> list[int]:
        split = self.split_cfg.get_split(split_name)
        return [d for d in available_days(self.features_dir) if split.contains_day(d)]

    def iter_days(
        self,
        days: Sequence[int],
        policy: SequencePolicy,
        tracker: SequenceCandidateRejectionTracker | None = None,
    ) -> Iterator[DaySequences]:
        return iter_day_sequences(self.features_dir, days, self.split_cfg, policy, self.excluded_users, tracker)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_inputs(
    features_root: str | Path,
    splits_config: str | Path,
    splits_manifest: str | Path,
    feature_config: str | Path,
    labels_dir: str | Path | None = None,
    allow_pilot_features: bool = False,
) -> SequenceInputs:
    """Load and cross-check the feature build, split policy and training exclusions."""
    features_root = Path(features_root)
    features_dir = features_root / "transformed" / "events"
    preprocessing_path = features_root / "preprocessing.json"
    if not features_dir.is_dir() or not preprocessing_path.is_file():
        raise FileNotFoundError(f"expected Task 2.4 output with transformed/events and preprocessing.json in {features_root}")
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    split_cfg = SplitConfig.from_file(splits_config)
    feature_cfg = FeatureConfig.from_file(feature_config)
    if not preprocessing.get("is_frozen"):
        raise ValueError("feature preprocessing is not frozen")
    if preprocessing.get("feature_config_fingerprint") != feature_cfg.fingerprint():
        raise ValueError("preprocessing was fitted with a different feature configuration")
    if preprocessing.get("split_policy_fingerprint") != split_cfg.fingerprint():
        raise ValueError("preprocessing was fitted with a different split policy")
    if not preprocessing.get("is_production", False) and not allow_pilot_features:
        raise ValueError("features come from a pilot run; pass allow_pilot_features for development only")
    manifest = json.loads(Path(splits_manifest).read_text(encoding="utf-8"))
    if manifest.get("config_fingerprint") != split_cfg.fingerprint():
        raise ValueError("splits manifest does not match the split configuration")
    excluded = sorted(str(u) for u in manifest["training_exclusions"]["excluded_users"])
    return SequenceInputs(
        features_dir=features_dir,
        preprocessing=preprocessing,
        preprocessing_path=preprocessing_path,
        feature_cfg=feature_cfg,
        split_cfg=split_cfg,
        excluded_users=excluded,
        labels_dir=Path(labels_dir) if labels_dir else None,
    )


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_samples(
    inputs: SequenceInputs,
    config: SequenceDetectorConfig,
    cache_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
):
    """Training sample (eligible train user-hours) and label-free validation monitor sample."""
    from dualscope.sequence.training import SequenceSample, sample_chunks

    settings = config.training
    length = config.policy.max_sequence_length
    key = f"L{length}_n{settings.train_sequences}_m{settings.monitor_sequences}_c{settings.max_chunks_per_user_hour}_s{settings.seed}"
    if config.policy.extra_binary_inputs:
        key += f"_x{len(config.policy.extra_binary_inputs)}"
    cache = Path(cache_dir) / f"samples_{key}.npz" if cache_dir else None
    fingerprint = f"{inputs.feature_info()['preprocessing_sha256']}:{inputs.split_cfg.fingerprint()}:{','.join(inputs.excluded_users)}"
    if config.policy.extra_binary_inputs:
        # Samples hold the model input columns, so a different input list needs its own sample.
        fingerprint += f":binary={','.join(config.policy.binary_inputs)}"
    if cache is not None and cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["fingerprint"]) == fingerprint:
            log(f"Using cached samples {cache}")
            samples = []
            for prefix in ("train", "monitor"):
                samples.append(
                    SequenceSample(*(stored[f"{prefix}_{name}"] for name in ("dense", "categorical", "starts", "lengths", "keys")))
                )
            return samples[0], samples[1], json.loads(str(stored["stats"]))

    rng = np.random.default_rng(settings.seed)
    started = time.time()
    train, train_stats = sample_chunks(
        inputs.iter_days(inputs.days_in("train"), config.policy),
        length,
        settings.train_sequences,
        settings.max_chunks_per_user_hour,
        rng,
        lambda day: day.fitting_eligible,
    )
    log(f"Training sample: {train_stats} ({time.time() - started:.0f}s)")
    started = time.time()
    monitor, monitor_stats = sample_chunks(
        inputs.iter_days(inputs.days_in("validation"), config.policy),
        length,
        settings.monitor_sequences,
        settings.max_chunks_per_user_hour,
        rng,
        lambda day: day.scorable_mask(),
    )
    log(f"Validation monitor sample (label-free): {monitor_stats} ({time.time() - started:.0f}s)")
    stats = {"train": train_stats, "monitor": monitor_stats}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"fingerprint": np.array(fingerprint), "stats": np.array(json.dumps(stats))}
        for prefix, sample in (("train", train), ("monitor", monitor)):
            for name in ("dense", "categorical", "starts", "lengths", "keys"):
                arrays[f"{prefix}_{name}"] = getattr(sample, name)
        np.savez(cache, **arrays)
    return train, monitor, stats


def train_trial(
    inputs: SequenceInputs,
    config: SequenceDetectorConfig,
    train,
    monitor,
    sample_stats: dict[str, Any],
    run_dir: str | Path,
    run_id: str,
    log: Callable[[str], None] = print,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one configuration and save checkpoint, config, log and run manifest."""
    from dualscope.sequence.model import ModelSpec
    from dualscope.sequence.training import (
        categorical_cardinalities,
        load_checkpoint,
        mean_reconstruction_loss,
        runtime_environment,
        save_checkpoint,
        train_autoencoder,
    )
    from dualscope.sequence.config import SEQUENCE_NUMERIC_INPUTS

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    spec = ModelSpec.from_settings(
        config.model, len(SEQUENCE_NUMERIC_INPUTS), len(config.policy.binary_inputs), categorical_cardinalities(inputs.preprocessing)
    )
    log_path = run_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    def record(entry: dict[str, Any]) -> None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        log(f"  epoch {entry['epoch']}: train={entry['train_loss']:.5f} monitor={entry['monitor_loss']:.5f} ({entry['epoch_seconds']}s)")

    started = time.time()
    model, history = train_autoencoder(spec, train, monitor, config.training, log=record, device=device)
    duration = time.time() - started
    metadata = {
        "run_id": run_id,
        "sequence_config_fingerprint": config.fingerprint(),
        "feature": inputs.feature_info(),
        "split_policy_fingerprint": inputs.split_cfg.fingerprint(),
        "seed": config.training.seed,
    }
    checkpoint_sha = save_checkpoint(run_dir / "checkpoint.pt", model, metadata)

    # Reload from disk and score the held-out validation monitor sample without updating weights.
    reloaded, _ = load_checkpoint(run_dir / "checkpoint.pt", checkpoint_sha)
    reload_loss = mean_reconstruction_loss(reloaded, monitor)
    if not np.isfinite(reload_loss):
        raise FloatingPointError("reloaded checkpoint produced non-finite scores")
    manifest = {
        "task": "3.2 — Implement and train the GRU autoencoder",
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_revision": git_revision(),
        "sequence_config": config.to_dict(),
        "sequence_config_fingerprint": config.fingerprint(),
        "model_spec": spec.to_dict(),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "feature": inputs.feature_info(),
        "split_policy_fingerprint": inputs.split_cfg.fingerprint(),
        "excluded_training_users": inputs.excluded_users,
        "samples": sample_stats,
        "training_seconds": round(duration, 1),
        "epochs_run": len(history),
        "selected_epoch": next(h["epoch"] for h in history if h["selected"]),
        "loss_history": history,
        "checkpoint": {"path": "checkpoint.pt", "sha256": checkpoint_sha},
        "reload_check": {
            "held_out": "label-free validation monitor sample",
            "sequences": len(monitor),
            "mean_reconstruction_error": reload_loss,
            "finite": True,
        },
        "environment": {**runtime_environment(), "training_device": device},
        "labels_used": False,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def score_split(
    inputs: SequenceInputs,
    policy: SequencePolicy,
    models: dict[str, Any],
    split_name: str,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Score all available user-hours of a split with several same-length models.

    Returns ``users``, ``hours``, ``days`` and ``n_events`` arrays for the scored
    units, and ``raw[model_name][aggregation]`` score arrays aligned with them.
    """
    from dualscope.sequence.scoring import score_day

    units: dict[str, list[np.ndarray]] = {"users": [], "hours": [], "days": [], "n_events": []}
    parts: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in models}
    for day in inputs.iter_days(inputs.days_in(split_name), policy):
        available = day.scorable_mask()
        units["users"].append(day.users[available])
        units["hours"].append(day.hour_starts[available])
        units["days"].append(np.full(int(available.sum()), day.dataset_day, dtype=np.int32))
        units["n_events"].append(day.counts[available])
        for name, model in models.items():
            started = time.time()
            scores = score_day(model, day, policy.max_sequence_length)
            for aggregation, values in scores.raw.items():
                parts[name].setdefault(aggregation, []).append(values[available])
            log(f"  day {day.dataset_day} {name}: {available.sum():,} user-hours, {len(scores.chunks):,} chunks ({time.time() - started:.0f}s)")
    result: dict[str, Any] = {key: np.concatenate(values) for key, values in units.items()}
    result["raw"] = {name: {agg: np.concatenate(chunks) for agg, chunks in parts[name].items()} for name in models}
    return result
