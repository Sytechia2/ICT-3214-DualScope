"""Task 3.2: train one GRU sequence autoencoder configuration.

Samples fitting-eligible training user-hour chunks and a label-free validation
monitor sample, trains with early stopping on the monitor loss, then saves
``checkpoint.pt``, ``config.json``, ``training_log.jsonl`` and
``run_manifest.json`` under ``<runs-dir>/<run-id>``. The checkpoint is
reloaded from disk and must give finite scores on the held-out monitor sample.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.sequence.config import SequenceDetectorConfig  # noqa: E402
from dualscope.sequence.pipeline import load_inputs, prepare_samples, train_trial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "data/processed/lanl_features_days_01_30")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features.json")
    parser.add_argument("--sequence-config", type=Path, default=REPO_ROOT / "config/sequence_detector.json")
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Override the configured L.")
    parser.add_argument("--model-size", default=None, help="Model size name from the search space.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "models/sequence/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/processed/sequence_cache")
    parser.add_argument("--train-sequences", type=int, default=None, help="Override the training sample size.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--allow-pilot-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SequenceDetectorConfig.from_file(args.sequence_config)
    if args.model_size or args.max_sequence_length:
        size = args.model_size or next(iter(config.search.model_sizes))
        config = config.with_trial(args.max_sequence_length or config.policy.max_sequence_length, size)
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
        config = replace(config, training=replace(config.training, **overrides))
    inputs = load_inputs(
        args.features_root, args.splits_config, args.splits_manifest, args.feature_config,
        allow_pilot_features=args.allow_pilot_features,
    )
    # Matches the run IDs used by select_sequence_model.py so completed runs are reused.
    run_id = args.run_id or f"seq-L{config.policy.max_sequence_length}-{args.model_size or 'default'}"
    train, monitor, stats = prepare_samples(inputs, config, args.cache_dir)
    manifest = train_trial(inputs, config, train, monitor, stats, args.runs_dir / run_id, run_id)
    print(
        f"Run {run_id}: {manifest['epochs_run']} epochs in {manifest['training_seconds']}s, "
        f"selected epoch {manifest['selected_epoch']}, reload monitor error "
        f"{manifest['reload_check']['mean_reconstruction_error']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
