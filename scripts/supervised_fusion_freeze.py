"""Experiment features-v2, Option A: freeze the final fusion model (docs/experiment_features_v2.md).

Refits the model that met the Option A success rule (HistGradientBoosting,
settings unchanged) on all validation days 8-16 and writes it with a manifest
(inputs, settings, training data, hashes). Uses the hourly counts cache and run
A's per-unit scores from ``supervised_fusion_validate.py``. Test days are never
loaded and test labels are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("supervised_fusion_validate", REPO_ROOT / "scripts" / "supervised_fusion_validate.py")
_fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fusion)

from dualscope.sequence.calibration import load_positive_user_hours, unit_labels  # noqa: E402
from dualscope.sequence.pipeline import git_revision  # noqa: E402
from dualscope.splits import SplitConfig  # noqa: E402

MODEL = "hist_gradient_boosting"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=Path, default=REPO_ROOT / "outputs/experiment_v2/units_A.parquet")
    parser.add_argument("--counts", type=Path, default=REPO_ROOT / "outputs/experiment_v2/fusion_counts.parquet")
    parser.add_argument("--gru-run", type=Path, default=REPO_ROOT / "models/sequence/runs_experiment_v2/A")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data/processed/lanl_auth_days_01_30/redteam_labels/labels")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/fusion/experiment_v2/frozen")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; a frozen model is never overwritten")

    split_cfg = SplitConfig.from_file(args.splits_config)
    positives = load_positive_user_hours(args.labels_dir, split_cfg, "validation")
    counts = pd.read_parquet(args.counts)
    gru = pd.read_parquet(args.units, columns=["user", "hour", "day", _fusion.GRU_COLUMN]).rename(columns={_fusion.GRU_COLUMN: "gru_max_event"})
    frame = counts.merge(gru, on=["user", "hour", "day"], how="inner", validate="1:1")
    if not len(frame) == len(counts) == len(gru):
        raise SystemExit("unit mismatch between counts and GRU scores")
    if sorted(frame["day"].unique()) != list(range(8, 17)):
        raise SystemExit("expected validation days 8-16 only")
    frame = frame.sort_values(["day", "user", "hour"], kind="stable").reset_index(drop=True)
    labels = unit_labels(frame["user"].to_numpy(), frame["hour"].to_numpy(), positives)
    if int(labels.sum()) != len(positives):
        raise SystemExit("not every validation positive matched a unit")

    started = time.time()
    model = _fusion.MODELS[MODEL]().fit(_fusion.model_inputs(frame), labels)
    print(f"{MODEL} fitted on days 8-16: {len(frame):,} user-hours, {int(labels.sum())} positive ({time.time() - started:.0f}s, {model.n_iter_} iterations)")

    args.output.mkdir(parents=True)
    model_path = args.output / "model.joblib"
    joblib.dump(model, model_path)
    gru_manifest = json.loads((args.gru_run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "model": MODEL,
        "experiment": "features-v2 Option A, final fusion model (docs/experiment_features_v2.md)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision(),
        "sklearn_version": sklearn.__version__,
        "settings": model.get_params(),
        "n_iter": int(model.n_iter_),
        "inputs": _fusion.INPUTS,
        "gru_score": {"run_dir": str(args.gru_run), "aggregation": "max_event", "checkpoint_sha256": gru_manifest["checkpoint"]["sha256"],
                      "sequence_config_fingerprint": gru_manifest["sequence_config_fingerprint"],
                      "preprocessing_sha256": gru_manifest["feature"]["preprocessing_sha256"]},
        "training": {"days": list(range(8, 17)), "units": int(len(frame)), "positives": int(labels.sum()),
                     "labels": "validation red-team user-hours (load_positive_user_hours)",
                     "units_sha256": sha256(args.units), "counts_sha256": sha256(args.counts)},
        "test_labels_used": False,
        "model_sha256": sha256(model_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Frozen model written to {args.output} (sha256 {manifest['model_sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
