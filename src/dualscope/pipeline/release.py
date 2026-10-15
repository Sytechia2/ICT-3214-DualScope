"""Load the frozen release model (``models/release``) with every hash and cross-check verified.

The release holds the run-of-record GRU run, the fusion model and the feature
preprocessing file (see ``models/release/README.md``). Nothing here retrains or
modifies them. Any mismatch raises ``ReleaseError`` naming the file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dualscope.fusion.hourly import INPUTS
from dualscope.sequence.config import SequenceDetectorConfig
from dualscope.sequence.training import file_sha256, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DIR = REPO_ROOT / "models" / "release"


class ReleaseError(RuntimeError):
    """A release file is missing, changed or inconsistent with the manifest."""


@dataclass
class Release:
    """The verified release: models, GRU configuration, preprocessing file and manifests."""

    release_dir: Path
    fusion_model: Any
    gru_model: Any
    gru_config: SequenceDetectorConfig
    preprocessing_path: Path
    preprocessing_sha256: str
    manifest: dict[str, Any]
    fusion_manifest: dict[str, Any]
    device: str = "cpu"
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReleaseError(f"cannot read {path}: {error}") from error


def verify_files(release_dir: Path, manifest: dict[str, Any]) -> None:
    """Check size and sha256 of every file the release manifest lists."""
    for rel, info in manifest.get("files", {}).items():
        path = release_dir / rel
        if not path.is_file():
            raise ReleaseError(f"release file missing: {path}")
        if path.stat().st_size != info["size_bytes"]:
            raise ReleaseError(f"release file size differs from the manifest: {path}")
        if file_sha256(path) != info["sha256"]:
            raise ReleaseError(f"release file hash differs from the manifest: {path}")


def load_release(
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> Release:
    """Verify and load the release. Version differences are recorded in ``Release.warnings``, not raised."""
    release_dir = Path(release_dir)
    manifest = _read_json(release_dir / "manifest.json")
    verify_files(release_dir, manifest)
    for rel in ("gru/checkpoint.pt", "gru/run_manifest.json", "fusion/model.joblib", "fusion/manifest.json", "features/preprocessing.json"):
        if rel not in manifest.get("files", {}):
            raise ReleaseError(f"release manifest does not list {release_dir / rel}")

    fusion_manifest = _read_json(release_dir / "fusion" / "manifest.json")
    model_path = release_dir / "fusion" / "model.joblib"
    if file_sha256(model_path) != fusion_manifest["model_sha256"]:
        raise ReleaseError(f"fusion model hash differs from fusion/manifest.json: {model_path}")
    if fusion_manifest["inputs"] != INPUTS:
        raise ReleaseError(f"fusion model inputs differ from the library's INPUTS: {release_dir / 'fusion' / 'manifest.json'}")

    gru = fusion_manifest["gru_score"]
    run_manifest = _read_json(release_dir / "gru" / "run_manifest.json")
    config = SequenceDetectorConfig.from_dict(run_manifest["sequence_config"])
    if config.fingerprint() != gru["sequence_config_fingerprint"]:
        raise ReleaseError(f"GRU configuration differs from the fusion manifest: {release_dir / 'gru' / 'run_manifest.json'}")
    preprocessing_path = release_dir / "features" / "preprocessing.json"
    preprocessing_sha = file_sha256(preprocessing_path)
    if preprocessing_sha != gru["preprocessing_sha256"]:
        raise ReleaseError(f"preprocessing hash differs from the fusion manifest: {preprocessing_path}")

    notes: list[str] = []
    import joblib
    import sklearn
    import torch

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fusion_model = joblib.load(model_path)
    notes.extend(str(w.message) for w in caught)
    made = manifest.get("created_with", {})
    for name, current in (("scikit-learn", sklearn.__version__), ("torch", torch.__version__)):
        if made.get(name) and made[name] != current:
            notes.append(f"{name} {current} is installed; the release was made with {made[name]}")
    try:
        gru_model, _ = load_checkpoint(release_dir / "gru" / "checkpoint.pt", gru["checkpoint_sha256"])
    except ValueError as error:
        raise ReleaseError(f"{release_dir / 'gru' / 'checkpoint.pt'}: {error}") from error
    gru_model.to(device)
    for note in notes:
        log(f"  release warning: {note}")
    return Release(
        release_dir=release_dir, fusion_model=fusion_model, gru_model=gru_model, gru_config=config,
        preprocessing_path=preprocessing_path, preprocessing_sha256=preprocessing_sha,
        manifest=manifest, fusion_manifest=fusion_manifest, device=device, warnings=notes,
    )
