import json
import shutil
from pathlib import Path

import pytest

from dualscope.pipeline.release import DEFAULT_RELEASE_DIR, ReleaseError, load_release, verify_files


def _copy_release(tmp_path: Path) -> Path:
    target = tmp_path / "release"
    shutil.copytree(DEFAULT_RELEASE_DIR, target)
    return target


def test_release_manifest_lists_existing_files_with_matching_hashes():
    manifest = json.loads((DEFAULT_RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    verify_files(DEFAULT_RELEASE_DIR, manifest)
    assert {"gru/checkpoint.pt", "fusion/model.joblib", "features/preprocessing.json"} <= set(manifest["files"])


def test_changed_file_raises_naming_the_file(tmp_path):
    release = _copy_release(tmp_path)
    with (release / "features" / "preprocessing.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ReleaseError, match="preprocessing.json"):
        load_release(release)


def test_missing_file_raises(tmp_path):
    release = _copy_release(tmp_path)
    (release / "gru" / "checkpoint.pt").unlink()
    with pytest.raises(ReleaseError, match="checkpoint.pt"):
        load_release(release)


def test_real_release_loads_on_cpu():
    release = load_release(DEFAULT_RELEASE_DIR, log=lambda _: None)
    assert release.gru_config.fingerprint() == release.fusion_manifest["gru_score"]["sequence_config_fingerprint"]
    assert release.preprocessing_sha256.startswith("9fc82c04")
