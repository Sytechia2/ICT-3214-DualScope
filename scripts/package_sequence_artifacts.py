"""Package the frozen sequence detector and exported scores for team sharing.

Model checkpoints and full score files are not stored in Git. This script
bundles them for the team's shared storage (for example Google Drive) as:

  sequence_detector_<model_version>.zip          frozen_detector.json + checkpoint.pt
  sequence_scores_<model_version>_<split>.zip    day partitions of one split
  sequence_scores_<model_version>_summary.json   export summary
  SHA256SUMS.txt                                 checksum of every file above
  README.txt                                     contents, source revision, how to use

Archive members keep repository-relative paths, so extracting a zip at the
repository root restores ``models/sequence/frozen/...`` and
``outputs/sequence_scores/...`` exactly where the scripts expect them.

Verify a downloaded package with ``--verify <package directory>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip(target: Path, members: list[tuple[Path, str]]) -> None:
    # Parquet and checkpoints are already compressed, so store without recompressing.
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, name in members:
            archive.write(path, name)


def package(frozen_dir: Path, scores_root: Path, output: Path) -> Path:
    record = json.loads((frozen_dir / "frozen_detector.json").read_text(encoding="utf-8"))
    summary = json.loads((scores_root / "summary.json").read_text(encoding="utf-8"))
    version = record["model_version"]
    if summary["model_version"] != version:
        raise SystemExit(f"scores were exported by {summary['model_version']}, not {version}")
    package_dir = output / version
    package_dir.mkdir(parents=True, exist_ok=True)

    written = []
    frozen_prefix = f"models/sequence/frozen/{frozen_dir.name}"
    detector_zip = package_dir / f"sequence_detector_{version}.zip"
    _zip(
        detector_zip,
        [
            (frozen_dir / name, f"{frozen_prefix}/{name}")
            for name in ("frozen_detector.json", record["checkpoint"]["path"])
        ],
    )
    written.append(detector_zip)

    scores_prefix = f"outputs/sequence_scores/{version}"
    for split, entry in summary["splits"].items():
        if "days" not in entry:
            continue
        members = [
            (path, f"{scores_prefix}/scores/{path.parent.name}/{path.name}")
            for day in entry["days"]
            for path in sorted((scores_root / "scores" / f"dataset_day={day:02d}").glob("*.parquet"))
        ]
        if split == next(iter(summary["splits"])):
            members.append((scores_root / "summary.json", f"{scores_prefix}/summary.json"))
        split_zip = package_dir / f"sequence_scores_{version}_{split}.zip"
        _zip(split_zip, members)
        written.append(split_zip)

    summary_copy = package_dir / f"sequence_scores_{version}_summary.json"
    summary_copy.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_copy)

    sums = "".join(f"{sha256(path)}  {path.name}\n" for path in written)
    (package_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    sizes = "\n".join(f"  {path.name}  ({path.stat().st_size / 1e6:,.1f} MB)" for path in written)
    readme = f"""DualScope sequence detector artifacts ({version})

Produced by Member 2 (Tasks 3.1-3.4). Source revision: {summary.get('source_revision')}
Frozen record content hash: {record['content_sha256']}
Checkpoint SHA-256: {record['checkpoint']['sha256']}

Files:
{sizes}

Use:
  1. Check the branch/commit above is in your checkout.
  2. Verify downloads:  python scripts/package_sequence_artifacts.py --verify <this folder>
  3. Extract each zip at the repository root. Paths inside the archives are
     repository-relative (models/sequence/frozen/..., outputs/sequence_scores/...).
  4. Read scores with dualscope.sequence.export.SequenceScoreStore; see
     docs/sequence_detector_handoff.md for fields, statuses and alignment rules.

Do not edit files in place. A new model or export gets a new version folder.
"""
    (package_dir / "README.txt").write_text(readme, encoding="utf-8")
    return package_dir


def verify(package_dir: Path) -> bool:
    ok = True
    for line in (package_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = package_dir / name
        actual = sha256(path) if path.is_file() else "missing"
        status = "OK" if actual == expected else "FAILED"
        ok &= status == "OK"
        print(f"{status}  {name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--scores-root", type=Path, help="outputs/sequence_scores/<model_version>")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/share")
    parser.add_argument("--verify", type=Path, default=None, help="Verify a downloaded package directory.")
    args = parser.parse_args()
    if args.verify is not None:
        return 0 if verify(args.verify) else 1
    if args.frozen_dir is None or args.scores_root is None:
        parser.error("--frozen-dir and --scores-root are required to build a package")
    package_dir = package(args.frozen_dir.resolve(), args.scores_root.resolve(), args.output)
    print(f"Package written to {package_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
