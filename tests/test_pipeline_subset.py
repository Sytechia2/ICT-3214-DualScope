"""Tests for the pipeline subset cutter and the gzip / line-map ingestion support."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow.dataset as ds
import pytest

from dualscope.ingestion.lanl import IngestionConfig, ingest_authentication, ingest_redteam_labels
from scripts.make_pipeline_subset import main as make_subset

DAY = 86_400


def _auth_lines() -> list[str]:
    """Synthetic auth rows: 2 red-team users (one heavy), 12 humans, 4 machines, plus day 10."""

    rows: list[tuple[int, str]] = []
    for i in range(12):
        for k in range(30 + i):
            rows.append((1 + i * 50 + k * 7, f"H{i}@DOM1"))
    for i in range(4):
        for k in range(40):
            rows.append((100 + i * 31 + k * 11, f"M{i}$@DOM1"))
    for k in range(25):
        rows.append((DAY + 5 + k * 13, "R1@DOM1"))
    for k in range(400):
        rows.append((2 * DAY + 5 + k * 3, "R2@DOM1"))
    rows.append((9 * DAY + 30, "H1@DOM1"))
    rows.append((9 * DAY + 90, "H2@DOM1"))
    rows.append((9 * DAY + 90, "H2@DOM1"))
    rows.append((10 * DAY + 5, "H1@DOM1"))
    rows.sort(key=lambda item: item[0])
    return [
        f"{ts},{user},{user},C{ts % 5},C{ts % 7},Kerberos,Network,LogOn,Success" for ts, user in rows
    ]


REDTEAM = [
    f"{DAY + 10},R1@DOM1,C1,C2",
    f"{DAY + 50},R1@DOM1,C3,C2",
    f"{2 * DAY + 20},R2@DOM1,C1,C2",
    f"{12 * DAY},R1@DOM1,C1,C2",
]
ARGS = ["--humans", "5", "--machines", "2", "--human-min", "10", "--machine-min", "10",
        "--redteam-cap", "100"]


@pytest.fixture()
def raw_files(tmp_path: Path) -> tuple[Path, Path]:
    auth = tmp_path / "auth.txt"
    auth.write_text("\n".join(_auth_lines()) + "\n", encoding="utf-8")
    redteam = tmp_path / "redteam.txt"
    redteam.write_text("\n".join(REDTEAM) + "\n", encoding="utf-8")
    return auth, redteam


def _gzip(path: Path) -> Path:
    target = path.with_name(path.name + ".gz")
    target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    return target


def _rows(root: Path) -> list[dict[str, object]]:
    table = ds.dataset(root, format="parquet", partitioning="hive").to_table()
    return sorted(table.to_pylist(), key=lambda row: int(row["source_line"]))


def test_gzip_input_matches_plain(tmp_path: Path, raw_files: tuple[Path, Path]) -> None:
    auth, redteam = raw_files
    config = IngestionConfig(day_end=10, chunk_rows=50)
    plain = ingest_authentication(auth, tmp_path / "plain", config)
    zipped = ingest_authentication(_gzip(auth), tmp_path / "zipped", config)
    assert plain["counts"] == zipped["counts"]
    assert _rows(tmp_path / "plain" / "events") == _rows(tmp_path / "zipped" / "events")
    ingest_redteam_labels(redteam, tmp_path / "rt_plain", config)
    ingest_redteam_labels(_gzip(redteam), tmp_path / "rt_zipped", config)
    assert _rows(tmp_path / "rt_plain" / "labels") == _rows(tmp_path / "rt_zipped" / "labels")


def test_line_map_rewrites_references(tmp_path: Path) -> None:
    source = tmp_path / "subset.txt"
    source.write_text(
        "10,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n"
        "bad,line\n"
        "11,U2@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "map.txt"
    mapping.write_text("500\n777\n9000\n", encoding="utf-8")
    summary = ingest_authentication(
        source, tmp_path / "out", IngestionConfig(line_map=mapping, chunk_rows=2)
    )
    rows = _rows(tmp_path / "out" / "events")
    assert [row["source_line"] for row in rows] == [500, 9000]
    assert [row["source_reference"] for row in rows] == ["auth.txt:500", "auth.txt:9000"]
    rejection = json.loads((tmp_path / "out" / "rejections.jsonl").read_text().splitlines()[0])
    assert rejection["source_line"] == 777 and rejection["physical_line"] == 2
    assert summary["line_map"] == str(mapping)


def test_line_map_length_mismatch_fails(tmp_path: Path) -> None:
    source = tmp_path / "subset.txt"
    source.write_text("10,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n", encoding="utf-8")
    mapping = tmp_path / "map.txt"
    mapping.write_text("5\n6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line map has 2 entries"):
        ingest_authentication(source, tmp_path / "out", IngestionConfig(line_map=mapping))


def _make_parquet_root(tmp_path: Path, auth: Path, redteam: Path) -> Path:
    root = tmp_path / "ingested"
    config = IngestionConfig(day_end=10, chunk_rows=100)
    ingest_authentication(auth, root / "authentication", config)
    ingest_redteam_labels(redteam, root / "redteam_labels", config)
    return root


def test_subset_is_deterministic_and_sources_agree(
    tmp_path: Path, raw_files: tuple[Path, Path]
) -> None:
    auth, redteam = raw_files
    root = _make_parquet_root(tmp_path, auth, redteam)
    outputs = {name: tmp_path / name for name in ("parquet_a", "parquet_b", "raw_plain", "raw_gz")}
    make_subset(["--ingested", str(root), "--output", str(outputs["parquet_a"]), *ARGS])
    make_subset(["--ingested", str(root), "--output", str(outputs["parquet_b"]), *ARGS])
    make_subset(["--auth", str(auth), "--redteam", str(redteam), "--output",
                 str(outputs["raw_plain"]), *ARGS])
    make_subset(["--auth", str(_gzip(auth)), "--redteam", str(redteam), "--output",
                 str(outputs["raw_gz"]), *ARGS])

    names = ["auth_subset.txt.gz", "line_map.txt.gz", "redteam_subset.txt"]
    reference = {name: (outputs["parquet_a"] / name).read_bytes() for name in names}
    for label in ("parquet_b", "raw_plain", "raw_gz"):
        for name in names:
            assert (outputs[label] / name).read_bytes() == reference[name], (label, name)
    assert (outputs["parquet_a"] / "manifest.json").read_bytes() == (
        outputs["parquet_b"] / "manifest.json"
    ).read_bytes()

    manifest = json.loads((outputs["parquet_a"] / "manifest.json").read_text())
    users = manifest["selected_users"]
    assert users["redteam"] == ["R1@DOM1"]
    assert [item["user"] for item in manifest["excluded_redteam_users"]] == ["R2@DOM1"]
    assert len(users["human"]) == 5 and len(users["machine"]) == 2
    assert not set(users["human"]) & {"R1@DOM1", "R2@DOM1"}
    assert manifest["redteam_rows_per_day"] == {"2": 2}
    assert "10" not in manifest["events_per_day"]
    assert manifest["events_total"] == sum(manifest["events_per_day"].values())

    kept = gzip.decompress(reference["auth_subset.txt.gz"]).decode().splitlines()
    original = auth.read_text().splitlines()
    numbers = [int(n) for n in gzip.decompress(reference["line_map.txt.gz"]).decode().split()]
    assert len(kept) == len(numbers) == manifest["events_total"]
    assert all(original[n - 1] == line for n, line in zip(numbers, kept))
    selected = set(users["redteam"]) | set(users["human"]) | set(users["machine"])
    assert {line.split(",")[1] for line in kept} <= selected
    assert reference["redteam_subset.txt"].decode().splitlines() == REDTEAM[:2]


def test_subset_feeds_real_ingestion_with_original_references(
    tmp_path: Path, raw_files: tuple[Path, Path]
) -> None:
    auth, redteam = raw_files
    output = tmp_path / "subset"
    make_subset(["--auth", str(auth), "--redteam", str(redteam), "--output", str(output), *ARGS])
    ingest_authentication(
        output / "auth_subset.txt.gz",
        tmp_path / "ingested" / "authentication",
        IngestionConfig(day_end=9, line_map=output / "line_map.txt.gz"),
    )
    original = auth.read_text().splitlines()
    rows = _rows(tmp_path / "ingested" / "authentication" / "events")
    assert rows
    for row in rows:
        assert row["raw_record"] == original[int(row["source_line"]) - 1]
        assert row["source_reference"] == f"auth.txt:{row['source_line']}"
