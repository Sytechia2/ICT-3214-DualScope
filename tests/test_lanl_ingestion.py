from __future__ import annotations

import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from dualscope.ingestion.lanl import (
    IngestionConfig,
    dataset_day,
    ingest_authentication,
    ingest_authentication_parallel,
    ingest_redteam_labels,
)


FIXTURE = Path("data/fixtures/lanl_auth_sample.txt")


def test_dataset_day_boundaries() -> None:
    assert dataset_day(1) == 1
    assert dataset_day(86_400) == 1
    assert dataset_day(86_401) == 2
    with pytest.raises(ValueError):
        dataset_day(0)


def test_auth_ingestion_reconciles_and_preserves_rows(tmp_path: Path) -> None:
    output = tmp_path / "auth"
    summary = ingest_authentication(FIXTURE, output, IngestionConfig(day_end=2, chunk_rows=3))

    assert summary["counts"] == {
        "input_rows_scanned": 9,
        "accepted_rows": 7,
        "rejected_rows": 2,
        "out_of_scope_rows": 0,
        "reconciled": True,
        "exact_duplicate_instances_beyond_first": 2,
        "distinct_exact_duplicate_groups": 2,
    }
    assert summary["rejection_reasons"] == {"field_count": 1, "invalid_timestamp": 1}
    assert len(summary["parts"]) == 4
    assert [part["part"] for part in summary["parts"] if part["dataset_day"] == 1] == [0, 1]

    tables = [pq.read_table(output / "events" / part["path"]) for part in summary["parts"]]
    rows = [row for table in tables for row in table.to_pylist()]
    assert len(rows) == 7
    assert rows[0]["source_reference"] == "auth.txt:1"
    assert rows[1]["exact_duplicate_ordinal"] == 2
    assert rows[0]["acting_user"] == rows[0]["source_user"]
    assert rows[2]["authentication_type"] == "?"
    assert {row["authentication_result"] for row in rows} == {"Success", "Fail"}
    assert "label" not in rows[0]
    assert json.loads((output / "summary.json").read_text())["counts"]["reconciled"]
    assert pq.ParquetFile(output / "events" / summary["parts"][0]["path"]).metadata.row_group(
        0
    ).column(0).compression == "ZSTD"


def test_redteam_labels_are_written_separately(tmp_path: Path) -> None:
    output = tmp_path / "labels"
    summary = ingest_redteam_labels(
        Path("data/fixtures/lanl_redteam_sample.txt"),
        output,
        IngestionConfig(day_end=2, chunk_rows=1),
    )

    assert summary["counts"]["accepted_rows"] == 2
    assert summary["counts"]["reconciled"] is True
    rows = [
        row
        for part in summary["parts"]
        for row in pq.read_table(output / "labels" / part["path"]).to_pylist()
    ]
    assert rows[0]["source_reference"] == "redteam.txt:1"
    assert "acting_user" not in rows[0]


def test_order_inversion_is_rejected_across_chunks(tmp_path: Path) -> None:
    source = tmp_path / "auth.txt"
    source.write_text(
        "2,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n"
        "1,U1,U1,C1,C1,NTLM,Network,LogOn,Success\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="order inversion"):
        ingest_authentication(source, tmp_path / "out", IngestionConfig(chunk_rows=1))


def test_parallel_ingestion_uses_exact_day_boundaries_and_reports_progress(
    tmp_path: Path,
) -> None:
    day_one = [
        "1,U1,U1,C1,C1,?,Network,LogOn,Success\n",
        "1,U1,U1,C1,C1,?,Network,LogOn,Success\n",
        "86400,U2,U3,C2,C3,NTLM,Network,LogOn,Fail\n",
    ]
    day_two = [
        "86401,U3,U4,C3,C4,Kerberos,?,TGS,Success\n",
        "86402,U4,U5,C4,C5,Negotiate,Service,LogOff,Success\n",
    ]
    source = tmp_path / "auth.txt"
    source.write_text("".join(day_one + day_two), encoding="utf-8", newline="")
    inspection = tmp_path / "inspection.json"
    inspection.write_text(
        json.dumps(
            {
                "authentication_stream_inspection": {
                    "row_counts_per_dataset_day": {"1": 3, "2": 2},
                    "byte_counts_per_dataset_day": {
                        "1": len("".join(day_one).encode()),
                        "2": len("".join(day_two).encode()),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "parallel"
    summary = ingest_authentication_parallel(
        source,
        output,
        IngestionConfig(day_end=2, chunk_rows=1),
        inspection,
        workers=2,
    )

    assert summary["counts"]["accepted_rows"] == 5
    assert summary["counts"]["reconciled"] is True
    assert summary["counts"]["exact_duplicate_instances_beyond_first"] == 1
    assert summary["parallel_workers"] == 2
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["percent_complete"] == 100.0
    assert progress["completed_days"] == [1, 2]
    rows = ds.dataset(output / "events", format="parquet", partitioning="hive").to_table()
    assert rows.num_rows == 5
    assert sorted(rows.column("source_reference").to_pylist()) == [
        "auth.txt:1",
        "auth.txt:2",
        "auth.txt:3",
        "auth.txt:4",
        "auth.txt:5",
    ]
