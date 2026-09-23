from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds
import pytest


SAMPLE_AUTH = Path("data/samples/lanl_ingestion_sample/authentication")


def test_lookup_resolves_normalized_event_and_original_source_record() -> None:
    from dualscope.evidence.lookup import AuthenticationEvidenceLookup

    lookup = AuthenticationEvidenceLookup(SAMPLE_AUTH)
    event = lookup.lookup("auth.txt:3")

    assert event is not None
    assert event["source_reference"] == "auth.txt:3"
    assert event["source_line"] == 3
    assert event["timestamp"] == 2
    assert event["source_user"] == "U1@DOM1"
    assert event["raw_record"] == (
        "2,U1@DOM1,U2@DOM1,C1,C2,?,Network,TGS,Success"
    )
    assert event["source_location"] == {"file": "auth.txt", "line": 3}


def test_lookup_distinguishes_duplicate_source_rows_and_returns_none_for_rejected_or_missing_ids() -> None:
    from dualscope.evidence.lookup import AuthenticationEvidenceLookup

    lookup = AuthenticationEvidenceLookup(SAMPLE_AUTH)
    first_duplicate = lookup.lookup("auth.txt:1")
    second_duplicate = lookup.lookup("auth.txt:2")

    assert first_duplicate is not None and second_duplicate is not None
    assert first_duplicate["raw_record"] == second_duplicate["raw_record"]
    assert first_duplicate["source_reference"] != second_duplicate["source_reference"]
    assert first_duplicate["exact_duplicate_ordinal"] == 1
    assert second_duplicate["exact_duplicate_ordinal"] == 2
    assert lookup.lookup("auth.txt:8") is None  # invalid timestamp; rejected during ingestion
    assert lookup.lookup("auth.txt:9") is None  # invalid field count; rejected during ingestion
    assert lookup.lookup("auth.txt:999") is None
    with pytest.raises(ValueError):
        lookup.lookup("not-a-source-reference")


def test_lookup_many_preserves_input_order_and_paginates_deterministically() -> None:
    from dualscope.evidence.lookup import AuthenticationEvidenceLookup

    lookup = AuthenticationEvidenceLookup(SAMPLE_AUTH)
    refs = [f"auth.txt:{line}" for line in range(1, 8)]

    many = lookup.lookup_many([refs[2], refs[0], refs[2], refs[1]])
    assert list(many) == [refs[2], refs[0], refs[1]]

    first = lookup.resolve_page(refs, offset=0, limit=3)
    second = lookup.resolve_page(refs, offset=3, limit=3)
    last = lookup.resolve_page(refs, offset=6, limit=3)
    assert first == {
        "offset": 0,
        "limit": 3,
        "total": 7,
        "items": [
            {"source_reference": ref, "event": lookup.lookup(ref)} for ref in refs[:3]
        ],
    }
    assert [item["source_reference"] for item in second["items"]] == refs[3:6]
    assert [item["source_reference"] for item in last["items"]] == refs[6:]
    duplicate_page = lookup.resolve_page(
        ["auth.txt:1", "auth.txt:1", "auth.txt:2"], offset=0, limit=2
    )
    assert duplicate_page["total"] == 3
    assert [item["source_reference"] for item in duplicate_page["items"]] == [
        "auth.txt:1",
        "auth.txt:1",
    ]
    assert duplicate_page["items"][0]["event"] == duplicate_page["items"][1]["event"]
    with pytest.raises(ValueError):
        lookup.resolve_page(refs, offset=-1, limit=2)


def test_feature_engine_keeps_each_source_reference_in_event_order() -> None:
    import pyarrow as pa

    from dualscope.features.engine import HistoricalFeatureEngine
    from dualscope.features.schemas import RAW_FEATURE_SCHEMA

    event_root = SAMPLE_AUTH / "events"
    table = ds.dataset(event_root, format="parquet", partitioning="hive").to_table()
    table = table.sort_by([("source_line", "ascending")])
    # The feature engine consumes normalized event columns and must preserve provenance.
    input_names = [
        "timestamp",
        "source_user",
        "destination_user",
        "source_computer",
        "destination_computer",
        "authentication_type",
        "logon_type",
        "authentication_orientation",
        "authentication_result",
        "acting_user",
        "exact_duplicate_ordinal",
        "source_line",
        "source_reference",
        "dataset_day",
    ]
    batch = table.select(input_names).to_batches(max_chunksize=table.num_rows)[0]

    output = HistoricalFeatureEngine().process_batch(batch)

    assert "source_reference" in RAW_FEATURE_SCHEMA.names
    assert output["source_reference"].to_pylist() == table["source_reference"].to_pylist()


def test_carriers_keep_ordered_event_references_and_paginate_large_sets() -> None:
    from dualscope.evidence.carriers import (
        attach_edge_references,
        attach_sequence_references,
        paginate_references,
    )

    events = [
        {"source_reference": "auth.txt:10", "split": "test"},
        {"source_reference": "auth.txt:11", "split": "test"},
        {"source_reference": "auth.txt:10", "split": "test"},
    ]
    sequence = attach_sequence_references({"sequence_id": "s1", "split": "test"}, events)
    edge = attach_edge_references({"source": "u1", "target": "c1"}, events)

    assert sequence["source_references"] == ["auth.txt:10", "auth.txt:11", "auth.txt:10"]
    assert sequence["split"] == "test"
    assert edge["source_references"] == sequence["source_references"]
    assert edge["evidence_count"] == 3
    with pytest.raises(ValueError, match="split"):
        attach_sequence_references(
            {"sequence_id": "s2", "split": "train"}, events
        )
    with pytest.raises(ValueError):
        attach_sequence_references(
            {"sequence_id": "s3", "period": "train"},
            [
                {"source_reference": "auth.txt:12", "period": "test"},
            ],
            split_field="period",
        )

    refs = (f"auth.txt:{line}" for line in range(1, 10_001))
    page_one = paginate_references(refs, offset=4_000, limit=250)
    assert page_one == {
        "source_references": [f"auth.txt:{line}" for line in range(4_001, 4_251)],
        "offset": 4_000,
        "limit": 250,
        "has_more": True,
        "next_offset": 4_250,
    }
