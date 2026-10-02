# Task 2.5: Evidence references

## Purpose and current scope

An accepted authentication event has a stable `source_reference` in the form `auth.txt:<source_line>`, for example `auth.txt:3`. The source line distinguishes identical duplicate records. The normalized event stores this reference, its `source_line`, and the original `raw_record`; ingestion also records the reference format in `authentication/summary.json`. Rejected or out-of-scope source lines do not have normalized event rows.

Task 2.5 provides lookup and carrier utilities so detector outputs can preserve these IDs. The sequence and graph builders now carry references into their generated evidence artifacts; the checked-in incident trace remains a small synthetic demonstration rather than a real incident. The local production Task 2.4 run processed and reconciled all 508,854,306 events, while its large generated artifacts remain outside Git.

## Lookup API

`AuthenticationEvidenceLookup` accepts either an authentication dataset root containing `summary.json` and `events/`, or the `events/` directory itself:

```python
from dualscope.evidence.lookup import AuthenticationEvidenceLookup

lookup = AuthenticationEvidenceLookup("data/samples/lanl_ingestion_sample/authentication")
event = lookup.lookup("auth.txt:3")
# event includes normalized fields, raw_record, source_reference,
# source_line, dataset_day, and source_location={"file": "auth.txt", "line": 3}
```

`lookup()` returns `None` when a well-formed reference is absent from the selected dataset. Malformed references raise `ValueError`. `lookup_many(references)` deduplicates IDs in first-seen order, preserves that order in its result mapping, and reads each candidate Parquet row group at most once. Separate source lines remain distinct even when their raw text is identical.

On initialization, the lookup reads the summary's part manifest and Parquet metadata, requiring `source_line` min/max statistics for every row group. It builds a sorted in-memory interval index without reading event rows. A lookup uses those ranges to select possible row groups, then checks exact `source_reference` equality (so gaps in a range do not produce false matches). Candidate groups are scanned in batches of 8,192 rows, avoiding materialization of an entire row group in Python memory. Initialization metadata work covers all manifest parts and row groups.

`resolve_page(references, offset=0, limit=100)` returns `{offset, limit, total, items}`. `total` counts all input references, including repeats; the page preserves input order and repeated IDs. It consumes the iterable once to calculate the exact total while retaining only IDs in the requested page, then scans only candidate row groups for those page IDs. Each item has `source_reference` and its event or `null`. The lookup API limits `limit` to 10,000. It rejects negative offsets, invalid limits, unsafe manifest paths, missing files, missing statistics, and duplicate stored rows for a queried reference.

## Carrying references into detector evidence

`dualscope.evidence.carriers` provides helpers that copy records and keep provenance in the caller's desired order:

- `attach_sequence_references(sequence, events)` adds `source_references` to a sequence. It preserves duplicates and checks split metadata when supplied.
- `attach_edge_references(edge, events)` adds ordered `source_references` and `evidence_count` to an aggregated graph edge. Duplicate references are preserved. Production graph snapshots bound the embedded preview to the first 100 source lines per edge, retain the exact contributing count, and mark truncation explicitly.
- `paginate_references(references, offset=0, limit=100)` returns a stable page, `has_more`, and `next_offset`. It uses one-item lookahead, so it can page a large iterable without materializing it. For a one-shot iterator, callers must supply a fresh iterator for each page.

The Task 2.4 raw and transformed feature schemas also preserve `source_reference` for each event. These fields let downstream builders carry event IDs into higher-level evidence. The helpers remain independent utilities; the detector builders and exporters are responsible for their own output construction.

## Demonstration and focused tests

Follow the [development environment setup](environment_setup.md) to create the Python 3.11, 3.12, or 3.13 virtual environment and install `requirements.txt`. From the repository root, run the synthetic incident trace:

```powershell
python scripts/trace_evidence.py
```

It uses the tracked ingestion sample by default. The sample incident packages U1's first-hour sequence (`auth.txt:1`, `auth.txt:2`, `auth.txt:3`) and a U1-to-C1 graph edge (`auth.txt:1`, `auth.txt:2`). The first two records are exact duplicate raw lines with distinct source references. The script resolves the sequence records and pages the edge evidence one reference at a time. To use another compatible ingestion output, pass `--dataset-root <authentication-output>`; change page size with `--page-size`.

Run the focused evidence tests with:

```powershell
python -m pytest tests/test_evidence.py
```

The tests cover lookup and raw-record recovery, duplicate IDs, absent and malformed references, ordered multi-lookup, pagination, feature reference preservation, sequence and edge carriers, split checks, and large iterable paging. The trace demonstrates the sample data contract; production graph evidence is additionally exercised by the graph detector tests and export pipeline.
