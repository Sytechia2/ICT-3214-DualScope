# Task 2.2: LANL ingestion and normalisation

## Locked decisions

- `acting_user` is the source user. Source/destination users and computers remain separate.
- Dataset time is stored as positive `int64` seconds. Dataset day is `((timestamp - 1) // 86400) + 1`.
- All non-empty category strings are preserved verbatim, including `?`, every authentication orientation, and both `Success` and `Fail`.
- Every structurally valid selected source row is retained. Ingestion never deduplicates.
- An event's stable evidence reference is `auth.txt:<source_line>`. The source line and complete raw record are also retained.
- Events use day-partitioned Parquet, Zstandard compression and ordered `part-NNNNNN.parquet` files. The default input chunk is 500,000 rows.
- Normalised red-team labels are stored in a separate tree and are absent from detector event columns.
- Task 2.2 defines no train/validation/test boundaries.

## Event dictionary

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | int64 | Positive dataset-relative seconds. |
| `dataset_day` | int32 | Hive partition field derived from `timestamp`; standard dataset readers expose it as a column. |
| `source_user` / `destination_user` | string | Original user/domain values in their distinct roles. |
| `source_computer` / `destination_computer` | string | Original computer values in their distinct roles. |
| `authentication_type` | string | Original category, including `?`. |
| `logon_type` | string | Original category, including `?`. |
| `authentication_orientation` | string | Original orientation (for example `LogOn`, `LogOff`, `TGT`, `TGS`, or `AuthMap`). |
| `authentication_result` | string | Original `Success` or `Fail` value; unfamiliar non-empty values are preserved and counted. |
| `acting_user` | string | Exact copy of `source_user`. |
| `exact_duplicate_ordinal` | int32 | Occurrence number for an exact raw record inside its timestamp group; 1 is the first occurrence. |
| `source_line` | int64 | One-based physical line in `auth.txt`. |
| `source_reference` | string | Stable `auth.txt:<source_line>` evidence reference. |
| `raw_record` | string | Complete source record without its line ending. |

An exact duplicate is a repeated nine-field authentication record within the same timestamp group. Duplicate instances beyond the first and distinct duplicate groups are reported in `summary.json`; all occurrences stay in the event table.

Rows are rejected only for the structural reasons `field_count`, `invalid_timestamp`, or `empty_field`. Rejected raw rows and line numbers are written to `rejections.jsonl`. A timestamp order inversion stops the run because ordered historical features would otherwise be unsafe. The summary reconciles scanned rows as accepted + rejected + out-of-scope.

## Run the tracked sample

```powershell
python scripts/ingest_lanl.py `
  --auth data/fixtures/lanl_auth_sample.txt `
  --redteam data/fixtures/lanl_redteam_sample.txt `
  --output data/samples/lanl_ingestion_sample `
  --day-end 2 `
  --chunk-rows 3
```

This deliberately small chunk size exercises boundaries. The generated sample is tracked as an early handoff for Members 2–4. Inspect `authentication/summary.json`, `authentication/events/dataset_day=NN`, and the physically separate `redteam_labels/labels/dataset_day=NN` tree.

## Completed Days 1–30 processing

Member 1 processed the selected dataset locally on 22 September 2026 using four independent day workers and 500,000-row chunks. The completed run produced:

| Result | Value |
| --- | ---: |
| Authentication input rows | 508,854,306 |
| Authentication rows accepted | 508,854,306 |
| Authentication rows rejected | 0 |
| Authentication Parquet parts | 1,032 |
| Red-team label rows accepted | 749 |
| Red-team label rows rejected | 0 |
| Red-team label Parquet parts | 18 |
| Authentication event size | 9.48 GiB |
| Dataset days | 30 |
| Count reconciliation | Passed |

The full result is stored locally at `data/processed/lanl_auth_days_01_30` and is excluded from Git because of its size. It should be distributed to Members 2–4 through the team's approved large-file storage. The tracked `data/samples/lanl_ingestion_sample` remains available for lightweight interface checks.

## Processed-data handoff

The shared directory has this structure:

```text
lanl_auth_days_01_30/
  authentication/
    events/dataset_day=01/part-000000.parquet
    ...
    summary.json
    rejections.jsonl
  redteam_labels/
    labels/dataset_day=02/part-000000.parquet
    ...
    summary.json
    rejections.jsonl
```

Members can load the detector events without the ingestion scripts:

```python
import pyarrow.dataset as ds

events = ds.dataset(
    "lanl_auth_days_01_30/authentication/events",
    format="parquet",
    partitioning="hive",
)
day_one = events.to_table(filter=ds.field("dataset_day") == 1)
```

Load labels only from `redteam_labels/labels` for split selection and evaluation; do not join labels into detector feature inputs. Verify the received `authentication/summary.json` reports 508,854,306 accepted rows with `reconciled: true`. Train/validation/test boundaries remain intentionally undefined until Task 2.3.

## Optional reproduction

The platform-neutral `scripts/ingest_lanl.py` command remains the reproducible ingestion entry point. A parallel reproduction requires extracted `auth.txt` and `redteam.txt`, the Task 2.1 inspection artifact containing exact day byte/row boundaries, PyArrow, and sufficient local storage. Local launch and monitoring wrappers are machine-specific and are intentionally not tracked.

## Gzip input and line maps

`scripts/ingest_lanl.py` accepts `.gz` files for `--auth` and `--redteam` when run with a single worker. Parallel workers (`--workers 2` or more) seek by byte offset and refuse gzip input.

`--line-map <file>` (plain or `.gz`, one integer per line of `--auth`) gives each event the original `auth.txt` line number in `source_line` and `source_reference`, so evidence references stay true when the input is a cut-down file such as `data/samples/lanl_pipeline_subset/auth_subset.txt.gz`. The map must be strictly increasing and have exactly one entry per line of the input, otherwise ingestion stops. A rejected row records the original number as `source_line` and its position in the input file as `physical_line`. Red-team labels always use the line number within the red-team file that was read, so for `redteam_subset.txt` the label references count lines of that small file.
