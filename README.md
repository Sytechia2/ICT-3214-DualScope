# DualScope

Multi-timescale intrusion detection using sequence and graph-based behavioural analysis.

This repository is being built from the shared workplans in:

- [DualScope WBS](DualScope_WBS.md)
- [Detailed task implementation guide](DualScope_Detailed_Tasks.md)
- [Scope and deliverables checklist](docs/scope_checklist.md)
- [Development environment setup](docs/environment_setup.md)

## Current status

Task 2.3 is complete. The chronological data split policy confirmed by Member 1 (and consumed by Member 4 for score fusion), training compromise exclusions, replay streaming engine, and evaluation unit definitions are established:
- **Training (Days 1–7, `[1, 604801)`)**: 113,699,496 events. Excludes seven red-team users (`32,659` events, 0.0287%) appearing in training compromise labels, leaving `113,666,837` eligible events (99.9713%) for normal model fitting, learned vocabularies, and scalers. Eligible training events are presumed normal background and are not guaranteed benign.
- **Validation (Days 8–16, `[604801, 1382401)`)**: 164,299,768 events, 640 label rows / 614 unique labels (600 matched, 14 unmatched). Used for model tuning, threshold selection, and fusion calibration.
- **Test (Days 17–30, `[1382401, 2592001)`)**: 230,855,042 events, 60 label rows / 52 unique labels. Evaluated strictly using frozen model checkpoints and calibration parameters.

The machine-readable split specification is in [the split configuration](config/lanl_splits.json), the verified totals and provenance are recorded in [the splits manifest](data/manifests/lanl_splits_v1.json), and the developer usage guide is in [the splits documentation](docs/lanl_splits.md).

Task 2.2 normalisation is complete. The selected Days 1–30 source was normalised into 508,854,306 authentication events and 749 separately stored red-team labels, with zero rejected rows and exact count reconciliation. The local result is a 9.48 GiB, day-partitioned Zstandard Parquet dataset ready to share through the team's large-file storage. The event contract, completed-run record and loading example are in [the LANL ingestion guide](docs/lanl_ingestion.md). A small processed sample remains tracked for interface development without the full dataset.

Task 1.1 foundation files are present; independent second-member setup verification has not yet been recorded. For Task 2.1, the inspected LANL authentication source has 1,051,430,459 records across 58 days. The initial source window is Days 1–30, retaining all 749 supplied red-team rows while reducing authentication input to 508,854,306 records. The selection rules, counts and limitations are recorded in [the dataset manifest](data/manifests/lanl_auth_days_01_30.json). Red-team labels remain separate from detector inputs, and authentication records without a matching label are unlabelled rather than proven benign. Full LANL data, trained models, generated outputs and credentials are not stored in Git.

## Quick smoke check

From the repository root, after creating the environment described in `docs/environment_setup.md`:

```powershell
python scripts/smoke_test.py
```

The smoke check uses only the tracked synthetic fixture and does not require LANL data, model checkpoints or external service credentials.

## Planned top-level layout

```text
config/       Shared configuration templates
data/         Tracked fixtures/samples; ignored raw and full processed local data
docs/         Scope, setup and technical documentation
logs/         Ignored run logs
models/       Ignored checkpoints and model artifacts
outputs/      Ignored predictions, incidents and reports
scripts/      Repeatable developer and pipeline commands
src/          Application packages
tests/        Automated tests
```
