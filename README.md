# DualScope

Multi-timescale intrusion detection using sequence and graph-based behavioural analysis.

This repository is being built from the shared workplans in:

- [DualScope WBS](DualScope_WBS.md)
- [Detailed task implementation guide](DualScope_Detailed_Tasks.md)
- [Scope and deliverables checklist](docs/scope_checklist.md)
- [Development environment setup](docs/environment_setup.md)

## Current status

Tasks 3.1–3.4, the short-term sequence detector, are implemented and frozen:

- **3.1 Sequences:** 11,846,723 acting-user/hour sequences, with zero boundary violations and Day 1 marked as warm-up.
- **3.2 Models:** six GRU sequence autoencoders trained on 2,238,183 eligible training user-hours.
- **3.3 Selection:** the settings were chosen on validation data only.
- **3.4 Export:** calibrated 0–1 user-hour scores with explicit statuses and retrievable `auth.txt` evidence were exported for Days 1–30.

The selected model (`seq-gru-ae-v1-L32-h32-91e4b11d34`) reaches validation average precision 0.0060 (102× the 5.9 × 10⁻⁵ prevalence) and ROC-AUC 0.88. At its max-F1 threshold it gives 37.8 alerts per day with 3.2% precision and 5.0% recall. These are weak standalone user-hour results, reported as found. Test labels were not used.

The detector was fitted on a production run of the Task 2.4 pipeline (all 508,854,306 events reconciled). Checkpoints and full scores are shared outside Git. See [the sequence detector guide](docs/sequence_detector.md), [the downstream handoff](docs/sequence_detector_handoff.md), and the [sequence](data/manifests/lanl_sequences_v1.json) and [selection](data/manifests/sequence_detector_v1.json) manifests.

Task 2.4 shared historical features are implemented. The streaming two-pass pipeline preserves `source_reference` in raw and transformed feature rows. A documented representative pilot processed and reconciled 15.8 million real authentication events; the full 30-day feature output is a separate run and has not been produced. See [the feature guide](docs/lanl_features.md) and [its manifest](data/manifests/lanl_features_v1.json).

Task 2.5 evidence lookup and provenance helpers are implemented. `AuthenticationEvidenceLookup` resolves accepted `auth.txt:<source_line>` references to normalized events and their retained raw record; carrier helpers attach ordered references to sequence and graph evidence and support paging. The checked-in sample demonstrates end-to-end tracing, but it is synthetic sample data rather than a real detector incident. See [the evidence guide](docs/evidence_references.md).

The chronological data split policy established for Task 2.3, including training compromise exclusions, replay boundaries and evaluation units, is:
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

## Incident dashboard (Task 7.1)

Install `requirements.txt`, then launch the read-only analyst dashboard from the repository root:

```powershell
streamlit run scripts/incident_dashboard.py
```

It opens the tracked **Synthetic fixture** (`data/fixtures/incidents_mock.jsonl`) by default. These records and their score and evidence fields are illustrative, not detector results, traceable evidence, or a live feed. Select **Local JSONL export** in the sidebar and enter a local path to view a saved full incident export from `scripts/align_and_fuse_scores.py --incidents-output <path>.jsonl`. The dashboard identifies this as a saved file, not a live feed, and reports missing or malformed files without switching to the fixture. Incident times are dataset-relative seconds, with half-open `[start, end)` ranges. The dashboard consumes packaged incidents only; it does not run detectors. The current Parquet incident export omits nested detector scores, so this dashboard accepts JSONL.

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
