# DualScope

Multi-timescale intrusion detection using sequence and graph-based behavioural analysis, on the LANL authentication dataset.

The work plan (WBS and detailed task guide) is kept outside the repository; task numbers below follow it. The scope checklist is [docs/scope_checklist.md](docs/scope_checklist.md) and environment setup is [docs/environment_setup.md](docs/environment_setup.md).

## Quick start

Step-by-step instructions are in [docs/user_manual.md](docs/user_manual.md). Times and resource needs for every run are in [docs/runtime_and_resources.md](docs/runtime_and_resources.md). Needs Python 3.11 to 3.13; no GPU, no LANL download and no credentials.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt      # about 5 minutes (mostly PyTorch)
python -m pytest -q                            # about 1.5 to 2 minutes; add -m "not slow" for about 1 minute
python scripts/run_pipeline.py --profile sample   # about 45 seconds, peak 0.9 GB, writes outputs/pipeline/sample
streamlit run scripts/incident_dashboard.py -- --incidents outputs/pipeline/sample/alerts/incidents.jsonl --investigations outputs/pipeline/sample/investigations   # starts in a few seconds
```

The full pipeline (`--profile full`) reuses stored outputs and takes about 8 minutes on a machine that holds them; rebuilding from the raw LANL files takes 4 to 7 hours. Making new AI summaries (`--llm`) needs a Google Cloud login. Both are optional.

## Current status (2026-10-06)

### Headline result

The final detection model is a **supervised fusion** (gradient boosting on the GRU score and hourly authentication counts), fitted on validation labels and tested once on Days 17–30. See [docs/supervised_fusion.md](docs/supervised_fusion.md).

| Days 13–16 (validation, 136 attack hours) | Caught at 38 alerts/day | AP |
| --- | ---: | ---: |
| GRU alone | 4 | 0.0046 |
| Final model | 16 (13–18 under numeric noise) | 0.050 |

| Days 17–30 (test, run once, 39 attack hours) | Caught at 38 alerts/day | AP |
| --- | ---: | ---: |
| GRU alone | 0 | 0.00020 |
| Final model | 1 | 0.00124 |

The model ranks attacks much better than the GRU on unseen days, but at 38 alerts per day it catches almost none of the test-period attacks; its pre-set primary success rule was not met.

### By work package

- **Data (2.1–2.5):** Days 1–30 of LANL authentication, 508,854,306 events and 749 red-team label rows, normalised with exact reconciliation; chronological split into training (Days 1–7), validation (Days 8–16) and test (Days 17–30); shared causal historical features; `auth.txt:<line>` evidence references throughout. See [lanl_ingestion.md](docs/lanl_ingestion.md), [lanl_splits.md](docs/lanl_splits.md), [lanl_features.md](docs/lanl_features.md), [evidence_references.md](docs/evidence_references.md).
- **Short-term sequence detector (3.1–3.4):** GRU autoencoder over user-hour event sequences, frozen; validation AP 0.0060 (102× prevalence), ROC-AUC 0.88. See [sequence_detector.md](docs/sequence_detector.md) and [sequence_detector_handoff.md](docs/sequence_detector_handoff.md).
- **Long-term graph detector (4.1–4.4):** graph autoencoder over daily user–computer graphs, plus a separate confirmed-relationship signature layer. Its reported test-period results (GAE F1 0.077; signature layer F1 0.84) are retrospective, because the test days had been inspected during development. See [graph_detector.md](docs/graph_detector.md), [graph_model_evaluation.md](docs/graph_model_evaluation.md), [graph_detector_handoff.md](docs/graph_detector_handoff.md).
- **Fusion and incidents (5.1–5.4):** alignment, maximum/average/temporal fusion and incident records with signature tagging and evidence.
- **Evaluation (9.1–9.4):** protocol ([evaluation_protocol.md](docs/evaluation_protocol.md)); comparison of all detectors and fusions on Days 13–16 at the same budget ([model_comparison_matrix.md](docs/model_comparison_matrix.md)); detection timing, precedence and errors with data-driven case studies ([detection_timing_and_errors.md](docs/detection_timing_and_errors.md)); the final model and test ([supervised_fusion.md](docs/supervised_fusion.md), full record in [experiment_features_v2.md](docs/experiment_features_v2.md)). The flat-feature Isolation Forest baseline (9.2) is the weakest detector: 0 of 136 attack hours on Days 13–16 and 0 of 39 on Days 17–30 at 38 alerts/day.
- **End-to-end runner (8.1):** `python scripts/run_pipeline.py --profile sample` takes the tracked subset in `data/samples/lanl_pipeline_subset` through ingestion, features, scoring with the frozen release model in `models/release`, alerts, investigation evidence and a dashboard check; `--profile full` reproduces the run of record and reuses verified outputs. See [docs/pipeline_runner.md](docs/pipeline_runner.md).
- **GenAI investigation, dashboard and reliability review (6.x, 7.x, 9.5):** ATT&CK retrieval, Gemini summaries with an automatic evidence check, the analyst dashboard and the reviewer sample are in the repository. See [docs/genai_investigation.md](docs/genai_investigation.md), [docs/attack_retrieval.md](docs/attack_retrieval.md), [docs/investigation_review.md](docs/investigation_review.md) and the dashboard section below.
- **Runtime and user manual (8.3, 8.4):** [docs/runtime_and_resources.md](docs/runtime_and_resources.md) and [docs/user_manual.md](docs/user_manual.md).

Large data, feature builds, model checkpoints, scores and logs are not stored in Git; each guide says where they come from or how to rebuild them.

## Quick smoke check

From the repository root, after creating the environment described in `docs/environment_setup.md`:

```powershell
python scripts/smoke_test.py
```

The smoke check uses only the tracked synthetic fixture and does not require LANL data, model checkpoints or external service credentials.

## Incident dashboard (Tasks 7.1 and 7.2)

Install `requirements.txt`, then launch the read-only analyst dashboard from the repository root:

```powershell
streamlit run scripts/incident_dashboard.py
```

It opens the tracked **Synthetic fixture** (`data/fixtures/incidents_mock.jsonl`) by default. These records and their score and evidence fields are illustrative, not detector results, traceable evidence, or a live feed. Select **Local JSONL export** in the sidebar and enter a local path to view a saved full incident export from `scripts/align_and_fuse_scores.py --incidents-output <path>.jsonl`. The dashboard identifies this as a saved file, not a live feed, and reports missing or malformed files without switching to the fixture. Incident times are dataset-relative seconds, with half-open `[start, end)` ranges. The dashboard consumes packaged incidents only; it does not run detectors. The current Parquet incident export omits nested detector scores, so this dashboard accepts JSONL.

Select **Alert handoff package** to open the final model's days 17–30 alerts (`outputs/handoff/final_test_alerts_v1/incidents.jsonl`, built by `scripts/export_alert_handoff.py`; see `docs/alert_handoff.md`). An incident's detail view then adds the Task 7.2 evidence panels:

- **Alerted hours:** rank, tie status and scores per hour.
- **Authentication events:** a time-ordered timeline of every cited event; selecting a row shows its full source record.
- **User–host relationships:** one row per source → destination pair, with first-time pairs listed first.
- **Graph detector context:** labelled as not used by the final model.

Event details come from `events.parquet` in the same folder as the incident file; without it, the view lists the event references only. The red-team answer key stays hidden unless **Show answer key (evaluation only)** is switched on in the sidebar.

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
