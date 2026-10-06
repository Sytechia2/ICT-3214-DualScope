# DualScope

Multi-timescale intrusion detection using sequence and graph-based behavioural analysis, on the LANL authentication dataset.

The work plan (WBS and detailed task guide) is kept outside the repository; task numbers below follow it. The scope checklist is [docs/scope_checklist.md](docs/scope_checklist.md) and environment setup is [docs/environment_setup.md](docs/environment_setup.md).

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
- **Evaluation (9.1–9.4):** protocol ([evaluation_protocol.md](docs/evaluation_protocol.md)); comparison of all detectors and fusions on Days 13–16 at the same budget ([model_comparison_matrix.md](docs/model_comparison_matrix.md)); detection timing, precedence and errors with data-driven case studies ([detection_timing_and_errors.md](docs/detection_timing_and_errors.md)); the final model and test ([supervised_fusion.md](docs/supervised_fusion.md), full record in [experiment_features_v2.md](docs/experiment_features_v2.md)). The Isolation Forest baseline (9.2) is implemented but not yet run on LANL.
- **Not yet in the repository:** the ATT&CK retrieval and LLM investigation component (6.x), the dashboard beyond a mock UI (7.x), end-to-end integration (8.x) and investigation reliability evaluation (9.5).

Large data, feature builds, model checkpoints, scores and logs are not stored in Git; each guide says where they come from or how to rebuild them.

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
