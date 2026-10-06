# Scope and deliverables checklist

This checklist captures the detailed-guide Task 1.1 decision. It should be reviewed by the team against the module brief before implementation choices are frozen.

## Core prototype

Status as of 2026-10-06 (ticked from the code on `main`).

- [x] Authentication records are the core input source.
- [x] A reproducible LANL subset is selected and documented (Days 1–30).
- [x] Chronological training, validation and test periods are defined.
- [x] A short-term sequence detector is trained and evaluated.
- [x] A long-term authentication-graph detector is trained and evaluated, including imbalanced metrics and a chronological overfitting check.
- [x] A flat-feature Isolation Forest baseline is included ([model_comparison_matrix.md](model_comparison_matrix.md)).
- [x] Maximum, average, weighted and temporal fusion are compared, plus supervised fusion ([model_comparison_matrix.md](model_comparison_matrix.md)).
- [x] Incidents retain detector score breakdowns and source evidence references.
- [ ] ATT&CK retrieval, investigation generation and evidence verification are compared in three modes: direct, RAG and RAG with verification.
- [ ] The dashboard supports incident navigation, detector evidence and investigation status.
- [ ] Detection performance and investigation reliability are evaluated separately. (Detection done, including the one-time test; investigation reliability waits for the investigation component.)

## Optional extensions

These are not completion blockers and should begin only after the core pipeline is working and the evaluation schedule is safe:

- [ ] Process-event integration.
- [ ] Network-flow integration.
- [ ] DNS integration.
- [ ] Separate evaluation of any added source.

## Team decisions still required

- [ ] Read the module brief and record actual submission files, marking criteria, page limits and presentation requirements.
- [ ] Replace provisional Member 1–6 labels with names.
- [ ] Nominate the report coordinator and accountable submitter.
- [x] Agree on the initial dataset subset and compute budget.
- [x] Agree on the primary scoring unit, acting-user definition and shared schemas (acting user × hour).
- [ ] Record what scope is reduced first if runtime is too high; the workplans recommend reducing data or model size and deferring optional sources while retaining both detectors and the baseline.

## Acceptance

Task 1.1 is complete when the team has reviewed this checklist against the module brief, filled the open decisions, and agreed that the core/optional boundary and deliverables are understood.
