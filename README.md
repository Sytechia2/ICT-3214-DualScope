# DualScope

Multi-timescale intrusion detection using sequence and graph-based behavioural analysis.

This repository is being built from the shared workplans in:

- [DualScope WBS](DualScope_WBS.md)
- [Detailed task implementation guide](DualScope_Detailed_Tasks.md)
- [Scope and deliverables checklist](docs/scope_checklist.md)
- [Development environment setup](docs/environment_setup.md)

## Current status

Task 1.1 foundation files are present; independent second-member setup verification has not yet been recorded. For Task 2.1, the inspected LANL authentication source has 1,051,430,459 records across 58 days. The initial source window is Days 1–30, retaining all 749 supplied red-team rows while reducing authentication input to 508,854,306 records. The selection rules, counts and limitations are recorded in [the dataset manifest](data/manifests/lanl_auth_days_01_30.json).

Train/validation/test boundaries are intentionally not defined by Task 2.1. Red-team labels remain separate from detector inputs, and authentication records without a matching label are unlabelled rather than proven benign. LANL data, trained models, generated outputs and credentials are not stored in Git.

## Quick smoke check

From the repository root, after creating the environment described in `docs/environment_setup.md`:

```powershell
python scripts/smoke_test.py
```

The smoke check uses only the tracked synthetic fixture and does not require LANL data, model checkpoints or external service credentials.

## Planned top-level layout

```text
config/       Shared configuration templates
data/         Tracked fixtures; ignored raw and processed local data
docs/         Scope, setup and technical documentation
logs/         Ignored run logs
models/       Ignored checkpoints and model artifacts
outputs/      Ignored predictions, incidents and reports
scripts/      Repeatable developer and pipeline commands
src/          Application packages
tests/        Automated tests
```
