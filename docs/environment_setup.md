# Development environment setup

This document implements the shared-environment requirements in Task 1.1 of both workplans.

## Prerequisites

- Python 3.11, 3.12 or 3.13
- Git
- Access to the shared repository

The repository currently has no dependency on LANL data for the setup smoke check.

## Create a local environment

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the machine uses a different Python launcher, create an equivalent Python 3.11+ virtual environment. Do not commit `.venv`.

## Create local configuration

```powershell
Copy-Item config/config.example.toml config/config.toml
```

Update only local paths or non-secret settings. Credentials must be supplied through environment variables, such as `DUALSCOPE_LLM_API_KEY`, and must never be placed in tracked files.

The configured local paths are:

- `data/raw`: downloaded LANL records; ignored by Git
- `data/processed`: normalised data and feature artifacts; ignored by Git
- `models`: checkpoints; ignored by Git
- `outputs`: predictions, incidents and generated reports; ignored by Git
- `logs`: run logs; ignored by Git

## Run the smoke check

```powershell
python scripts/smoke_test.py
```

The check loads `data/fixtures/authentication_fixture.csv` and verifies successful and failed authentication, a new destination, repeated timestamps and chronological ordering.

## Second-member verification

A second member should clone or pull the repository, follow the commands above, and run the smoke check using only this document. Record the date, Python version and result in the team tracker. This is the required independent setup verification for Task 1.1.

## Dependency changes

Each component owner adds required packages to `requirements.txt`, records why they are needed, and reruns the smoke check. Keep model-specific, retrieval and dashboard packages documented before the first complete pipeline run.
