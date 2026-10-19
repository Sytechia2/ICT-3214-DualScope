# DualScope

DualScope looks for intruders in Windows login logs. It reads the authentication events of the LANL "Comprehensive, Multi-Source Cyber-Security Events" dataset, ranks each user's hours by how unusual they look, and turns the most unusual hours into incidents. An analyst can open each incident in a dashboard, follow it back to the original log lines, and read an AI summary that has been checked against those lines.

## How it works

1. **Ingest.** The raw `auth.txt` log is converted to Parquet. Every event keeps a reference to its line in the original file (`auth.txt:<line>`), so any result can be traced back to the source.
2. **Build features.** Each user's events are grouped by hour and described with counts and history, such as first-time logins to a computer. Days 1–7 are used for training, days 8–16 for validation and days 17–30 for the final test.
3. **Score.** Two detectors look at the data on different time scales. A GRU autoencoder learns the normal order of a user's logins within an hour and scores how far an hour strays from it. A graph autoencoder looks at which users log in to which computers each day. A gradient-boosting model then combines the GRU score with hourly login counts into one ranking score. The graph detector was studied in depth but is not part of the final model.
4. **Raise alerts.** The 38 highest-scoring user-hours of each day become alerts, and alerts close together for the same user are grouped into incidents.
5. **Investigate.** For each incident, Google Gemini writes a summary and suggests matching MITRE ATT&CK techniques. A rule-based check, which uses no AI, then marks each technique as supported, uncertain or rejected, based on whether the cited log lines show that behaviour.
6. **Review.** A Streamlit dashboard shows the daily alert queue, each incident's timeline and connections, and its checked AI summary.

## Results

The final model was fitted on the validation days and tested once on days 17–30. The test days are never scored again.

| Days 13–16 (validation, 136 attack hours) | Attack hours caught at 38 alerts a day | Average precision |
| --- | ---: | ---: |
| GRU alone | 4 | 0.0046 |
| Final model | 16 | 0.050 |

| Days 17–30 (test, 39 attack hours) | Attack hours caught at 38 alerts a day | Average precision |
| --- | ---: | ---: |
| GRU alone | 0 | 0.00020 |
| Final model | 1 | 0.00124 |

The final model ranks attacks well above the GRU on days it has never seen, but at 38 alerts a day it catches only 1 of the 39 attack hours in the test period, so the success target set before the test was not met. An Isolation Forest baseline caught none of the attack hours in either period. The full comparison is in [docs/evaluation/model_comparison_matrix.md](docs/evaluation/model_comparison_matrix.md) and [docs/evaluation/supervised_fusion.md](docs/evaluation/supervised_fusion.md).

## Getting started

You need Python 3.11, 3.12 or 3.13. The quick start below needs no GPU, no LANL download and no cloud account.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                       # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt          # about 5 minutes, mostly PyTorch
python scripts/smoke_test.py                       # quick install check
python -m pytest -q                                # full test suite, a few minutes
python scripts/run_pipeline.py --profile sample    # about 45 seconds
streamlit run scripts/incident_dashboard.py -- --incidents outputs/pipeline/sample/alerts/incidents.jsonl --investigations outputs/pipeline/sample/investigations
```

The sample run takes a small slice of the LANL log that is stored in this repository (526,922 events from 260 users) through every stage, using the frozen models in `models/release/`. It checks that the parts fit together. Its scores are not an accuracy result.

The [user manual](docs/user_manual.md) covers everything else, step by step: opening the dashboard on the final-test alerts and stored AI summaries, rerunning the full pipeline from the raw LANL files, making new AI summaries, retraining the models, and troubleshooting. How long each run takes and how much memory it needs is in [docs/running/runtime_and_resources.md](docs/running/runtime_and_resources.md).

## Documentation

| Topic | Documents |
| --- | --- |
| Using DualScope | [User manual](docs/user_manual.md), [runtime and resources](docs/running/runtime_and_resources.md), [pipeline runner](docs/running/pipeline_runner.md), [environment setup](docs/running/environment_setup.md) |
| Data | [Ingestion](docs/data/lanl_ingestion.md), [time splits](docs/data/lanl_splits.md), [features](docs/data/lanl_features.md), [evidence references](docs/data/evidence_references.md) |
| Detectors | [Sequence detector (GRU)](docs/detectors/sequence_detector.md), [graph detector](docs/detectors/graph_detector.md), [graph evaluation](docs/detectors/graph_model_evaluation.md), [graph overfitting check](docs/detectors/graph_overfitting.md) |
| Final model and evaluation | [Evaluation protocol](docs/evaluation/evaluation_protocol.md), [model comparison](docs/evaluation/model_comparison_matrix.md), [final model](docs/evaluation/supervised_fusion.md), [feature experiment record](docs/evaluation/experiment_features_v2.md), [detection timing and errors](docs/evaluation/detection_timing_and_errors.md) |
| Alerts and investigation | [Alert package](docs/investigation/alert_handoff.md), [ATT&CK retrieval](docs/investigation/attack_retrieval.md), [AI investigation](docs/investigation/genai_investigation.md), [review of the AI summaries](docs/investigation/investigation_review.md) |
| Testing and release | [Failure handling tests](docs/release/interface_failure_tests.md), [release validation](docs/release/release_validation.md) |

## Repository layout

```text
config/     Configuration template
data/       Small samples, fixtures, manifests and reference data (the full LANL data is not stored here)
docs/       Technical documentation and the user manual
models/     Frozen release models in models/release/, with a manifest of file hashes
scripts/    Command-line entry points: pipeline runner, training, evaluation, dashboard
src/        The dualscope Python package
tests/      Automated tests
```

The full LANL data, the processed data, intermediate model runs, scores and logs are too large for Git. `outputs/`, `logs/` and most of `data/` and `models/` are ignored. The larger data and results are kept in a [shared Google Drive folder](https://drive.google.com/drive/folders/1uccDsuaxZahnTwdyiIGJIW2MzZ1dBQdq?usp=sharing). The file `google_drive_contents.txt`, which comes with the submission, lists each file and where it goes. The user manual explains when each one is needed and how to rebuild it.

## Limitations

* At 38 alerts a day the final model catches 1 of 39 attack hours in the test period.
* The final model was fitted on labelled attacks, which a real organisation would not normally have, and it would need retraining for another network.
* Graph detector results on the test period are retrospective, because those days had been looked at during development.
* AI summaries are starting points for an analyst. The check confirms that the log lines match a technique, and the analyst still decides whether it was an attack.
* Only authentication events are used. The dataset's process, network flow and DNS logs are not.

## Data

The dataset is the Los Alamos National Laboratory "Comprehensive, Multi-Source Cyber-Security Events" collection (A. D. Kent, 2015, DOI [10.17021/1179829](https://doi.org/10.17021/1179829)), available from <https://csr.lanl.gov/data/cyber1/>. The small slices stored in this repository are cut from it.
