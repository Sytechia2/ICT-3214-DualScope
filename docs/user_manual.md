# DualScope user manual (draft for Tasks 8.4 and 10.2)

DualScope ranks the most unusual user-hours in Windows login logs from the LANL dataset, groups them into incidents, and shows each incident to an analyst with its evidence and an optional AI summary. This manual takes you from a fresh copy of the code to a working dashboard.

How long each step takes and what it needs is in [runtime_and_resources.md](runtime_and_resources.md). Read its first table before you choose what to run.

Commands are given for Windows PowerShell first. Where macOS or Linux differ, the line is marked. Commands marked **not verified here** need things that were not available when this manual was written (Google Cloud access, the full raw data, another operating system).

## 1. What you can do, and what you need

| Goal | Needs | Time |
|---|---|---|
| Install, run the tests, run the sample pipeline, open the dashboard on the sample | Only this repository | About 15 minutes |
| Open the dashboard on the final-test alerts and stored AI summaries | Two folders from the team drive (section 5) | A minute on top |
| Rerun the full pipeline from the raw LANL files | The LANL download (7.6 GB), 32 GB RAM, about 80 GB disk | 4 to 7 hours |
| Make new AI summaries | A Google Cloud login with Vertex AI access | About 1 hour for all 497 incidents |

## 2. Prerequisites

* Python 3.11, 3.12 or 3.13. Python 3.14 is not supported (`pyproject.toml` requires below 3.14).
* Git, if you clone. A zip of the repository works too.
* About 2 GB of free disk for the sample route (1.2 GB for the packages plus the repository). The other routes need more, see the table above.
* An internet connection for the package install.
* 4 GB of RAM is enough for the sample route. The sample pipeline peaks at 0.9 GB.

## 3. Install

From the folder that holds the repository (the one that contains `README.md`):

```powershell
# Windows PowerShell
git clone <repository-url> ICT-3214-DualScope      # or unzip the submission
cd ICT-3214-DualScope
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
# macOS or Linux (not verified here)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` and try again. If your default `python` is 3.14 or older than 3.11, create the venv with a supported interpreter, for example `py -3.13 -m venv .venv` on Windows.

The install took 5.5 minutes on the development laptop. It is mostly PyTorch. Nothing else has to be installed: the scripts add `src/` to the Python path themselves, and the tests do the same through `pyproject.toml`, so `pip install -e .` is not needed. Run every command from the repository root.

No GPU is needed. The code runs on the CPU.

Check the install:

```powershell
python scripts/smoke_test.py
```

It should end with `Smoke check passed.`

## 4. Run the tests

```powershell
python -m pytest -q                  # all tests, about 1.6 minutes
python -m pytest -q -m "not slow"    # skips the one test that runs the sample pipeline, about 1 minute
```

Expect all tests to pass with one skipped (`CUDA is not available`, which is normal on a machine without an NVIDIA GPU). pytest may print a warning that it could not write its cache; this is harmless.

## 5. Run the sample pipeline

The sample pipeline takes a small slice of the LANL log that is stored in the repository (`data/samples/lanl_pipeline_subset`, 526,922 events of days 1 to 9) through ingestion, feature building, scoring with the frozen model, alert selection, investigation evidence and a dashboard check.

```powershell
python scripts/run_pipeline.py --profile sample
```

It takes about 45 seconds. It prints one line per stage and a summary table like this:

```text
stage        status  seconds  peak MB  note
ingest       done    12.7     738
features     done    17.7     490
score        done    0.9      817
alerts       done    0.2      865
investigate  done    3.8      188
dashboard    done    0.6      881

All selected stages finished.
```

The last lines of the output give the command that opens the dashboard on this run. Running the same command again skips every stage that is up to date (about 7 seconds). Use `--force` to run everything again, or `--from features` to start at one stage. Details are in [pipeline_runner.md](pipeline_runner.md).

Everything goes into `outputs/pipeline/sample`. The run is a check that the parts fit together. Its scores are not an accuracy result, because only 260 users are in the slice.

The investigation stage in this run is a dry run. It builds the evidence packages and prompts and makes no AI calls. The dashboard therefore shows "No AI summary for this data" for the sample, which is expected.

## 6. Open the dashboard

### On the sample run

```powershell
streamlit run scripts/incident_dashboard.py -- --incidents outputs/pipeline/sample/alerts/incidents.jsonl --investigations outputs/pipeline/sample/investigations
```

Streamlit starts in a couple of seconds and opens `http://localhost:8501` in your browser. If it does not, open that address yourself. Stop it with Ctrl+C. The two folders can also be given as the environment variables `DUALSCOPE_INCIDENTS` and `DUALSCOPE_INVESTIGATIONS`. Keep the space after the lone `--`; it separates Streamlit's options from the dashboard's.

If the `streamlit` command is not found, the virtual environment is not active. Activate it, or use `python -m streamlit run ...`.

### On the final-test alerts with the stored AI summaries

These are the alerts of days 17 to 30 and the stored Gemini summaries. They are too large for Git, so they come as two zip files from the team drive: `handoff.zip` and `investigations.zip`. Unzip both into the `outputs` folder of the repository, so that these paths exist:

```text
outputs/handoff/final_test_alerts_v1/incidents.jsonl      (with alerts.parquet, events.parquet, manifest.json)
outputs/investigations/gemini_v1/summary.json             (with the other files of that run)
```

Then start the dashboard without options:

```powershell
streamlit run scripts/incident_dashboard.py
```

It opens the final-test package when the folder exists. Otherwise it falls back to a synthetic fixture (labelled "Synthetic fixture" in the sidebar), so check the source label in the sidebar. The first page takes about 8 seconds to render.

### Choosing the data source

The sidebar has an expander called **Data source**. For a package opened with the options above it reads "Alert package (this run)". The default package reads "Alert package (days 17–30)". The other choices are "Synthetic fixture" (invented records for a first look) and "Other JSONL export" (type a path). The sidebar also has an **Answer key** switch. It shows the dataset's red-team labels, which are meant for evaluation. Keep it off while you review incidents as an analyst would.

## 7. Using the dashboard

The dashboard is read-only. It shows what the detector produced. It does not score anything itself.

The pages are in the sidebar: **Queue**, **Incidents**, **Evidence** and **About the model**.

1. **Queue.** Pick a day with the day box. The page lists that day's incidents, the most unusual first, with a summary strip above the list. The queue holds the 38 most unusual user-hours of the day for the final-test package (10 for the sample). Each row is an incident: one user, one flagged hour or a few hours close together. The **HIGH** label means the incident scored above the day's cut-off. **MEDIUM** means it landed exactly on the cut-off with other hours and a tie-break chose which ones to include, so their order means nothing. The search box finds a user, host or event ID. The panel on the right previews the selected incident.
2. **Incidents.** Open an incident from the queue (or choose one in the "Open an incident" box). Use **Previous incident** and **Next incident** to move through the day. The page has five tabs:
   * **Overview**: why it was flagged, with the four simple checks (failed logins, an NTLM login to a new computer, a network login to a new computer, a login from a computer the user has not used before) and the scores. "No rule matched" means the model found the hour unusual and none of the four checks fired.
   * **Timeline (n)**: every cited login event in time order. Select a row to see the full original record and its reference, such as `auth.txt:463603187`.
   * **Connections (n)**: one row per user-to-computer pair, with first-time pairs first.
   * **Model details**: the fusion score (a ranking value, not a probability), the sequence percentile and the graph score. The graph score is shown for background only and does not change the ranking.
   * **Investigation**: the AI summary (section 8).
3. **Evidence.** Type an event ID such as `auth.txt:463603187` to see the raw record and which incidents cite it.
4. **About the model.** The glossary of every term used in the dashboard, and notes on the model.

To get from an incident to raw evidence: open the incident, go to **Timeline**, select an event row, and copy its `auth.txt:<line>` reference. Every claim in an AI summary cites such references, and the **Evidence** page resolves them.

## 8. Reading the AI summary

The **Investigation** tab has a selector with three views of the same incident:

| View | What it is |
|---|---|
| **AI + ATT&CK, checked** | The AI answer with a MITRE ATT&CK candidate list, after the automatic check. Anything the log lines do not back up has been removed. This is the view to read first. |
| **AI + ATT&CK** | The same answer before the check. |
| **AI only** | The AI reading the log lines with no candidate list. |

Each technique carries a verification status from fixed rules that use no AI:

| Status | Meaning |
|---|---|
| **Supported** | The log lines show the behaviour the technique describes. Whether it was an attack is still for you to decide. |
| **Uncertain** | The log lines only partly match, or login records do not carry enough detail to confirm it. |
| **Rejected** | Removed because the technique does not exist, the cited log lines do not exist, or those lines do not show the behaviour. |

Individual sentences can be marked **Partly supported** when some computer names or times in them appear elsewhere in the incident but not in the lines they cite.

**No supported mapping** means none of the known ATT&CK techniques fit the incident. This is a normal result and happens often (147 of the 497 stored incidents). The summary text still reads as an explanation of the events.

The AI states a confidence level itself (Low, Medium, High). The automatic check does not change it. Treat a summary as a lead to check against the log lines, not as a finding.

If the tab says "No AI summary for this data", the folder given for investigations has no real model output. This is the case for the sample. If it says "No AI summary yet", the summary was not generated for that incident yet. A failed generation shows its reason.

## 9. Run the full pipeline

### Reusing the stored results

On the development machine, which holds the processed data and stored runs:

```powershell
python scripts/run_pipeline.py --profile full
```

It reuses the verified ingestion, feature build, test alerts and Gemini run, and rescores the validation days 8 to 16 on the CPU with the frozen models. It takes about 8 minutes and peaks at 8.8 GB of memory. Without `data/processed/lanl_auth_days_01_30` and `data/processed/lanl_features_v2_days_01_30` the first two stages fall through to a rebuild and need the raw files. Not verified on a machine without those folders.

### From the raw files

1. Download `auth.txt.gz` and `redteam.txt.gz` from the LANL site (<https://csr.lanl.gov/data/cyber1/>, "Comprehensive, Multi-Source Cyber-Security Events", DOI 10.17021/1179829). Put them in a folder called `dataset` at the repository root, which Git ignores: `dataset/auth.txt.gz` (7.6 GB) and `dataset/redteam.txt.gz`. These are the default paths of the `full` profile. Any other location works with `--auth` and `--redteam`.
2. Choose how to read the file:
   * Simple route, one worker, slow (about 3 hours for ingestion by estimate; the whole run about 7 hours):

     ```powershell
     python scripts/run_pipeline.py --profile full --workers 1 --auth dataset/auth.txt.gz --redteam dataset/redteam.txt.gz
     ```

   * Fast route, four workers (ingestion in 49 minutes; the whole run about 4 hours): extract the file first (34.9 GB for days 1 to 30, 73.4 GB for the whole file) and the runner uses the tracked ingestion inspection file `data/manifests/lanl_auth_inspection.json` (byte offsets of each day in the official `auth.txt`):

     ```powershell
     python scripts/run_pipeline.py --profile full --auth dataset/auth.txt --redteam dataset/redteam.txt
     ```

   The default `--workers 4` cannot read a `.gz` file. The runner stops with a message from the ingestion script if you try.
3. Needs 32 GB RAM for the feature build and about 80 GB of free disk. The rebuild writes into `outputs/pipeline/full` and never overwrites `data/processed/` or the stored results.

Both routes are **not verified here** because they take hours. The reuse checks and the code path were tested on the development machine. See [lanl_ingestion.md](lanl_ingestion.md) and [lanl_features.md](lanl_features.md) for the underlying scripts.

## 10. Make new AI summaries (optional)

This step calls Google's Gemini model through Vertex AI. The stored run in `outputs/investigations/gemini_v1` already holds the summaries for all 497 test incidents, so you do not need this to use the dashboard. It is **not verified here** for a new account.

```powershell
gcloud auth login
gcloud config set project <your-project>        # or set the environment variable GOOGLE_CLOUD_PROJECT
python scripts/run_pipeline.py --profile sample --only investigate --llm
```

The sample run makes about 24 model calls (12 incidents, two modes), uses about 245,000 tokens and should take a couple of minutes (an estimate from the figures in [runtime_and_resources.md](runtime_and_resources.md)). On the full profile, `--llm` generates the 497 test incidents into `outputs/investigations/gemini_v1` (about 66 minutes, about 10 million tokens). Generation can be stopped and restarted; it carries on where it left off and retries failed replies. No key is stored: the code uses your `gcloud` login. See [genai_investigation.md](genai_investigation.md) for the settings and the way summaries are checked.

Without `--llm` the investigation stage is a dry run that makes no calls.

## 11. Training and inference

The pipeline runner only scores. It never trains. The frozen models are in `models/release/` (265 KB, with a manifest of hashes that is checked before every use):

* `models/release/gru/`: the GRU autoencoder that scores each user-hour.
* `models/release/fusion/`: the gradient boosting model that combines the GRU score with hourly event counts.
* `models/release/features/preprocessing.json`: the feature scaling these models were trained with. A feature build must use this exact file.

To retrain, run the scripts of the earlier stages. They need the full feature build and are not part of the runner:

| Step | Script | Guide |
|---|---|---|
| Time splits | `scripts/create_lanl_splits.py` | [lanl_splits.md](lanl_splits.md) |
| Features | `scripts/build_lanl_features.py` | [lanl_features.md](lanl_features.md), [experiment_features_v2.md](experiment_features_v2.md) |
| Sequence detector (GRU) | `scripts/build_sequences.py`, `scripts/train_sequence_model.py` | [sequence_detector.md](sequence_detector.md) |
| Graph detector | `scripts/build_graph_snapshots.py`, `scripts/train_graph_model.py` | [graph_detector.md](graph_detector.md) |
| Fusion model | `scripts/supervised_fusion_validate.py`, `scripts/supervised_fusion_freeze.py` | [supervised_fusion.md](supervised_fusion.md) |

Training the GRU takes about 7.5 minutes on the CPU once the features exist.

## 12. The test-day rule

Days 17 to 30 were the test period. The final model was tested on them once, on 2026-10-06, and the result is final. They are never scored again. `scripts/supervised_fusion_final_test.py` is the script that did it. **Do not run it.** The pipeline runner refuses a profile that lists a day from 17 to 30, and the scoring library refuses these days on its own. The days 17 to 30 alerts you see in the dashboard come from the stored scores of that one test.

Days 8 to 16 were used to fit the fusion model. Red-team catches in the sample and validation queues are therefore in-sample and are not accuracy figures.

## 13. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `No module named ...` | The virtual environment is not active, or `requirements.txt` was not installed. Activate `.venv` and rerun the install. |
| `streamlit` is not recognised | Same cause. Use `python -m streamlit run ...` after activating. |
| The install fails or picks no torch wheel | Python is 3.14 or older than 3.11. Create the venv with Python 3.11 to 3.13. |
| A hash mismatch message from the release loader or the subset check | A file in `models/release/` or `data/samples/lanl_pipeline_subset/` was changed or checked out with different line endings. Restore it from Git (`git checkout -- models/release data/samples/lanl_pipeline_subset`). Do not edit these files. |
| `raw LANL files not found ...; pass --auth and --redteam` | The full profile cannot reuse stored outputs and has no raw files. Put the files in `dataset/` or pass their paths (section 9). |
| `cannot read gzip input` or a message to use `--workers 1` | A `.gz` file was given with several workers. Add `--workers 1` or extract the file. |
| Missing `data/manifests/lanl_auth_inspection.json` | Needed by the four-worker ingestion. It is tracked in the repository; restore it with `git checkout data/manifests/lanl_auth_inspection.json`. |
| The investigate stage fails with `gcloud is not installed or not on PATH` or an authentication error | Install the Google Cloud CLI and run `gcloud auth login`. The run still finishes the other stages and exits with a non-zero code. The alerts stay viewable in the dashboard. Resume with `--only investigate --llm`. |
| The dashboard shows "Synthetic fixture" | The final-test folders were not found. Check the paths in section 6. |
| The dashboard says "No AI summary for this data" | The investigations folder holds only a dry run. Expected for the sample. |
| Out of memory or the machine swaps | The feature build and the full score stage need a lot of RAM (up to the whole of 31 GB, and 8.8 GB). Close other programs or use the sample profile. |
| `Permission denied` on `.pytest_cache` | Harmless warning. Add `-p no:cacheprovider` to silence it. |
| Port 8501 is taken | `streamlit run ... --server.port 8502` |

Each stage logs to `outputs/pipeline/<run>/logs/<stage>.log`, and `pipeline_state.json` in the run folder records status, time and memory per stage.

## 14. Known limitations

* The final model catches 1 of the 39 attack hours of days 17 to 30 at 38 alerts a day. Its pre-set success rule was not met. See [supervised_fusion.md](supervised_fusion.md).
* Graph results on the test period are retrospective: those days had been looked at during development ([graph_model_evaluation.md](graph_model_evaluation.md)).
* The sample run uses 260 users, so its scores differ from the full run ([pipeline_runner.md](pipeline_runner.md)).
* AI summaries are leads. The check confirms that the log lines are consistent with a technique, not that an attack happened. The model is not deterministic, so a new run gives different text ([genai_investigation.md](genai_investigation.md), [investigation_review.md](investigation_review.md)).
* The fusion score ranks hours. It is not a calibrated probability. Many hours tie at the cut-off ([alert_handoff.md](alert_handoff.md)).
* The dashboard shows alerts that were produced earlier. It does not score new logs.
* Only authentication events are used. Other LANL sources (processes, flows, DNS) are not.
