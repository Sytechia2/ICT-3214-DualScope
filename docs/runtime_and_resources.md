# Runtime and resource use (Task 8.3)

This page tells you how long each part of DualScope takes and what it needs, so you can choose what to run. The setup steps are in [user_manual.md](user_manual.md).

Short version: everything a reviewer needs to check the code runs from the submission alone in about ten minutes (setup, tests, sample pipeline, dashboard). The larger runs need the LANL download or Google Cloud access, and their times are given so you can decide whether to run them.

## What to expect

All times were measured on the development laptop described below unless the last column says otherwise. A slower machine will take proportionally longer. Disk is the extra space the step writes.

| What you run | Needs | Time | Peak memory | Disk | How obtained |
|---|---|---|---|---|---|
| **Setup:** `pip install -r requirements.txt` into a new venv | Python 3.11 to 3.13, internet | 5.5 minutes | low | 1.2 GB venv | Measured |
| **Smoke check:** `python scripts/smoke_test.py` | Setup only | under 1 second | low | none | Measured |
| **Tests:** `python -m pytest -q` | Setup only | 1.6 minutes (1.0 minute with `-m "not slow"`) | about 1 GB | none | Measured |
| **Sample pipeline:** `python scripts/run_pipeline.py --profile sample` | Setup only (the 4 MB subset is in the repository) | 35 seconds of stages, about 43 seconds in all; a rerun takes 6.5 seconds | 0.9 GB | 38 MB | Measured, twice |
| **Dashboard on the sample run** | Sample pipeline done | starts in 1.7 seconds; the first page renders in about 4 seconds | 0.2 to 0.3 GB | none | Measured |
| **Dashboard on the shared final-test package** | The alert and Gemini folders from the team drive (about 45 MB) | first page renders in about 8 seconds | 0.3 GB | none | Measured |
| **Full pipeline, reusing the outputs of record:** `--profile full` | The processed data, features, scores and stored runs of the development machine (about 33 GB, not in the submission) | 8 minutes | 8.8 GB | 87 MB | Measured |
| **Full pipeline from raw data:** `--profile full --auth ... --redteam ...` | Download of `auth.txt.gz` (7.6 GB) and `redteam.txt.gz`; 32 GB RAM; about 80 GB free disk | about 4 hours with extracted text and the ingestion inspection file; about 7 hours with the gzip file | close to all of 31 GB during the feature build | about 40 GB new, plus 7.6 GB download and 34.9 GB if you extract it | Ingestion and features from the logs of the original build; the rest measured; the single-worker figure is an estimate |
| **LLM investigations for 497 incidents:** `--llm` | Google Cloud login with Vertex AI access | about 66 minutes with 8 workers | low | 38 MB | From the run log |
| **Retraining the GRU** (not part of the runner) | Full feature build | about 7.5 minutes of CPU training | not recorded | small | From the training log |

Which of these a reviewer can run from the submission alone: setup, smoke check, tests, sample pipeline and the dashboard on the sample run. The dashboard on the final-test package also needs the two folders from the team drive (see the manual). The full pipeline and the LLM run need data or credentials that are not in the submission. Their stored results are included so that nothing has to be rerun to see them.

## Hardware and software used

| | |
|---|---|
| Machine | Laptop, AMD Ryzen AI 9 HX 370 (12 cores, 24 threads), 31.1 GB RAM |
| System | Windows 11 Pro |
| Python | 3.13.11 (the project supports 3.11 to 3.13) |
| Main packages | torch 2.11.0+cu128 (run on the CPU; PyTorch does not see the GPU), scikit-learn 1.9.1, numpy 2.5.3, pandas 2.3.3, pyarrow 21.0.0, streamlit 1.65.0 |
| Disk | About 98 GB free before the measurements |

No GPU is needed or used by any script in the runner. The setup measurement used a new venv and let pip pick the newest allowed versions (torch 2.14.1, CPU build). The sample pipeline ran in that venv too, in 51 seconds on its first run, with a peak of 0.55 GB. The numbers in this document otherwise come from the development venv, which has the larger CUDA build of torch (4.9 GB instead of 1.2 GB).

## Data sizes

| | Sample subset | Full data used |
|---|---|---|
| Source | `data/samples/lanl_pipeline_subset` (4.1 MB) | LANL authentication data, days 1 to 30 of 58 |
| Events | 526,922 | 508,854,306 |
| Days | 1 to 9 | 1 to 30 (train 1 to 7, validation 8 to 16, test 17 to 30) |
| Users | 260 selected users | all users in the log |
| Red-team rows | 296 | 749 |
| User-hours scored per day | 2,868 (days 8 and 9, 5,736 in all) | about 330,000 to 460,000 |
| Alerts produced | 20 alert hours, 12 incidents, 3,119 events | validation 342 alerts and 309 incidents; test 532 alerts, 497 incidents, 106,730 events |

The scored user-hours of days 17 to 30 total 5,537,311. Days 13 to 16 hold 1,776,074.

Files on disk:

| Item | Size |
|---|---|
| Raw `auth.txt.gz` (all 58 days) | 7.6 GB |
| Raw `auth.txt`, days 1 to 30 only | 34.9 GB (all 58 days: 73.4 GB) |
| Ingested Parquet, `data/processed/lanl_auth_days_01_30` | 9.5 GB |
| Feature build, `data/processed/lanl_features_v2_days_01_30` | 23 GB |
| Graph scores, `outputs/graph_scores_v1` | 284 MB |
| Final-test scores, `outputs/final_test` | 225 MB |
| Final-test alert package, `outputs/handoff/final_test_alerts_v1` | 6.4 MB |
| Gemini run, `outputs/investigations/gemini_v1` | 38 MB |
| Frozen release models, `models/release` | 265 KB |
| Development venv (CUDA torch) | 4.9 GB |
| New venv (CPU torch) | 1.2 GB |

The raw sizes come from the inspection file `data/manifests/lanl_auth_inspection.json` (exact byte counts per day range).

## Sample pipeline in detail

Command: `python scripts/run_pipeline.py --profile sample` into an empty run directory. Two runs on 2026-10-09 gave:

| Stage | Run 1 (s) | Run 2 (s) | Peak MB |
|---|---:|---:|---:|
| ingest | 12.7 | 12.6 | 738 to 740 |
| features | 17.7 | 17.1 | 490 to 496 |
| score | 0.9 | 0.8 | 817 to 821 |
| alerts | 0.2 | 0.2 | 865 to 868 |
| investigate (dry run) | 3.8 | 3.6 | 188 |
| dashboard check | 0.6 | 0.5 | 881 |
| Total of stages | 35.9 | 34.8 | 0.9 GB |
| Wall clock, including interpreter start | 42.9 | 42.0 | |

A second invocation with the same run directory skips all six stages and takes 6.5 seconds. The run directory is 38 MB. `docs/pipeline_runner.md` lists an earlier run of 32 seconds; the difference is normal variation between runs.

The ingest stage runs with one worker at about 48,000 events per second (526,922 events in 11 to 13 seconds including start-up). This rate is the basis for the single-worker estimate for the full data below.

## Full pipeline in detail

### Reusing the outputs of record (`--profile full`)

Measured by the runner (also listed in [pipeline_runner.md](pipeline_runner.md)). The checks in the first three rows only read summaries and hashes.

| Stage | Seconds | Peak MB | How obtained |
|---|---:|---:|---|
| ingest (reuse check) | under 1 | none | Measured |
| features (reuse check) | under 1 | none | Measured |
| score, days 8 to 16 on the CPU, with the day-16 reproduction check | 432.2 | 8,788 | Measured |
| alerts, 342 alerts, 309 incidents, 131,288 events | 16.6 | 5,418 | Measured |
| test_package (reuse check) | under 1 | none | Measured |
| investigate (reuse check and validation dry run) | 10.6 | 309 | Measured |
| dashboard check on both packages | 10.1 | 3,980 | Measured |
| Total | about 8 minutes | 8.8 GB | |

The score stage dominates. It reads the full feature build for nine days and runs the GRU on about 3.7 million user-hours. Per day it needs 30 to 45 seconds for the GRU and about 13 seconds for the hourly counts (day 16 check: GRU 37.6 s, counts 13.1 s).

### Rebuilding from the raw files

None of this was rerun for this page, because the parts take hours. The figures are from the logs of the original build.

| Stage | Time | Workers | How obtained |
|---|---|---|---|
| ingest, days 1 to 30 | 48.7 minutes (2,924 s) | 4 | `progress.json` of the ingestion records start 2026-09-22 07:31:07 UTC and end 08:19:51 UTC |
| ingest, one worker on `auth.txt.gz` | about 3 hours | 1 | Estimate: 508.9 million events at 48,000 per second is 10,600 seconds. Not run |
| features, Pass 1 (user shards, host novelty, preprocessing) | 6,797.7 s (113 minutes) | 2 (`--low-memory-read`) | `logs/build_v2_01_30.log` |
| features, Pass 2 (assemble and transform 30 days) | 3,875.4 s (65 minutes) | 2, 1 assembly worker | `logs/build_v2_01_30.log` |
| features, total | 10,674 s (about 3 hours) | | Sum of the two passes |
| score, alerts, checks | about 8 minutes | | Same as above |
| Total with 4-worker ingestion | about 4 hours | | Sum |
| Total with 1-worker ingestion | about 7 hours | | Sum, using the estimate above |

Memory during the feature build: the memory log `logs/build_v2_01_30_memory.log` records available memory every minute on the 31 GB machine. It fell as low as 0.3 GB, so the build used nearly all of it. Do not start it on a machine with less than 32 GB.

Two points about starting from the raw file:

* The 4-worker ingestion seeks by byte offset, so it needs `auth.txt` extracted (34.9 GB for days 1 to 30) and the inspection file `data/manifests/lanl_auth_inspection.json`, which the full profile passes automatically. The inspection itself ran in 329 seconds on the original file.
* `auth.txt.gz` can be read only with `--workers 1`. The runner passes the same `--workers` value to the feature build, which then also runs with one worker (about 4 hours by the estimate in [lanl_features.md](lanl_features.md)).

A rebuild writes into the run directory and never changes `data/processed/` or the outputs of record.

### One-time test (not repeatable)

The final test on days 17 to 30 scored 5,537,311 user-hours once, on 2026-10-06 (GRU 28 to 57 seconds per day, 427 seconds cumulative; hourly counts 7 to 11 seconds per day). It must not be rerun and the runner refuses to score these days. The alert package export from the stored scores takes about 30 seconds ([alert_handoff.md](alert_handoff.md)).

### Training times

The runner never trains. For reference, from `logs/final_test/`:

| Step | Time |
|---|---|
| GRU run A: sampling 95 s, then 8 epochs at 44 s each (356 s) | about 7.5 minutes on the CPU |
| Fusion model: seven-seed stability check on days 8 to 16 | 139 seconds |

## LLM cost

The stored run is `outputs/investigations/gemini_v1`: model `gemini-3.8-flash` on Vertex AI, thinking level MEDIUM, 497 incidents, two generation modes. The third mode (`rag_verified`) makes no model call.

| | direct (AI only) | rag (AI + ATT&CK) | Both |
|---|---:|---:|---:|
| Replies, all ok | 497 | 497 | 994 |
| Prompt tokens | 2,835,127 | 3,279,222 | 6,114,349 |
| Output tokens | 641,750 | 639,317 | 1,281,067 |
| Thinking tokens | 1,366,901 | 1,362,706 | 2,729,607 |
| Total tokens | 4,843,778 | 5,281,245 | 10,125,023 |
| Mean latency per reply | 29.5 s | 32.3 s | |
| Wall time of the mode | 1,867 s (31 minutes) | 2,065 s (34 minutes) | about 3,930 s (66 minutes) |

The wall times come from the run log (`gemini_v1_run.log`). The log does not record the worker count. The numbers fit 8 workers: 497 replies at 29.5 s divided by 1,867 s is 7.9, and 497 at 32.3 s divided by 2,065 s is 7.8.

Per incident, both modes together: about 12,300 prompt tokens, 2,600 output tokens and 5,500 thinking tokens, in all about 20,400 tokens and two model calls. Run one at a time, one incident takes about one minute (30 s per call). Run with 8 workers, the whole queue takes about 66 minutes. The sample run has 12 incidents, so a real run with `--llm` is 24 calls, about 245,000 tokens and an estimated two minutes (estimate from the per-incident figures). This page gives tokens and not money, because the price depends on the account and the current rate card.

The default `investigate` stage is a dry run. It builds the evidence packages and prompts and takes 4 seconds on the sample.

## Fits the schedule?

Yes. The final test is done and must not be repeated. What remains is rerunning the sample pipeline (under a minute), the full pipeline in reuse mode (8 minutes) and producing report figures from stored outputs. All of these take minutes. The only long steps (the 3-hour feature build, the 66-minute Gemini run and the 49-minute ingestion) are finished and stored, and no part of the plan needs them again. Reviewers who want to repeat them have the times above to plan with.

## How memory was measured

The runner samples the resident memory (working set on Windows) every 0.2 seconds and keeps the highest value. For stages that start a script (`ingest`, `features`, `investigate`) it adds up the whole process tree. For stages that run inside the runner (`score`, `alerts`, `dashboard`) it is the runner process itself, which also holds everything earlier stages loaded, so it is a peak for the process and not the increase from one stage. Values are in each run's `pipeline_state.json` (`peak_memory_mb`, `memory_method`). The dashboard figures were taken from the resident memory of a Streamlit test session after the first page had rendered. The feature build memory is the available-memory log of the original build. Times for the pip install, the tests and the dashboard start were taken with the shell clock.

## Not measured

* Download size and time for the LANL files. They depend on the connection.
* Peak memory of the test suite as a separate number. It is mostly the sample run inside the slow test, so about 1 GB.
* Ingestion with one worker on the full file, and any rebuild on another machine. These are estimates, marked above.
* The clean-machine check of the manual (Task 8.4) on a computer other than this one.
