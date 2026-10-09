# Pipeline runner (Task 8.1)

`scripts/run_pipeline.py` runs the whole system from log files to dashboard incidents in one command. Every stage calls the same documented scripts and library functions that are described in the other guides, so the runner shows that they fit together and gives a reviewer one place to start.

    python scripts/run_pipeline.py --profile sample     # about half a minute, works on a fresh clone
    python scripts/run_pipeline.py --profile full       # the run of record; reuses verified outputs

At the end it prints a table with the status, seconds and peak memory of each stage and the command that opens the dashboard on the result.

## Profiles

Both profiles are plain JSON files, `config/pipeline_sample.json` and `config/pipeline_full.json`. They hold the stage list, paths, days, alert budget, incident ID prefix and reuse rules. Nothing is hard-coded per profile in Python.

| | `sample` | `full` |
|---|---|---|
| Data | The tracked subset in `data/samples/lanl_pipeline_subset` (days 1-9, 260 users, 526,922 events) | The full days 1-30 outputs of record |
| Scored days | 8-9 | 8-16 (validation) |
| Queue | 10 alerts per day, IDs `INC-SMP-...` | 38 alerts per day, IDs `INC-VAL-...`, plus the stored test queue for days 17-30 |
| Investigations | Evidence packages, ATT&CK retrieval and prompts only (dry run) | The stored Gemini run for the test queue, plus a dry run for the validation queue |
| Run directory | `outputs/pipeline/sample` | `outputs/pipeline/full` |

`--run-dir DIR` chooses another run directory. `--list-stages` prints the stages of a profile.

## Stages

Each stage writes into its own folder of the run directory, logs to `<run>/logs/<stage>.log` and to the console, and records itself in `<run>/pipeline_state.json`.

### sample

| Stage | Reads | Writes | What runs |
|---|---|---|---|
| `ingest` | `auth_subset.txt.gz`, `redteam_subset.txt`, `line_map.txt.gz` (hashes checked against the subset manifest first) | `ingested/` | `scripts/ingest_lanl.py ... --line-map ... --day-end 9`, one worker |
| `features` | `ingested/authentication/events` | `features/` | `scripts/build_lanl_features.py --raw-only`, then the release `preprocessing.json` replaces the one fitted on the subset, then `--transform-only` |
| `score` | `features/`, `models/release` | `scores/scores.parquet`, `scores_summary.json` | `score_days` for days 8 and 9 with the frozen GRU and fusion model |
| `alerts` | `scores/`, `features/`, red-team labels from `ingested/` | `alerts/` (the four-file alert package) | `build_alert_package`, 10 per day |
| `investigate` | `alerts/` | `investigations/` | `scripts/run_investigations.py --dry-run` (or the real run with `--llm`) |
| `dashboard` | `alerts/`, `investigations/` | `dashboard/check.json` | Headless check, see below |

The release `preprocessing.json` is swapped in because the frozen GRU was trained on inputs scaled with that exact file. Fitting a new file on a 260-user subset would give different scaling, and the scoring library refuses a build whose preprocessing hash differs from the release. The subset-fitted file stays next to it as `features/preprocessing_fitted_on_subset.json`.

### full

| Stage | Reuse rule | Otherwise |
|---|---|---|
| `ingest` | Reuse `data/processed/lanl_auth_days_01_30` when its `authentication/summary.json` shows days 1-30, reconciled and 508,854,306 rows | Run `ingest_lanl.py` on `dataset/auth.txt.gz` and `dataset/redteam.txt.gz` (`--auth` and `--redteam` choose other files) with 4 workers into `<run>/ingested` |
| `features` | Reuse `data/processed/lanl_features_v2_days_01_30` when its `preprocessing.json` hash equals the release file and its summary shows 508,854,306 events | Run `build_lanl_features.py --workers 2 --assembly-workers 1 --low-memory-read` into `<run>/features` |
| `score` | Never reused | Score validation days 8-16 on CPU with the frozen models, then run the day-16 reproduction check |
| `alerts` | Never reused | Validation queue, 38 per day, into `<run>/alerts_validation` |
| `test_package` | Reuse `outputs/handoff/final_test_alerts_v1` when its manifest shows 532 alerts, 497 incidents and 1 red-team alert | If the folder is missing, run `scripts/export_alert_handoff.py`, which rebuilds the package from the stored test scores. A folder with different counts makes the stage fail and is left unchanged |
| `investigate` | Reuse `outputs/investigations/gemini_v1` when its `summary.json` shows 497 incidents with every reply ok | With `--llm`, run `run_investigations.py` into that folder (resumable). Without `--llm` and without a complete run the stage is skipped. A dry-run investigation of the validation queue goes into `<run>/investigations_validation` in both cases |
| `dashboard` | Never reused | The same headless check on the test package with `gemini_v1`, and on the validation package with its dry-run folder |

A rebuild never writes into `data/processed/` or the other outputs of record. It goes to the run directory.

The day-16 reproduction check compares the fresh CPU scoring with `outputs/experiment_v2/units_A.parquet` and `fusion_counts.parquet`, which were scored on a GPU. The user-hour sets must match, the hourly counts must be equal, the largest GRU difference must stay within 5e-3 (float32 differences between GPU and CPU reach about 3e-3), and the 38 alerts of day 16 must be the same set. The result, including the largest and median difference, is saved in `scores/reproduction_check.json`. The stage fails if any of these differ.

### The dashboard check

The `dashboard` stage opens each alert package the way the dashboard does (`load_incidents`, `load_events`, `summarise_incidents` and `investigation.load_run`) without a browser. It fails unless every incident loads and every event reference of every incident exists in `events.parquet`. `dashboard/check.json` records the counts: incidents, alert hours, events, references, incidents with ATT&CK candidates, red-team incidents and, for each investigation mode, how many incidents have a finished summary, are pending, failed or were not generated. A folder holding only dry-run files counts as not generated, and the dashboard shows "No AI summary for this data" for it.

## Rerunning

A stage that finished (`done`) and whose inputs are unchanged is skipped on the next run. Its inputs are its configuration, the hashes of its small input files and the finish marker of the stages it needs, so a rerun of an earlier stage makes every later stage run again.

    python scripts/run_pipeline.py --profile sample --from features      # features and everything after it
    python scripts/run_pipeline.py --profile sample --only investigate   # just that stage
    python scripts/run_pipeline.py --profile sample --force              # everything, even when up to date
    python scripts/run_pipeline.py --profile sample --only score --force

Stages named by `--only`, or at and after `--from`, always run. `--force` also runs stages that are up to date. With `--only`, the stages it needs must already have finished once. A stage whose output was deleted counts as not up to date.

A stage with status `reused` used a verified output that already existed. Reuse is checked again on every run, which is cheap, and it never copies or changes the existing output.

## Failures

* If the `investigate` stage fails (no gcloud, timeouts, replies that are not usable), the runner marks it failed, leaves every earlier output untouched, still runs the dashboard check and exits with a non-zero code. The alerts and their evidence stay viewable. The summary prints the command to resume: `python scripts/run_pipeline.py --profile sample --only investigate --llm`. Generation resumes where it stopped, and failed replies are retried.
* If any other stage fails, the stages that need it are skipped. The summary names the failed stage and the command to rerun from it with `--from`.
* A stage that cannot run in this invocation, such as the full-profile investigation without `--llm` and without a complete stored run, is `skipped` with a reason and does not make the run fail.
* Test days are refused. A profile that lists a day from 17 to 30, or an alert split named `test`, stops the runner before any stage starts, and the scoring library refuses them again on its own. Days 17-30 only ever come from the stored one-time test.

## LLM calls

The default investigation stage makes no model calls. `--llm` runs `scripts/run_investigations.py` for real. It needs a working `gcloud` login and sends requests to Vertex AI, so it is opt-in. See [genai_investigation.md](genai_investigation.md).

## State and logs

`<run>/pipeline_state.json` holds one record per stage:

* `status` (`done`, `reused`, `skipped` or `failed`), `reason`, `started_utc`, `wall_seconds` and `error`
* `peak_memory_mb` and `memory_method`
* `commands` (the command lines of scripts it ran) and `functions` (library functions it called)
* `inputs` and `outputs` with paths and sha256 for files up to 64 MB, size and time for larger files, and file counts and sizes for folders
* `fingerprint`, and `completed`, the last successful run, which the next invocation compares against

Peak memory is the highest resident memory (working set on Windows) seen by a sampler that checks every 0.2 seconds. For stages that run scripts (`ingest`, `features`, `investigate`) it is the sum over the script's process tree. For stages that run in the runner process (`score`, `alerts`, `dashboard`) it is the runner process itself, which includes the Python interpreter and every library loaded by earlier stages in the same invocation, so it is a peak and not the stage's own increase.

## Opening the dashboard

    streamlit run scripts/incident_dashboard.py -- --incidents outputs/pipeline/sample/alerts/incidents.jsonl --investigations outputs/pipeline/sample/investigations

The two options can also be set as the environment variables `DUALSCOPE_INCIDENTS` and `DUALSCOPE_INVESTIGATIONS`. With a custom package open, the source in the sidebar reads "Alert package (this run)", the queue size comes from the package manifest and the Investigation tab reads the folder given (no folder means no summaries). Without them the dashboard opens `outputs/handoff/final_test_alerts_v1` and `outputs/investigations/gemini_v1` as before.

## What the results mean

* Scores from the sample differ from the full run. Features that use other users' history on the same computers see only the 260 selected users in the subset. Use the sample to check that the pipeline works, not to report accuracy.
* Days 8-9 (sample) and days 8-16 (full validation queue) were the fusion model's fitting days. Red-team catches in these queues are in-sample and are not an accuracy result. The alert manifests say so under `pipeline`.
* The sample and validation packages carry the red-team answer key (`ground_truth_redteam`) for evaluation, hidden by default in the dashboard and never sent to the model.
* Days 17-30 come only from the stored one-time test. The runner never scores them.

## Expected run times

Measured on the development laptop (CPU only, 31 GB RAM). A later document gives the full runtime breakdown.

Sample profile, from an empty run directory (about 32 seconds in total):

| Stage | Seconds | Peak MB |
|---|---|---|
| ingest | 11.0 | 738 |
| features | 15.9 | 496 |
| score | 0.8 | 821 |
| alerts | 0.2 | 863 |
| investigate (dry run) | 3.9 | 187 |
| dashboard check | 0.5 | 864 |

A second run skips all six stages. The peak for `score`, `alerts` and `dashboard` is the runner process, which also holds what earlier stages loaded.

Full profile with ingestion, features, the test package and the Gemini run reused (about 8 minutes in total):

| Stage | Seconds | Peak MB |
|---|---|---|
| ingest, features, test_package | under 1 each (checks only) | - |
| score (days 8-16 on CPU, plus the day-16 check) | 432.2 | 8,788 |
| alerts (342 alerts, 309 incidents, 131,288 events) | 16.6 | 5,418 |
| investigate (reuse check and validation dry run) | 10.6 | 309 |
| dashboard check (test and validation packages) | 10.1 | 3,980 |

Day-16 reproduction: the user-hour sets and counts are equal, the 38 alerts are the same set, the largest GRU difference to the stored GPU scores is 2.86e-3 and the median is 7.7e-5.

Rebuilding the reused parts is not measured here. The feature build of record took 10,674 seconds (about 3 hours) with `--workers 2`, and the sequential estimate is about 4 hours ([lanl_features.md](lanl_features.md)). The Gemini run makes one request per incident and generation mode, so about 1,000 requests for the 497 test incidents.
