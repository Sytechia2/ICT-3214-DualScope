# Release candidate validation (Task 8.4)

This record shows that the user manual ([user_manual.md](user_manual.md)) takes a person from a fresh copy of the repository to a working dashboard. The manual was followed in order in a new clone with a new virtual environment.

Release tag: `v1.0-rc1`, on the merge of this work into `main`.

## What was validated

| Item | Value |
|---|---|
| Commit | `9410a87` ("Runtime and resource doc, user manual draft (Tasks 8.3, 8.4)"), with the manual fixes listed below in the working tree |
| Date | 2026-10-09 |
| Machine | Developer laptop, AMD Ryzen AI 9 HX 370, 31.1 GB RAM, Windows 11 Pro |
| Copy | `git clone --branch feature/pipeline-runner` of the local repository into an empty folder, so nothing came from GitHub and no file of the development checkout was reused |
| Virtual environment | New `.venv` inside the clone, Python 3.13.11 (the default `python` on the machine is 3.14, which the project does not support, so the 3.13 interpreter was called by its full path) |
| Packages | Newest versions allowed by `requirements.txt`: torch 2.14.1 (CPU build), scikit-learn 1.9.1, pandas 2.3.3, pyarrow 21.0.0, streamlit 1.65.0, numpy 2.5.3, pytest 8.4.2 |
| Configs | `config/pipeline_sample.json` (sample profile); `config/lanl_features_v2.json`, `config/lanl_splits.json` and `data/manifests/lanl_splits_v1.json` as called by the runner |
| Network use | Package install only. No Google Cloud call was made. |
| Not run | `scripts/supervised_fusion_final_test.py`, any scoring of days 17 to 30, the full profile from raw data |

Release model files (`models/release/manifest.json`), all verified by the loader in the clone:

| File | SHA-256 |
|---|---|
| `gru/checkpoint.pt` | `2f63843ad037b5d54d5a3275797b333f0313ed7d7efb2d5f547e1e919c7ab3ef` |
| `gru/config.json` | `3fb31dff4e708218842533c3e753df8e5496db3c0b4a35181ea0f12b6945f721` |
| `gru/run_manifest.json` | `e2ceb6cacafa28990a9117af2800cfd952f228a2b04ee0a6a6fdb7cc8c1932a3` |
| `fusion/model.joblib` | `02b4e6f7e3d10d80dea0680b2850780cac3b7eb6f7ced49b52fc2a18fb6a554d` |
| `fusion/manifest.json` | `6827e695597f8e08c6d9048df58fe2646c91b338ea37cd0ddb20df9e277a2dac` |
| `features/preprocessing.json` | `9fc82c04606674c6eebc8eecbcf842d90ef6a37e33b08d710a2ab99c801c5736` |

Sample subset files (`data/samples/lanl_pipeline_subset/manifest.json`), all matching after checkout on Windows: `auth_subset.txt.gz` `1686a1e7...f75207d`, `line_map.txt.gz` `3508d4fc...c151e596`, `redteam_subset.txt` `64e6a078...54b44`, `README.md` `130f881d...2b403b`.

## Steps and results

Manual section numbers refer to [user_manual.md](user_manual.md).

| # | Step (manual section) | Command | Expected | Observed | Time | Result |
|---|---|---|---|---|---|---|
| 1 | Create venv (3) | `python -m venv .venv` with Python 3.13 | Venv created | Created | 14 s | Pass |
| 2 | Install (3) | `python -m pip install -r requirements.txt` | About 5.5 minutes | Installed, no errors | 5.6 min | Pass |
| 3 | Smoke check (3) | `python scripts/smoke_test.py` | `Smoke check passed.` | As expected | 0.2 s | Pass |
| 4 | Tests (4) | `python -m pytest -q` | All pass, one skipped, about 1.6 minutes | 331 passed, 2 skipped (`run the sample profile first`, `CUDA is not available`) | 2.1 min | Pass, manual corrected |
| 4b | Tests, repeat after the sample run | `python -m pytest -q` | All pass | 332 passed, 1 skipped (CUDA) | 1.5 min | Pass |
| 4c | Tests without slow | `python -m pytest -q -m "not slow"` | About 1 minute | 331 passed, 1 skipped, 1 deselected | 59 s | Pass |
| 5 | Sample pipeline (5) | `python scripts/run_pipeline.py --profile sample` | Six stages done, about 45 s | All six done, exit 0; 12 incidents, 20 alert hours, 0 unresolved references; peak 735 MB | 40 s | Pass |
| 5b | State file | `outputs/pipeline/sample/pipeline_state.json` | Status per stage | Present, one entry per stage with time, memory and fingerprint | | Pass |
| 5c | Rerun | same command | Every stage skipped, about 7 s | All six "skipped, up to date", exit 0 | 6.7 s | Pass |
| 6 | Hash checks | `load_release()` and subset hashes in the clone | No mismatch | Release loaded; four subset files match the manifest; `dualscope` imported from the clone | | Pass |
| 7 | Dashboard on the sample (6) | the command printed by the runner, headless on port 8611 | Starts, health check ok | `/_stcore/health` returned `ok` after 1 s, page returned HTTP 200; server stopped afterwards | | Pass |
| 7b | First render, sample | `streamlit.testing.v1.AppTest` with the same arguments | No exception | 0 exceptions, 4.5 s, tables drawn | 4.5 s | Pass |
| 8 | Unzip the drive packages (6) | `handoff.zip` and `investigations.zip` unzipped into `outputs/` | Paths in the manual exist | `outputs/handoff/final_test_alerts_v1/incidents.jsonl` and `outputs/investigations/gemini_v1/summary.json` exist | | Pass |
| 9 | Dashboard, default command (6) | `streamlit run scripts/incident_dashboard.py` (AppTest of the script, no arguments) | Opens the final-test package | Sidebar source "Alert package (days 17-30)", page "Alert queue", 0 exceptions | 8.4 s | Pass |
| 9b | Incident with investigation | Incident page for `INC-TEST-D28-U737_DOM1-003` | Loads with its summary | Title "U737@DOM1 - NTLM logon to a new host"; tabs Overview, Timeline (130), Connections (45), Model details, Investigation; view "AI + ATT&CK, checked" shows verification statuses and the cited events; 0 exceptions | 8.8 s | Pass |

The dashboard checks used the Streamlit test runner and one real headless server. No browser window was opened, so the automatic browser launch and the look of the page were not checked in this run.

## Failure case: evidence stays available when generation is unavailable

Command, run in the clone with the `PATH` reduced to the Windows system folders (so `gcloud` could not be found) and `GOOGLE_CLOUD_PROJECT` unset:

```powershell
python scripts/run_pipeline.py --profile sample --only investigate dashboard --llm
```

Without a `gcloud` command the Gemini client stops while looking up the project name, before any request exists, so no network call can be made.

| Check | Expected | Observed | Result |
|---|---|---|---|
| Investigate stage | Failed with a clear cause | `failed`, "gcloud is not installed or not on PATH" (the log also holds a Python traceback) | Pass |
| Dashboard check | Passes | `done`; 12 incidents, 3119 events, 0 unresolved references | Pass |
| Alert files | Unchanged | SHA-256 of `incidents.jsonl`, `alerts.parquet`, `events.parquet` and `manifest.json` identical before and after | Pass |
| Exit code | Non-zero | 1 | Pass |
| Resume hint | Printed | "After fixing the cause, resume with: python scripts/run_pipeline.py --profile sample --only investigate --llm" | Pass |
| Restore | Normal sample run completes | Investigate and dashboard done, exit 0, 11 s | Pass |

## Problems found in the manual and what was done

| Problem | Fix |
|---|---|
| Section 4 said one test is skipped. On a fresh copy two are skipped: the test that needs the sample run, and the CUDA test. | Section 4 now names both and says that only the CUDA test is skipped after the sample run. |
| The test time of 1.6 minutes was low for a new venv (2.1 minutes on the first run, 1.5 on the next). | Times changed to "about 1.5 to 2 minutes" in the manual, the README Quick start and `runtime_and_resources.md`. |
| The sample run prints a `release warning` about the torch version (2.14.1 installed, 2.11.0 used for the release). The manual did not mention it, and a reader could take it for an error. | Section 5 explains the warning. |
| The manual suggests `py -3.13 -m venv .venv`, but the `py` launcher was not installed on the test machine and the default `python` was 3.14. | Section 3 adds the option of giving the full path of a supported `python.exe`. |

No step was out of order and no step was missing. The install, sample, rerun and dashboard times in the manual were within about 10 percent of what was measured.

## Remaining limitations

* The validation ran on the developer's laptop, in a separate clone and a new venv. It has not been run on a second machine, and the macOS and Linux commands were not tried.
* The full profile, the rebuild from the raw LANL files and new Gemini summaries were not run (they need the 7.6 GB download, hours of compute, or a Google Cloud login).
* No browser was opened. The dashboard was checked with the Streamlit test runner and a headless server.
* The final-test packages came from the team drive zip files kept on the developer's machine. Whether the drive links work for a marker is not checked here.
