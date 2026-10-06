# Supervised fusion: what we found and how to run the final test

From Member 1 (Peter), 2026-10-02. Full record: [experiment_features_v2.md](experiment_features_v2.md).

## What we found

The unsupervised detectors catch few attacks. At 38 alerts per day, the GRU (Member 2's model) catches 8 of the 220 attack user-hours on validation days 8–16. A one-line rule that counts "first-time" logins does about as well. Adding new features to the GRU didn't help.

So we tried **supervised fusion**: a model that learns from the labelled validation attacks how to combine the GRU score with simple hourly counts (events, failures, distinct computers, first-time logins, NTLM, Network logons, LogOns, machine account or not). No user or computer names are used. It was trained on days 8–12 and graded on days 13–16, which it never saw. The settings and the pass/fail rule were written down before any results.

| Days 13–16 (136 attack user-hours) | Attacks in top 38 alerts/day | Average precision |
| --- | ---: | ---: |
| GRU alone | 4 | 0.0046 |
| Counting rule | 4 | 0.0067 |
| Logistic regression | 4 | 0.019 |
| **Gradient boosting** | **16** | **0.050** |

**Gradient boosting passed the rule** (≥ 2× the GRU's attacks and a higher AP). The 16 hits come from 13 different users across all four days, and the result doesn't depend on how ties are broken.

Caveats for the report:
- About half of the gain comes from learning "never alert on machine accounts" (no attack in the data is a machine account). If we just filter the GRU to human accounts, it catches 8. The model still doubles that.
- It learned mostly from one day (82 of the 84 training attacks are on day 9) and from one red team, so it may miss attacks that look different.
- It's supervised: the report must say clearly that validation labels were used to train the fusion (the proposal's "validation-tuned weighted fusion").

**This is the final model.** Settings are fixed in code. Don't tune anything.

## What's left: the one-time test on days 17–30

> **Done (2026-10-06, Member 1). Do not run step 6 again.** Result: the frozen model caught 1 of 39 test attack hours at 38 alerts/day (GRU 0); AP 0.00124 vs 0.00020. Primary rule not met, secondary rule met. Details: "Final test results" in [experiment_features_v2.md](experiment_features_v2.md). The steps below are kept as the record of how it was run.

Everything below is on branch `experiment/features-v2` (not merged into `main`). Our trained models aren't in Git, so you rebuild them; the settings are fixed and the steps are deterministic, so your numbers should match ours (checkpoints below). Use the project's `.venv` (Python 3.12 or 3.13).

**Memory (measured):** each pass-1 worker uses about 5 GB plus about 3 GB for the host-connection task, and each pass-2 assembly worker up to 11–12 GB. On a 32 GB machine use `--workers 2 --assembly-workers 1 --low-memory-read` with other apps closed (the actual run used `--assembly-workers 2` and went into the page file; it took about 3 h). Any worker count gives the same result because everything is rebuilt from the same build.

```powershell
git fetch origin
git checkout experiment/features-v2
git pull
```

**0. Data.** You need the ingested LANL data in `data/processed/lanl_auth_days_01_30/` (with `authentication/events` and `redteam_labels/labels`). Member 2 has it. If you don't, follow [lanl_ingestion.md](lanl_ingestion.md).

**1. Features, days 1–30** (about 1.5–3 h):
```powershell
.venv\Scripts\python.exe -u scripts\build_lanl_features.py --feature-config config\lanl_features_v2.json --workers 4 --output data\processed\lanl_features_v2_days_01_30 > logs\build_v2_01_30.log 2>&1
```

**2. Train the GRU (run A; CPU, about 10 min):**
```powershell
.venv\Scripts\python.exe -u scripts\train_sequence_model.py --features-root data\processed\lanl_features_v2_days_01_30 --feature-config config\lanl_features_v2.json --sequence-config config\sequence_detector.json --max-sequence-length 32 --model-size small --torch-threads 5 --runs-dir models\sequence\runs_experiment_v2 --run-id A --cache-dir data\processed\sequence_cache_experiment_v2
```

**3. Score validation with the GRU and save per-hour scores:**
```powershell
.venv\Scripts\python.exe -u scripts\evaluate_sequence_runs.py --device cuda --features-root data\processed\lanl_features_v2_days_01_30 --run A=models\sequence\runs_experiment_v2\A --output outputs\experiment_v2\results_A.json --units-output outputs\experiment_v2\units_A.parquet
```
Checkpoint: `A max_event` should show **AP ≈ 0.006 and about 8/220** (ours: 0.00598 and 8/220; a different worker count can shift it very slightly) (use `--device cpu` if you have no GPU).

**4. Validation fusion test (repeats our result):**
```powershell
.venv\Scripts\python.exe -u scripts\supervised_fusion_validate.py --features-root data\processed\lanl_features_v2_days_01_30 --units outputs\experiment_v2\units_A.parquet
```
Checkpoint: gradient boosting **about 16/136**, GRU **about 4/136**, and `Success rule ... 'hist_gradient_boosting': True`. If your numbers are clearly different, stop and tell Peter.

**5. Freeze the final model** (refits on days 8–16, saves to `models\fusion\experiment_v2\frozen`):
```powershell
.venv\Scripts\python.exe -u scripts\supervised_fusion_freeze.py
```

**6. Final test, run once only:**
```powershell
.venv\Scripts\python.exe -u scripts\supervised_fusion_final_test.py --device cuda --features-root data\processed\lanl_features_v2_days_01_30 > logs\final_test.log 2>&1
```
It first checks that day 16 reproduces the validation inputs exactly, then scores days 17–30, saves the scores, and only then reads the test labels. It refuses to run a second time. **Don't rerun it with changed settings**: that would turn the test set into a tuning set.

## Reporting

- Paste the final table from `logs\final_test.log` into the "Final test" section of [experiment_features_v2.md](experiment_features_v2.md), with whether the primary and secondary rules were met. Numbers are in `outputs\experiment_v2\final_test.json`.
- **Primary rule:** at least 2× the GRU's attacks at 38 alerts/day and a higher AP on days 17–30.
- **Secondary rule:** more attacks and a higher AP than the GRU filtered to human accounts. This shows whether the model adds anything beyond ignoring machine accounts.
- Report the result honestly whichever way it goes, with the caveats above.
- The fusion score is the input for the LLM investigation (top alerts per day) and the dashboard: `outputs\experiment_v2\final_test_scores.parquet`, column `fusion`.
