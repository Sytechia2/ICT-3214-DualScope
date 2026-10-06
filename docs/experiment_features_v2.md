# Experiment plan: better prepared data for the short-term detector

Owner: Member 1 (data prep), building on Member 2's sequence detector.
Results: `outputs/experiment_v2/results.json` and `rules.json` (not in Git). Logs: `logs/experiment_v2/`.
Branch: `experiment/features-v2`. Status: **done (2026-10-02)**. Result: **no clear improvement**; the success rule was not met (see results log). Follow-up Option A (supervised fusion): **success rule met** by gradient boosting (16 vs 4 attacks at 38 alerts/day on days 13–16); see its section below. Final test on days 17–30 (2026-10-06): **primary rule not met** (1 of 39 attacks at 38/day vs the GRU's 0; AP 0.00124 vs 0.00020); see "Final test results".

## In plain words

The short-term detector catches few attacks. We think one reason is that the prepared data leaves out the strongest warning sign ("this person has never logged in from this computer before"), and another is that the detector lets timing noise drown out real warning signs. We will add the missing signs, rerun Member 2's detector on the same data three ways (before, with better data, with better data and a scoring fix), and compare. If the improvement is real, we rerun Member 2's full model search later.

## Ground rules (check before every step)

- [ ] Work only on branch `experiment/features-v2`. Do not modify `main`.
- [ ] Do not overwrite Member 2's artifacts: `models/sequence/frozen/`, `outputs/sequence_scores/`, `data/manifests/sequence_detector_v1.json`, `data/manifests/lanl_sequences_v1.json`.
- [ ] Do not change `config/lanl_features.json` or `config/sequence_detector.json`. New settings go in new files (`*_v2.json`), so the v1 build stays reproducible.
- [ ] **Days 1–16 only.** No test-day (17–30) features, scores or labels are produced or read. Labels are used for validation days 8–16 only, exactly as Member 2 did.
- [ ] The success rule below is fixed now and is not changed after seeing results.
- [ ] Every new output path contains `v2` or `experiment` so it cannot be confused with v1.

## Baseline facts (already verified in this repo)

| Fact | Value | Source |
| --- | --- | --- |
| Selected v1 model | `seq-L32-small` (L = 32, hidden 32, latent 16, embedding 4), `max_event` aggregation, seed 42 | `data/manifests/sequence_detector_v1.json` |
| v1 validation AP / ROC-AUC | 0.005981 / 0.8801 | same |
| v1 at threshold | 340 alerts (37.8/day), TP 11, FP 329, precision 3.2%, recall 5.0% | same |
| Validation positives | 220 user-hours (all scored), prevalence 5.87e-5, 3,745,355 scorable user-hours | same |
| Labelled events' profile | 100% NTLM / Network / LogOn; 98% from C17693; first-ever (src,dst) 42% vs 0.28% for unlabelled; first-ever (user,src) 11% vs 0.11% | own analysis of `data/processed/lanl_auth_days_01_30` |
| Label incompleteness | C17693: 1,714 events days 1–30, only 670 labelled; unlabelled ones are all NTLM LogOn | same |
| Events, days 1–16 | 277,999,264 | `authentication/summary.json` |
| Feature build throughput (v1) | pass 1 ≈ 55k events/s, pass 2 ≈ 95k events/s → ≈ 84 + 49 min for days 1–16 | `data/manifests/lanl_features_v1.json` |
| Hardware | 24 logical CPUs; NVIDIA RTX 5060 Laptop 8 GB (driver 591.84); 90 GB free on C: | `nproc`, `nvidia-smi`, `df` |
| Python | system Python is 3.14; project requires `>=3.11,<3.14` and `pyarrow<22` | `pyproject.toml`, `requirements.txt` |

## Step 1 — Environment (≈ 20 min)

1. Create branch `experiment/features-v2` from `main`.
2. Install Python 3.12 or 3.13 (system 3.14 is outside `requires-python`). Create `.venv` (already git-ignored).
3. `pip install -r requirements.txt`, then replace torch with a CUDA build that supports the RTX 5060 (Blackwell, sm_120; needs a CUDA 12.8+ wheel, e.g. `--index-url https://download.pytorch.org/whl/cu128`).
4. Run the existing test suite as a baseline.

**Verify**
- [ ] `python --version` is 3.12.x or 3.13.x inside `.venv`.
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); torch.ones(1,device='cuda')+1"` prints `True` and the RTX 5060 without a "no kernel image" error. If it fails: continue on CPU and note it here.
- [ ] `python -m pytest` passes (record the count) before any code change.

## Step 2 — Speed-ups (≈ 1–2 h of coding)

### 2a. Parallel feature build (pass 1 and pass 2)

- Pass 1 state is per `acting_user`, except `is_new_host_connection`, which uses a global (source, destination) set.
- Plan:
  - Shard users by a stable hash into N workers (N ≈ 10–12). Each worker runs `FeatureEngine` on its users' events in the original order.
  - Compute `is_new_host_connection` in a separate vectorised single pass (first occurrence of the (src, dst) pair in `(timestamp, source_line)` order) and join it back.
  - Merge training statistics with the existing `combine_batch` (Chan merge), and merge vocabularies as set unions.
- Pass 2 (transform) is stateless given the frozen preprocessing, so parallelise it by day.
- Keep the sequential path as the default. The parallel path is opt-in (`--workers N`).

**Verify**
- [ ] On days 1–2 (pilot), parallel and sequential outputs are **identical**: same row count, and every column equal after sorting by `source_line`.
- [ ] `preprocessing.json` statistics agree to ≤ 1e-9 relative and vocabularies are identical.
- [ ] Measured speed-up recorded below. If outputs differ and the cause isn't found quickly, fall back to the sequential build (≈ 2 h 15 min) and say so.

### 2b. Optional GPU for training and scoring

- Add a `device` setting, default `cpu`, so v1 results stay reproducible. Move the model and batch tensors to the device in `training.py` and `scoring.py`.
- `pack_padded_sequence` already takes CPU lengths (`model.py:114`). Keep that.

**Verify**
- [ ] With `device=cpu`, existing sequence tests pass unchanged.
- [ ] On a small sample, CPU and GPU event errors agree within 1e-4 (absolute). GPU kernels are not bit-identical, so exact equality isn't expected.
- [ ] Epoch time on GPU vs CPU recorded below.

## Step 3 — New features (Member 1's part, Task 2.4)

| Column | Type | Definition | Why |
| --- | --- | --- | --- |
| `is_new_user_source` | bool, model input (binary) | `(acting_user, source_computer)` never seen before this event, cumulative from day 1, causal (computed before the state update), same as `is_new_user_destination` | General lateral-movement indicator (stolen credentials used from an unusual machine). Strongest signal in the labelled data that the v1 features lack. |
| `is_machine_account` | bool, model input (binary) | Account name part of `acting_user` (before `@`) ends with `$` | Lets the model separate machine accounts (all v1 alerts in the score sample) from human accounts (all 749 labels) |

Implementation:
- `features/engine.py`: add a `seen_user_sources` set, plus memory accounting in `get_memory_breakdown`.
- `features/schemas.py`, `features/config.py`: new columns. New `config/lanl_features_v2.json` with `feature_version` `2.0.0` and the two binary inputs added. v1 config untouched.
- `sequence/config.py`: model inputs currently hard-coded from `DEFAULT_MODEL_INPUTS_*` (`config.py:28-30`). Make them come from the sequence config (new `inputs` block), defaulting to the v1 list so v1 is unchanged.
- **Bug risk:** the training-sample cache key (`pipeline.py:133`) contains the preprocessing hash, split fingerprint and excluded users, **but not the input list**. Add the input list (or the sequence config fingerprint) to the key, otherwise runs A and B could silently share a cached sample with the wrong columns.
- Tests: extend `tests/test_lanl_features.py` with the hand-worked example in `docs/lanl_features.md` (expected `is_new_user_source` per event, a `$` account and an `@`-less name). Add a cache-key test.

**Verify**
- [ ] Hand-worked example values match.
- [ ] v1 config still produces byte-identical v1 columns on the pilot (new columns are additive).
- [ ] Full `pytest` passes.
- [ ] On the days 1–16 build: `is_new_user_source` rate among labelled validation events ≈ 0.11 and among all others ≈ 0.001 (matches the analysis above). If it's wildly different, stop and investigate.

## Step 4 — Build and run (≈ 1–1.5 h compute)

1. Feature build, days 1–16, v2 config:
   `python scripts/build_lanl_features.py --feature-config config/lanl_features_v2.json --pilot-days 16 --output data/processed/lanl_features_v2_days_01_16 [--workers N]`
   (Correction after reading the code: `--pilot-days` does **not** mark the build as pilot. Only `--pilot-mode` does. The build is labelled production, and its `summary.json` records `max_dataset_day: 16`. The `days_01_16` folder name makes the scope explicit.)
   Actual command: `python -u scripts/build_lanl_features.py --feature-config config/lanl_features_v2.json --pilot-days 16 --workers 12 --output data/processed/lanl_features_v2_days_01_16`, log `logs/experiment_v2/full_build.log`.
2. Sequences: `scripts/build_sequences.py --days 1..16 --output outputs/experiment_v2/lanl_sequences_v2.json` (never the v1 manifest path).
3. Runs. All use L = 32, `small`, seed 42, 300,000 training chunks, 8 epochs, `--torch-threads 5` (same as the v1 winner). **A and B both train on CPU**, so the device can't explain any difference. Separate cache dir `data/processed/sequence_cache_experiment_v2`, so Member 2's cache is never touched:

| Run | Inputs | Scoring | Retrain? |
| --- | --- | --- | --- |
| **A. Control** | v1 12 inputs (on the v2 build) | v1 (`max_event` of mean per-feature loss) | yes |
| **B. New features** | v1 + `is_new_user_source` + `is_machine_account` (14) | v1 | yes |
| **C. B + scoring fix** | as B | each feature's per-event loss divided by its mean loss on the **training** sample (label-free), then averaged. Aggregation still `max_event`, plus the other two for reference | no (re-score B) |

4. Score validation days 8–16 with each run and evaluate against validation labels using Member 2's existing evaluation code (same positive definition and AP implementation).
5. Write runs to `models/sequence/runs_experiment_v2/` and results to `outputs/experiment_v2/`.

**Verify**
- [ ] Run A on **CPU** reproduces v1 validation AP ≈ 0.005981 (within ±10%). This proves the rebuild and pipeline match Member 2's. If it's outside ±10%, stop and find out why before reading B or C.
- [ ] Every run's validation coverage = 220/220 positives scored.
- [ ] No file for days ≥ 17 exists under the v2 output folders (`ls`/glob check).
- [ ] Run C's normalisation constants come from the training sample only (logged in its manifest).

## Step 5 — Compare (rule fixed in advance)

Report for A, B and C:
1. Validation average precision (scikit-learn step-wise, same as v1) and ROC-AUC.
2. **Recall at a fixed alert budget:** top 38 user-hours per validation day (≈ v1's 37.8/day). Report TP and recall.
3. The same two measures at **user-day** level (max over the day's hours), for comparison with Member 3.
4. Alerts by account type (machine `$` vs human).

**Success rule:** B or C counts as a real gain **only if** it reaches **AP ≥ 2× run A** **and** catches **more true positives than A at the 38/day budget**. Anything smaller is reported as "no clear improvement": with 220 positives concentrated on two days, small differences can be chance.

## Results log (fill in as we go)

| Item | Value |
| --- | --- |
| Python / torch / CUDA available | Python 3.13.11 (`.venv`), torch 2.11.0+cu128, CUDA available on the RTX 5060 (capability 12.0) ✅ |
| Baseline pytest count | 100 passed before changes; 118 passed after steps 2–3 ✅ |
| Parallel build speed-up (pilot) | Days 1–2 (33.3M events), 12 shards, while the sequential build ran at the same time: pass 1 233 s, pass 2 246 s, total 8.0 min. Each shard reads every event and keeps its own, so pass 1 is read-bound (≈ 13k events/s per shard). Projected ≈ 35–40 min for days 1–16 (vs ≈ 2 h 15 min sequential). A pre-partitioned design could reach ≈ 15 min; not worth the extra risk now. |
| Parallel == sequential (pilot) | Synthetic fixture: identical ✅. Real days 1–2 (33,303,322 rows): every raw and transformed value identical (max float diff 0), vocabularies and fitting counts equal, scaling stats within 1.4e-15 relative ✅. Sequential took 20.0 min (pass 1 841 s, pass 2 356 s) |
| Days 1–16 build time | 46.3 min with 12 shards (pass 1 2,164 s, pass 2 611 s); 277,999,264 events, 12 GB; 0 test rows. New-feature check: `is_new_user_source` 0.1117 on the 600 matched labelled validation events vs 0.0011 on the same users' other validation events ✅ |
| GPU vs CPU epoch time | Training was done on CPU for both runs (≈ 46 s/epoch, 6.2 min per run with 5 threads), so the GPU wasn't needed. Validation scoring ran on GPU; GPU and CPU agree within 1e-4 ✅ |
| Run A AP / ROC-AUC / TP@38 | 0.00598 / 0.880 / 8 of 220 (`max_event`). **Reproduces v1 exactly** (v1 AP 0.005981; train/monitor loss 0.22433/0.22507 vs v1 0.2243/0.2251) ✅ |
| Run B AP / ROC-AUC / TP@38 | 0.00486 / 0.892 / 3 of 220 (`max_event`) |
| Run C AP / ROC-AUC / TP@38 | 0.00546 / 0.964 / 0 of 220 (`max_event`, B + per-feature scaling). Informational A+scaled: 0.00284 / 0.952 / 0 |
| User-day AP (A / B / C) | 0.0125 / 0.0103 / 0.0086; TP at 38 alerts/day: 13 / 6 / 0 of 135 positive user-days |
| Success rule met? | **No.** Neither B nor C reaches 2× A's AP (0.012), and both catch fewer attacks than A at 38 alerts/day. ROC-AUC rose from 0.880 to 0.964 (C): attack hours move up the overall ranking, but not into the top alerts. Full numbers: `outputs/experiment_v2/results.json` |

## Follow-up: label-free counting rules (2026-10-02)

`scripts/experiment_v2_rules.py`. Each validation user-hour is scored by counting events with a novelty flag. No model and no training. Rules were fixed before the run, and ties are broken by a fixed random order. Same units (3,745,355 user-hours, 220 positives), labels and budget as above.

| Scorer | AP | ROC-AUC | TP at 38/day | Alerts machine / human | User-day AP | User-day TP at 38/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU, run A (`max_event`) | 0.00598 | 0.880 | 8 / 220 | 158 / 184 | 0.0125 | 13 / 135 |
| Count `is_new_user_source` | 0.00581 | 0.863 | 8 / 220 | 170 / 172 | 0.0141 | 8 / 135 |
| Count `is_new_host_connection` | 0.00562 | 0.787 | 5 / 220 | 284 / 58 | 0.0132 | 6 / 135 |
| Count either flag | 0.00733 | 0.895 | 7 / 220 | 264 / 78 | 0.0191 | 8 / 135 |

Reading: a one-line counting rule roughly matches the trained GRU (slightly higher AP, about the same number of attacks in the top alerts). So the GRU adds little beyond counting novel logins. The ceiling at the top of the list is shared: novel logins are common, and busy machine accounts fill the alert budget for every scorer.

## Follow-up: Option A, supervised fusion quick test (plan fixed 2026-10-02, before any results)

**Question.** If a simple model learns from the labelled validation attacks how to combine the GRU score with hourly counts, does it clearly beat the GRU alone? This tests the proposal's "validation-tuned weighted fusion" in its most flexible form. Unlike everything above, it is **supervised**: it uses validation labels for training, not only for evaluation.

**Units.** One row per validation user-hour (acting user, hour), days 8–16: the same 3,745,355 units and 220 positives as above.

**Inputs (label-free, no user or computer names):**

| Input | Definition |
| --- | --- |
| `gru_max_event` | Run A's raw `max_event` score (= Member 2's v1 model, rebuilt), re-scored on GPU |
| `n_events` | events in the user-hour |
| `n_failures` | `authentication_result == "Fail"` |
| `n_sources`, `n_destinations` | distinct source / destination computers |
| `n_new_user_source`, `n_new_host_connection`, `n_new_user_destination` | events with that novelty flag |
| `n_ntlm`, `n_network_logon`, `n_logon` | `authentication_type == "NTLM"`, `logon_type == "Network"`, `authentication_orientation == "LogOn"` |
| `is_machine_account` | account name ends with `$` |

**Labels:** `load_positive_user_hours` (validation only).

**Split by time:** fit on days 8–12, evaluate on days 13–16. Days 17–30 are not built, scored or read.

Label counts per day (counted before fitting anything; these are counts, not results): day 8: 1, day 9: 82, day 10: 1, days 11–12: 0, so **84 training positives, 82 of them from one day**; days 13–16: 68 / 36 / 15 / 17 = **136 test positives**. 97 users in total, 17 of them in both halves. The models therefore learn mostly from a single day's attack burst.

**Models (settings fixed now, no tuning on days 13–16):**
- Logistic regression: `log1p` of every count, `log(gru_max_event + 1e-12)`, `is_machine_account` as 0/1, then `StandardScaler`; `LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)`.
- `HistGradientBoostingClassifier(class_weight="balanced", random_state=0)`, other settings scikit-learn defaults (its default early stopping holds out 10% of the **training** days).

**Compared on days 13–16:** GRU alone, counting rule "either" (events with `is_new_user_source` or `is_new_host_connection`, as in the rules follow-up; a comparator only, not a model input), logistic regression, gradient boosting. Metrics from `experiment_v2_evaluate.evaluate`: AP, ROC-AUC, TP in the top 38 user-hours per day, machine/human split of those alerts, the user-day versions. Ties are broken by the same fixed random order (seed 0) as the rules. Feature importance: standardised coefficients (LR), and permutation importance on days 13–16 scored by AP (5 repeats; for reporting only).

**Success rule (fixed):** a real gain only if a model catches **≥ 2× the GRU's true positives at 38/day on days 13–16** **and** has **higher AP** than the GRU there. Edge case fixed in advance: if the GRU catches 0 on days 13–16, the model must catch at least 2.

### Option A results (2026-10-02)

Run: `scripts/experiment_v2_evaluate.py --run A=... --units-output outputs/experiment_v2/units_A.parquet` (re-score on GPU; reproduces AP 0.00598 and 8 of 220 exactly, log `rescore_A.log`), then `scripts/experiment_v2_fusion.py` (log `fusion.log`, numbers `outputs/experiment_v2/fusion.json`). Fit on days 8–12 (1,969,281 user-hours, 84 positive). All rows below are days 13–16: 1,776,074 user-hours, 136 positive user-hours, 88 positive user-days, 38 alerts per day = 152 alerts.

| Scorer | AP | ROC-AUC | TP at 38/day | Alerts machine / human | User-day AP | User-day TP at 38/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU, run A (`max_event`) | 0.00457 | 0.839 | 4 / 136 | 82 / 70 | 0.0114 | 6 / 88 |
| Count either flag | 0.00670 | 0.849 | 4 / 136 | 120 / 32 | 0.0149 | 3 / 88 |
| Logistic regression | 0.01901 | 0.978 | 4 / 136 | 0 / 152 | 0.0508 | 18 / 88 |
| **Gradient boosting** | **0.04984** | 0.817 | **16 / 136** | 0 / 152 | **0.0802** | 17 / 88 |

**Success rule: met by gradient boosting** (16 ≥ 2 × 4, and AP 0.0498 > 0.00457, about 11×). **Not met by logistic regression** (same 4 TP at 38/day, although its AP is 4× the GRU's and it finds the most attack user-days).

Feature importance (permutation, AP drop on days 13–16):
- Gradient boosting: `n_ntlm` 0.046, `is_machine_account` 0.045, `n_new_host_connection` 0.042, `n_new_user_source` 0.038, `gru_max_event` 0.034. It uses the GRU score, but as one input among several.
- Logistic regression: `n_network_logon`, `n_ntlm`, `n_new_user_source`, `gru_max_event`, `n_events`, each about 0.015. Its top alerts are very busy human accounts (median 5,460 events per hour); it ranks attacks well overall (ROC-AUC 0.978) but not at the very top.

Checks (for interpretation only; they don't change the verdict):
- **Not one burst:** the 16 hits come from 13 different users and all four days (10 / 2 / 3 / 1 on days 13 / 14 / 15 / 16). The GRU's 4 hits come from 4 users on days 13–14.
- **Ties don't explain it:** gradient boosting outputs only 1,204 distinct scores, but the 38th-place cut-off only splits ties without positives, except day 15, where all tied rows fit in the budget. TP at 38/day is 16 under 20 different tie-break orders (and "either" is 4 under all 20).
- **About half of the gain is "ignore machine accounts":** both models raise zero machine-account alerts, and no validation positive is a machine account. Post hoc, restricting the label-free scorers to human accounts gives GRU 8 TP (AP 0.0100) and "either" 6 TP (AP 0.0150). Gradient boosting still catches 2× that, with 3–5× the AP.
- **Overfitting:** gradient boosting's AP on its own training days is 0.39 vs 0.050 on days 13–16, and it learned mostly from one day (day 9). It still generalised to later days and new users, but it has only ever seen one red-team campaign.

**Reading.** Learning from labelled attacks clearly beats the GRU at our alert budget on unseen days: 4× the attacks in the top 38 per day. Part of that comes from learning that machine accounts are never attackers here, which a fixed filter could also do; the rest comes from combining NTLM use, novelty counts and the GRU score. The model learns *this* red team's profile (NTLM network logons to new hosts), so it may miss attacks that look different. This is a limitation to state in the report.

**Next:** refit on days 8–16 with the same fixed settings, freeze, score the test days once (below). Decision by Member 1 (2026-10-02): this is the final model; the graph score is not added.

## Final test on days 17–30 (plan fixed 2026-10-02, before any test-day data was built)

**Frozen model.** `scripts/experiment_v2_freeze_fusion.py` → `models/fusion/experiment_v2/frozen/` (`model.joblib` + `manifest.json`): gradient boosting, settings unchanged, refitted on all of days 8–16 (3,745,355 user-hours, 220 positives; 39 boosting iterations), code at commit `85b0ff1`, model sha256 `bf9ffc58be29…`. GRU input: run A `max_event`. Nothing is changed after this point.

**Who runs it:** this laptop (31 GB RAM) couldn't run the days 1–30 build with 12 workers. The team reruns steps 1–6 of [supervised_fusion_handover.md](supervised_fusion_handover.md) from branch `experiment/features-v2` on their machine: same code and fixed settings, so the refrozen model is the same model up to tiny numeric differences, and its own manifest records its hash.

**Test features.** New v2 build of days 1–30 (`data/processed/lanl_features_v2_days_01_30`, same config, 12 workers). The novelty flags are cumulative from day 1, so days 1–16 must be rebuilt too. Before any test label is read, the run checks that this build reproduces the validation inputs: same preprocessing file hash as run A was trained with, and day 16's GRU scores and hourly counts equal the ones the model was fitted on.

**Units.** User-hours on days 17–30 that the GRU scores (status available). Test positives outside those units count as misses (recall is over all test positives).

**Scored once, in this order:** GRU scores and counts for all test days → frozen model predictions → saved to disk → only then are test labels read (`load_positive_user_hours(..., "test", allow_test=True)`). The script refuses to run again if its output exists.

**Reported (same metrics and tie-break as Option A):** for the frozen model, the GRU alone, the counting rule "either", and, because Option A showed about half the gain comes from ignoring machine accounts, GRU and "either" restricted to human accounts. Also per-day TP and distinct users among hits.

**Test success rules (fixed):**
1. Primary: the frozen model catches **≥ 2× the GRU's TP at 38/day** (at least 2 if the GRU catches 0) **and** has **higher AP** than the GRU on days 17–30.
2. Secondary: the frozen model catches **more TP at 38/day and has higher AP** than the GRU restricted to human accounts. This shows whether the model adds anything beyond a machine-account filter.

### Final test results (2026-10-06, run once by Member 1)

**Rebuild and refreeze.** Run on this laptop after all, with the steps of the handover writing to separate folders so the Option A files stay untouched. Logs: `logs/final_test/`, results: `outputs/final_test/`.

| Step | Command (deviations from the handover) | Result |
| --- | --- | --- |
| Features, days 1–30 | `build_lanl_features.py --workers 2 --assembly-workers 2 --low-memory-read` → `data/processed/lanl_features_v2_days_01_30` | 508,854,306 events, reconciled; 2 h 58 min (pass 1 1 h 53 min, pass 2 1 h 5 min). Preprocessing differs from the days 1–16 build only in 13 floating-point values (max relative difference 2.2 × 10⁻¹⁵, from merging 2 instead of 12 shards) and in the row counts that now include test days |
| GRU run A | `--runs-dir models/sequence/runs_final_test --cache-dir data/processed/sequence_cache_final_test` | Same loss at every epoch as the original run A to 5 decimals (final 0.22433 / 0.22507) |
| Validation scoring | CPU (the GPU was not available to PyTorch), `--units-output outputs/final_test/units_A.parquet` | **Reproduces** AP 0.00598, 8 / 220 and every other row of the original table |
| Fusion check | `--counts-cache outputs/final_test/fusion_counts.parquet` (a new cache, so counts come from the new build) | Hourly counts identical to Option A. GRU scores differ slightly (median relative difference 0.02%, rank correlation 0.9999998; original scored on GPU, these on CPU). Gradient boosting: **13 / 136**, AP 0.0348 (Option A: 16, 0.0498). Still meets the success rule |
| Freeze | `--gru-run models/sequence/runs_final_test/A --output models/fusion/final_test/frozen` | Same settings, days 8–16, 45 iterations, model sha256 `02b4e6f7e3d1…`. Nothing changed after this point |

**Stability check (validation only, before the test).** Because a 0.02% change in the GRU scores moved gradient boosting from 16 to 13, `scripts/experiment_v2_stability.py` refitted both models on days 8–12 ten times with the GRU scores multiplied by (1 + e), e ~ Normal(0, 3 × 10⁻⁴) (the observed size of the change), and scored days 13–16 (`outputs/final_test/stability.json`). Gradient boosting: **median 16 / 136, range 13–18; AP median 0.042, range 0.035–0.051**. Logistic regression: 4 / 136 in every run. Every run meets the success rule. The validation result to report is therefore "16 of 136 (13–18 under numeric noise)", 3–4.5× the GRU's 4.

**Test (days 17–30).** Day-16 reproduction check: 431,200 user-hours, max GRU score difference 0, counts equal. 5,537,311 test user-hours, all with events scored. **39 positive user-hours**, all scored, from 34 positive user-days; 28 of the 39 are on days 27 (20) and 28 (8).

| Scorer | AP | ROC-AUC | TP at 38/day | Alerts machine / human | User-day AP | User-day TP at 38/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Frozen gradient boosting** | **0.00124** | **0.969** | **1 / 39** | 0 / 532 | 0.00368 | 1 / 34 |
| GRU, run A (`max_event`) | 0.00020 | 0.846 | 0 / 39 | 303 / 229 | 0.00096 | 1 / 34 |
| Count either flag | 0.00006 | 0.598 | 0 / 39 | 343 / 189 | 0.00061 | 0 / 34 |
| GRU, human accounts only | 0.00043 | 0.944 | 0 / 39 | 0 / 532 | 0.00236 | 1 / 34 |
| Either flag, human accounts only | 0.00010 | 0.811 | 0 / 39 | 0 / 532 | 0.00155 | 2 / 34 |

The model's one hit is on day 28.

**Rules:**
1. Primary: **not met.** The GRU caught 0, so the model needed at least 2. It caught 1 (its AP is about 6× the GRU's).
2. Secondary: **met** (1 > 0 TP, AP 0.00124 > 0.00043).

**Where the attacks rank (descriptive, after the test; nothing was changed).** Median daily rank of the 39 attack hours: frozen model 1,958, GRU 19,917 (of about 325,000–453,000 user-hours per day). Attack hours within each day's top 100 / 500 / 1,000: frozen model 2 / 7 / 14, GRU 0 / 3 / 3.

**Reading.** On unseen test days the supervised fusion still ranks attacks far better than the GRU (AP about 6×, ROC-AUC 0.969 vs 0.846; the typical attack hour about 10× higher in the ranking), but at 38 alerts per day it catches almost nothing: 1 of 39. The validation gain at the budget (16 of 136 vs 4) does not carry over to days 17–30. With 39 positives concentrated on two days, the test is also small: one hit more or less changes the verdict. For the report: the model is a better ranker than the GRU, but not an effective detector at this alert budget on this test period.

## Out of scope (for now)

- Member 3's graph detector (not pushed; no code in any branch, PR or fork as of 2026-10-02).
- Run D: dropping LogOff / TGS / TGT events, or adjusting for busy accounts. Only if B or C help.
- Re-running Member 2's full 6-model search, re-freezing, test-day scoring. Only after a clear gain, and agreed with Member 2.

## Risks

| Risk | Mitigation |
| --- | --- |
| RTX 5060 not supported by the installed torch build | Use a cu128 wheel. Otherwise run on CPU (≈ 6–20 min per training run is acceptable). |
| Parallel build subtly differs from sequential | Exact-equality check on the pilot. Otherwise fall back to sequential. |
| Disk: 90 GB free | Days 1–16 only. Estimate 15–25 GB build + ≈ 5 GB venv. Check free space before step 4. |
| New feature overfits this one campaign | It's a standard lateral-movement indicator defined without labels. Report this limitation anyway. |
| Validation is also where we choose things | No thresholds or settings are tuned on these comparisons. Test days stay untouched for the final evaluation. |
