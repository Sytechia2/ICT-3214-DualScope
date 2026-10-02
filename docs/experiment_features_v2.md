# Experiment plan: better prepared data for the short-term detector

Owner: Member 1 (data prep), building on Member 2's sequence detector.
Results: `outputs/experiment_v2/results.json` and `rules.json` (not in Git). Logs: `logs/experiment_v2/`.
Branch: `experiment/features-v2`. Status: **done (2026-10-02)**. Result: **no clear improvement**; the success rule was not met (see results log).

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
