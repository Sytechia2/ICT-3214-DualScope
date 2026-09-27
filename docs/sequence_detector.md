# Tasks 3.1–3.4: Short-term sequence detector

Owner: Member 2 (sequence detector). Code: `src/dualscope/sequence/`. Configuration: [`config/sequence_detector.json`](../config/sequence_detector.json).

The detector models short-term authentication behaviour of one acting user within one dataset hour. A GRU sequence autoencoder learns to reconstruct presumed-normal training sequences; a user-hour whose events it reconstructs poorly receives a high anomaly score. Scores, statuses and evidence are exported per user-hour for fusion (Member 4), investigation (Member 5) and the dashboard (Member 6).

A high score means the model's reconstruction deviated from learned behaviour. It is not proof of compromise, and the 0–1 score is a relative-rarity scale, not an attack probability.

## Inputs

| Input | Source | Use |
| --- | --- | --- |
| Transformed features (`transformed/events`, `preprocessing.json`) | Task 2.4 (`scripts/build_lanl_features.py`) | Per-event model inputs and category vocabularies |
| Split policy and manifest | Task 2.3 (`config/lanl_splits.json`, `data/manifests/lanl_splits_v1.json`) | Split boundaries, training exclusions |
| Evidence references | Task 2.2/2.5 (`auth.txt:<source_line>`) | Every exported event reference resolves through `AuthenticationEvidenceLookup` |
| Red-team labels | Task 2.2 (`redteam_labels/labels`) | **Validation only**, for model selection and threshold choice (Task 3.3). Never a model input. Test labels are refused by `load_positive_user_hours` unless explicitly allowed for frozen final evaluation. |

`load_inputs` refuses to run when the preprocessing artifact was fitted with a different feature configuration or split policy, when the split manifest does not match the split configuration, or when the features come from a pilot build (unless `--allow-pilot-features` is passed for development).

## Task 3.1 — User-hour sequences

| Decision | Rule |
| --- | --- |
| Unit | `(acting_user, hour_start)`; `acting_user` is the source user (Task 2.2) and `hour_start = 1 + floor((timestamp − 1) / 3600) × 3600` (Task 2.3). Window is `[hour_start, hour_start + 3600)`. |
| Order | `timestamp`, then `source_line`, both ascending. Users are processed alphabetically so output order is deterministic. |
| Per-event inputs (12) | 5 standardised numeric features (`log1p_prior_auth_count_1h_scaled`, `log1p_prior_failure_count_1h_scaled`, `log1p_seconds_since_previous_auth_scaled`, `log1p_prior_unique_destinations_24h_scaled`, `log1p_prior_user_destination_count_24h_scaled`), 3 binary flags (`has_user_history`, `is_new_user_destination`, `is_new_host_connection`) and 4 categorical IDs (`auth_type_id`, `logon_type_id`, `auth_orientation_id`, `auth_result_id`). All are causal Task 2.4 features. |
| Length and long hours | A maximum length `L` is chosen on validation from {32, 64, 128}. An hour with `n > L` events becomes `ceil(n / L)` **balanced consecutive chunks** (lengths differ by at most one), so no event is dropped or truncated and there is no one-event trailing chunk. Chunks never cross their hour. |
| Padding | Right padding to the longest sequence in a batch, with a boolean mask. Padded steps carry zero inputs and category ID 0, and contribute exactly zero loss (tested). |
| Event IDs | Model arrays keep each event's `source_line`; `auth.txt:<source_line>` is its Task 2.2 evidence reference (checked on a strided sample of every day). |
| Warm-up | Hours whose events lack a complete 24-hour feature history (all of dataset day 1) are `insufficient_history` with detail `warmup_incomplete_24h_history`. This matches the graph detector's 24-hour warm-up. |
| Low activity | `min_events = 1`: every hour with at least one event is scored. A larger `min_events` marks shorter hours `insufficient_history` with detail `fewer_than_min_events`. User-hours without events are not scored; they are exported only as explicit `no_activity` rows when requested. |
| Split boundaries | Split boundaries are whole days and hours never cross a day, so a user-hour cannot cross a split. This is still verified for every event, and every candidate is recorded through `SequenceCandidateRejectionTracker` (Task 2.3 contract). |
| Fitting eligibility | A user-hour may train the model only if it is in the train split, available, and none of its events has a training-excluded source or destination user. The whole hour is dropped from fitting (not just the event), so fitting never sees an altered sequence. Replay and scoring still include those hours. |

Run:

```powershell
python scripts/build_sequences.py `
  --features-root data/processed/lanl_features_days_01_30 `
  --sample-output outputs/sequence/tensor_sample/sequence_tensors
```

It writes [`data/manifests/lanl_sequences_v1.json`](../data/manifests/lanl_sequences_v1.json) with per-split status counts, fitting-eligible counts, events-per-hour percentiles, chunk counts per candidate `L` and the tracker summary. `--sample-output` saves padded tensors, masks and their ordered evidence references for inspection (a tracked copy is in `data/samples/sequence_tensors_sample/`).

## Task 3.2 — GRU autoencoder

**Architecture** (`dualscope.sequence.model`): each categorical ID is embedded, concatenated with the numeric and binary inputs, and read by a GRU encoder (packed, so padding is never seen). The final hidden state is projected to a latent vector `z`. A GRU decoder receives only `z` at every step, with its initial state derived from `z`, so it cannot copy inputs. Heads reconstruct every input.

**Objective:** for each event and input feature, squared error (numeric), binary cross-entropy (binary) or cross-entropy (categorical), multiplied by the group weight (all 1.0). An event's reconstruction error is the mean of its 12 per-feature losses, and the training loss is the mean event error over unpadded events.

**Training data:** chunks from fitting-eligible training user-hours (days 2–7 after warm-up), with at most two random chunks per hour so very busy accounts do not dominate. From those candidates, `train_sequences` (300,000) are drawn uniformly at random (a bottom-k sample of random keys across days). A label-free sample of 30,000 validation chunks monitors reconstruction loss. Adam optimiser (learning rate 1e-3), batches of 512 grouped by length, gradient clipping at 1.0, at most 8 epochs, and early stopping after 2 epochs without monitor improvement. The best-monitor epoch is restored.

**Saved per run** (`models/sequence/runs/<run_id>/`): `checkpoint.pt` (weights, architecture, metadata; loaded with `weights_only=True`), `config.json`, `training_log.jsonl` (per-epoch losses and time), and `run_manifest.json`. The manifest records the seed, feature version and preprocessing hash, split fingerprint, excluded users, sample accounting, parameter count, duration, loss history, environment and source revision. After saving, the checkpoint is reloaded from disk and scores the held-out monitor sample with no weight updates. The run fails unless those scores are finite.

```powershell
python scripts/train_sequence_model.py --max-sequence-length 64 --model-size medium
```

## Task 3.3 — Selection and calibration

**Search space** (bounded to the schedule): `L ∈ {32, 64, 128}` × model size {`small`: hidden 32, latent 16, embedding 4; `medium`: hidden 64, latent 32, embedding 8}, giving 6 trained models. Each model is scored with three user-hour aggregations, giving 18 candidates:

- `max_chunk_mean`: highest mean event error of any chunk in the hour;
- `max_event`: the single worst event error in the hour;
- `hour_mean`: mean event error over the whole hour.

**Evaluation unit:** validation user-hour; positive if any deduplicated validation red-team label (user = source user) falls in it. Positives with no authentication events cannot be scored and are reported through `positive_coverage` and `recall_including_unscored_positives`.

**Selection rule** (fixed before any test scoring): highest validation average precision (scikit-learn step-wise PR-AUC) over scorable user-hours. Ties are broken by fewer parameters, then shorter `L`, then aggregation order. Failed runs are recorded rather than dropped.

**Score normalisation** (`QuantileTailCalibrator`), fitted on the selected model's label-free validation raw scores:

- `score = empirical quantile of log(raw)` among reference user-hours, interpolated between 1,000 knots (0 to 99.9th percentile);
- below the lowest knot: 0 (clipped);
- above the 99.9th percentile knot `x₀`: `score = 1 − 0.001 / (1 + (log(raw) − x₀) / s)`, a generalized-Pareto (shape 1) tail whose initial slope matches the 99th–99.9th percentile spacing (`s = (x₀ − x₀.₉₉) / ln 10`).

The transformation is monotonic, maps to `[0, 1)`, never saturates at 1 (so extreme hours stay ranked for fusion), and uses no test data.

**Alert threshold:** maximises validation F1 of `score ≥ threshold` on calibrated scores; ties keep the higher threshold.

**Freeze:** `models/sequence/frozen/<run_id>/frozen_detector.json` stores:
- the checkpoint copy and its SHA-256;
- the full sequence configuration and its fingerprint;
- the aggregation, calibration knots and threshold;
- the feature, preprocessing and split fingerprints;
- the selection record and `test_data_used: false`;
- a content hash.

`FrozenSequenceDetector.load` rejects a modified record, a changed checkpoint, or mismatched feature or split versions. The tracked [`data/manifests/sequence_detector_v1.json`](../data/manifests/sequence_detector_v1.json) records every trial and the selection.

```powershell
python scripts/select_sequence_model.py `
  --features-root data/processed/lanl_features_days_01_30 `
  --labels-dir <lanl_auth_days_01_30>/redteam_labels/labels
```

Completed runs with the same configuration fingerprint and preprocessing hash are reused, so selection can resume after an interruption.

## Task 3.4 — Output schema and evidence

`scripts/export_sequence_scores.py --frozen-dir models/sequence/frozen/<run_id>` writes `outputs/sequence_scores/<model_version>/scores/dataset_day=NN/*.parquet` and `summary.json`. One row per user-hour:

| Field | Type | Meaning |
| --- | --- | --- |
| `detector` | string | `sequence_gru_autoencoder` |
| `model_version` | string | `seq-gru-ae-v1-L<L>-h<hidden>-<checkpoint hash prefix>` |
| `run_id` | string | Selected training run |
| `user_id` | string | Canonical acting user (source user) |
| `window_start`, `window_end` | int64 | Half-open dataset-second window `[start, end)` |
| `score_available_at` | int64 | `window_end`; a decision at time `t` may use only rows with `score_available_at ≤ t` |
| `dataset_day` | int32 | Hive partition |
| `split` | string | `train` / `validation` / `test` |
| `status` | string | `available`, `insufficient_history`, or `no_activity` |
| `status_detail` | string | Reason for a non-available status |
| `raw_score` | float64, nullable | Aggregated reconstruction error |
| `score` | float64, nullable | Calibrated 0–1 score |
| `alert_threshold` | float64 | Frozen validation threshold |
| `is_alert` | bool, nullable | `score ≥ alert_threshold` |
| `aggregation` | string | Aggregation that produced `raw_score` |
| `n_events`, `n_chunks` | int32 | Events in the hour; sequences they formed |
| `source_lines` | list<int64> | Every event of the hour in sequence order; reference `auth.txt:<line>` |
| `evidence_chunk_index`, `_offset`, `_length`, `_error` | nullable | The chunk (slice of `source_lines`) that set the score and its mean error |
| `top_events` | list<struct> | Up to 5 highest-error events: `source_reference`, `timestamp`, `position`, `event_error`, `top_feature` |
| `top_feature_contributions` | list<struct> | Top 3 input features by share of the evidence chunk's error |

**Missing scores:** `raw_score`, `score` and `is_alert` are null unless `status == "available"`. Fusion must treat them as unavailable, never as zero. Hours without events have no row by default; pass `--requested-units <csv|parquet with user_id, window_start>` to add explicit `no_activity` rows for an evaluation cohort.

**Reading evidence:**

```python
from dualscope.sequence.export import SequenceScoreStore
from dualscope.evidence.lookup import AuthenticationEvidenceLookup

store = SequenceScoreStore("outputs/sequence_scores/<model_version>/scores")
record = store.get("U123@DOM1", 777601)       # adds source_references, evidence_source_references
events = AuthenticationEvidenceLookup("<lanl_auth_days_01_30>/authentication").lookup_many(record["source_references"])
alerts = store.alerts(columns=["user_id", "window_start", "score"])
```

**Consistency:** the export re-scores one validation day and fails unless raw scores match within `1e-6` and alert decisions are identical (recorded in `summary.json`). `--sample-output` writes a small handoff sample (Parquet + JSONL) covering ordinary, high-error, alerting, multi-chunk, warm-up and `no_activity` rows; a tracked copy is in `data/samples/sequence_scores_sample/`.

## Sharing artifacts

Checkpoints and full score files stay out of Git. Package them for the team's shared Google Drive:

```powershell
python scripts/package_sequence_artifacts.py `
  --frozen-dir models/sequence/frozen/<run_id> `
  --scores-root outputs/sequence_scores/<model_version>
```

This writes `outputs/share/<model_version>/`:
- a detector zip;
- one score zip per split;
- the export summary;
- `SHA256SUMS.txt`;
- `README.txt`.

Upload that folder unchanged to `DualScope/artifacts/sequence_detector/<model_version>/`. Recipients verify with `--verify <folder>` and extract the zips at the repository root. What each downstream member needs is in [the handoff note](sequence_detector_handoff.md).

## Tests

```powershell
python -m pytest tests/test_sequence_builder.py tests/test_sequence_model.py tests/test_sequence_calibration.py tests/test_sequence_export.py
```

The tests run synthetic events through the real Task 2.4 feature engine (`tests/conftest.py`). They cover:
- ordering and assignment of every event to exactly one user-hour;
- split and hour boundaries, warm-up status and tracker counts;
- excluded-user eligibility and balanced chunking without loss;
- zero loss at padded positions;
- training, checkpoint reload and seed reproducibility;
- calibrator monotonicity, bounds and ties, and a hand-calculated F1 threshold;
- the test-label guard;
- output schema conformance and retrieval of every exported reference through `AuthenticationEvidenceLookup`;
- evidence-chunk correctness, `no_activity` and warm-up statuses;
- repeat-inference consistency and frozen-record tamper detection.

## Installation note

`torch` is listed in `requirements.txt`; CPU wheels are sufficient. On Linux, `pip install torch --index-url https://download.pytorch.org/whl/cpu` avoids downloading the CUDA build.

## Limitations

- Training data is eligible training activity presumed normal, not verified benign. Unlabelled validation activity is treated as negative for selection, which undercounts precision if it contains unlabelled attacks.
- Validation labels are concentrated in two campaign days, so the selected settings and threshold are fitted to those campaigns. The calibration reference also includes validation attack hours (a negligible fraction).
- Features count authentication records (including LogOff, TGS, AuthMap), not interactive sessions. Heavy service accounts produce many chunks per hour, and `max_chunk_mean` and `max_event` give such hours more chances of a high score.
- An hour with one event is scored with little sequence context.
- A score becomes available only when its hour closes, so the sequence detector gives no intra-hour early warning.
