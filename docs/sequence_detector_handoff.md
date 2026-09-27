# Sequence detector handoff (Member 2 → Members 4, 5 and 6)

This note lists what the short-term sequence detector (Tasks 3.1–3.4) gives downstream components, and the rules for using it correctly. Method details are in [sequence_detector.md](sequence_detector.md). Frozen settings and validation results are in [`data/manifests/sequence_detector_v1.json`](../data/manifests/sequence_detector_v1.json).

## At a glance

| Item | Value |
| --- | --- |
| `model_version` | `seq-gru-ae-v1-L32-h32-91e4b11d34` |
| Selected run | `seq-L32-small`: L = 32, hidden 32, latent 16, embedding 4, `max_event` aggregation |
| Frozen record hash | `content_sha256` in `frozen_detector.json`, also in the manifest |
| Alert threshold | `score ≥ 0.999633` (validation max-F1) |
| Validation (days 8–16) | AP 0.00598 (102× random), ROC-AUC 0.880; at threshold: 340 alerts (37.8/day), precision 3.2%, recall 5.0%, FPR 8.8 × 10⁻⁵ |
| Rows exported | 11,846,723 user-hours (2,564,057 train / 3,745,355 validation / 5,537,311 test) + 1 example `no_activity` row |
| Alert volume | train 211, validation 340, test 233 (volumes only; no test labels were read) |
| Test labels used | none |
| Frozen record / checkpoint SHA-256 | `7f788ee6…f79291b` / `91e4b11d…c097ca39` |
| Feature build it depends on | Production Task 2.4 run of `scripts/build_lanl_features.py` on the shared Days 1–30 dataset (Member 1's pipeline, unchanged). `preprocessing.json` SHA-256 `a05c68be…bf35f2908`; feature config `e68cb98f…`; split policy `c5e3d5d8…` |

Re-scoring or re-exporting requires that exact feature build. `export_sequence_scores.py` refuses a different preprocessing artifact, and `FrozenSequenceDetector.load` refuses a different feature configuration or split policy. If the team shares one feature build (recommended: Member 1's call), check its `preprocessing.json` hash against the one above. Reading the exported scores does not require the feature build.

## 1. What is delivered and where

| Artifact | Location | In Git? |
| --- | --- | --- |
| Code, configuration, tests, documentation | this repository | Yes |
| Sequence manifest (Task 3.1 counts and statuses) | `data/manifests/lanl_sequences_v1.json` | Yes |
| Selection, calibration and freeze record (Task 3.3) | `data/manifests/sequence_detector_v1.json` | Yes |
| Small handoff samples (score rows, padded tensors) | `data/samples/sequence_scores_sample/`, `data/samples/sequence_tensors_sample/` | Yes |
| Frozen detector (`frozen_detector.json` + `checkpoint.pt`), 0.1 MB zip | team Google Drive: `DualScope/artifacts/sequence_detector/seq-gru-ae-v1-L32-h32-91e4b11d34/` | No |
| Full user-hour scores, days 1–30: train 448 MB, validation 692 MB, test 999 MB zips + `summary.json` | same Drive folder | No |

The Drive folder contains `README.txt` and `SHA256SUMS.txt`. After downloading:

```powershell
python scripts/package_sequence_artifacts.py --verify <downloaded folder>
```

Then extract each zip at the repository root. Archive paths are repository-relative, so files land in `models/sequence/frozen/<run_id>/` and `outputs/sequence_scores/<model_version>/`. You do not need the Task 2.4 feature build to read scores. You do need the shared `lanl_auth_days_01_30` dataset to resolve evidence to raw records.

Every row carries `model_version`. A retrained or re-exported detector gets a new version and a new Drive folder; existing files are never edited in place.

## 2. For Member 4 — fusion (5.1–5.4) and evaluation (9.x)

### Join key and time

- `user_id` is the canonical acting user = Task 2.2 `acting_user` = `source_user`, verbatim (for example `U123@DOM1`, `C456$@DOM1`).
- Window is `[window_start, window_end)` in dataset seconds. `window_start = 1 + floor((t − 1) / 3600) × 3600` (Task 2.3 hour origin), and `window_end = window_start + 3600`.
- **`score_available_at = window_end`.** A fused decision at time `t` may use a sequence row only if `score_available_at ≤ t`. The sequence score of an hour is not available inside that hour.

### Statuses and missing scores

| `status` | Meaning | `raw_score` / `score` / `is_alert` |
| --- | --- | --- |
| `available` | Hour had ≥ 1 event and complete feature history; scored | present |
| `insufficient_history` | Dataset day 1 (24-hour feature warm-up, detail `warmup_incomplete_24h_history`) | null |
| `no_activity` | Explicitly requested unit with no authentication events by that user | null |

- A user-hour with no events has **no row** unless requested; absence means `no_activity`.
- Null scores are **unavailable, never 0**. Sequence-only, graph-only and neither cases should come from these statuses, not from zero-filling.
- To get explicit `no_activity` rows for your evaluation cohort (for example labelled user-hours or controls), send a CSV or Parquet file with `user_id, window_start`. I re-export with `--requested-units`, which adds only rows; existing scores are unchanged. The current export includes one demonstration row (`U10002@DOM1`, window 756001), built from a label-free gap in that user's activity.
- In validation, all 220 labelled user-hours had events and received a score (coverage 1.0). Positives without events would still be reported as unscored rather than dropped.

### Score meaning and threshold

- `score ∈ [0, 1)` is a frozen, monotonic transform of `raw_score`. It is the empirical quantile of the raw reconstruction error among all available **validation** user-hours (label-free), with a generalized-Pareto tail above the 99.9th percentile. So 0.99 means rarer than about 99% of validation user-hours. It never reaches 1, so extreme hours stay ranked (few ties for max/average fusion). **It is not an attack probability.**
- `alert_threshold` is the frozen validation max-F1 threshold, and `is_alert = score ≥ alert_threshold`. Use it as the sequence detector's "crossed its validation threshold" signal in temporal fusion (5.3).
- The frozen threshold, aggregation, sequence length and validation metrics are in `data/manifests/sequence_detector_v1.json` → `frozen_detector` and `selection`.

### Evaluation facts you need to report

- **Unit:** `(acting_user, hour_start)`. Positive if any deduplicated red-team label (label `user` = source user) falls in the hour. Positives without authentication events cannot be scored; they are reported as `positive_coverage` and `recall_including_unscored_positives`.
- **Training:** eligible train user-hours on days 2–7. Hours touching the 7 training-excluded users are dropped whole, and day 1 is warm-up.
- **Tuning:** model settings, aggregation, calibration and threshold were chosen on validation days 8–16 only.
- **Test scores:** test days 17–30 were scored once with the frozen detector. **No test labels were read** and no test metrics were computed by Member 2, so the test split is untouched for 9.3.
- **Cohort:** detector coverage is every hour with events from day 2 on. For a common eligible cohort with the graph detector, both start after the same 24-hour warm-up.
- **PR-AUC:** "average precision" in the manifest is scikit-learn's step-wise average precision.

### Loading

```python
import pyarrow.dataset as ds

scores = ds.dataset(
    "outputs/sequence_scores/seq-gru-ae-v1-L32-h32-91e4b11d34/scores", format="parquet", partitioning="hive"
)
validation = scores.to_table(
    columns=["user_id", "window_start", "window_end", "score_available_at", "status", "raw_score", "score", "is_alert"],
    filter=ds.field("split") == "validation",
)
```

## 3. For Members 5 and 6 — evidence (6.2, 7.2)

Each row carries:

- `source_lines`: every event of the hour in order. Its evidence reference is `auth.txt:<line>`, which resolves to normalised and raw records through `AuthenticationEvidenceLookup(<lanl_auth_days_01_30>/authentication)`.
- `evidence_chunk_offset` / `evidence_chunk_length`: the slice of `source_lines` that set the score (the most anomalous consecutive run of at most `L` events).
- `top_events`: up to 5 highest-error events. Each has `source_reference`, `timestamp`, `position` in the hour, `event_error` and `top_feature` (the input the model reconstructed worst for it).
- `top_feature_contributions`: top 3 inputs by share of the evidence chunk's error.

`SequenceScoreStore(...).get(user_id, window_start)` returns a record with `source_references` and `evidence_source_references` already expanded.

**How to describe this evidence:** "the sequence model reconstructed these events poorly, mostly because of `<feature>`". It is model deviation, **not** proof that an event was malicious. It does not by itself support an ATT&CK mapping; the underlying events must.

Plain-language glossary for `top_feature` names:

| Feature | Plain meaning |
| --- | --- |
| `log1p_prior_auth_count_1h_scaled` | How many authentications this user made in the previous hour |
| `log1p_prior_failure_count_1h_scaled` | How many of this user's authentications failed in the previous hour |
| `log1p_seconds_since_previous_auth_scaled` | Time since this user's previous authentication |
| `log1p_prior_unique_destinations_24h_scaled` | Number of distinct destination computers in the previous 24 hours |
| `log1p_prior_user_destination_count_24h_scaled` | How often this user reached this destination in the previous 24 hours |
| `has_user_history` | Whether the user had any earlier authentication |
| `is_new_user_destination` | First time this user authenticated to this destination |
| `is_new_host_connection` | First time this source computer authenticated to this destination computer |
| `auth_type_id` | Authentication package (for example Kerberos, NTLM) |
| `logon_type_id` | Logon type (for example Network, Interactive) |
| `auth_orientation_id` | LogOn, LogOff, TGT, TGS or AuthMap |
| `auth_result_id` | Success or failure |

## 4. Open points to agree in Task 5.1

1. Whether the common scoring unit for fusion is the sequence user-hour. If so, the graph detector should expose a score available at or before `window_end` for the same user.
2. Maximum age of a graph score reused for a sequence hour (stale-score rule). The sequence detector has no stale scores: each row describes only its own hour.
3. The evaluation cohort file (if any) for explicit `no_activity` rows.
4. Whether any field name in the sequence schema should change. It can change once, before the frozen test run is used for reporting.
