# Task 2.3: Chronological Data Splits and Replay Policy

## 1. Overview and Context

DualScope Task 2.3 establishes the chronological dataset partitioning, training compromise exclusion rules, replay streaming protocol, and evaluation unit definitions for the Days 1–30 LANL authentication subset (`508,854,306` events, `749` red-team label rows, `715` unique compromise records).

This policy builds on the Task 2.1 inspection manifest (`data/manifests/lanl_auth_days_01_30.json`) and Task 2.2 chunked normalisation (`src/dualscope/ingestion/lanl.py`). Downstream tasks (Task 2.4 historical features, Task 3.1 sequences, Task 4.1 rolling graphs, and Task 9.1 evaluation protocol) consume this specification.

> [!NOTE]
> **Ownership and Policy Confirmation**:
> Member 1 owns and has confirmed this chronological split policy (Days 1–7 Train, Days 8–16 Validation, Days 17–30 Test). Member 4 consumes this policy for detector fusion and evaluation coordination; Member 4 approval is not required. Boundaries must strictly not be altered based on test-set performance.


---

## 2. Chronological Split Intervals

All splits use **half-open integer timestamp intervals** $[t_{\text{start}}, t_{\text{end}})$ with dataset day defined as:
$$\text{dataset\_day} = \left\lfloor \frac{\text{timestamp} - 1}{86400} \right\rfloor + 1$$

Splits form a continuous, gap-free, non-overlapping chain covering the entire selected 30-day period $[1, 2592001)$:

| Split | Dataset Days | Timestamp Interval $[t_{\text{start}}, t_{\text{end}})$ | Duration | Authentication Events | Label Rows | Unique Labels | User-Hours (Positives) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **Train** | Days 1–7 | $[1, 604801)$ | 7 days (604,800 s) | 113,699,496 | 49 | 49 | 23 |
| **Validation** | Days 8–16 | $[604801, 1382401)$ | 9 days (777,600 s) | 164,299,768 | 640 | 614 | 220 |
| **Test** | Days 17–30 | $[1382401, 2592001)$ | 14 days (1,209,600 s) | 230,855,042 | 60 | 52 | 39 |
| **Total** | **Days 1–30** | **$[1, 2592001)$** | **30 days (2,592,000 s)** | **508,854,306** | **749** | **715** | **282** |

### Purpose and Design Rationale
* **Training (Days 1–7)**: Captures early enterprise traffic and establishes baseline user behavioural distributions. Excludes known compromise events to provide clean normal-training data for the sequence autoencoder, graph autoencoder, and Isolation Forest baseline.
* **Validation (Days 8–16)**: Captures the primary high-volume attack campaigns (Days 9 and 13). Used by Members 2, 3, and 4 for model selection, sequence length tuning, score calibration, alert threshold selection, and fusion weight estimation.
* **Test (Days 17–30)**: Held-out final evaluation period containing secondary multi-hop lateral movement and credential-reuse campaigns (Days 21–23, 27–30). Evaluated exclusively using **frozen** model checkpoints, fixed normalisation parameters, and frozen decision thresholds.

> [!WARNING]
> **Uneven Label Distribution Limitation**:
> Red-team labels are highly concentrated in the validation split (85.4% of label rows; 273 rows on Day 9 and 209 rows on Day 13). The test split contains 60 label rows across Days 21–23 and 27–30. Splitting boundaries must strictly not be adjusted based on model test performance.

---

## 3. Training Exclusions

### Derivation Rule
Excluded users are derived **strictly and exclusively** from red-team compromise records whose timestamp falls within the training interval $[1, 604801)$. Validation and test labels are never inspected or used to derive training exclusions.

In the 30-day LANL dataset, red-team events on Days 2, 3, 6, and 7 derive exactly **seven excluded users**:
1. `U1723@DOM1`
2. `U6115@DOM1`
3. `U620@DOM1`
4. `U636@DOM1`
5. `U737@DOM1`
6. `U748@DOM1`
7. `U825@DOM1`

### Fitting Eligibility Rule
A training event is excluded from model fitting if **either**:
* `source_user` is in the excluded user set, **or**
* `destination_user` is in the excluded user set.

Host-wide exclusions are not applied; clean activity by other users on the same machines remains eligible. The exclusion window covers the **entire training period** for each affected user.

### Scope of the Eligibility Filter
The exact same eligibility filter must be applied to:
1. **Model parameter fitting** (GRU autoencoder, Graph autoencoder, Isolation Forest baseline).
2. **Learned numeric transformations** (standard scalers, min-max normalisers, quantile transforms).
3. **Learned categorical vocabularies** (entity and categorical embeddings, frequency encoders).

### Reconciled Counts
* **Total Training Events**: `113,699,496`
* **Excluded Training Events**: `32,659` (0.0287%)
* **Eligible Training Events**: `113,666,837` (99.9713%)

> [!IMPORTANT]
> **Benign Disclaimer**:
> Eligible training data is **not guaranteed benign**. Red-team labels only identify known recorded attacks; remaining unlabelled authentication events represent operational background traffic that may contain unlabelled administrative anomalies or undetected lateral movement. Downstream documentation must describe this data as "eligible training data" or "presumed normal background", not "verified ground-truth benign". Future sequence and graph builders must not reintroduce excluded events into fitting samples.

---

## 4. Chronological Replay vs. Model-Fitting Eligibility

Model-fitting eligibility is strictly decoupled from chronological event replay:

```
[Day 1 ......................................... Day 7] [Day 8 ............ Day 16] [Day 17 ............ Day 30]
<-------------- Training Split -----------------------> <--- Validation Split ----> <------- Test Split ------->
[All events replayed in chronological order ----------->]
               [All events replayed in chronological order ------------------------->]
                              [All events replayed in chronological order ------------------------------------>]
```

1. **Replay Retains All Events**:
   Chronological replay preserves all observed authentication events, **including training exclusions**. Validation and test labels must never selectively clean replay history.
2. **Full History Context**:
   Default replay for validation or test starts from timestamp 1 (Day 1), enabling downstream cumulative state (such as destination novelty, unique computer counts, and rolling degree) to initialise with genuine enterprise history.
3. **Optional Bounded History**:
   Callers restoring saved state or running bounded-memory graph snapshots can supply an explicit `history_start_timestamp`.
4. **Target Window Distinction**:
   Replay distinguishes earlier history from the split's own evaluation targets:
   $$\text{is\_target\_event} = (t_{\text{start}} \le \text{timestamp} < t_{\text{end}})$$
5. **Strict Upper Bound**:
   Replay must never include events at or beyond the requested split's exclusive end ($t \ge t_{\text{end}}$).
6. **Frozen Inference**:
   During validation and test replay, model weights, learned feature scalers, categorical vocabularies, and score normalisation parameters remain strictly frozen.

---

## 5. Graph Warm-Up and Sequence Policies

### Graph Warm-Up Policy
* **Initial lookback window**: $L = 86,400\text{ seconds}$ (24 hours).
* **Graph history at score time $t$**: $[t - 86400, t)$.
* **First scorable timestamp**: $t = 86,401$ (allowing a complete 24-hour history from $t=1$).
* **Earlier score times ($t < 86401$)**: Return explicit status `insufficient_history`. These must **not** become zeros or imputed normal scores.
* **Exclusive dataset end ($t = 2,592,001$)**: A graph window ending exactly at $t=2592001$ is valid and covers $[2505601, 2592001)$.
* **Real Graph Scoring Eligibility Breakdown (Recorded in Manifest)**:
  * **Train**: `97,958,544` available, `15,740,952` insufficient_history (entire Day 1), `0` out_of_bounds.
  * **Validation**: `164,299,768` available, `0` insufficient_history, `0` out_of_bounds.
  * **Test**: `230,855,042` available, `0` insufficient_history, `0` out_of_bounds.
  * **Total Dataset**: `493,113,354` available, `15,740,952` insufficient_history, `0` out_of_bounds.
* **Activity caveat**: Elapsed time does not guarantee sufficient user activity. Nodes with zero or minimal degree in the window must be handled explicitly by downstream graph models.

### Sequence Boundary Policy and Scoring Status Contract
* **Hour start origin**: 1-indexed dataset seconds:
  $$\text{hour\_start} = 1 + \left\lfloor \frac{\text{timestamp} - 1}{3600} \right\rfloor \times 3600$$
* **Boundary constraint**: Sequence target windows must stay strictly within **one assigned split** and **one dataset hour**. Events crossing a split boundary or hour boundary must not be concatenated into the same sequence target.
* **Explicit Scoring-Status Representation**:
  Candidate statuses are formally enumerated in `dualscope.splits.ScoringStatus`:
  * `available`: Candidate is temporally valid and has sufficient history.
  * `insufficient_history`: Fewer events or shorter history than required.
  * `split_boundary_violation`: Timestamp falls outside the assigned split.
  * `hour_boundary_violation`: Timestamp does not match the target dataset hour.
  * `out_of_bounds`: Timestamp falls outside the dataset boundaries.
* **Task 3.1 Rejection Tracking Contract**:
  Sequence instances are not materialized until Task 3.1. Task 3.1 must use `dualscope.splits.SequenceCandidateRejectionTracker` to record rejected candidate instances and reason codes rather than silently discarding them. Sequence counts are not fabricated prior to Task 3.1 sequence generation.


---

## 6. Label Handling and Evaluation Units

1. **Provenance Retention**:
   Original label rows are preserved for provenance. For event-level coverage, duplicate rows are deduplicated by $(t, u, c_s, c_d)$.
2. **Breakdown by Split**:
   * **Train**: 49 rows, 49 unique labels, 0 duplicates.
   * **Validation**: 640 rows, 614 unique labels, 26 duplicates.
   * **Test**: 60 rows, 52 unique labels, 8 duplicates.
3. **Imported Match Provenance (Task 2.1)**:
   * **Train**: 49 unique labels matched to authentication rows (100.0% coverage).
   * **Validation**: 600 unique labels matched; **14 unique labels unmatched** (97.72% coverage).
   * **Test**: 52 unique labels matched (100.0% coverage).
4. **Treatment of the 14 Unmatched Labels**:
   All 14 unmatched labels occur on Day 9 (12 labels) and Day 13 (2 labels) during validation. In accordance with Task 2.1 decisions, they are retained in the label source and will be evaluated under the Task 9.1 evaluation protocol. They must not be silently dropped or fabricated.
5. **User-Hour Positive Units**:
   For primary detector scoring, events are aggregated to $(u, \text{hour\_start})$ aligned with source-user acting identity. Multiple labelled events in the same user-hour evaluate as **one positive unit**:
   * **Train**: 23 positive user-hours
   * **Validation**: 220 positive user-hours
   * **Test**: 39 positive user-hours

---

## 7. Python API and Usage Examples

### Example 1: Fitting-Eligible Filter (Model Training & Scaler Fitting)

```python
import pyarrow.dataset as ds
from dualscope.splits import SplitConfig, derive_excluded_users, filter_fitting_eligible

# 1. Load configuration and training split
cfg = SplitConfig.default()
train_split = cfg.get_split("train")

# 2. Derive excluded users from training labels
labels_ds = ds.dataset("data/processed/lanl_auth_days_01_30/redteam_labels/labels", format="parquet", partitioning="hive")
excluded_users = derive_excluded_users(labels_ds, train_split.timestamp_start, train_split.timestamp_end)
print(f"Excluded users ({len(excluded_users)}): {excluded_users}")

# 3. Stream training events and filter for fitting eligibility
events_ds = ds.dataset("data/processed/lanl_auth_days_01_30/authentication/events", format="parquet", partitioning="hive")
train_filter = (ds.field("timestamp") >= train_split.timestamp_start) & (ds.field("timestamp") < train_split.timestamp_end)

for batch in events_ds.to_batches(filter=train_filter):
    # Retain only eligible events for autoencoder training or scaler fitting
    eligible_batch = filter_fitting_eligible(batch, excluded_users)
    # Train autoencoder / fit vocabulary on eligible_batch...
```

### Example 2: Chronological Replay Streaming (Inference & Feature Extraction)

```python
from dualscope.splits import SplitConfig, stream_split_events

cfg = SplitConfig.default()
val_split = cfg.get_split("validation")

# Stream events for validation: replays from Day 1 to allow cumulative features to warm up,
# strictly cutting off before validation split end (timestamp 1382401).
event_stream = stream_split_events(
    "data/processed/lanl_auth_days_01_30/authentication/events",
    target_split=val_split,
    include_prior_history=True,
    columns=["timestamp", "source_line", "acting_user", "destination_computer", "authentication_result"],
    batch_size=65536,
)

for batch in event_stream:
    # Check whether events are prior history or scoring targets
    ts_array = batch["timestamp"]
    is_target = val_split.contains_timestamp(ts_array[0].as_py())
    # Update rolling feature state on all events; score only target events...
```

### Example 3: Graph History Bounds and Sequence Verification

```python
from dualscope.splits import SplitConfig, graph_history_bounds, check_graph_history_status, check_sequence_boundary

cfg = SplitConfig.default()
train_split = cfg.get_split("train")

# Check graph status at timestamp 50,000
status_ok, reason = check_graph_history_status(50_000, dataset_start=1, lookback_seconds=86400)
assert not status_ok
assert reason == "insufficient_history"

# Check sequence boundary for event at timestamp 100 in hour 1
valid_seq, reason = check_sequence_boundary(100, split=train_split, expected_hour_start=1)
assert valid_seq
```

---

## 8. CLI Manifest Generation and Reproduction

To regenerate or verify the split manifest against the processed dataset:

```powershell
python scripts/create_lanl_splits.py `
  --config config/lanl_splits.json `
  --events-dir data/processed/lanl_auth_days_01_30/authentication/events `
  --auth-summary data/processed/lanl_auth_days_01_30/authentication/summary.json `
  --labels-dir data/processed/lanl_auth_days_01_30/redteam_labels/labels `
  --labels-summary data/processed/lanl_auth_days_01_30/redteam_labels/summary.json `
  --selection-manifest data/manifests/lanl_auth_days_01_30.json `
  --output-manifest data/manifests/lanl_splits_v1.json
```

The script verifies Parquet metadata across all 1,032 event files and 18 label files, computes training exclusions using vectorized Arrow projection, reconciles all row counts, and writes `data/manifests/lanl_splits_v1.json`.
