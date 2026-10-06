# Task 2.4: Shared Historical Features

## 1. Overview and Architecture

DualScope Task 2.4 provides a streaming, causal feature extraction pipeline that converts normalized authentication events into behavioral features. The pipeline processes records in strictly non-decreasing `(timestamp, source_line)` order without using future information.

The pipeline enforces an **explicit two-pass architecture**:
1. **Pass 1 (Raw Causal Feature Generation & Training Statistics Accumulation)**:
   - Streams input authentication events in deterministic chronological order.
   - For every event, expires rolling records, calculates causal historical features from prior state, emits a raw feature row, and updates state with the current event.
   - Concurrently updates running training statistics (mean, variance, category sets) **exclusively on eligible training rows** (timestamp $< 604,801$ and neither source nor destination user in the 7 training compromise exclusion users).
   - Excluded training rows and validation/test rows update replay state but contribute zero statistics to scalers and vocabularies.
2. **Preprocessing Freeze**:
   - At the completion of training data scanning, standardization parameters ($\mu, \sigma$) and categorical vocabularies are frozen and serialized to a versioned `preprocessing.json` artifact.
3. **Pass 2 (Frozen Transformation)**:
   - Streams the saved raw feature batches and transforms them into model-ready numerical and categorical arrays using the frozen preprocessor.
   - Applies post-scaling insertion of a finite placeholder (`0.0`) for missing gaps, retaining `has_user_history` so models can distinguish genuine zero-second gaps from missing gaps.

---

## 2. Feature Dictionary

| Feature Name | Data Type | Grouping Key | Lookback | Same-Timestamp Semantics | Missing-Value Handling | Role | Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `prior_auth_count_1h` | `int64` | `acting_user` | 3,600 s | Records at same timestamp with strictly earlier `source_line` count as prior. | Default 0 if no prior records in window. | Raw Feature / Transformed Model Input | $\log(1 + x)$, then standardized $(z - \mu)/\sigma$. |
| `prior_failure_count_1h` | `int64` | `acting_user` | 3,600 s | Prior records in window with `authentication_result == "Fail"`. | Default 0 if no prior failures. | Raw Feature / Transformed Model Input | $\log(1 + x)$, then standardized $(z - \mu)/\sigma$. |
| `seconds_since_previous_auth` | `float64` (nullable) | `acting_user` | Entire replay | Difference $t - t_{\text{prev}}$. Can be exactly $0.0$ if user had an earlier event at the same timestamp. | `null` for first-seen user. Excluded from training $\mu, \sigma$ fitting. | Raw Feature / Transformed Model Input | Observed: $\log(1 + x)$ standardized. Null: assigned `0.0` placeholder post-scaling. |
| `has_user_history` | `bool` / `float32` | `acting_user` | Entire replay | True if user had any strictly earlier event in replay. | None (always True or False). | Model Input (Binary) | Cast to `float32` (`1.0` if True, `0.0` if False). |
| `prior_unique_destinations_24h` | `int64` | `acting_user` | 86,400 s | Count of distinct `destination_computer`s accessed by user in window. | Default 0 if no prior records in window. | Raw Feature / Transformed Model Input | $\log(1 + x)$, then standardized $(z - \mu)/\sigma$. |
| `prior_user_destination_count_24h` | `int64` | `(acting_user, destination_computer)` | 86,400 s | Count of prior records for this user-destination pair in window. | Default 0 if no prior records for this pair. | Raw Feature / Transformed Model Input | $\log(1 + x)$, then standardized $(z - \mu)/\sigma$. |
| `is_new_user_destination` | `bool` / `float32` | `(acting_user, destination_computer)` | Entire replay | True only on the first appearance of this `(user, destination)` pair. | None (always True or False). | Model Input (Binary) | Cast to `float32` (`1.0` if True, `0.0` if False). |
| `is_new_host_connection` | `bool` / `float32` | `(source_computer, destination_computer)` | Entire replay | True only on the first appearance of this directed host edge. | None (always True or False). | Model Input (Binary) | Cast to `float32` (`1.0` if True, `0.0` if False). |
| `is_new_user_source` | `bool` / `float32` | `(acting_user, source_computer)` | Entire replay | True only on the first appearance of this `(user, source)` pair, i.e. the first time the user authenticates from this computer. | None (always True or False). | Model Input (Binary) in `lanl_features_v2.json` only | Cast to `float32` (`1.0` if True, `0.0` if False). |
| `is_machine_account` | `bool` / `float32` | `acting_user` | None | True when the account name before `@` ends with `$` (a computer account). | None (always True or False). | Model Input (Binary) in `lanl_features_v2.json` only | Cast to `float32` (`1.0` if True, `0.0` if False). |
| `history_complete_1h` | `bool` | Dataset time | 3,600 s | True when $t \ge t_{\text{start}} + 3600$. | None (always True or False). | Metadata / Filter Flag | Retained as metadata; excluded from default model inputs. |
| `history_complete_24h` | `bool` | Dataset time | 86,400 s | True when $t \ge t_{\text{start}} + 86400$. | None (always True or False). | Metadata / Filter Flag | Retained as metadata; excluded from default model inputs. |
| `auth_type_id` | `int32` | None | N/A | Preserves original string in metadata. | Categorical vocabulary. | Model Input (Categorical) | `0` = UNSEEN, `1` = MISSING, `2+` = sorted training categories. |
| `logon_type_id` | `int32` | None | N/A | Preserves original string in metadata. | Categorical vocabulary. | Model Input (Categorical) | `0` = UNSEEN, `1` = MISSING, `2+` = sorted training categories. |
| `auth_orientation_id` | `int32` | None | N/A | Preserves original string in metadata. | Categorical vocabulary. | Model Input (Categorical) | `0` = UNSEEN, `1` = MISSING, `2+` = sorted training categories. |
| `auth_result_id` | `int32` | None | N/A | Preserves original string in metadata. | Categorical vocabulary. | Model Input (Categorical) | `0` = UNSEEN, `1` = MISSING, `2+` = sorted training categories. |

> [!IMPORTANT]
> **Distinction Between Feature G (`is_new_user_destination`) and Feature H (`is_new_host_connection`)**:
> - Feature G tracks user-to-destination access: whether `acting_user` has ever previously logged into `destination_computer`.
> - Feature H tracks host-to-host network topology: whether `source_computer` has ever previously established an authentication connection to `destination_computer`.
> - For example, if User A logs into Server 1 from Workstation 1, both G and H are True. If User B later logs into Server 1 from Workstation 1, Feature G is True (new for User B) while Feature H is False (Workstation 1 -> Server 1 has been seen before).

> [!NOTE]
> **Record Counts vs. Authentication Attempts**:
> The counts represent preserved authentication records (including LogOn, LogOff, TGT, TGS, and AuthMap). They count retained records, not deduplicated interactive login sessions.

---

## 3. Manually Worked Example

Consider the following deterministic 7-event sequence:

| Event | Timestamp $t$ | Source Line | User $u$ | Source Host $sc$ | Dest Host $dc$ | Orientation | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 1 | `U1` | `C1` | `C1` | LogOn | Success |
| **2** | 1 | 2 | `U1` | `C1` | `C1` | LogOn | Success |
| **3** | 2 | 3 | `U1` | `C1` | `C2` | TGS | Success |
| **4** | 3 | 4 | `U2` | `C2` | `C3` | LogOn | Fail |
| **5** | 3602 | 5 | `U1` | `C3` | `C2` | LogOn | Fail |
| **6** | 3603 | 6 | `U1` | `C1` | `C2` | LogOn | Success |
| **7** | 86401 | 7 | `U1` | `C1` | `C4` | LogOn | Success |

### Step-by-Step Causal State Evolution:

1. **Event 1** ($t=1$, line 1):
   - Prior state: None.
   - Features: `prior_auth_count_1h = 0`, `prior_failure_count_1h = 0`, `seconds_since_previous_auth = null`, `has_user_history = False`, `prior_unique_destinations_24h = 0`, `prior_user_destination_count_24h = 0`, `is_new_user_destination = True`, `is_new_host_connection = True`, `history_complete_1h = False`, `history_complete_24h = False`.
   - Update: `U1` last auth = 1. `(U1, C1)` and `(C1, C1)` recorded as seen.

2. **Event 2** ($t=1$, line 2, identical timestamp):
   - Expiry: $1 \ge 1 - 3600$, nothing expires.
   - Features: Sees Event 1! `prior_auth_count_1h = 1`, `prior_failure_count_1h = 0`, `seconds_since_previous_auth = 0.0` (legitimate zero-second gap), `has_user_history = True`, `prior_unique_destinations_24h = 1`, `prior_user_destination_count_24h = 1`, `is_new_user_destination = False`, `is_new_host_connection = False`.
   - Update: `U1` last auth = 1.

3. **Event 3** ($t=2$, line 3):
   - Features: `prior_auth_count_1h = 2`, `seconds_since_previous_auth = 1.0`, `has_user_history = True`, `prior_unique_destinations_24h = 1` (only `C1`), `prior_user_destination_count_24h = 0` (`C2` not in window), `is_new_user_destination = True`, `is_new_host_connection = True`.
   - Update: `U1` last auth = 2. `(U1, C2)` and `(C1, C2)` recorded as seen.

4. **Event 4** ($t=3$, line 4, `U2`):
   - Features: `U2` has never appeared: `prior_auth_count_1h = 0`, `seconds_since_previous_auth = null`, `has_user_history = False`, `is_new_user_destination = True`, `is_new_host_connection = True`.
   - Update: `U2` last auth = 3. `U2` failure count in 1h = 1.

5. **Event 5** ($t=3602$, line 5):
   - Expiry: 1h lower bound is $3602 - 3600 = 2$. Events 1 & 2 ($t=1$) expire! Event 3 ($t=2$) satisfies $2 \ge 2$ and remains.
   - Features: `prior_auth_count_1h = 1`, `prior_failure_count_1h = 0`, `seconds_since_previous_auth = 3600.0` ($3602 - 2$), `has_user_history = True`, `prior_unique_destinations_24h = 2` (`C1`, `C2`), `prior_user_destination_count_24h = 1` (`C2` in Event 3), `is_new_user_destination = False` (`(U1, C2)` seen in Event 3), `is_new_host_connection = True` (`(C3, C2)` has never connected before!), `history_complete_1h = True`, `history_complete_24h = False`.
   - Update: `U1` last auth = 3602. `U1` failure count in 1h = 1.

6. **Event 6** ($t=3603$, line 6):
   - Expiry: 1h lower bound is $3603 - 3600 = 3$. Event 3 ($t=2$) now expires! Only Event 5 ($t=3602$, Fail) remains in 1h window.
   - Features: `prior_auth_count_1h = 1`, `prior_failure_count_1h = 1` (Event 5 failed!), `seconds_since_previous_auth = 1.0` ($3603 - 3602$), `has_user_history = True`, `prior_unique_destinations_24h = 2`, `prior_user_destination_count_24h = 2` (Events 3 and 5), `is_new_user_destination = False`, `is_new_host_connection = False` (`(C1, C2)` seen in Event 3).

7. **Event 7** ($t=86401$, line 7):
   - Expiry: 1h lower bound = $86401 - 3600 = 82801$ (all 1h records expire). 24h lower bound = $86401 - 86400 = 1$ (all records $t \ge 1$ remain in 24h window).
   - Features: `prior_auth_count_1h = 0`, `seconds_since_previous_auth = 82798.0` ($86401 - 3603$), `has_user_history = True`, `prior_unique_destinations_24h = 2` (`C1`, `C2`), `prior_user_destination_count_24h = 0` (`C4` not seen), `is_new_user_destination = True`, `is_new_host_connection = True`, `history_complete_1h = True`, `history_complete_24h = True`.

---

## 4. Loading Examples for Downstream Tasks

### Member 2: Short-Term Sequence Detector (Task 3.1 & 3.2)
Member 2 constructs user-hour sequences from the transformed features:

```python
import pyarrow.dataset as ds

# Load transformed features with projection
features = ds.dataset(
    "data/processed/lanl_features_days_01_30/transformed/events",
    format="parquet",
    partitioning="hive",
)

# Project model inputs and evidence metadata
model_cols = [
    "timestamp", "acting_user", "source_reference", "dataset_day",
    "log1p_prior_auth_count_1h_scaled",
    "log1p_prior_failure_count_1h_scaled",
    "log1p_seconds_since_previous_auth_scaled",
    "has_user_history",
    "auth_type_id", "logon_type_id", "auth_orientation_id", "auth_result_id",
]

day_filter = ds.field("dataset_day") == 1
day_one_events = features.to_table(filter=day_filter, columns=model_cols)
```

### Member 3: Long-Term Graph Detector (Task 4.1)
Member 3 builds the unsupervised GAE from `acting_user → destination_computer` edges. The separate confirmed-relationship extension also retains `source_computer` and matches exact user/source/destination triples. Both paths preserve source references:

```python
import pyarrow.dataset as ds

features = ds.dataset(
    "data/processed/lanl_features_days_01_30/transformed/events",
    format="parquet",
    partitioning="hive",
)

graph_cols = [
    "timestamp", "acting_user", "source_computer", "destination_computer",
    "source_reference", "is_new_host_connection", "authentication_result",
]

# Query temporal snapshot [t - 86400, t)
window_filter = (ds.field("timestamp") >= 1) & (ds.field("timestamp") < 86401)
graph_records = features.to_table(filter=window_filter, columns=graph_cols)
```

### Member 4: Isolation Forest Baseline (Task 9.2)
Member 4 extracts the full flat feature matrix for eligible training evaluation units:

```python
import pyarrow.dataset as ds
import numpy as np

features = ds.dataset(
    "data/processed/lanl_features_days_01_30/transformed/events",
    format="parquet",
    partitioning="hive",
)

feature_cols = [
    "log1p_prior_auth_count_1h_scaled",
    "log1p_prior_failure_count_1h_scaled",
    "log1p_seconds_since_previous_auth_scaled",
    "log1p_prior_unique_destinations_24h_scaled",
    "log1p_prior_user_destination_count_24h_scaled",
    "has_user_history",
    "is_new_user_destination",
    "is_new_host_connection",
    "auth_type_id",
    "logon_type_id",
    "auth_orientation_id",
    "auth_result_id",
]

# Load training split
train_filter = (ds.field("timestamp") >= 1) & (ds.field("timestamp") < 604801)
batch_iter = features.to_batches(filter=train_filter, columns=feature_cols)
for batch in batch_iter:
    X_batch = np.column_stack([batch[col].to_numpy() for col in feature_cols])
    # Pass to baseline model
```

---

## 5. Reproduction Commands

### 1. Build Synthetic Tracked Sample
```powershell
python scripts/build_lanl_features.py `
  --events data/samples/lanl_ingestion_sample/authentication/events `
  --splits-config data/fixtures/fixture_splits.json `
  --splits-manifest data/manifests/fixture_splits_v1.json `
  --feature-config data/fixtures/fixture_features.json `
  --output data/samples/lanl_features_sample `
  --pilot-mode `
  --overwrite
```

### 2. Run Full 24-Hour Expiry Pilot on Real Data
```powershell
python scripts/build_lanl_features.py `
  --events data/processed/lanl_auth_days_01_30/authentication/events `
  --splits-config config/lanl_splits.json `
  --splits-manifest data/manifests/lanl_splits_v1.json `
  --feature-config config/lanl_features.json `
  --output data/processed/lanl_features_pilot `
  --pilot-rows 15800000 `
  --batch-size 131072 `
  --pilot-mode `
  --overwrite
```

### 3. Full 30-Day Production Generation
```powershell
python scripts/build_lanl_features.py `
  --events data/processed/lanl_auth_days_01_30/authentication/events `
  --splits-config config/lanl_splits.json `
  --splits-manifest data/manifests/lanl_splits_v1.json `
  --feature-config config/lanl_features.json `
  --output data/processed/lanl_features_days_01_30 `
  --batch-size 131072
```

### 4. Parallel Build and v2 Features (experiment)
`--workers N` shards users across `N` processes (`src/dualscope/features/parallel.py`). Every feature except `is_new_host_connection` depends only on the acting user's own history; that one is computed in a separate pass over all events and joined back by `source_line`. Output rows and values match the sequential build (`tests/test_parallel_features.py`), but rows within a day are grouped by shard, so sort by `(timestamp, source_line)` when order matters. Scaling statistics can differ in the last floating-point digits because moments are merged in a different order.

`config/lanl_features_v2.json` adds `is_new_user_source` and `is_machine_account` as model inputs. Both columns are computed in every build, so the v1 configuration and its columns are unchanged.

```powershell
python scripts/build_lanl_features.py `
  --feature-config config/lanl_features_v2.json `
  --pilot-days 16 --workers 12 `
  --output data/processed/lanl_features_v2_days_01_16
```

---

## 6. Measured Performance and Full Dataset Extrapolation

The two-pass streaming pipeline was benchmarked using a representative performance pilot over the first **15,800,000 real authentication events** from `data/processed/lanl_auth_days_01_30/authentication/events`, spanning from timestamp 1 past timestamp 86,400 into Day 2 to complete a full 24-hour sliding window-expiry cycle.

### Measured Pilot Results (15,800,000 events)

| Metric | Measured Value |
| :--- | :--- |
| **Pass 1 Duration** | 287.30 s (4.79 min) |
| **Pass 1 Throughput** | **54,994 events/s** |
| **Pass 2 Duration** | 166.13 s (2.77 min) |
| **Pass 2 Throughput** | **95,108 events/s** |
| **Total Wall-Clock Time** | 453.43 s (7.56 min) |
| **Combined Effective Throughput** | **34,845 events/s** |
| **Raw Events Written** | 15,800,000 |
| **Transformed Events Written** | 15,800,000 |
| **Count Reconciliation** | **Reconciled (100%)** |
| **Distinct Users Tracked** | 33,601 |
| **Distinct Computers Tracked** | 10,211 |
| **Distinct User-Destinations Ever Seen** | 213,329 |
| **Distinct Host Connections Ever Seen** | 134,752 |
| **Rolling History Memory (24h deques)** | **319.33 MB** (15,773,726 events held) |
| **Ever-Seen Sets Memory (64-bit packed)** | **12.00 MB** |
| **Entity Tables Memory** | **2.37 MB** |
| **Total Internal State Memory** | **333.70 MB** |

### Extrapolation to Full Selected Dataset (508,854,306 events, Days 1–30)

| Dimension | Projected Value | Rationale |
| :--- | :--- | :--- |
| **Pass 1 Runtime** | ~2.57 hours | At 55,000 events/s sustained rate. |
| **Pass 2 Runtime** | ~1.48 hours | At 95,000 events/s sustained rate. |
| **Total Full Run Time** | **~4.05 hours** | Single commodity workstation, no distributed cluster needed. |
| **Peak Rolling History Memory** | **~350 MB** | Rolling state is strictly bounded to the active 24-hour window by the inactive user sweeper. |
| **Peak Ever-Seen Sets Memory** | **~60 MB** | Over 30 days, distinct (user, host) and (host, host) pairs grow to at most ~1.5M packed 64-bit ints. |
| **Peak Internal Engine Memory** | **< 500 MB** | Total internal state remains well within desktop memory constraints. |
| **Output Disk Footprint (Zstd)** | ~15 GiB raw, ~11 GiB transformed | Day-partitioned Parquet with compressed columnar storage. |

---

## 7. Known Limitations and Operational Considerations

1. **Authentication Records vs. Logins**: Features count retained log records. Orientation types (such as `LogOff`, `TGT`, `TGS`) are preserved in historical state counts and are not collapsed into distinct human logon sessions.
2. **Dataset Start Startup Transient**: Historical features at the beginning of Day 1 describe observed dataset history only. Early events legitimately lack prior history; downstream models should use `history_complete_1h` and `history_complete_24h` for warm-up eligibility analysis.
3. **Presumption of Normality**: Eligible training rows (with known red-team compromise users excluded) represent presumed operational background, not guaranteed ground-truth benign activity.
4. **Memory Dynamics**: Ever-seen relationship sets grow with distinct entities observed over time. Compact packed 64-bit integer sets and active purging of inactive rolling states bound memory consumption.
