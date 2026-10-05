# Detection Timing, Precedence, and Forensic Error Analysis (Task 9.4)

## 1. Executive Summary & Objective

In operational Security Operations Centers (SOCs), detection performance is not measured solely by precision and recall. Security teams require:
1. **Low Detection Latency**: Catching lateral movement before domain persistence is established.
2. **Causal Decision Availability**: Accounting for the exact timestamp when a model's score is available to an analyst.
3. **Forensic Transparency**: Understanding why specific alerts fired (True Positives), why benign operations triggered alerts (Assumed False Positives), and why stealthy activities were missed (False Negatives).

This report evaluates detection timing and error distributions across the **unseen validation test holdout (Days 13–16)**, using models trained strictly on **Days 08–12**.

---

## 2. Evaluation Protocol & Timing Definitions

### 2.1 Causal Score Availability
- **Ground-Truth Attack Timestamp (`t_attack`)**: The exact epoch second recorded in `redteam.txt` for an attack event.
- **Sequence Score Availability (`t_avail`)**: A short-term sequence window spanning `[window_start, window_start + 3600)` completes and outputs its score at `t_avail = window_start + 3600`.
- **Graph Score Availability**: Long-term graph embeddings are computed on daily snapshots (Day D - 1) and become causally available at day boundary `(D - 1) * 86400 + 1`.
- **First-Alert Offset (`Delta_t`)**:
  `Delta_t = t_avail - t_first_attack`
  - **Early Warning (`Delta_t < 0`)**: The model alerted before the earliest officially labelled red-team event.
  - **Immediate (`0 <= Delta_t <= 3600`)**: The model alerted during the very first hour of the attack campaign.
  - **Delayed (`Delta_t > 3600`)**: The model alerted during subsequent attack hours (e.g. lateral movement).
  - **Missed (`Delta_t = None`)**: No alert was produced within the daily SOC alert budget (38 alerts/day).

---

## 3. First-Alert Timing & Lead-Time Results (Days 13–16)

Evaluated across 88 distinct `(user, day)` attack campaigns on Days 13–16 at the empirical budget of 38 alerts/day:

| Metric | Result | Operational Significance |
| :--- | :---: | :--- |
| **Total Attack Campaigns** | 88 campaigns | 136 ground-truth attack user-hours |
| **Campaigns Caught @ Budget 38** | **10 / 88 (11.36%)** | 2.5x higher than GRU alone (4 / 88) |
| **Campaigns Missed @ Budget 38** | 78 campaigns | Mostly single isolated logons |
| **Immediate Detections (`0 <= Delta_t <= 1h`)** | **4 campaigns** | Caught in the very first hour of compromise |
| **Early Warning (`Delta_t < 0`)** | **2 campaigns** | Flagged prior to official red-team label |
| **Delayed Detections (`Delta_t > 1h`)** | **4 campaigns** | Caught during subsequent lateral movement |
| **Median Detection Offset** | **+0.84 hours** | Median catch occurs in under 51 minutes |
| **Mean Detection Offset** | **-0.36 hours** | Balanced by early-activity alerts |

---

## 4. Detector Precedence & Lead-Lag Dynamics

When both short-term sequence and long-term graph detectors participate, their operational interaction exhibits clear specialization:

1. **Short-Term Sequence Detector (GRU)**:
   - **Timescale**: 1-hour rolling window.
   - **Role**: Detects rapid authentication velocity, unexpected logon types (e.g. Network vs Interactive), and authentication failure bursts.
   - **Operational Lead**: Leads during sudden credential stuffing and rapid lateral hops.
2. **Long-Term Graph Detector (GAE)**:
   - **Timescale**: 24-hour network topology.
   - **Role**: Detects novel user-to-host edges, abnormal connection fan-out, and structural degree growth.
   - **Operational Lead**: Provides standing corroboration available at day start, anchoring the baseline against sporadic single-detector false alarms.
3. **DualScope Temporal Fusion**:
   - Applies an exponential co-occurrence boost (`tau = 6h`, `boost = 0.15`) when a sequence anomaly occurs near a historical graph structural anomaly.
   - Elevates stealthy compromises that fall below single-detector thresholds into the analyst queue (e.g. user `U4448@DOM1`).

---

## 5. Qualitative Error Case Studies

### Case Study 1: True Positive (Dual-Detector Corroboration & Rapid Catch)
* **Target:** User `U66@DOM1` on Day 13
* **Window:** `[1065601, 1069201)` (Day 13, Hour 9)
* **Model Scores:** Fused = 0.914370 | Sequence = 0.9994 | Graph = 0.9996 | Temporal Boost = 0.0000
* **Forensic Findings:**
  - Red-team attacker initiated lateral movement from host `C1783` at timestamp `1066394`.
  - The sequence detector flagged an immediate surge in authentication attempts (`seq_score = 0.9994`).
  - Simultaneously, the prior day's graph embedding flagged `U66@DOM1` for anomalous host fan-out (`graph_score = 0.9996`).
  - The alert became available at window completion (`1069201`), producing an offset of **+0.78 hours (46 minutes)** after initial compromise.
* **Operational Implication:** Dual-timescale consensus rapidly catches high-velocity lateral movement while cross-corroboration prevents single-detector false alarms.

---

### Case Study 2: True Positive (Sub-Hour Catch via Temporal Lead Time)
* **Target:** User `U4448@DOM1` on Day 14
* **Window:** `[1177201, 1180801)` (Day 14, Hour 16)
* **Model Scores:** Fused = 0.887215 | Sequence = 0.9972 | Graph = 0.9984 | Temporal Boost = 0.0312
* **Forensic Findings:**
  - Attacker compromised `U4448@DOM1` at timestamp `1179675` and hopped to `C529`.
  - Standalone GRU ranked this event below budget 38 (`seq_score = 0.9972`) because the event volume was modest.
  - However, Graph GAE detected prior structural shifts on the user's historical graph (`graph_score = 0.9984`).
  - The temporal engine applied a co-occurrence boost (+0.0312), elevating the alert into the Top 38 queue.
  - The alert became available at timestamp `1180801`, just **18 minutes (+0.31 hours)** after the attack event.
* **Operational Implication:** Demonstrates how multi-timescale fusion catches stealthy, low-volume compromises that fall below single-detector thresholds.

---

### Case Study 3: Assumed False Positive (Automated Domain Controller Synchronization)
* **Target:** Machine Account `C395$@DOM1` on Day 13
* **Window:** `[1087201, 1090801)` (Day 13, Hour 15)
* **Model Scores:** Fused = 0.914488 | Sequence = 0.9997 | Graph = 0.9992 | Temporal Boost = 0.0000
* **Forensic Findings:**
  - Computer account `C395$` (denoted by the `$` suffix) initiated dozens of rapid Kerberos authentications across multiple domain controllers.
  - The sequence model flagged the velocity burst as anomalous, and the graph model flagged the multi-host fan-out.
  - In standard security analytics, unlabelled events are assumed negative, classifying this as a False Positive.
* **Operational Implication:**
  - Real-world enterprise maintenance and Active Directory replication scripts mimic attacker lateral movement.
  - In production SOC operations, machine service accounts (`$` suffix) should be partitioned into a dedicated service baseline or filtered via deterministic rules, preventing routine domain maintenance from consuming analyst alert capacity.

---

### Case Study 4: False Negative (Single-Credential Living-off-the-Land Logon)
* **Target:** User `U3277@C2519` on Day 13
* **Window:** `[1069201, 1072801)` (Day 13, Hour 10)
* **Model Scores:** Fused = 0.006001 | Sequence = 0.1590 | Graph = 0.0000 | Temporal Boost = 0.0000
* **Forensic Findings:**
  - Attacker executed a single interactive Kerberos logon at timestamp `1072660` to workstation `C2519`.
  - The user had only 1 authentication event in that hour with zero failures, producing a sequence anomaly score near the dataset median (`seq_score = 0.1590`).
  - The user had no prior graph anomalies (`graph_score = 0.0000`).
  - Consequently, the fused compromise probability was 0.006, ranking at #421,307 out of 433,399 units.
* **Operational Implication:**
  - Single isolated authentications using valid credentials have zero statistical footprint in authentication anomaly models.
  - Detecting such stealthy attacks requires complementary telemetry, such as endpoint process execution logs (EDR process trees) or threat intelligence indicators.

---

## 6. Uncertainty & Label Limitations

In accordance with project evaluation requirements:
1. **Assumed Negatives vs. True Benignity**: Unlabelled enterprise authentication events are assumed negative under standard security evaluation. However, genuine enterprise environments contain background noise, misconfigurations, and unlabelled attacker activity.
2. **Early-Warning Uncertainty (e.g. `U1653@DOM1`, `U78@DOM1`)**: When an alert fires hours before the earliest official red-team label, we do **not** automatically claim it is an early-warning precursor. It may represent unlabelled reconnaissance, or it may simply reflect legitimate morning baseline variance before an adversary hijacked the account.
3. **Unmatched Label Cohort**: 14 ground-truth red-team events on Days 9 and 13 did not match raw `auth.txt` records (documented in `lanl_splits.md`). They remain accounted for in all total recall denominators.

---

## 7. Reproduction Command

To reproduce this timing and error analysis and generate the output JSON:

```powershell
.\.venv\Scripts\python scripts/analyse_timing_and_errors.py `
  --seq-scores-dir seq-gru-ae-v1-L32-h32/sequence_scores_seq_validation/outputs/sequence_scores/seq-gru-ae-v1/scores `
  --graph-scores-dir "Long Term Graphs/scores" `
  --labels-dir data/processed/lanl_auth_days_01_30/redteam_labels/labels `
  --budget 38 `
  --output-dir outputs/evaluation
```
