# Task 9.1: Formal Evaluation Protocol

**Owner:** Member 4 (Justin Teo / `TeoJustin`)  
**Scope:** Team-wide evaluation contract for DualScope intrusion detection and comparative experiments.  
**Relevant Issues:** [#30 (Task 9.1)](https://github.com/Sytechia2/ICT-3214-DualScope/issues/30), [#31 (Task 9.2)](https://github.com/Sytechia2/ICT-3214-DualScope/issues/31), [#32 (Task 9.3)](https://github.com/Sytechia2/ICT-3214-DualScope/issues/32).

---

## 1. Protocol Objective

This protocol predefines the scoring units, ground-truth label mappings, performance metrics, threshold-selection rules, and negative class assumptions across all detectors in the DualScope project. 

By fixing these definitions prior to test split scoring, we ensure that:
1. All models (Isolation Forest Baseline, Short-Term GRU Autoencoder, Long-Term Graph Autoencoder, and Fused DualScope) are evaluated on identical, strictly comparable cohorts.
2. Target leakage and data snooping are strictly prevented.
3. Metric trade-offs (e.g. Precision vs. Recall under extreme class imbalance) reflect genuine operational security requirements.

---

## 2. Evaluation Unit & Temporal Alignment

### 2.1 Primary Scoring Unit
The primary evaluation unit is the **User-Hour**:
$$\text{Unit} = (\text{acting\_user}, \text{hour\_start})$$

* `acting_user`: The canonical source user account initiating authentication (Task 2.2).
* `hour_start`: Half-open hourly window boundary defined as:
  $$\text{hour\_start} = 1 + \left\lfloor \frac{\text{timestamp} - 1}{3600} \right\rfloor \times 3600$$
  Window interval: $[\text{hour\_start}, \text{hour\_start} + 3600)$.

### 2.2 Decision Time & Causal Availability
Every prediction record carries `score_available_at`:
* For the **Sequence Detector** and **Baseline**: `score_available_at = window_end` (end of the hour).
* For the **Graph Detector**: `score_available_at = cutoff_time` (daily cutoff).
* **Causal Decision Rule:** A decision rendered at time $t$ may only consume scores where:
  $$\text{score\_available\_at} \le t$$
* **Graph Staleness Horizon:** Because graph scores are updated daily, a graph score may be carried forward into subsequent hourly windows for at most **24 hours (86,400 seconds)**:
  $$t - \text{score\_available\_at} \le 86,400$$
  Older graph scores are marked `stale_graph` and treated as unavailable.

---

## 3. Ground Truth & Label Matching Policy

### 3.1 Positive Label Assignment
Ground-truth compromise events are sourced from the official LANL Red-Team dataset (`redteam.txt`):
* Each record specifies `(timestamp, user, source_computer, destination_computer)`.
* An evaluation unit `(acting_user, hour_start)` is assigned ground-truth label $y = 1$ if:
  $$\exists \text{ event} \in \text{RedTeam} \quad \text{s.t.} \quad \text{event.user} = \text{acting\_user} \quad \land \quad \text{hour\_start} \le \text{event.timestamp} < \text{hour\_start} + 3600$$
* Positive labels are deduplicated per unit: multiple red-team authentications by the same user within the same hour constitute a single positive user-hour.

### 3.2 Treatment of the 14 Unmatched Red-Team Labels
During Task 2.1 inspection and Task 2.3 split creation, **14 red-team events** (12 on Day 9, 2 on Day 13) were identified that have no corresponding matching event in `auth.txt`:
* **Policy:** In accordance with the Task 2.1 decision, these 14 labels are **retained in the label ground-truth** and are never discarded or altered.
* **Dual Recall Reporting:** Because a user-hour without authentication events cannot produce an available anomaly score by an unsupervised sequence model, all evaluations must explicitly report two recall figures:
  $$\text{Scorable Recall} = \frac{TP}{\text{Positives with Scorable Activity}}$$
  $$\text{Full Recall} = \frac{TP}{\text{Total Ground Truth Positives (including unscored)}}$$
  This transparency ensures detection rates cannot be artificially inflated by filtering unscored attack windows.

### 3.3 Presumed Operational Negatives
In enterprise authentication logs, exhaustive negative ground truth does not exist (some benign-appearing accounts may be undiscovered compromises or background red-team reconnaissance):
* **Operational Assumption:** Every active user-hour lacking a matching red-team record is treated as an operational negative ($y = 0$).
* **Prevalence Context:** The dataset exhibits extreme class imbalance (validation split prevalence is $\approx 5.87 \times 10^{-5}$, or 1 attack per 17,000 user-hours). Metrics that are sensitive to raw negative volume (e.g. plain accuracy) are misleading and are not used for model ranking.

---

## 4. Primary & Secondary Evaluation Metrics

Model comparison and threshold tuning are governed by the following formal hierarchy:

| Metric | Formulation | Role in Protocol |
| :--- | :--- | :--- |
| **Average Precision (PR-AUC)** | $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$ | **Primary Model Selection Metric.** Evaluates ranking quality across all thresholds under severe class imbalance without relying on arbitrary cut-offs. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | **Secondary Discrimination Metric.** Measures pairwise ranking separation between positive and negative user-hours. |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | **Decision Threshold Objective.** The operational alert threshold $T^*$ is selected to maximize $F_1$ on the validation split. |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | **SOC Operational Feasibility.** In a network with 450,000 user-hours/day, an acceptable FPR must remain $< 0.05\%$ ($< 225$ false alarms/day). |
| **Lift Over Random** | $\frac{\text{AP}}{\text{Prevalence}}$ | **Signal Quality.** Measures factor improvement in precision over a naive random baseline. |
| **Detection Lead Time** | $t_{\text{alert}} - t_{\text{first\_compromise}}$ | **Early Warning Metric.** Evaluates which detector alerted first and by how many hours. |

---

## 5. Non-Zero Missingness Policy

* Unsupervised models may have hours with `insufficient_history` (e.g. Day 1 warm-up, incomplete 24h context) or `no_activity`.
* **Strict Rule:** Missing detector scores must remain `null` / `NaN`, and must **never be imputed as zero (`0.0`)**.
* Imputing missing values with zero falsely asserts that the user's behavior was completely normal and distorts ranking.
* The fusion engine handles single-detector availability via dynamic weight renormalisation:
  $$\text{If } S_{\text{graph}} \text{ is missing:} \quad S_{\text{fused}} = \frac{w_{\text{seq}}}{w_{\text{seq}} + 0} \cdot S_{\text{seq}} = S_{\text{seq}}$$

---

## 6. Threshold Selection & Data Isolation

1. **Training Split (Days 1–7):** Strictly unsupervised model parameter fitting. Training compromise accounts are dropped to prevent contamination. No threshold selection or evaluation is performed on training data.
2. **Validation Split (Days 8–16):** Hyperparameter grid search, fusion weight tuning ($w$), temporal decay tuning ($\tau$), score calibration, and alert threshold selection ($T^*$).
3. **Test Split (Days 17–30):** Strictly held-out benchmark data. Model weights, fusion parameters, calibration knots, and alert thresholds are **frozen** before test scoring commences. Test labels are read only once for final reporting.

---

## 7. Comparative Benchmark Matrix (Task 9.3 Structure)

Final experimental reporting will populate the comparative benchmark table on identical validation and test splits:

| Model Architecture | Task | Timescale | Input Representations | Primary Metric (PR-AUC) | Best F1 | Alert Volume / Day |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Flat Isolation Forest** | 9.2 | 1 Hour | 16 tabular aggregated features | *TBD* | *TBD* | *TBD* |
| **Short-Term GRU Autoencoder** | 3.3 | 1 Hour | Event sequences ($L=32$, 12 features) | 0.00598 | 0.039 | 37.8 |
| **Long-Term Graph GAE** | 4.3 | 24 Hours | Dynamic bipartite user-computer graph | 0.02640 | 0.077 | 42.1 |
| **DualScope Fused Engine** | 5.3 / 9.3 | Multi (1h + 24h) | Fused sequence + graph + temporal decay | *TBD* | *TBD* | *TBD* |

---

## 8. GenAI Investigation Review Rubric (Task 9.5)

For qualitative investigation validation (coordinated with Member 5):
* **Factual Grounding:** Every claim in the LLM-generated incident summary must link directly to an existing `auth.txt:<source_line>` reference.
* **Hallucination Rate:** Percentage of generated incident statements lacking empirical backing in the evidence chunk. Target: 0.0%.
* **Triage Accuracy:** Agreement between LLM-recommended priority and fused quantitative priority (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
