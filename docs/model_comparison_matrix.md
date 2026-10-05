# Master Comparative Evaluation Matrix (Tasks 5.4 & 9.3)

## 1. Executive Summary & Evaluation Protocol

This report provides the unified comparative benchmark for **DualScope**, evaluated under a rigorous out-of-sample temporal holdout protocol:

1. **Identical Unit of Evaluation**: Every model is evaluated on the exact same unit: individual **user-hours** (`[window_start, window_start + 3600)`).
2. **Proper Out-of-Sample Split**:
   - **Training Set (Days 08–12)**: Used exclusively for fitting the `SupervisedFusionModel` and tuning fusion parameters. Contains 1,969,281 user-hours and 84 ground-truth red-team attack user-hours.
   - **Test Set (Days 13–16)**: Held-out validation days completely unseen during model training. Contains 1,776,074 user-hours and 136 ground-truth red-team attack user-hours.
   - **Frozen Test Set (Days 17–30)**: Kept strictly untouched and held back for the final one-time evaluation.
3. **Empirical Alert Budget**: Fixed alert budget of **38 alerts/day** (152 total alerts across the 4-day evaluation period), established from Member 2's empirical GRU alert rate across validation days (340 alerts / 9 days ≈ 37.8 ≈ 38 alerts/day).
4. **Primary vs. Operational Evaluation**:
   - **Hourly Matrix (Primary)**: Standard benchmark across all individual detector architectures and fusion methods.
   - **Incident Triage Matrix (Operational)**: Evaluates multi-hour incident envelopes from Task 5.4, with strict ground-truth temporal overlap and explicit analyst review hours reported.

---

## 2. Primary Hourly Benchmark Matrix (Test Days 13–16)

Evaluated at the fixed budget of **38 alerts/day** (152 total alerts over 1,776,074 user-hours across Days 13–16; total ground-truth attacks = 136):

| Model / Detection Architecture | Hits @ 152 Budget | Precision @ 152 | Recall @ 152 | PR-AUC (AP) | Best F1 | Architecture Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GRU Alone (Sequence)** | 4 / 152 | 2.63% | 2.94% | 0.004572 | 0.0407 | Single-timescale short-term RNN autoencoder |
| **Graph GAE Alone** | 1 / 152 | 0.66% | 0.74% | 0.001917 | 0.0206 | Single-timescale long-term graph autoencoder |
| **Baseline: Maximum Fusion** | 0 / 152 | 0.00% | 0.00% | 0.003127 | 0.0241 | Simple naive baseline: max(seq, graph) |
| **Baseline: Average Fusion (50/50)** | 8 / 152 | 5.26% | 5.88% | 0.012272 | 0.0714 | Simple linear baseline: 0.5*seq + 0.5*graph |
| **DualScope: Temporal Fusion** | 8 / 152 | 5.26% | 5.88% | 0.013969 | 0.0714 | Multi-timescale heuristic (tau=6h, boost=0.15) |
| **DualScope: Supervised Fusion** | **10 / 152** | **6.58%** | **7.35%** | **0.017321** | **0.0733** | **DualScope Core: Trained on Days 8–12** |
| **Known-Attack Lookup** | 0 / 152 | 0.00% | 0.00% | 0.004253 | 0.0240 | Non-ML signature baseline (ThreatHistory) |

---

## 3. Budget Sensitivity Curves (Days 13–16)

Performance across operational daily alert budgets (10, 25, 38, 50, and 100 alerts/day) over the 4 test days:

| Alert Budget / Day | Total Alerts (4 Days) | GRU Alone (Hits & Recall) | Supervised Fusion (Hits & Recall) | Supervised Precision | Supervised Recall (of 136) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **10 alerts/day** | 40 alerts | 0 / 40 (0.00%) | **6 / 40 (4.41%)** | **15.00%** | **4.41%** |
| **25 alerts/day** | 100 alerts | 1 / 100 (0.74%) | **8 / 100 (5.88%)** | **8.00%** | **5.88%** |
| **38 alerts/day** | 152 alerts | 4 / 152 (2.94%) | **10 / 152 (7.35%)** | **6.58%** | **7.35%** |
| **50 alerts/day** | 200 alerts | 4 / 200 (2.94%) | **10 / 200 (7.35%)** | **5.00%** | **7.35%** |
| **100 alerts/day** | 400 alerts | 8 / 400 (5.88%) | **17 / 400 (12.50%)** | **4.25%** | **12.50%** |

### Key Observations from Budget Curves:
- **Low-Budget Dominance**: At 10 alerts/day (a constrained SOC shift), GRU alone catches 0 attacks, whereas Supervised Fusion catches 6 attacks with 15.00% precision.
- **Sustained Superiority**: Across all tested budgets, DualScope Supervised Fusion catches between 2.1x and 8.0x more attacks than the single-timescale GRU baseline.

---

## 4. Key Findings & Discussion

### Did Fusion Outperform Single-Timescale Models on Unseen Days?
- **Yes.** On out-of-sample Days 13–16, GRU alone caught 4 attacks (2.63% precision, 2.94% recall), and Graph GAE alone caught 1 attack.
- DualScope Supervised Fusion caught **10 attacks (6.58% precision, 7.35% recall)**, representing a **2.5x increase in attack catch rate** over GRU alone and a **3.8x lift in PR-AUC** (0.017321 vs 0.004572).
- Simple Average Fusion also outperformed the single models, catching 8 attacks (5.26% precision, 5.88% recall).

### Comparison with Teammate Baseline
- A standard logistic regression tested strictly on Days 13–16 without temporal features caught 4/136 attacks (identical to GRU alone).
- By incorporating multi-timescale temporal proximity (`temporal_boost`) and cross-detector lead time (`log_lead_time`), DualScope Supervised Fusion increased caught attacks from 4 to 10 out of the 136 ground-truth attacks.

### Why Maximum Fusion Failed
- Maximum Fusion achieved 0 hits at budget 152 (AP 0.0031).
- Taking the element-wise maximum combines the extreme false-positive tails of both detectors, allowing uncorroborated single-detector spikes to displace true attacks.

### Non-ML Signature Baseline
- Known-Attack Lookup produced 0 hits at the budget threshold because lateral movement on Days 13–16 involved newly compromised destination hosts that did not match frozen prior-day threat triples.

---

## 5. Task 5.4 Incident-Level Triage & Workload Reduction

In security operations, analysts review clustered multi-hour incident envelopes rather than isolated hourly alerts. 

### Counting Methodology Fix
To prevent inflated triage metrics:
1. **Strict Temporal Overlap**: An incident envelope is counted as a hit if and only if an actual labelled red-team attack hour for that user falls within the incident window `[start_time, end_time)`. Merely being a compromised user on that calendar day is insufficient.
2. **True Attack Hours Caught**: Only the actual ground-truth attack hours inside the incident are credited towards recall.
3. **Explicit Analyst Workload**: Because multi-hour incidents require more review time than single-hour alerts, the total hours reviewed (`sum(duration_hours)`) is explicitly reported alongside incident counts.

### Results on Test Days 13–16:
- **Raw Fused Hourly Alerts**: 161 alerts across the 4 test days
- **Clustered Incident Envelopes**: 74 incidents (**54.0% reduction in triage ticket volume**)
- **Analyst Hours Reviewed**: 183 hours across all 74 incidents (average 2.47 hours per incident)
- **Malicious Incidents Caught**: 4 / 74 (2.63% incident precision)
- **Ground-Truth Attack Hours Caught**: 8 / 136 (5.88% recall of attack hours)
- **Distinct Attacker Accounts Caught**: 3 accounts (`U1653@DOM1`, `U4448@DOM1`, `U66@DOM1`)

---

## 6. Ablation Notes

### Feature Set Selection
The production `SupervisedFusionModel` uses 4 multi-timescale features:
1. `seq_score`: Short-term sequence anomaly score (1-hour window)
2. `graph_score`: Long-term graph novelty score (24-hour window)
3. `temporal_boost`: Exponentially decayed co-occurrence bonus (tau = 6h)
4. `log_lead_time`: Log-scaled lead time between detector alerts

An exploratory test incorporating raw graph structural counters (`new_edge_count` and `degree_growth`) was evaluated on Day 09. These structural counters degraded ranking precision because routine IT administration scripts generate large legitimate network fan-out. Note that these tested counters were graph snapshot counters rather than Zachary's 5-minute burst counter. The 4-feature configuration remains the verified baseline.

---

## 7. Reproduction Command

To reproduce the complete out-of-sample evaluation matrix, budget curves, and incident triage metrics:

```powershell
.\.venv\Scripts\python scripts/evaluate_comparison_matrix.py `
  --seq-scores-dir seq-gru-ae-v1-L32-h32/sequence_scores_seq_validation/outputs/sequence_scores/seq-gru-ae-v1/scores `
  --graph-scores-dir "Long Term Graphs/scores" `
  --labels-dir data/processed/lanl_auth_days_01_30/redteam_labels/labels `
  --output-dir outputs/evaluation
```
