# Master Comparative Evaluation Matrix (Tasks 5.4 & 9.3)

## 1. Evaluation protocol

1. **Unit**: every model is scored on the same user-hours (`[window_start, window_start + 3600)`).
2. **Split**:
   - **Training (Days 08–12)**: 1,969,281 user-hours, 84 red-team attack user-hours (82 of them on Day 09). Used to fit the supervised models and every label-based threshold.
   - **Evaluation (Days 13–16)**: 1,776,074 user-hours, 136 attack user-hours. Never used for fitting or threshold choice.
   - **Days 17–30**: not read by this evaluation. They are reserved for the final one-time test of the frozen model.
3. **Thresholds**: the sequence and graph alert cut-offs (which drive the temporal boost and lead time) and the temporal fusion alert threshold (which feeds incident generation) are all chosen by best F1 on Days 08–12. The `is_alert` flags in the exported detector scores are not used, because their thresholds were chosen on all validation days 8–16, which include the evaluation days.
4. **Alert budget**: 38 alerts/day (152 over the 4 evaluation days), the top 38 user-hours per day by score. 38 is the GRU's alert rate at its validation-chosen threshold (340 alerts over 9 validation days ≈ 37.8/day), used so every model is compared at the same workload. It is not a measured SOC capacity. Because the budget is per day, recall is capped: Days 13–16 have 68, 36, 15 and 17 attack hours, so at most 38 + 36 + 15 + 17 = 106 of 136 can be caught (77.9%). (Day 09 alone, with 82 attack hours, would be capped at 38 / 82 = 46.3%.)
5. **Primary vs. operational view**: the hourly matrix is the primary comparison. Incident-level triage (Task 5.4) is reported separately with its review workload.

---

## 2. Hourly matrix (Days 13–16, 38 alerts/day)

| Model | Hits @ 152 | Precision | Recall (of 136) | PR-AUC (AP) | Best F1 | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| GRU alone (sequence) | 4 | 2.63% | 2.94% | 0.004572 | 0.0407 | Short-term detector |
| Graph GAE alone | 1 | 0.66% | 0.74% | 0.001917 | 0.0206 | Long-term detector |
| Maximum fusion | 0 | 0.00% | 0.00% | 0.003127 | 0.0241 | Baseline: max(seq, graph) |
| Average fusion | 8 | 5.26% | 5.88% | 0.012272 | 0.0714 | Baseline: 0.5·seq + 0.5·graph |
| Temporal fusion | 8 | 5.26% | 5.88% | 0.014148 | 0.0711 | Average fusion + co-alert boost (τ = 6 h, boost 0.15) |
| Supervised fusion: logistic regression | 10 | 6.58% | 7.35% | 0.016695 | 0.0733 | seq + graph + temporal boost + lead time |
| Known-attack lookup | 0 | 0.00% | 0.00% | 0.004253 | 0.0240 | Non-ML: users in confirmed history before Day 13 |
| **DualScope final: gradient boosting** | **16** | **10.53%** | **11.76%** | **0.049843** | **0.1333** | **GRU score + hourly authentication counts** |

The final model is the HistGradientBoosting fusion from branch `experiment/features-v2` (`docs/experiment_features_v2.md`), fitted on Days 08–12 with its settings fixed in advance. Its inputs are the GRU score and per-hour counts (events, failures, distinct sources and destinations, first-time user→source, host→host and user→destination connections, NTLM, Network logon type, LogOn) plus a machine-account flag. Its scores are exported by that experiment's code and passed in with `--final-model-scores`. Ties are broken by that experiment's fixed random order (seed 0), so the row matches its recorded result (16/136, AP 0.04984).

---

## 3. Budget sensitivity (Days 13–16)

Hits / precision / recall (of 136) at each daily budget:

| Alerts/day | Total | GRU alone | Logistic regression fusion | Final: gradient boosting |
| :--- | :---: | :---: | :---: | :---: |
| 10 | 40 | 0 / 0.00% / 0.00% | **6 / 15.00% / 4.41%** | 5 / 12.50% / 3.68% |
| 25 | 100 | 1 / 1.00% / 0.74% | 9 / 9.00% / 6.62% | **14 / 14.00% / 10.29%** |
| 38 | 152 | 4 / 2.63% / 2.94% | 10 / 6.58% / 7.35% | **16 / 10.53% / 11.76%** |
| 50 | 200 | 4 / 2.00% / 2.94% | 10 / 5.00% / 7.35% | **16 / 8.00% / 11.76%** |
| 100 | 400 | 8 / 2.00% / 5.88% | 18 / 4.50% / 13.24% | **25 / 6.25% / 18.38%** |

- Both supervised fusions beat the GRU at every budget.
- The gradient boosting model catches the most attacks from 25 alerts/day upward. At 10 alerts/day the logistic regression fusion is one hit ahead (6 vs 5); a one-hit difference is within chance at these counts.

---

## 4. Findings

### Fusion vs. single detectors
- On unseen Days 13–16 the GRU alone caught 4 attacks and the graph detector 1. Every fusion except maximum fusion did better.
- The final gradient boosting model caught **16 (4× the GRU)**, with AP **0.0498 (about 11× the GRU's 0.0046)**.
- The final model's hits are spread over all four days (10, 2, 3, 1). The logistic regression fusion's are on Days 13–14 only (7, 3, 0, 0).
- The logistic regression fusion caught 10 (2.5× the GRU). Average and temporal fusion caught 8 each. The gaps between these three (8, 8, 10) are too small to rank them reliably with 136 attacks.

### What drives the logistic regression fusion
Fitted coefficients (raw inputs): seq_score 6.75, graph_score 1.78, temporal_boost 0.004, log_lead_time 2.99. The temporal boost contributes almost nothing. The lead-time input is non-zero only when both detectors alert, so it acts mainly as a "both detectors alerted" indicator. The gain over the GRU comes from adding the graph score and that co-alert signal, not from the exponential boost.

### Why maximum fusion failed
Taking the maximum keeps the highest false-positive scores of both detectors, so uncorroborated single-detector spikes fill the budget.

### Known-attack lookup
This baseline gives the same score to every hour of a user who appears in the confirmed history before Day 13, so it cannot rank among those hours, and its top 38 per day contained no attack hours.

### Effect of fitting the cut-offs on Days 08–12
Previously the temporal features used the exported `is_alert` flags (thresholds chosen on Days 8–16). Fitting the cut-offs on Days 08–12 instead changed little: the logistic regression fusion stays at 10/152 (AP 0.0173 → 0.0167), temporal fusion stays at 8/152 (AP 0.0140 → 0.0141), and the temporal fusion alert threshold is unchanged (0.998188).

---

## 5. Incident-level triage (Task 5.4, Days 13–16)

Incidents are built from temporal-fusion alerts (threshold fitted on Days 08–12) merged per user with a 2-hour gap.

Counting rules:
1. An incident is a hit only if at least one labelled attack hour of that user falls inside `[start_time, end_time)`.
2. Only labelled attack hours inside hit incidents count towards recall.
3. Review workload is reported as the total hours covered by the reviewed incidents.

| Measure | Value |
| :--- | :--- |
| Raw temporal-fusion alerts | 161 |
| Incidents | 74 (54.0% fewer tickets) |
| Hours reviewed (all 74 incidents) | 183 (2.47 per incident) |
| Malicious incidents | 4 / 74 (5.41%) |
| Attack hours caught | 8 / 136 (5.88%) |
| Attacker accounts caught | 3 (`U1653@DOM1`, `U4448@DOM1`, `U66@DOM1`) |

Incidents reduce the number of tickets, not the work: the 74 incidents cover 183 hours, more than the 161 alerting hours, because they include the gaps between alerts. They also do not improve detection here: 183 reviewed hours caught 8 attack hours, while the final model's top 152 single hours caught 16.

---

## 6. Ablation note

An earlier test that added graph structural counters (`new_edge_count`, `degree_growth`) to the logistic regression fusion was fitted and scored on Day 09 only (in-sample), with unscaled inputs. Its result is not evidence either way and is not reported. The tested counters were graph snapshot counters, not the 5-minute burst counter (`peak_unique_destinations_300s`).

---

## 7. Reproduction

1. Export the final model's Days 13–16 scores (fitted on Days 08–12). The exporter reuses the unchanged supervised fusion code of branch `experiment/features-v2` and its cached inputs (`outputs/experiment_v2/units_A.parquet`, `fusion_counts.parquet`), and refuses to write unless it reproduces 16/136 and AP 0.04984. Until that branch is merged, pass a checkout of it:

```powershell
.\.venv\Scripts\python scripts/export_final_model_scores.py --experiment-root <checkout of experiment/features-v2>
```

2. Run the matrix (detector score paths are the defaults: `outputs/sequence_scores/seq-gru-ae-v1-L32-h32-91e4b11d34/scores` from Member 2's validation package and `outputs/graph_scores_v1/scores` from the graph package):

```powershell
.\.venv\Scripts\python scripts/evaluate_comparison_matrix.py `
  --final-model-scores outputs/experiment_v2/final_model_scores_days13_16.parquet `
  --output-dir outputs/evaluation
```

Without `--final-model-scores` the final model's row is omitted.
