# Long-Term Graph Detector: Evaluation and Improvement Analysis

## Executive summary

The graph work now has two results that must remain separate:

- the unsupervised graph autoencoder has useful ranking ability but weak standalone alert classification; its original test F1 is `0.0273` and the validation-selected counter enhancement raises it to `0.0769`; and
- the supervised confirmed-relationship history reaches retrospective test F1 `0.8406`, but detects repeated confirmed relationships rather than novel anomalies.

The chronological overfitting check for the confirmed-history extension currently passes, with no later-period F1 collapse. This is limited evidence rather than proof, and the hybrid result still requires a fresh holdout. See [the dedicated overfitting guide](graph_overfitting.md).

The following table is the original GAE baseline:

| Metric | Validation | Held-out test |
|---|---:|---:|
| F1 | 0.0940 | **0.0273** |
| Precision | 0.0815 | **0.0154** |
| Recall | 0.1111 | **0.1176** |
| ROC-AUC | 0.8886 | **0.9210** |
| Average precision (PR-AUC) | 0.0295 | **0.00874** |
| Positive prevalence | 0.0492% | **0.00919%** |
| False-positive rate | 0.0616% | **0.0689%** |

The validation-selected threshold is `0.9992886186840491`. On the test split it produced:

- 4 true positives;
- 255 false positives;
- 30 false negatives;
- 369,821 true negatives; and
- 259 total alerts across 370,110 scored user-day units.

Therefore:

```text
precision = 4 / (4 + 255) = 0.01544
recall    = 4 / (4 + 30)  = 0.11765
F1        = 2PR / (P + R) = 0.02730
```

Accuracy is deliberately omitted. With only 34 positive test units among 370,110 scored units, predicting every unit as normal would appear highly accurate while detecting nothing.

## What the metrics mean

The ROC-AUC of 0.921 shows that a randomly chosen compromised user-day is usually ranked above a randomly chosen normal user-day. This is promising, but ROC-AUC can look strong when negative examples greatly outnumber positives.

Average precision is more informative for this workload. Test average precision is 0.00874, or about 95 times the random baseline prevalence of 0.0000919. The detector therefore contains useful signal, but the top-ranked tail is not clean enough to generate precise alerts. At the frozen threshold, roughly 65 false alerts are produced for every true alert.

The validation/test difference also shows temporal distribution shift. Validation contains 135 positive user-day units and test contains only 34. Attack behavior, affected users, and background activity differ between the periods, so a threshold or structural rule that performs well during validation may not transfer to test.

## Experiments performed

### Short-window authentication counters

Three causal node features were added and materialized for all 508.9 million events:

- peak authentications in an exact rolling 60-second interval;
- peak authentications in an exact rolling five-minute interval; and
- peak unique destination computers in an exact rolling five-minute interval.

The burst-aware autoencoder reduced training reconstruction loss from 0.3954 to 0.3105 and increased test recall, but using the counters only as encoder inputs did not improve alert quality:

| Detector | Test precision | Test recall | Test F1 | Test AP |
|---|---:|---:|---:|---:|
| Original GAE | 0.0154 | 0.1176 | 0.0273 | 0.00874 |
| Burst features inside GAE | 0.00730 | 0.2059 | 0.0141 | 0.00727 |

The counters were therefore tested as explicit anomaly evidence rather than only latent encoder inputs. A bounded set of weights was compared using validation F1. The validation-selected formula was:

```text
improved score = 0.25 × original graph score
               + 0.75 × validation-scaled peak unique destinations in rolling 5 minutes
```

This candidate improved the held-out result without using test metrics for selection:

| Metric | Original GAE | Counter-enhanced score | Change |
|---|---:|---:|---:|
| Validation F1 | 0.0940 | **0.1196** | +27.2% |
| Test F1 | 0.0273 | **0.0769** | +181.7% |
| Test precision | 0.0154 | **0.0450** | +191.4% |
| Test recall | 0.1176 | **0.2647** | +125.0% |
| Test average precision | 0.00874 | **0.02640** | +201.9% |
| Test ROC-AUC | **0.9210** | 0.9106 | -1.1% |

At the validation-selected threshold of `0.9981996450132613`, the improved test score produced 9 true positives, 191 false positives, 25 false negatives, and 369,885 true negatives. The large F1 improvement comes from both finding more compromises and reducing false alerts. The small ROC-AUC decrease means global ranking became slightly worse even though the operational high-score tail became substantially better.

### Latent dimension

Models with latent dimensions 8, 16, and 32 were trained for 20 epochs on the same leakage-safe Days 2–7 snapshots. Their final reconstruction losses were:

| Latent dimension | Final training loss |
|---:|---:|
| 8 | 0.4141 |
| 16 | **0.3954** |
| 32 | 0.3996 |

The existing 16-dimensional model retained the lowest reconstruction loss. Reconstruction loss alone must not be treated as detection quality, so the candidates were also compared using validation labels.

### Aggregation and structural blending

The bounded comparison included maximum, top-3 mean, top-5 mean, and new-edge-only model aggregation. It also tested validation-scaled new-edge counts, new-edge ratios, degree growth, and fixed model/structure blend weights.

The validation winner was an equal blend of the 16-dimensional top-5 model score and new-edge count:

| Metric | Validation | Test |
|---|---:|---:|
| Average precision | 0.0991 | 0.00446 |
| F1 | 0.2275 | **0.0000** |
| True positives | 29 | **0** |
| False positives | 91 | 59 |

This candidate must be rejected. Although it substantially improved validation results, it detected none of the held-out test positives. This is direct evidence that validation red-team activity contains a new-edge pattern that does not generalize to the later campaign. Selecting it using test results would leak test information.

### Threshold policies

Changing only the threshold trades precision for recall but does not materially improve test F1:

| Validation-only threshold policy | Test precision | Test recall | Test F1 | Test alerts |
|---|---:|---:|---:|---:|
| Maximum validation F1 | 0.0154 | 0.1176 | 0.0273 | 259 |
| At least 20% validation recall | 0.0100 | 0.2059 | 0.0192 | 697 |
| At least 25% validation recall | 0.0112 | 0.3529 | 0.0217 | 1,071 |
| Maximum validation F2 | 0.0102 | 0.4118 | 0.0198 | 1,379 |
| At least 50% validation recall | 0.0056 | 0.6765 | 0.0112 | 4,075 |

Threshold adjustment is useful for choosing an operational alert budget, but it cannot repair weak separation in the high-score tail.

### Option A: supervised graph-feature classifier

A supervised alternative was trained from labelled user-days in Days 8–12. Its inputs remain graph-derived: the original and burst GAE scores, edge and degree statistics, short-window graph counters, and source-host structural counts. Logistic regression, histogram gradient boosting, random forest and extra trees were compared using Days 13–16 average precision. The histogram gradient-boosting classifier won; its alert threshold was also frozen on Days 13–16 before the Days 17–30 comparison.

| Metric | Validation, Days 13–16 | Retrospective test, Days 17–30 |
|---|---:|---:|
| Precision | 0.1982 | 0.0272 |
| Recall | 0.2500 | 0.1471 |
| F1 | **0.2211** | **0.0459** |
| Average precision | 0.1167 | 0.0120 |
| ROC-AUC | 0.9361 | 0.8504 |

The supervised model overfits the earlier attack campaigns and does not replace the selected counter-enhanced unsupervised score, whose test F1 is `0.0769` and average precision is `0.0264`. Option A is retained as a negative experiment. Only 47 positive user-days were available for supervised fitting, which is inadequate for learning the later campaign reliably.

The reproducible trainer is `scripts/train_supervised_graph_detector.py`; its frozen model and metrics are written to `outputs/graph_evaluation/supervised_detector/`. Because the project test period had already been inspected during earlier development, this result is explicitly retrospective and would need confirmation on a fresh holdout.

### Confirmed relationship history (hybrid extension)

The authentication graph originally discarded `source_computer`, even though the labels identify the full user/source/destination relationship. A separate causal signature layer was therefore evaluated. It stores exact `(user, source computer, destination computer)` relationships from confirmed incidents in Days 8–16, freezes that history before Day 17, and alerts when the exact relationship recurs in Days 17–30.

This hybrid extension crosses the requested 80% target on the existing test period:

| Metric | Confirmed relationship history |
|---|---:|
| Precision | **0.8286** |
| Recall | **0.8529** |
| F1 | **0.8406** |
| True positives | 29 |
| False positives | 6 |
| False negatives | 5 |
| True negatives | 370,070 |

No Day 17–30 label is inserted into the history. The history contains 410 relationships and is frozen at timestamp `1382401` before test events are scanned.

This result must not be described as an 84% F1 for the graph autoencoder. It is a supervised threat-history/signature result that detects recurrence of already confirmed malicious relationships. Its advantage is high precision on repeated campaigns; its limitation is that it cannot detect a completely new relationship. The counter-enhanced GAE remains the appropriate novelty detector and has test F1 `0.0769`.

The two channels are now exported together without conflating their metrics. Confirmed recurrence is the primary high-confidence decision, and GAE anomalies are retained for novel-behaviour review. On the same retrospective test:

| Exported decision | Precision | Recall | F1 |
|---|---:|---:|---:|
| Confirmed relationship, primary | **0.8286** | 0.8529 | **0.8406** |
| Counter-enhanced GAE alert | 0.0450 | 0.2647 | 0.0769 |
| Union: any graph signal | 0.1327 | **0.8824** | 0.2308 |

The union is useful as an analyst-review queue but is not a high-precision detector. Its additional GAE false positives reduce F1 sharply.

Because this extension was developed after inspecting the original test outcome, `0.8406` is a retrospective backtest estimate. A fresh chronological holdout or a different dataset is required for an unbiased final performance claim.

#### Temporal overfitting check

The automated check compares two non-overlapping rolling-origin evaluations. Days 8–12 form the relationship history for an evaluation on Days 13–16; the expanded Days 8–16 history is then frozen for the later Days 17–30 holdout. It fails when later F1 drops by both more than 0.10 absolutely and more than 50% relatively, or when the holdout contains fewer than 30 positive units.

```bash
.venv/bin/python scripts/check_graph_overfitting.py \
  --development outputs/graph_evaluation/threat_history_temporal_validation_metrics.json \
  --holdout outputs/graph_evaluation/threat_history_test_metrics.json \
  --output outputs/graph_evaluation/threat_history_overfitting_check.json \
  --fail-on-overfitting
```

The current run passes: F1 increases from `0.2222` in Days 13–16 to `0.8406` in Days 17–30, so there is no temporal F1 collapse. This pass is limited evidence, not proof that the detector is free from overfitting. In particular, it does not remove the need for a fresh holdout because the hybrid method was devised after the original test results were inspected.

## Likely causes

1. **Self-reconstruction is easier than anomaly prediction.** The current graph is provided to the encoder and then reconstructed. A new relationship participates in message passing before it is scored, which can make the relationship easier to reconstruct. This limitation was documented in the original design but is now visible in the results.

2. **Random negatives are too easy.** Uniformly sampled absent user-computer pairs are often implausible. The model can learn broad node-type and activity differences without learning the subtle distinction between a normal new destination and lateral movement.

3. **Daily user-level labels are extremely sparse.** Authentication events are aggregated into user-day graph scores. A compromised event can be diluted by many normal relationships from the same user, while a normal high-volume user may naturally create several new edges.

4. **The structural features are weak.** Node type, degree, strength, failure volume, and degree growth omit relationship recency, historical frequency, destination popularity, user role/cohort, and repeated uncommon access.

5. **A single global threshold is brittle.** Service accounts, administrators, computer accounts, and ordinary users have very different degree and authentication distributions.

6. **Validation and test attacks differ.** The failed new-edge blend demonstrates that a feature strongly associated with the validation campaign can disappear from the later campaign.

## Recommended improvements

### 1. Change reconstruction into causal link prediction

This is the highest-priority model change. Encode only historical relationships ending at time `T`, then score authentication edges in `(T, T + horizon]`. A candidate edge must not be included in the graph used to predict itself. Compare one-hour, six-hour, and one-day prediction horizons using validation average precision.

Expected benefit: truly unseen or poorly predicted user-host relationships cannot help reconstruct themselves.

### 2. Add edge-history features

For each candidate user-host relationship, include features calculated strictly before its event/window:

- whether the edge has ever appeared before;
- seconds or days since last access;
- historical authentication count;
- number of active days containing the edge;
- user-specific destination frequency;
- destination popularity across other users;
- success/failure ratio; and
- user degree growth over 1-, 3-, and 7-day histories.

These can feed an edge decoder or a small calibrated classifier alongside the GAE score. All preprocessing and scaling must be fitted on training/validation only.

The completed burst experiment supports this approach: explicit five-minute unique-destination evidence generalized better than embedding the same signal only inside the autoencoder.

### 3. Use hard bipartite negatives

Retain valid user-computer sampling, but replace many uniform negatives with harder examples:

- popular hosts not previously accessed by the user;
- hosts accessed by peer users but not the current user;
- destinations in a different historical neighborhood; and
- temporally plausible absent edges for active users.

This teaches the decoder to separate realistic-but-unexpected relationships rather than obvious random non-edges.

### 4. Increase temporal resolution

Keep a multi-day historical graph but emit scores every hour or six hours. This preserves long-term context while aligning more closely with red-team event times and the sequence detector. It also reduces dilution from a full day of normal activity.

### 5. Calibrate by user cohort

Create training-derived cohorts such as ordinary users, service/computer accounts, and high-degree administrative users. Fit separate score calibrators or include cohort-conditioned baselines. Do not fit a separate threshold for individual users with insufficient history.

### 6. Use fusion rather than relying on graph alerts alone

The graph model's ROC-AUC indicates useful ranking signal even though standalone F1 is weak. Its best role may be adding long-term context to the sequence detector. Evaluate sequence-only, graph-only, maximum, average, weighted, and temporal fusion using the same validation-selected thresholds and held-out test split.

### 7. Report alert-budget curves

Alongside maximum-F1 results, report precision and recall at fixed alert budgets such as 10, 25, 50, and 100 alerts per day. This is more operationally meaningful than assuming one threshold transfers across campaigns with different prevalence.

## Recommended next experiment

Implement a causal next-window edge predictor with:

1. a seven-day historical graph;
2. six-hour prediction windows;
3. historical edge frequency and recency features;
4. a 50/50 mix of uniform and hard negatives;
5. top-3 user aggregation; and
6. validation selection by average precision, followed by a threshold chosen for a fixed daily alert budget.

The current model should remain as the documented graph-autoencoder baseline. The structural blend should not be promoted because it failed held-out evaluation.

## Reproducibility artifacts

- Baseline validation metrics: `outputs/graph_evaluation/validation_metrics.json`
- Baseline test metrics: `outputs/graph_evaluation/test_metrics.json`
- Improvement comparison: `outputs/graph_evaluation/improvement_trials/results.json`
- Burst-aware validation metrics: `outputs/graph_evaluation/burst_validation_metrics.json`
- Burst-aware test metrics: `outputs/graph_evaluation/burst_test_metrics.json`
- Counter-signal candidates: `outputs/graph_evaluation/improvement_trials/burst_signal_results.json`
- Validation-F1-selected counter result: `outputs/graph_evaluation/improvement_trials/burst_f1_selected.json`
- Threshold comparison: `outputs/graph_evaluation/improvement_trials/threshold_policies.json`
- Training history: `outputs/graph_evaluation/trained_model/training_log.json`
- Frozen detector: `outputs/graph_evaluation/frozen_detector/`
- Confirmed-relationship test metrics: `outputs/graph_evaluation/threat_history_test_metrics.json`
- Temporal validation metrics: `outputs/graph_evaluation/threat_history_temporal_validation_metrics.json`
- Temporal overfitting check: `outputs/graph_evaluation/threat_history_overfitting_check.json`
- Exploratory supervised stack: `outputs/graph_evaluation/improvement_trials/supervised_stack_results.json`
- Source-host feature study: `outputs/graph_evaluation/improvement_trials/source_host_results.json`
- Five-minute feature study: `outputs/graph_evaluation/improvement_trials/five_minute_results.json`
- Causal threat-history audit: `outputs/graph_evaluation/improvement_trials/causal_threat_history_results.json`
- Supervised Option A metrics: `outputs/graph_evaluation/supervised_detector/metrics.json`
- Supervised Option A frozen classifier: `outputs/graph_evaluation/supervised_detector/supervised_graph_detector.joblib`
- Combined graph alert metrics: `outputs/graph_evaluation/hybrid_alerts/metrics.json`
- Combined graph user-day export: `outputs/graph_evaluation/hybrid_alerts/dataset_day=NN/*.parquet`

Baseline and counter-enhanced model selection used validation data only. The threat-history dictionary also excludes all test labels, but the extension itself was proposed after the baseline test had been inspected; it therefore needs confirmation on a fresh holdout.
