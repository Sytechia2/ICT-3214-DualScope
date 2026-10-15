# Long-term graph detector (Tasks 4.1–4.4)

The detector has two explicitly separated layers:

1. an unsupervised graph autoencoder (GAE) for novel or structurally unusual `acting_user → destination_computer` relationships; and
2. a supervised confirmed-relationship history for exact recurrences of previously confirmed malicious `(user, source computer, destination computer)` relationships.

The GAE builds directed bipartite authentication graphs. A successful authentication contributes `1.0` to an edge and a failed authentication contributes `0.25`; both counts remain separately visible. The default snapshot is the half-open 24-hour interval `[cutoff - 86400, cutoff)`, emitted daily. Its score becomes available at the cutoff. Day 1 is warm-up and cannot produce an available score.

## Causality and evidence

Construction reads only rows before the cutoff. New edges and degree growth are compared with relationships that existed before the current window. Every retained evidence reference is an `auth.txt:<source_line>` reference. To bound artifacts on the 508.9-million-event corpus, the default stores the first 100 references per edge together with the exact contributing count and a `references_truncated` flag. The raw events remain resolvable through `AuthenticationEvidenceLookup`.

The builder also writes separate fitting snapshots for Days 1–7. Those omit every event whose source or destination user is one of the seven training-compromise users from the Task 2.3 manifest. Normal replay snapshots retain all observed activity, so validation/test historical state is not silently altered.

Snapshot files contain typed user and computer dictionaries, edge endpoints, success/failure counts, weights, source lines, exact source counts, new-edge flags, and prior/current user degrees. The default 24-hour window is longer than the one-hour sequence window; the validation search is bounded to 1, 3, and 7 days.

## Model and limitations

The graph autoencoder uses nine identity-free node features: node type, log-degree, log-strength, failure activity, positive degree growth, peak authentication count in exact rolling 60-second and five-minute intervals, and peak unique destinations in an exact rolling five-minute interval. It has two normalized graph-convolution layers and a dot-product link decoder. Negative examples are sampled only from absent user-computer pairs. Because there are no learned node-ID embeddings, a first-seen user or computer remains scorable from its type and structural features.

This is observed-graph reconstruction, not future-link prediction. An edge anomaly is `1 - P(edge)` from the decoder. User scores use the mean of the three highest edge anomalies by default, limiting dilution by activity volume. Structural facts (`new_edge_count`, degrees, edge weights) are exported separately from learned anomaly scores. Neither is proof of compromise.

The best validation-selected counter-enhanced GAE reaches test precision `0.0450`, recall `0.2647`, F1 `0.0769`, average precision `0.02640`, and ROC-AUC `0.9106`. Its ranking signal is useful, but it is not an 80%-F1 classifier.

## Confirmed relationship history

`ThreatHistory` stores exact triples from confirmed incidents strictly before a configured freeze timestamp. A later authentication matches only when its acting user, source computer, and destination computer all match a stored triple. This preserves the source-computer signal that the GAE graph does not currently encode.

With Days 8–16 used only as confirmed history and the dictionary frozen before Day 17, the retrospective Days 17–30 result is precision `0.8286`, recall `0.8529`, and F1 `0.8406` (29 TP, 6 FP, 5 FN, 370,070 TN). This is a recurrence signature, not an anomaly score and not the GAE's F1. It will miss a completely new malicious relationship unless the GAE or another detector identifies it.

The hybrid extension was devised after the original test results had been inspected. Although no Day 17–30 label enters its frozen dictionary, the `0.8406` result is a retrospective backtest and must be confirmed on a fresh chronological holdout or another dataset.

## Reproducible run

```bash
.venv/bin/python scripts/build_graph_snapshots.py
.venv/bin/python scripts/train_graph_model.py
.venv/bin/python scripts/calibrate_graph_model.py --positive-units path/to/validation_graph_labels.json
.venv/bin/python scripts/export_graph_scores.py
.venv/bin/python scripts/evaluate_graph_detector.py --scores outputs/graph_scores_v1 --labels path/to/redteam_labels/labels --split test
```

For the full bounded Task 4.3 search, build snapshot directories using 1-, 3-, and 7-day policy configurations, then run `select_graph_model.py` with repeated arguments such as `--candidate 86400=path/to/one_day --candidate 259200=path/to/three_day --candidate 604800=path/to/seven_day --labels path/to/redteam_labels/labels`. It compares the configured latent sizes and aggregation rules by validation average precision, records every candidate, and freezes the winner without opening test labels.

Calibration fits its monotonic 0–1 rarity transform on validation scores only. If validation labels are supplied, the alert threshold maximises validation F1; otherwise the script records and uses the 99.9th validation-score percentile. Test data is never used for fitting, scaling, or threshold choice. The exported score is the latest daily score and downstream fusion may carry it forward for at most 24 hours.

Classification reporting should include precision, recall, F1, false-positive rate, average precision (PR-AUC), and ROC-AUC. Accuracy is intentionally not primary because red-team positives are sparse. MAE/RMSE and MASE/RMSSE are regression/forecasting metrics and do not apply to this reconstruction-based binary anomaly detector unless a separate continuous or forecasting target is introduced.

Evaluate the frozen confirmed-history layer with:

```bash
.venv/bin/python scripts/evaluate_graph_threat_history.py \
  --output outputs/graph_evaluation/threat_history_test_metrics.json
```

Export both GAE novelty and confirmed-relationship evidence in one user-day artifact with:

```bash
.venv/bin/python scripts/export_hybrid_graph_alerts.py
```

The combined artifact keeps separate fields for each channel. Confirmed recurrence is the primary high-confidence alert; GAE alerts remain lower-confidence novelty review items. A raw OR of both signals is exported for visibility but is not the source of the 84.06% result.

Run the chronological overfitting check with:

```bash
.venv/bin/python scripts/check_graph_overfitting.py \
  --development outputs/graph_evaluation/threat_history_temporal_validation_metrics.json \
  --holdout outputs/graph_evaluation/threat_history_test_metrics.json \
  --output outputs/graph_evaluation/threat_history_overfitting_check.json \
  --fail-on-overfitting
```

The current check passes because F1 does not collapse on the later period (`0.2222` on Days 13–16 versus `0.8406` on Days 17–30). A pass means only that this configured temporal-collapse test found no evidence of overfitting. See [the overfitting-check guide](graph_overfitting.md) and [the full model evaluation](graph_model_evaluation.md).
