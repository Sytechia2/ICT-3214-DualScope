# Long-Term Graph Detector screenshots

The figures in `docs/images/graph_results/` are generated from the frozen Days 17–30 hybrid evaluation. Regenerate them with:

```bash
.venv/bin/python scripts/plot_graph_evaluation.py
```

The set contains:

1. `01_metric_summary.png`: precision, recall, and F1 comparison;
2. `02_confusion_matrices.png`: confusion counts and class-normalized percentages;
3. `03_precision_recall_curve.png`: counter-enhanced GAE precision–recall curve;
4. `04_roc_curve.png`: counter-enhanced GAE ROC curve;
5. `05_training_loss.png`: recorded GAE training loss;
6. `06_scores_over_time.png`: graph anomaly scores over test days;
7. `07_threshold_tradeoff.png`: precision, recall, and F1 across GAE thresholds; and
8. `08_relationship_example.png`: one evidence-backed confirmed relationship.

The training run did not record validation loss per epoch, so figure 5 intentionally shows only the real training-loss history. The threshold and ranking curves belong to the continuous counter-enhanced GAE score. The `84.06%` F1 belongs to the supervised confirmed-relationship recurrence channel, not the GAE.
