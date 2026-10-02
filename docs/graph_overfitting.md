# Graph detector overfitting check

## Purpose

The check looks for temporal performance collapse. It uses chronological periods because a random row split would mix the same users, computers, and attack campaign across both sides and would give an unrealistically easy estimate.

This check applies to the confirmed-relationship history extension. It does not convert that signature layer into an unsupervised model, and it does not prove that the model is free from every form of overfitting.

## Evaluation design

The rolling-origin comparison is:

| Run | Confirmed history | Evaluation period | F1 |
|---|---|---|---:|
| Development | Days 8–12 | Days 13–16 | 0.2222 |
| Later holdout | Days 8–16 | Days 17–30 | 0.8406 |

For each run, the relationship dictionary is frozen before the first evaluation event. Labels from the evaluation period cannot enter that run's dictionary. The two evaluation periods do not overlap.

The development period produces 11 TP, 0 FP, and 77 FN. The later holdout produces 29 TP, 6 FP, and 5 FN. Performance improves rather than collapsing, so the current check reports `PASS`.

## Failure policy

The command reports overfitting evidence when either condition holds:

1. later F1 drops by more than `0.10` absolutely **and** more than `50%` relative to development F1; or
2. the later holdout contains fewer than 30 positive evaluation units.

Both F1 limits must be crossed so that a small absolute change from a small baseline does not fail only because its relative percentage looks large. The minimum-positive rule prevents a tiny holdout from producing a misleading pass.

The defaults can be changed with `--max-absolute-f1-drop`, `--max-relative-f1-drop`, and `--minimum-holdout-positives`. Any changed limits must be decided before reading the resulting holdout metric.

## Reproduce the check

Generate the earlier temporal evaluation:

```bash
.venv/bin/python scripts/evaluate_graph_threat_history.py \
  --history-start-day 8 \
  --history-end-day 12 \
  --evaluation-start-day 13 \
  --evaluation-end-day 16 \
  --output outputs/graph_evaluation/threat_history_temporal_validation_metrics.json
```

Generate the later holdout evaluation:

```bash
.venv/bin/python scripts/evaluate_graph_threat_history.py \
  --history-start-day 8 \
  --history-end-day 16 \
  --evaluation-start-day 17 \
  --evaluation-end-day 30 \
  --output outputs/graph_evaluation/threat_history_test_metrics.json
```

Compare them and return a non-zero exit status on failure:

```bash
.venv/bin/python scripts/check_graph_overfitting.py \
  --development outputs/graph_evaluation/threat_history_temporal_validation_metrics.json \
  --holdout outputs/graph_evaluation/threat_history_test_metrics.json \
  --output outputs/graph_evaluation/threat_history_overfitting_check.json \
  --fail-on-overfitting
```

The JSON output records the F1 values, absolute and relative drop, positive count, configured limits, status, and reasons. Without `--fail-on-overfitting`, the report is still written but a failed check does not stop an automated pipeline.

## Interpretation and limitations

A `FAIL` is evidence consistent with overfitting or temporal distribution shift. It does not by itself identify which one caused the decline. Investigate feature drift, changed users/hosts, label prevalence, alert thresholds, and campaign behavior.

A `PASS` only means that this later-period F1 did not materially collapse under the configured policy. The current 84.06% result is still a retrospective estimate because the confirmed-history extension was designed after the original test outcome had been reviewed. The strongest next check is to freeze the code and policy, then evaluate once on Days 31–58 or another untouched authentication dataset.

The unsupervised GAE should be monitored separately using validation-selected thresholds, average precision, ROC-AUC, alert-budget precision/recall, and cohort-specific drift. Its current counter-enhanced held-out F1 is 0.0769; the hybrid signature's 0.8406 must never be reported as the GAE result.

## Automated tests

`tests/test_graph_evaluation.py` verifies that:

- a large later-period F1 collapse fails;
- stable later-period performance passes;
- future labels are excluded from `ThreatHistory`; and
- exact relationship matching and user-day classification counts are correct.

Run the complete suite with:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```
