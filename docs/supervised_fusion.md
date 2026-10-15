# Final detection model: supervised fusion

DualScope's final detection model is a gradient boosting classifier that combines the short-term GRU detector's score with simple hourly authentication counts. It was chosen on validation days, frozen, and tested once on Days 17–30. This page summarises the model and its results. The full experiment record, including what was tried and failed, is [experiment_features_v2.md](experiment_features_v2.md).

## The model

- **Unit:** one acting user in one hour (user-hour), the same unit as every other detector.
- **Inputs** (no user or computer names):
  - the GRU's raw `max_event` score for the hour;
  - counts of the hour's events: total, failures, distinct source and destination computers, first-time user→source, first-time host→host and first-time user→destination connections, NTLM, Network logon type, LogOn;
  - whether the account is a machine account (`$`).
- **Classifier:** scikit-learn `HistGradientBoostingClassifier(class_weight="balanced", random_state=0)`. Settings were fixed before any result was seen and never tuned.
- **Labels:** validation red-team labels (Days 8–16). This is supervised fusion, as in the proposal's "validation-tuned weighted fusion"; the report must say that validation labels trained it.
- **Alerts:** the top 38 user-hours per day (the GRU's alert rate at its validation threshold, used as the common workload for every model).

## Validation (fitted on Days 8–12, scored on Days 13–16)

Days 13–16: 1,776,074 user-hours, 136 attack user-hours.

| Scorer | Caught at 38/day | AP |
| --- | ---: | ---: |
| GRU alone | 4 | 0.0046 |
| Counting rule (first-time logins) | 4 | 0.0067 |
| GRU, human accounts only (post hoc) | 8 | 0.0100 |
| Logistic regression (same inputs) | 4 | 0.019 |
| **Gradient boosting** | **16 (13–18)** | **0.050 (0.035–0.051)** |

The range comes from a numeric stability check: refitting with 0.03% random noise on the GRU score (the size of the difference between GPU and CPU scoring) gave 13 to 18 caught, median 16, over ten runs. Gradient boosting met the pre-set success rule (at least 2× the GRU's catches and a higher AP) in every run.

About half of the gain comes from never alerting on machine accounts (no attack in the data uses one); the GRU restricted to human accounts catches 8.

## Final test (Days 17–30, run once on 2026-10-06)

The model was refitted on all of Days 8–16 with the same settings, frozen, and scored on the test days. Test labels were read only after all scores were saved. Days 17–30: 5,537,311 user-hours, **39 attack user-hours**, 28 of them on Days 27–28.

| Scorer | Caught at 38/day (of 39) | AP | ROC-AUC |
| --- | ---: | ---: | ---: |
| **Final model** | **1** | **0.00124** | **0.969** |
| GRU alone | 0 | 0.00020 | 0.846 |
| GRU, human accounts only | 0 | 0.00043 | 0.944 |
| Counting rule | 0 | 0.00006 | 0.598 |
| Isolation Forest baseline | 0 | 0.00002 | 0.675 |

- **Primary rule** (at least 2× the GRU's catches, at least 2 when the GRU catches 0, and a higher AP): **not met**.
- **Secondary rule** (more catches and higher AP than the GRU on human accounts only): **met**.
- The model ranks attacks much better than the GRU: the median attack hour is at daily rank 1,958 versus 19,917 for the GRU, and 14 of 39 attack hours are in its daily top 1,000 versus 3. But only one reaches the top 38.

**Conclusion:** the supervised fusion is a better ranker than the GRU on unseen test days (AP about 6×), but at 38 alerts per day it catches almost none of the test-period attacks. The validation gain did not carry over.

## Limitations

- **One red team, one dominant day:** 82 of the 84 training attacks (Days 8–12) are on Day 9. The model learns that campaign's profile (NTLM network logons to new hosts) and may miss attacks that look different.
- **Small, clustered test set:** 39 attack hours, mostly on two days. One catch more or less changes the verdict.
- **Numerically fragile at the budget:** tiny score differences move the validation result between 13 and 18.
- **Incomplete labels:** unlabelled events from the red-team source computer exist, so some "false alarms" may be real attacker activity.
- **Graph detector not used:** the final model uses only the 1-hour timescale. A logistic regression fusion that adds the graph score caught 10 of 136 on Days 13–16 ([model_comparison_matrix.md](model_comparison_matrix.md)) and was not taken forward.

## Reproduction

Scripts, in order (each writes its output path; the run of record used the `outputs/final_test/` and `models/*/final_test` folders, see [supervised_fusion_handover.md](supervised_fusion_handover.md)):

1. `scripts/build_lanl_features.py` with `config/lanl_features_v2.json`: feature build, Days 1–30. On a 32 GB machine use `--workers 2 --assembly-workers 1 --low-memory-read` (about 3 h).
2. `scripts/train_sequence_model.py`: GRU run A (L = 32, small, CPU).
3. `scripts/evaluate_sequence_runs.py --units-output …`: GRU scores for every validation user-hour.
4. `scripts/supervised_fusion_validate.py`: hourly counts, fit on Days 8–12, compare on Days 13–16.
5. `scripts/supervised_fusion_stability.py`: numeric stability check (optional).
6. `scripts/supervised_fusion_freeze.py`: refit on Days 8–16 and freeze.
7. `scripts/supervised_fusion_final_test.py`: the one-time test. **Already run; do not run again.**

`scripts/evaluate_counting_rules.py` produces the counting-rule baseline, and `scripts/export_final_model_scores.py` exports the Days 13–16 scores used by the 9.3 and 9.4 analyses.

Large artifacts (feature build, models, scores, logs) are not in Git. The run of record is on Member 1's machine: results in `outputs/final_test/final_test.json`, frozen model `models/fusion/final_test/frozen/` (sha256 `02b4e6f7e3d1…`).
