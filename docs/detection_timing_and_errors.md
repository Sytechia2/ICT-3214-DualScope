# Detection timing, detector precedence and errors (Task 9.4)

All figures below are produced by `scripts/analyse_timing_and_errors.py` from the score files, the red-team labels and the authentication events. Case studies are selected by fixed rules and their facts are read from the data; none are written by hand. Results are in `outputs/evaluation/timing_and_errors.json`.

## 1. Setup

- **Models**: the final gradient boosting fusion (GRU score + hourly counts) and, for comparison, the GRU alone and the logistic regression fusion (seq + graph + temporal features). Both supervised models are fitted on Days 08–12 only (see [model_comparison_matrix.md](model_comparison_matrix.md)).
- **Analysed days**: 13–16 (unseen by the models). Days 17–30 are not read.
- **Alerts**: the top 38 user-hours per day for each model.
- **Campaign**: one (user, day) pair with at least one labelled red-team event. Days 13–16 have **88 campaigns** covering 136 attack user-hours.
- **Timing**: a score for hour `[t, t + 3600)` is available at `t + 3600`. The offset is the time from a campaign's first labelled event to the availability of the first alert for that user and day:
  - *early*: offset < 0 (the alert is on an hour before the first labelled event, so it is not itself an attack hour);
  - *immediate*: 0 ≤ offset ≤ 1 h;
  - *delayed*: offset > 1 h;
  - *missed*: no alert for that user and day within the budget.
- **Detector cut-offs** for precedence (GRU ≥ 0.999415, graph ≥ 0.999509) are fitted by best F1 on Days 08–12.

## 2. Timing (Days 13–16, 38 alerts/day)

| Model | Campaigns with any alert | …with an attack hour alerted | Early | Immediate | Delayed | Median offset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GRU alone | 5 / 88 | 4 | 1 | 2 | 2 | 0.31 h |
| Logistic regression fusion | 10 / 88 | 6 | 2 | 4 | 4 | 0.84 h |
| **Final: gradient boosting** | **16 / 88** | **14** | 1 | **11** | 4 | 0.68 h |

- The final model raises an alert on an actual attack hour in **14 of 88 campaigns** (16%), against 4 for the GRU and 6 for the logistic regression fusion.
- For 11 of its 16 alerted campaigns, the first alert is available within an hour of the first labelled event.
- "Early" alerts are counted separately: they fall on hours with no labelled event. The final model has one (U3486@DOM1, Day 13, 6.45 h before the first label). It may be unlabelled attacker activity or a coincidental false alarm; the data cannot tell which.
- 72 of 88 campaigns get no alert from the final model at this budget.

## 3. Detector precedence (all 88 campaigns)

Using the Days 08–12 cut-offs, the GRU crosses its cut-off on some hour of the campaign day in 14 campaigns, the graph detector (previous day's score, available at the start of the day) in 5, and neither in 73:

| Precedence | Campaigns |
| :--- | :---: |
| GRU only | 10 |
| Graph first (graph alert at day start, GRU alert later that day) | 4 |
| Graph only | 1 |
| Neither detector alerts | 73 |

The two detectors rarely alert on the same campaign (4 of 88), and when they do the graph score is available first because it is computed from the previous day. Most campaigns trigger neither detector's own alert threshold, which is why ranking by a fused score, rather than requiring a detector alert, matters.

## 4. Errors of the final model

**False alarms** (136 of 152 alerts):
- All 136 are human accounts; none are machine accounts (`$`). The model learned from the training labels that no attack in the data uses a machine account.
- 4 are non-attack hours of a user who was attacked the same day.

**Missed attack hours** (120 of 136) are mostly busy hours, not isolated logons:

| Attack hours | Hours | Median events in the hour | Hours with 1–2 events | Hours with > 100 events |
| :--- | :---: | :---: | :---: | :---: |
| Caught | 16 | 54 | 1 | 7 |
| Missed | 120 | 77.5 | 9 | 53 |

A typical labelled attack is one or a few NTLM network logons inside an hour with dozens to thousands of the account's normal events, so the hour as a whole looks ordinary.

## 5. Case studies

Selected by rule from the final model's ranking on Days 13–16. All event references can be looked up in `auth.txt` and `redteam.txt`.

### 5.1 Caught attack: highest-ranked attack hour inside the budget
- **Unit**: `U3635@DOM1`, Day 14, hour `[1170001, 1173601)`.
- **Scores**: final model 0.9943, rank 3 of 454,365 that day. GRU 0.9970, rank 1,193 (outside the GRU's own top 38). Graph (Day 13 snapshot) 0.9859, below its cut-off.
- **Labelled attack**: `redteam.txt:593`, C17693 → C22766 at 1172817, matching `auth.txt:230868761` (NTLM, Network, LogOn).
- **Hour profile**: 402 events, 3 failures, 34 sources, 42 destinations; 15 first-time user→source logins and 17 first-time host→host connections. Types: 248 unknown (`?`), 137 Kerberos, 14 NTLM, 3 Negotiate.
- **Detector evidence**: the GRU's highest-error events in this hour are `auth.txt:230473160` (feature `is_new_user_destination`), `auth.txt:230479342` (`logon_type_id`) and `auth.txt:231064707` (`is_new_user_destination`). None of them is the labelled attack event. The graph snapshot has 29 edges, 5 of them new; its most anomalous edge is to C8209 (new edge; `auth.txt:210302264`, …).
- **Reading**: the GRU alone would not have alerted. The final model ranked the hour third, consistent with its many first-time logins, which are among its inputs. The single labelled event is one of 402 in the hour.

### 5.2 False alarm: highest-ranked hour without an attack label
- **Unit**: `U6855@DOM1`, Day 13, hour `[1090801, 1094401)`.
- **Scores**: final model 0.9943, rank 1 of 433,399 that day (tied with 5.1's score; ties are ordered by the fixed seed-0 order). GRU 0.9886, rank 5,915. Graph (Day 12 snapshot) 0.6281.
- **Labelled attack events**: none.
- **Hour profile**: 163 events, 0 failures, 11 sources, 16 destinations; 3 first-time user→source logins and 7 first-time host→host connections. Types: 98 unknown, 56 Kerberos, 9 NTLM.
- **Detector evidence**: the GRU's highest-error events are `auth.txt:211883620`, `auth.txt:211909766` and `auth.txt:211873377`, all with top feature `is_new_host_connection`. The graph snapshot has 9 edges, none new.
- **Reading**: the hour has the same kind of first-time connection activity as the caught attack in 5.1, without a label. Whether it is benign or unlabelled attacker activity cannot be determined from the labels, which are incomplete (see section 6).

### 5.3 Missed attack: median-ranked miss
- **Unit**: `U8170@DOM1`, Day 14, hour `[1180801, 1184401)`.
- **Scores**: final model 0.0239, rank 1,386 of 454,365 (budget is 38). GRU 0.2775, rank 331,652. Graph (Day 13 snapshot) 0.5502.
- **Labelled attack**: `redteam.txt:625`, C17693 → C313 at 1184399, matching `auth.txt:234340102` (NTLM, Network, LogOn).
- **Hour profile**: 1 event (the attack itself); 1 first-time user→source login.
- **Detector evidence**: the GRU's top event is the attack event, `auth.txt:234340102`, with feature `is_new_user_destination`, but its reconstruction error is low (0.358). The graph snapshot has 2 edges, none new.
- **Reading**: one successful NTLM network logon carries little signal for an hourly model. This miss is a single-event hour, but such hours are the minority of misses (9 of 120; see section 4).

## 6. Limitations

- Labels are incomplete: the red-team source computer C17693 has 1,714 authentication events in Days 1–30 but only 670 are labelled. Unlabelled hours are treated as benign, so some "false alarms" (and the one early alert) may be attacker activity.
- 88 campaigns and 136 attack hours from one red team: differences of a few campaigns are within chance.
- The final model's top scores are tied (5.1 and 5.2 share 0.9943); the fixed tie order decides their ranks. Its recorded result does not depend on the tie order.

## 7. Reproduction

```powershell
.\.venv\Scripts\python scripts/analyse_timing_and_errors.py `
  --final-model-scores outputs/experiment_v2/final_model_scores_days13_16.parquet
```

It needs the GRU and graph score packages (default paths as in [model_comparison_matrix.md](model_comparison_matrix.md)), the red-team labels, `outputs/experiment_v2/fusion_counts.parquet`, and the Days 1–16 feature-build events (`data/processed/lanl_features_v2_days_01_16/raw/events`) for the hour profiles and `auth.txt` references.
