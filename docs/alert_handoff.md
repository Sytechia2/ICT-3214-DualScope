# Final-model alert handoff (days 17–30)

Input for GenAI investigation (Tasks 6.x) and the analyst dashboard (Tasks 7.x). These are the 532 alerts the final model raised in the one-time final test: the top 38 user-hours per day on days 17–30 (see [supervised_fusion.md](supervised_fusion.md)). Nothing is re-scored, so building this package does not rerun the test.

## Build it

```powershell
python scripts/export_alert_handoff.py
```

It needs the local test scores (`outputs/final_test/`), the days 1–30 feature build (`data/processed/lanl_features_v2_days_01_30`) and the graph score package (`outputs/graph_scores_v1/scores`). It takes about 30 seconds. It refuses to write unless it reproduces the test record's 532 alerts and 1 red-team hour, and it checks that every alert's event count matches the scored `n_events`.

The package (about 6.5 MB) is written to `outputs/handoff/final_test_alerts_v1/`. It is not in Git: share the folder directly.

## What's in it

| File | Rows | Contents |
| --- | --- | --- |
| `alerts.parquet` | 532 | One row per alerted user-hour: `alert_id`, `incident_id`, user, day, window, `rank_in_day`, `fusion_score`, GRU score, hourly counts, `ground_truth_redteam` |
| `events.parquet` | 106,730 | Every authentication event in those user-hours, with its `source_reference` (`auth.txt:<line>`), hosts, auth type, logon type, result and novelty flags |
| `incidents.jsonl` | 497 | Alerts grouped per user (hours up to 2 h apart join one incident), with all evidence references and the graph detector's view of the same day |
| `manifest.json` | – | Input file hashes, counts and field notes |

`incidents.jsonl` uses the Task 5.4 field layout, so the Task 7.1 dashboard (`streamlit run scripts/incident_dashboard.py` on Matt's branch, **Local JSONL export**) opens it unchanged.

## Summary

- 532 alerts, 413 users, 497 incidents (477 are a single hour; the longest has 7 alert hours over 11 hours).
- 1 alert is red-team activity: `U737@DOM1`, day 28, 14:00–15:00 (`INC-TEST-D28-U737_DOM1-003`, rank 7 that day).
- The graph detector also alerted on the same user-day in 36 incidents.

## Reading the fields

- **Times** are dataset-relative seconds; windows are half-open `[start, end)`. Day *d* starts at second `1 + (d − 1) × 86,400`.
- **`fusion_score`** ranks user-hours. It is not a calibrated attack probability.
- **Ties.** The model gives many hours exactly the same score. On most days, hours tied at the lowest queued score fill the last places, and the test's fixed seed-0 order decided which of them made the queue. The 162 alerts with `tied_at_cutoff = true` are those; their `rank_in_day` among each other carries no meaning.
- **`priority`**: `HIGH` = at least one hour scored above its day's cut-off, so it is queued whatever the tie-break (340 incidents). `MEDIUM` = every hour was tied at the cut-off (157 incidents). No `CRITICAL` or `LOW` is used.
- **`detector_scores.sequence.max_score`** is the GRU score as a within-day percentile (0–1), so it fits the dashboard's 0–1 scale; the raw value is in `alert_hours[].gru_max_event`. The GRU has no alert flag here (`any_alert` is null).
- **`detector_scores.graph`** and **`graph_context`** come from the graph detector's daily score for the same user (`top_edges` lists the hosts it weighted most, with their evidence). The final model does not use them, and the graph team inspected days 17–30 during development, so treat them as context, not as an independent test result.
- **`ground_truth_redteam`** is the answer key, for evaluation (9.5) and demos only. Hide it from the analyst view by default, and never include it in an LLM evidence package.

## Notes for 6.2 evidence packages

The median incident has 82 events, but the largest has 13,800, so evidence packages need a selection rule (for example: failures, NTLM, new hosts and new sources first, then a sample). Resolve references through `events.parquet`; the full LANL data is not needed.
