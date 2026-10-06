# Graph detector downstream handoff

## Unsupervised graph score

`outputs/graph_scores_v1/dataset_day=NN/*.parquet` contains one row per active user and daily graph cutoff. Fields shared with fusion are `user_id`, `window_start`, `window_end`, `score_available_at`, `status`, `raw_score`, `score`, `alert_threshold`, `is_alert`, and `evidence_nodes`.

The 0–1 `score` is validation-relative rarity, not a probability of attack. Fusion must use a row only when `score_available_at <= decision_time`, must not carry it more than `maximum_score_age_seconds` (86,400 by default), and must preserve null scores. The first warm-up snapshot has `insufficient_history` rather than a zero.

Investigation fields distinguish direct observations (`new_edge_count`, `prior_degree`, `current_degree`, `degree_growth`, edge success/failure counts and source lines) from `edge_raw_score` and the user-level model reconstruction score. `top_edges.references_truncated` signals that only the bounded evidence preview is embedded; `source_reference_count` is the exact number of contributing raw events.

The validation-selected counter-enhanced GAE has held-out precision `0.0450`, recall `0.2647`, and F1 `0.0769`. Consumers must not substitute the hybrid signature result for this continuous graph anomaly score.

## Confirmed-relationship signal

The optional `ThreatHistory` layer consumes analyst-confirmed relationships available before a freeze timestamp and matches exact `(acting_user, source_computer, destination_computer)` triples. Treat a match as a categorical high-confidence recurrence signal:

- do not interpret it as a calibrated probability;
- retain the matching authentication event as direct evidence;
- keep its provenance separate from the GAE reconstruction explanation; and
- do not use labels at or after the decision time to update the dictionary.

The retrospective Days 17–30 backtest produced precision `0.8286`, recall `0.8529`, and F1 `0.8406`. This number belongs to the signature layer, not the GAE.

`scripts/export_hybrid_graph_alerts.py` now packages both channels in `outputs/graph_evaluation/hybrid_alerts/dataset_day=NN/*.parquet`. Important fields are:

- `gae_score` and `gae_is_alert`: the selected counter-enhanced unsupervised novelty evidence;
- `base_gae_score` and `base_gae_is_alert`: the original GAE result before the five-minute counter enhancement;
- `peak_unique_destinations_300s`: the structural counter used by the enhancement;
- `confirmed_relationship_alert`: exact recurrence of an analyst-confirmed relationship;
- `relationship_match_count` and `relationship_matches`: direct event evidence;
- `primary_high_confidence_alert`: the confirmed-relationship decision;
- `any_graph_signal`: union of both channels for analyst visibility; and
- `priority_reason`: `confirmed_relationship`, `graph_anomaly_review`, or `none`.

Do not use `any_graph_signal` when quoting the 84.06% result. On the retrospective test it reaches recall `0.8824` but only precision `0.1327` and F1 `0.2308`. The 84.06% result belongs specifically to `primary_high_confidence_alert` / `confirmed_relationship_alert`.

## Validation and monitoring

Before handoff, run `scripts/check_graph_overfitting.py` with non-overlapping chronological reports. With `--fail-on-overfitting`, it exits with status 1 when later F1 drops beyond both configured limits or the holdout has too few positives. The current run passes, but the result remains retrospective and needs a fresh holdout. Full commands and interpretation are in [graph_overfitting.md](graph_overfitting.md).
