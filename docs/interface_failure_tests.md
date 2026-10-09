# Interface and failure-handling tests (Task 8.2)

This page records what happens at each hand-off between components when the input is wrong, missing or late: what we expected, what we observed, and which test checks it. The tests are in `tests/test_failure_handling.py` (54 tests, about 14 seconds plus the 6 second torch import that the other pipeline tests also pay). They use synthetic data, the tracked pipeline subset and a fake model, so they make no Gemini or gcloud calls and never touch days 17-30. Cases that earlier tasks already test are listed with their existing test instead of being repeated.

Run them with

    python -m pytest tests/test_failure_handling.py

One test (`test_sample_run_references_resolve_and_map_back_to_the_subset_records`) reads the finished sample run in `outputs/pipeline/sample` and is skipped when that folder does not exist. It never writes there.

## Rules the tests check

* A bad record, file or model reply stops or marks only the thing it belongs to. Nothing is dropped without a count, a reason or an error that names the file.
* The detector results (`alerts.parquet`, `events.parquet`, `incidents.jsonl`, `manifest.json`) stay byte-identical when the investigation fails, and the dashboard keeps loading them.
* A refused or failed step leaves no folder that looks finished. Ingestion writes `summary.json` last and the alert package writes its folder only after every check has passed.

## Results

Status: **pass** means the behaviour was already right. **fixed** means the test found a bug that is now fixed. **open** means the behaviour is a known limitation, listed under Issues found.

| # | Case | What was done | Expected outcome | Observed outcome | Status |
|---|---|---|---|---|---|
| 1a | Malformed raw records: eight fields, text, zero, negative, fractional or empty timestamp, empty field | `test_malformed_raw_records_are_rejected_with_line_numbers_and_counts_reconcile` (also `test_parallel_ingestion_rejects_malformed_rows_with_line_numbers_too` for the per-day workers). Existing: `test_auth_ingestion_reconciles_and_preserves_rows`, `test_line_map_rewrites_references` | Each bad line goes to `rejections.jsonl` with its line number, reason and raw text. Accepted + rejected + out of scope equals rows scanned. | As expected. A malformed line that also has an earlier timestamp is rejected as malformed and does not trigger the order check. | pass |
| 1b | Timestamp goes backwards | `test_timestamp_going_backwards_stops_the_run_and_leaves_no_summary`. Existing: `test_order_inversion_is_rejected_across_chunks` | The run stops with the file, line and both timestamps. No `summary.json` is written. A rerun with `overwrite` works. | Message `timestamp order inversion at auth.txt:3: 7 < 9`. Parts written before the stop stay on disk without `summary.json`, and the runner clears the folder before it retries. | pass |
| 1c | Line map does not fit the source | `test_bad_line_maps_are_refused_before_any_output` (non-integer, repeated, zero). Existing: `test_line_map_length_mismatch_fails` | A clear error before any event is written. | As expected. | pass |
| 2a | `incidents.jsonl` with a missing field or a bad JSON line | `test_bad_incident_lines_name_the_line_and_field` (dashboard loader). `test_investigation_inputs_with_bad_files_fail_with_the_file_named` (investigation script). Existing: `test_loader_empty_and_malformed_files` | The error names the file and line. | The dashboard loader already did. The investigation script showed a bare `JSONDecodeError` or `KeyError`; it now names the file and line. | fixed (Issue 1) |
| 2b | `events.parquet` with a missing column | Same test as 2a for the investigation script. Existing: `test_load_events_rejects_bad_files` (dashboard) | The error names `events.parquet` and the column. | The script failed later with a `KeyError` inside the package builder. It now checks the columns up front. | fixed (Issue 1) |
| 2c | Release manifest hash does not match | Existing: `test_changed_file_raises_naming_the_file`, `test_missing_file_raises` in `tests/test_pipeline_release.py`; the runner's `stage_test_package` also checks the incident file hash | `ReleaseError` naming the file. | As expected. | pass (existing) |
| 2d | Feature build with a different preprocessing file | `test_a_build_with_a_different_preprocessing_file_is_refused_by_name` | `ScoringError` naming the file, and no scoring starts. | As expected. | pass |
| 2e | Alert package refusals leave nothing behind | `test_alert_package_refusals_leave_no_half_written_folder`, `test_events_that_do_not_match_the_scored_count_refuse_the_package` | Bad arguments, an unreadable graph file or an event count mismatch raise before the output folder is created. | As expected. | pass |
| 3a | User-hours with events but no GRU score | `test_user_hours_with_events_but_no_gru_score_are_counted_not_hidden` | They cannot be queued, and the number per day is reported. | The rows were dropped without any count. The scoring log now reports them, `score_days` keeps them in `frame.attrs["unscored_user_hours"]` and `scores_summary.json` records `user_hours_with_events_but_no_gru_score`. The sample run has none, because days 8-9 all have 24 hours of history. | fixed (Issue 2) |
| 3b | GRU score without events | `test_a_gru_score_without_events_stops_scoring` | `ScoringError`. | As expected. | pass |
| 3c | Score frame with a missing column or a missing fusion score | `test_score_frames_with_missing_columns_or_scores_are_refused_by_name` | A `ValueError` naming the column or counting the rows, before anything is written. | A missing column gave a bare `KeyError`. A missing fusion score (NaN) was never picked for the queue and nothing said so. Both are now refused. | fixed (Issue 3) |
| 3d | Alert whose events do not match `n_events` | `test_events_that_do_not_match_the_scored_count_refuse_the_package`. Existing: `test_build_alert_package_guards` | The package is refused and no folder is written. | As expected. | pass |
| 4 | No graph context, or graph rows for another day | `test_graph_context_is_empty_when_there_is_no_graph_or_it_is_for_another_day`, `test_packages_without_graph_rows_still_build_and_load_in_the_dashboard` | The incident is still built with empty graph fields (`null` scores, no edges). The evidence package gets a graph section without edges. The dashboard check loads it. | As expected. A day-1 snapshot is not carried over to a day-2 incident. | pass |
| 5 | Empty retrieval | `test_evidence_with_no_matching_rule_gives_no_candidates_and_no_invented_technique`, `test_dashboard_shows_incidents_with_no_matching_rule`. Existing: `test_rag_without_candidates_asks_for_no_mapping`, `test_no_mapping_outcome`, `test_mode_switch_and_no_supported_mapping` | No rule match gives `insufficient_evidence`, no candidates and a prompt that asks for no mapping. A reply that names techniques anyway has every one rejected. The dashboard shows "No rule matched". | As expected. | pass |
| 6a | Model timeout, HTTP 429, HTTP 500, reply that is not JSON | `test_model_failures_are_isolated_per_incident_and_retry_failed_recovers_them`, `test_client_turns_every_transport_failure_into_a_generation_error`, `test_investigate_records_a_timeout_and_an_unreadable_reply_as_failed`. Existing: `test_client_retries_rate_limits_then_gives_up`, `test_investigate_records_ok_invalid_and_failed` | Each incident gets its own `failed` or `invalid` record with the error. The other incidents finish. The summary and the dashboard states count them. | Isolation was right for timeouts, 429 and 500. A reply with HTTP 200 and a body that is not JSON (for example a proxy page) escaped the client as a `ValueError`, stopped the whole generation run and skipped the summary. It is now an ordinary `failed` record, retried like other transient errors. | fixed (Issue 4) |
| 6b | Resume and retry | Same test as 6a | A plain rerun makes no model calls. `--retry-failed` asks again only for the `failed` and `invalid` replies, in both modes. | As expected: 0 calls, then 6 calls for 3 incidents in 2 modes. | pass |
| 6c | gcloud missing, end to end through the runner | `test_missing_gcloud_fails_with_a_clear_message`, `test_runner_with_gcloud_missing_marks_investigate_failed_and_keeps_the_detector_results` (runs the real script in a child process without gcloud) | `investigate` is marked failed with the reason, the four package files stay byte-identical, the dashboard check still passes with every incident "not generated", the exit code is 1 and the report gives `--only investigate --llm`. | As expected. The reason ends with `GenerationError: gcloud is not installed or not on PATH`. | pass |
| 7a | Day and hour formulas across components | `test_day_and_hour_formulas_agree_across_components` at t = 1, 3600, 3601, 7200, 7201, 86400, 86401, 172800, 172801; `test_dataset_day_refuses_zero_and_negative_seconds` | Ingestion, the sequence builder, `splits.dataset_hour_start`, the hourly counts and both time displays agree. | As expected. | pass |
| 7b | Hourly counts, alert windows and attached events | `test_hourly_counts_alert_windows_and_attached_events_use_the_same_hours`, `test_red_team_labels_use_the_same_hour_as_the_scores` | Windows are half-open `[start, end)`. Second 3600 belongs to the hour starting at 1 and second 3601 to the one starting at 3601. The last second of a day and the first of the next fall in different hours and days. Every event lands in exactly one alert. | As expected. | pass |
| 7c | Incident grouping gap | `test_incident_grouping_gap_is_inclusive_at_7200_seconds`, `test_hour_aligned_alerts_three_hours_apart_form_one_incident_and_four_hours_apart_two`. Existing: `test_group_incidents_merges_hours_within_gap` | A gap of 0 or 7200 seconds merges, 7201 and more splits. | As expected. Because windows are whole hours the gap is always a multiple of 3600, so the 7201 case only arises for hand-made windows. | pass |
| 8a | Evidence link integrity | `test_synthetic_package_has_a_one_to_one_link_from_every_reference_to_an_event`, `test_sample_run_references_resolve_and_map_back_to_the_subset_records` (sample run: every incident reference resolves to one event, each event is inside its alert window and belongs to its user, and its `auth.txt:<line>` maps through `line_map.txt.gz` to the same record in `auth_subset.txt.gz`). The dashboard check in the runner also fails on any unresolved reference (existing: `test_dashboard_check_fails_when_an_evidence_reference_is_missing`) | Reference to event is one to one. No event is cited by two incidents or by none. | As expected. | pass |
| 8b | Subset ingestion keeps the original line numbers | `test_subset_ingestion_keeps_each_event_tied_to_its_subset_line`. Existing: `test_subset_feeds_real_ingestion_with_original_references` | Raw record, line number and `source_reference` agree for each event. | As expected. | pass |
| 8c | Answer key never reaches the model | `test_the_answer_key_is_refused_wherever_it_could_enter_a_prompt`, and in the sample test the packages, retrieval, both prompts, the system prompt and the saved dry-run files. Existing: `test_package_never_contains_the_answer_key` | No `ground_truth` or `redteam` text in anything sent to the model. | As expected. | pass |
| 9a | A stage fails in the runner | `test_a_failed_stage_blocks_its_dependents_keeps_earlier_outputs_and_rerun_resumes` for each of the six sample stages. Existing: `test_failed_stage_blocks_its_dependents_and_names_the_rerun`, `test_failed_investigation_keeps_alerts_and_the_dashboard_check_runs` | Dependents are skipped as "blocked", earlier outputs are untouched, the report names `--from <stage>` (for `investigate`: `--only investigate --llm`), and a rerun after the fix runs only the failed stage and what needs it. A failed `investigate` does not block `dashboard`. | As expected. | pass |
| 9b | A script is killed | `test_a_killed_script_fails_its_stage_with_the_exit_code_and_last_output`. Existing: `test_peak_memory_and_run_command` (exit code and timeout) | `StageFailed` with the exit code and the last output line. | As expected. | pass |
| 9c | Damaged `pipeline_state.json` | `test_a_damaged_state_file_means_every_stage_runs_again` | The file is ignored and every stage runs again. | Truncated JSON and an empty file were already handled. A file that held `[]` or had no `stages` crashed the runner with an `AttributeError` or `TypeError`. Both are now ignored too. The runner writes the file atomically, so this is unlikely in practice. | fixed (Issue 5) |
| 10a | Test days are refused | `test_a_profile_with_test_days_stops_the_command_before_any_stage` (command exits with code 2, no run folder). Existing: `test_test_days_refused_before_loading`, `test_validation_day_passes_the_guard`, `test_test_days_are_refused_before_any_stage_runs` | Days 17-30 are refused before any file is opened or stage starts. | As expected. | pass |
| 10b | Test split is refused | `test_test_labels_and_test_alert_stage_are_refused_before_reading_anything`. Existing: `test_build_alert_package_guards` | Test labels and a `test` alert split raise before any read. | As expected. | pass |

## Stale graph snapshots are out of scope for the final pipeline

The final model is the gradient-boosting fusion of the GRU score and the hourly counts. It does not read graph scores, and the incident records say so (`graph_context.used_by_final_model` is false). The sample and validation packages are built without graph scores, and the full-profile test package carries the graph view of the stored export as context for the analyst only. The 24-hour carry-forward limit for stale graph scores belongs to the earlier average and temporal fusion ([evaluation_protocol.md](evaluation_protocol.md)), which the final pipeline does not run. What the pipeline does guarantee is tested in case 4: graph rows are matched on user and dataset day, a snapshot from another day is never carried over, and a missing or old snapshot leaves the graph fields empty without losing the incident.

## Issues found

Owner is Peter (Member 1) for all of them, including the investigation and dashboard items he took over.

1. **Investigation script gave unclear errors for damaged input files.** Status: fixed.
   * Reproduce: add a line `{broken` to `incidents.jsonl` of an alert package, or drop a column such as `authentication_result` from `events.parquet`, then run `python scripts/run_investigations.py --handoff-dir <package> --dry-run`.
   * Before: a bare `JSONDecodeError` or a `KeyError` from inside the package builder, without the file name.
   * Fix: `read_jsonl` and `load_inputs` in `scripts/run_investigations.py` stop with the file, line and field or column.
   * Test: `test_investigation_inputs_with_bad_files_fail_with_the_file_named`.
2. **User-hours with events but no GRU score were dropped without a count.** Status: fixed.
   * Reproduce: call `score_days` where the count frame has user-hours that the GRU frame does not (the first 24 hours of history, or users without enough history).
   * Before: `score_days` returned only the scored rows and reported nothing about the rest.
   * Fix: `score_days` in `src/dualscope/pipeline/scoring.py` logs the number per day and sets `frame.attrs["unscored_user_hours"]`. The runner writes it to `scores_summary.json`.
   * Test: `test_user_hours_with_events_but_no_gru_score_are_counted_not_hidden`.
3. **A score frame with a missing column or a NaN fusion score gave a bare `KeyError` or was silently ignored.** Status: fixed.
   * Reproduce: pass `scores.drop(columns=["gru_max_event"])`, or a frame with one NaN in `fusion`, to `build_alert_package`.
   * Fix: `build_alert_package` in `src/dualscope/pipeline/alerts.py` raises a `ValueError` that names the missing columns or counts the rows without a fusion score, before it writes anything.
   * Test: `test_score_frames_with_missing_columns_or_scores_are_refused_by_name`.
4. **An HTTP 200 reply that is not JSON stopped the whole generation run.** Status: fixed.
   * Reproduce: make the Gemini endpoint (or a proxy in front of it) return status 200 with an HTML body, then run `scripts/run_investigations.py`.
   * Before: `response.json()` raised a `ValueError` in `GeminiClient.generate`, `investigate` only caught `GenerationError`, so the run crashed, the final tidy-up and `summary.json` were skipped and other incidents in flight were not recorded as failed.
   * Fix: `generate` in `src/dualscope/investigation/gemini.py` treats an unreadable 200 reply as a transient failure, retries it and ends with `GenerationError`, which becomes a `failed` record for that incident.
   * Tests: `test_client_turns_every_transport_failure_into_a_generation_error`, `test_investigate_records_a_timeout_and_an_unreadable_reply_as_failed`.
5. **A damaged `pipeline_state.json` could crash the runner.** Status: fixed.
   * Reproduce: write `[]` into `<run>/pipeline_state.json` and start the runner.
   * Fix: `PipelineState` in `src/dualscope/pipeline/runner.py` ignores a file that is not an object with a `stages` object, as it already did for unreadable JSON.
   * Test: `test_a_damaged_state_file_means_every_stage_runs_again`.
6. **A generation run killed in the middle of writing a reply leaves a cut-off last line.** Status: open, low priority.
   * Reproduce: truncate the last line of `generated_rag.jsonl` in an investigation folder (as a kill during the write would) and run `scripts/run_investigations.py` again.
   * Observed: the run stops with `<file> line N is not valid JSON; fix or remove it and run again`. No data is lost and the message is clear, but the user has to delete the line by hand. Each reply is written as one flushed line, so the window is very small.
   * Possible fix: drop a cut-off final line when reading and let the resume path generate that incident again.
7. **The per-day (parallel) ingestion path splits lines on commas, the sequential path uses a CSV reader.** Status: open, low priority.
   * Reproduce: a line with a quoted field that contains a comma gives different field counts in the two paths.
   * Observed: the LANL authentication file has no quoted fields, so the full run is not affected. The sample profile uses the sequential path.
   * Possible fix: use `csv.reader` in `_ingest_authentication_day` too, if the speed cost on the full file is acceptable.
8. **The dashboard skips unreadable lines in the investigation files without a message.** Status: open, by design.
   * Reproduce: put a bad line into `rag_verified.jsonl`.
   * Observed: that incident shows "No AI summary yet" while the rest of the run displays normally. The detector evidence tabs are never blocked. The headless dashboard check counts the states, so a run with damaged investigation files shows up as a rise in `pending`. We kept the tolerant reader because the alerts must stay viewable.
9. **One order inversion stops a whole ingestion run.** Status: open, by design.
   * Observed: the 508 million line file is read once and the first order inversion in a valid line raises, so a single bad timestamp costs the run up to that point. The run leaves no `summary.json`, and the message gives the line. The LANL file is sorted, so this has not occurred.

## What the checks do not cover

* Real Gemini behaviour. Timeouts, rate limits and bad replies are simulated, so retry timing and token refresh against Vertex AI are untested.
* The GRU and fusion models themselves. Scoring failures are tested with fake models and frames. The real release is covered by `tests/test_pipeline_release.py` and `tests/test_pipeline_scoring.py`.
* The Streamlit pages beyond what the earlier dashboard tests render. Here the dashboard side is checked through the headless check (`check_package`), which uses the same loaders.
