# Investigation review: reference labels (Task 9.5)

## Purpose

Reference ATT&CK techniques that the evidence supports are labelled for 25 incidents **before any LLM output is seen**. The three investigation modes (`direct`, `rag`, `rag_verified`; Tasks 6.2-6.4) are then scored against these labels.

## The sample

`data/reference/evaluation/investigation_review_sample_v1.json` holds 25 incidents from the 497 in `outputs/handoff/final_test_alerts_v1`. They were chosen with detector and evidence fields only (priority, graph detector agreement, the Task 6.1 behaviour rules, event count, dataset day), never with LLM output. Incidents are picked at random (seed 0) until each minimum below is met, with at most 3 incidents per dataset day. Re-running the script gives the same list.

| Stratum | Sampled | Minimum |
| --- | --- | --- |
| Priority MEDIUM | 11 | 7 |
| Priority HIGH | 14 | |
| Graph detector also alerted | 4 | 4 |
| No behaviour rule matched | 5 | 5 |
| `failed_logons` | 6 | 4 |
| `ntlm_logon_to_new_destination` | 10 | 6 |
| `network_logon_to_new_destination` | 12 | 6 |
| `logon_from_new_source` | 9 | 5 |
| More than 1000 events | 4 | 3 |
| Fewer than 60 events | 10 | 5 |

Behaviour rows overlap (an incident can match several). The table is printed by `select_review_sample.py`; the figures above are from the committed sample.

## Rules for reviewers

1. Label each incident before you see any LLM output and before you open the dashboard's Investigation tab.
2. Use only the incident sheet (`incidents/<incident_id>.html`, open `index.html` first) and <https://attack.mitre.org/>. The sheet shows exactly what the LLM sees.
3. Fill a label workbook (`labels_reviewer_A.xlsx`). The sheet `labels` has one pre-filled row per incident. For each technique write one row (copy the `incident_id` for extra rows):
   - `technique_id`, in the form `T####` or `T####.###`. Prefer the parent technique when the sub-technique cannot be told from logon records.
   - `judgement`:
     - **Supported**: the cited events show the behaviour the technique describes. This means consistent with it, not proof of intent.
     - **Uncertain**: plausible, but the logs cannot show it.
     - **No supported mapping**: use one row with an empty `technique_id` if nothing fits.
   - `evidence_refs`: the `auth.txt:N` references, separated by semicolons.
   - `notes`: anything that explains the choice.
4. The final labels go in `labels_consensus.xlsx` (same layout, reviewer `consensus`); this is the file the scores use.

## How the labels are used

`scripts/evaluate_investigations.py` scores `direct`, `rag` and `rag_verified` against the consensus labels. It reports technique precision, the unsupported-generation rate, empty outputs, abstentions, and supported mappings that verification removed. Valid ATT&CK identifiers are kept separate from evidence-supported mappings, and the sample size and denominators are reported with every figure.

## Regenerate

```
python scripts/select_review_sample.py
python scripts/build_review_pack.py
```

The first writes the committed sample file. The second writes the pack to `outputs/evaluation/review_sample_v1/` (not committed) and fails if any file mentions the answer key or an incident sheet contains an ATT&CK technique ID. It needs `outputs/handoff/final_test_alerts_v1` and `openpyxl`.

## How the labels were made

The first labels came from two separate Claude Sonnet agents (reviewers A and B). Each labelled all 25 incidents on its own, without seeing the other's labels. They worked only from the incident sheets (`outputs/evaluation/review_sample_v1/incidents/`) and the ATT&CK Enterprise 19.2 technique list (`data/reference/attack/enterprise_techniques.json`). They did not see any Gemini output, the 6.1 retrieved candidates, the dashboard's Investigation tab or the answer key. Claude is a different model from the Gemini model being scored, so the scored model is not marking its own work.

The two reviewers agreed on 19 incidents, and those labels were kept as given. Peter decided the other 6. For two of them (`INC-TEST-D21-U8887_DOM1-001` and `INC-TEST-D29-U1179_DOM1-002`) he also checked 30 days of log context: how widely the destination computers are used, and how active each user normally is in a day. He did not use the answer key or any model output. The final labels are in `labels_consensus.xlsx`.

The labels were finished on 2026-10-07, before `scripts/evaluate_investigations.py` was first run on them. The notes column was reworded afterwards without changing any label, and rerunning the scores gave the same results.

## Results (2026-10-07)

The 25 consensus labels hold 8 incidents with an Uncertain technique (T1021 x5, T1078 x2, T1110 x1) and 17 with no supported mapping. There are 0 Supported labels. The labels were recorded before scoring.

Scores are in `outputs/evaluation/investigation_scores_v1/` (`report.md`, `scores.json`, `per_incident.csv`). Rates are numerator/denominator.

### Mapping quality

| Metric | direct | rag | rag_verified |
| --- | --- | --- | --- |
| Valid ATT&CK ID | 100.0% (25/25) | 100.0% (22/22) | 100.0% (22/22) |
| Precision (strict) | 0.0% (0/25) | 0.0% (0/22) | 0.0% (0/22) |
| Precision (parent-level) | 0.0% (0/25) | 0.0% (0/22) | 0.0% (0/22) |
| Uncertain matches | 20.0% (5/25) | 13.6% (3/22) | 13.6% (3/22) |
| Unsupported generation | 80.0% (20/25) | 72.7% (16/22) | 72.7% (16/22) |
| Recall (parent-level) | n/a (0/0) | n/a (0/0) | n/a (0/0) |

### Empty outputs, abstentions and false mappings

| Metric | direct | rag | rag_verified |
| --- | --- | --- | --- |
| Empty outputs (failed/invalid/missing) | 0.0% (0/25) | 0.0% (0/25) | 0.0% (0/25) |
| False mappings on no-mapping incidents | 58.8% (10/17) | 47.1% (8/17) | 47.1% (8/17) |
| Correct abstentions | 100.0% (8/8) | 100.0% (11/11) | 100.0% (11/11) |
| Missed abstentions (reference has Supported) | 0.0% (0/8) | 0.0% (0/11) | 0.0% (0/11) |

### Verification impact (rag to rag_verified)

| Metric | Value |
| --- | --- |
| Techniques removed | 0 |
| Correctly removed (parent not Supported) | n/a (0/0) |
| Supported mappings wrongly removed | n/a (0/0) |

Kept techniques: verifier status against the reference label (exact ID).

| Verifier / Reference | Supported | Uncertain | No label |
| --- | --- | --- | --- |
| Supported | 0 | 2 | 14 |
| Uncertain | 0 | 1 | 5 |

### Reading the results

- **Do not quote "precision 0%" on its own.** No reference label is Supported, so strict and parent-level precision are 0 by construction and recall is undefined. The meaningful measure is agreement with any reference label (Supported or Uncertain), which is 1 minus unsupported generation: direct 5/25 (20%), rag 6/22 (27%), rag_verified 6/22 (27%).
- RAG mapped techniques on fewer no-mapping incidents (8/17 vs 10/17) and produced fewer unsupported mappings (16/22 vs 20/25) than direct. This is a modest improvement on a small sample.
- All modes over-label relative to the reference: about half of the incidents judged routine still got a technique.
- Verification removed nothing within the sample. Of the kept techniques, 14 marked Supported by the verifier fall on incidents the reference labels as no mapping. The verifier checks that the cited events exist and show the pattern, not whether the activity is malicious or routine. This is a limitation, not a bug.
- No mode abstained wrongly (0 missed abstentions), and there were 0 empty outputs.
- The sample is 25 incidents. Differences of 1 to 2 incidents are within noise.

### Reproduce

```
python scripts/evaluate_investigations.py
```

All options use their defaults. Output goes to `outputs/evaluation/investigation_scores_v1/` (not committed).
