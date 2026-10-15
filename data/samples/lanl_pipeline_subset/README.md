# LANL pipeline subset

A small fixed slice of the LANL authentication data that the pipeline runner's
`sample` profile processes from raw lines to dashboard incidents.

- `auth_subset.txt.gz`: 526,922 original auth.txt lines in original order.
- `line_map.txt.gz`: the original auth.txt line number of each subset line, so
  ingestion keeps true `auth.txt:<line>` evidence references.
- `redteam_subset.txt`: the original red-team lines for the selected users on
  days 1-9 (296 rows).
- `manifest.json`: selection rules, seed, selected users, per-day counts and
  sha256 sums of every file here.

## What is in it

Days 1-9 only. Events of 50 red-team users, 200 randomly sampled human users
and 10 randomly sampled machine accounts. Red-team users above the event cap were left out: U66@DOM1 (1,594,934 events).

Days 17-30 (the test days) are never read. Day 9 was part of the fusion model's
training data, so catches on that day are not a fair accuracy result.

## Scores differ from the full run

Features that use other users' history on the same hosts see only the selected
users here, so scores from the subset differ from the full days 1-30 run. Use it
to check that the pipeline works end to end, not to report accuracy.

## Regenerate

    python scripts/make_pipeline_subset.py

This reads the Task 2.2 Parquet output under data/processed/lanl_auth_days_01_30. Without that output,
stream the raw files instead:

    python scripts/make_pipeline_subset.py --auth path/to/auth.txt.gz --redteam path/to/redteam.txt

Both routes give byte-identical files. Seed: 0.
