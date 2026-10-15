# Release model files

These are the frozen files of the final DualScope detection model, copied unchanged from the run of record:

- `gru/` the GRU autoencoder (run A) that produces the `max_event` score.
- `fusion/` the HistGradientBoosting model that combines that score with hourly event counts.
- `features/preprocessing.json` the feature preprocessing fitted on the training days. Any feature build scored with this model must use this exact file.

`manifest.json` lists every file with its sha256, size and source path. The loader in `dualscope.pipeline.release` checks all of them before use.

The models are never retrained. The one-time final test on days 17-30 was run on 2026-10-06 and must not be repeated.
