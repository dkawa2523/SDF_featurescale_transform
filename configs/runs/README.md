# configs/runs

This directory is for developer smoke checks, not everyday user examples.

Normal users should start from `configs/examples/`.

The files here are intentionally small wrappers around public workflows:

- `dataset_t08_vs_run0010.yaml` uses `batch-compare` with the fixed primary
  metric set on the original VTI data. Use this for the existing all-run
  ranking check.
- `dataset_t08_compare_eval.yaml` uses `compare-eval` to compare a few metric
  candidates on the original VTI case pairs. Use this for the existing all-run
  metric-set check.
- `dataset_t08_npz_compare_eval.yaml` uses `compare-eval` on selected restored
  NPZ cases. Use this as the restored-evaluation-data smoke check.
- `dataset_t08_npz_transform_eval.yaml` uses `transform-eval` on the same
  selected restored NPZ cases. Use this to reproduce the feature-method
  evaluation, including final-shape features and process-delta features.

In short:

- NPZ is the restored evaluation-data smoke path.
- VTI is the existing all-run evaluation path.
- `transform-eval` checks whether each feature method expresses useful
  variation in the input geometry. `compare-eval` checks which metric set is
  useful for target comparison.

They must not introduce a new public pipeline or revive old benchmark/report/
manifest concepts. Results are regenerated under `outputs/` and should not be
committed.
