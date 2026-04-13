# configs/runs

This directory is for developer smoke checks, not everyday user examples.

Normal users should start from `configs/examples/`.

The files here are intentionally small wrappers around public workflows:

- `dataset_t08_vs_run0010.yaml` uses `batch-compare` with the standard shape
  metrics on the original VTI data. Use this for the existing all-run
  ranking check.
- `dataset_t08_compare_eval.yaml` uses `compare-eval` to compare evaluation
  axes on the original VTI case pairs. Use this for the existing all-run
  comparison-axis check.
- `dataset_t08_npz_compare_eval.yaml` uses `compare-eval` on selected restored
  NPZ cases. Use this as the restored-evaluation-data smoke check.
- `dataset_t08_npz_transform_eval.yaml` uses `transform-eval` on the same
  selected restored NPZ cases. Use this to reproduce the `target_shape x
  method` evaluation.

In short:

- NPZ is the restored evaluation-data smoke path.
- VTI is the existing all-run evaluation path.
- `transform-eval` checks whether each `target_shape x method` feature
  expresses useful variation in the input geometry. SDF-derived relation files
  expose material interfaces and process transitions without adding new SDF
  method names. `compare-eval` checks height-CD, SDF shape distance, material
  distance, and boundary-band distance as separate evaluation axes for target
  comparison.

They must not introduce a new public pipeline. Results are regenerated under
`outputs/` and should not be committed.
