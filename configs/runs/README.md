# configs/runs

This directory is for developer smoke checks, not everyday user examples.

Normal users should start from `configs/examples/`.

The files here are intentionally small wrappers around the public `batch-compare`
workflow. They must not introduce a new public pipeline or revive old
benchmark/report/manifest concepts. Results are regenerated under `outputs/`
and should not be committed.
