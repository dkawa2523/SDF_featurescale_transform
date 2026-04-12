## Summary

- What changed:
- Why it is needed:

## Scope Guard

- [ ] Public workflows are still only `transform`, `compare`, and `batch-compare`.
- [ ] YAML top-level shape remains `task / input / view / features / metrics / output`.
- [ ] No manifest/report/surrogate/assimilation/benchmark/preview/audit concepts were reintroduced.
- [ ] New logic is contained in the appropriate layer: loader / feature / metric / output / docs / tests.
- [ ] Runner and CLI orchestration do not contain metric math or feature algorithms.

## User Simplicity

- [ ] Normal examples still start from `metrics.use: [cd, sdf, iou]`.
- [ ] Diagnostic metrics are opt-in and documented in `docs/Scoring.md`.
- [ ] New outputs are CSV/JSON first; PNG is only a lightweight helper.
- [ ] No generated `outputs/`, `site/`, or cache files are committed.

## Validation

- [ ] `py -3.13 -m ruff check wafergeo tests`
- [ ] `py -3.13 -m mypy wafergeo`
- [ ] `py -3.13 -m pytest -q`
- [ ] `py -3.13 -m mkdocs build --strict`

## Notes

- Residual risks:
- Follow-up work:
