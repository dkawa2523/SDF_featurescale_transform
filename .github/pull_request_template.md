## Summary

- What changed:
- Why it is needed:

## Scope Guard

- [ ] Public workflows stay within `docs/WorkflowRoadmap.md`.
- [ ] Normal YAML top-level shape remains `task / input / view / features / metrics / output`.
- [ ] Eval YAML uses only the planned `eval.candidates` extension.
- [ ] No manifest/report/surrogate/assimilation/benchmark/preview/audit concepts were reintroduced.
- [ ] New logic is contained in the appropriate layer: loader / feature / metric / output / docs / tests.
- [ ] Runner and CLI orchestration do not contain metric math or feature algorithms.

## User Simplicity

- [ ] Normal examples still start from `metrics.use: [cd, sdf, iou]`.
- [ ] New features or metrics are opt-in and documented.
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
