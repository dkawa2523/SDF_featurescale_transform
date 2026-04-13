# AGENTS.md

This file defines the operating rules for Codex and other coding agents working
in this repository. Keep the package useful, small, and easy to extend.

## Product Direction

`wafergeo` exists to:

- transform simulation or measurement-derived geometry into reusable features;
- compare simulation and target observations;
- create feature datasets that can be used by external analysis or surrogate
  learning code;
- make it easy for third-party developers to add loaders, features, metrics,
  and outputs without changing the public usage model.

The roadmap is documented in `docs/WorkflowRoadmap.md`. Treat that file as the
source of truth for planned workflow expansion.

## Public Workflows

Current implemented workflows:

- `transform`
- `batch-transform`
- `transform-eval`
- `compare`
- `batch-compare`

Planned workflow family:

- Feature workflows: `transform`, `batch-transform`, `transform-eval`
- Compare workflows: `compare`, `batch-compare`, `compare-eval`

Do not add workflows outside this family unless the user explicitly approves a
new product direction. When adding a planned workflow, keep its role aligned
with `docs/WorkflowRoadmap.md`.

## Do Not Reintroduce Old User-Facing Concepts

Do not bring these concepts back into the normal CLI, YAML, docs, or examples:

- `manifest`
- `report`
- `surrogate`
- `assimilation`
- `benchmark`
- `preview`
- `audit`

Surrogate learning is an external downstream use. This package should create
feature datasets, not train surrogate models.

Here `preview` means the old user-facing preview workflow or CLI concept.
Eval visualization is intentionally absent for now. Do not add it unless the
user provides a new explicit design.

## YAML Policy

Keep normal YAML shallow:

- `task`
- `input`
- `view`
- `features`
- `metrics`
- `output`

Eval workflows may use one additional block:

- `eval.candidates`

Process-aware transform may use one additional explicit block:

- `process.enabled`

Do not introduce profiles, hidden run specs, multiple config layers, or deeply
nested options unless the user explicitly asks for that complexity.

## Extension Rules

Classify every change before editing:

- New input format: loader.
- New 3D or 2D feature transform: feature code and feature output.
- New comparison method: metric registry and focused tests.
- New result file: small output writer, with CSV/JSON as authoritative data.
- Documentation-only clarification: docs.

Keep runners orchestration-only. Do not put metric math, loader parsing, or
feature algorithms directly into CLI or runner code.

## Method Implementation Contract

For new methods, default to:

- implement one method per change;
- do not make new methods default in the same change;
- use `features.use` or `metrics.use` for opt-in behavior;
- keep output data flat and readable;
- add only focused synthetic tests;
- avoid heavy dataset tests in the default suite.

Completed roadmap work:

- `sdf_raw`, `tsdf_views`, `udf`, `material_sdf`
- `material_profile`
- `batch-transform`
- `transform-eval`
- `process.enabled` and `input.reference`
- `process_delta_profile`
- `process_delta_sdf`
- transform-eval scalar/profile summaries
- minimal process-delta preview artifacts
- `compare-eval`

Next roadmap workflow work should focus on feature and metric methods, not on
visualization, until the user provides a new visualization design.

## Broad Request Handling

When the user asks for broad cleanup or "all remaining work", do not create a
new architecture by default. Pick the smallest useful layer and proceed.

Prefer deleting or shortening confusing docs over adding more guard documents.
If an old document conflicts with `WorkflowRoadmap.md`, update it to point to
the roadmap instead of preserving both versions.

## Cleanup Rules

Do not commit generated artifacts:

- `outputs/`
- `site/`
- caches
- `__pycache__/`
- temporary experiment directories

Small official examples under `data/examples/` and `configs/examples/` are
allowed when required for quickstart or tests.

## Test Rules

Before finalizing code changes, run the relevant checks:

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
```

When docs change, also run:

```powershell
py -3.13 -m mkdocs build --strict
```

Remove generated `site/` after build validation.
