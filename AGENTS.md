# AGENTS.md

This file defines the operating rules for Codex and other coding agents working
in this repository. Keep the project focused, small, and easy to extend.

## Project Direction

The public workflows are intentionally limited to:

- `transform`
- `compare`
- `batch-compare`

Do not add a new public workflow unless the user explicitly asks for it and the
design tradeoff is discussed first.

## Do Not Reintroduce Removed User-Facing Concepts

Do not bring these concepts back into the normal CLI, YAML, docs, or examples:

- `manifest`
- `report`
- `surrogate`
- `assimilation`
- `benchmark`
- `preview`
- `audit`

If one of these is needed for investigation, keep it outside the public user
path and ask before adding it back to the package.

## Keep YAML Shallow

Public YAML should stay close to this shape:

- `task`
- `input`
- `view`
- `features`
- `metrics`
- `output`

Do not introduce profiles, multiple config layers, hidden run specs, or deeply
nested configuration unless the user explicitly approves the added complexity.

## Extension Rules

Classify every feature request before editing:

- New input format: add or adjust a loader.
- New feature transform: add or adjust feature extraction or feature output code.
- New metric: add or adjust the metric registry and focused tests.
- New output artifact: add a small writer, but keep CSV/JSON as the authoritative data.

Keep runners orchestration-only. Do not put metric math, loader parsing, or
feature algorithms directly into CLI or runner code.

## Risk Control For New Methods

When adding methods, follow `docs/RiskControlPlan.md`.

Do not implement multiple new methods in one change. A single change should
usually cover one of:

- one metric
- one feature
- one loader
- one output artifact
- docs-only planning

New metrics or features must not become defaults in the same change that
introduces them. They should run only when explicitly listed in `metrics.use` or
`features.use`.

Before implementing a new method, confirm:

- which existing metric or feature it improves on
- what CSV/JSON output explains it
- when it should return `SKIPPED`
- which small synthetic tests prove it works
- what is explicitly out of scope

## Required Kickoff Before Implementation

Before editing files for a new method or broad improvement, provide a concise
implementation kickoff note to the user. It must include:

- target layer: one of `loader`, `feature`, `metric`, `output`, `docs`, or `tests`
- user-facing behavior: command, YAML name, and output files that will change
- non-goals: what will not be implemented in this change
- risk controls: how YAML, runner, tests, and outputs will stay small
- confirmation needs: only the concrete decisions that cannot be safely inferred

If there are no blocking confirmation needs, state the assumptions and proceed.
Do not ask broad open-ended questions.

For method implementation, use this default contract unless the user says
otherwise:

- implement exactly one method
- do not add a public workflow
- do not add the method to default metrics or default features
- do not add top-level YAML sections
- keep runner and CLI orchestration-only
- prefer CSV/JSON outputs over PNG
- keep tests focused and synthetic

## Scope Control For Broad Requests

When the user asks for broad work such as "残件を全部進めて" or
"不要なものを削除して", do not expand the architecture by default.

First constrain the task to the smallest useful layer:

- `loader`
- `feature`
- `metric`
- `output`
- `docs`
- `tests`

If the change would add a public concept, a new workflow, or a deeper YAML
structure, pause and ask before implementing.

## Cleanup Rules

Do not commit generated artifacts:

- `outputs/`
- `site/`
- caches
- temporary experiment directories

Small official examples under `data/examples/` and `configs/examples/` are
allowed when they are required for quickstart or tests.

## Test Rules

Prefer focused behavior tests over broad legacy tests.

For a new loader, feature, or metric, add only the smallest tests that prove:

- the normal path works
- one important invalid input fails clearly
- the public workflow still works

Do not revive tests for removed public concepts. Do not add heavy dataset tests
to the default test suite.

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

Remove `site/` after build validation if it was generated.
