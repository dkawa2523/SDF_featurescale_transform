# Developer Manual

Keep the code structure simple.

## Responsibilities

| layer | responsibility |
| --- | --- |
| schema | Load shallow YAML into typed specs. |
| runner | Resolve paths, run workflow steps, collect rows, call writers. |
| loader | Read input files into domain objects. |
| feature | Convert geometry into CSV/JSON/NPZ feature outputs. |
| metric | Compare simulation and target features. |
| figure writer | Read existing outputs and write diagnostic PNG/CSV files. |

Do not put loader parsing, feature math, or metric math directly into runners.

## Naming

Use [Terminology](Terminology.md) for feature wording. Code-specific feature
names must map to:

```text
target_shape x method
```

Keep that mapping in `wafergeo/compare/feature_taxonomy.py`. User-facing
transform-eval docs should show only `target_shape`, `method`, and `code_name`.

## Important Files

| path | purpose |
| --- | --- |
| `wafergeo/compare/schema_types.py` | YAML-facing spec types. |
| `wafergeo/compare/runner.py` | `transform` and `compare`. |
| `wafergeo/compare/batch_transform_runner.py` | `batch-transform`. |
| `wafergeo/compare/transform_eval_runner.py` | `transform-eval`. |
| `wafergeo/compare/batch_runner.py` | `batch-compare`. |
| `wafergeo/compare/compare_eval_runner.py` | `compare-eval`. |
| `wafergeo/compare/feature_outputs.py` | Transform feature dispatch. |
| `wafergeo/compare/feature_taxonomy.py` | Feature naming map. |
| `wafergeo/compare/transform_eval_figures.py` | Transform-eval diagnostics. |
| `wafergeo/compare/compare_eval_figures.py` | Compare-eval diagnostics. |
| `wafergeo/compare/metric_defs.py` | Metric registry. |
| `wafergeo/compare/loader.py` | Input loader dispatch. |

## Checks

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
py -3.13 -m mkdocs build --strict
```

Do not commit `outputs/`, `site/`, caches, or temporary experiment files.
