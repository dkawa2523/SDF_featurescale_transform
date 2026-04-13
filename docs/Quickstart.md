# Quickstart

Run from the repository root.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

Add `vtk` only when you need VTI input.

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## Run Examples

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run batch-transform --config .\configs\examples\batch-transform.simple.yaml
python -m wafergeo run transform-eval --config .\configs\examples\transform-eval.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
python -m wafergeo run compare-eval --config .\configs\examples\compare-eval.simple.yaml
```

## Check Outputs

| workflow | first files to inspect |
| --- | --- |
| `transform` | `features/`, `feature_summary.json` |
| `batch-transform` | `dataset_index.csv`, `features_summary.csv` |
| `transform-eval` | `feature_scores.csv`, `figures/by_target_shape/` |
| `compare` | `objective.json`, `metrics.csv`, `difference.png` |
| `batch-compare` | `objectives.csv`, `ranking.csv` |
| `compare-eval` | `axis_agreement.csv`, `case_scores.csv`, `figures/` |

Generated outputs live under `outputs/` and should not be committed.

## Read Eval Results

For `transform-eval`, start with:

1. `figures/input_shape_sections.png`
2. `figures/feature_scores.csv`
3. `figures/by_target_shape/<target_shape>/<method>/field.png`
4. `figures/by_target_shape/<target_shape>/<method>/case_distance.png`

Use `target_shape x method` when reading transform results. For example,
`material_shape/sdf` means SDF applied to each material shape.

For `compare-eval`, start with:

1. `axis_agreement.csv`
2. `figures/cd_vs_sdf_scatter.png`
3. `figures/comparison_loss_heatmap.png`
4. `figures/representative_differences/`

`cd_vs_sdf_scatter.png` compares evaluation-axis `comparison_loss` values:
the height-CD baseline axis against the SDF shape-distance axis. Use
`case_scores.csv` when you need the raw per-metric losses.

## Validate

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
py -3.13 -m mkdocs build --strict
```

Remove generated `site/` after a docs build.
