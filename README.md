# wafergeo

`wafergeo` converts simulation shape data into features and compares those
features with contour or label-volume target data.

The public workflow is intentionally small:

- `transform`: build features from one simulation label volume.
- `batch-transform`: build features from multiple simulation label volumes.
- `transform-eval`: evaluate transform features on the same inputs.
- `compare`: compare one simulation label volume with one target.
- `batch-compare`: compare multiple simulation-target pairs and write a ranking.
- `compare-eval`: compare evaluation axes on the same pairs.

Downstream tools should consume the files written by this package instead of
being built into the public workflow.

## Install

```powershell
Set-Location <repo>
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

Use `vtk` only when you need VTI input:

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## Run

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run batch-transform --config .\configs\examples\batch-transform.simple.yaml
python -m wafergeo run transform-eval --config .\configs\examples\transform-eval.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
python -m wafergeo run compare-eval --config .\configs\examples\compare-eval.simple.yaml
```

Users edit YAML files and read files under `outputs/`.

## Docs

The user and developer manuals are under `docs/`. To view them as a local
MkDocs site:

```powershell
py -3.13 -m pip install -e ".[dev]"
py -3.13 -m mkdocs serve
```

To validate the documentation build:

```powershell
py -3.13 -m mkdocs build --strict
```

## Inputs

Simulation inputs:

- `npz_label`
- `vti_label`

Target inputs:

- `contour_json`
- `npz_label`
- `vti_label`

`npz_label` is user-facing `[X,Y,Z]`:

```text
labels: integer array, shape [X,Y,Z]
spacing: optional float array, shape [3], order [X,Y,Z]
origin: optional float array, shape [3], order [X,Y,Z]
material_ids: optional integer array
material_names: optional string array
void_id: optional integer, required when material id 0 is not present
```

`contour_json` accepts 2D or 3D points. YAML `view.axes` selects the two axes
used for comparison.

```json
{
  "schema_version": "contour/v1",
  "units": "nm",
  "coordinate_axes": ["x", "y", "z"],
  "contours": [
    {
      "id": "outer",
      "label": "global",
      "material_id": null,
      "closed": true,
      "points": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 80.0, 0.0]]
    }
  ]
}
```

For label-volume targets, use the same `npz_label` or `vti_label` contract as
simulation input. This is the recommended path when both simulation and target
are material labels and internal material geometry matters.

`cd` is a cross-section critical-dimension profile metric. Select `[x,z]` or
`[y,z]` in `view.axes` when using `cd`; top-view `[x,y]` comparisons should use
`sdf` and `iou` first, adding `chamfer` only when boundary-point diagnostics are
needed. For label-volume targets, the default CD compares
internal material-boundary positions by height and also records the center-line
width profile, so it can catch internal shape shifts even when the outer width
is unchanged. `metrics.cd.material_ids` can focus CD on a specific material.
Optional `metrics.cd.gauge` makes the CD measurement location explicit:

```yaml
metrics:
  use: [cd, sdf, iou]
  cd:
    material_ids: [2]
    gauge:
      axis: x
      height_axis: z
      center: 4.0
      height_range: [20.0, 120.0]
```

When `gauge` is omitted, CD defaults to `height_axis: z`, the non-`z` view axis
as the width axis, the center of the projected view, all available height
samples, and automatic material-boundary transition scoring.
`sdf_material` reports per-material SDF losses for every detected non-void
material, so users do not need to list material ids just to find the material
driving an error.
Its total loss is weighted by projected union area so tiny materials do not
dominate the ranking by default.
`sdf_band` uses the existing SDF feature but scores only a default `10 nm`
boundary neighborhood, which makes interface placement errors easier to see.
Treat `chamfer`, `sdf_material`, `sdf_band`, `profile`, and `corner` as
diagnostic additions rather than the first comparison axis to try.

## Outputs

`compare` writes:

```text
objective.json
score.json
metrics.csv
metric_details.json
per_material_sdf.csv
difference.png
difference_legend.json
difference_summary.json
simulation_label_summary.json
target_label_summary.json
cd_profile.csv
cd_profile.png
cd_profile_summary.json
features/
_run/
```

`batch-compare` also writes:

```text
objectives.csv
ranking.csv
ranking_top.png
metric_summary.csv
score_summary.json
difference_summary.csv
shared_targets/
differences/
cases/
```

`ranking.csv` is sorted by `normalized_total_score`, which uses metric scales
from the registry. Raw `total_score` is also written for debugging in the
original units.
Repeated label-volume targets are stored once under `shared_targets/` so case
directories do not duplicate the same target feature files.
`metric_details.json` records SDF and IoU component details, such as label-field
loss, per-material loss, and projected material-boundary loss.
`per_material_sdf.csv`, `metric_summary.csv`, `cd_profile.png`, and
`ranking_top.png` are lightweight derived outputs for quick inspection. CSV/JSON
files remain the authoritative data.

`transform-eval` writes:

```text
eval_feature_summary.csv
eval_feature_signal.csv
case_summary.csv
case_variation_summary.csv
feature_stats.csv
material_coverage.csv
summary.json
figures/
```

Use it to compare feature methods for dataset creation. Read it as
`target_shape x method`: full shape, material-specific shapes, and
process-delta shape are separate target shapes. Material-interface and
process-transition relation files are derived from SDF fields; they are not
extra SDF methods. `feature_scores.csv` separates `shape_match`,
`boundary_match`, `interface_match`, `transition_match`, `case_sensitivity`,
and `data_cost`.
`case_distance.csv` and `distance_correlation.csv` show how each explicit
`target_shape` and `method` pair separates cases and whether two outputs carry
similar information. PNG diagnostics are written under
`figures/by_target_shape/<target_shape>/<method>/`; relation diagnostics are
under `figures/by_target_shape/<target_shape>/relations/<relation>/`.

`compare-eval` writes:

```text
metric_set_summary.csv
case_scores.csv
metric_summary.csv
ranking_consistency.csv
axis_agreement.csv
summary.json
figures/
```

Use it to choose an evaluation axis for comparison. It reuses existing compare
behavior and does not introduce new metrics or features. It keeps per-axis case
details in the root CSV files and writes only small diagnostic figures under
`figures/`.
The YAML block is named `eval.metric_sets` for compatibility; each key in that
block is an evaluation axis such as `height_cd` or `shape_distance`.
`case_scores.csv` exposes `comparison_loss` for ranking or optimization.
`metric_summary.csv` includes `metric_family` so shape-distance,
boundary-band-distance, material-distance, shape-overlap, and geometry-measure
losses are easy to read.
`ranking_consistency.csv` exposes `ranking_shift` relative to the baseline
evaluation axis.
`axis_agreement.csv` and `figures/cd_vs_sdf_scatter.png` compare evaluation-axis
`comparison_loss` values to show whether the SDF axis adds information beyond
the conventional height-wise CD baseline.

`_run/` contains `used_config.yaml` and `run_info.json` for debugging. It is not
an input contract and can be deleted when not needed.

## Development

Keep public concepts simple:

- Add new input formats as loaders.
- Add new feature conversions under the feature layer.
- Add new comparison methods as metrics.
- Keep YAML shallow and avoid adding new public pipelines unless there is a
  clear user workflow.

Run checks:

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
```

Generated outputs and caches are local artifacts. Remove them with `make clean`
when you want a tidy workspace.
