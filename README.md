# wafergeo

`wafergeo` converts simulation shape data into features and compares those
features with contour or label-volume target data.

The public workflow is intentionally small:

- `transform`: build features from one simulation label volume.
- `compare`: compare one simulation label volume with one target.
- `batch-compare`: compare multiple simulation-target pairs and write a ranking.

Downstream tools should consume the files written by this package instead of
being built into the public workflow.

## Install

```powershell
Set-Location C:\Users\user\Desktop\SDF_fs
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

Use `vtk` only when you need VTI or optional mesh tooling:

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## Run

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
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
diagnostic additions rather than the first metric set to try.

## Outputs

`compare` writes:

```text
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

`_run/` contains `used_config.yaml` and `run_info.json` for debugging. It is not
an input contract and can be deleted when not needed.

## Development

Before asking Codex or another coding agent to change the repository, read
`AGENTS.md` and `docs/MaintenancePolicy.md`. The project should stay focused on
the three public workflows and avoid reintroducing removed user-facing concepts
such as manifest, report, surrogate, assimilation, benchmark, preview, or audit.

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
