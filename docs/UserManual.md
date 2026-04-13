# User Manual

`wafergeo` converts geometry data into reusable feature files and compares
simulation geometry with target geometry. External learning or optimization
code should consume the CSV/JSON/NPZ outputs written here.

## Inputs

| kind | use |
| --- | --- |
| `npz_label` | Label volume in an NPZ file. Use `labels` with shape `[X,Y,Z]`. Optional arrays: `spacing`, `origin`, `material_ids`. |
| `vti_label` | Label volume stored in VTI. |
| `contour_json` | Contour target for comparison metrics. |

If material id `0` is not void in your data, set `void_id` in YAML or the index
CSV.

## YAML Shape

Keep YAML shallow.

```yaml
task: transform

input:
  simulation:
    kind: npz_label
    path: data/final.npz

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf_raw, material_sdf, material_tsdf_views]

output:
  dir: outputs/example_transform
```

Process-aware features need a reference label and `process.enabled`.

```yaml
task: transform

input:
  simulation:
    kind: npz_label
    path: data/final.npz
  reference:
    kind: npz_label
    path: data/initial.npz

process:
  enabled: true

features:
  use: [process_delta_sdf, process_delta_tsdf_views]

output:
  dir: outputs/process_delta
```

## Workflows

| workflow | use |
| --- | --- |
| `transform` | Convert one case into feature files. |
| `batch-transform` | Convert many cases with the same feature list. |
| `transform-eval` | Compare feature methods on many cases. |
| `compare` | Compare one simulation with one target. |
| `batch-compare` | Compare many simulation/target pairs. |
| `compare-eval` | Compare evaluation axes on the same cases. |

## Transform Features

Read transform features with the terms defined in
[Terminology](Terminology.md):

```text
target_shape x method, plus relation outputs derived from SDF stacks
```

| feature | target_shape | method or relation |
| --- | --- | --- |
| `sdf_raw` | `full_shape` | `sdf` |
| `tsdf_views` | `full_shape` | `multi_scale_tsdf` |
| `udf` | `full_shape` | `udf` |
| `material_sdf` | `material_shape` | `sdf` |
| `material_tsdf_views` | `material_shape` | `multi_scale_tsdf` |
| `material_udf` | `material_shape` | `udf` |
| `material_interface_relation` | `material_shape` | relation |
| `process_delta_sdf` | `process_delta_shape` | `sdf` |
| `process_delta_tsdf_views` | `process_delta_shape` | `multi_scale_tsdf` |
| `process_delta_udf` | `process_delta_shape` | `udf` |
| `process_transition_relation` | `process_delta_shape` | relation |

`material_sdf` and `process_delta_sdf` are not separate methods. They are SDF
applied to different target shapes. Relation outputs describe material
interfaces or reference-to-final material transitions.

## Transform-Eval Outputs

`transform-eval` writes the normal eval CSV files plus diagnostic figures.
Use the figure directory as a quick inspection layer, not as the source of truth.

Important figure files:

| output | use |
| --- | --- |
| `figures/input_shape_sections.png` | Original material sections for all cases. |
| `figures/by_target_shape/<target_shape>/<method>/field.png` | Field report for one explicit target shape and method. |
| `figures/by_target_shape/<target_shape>/<method>/scores.png` | Scores for that target shape and method. |
| `figures/by_target_shape/<target_shape>/<method>/case_distance.png` | Case distance in that feature space. |
| `figures/by_target_shape/<target_shape>/relations/<relation>/field.png` | Relation report derived from SDF fields. |
| `figures/feature_scores.csv` | Machine-readable scores with separate `role`, `target_shape`, `method`, `relation`, and `code_name` columns. |
| `figures/case_distance.csv` | Machine-readable case distances with separate `role`, `target_shape`, `method`, and `relation` columns. |

Avoid reading one mixed score across all feature types. `shape_match`,
`boundary_match`, `interface_match`, `transition_match`, `case_sensitivity`,
and `data_cost` answer different questions.

## Compare Metrics

| metric | use |
| --- | --- |
| `cd` | Cross-section CD / edge position difference. |
| `chamfer` | Contour point distance. |
| `sdf` | 2D SDF loss. |
| `sdf_band` | Boundary-band SDF loss. |
| `sdf_material` | Per-material SDF loss. |
| `iou` | Mask overlap. |
| `profile` | Profile value difference. |
| `corner` | Local corner-shape difference. |
| `topology` | Connectivity and large-shape checks. |

See [Scoring](Scoring.md) for metric details.

## Compare-Eval Outputs

`compare-eval` compares named evaluation axes on the same cases. The YAML field
is still `eval.metric_sets`, but each key should be read as an evaluation axis:

| axis | use |
| --- | --- |
| `height_cd` | Height-wise CD baseline on `[x,z]` or `[y,z]` views. |
| `shape_distance` | SDF and IoU shape comparison. |
| `material_distance` | Per-material SDF diagnostic for label-volume targets. |
| `boundary_band_distance` | Boundary-neighborhood SDF diagnostic. |

Read compare-eval outputs in this order:

1. `axis_agreement.csv`
2. `figures/cd_vs_sdf_scatter.png`
3. `figures/comparison_loss_heatmap.png`
4. `figures/representative_differences/`

`comparison_loss` is the normalized value used for ranking within an evaluation
axis. `case_scores.csv` contains the raw per-metric columns such as `cd_loss`,
`sdf_loss`, `iou_loss`, `sdf_material_loss`, and `sdf_band_loss`.

## Generated Files

CSV/JSON/NPZ outputs are authoritative. PNG outputs are diagnostics.
Generated `outputs/`, `site/`, and caches should not be committed.
