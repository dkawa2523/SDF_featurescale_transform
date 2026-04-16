# Config Examples

Use these YAML files for normal feature conversion and comparison work:

- `transform.simple.yaml`
- `batch-transform.simple.yaml`
- `transform-eval.simple.yaml`
- `compare.simple.yaml`
- `batch-compare.simple.yaml`
- `compare-eval.simple.yaml`

No other example YAML files are part of the public workflow.

Relative paths in YAML are resolved from the YAML file location. Relative paths
inside a batch index CSV are resolved from the CSV file location.

The compare examples use `[x,z]` because `cd` is a cross-section edge/width
metric. Use `[x,y]` for top-view checks when you do not need CD.

In compare YAML, `features.use` builds representations and `metrics.use`
computes losses from them. Required features:

| metric | required feature |
| --- | --- |
| `cd`, `chamfer`, `profile`, `corner` | `contour` |
| `sdf`, `sdf_band`, `sdf_material` | `sdf` |
| `iou`, `topology` | none |

The default example comparison axis is intentionally small: `cd`, `sdf`, and `iou`.
Add `chamfer`, `sdf_material`, or `sdf_band` only when you need diagnostic
detail. Compare runs write `per_material_sdf.csv` when `sdf_material` is
requested.
Batch runs write `metric_summary.csv` and `ranking_top.png` for quick checks;
use the CSV files for downstream analysis.

`compare-eval.simple.yaml` runs height-CD and SDF-based evaluation axes on the
same compare pairs.
It is for choosing a loss set, not for adding new metrics or features.
`transform-eval.simple.yaml` compares transform features on the same tiny
cases. Read feature names with `target_shape x method`:

- `sdf_raw`: `full_shape x sdf`
- `tsdf_views`: `full_shape x multi_scale_tsdf`
- `udf`: `full_shape x udf`
- `material_sdf`: `material_shape x sdf`
- `material_tsdf_views`: `material_shape x multi_scale_tsdf`
- `material_udf`: `material_shape x udf`

Relation files such as `material_interface_relation` are derived from SDF
stacks and are not separate SDF methods.
