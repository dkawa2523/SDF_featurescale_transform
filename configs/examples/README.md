# Config Examples

Use these three YAML files for normal feature conversion and comparison work:

- `transform.simple.yaml`
- `compare.simple.yaml`
- `batch-compare.simple.yaml`

No other example YAML files are part of the public workflow.

Relative paths in YAML are resolved from the YAML file location. Relative paths
inside a batch index CSV are resolved from the CSV file location.

The compare examples use `[x,z]` because `cd` is a cross-section edge/width
metric. Use `[x,y]` for top-view checks when you do not need CD.

The default example metric set is intentionally small: `cd`, `sdf`, and `iou`.
Add `chamfer`, `sdf_material`, or `sdf_band` only when you need diagnostic
detail. Compare runs write `per_material_sdf.csv` when `sdf_material` is
requested.
Batch runs write `metric_summary.csv` and `ranking_top.png` for quick checks;
use the CSV files for downstream analysis.
