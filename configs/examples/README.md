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
They also include `sdf_band`, which reuses the SDF feature but scores only a
default 10 nm boundary neighborhood.
`sdf_material` is included to report per-material SDF loss without requiring
users to list material ids in YAML.
Compare runs write `per_material_sdf.csv` when that metric is requested.
Batch runs write `metric_summary.csv` and `ranking_top.png` for quick checks;
use the CSV files for downstream analysis.
