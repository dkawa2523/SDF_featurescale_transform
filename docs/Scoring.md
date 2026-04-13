# Scoring

`compare` computes metrics on a 2D view of simulation and target geometry.
Metrics write raw values to `metrics.csv`; weighted losses are combined into
`objective.json`.

## Metrics

| metric | use |
| --- | --- |
| `cd` | Cross-section width or edge-position difference. |
| `sdf` | Signed-distance loss over the view. |
| `sdf_band` | SDF loss near the boundary. |
| `sdf_material` | Per-material SDF loss. |
| `iou` | Mask overlap. |
| `chamfer` | Contour point distance. |
| `profile` | Profile value difference. |
| `corner` | Local corner-shape difference. |
| `topology` | Connectivity and large-shape checks. |

## Objective

```yaml
metrics:
  use: [cd, sdf, iou]
  weights:
    cd: 1.0
    sdf: 1.0
    iou: 1.0
```

Use `objective.json` for optimization or ranking. Use `metrics.csv` to debug
which metric contributed to the objective.

`compare-eval` compares evaluation axes on the same cases. Keep axis names tied
to the geometry question, for example:

| axis | intended use |
| --- | --- |
| `height_cd` | Conventional height-wise CD baseline on an `[x,z]` or `[y,z]` view. |
| `shape_distance` | Overall shape overlap and SDF distance. |
| `material_distance` | Per-material distance when label volumes are available. |
| `boundary_band_distance` | Near-boundary SDF sensitivity check. |

Use process-delta comparison only when a reference geometry is part of the
compare input.

Use `axis_agreement.csv` and `cd_vs_sdf_scatter.png` to check whether the SDF
evaluation axis adds information beyond height-wise CD. The scatter plot uses
axis-level `comparison_loss`; use `case_scores.csv` when you need raw `cd_loss`
or `sdf_loss`. If rankings are identical, SDF is not changing the decision for
that case set. If rankings differ, inspect the representative differences
before choosing an optimization objective.
