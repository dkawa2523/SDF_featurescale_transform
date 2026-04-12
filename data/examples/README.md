# Example Data

Tiny NPZ label volumes used by the bundled YAML examples.

The arrays follow the public `npz_label` contract:

```text
labels: [X,Y,Z]
spacing: [X,Y,Z]
origin: [X,Y,Z]
material_ids: [0,1,2]
```

`target_case.npz` and `sim_case.npz` are intentionally small cross-section
shapes with both width and internal material-boundary differences, so `cd`,
`chamfer`, `sdf`, and `iou` can all be smoke-tested quickly.
