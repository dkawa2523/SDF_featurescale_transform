# View

`view` selects the 2D observation plane used by compare metrics and shape
inspection images.

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

| key | meaning |
| --- | --- |
| `axes` | The two axes shown in the 2D view. |
| `depth_axis` | The remaining axis used for projection. |

`axes` and `depth_axis` must use `x`, `y`, and `z` exactly once.

For label volumes, projection keeps the first visible non-void material along
the depth axis. If no material is visible, the pixel stays void.

Material-aware metrics may use both the outer boundary and internal material
boundaries.

Use `[x,z]` or `[y,z]` when comparing height-wise CD. A top-view `[x,y]`
comparison can be useful for projected shape overlap, but it cannot answer
whether SDF shape-distance is better than height-wise CD.
