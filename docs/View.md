# View

`view` は、3D label volume をどの 2D 観察面にそろえるかを決める設定です。
`compare` 系 metric は、この 2D view 上で計算されます。

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

この例では `x-z` 断面を使い、`y` 方向から見た観察にそろえます。

## 軸の意味

| 設定 | 意味 |
| --- | --- |
| `axes` | 比較・観察に使う 2 軸 |
| `depth_axis` | 投影方向。`axes` に含めない残り 1 軸 |

`axes` と `depth_axis` は `x/y/z` をちょうど一度ずつ使う必要があります。

## label volume の投影

label volume target の場合、各 pixel は `depth_axis` の正方向側から見て、
最初に見える non-void material になります。

これにより、単なる最大 ID ではなく、観察に近い topmost non-void 表現になります。
depth 方向に non-void が無い pixel は `void_id` のままです。

## boundary の扱い

material-aware compare では、外形だけでなく内部 material 境界も使います。

material boundary は次の条件で定義します。

- 隣接 pixel がどちらも non-void
- material ID が異なる

内部 material boundary が無い単一 material 形状では、外形 boundary に fallback します。

## contour_json の扱い

`contour_json` は SEM 専用ではなく、一般の輪郭座標データです。

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
      "points": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    }
  ]
}
```

YAML の `view.axes` で、3D 点 `[x,y,z]` のうち比較に使う 2 軸を選びます。
`units` は未指定なら `nm` として扱います。
