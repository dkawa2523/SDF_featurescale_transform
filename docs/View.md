# view の考え方

`view` は、3D の label volume や contour をどの 2D 面で比較するかを決める設定です。

```yaml
view:
  kind: topview
  axes: [x, z]
  depth_axis: y
```

この例では、`x-z` 面を比較し、`y` 方向に投影します。

## 基本動作

- label volume は選択した 2D 面へ投影されます。
- `label2d` は `depth_axis` の最大 index 側から見た topmost non-void material です。
- material-aware metric のため、non-void material ごとの投影 mask も保持します。
- material boundary は「隣接 voxel がどちらも non-void で、material id が異なる場所」です。
- 内部 material boundary が無い場合は、外形 boundary に fallback します。
- `contour_json` target は同じ `axes` を使って 2D に投影されます。

## CD と view

CD は半導体断面の高さごとの幅・edge 位置を評価する metric です。
そのため通常は `z` を含む view を使います。

推奨:

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

または:

```yaml
view:
  axes: [y, z]
  depth_axis: x
```

`[x,y]` の top view では高さ方向 CD が定義しにくいため、`cd` を外す運用が分かりやすいです。

## ViewFeature の内部契約

metric は raw file ではなく `ViewFeature` を受け取ります。

| field | 内容 |
|---|---|
| `mask` | 2D view 上の non-void 領域 |
| `label2d` | 投影された material label |
| `contours` | contour 距離 metric が使う boundary 点 |
| `sdf_nm` | 2D signed distance map |
| `boundary_mask` | material boundary の 2D mask |
| `material_masks` | material ごとの 2D projected mask |
| `axes` | YAML で指定された比較軸 |

## 注意点

label volume 同士を比較する場合、simulation と target は同じ projected grid
である必要があります。`shape`, `spacing`, `origin`, `axes` が違う場合は、
暗黙に resample せずエラーにします。

異なる grid を比較したい場合は、比較前に同じ grid へそろえる処理を追加してください。
