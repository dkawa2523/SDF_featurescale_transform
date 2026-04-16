# ビュー

`view` は、compare metrics と形状確認画像で使う 2D 観察面を選びます。

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

| key | 意味 |
| --- | --- |
| `axes` | 2D view に表示する 2 軸 |
| `depth_axis` | 投影方向として使う残りの軸 |

`axes` と `depth_axis` は、`x`, `y`, `z` をちょうど 1 回ずつ使う必要があります。

label volume では、depth axis に沿って最初に見える non-void material を投影します。
material が見えない pixel は void のままです。

material-aware metrics は、外形 boundary と内部 material boundary の両方を使う場合があります。

height-wise CD を比較する場合は `[x,z]` または `[y,z]` を使います。
top-view の `[x,y]` は projected shape overlap の確認には有用ですが、
height-wise CD より SDF shape-distance が良いかどうかを判断する view ではありません。
