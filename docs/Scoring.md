# スコアリング

`compare` と `batch-compare` は、simulation と target を同じ 2D view にそろえてから
metric を計算します。

target は `contour_json` または label volume (`npz_label` / `vti_label`) を使えます。
label volume target の場合は material-aware な評価になります。

## 公開 metric

| metric | loss の意味 | 主な用途 |
|---|---|---|
| `cd` | 断面 CD の幅・edge 位置誤差 nm | 高さごとの CD 評価 |
| `chamfer` | contour / boundary 点の近傍距離 nm | 境界形状のずれ |
| `sdf` | 2D SDF 差分 nm | 全体的な形状差 |
| `sdf_material` | material ごとの SDF 差分 nm | どの媒質が差分に効くか |
| `sdf_band` | boundary 近傍 10 nm の SDF 差分 nm | edge / interface 近傍の差 |
| `iou` | overlap の不一致。loss は `1 - iou` | 重なり、一致率 |

推奨の最初の設定:

```yaml
metrics:
  use: [cd, chamfer, sdf, sdf_material, sdf_band, iou]
```

## CD

CD は、半導体断面で高さごとの幅や edge 位置を見る metric です。
通常は `view.axes` に `z` を含めます。

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

デフォルトでは、material boundary transition と中心線付近の幅 profile を使います。
外形幅が同じでも、内部 material boundary がずれた場合に検出しやすくしています。

測定対象 material や gauge を明示したい場合だけ、次を追加します。

```yaml
metrics:
  use: [cd, chamfer, sdf, sdf_material, sdf_band, iou]
  cd:
    material_ids: [2]
    gauge:
      axis: x
      height_axis: z
      center: 4.0
      height_range: [20.0, 120.0]
```

`cd_profile.csv` と `cd_profile.png` に高さごとの結果が出ます。

## SDF 系 metric

### `sdf`

2D view の SDF 差分を評価します。
label volume 同士の比較では material label field と material boundary field を見ます。

### `sdf_material`

non-void material を自動検出し、material ごとの SDF loss を出します。
material id を YAML に列挙する必要はありません。

総合 loss は projected union area による重み付き平均です。
詳細は `metric_details.json` と `per_material_sdf.csv` に出ます。

### `sdf_band`

boundary 近傍だけを評価する SDF loss です。
デフォルト band は 10 nm です。
遠い背景領域より、edge / interface の位置ずれを重視したい場合に使います。

## IoU

`iou` は overlap を評価します。
`value` は大きいほど良く、ranking 用の `loss` は `1 - iou` です。

label volume target では、material label IoU と material boundary IoU のうち、
より悪い方を採用します。

## score

`score.json` には次の 2 種類の score が出ます。

| field | 内容 |
|---|---|
| `total_score` | raw loss の重み付き合計 |
| `normalized_total_score` | metric ごとの scale で正規化した ranking 用 score |

ranking は `normalized_total_score` の昇順です。小さいほど target に近いです。

metric scale:

| metric | scale |
|---|---|
| `cd` | `10 nm` |
| `chamfer` | `10 nm` |
| `sdf` | `10 nm` |
| `sdf_material` | `10 nm` |
| `sdf_band` | `10 nm` |
| `iou` | `1.0` |

指定した metric が評価できない場合、`score.json` に `skipped_metrics` が出ます。
`metrics.csv` でも `status=SKIPPED` として確認できます。

## compare の主な出力

```text
score.json
metrics.csv
metric_details.json
per_material_sdf.csv
difference.png
difference_legend.json
difference_summary.json
simulation_label_summary.json
target_label_summary.json
cd_profile.csv
cd_profile.png
cd_profile_summary.json
```

PNG は確認用です。後段解析では CSV/JSON を使ってください。

## batch-compare の主な出力

```text
ranking.csv
ranking_top.png
metrics.csv
metric_summary.csv
per_material_sdf.csv
score_summary.json
difference_summary.csv
cases/
shared_targets/
```

`shared_targets/` は、同じ label-volume target を複数 case で使う場合に重複出力を避けるための場所です。

## 拡張ルール

- 新しい metric は `wafergeo.compare.metric_defs` に登録する。
- metric は `value` と ranking 用の `loss` を返す。
- 必要な feature は metric registry に宣言する。
- metric 固有の表や軽量 PNG は `output_artifacts.py` に置く。
- 同一形状が良く、ずらした形状が悪くなる test を追加する。
