# スコアリング

`compare` と `batch-compare` は、simulation と target を同じ 2D view にそろえてから
metric を計算します。

target は `contour_json` または label volume (`npz_label` / `vti_label`) を使えます。
label volume target の場合は material-aware な評価になります。

## metric の分類

`wafergeo` には複数の metric がありますが、最初から全部を score に入れる必要はありません。
通常は Primary metric から始め、原因分析が必要になったら Diagnostic metric を足します。

| 分類 | metric | loss の意味 | 主な用途 |
|---|---|---|---|
| Primary | `cd` | 断面 CD の幅・edge 位置誤差 nm | 高さごとの CD 評価 |
| Primary | `sdf` | 2D SDF 差分 nm | 全体的な形状差 |
| Primary | `iou` | overlap の不一致。loss は `1 - iou` | 重なり、一致率 |
| Diagnostic | `chamfer` | contour / boundary 点の近傍距離 nm | 境界形状のずれ |
| Diagnostic | `sdf_material` | material ごとの SDF 差分 nm | どの媒質が差分に効くか |
| Diagnostic | `sdf_band` | boundary 近傍 10 nm の SDF 差分 nm | edge / interface 近傍の差 |
| Optional Diagnostic | `profile` | 高さごとの幅・中心・edge 差 nm | CD の内訳確認 |
| Optional Diagnostic | `corner` | bottom corner 位置差 nm | 局所形状の sanity check |
| Optional Diagnostic | `topology` | 2D component 数差 | 分断、接続、bridge / pinch-off の sanity check |

推奨の最初の設定:

```yaml
metrics:
  use: [cd, sdf, iou]
```

原因分析をしたい場合:

```yaml
metrics:
  use: [cd, sdf, iou, chamfer, sdf_material, sdf_band]
```

## 最短の診断手順

通常は `cd`, `sdf`, `iou` だけを見ます。
原因が分からない時だけ、次のように1つずつ追加します。

| 症状 | 追加するもの | 見る出力 |
|---|---|---|
| `sdf` が悪い | `sdf_material` | `per_material_sdf.csv` |
| edge 近傍だけ確認したい | `sdf_band` | `metrics.csv`, `metric_details.json` |
| 境界点群のずれを見たい | `chamfer` | `metrics.csv` |
| CD の高さ方向の内訳を見たい | `profile` | `profile.csv` |
| bottom corner だけ確認したい | `corner` | `corner_summary.json` |
| 分断や接続の有無だけ確認したい | `topology` | `metric_details.json` |

診断 metric は「全部を常に入れる」ものではありません。
score の主軸を分かりやすく保つため、必要な症状に合わせて追加してください。

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
  use: [cd, sdf, iou]
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
| `profile` | `10 nm` |
| `corner` | `10 nm` |
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
difference.png
difference_legend.json
difference_summary.json
simulation_label_summary.json
target_label_summary.json
cd_profile.csv
cd_profile.png
cd_profile_summary.json
```

`sdf_material` を指定した場合は `per_material_sdf.csv`、`profile` を指定した場合は
`profile.csv` / `profile_summary.json`、`corner` を指定した場合は `corner_summary.json` も出ます。

PNG は確認用です。後段解析では CSV/JSON を使ってください。

## batch-compare の主な出力

```text
ranking.csv
ranking_top.png
metrics.csv
metric_summary.csv
score_summary.json
difference_summary.csv
cases/
shared_targets/
```

`sdf_material` を指定した batch では `per_material_sdf.csv` が root に集約されます。

`shared_targets/` は、同じ label-volume target を複数 case で使う場合に重複出力を避けるための場所です。

## 拡張ルール

- 新しい metric は `wafergeo.compare.metric_defs` に登録する。
- metric は `value` と ranking 用の `loss` を返す。
- 必要な feature は metric registry に宣言する。
- metric 固有の表や軽量 PNG は `output_artifacts.py` に置く。
- 同一形状が良く、ずらした形状が悪くなる test を追加する。

## 任意 metric: `profile`

`profile` は、断面 view に `z` が含まれる場合だけ使う軽量な診断 metric です。半導体断面でよく見る「高さごとの幅」「中心位置」「左右エッジ位置」を `profile.csv` と `profile_summary.json` に出します。

```yaml
features:
  use: [contour]

metrics:
  use: [profile]
```

通常の ranking では `cd`, `sdf`, `iou` から始め、断面プロファイルを確認したい時だけ `profile` を明示してください。`cd` と `profile` は近い情報を持つため、両方を score に入れる場合は二重評価になっていないか確認してください。

`profile` は `view.axes: [x, z]` または `[y, z]` のような断面比較で有効です。`[x, y]` の top view では高さ軸が無いため `SKIPPED` になります。

## open contour の距離評価

`contour_json` の contour に `closed: false` が含まれる場合、その target は面ではなく線として扱います。無理に polygon mask に変換せず、SDF 系 metric は open polyline からの unsigned distance として計算します。

```json
{
  "schema_version": "contour/v1",
  "units": "nm",
  "contours": [
    {
      "id": "edge",
      "closed": false,
      "points": [[0.0, 0.0, 0.0], [100.0, 20.0, 0.0]]
    }
  ]
}
```

open contour では面積 overlap が定義できないため、`iou` は `SKIPPED` になります。距離比較には `chamfer`, `sdf`, `sdf_band` を使ってください。`metric_details.json` には `distance_semantics: unsigned` が出ます。

## material confusion

label volume 同士を比較した場合は、score とは別に `material_confusion.csv` と `material_confusion_summary.json` が出ます。これは metric ではなく診断用の出力です。

`material_confusion.csv` では、simulation の material id と target の material id の組み合わせごとに pixel 数を出します。material id の取り違え、入れ替わり、局所的な誤分類を確認するために使います。

`material_confusion_summary.json` には、全 pixel 数、一致 pixel 数、不一致 pixel 数、最も大きい material の混同ペアが入ります。`batch-compare` では各 case の `material_confusion.csv` に加えて、root の `material_confusion.csv` に `case_id` 付きで集約されます。

## 任意 metric: `corner`

`corner` は断面 view の bottom corner 位置差を見る最小 metric です。`view.axes` に `z` が含まれる場合だけ有効で、non-void 領域の最小 z 側にある left / right 端点を bottom-left / bottom-right corner として扱います。

```yaml
view:
  axes: [x, z]
  depth_axis: y

features:
  use: [contour]

metrics:
  use: [corner]
```

出力は `corner_summary.json` です。`corner` は default metric ではありません。corner radius, curvature, wall angle はまだ扱わず、まずは corner の位置差だけを診断します。

## 任意 metric: `topology`

`topology` は projected 2D view の connected component 数差を見る最小 metric です。
材料が分断された、bridge した、pinch-off した、といった大きな位相差の sanity check に使います。

```yaml
metrics:
  use: [topology]
```

初期実装は 4-neighbor の component count のみです。3D topology、穴数、persistent homology は扱いません。
詳細は `metric_details.json` に `mode: projected_2d_component_count` として出ます。
値が 0 の場合は、少なくとも projected 2D view 上の component 数は一致しています。
