# User Manual

このページは、利用者が YAML を編集して実行し、出力を読むための説明です。
コードを改修する場合は [Developer Manual](DeveloperManual.md) を参照してください。

## 入力データ

| kind | 用途 | 必須内容 |
| --- | --- | --- |
| `npz_label` | 推奨の label volume 入力 | `labels` shape `[X,Y,Z]`, `spacing`, `origin` |
| `vti_label` | VTI label volume 入力 | VTI の label array |
| `contour_json` | target 輪郭入力 | `contours[].points` と `units` |

`npz_label` はユーザー向けには `[X,Y,Z]` です。
内部では `[Z,Y,X]` に変換して処理します。

`void_id` は「媒質がない」領域です。
データにより void ID が異なるため、必要なら YAML または index CSV で明示してください。

## YAML の基本形

```yaml
task: transform

input:
  simulation:
    kind: npz_label
    path: data/examples/sim_case.npz

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf_raw, material_profile]

output:
  dir: outputs/example_transform
```

通常 YAML は次のブロックだけを使います。

| block | 役割 |
| --- | --- |
| `task` | 実行する workflow |
| `input` | simulation, target, index CSV |
| `view` | 3D 形状をどの 2D 面で見るか |
| `features` | 計算する特徴量 |
| `metrics` | 比較に使う metric |
| `output` | 出力先 |

加工前後差分を使う場合だけ、追加で `process.enabled` と
`input.reference` を使います。

```yaml
task: transform

input:
  simulation:
    kind: npz_label
    path: data/final.npz
  reference:
    kind: npz_label
    path: data/initial.npz

process:
  enabled: true

features:
  use: [process_delta_profile, process_delta_sdf]

output:
  dir: outputs/process_delta
```

## workflow の使い分け

| やりたいこと | workflow |
| --- | --- |
| 1 case を特徴量化したい | `transform` |
| 複数 case を同じ特徴量で変換したい | `batch-transform` |
| 複数の特徴量候補を比較したい | `transform-eval` |
| simulation と target を 1 対 1 で比較したい | `compare` |
| 複数 case を target に近い順に並べたい | `batch-compare` |
| 複数 metric set を比較したい | `compare-eval` |

## transform の特徴量

| feature | 出力 | 主な用途 |
| --- | --- | --- |
| `sdf` | 2D SDF | compare 用の 2D 距離場 |
| `sdf_raw` | 3D raw SDF | 全体形状の距離場表現 |
| `tsdf_views` | clip 幅違いの TSDF | 学習用の距離場候補 |
| `udf` | unsigned distance | 開いた輪郭や境界距離 |
| `material_sdf` | material ごとの SDF | material-aware な形状表現 |
| `material_profile` | CSV/JSON profile | material 量、範囲、z profile |
| `process_delta_profile` | CSV/JSON profile | 加工前後の transition 集計 |
| `process_delta_sdf` | NPZ/JSON | 加工差分領域の距離場 |
| `mesh` | mesh file | 3D surface の外部利用 |
| `contour`, `slice` | JSON/NPZ | 2D 確認や compare 補助 |

## compare の metric

| metric | 主な意味 |
| --- | --- |
| `cd` | 断面 CD。高さごとの幅や edge 位置の差 |
| `chamfer` | contour 点群間の距離 |
| `sdf` | 2D SDF 差分 |
| `sdf_band` | 境界近傍に絞った SDF 差分 |
| `sdf_material` | material ごとの SDF 差分 |
| `iou` | label / mask の重なり |
| `profile` | profile 量の差 |
| `corner` | 局所 corner 形状の差 |
| `topology` | 連結性などの大域形状差 |

詳しい使い分けは [Scoring](Scoring.md) を参照してください。

## 出力の見方

| 出力 | 意味 |
| --- | --- |
| `features/` | 特徴量本体 |
| `feature_summary.json` | どの feature を何として出したか |
| `objective.json` | 最適化や外部 sampler が読む代表 loss |
| `metrics.csv` | metric ごとの値 |
| `difference.png` | compare の補助確認画像 |
| `ranking.csv` | batch-compare の順位 |
| `candidate_summary.csv` | eval の候補別 summary |
| `_run/used_config.yaml` | 実行に使った設定の控え |
| `_run/run_info.json` | 実行日時、入力、task の記録 |

`_run/` は再現性のための補助情報です。
ユーザーが編集する入力ではありません。
