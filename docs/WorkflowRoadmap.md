# 特徴量化・評価 workflow ロードマップ

この文書は、`wafergeo` の今後の拡張計画を、ユーザー、3D 構造評価の専門家、
運用エンジニア、手法追加を行う開発者が同じ前提で参照できるように整理したものです。

## 1. 絶対指針

`wafergeo` の目的は、入口と出口を固定したうえで、複数の特徴量化手法を計算し、
実測データとの比較やサロゲート学習用 dataset 作成に使える基盤を作ることです。

今後の改良では、次を守ります。

| 原則 | 内容 |
| --- | --- |
| 入口を固定する | workflow は用途ごとに限定し、場当たり的に増やさない |
| 出口を固定する | CSV/JSON/NPZ を主出力にし、後段処理で使いやすくする |
| transform と compare を分ける | 3D field 特徴量化は transform 系、2D observation 比較は compare 系 |
| 手法追加を局所化する | loader / feature / metric / output のどこを触るか明確にする |
| 評価できる形で増やす | 新手法は runtime、出力サイズ、数値品質、ranking 性能を確認できるようにする |
| サロゲート学習器は入れない | 本パッケージは学習器ではなく、学習に渡せる特徴量 dataset を作る |

## 2. 現行と将来の入口

現時点の正式 workflow は次の 3 つです。

| workflow | 現在の役割 |
| --- | --- |
| `transform` | 単一データを特徴量化する |
| `compare` | 1 ペアを 2D observation view 上で比較する |
| `batch-compare` | 複数ペアを比較し ranking を出す |

将来的には、用途を明確にしたうえで次の 6 workflow へ拡張する計画です。

| 系統 | workflow | 役割 |
| --- | --- | --- |
| 特徴量化 | `transform` | 単一データを固定手法で特徴量化する |
| 特徴量化 | `batch-transform` | 複数データを固定手法で特徴量化し、学習 dataset を作る |
| 特徴量化 | `transform-eval` | 複数の特徴量化手法の生成品質と計算コストを評価する |
| 比較 | `compare` | 1 ペアを固定 feature / metric で比較する |
| 比較 | `batch-compare` | 複数ペアを固定 feature / metric で比較し ranking する |
| 比較 | `compare-eval` | 複数 feature / metric 候補の比較性能を評価する |

この 6 つを超える public workflow は、原則として追加しません。

## 3. 2D observation 評価と 3D field 評価

今後の設計では、評価対象を明確に分けます。

| 評価対象 | 主な workflow | 目的 |
| --- | --- | --- |
| 3D field 評価 | `transform`, `batch-transform`, `transform-eval` | SDF/UDF/TSDF/material SDF などが特徴量として妥当か確認する |
| 2D observation 評価 | `compare`, `batch-compare`, `compare-eval` | SEM 輪郭、断面、topview など観察 view 上で比較する |

この分離により、次の混乱を避けます。

- `transform` で ranking まで行わない。
- `compare` でサロゲート学習 dataset を作らない。
- `transform-eval` で target 比較を主目的にしない。
- `compare-eval` で 3D feature の品質統計を深追いしない。

## 4. 3D 構造評価の専門家視点での課題と対策

| 課題 | 問題 | 対策 |
| --- | --- | --- |
| 2D view だけでは 3D 差分を見落とす | 奥行方向の欠陥、buried material、局所 void が潰れる | 3D field feature を `transform` の主力出力にする |
| projection の意味が曖昧になりやすい | topview と断面で比較対象が変わる | `view.axes`, `depth_axis`, `projection` を出力 summary に残す |
| material 差分が外形 metric に埋もれる | CD/Chamfer だけでは内部構造を拾いにくい | `material_sdf` と material coverage を transform 系で評価する |
| open contour を signed SDF に無理に変換する | inside/outside が曖昧になり、意味が壊れる | contour はまず UDF として扱う |
| spacing / units の不一致 | nm 単位の比較結果が信用できなくなる | `spacing_xyz`, `units`, `axis_order` を feature summary に必ず残す |

## 5. 運用エンジニア視点での課題と対策

| 課題 | 問題 | 対策 |
| --- | --- | --- |
| workflow が増えたように見える | ユーザーが入口を迷う | 6 workflow を「特徴量化系」と「比較系」に分けて説明する |
| eval 出力が膨らむ | candidate × case × metric でファイルが増える | eval 系の正式出力は CSV/JSON 中心にする |
| 大量データで重い | runtime / output size を見ずに手法採用してしまう | `candidate_summary.csv` に runtime と出力サイズを必ず出す |
| memory 測定が複雑 | 正確な peak memory 計測で実装が重くなる | 初期は runtime と output size を採用し、memory は後段 optional |
| cache が複雑化する | 再利用状態のバグが増える | 初期実装では cache しない。必要になってから局所的に追加する |

## 6. 手法追加エンジニア視点での課題と対策

| 課題 | 問題 | 対策 |
| --- | --- | --- |
| feature 追加場所が分かりにくい | runner や output に処理が散る | feature registry を整備し、runner は呼び出しだけにする |
| metric と feature の責務が混ざる | 新手法追加時に影響範囲が広がる | feature は transform 系、metric は compare 系に閉じる |
| 新手法の採用判断が属人的 | 良い手法かどうか比較できない | transform-eval / compare-eval の summary で判断材料を出す |
| テストが増えすぎる | CI と読解コストが重くなる | synthetic test を 2-4 件に抑え、重い dataset は手動 smoke にする |
| default が膨らむ | ユーザーが通常実行で迷う | 新 feature / metric は明示指定時だけ使い、すぐ default にしない |

## 7. transform 系の仕様案

### 7.1 `transform`

単一データを固定手法で特徴量化します。

```yaml
task: transform

input:
  simulation:
    kind: vti_label
    path: data/run_0001/vox_t08.vti

features:
  use: [sdf_raw, tsdf_views]

output:
  dir: outputs/run_0001_features
```

主な出力:

```text
features/
  sdf_raw.npz
  tsdf_views.npz
feature_summary.json
_run/
  used_config.yaml
  run_info.json
```

### 7.2 `batch-transform`

複数データを固定手法で特徴量化し、サロゲート学習に渡せる dataset を作ります。

初期の index CSV は最小形にします。

```csv
case_id,input_kind,input_path
run_0000,vti_label,data/run_0000/vox_t08.vti
run_0001,vti_label,data/run_0001/vox_t08.vti
```

YAML:

```yaml
task: batch-transform

input:
  index: data/sim_cases.csv

features:
  use: [sdf_raw, tsdf_views]

output:
  dir: outputs/training_dataset
```

主な出力:

```text
dataset_index.csv
features_summary.csv
cases/
  run_0000/
    features/
      sdf_raw.npz
      tsdf_views.npz
  run_0001/
    features/
      sdf_raw.npz
      tsdf_views.npz
```

### 7.3 `transform-eval`

複数の特徴量化手法を同じ入力群で計算し、生成品質と運用コストを評価します。
target との比較性能は主目的にしません。

```yaml
task: transform-eval

input:
  index: data/sim_cases.csv

eval:
  candidates:
    raw:
      features:
        use: [sdf_raw]

    tsdf:
      features:
        use: [tsdf_views]

    material:
      features:
        use: [material_sdf]

output:
  dir: outputs/transform_eval
```

正式出力:

| 出力 | 内容 |
| --- | --- |
| `candidate_summary.csv` | candidate ごとの成功数、runtime、出力サイズ |
| `feature_stats.csv` | feature ごとの shape、dtype、min/max/mean/std、NaN/inf |
| `case_summary.csv` | case ごとの成功/失敗、runtime、出力サイズ |

## 8. compare 系の仕様案

### 8.1 `compare`

1 ペアを 2D observation view 上で比較します。

```yaml
task: compare

input:
  simulation:
    kind: vti_label
    path: data/run_0001/vox_t08.vti
  target:
    kind: vti_label
    path: data/run_0010/vox_t08.vti

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf, contour]

metrics:
  use: [cd, sdf, iou]

output:
  dir: outputs/compare_run0001
```

### 8.2 `batch-compare`

固定 feature / metric で複数ペアを比較し、ranking を出します。

### 8.3 `compare-eval`

複数 feature / metric 候補を同じ入力群で比較し、どの組み合わせが有用か判断する材料を出します。

```yaml
task: compare-eval

input:
  index: data/compare_pairs.csv

eval:
  candidates:
    baseline:
      features:
        use: [sdf, contour]
      metrics:
        use: [cd, sdf, iou]

    material:
      features:
        use: [material_sdf]
      metrics:
        use: [sdf_material, iou]

output:
  dir: outputs/compare_eval
```

正式出力:

| 出力 | 内容 |
| --- | --- |
| `candidate_summary.csv` | candidate ごとの score、rank 指標、runtime |
| `case_scores.csv` | case × candidate の score と metric 値 |
| `metric_summary.csv` | metric ごとの平均、中央値、分散、skip 数 |
| `candidate_rankings.csv` | candidate ごとの ranking 結果 |

## 9. 評価指標

### 9.1 transform-eval

| 指標 | 意味 |
| --- | --- |
| `success_count` / `failed_count` | 実データで安定して特徴量化できるか |
| `mean_runtime_sec` | 大量データに適用できる速度か |
| `total_size_mb` / `mean_size_mb` | 学習 dataset として重すぎないか |
| `shape_xyz` | 学習モデル入力として扱える形か |
| `dtype` | float32/float16 など |
| `nan_count` / `inf_count` | 数値異常がないか |
| `min/max/mean/std` | 特徴量のスケール |
| `material_coverage` | material ごとの voxel 数や比率 |

### 9.2 compare-eval

| 指標 | 意味 |
| --- | --- |
| `mean_total_score` | 平均的に良い score か |
| `median_total_score` | 外れ値に左右されにくい傾向 |
| `self_rank1_rate` | 自己比較が 1 位になる率 |
| `known_best_rank1_rate` | 正解 case がある場合に 1 位になる率 |
| `top3_hit_rate` | 候補絞り込みに使えるか |
| `skipped_metric_count` | 入力と metric の不整合がないか |
| `mean_runtime_sec` | 運用可能な速度か |

## 10. 特徴量化手法ロードマップ

次の順で実装します。

| 優先度 | feature | 初期定義 | 目的 |
| --- | --- | --- | --- |
| P1 | `sdf_raw` | non-void union の signed SDF | 距離場特徴量の基準 |
| P1 | `tsdf_views` | 固定 clip 幅の TSDF 派生 view | 学習しやすい multi-scale 表現 |
| P1 | `udf` | contour_json から unsigned distance | 実測 contour / open contour 向け |
| P2 | `material_sdf` | material id ごとの signed SDF stack | material-aware な 3D 特徴量 |
| P3 | `interface_sdf` | material interface までの距離 | 内部境界評価 |

初期判断として、次を採用します。

- `sdf_raw` は non-void union の signed SDF から開始する。
- `tsdf_views` の clip 幅はまず固定値にする。
- `batch-transform` の index CSV は `case_id,input_kind,input_path` の最小形から開始する。

## 11. 可視化方針

可視化は最初から増やしません。CSV/JSON を正式出力にし、PNG は補助に限定します。

| workflow | 初期可視化 | 後段候補 |
| --- | --- | --- |
| `transform` | なし | 代表 slice preview |
| `batch-transform` | なし | feature size bar |
| `transform-eval` | なし | runtime / size bar |
| `compare` | `difference.png` | metric overlay |
| `batch-compare` | optional differences | ranking plot |
| `compare-eval` | なし | score box plot, rank heatmap |

## 12. 残る設計論点

実装前に議論が必要なものは次です。

| 論点 | 推奨初期判断 |
| --- | --- |
| `tsdf_views` の clip 幅 | 固定値から開始。候補は実データ spacing を見て決める |
| `udf` の grid 解像度 | contour_json の units と比較対象 view に合わせる |
| `compare-eval` の正解情報 | self comparison と optional `expected_best_case_id` から開始 |
| eval 系の plots | 初期実装では出さない |
| memory 計測 | 初期実装では入れない |
| feature cache | 初期実装では入れない |

## 13. 実装順

次の順で進めると、手戻りが少なくなります。

1. `sdf_raw` feature と `feature_summary.json`
2. `tsdf_views` feature
3. `udf` feature
4. `material_sdf` feature
5. `batch-transform`
6. `transform-eval`
7. `compare-eval`
8. 最小可視化

`compare-eval` を先に作ると、比較対象となる feature が少なく、summary だけが増えます。
そのため、まず transform 系の 3D field feature を強くします。
