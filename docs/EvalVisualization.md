# Eval Visualization Design

このページは `transform-eval` と `compare-eval` に追加する診断用 PNG の設計メモです。
CSV/JSON/NPZ を正本とし、PNG は候補を選ぶ判断を速くするための補助出力として扱います。

## 目的

eval 可視化では次の 3 点を分けて確認します。

| 確認したいこと | 対象 workflow | 主な図 |
| --- | --- | --- |
| 特徴量化手法が何を表しているか | `transform-eval` | method overview, representative feature slices |
| 特徴量が元データをどの程度表現できているか | `transform-eval` | representation score, feature-input alignment |
| metric set が比較目的に合うか | `compare-eval` | objective heatmap, rank delta, metric contribution |

## 出力方針

- 出力先は各 eval output の `figures/` 配下に限定します。
- 図を YAML の新しい workflow や深い設定にしません。
- まずは既存 CSV/JSON から作れる summary 図を優先します。
- case ごとの重い断面図は、代表 case だけに絞ります。
- PNG の判断根拠になる数値は CSV/JSON にも残します。

## 特徴量化手法ごとの見せ方

各 feature method について、「入力のどの構造を見るか」「図では何を見ればよいか」を明示します。

| feature | 表す内容 | 推奨する図 | 読み方 |
| --- | --- | --- | --- |
| `sdf` | 2D view 上の signed distance | boundary + SDF slice | 0 近傍が境界に沿い、内外の距離勾配が滑らかかを見る |
| `sdf_raw` | 3D label volume 全体の signed distance | orthogonal mid slices, min-abs slices | 元形状の穴、溝、膜境界が 3D field に残っているかを見る |
| `tsdf_views` | clip された複数 2D TSDF view | channel grid | clip 幅内で境界近傍の差分だけを強調できているかを見る |
| `udf` | unsigned distance | min/max slice, distance histogram | 内外符号を使わず境界からの距離分布だけを見たい時に使う |
| `material_sdf` | material ごとの SDF channel | material channel grid | material ごとの形状差や欠損が分離されているかを見る |
| `material_profile` | material 量、厚み、断面統計 | stacked bars, profile lines | 形状 field ではなく集計値として case 差分を表せるかを見る |
| `process_delta_profile` | initial/final 間の etch/deposit/change 集計 | transition bars | 加工差分の量と material transition が case 差として出るかを見る |
| `process_delta_sdf` | 加工差分領域の distance field | change-region slices | 差分の場所と周辺距離が局所 feature として妥当かを見る |
| `mesh` | 3D surface | surface stats, optional rendered snapshot | 面数、体積、境界の粗さが case 差を反映するかを見る |
| `contour`, `slice` | 2D geometry extraction | contour overlay, label slice | compare 前の 2D view が意図した断面を表しているかを見る |

## transform-eval の図

### 0. Input Shape Sections

`figures/input_shape_sections.png`

元 label volume の形状を case ごとに `[x,z]` と `[y,z]` の 2 断面で並べます。

| 列 | 内容 |
| --- | --- |
| `[x,z] mid-y` | y 中央位置で切った x-z 断面 |
| `[y,z] mid-x` | x 中央位置で切った y-z 断面 |

この図は feature を見る前に、入力形状そのものの奥行き方向や高さ方向の違いを確認するために使います。

### 1. Method Overview

`figures/feature_method_overview.png`

feature method ごとに、代表 case の入力断面と feature 出力を 1 行で並べます。

| 列 | 内容 |
| --- | --- |
| input slice | 元 label の代表断面 |
| feature slice | SDF/channel/profile など feature の代表表示 |
| feature summary | shape, channel count, material count, units |
| signal note | varies / constant / sparse などの短い診断 |

この図は「この candidate は何を作っているのか」を最初に読むための一覧です。

### 2. Candidate Signal Heatmap

`figures/candidate_signal_heatmap.png`

既存の `candidate_eval_summary.csv`, `case_variation_summary.csv`,
`scalar_variation_summary.csv`, `profile_variation_summary.csv` から作ります。

| 軸 | 内容 |
| --- | --- |
| row | candidate |
| column | feature output, scalar, profile key |
| color | constant / varies / high variation |

この図では、case 間で反応する feature output がどこにあるかを見ます。
候補選定では、信号が多いこと自体よりも、目的に合う feature が変動しているかを重視します。

### 3. Representation Score

`figures/feature_representation_score.png`

特徴量が元データをどの程度表現できているかを、feature method ごとに別グラフで示します。
これは downstream score ではなく、入力 label に対する自己診断です。

| score | 対象 feature | 意味 |
| --- | --- | --- |
| occupancy_coverage | SDF, UDF, material SDF | feature から復元した non-void 領域が元 label の non-void 領域をどれだけ覆うか |
| boundary_alignment | SDF, TSDF, material SDF | feature の 0 近傍または最小距離帯が元 label 境界とどれだけ重なるか |
| material_separation | material SDF, material profile | material ごとの領域や量が元 label の material 分布を保っているか |
| variation_capture | all eval candidates | case 間の入力差分に対して feature 統計がどれだけ変化するか |
| compactness | profile, scalar features | 少ない出力サイズでどれだけ case 差分を説明できるか |

推奨表示は grouped bar です。

| x 軸 | group | y 軸 |
| --- | --- | --- |
| feature method | score type | 0.0 から 1.0 の normalized score |

`sdf_raw` や `material_sdf` は representation が高くなりやすく、
`material_profile` は compactness が高くなりやすい、というように特徴の違いを読めるようにします。

### 4. Feature-Input Alignment

`figures/feature_input_alignment.png`

代表 case だけ、元 label と feature-derived mask を重ねます。

| 色 | 意味 |
| --- | --- |
| neutral | 元 label と feature-derived mask が一致 |
| red | 元 label にあるが feature が落とした領域 |
| green | feature が追加的に表した領域 |
| blue | material または channel の取り違え |

この図は representation score の理由を形状として確認するためのものです。
全 case 全 candidate に出すと重くなるため、score が低い case、または variation が最大の case に絞ります。

### 5. Signal vs Cost

`figures/candidate_signal_cost.png`

| 軸 | 内容 |
| --- | --- |
| x | mean_size_mb または mean_runtime_sec |
| y | varying_output_count または representation score |
| point | candidate |

重いが表現力が伸びない candidate、軽いが目的に十分な candidate を見分けます。

## compare-eval の図

### 1. Objective Heatmap

`figures/objective_heatmap.png`

`case_scores.csv` から作ります。

| 軸 | 内容 |
| --- | --- |
| row | metric candidate |
| column | case_id |
| color | normalized_total_score |

metric set がどの case を厳しく評価するかを見ます。
低いほど良い score なので、色の legend には direction を明記します。

### 2. Rank Delta Heatmap

`figures/rank_delta_heatmap.png`

`ranking_consistency.csv` から作ります。

| 軸 | 内容 |
| --- | --- |
| row | metric candidate |
| column | case_id |
| color | baseline からの rank_delta |

metric set を変えた時に順位がどれだけ揺れるかを見ます。
rank が安定していれば optimizer/sampler の選別に使いやすく、rank が大きく変わる場合は metric の意図を確認します。

### 3. Metric Contribution

`figures/metric_contribution_heatmap.png`

`metric_summary.csv` から作ります。

| 軸 | 内容 |
| --- | --- |
| row | metric candidate |
| column | metric name |
| color | mean_normalized_loss |

objective の差が `cd`, `sdf`, `iou`, `sdf_material`, `profile` など、どの metric に由来するかを見ます。
metric set の説明責任を持たせるため、candidate summary だけでなくこの図を併用します。

### 4. Metric Evaluation Score

`figures/metric_evaluation_score.png`

metric set 自体を評価するための bar chart です。

| score | 意味 |
| --- | --- |
| case_coverage | OK で評価できた case の比率 |
| metric_coverage | SKIPPED にならなかった metric の比率 |
| ranking_stability | baseline との rank delta が小さいほど高い |
| objective_spread | case 差を score として分離できている度合い |
| runtime_efficiency | 実行時間に対する評価可能 case 数 |

この図は「metric が評価できるか」を見るための入口です。
良い metric set は、case_coverage と metric_coverage が高く、目的に応じて ranking_stability または objective_spread が十分にあります。

### 5. Representative Difference Panels

`figures/representative_differences/`

既存 `figs` の triptych 型を踏襲し、代表 case だけを出します。

| panel | 内容 |
| --- | --- |
| target | target contour or label boundary |
| simulation | simulation contour or label boundary |
| overlay + diff | overlap, simulation-only, target-only, major metric values |

代表 case は次の優先順で選びます。

1. baseline で最も良い case
2. baseline で最も悪い case
3. rank_delta が最大の case
4. material/profile metric が大きく効いた case

## 読み方の標準手順

1. `feature_method_overview.png` で、各 feature candidate が何を出しているかを見る。
2. `feature_representation_score.png` で、元データ表現力を feature method ごとに見る。
3. `candidate_signal_heatmap.png` で、case 差分に反応する output を確認する。
4. `candidate_signal_cost.png` で、表現力と出力コストの釣り合いを見る。
5. `objective_heatmap.png` で、metric set が case をどう評価するかを見る。
6. `rank_delta_heatmap.png` で、metric set の変更が順位を揺らすかを見る。
7. `metric_contribution_heatmap.png` と `metric_evaluation_score.png` で、metric set が評価可能で説明可能かを見る。
8. 代表差分パネルで、score の理由を形状として確認する。

## 最小実装セット

最初に実装するなら、以下の 5 枚を優先します。

| workflow | figure | 理由 |
| --- | --- | --- |
| `transform-eval` | `feature_representation_score.png` | 元データをどの程度表現できているかを直接見る |
| `transform-eval` | `candidate_signal_heatmap.png` | candidate ごとの case 差分への反応を見る |
| `compare-eval` | `objective_heatmap.png` | metric set ごとの評価結果を俯瞰する |
| `compare-eval` | `rank_delta_heatmap.png` | metric set 変更による順位変動を見る |
| `compare-eval` | `metric_evaluation_score.png` | metric set が評価可能かをまとめて見る |

その後に `feature_method_overview.png`, `feature_input_alignment.png`,
`metric_contribution_heatmap.png`, representative difference panels を追加します。

## 実装メモ

- `wafergeo/compare/eval_figures.py` の小さな writer に分離します。
- runner は既存 row と path を渡すだけにし、図の score 計算や描画は writer 側で扱います。
- Matplotlib は optional `viz` extra のまま、関数内 import と `Agg` backend で使います。
- Matplotlib が使えない場合は `figures/figure_manifest.json` に `SKIPPED` と理由を残し、workflow は成功させます。
- 図の有無で workflow の成否を変えません。CSV/JSON/NPZ が正本です。

## 再実行時の注意

- eval workflow は再実行時に自分が管理する `candidates/` と `figures/` を作り直します。
- `transform-eval/candidates/*/cases/*/preview.png` は feature 可視化ではなく、入力 label view の確認画像です。
- feature の違いは `figures/representative_feature_slices/` と `features/` 配下の NPZ/CSV を見ます。
- 図の読み方は `figures/figure_manifest.json` の `how_to_read` にも出力されます。

## グラフの読み方

| figure | 見ること |
| --- | --- |
| `input_shape_sections.png` | 元 label volume の `[x,z]` と `[y,z]` 断面を case ごとに見ます。 |
| `representative_feature_slices/*.png` | 代表 feature ごとに診断内容を変えます。`material_sdf` は SDF channel を `sdf<=0` で label に復元し、元 label との view/断面/IoU を見ます。`material_profile` は CSV の material fraction と bbox z range が元 label の集計と合うかを見ます。`process_delta_profile` は reference から final への changed mask と transition 量を見ます。 |
| `feature_representation_score.png` | 高いほど元 geometry/material/profile を保っています。空欄 score は平均から除外します。 |
| `candidate_signal_heatmap.png` | 明るいセルほど case 間で変動した output です。目的に合う feature が変動しているかを見ます。 |
| `candidate_signal_cost.png` | 上にあるほど変動 output が多く、右にあるほど出力サイズが大きいです。 |
| `objective_heatmap.png` | 低い score ほど良い評価です。候補 metric set が case をどう厳しく見るかを確認します。 |
| `rank_delta_heatmap.png` | baseline candidate から順位がどれだけ変わったかを見ます。0 に近いほど順位は安定です。 |
| `metric_contribution_heatmap.png` | どの metric が loss に効いているかを見ます。明るい metric は寄与が大きいです。 |
| `metric_evaluation_score.png` | coverage、ranking stability、objective spread をまとめた metric set 診断です。 |
| `representative_differences/*.png` | 代表 case ごとに、compare view に加えて simulation と label target の `[x,z]` / `[y,z]` 断面を見ます。 |
