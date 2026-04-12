# ユーザーマニュアル

このパッケージは、シミュレーションや実験由来の形状データを
**特徴量化**し、必要に応じて **形状を比較評価**するためのツールです。

ユーザーが普段意識するものは次の 3 種類です。

- YAML 設定ファイル
- 入力データ
- `outputs/` に出る結果

`manifest` や `report` を手で編集する運用は行いません。

## 1. 実行コマンド

現在実装済みの実行入口は次の 3 種類です。

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
```

| task | 目的 |
|---|---|
| `transform` | 1 つの simulation label から SDF、mesh、contour、slice などの特徴量を出す |
| `compare` | simulation 1 件と target 1 件を比較する |
| `batch-compare` | 複数の simulation/target ペアを比較し、ranking を出す |

まずは example YAML をコピーして、`input.*.path` と `output.dir` だけを変えるのが安全です。

## 2. インストール

開発環境では次を使います。

```powershell
Set-Location C:\Users\user\Desktop\SDF_fs
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

VTI を読む場合は `vtk` extra も入れます。

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## 3. 入力データ

正式対応している simulation 入力は次の 2 種類です。

| kind | 内容 |
|---|---|
| `npz_label` | `labels` を持つ NumPy npz。ユーザー向け軸順は `[X,Y,Z]` |
| `vti_label` | VTI label volume |

target 入力は次の 3 種類です。

| kind | 内容 |
|---|---|
| `npz_label` | label volume target |
| `vti_label` | label volume target |
| `contour_json` | 輪郭座標 target |

`npz_label` の最小形式:

```text
labels: integer array, shape [X,Y,Z]
spacing: optional [X,Y,Z], default [1,1,1]
origin: optional [X,Y,Z], default [0,0,0]
material_ids: optional
material_names: optional
void_id: optional, material id 0 が無い場合は必須
```

内部では `[Z,Y,X]` に変換されます。ユーザーは npz では `[X,Y,Z]` と考えてください。

## 4. view の考え方

`view` は 3D volume をどの 2D 面で比較するかを決めます。

```yaml
view:
  axes: [x, z]
  depth_axis: y
```

この例では、`x-z` 断面を比較し、`y` 方向に投影します。

CD を評価したい場合は、通常 `[x,z]` または `[y,z]` のように `z` を含む view を使います。
Top view の `[x,y]` では高さ方向 CD が定義しにくいため、`cd` を外して
`sdf`, `iou` から始めるのが分かりやすいです。境界点のずれを詳しく見たい場合だけ
`chamfer` を追加します。

## 5. transform の YAML

特徴量化だけを行う最小例です。

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
  use: [sdf, contour, slice]

output:
  dir: outputs/transform_case001
```

主な出力:

| 出力 | 内容 |
|---|---|
| `features/simulation_sdf.npz` | 2D view の SDF |
| `features/sdf_raw.npz` | `sdf_raw` を指定した場合の raw 3D signed distance |
| `features/simulation_contours.json` | 投影後の contour / material boundary |
| `features/simulation_slice.npy` | 投影後の 2D label |
| `features/mesh.npz` | mesh を指定した場合の mesh feature |
| `feature_summary.json` | transform feature の shape、単位、統計 |
| `preview.png` | 入力 view の簡易確認画像 |
| `label_summary.json` | 入力 volume と view の要約 |

`sdf` は軽量な 2D view SDF です。3D field feature が必要な場合は、まず `sdf_raw` を指定します。
`sdf_raw` は non-void union の raw signed distance で、inside は負、outside は正です。

外部解析や学習用に SDF 派生特徴量をまとめて出したい場合は、任意 feature として `sdf_views` を指定します。

```yaml
features:
  use: [sdf_views]
```

`features/sdf_views.npz` には `sdf_nm`, `tsdf_10nm`, `tsdf_50nm`, `log_abs_sdf`, `mask`, `spacing`, `origin` が入ります。これは `transform` 専用で、`compare` の score には影響しません。

## 6. compare の YAML

simulation 1 件と target 1 件を比較する例です。

```yaml
task: compare

input:
  simulation:
    kind: npz_label
    path: data/examples/sim_case.npz

  target:
    kind: npz_label
    path: data/examples/target_case.npz

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf, contour]

metrics:
  use: [cd, sdf, iou]

output:
  dir: outputs/compare_case001
  difference_image: true
```

label volume 同士を比較する場合、simulation と target は同じ projected grid
である必要があります。つまり、比較後の 2D `shape`, `spacing`, `origin`, `axes`
が一致している必要があります。異なる grid を暗黙に resample することはありません。

## 7. batch-compare の YAML

複数ケースを比較して ranking を出す例です。

```yaml
task: batch-compare

input:
  index: data/examples/compare_pairs.csv

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf, contour]

metrics:
  use: [cd, sdf, iou]

output:
  dir: outputs/batch_compare_001
  ranking: true
  difference_images: true
```

index CSV の例:

```csv
case_id,simulation_kind,simulation_path,target_kind,target_path
case_001,npz_label,data/sim/case_001.npz,npz_label,data/target/case_001.npz
case_002,vti_label,data/sim/case_002.vti,vti_label,data/target/case_002.vti
```

YAML 内の相対パスは YAML ファイル基準です。
batch index CSV 内の相対パスは CSV ファイル基準です。

## 8. metric の選び方

| metric | 見ているもの | 使いどころ |
|---|---|---|
| `cd` | 高さごとの幅、edge 位置、material boundary transition | 半導体断面の CD 評価 |
| `sdf` | 2D SDF field と material boundary SDF | 全体的な形状差 |
| `iou` | overlap | 一致率、重なり |

最初は次の組み合わせを推奨します。

```yaml
metrics:
  use: [cd, sdf, iou]
```

原因分析を詳しくしたい場合だけ、`chamfer`, `sdf_material`, `sdf_band` を追加します。
追加の診断 metric は [Scoring.md](Scoring.md) にまとめています。

CD の測定位置を明示したい場合だけ `metrics.cd.gauge` を追加します。

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

## 9. 出力の読み方

`compare` の重要出力:

| 出力 | 内容 |
|---|---|
| `score.json` | 総合 score と metric 詳細 |
| `metrics.csv` | metric ごとの loss/value/status |
| `metric_details.json` | SDF/IoU/CD などの内訳 |
| `difference.png` | 差分確認画像 |
| `difference_summary.json` | 差分 pixel 数 |
| `cd_profile.csv` | 高さごとの CD profile |
| `cd_profile.png` | CD profile の簡易グラフ |
| `simulation_label_summary.json` | simulation の入力・view 要約 |
| `target_label_summary.json` | target の入力・view 要約 |

`sdf_material` を指定した場合は `per_material_sdf.csv`、`profile` を指定した場合は
`profile.csv`、`corner` を指定した場合は `corner_summary.json` も出ます。

`batch-compare` の重要出力:

| 出力 | 内容 |
|---|---|
| `ranking.csv` | `normalized_total_score` 昇順の ranking |
| `ranking_top.png` | ranking 上位の簡易グラフ |
| `metrics.csv` | 全 case の metric |
| `metric_summary.csv` | metric ごとの min/max/mean/std |
| `score_summary.json` | best case や metric scale の要約 |
| `difference_summary.csv` | 全 case の差分 pixel 数 |
| `cases/` | case ごとの compare 出力 |
| `shared_targets/` | 繰り返し target の共通出力 |

`sdf_material` を指定した batch では、root に `per_material_sdf.csv` が集約されます。

ランキングは `normalized_total_score` を基準にします。小さいほど target に近いです。

## 10. よくある注意点

- `output.dir` は実行結果なので、不要になったら削除してかまいません。
- `_run/used_config.yaml` と `_run/run_info.json` はデバッグ補助です。ユーザーが編集する入力ではありません。
- `cd` が `SKIPPED` の場合は、view に `z` が含まれているか確認してください。
- label volume target と simulation の grid が違う場合は、現状では明示的に失敗します。比較前に同じ grid にそろえてください。
- material id `0` が void でないデータでは、YAML か npz 内で `void_id` を指定してください。

## 11. 結果を消す

生成物や cache をまとめて消す場合:

```powershell
make clean
```

`outputs/` は再生成可能なローカル成果物として扱います。

## 12. 詳しく診断したい場合

通常は `metrics.use: [cd, sdf, iou]` から始めます。
score が悪い理由を詳しく見たい場合だけ、次の診断 metric を追加します。

| 追加 metric | 使う場面 |
|---|---|
| `chamfer` | 境界点群のずれを見たい |
| `sdf_material` | どの material が差分に効いているか見たい |
| `sdf_band` | edge / interface 近傍だけを見たい |
| `profile` | CD の高さごとの内訳を見たい |
| `corner` | bottom corner 位置だけを見たい |
| `topology` | 分断や接続の有無だけを確認したい |

詳しい意味、`SKIPPED` になる条件、追加出力は [Scoring.md](Scoring.md) に集約しています。
