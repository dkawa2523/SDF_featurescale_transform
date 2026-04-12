# クイックスタート

このページは、最短で動かすための手順です。
詳しい実行方法や出力の読み方は [UserManual.md](UserManual.md) を見てください。

## 1. インストール

```powershell
Set-Location C:\Users\user\Desktop\SDF_fs
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

VTI を使う場合:

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## 2. まず example を実行する

正式な実行入口は次の 3 つだけです。

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
```

## 3. YAML で変更する場所

通常ユーザーが編集するのは次の場所です。

```yaml
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

metrics:
  use: [cd, sdf, iou]

output:
  dir: outputs/compare_case001
```

`transform` では `target` と `metrics` は使いません。

## 4. path の基準

- YAML 内の相対パスは YAML ファイルの場所が基準です。
- `batch-compare` の CSV 内の相対パスは CSV ファイルの場所が基準です。

## 5. 出力を見る

`compare` ではまず次を見ます。

- `score.json`
- `metrics.csv`
- `difference.png`
- `cd_profile.png`

原因分析として `sdf_material` を追加した場合は、`per_material_sdf.csv` も確認します。

`batch-compare` ではまず次を見ます。

- `ranking.csv`
- `ranking_top.png`
- `metric_summary.csv`
- `metrics.csv`

CSV/JSON が正式なデータです。PNG は確認用の軽量な可視化です。

## 6. よくある詰まりどころ

- `cd` は `[x,z]` や `[y,z]` のように `z` を含む view で使います。
- label volume 同士の比較では、projected grid の `shape`, `spacing`, `origin`, `axes` が一致している必要があります。
- material id `0` が void でない場合は `void_id` を指定してください。
- `outputs/` は生成物です。不要なら削除してかまいません。

生成物と cache をまとめて消す場合:

```powershell
make clean
```
