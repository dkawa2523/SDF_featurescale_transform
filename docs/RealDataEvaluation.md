# 実データ評価手順

このページは開発者向けです。通常ユーザーは `configs/examples/` の YAML だけを使えば十分です。

実データに近い dataset を使って、現在の比較 metric が壊れていないかを確認するための手順です。
旧 `benchmark` pipeline は使いません。正式 workflow の `batch-compare` だけで評価します。
ここで使う metric set は「推奨 default」ではなく、退行確認用の全部入り smoke set です。

## 目的

新しい metric や SDF helper を追加したあと、次を確認します。

- 自己比較が ranking 1 位になる。
- `cd`, `chamfer`, `profile`, `corner`, `sdf`, `sdf_material`, `sdf_band`, `iou`, `topology` が出力される。
- `difference.png` と CSV/JSON で差分原因を確認できる。
- 実行結果は `outputs/` にだけ出て、リポジトリには混ぜない。

## 入力 dataset

使用する設定:

```text
configs/runs/dataset_t08_vs_run0010.yaml
configs/runs/dataset_t08_vs_run0010_pairs.csv
```

対象データ:

```text
dataset/ataset_3d_test2/run_0000/vox_t08.vti
...
dataset/ataset_3d_test2/run_0010/vox_t08.vti
```

`run_0010` を擬似 target とし、`run_0000` から `run_0010` までを比較します。
`run_0010` 自身も比較に含め、自己比較の sanity check に使います。

## 実行

PowerShell でリポジトリ root から実行します。

```powershell
py -3.13 -m wafergeo run batch-compare --config .\configs\runs\dataset_t08_vs_run0010.yaml
```

出力先:

```text
outputs/realdata_dataset_t08_vs_run0010/
```

主な確認ファイル:

| file | 見る内容 |
|---|---|
| `ranking.csv` | `run_0010` が 1 位か |
| `metrics.csv` | metric ごとの loss / value / status |
| `score_summary.json` | 全体 summary |
| `material_confusion.csv` | material id の一致/不一致 |
| `per_material_sdf.csv` | material ごとの SDF loss |
| `profile.csv` | 高さごとの CD/profile 差分 |
| `corner_summary.json` | bottom corner 差分 |
| `differences/*.png` | 目視確認用の差分画像 |

## 結果確認

最低限の確認:

```powershell
Import-Csv .\outputs\realdata_dataset_t08_vs_run0010\ranking.csv | Select-Object -First 5
Import-Csv .\outputs\realdata_dataset_t08_vs_run0010\metrics.csv | Group-Object name | Select-Object Name,Count
```

期待する状態:

- `ranking.csv` の 1 行目が `run_0010`。
- `run_0010` の `normalized_total_score` と `total_score` が 0 に近い。
- `metrics.csv` に `cd`, `chamfer`, `profile`, `corner`, `sdf`, `sdf_material`, `sdf_band`, `iou`, `topology` がある。
- `metrics.csv` の `status` が想定外に `SKIPPED` だらけになっていない。
- `differences/run_0010.png` は自己比較なので差分がほぼ無い。

より詳しく見る場合:

```powershell
Import-Csv .\outputs\realdata_dataset_t08_vs_run0010\metrics.csv |
  Where-Object case_id -eq run_0010 |
  Format-Table case_id,name,loss,value,status

Import-Csv .\outputs\realdata_dataset_t08_vs_run0010\per_material_sdf.csv |
  Sort-Object {[double]$_.sdf_loss_nm} -Descending |
  Select-Object -First 10
```

## 精度・速度・メモリの見方

この手順は、大規模な性能 benchmark ではありません。
目的は、手法追加後の退行を小さく早く見つけることです。

| 観点 | 確認するもの |
|---|---|
| 精度 | 自己比較が 1 位、差がある case の SDF/material/profile が悪化する |
| 速度 | 11 case が通常の開発環境で短時間に終わる |
| メモリ | `outputs/` が過剰に肥大化しない。巨大な中間ファイルを増やさない |
| 保守性 | 新 metric 追加後も YAML block を増やさず `metrics.use` だけで評価できる |

今回の dataset では `topology` が全 case で 0 になる場合があります。
これは分断・接続の component 数差が無いという sanity check であり、形状差が無いという意味ではありません。
形状差は `cd`, `sdf`, `sdf_material`, `sdf_band`, `iou` を見ます。

## 掃除

実行結果は再生成可能です。不要になったら削除して構いません。

```powershell
Remove-Item .\outputs\realdata_dataset_t08_vs_run0010 -Recurse -Force
```

`outputs/` は git 管理対象にしません。
