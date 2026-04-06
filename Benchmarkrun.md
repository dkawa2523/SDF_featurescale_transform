# Benchmark / Preview 実行手順

## 1. このドキュメントの目的

このドキュメントは、`dataset/` 配下の実データを使って、

1. 環境構築する
2. benchmark を走らせる
3. preview / audit を走らせる
4. 出力を確認して評価する

までを、初めてこのリポジトリを見る第三者でも再現しやすいようにまとめた手順書です。  
コマンドは **Windows PowerShell** を前提に書いています。

このリポジトリでは、次の 3 つが主な実行入口です。

| 用途 | スクリプト | 役割 |
|---|---|---|
| benchmark | `scripts/run_correspondence_benchmark.py` | synthetic ケース + 実 VTI をまとめて比較評価する |
| preview | `scripts/run_vti_preview.py` | 1 つの VTI を label / SDF / mesh / 図版へ展開する |
| audit | `scripts/run_vti_correspondence_audit.py` | raw ラベルと変換後ラベルの一致度を確認する |

---

## 2. 前提条件

### 2.1 必要なもの

| 項目 | 推奨 |
|---|---|
| OS | Windows 10 / 11 |
| Python | 3.11 以上 |
| シェル | PowerShell |
| 空き容量 | 少なくとも数 GB |

`pyproject.toml` では `requires-python = ">=3.11"` です。  
手元での確認は `py -3.13` を使っていますが、`3.11` 以上であればよいです。

### 2.2 リポジトリ直下で作業する

このドキュメントのコマンドは、リポジトリ直下 `SDF_fs/` をカレントディレクトリにした状態で実行してください。

```powershell
Set-Location C:\Users\user\Desktop\SDF_fs
```

---

## 3. データセットの準備

### 3.1 `dataset/` がすでにある場合

まず、代表ケースの VTI が見えているか確認します。

```powershell
Test-Path .\dataset\ataset_3d_test2\run_0000\vox_t08.vti
```

`True` が返れば、そのまま次へ進めます。

### 3.2 `dataset/` がない場合

リポジトリ直下に `dataset.zip` があるので、展開します。  
この zip は `dataset/` フォルダごと含んでいるため、**展開先は `.`** にしてください。

```powershell
Expand-Archive -Path .\dataset.zip -DestinationPath . -Force
```

展開後に再確認します。

```powershell
Test-Path .\dataset\ataset_3d_test2\index.csv
Test-Path .\dataset\ataset_3d_test2\run_0000\vox_t08.vti
```

---

## 4. 環境構築

## 4.1 仮想環境を作る

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`py -3.13` がない環境では、`py -3.11` や `python` に置き換えてください。

## 4.2 推奨インストール

### フル構成

benchmark、preview、audit を素直に回したい場合の推奨構成です。

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

この構成で入る主な optional dependency は次の通りです。

| extra | 主な用途 |
|---|---|
| `scipy` | SDF の距離変換 |
| `vtk` | VTI 読み込み、VTK mesh backend、shell 抽出 |
| `viz` | matplotlib による図出力 |
| `dev` | pytest などの開発用ツール |

### 軽量構成

VTK の導入が難しい環境では、次の構成でも benchmark と preview 自体は動きます。

```powershell
python -m pip install -e ".[scipy,viz,dev]"
```

ただしこの場合は次の挙動になります。

- benchmark の `mesh_backend: vtk` 行は `naive_interface` へ fallback して `status: WARN` になりやすい
- preview は `read_backend_used: xml_fallback` で VTI を読み込むことがある
- preview / audit の shell 抽出ができず、3D mesh の中身が 0 face になることがある

つまり、**数値確認はできるが、3D mesh 系の確認は弱くなる** と考えてください。

---

## 5. benchmark を走らせる

## 5.1 そのまま使える spec

このリポジトリには、`dataset/` を直接参照する benchmark spec を追加しています。

```text
configs/correspondence_bench_dataset_t08.yaml
```

この YAML は、`dataset/ataset_3d_test2/run_0000/vox_t08.vti` を `real_vti` ケースとして使います。

## 5.2 実行コマンド

```powershell
python .\scripts\run_correspondence_benchmark.py `
  --spec .\configs\correspondence_bench_dataset_t08.yaml `
  --out .\outputs\bench_dataset_t08
```

### 実行内容

この benchmark は、次の条件の組み合わせを評価します。

- scenarios: `cube`, `layers3`, `thin_shell`, `diagonal`, `real_vti`
- point-to-cell policy: `nearest`, `majority_nearest_tie`
- mesh backend: `naive_interface`, `vtk`
- mesh mode: `material_shell`, `interface_mesh`

つまり、`5 x 2 x 2 x 2 = 40` 行の stage 結果が出ます。

## 5.3 出力先

正常に完了すると、次のファイルができます。

```text
outputs/bench_dataset_t08/
  benchmark_manifest.json
  figures/
    mesh_boundary_iou.png
  tables/
    stage_metrics.csv
    summary_metrics.csv
```

## 5.4 まず何を見ればよいか

### 1. manifest の status を確認

```powershell
Get-Content .\outputs\bench_dataset_t08\benchmark_manifest.json -TotalCount 80
```

見るべき主な項目は次の通りです。

| 項目 | 意味 |
|---|---|
| `status` | 全体状態。`OK` か `WARN` |
| `messages` | fallback や不足依存の警告 |
| `summary.strict_overall_pass` | 閾値ベースの総合 pass / fail |
| `summary.policy_gap_max` | policy 間の差の大きさ |
| `diagnosis.root_cause_candidates` | 自動診断結果 |

### 2. summary テーブルを見る

```powershell
Import-Csv .\outputs\bench_dataset_t08\tables\summary_metrics.csv | Format-Table -AutoSize
```

特に重要なのは次の指標です。

| 指標 | 意味 |
|---|---|
| `sdf_roundtrip_acc_mean` | label -> TSDF -> label の復元精度 |
| `material_shell_mesh_iou_mean` | material shell での境界一致度 |
| `interface_mesh_iou_mean` | interface mesh での境界一致度 |
| `material_shell_mesh_chamfer_nm_mean` | shell の境界距離 |
| `policy_gap_max` | point-to-cell policy の差 |
| `strict_overall_pass` | 閾値基準で全行 pass したか |

### 3. `real_vti` 行だけ見る

```powershell
Import-Csv .\outputs\bench_dataset_t08\tables\stage_metrics.csv `
  | Where-Object { $_.scenario -eq 'real_vti' } `
  | Format-Table scenario,policy,mesh_backend,mesh_backend_used,mesh_mode,status,point_to_cell_match,mesh_boundary_iou,mesh_boundary_chamfer_nm -AutoSize
```

この表を見ると、

- `nearest` と `majority_nearest_tie` の差
- `vtk` が本当に使われたか
- fallback で `mesh_backend_used=naive_interface` になっていないか

がすぐ分かります。

## 5.5 benchmark の合格基準

spec に入っている閾値は次の通りです。

| 指標 | 閾値 |
|---|---:|
| mesh boundary IoU | 0.80 以上 |
| mesh boundary chamfer | 2.0 nm 以下 |
| mesh boundary coverage | 0.70 以上 |
| SDF roundtrip accuracy | 0.999 以上 |
| render diff rate | 0.10 以下 |
| policy gap | 0.05 以下 |

これらは `configs/correspondence_bench_dataset_t08.yaml` の `thresholds` にあります。

---

## 6. preview を走らせる

benchmark は比較全体を見るのに向いていますが、**1 個の VTI の中身を丁寧に見る** には preview が便利です。

## 6.1 実行コマンド

```powershell
python .\scripts\run_vti_preview.py `
  --vti .\dataset\ataset_3d_test2\run_0000\vox_t08.vti `
  --out .\outputs\vti_preview_t08 `
  --outside-id 2
```

## 6.2 出力先

```text
outputs/vti_preview_t08/
  audit_manifest.json
  preview_manifest.json
  figures/
  sdf/
    sdf_summary_full.json
    tsdf_full_stack.npy
  tables/
    material_volume_compare.csv
    slice_metrics.csv
```

## 6.3 まず見るべきファイル

### preview manifest

```powershell
Get-Content .\outputs\vti_preview_t08\preview_manifest.json -TotalCount 120
```

主な確認点は次の通りです。

| 項目 | 意味 |
|---|---|
| `status` | preview 全体状態 |
| `read_backend_used` | `vtk` か `xml_fallback` か |
| `tsdf_shape` | 生成された TSDF stack の shape |
| `selected_material_ids` | TSDF 化された材料 |
| `metrics.slice.*` | 断面一致度 |
| `postprocess.status` | VTK 後処理の QA 状態 |

### SDF summary

```powershell
Get-Content .\outputs\vti_preview_t08\sdf\sdf_summary_full.json
```

ここでは次を見ます。

- `shape`
- `dtype`
- `mu_nm`
- `tsdf_min`, `tsdf_max`
- `nan_count`, `inf_count`

### slice 指標

```powershell
Import-Csv .\outputs\vti_preview_t08\tables\slice_metrics.csv | Format-Table -AutoSize
```

IoU / Dice / Boundary Chamfer が揃っているので、raw ラベルと変換後ラベルがどれだけ一致しているかを確認できます。

---

## 7. audit を単独で走らせる

preview は audit を内包していますが、raw / converted の対応だけを見たい場合は audit 単体でも実行できます。

```powershell
python .\scripts\run_vti_correspondence_audit.py `
  --vti .\dataset\ataset_3d_test2\run_0000\vox_t08.vti `
  --out .\outputs\vti_audit_t08 `
  --outside-id 2
```

audit では主に次を見ます。

- `audit_manifest.json`
- `tables/material_volume_compare.csv`
- `tables/slice_metrics.csv`
- `figures/slice_*`
- `figures/3d_*`

---

## 8. 結果の見方

## 8.1 benchmark の見方

benchmark は「どの設定が安定しているか」を見るためのものです。  
特に次の 3 点を見ると、だいたい状況が分かります。

### SDF が安定しているか

- `sdf_roundtrip_acc_mean`
- 各 row の `sdf_roundtrip_acc`

ここが 1.0 に近ければ、label -> TSDF -> label の往復で形状が崩れていません。

### mesh backend の差が大きいか

- `mesh_boundary_iou`
- `mesh_boundary_chamfer_nm`
- `mesh_backend` と `mesh_backend_used`

ここで `mesh_backend_used` が期待値から変わっていれば、fallback が起きています。

### policy の差が大きいか

- `point_to_cell_match`
- `policy_gap_max`
- `policy_gap_real_vti`

ここが大きいと、PointData -> CellData の変換ポリシーの選択が結果に効いています。

## 8.2 preview の見方

preview は「1 サンプルの構造が本当に保たれているか」を見るためのものです。  
特に次の 4 点が重要です。

1. `slice_metrics.csv`  
   断面一致度を見る

2. `material_volume_compare.csv`  
   材料ごとの体積差が変化していないかを見る

3. `sdf/sdf_summary_full.json`  
   TSDF の shape と数値破綻の有無を見る

4. `figures/sdf_*`  
   SDF のチャンネルや最小絶対値分布を視覚確認する

---

## 9. 実行例

## 9.1 benchmark

```powershell
python .\scripts\run_correspondence_benchmark.py `
  --spec .\configs\correspondence_bench_dataset_t08.yaml `
  --out .\outputs\bench_dataset_t08
```

## 9.2 preview

```powershell
python .\scripts\run_vti_preview.py `
  --vti .\dataset\ataset_3d_test2\run_0000\vox_t08.vti `
  --out .\outputs\vti_preview_t08 `
  --outside-id 2
```

## 9.3 audit

```powershell
python .\scripts\run_vti_correspondence_audit.py `
  --vti .\dataset\ataset_3d_test2\run_0000\vox_t08.vti `
  --out .\outputs\vti_audit_t08 `
  --outside-id 2
```

---

## 10. よくある詰まりどころ

## 10.1 `vtk` が無い

症状:

- benchmark が `status: WARN`
- `messages` に `mesh backend fallback` が出る
- preview が `read_backend_used: xml_fallback` になる

対処:

```powershell
python -m pip install -e ".[vtk]"
```

## 10.2 `matplotlib` が無い

症状:

- 図が出ない
- `figure warning` が manifest に入る

対処:

```powershell
python -m pip install -e ".[viz]"
```

## 10.3 `dataset/` が無い

症状:

- `FileNotFoundError`
- `real_vti_path` が見つからない

対処:

```powershell
Expand-Archive -Path .\dataset.zip -DestinationPath . -Force
```

## 10.4 余計な CLI オプションを入れてしまう

`run_vti_preview.py` や `run_vti_correspondence_audit.py` は、現在の CLI では `--mesh-mode` のようなオプションを受け取りません。  
受け付けるのは基本的に次だけです。

- preview: `--vti`, `--out`, `--outside-id`
- audit: `--vti`, `--out`, `--outside-id`
- benchmark: `--spec`, `--out`

---

## 11. 手元で確認した結果

2026-04-06 に、このリポジトリ上で次を確認しました。

### benchmark

```powershell
python .\scripts\run_correspondence_benchmark.py `
  --spec .\configs\correspondence_bench_dataset_t08.yaml `
  --out .\outputs\bench_dataset_t08
```

確認結果:

- コマンド自体は正常終了
- `status` は `WARN`
- 理由は `vtk` 未導入による backend fallback
- ただし `summary.strict_overall_pass` は `true`

### preview

```powershell
python .\scripts\run_vti_preview.py `
  --vti .\dataset\ataset_3d_test2\run_0000\vox_t08.vti `
  --out .\outputs\vti_preview_t08 `
  --outside-id 2
```

確認結果:

- コマンド自体は正常終了
- `read_backend_used` は `xml_fallback`
- `slice_metrics` はすべて 1.0 / 0.0 系で良好
- `vtk` 未導入のため shell 抽出は fallback になり、3D mesh の face 数は 0

このため、**数値再現だけなら軽量構成でも進められますが、3D mesh の確認まで行うなら `vtk` を入れたフル構成が推奨** です。

---

## 12. 最短ルート

迷ったら、まず次の 5 コマンドで進めてください。

```powershell
Set-Location C:\Users\user\Desktop\SDF_fs
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[scipy,viz,dev]"
python .\scripts\run_correspondence_benchmark.py --spec .\configs\correspondence_bench_dataset_t08.yaml --out .\outputs\bench_dataset_t08
```

その後、結果確認は次で十分です。

```powershell
Import-Csv .\outputs\bench_dataset_t08\tables\summary_metrics.csv | Format-Table -AutoSize
Get-Content .\outputs\bench_dataset_t08\benchmark_manifest.json -TotalCount 80
```

3D mesh まで確認したくなったら、最後に `vtk` を追加してください。

```powershell
python -m pip install -e ".[vtk]"
```
