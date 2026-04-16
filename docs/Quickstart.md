# クイックスタート

以下のコマンドはリポジトリ root から実行します。

まず example config で環境が動くことを確認してください。その後、
入力パスと出力ディレクトリだけを差し替え、最後に feature や metric の設定を
変更するのが安全です。

## インストール

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[scipy,viz,dev]"
```

VTI 入力が必要な場合だけ `vtk` を追加します。

```powershell
python -m pip install -e ".[scipy,vtk,viz,dev]"
```

## example を実行する

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run batch-transform --config .\configs\examples\batch-transform.simple.yaml
python -m wafergeo run transform-eval --config .\configs\examples\transform-eval.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
python -m wafergeo run compare-eval --config .\configs\examples\compare-eval.simple.yaml
```

## 最初に見る出力

| workflow | 最初に確認するファイル |
| --- | --- |
| `transform` | `features/`, `feature_summary.json` |
| `batch-transform` | `dataset_index.csv`, `features_summary.csv` |
| `transform-eval` | `figures/feature_scores.csv`, `figures/by_target_shape/` |
| `compare` | `objective.json`, `metrics.csv`, `difference.png` |
| `batch-compare` | `objectives.csv`, `ranking.csv` |
| `compare-eval` | `axis_agreement.csv`, `case_scores.csv`, `figures/` |

生成物は `outputs/` 配下に出力されます。通常はコミットしません。

## データサイエンス用途の流れ

1. 各形状ケースを `npz_label` または `vti_label` にする。
2. `batch-transform` で特徴量を作る。
3. `transform-eval` で `full_shape`, `material_shape`, `process_delta_shape`
   の特徴量がケース差を有用に表現しているか確認する。
4. `compare-eval` で従来の CD 軸と SDF 系の shape 軸を比較し、
   最適化 loss として使う評価軸を選ぶ。
5. CSV/JSON/NPZ 出力をサロゲートモデルや optimizer に渡す。

実データの CSV を用意する前に [入力データ](InputData.md) を確認してください。

## eval 結果の読み方

`transform-eval` では次の順で確認します。

1. `figures/input_shape_sections.png`
2. `figures/feature_scores.csv`
3. `figures/by_target_shape/<target_shape>/<method>/field.png`
4. `figures/by_target_shape/<target_shape>/<method>/case_distance.png`

transform 結果は `target_shape x method` として読みます。例えば
`material_shape/sdf` は、各 material shape に SDF を適用した結果です。

`compare-eval` では次の順で確認します。

1. `axis_agreement.csv`
2. `figures/cd_vs_sdf_scatter.png`
3. `figures/comparison_loss_heatmap.png`
4. `figures/representative_differences/`

`cd_vs_sdf_scatter.png` は、height-CD baseline 軸と SDF shape-distance 軸の
`comparison_loss` を比較します。metric ごとの raw loss が必要な場合は
`case_scores.csv` を見てください。

## 検証

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
py -3.13 -m mkdocs build --strict
```

docs build 後に生成される `site/` はコミットしません。
