# Quickstart

最短で動作確認するための手順です。
詳しい YAML と出力の意味は [User Manual](UserManual.md) を参照してください。

## 1. セットアップ

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

## 2. example を実行する

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run batch-transform --config .\configs\examples\batch-transform.simple.yaml
python -m wafergeo run transform-eval --config .\configs\examples\transform-eval.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
python -m wafergeo run compare-eval --config .\configs\examples\compare-eval.simple.yaml
```

## 3. 出力を見る

| workflow | まず見るファイル |
| --- | --- |
| `transform` | `features/`, `feature_summary.json` |
| `batch-transform` | `dataset_index.csv`, `features_summary.csv` |
| `transform-eval` | `candidate_eval_summary.csv`, `case_variation_summary.csv` |
| `compare` | `objective.json`, `metrics.csv`, `difference.png` |
| `batch-compare` | `objectives.csv`, `ranking.csv` |
| `compare-eval` | `candidate_summary.csv`, `case_scores.csv` |

`outputs/` は実行結果です。不要なら削除して構いません。

## 4. よく使う確認コマンド

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
py -3.13 -m mkdocs build --strict
```

`mkdocs build` 後に作られる `site/` は生成物です。
Git 管理には入れません。
