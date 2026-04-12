# 設定の所在マップ

このパッケージでは、ユーザーが編集する設定を YAML に集約します。

## ユーザーが編集するもの

| 場所 | 内容 |
|---|---|
| YAML | task、input、view、features、metrics、output |
| batch index CSV | `batch-compare` の case 一覧 |
| 入力データ | `npz_label`, `vti_label`, `contour_json` |

## ユーザーが編集しないもの

| 場所 | 内容 |
|---|---|
| `outputs/` | 実行結果 |
| `_run/used_config.yaml` | 実行時に使った YAML のコピー |
| `_run/run_info.json` | 実行情報 |
| `score.json` / `metrics.csv` | 評価結果 |

`_run/` はデバッグ補助です。入力設定ではありません。
不要なら削除してかまいません。

## 正式 task

| task | 目的 | `metrics` |
|---|---|---|
| `transform` | 特徴量化 | 使わない |
| `compare` | 1 対 1 比較 | 使う |
| `batch-compare` | 複数比較と ranking | 使う |

## YAML の基本形

```yaml
task: compare

input:
  simulation:
    kind: npz_label
    path: data/sim_case.npz
  target:
    kind: npz_label
    path: data/target_case.npz

view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf, contour]

metrics:
  use: [cd, chamfer, sdf, sdf_material, sdf_band, iou]

output:
  dir: outputs/compare_case001
```

## path 解決

- YAML 内の相対パスは YAML ファイルの場所が基準です。
- batch index CSV 内の相対パスは CSV ファイルの場所が基準です。

## 拡張ポイント

| 追加したいもの | 主な場所 |
|---|---|
| 入力形式 | `wafergeo/compare/label_loaders.py` または `contour_loaders.py` |
| 特徴量出力 | `wafergeo/compare/feature_outputs.py` |
| metric | `wafergeo/compare/metric_defs.py` と `metric_*.py` |
| 軽量出力 | `wafergeo/compare/output_artifacts.py` |
