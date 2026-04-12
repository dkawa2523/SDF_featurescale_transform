# 開発者マニュアル

この資料は、初めてこのコードを改修する開発者が、
**どこを読み、どこを変更し、どう検証するか**を把握するための入口です。

具体的な loader / feature / metric の追加手順は
[ExtensionGuide.md](ExtensionGuide.md) を参照してください。

## 1. このパッケージの目的

このコードの主目的は、simulation や実験由来の形状データを扱いやすい特徴量に変換し、
必要に応じて形状比較の metric を計算することです。

現在実装済みの正式 workflow は 3 つです。

| workflow | 目的 |
|---|---|
| `transform` | 1 つの入力 label volume を特徴量化する |
| `compare` | simulation 1 件と target 1 件を比較する |
| `batch-compare` | 複数 case を比較して ranking を出す |

今後の workflow 拡張は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) に従います。
場当たり的に public pipeline を増やさず、特徴量化系と比較系の用途に沿って追加します。

## 2. 最初に読むファイル

コードを読む順番は次がおすすめです。

| 順番 | ファイル | 目的 |
|---|---|---|
| 1 | `wafergeo/application/runtime/cli.py` | CLI の入口 |
| 2 | `wafergeo/application/runtime/runner.py` | task 名から runner へ振り分け |
| 3 | `wafergeo/compare/schema_loader.py` | YAML の読み込み |
| 4 | `wafergeo/compare/schema_types.py` | YAML から作られる型 |
| 5 | `wafergeo/compare/runner.py` | `transform` / `compare` の実行本体 |
| 6 | `wafergeo/compare/batch_runner.py` | `batch-compare` の実行本体 |
| 7 | `wafergeo/compare/features.py` | 2D view feature の生成 |
| 8 | `wafergeo/compare/metric_defs.py` | metric registry |

全体像を掴むまでは、古い internal module や domain backend を先に深追いしない方が安全です。

## 3. ディレクトリの役割

| 場所 | 役割 |
|---|---|
| `wafergeo/application/runtime/` | CLI から public task を起動する薄い層 |
| `wafergeo/compare/` | 現在の公開 workflow の中心 |
| `wafergeo/core/` | grid、型、hash などの共通部品 |
| `wafergeo/io/` | VTI などの低レベル入力処理 |
| `wafergeo/label/` | label volume 正規化 |
| `wafergeo/sdf/` | SDF 生成の domain 処理 |
| `wafergeo/mesh/` | mesh 生成の domain 処理 |
| `configs/examples/` | ユーザー向けの最小 YAML |
| `tests/compare/` | public workflow の中心テスト |

今の開発では、まず `wafergeo/compare/` と `tests/compare/` を中心に見れば十分です。

## 4. データの流れ

`compare` の流れは次の通りです。

```text
YAML
  -> schema_loader
  -> SimulationInputSpec / TargetInputSpec / ViewSpec / MetricSpec
  -> loader
  -> LabelVolume または ContourData
  -> ViewFeature
  -> metric registry
  -> score.json / metrics.csv / difference.png
```

重要なのは、metric が raw file を直接読まないことです。
metric は `ViewFeature` だけを見て計算します。

## 5. 内部契約

### LabelVolume

3D label volume の内部形式です。

```text
material_id: [Z,Y,X]
spacing: [Z,Y,X]
origin: [Z,Y,X]
```

ユーザー向け `npz_label` は `[X,Y,Z]` ですが、loader で `[Z,Y,X]` に変換します。

### ViewFeature

metric が受け取る 2D 比較用特徴です。

| field | 内容 |
|---|---|
| `mask` | non-void 領域 |
| `label2d` | topmost non-void material label |
| `sdf_nm` | 2D SDF |
| `contours` | contour / boundary 点 |
| `boundary_mask` | material boundary |
| `material_masks` | material ごとの projected mask |
| `grid2d` | 2D grid |

新しい metric は、まず `ViewFeature` の既存 field で計算できるかを検討してください。
新しい feature field を増やすのは最後でよいです。

## 6. 変更内容ごとの判断表

| やりたいこと | 変更する場所 |
|---|---|
| 新しい入力形式を読む | `label_loaders.py` または `contour_loaders.py` |
| YAML の kind を増やす | `schema_types.py` |
| `transform` の出力を増やす | `feature_outputs.py`, `transform_features.py` |
| 比較 metric を増やす | `metric_*.py`, `metric_defs.py` |
| metric の詳細 CSV/PNG を増やす | `output_artifacts.py` |
| 実行 task を増やす | `WorkflowRoadmap.md` にある計画済み workflow の範囲で追加する |

## 7. metric 追加の最小手順

1. `metric_*.py` に `compute_xxx` を追加する。
2. `metric_defs.py` に `MetricDefinition` を登録する。
3. 必要な feature を `required_features` に書く。
4. `tests/compare/` に同一形状とずれ形状の test を追加する。
5. `docs/Scoring.md` に metric の意味を書く。

例:

```python
METRIC_DEFINITIONS["my_metric"] = MetricDefinition(
    name="my_metric",
    required_features=frozenset({"sdf"}),
    compute=compute_my_metric,
    loss_scale=10.0,
)
```

`loss_scale` は `normalized_total_score` に使われます。
ランキングで既存 metric と釣り合う値にしてください。

## 8. feature 追加の最小手順

1. `schema_types.py` の `FEATURE_NAMES` に追加する。
2. feature を作る関数を追加する。
3. `feature_outputs.py` の writer に登録する。
4. `transform` で出力されることを test する。
5. `compare` で使う feature なら、対応 metric の `required_features` に入れる。

重い feature はデフォルトで出さないでください。
例えば full 3D SDF は `sdf3d` を明示したときだけ出します。

## 9. loader 追加の最小手順

1. raw file を読み、軸順と単位を内部形式へ変換する。
2. `LabelVolume` または `ContourData` を返す。
3. dispatch dict に登録する。
4. YAML kind を schema に追加する。
5. loader 単体 test と workflow test を追加する。

loader の中で metric を計算しないでください。
loader は「読む、検証する、内部形式に変換する」だけです。

## 10. 出力追加の方針

出力は増やしすぎるとユーザーが迷います。
追加する場合は、次のどれかに該当するものだけにしてください。

- ユーザーが評価結果を判断しやすくなる。
- データサイエンティストが notebook / spreadsheet で扱いやすくなる。
- metric の内訳を説明できる。

軽量出力は `output_artifacts.py` に置きます。
metric compute 関数の中でファイルを書かないでください。

## 11. テストの考え方

最小テストは次の 4 種類です。

- 正常入力で出力が作られる。
- 不正入力で分かりやすく失敗する。
- 同一形状は score が良い。
- ずらした形状は score が悪化する。

実行:

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
```

公式 example も確認します。

```powershell
py -3.13 -m wafergeo run transform --config configs\examples\transform.simple.yaml
py -3.13 -m wafergeo run compare --config configs\examples\compare.simple.yaml
py -3.13 -m wafergeo run batch-compare --config configs\examples\batch-compare.simple.yaml
```

## 12. 複雑化を避ける判断

次の変更は慎重に扱ってください。

- 新しい public task
- 深い YAML 階層
- 大きな report 生成
- metric ごとの独自 runner
- raw file を metric から直接読む実装
- 暗黙の resampling / alignment

特に alignment や resampling は評価結果の意味を大きく変えます。
必要になった場合は、仕様を docs に明記してから実装してください。

## 13. 迷ったときの原則

- ユーザーは YAML と `outputs/` だけ見ればよい状態にする。
- 開発者は loader / feature / metric / output artifact のどこを触るか分かる状態にする。
- 既存の metric の意味を変える場合は docs と tests を同時に更新する。
- 追加したものは、最小 example か test で使い方を示す。
- 使わない旧導線を残して複雑にしない。
