# 拡張ガイド

この資料は、新しい loader / feature / metric を追加する開発者向けです。
目的は、公開 workflow を複雑にせずに手法を追加できる状態を保つことです。

初めてコード全体を読む場合は、先に [DeveloperManual.md](DeveloperManual.md) を読んでください。
この資料は、具体的な追加作業の手順書です。

正式 CLI は変えません。

```text
python -m wafergeo run transform --config <yaml>
python -m wafergeo run compare --config <yaml>
python -m wafergeo run batch-compare --config <yaml>
```

新しい手法を追加するときは、まず既存の YAML block に追加できるかを考えます。
新しい public task は最後の手段です。

## 基本ルール

- YAML は浅く保つ。基本 block は `task`, `input`, `view`, `features`, `metrics`, `output`。
- raw file の仕様は loader で吸収する。
- algorithm は `LabelVolume` や `ViewFeature` のような内部契約を受け取る。
- simulation と target の比較は、同じ projected 2D grid 上で行う。
- 異なる grid を比較したい場合は、scoring 前に明示的な alignment/resampling を追加する。
- 出力は `output.dir` の下に単純な CSV/JSON/PNG として出す。
- `manifest` や大きな report framework を復活させない。
- 抽象化を増やす前に、小さい test を追加する。

## 拡張ポイント

| 追加したいもの | 主なファイル | YAML |
|---|---|---|
| simulation loader | `label_loaders.py`, `schema_types.py` | `input.simulation.kind` |
| target contour loader | `contour_loaders.py`, `schema_types.py` | `input.target.kind` |
| target label loader | `label_loaders.py`, `schema_types.py` | `input.target.kind` |
| feature output | `feature_outputs.py`, `transform_features.py` | `features.use` |
| metric | `metric_*.py`, `metric_defs.py` | `metrics.use` |
| 軽量出力 | `output_artifacts.py` | 原則 YAML 追加なし |

## simulation loader を追加する

新しい simulation file format を正式入力にしたい場合の手順です。

1. `schema_types.py` の `SimulationKind` と validation に kind 名を追加する。
2. `label_loaders.py` に loader 関数を追加する。
3. `LABEL_LOADERS` に登録する。
4. loader test を追加する。

loader は必ず `LabelVolume` を返します。

```python
def load_my_label(path: str | Path, *, void_id: int | None = None) -> LabelVolume:
    ...
```

loader の責務:

- raw file を読む。
- 必要な array / metadata を検証する。
- ユーザー向け軸順を内部 `[Z,Y,X]` に変換する。
- `GridSpec` と `MaterialSpec` を作る。
- `LabelVolume` を返す。

SDF、mesh、metric 側に raw file の癖を持ち込まないでください。

## target loader を追加する

輪郭 target を追加する場合は `ContourData` を返します。

```python
def load_my_contour(
    path: str | Path,
    *,
    units_override: str | None = None,
    view_axes: tuple[AxisName, AxisName] = ("x", "y"),
) -> ContourData:
    ...
```

target loader の責務:

- schema や columns を検証する。
- 単位を nm に変換する。
- `view.axes` に合わせて 2D に投影する。
- `ContourData` を返す。

label volume target の場合は simulation loader と同じく `LabelVolume` を返し、
`LABEL_LOADERS` に登録します。

## feature を追加する

`transform` で新しい特徴量を出したい場合の手順です。

1. `schema_types.py` の `FEATURE_NAMES` に feature 名を追加する。
2. 2D view feature なら `features.py`、3D transform output なら `transform_features.py` に実装する。
3. `feature_outputs.py` の dispatch に writer を追加する。
4. 出力ファイルの shape、axis、units を test と docs に書く。

feature 実装では YAML を直接読まないでください。
YAML は schema / runner が解決し、feature は内部契約だけを受け取ります。

## metric を追加する

新しい比較方法を `metrics.use` に追加したい場合の手順です。

1. 近い責務の `metric_*.py` に compute 関数を追加する。
2. 必要なら新しい `metric_*.py` を作る。
3. `metric_defs.py` に登録する。
4. `tests/compare/` に test を追加する。
5. `docs/Scoring.md` に metric の意味を書く。

compute 関数の形:

```python
def compute_my_metric(
    sim: ViewFeature,
    target: ViewFeature,
    context: MetricContext,
) -> MetricComputation:
    value = ...
    loss = ...
    return MetricComputation(name="my_metric", value=float(value), loss=float(loss))
```

metric の考え方:

- `value` は人が読む値。
- `loss` は ranking が最小化する値。
- 大きいほど良い metric は、`loss` に変換する。例: `iou` は `loss = 1 - iou`。
- 詳細内訳は `details` に入れる。
- metric 固有の CSV/PNG は compute 関数内では作らず、`output_artifacts.py` に置く。

registry への登録例:

```python
METRIC_DEFINITIONS["my_metric"] = MetricDefinition(
    name="my_metric",
    required_features=frozenset({"sdf"}),
    compute=compute_my_metric,
    loss_scale=10.0,
)
```

`loss_scale` は `normalized_total_score` に使われます。
nm 単位の距離 metric なら、まずは工程上意味のあるスケールを選んでください。

## 軽量出力を追加する

新しい CSV や簡易 PNG が必要な場合は `output_artifacts.py` に追加します。

追加してよいもの:

- 既存の score / metric / feature から派生できる CSV
- 確認用の小さい PNG
- notebook や spreadsheet で扱いやすい flat table

避けるもの:

- 大きな HTML report
- 複雑な plot 設定
- metric 計算中にファイルを書く処理

## test 方針

最低限、次を確認します。

- 正常入力で出力が作られる。
- 不正入力で分かりやすく失敗する。
- 同一形状は score が良い。
- ずらした形状は score が悪化する。
- 新 metric は `metrics.csv` に出る。
- YAML の feature dependency が間違っている場合は load 時に失敗する。

実行:

```powershell
py -3.13 -m ruff check wafergeo tests
py -3.13 -m mypy wafergeo
py -3.13 -m pytest -q
```

## 新しい public task を追加しない判断

次のような理由だけでは、新しい task を増やさないでください。

- 新しい file format
- 新しい SDF 手法
- 新しい mesh 手法
- 新しい metric
- 既存 compare 結果の別可視化

これらは loader / feature / metric / output artifact として追加します。

新しい task を検討するのは、入力、処理、出力、ユーザー workflow が
`transform`, `compare`, `batch-compare` と明確に違う場合だけです。
