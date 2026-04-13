# Extension Guide

このページは、新しい loader / feature / metric を追加するための最小手順です。
入口、YAML、runner をむやみに増やさず、手法だけを追加することを優先します。

## 追加前の判断

| 追加したいもの | 分類 |
| --- | --- |
| 新しい file format を読みたい | loader |
| 形状を新しい特徴量へ変換したい | feature |
| simulation と target の差を新しい方法で測りたい | metric |
| 結果 table を増やしたい | output |

分類できない場合は、設計が広がりすぎている可能性があります。
先に [Workflow Roadmap](WorkflowRoadmap.md) に照らして整理してください。

## 新しい loader

1. `wafergeo/compare/loader.py` に読み込み関数を追加する。
2. label volume なら `LABEL_LOADERS` に登録する。
3. contour なら `CONTOUR_LOADERS` に登録する。
4. `schema_types.py` の `SimulationKind` / `TargetKind` を更新する。
5. 小さい synthetic file で loader test を追加する。

loader は `LabelVolume` または `ContourData` を返します。
runner に file parse ロジックを書かないでください。

## 新しい feature

1. feature 本体を `transform_features.py` または domain module に追加する。
2. `feature_outputs.py` に writer を 1 件登録する。
3. `schema_types.py` の `FEATURE_NAMES` に追加する。
4. 出力は `features/<feature_name>.*` にまとめる。
5. `feature_summary.json` に shape、units、semantics を残す。
6. synthetic test を追加する。

process-aware feature の場合は、`PROCESS_FEATURE_NAMES` に追加し、
`process.enabled: true` と `input.reference` を必須にします。

## 新しい metric

1. `wafergeo/compare/metric_<name>.py` を追加する。
2. `MetricComputation` を返す `compute_<name>` を実装する。
3. `metric_defs.py` に `MetricDefinition` を登録する。
4. 必要な feature を `required_features` に書く。
5. `Scoring.md` に意味と使い分けを書く。
6. 同一形状、ずらした形状、不正入力の synthetic test を追加する。

metric は `ViewFeature` を受け取ります。
raw file を直接読んだり、runner に計算式を書いたりしないでください。

最小例:

```python
from wafergeo.compare.metric_types import MetricComputation, MetricContext


def compute_example(context: MetricContext) -> MetricComputation:
    sim = context.sim
    target = context.target
    value = float((sim.mask != target.mask).mean())
    return MetricComputation(
        name="example",
        value=value,
        loss=value,
        details={"meaning": "mask mismatch fraction"},
    )
```

`metric_defs.py`:

```python
"example": MetricDefinition(
    "example",
    frozenset({"sdf"}),
    compute_example,
    loss_scale=1.0,
)
```

## 新しい output

新しい output は、CSV/JSON/NPZ のどれかを優先します。
後段解析で直接読める形式にしてください。

| output | 向いている内容 |
| --- | --- |
| CSV | case ごとの table、summary、ranking |
| JSON | metadata、objective、詳細 dict |
| NPZ | feature tensor |
| PNG | 補助確認だけ |

## やらないこと

- 新しい workflow を先に増やす。
- profile / preset / hidden config layer を増やす。
- runner に loader / feature / metric の詳細を書く。
- 重い実データ test を通常 test suite に入れる。

小さく追加し、CSV/JSON/NPZ で効果を確認し、必要なら次の手法を追加してください。
