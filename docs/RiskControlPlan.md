# 実装リスク対策

この文書は、`wafergeo` に新しい特徴量化手法や比較手法を追加するときの最小ルールです。
詳細計画は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) を優先してください。

## 基本方針

| リスク | 対策 |
| --- | --- |
| YAML が複雑になる | 通常 workflow は `task/input/view/features/metrics/output` を維持する |
| eval が肥大化する | eval workflow だけ `eval.candidates` を許可する |
| runner が太る | runner は orchestration に限定する |
| 出力が増えすぎる | CSV/JSON/NPZ を正式出力、PNG は補助にする |
| 手法が default 化されすぎる | 新手法は明示指定時だけ動かす |
| テストが重くなる | synthetic test 中心、実データは手動 smoke にする |

## 実装前チェック

新しい作業は、次のどれかに分類します。

- loader
- feature
- metric
- output
- docs
- tests

分類できない場合は、設計が広がりすぎている可能性があります。

## 新手法の最小条件

新しい feature / metric は、実装前に次を決めます。

| 項目 | 内容 |
| --- | --- |
| 目的 | 既存手法より何を改善するか |
| 入力 | どの内部型を使うか |
| 出力 | どの CSV/JSON/NPZ を出すか |
| 対象外 | 何を今回はやらないか |
| テスト | 自己比較、意図した差分、不正入力のうち必要最小限 |

## 採用判断

次を満たすものから採用します。

- ユーザーに説明できる。
- 出力が後段解析に使える。
- 実装範囲が小さい。
- 既存 workflow の目的に合う。

次に当てはまる場合は保留します。

- 新しい概念を大量に増やす。
- metric と feature の責務が混ざる。
- runner に計算ロジックが漏れる。
- 重い dataset test なしでは守れない。

## 実装順

ロードマップ上の優先順は次です。

1. `sdf_raw` feature
2. `tsdf_views` feature
3. `udf` feature
4. `material_sdf` feature
5. `batch-transform`
6. `transform-eval`
7. `compare-eval`

細かい診断出力や可視化は、この順番を邪魔しない場合だけ追加します。
