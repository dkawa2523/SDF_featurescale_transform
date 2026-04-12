# 手法調査と実装メモ

この文書は、新しい特徴量化手法や比較手法を追加するときのメモです。
全体方針と実装順は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) を優先してください。

## 採用したい手法の方向性

| 手法 | 主な用途 | 追加先 |
| --- | --- | --- |
| `sdf_raw` | 3D raw signed distance feature | feature |
| `tsdf_views` | clip 幅違いの学習用 distance view | feature |
| `udf` | open contour / SEM contour の unsigned distance | feature |
| `material_sdf` | material ごとの 3D distance field | feature |
| `sdf`, `sdf_band`, `sdf_material` | 2D observation 上の距離場比較 | metric |
| `profile`, `corner` | 断面観察の診断 | metric |

## 手法カードの最小テンプレート

新しい手法を実装する前に、次だけ決めます。

```text
name:
layer: feature | metric | loader | output
purpose:
input:
output:
when_skipped:
non_goals:
tests:
```

## 採用基準

採用する条件:

- 既存手法では見えない差を拾える。
- ユーザーが出力を見て判断できる。
- CSV/JSON/NPZ として後段解析に使える。
- 追加箇所が loader / feature / metric / output のどれかに閉じる。

保留する条件:

- workflow や YAML 階層を増やさないと説明できない。
- 既存手法との差が説明できない。
- heavy dataset test がないと壊れやすい。
- サロゲート学習器や最適化器の内蔵が必要になる。

## 検証の最小セット

| 検証 | 内容 |
| --- | --- |
| self comparison | 同一入力で最良になる |
| intended perturbation | 狙った差分で値が悪化する |
| unsupported input | 対象外は `SKIPPED` または明確な `ValueError` |
| workflow smoke | 公式 YAML から実行できる |

実データ評価は通常テストには入れず、必要なときだけ手動 smoke として行います。
