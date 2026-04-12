# 改良計画

今後の改良計画は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) に集約します。
この文書は、実装時に迷わないための短い要約です。

## 目的

`wafergeo` は、形状データを複数の特徴量化手法で変換し、実測データ比較や
サロゲート学習用 dataset 作成に使える出力を作るためのパッケージです。

サロゲート学習器や最適化器は内蔵しません。外部ツールが使える特徴量を出力します。

## workflow 方針

現在の実装済み workflow:

- `transform`
- `compare`
- `batch-compare`

計画済み workflow:

- `batch-transform`
- `transform-eval`
- `compare-eval`

特徴量化系と比較系を分けます。

| 系統 | workflow | 目的 |
| --- | --- | --- |
| 特徴量化 | `transform`, `batch-transform`, `transform-eval` | 3D field feature と dataset 作成 |
| 比較 | `compare`, `batch-compare`, `compare-eval` | 2D observation view 上の比較と ranking |

## 優先実装

| 優先 | 内容 | 理由 |
| --- | --- | --- |
| P1 | `sdf_raw` | 距離場特徴量の基準 |
| P1 | `tsdf_views` | 学習に使いやすい multi-scale 表現 |
| P1 | `udf` | contour / open contour 由来の実測データ向け |
| P2 | `material_sdf` | material-aware な 3D 特徴量 |
| P2 | `batch-transform` | サロゲート学習 dataset 作成 |
| P3 | `transform-eval` | 特徴量生成品質とコストの比較 |
| P3 | `compare-eval` | feature / metric 候補の比較性能評価 |

## 戻さないもの

次の旧概念は通常導線に戻しません。

- `manifest`
- `report`
- `surrogate`
- `assimilation`
- `benchmark`
- `preview`
- `audit`

## 実装時の判断

- feature は `transform` 系に追加する。
- metric は `compare` 系に追加する。
- eval は候補比較の summary を出すだけにし、自動採用しない。
- CSV/JSON/NPZ を正式出力にする。
- PNG は必要になってから補助として追加する。
