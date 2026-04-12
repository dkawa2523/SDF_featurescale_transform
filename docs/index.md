# wafergeo ドキュメント

`wafergeo` は、シミュレーションや実験由来の形状データを
**特徴量化**し、必要に応じて **形状比較の指標を計算**するためのパッケージです。

このドキュメントは、初めて使う人と、あとから手法を追加する開発者の両方が迷わないように、
「どこを編集するか」「何が出力されるか」「どこを改修すればよいか」を中心に整理しています。

## まず読む資料

| 読むもの | 対象 | 内容 |
| --- | --- | --- |
| [ユーザーマニュアル](UserManual.md) | 利用者 | 実行方法、YAML の書き方、出力の読み方 |
| [クイックスタート](Quickstart.md) | 利用者 | 最短で `transform / compare / batch-compare` を動かす手順 |
| [設定の所在マップ](SettingsMap.md) | 利用者、開発者 | どこを設定し、どこが生成物か |
| [開発者マニュアル](DeveloperManual.md) | 改修者 | コード全体の読み方、変更判断、検証方法 |
| [拡張ガイド](ExtensionGuide.md) | 改修者 | loader / feature / metric の追加手順 |
| [特徴量化・評価ロードマップ](WorkflowRoadmap.md) | 利用者、改修者、データサイエンティスト | transform / compare 系 workflow の将来像、評価指標、実装順 |
| [保守運用ポリシー](MaintenancePolicy.md) | 改修者、Codex 利用者 | 設計の肥大化、過剰実装、過度なテストを防ぐルール |
| [改良計画](ImprovementPlan.md) | 改修者、データサイエンティスト | 現在の設計を崩さずに進める機能改善ロードマップ |
| [手法調査と実装計画](MethodResearch.md) | 改修者、データサイエンティスト | 新しい手法の効果、実装場所、検証方法を具体化 |
| [実装リスク対策](RiskControlPlan.md) | 改修者、Codex 利用者 | 新手法を安全に実装するための段階ゲートと撤退基準 |
| [実データ評価 smoke](RealDataEvaluation.md) | 改修者 | 実データに近い dataset で metric 退行を確認する開発者向け手順 |

## 正式な利用導線

現在の通常ユーザー向けの正式入口は次の 3 つです。

```powershell
python -m wafergeo run transform --config .\configs\examples\transform.simple.yaml
python -m wafergeo run compare --config .\configs\examples\compare.simple.yaml
python -m wafergeo run batch-compare --config .\configs\examples\batch-compare.simple.yaml
```

| workflow | 目的 | 主な出力 |
| --- | --- | --- |
| `transform` | 1 つの simulation 入力を特徴量化する | `features/`, `summary.json` |
| `compare` | 1 つの simulation と 1 つの target を比較する | `score.json`, `metrics.csv`, `difference.png` |
| `batch-compare` | 複数 case を比較し、順位付けする | `ranking.csv`, `metric_summary.csv`, `ranking_top.png` |

`manifest`, `report`, `surrogate`, `assimilation`, `benchmark`, `preview`, `audit`
は通常の利用導線には含めません。必要な出力は YAML と `outputs/` だけで追える設計にしています。

今後は、特徴量化と比較評価をそれぞれ「単一実行」「batch 実行」「手法評価」に分ける計画です。
詳細は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) を参照してください。

## 処理内容を理解する資料

| 資料 | 内容 |
| --- | --- |
| [入力 label volume](1_IngestLabel.md) | `npz_label` / `vti_label` の形、軸順、単位 |
| [View と 2D 投影](View.md) | 3D データを比較用の 2D 面に変換する考え方 |
| [SDF](2_sdf.md) | SDF 特徴量と SDF 系 metric の役割 |
| [Mesh](3_mesh.md) | mesh 特徴量の位置づけ |
| [Scoring](Scoring.md) | Primary / Diagnostic metric の意味と使い分け |

## Codex に作業を依頼するとき

Codex などの coding agent に作業を依頼するときは、root の `AGENTS.md` と
[保守運用ポリシー](MaintenancePolicy.md) を前提にしてください。

広い依頼ほど、不要な workflow、深い YAML、過度な互換処理、重いテストが増えやすくなります。
依頼時は「変更対象を loader / feature / metric / output / docs / tests のどれに限定するか」を
明示すると、現在の設計を崩さずに改良しやすくなります。

## MkDocs で見る

ローカルでドキュメントを確認する場合は、開発用依存を入れてから MkDocs を起動します。

```powershell
py -3.13 -m pip install -e ".[dev]"
py -3.13 -m mkdocs serve
```

静的サイトとして確認する場合は次を実行します。

```powershell
py -3.13 -m mkdocs build --strict
```

`site/` は生成物なので Git 管理には含めません。
