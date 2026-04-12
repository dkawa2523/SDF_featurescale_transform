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

## 正式な利用導線

通常ユーザー向けの正式入口は次の 3 つです。

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

## 処理内容を理解する資料

| 資料 | 内容 |
| --- | --- |
| [入力 label volume](1_IngestLabel.md) | `npz_label` / `vti_label` の形、軸順、単位 |
| [View と 2D 投影](View.md) | 3D データを比較用の 2D 面に変換する考え方 |
| [SDF](2_sdf.md) | SDF 特徴量と SDF 系 metric の役割 |
| [Mesh](3_mesh.md) | mesh 特徴量の位置づけ |
| [Scoring](Scoring.md) | `cd`, `chamfer`, `sdf`, `sdf_material`, `sdf_band`, `iou` の意味 |

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
