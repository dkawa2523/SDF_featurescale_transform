# 保守運用ポリシー

この文書は、`wafergeo` を小さく保ちながら、特徴量化手法と比較手法を増やすための運用方針です。
詳細な将来計画は [特徴量化・評価ロードマップ](WorkflowRoadmap.md) を参照してください。

## 守ること

| 方針 | 内容 |
| --- | --- |
| 入口を用途で固定する | 特徴量化系と比較系の workflow だけを増やす |
| 出口を固定する | CSV/JSON/NPZ を正式出力にする |
| runner を薄くする | runner は loader / feature / metric / output を呼ぶだけにする |
| 手法を局所追加する | loader, feature, metric, output のどれかに閉じる |
| default を増やしすぎない | 新手法は明示指定時だけ動かす |

## workflow 方針

現在の実装済み workflow:

- `transform`
- `compare`
- `batch-compare`

計画済み workflow:

- `batch-transform`
- `transform-eval`
- `compare-eval`

この 6 つを超える public workflow は原則として追加しません。

## 追加してよいもの

| 追加したいもの | 追加場所 |
| --- | --- |
| 入力形式 | loader |
| 特徴量化手法 | feature |
| 比較指標 | metric registry |
| 結果ファイル | output writer |
| 説明 | docs |

## 戻さないもの

`manifest`, `report`, `surrogate`, `assimilation`, `benchmark`, `preview`, `audit`
は通常導線に戻しません。

サロゲート学習は外部で行います。このパッケージは、学習に渡せる特徴量 dataset を作ります。

## Codex への依頼文

広い作業を依頼するときは、次の方針を含めてください。

```text
WorkflowRoadmap.md と AGENTS.md を前提にしてください。
入口と出口を増やしすぎず、変更を loader / feature / metric / output / docs / tests のどれかに閉じてください。
runner と CLI に計算ロジックを入れないでください。
生成物、重いテスト、不要な互換処理は追加しないでください。
```

## レビュー観点

| 観点 | 確認すること |
| --- | --- |
| ユーザー | YAML と出力の見方が分かりやすいか |
| データサイエンティスト | 評価指標と特徴量の意味、単位、shape が追えるか |
| 開発者 | 追加箇所が明確か |
| 運用 | runtime、出力サイズ、生成物管理が破綻しないか |
