# 保守運用ポリシー

この文書は、`wafergeo` を今後もシンプルに保ちながら、手法追加や改良を続けるための方針です。

特に Codex などの coding agent に作業を依頼するときは、指示が広すぎると
「親切な過剰実装」が起きやすくなります。新しい入口、設定層、互換処理、過度なテストが増えると、
第三者がコードを理解しにくくなります。

## 1. 守るべき現在の形

通常ユーザー向けの workflow は次の 3 つに固定します。

| workflow | 目的 |
| --- | --- |
| `transform` | 1 つの simulation 入力を特徴量化する |
| `compare` | 1 つの simulation と 1 つの target を比較する |
| `batch-compare` | 複数 case を比較して ranking を出す |

通常のユーザーは、YAML と `outputs/` だけを意識すればよい状態を維持します。

## 2. 増やしてよいもの、増やさないもの

増やしてよいものは、既存 workflow の中に自然に収まるものです。

| 追加したいもの | 追加場所 |
| --- | --- |
| 入力形式 | loader |
| 特徴量変換 | feature |
| 評価指標 | metric registry |
| 軽い確認用出力 | output artifact |
| 説明 | docs |

増やさないものは、ユーザーが覚える入口や概念を増やすものです。

| 避けるもの | 理由 |
| --- | --- |
| 新しい public pipeline | 実行入口が増えて迷いやすい |
| `manifest` の復活 | 入力設定と生成物の区別が曖昧になる |
| `report` の復活 | 出力責務が膨らみやすい |
| `surrogate` / `assimilation` の内蔵 | 本パッケージの主目的から外れやすい |
| 深い YAML profile | 人手編集が難しくなる |
| 重い実データテスト | 通常検証が遅くなり、保守されにくい |

## 3. 変更前の判断ルール

実装前に、変更内容を必ず次のどれかに分類します。

| 分類 | 判断 |
| --- | --- |
| loader | 新しいファイル形式、座標形式、入力規約を扱う |
| feature | 入力データを SDF、contour、mesh、slice などへ変換する |
| metric | 既存 feature から評価値を計算する |
| output | 既に計算された結果を CSV、JSON、PNG で出す |
| docs | 使い方、設計意図、拡張方法を説明する |

どれにも分類できない場合は、設計が広がりすぎている可能性があります。

## 4. Codex への依頼テンプレート

Codex に作業を依頼するときは、次の文を先頭に付けると過剰実装を防ぎやすくなります。

```text
AGENTS.md の方針を必ず守ってください。
公式 workflow は transform / compare / batch-compare の3つだけです。
新しい public workflow、manifest、report、surrogate、assimilation、benchmark、preview、audit を復活させないでください。
YAML のトップレベル構造は task / input / view / features / metrics / output を維持してください。
変更が loader / feature / metric / output / docs / tests のどれに属するかを判断してから実装してください。
不要な互換処理、過度なテスト、生成物の commit は避けてください。
```

## 5. 依頼の粒度を小さくする

広すぎる依頼は避けます。

避けたい依頼:

```text
残件を全部実装してください。不要なものも削除してください。
```

望ましい依頼:

```text
AGENTS.md を守って、metric registry だけを確認してください。
新しい public workflow は追加しないでください。
必要なら metric 追加と最小テストだけにしてください。
```

さらに安全にしたい場合は、変更予算を明示します。

```text
今回の変更は最大3ファイルまで。
新規 public YAML 項目は追加しないでください。
テスト追加は1件まで。
```

## 6. テストを増やしすぎない方針

テストは「壊れやすい仕様を守るため」に追加します。

| 対象 | 最小テスト |
| --- | --- |
| loader | 正常入力、代表的な不正入力 |
| feature | 小さい synthetic データで期待 shape と主要値 |
| metric | 同一形状で score が良い、ずれた形状で悪化 |
| runner | 公式 YAML の smoke test |
| docs | `mkdocs build --strict` |

画像出力は完全一致を避けます。ファイル生成、凡例、主要 JSON/CSV 値を確認する方が保守しやすいです。

## 7. 出力物を Git に入れない

次のものは生成物です。

- `outputs/`
- `site/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- 一時的な実験ディレクトリ

公式 example に必要な小さな入力データだけを `data/examples/` に置きます。

## 8. レビュー観点

PR や Codex の作業結果を見るときは、次を確認します。

| 観点 | 確認すること |
| --- | --- |
| ユーザー視点 | 実行入口や YAML が増えていないか |
| データサイエンティスト視点 | 出力 CSV/JSON が評価や分析に使いやすいか |
| 開発者視点 | 追加箇所が loader / feature / metric / output に閉じているか |
| アーキテクト視点 | runner や CLI に計算ロジックが漏れていないか |
| 保守視点 | テストと docs が必要最小限か |

## 9. 判断に迷ったとき

迷ったときは、次の順で判断します。

1. 既存 workflow の中で表現できるか。
2. YAML を増やさずにデフォルトで動くか。
3. registry や小さな helper に閉じ込められるか。
4. 出力は CSV/JSON を主にし、PNG は補助にできるか。
5. docs に1段落で説明できるか。

この5つを満たせない場合は、いったん実装を止めて設計確認を行います。
