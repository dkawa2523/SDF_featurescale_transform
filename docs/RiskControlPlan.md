# 実装リスク対策

この文書は、`ImprovementPlan.md` と `MethodResearch.md` の内容を実装へ進めるときに、
コードや仕様が無駄に肥大化しないようにするためのリスク対策です。

結論から言うと、複数の新手法を一括で実装しません。
1つの issue では、1つの手法、1つのレイヤ、必要最小限の出力だけを扱います。

## 1. 主なリスクと対策

| リスク | 起きる問題 | 対策 |
| --- | --- | --- |
| YAML が複雑になる | ユーザーが何を設定すべきか分からなくなる | 追加設定は `metrics.<name>` または `features.use` に限定する |
| 出力が増えすぎる | どのファイルを見ればよいか分からなくなる | authoritative data は CSV/JSON、PNG は補助に限定する |
| metric が重複する | total score の意味が曖昧になる | 新 metric は default に入れず、明示指定時だけ使う |
| runner が肥大化する | orchestration と計算ロジックが混ざる | 計算は metric / feature module に閉じる |
| `ViewFeature` が万能化する | 中間表現が理解しにくくなる | 新しい field は必要性を確認してから追加する |
| テストが増えすぎる | 第三者が把握しにくく、CI が重くなる | synthetic test を中心に 2-4 件へ抑える |
| 例外処理が過剰になる | 内部エラーが隠れる | user error は `ValueError`、対象外は `SKIPPED`、内部エラーは落とす |
| 旧概念が復活する | manifest/report/surrogate/assimilation に戻る | AGENTS.md の禁止事項を守る |

## 2. 実装を進める段階

### Stage -1: 実装開始チェック

コードを編集する前に、Codex は次の内容をユーザーへ提示します。

| 項目 | 内容 |
| --- | --- |
| 対象レイヤ | `loader / feature / metric / output / docs / tests` のどれか |
| 追加するもの | 手法名、YAML 名、出力ファイル名 |
| 変えないもの | public workflow、default metric、YAML トップレベル、runner の責務 |
| 確認事項 | 勝手に決めると危険な仕様だけ |
| 仮定 | 確認不要と判断して進める前提 |
| テスト範囲 | synthetic test と smoke test の範囲 |

提示例:

```text
対象レイヤ: metric
追加するもの: profile metric。metrics.use: [profile] で明示指定された場合だけ動く。
出力: profile.csv, profile_summary.json
変えないもの: default metrics, public workflow, YAML top-level, runner/CLI
確認事項: なし。height_axis は z、width_axis は view 内の非 z 軸をデフォルトにする。
テスト: 自己比較、幅差、center shift の synthetic 3件。
```

確認事項がない場合は、仮定を明示してそのまま実装します。
確認事項がある場合でも、質問は具体的な1から3点に絞ります。

### Stage 0: 手法カード

実装前に、`MethodResearch.md` の形式で手法カードを作ります。

必須項目:

- 何を測るか。
- 既存 metric と何が違うか。
- どの入力と view で使うか。
- 出力 CSV/JSON は何か。
- どの条件で `SKIPPED` になるか。
- やらないことは何か。

この段階ではコードを書きません。

### Stage 1: 非 default 実装

新手法は、最初から default metric に入れません。

例:

```yaml
metrics:
  use: [profile]
```

ユーザーが明示した場合だけ動くようにします。

受け入れ条件:

- 既存 example の default 結果が変わらない。
- `metrics.use` に指定した場合だけ新手法が実行される。
- YAML のトップレベル構造が増えていない。

### Stage 2: synthetic 検証

小さい `npz_label` だけで検証します。

最小テスト:

- 自己比較で最良になる。
- 意図した perturbation で悪化する。
- 対象外 input / view では `SKIPPED` または明確な `ValueError` になる。

画像の完全一致テストは行いません。

### Stage 3: example 検証

`configs/examples/` の既存 workflow で確認します。

確認項目:

- `transform / compare / batch-compare` が壊れていない。
- `score.json` と `metrics.csv` が読める。
- 新しい出力ファイルの名前と内容が docs と一致する。
- `metric_details.json` に評価条件が残る。

### Stage 4: DS レビュー

データサイエンティスト観点で、手法が本当に有用かを確認します。

確認項目:

| 観点 | 確認内容 |
| --- | --- |
| 差分検出 | 既存 metric では見えない差を拾えるか |
| 解釈性 | CSV/JSON から原因を説明できるか |
| 安定性 | 小さいノイズや tiny material に過剰反応しないか |
| ranking | `normalized_total_score` を悪く歪めないか |
| 相関 | 既存 metric と完全に同じ情報になっていないか |

有用性が弱い場合は、default 化せず experimental 扱いにします。

### Stage 4.5: 実データ smoke

synthetic test で守れた後に、必要な場合だけ実データに近い小さな dataset で確認します。
これは通常 CI に入れる heavy test ではなく、開発者が手元で実行する退行確認です。

実行:

```powershell
py -3.13 -m wafergeo run batch-compare --config .\configs\runs\dataset_t08_vs_run0010.yaml
```

確認項目:

- `run_0010` の自己比較が `ranking.csv` の 1 位になる。
- 追加した metric が `metrics.csv` に出る。
- `status` が想定外に `SKIPPED` になっていない。
- `outputs/` のファイル数と容量が過剰に増えていない。
- 実行結果を commit しない。

### Stage 5: 採用または撤退

採用する条件:

- ユーザーに説明できる。
- 出力が後段分析に使える。
- 実装箇所が小さい。
- テストが小さい。
- 既存 workflow を壊していない。

撤退または保留する条件:

- YAML が深くなる。
- 出力が多すぎる。
- 既存 metric と差がない。
- batch 実行で重すぎる。
- 実装が runner に漏れる。

## 3. 手法別のリスク制御

| 手法 | 主なリスク | 制御方法 |
| --- | --- | --- |
| `profile` | `cd` と役割が重なる | default に入れず、`profile.csv` で診断用に始める |
| open contour unsigned distance | IoU/SDF の意味が混ざる | distance semantics を `metric_details.json` に出し、不適切な IoU は `SKIPPED` |
| `sdf_views` | 3D dense 出力で容量が増える | `transform` の明示指定時だけ出力し、圧縮 NPZ に限定 |
| `corner` | 形状定義が曖昧になりやすい | 断面 view のみ、最初は位置差だけ |
| material confusion summary | metric が増えたように見える | score には入れず、診断 CSV として出す |
| topology | 実装と依存が重くなりやすい | 2D component count の最小実装に限定し、3D topology や persistent homology は入れない |

## 4. Codex に実装を依頼するときの追加文

新手法を実装するときは、通常の依頼文に次を追加します。

```text
docs/RiskControlPlan.md を守ってください。
今回実装する手法は1つだけです。
default metrics には追加しないでください。
YAML のトップレベル構造は増やさないでください。
runner / CLI に計算ロジックを入れないでください。
出力は CSV/JSON を主にし、PNG は補助にしてください。
テストは synthetic 2-4 件に抑えてください。
実装前に、対象レイヤ、追加する YAML 名、出力ファイル、やらないこと、確認事項を提示してください。
```

## 5. PR 前チェックリスト

実装後に次を確認します。

- `metrics.use` または `features.use` に明示した場合だけ新機能が動く。
- 既存 example YAML の結果が壊れていない。
- `score.json` と `metrics.csv` の意味が変わっていない。
- 新しい出力は docs に説明がある。
- 新しい出力は `outputs/` 以外に生成されない。
- `outputs/` と `site/` を commit していない。
- `ruff`, `mypy`, `pytest` が通る。
- docs を変更した場合、`mkdocs build --strict` が通る。
- `.github/pull_request_template.md` の Scope Guard を満たしている。

## 6. 最初に進める安全な単位

最初に進めるなら、`profile` metric を次の範囲に限定します。

やること:

- `metrics.use: [profile]` で明示指定された場合だけ動かす。
- 断面 view `[x,z]` または `[y,z]` のみ対応する。
- `profile.csv` と `profile_summary.json` を出す。
- width、center、left/right edge の差だけを評価する。

やらないこと:

- default metric に入れない。
- wall angle や curvature は入れない。
- `MetricBundle` は作らない。
- 新しい workflow は作らない。
- heavy dataset test は入れない。

この範囲なら、現在の思想を崩さずに有用性を確認できます。
