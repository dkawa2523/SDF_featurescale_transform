# 改良計画

この文書は、現在の `wafergeo` の設計を崩さずに、今後の改良を進めるための計画です。

`sdf_extension_docs_v5` は有用な拡張案を含みますが、既存コードを知らない前提で作られており、
現在のシンプル化方針とは合わない案も含まれています。したがって、この文書では外部案を
そのまま取り込まず、現在の正式 workflow に自然に入るものだけを採用候補にします。

## 1. 現在の設計で守ること

現在の public workflow は次の 3 つです。

| workflow | 目的 |
| --- | --- |
| `transform` | simulation 入力を特徴量化する |
| `compare` | 1 件の simulation と target を比較する |
| `batch-compare` | 複数 case を比較し、ranking を出す |

この3つを維持し、次の概念は通常のユーザー導線に戻しません。

- `manifest`
- `report`
- `surrogate`
- `assimilation`
- `benchmark`
- `preview`
- `audit`

新しい機能は、原則として次のどれかに分類して追加します。

| 分類 | 追加場所 |
| --- | --- |
| 入力形式 | loader |
| 特徴量 | feature |
| 評価指標 | metric registry |
| 出力 | output artifact |
| 説明 | docs |

runner と CLI は orchestration に専念させ、計算ロジックを入れません。

## 2. 外部案から採用するもの、しないもの

| 外部案の要素 | 現在コードでの扱い | 判断 |
| --- | --- | --- |
| raw SDF を中心にする | まず内部 feature 品質の改善として扱う | 採用候補 |
| TSDF / tanh / logsigned view | `transform` の追加出力として扱う | 採用候補 |
| profile metric | `cd` の自然な拡張として扱う | 優先採用 |
| corner metric | optional metric として後段で扱う | 条件付き採用 |
| MetricBundle | `metric_details.json` の構造改善として吸収する | public API としては不採用 |
| open contour の unsigned distance | `contour_json` 比較の精度改善として扱う | 採用候補 |
| Field3D | 今すぐ導入せず、重複が増えたら内部型として検討する | 保留 |
| observer 層 | `ViewFeature` helper に吸収する | 原則不採用 |
| surrogate export | 本パッケージの外部利用に任せる | 不採用 |
| assimilation | 本パッケージ内では最適化しない | 不採用 |
| manifest / report | 複雑化要因のため戻さない | 不採用 |

## 3. アーキテクト観点の改良方針

アーキテクト観点では、最大の目的は **機能追加のたびに構造を増やさないこと**です。

### 3.0 ユーザーと第三者開発者に対する設計原則

このパッケージは、利用者と改修者の両方にとって、次の状態を目指します。

| 視点 | 望ましい状態 |
| --- | --- |
| ユーザー | YAML を1つ指定すれば実行でき、出力 CSV/JSON/PNG を見れば結果を理解できる |
| データサイエンティスト | metric の意味、単位、集約方法、ranking への効き方を追える |
| 第三者開発者 | loader / feature / metric / output のどこを触ればよいかすぐ分かる |
| アーキテクト | public workflow、YAML、runner が肥大化していない |

追加機能は「ユーザーが覚える概念」を増やすのではなく、既存の `features.use` や
`metrics.use` の候補を増やす形を優先します。

新しい手法が便利でも、次の状態になるなら採用を保留します。

- YAML の階層が深くなる。
- runner に分岐が増える。
- 出力の意味を docs で短く説明できない。
- 既存 metric と何が違うか説明できない。
- テストを大量に追加しないと守れない。

### 3.1 判断基準

新しい要望が来たら、次の順に判断します。

1. `transform / compare / batch-compare` のどれかで表現できるか。
2. YAML のトップレベルを増やさずに表現できるか。
3. 追加箇所を loader / feature / metric / output に閉じ込められるか。
4. runner や CLI に計算ロジックを書かずに済むか。
5. docs に短く説明できるか。

この5つを満たせない場合は、実装前に設計確認を行います。

### 3.2 リファクタリングの開始条件

抽象化は早く入れすぎない方針です。次の条件を満たしたときだけ、内部型や helper を増やします。

| 条件 | 対応 |
| --- | --- |
| 同じ SDF 計算が3か所以上に重複した | 小さな SDF helper に切り出す |
| metric ごとに同じ profile 抽出が重複した | profile helper を追加する |
| `ViewFeature` の責務が曖昧になった | `ViewFeature` のフィールド意味を docs と型で整理する |
| 出力 writer が runner に漏れ始めた | `output_artifacts` に寄せる |
| YAML validation が分散した | schema loader に集約する |

逆に、次の理由だけでは抽象化しません。

- 将来使いそうだから。
- 外部案に書いてあるから。
- きれいな architecture に見えるから。
- サロゲートや同化でいつか必要そうだから。

### 3.3 複雑化を防ぐ実装予算

各 issue は、原則として次の予算内に収めます。

| 項目 | 目安 |
| --- | --- |
| 変更レイヤ | 1つ。例: metric だけ、feature だけ |
| 新規 public YAML 項目 | 0。必要な場合でも `metrics.<name>` または `features.use` 内に限定 |
| 新規 module | 1つまで |
| 新規 output | CSV/JSON を優先。PNG は補助 |
| 新規 test | 2から4件程度 |
| 新規 docs | 既存 docs への追記を優先 |

この予算を超える場合は、作業を分割します。

たとえば `profile` metric なら、最初の issue では `width / center / left/right edge`
までに限定し、wall angle や curvature は別 issue にします。

### 3.4 テストの増やし方

テストは「現在の public behavior を守るため」に追加します。

| 変更 | 最小テスト |
| --- | --- |
| loader 追加 | 正常入力と代表的な不正入力 |
| metric 追加 | 自己比較で良い値、ずれた形状で悪化 |
| output 追加 | ファイル生成と主要 CSV/JSON フィールド |
| docs 変更 | `mkdocs build --strict` |

重い dataset test、画像完全一致 test、旧導線の復活 test は追加しません。

### 3.5 例外処理の増やし方

例外処理は、ユーザーが直せる失敗を分かりやすくするために追加します。

| 失敗 | 方針 |
| --- | --- |
| YAML の未知 key / unknown metric | load 時に `ValueError` |
| 入力 shape / axis / units の不整合 | loader または schema で `ValueError` |
| metric が対象外 view だった | `SKIPPED` と `metric_details.json` に理由 |
| SciPy など optional dependency 不在 | 明確な `ImportError` または低速 fallback |
| 内部計算エラー | 握りつぶさず失敗させる |

`except Exception` で広く握りつぶす実装は避けます。

## 4. データサイエンティスト観点の活用方針

データサイエンティスト観点では、今の機能だけでも次の使い方ができます。

### 4.1 断面 CD 評価

目的:

- 高さごとの幅を評価する。
- 左右 edge のずれを見る。
- 内部 material boundary のずれを見る。

推奨設定:

```yaml
view:
  axes: [x, z]
  depth_axis: y

features:
  use: [contour, sdf]

metrics:
  use: [cd, sdf, sdf_band, iou]
```

見る出力:

| 出力 | 使い方 |
| --- | --- |
| `cd_profile.csv` | 高さごとの幅、左右 edge 差を見る |
| `cd_profile.png` | どの高さで差が大きいかを素早く見る |
| `cd_profile_summary.json` | CD の集約値を見る |
| `difference.png` | material 不一致の場所を確認する |

### 4.2 material 差分評価

目的:

- 外形ではなく、材料配置や内部構造の違いを見る。
- topmost 投影では見えない material も projected material mask で拾う。

推奨設定:

```yaml
features:
  use: [sdf]

metrics:
  use: [sdf_material, sdf_band, iou]
```

見る出力:

| 出力 | 使い方 |
| --- | --- |
| `per_material_sdf.csv` | どの material が誤差を支配しているかを見る |
| `metric_details.json` | material ごとの SDF loss と集約方法を見る |
| `difference_summary.json` | 不一致 pixel 数や差分概要を見る |

### 4.3 輪郭比較

目的:

- 外部で抽出した輪郭座標と simulation を比較する。
- SEM 特化ではなく、一般の contour JSON として扱う。

推奨設定:

```yaml
features:
  use: [contour, sdf]

metrics:
  use: [chamfer, sdf, iou]
```

注意:

- `contour_json` が open contour の場合、現在は closed contour より解釈が難しいです。
- 今後の改善では、open contour を unsigned distance として扱う方針が有力です。

### 4.4 batch ranking

目的:

- 複数 simulation case を target と比較して順位付けする。
- どの metric が ranking に効いているかを確認する。

見る出力:

| 出力 | 使い方 |
| --- | --- |
| `ranking.csv` | case の順位を見る |
| `metric_summary.csv` | metric ごとの分布を見る |
| `ranking_top.png` | 上位 case を素早く確認する |
| `cases/<case_id>/score.json` | 個別 case の詳細を見る |

## 5. 実装ロードマップ

### Phase 1: profile metric を追加する

目的:

- CD を単一値ではなく、断面 profile としてより使いやすくする。
- 幅、center shift、left/right edge、必要なら wall angle を扱う。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| metric | `profile` を metric registry に追加 |
| helper | `ViewFeature` から profile rows を作る小 helper |
| output | `profile.csv`, `profile_summary.json` |
| docs | `Scoring.md`, `UserManual.md` に追記 |
| tests | synthetic 断面の自己比較、幅差、center shift |

やらないこと:

- 新しい workflow を作らない。
- `MetricBundle` を public API にしない。
- YAML のトップレベルを増やさない。

受け入れ条件:

- `metrics.use: [profile]` で実行できる。
- 自己比較で loss が 0 になる。
- 幅差や center shift で loss が悪化する。
- `compare` と `batch-compare` の既存出力が壊れない。

### Phase 2: open contour の unsigned distance 対応

目的:

- `contour_json` の `closed: false` を自然に扱う。
- open contour を無理に mask 化せず、polyline distance として比較する。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| feature | open contour から unsigned distance map を作る |
| metric | `sdf` / `sdf_band` が distance semantics を見る |
| output | `metric_details.json` に `distance_semantics` を出す |
| docs | `contour_json` の closed/open の意味を明記 |
| tests | open contour smoke test と invalid contour test |

やらないこと:

- SEM 専用 API を作らない。
- open contour で IoU を無理に計算しない。

受け入れ条件:

- `closed: false` contour で compare が失敗しない。
- open contour では IoU が適切に `SKIPPED` になるか、明確な fallback になる。
- distance metric の意味が `metric_details.json` で追える。

### Phase 3: transform の SDF view 出力を追加する

目的:

- 外部のサロゲート学習や解析で使いやすい SDF 派生特徴量を出す。
- ただし `surrogate` workflow は作らない。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| feature | `sdf_views` を追加 |
| output | `features/sdf_views.npz` |
| docs | `2_sdf.md`, `UserManual.md` に説明 |
| tests | 小さい NPZ で出力 shape と値域を確認 |

初期出力候補:

- `sdf_nm`
- `tsdf_10nm`
- `tsdf_50nm`
- `log_abs_sdf`

やらないこと:

- `Field3D` を public API にしない。
- 学習用 dataset builder を作らない。
- manifest を戻さない。

受け入れ条件:

- `features.use: [sdf_views]` で transform が動く。
- 出力の shape、spacing、origin が追える。
- 既存 `sdf` 出力と混乱しない名前になっている。

### Phase 4: corner metric を optional に追加する

目的:

- bottom corner、shoulder、rounding など局所形状差を拾う。
- CD や SDF だけでは説明しにくい差分を補助する。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| metric | `corner` を metric registry に追加 |
| helper | 2D mask/profile から代表 corner を抽出 |
| output | `corner_summary.json` |
| docs | 用途と制約を説明 |
| tests | 単純矩形と corner shift の synthetic test |

やらないこと:

- 曲率推定を最初から複雑にしない。
- process 専用名を API に入れない。
- topview で無理に corner を定義しない。

受け入れ条件:

- 断面 `[x,z]` または `[y,z]` で動く。
- corner が検出できない場合は明確に `SKIPPED` になる。
- 他 metric の結果に影響しない。

### Phase 5: 内部 SDF helper の整理

目的:

- SDF 計算や distance semantics が複数箇所に重複した場合だけ、内部 helper に切り出す。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| helper | signed / unsigned / clipped SDF を小さく整理 |
| docs | 内部 contract を `2_sdf.md` に追記 |
| tests | helper 単位の最小 test |

やらないこと:

- 最初から `Field3D` を全面導入しない。
- `wafergeo/observe` や `wafergeo/metrics` を復活しない。

受け入れ条件:

- 重複が減る。
- public YAML が変わらない。
- 既存 metric の数値が大きく変わらない。

### Phase 6: 実データ評価 smoke の固定

目的:

- 新しい metric や SDF helper の追加後に、実データに近い dataset で退行を早く見つける。
- 旧 `benchmark` pipeline は復活させず、正式 workflow の `batch-compare` だけで確認する。

実装範囲:

| 種別 | 内容 |
| --- | --- |
| config | `configs/runs/dataset_t08_vs_run0010.yaml` |
| index | `configs/runs/dataset_t08_vs_run0010_pairs.csv` |
| docs | `RealDataEvaluation.md` に実行方法と結果確認手順を書く |
| tests | config と参照 path の存在確認だけを行う |

やらないこと:

- heavy dataset 実行を通常 test に入れない。
- 実行結果を git 管理しない。
- 新しい public workflow を増やさない。
- 性能測定 framework を作らない。

受け入れ条件:

- `batch-compare` で 11 case が実行できる。
- `run_0010` の自己比較が ranking 1 位になる。
- 代表 metric がすべて `metrics.csv` に出る。
- 実行結果は `outputs/` 配下だけに作られる。

## 6. 実装 issue テンプレート

新しい作業を切るときは、次のテンプレートを使います。

```text
AGENTS.md と docs/MaintenancePolicy.md を守る。
対象は [loader / feature / metric / output / docs / tests] のみ。
新しい public workflow は追加しない。
manifest/report/surrogate/assimilation/benchmark/preview/audit は復活させない。
YAML のトップレベル構造 task/input/view/features/metrics/output は維持する。

目的:
- ...

変更範囲:
- ...

受け入れ条件:
- ...

テスト:
- ...

やらないこと:
- ...
```

## 7. 実装前ゲート

実装に入る前に、次を確認します。

| ゲート | 通す条件 |
| --- | --- |
| ユーザー価値 | 何が分かりやすくなるかを1文で説明できる |
| DS 価値 | 既存出力では見えない差、または解釈しにくい差を補える |
| 追加場所 | loader / feature / metric / output / docs / tests のどれかに閉じている |
| YAML | 既存トップレベル構造を維持している |
| 出力 | CSV/JSON が主で、PNG は補助になっている |
| テスト | synthetic test で守れる |
| 削除/整理 | 古い概念や生成物を増やしていない |

1つでも満たせない場合は、実装せずに設計確認へ戻します。

## 8. 次に進める推奨順

推奨順は次の通りです。

1. `profile` metric
2. open contour の unsigned distance
3. `sdf_views` transform 出力
4. `corner` metric
5. SDF helper 整理
6. 実データ評価 smoke の固定

ここまで進んだ後は、新機能を増やすより、実データ評価 smoke で各 metric の有用性と重さを見てから
次の手法を選びます。
