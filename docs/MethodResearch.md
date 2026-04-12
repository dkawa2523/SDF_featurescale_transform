# 手法調査と実装計画

この文書は、新しい手法を追加するときに、データサイエンス上の効果が得られる形で
調査・設計・実装するための計画です。

前提として、public workflow は `transform / compare / batch-compare` の 3 つだけを維持します。
新しい手法は、新しい workflow ではなく、原則として **feature** または **metric** として追加します。

## 1. データサイエンス観点の目的

このパッケージで新手法を追加する目的は、単に metric を増やすことではありません。

目的は次の 4 つです。

| 目的 | 説明 |
| --- | --- |
| 評価の分解 | total score だけでなく、どの形状差が効いたかを説明する |
| 形状差の検出 | CD だけでは見えない内部 material、sidewall、corner、局所差を拾う |
| ranking の安定化 | ノイズや小領域に過剰反応せず、実験判断に使える順位を出す |
| 外部利用 | サロゲート学習や後段解析に使える特徴量を `transform` で出す |

そのため、新しい手法は必ず次を満たす必要があります。

- 自己比較で良い値になる。
- 意図した perturbation で悪化する。
- 既存 metric と違う情報を持つ。
- CSV/JSON で後段分析できる。
- ユーザーが YAML を深く書かなくてもデフォルトで動く。

## 2. 手法追加の基本ルール

| 追加したいもの | 実装場所 | public YAML |
| --- | --- | --- |
| 新しい評価指標 | `wafergeo.compare.metric_defs` と小さな metric module | `metrics.use` に名前を追加 |
| 新しい特徴量 | `feature_outputs` または `transform_features` | `features.use` に名前を追加 |
| 新しい出力 | `output_artifacts` | 原則デフォルト出力。必要な場合だけ `output` に最小 option |
| 新しい入力 | loader registry | `input.kind` に名前を追加 |

runner や CLI に計算ロジックを入れません。

`manifest`, `report`, `surrogate`, `assimilation`, `benchmark`, `preview`, `audit`
は通常の public path に戻しません。

## 2.1 手法カードの採否基準

新しい手法は、実装前に必ず手法カードとして整理します。

採用する条件:

- 既存 metric では見えない差を拾える。
- 既存 metric より解釈しやすい。
- ranking の安定性が上がる。
- 後段解析で使える CSV/JSON が出せる。
- 追加先が metric / feature / output のどれかに閉じる。

保留または不採用にする条件:

- 新しい workflow が必要になる。
- YAML のトップレベル構造を増やす必要がある。
- `manifest` や `report` を前提にしないと説明できない。
- 重い実データ test がないと正しさを守れない。
- 既存 metric との差分が説明できない。

## 2.2 実装しやすさの観点

第三者が手法を追加しやすい状態とは、次の状態です。

| 条件 | 内容 |
| --- | --- |
| 入口が明確 | metric なら `metric_defs`、feature なら `feature_outputs` を見ればよい |
| 依存が少ない | 新手法が runner、schema、output に広く散らばらない |
| 出力が予測できる | CSV/JSON の列名と単位が docs で分かる |
| 失敗が分かる | `SKIPPED` や `ValueError` の理由がユーザーに伝わる |
| テストが小さい | synthetic データだけで基本挙動を確認できる |

この条件を満たさない手法は、まず helper や docs の設計を見直します。

## 3. 効果検証の標準プロトコル

新しい手法は、実装前に「何に効くか」を明確にし、実装後に次の順で検証します。

### 3.1 synthetic 検証

小さい `npz_label` を作り、意図した差だけを入れます。

| case | 目的 |
| --- | --- |
| identical | 自己比較で loss が 0 または最良になること |
| shift | 位置ずれで loss が悪化すること |
| width change | CD / profile 系が幅差を拾うこと |
| center shift | 幅が同じでも edge 位置差を拾うこと |
| material swap | 外形が同じでも material 差を拾うこと |
| hidden material | topmost では見えにくい material 差を拾うこと |
| open contour | open contour が無理に closed mask 扱いされないこと |

### 3.2 example 検証

`configs/examples/` と `data/examples/` で smoke test します。

確認するもの:

- コマンドが成功する。
- `score.json` と `metrics.csv` が出る。
- 追加した CSV/JSON 出力が出る。
- PNG は補助出力として生成される。

### 3.3 実データ検証

重い実データは通常テストに入れません。手動検証として扱います。

確認するもの:

- target 自己比較が ranking 1 位になる。
- 既知の近い case が上位に来る。
- metric ごとの傾向が `metric_summary.csv` で説明できる。
- runtime と memory が許容範囲か。

### 3.4 採用判断

採用する metric は、次のどれかを満たす必要があります。

- 既存 metric では見えない差を拾う。
- 既存 metric より解釈しやすい。
- ranking の安定性が上がる。
- 出力特徴量として後段解析に使いやすい。

満たさない場合は、実装しないか、docs 上の将来候補に留めます。

### 3.5 テスト予算

新しい手法ごとのテストは、最初から増やしすぎないようにします。

| テスト | 必須度 | 内容 |
| --- | --- | --- |
| self comparison | 必須 | 自己比較で最良になる |
| intended perturbation | 必須 | 手法が狙う差で悪化する |
| unsupported input | 必要時のみ | 対象外 view や open contour で `SKIPPED` / `ValueError` |
| workflow smoke | 必要時のみ | public YAML で実行できる |
| heavy dataset | 通常 test には入れない | 手動検証または docs に残す |

画像の完全一致 test は避けます。PNG は補助出力なので、存在確認や summary の値で検証します。

## 4. 手法カード

### 4.1 `profile` metric

優先度: P1

目的:

- 断面 `[x,z]` または `[y,z]` における高さ方向 profile を評価する。
- CD を単一値ではなく、幅、center、left edge、right edge の系列として扱う。

効くケース:

- 高さごとの幅差。
- 幅は同じだが中心がずれているケース。
- sidewall の傾きや bowing の傾向。
- 内部 material boundary の profile 差。

入力:

- `ViewFeature.label2d`
- `ViewFeature.mask`
- `ViewFeature.boundary_mask`
- `ViewFeature.grid2d`

最小 YAML:

```yaml
features:
  use: [contour, sdf]

metrics:
  use: [profile]
```

任意設定:

```yaml
metrics:
  use: [profile]
  profile:
    material_ids: [2]
    height_axis: z
    width_axis: x
    height_range: [20.0, 120.0]
```

デフォルト:

- `height_axis` は `z`。
- `width_axis` は view 内の非 `z` 軸。
- `height_range` は全範囲。
- `material_ids` 未指定時は non-void または material boundary を自動評価。

出力:

| 出力 | 内容 |
| --- | --- |
| `profile.csv` | 高さごとの width、center、left/right edge、差分 |
| `profile_summary.json` | mean / max / p95、support count |
| `metric_details.json` | 評価モードと対象 material |

実装場所:

- `wafergeo.compare.metric_profile.py`
- `wafergeo.compare.metric_defs`
- `wafergeo.compare.scoring`
- `wafergeo.compare.output_artifacts`

テスト:

- 自己比較で loss 0。
- 幅差で `width_loss > 0`。
- center shift で `center_loss > 0`。
- `[x,y]` topview では `SKIPPED` または明確な validation。

肥大化防止:

- `cd` を削除しない。
- `profile` は `cd` の上位互換にしすぎず、診断用 metric とする。
- wall angle や curvature は最初から入れない。必要になったら別 field として追加する。

### 4.2 open contour unsigned distance

優先度: P2

目的:

- `contour_json` の `closed: false` を自然に扱う。
- open polyline を無理に mask 化せず、unsigned distance として比較する。

効くケース:

- SEM などから抽出した一部輪郭。
- 全体領域が閉じていない edge line。
- 外形の一部だけを比較したいケース。

入力:

- `contour_json`
- `closed: false`
- `points`

最小 YAML:

```yaml
input:
  target:
    kind: contour_json
    path: data/target/open_edge.json

features:
  use: [sdf, contour]

metrics:
  use: [chamfer, sdf_band]
```

出力:

| 出力 | 内容 |
| --- | --- |
| `metric_details.json` | `distance_semantics: unsigned` |
| `features/target_sdf.npz` | polyline からの unsigned distance |
| `metrics.csv` | IoU が不適切な場合は `SKIPPED` |

実装場所:

- `wafergeo.compare.features`
- `wafergeo.compare.contour_loaders`
- `wafergeo.compare.metric_region`

テスト:

- open contour で compare が成功する。
- open contour に対して IoU を無理に計算しない。
- closed contour の既存挙動が壊れない。

肥大化防止:

- SEM 専用 loader を作らない。
- `contour_json` の contract 内で扱う。
- 新しい workflow は作らない。

### 4.3 `sdf_views` transform feature

優先度: P3

目的:

- 後段解析や外部サロゲート学習に使いやすい SDF 派生特徴量を出す。
- ただしパッケージ内に surrogate 学習機能は作らない。

効くケース:

- raw SDF だけではスケールが大きすぎる。
- boundary 近傍を強調したい。
- 学習入力として複数スケールの SDF が欲しい。

最小 YAML:

```yaml
task: transform

features:
  use: [sdf_views]
```

初期出力候補:

| field | 内容 |
| --- | --- |
| `sdf_nm` | 距離 nm |
| `tsdf_10nm` | 10 nm clip の TSDF |
| `tsdf_50nm` | 50 nm clip の TSDF |
| `log_abs_sdf` | `log1p(abs(sdf_nm))` |

出力:

```text
features/sdf_views.npz
```

実装場所:

- `wafergeo.compare.transform_features`
- `wafergeo.compare.feature_outputs`

テスト:

- 出力 shape が label grid と合う。
- TSDF が `[-1, 1]` に入る。
- `sdf_views` なしの既存 `sdf` 出力が壊れない。

肥大化防止:

- `Field3D` を public API として導入しない。
- manifest を出さない。
- 学習用 dataset builder を作らない。

### 4.4 `corner` metric

優先度: P4

目的:

- bottom corner、shoulder、rounding など、局所的な形状差を拾う。

効くケース:

- 幅は合っているが角が丸い。
- trench bottom がずれている。
- shoulder の削れ方が違う。

最小 YAML:

```yaml
view:
  axes: [x, z]
  depth_axis: y

features:
  use: [sdf, contour]

metrics:
  use: [corner]
```

初期実装:

- 2D 断面 mask から bounding box 近傍の代表 corner を抽出。
- corner 位置差を評価。
- corner 近傍の SDF 差を補助的に評価。

出力:

| 出力 | 内容 |
| --- | --- |
| `corner_summary.json` | 検出 corner、位置差、status |
| `metric_details.json` | corner loss の詳細 |

実装場所:

- `wafergeo.compare.metric_corner.py`
- `wafergeo.compare.metric_defs`
- `wafergeo.compare.output_artifacts`

テスト:

- 矩形の自己比較で loss 0。
- corner shift で loss が悪化。
- corner 検出できない形状で `SKIPPED`。

肥大化防止:

- 曲率推定や corner radius は最初から入れない。
- process-specific 名を入れない。
- topview では無理に評価しない。

### 4.5 material confusion summary

優先度: P2.5

目的:

- label-volume 比較で、どの material がどの material に置き換わっているかを見る。
- `sdf_material` の解釈を補助する。

効くケース:

- material id の入れ替わり。
- 薄膜や buried material の誤分類。
- IoU は悪いが、どの material が原因か分かりにくいケース。

最小 YAML:

```yaml
metrics:
  use: [sdf_material, iou]
```

追加出力:

| 出力 | 内容 |
| --- | --- |
| `material_confusion.csv` | simulation label と target label の pixel confusion |
| `material_confusion_summary.json` | major confusion pair |

実装場所:

- `wafergeo.compare.output_artifacts`
- 必要なら小 helper

テスト:

- swapped material case で off-diagonal が増える。
- 自己比較で diagonal のみになる。

肥大化防止:

- metric として total score に入れない。
- まずは診断用 output に限定する。
- YAML option は増やさない。

### 4.6 topology metric

優先度: P5

実装状態: 最小版を実装済み。2D projected view の connected component count 差だけを扱う。

目的:

- 穴、分断、連結性など、SDF や CD では見えにくい位相差を拾う。

効くケース:

- 材料が分断された。
- void が閉じた、または開いた。
- bridge / pinch-off の有無。

初期候補:

- connected component count 差。採用済み。
- hole count 差。保留。
- major component area ratio 差。保留。

実装場所:

- `wafergeo.compare.metric_topology.py`
- `metric_defs`

テスト:

- connected component 数が違う synthetic mask。
- 自己比較で loss 0。

肥大化防止:

- 初期は 2D projected view のみ。
- 3D topology や persistent homology は入れない。
- scipy が無い場合の fallback を複雑にしすぎない。

## 5. 優先順位

| 優先度 | 手法 | 理由 |
| --- | --- | --- |
| P1 | `profile` metric | CD の自然な拡張で、半導体断面評価に直結する |
| P2 | open contour unsigned distance | 外部輪郭データの扱いが自然になる |
| P2.5 | material confusion summary | 既存 `sdf_material` の解釈性が上がる |
| P3 | `sdf_views` feature | 外部解析・学習に使いやすい特徴量になる |
| P4 | `corner` metric | 局所形状差を拾えるが、定義を慎重にする必要がある |
| P5 | topology metric | 有用だが、過剰実装になりやすい |

最初に実装するなら `profile` metric が最も安全です。

理由:

- 既存の `cd_profile.csv` と発想が近い。
- `compare` の metric registry に閉じられる。
- データサイエンティストが結果を解釈しやすい。
- YAML を大きく変えずに追加できる。

## 6. 手法実装の issue テンプレート

新手法を実装するときは、次のテンプレートで issue 化します。

```text
AGENTS.md と docs/MaintenancePolicy.md を守る。
対象は metric / feature / output / docs / tests のうち必要最小限に限定する。
新しい public workflow は追加しない。
manifest/report/surrogate/assimilation/benchmark/preview/audit は復活させない。

手法名:
- ...

目的:
- ...

既存 metric との差分:
- ...

入力:
- ...

出力:
- ...

実装場所:
- ...

YAML:
- デフォルトで動くこと
- 追加 option が必要なら metrics.<name> 以下に限定すること

検証:
- 自己比較
- 意図した perturbation
- 既存 example

やらないこと:
- ...

実装開始時に提示すること:
- 対象レイヤ
- 追加する YAML 名
- 追加する出力
- default に入れないこと
- runner / CLI を変えないこと
- 確認事項または仮定
```

## 7. 実装前レビュー観点

実装に入る前に、次を確認します。

| 観点 | 確認内容 |
| --- | --- |
| DS 価値 | 既存 metric では見えない差を拾えるか |
| 解釈性 | CSV/JSON から原因を説明できるか |
| 安定性 | 小さいノイズや tiny material に過剰反応しないか |
| 計算量 | batch-compare で現実的な時間に収まるか |
| メモリ | 3D dense feature を不要に増やしていないか |
| 設計 | runner / CLI / YAML が肥大化していないか |
| テスト | synthetic test で十分に守れるか |

この表を満たせない手法は、まず docs 上の候補に留め、実装しません。

## 8. ユーザー向け仕様に落とすときのルール

新しい手法をユーザーに見せるときは、次の形にそろえます。

### 8.1 YAML

基本は `metrics.use` または `features.use` に名前を追加するだけにします。

望ましい例:

```yaml
features:
  use: [sdf, contour]

metrics:
  use: [profile]
```

追加設定が必要な場合も、対象 metric の下だけに置きます。

```yaml
metrics:
  use: [profile]
  profile:
    material_ids: [2]
```

避ける例:

```yaml
profiles:
  default:
    metrics:
      profile:
        runtime:
          policy:
            advanced_mode: true
```

### 8.2 出力

出力は次の優先順位にします。

1. `score.json` と `metrics.csv`
2. 手法別の `*_summary.json`
3. 手法別の `*.csv`
4. 補助 PNG

CSV/JSON を authoritative data とし、PNG は確認用にします。

### 8.3 docs

手法ごとに、最低限次を説明します。

- 何を測るか。
- どの view で使うか。
- 単位は何か。
- どの出力を見るか。
- どのケースでは `SKIPPED` になるか。

## 9. 実装後レビュー観点

実装後は、コード量だけでなく利用体験も確認します。

| 観点 | 確認内容 |
| --- | --- |
| ユーザー | 最小 YAML で動くか。出力名から意味が分かるか |
| DS | metric の単位、集約方法、ranking への影響が追えるか |
| 開発者 | 追加箇所が registry と小 module に閉じているか |
| 保守 | テストが増えすぎていないか |
| 性能 | batch 実行で極端に遅くならないか |
| メモリ | 不要な 3D dense array を保存していないか |

問題がある場合は、機能追加を広げるのではなく、まず出力名、summary、docs を改善します。

## P4 実装メモ: `corner`

`corner` は最小実装として採用します。対象は断面 view の bottom-left / bottom-right 位置差だけです。

やること:

- `metrics.use: [corner]` で明示指定された場合だけ動かす。
- `view.axes` に `z` を含む断面だけ対応する。
- `corner_summary.json` に left / right の位置差と平均 loss を出す。

やらないこと:

- default metric には入れない。
- corner radius, curvature, wall angle は入れない。
- top view で無理に corner を定義しない。
