以下は **Metrics層（`wafergeo/metrics/*`）**を、あなたの全体設計（Artifact駆動／SDF・Meshサロゲート両対応／SEM同化は最終的に2D輪郭比較）に合わせて、**実装・レビューにそのまま使える“設計図レベル”**で詳細化したものです。

ここでの Metrics層は、単なる「評価関数の寄せ集め」ではなく、

* **同化（最適化）の目的関数を構成する Loss**
* **学習時の評価（2D観測整合）**
* **手法比較（SDF手法・mesh手法・observer定義差）**
* **運用デバッグ（なぜ合わないかの可視化）**

を **同じ契約（Obs2D/TSDF/Contour/MeasurementSpec）**で支える「中核インフラ」です。

---

# 1. Metrics層の責務と、最重要設計方針

## 1.1 Metrics層の責務（何を提供する層か）

Metrics層は、主に次を提供します。

1. **Obs2D（予測 vs 観測）を比較して、損失・評価指標を返す**

   * 2D TSDF差（ロバスト、band、weight）
   * contour距離（Chamfer/Hausdorff）
   * CD（Critical Dimension）など、半導体工程で重要な測定指標
   * 面積/周長/オフセット/ラフネス等の補助指標

2. **複数Observer（topdown + slice複数）を束ねて “総合スコア” を作る**

   * 同化の objective = Σ w_k * metric_k(Obs2D_k)

3. **高速化のための precompute / cache を提供**

   * 同化は同じ観測に対して何百回も評価するため、obs側のKDTree等をキャッシュする（重要）

4. **結果の説明性（デバッグ用の map/レポート）**

   * residual map（どこが合ってないか）
   * band mask
   * CD測定点と結果
   * 失敗理由（contourが閉じてない、maskが空、など）

5. **Metric仕様のYAML化・バージョニング**

   * “同じモデルでも評価関数が変わる”と運用が壊れるので、MetricSpecを固定する

---

## 1.2 最重要設計方針：LossとReportを分離せず “同じMetricが両方返す”

実務では、「最適化の損失」と「レポート指標」がバラバラになり、結果の追跡が困難になります。

そこで設計としては：

* **Metricは必ず**

  * `loss_value`（最適化に使えるスカラー）
  * `report`（人間が読む指標・統計）
  * `maps`（可視化用）
  * `status`（OK/WARN/FAIL）
    を返す

ようにします。

---

## 1.3 “比較ドメイン”は Obs2D を最優先にする

あなたの前提では、SDFサロゲートでもMeshサロゲートでも最終比較は2D輪郭です。
したがって Metrics層の第一級I/Fは：

* **Obs2D × Obs2D → MetricResult**

です。

（3D TSDFやmeshの学習損失は surrogate/training 側に置いてもよいですが、**“2D観測整合損失”は必ずMetrics層のロジックを共有**するのが運用上強いです。）

---

# 2. コード構成（Metrics層の責務分割）

推奨構成（これがそのまま “レビューしやすい単位” になります）：

```text
wafergeo/metrics/
  base.py                # Metric I/F, MetricSpec, MetricResult, registry
  context.py             # Precompute/Cache: Obs2DPrecomp, KDTree, band masks
  robust.py              # robust penalty ρ() とその設定
  weights.py             # weight map / band weighting / spatial weighting
  tsdf_loss.py           # 2D TSDF loss（band+robust+weight）中核
  mask_metrics.py        # IoU/Dice/area/perimeterなど（補助）
  contour_metrics.py     # Chamfer/Hausdorff/curve alignment等
  cd_metrics.py          # CD（測定定義YAML）: line-scan TSDF zero crossing
  alignment_metrics.py   # alignment residual / optional alignment estimator（別層でもOK）
  aggregate.py           # multi-observer / multi-metric 集約（objective builder）
  qa.py                  # metric側の失敗判定・例外扱い規約
  reports.py             # 結果の整形（JSON/CSV/plots用の構造化）
```

---

# 3. 主要データ型（契約）— ここを固定すると運用が壊れない

## 3.1 MetricSpec（YAMLで固定する設定）

Metricは「同じ名前でもパラメータで意味が変わる」ので、Specを必ず明文化します。

例（概念）：

```yaml
schema_version: "metric/v2"
metric_set_id: "assim_objective_v2"
metrics:
  - name: "tsdf_l1_robust"
    weight: 1.0
    params:
      mu_nm: 200.0
      band: "obs_band"          # |phi_obs|<mu
      robust: {type: "huber", delta: 0.1}
      weight_mode: "sem_weight" # obs.weight を使用
      spatial_weight: {type: "boundary_emphasis", gamma: 2.0}  # optional

  - name: "contour_chamfer"
    weight: 0.2
    params:
      sampling_points: 1024
      robust: {type: "cauchy", c: 50.0}  # nmスケールなど
      use_holes: false

  - name: "cd_linescan"
    weight: 0.5
    params:
      lines:
        - axis: "x"
          y_nm: 5000.0
          x_range_nm: [2000, 18000]
          expected_edges: 2
          edge_pair: "outer"     # left-right
          method: "tsdf_zero_cross"
```

重要：このYAML（MetricSpec）は `AssimilationRunArtifact` / `ModelPackage` / `EvalReport` に必ず埋め込みます。

---

## 3.2 MetricResult（共通の戻り型）

Metricが返すものを固定すると、同化・学習・評価で共通化できます。

```python
@dataclass(frozen=True)
class MetricResult:
    name: str
    version: str
    loss: float                 # objectiveに足し込むスカラー（NaN禁止）
    report: dict                # 例: {"mean_nm":..., "p95_nm":...}
    maps: dict[str, np.ndarray] # 例: residual_map(Y,X), band_mask(Y,X)
    status: str                 # "OK" | "WARN" | "FAIL"
    messages: list[str]
    meta: dict                  # spec_hash, params, units, timing, etc.
```

> **loss** は最適化で使われるので、例外ではなく “FAILでもlossを返す” 規約にします
> （例：maskが空→大きいペナルティ＋status=FAIL）。同化が落ちない運用が可能です。

---

## 3.3 Precompute/Cache（同化高速化の核）

同化は「同じobsに対して何百回もpredを評価」します。
そこで **obs側の前計算**をキャッシュします。

```python
@dataclass(frozen=True)
class Obs2DPrecomp:
    grid2d: GridSpec
    mu_nm: float
    band_mask_obs: np.ndarray          # |phi_obs| < mu
    phi_obs_nm: np.ndarray             # obs.tsdf * mu（band内用途）
    contour_points_xy: np.ndarray      # (N,2) nm（全ループ結合 or ループ別）
    contour_kdtree: object | None      # KDTree（scipy等）
    mask_area_nm2: float
    perimeter_nm: float
    notes: dict
```

`context.py` で `precompute_obs(obs2d, spec)` を提供し、同化ループ内で再利用します。

---

# 4. Metricsの分類（本プロジェクトで必須の“4本柱”）

あなたの用途（サロゲート＋SEM同化）で必須なのは次の4種類です。

1. **2D TSDF損失（Field-level）**：同化の主力・安定
2. **Contour距離（Curve-level）**：現場説明性が高い・レポートに必須
3. **CD/測定指標（Measurement-level）**：半導体エンジニアリング的に必須
4. **集約（Aggregation）**：複数Observer/複数Metricの総合objective化

以下、それぞれを「実装できる粒度」で説明します。

---

# 5. 2D TSDF損失（`tsdf_loss.py`）— 同化の中核

## 5.1 目的と基本式

obs/pred の 2D TSDF（[-1,1]）を比較します。

* `phi_obs_nm = obs.tsdf * mu_nm`
* `phi_pred_nm = pred.tsdf * mu_nm`

差分 `r = phi_pred_nm - phi_obs_nm`

同化・評価の標準は：

* **band内**（境界近傍のみ）
* **robust**（SEM抽出誤差・外れ値に強い）
* **weight map**（SEM信頼度・ノイズ領域を抑える）

### 標準loss

[
L = \frac{1}{\sum w}\sum_{x \in \Omega} w(x),\rho(r(x))
]

* Ω：band領域（例：(|\phi_{obs}| < \mu)）
* w(x)：SEM confidence（obs.weight）や空間重み
* ρ：Huber / Cauchy / Tukey など

---

## 5.2 bandの定義（設計で固定すべき）

bandは “どこを重視するか” の定義なので、Specで固定します。

推奨は次の3種を用意し、Specで選択：

* `obs_band`：`|phi_obs_nm| < mu_nm`（推奨：観測基準で安定）
* `pred_band`：`|phi_pred_nm| < mu_nm`（予測形状追随）
* `union_band`：`obs_band OR pred_band`（強めだが安定性は落ちる）

---

## 5.3 weight map設計（SEM同化の実用ポイント）

`weights.py` で重み設計を統一します。

### 代表weightモード

* `uniform`：w=1
* `sem_weight`：obs.weight を使用（SEM側が信頼度を出す場合）
* `boundary_emphasis`：境界近傍を強調（例：w = exp(-|phi_obs|/σ)）
* `roi_mask`：測定対象ROI以外は0（SEM視野外など）

重みは必ず正規化して `Σw` で割り、比較可能なスカラーにします。

---

## 5.4 robust penalty（`robust.py`）

同化では外れ値が頻出するので、ρを統一的に選べるようにします。

* `L1`：ρ(r)=|r|
* `L2`：ρ(r)=0.5 r^2（外れ値に弱い）
* `Huber(δ)`：小さい誤差はL2、大きい誤差はL1
* `Cauchy(c)`：外れ値の影響を緩やかに抑える
* `Tukey`：外れ値をほぼ無視（強いが局所最適化に注意）

設計としては `RobustSpec` を持ち、全Metricで共通利用します。

---

## 5.5 出力（デバッグに効く maps を必ず返す）

TSDF loss は “どこが合わないか” が重要なので、

* `residual_map_nm`（r）
* `band_mask`
* `weight_map`
* `robust_map`（ρ(r)）
* `loss_contrib_map`（w*ρ(r)）

を `maps` に入れられるようにします（デバッグオン時のみ保存でもOK）。

---

# 6. contour距離（`contour_metrics.py`）— 説明性の高い指標

TSDF lossは安定ですが、現場説明は輪郭距離の方が分かりやすいことが多いです。

## 6.1 入力の前提（Observer層で統一済み）

* `Obs2D.loops`（outer/holeのループ列）
* 点数は observer 側で resample 済み（例：1024点）

> **設計方針**：輪郭生成の“定義差”が出ないよう、Metrics層は輪郭生成をしません。
> 使うのはObs2Dの輪郭だけ。

---

## 6.2 代表指標

### A) Symmetric Chamfer distance（推奨）

輪郭点集合 A（pred）と B（obs）について、

* `d(A→B) = mean_{a∈A} min_{b∈B} ||a-b||`
* `d(B→A)` 同様
* `Chamfer = (d(A→B) + d(B→A))/2`

実装はKDTreeで高速化し、**obs側KDTreeは precompute でキャッシュ**します。

### B) Hausdorff distance（品質チェック向け）

* `H(A,B) = max( max_{a∈A} min_{b∈B}||a-b|| , max_{b∈B} min_{a∈A}||b-a|| )`
  外れ値に弱いので、運用では `p95 Hausdorff` などの分位版を標準にすると良いです。

### C) Area / Perimeter difference

* `ΔArea = |Area_pred - Area_obs|`
* `ΔPerimeter = |P_pred - P_obs|`
  輪郭抽出の異常や、topdown露出定義の差異検知に効きます。

---

## 6.3 ループ（outer/hole）の扱い（事故防止の仕様）

* デフォルトは **outerのみ**
* holeを使う場合は `use_holes=true` をSpecで明示
* 複数コンポーネントがある場合：

  * “最大面積ループのみ”を使うモード
  * “全ループ結合”で使うモード
    をSpecで選べるようにする

---

# 7. CD（Critical Dimension）・測定指標（`cd_metrics.py`）— 半導体エンジニアリングの核

CDは「輪郭が一致しているか」を工程的に解釈する最重要指標です。
そして **輪郭点から測るより、2D TSDFからゼロ交差を測る方が頑健**です。

## 7.1 CD測定の基本戦略（TSDF line-scan）

測定線（例：y=y0 の水平線）上で `phi(x)` をサンプルし、

* `phi(x)` が符号変化する区間を探し
* 線形補間でゼロ交差点を求め
* 交差点の組み合わせからCDを計算する

### なぜTSDF line-scanが良いか

* 輪郭抽出の細部差（スムージング・再サンプル差）に依存しにくい
* sub-pixelにゼロ交差を推定できる
* bandの情報と一致する（同化の損失と整合）

---

## 7.2 MeasurementSpec（CD定義をYAML固定）

CDの測定定義は “現場仕様” なので YAMLで固定し、Artifactに保存します。

例（概念）：

```yaml
schema_version: "measurement/v1"
name: "cdset_topdown_v1"
lines:
  - id: "cd_center"
    axis: "x"
    y_nm: 10000.0
    x_range_nm: [2000.0, 18000.0]
    expected_edges: 2
    edge_pair: "outer"           # left-most and right-most
    method: "tsdf_zero_cross"
  - id: "cd_upper"
    axis: "x"
    y_nm: 12000.0
    x_range_nm: [2000.0, 18000.0]
    expected_edges: 2
```

### edge選択ルール（事故が起きるので明文化）

* `outer`：最左と最右
* `inner`：中央の2つ（4交差のトレンチなど）
* `nearest_to(x_ref)`：基準位置に近い2つ
* `pair_by_expected_cd(cd_nm)`：期待CDに最も近いペア

---

## 7.3 出力（CDは “数値 + 位置” が必要）

CD metric の result は次を report/maps に入れます。

* `cd_nm`（1本ごと）
* `cd_error_nm`（pred-obs）
* 交差点座標（x_left, x_right）
* 交差点探索が失敗した場合の理由（交差が見つからない/多すぎる等）

これがあると、同化が失敗する理由が追えます。

---

# 8. alignment_metrics（任意：残差評価と、軽量推定）

あなたの設計ではアライメント推定は assimilation 側に置くのが自然ですが、Metrics層にも “残差の見え方” を揃えるための部品が必要です。

## 8.1 最低限 Metrics層で持つべきもの

* `alignment_residual`：Transform適用後の指標（例：Chamfer after T）
* `sanity_check`：Transformが大きすぎる・スケールが異常など

## 8.2 optional：初期align推定（使うなら）

* TSDFの相互相関（phase correlation的）
* contour点のProcrustes（平行移動+回転+スケール）
  ただしこれは運用上の難所なので、**最初はassimilationで段階処理（粗→精）**が安全です。

---

# 9. aggregate.py（総合objective builder）— 同化と学習評価の“統一口”

同化・学習評価・ベンチ比較で「同じ指標セット」を使うために、集約器を固定します。

## 9.1 Aggregator設計（概念）

* 入力：`MetricSpec`（YAML）
* 入力：`pred_obs2d_dict`, `obs_obs2d_dict`（observer名→Obs2D）
* 出力：`ObjectiveResult`

  * `total_loss`
  * `metric_results`（各metricの詳細）
  * `by_observer`（observer別合計）

### 合計の例

[
L_{\text{total}} = \sum_{o \in observers} \sum_{m \in metrics} w_{o,m},L_{o,m}
]

observer別の重みもSpecで持てるようにすると運用が強いです：

* topdownは重い
* sliceは補助
  など。

---

# 10. 失敗・例外設計（qa.py）：同化が落ちないための規約

Metrics層は同化ループ内で呼ばれるので、例外で落とすのは避け、**statusとpenaltyで返す**設計が強いです。

## 10.1 典型的な失敗ケース

* obsのmaskが空（SEM抽出失敗）
* predのmaskが空（予測形状が飛んだ）
* contourが閉じない
* CD交差点が見つからない／多すぎる

## 10.2 規約（推奨）

* `status="FAIL"` とする
* `loss` は **大きい固定値**（または段階的ペナルティ）を返す
* `messages` に理由・必要な修正（ObserverSpec、ROI、threshold等）を入れる

これで最適化は継続でき、ログから原因追跡できます。

---

# 11. 性能設計（同化で“何百回も回る”前提の高速化）

Metrics層で同化が遅くなるポイントは主に2つ：

1. contour距離の近傍探索（KDTree）
2. CD line-scan の繰り返し

なので設計として：

* **obs側KDTreeは precompute で1回だけ作る**
* CD測定線（サンプル座標列）も precompute して保持する
* TSDF lossのband_mask（obs_band）は precompute
* `Obs2DPrecomp` を `MetricContext` として同化ループに渡す

を必須にします。

---

# 12. Metrics層の“最小実装セット”（まず運用を回すために必須）

プロジェクトを早く回すための最小セットはこれです（これが揃うと同化・評価が成立）：

1. `tsdf_l1_robust`（band + weight + huber）
2. `contour_chamfer`（KDTree cache）
3. `cd_linescan`（TSDFゼロ交差）
4. `aggregate`（MetricSpecから総合lossを作る）
5. `precompute_obs`（band_mask・KDTree・measurement lines）
6. `qa`（空mask等のFAIL処理）

---

# 13. 第三者が追加・編集しやすい設計（プラグイン化）

あなたのプロジェクトは「様々な手法の利用・評価」が目的なので、Metrics層もプラグイン前提が必須です。

## 13.1 MetricプラグインI/F（設計案）

```python
class Metric(Protocol):
    name: str
    version: str
    def required_inputs(self) -> dict: ...      # needs tsdf? needs contour? needs weight?
    def precompute_obs(self, obs: Obs2D, spec: dict) -> object | None: ...
    def compute(self, pred: Obs2D, obs: Obs2D, spec: dict, ctx: object | None) -> MetricResult: ...
```

* `precompute_obs` を各Metricが持つと、obs側キャッシュを自然に設計できます
* `required_inputs` で “このMetricはcontour必須” などを実行前に検証できる

## 13.2 新Metric追加の必須条件（運用を壊さないための規約）

* MetricSpecのYAML例を `docs/` に追加
* `FAIL時のloss返し方` を `qa.py` 規約に従う
* ゴールデン形状で単体テスト（obs/predが一致→loss=0に近い）
* `meta` に spec_hash と version を必ず含める

---

# 14. データサイエンティスト視点：各Metricが意味するもの（短く整理）

* **TSDF loss**：境界近傍の“形状差”を滑らかに最適化できる（同化の主エンジン）
* **Chamfer/Hausdorff**：輪郭のズレを直感的に説明できる（現場レポート向き）
* **CD**：工程的に意味がある（ライン幅・トレンチ幅・段差位置など）
* **Area/Perimeter**：観測定義（exposedルールやROI）の不整合検知
* **multi-observer aggregation**：2D同化の不定性を緩和（topdown+sliceで形状制約が増える）

---

## ここまでの要点（Metrics層の設計の芯）

* **Obs2D比較を第一級I/Fに固定**して、SDF/meshサロゲートの差を吸収する
* 同化が安定する **TSDF band + robust + weight** を標準損失にする
* 半導体向けの **CDをTSDFゼロ交差で定義し、YAMLで測定仕様を固定**する
* **precompute/cache** を設計に組み込み、同化ループの速度を担保する
* Metricは **loss + report + maps + status** を返して、最適化と説明性を両立する

---

必要なら次に、Metrics層について **“実装テンプレ（ファイルごとの関数一覧・擬似コード・例外とFAIL規約・テスト雛形・MetricSpecスキーマ）”**を、リポジトリにそのまま貼れる粒度で提示できます。
