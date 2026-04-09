以下は **Assimilation層（ただしこのプロジェクトにおける役割は「最適化」ではなく「比較・評価（＝同化のための目的関数評価器／レポータ）」）**としての、**設計図レベルの詳細仕様**です。
（全体設計はこれまで通り：Artifact駆動、SDF/Meshサロゲート両対応、観測はObserver層で `Obs2D` に統一、Metrics層で `Obs2D vs Obs2D` を比較。）

---

# 1. このプロジェクトにおける Assimilation層の定義

一般に「同化」は *最適化（パラメータ推定）＋観測比較* を含みますが、あなたの要件では

* **最適化・実行部（候補生成・探索戦略・収束管理）は本コードの役割ではない**
* 本コードは **比較・評価**（観測と予測のずれを定量化し、外部最適化器が使える形で返す）に徹する

という前提です。

したがって、この層は実質的に：

> **“Data Assimilation Objective Evaluation Layer”**
> （外部最適化器に渡せる、決定的で再現可能な objective / report を返す層）

です。

---

# 2. Assimilation層の責務と非責務（境界を厳密に）

## 2.1 責務（この層がやる）

1. **評価セッションの準備**

   * SEM観測（`SEMObsArtifact` / `Obs2D`）の読み込み
   * ObserverSpec・MetricSpecの読み込み（評価定義の固定）
   * Metricsのobs側前計算（band mask、KDTree、CD測定線など）をキャッシュ
   * surrogateモデルのロード（推論器の準備）

2. **候補パラメータ（外部から渡される）に対して objective を評価**

   * `surrogate.predict(params)` → geometry（TSDF or Mesh）
   * `Observer` → `pred Obs2D`（複数Observer対応）
   * `Metrics` → `loss + report + maps + status`
   * 複数Observer/複数Metricを **合成**して総合スコアを返す

3. **デバッグ・説明性のある出力**

   * 残差マップ、輪郭オーバレイ、CD測定結果、失敗理由
   * “なぜ合わないか” を追跡できる構造化レポート

4. **再現性・監査性（Artifact化は“選択可能”だが設計として担保）**

   * 設定（ObserverSpec/MetricSpec/SEM入力hash/モデルversion）を必ず紐付け
   * trial（評価1回）を保存するかどうかはポリシーで切替（後述）

---

## 2.2 非責務（この層がやらない）

* **次の候補パラメータを決めない**（CMA-ES/BO/NM/勾配法などは外部）
* **同化のループ（反復・収束判定・予算管理）を持たない**
* **ジョブスケジューリング（並列探索の制御）を持たない**
* **SEMアライメントの“大域探索”をしない**（やるなら別サービス／外部。ここでは「与えられたTransformの適用」＋「必要なら軽量な検証」まで）

> ただし、“比較の前提を揃えるための deterministic な前処理” は含めて良い
> 例：Obs側のKDTree構築、band mask作成、CD測定線のサンプル点列の作成

---

# 3. コード構成（最適化を含めない Assimilation層の完成形）

前回案の `assimilation/optimizers/` は **本プロジェクトの責務外**として外します（または `external/` に移し、インターフェースだけ残す）。

推奨構成：

```text
wafergeo/assimilation/
  README.md                 # 「最適化は外部」方針の明文化（重要）
  types.py                  # ParamSpec/Candidate/EvalResult/CaseSpec など型
  case.py                   # AssimilationCase（評価対象の束：SEM/Observer/Metric/Model）
  session.py                # EvaluationSession（キャッシュ・依存を保持）
  objective.py              # ObjectiveBuilder（geom→obs→metrics の合成器）
  evaluator.py              # evaluate_one / evaluate_batch（外部最適化器が呼ぶAPI）
  cache.py                  # obs前計算/中間結果のメモ化（in-memory + optional disk）
  policies.py               # LoggingPolicy / FailurePolicy / TransformPolicy
  artifacts.py              # TrialArtifact / ReportArtifact の保存（任意）
  reports.py                # 人間向け要約（JSON/CSV/HTML等の整形）
  cli.py                    # sem-case validate / evaluate / summarize
```

> **レビューしやすさの肝**：
>
> * `objective.py` は “純粋に組み立てる”
> * `session.py` は “重いもの（モデル・キャッシュ）を保持”
> * `evaluator.py` は “外部が呼びやすい薄いAPI”
> * `artifacts.py/reports.py` は “保存と整形を分離”

---

# 4. 主要データ型（外部最適化器が使いやすい契約）

## 4.1 ParamSpec（パラメータの表現：エンコード/デコードが重要）

外部最適化器はふつう `x: R^d` を扱います。サロゲートは辞書や構造体を要求しがちです。
Assimilation層は **その橋渡し**を担います。

```python
@dataclass(frozen=True)
class ParamAxis:
    name: str
    kind: Literal["continuous", "int", "categorical"]
    bounds: tuple[float, float] | None
    units: str | None
    transform: Literal["identity", "log", "logit"]  # 例
    default: float | int | str
    choices: list[str] | None

@dataclass(frozen=True)
class ParamSpec:
    axes: list[ParamAxis]
    vector_order: list[str]  # axes順固定（再現性）
    def encode(self, params_dict: dict) -> np.ndarray: ...
    def decode(self, x: np.ndarray) -> dict: ...
```

**ポイント**

* `transform`（log等）を仕様化すると、外部最適化器の探索が安定しやすい
* ただし “最適化自体” は外部なので、ここは **変換器の提供**に留める

---

## 4.2 AssimilationCase（評価対象の束）

1ケース＝「このSEM観測に対して、この観測定義と指標で、このモデルを評価する」。

```python
@dataclass(frozen=True)
class AssimilationCase:
    case_id: str
    sem_obs_ids: dict[str, str]          # observer_name -> SEMObsArtifact id（Obs2D）
    observer_specs: dict[str, ObserverSpec]
    metric_spec: MetricSpec              # aggregate含む
    model_package_id: str                # surrogate model package
    param_spec: ParamSpec
    transform_policy: TransformPolicy
    failure_policy: FailurePolicy
    logging_policy: LoggingPolicy
```

**sem_obs_ids を observer_name ごとに持つ**のが実務で効きます：

* topdown観測とslice観測で、SEMの前処理（ROI・grid・tsdf2d）が異なることが多い
* 1つのObs2Dに無理に詰めず、Observerごとに正規化したSEMObsArtifactを作る、という前提に合います

---

## 4.3 EvaluationSession（重い依存とキャッシュを保持する“状態オブジェクト”）

外部最適化器は何百回も objective を呼ぶので、毎回ロードすると終わります。
Sessionに集約して “一度だけ重い準備” をします。

```python
@dataclass
class EvaluationSession:
    case: AssimilationCase
    store: ArtifactStore
    surrogate: SurrogateModel
    observers: dict[str, Observer]                 # name -> instance
    obs_sem: dict[str, Obs2D]                      # name -> SEM Obs2D
    obs_precomp: dict[str, Obs2DPrecomp]           # metrics用キャッシュ
    metric_aggregator: MetricAggregator            # MetricSpecから生成
    caches: EvalCaches                             # memoization（任意）
```

---

## 4.4 EvalResult（外部に返す標準結果）

```python
@dataclass(frozen=True)
class EvalResult:
    candidate_id: str
    x: np.ndarray                                  # canonical vector
    params: dict                                   # decoded params
    total_loss: float
    per_observer: dict[str, float]
    metric_results: list[MetricResult]             # loss+report+maps+status
    status: Literal["OK","WARN","FAIL"]
    messages: list[str]
    timings: dict[str, float]                      # inference/observe/metrics etc
    artifacts: dict[str, str] | None               # optional: saved artifact ids
```

**外部最適化器が欲しいのは通常 `total_loss`**ですが、
デバッグや後分析のために、`metric_results` と `timings` を標準で返せる設計が重要です。

---

# 5. 評価パイプライン（1候補の評価手順：固定フロー）

ここが Assimilation層の中核です。**最適化ではなく、評価だけ**を行います。

## 5.1 evaluate_one の固定手順

入力：`x`（パラメータベクトル）

1. **decode**

   * `params = ParamSpec.decode(x)`
   * （必要なら）範囲外の扱い：clamp/penalty/fail（FailurePolicyで決める）

2. **surrogate推論**

   * `geom = surrogate.predict(params)`
   * 出力は `TSDFVolume` か `MeshGeom`（サロゲートに依存）

3. **Observerで予測観測を生成（複数Observer）**

   * `pred_obs[name] = observers[name].observe(geom, observer_specs[name])`
   * ここで “比較ドメインが Obs2D に統一” されます

4. **Transformの適用（TransformPolicy）**

   * 基本：SEM側 `obs_sem[name].transform` を適用済みとみなす、または pred側に適用
   * ルールは必ずPolicyで固定し、勝手に最適化しない
     例：
   * `fixed`：何もしない（strict）
   * `apply_sem_transform`：SEM→sim整列済みを前提に比較
   * `apply_pred_transform`：predをSEM座標へ戻して比較（用途次第）

5. **Metricsで比較（合成）**

   * `MetricAggregator.compute(pred_obs, obs_sem, precomp=obs_precomp)`
   * 返り値：

     * `total_loss`
     * observer別・metric別の詳細（MetricResult）
     * residual maps（任意）
     * status（OK/WARN/FAIL）

6. **ロギング／Artifact保存（LoggingPolicy）**

   * `none`：保存しない（外部最適化器が高速に回す用）
   * `best_only`：現在ベストだけ保存
   * `periodic(k)`：k回に1回保存
   * `all`：全trial保存（重いが解析には強い）

7. **EvalResult を返す**

   * 外部は `total_loss` を使って探索
   * 内部・人間は `metric_results` や `maps` で原因追跡

---

# 6. objective.py（ObjectiveBuilder）：geom→obs→metrics を“組み立てるだけ”にする

`objective.py` はコードの中で最もレビューされる場所になります。
ここを “薄い・決定的・設定駆動” にすると保守が圧倒的に楽です。

## 6.1 ObjectiveBuilderの責務

* サロゲート表現（TSDF/Mesh）を気にしない（Observerが吸収）
* ObserverSpecに基づき `Obs2D` を作る
* MetricSpecに基づき比較し、合成lossを返す
* 例外では落とさず status と penalty で返す（FailurePolicy）

## 6.2 擬似コード（設計の骨格）

```python
def evaluate_candidate(session: EvaluationSession, x: np.ndarray, candidate_id: str,
                       return_maps: bool) -> EvalResult:

    t0 = now()
    params = session.case.param_spec.decode(x)

    geom = session.surrogate.predict(params)

    pred_obs = {}
    for obs_name, observer in session.observers.items():
        pred_obs[obs_name] = observer.observe(geom, session.case.observer_specs[obs_name])

    # transform policy
    pred_obs, obs_sem = apply_transform_policy(pred_obs, session.obs_sem, session.case.transform_policy)

    agg = session.metric_aggregator.compute(pred_obs, obs_sem, session.obs_precomp, return_maps=return_maps)

    # logging policy
    art = maybe_save_trial(session, x, params, geom, pred_obs, agg, session.case.logging_policy)

    return build_eval_result(...)
```

---

# 7. evaluator.py：外部最適化器向けAPI（最重要）

## 7.1 外部が呼びやすいAPI設計

外部最適化器は次の2形態を好みます。

### A) 単発評価

* `f(x) -> loss`
* `f(x) -> (loss, aux)`（補助情報）

### B) バッチ評価（探索アルゴリズムが並列候補を投げる）

* `F(X) -> losses`（X: [B,d]）

これを吸収します。

```python
def evaluate_one(session, x, *, return_debug=False) -> EvalResult
def evaluate_batch(session, X, *, return_debug=False, parallel="process", num_workers=...) -> list[EvalResult]
def objective_only(session, x) -> float   # 外部向け最小API
```

**設計上の注意**

* 並列は Session 内の surrogate が thread-safe とは限らない
  → “プロセス並列” を推奨（ただしモデルロードコストが大きい場合は要検討）
* 本プロジェクトは最適化をしないので、並列戦略は “評価を早く返す” ための補助機能に留める

---

# 8. Policy設計（運用で壊れないための“挙動固定”）

同化評価で運用が壊れるのは「例外処理」「Transform」「保存量」が曖昧だからです。
これを Policy として明文化します。

## 8.1 FailurePolicy（範囲外・観測失敗の扱い）

例：

* `out_of_bounds = "clamp" | "penalty" | "fail"`
* `empty_mask = penalty_value`（巨大値を返す）
* `open_contour = allow_unsigned_tsdf`（SEMがopen輪郭なら距離定義を切替）
* `nan_loss = penalty_value`

**重要**：最適化器は外部なので、ここは “評価器として一貫した応答” を返すことが目的です。

---

## 8.2 TransformPolicy（比較座標系の固定）

* `strict_sim_grid`：SEMObsはすでにsim座標に整列済み前提、追加変換なし
* `apply_sem_to_sim`：SEMObsに保存された transform を適用して比較
* `compare_in_sem`：predをSEM座標に戻して比較（可視化用途）

> “Transformを推定する”のは最適化に近くなるので、この層では原則しません。
> 推定が必要なら `sem/align.py`（前処理）や外部の同化システムで行い、その結果transformをSEMObsArtifactに保存する、という運用が安全です。

---

## 8.3 LoggingPolicy（trialの保存量）

外部最適化は評価回数が多いので、全保存は重くなります。
設計として必ずポリシー化します。

* `none`（高速、最適化器向け）
* `best_only`（解析に十分なことが多い）
* `periodic(n)`（トレース可能）
* `all`（研究用途）

---

# 9. Artifact化（“最適化は外部”でも、評価ログは残せる設計に）

最適化は外部でも、**評価器としての結果（特にベスト候補や失敗例）は残したい**ことが多いです。
なので、Artifactは “任意” だが **いつでも有効にできる**設計にします。

## 9.1 TrialArtifact（評価1回）

保存する場合の最小セット（heavyにしない）：

* `x / params`
* `total_loss`
* `per_observer`
* `metric report`（mapsはオプション）
* `timings`
* `meta`（case_id, spec hashes, model id, sem ids）

**pred Obs2D の保存**は重いので原則オプション：

* `save_pred_obs = false`（通常）
* `save_pred_obs = true`（デバッグ時だけ）

## 9.2 SummaryReportArtifact（ケースのまとめ）

外部最適化が終わった後に、外部から “best x” を受け取り、

* best候補の予測Obs2D
* residual maps
* CD誤差表
* 観測定義（ObserverSpec）と評価定義（MetricSpec）

をまとめて保存するコマンドを用意すると運用が回ります。
（探索は外部、レポート生成は本プロジェクト、という分担が明確。）

---

# 10. CLI（最適化を含めない同化評価の操作体系）

“同化の実行” は外部でも、評価器の整合性検証やレポート生成は本プロジェクトで行えるべきです。

推奨コマンド：

1. `assim validate-case case.yaml`

* SEMObsArtifact / ObserverSpec / MetricSpec / ModelPackage が揃っているか
* grid2d一致、mu一致、units一致、などを検証

2. `assim evaluate case.yaml --x "..."`

* 1点評価（デバッグ用）
* residual maps を出力できる

3. `assim evaluate-batch case.yaml --candidates candidates.csv`

* 外部が吐いた候補列をまとめて評価（ランキング表を作る）

4. `assim summarize case.yaml --best best.json`

* best候補の詳細レポート生成（CD表、輪郭オーバレイ、指標一覧）

---

# 11. テスト設計（この層は“壊れると全滅”なので回帰テスト必須）

Assimilation層は複数層（surrogate/observer/metrics/sem）をつなぐため、統合テストが重要です。

## 11.1 ゴールデンテスト（最低限）

* “同一Obs2Dをpred=obsで与える” → total_loss≈0
* “predを既知量だけ平行移動” → chamfer/CD/tsdf_loss が期待通り増える
* “mask空/輪郭open” → FAILになり、penaltyが返る（落ちない）
* “Observerを複数” → 合成lossがSpec通りの重みで合算される

## 11.2 再現性テスト

* 同じ case.yaml + 同じ x → 同じ loss（浮動小数の誤差許容内）
* 乱数が関わる要素（点群サンプル等）は、Observer/Metricでは原則使わない or seed固定

---

# 12. まとめ：このAssimilation層の“設計の芯”

* **最適化（探索）は外部**。本層は **評価器（objective function）**として設計する
* 入力（SEMObs/ObserverSpec/MetricSpec/ModelPackage/ParamSpec）を固定し、
  **決定的に `total_loss + report` を返す**
* 複数Observer・複数Metricを **Spec駆動で合成**し、比較定義をブレさせない
* 失敗しても落とさず、status/penaltyで返す（外部探索が止まらない）
* ログ/Artifactは “任意だがいつでもONにできる” ポリシー化（運用と研究の両立）

---

