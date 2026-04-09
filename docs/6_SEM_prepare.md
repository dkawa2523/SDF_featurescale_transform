以下は **SEM取り込み・同化準備層**（SEM画像／輪郭座標 → “同化可能な観測表現”へ正規化し、`SEMObsArtifact` として保存する層）を、**設計図（実装に落ちる粒度）**としてできるだけ詳細にまとめたものです。
全体設計（Artifact駆動、Observerで2D観測に統一、SDF/meshサロゲート両対応、同化比較は最終的に2D輪郭/2D距離場）は前回の通りで、ここでは **この層だけ**に集中します。

---

# 1. この層の目的と責務（アーキテクト視点での位置づけ）

## 1.1 目的

この層のゴールは、SEM由来の情報（画像・輪郭座標）を、以降の同化・評価が **完全に表現非依存**（SDFサロゲートでもMeshサロゲートでも同じ）で扱えるように、**共通の観測表現に変換すること**です。

> ここで作る最終成果物は **Obs2D（mask/2D TSDF/contours/weight/transform）** です。
> 以降の同化・評価は「Obs2D vs Obs2D」だけで成立します。

---

## 1.2 この層が担当すること（責務）

1. SEMデータの取り込み（画像・輪郭）
2. 輪郭の正規化（座標系、単位、向き、閉曲線判定、再サンプリング、複数ループ/穴の扱い）
3. 同化の比較ドメインに合わせて **2D観測（Obs2D）** を生成

   * 2D mask（閉曲線なら内部塗りつぶし）
   * 2D TSDF（距離場：mask→EDT または polyline解析距離）
   * 2D contour（比較用：resample固定）
4. SEM固有の重み（confidence/weight map）生成
5. 初期アライメント（Transform）の推定・保存（任意だが運用で重要）
6. QA（データ破綻・座標系ミス・輪郭異常の早期検知）
7. Artifact化（再現性：入力ハッシュ・設定・バージョン・結果を保存）

---

## 1.3 この層がやらないこと（非責務）

* 3D形状（SDF/mesh）からの2D観測生成（それは Observer層の責務）
* 同化最適化（assimilation層）
* サロゲート学習（surrogate層）
* vti ingest / label正規化（label層）

> ただし「SEM観測を**ObserverSpecと一致する定義**でObs2Dにする」ために、ObserverSpec（target grid、mu、resample点数など）を参照します。

---

# 2. コード構成（SEM取り込み・同化準備層の“読みやすい分割”）

全体設計の `io/` と `observe/` を活かしつつ、SEM固有のオーケストレーションを **独立モジュール**にすると、第三者が理解しやすくなります。

推奨構成（新規 `wafergeo/sem/` を追加：既存の層を汚さない）：

```text
wafergeo/sem/
  base.py                 # 型定義：SEMSpec/SEMCalibration/SEMObsSet 等
  ingest.py               # SEM入力の取り込み（image + contour）
  readers/
    image_reader.py       # TIFF/PNG + メタデータ抽出（pixel size等）
    contour_reader.py     # CSV/JSON/DXF等の輪郭読み込み
  normalize.py            # 輪郭正規化（単位/座標系/閉曲線/穴/再サンプル）
  build_obs2d.py          # contour/image -> Obs2D生成（mask/tsdf/contour/weight）
  weight.py               # weight map生成（confidence/ROI/band重み）
  align.py                # 初期transform推定（任意：Procrustes/ICP/相関など）
  qa.py                   # SEM側QA（輪郭・マスク・TSDF・transform sanity）
  artifacts.py            # SEMObsArtifact/SEMRawArtifact のI/O（ArtifactStore連携）
```

依存関係（重要）：

* `sem/*` は `io/*`（読み込み）と `observe/tsdf2d.py`（2D TSDF生成）、`observe/contour_extract.py`（再サンプリング・整形）を再利用する
* **sem層が observe層の“仕様（grid2d, mu, contour resample）”に揃える**
* metrics/assimilationには依存しない（ただし align推定を metrics側に置く場合は依存方向に注意）

---

# 3. 入出力（Artifactと型）— ここを固定すると運用が壊れない

## 3.1 入力データ（想定）

* SEM画像（任意だが推奨）

  * TIFF/PNG（できればTIFF：メタデータが取りやすい）
* SEM輪郭座標データ（必須）

  * 形式：CSV/JSON（最初はここに絞るのが運用的に強い）
  * 輪郭が複数ある可能性（外形、穴、複数エッジ、複数ROI）

## 3.2 出力データ（必須成果物）

* **SEMObsArtifact**（= 同化に投入する観測）

  * `Obs2D` と同形式（mask/tsdf/contours/weight/transform/meta）
  * 可能なら observerごと（topdown / slice etc）に複数持てる

* **SEMRawArtifact**（任意だが強く推奨）

  * 生の画像
  * 生の輪郭（読み込んだまま）
  * 画像メタ（pixel size, scale bar等）
  * 入力hash、読み込み設定

> SEMObsArtifactは「同化の入力」
> SEMRawArtifactは「監査・再現・トラブル解析の保険」

---

# 4. 座標系と単位（SEM取り込みで最も事故が起きる部分）

SEM取り込みで壊れる原因の8割は **座標系/単位**です。ここを設計として“型”に落とします。

## 4.1 3つの座標系を明示する

1. **Image pixel座標**

   * (u,v) in pixels
   * 原点：画像左上が多い
   * v軸は下向きが多い

2. **SEM物理座標（nm）**

   * (x_sem, y_sem) in nm
   * 原点：画像左上 or 任意基準点（データによる）
   * 軸方向：装置・出力仕様で反転の可能性

3. **Sim/Observer物理座標（nm）**

   * (x_sim, y_sim) in nm
   * ObserverSpecが定義する `grid2d.origin/spacing` に一致させる

## 4.2 変換チェーンをArtifactに保存する

最低限：

* `T_px_to_sem_nm`（pixel→sem物理）
* `T_sem_nm_to_sim_nm`（alignment：similarity/affine）
* 合成 `T_px_to_sim = T_sem_to_sim ∘ T_px_to_sem`

を `SEMObsArtifact.meta.extra` に保存します。

---

# 5. SEM取り込みパイプライン（ステップ別：実装に落ちる粒度）

ここから `sem/ingest.py → normalize.py → build_obs2d.py` の処理を固定手順として定義します。

---

## Step 0: 入力読み込み（readers）

### 0-1 画像読み込み（任意）

* `image_reader.read(path) -> SEMImageRaw`

  * `image: np.ndarray (H,W)`（グレースケール想定）
  * `pixel_size_nm`（取得できれば）
  * `meta`（装置情報、倍率、スケールバー等）

### 0-2 輪郭読み込み（必須）

* `contour_reader.read(path) -> SEMContoursRaw`

  * 複数ループ対応
  * 各ループに `label`（任意：outer/hole/material/interface名）を付けられる構造

**推奨：raw輪郭スキーマ（JSON）**

```json
{
  "coord_system": "pixel" | "nm",
  "units": "px" | "nm" | "um",
  "loops": [
    {"id":"outer_0","role":"outer","points":[[u,v],[u,v],...]},
    {"id":"hole_0","role":"hole","points":[[u,v],[u,v],...]}
  ]
}
```

CSVの場合も同等を表現できるように「loop_id列」「role列」を持たせるのが理想です。

---

## Step 1: Calibration（pixel→nm）の確定

SEM輪郭がpixel座標の場合、nm換算が必須です。

### 1-1 pixel size の優先順位

1. 画像メタから取得できる場合（最優先）
2. 入力Configで明示（運用で安定）
3. スケールバー解析（自動は不確実：評価用途ならOK、運用は推奨しない）

### 1-2 軸向きの統一（重要）

* 画像座標 vが下向きで、sim座標 yが上向きの場合など
  → `flip_y` を `T_px_to_sem_nm` に含める（ここで統一しないと後で地獄）

---

## Step 2: 輪郭正規化（normalize.py）

この工程は “観測定義のブレ” を防ぐために必須です。

### 2-1 ループの健全性チェック

* 重複点の除去（連続で同一点など）
* NaN/Inf除去
* 点数が少なすぎるループは除外（例：N<10）
* 自己交差の簡易検知（任意：警告でよい）

### 2-2 閉曲線判定と閉じ処理

* `||p0 - p_last|| < close_tol_nm` なら閉曲線扱い
* 閉曲線なら `p_last = p0` に強制
* 閉曲線でない場合：

  * `role=open_curve` として残す（後述：距離場は “曲線距離” を使う）
  * 無理に閉じない（誤差が入りやすい）

> 実務では “ROIで切れて閉じない輪郭” が普通に出ます。
> この層が open contour を扱えないと運用で詰みます。

### 2-3 ループの向き（winding）統一

* outerは反時計回り、holeは時計回りなどの規約を決めて強制
* 面積符号で判定し、必要なら反転

### 2-4 再サンプリング（固定点数化）

* `resample_points = 1024` など ObserverSpecと同じ点数に統一
* “弧長”で等間隔にサンプル（uniform indexはNG）

### 2-5 平滑化（任意だが仕様化）

* ノイズが強い場合のみ
* σは **nm単位で指定**（gridが違っても意味が変わらない）
* 適用したら必ず `meta` に記録（同化結果の再現性に影響する）

---

## Step 3: 目標2Dグリッド決定（ObserverSpecに合わせる）

SEMObs2Dは、**sim側Observerが出すObs2Dと同じgrid2d**に落とします。

* `grid2d = ObserverSpec.target_grid_2d`
* ROIもObserverSpecのROIに合わせる（完全一致が基本）

> ここを一致させると
>
> * 画像解像度が違っても比較可能
> * 同化・評価コードが単純
> * Transform適用も一貫

---

## Step 4: SEM輪郭→2D mask 生成（閉曲線の場合）

`build_obs2d.py` の中核です。

### 4-1 塗りつぶし（outer-holes）

* rasterize: outerで塗りつぶし → holesを引く
* 結果：`mask[y,x] ∈ {0,1}`

### 4-2 anti-aliasing（任意）

* CDが数pixel級の時は aliasing が効く
* ただし運用の複雑さが増えるので、最初は **二値maskで固定**が推奨
* どうしても必要なら “subpixel coverage（0..1）” を持てる設計にしておく
  （MetricsのTSDF lossは二値の方が単純・堅牢）

---

## Step 5: 2D TSDF生成（必須）

SEM同化は “輪郭比較” が最終でも、**2D TSDF（距離場）を中間に持つと安定**します。

### 5-1 maskベース（標準）

* `phi = EDT(~mask) - EDT(mask)`（nm）
* `tsdf = clip(phi, -mu, +mu)/mu`

### 5-2 open contour（閉曲線でない場合）の扱い

open contour は maskが作れない/意味が曖昧なケースが多いので、設計として2つのモードを用意します。

* **Mode A（推奨：曲線距離場）**

  * polylineへの最短距離 `d(x)` を計算（nm）
  * `tsdf = clip(d, 0, mu)/mu`（※符号なし：0..1）
  * metadataに `distance_type="unsigned_curve"` を刻む
  * 同化側は TSDF差ではなく “曲線距離” 指標（Chamfer等）を主に使うか、pred側も同じ距離定義に合わせる

* **Mode B（チューブ状領域化）**

  * polylineを太さ `r_nm` のチューブで塗りつぶして疑似maskを作る
  * TSDFを生成（符号は近似）
  * “線の近傍一致” を促す用途に使えるが、r設定が意味を変えるので慎重

このモードは `SEMSpec` で固定します（auto判定も可）。

---

## Step 6: 重み（confidence/weight map）生成（同化のロバスト性に直結）

SEMは「抽出が正しい場所／怪しい場所」があるので、重みを持つと同化が安定します。

### 6-1 weightの標準実装（まずこれで運用が回る）

* `weight = 1`（uniform）
* ただし band内だけ（`|phi|<mu`）に限定するのは強く推奨
  → 形状と関係ない遠方で損失が増えない

### 6-2 画像がある場合の重み（発展）

* エッジ強度（gradient magnitude）が強い箇所を重くする
* ノイズが強い領域、チャージング領域などを軽くする（手動ROIマスクでも良い）

### 6-3 contour品質由来の重み（輪郭しか無い場合）

* 輪郭点の局所曲率が極端に高い箇所を軽くする（ノイズ由来の尖り）
* “輪郭抽出時のconfidence” があればそれを点群→画像へ拡散してweight mapにする

---

## Step 7: 初期アライメント（Transform）推定（任意だが現場では重要）

この層で「初期transform」を作るかどうかは運用ポリシーですが、設計としては **ここで生成してArtifactに保存**できるようにしておくと強いです。

### 7-1 Transformモデル（まずはこれ）

* similarity（scale + rotation + translation）
* 必要なら affine（shear含む）

### 7-2 推定手法（選べる設計：プラグイン化）

* **(A) contour-based（Procrustes/ICP）**

  * SEM contour点集合と、sim側の初期予測 contour（または既知テンプレ）を合わせる
* **(B) TSDF相関（phase correlation風）**

  * obs.tsdf と pred.tsdf の相関で平行移動を粗推定 → 回転/スケールは探索
* **(C) fiducial / mark**

  * パターンマークがある場合は最強（現場運用向き）
* **(D) 手動seed**

  * 実務で一番早いことも多い。seedだけ入れて同化で微調整

> 初期transformは “正解” である必要はなく、
> **同化が探索しやすい範囲に収まること**が目的です。

---

## Step 8: QA（sem/qa.py）

SEM側のQAは運用の生命線です。特に座標系ミスを早期検出します。

最低限のQA：

* units/pixel sizeが妥当か（nmに換算できたか）
* contourがROIの範囲に入っているか（大きく外れていないか）
* maskが空でないか／極端に小さすぎないか
* tsdfが[-1,1]に収まるか、NaN/Infがないか
* transformが異常でないか（スケールが極端、回転が大きすぎる等）
* ループ数が異常でないか（ノイズ抽出）

QA結果は `SEMObsArtifact.qa.json` として保存し、FAILでもartifactを作るか（運用次第）を規約化します。

* 例：FAILなら `status=FAIL` として lossにペナルティを入れられるようにする（同化が落ちない）

---

# 6. SEMObsArtifact のスキーマ（設計図として固定）

SEMObsArtifact は “同化に投入する唯一の観測” なので、メタと再現性情報を厚くします。

推奨レイアウト（Zarr/ディレクトリでも可）：

```text
sem_obs/{artifact_id}/
  obs2d/
    grid2d.json
    mask.npy                # (Y,X) uint8  optional（open curveなら無い/空でもよい）
    tsdf.npy                # (Y,X) float16 [-1,1] or [0,1]（distance_typeで区別）
    weight.npy              # (Y,X) float16 optional
    contours.json           # loops（open/closed, role, points_nm）
    debug/
      image_thumbnail.png   # optional
      overlay.png           # optional（運用で効く）
      band_mask.npy
      exposed_hint.npy      # optional
  transform/
    T_px_to_sem_nm.json
    T_sem_nm_to_sim_nm.json
  raw/
    image_path.txt          # optional
    contour_path.txt
    raw_contours.json       # optional（追跡用）
  qa.json
  meta.json                 # schema_version/profile_id/config_hash/generator_version/input_hash/...
```

**重要メタ（必須）**

* `observer_spec_name`（どの観測定義に揃えたか）
* `mu_nm`
* `distance_type`（signed_region / unsigned_curve / tubular 等）
* `coord_convention`（y軸反転等）
* `input_hash`（画像・輪郭ファイルのhash）
* `config_hash`（正規化、rasterize、weight、alignの設定）

---

# 7. 設計の拡張性（後から手法追加・編集しても運用が壊れない工夫）

SEM取り込みは「現場都合で入力形式も抽出方法も変わる」ので、**プラグイン境界を明確に**しておきます。

## 7.1 プラグイン化すべき部品

1. **ContourReader**（CSV/JSON/装置独自）
2. **ImageReader**（TIFF/PNG/メタ解析）
3. **ContourExtractor**（画像から輪郭抽出：任意）
4. **TSDF2DEngine**（EDT / polyline解析距離 / 近似chamfer）
5. **WeightBuilder**（uniform / image-gradient / confidence / ROI）
6. **AlignmentEstimator**（Procrustes/ICP/相関/fiducial）
7. **QA checks**（現場ルールが増える）

各プラグインに

* `name/version`
* `capabilities`（入力タイプ、次元、必須メタ等）
* `method_card`（説明・注意点・推奨用途）
  を持たせると、手法追加後も運用が壊れにくいです。

---

# 8. 実行単位（CLI/ワークフロー）— “同化準備”をDAGの1ノードにする

この層の処理は重く、また定義が変わると結果が変わるので、DAGの単位を固定します。

## 8.1 コマンド例（設計）

* `sem ingest`：RawArtifactを作る
* `sem build-obs2d --observer topdown_cdsem_v2`：SEMObsArtifactを作る（キャッシュあり）
* `sem align --method procrustes`：初期transformを付与（任意）

これにより、

* 「どのObserver定義で作った観測か」
* 「どのSDF手法で2D TSDFを作ったか」
  がartifactで追えるようになります。

---

# 9. この層の“現場での典型的な問題”と、設計での対策

## 9.1 単位ミス（um↔nm、pixel size未設定）

**対策**

* Calibrationを必須（pixel→nmが決まらない場合はFAIL）
* metaに必ず units/pixel_size_nm を保存
* QAで “CDがあり得ない桁” を検知（例：nmで10^9はおかしい）

## 9.2 y軸反転・回転（左右反転等）

**対策**

* coordinate convention を型として持つ（flip_y等）
* 初期transform推定で mirror を許す/許さないを設定化
* overlay画像（raw image + contour + mask）をartifactに保存（運用で即効）

## 9.3 輪郭が閉じていない（ROIで切れている）

**対策**

* open contour を第一級として扱う（distance_type="unsigned_curve"）
* 目的関数は contour chamfer を主に、TSDF差は補助か “曲線距離同士” に限定

## 9.4 複数輪郭（外形+穴、複数コンポーネント）

**対策**

* loopsに outer/hole/unknown を持たせる
* “最大ループだけ使う”などをSpecで選べる
* QAでループ数が急増したら警告

---

# 10. まとめ（SEM取り込み・同化準備層の設計要点）

* **Obs2D（mask/2D TSDF/contour/weight/transform）**に正規化してArtifact化するのがゴール
* 座標系と単位を“型”と“Transform chain”で固定し、metaに必ず刻む
* 輪郭は閉曲線だけでなく **open contourも第一級**として扱い、距離場生成モードを持つ
* weight（confidence）と初期transformは運用安定性に直結するので、生成・保存できる設計にする
* すべてをプラグイン境界で分割し、後で手法追加しても運用が壊れないようにする
* QAとoverlay（可視化）をartifactに残し、「なぜ合わないか」を後で追えるようにする

---

