以下は **Observer層（`wafergeo/observe/*`）**を、あなたの全体設計（Artifact駆動／SDF・Meshサロゲート両対応／SEM同化は最終的に2D輪郭比較）に合わせて、**設計図としてそのまま実装・レビューに使える粒度**で詳細化したものです。
（ここでは「Observer層＝観測オペレータ」を、データ同化の観点で **H(x)**：3D形状→2D観測へ写像する“中核”として設計します。）

---

# 1. Observer層の役割（責務）と最重要な設計方針

## 1.1 Observer層の役割（何を提供する層か）

Observer層は、**どんな形状表現（SDF/TSDF, Label, Mesh）でも、最終的に同じ形式の 2D観測（Obs2D）を生成する**層です。

* 入力：

  * `TSDFVolume`（材料別TSDF stack）
  * `LabelVolume`（material_id）
  * `MeshGeom`（三角形メッシュ＋界面属性）
* 出力（固定）：

  * `Obs2D`：

    1. `mask2d`（2Dのinside/outside定義が固定）
    2. `tsdf2d`（mask→2D距離場→TSDF化、符号・μ・単位が固定）
    3. `contours2d`（TSDF=0等値線から抽出し、再サンプリング・座標系が固定）
    4. optional：`weight2d`（SEM観測のconfidence等）
    5. optional：`debug_maps`（露出材料ID、露出高さ、スライス高さなど：運用と解析に効く）

> **この層で“2D観測の定義（何を輪郭と呼ぶか）”を固定**しないと、
> 同化・評価・学習の定義が分岐して、運用の再現性が壊れます。
> そのため ObserverSpec（YAML）で観測定義を管理し、Artifactに埋め込みます。

---

## 1.2 最重要設計方針：Observerは「比較ドメインを統一する装置」

あなたの状況では

* サロゲートの学習表現：`SDF/TSDF` か `Mesh` の2系統
* ただし SEM同化の比較は最終的に `2D輪郭（または2D TSDF）`

です。そこで Observer層は **表現差（SDF vs Mesh）を吸収し、同じ2D観測を出す**ことに集中します。

**結果として：**

* 同化・評価は `Obs2D` 同士を比較するだけで成立
* 2系統のサロゲートでも同じ同化ループに乗る
* “meshだと輪郭の定義が違う” 事故が起きない

---

## 1.3 Observer層の非責務（ここでやらない）

* vti読み込み、材料IDの揺れ吸収（Ingest/Label層）
* SDF生成（SDF層）
* mesh生成（Mesh層）
* 同化最適化（assimilation層）
* アライメント推定（alignmentは metrics/assimilation 側で扱うのが原則）

  * ただし **Transform適用（観測座標変換の適用）**は Observer/Obs2D側で扱えるように設計します（後述）

---

# 2. Observer層のコード構成（レビューしやすい責務分割）

推奨構成（前回案に「拡張・運用」を織り込んだ完成版）：

```text
wafergeo/observe/
  base.py              # 型・I/F・Observer registry、共通ユーティリティ
  factory.py           # ObserverSpec(YAML) -> Observer インスタンス生成
  topdown.py           # TopDownExposedObserver（SDF/Label/Mesh）
  slice.py             # SliceObserver（SDF/Label/Mesh）
  mask_def.py          # "何をmaskとするか" の定義（mask DSL）
  rasterize.py         # mesh/contour -> 2D mask（Z-buffer/scanline/voxelize）
  sample_3d.py         # TSDFの連続スライス（trilinear sampling等）
  tsdf2d.py            # 2D mask/2D polyline -> 2D TSDF（EDT/解析SDFもここ）
  contour_extract.py   # 2D TSDF -> contours（marching squares + resample）
  transform.py         # Obs2Dの座標変換適用（SEM alignment適用）
  qa.py                # Observer QA（contour閉曲線、面積、TSDF範囲など）
  debug.py             # デバッグマップ出力（exposed_id, height_map, etc）
```

### 設計原則（第三者が理解・追加しやすい）

* `topdown.py` / `slice.py` は **観測ロジック（H）**に集中
* `mask_def.py` は「観測対象領域（mask定義）」を宣言的に
* `rasterize.py`, `tsdf2d.py`, `contour_extract.py` は **汎用部品**として独立
* QAは必ず `qa.py` に集約し、Observerが増えても共通で検証できる

---

# 3. データ型とI/F（Observer層の“契約”を固定する）

## 3.1 ObserverSpec（観測定義：YAMLで固定・versioning必須）

Observer層の運用を壊さない最大の鍵は、観測定義をコードに埋め込まず **YAMLで固定**することです。

**ObserverSpec の設計例（概念）**

```yaml
schema_version: "observer/v2"
name: "topdown_cdsem_v2"
kind: "topdown_exposed"     # or "slice"
target_grid_2d:
  dim: 2
  spacing: [2.0, 2.0]       # (sy, sx) nm
  origin: [0.0, 0.0]        # (oy, ox) nm
  axis_order: "YX"
  sample_location: "cell_center"
  units: "nm"

roi:                        # 2D ROI in physical coordinates (nm)
  x_min: 0.0
  x_max: 20000.0
  y_min: 0.0
  y_max: 20000.0

mask_definition:
  kind: "exposed_union"     # 観測したい輪郭の定義（重要）
  axis: "z+"
  include_materials: ["resist","oxide"]    # 露出対象（例）
  ignore_materials: ["void"]               # 露出探索で無視する材料（例）
  slab_thickness_nm: 0.0                   # >0なら薄い厚みでOR（任意）

tsdf2d:
  mu_nm: 200.0
  engine: "edt"              # "edt" | "polyline_exact" 等
  band_only: true

contour:
  source: "tsdf"             # "mask" or "tsdf"
  level: 0.0
  smoothing_sigma_nm: 0.0
  resample_points: 1024
  simplify_tolerance_nm: 0.0

debug:
  save_exposed_id: true
  save_height_map: true

qa:
  check_closed: true
  max_open_contours: 0
  min_area_nm2: 1.0e4
```

> **これが“測定定義そのもの”**です。
> 同じ名前（`topdown_cdsem_v2`）なら、誰がいつ走らせても同じ輪郭になります。

---

## 3.2 Obs2D（比較の共通ドメイン）

Observer層が出すものは常に `Obs2D`（固定）です。

**Obs2D（推奨の拡張版）**

```python
@dataclass(frozen=True)
class ContourLoop:
    points_xy: np.ndarray     # (N,2) float32 in physical (nm), closed (first==last) optional
    is_hole: bool
    label: str | None         # "exposed_union"など（任意）
    meta: dict

@dataclass(frozen=True)
class Obs2D:
    grid2d: GridSpec
    mask: np.ndarray          # (Y,X) uint8, inside=1 outside=0
    tsdf: np.ndarray          # (Y,X) float16/float32, [-1,1]
    loops: list[ContourLoop]  # outer + holes
    weight: np.ndarray | None # (Y,X) float16 (SEM only)
    transform: dict | None    # alignment applied or to be applied
    debug_maps: dict[str, np.ndarray]       # optional (exposed_id, height_map, etc)
    meta: Meta                # includes observer spec hash/version
```

---

## 3.3 Observerインターフェース（最小で、追加しやすい）

Observerは基本的に

* **Geometry（TSDF/Label/Mesh）**
* **ObserverSpec**

から `Obs2D` を返します。

推奨は **GeometryAdapter** による一本化ですが、Observer層だけで完結させるならこうします：

```python
class Observer(Protocol):
    name: str
    def observe(self, geom: object, spec: ObserverSpec) -> Obs2D:
        ...
```

内部では

* `isinstance(geom, TSDFVolume)`
* `isinstance(geom, LabelVolume)`
* `isinstance(geom, MeshGeom)`
  で分岐します（この分岐はbase.pyに寄せ、各observer実装は“必要最小限”に）。

---

# 4. Observer層の共通処理フロー（すべてのObserverが従う固定手順）

Observer層の処理は、**手法や入力形式が違っても必ず同じ“型”の手順**にします。
これがレビューしやすさと、運用の一貫性の鍵です。

## 4.1 共通手順（固定）

1. **ObserverSpecの検証**

   * grid2dのunits、spacing>0、ROI整合、μ>0 など
2. **2Dターゲット格子を確定**

   * `grid2d`（origin/spacing/size）を決定（ROIと整合）
3. **Mask定義に従って 2D mask を生成**

   * TSDF/Label入力：投影/スライス/露出探索
   * Mesh入力：ラスタライズ or 平面切断→ラスタライズ
4. **2D TSDF（距離場）を生成**

   * `mask -> 2D signed distance -> TSDF化`
   * μと符号規約を固定
5. **Contour抽出**

   * `tsdf=0` 等値線（推奨）または mask境界（可）
   * 再サンプリング（N点固定）
6. **Debug map生成（任意）**

   * exposed_material_id_2d、height_map、slice_zなど
7. **QA**

   * TSDF値域、contour閉曲線性、面積・周長など
8. **Obs2Dを返す**

   * metaに observer spec hash / version / engine signature を必ず記録

---

# 5. “mask_definition” 設計（観測の意味を宣言的に固定する）

Observer層で最も事故が起きるのは「何を輪郭として比較しているか」が曖昧になることです。
そこで `mask_definition` を **DSL（宣言的定義）**として設計します。

## 5.1 mask_definition の主要パターン（本用途で必要なもの）

### A) binary_solid（solid vs void）

* inside = 非void（または指定材料の和集合）
* cross-section SEMで “形状外形” を比べるときに有効

### B) material_union（特定材料の和集合）

* inside = {resist, oxide} のような union
* “レジスト形状だけ”“膜だけ”など材料依存輪郭に使う

### C) exposed_union（topdown露出領域）

* +z方向から見て最初に当たる材料が対象材料なら inside
* CD-SEM（トップダウン）で “露出エッジ” を比較したいときの標準

### D) exposed_material_equal（露出材料が特定材料）

* inside = topdownで露出材料が “resist” の画素
* 材料識別が一部できる場合や、工程ステップごとの定義に使う

### E) interface_pair（材料i-j界面）

* inside = “i側” と定義し、輪郭（界面）を比較
* 断面SEMで材料界面（例：oxide-si）を比べる等

### F) slab（厚み付きスライス）

* z=z0 ではなく、z0±t/2 のスラブで OR
* SEMが有限厚みで観測される/ノイズ低減したいときに使える

> これらを `mask_def.py` に集約し、Observer（topdown/slice）は
> 「どうやって3D→2Dのサンプルを得るか」だけに集中させます。

---

# 6. TopDownExposedObserver（topdown.py）を“実装できる粒度”で詳細化

トップダウンSEM（CD-SEM）で最重要なのがこれです。

## 6.1 目的（何を観測として作るか）

* 2D mask：**露出領域（exposed_union / exposed_material）**
* 2D TSDF：その露出領域の境界からの距離（符号付き）
* contour：露出領域の境界（閉曲線）を固定点数に再サンプル

## 6.2 TSDF/Label入力：露出材料IDと露出マスク生成（高速・決定的）

### 6.2.1 露出探索（z+ の場合）

入力：`label[z,y,x]`（cell_center, ZYX）

定義：

* `ignore_ids`（例：void）を無視しながら +z 方向（上側）から探索し、
* 各 (y,x) に対して「最初に見つかる非ignore材料」を露出材料とする

**ベクトル化アルゴリズム（設計案）**

```python
valid = ~isin(label, ignore_ids)             # (Z,Y,X) bool
# z+ 視点なら上から（Z-1→0）探索
valid_rev = valid[::-1, :, :]                # reverse z

any_hit = valid_rev.any(axis=0)              # (Y,X)
k = valid_rev.argmax(axis=0)                 # (Y,X) 0..Z-1 (注意: any_hit=Falseでも0になる)
z_hit = (Z - 1) - k                          # (Y,X)

exposed_id = full((Y,X), fill=void_id)
exposed_id[any_hit] = label[z_hit[any_hit], y, x]
height_map_nm[any_hit] = origin_z + (z_hit+0.5)*spacing_z
```

### 6.2.2 exposed_union mask の作り方

* `exposed_id_2d` が `include_materials` に含まれる画素を inside=1
* それ以外は outside=0

**注意（工程ルール）**

* resistを無視する/するは工程によって変わる
  → `materials.yaml` の `ignore_in_exposure` と、ObserverSpecの `ignore_materials` を併用して決める

### 6.2.3 TSDF入力の場合（labelが無い／推論出力）

TSDFサロゲート推論後は label が無いことが多いので、露出探索はTSDFから行います。

**推奨ルート**

1. `label_zyx = label_from_tsdf(tsdf_stack)`（SDF層で規約固定済み）
2. 上の labelルートと同じアルゴリズムで exposed_id を作る

> 露出探索の規約が散ると結果が変わるので
> TSDF→label化規約は SDF層に固定し、Observerはそれを使う設計が安全です。

---

## 6.3 Mesh入力：topdown観測の作り方（2方式を設計として用意）

Mesh入力のtopdownは難所です。運用・保守を考えると、2つの実装ルートを用意しておくのが強いです。

### 方式1：Z-bufferラスタライズ（高速・連続、実装は中程度）

入力：`MeshGeom(vertices, faces, face_is_exposed, face_mat_in/out)`

手順：

1. rasterize対象 face を選ぶ

   * `face_is_exposed=True` を基本
   * さらに +z視点なら “上向き成分がある面” を優先する等はオプション
2. 2D格子（grid2d）上に各faceを投影し、**深度（z）最大**を取る（z-buffer）
3. 各pixelで最大zを与えたfaceの材料を `exposed_id_2d` とする
4. `exposed_union mask` → 2D TSDF → contour

**rasterize.py の責務**

* triangle → 2D pixel coverage（barycentric）
* depth interpolation
* z-buffer更新（max z）
* materials assignment

### 方式2：Voxelize→Labelルート（実装が簡単・精度は格子依存）

Meshを一旦ターゲットの3Dグリッドにボクセライズし、Label/TSDFルートでtopdownを作ります。

* 利点：既存の label/topdown 実装が再利用できる（保守が楽）
* 欠点：格子依存、薄膜や微細形状でエイリアシング

**設計としての扱い**

* `spec.params["mesh_topdown_mode"] = "zbuffer" | "voxelize"`
* まず運用開始は voxelizeでもよい（壊れにくい）
* 需要・精度要求が上がった段階で z-buffer実装を追加しても、I/Fは変わらない

---

## 6.4 露出エッジ（輪郭）のための2D TSDF生成

topdownで最終比較は輪郭であることが多いので、`mask→2D TSDF` を必ず生成します。

### 2D TSDF生成（tsdf2d.py）

* `dist_in = EDT(mask)`
* `dist_out = EDT(~mask)`
* `phi = dist_out - dist_in`
* `tsdf = clip(phi, -mu, +mu)/mu`

**2D spacing の扱い**

* grid2dの `spacing=(sy,sx)` を EDT に渡す
* これを統一すると、SEM pixelサイズが違っても物理距離で比較できます

---

# 7. SliceObserver（slice.py）を“実装できる粒度”で詳細化

断面SEM（X-SEM）や「高さでのCD」など、topdown以外を扱うための観測です。

## 7.1 仕様（最低限の第一段階：軸平行スライス）

最初は運用を壊さないために **軸平行**から入るのが良いです：

* `z = z0` のスライス（XY断面）
* `x = x0` のスライス（YZ断面）
* `y = y0` のスライス（XZ断面）

ObserverSpecで `axis` と `coord_nm` を指定します。

```yaml
kind: "slice"
mask_definition:
  kind: "binary_solid"   # or interface_pair etc
params:
  axis: "z"
  coord_nm: 3500.0
  slab_thickness_nm: 20.0   # 任意（厚み付き）
```

---

## 7.2 TSDF入力：連続スライス（trilinear sampling を標準にする）

Meshスライスは連続ですが、TSDFを単に `z_index` で切ると “meshと比較した時にズレ” が出ます。
そこで SDF層のTSDFを **連続場としてサンプリング**できるようにします（sample_3d.py）。

### 推奨：TSDFを平面上にサンプル → 2D mask → 2D TSDF

手順：

1. 3D TSDF stack（または binary TSDF）を指定平面上にサンプル

   * まずは axis-aligned なら 2Dスライス + 1D補間
2. `mask_definition` に従って2D maskを作る

   * 例：binary_solidなら `tsdf_binary < 0`
   * material_unionなら `argmin(tsdf_stack)` で label化して条件を満たすか
3. 2D mask→2D TSDF→contour

### slab_thickness（厚み付き）の実装（任意）

* `z0±t/2` の範囲で複数スライスを取り OR（maskの和集合）
* “SEMの観測厚み”を近似でき、ノイズ耐性が上がる

---

## 7.3 Mesh入力：平面切断 → polyline → mask化

Meshのスライスは以下が標準です：

1. plane（例：z=z0）と各triangleの交差を計算し、線分を得る
2. 線分を接続して polyline（閉曲線/開曲線）へ
3. polylineを2D格子にラスタライズして mask化
4. 2D TSDF→contour（同じ経路に統一）

**設計上の重要点**

* 2D TSDF生成と contour抽出は SDF入力と共通にする（比較の定義が一致する）
* meshの切断polylineは debugとして保存できるが、最終はtsdf経由で統一

---

# 8. rasterize.py の設計（mesh/contour→maskの共通部品）

Observerの実装が増えるほど、ラスタライズを共通化しないと保守不能になります。

## 8.1 提供するべき関数群（最小セット）

### A) polyline → mask（2D）

* 目的：SEM輪郭、meshスライス輪郭、mask定義の補助に使う
* 必須：

  * outer + holes を扱える（穴を塗りつぶさない）
  * 物理座標→grid2d index変換が統一される

### B) triangles → z-buffer + id（topdown mesh）

* 目的：meshからexposed_id_2dを作る
* 必須：

  * 深度のmax/min
  * material idの格納

### C) voxelize（mesh→label）オプション

* 目的：保守のための簡易ルート
* 必須：

  * 速度はそこそこで良いが、再現性・決定性が必要

---

# 9. tsdf2d.py の設計（mask/contour から 2D TSDF を作る “唯一の入口”）

2D TSDF生成は同化のロバスト性に直結します。
ここが分岐すると結果が変わるため、**Observer層内で唯一の入口に固定**します。

## 9.1 API設計

```python
def tsdf2d_from_mask(mask: np.ndarray, grid2d: GridSpec, mu_nm: float, engine: str="edt") -> np.ndarray
```

### engine の選択（将来の拡張）

* `"edt"`：ラスタ化maskからEDT（標準、決定的）
* `"polyline_exact"`：輪郭線分距離＋windingで解析SDF（SEMのエイリアシング低減に有効）
* `"approx_chamfer"`：高速近似（探索用途）

ただし、**同化や教師生成の標準は “edt”**として固定し、他は評価対象にするのが運用上安全です。

---

# 10. contour_extract.py の設計（2D TSDF→輪郭の標準化）

輪郭比較の事故は「輪郭抽出の細部差」で起きます。
そこで contour抽出を標準化します。

## 10.1 入力・出力と規約

* 入力：2D TSDF（[-1,1]）
* 出力：

  * `ContourLoop[]`（outer/holeが識別される）
  * pointsは物理座標（nm）
  * resample_points により固定点数化

## 10.2 標準ステップ

1. 事前平滑化（任意、σをnm指定、適用したらメタに保存）
2. 等値線抽出（level=0）
3. loop判定（閉曲線に近いか）
4. ループの向きで outer/hole を推定（必要なら面積符号）
5. 再サンプリング（弧長でN点に）
6. 簡略化（toleranceがあれば）

> 重要：mask境界から取る方法もありますが、SDF/mesh/SEMで統一しやすいのは
> **TSDF=0等値線**の方です（同化の距離場と同じものから輪郭が出る）。

---

# 11. transform.py（Obs2Dへの座標変換適用）

SEM同化ではアライメント（回転・スケール・平行移動）が絡みます。
推定は metrics/assimilation 側でやるとしても、**適用はObs2Dに対して確実にできる必要**があります。

Observer層に `transform.py` を置く理由：

* 「輪郭」「mask」「tsdf」全部を同じTransformで扱うための標準実装が必要
* 同化中に transform を更新しても、適用の実装が一箇所にまとまる

**設計**

* Transformは 2D similarity/affine をまず標準化
* `apply_transform_to_obs2d(obs, T, target_grid2d)` を提供

  * contourの座標変換
  * mask/tsdfのワーピング（補間：nearest/linearを指定）
  * weight mapのワーピング

---

# 12. Observer QA（qa.py）：観測結果を“入口で止める”

Observer出力が壊れると同化が暴れます。
QAは必須で、Artifactに保存します。

## 12.1 最低限のQA項目

* TSDFが [-1,1] に収まる
* NaN/Infなし
* mask面積が極端に小さい/大きい（min_area/max_area）
* contourが閉じているか（許容open数）
* contourがROI外へ逸脱していないか
* ループ数が異常に多い（ノイズの兆候）

## 12.2 デバッグ出力（運用で効く）

* exposed_id_2d のヒストグラム（材料が想定通りか）
* height_map の統計（露出高さが想定範囲か）
* slice位置の確認（z0が正しいか）

---

# 13. ObserverFactory / Registry（運用で壊れない生成・追加方式）

Observerを増やすプロジェクトでは、**生成と登録**を固定しないと管理不能になります。

## 13.1 registry（base.py）

* `register_observer(kind, class)`
* `get_observer(kind)`
* `list_observers()`

## 13.2 factory（factory.py）

* YAML読み込み→スキーマ検証→ObserverSpec生成
* `kind` を見てObserverインスタンス生成
* spec hash（定義のハッシュ）を計算してMetaに埋め込み

---

# 14. ワークフロー統合（Artifact駆動での build_observations）

Observer層は “同化中に毎回呼ぶ” だけでなく、**GT生成・学習評価の前処理**でも呼びます。
ここが運用の要なので、実行単位を固定します。

## 14.1 build_observations の実行単位

入力：

* geometry artifact（SDFArtifact / MeshArtifact / LabelArtifact）
* observers（YAML名のリスト）

出力：

* `Obs2DArtifact` を observerごとに生成（キャッシュ）

例：

* `obs2d/sim_gt/{artifact_id}/{observer_name}/...`
* `obs2d/pred/{run_id}/{observer_name}/...`

> 同じObserverSpecで同じ入力なら同じObs2Dが得られる（決定性）
> → 再現性・比較可能性が担保されます。

---

# 15. 追加・修正がしやすい設計（第三者向けの拡張ガイド）

## 15.1 新しいObserverを追加する手順（必須要件）

1. `observe/new_kind.py` を追加し、`Observer` を実装
2. `register_observer("new_kind", NewObserver)`
3. `configs/observers/new_kind_v1.yaml` を追加
4. ゴールデン形状でテストを追加（SDF入力とMesh入力でObs2D一致のテストが理想）
5. `docs/OBSERVERS.md` に測定定義を書き、運用に残す

## 15.2 新しいmask_definitionを追加する手順

* `mask_def.py` に `kind` を追加
* Observerは mask_defを呼ぶだけなので、topdown/slice側の変更は最小になる

## 15.3 新しい2D TSDFエンジン（例：polyline解析SDF）

* `tsdf2d.py` に engine を追加
* ただし **標準はedtのまま**にして、評価ベンチで比較可能にする

---

# 16. データサイエンティスト観点でのObserver層の位置づけ（役割まとめ）

Observer層は、データ同化の言葉で言うと **観測オペレータ H(x)** です。

* `x`：サロゲートが出す3D形状（TSDF or mesh）
* `H(x)`：SEMが見ているものに合わせた2D表現（mask/2D TSDF/輪郭）
* 比較：`y_obs`（SEM観測）と `H(x)` を損失で結び、パラメータ推定やモデル更新を行う

**この層の価値**

* サロゲート表現が変わっても同化の定義が変わらない
* 学習・評価・同化が同じ観測定義で整合する
* SEM輪郭の定義（どの材料／露出ルール／スライス位置）がYAMLで固定され、実験・運用が追跡可能

---

# 17. 実装のための“最小セット”チェックリスト（Observer層）

運用開始に必要な最小実装（これが揃うと回り始める）：

1. `ObserverSpec` 読み込み＋検証（YAML→dataclass）
2. `TopDownExposedObserver`（TSDF/Label入力対応、meshはvoxelizeでも可）
3. `SliceObserver`（TSDF入力対応、meshは後追いでも可）
4. `mask_def`（exposed_union / binary_solid / material_union）
5. `tsdf2d_from_mask`（EDT）
6. `contour_extract`（TSDF=0、resample）
7. QA（TSDF範囲、閉曲線性、面積）
8. Artifact保存（Obs2DArtifact）

その後、精度と速度要求に応じて

* meshのz-buffer topdown
* meshのplane cut slice
* polyline解析SDF
  などを追加しても、I/Fと運用は壊れません（これが設計の狙いです）。

---

必要なら次に、Observer層について **実装テンプレ（関数一覧＋擬似コード＋ログ仕様＋テスト雛形）**を、ファイル単位でそのまま貼れるレベルに落として提示します。
