以下は、これまで検討した内容（**multi-material ≤5 の vti voxel**, **SDF学習サロゲート / SDF→メッシュ学習サロゲート**, ただし **SEM同化は両者とも最終的に2D輪郭比較**）を前提にした、**コードの設計図（アーキテクト視点）**です。

* コード全体設計（リポジトリ構成／責務分離／依存分割）
* ワークフロー設計（DAG・Artifact・キャッシュ・QA）
* 拡張設計（新規SDF手法・新規観測・新規メッシュ手法追加のやり方）
* データサイエンティスト視点での「各層の役割」「SDF手法」「データ変換」「SEM同化」まとめ

を、**レビューワーが把握しやすく、第三者が修正・追加しやすい**ことを最優先に、できるだけ欠けなく具体化します。

---

# 0. この設計の最重要コンセプト（ブレ防止）

## 0.1 “同化・評価”の共通インターフェースを固定する

サロゲートは2系統に分かれても、SEM同化では最終的に2D輪郭を比較します。
そこで **SDF系でもMesh系でも最終的に同じ `Obs2D` を出す**ことをコード設計の中心に置きます。

> **Canonical Comparison Domain = 2D Observation（mask / 2D TSDF / contour）**

これにより、

* サロゲート（SDF/mesh）の違いは **GeometryAdapter** が吸収
* 同化・評価コードは **Obs2D同士の比較**だけで完結
* 実務で頻出の “評価コードが分岐して再現性が壊れる” を防げます

## 0.2 SDFは「基盤の正規形（canonical）」として常に残す

Mesh学習をしたい場合も、

* 入力vtiの揺れ吸収
* QA（距離性・ラベル整合）
* 2D観測生成（mask→2D TSDF→contour）
  の安定性は SDF が最も高いので、

> **SDFを生成→必要ならMesh派生（Artifact）を作る**
> を基本線にします。

## 0.3 Artifact駆動（再現性・キャッシュ・監査性）を基盤に埋め込む

全処理は「入力＋設定＋バージョン → 出力Artifact」の関数として扱い、

* 再計算を避ける（重い前処理を守る）
* どの条件で作ったか追跡可能
* 学習・同化の再現性担保

を最初から設計に入れます。

---

# 1. 全体アーキテクチャ（責務分離：レビューしやすい層構造）

## 1.1 レイヤ構造（固定）

コードは次のレイヤに分割し、**上位が下位に依存する一方向**にします（循環依存禁止）。

1. **core**：型・メタ・共通ユーティリティ（純粋・副作用なし中心）
2. **io**：vti/画像/輪郭/Artifact入出力（副作用をここに隔離）
3. **label**：ラベル正規化・ラベルQA（multi-material入口の堅牢化）
4. **sdf**：EDT/TSDF/narrow-band/reinit/境界特徴（数学処理）
5. **mesh**：SDF→mesh、mesh属性付与、点群生成、mesh QA
6. **observe**：SDF/mesh→2D観測（mask/2D TSDF/contour）を統一生成
7. **metrics**：2D TSDF差・輪郭距離・CD誤差などの比較指標
8. **assimilation**：最適化ループ（optimizerプラグイン）、ログ、成果物化
9. **datasets**：manifest、split、loader（DSが触る入口）
10. **surrogate**：モデルI/F・学習/推論ラッパ（SDF系／mesh系）
11. **qa**：チェック群、ゴールデンテスト、回帰テスト
12. **cli**：コマンド群（研究→運用の導線）
13. **docs**：仕様（スキーマ・プロファイル・Observer定義）と設計書

---

# 2. リポジトリ構成（具体案：レビューワーの読み順つき）

## 2.1 ディレクトリ構成

```text
wafer_sdf/
  README.md
  pyproject.toml
  docs/
    ARCHITECTURE.md
    SCHEMA.md
    PROFILES.md
    OBSERVERS.md
    QA.md
    CONTRIBUTING.md
  configs/
    materials.yaml
    profiles/
      sem_assim_profile_v2.yaml
      surrogate_sdf_profile_v2.yaml
      surrogate_mesh_profile_v2.yaml
    observers/
      topdown_cdsem_v1.yaml
      slice_xsem_v1.yaml
  wafergeo/
    core/
      types.py
      grid.py
      meta.py
      hashing.py
      registry.py
      errors.py
      logging.py
      math_utils.py
    io/
      vti_reader.py
      image_reader.py
      contour_io.py
      artifact_store.py
      zarr_io.py
      mesh_io.py
    label/
      normalize.py
      qa.py
      materials.py
    sdf/
      edt.py
      tsdf.py
      band.py
      boundary_features.py
      reinit.py
      qa.py
    mesh/
      build.py
      attrib.py
      sampling.py
      qa.py
    observe/
      base.py
      topdown.py
      slice.py
      rasterize.py
      tsdf2d.py
      contour_extract.py
      qa.py
    metrics/
      tsdf_loss.py
      contour_metrics.py
      cd_metrics.py
      alignment_metrics.py
    assimilation/
      runner.py
      optimizers/
        base.py
        nelder_mead.py
        cmaes.py
      objective.py
      reports.py
    datasets/
      manifest.py
      splits.py
      loaders.py
      schema.py
    surrogate/
      base.py
      sdf_models/
      mesh_models/
      training/
        trainer.py
        eval.py
        packaging.py
    qa/
      golden_shapes.py
      regression.py
      assertions.py
  tests/
    test_label_normalize.py
    test_sdf_edt.py
    test_observer_topdown.py
    test_observer_slice.py
    test_mesh_build.py
    test_sem_pipeline.py
    test_assimilation_objective.py
```

## 2.2 レビューワーの推奨読み順（把握の最短経路）

1. `docs/ARCHITECTURE.md`（全体像）
2. `wafergeo/core/types.py, grid.py, meta.py`（型とメタ設計）
3. `wafergeo/io/artifact_store.py`（Artifact思想・キャッシュ）
4. `wafergeo/observe/base.py`（2D観測統一I/F）
5. `wafergeo/sdf/edt.py`（SDF基礎）
6. `wafergeo/mesh/build.py`（SDF→mesh）
7. `wafergeo/assimilation/runner.py`（同化ループの骨格）

---

# 3. データモデル（型定義：第三者が安全に拡張できる骨格）

以降は **“型（dataclass）中心の設計”**にします。
処理のI/Oが明確で、レビュー・テスト・追加が圧倒的に楽になります。

## 3.1 Grid/Volume/Meta（核心）

```python
# wafergeo/core/grid.py
from dataclasses import dataclass
from typing import Literal, Tuple, Sequence, Optional, Dict

AxisOrder = Literal["ZYX", "YXC", "XYZ", "YX"]  # 例：内部は ZYX を標準に固定推奨
SampleLocation = Literal["cell_center", "grid_point"]

@dataclass(frozen=True)
class GridSpec:
    dim: int                      # 2 or 3
    spacing: Tuple[float, ...]    # (sz, sy, sx) [nm] for ZYX
    origin: Tuple[float, ...]     # (oz, oy, ox) [nm]
    axis_order: AxisOrder         # how ndarray is ordered
    sample_location: SampleLocation
    units: str                    # "nm" etc

@dataclass(frozen=True)
class Meta:
    schema_version: str
    profile_id: str
    config_hash: str
    generator_version: str
    git_commit: str
    input_hash: str
    created_at: str
    extra: Dict[str, str]
```

## 3.2 Label / TSDF（multi-material ≤5）

```python
# wafergeo/core/types.py
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class MaterialSpec:
    ids: List[int]               # e.g. [0,1,2,3,4]
    names: List[str]             # e.g. ["void","resist","oxide","si","metal"]
    void_id: int                 # typically 0
    priority: List[int]          # for resolving conflicts if needed
    ignore_in_exposure: List[bool]  # e.g. resist ignored? etc

@dataclass(frozen=True)
class LabelVolume:
    grid: GridSpec
    material: MaterialSpec
    material_id: np.ndarray      # shape (Z,Y,X), dtype uint16/uint32
    meta: Meta

@dataclass(frozen=True)
class TSDFVolume:
    grid: GridSpec
    material: MaterialSpec
    mu_nm: float
    tsdf: np.ndarray             # shape (M,Z,Y,X) float16/float32, values in [-1,1]
    # optional auxiliaries
    d_boundary: np.ndarray | None   # (Z,Y,X), float16
    pair_code: np.ndarray | None    # (Z,Y,X), uint8
    meta: Meta
```

> **設計のキモ**：TSDFのチャネル順は `material.ids` と一致させ、常にメタに保存。
> これで third party が混乱しません。

## 3.3 Mesh / PointCloud

```python
@dataclass(frozen=True)
class MeshGeom:
    vertices: np.ndarray     # (V,3) float32 [nm]
    faces: np.ndarray        # (F,3) int32
    face_mat_in: np.ndarray  # (F,) uint8  material index (0..M-1)
    face_mat_out: np.ndarray # (F,) uint8  neighbor material index
    face_is_exposed: np.ndarray # (F,) bool
    grid: GridSpec
    material: MaterialSpec
    meta: Meta

@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray       # (N,3)
    normals: np.ndarray      # (N,3)
    pair_code: np.ndarray    # (N,) uint8
    meta: Meta
```

## 3.4 2D観測（同化・評価の共通出口）

```python
@dataclass(frozen=True)
class ObserverSpec:
    kind: str                  # "topdown_exposed" | "slice"
    target_grid_2d: GridSpec   # dim=2
    roi: Dict[str, float]      # physical ROI or index ROI
    params: Dict[str, float]   # z0, thickness, etc
    contour: Dict[str, float]  # resampling, smoothing, etc

@dataclass(frozen=True)
class Obs2D:
    grid2d: GridSpec
    mask: np.ndarray           # (Y,X) uint8
    tsdf: np.ndarray           # (Y,X) float16/float32, [-1,1]
    contours: list             # list of polylines (N_i,2) in physical coordinates
    weight: np.ndarray | None  # (Y,X) float16 (SEM only)
    transform: Dict | None     # alignment parameters (SEM only)
    meta: Meta
```

---

# 4. Artifactストア設計（再現性・キャッシュ・運用をコードに固定）

## 4.1 Artifactの思想

すべての処理は「入力Artifact + 設定 → 出力Artifact」の形で実装します。

* ステップは副作用を持たず（＝同じ入力なら同じ出力）
* 出力はartifact_idで一意に管理
* 生成に使った設定・バージョン・入力ハッシュを必ず埋め込む

## 4.2 ArtifactStore API（設計案）

```python
class ArtifactStore:
    def exists(self, artifact_id: str) -> bool: ...
    def read_meta(self, artifact_id: str) -> Meta: ...
    def write(self, artifact_type: str, payload: object, meta: Meta) -> str: ...
    def load(self, artifact_id: str) -> object: ...
```

### artifact_id 生成規約（必須）

* `input_hash`：入力ファイル（vti/SEM輪郭）hash
* `profile_id`：生成する特徴セット（後述）
* `config_hash`：mu、ObserverSpec、メッシュ簡略化条件など
* `generator_version`：コードバージョン

→ `artifact_id = sha256(input_hash + profile_id + config_hash + generator_version)`

## 4.3 ディスク上のレイアウト（例）

```text
artifacts/
  label/{artifact_id}/...
  sdf/{artifact_id}/...
  mesh/{artifact_id}/...
  obs2d/{artifact_id}/...
  sem/{artifact_id}/...
  runs/assim/{run_id}/...
  models/{model_id}/...
```

> レビューワーが「どの成果物が何か」を辿れるよう、artifact_idと同時に `manifest.json` を必ず書く設計にします。

---

# 5. Config / Profile / Observer定義（運用と保守の鍵）

## 5.1 MaterialSpec（materials.yaml）

```yaml
schema_version: "materials/v1"
void_id: 0
materials:
  - id: 0
    name: void
    ignore_in_exposure: true
    priority: 0
  - id: 1
    name: resist
    ignore_in_exposure: false
    priority: 10
  - id: 2
    name: oxide
    ignore_in_exposure: false
    priority: 20
  - id: 3
    name: si
    ignore_in_exposure: false
    priority: 30
  - id: 4
    name: metal
    ignore_in_exposure: false
    priority: 40
```

## 5.2 FeatureProfile（SDFサロゲート用）

```yaml
schema_version: "profile/v2"
profile_id: "surrogate_sdf_profile_v2"
mu_nm: 200.0
build:
  tsdf_per_material: true      # M<=5なら true を標準
  boundary_features:
    d_boundary: true
    pair_code: true
  process_features:
    top_height: true
    thickness_z: true
  stats:
    per_material: true
    per_interface: true
storage:
  tsdf_dtype: float16
  chunk: [1, 64, 256, 256]     # (M, Z, Y, X) の例
  compressor: zstd
qa:
  label_qa: true
  sdf_grad_check: true
```

## 5.3 FeatureProfile（Meshサロゲート用）

```yaml
schema_version: "profile/v2"
profile_id: "surrogate_mesh_profile_v2"
mesh:
  from: "tsdf_material"     # or "binary"
  simplify:
    enabled: true
    target_faces: 200000
  pointcloud:
    enabled: true
    num_points: 65536
  face_attrib:
    enable_interface_labels: true
    enable_exposed: true
observe:
  default_observers:
    - "topdown_cdsem_v1"
    - "slice_xsem_v1"
```

## 5.4 ObserverSpec（topdown_cdsem_v1.yaml）

```yaml
schema_version: "observer/v1"
name: "topdown_cdsem_v1"
kind: "topdown_exposed"
target_grid_2d:
  dim: 2
  spacing: [2.0, 2.0]     # (sy, sx) nm
  origin: [0.0, 0.0]
  axis_order: "YX"
  sample_location: "cell_center"
  units: "nm"
roi:
  x_min: 0.0
  x_max: 20000.0
  y_min: 0.0
  y_max: 20000.0
params:
  exposure_axis: "z+"
  exposed_materials: ["resist","oxide","si","metal"]   # voidは除外
contour:
  extract_level: 0.0
  smoothing_sigma: 0.0
  resample_points: 1024
tsdf2d:
  mu_nm: 200.0
```

> **ObserverSpecのYAML化**は運用で極めて重要です。
> 「SEM輪郭の定義」「比較しているCDの定義」をコードから切り離し、Artifactに保存できるようになります。

---

# 6. 処理層の設計（各層の責務・入出力・拡張点）

ここからは層ごとに「何をするか」「I/O」「実装ポイント」「拡張方法」を明確にします。

---

## 6.1 Ingest / Label正規化層（label/normalize.py）

### 目的（DS視点）

* vtiの揺れ・材料IDの不整合を吸収し、**“学習/同化に使える安定なLabelVolume”**を作る
* 異常（未割当、飛び地、ID変換ミス）を最初に検出する

### 主な処理

1. `.vti`読み込み（io/vti_reader.py）
2. axis_orderを内部標準（例：ZYX）に変換
3. spacing/origin/unitsをGridSpecへ
4. material_id正規化（void埋め、範囲チェック）
5. label QAレポート生成（label/qa.py）

### 主要I/F

```python
def ingest_vti_to_label(path: str, materials: MaterialSpec, grid_override: GridSpec|None) -> LabelVolume
```

### 拡張点

* vti内のスカラ名が変わる、複数配列（mask群）を受けたい等
  → `io/vti_reader.py` で吸収、`normalize.py` は出力を統一するだけにする

---

## 6.2 SDF層（sdf/edt.py, tsdf.py, boundary_features.py）

### 目的（DS視点）

* 離散ラベルを連続幾何場に変換し、学習しやすい形（TSDF）にする
* multi-material ≤5 を活かし、**材料別TSDFスタック**を標準化

### コア手法：EDT SDF（材料ごと）

* `mask_m = (label==m)`
* `phi_m = EDT(~mask_m) - EDT(mask_m)`（inside negative）
* `tsdf_m = clip(phi_m, -mu, +mu)/mu`（[-1,1]）

### 追加の境界特徴（軽量で効果が高い）

* `d_boundary = min_m |phi_m|`
* `pair_code = encode(argmin|phi_m|, second_min|phi_m|)`

### 主要I/F

```python
class SDFBuilder:
    def build_tsdf(self, label: LabelVolume, mu_nm: float, build_boundary_features: bool) -> TSDFVolume
```

### 品質保証（sdf/qa.py）

* TSDF範囲（[-1,1]）
* band内 |∇φ| 統計（任意）
* mu・spacing整合

### 拡張点（新SDF手法の追加）

* `SDFEngine` インターフェースを用意し、EDT実装は `EDTSDFEngine` として登録
* 将来：subvoxel SDF、GPU EDT等は別Engineで追加

```python
class SDFEngine(Protocol):
    name: str
    def compute_phi_per_material(self, label: LabelVolume) -> np.ndarray  # (M,Z,Y,X)
```

---

## 6.3 Mesh層（mesh/build.py, attrib.py, sampling.py）

### 目的（DS視点）

* メッシュ学習をしたい場合に、**SDFを介して一貫したメッシュ**を生成する
* meshの可変トポロジ問題を避けるため、**学習用点群（PointCloud）**も標準生成する

### Mesh生成の方針（multi-material対応）

* 材料mの `phi_m=0` 等値面から面を作る（またはbinary phi）
* faceごとに `mat_in`, `mat_out`（界面ペア）を付与
* `is_exposed`（外気に接する面）を付与（topdown観測と直結）

### 主要I/F

```python
class MeshBuilder:
    def build(self, tsdf: TSDFVolume, mode: str, simplify: dict, attrib: dict) -> MeshGeom

class PointSampler:
    def sample(self, mesh: MeshGeom, num_points: int) -> PointCloud
```

### Mesh QA

* 面積/体積の極端な異常
* face属性（mat_in/out）がmaterial範囲内か
* 露出面の存在/欠落チェック（topdown想定なら重要）

### 拡張点

* meshing手法の変更（VTK FlyingEdges / marching cubes / OpenVDBなど）
* リメッシュ/簡略化アルゴリズムの差し替え
* 点群サンプリング戦略（曲率重み、露出面優先など）

---

## 6.4 Observer層（observe/*）— “同化と評価の中心”

### 目的（DS視点）

* **SDF/meshどちらの形状でも、最終的に同じObs2D（mask/2D TSDF/contour）を作る**
* SEM同化の比較対象を統一し、評価・同化コードを共通化する

### 観測の出力仕様（固定）

1. `2D mask`（Y,X）
2. `2D TSDF`（mask→EDT→TSDF）
3. `contours`（TSDF=0の等値線、再サンプリング）

### I/F（共通）

```python
class Observer(Protocol):
    name: str
    def observe_from_tsdf(self, tsdf: TSDFVolume, spec: ObserverSpec) -> Obs2D
    def observe_from_mesh(self, mesh: MeshGeom, spec: ObserverSpec) -> Obs2D
```

> **重要**：両方のメソッドを必須にすると実装が重くなるので、
> 実際は `GeometryAdapter` を設けて `observe(geometry)` に一本化するのが運用上おすすめです。

### 具体Observer

* `TopDownExposedObserver`

  * `exposed_material_id(x,y)` を作り、exposed mask → 2D TSDF → contour
  * mesh入力の場合は、露出面をZ-buffer的に2D rasterize → 2D TSDF → contour
* `SliceObserver`

  * tsdfなら断面を切ってmask化
  * meshなら平面切断でpolyline→mask化→2D TSDF

### contour抽出の標準化（observe/contour_extract.py）

* TSDF=0等値線抽出（実装差で結果がブレやすいので統一）
* スムージング有無
* 再サンプリング（固定点数に統一）

### Observer QA（observe/qa.py）

* contourが閉じているか
* mask面積が異常に小さい/大きい
* ROI外への逸脱がないか
* 2D TSDFが[-1,1]に収まるか

---

## 6.5 Metrics層（metrics/*）

### 目的（DS視点）

* 同化・評価の目的関数を、サロゲート表現に依存せず定義する
* SEM抽出誤差を吸収するため **2D TSDF差（band+robust+weight）を標準損失**にする
* レポート用に輪郭距離・CD誤差も標準装備

### 標準損失（推奨）

* `L_tsdf = Σ w(x,y) * ρ(φ_pred - φ_obs)` ただし `|φ_obs|<μ` のbandだけ
* `w` は SEM confidence を含む（SEMObsArtifactに保存）

### 補助指標

* 対称Chamfer（輪郭）
* Hausdorff（品質チェック）
* CD誤差（測定定義付き）

---

## 6.6 SEM取り込み・同化準備（io/contour_io.py, observe/tsdf2d.py, metrics/alignment_metrics.py）

### 目的（DS視点）

* SEM輪郭を “同化可能な観測表現（Obs2D）” に正規化
* 必ずメタ（pixel_size, origin, transform）を保持し再現性を担保

### SEMObsArtifact生成フロー

1. contour → 2D mask（rasterize）
2. mask → 2D TSDF（EDT→TSDF）
3. confidence/weight map（あるなら）生成
4. アライメント推定（任意）→ Transform 保存

> 同化・評価側は「Obs2D同士の比較」だけにする
> SEM固有処理はここで完結させる

---

## 6.7 Assimilation層（assimilation/*）

### 目的（DS視点）

* surrogate model（SDF/meshどちらでも）を呼び出し、Observerで2D観測を作ってSEMと比較し、パラメータを最適化する
* 最適化手法は後から変えられる（Optimizerプラグイン）

### 設計ポイント

* 同化は「objective関数」を最小化するだけに抽象化
* objective内は必ず

  1. surrogate推論
  2. observerで2Dへ
  3. loss計算
     の3手順に固定（ログもここで統一）

### 主要I/F

```python
class Optimizer(Protocol):
    name: str
    def minimize(self, objective_fn, x0, bounds, budget, seed) -> dict

class AssimilationRunner:
    def run(self, sem_obs: Obs2D, surrogate, observer_spec: ObserverSpec,
            optimizer: Optimizer, x0, bounds, config) -> "AssimilationRunArtifact"
```

### 出力（AssimilationRunArtifact）

* best params
* best predicted Obs2D（tsdf, contour）
* loss履歴、CD誤差履歴
* 使ったObserverSpec、profile、versions

---

## 6.8 Surrogate層（surrogate/*）— 表現が2系統でもI/Fは統一

### 設計方針

* SDFサロゲート：出力は `TSDFVolume`（材料別TSDFスタック）
* Meshサロゲート：出力は `MeshGeom` または `PointCloud`（推奨はMeshGeom or pointcloud→mesh復元は別途）

しかし同化では `Obs2D` が必要なので、**GeometryAdapter** を用意します。

### Surrogate I/F（最小）

```python
class SurrogateModel(Protocol):
    name: str
    def predict(self, recipe_params: dict) -> object  # TSDFVolume or MeshGeom

class GeometryAdapter:
    def to_obs2d(self, geom: object, observer: Observer, spec: ObserverSpec) -> Obs2D
```

### 学習パイプライン（trainer）

* 学習時にも ObserverSpec を使い、2D観測損失（または評価）を同じコードで回せるようにする

---

# 7. ワークフロー設計（DAG：処理単位・キャッシュ単位を固定する）

ここは「実行をどう組み立てるか」を具体化します。
**全ステップが Artifact を入出力**にすることで、研究と運用を一致させます。

---

## 7.1 データ生成（Ground Truth）DAG

### SDF系（必須）

1. `ingest_vti` → `LabelArtifact`
2. `build_tsdf_features` → `SDFArtifact`
3. `build_observations`（topdown/slice複数）→ `Obs2DArtifact(sim_gt)`

### Mesh系（必要なら）

4. `build_mesh_from_tsdf` → `MeshArtifact`
5. `build_observations_from_mesh` → `Obs2DArtifact(mesh_gt)`
   ※基本は (3) で十分。mesh経由の2D観測は検証用（同一性チェック）に使う。

---

## 7.2 SEM観測DAG

1. `ingest_sem_contours` → `SEMContourArtifact`
2. `build_sem_obs2d` → `SEMObsArtifact`（Obs2D形式）
3. `align_sem_to_sim`（任意）→ Transformを `SEMObsArtifact` に埋め込む or 別Artifact

---

## 7.3 サロゲート学習DAG（2系統）

### SDFサロゲート

* dataset：`(recipe_params, SDFArtifact or Obs2DArtifact(sim_gt))`
* 学習目標：

  * 3D TSDF（材料別）
  * +λ * 2D観測損失（ObserverSpecを固定して生成）

### Meshサロゲート

* dataset：`(recipe_params, MeshArtifact/PointCloud + Obs2DArtifact(sim_gt))`
* 学習目標：

  * mesh/point損失（自由）
  * +λ * 2D観測損失（同じObserverSpec）

> **同じObserverSpecを学習・評価・同化で共有**することで、定義ブレを消します。

---

## 7.4 同化（Optimization）DAG（共通）

1. SEMObsArtifact（Obs2D）を入力
2. optimizerが候補パラメータ生成
3. surrogate.predict → geometry
4. observer → predicted Obs2D
5. loss（2D TSDF差＋輪郭補助）計算
6. best更新
7. 結果を AssimilationRunArtifact として保存

---

# 8. “第三者が追加しやすい”拡張ガイド（新手法追加の最短ルート）

## 8.1 新しいSDF手法を追加したい（例：subvoxel、GPU）

1. `wafergeo/sdf/engines/xxx.py` を作り `SDFEngine` を実装
2. `registry.py` に登録（またはentry point）
3. `profiles/*.yaml` の `sdf_engine: "xxx"` で切替可能にする
4. `tests/test_sdf_xxx.py` を追加（ゴールデン形状で回帰）

## 8.2 新しいmesh生成手法を追加したい

1. `MeshBuilder` を実装（SDF→mesh）
2. 生成後に `attrib.py` で `mat_in/out`, `is_exposed` を付ける工程は共通にする
   → mesh手法を変えても下流が壊れない

## 8.3 新しいObserver（測定定義）を追加したい

1. `observe/*.py` に新しい Observer クラス追加
2. YAML（configs/observers）を追加
3. `tests/test_observer_new.py` を追加（SDF入力とmesh入力で同じObs2Dが出るか）

## 8.4 新しい同化損失・最適化手法を追加したい

* `metrics/*` にloss追加
* `assimilation/optimizers/*` にoptimizer追加
* `AssimilationRunner` は objective を差し替え可能な設計にする

---

# 9. QA・回帰テスト設計（研究→運用の橋渡し）

## 9.1 QAはArtifactに保存する（監査可能）

各ステップのQA結果（統計・閾値判定）を `qa.json` としてArtifact内に保存する。

### Label QA（必須）

* 材料別体積・割合
* 連結成分数（飛び地）
* void比率
* material_id範囲チェック

### SDF QA（必須）

* TSDF値域
* mu・spacing・符号規約の整合
* band内 |∇φ| 統計（任意、ただし同化で曲率等を使うなら必須）

### Mesh QA（推奨）

* face属性整合（mat_in/out）
* 極端な面積・体積
* non-manifold検知（できる範囲で）

### Observer QA（必須）

* contour閉曲線性
* mask面積の極端値
* TSDF値域

## 9.2 ゴールデン形状（回帰テストの核）

* 積層膜（薄膜＋段差）
* 単純トレンチ
* 2材料界面、3材料界面
* topdownとsliceで期待輪郭が分かる形状

これを `qa/golden_shapes.py` で生成し、SDF/mesh/observer/metricsの回帰基準にする。

---

# 10. データサイエンティスト観点まとめ（各層の“役割”と“使い方”）

ここはDSがプロジェクトに参加しやすくするための説明です。

## 10.1 SDF系サロゲート（学習の狙いと利点）

* 出力（教師）を **材料別TSDF** にすることで、

  * 材料ごとの形状差が自然に学習される
  * 連続場なので学習が安定
* 同化・評価は Observer で2Dへ落として輪郭比較
  → “最終指標”と学習が整合しやすい

推奨データ：

* 入力：recipe params +（必要なら初期TSDF）
* 出力：TSDFVolume（M≤5）

損失：

* 3D TSDF損失（材料別）
* * 2D観測損失（ObserverSpec固定）

## 10.2 Mesh系サロゲート（学習の狙いと利点）

* meshは構造が可変なので、基盤側で

  * mesh属性（界面ペア、露出面）
  * 学習用点群（固定N）
    を用意して、モデル選定の自由度を確保する
* 最終比較は同じく2D輪郭（Obs2D）
  → メッシュで学習しても同化コードは共通

推奨データ：

* 入力：recipe params
* 出力：MeshGeom or PointCloud（+必要なら mesh QA）

損失：

* mesh/point損失（モデルに応じて）
* * 2D観測損失（ObserverSpec固定）

## 10.3 SDF手法の要点（基盤の標準）

* 多材料（≤5）なので **材料別SDFを全部生成**してもコストが現実的
* 学習・同化では **TSDF（±μでクリップ）**が基本
* `d_boundary` と `pair_code` は軽量で界面情報が増える“効きが良い補助特徴”

## 10.4 データ変換（SDF↔mesh、3D→2D）

* SDF→mesh：等値面抽出＋属性付与（mat_in/out, exposed）
* 3D→2D（Observer）：topdown/slice を明示的に定義し、必ず2D TSDFと輪郭を生成

## 10.5 SEM同化（DS視点の要点）

* 観測（SEM輪郭）は“真値”ではない → confidence/weight map を扱う
* 比較は `Obs2D`（2D TSDF + contour）に統一
  → 同化目的関数が安定・レビューしやすい
* 2D輪郭だけの同化は不定性がある
  → 複数Observer（topdown + 複数slice）か、物理指標制約（膜厚など）を追加できる設計にしておく

---

# 11. 実装のための最終チェックリスト（設計図としての抜け防止）

この設計をコードに落とすときの「必須」項目です（抜けると後で破綻しやすい）。

1. **Obs2Dを同化・評価の唯一の比較インターフェース**にする
2. **ObserverSpecをYAMLで管理**し、Artifactに保存する
3. **材料別TSDF（≤5）を標準生成**し、mu・符号規約・順序をメタに保存
4. **ArtifactStore + キャッシュキー**を最初から実装し、再計算地獄を防ぐ
5. **QA結果をArtifactに保存**し、CIでゴールデン形状回帰を回す
6. mesh系は **MeshGeom + PointCloud** を標準成果物にして学習の自由度を確保
7. 同化Runnerは **surrogate表現に依存しない**（GeometryAdapterで吸収）

---

必要であれば、次のステップとしてこの設計図を「そのまま開発開始できる」状態にするために、以下を具体例付きで提示できます（追加で質問は不要です）：

* `Artifact schema（Zarrキー一覧）`を確定した `SCHEMA.md`
* `ObserverSpec`（topdown/slice複数）の標準セット
* `CLIコマンド設計`（引数、入出力、生成Artifact）
* “新SDF手法追加”のテンプレコード（`SDFEngine` 実装例）
* “新Observer追加”のテンプレコード（テスト付き）

この回答自体が設計図になるように、I/F・モジュール境界・ワークフロー・拡張点を欠けなく書きました。
