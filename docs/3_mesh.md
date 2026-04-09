以下は **Mesh層（`wafergeo/mesh/*`）**を、**設計図（コード全体設計の一部）**として “実装に落ちる粒度” で詳細化したものです。
（全体設計・ワークフローは先ほどの通り：Artifact駆動、Observerで2D観測に統一、SDFサロゲート／Meshサロゲート両対応、SEM同化は最終的に2D輪郭比較。）

---

# 1. Mesh層の位置づけと責務

## 1.1 Mesh層の目的

Mesh層は、主に次の3つを担当します。

1. **SDF（TSDF）や multi-label（material_id）から、三角形メッシュ（および点群）を生成**

   * サロゲート学習（Meshベース）
   * 解析・可視化
   * 別表現（mesh）を介した処理・特徴量化

2. **multi-material（≤5）を“壊さず”に扱える形で、メッシュに材料・界面属性を付与**

   * `mat_in / mat_out`（界面ペア）
   * `is_exposed`（外気/voidに露出している面）
   * 面積などの統計（界面面積行列など）

3. **下流（observe/SEM同化）で使いやすい形に整える**

   * plane cut を取りやすい
   * topdown rasterize をしやすい（exposed面が分かる）
   * QA（壊れたmeshを早期発見）

> 重要：SEM同化の最終比較は `Obs2D（2D TSDF + contour）` に統一しているので、
> Mesh層の設計でも **「mesh → Observer → Obs2D」** の安定性を最優先します。

---

## 1.2 Mesh層の非責務（ここでやらない）

* `.vti`読み込み・ラベル揺れ吸収（Ingest/Label層の責務）
* SDF生成・TSDF化（SDF層の責務）
* 2D輪郭の抽出・2D TSDF化（observe層の責務）
* 同化最適化（assimilation層の責務）

---

# 2. Mesh層の課題整理（multi-material・運用で必ず詰まるところ）

Mesh層は、単に「TSDF=0等値面を取ればよい」では運用で破綻しやすいです。設計で潰すべき課題は次。

## 2.1 multi-materialの“界面重複”問題

材料 m ごとに `phi_m=0` をメッシュ化すると、材料mと材料nの界面が **両方のメッシュに重複**して現れ得ます。
→ 面積計算・学習・可視化がややこしくなる

**対策（設計として選択可能にする）**

* **Interface Meshモード**：界面ペア(i,j)の面だけを採用（重複なし）
* **Material Shellモード**：材料ごとに外殻を保持（重複はあるが材料単位で扱いやすい）

両方を持てる設計にして、用途で切替します。

---

## 2.2 マルチラベルの“隙間・クラック”問題

ラベル境界（i-j）に対して別々の手法で面を生成すると、微小なズレが生じて **隙間や自交差**が起きることがあります。

**対策**

* multi-labelから直接境界面を生成できるメッシング手法を“候補”として持つ

  * VTKは segmentation/labelマスク向けに `vtkDiscreteFlyingEdges3D` や `vtkSurfaceNets3D` の利用を推奨しています。 ([VTK][1])
  * multi-label SurfaceNetsの研究では「多ラベルで非重複・高品質な面を生成し、シャープ境界の保持」も扱っています。 ([PMC][2])
  * 2023年の multi-label ボリュームからの境界面復元手法（数理形態学）も「薄い構造や曖昧ケースでも構成曖昧性が少なく、Marching Cubesより疎なメッシュ」と主張しています。 ([サイエンスダイレクト][3])

Mesh層は「SDF→mesh」だけでなく、「label→mesh」も第一級に扱える設計が強いです。

---

## 2.3 大規模3Dでの性能とメッシュ品質

* Marching Cubesは古典的で広く使われます。 ([ACM Digital Library][4])
* VTKの Flying Edges は大規模向けの最適化・並列化を重視した isocontouring 実装で、性能面で有利なことが多い一方、**退化三角形が出る可能性**がある旨も明記されています。 ([VTK][1])

→ **mesh生成後の “標準クリーンアップ工程” を設計に固定**します（後述）。

---

# 3. コード構成（Mesh層のファイル責務と読みやすさ）

先ほどの全体設計に沿いつつ、Mesh層は “手法評価・追加” を想定して、**Extractor（生成）と Postprocess（整形）を分離**します。

推奨（Mesh層のみ拡張版）：

```text
wafergeo/mesh/
  build.py              # パイプライン組み立て（extract→clean→attrib→simplify→sample→qa）
  extractors/
    base.py             # MeshExtractorインターフェース + capabilities
    iso_flying_edges.py # TSDF/スカラー場 → mesh（VTK FlyingEdges）
    iso_marching_cubes.py
    discrete_flying_edges.py # label → mesh（VTK DiscreteFlyingEdges）
    surfacenets_multilabel.py # multi-label SurfaceNets（検証/比較用）
    dual_contouring.py  # sharp feature重視の候補（Dual Contouring）
  attrib.py             # mat_in/out, is_exposed, area 等の付与
  repair.py             # degenerate除去、重複頂点統合、法線向き、穴検出 等
  simplify.py           # decimation/remesh（オプション）
  sampling.py           # 点群生成（学習用固定N）
  stats.py              # interface area matrix など集約統計
  qa.py                 # Mesh QA（品質保証）
```

> 既存構成（`build.py, attrib.py, sampling.py, qa.py`）を維持しつつ、
> `extractors/repair/simplify/stats` を追加すると “手法を増やしても壊れない” 設計になります。

---

# 4. Mesh層の入出力契約（型・Artifact）

## 4.1 入力

Mesh層は入力を2系統で受けます。

* **(A) TSDFVolume**（SDFサロゲートやSDF層の出力）

  * `tsdf[m,z,y,x]`（M≤5）
  * `GridSpec`（nm, spacing/origin）
  * `MaterialSpec`

* **(B) LabelVolume**（multi-label segmentationとして直接メッシュ化したい場合）

  * `material_id[z,y,x]`
  * `GridSpec`（cell_center基準が前提）
  * `MaterialSpec`

## 4.2 出力（MeshGeom + PointCloud）

* `MeshGeom`

  * vertices, faces
  * `face_mat_in / face_mat_out`
  * `face_is_exposed`
  * （任意）face_area, normals

* `PointCloud`（学習用、固定N）

  * points, normals
  * pair_code（界面ID）
  * 追加属性（is_exposedなど）

## 4.3 MeshArtifact（保存設計）

Mesh層は “学習・再現性” を前提に、少なくとも以下をArtifactに残します。

* `mesh/main.(vtp|ply|npz)`（最小はnpzで vertices/faces）
* `mesh/attrs/*.npy`（face属性）
* `pointcloud/*`（固定N点群と属性）
* `stats/interface.json`（界面面積行列など）
* `qa.json`
* `meta.json`（extractor名・version、設定、入力hash、grid_spec）

---

# 5. Mesh生成モード（このプロジェクトで扱うべき “2大系列”）

Meshは大別して2系列が必要です。
（“SDF→mesh”と“label→mesh”を両方持てる設計が、本プロジェクトの強みになります。）

---

## 5.1 TSDF（スカラー場）→ mesh（等値面抽出：isocontouring）

### 代表Extractor

* **Marching Cubes**（古典・広く普及） ([ACM Digital Library][4])
* **Flying Edges**（大規模向け・並列化） ([VTK][1])
* （候補）Dual Contouring（鋭い特徴保持・簡略化に強い） ([ワシントン大学コンピュータサイエンス部][5])

### パイプライン（設計として固定）

1. **対象フィールド選択**

   * `field = tsdf[m]`（材料ごと）または binary tsdf
2. **等値面抽出**

   * iso_value = 0.0
   * grid spacing/origin を正しく設定（nm）
3. **標準クリーンアップ**

   * 退化三角形除去（FlyingEdgesで出る可能性がある） ([VTK][1])
   * 重複頂点統合（許容誤差付き）
   * 法線計算（必要なら）
4. **属性付与（mat_in/out, exposed）**
5. **（任意）簡略化・リメッシュ**
6. **点群サンプリング**
7. **QA + 統計生成**
8. Artifact化

### multi-materialでの重要設計：重複界面をどう扱うか

TSDFチャネルを材料ごとに等値面抽出すると界面が重複しやすいので、**2つのモードを実装**します。

* **Material Shellモード**

  * 各材料mの殻（phi_m=0）をそのまま保持
  * `mat_in=m` を固定、`mat_out` は後段で推定
  * “材料別に学習したい”ケースで扱いやすい

* **Interface Meshモード（推奨：同化・統計に強い）**

  * 生成した面に `mat_in/out` を付与した後、

    * `(mat_in < mat_out)` の面だけを保持
    * あるいは `(mat_in, mat_out)` を正規化して重複排除
  * 界面が一意になるので、`interface_area[i,j]` が安定する

---

## 5.2 Label（multi-label segmentation）→ mesh（離散境界面抽出）

### 代表Extractor（候補を複数持つ価値が高い）

* VTKは「segmented label mask をメッシュ化したいなら `vtkSurfaceNets3D` や `vtkDiscreteFlyingEdges3D` を検討」と明記しています。 ([VTK][1])
* multi-label SurfaceNetsは「多ラベルで非重複・高品質な面、シャープ境界保持」などを扱っています。 ([PMC][2])
* 2023の multi-label 境界面復元手法（数理形態学）も、薄い構造や曖昧ケースでの頑健性を主張しています。 ([サイエンスダイレクト][3])
* multi-label Marching Cubes（M3C系）は最近の研究でも言及されます。 ([arXiv][6])

### この系列を持つメリット（運用目線）

* multi-materialの **隙間/クラック**が起きにくい（手法次第）
* `mat_in/out` をラベルから直接付与しやすい（界面ペアが明確）
* SDFの“実装差”に依存しない基準メッシュとして使える
  → SDF手法評価（ベンチ）にも効く

### パイプライン（設計として固定）

1. label volume を入力
2. discrete surfacing（ラベル境界抽出）
3. 標準クリーンアップ
4. `mat_in/out` を “ラベル遷移” で確定
5. exposed面（片側がvoid）
6. 点群、QA、統計
7. Artifact化

---

# 6. Extractorプラグイン設計（多手法評価・追加に耐える）

本プロジェクトは「様々なSDF手法を評価できる」だけでなく、Mesh手法も増えます。
そこで Mesh層も **Extractorをプラグイン化**します（SDF Engineと同じ思想）。

## 6.1 MeshExtractorインターフェース（設計案）

```python
class MeshCapabilities(TypedDict):
    input_types: list[str]    # ["tsdf", "label"]
    dim: list[int]            # [3] (2Dはobserveへ、など)
    supports_anisotropic_spacing: bool
    can_output_nonoverlapping_multilabel: bool
    deterministic: bool

class MeshExtractor(Protocol):
    name: str
    version: str
    def capabilities(self) -> MeshCapabilities: ...
    def extract_from_tsdf(self, tsdf: TSDFVolume, req: MeshRequest) -> RawMesh: ...
    def extract_from_label(self, label: LabelVolume, req: MeshRequest) -> RawMesh: ...
```

> RawMeshは “最小の頂点・面” だけ返す（属性はattrib.pyで付ける）
> → Extractorの責務が膨張しない

## 6.2 MeshRequest（手法差を吸収するパラメータ束）

* `mode`: `"material_shell"` / `"interface_mesh"`
* `materials`: 対象材料（≤5なので全材料でも可）
* `iso_value`: 0.0（TSDF）
* `cleanup`: 退化除去、重複統合など
* `simplify`: target_faces, max_error, preserve_feature_edges
* `seed`: 点群サンプリングの再現性

---

# 7. 属性付与（attrib.py）：meshを“学習・同化・統計”に耐える形にする

## 7.1 face_mat_in / face_mat_out の付与（TSDF由来の場合）

TSDF等値面抽出だけでは界面相手が分かりません。そこで **幾何的に推定**します。

1. faceの法線 `n` を計算（面の向きを仮決め）
2. face重心 `c` を取る
3. 微小オフセット `ε`（例：0.25 voxel）で

   * `c_in = c - ε n`
   * `c_out = c + ε n`
4. それぞれの位置で **ラベル（またはtsdf stackからargmin）を参照**

   * `mat_in = label(c_in)`
   * `mat_out = label(c_out)`

これで界面ペアが得られます。

### TSDFからlabel参照する場合の規約

* `label = argmin(tsdf_m)`（最も負のチャネル）
* 全チャネルが正ならvoid
* tie-breakはMaterialSpec.priority

この規約は SDF層に固定してある想定（ブレ防止）。

## 7.2 exposed面の付与

* `face_is_exposed = (mat_out == void) or (mat_in == void)`
  ※向きの定義次第でどちらを見るかは統一する

## 7.3 面積・界面統計の生成（stats.py）

* `interface_area[i,j]`（i<jで対称化）
* `exposed_area_by_material[m]`
* `num_faces_by_pair_code`

この統計は

* DS特徴量
* QA（異常検知）
* 同化の正則化（露出面積一致など）
  に使えます。

---

# 8. 標準クリーンアップ（repair.py）：必須工程として固定する

Mesh抽出は手法により退化面が出ます（FlyingEdgesでも注意が書かれています）。 ([VTK][1])
そのため Mesh層は “抽出後の標準整形” を必須ステップにします。

## 8.1 クリーニングの最低限セット

* degenerate triangle除去（面積が閾値以下）
* duplicate vertex / duplicate face の削除
* weld（近接点統合：許容誤差 `tol = 1e-6 * bbox_diag` など）
* 法線の再計算（必要なら）
* connected component 分解（必要なら最大成分だけ残す、など）

## 8.2 watertight性が必要な場合（任意）

* もし “mesh→SDFへ戻す” を頑健にしたい場合、内外判定が不安定になることがあります。
  その場合は generalized winding numbers を使ったinside/out判定の採用余地があります。 ([ユーザーサーバー][7])
  （ただし、これは optional backend として分離推奨）

---

# 9. 簡略化・リメッシュ（simplify.py）：運用で壊れない設計

Meshサロゲートは、頂点数が一定でないと学習が難しいので「簡略化」が欲しくなりますが、ここは危険です。

## 9.1 設計上の原則

* **幾何を変える処理はすべて optional にし、必ずパラメータをArtifactに刻む**
* 同化・評価（2D輪郭）に影響が出るので、簡略化は

  * “学習用の派生メッシュ” として保持し、
  * “解析用/比較用のメッシュ” は別に保存する
    が運用上安全です

## 9.2 preserveすべきもの（multi-material向け）

* 界面ペア境界（material境界の線）
* exposed edge（SEM topdownで重要）
* sharp features（必要なら）
  → Dual Contouringは特徴保持と簡略化の枠組みを持つ手法として有名です。 ([ワシントン大学コンピュータサイエンス部][5])
  SurfaceNets拡張でも sharp boundary preservation を扱っています。 ([PMC][2])

---

# 10. 点群生成（sampling.py）：Meshサロゲートを運用可能にする核

Meshを直接学習するのは難しいので、このプロジェクトでは **点群（固定N）を標準生成**するのが強いです。

## 10.1 サンプリング戦略（複数実装を用意）

* **Area-uniform**：面積比例でサンプル（基本）
* **Stratified by interface**：界面ペアごとに均等or重み付け（multi-materialで効く）
* **Exposed-prior**：exposed面を多め（SEM同化・topdownが主なら効く）
* **Curvature-prior（任意）**：曲率が高い領域を多め（エッジ学習に効く）

## 10.2 点群に持たせる属性（学習で効く）

* `normal`
* `pair_code`（界面種別）
* `is_exposed`
* （任意）局所曲率や面積ウェイト

すべて **再現性のためseed固定**を設計に入れます。

---

# 11. Mesh QA（qa.py）：壊れたmeshを入口で止める

Meshは壊れても「見た目で気づきづらい」ことが多いので、数値QAが必須です。
SDF層と同じく、QA結果はArtifactへ保存します。

## 11.1 QA項目（最低限）

* NaN/Infの存在
* degenerate triangle率
* connected component数、最大成分比
* face_mat_in/out が material範囲内か
* interface area行列の対称性（i-j と j-i）
* exposed面の存在（topdown運用なら重要）
* bboxが異常でない（origin/units崩れ検知）

---

# 12. MeshBuilder（build.py）：Mesh層の“組み立て”を固定する

Mesh層は「抽出手法が増える」前提なので、**build.pyは薄く、手順を固定**します。

## 12.1 buildの擬似コード（レビューしやすい粒度）

```python
def build_mesh_artifact(input_geom, mesh_profile, store) -> MeshArtifactId:
    # 1) Extract
    raw_mesh = extractor.extract(input_geom, req)

    # 2) Repair/Cleanup (standard)
    mesh = repair.clean(raw_mesh, cfg.clean)

    # 3) Attribute assignment (mat_in/out, exposed, area)
    mesh = attrib.annotate(mesh, input_geom, cfg.attrib)

    # 4) Mode enforcement (interface_mesh / material_shell)
    mesh = attrib.filter_mode(mesh, cfg.mode)

    # 5) Simplify (optional)
    mesh_simpl = simplify.apply(mesh, cfg.simplify)

    # 6) PointCloud (optional but recommended)
    pc = sampling.sample(mesh_simpl or mesh, cfg.pointcloud)

    # 7) Stats + QA
    stats = stats.compute(mesh, pc)
    qa = qa.run(mesh, stats, cfg.qa)

    # 8) Write artifact (mesh + attrs + pc + stats + qa + meta)
    return store.write("mesh", payload, meta)
```

---

# 13. “様々な手法を評価できる”ためのMesh層側の設計ポイント

このプロジェクトはSDF手法だけでなく、**mesh生成法の違い**が最終的な2D輪郭比較に影響し得ます。
そこで Mesh層でも以下を規約化すると運用が回ります。

## 13.1 Method Card（手法説明カード）を必須にする

Extractorごとに

* name/version
* 入力種（TSDF/label）
* 長所短所（multi-label non-overlap 等）
* 推奨用途（同化向け／学習向け／可視化向け）
* 既知の注意（退化面が出る等）

を `method_card.json` として持つ（CIで存在チェック）。

## 13.2 ベンチ（bench）との接続を意識した出力

Mesh層が出す `Obs2D` 生成は observe層ですが、Mesh層としては

* interface面積
* exposed面積
* degenerate率
  を出せば、「手法を変えたとき何が変わったか」を切り分けしやすいです。

---

# 14. 実装上の推奨デフォルト（本目的に最適化）

最後に「このプロジェクト目的（サロゲート＋SEM同化）でのデフォルト」を明確にします。

## 14.1 推奨デフォルト

* **mesh生成入力**

  * ground truth（sim）由来：

    * **label→mesh（離散）**を1本持つ（基準メッシュ）
    * **tsdf→mesh（連続）**も持つ（SDFサロゲート出力整合用）
  * surrogate出力がTSDF：tsdf→mesh
* **extractor**

  * 大規模向け：FlyingEdges（isocontour） ([VTK][1])
  * multi-label向け：DiscreteFlyingEdges/SurfaceNets系（比較候補） ([VTK][1])
* **mode**

  * 同化・統計：interface_mesh（重複排除）
  * 材料単位の学習：material_shell（必要なら）
* **点群**

  * 固定N（例：65k）＋ `pair_code, is_exposed, normal` を付与

## 14.2 2Dデータ（Z=1）への扱い

* Mesh層で無理に三角形化せず、基本は observe層の `contour抽出` を使う
* どうしてもmeshモデルで統一したい場合は

  * “2D contour mesh（線分）” を別型として用意するか
  * 薄い押し出し（extrude）で3D化（ただし同化の意味が変わるので慎重に）

---

## 参考として、今回のMesh層設計で言及した主要技術（根拠）

* Flying Edges（VTK）は大規模向け最適化・並列化の isocontouring 実装で、退化三角形が出る可能性が明記されています。 ([VTK][1])
* Marching Cubes（1987）は等値面抽出の古典で広く使われます。 ([ACM Digital Library][4])
* VTKは segmented label mask のサーフェシングとして `vtkSurfaceNets3D` や `vtkDiscreteFlyingEdges3D` を検討するよう示しています。 ([VTK][1])
* multi-label SurfaceNetsは多ラベル非重複面生成・シャープ境界保持などを扱います。 ([PMC][2])
* multi-labelボリュームからの境界面復元（数理形態学）も、薄い構造等での頑健性や疎なメッシュ生成を主張しています。 ([サイエンスダイレクト][3])
* Dual Contouringは特徴保持や簡略化と関連する手法です。 ([ワシントン大学コンピュータサイエンス部][5])
* generalized winding numbers は壊れたmeshでも内外判定を頑健化する文脈で知られています。 ([ユーザーサーバー][7])
* multi-label Marching Cubes（M3C）系は最近の研究でも言及されています。 ([arXiv][6])

---



[1]: https://vtk.org/doc/nightly/html/classvtkFlyingEdges3D.html "VTK: vtkFlyingEdges3D Class Reference"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9623606/ "
            SurfaceNets for Multi-Label Segmentations with Preservation of Sharp Boundaries - PMC
        "
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0097849323002406 "Efficient boundary surface reconstruction from multi-label volumetric data with mathematical morphology - ScienceDirect"
[4]: https://dl.acm.org/doi/pdf/10.1145/37402.37422?utm_source=chatgpt.com "Marching cubes: A high resolution 3D surface construction ..."
[5]: https://www.cs.wustl.edu/~taoju/research/dualContour.pdf?utm_source=chatgpt.com "Dual Contouring of Hermite Data"
[6]: https://arxiv.org/html/2506.02938v1?utm_source=chatgpt.com "Material Interface Generation from UDFs for Non-Manifold ..."
[7]: https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.pdf?utm_source=chatgpt.com "Robust Inside-Outside Segmentation using Generalized ..."
