以下は **Surrogate層（ただし本プロジェクトにおける役割は「学習・推論」ではなく「SDFサロゲート用／Meshサロゲート用の“データ作成・パッケージ化”」）**としての、**設計図レベル**の詳細仕様です。
（全体設計はこれまで通り：Artifact駆動、SDF/Mesh両対応、観測はObserver層で `Obs2D` に統一、SEM同化の比較は `Obs2D vs Obs2D`。）

---

# 1. Surrogate層の定義（この層の責務を厳密に）

## 1.1 この層のゴール

Surrogate層（このプロジェクトでの定義）は、外部の学習基盤（PyTorch/JAX/社内基盤など）がそのまま使える形で

* **入力（プロセス条件・レシピ・初期状態など）**
* **教師（SDF表現 or Mesh/PointCloud表現）**
* **評価用ターゲット（Observerで落とした 2D観測：mask/2D TSDF/輪郭）**
* **分割（train/val/test）**
* **統計（正規化・分布・QA）**
* **再現性（定義のハッシュ・バージョン・依存）**

を揃えた **データパッケージ（DatasetPackage）** を作ることです。

> 学習・推論の実行（Trainer/InferenceRunner）は本プロジェクトの責務外。
> ここは **「誰が学習しても同じデータが供給される」**ことに徹します。

---

## 1.2 何を入力として扱うか（Surrogateの入口）

この層は「生のvti」から直接作りません。**必ず上流Artifactを入力**にします。

* `LabelArtifact`（vti正規化済み、multi-material ≤5）
* `SDFArtifact / TSDFVolume`（材料別TSDF、d_boundary、pair_code…）
* `MeshArtifact / MeshGeom + PointCloud`（界面属性、露出属性付き）
* `Obs2DArtifact(sim_gt)`（Observer定義に基づく2D観測：topdown/slice 等）
* `Recipe/ParamArtifact`（プロセス条件：キー・単位・範囲が確定したもの）

---

## 1.3 この層がやらないこと（非責務）

* モデル定義、学習ループ、損失、最適化（外部）
* 推論サーバ、推論バッチ処理（外部）
* “新しいSDF手法の計算”そのもの（SDF層の責務）
  ※ただし「どのSDF手法のArtifactを教師として採用するか」の**選択**と**評価用データ生成**はここで行う

---

# 2. コード構成（Surrogate“データ作成”としての読みやすい分割）

既存の `datasets/`（manifest/splits/loader）を活かしつつ、Surrogate層は「サロゲート用データの定義・構築・出力」に集中させます。

推奨構成：

```text
wafergeo/surrogate/
  README.md                    # 「学習/推論は外部」方針の明文化
  schema.py                    # DatasetPackage/Record/Spec 型定義
  profiles/
    sdf_surrogate_v*.yaml      # SDFデータ作成プロファイル
    mesh_surrogate_v*.yaml     # Meshデータ作成プロファイル
    implicit_surrogate_v*.yaml # (任意) 点SDF/占有などquery系プロファイル
  builder/
    base.py                    # Builder I/F + registry
    sdf_builder.py             # SDF教師（TSDF stack 等）のパッケージ化
    mesh_builder.py            # Mesh/PointCloud教師のパッケージ化
    obs_builder.py             # sim側Obs2Dターゲット生成の束ね（Observer層呼び出し）
    pair_builder.py            # (任意) sim-SEM紐付けセット生成
  transforms/
    crop_pad.py                # ROI、固定shape化
    resample_grid.py           # spacing変更・grid統一
    param_encode.py            # ParamSpecに基づくvector化
  splits.py                    # group split、リーク防止（lot/wafer/seed）
  stats.py                     # 分布・正規化統計（paramsなど）
  qa.py                        # サンプルQA + データセットQA
  export/
    zarr_export.py             # 固定shape配列に最適
    npz_export.py              # 軽量配布用
    parquet_export.py          # tabular（params/QA/リンク）
    mesh_export.py             # (任意) ply/gltf等（可視化向け）
  cli.py                       # build/validate/export
```

> 重要：Surrogate層は **I/O形式（export）とデータ定義（schema/profile）を強く分離**します。
> これにより「内部のArtifact構造が変わってもexport互換を保ちやすい」＝運用が壊れにくい。

---

# 3. データモデル（DatasetPackageの中身を固定する）

## 3.1 DatasetPackage の2モード（運用で必須）

学習環境とデータ配置が分かれるため、2モードを設計として持ちます。

### A) Linked dataset（参照型：軽い・速い・運用向き）

* `manifest` に **Artifact ID** を保存
* 実データは `ArtifactStore` から読み出す
* 長所：重複保存しない、更新が容易
* 短所：学習環境がArtifactStoreにアクセスできる必要

### B) Packed dataset（梱包型：ポータブル・共有向き）

* 主要テンソルを `data.zarr` 等に **物理的に書き出す**
* 長所：学習クラスタへ配布しやすい
* 短所：容量が増える、更新が重い

**同じmanifest/split/specで両方作れる**設計にします（exportを差し替えるだけ）。

---

## 3.2 SampleRecord（1サンプルの最小契約）

各サンプルは「入力パラメータ」と「教師（SDF/Mesh）」と「評価用2Dターゲット」をリンクします。

```python
@dataclass(frozen=True)
class SampleRecord:
    sample_id: str                    # stable
    group_id: str                     # leakage防止（lot/wafer/pattern等）
    recipe_params: dict               # 物理量（単位付き）を保持
    param_vector: list[float]         # ParamSpecに基づく固定順ベクトル（任意）
    # pointers (linked) or paths (packed)
    label_artifact_id: str | None
    tsdf_artifact_id: str | None
    mesh_artifact_id: str | None
    obs2d_sim_ids: dict[str, str]     # observer_name -> Obs2DArtifact id
    qa: dict                          # sample QA summary
    meta: dict                        # grid/materials/profile hashes
```

---

## 3.3 DatasetManifest（再現性の核）

```json
{
  "schema_version": "surrogate_dataset/v3",
  "dataset_id": "sdf_surrogate_v3_20260301_xxx",
  "profile_id": "sdf_surrogate_v3",
  "created_at": "2026-03-01T...",
  "generator_version": "...",
  "git_commit": "...",
  "materials": { "ids":[0,1,2,3,4], "names":[...], "void_id":0 },
  "grid3d": { "dim":3, "spacing_zyx":[...], "origin_zyx":[...], "axis_order":"ZYX", "units":"nm" },
  "observers": ["topdown_cdsem_v2", "slice_xsem_v1"],
  "metric_reference": "assim_objective_v2 (optional)",
  "param_spec": { "axes":[...], "vector_order":[...] },
  "storage_mode": "linked|packed",
  "samples": [ ... ],
  "splits": { "train":[...], "val":[...], "test":[...] },
  "stats": { "param_mean":[...], "param_std":[...], "notes":{...} }
}
```

> **ここに ObserverSpec / SDFProfile / MeshProfile / ParamSpec の hash を必ず刻む**ことで、
> 「同じdataset_idなら比較定義が同一」になります。

---

# 4. SDFサロゲート用データ作成（SDFBuilderの詳細）

## 4.1 何を“教師”として提供するか（SDF系の想定出力）

目的により「教師」を複数タイプに分けます。プロファイルで切替可能にします。

### タイプS1：材料別TSDFスタック（標準・最も汎用）

* `tsdf`: `(M,Z,Y,X)` float16, 値域 `[-1,1]`
* optional：

  * `d_boundary`: `(Z,Y,X)` float16, `[0,1]`
  * `pair_code`: `(Z,Y,X)` uint8（界面ペア）
  * `present_mask`: `(M,)` bool（材料が存在するか）

> M≤5ならチャネル数は現実的。サロゲートにとっても扱いやすい標準教師。

### タイプS2：二値TSDF（solid vs void）

* 工程が “外形のみ” に支配される場合や、まず粗いサロゲートで試す時に有効
* multi-materialを捨てるので用途限定（評価はObserver/SEMで担保）

### タイプS3：点SDF（implicit/query型：巨大3D向け）

* dense TSDFが重い場合に効く“データ供給形態”
* `N`点の `(xyz, sdf, material_label or tsdf_stack)` を生成
* 例：

  * boundary近傍重点サンプル
  * exposed近傍重点サンプル
  * interfaceペアごとのバランス

> 学習は外部ですが、この形式は「新しいモデル（implicit NN等）」にも対応しやすいので、パッケージとして価値が高いです。

---

## 4.2 SDFデータ作成パイプライン（固定フロー）

1. **入力の確定**

   * `LabelArtifact` or `TSDFArtifact` を入力にする
   * 既にTSDFがあるなら再計算しない（Artifact駆動）

2. **grid統一（必要なら）**

   * spacing/origin/shapeを dataset profile の固定gridへ
   * crop/pad/resample を `transforms/` で明示的に実施
   * 変更したら必ずmetaに記録

3. **出力テンソル生成**

   * TSDF（Mチャネル）
   * 境界特徴（d_boundary/pair_code）
   * （任意）工程特化の2D特徴：`top_height_map`, `top_material_id_2d` など
     ※ただし “教師に入れるか、補助評価に使うか” はprofileで分ける

4. **sim側Obs2Dターゲット生成（強く推奨）**

   * `obs2d_sim_ids[observer_name]` を作る（Observer層を呼ぶ）
   * これにより

     * SDF学習でも
     * mesh学習でも
       **同じ2D観測定義で評価可能**になる

5. **QA**

   * TSDF値域、NaN/Inf、材料存在、pair_code妥当性
   * `Obs2D`の閉曲線性・面積・TSDF値域
   * sample QA summary を record に付与

6. **DatasetPackageへ書き込み（linked/paked）**

---

## 4.3 multi-material ≤5 を最大限活かす“教師チャンネル設計”

SDF教師の設計で重要なのは「材料差」を学習可能にすることです。推奨は：

* **チャネルは必ず固定順（materials.yamlに一致）**
* 欠損材料はチャネルを消さず、**全+1（外側）**で埋める（present_maskで区別）
* `pair_code` は Mが小さいほど強い特徴になる

  * 例：界面 `(resist, oxide)` がどこに多いか、学習が分かりやすくなる

> “材料は最大5種”という条件は、**「多チャネルでも破綻しない」**大きな利点です。

---

## 4.4 SDFデータセットの典型ストレージ（Packed, Zarr）

固定shape（Z,Y,X）が揃う場合：

* `data.zarr/tsdf` : `(N,M,Z,Y,X)` float16
* `data.zarr/d_boundary` : `(N,Z,Y,X)` float16（任意）
* `data.zarr/pair_code` : `(N,Z,Y,X)` uint8（任意）
* `data.zarr/obs2d/{observer}/tsdf2d` : `(N,Y2,X2)` float16
* `data.zarr/obs2d/{observer}/mask2d` : `(N,Y2,X2)` uint8
* `meta/contours/{observer}.jsonl` : 可変長（必要なら）

巨大3Dで固定shapeが厳しい場合は：

* パッチ化（後述）か
* linked dataset（artifact参照）へ

---

## 4.5 パッチ化（巨大3Dに対する運用設計）

学習は外部でも、データはパッチで渡したいことが多いので、Surrogate層で「パッチインデックス」を生成できます。

* `patch_size_zyx`, `stride_zyx`
* `sampling_policy`：

  * uniform
  * boundary-focused（|phi|<band を含むパッチ優先）
  * material-balanced（材料割合が偏らないようサンプル）

出力：

* `patch_index.parquet`（sample_id, z0,y0,x0, weight, split…）
* `data.zarr` は元ボリュームを持つか、パッチを直接持つかを選択（profile）

---

# 5. Meshサロゲート用データ作成（MeshBuilderの詳細）

## 5.1 “学習可能な形”として何を提供するか

メッシュは可変長で扱いにくいので、このプロジェクトでは **PointCloud（固定N）を第一級教師**にするのが運用上強いです。
（メッシュ自体も保存可能だが、学習の主経路は点群を想定。）

### タイプM1：固定N点群（推奨：学習実装が簡単）

* `points_nm`: `(N,3)` float32
* `normals`: `(N,3)` float32
* `pair_code`: `(N,)` uint8（界面種別）
* `is_exposed`: `(N,)` uint8/bool（SEM topdownに効く）
* optional：

  * `material_in/out`（点が属する界面の両側）
  * `curvature` 等（必要なら）

### タイプM2：MeshGeom（可視化・解析・一部学習用）

* vertices/faces + face attributes
* variable lengthなので packed方式では「サンプルごとファイル」または「CSR方式の連結保存」を用意

### タイプM3：点＋SDF（implicitにも使える）

* mesh表現でも “点に距離値” を付けたい場合
  → SDF層のTSDFからサンプルして `(xyz, sdf)` を持つ派生教師を作れる

---

## 5.2 Meshデータ作成パイプライン（固定フロー）

1. **入力の確定**

   * `MeshArtifact` があるならそれを利用（Mesh層で生成）
   * なければ `TSDFArtifact` から Mesh層を呼んで生成（ただしこの層は“生成オーケストレーション”のみ）

2. **meshモード選択**

   * `interface_mesh`（界面重複なし：統計やexposed判定が安定）
   * `material_shell`（材料別殻が欲しい場合）

3. **点群サンプリング（固定N、再現性）**

   * stratified：

     * interfaceペアごとにバランス
     * exposed面を優遇（SEM topdownが主なら重要）
   * `seed` をprofileで固定し、同じ入力なら同じ点群にする

4. **共通2Dターゲットの生成**

   * `Obs2D(sim_gt)` を **TSDF/Label由来で作る**のを推奨
     （mesh discretizationの差で観測が揺れるのを避ける）

5. **QA**

   * 点群のbbox、NaN、pair_code範囲
   * exposed点の割合が極端でないか（topdown運用では重要）
   * sim側Obs2Dの健全性も合わせて確認

6. **DatasetPackageへ書き込み**

---

## 5.3 Mesh（可変長）を“パッケージとして運用可能”にする保存設計

### A) 点群中心（推奨）

* `data.zarr/pc_points` : `(N_samples, N_points, 3)`
* `data.zarr/pc_normals` : 同様
* `data.zarr/pc_pair_code` : `(N_samples, N_points)`
* `data.zarr/pc_is_exposed` : `(N_samples, N_points)`

### B) メッシュも保存したい場合（オプション）

* `meshes/{sample_id}.npz`（vertices/faces/attrs）
* `mesh_index.parquet`（sample_id -> path）
* これなら学習は点群で行いつつ、解析・再メッシュなどに使える

---

# 6. 2D観測ターゲット（Obs2D）をSurrogateデータに含める理由と設計

あなたの要件では **SEM同化の最終比較は2D輪郭（または2D TSDF）**です。
したがって Surrogateデータ作成でも、**sim側の2D観測（Obs2D_gt）を同じObserverSpecで生成**しておくのが極めて重要です。

## 6.1 何を保存するか（学習用／評価用の両立）

* 学習に必要：

  * `tsdf2d`（損失を作りやすい）
  * `mask2d`（補助）
* 評価・レポートに必要：

  * `contours`（可変長なので jsonl などで）

プロファイルで

* `save_contours: true/false`
  を切り替え可能にします。

## 6.2 複数Observerに対応

* topdown（CD-SEM）
* slice（X-SEM、複数断面）
* ROI違い

を `obs2d_sim_ids[observer_name]` として持てるようにし、**同じデータセットで多角評価**できるようにします。

---

# 7. Split設計（リーク防止が最重要：半導体データ特有）

サロゲート学習で一番やってはいけないのが、**リークしたsplit**です（同一wafer/lotがtrainとtestに混ざる等）。

## 7.1 group split を標準にする

`group_id` を必須フィールドとし、以下のいずれか（または組合せ）で作ります：

* `lot_id`
* `wafer_id`
* `mask/pattern_id`
* `process_recipe_id`
* `simulation_seed_id`

推奨：**wafer単位**または**lot単位**で完全分離。

## 7.2 分割の保存

* `splits.json` に sample_id のリスト
* split生成の `seed` と `rules` を meta に保存
  → 同じdataset_idなら同じsplitを再現できる

---

# 8. QA設計（Surrogateデータは“静かに壊れる”ので必須）

## 8.1 サンプルQA（最低限）

* TSDF値域（[-1,1]）、NaN/Infなし
* pair_codeが許容範囲内（0..K, 255など）
* present_mask と実データの整合
* 2D観測が空でない（mask面積）
* grid・units の整合（nm）

## 8.2 データセットQA（分布と偏り）

* parameter分布（min/max、外れ値）
* material存在率（欠損材料が多いと学習が偏る）
* interfaceペア頻度（pair_codeヒストグラム）
* 2D観測の面積・CDの統計（評価が極端なサンプルを検出）

QAは `qa_report.json` と `qa_table.parquet` で保存し、運用で追跡します。

---

# 9. 拡張・追加が容易な設計（手法が増える前提のプラグイン境界）

Surrogate層は「新しいSDF手法」「新しいMesh表現」「新しいObserver」が増える前提なので、以下をプラグイン化します。

## 9.1 Builderプラグイン

* `SDFDatasetBuilder`
* `MeshDatasetBuilder`
* `ImplicitDatasetBuilder`（点SDF/占有等）

共通I/F（概念）：

```python
class DatasetBuilder(Protocol):
    name: str
    version: str
    def build(self, input_manifest, profile, store, out_dir) -> DatasetManifest: ...
```

## 9.2 Exporterプラグイン

* `ZarrExporter`（固定shapeテンソル）
* `ParquetExporter`（tabular）
* `NPZExporter`（軽量）
* `MeshFileExporter`（ply/gltf等）

これにより「内部はlinkedで運用、外部学習用にpackedへ出力」が簡単になります。

## 9.3 Feature/Targetの追加

* SDF教師に `height_map` を足す
* 点群に `curvature` を足す
* 2D観測ターゲットに新しいobserverを足す

→ profileに項目を足し、対応するBuilderの小変更で済むようにする（設計での目的）。

---

# 10. CLI / ワークフロー（Surrogateデータ作成をDAGの1ノードにする）

推奨CLI：

1. **`surrogate build --profile sdf_surrogate_v3.yaml --in sim_runs.parquet --out dataset_sdf_v3/`**

   * linked/packedはprofileで指定
2. **`surrogate validate dataset_sdf_v3/`**

   * QA結果と要約出力
3. **`surrogate export dataset_sdf_v3/ --format zarr`**

   * 外部学習環境へ渡す
4. **`surrogate build --profile mesh_surrogate_v2.yaml ...`**

   * mesh/pointcloud版も同様

> ここまでが本プロジェクトのSurrogate層の責務。
> 学習は「このdatasetを外部が読む」だけで成立します。

---

# 11. まとめ（Surrogate層＝データ作成の設計の芯）

* Surrogate層は **学習/推論をしない**。代わりに
  **SDF用／Mesh用の教師データ＋評価用2D観測（Obs2D）を、再現可能なDatasetPackageとして生成する**。
* `linked` と `packed` の2モードを持ち、運用と学習環境の分離に耐える。
* multi-material ≤5 を活かし、SDF教師は **材料別TSDFスタック + 境界特徴（pair_code等）**を標準化する。
* Mesh教師は **固定N点群（属性付き）**を第一級にし、可変長meshはオプションとして保存する。
* 2D観測ターゲット（Obs2D_gt）を必ず同梱できる設計にし、SEM同化の定義と学習評価を一致させる。
* split/QA/manifest/spec hash を必ず保存し、「同じdataset_idなら同じ比較定義」が成立するようにする。

---

もし次に進めるなら、Surrogate層を“設計図→実装仕様”に落とすために、以下を具体のYAMLとスキーマで提示できます（学習コードは含めません）：

* `sdf_surrogate_v3.yaml` / `mesh_surrogate_v2.yaml` のテンプレ（fixed grid、出力項目、linked/packed、QA、splitルール込み）
* `DatasetManifest` のJSON Schema（厳密版）
* Zarrのグループ構造（キー一覧、dtype、chunk推奨）
* 典型的な `sim_runs.parquet`（入力manifest）の列定義（sample_id, group_id, recipe_params, artifact_ids…）
