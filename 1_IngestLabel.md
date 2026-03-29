以下は **Ingest / Label正規化層**（`.vti` から multi-material（≤5）ラベルを読み込み、揺れを吸収して **学習・同化に耐える LabelVolume（LabelArtifact）** を作る層）の **設計図レベルの詳細仕様**です。
全体設計（レイヤ構成・Artifact駆動・Observerで2Dへ統一）は前回の通りで、ここでは **この層だけをアーキテクト観点で徹底的に具体化**します。

---

# 1. Ingest / Label正規化層の責務（この層が「どこまで」やるか）

## 1.1 この層の役割（アーキテクト視点）

この層の責務は **「入力の揺れ・不整合・危険を入口で全て押さえる」**ことです。
後段（SDF/mesh/observer/surrogate/assimilation）は、この層の出力を **“信じて良い前提”**にします。

具体的には：

1. **`.vti` の読み込み（I/O）とメタ抽出**

   * spacing/origin/extent、cell/point dataの区別、配列名の揺れ
2. **内部標準（canonical）への正規化**

   * ndarray軸順、dtype、材料IDの集合、void埋め、未知ラベル処理
3. **幾何意味の統一**

   * cell中心/格子点の解釈、originの補正、units統一
4. **品質保証（QA）**

   * 体積・連結成分・飛び地・異常率の検出（ラベルQAレポート）
5. **Artifact出力・再現性**

   * `LabelArtifact` として保存、再計算を避けるキャッシュキーを確立

> **重要な方針**：この層は「形状を改善する（ノイズ除去など）」は基本しません。
> それは物理的真値を変えうるため、別レイヤ（optional postprocess）に分離します。
> ただし **“入力として壊れている”**場合（未割当・多重割当等）は正規化ルールで修正し、必ずログ/QAに残します。

---

# 2. 入出力仕様（契約）— ここがズレると後段が崩れる

## 2.1 入力（入力形式の揺れを許容）

* `.vti`（VTK ImageData）
* ラベルの格納形態の揺れ：

  * **(A)** 1本のスカラー配列：`material_id[z,y,x]`（推奨・多い）
  * **(B)** 材料ごとの複数配列：`mask_resist`, `mask_oxide` …（稀だがあり得る）
* ラベルの格納位置：

  * CellData（voxel/セルに付与）
  * PointData（格子点に付与） ← “voxelデータ”としては要注意（後述の変換を実装）

## 2.2 出力（この層が保証する “内部正規形”）

* **LabelVolume**（内部はこれだけを使う）

  * `grid.axis_order = "ZYX"` 固定
  * `material_id.shape = (Z,Y,X)` 固定（Z=1でもOK：2D相当）
  * `material_id` は **MaterialSpec.ids の集合に必ず収まる**
  * `grid.sample_location` は **必ず "cell_center"** を原則にする

    * PointData入力でも最終的に cellに落とす（voxelとして意味が揃う）

* **LabelQA**（必須生成）

  * 材料別ボクセル数/体積
  * 連結成分数
  * 未割当率、未知ID率、（mask入力なら）多重割当率
  * 基本的な隣接（界面）統計（任意だが推奨）

* **LabelArtifact**（ArtifactStoreへ保存）

  * `material_id`、`GridSpec`、`MaterialSpec`、`Meta`、`LabelQA`

---

# 3. コード構成（この層のファイル責務と読みやすさの設計）

前回のリポジトリ構成のうち、Ingest/Label正規化に関係する部分だけを詳細化します。

```
wafergeo/
  io/
    vti_reader.py        # vti -> RawVtiImage（副作用I/O）
    artifact_store.py    # Artifact入出力（副作用I/O）
  label/
    materials.py         # MaterialSpec 読み込み + ID/Index変換
    normalize.py         # Raw -> LabelVolume（純粋関数中心）
    qa.py                # Label QA（純粋関数中心）
    errors.py            # 入力異常の例外型
```

## 3.1 重要な設計原則（レビューしやすさ）

* **Functional core, imperative shell**

  * `io/*` が副作用（ファイル読み書き）
  * `label/normalize.py` と `label/qa.py` は **ほぼ純粋関数**（入力→出力が決まる）
* 「入口の揺れ吸収」を **normalize.py に集約**し、後段に条件分岐を持ち込まない
* 例外型を細かくし、上位（CLI）で “ユーザに何を直せばよいか” を明確に出す

---

# 4. 主要データ型（Raw → Canonical への変換のための中間表現）

## 4.1 RawVtiImage（I/O層から返す中間型）

`vti_reader.py` は、VTK依存をここに閉じ込めるために Raw 型を返します。

```python
# wafergeo/io/vti_reader.py
from dataclasses import dataclass
import numpy as np
from typing import Literal, Dict, Optional, Tuple

ArrayLocation = Literal["cell", "point"]

@dataclass(frozen=True)
class RawVtiImage:
    spacing_xyz: Tuple[float, float, float]   # VTKの (sx, sy, sz)
    origin_xyz: Tuple[float, float, float]    # VTKの origin (ox, oy, oz)
    dims_xyz: Tuple[int, int, int]            # VTKの dims (nx, ny, nz)
    arrays: Dict[str, np.ndarray]             # array_name -> raw ndarray
    array_location: Dict[str, ArrayLocation]  # array_name -> "cell"/"point"
    vtk_meta: Dict[str, str]                  # optional
```

> **Rawは“VTK座標・VTK並び”のまま**で良いです。
> Canonicalへの変換は normalize.py が責務です。

---

# 5. 設定（Config）— 揺れを吸収するための明示パラメータ

この層は「入力が常に同じ」前提にしないので、設定で揺れを吸収できるようにします。

## 5.1 LabelNormalizeConfig（設計）

```python
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional

UnknownLabelPolicy = Literal["error", "map_to_void"]
PointToCellPolicy = Literal["majority", "nearest", "error"]
MaskMergePolicy = Literal["priority", "error_on_conflict"]

@dataclass(frozen=True)
class LabelNormalizeConfig:
    label_array_candidates: List[str]         # ["material_id", "MaterialId", ...]
    prefer_cell_data: bool = True
    unknown_label_policy: UnknownLabelPolicy = "error"
    unknown_to_void_id: int = 0               # unknown_label_policy == map_to_void のとき
    remap_ids: Dict[int, int] = None          # raw_id -> canonical_id（工具差吸収）
    point_to_cell_policy: PointToCellPolicy = "majority"
    mask_merge_policy: MaskMergePolicy = "priority"
    force_units: Optional[str] = "nm"         # vtiのunitsが不明なら強制
    spacing_override_xyz: Optional[tuple] = None
    origin_override_xyz: Optional[tuple] = None
    enforce_cell_center: bool = True          # 原則true：voxelとして統一
```

## 5.2 materials.yaml（MaterialSpec）

* 材料種≤5なので、IDは安定運用しやすい（例：0..4）
* ここに **priority** を持たせると、mask入力で多重割当があっても deterministic に解決できる

---

# 6. 正規化の処理内容（ステップ別：できるだけ具体的に）

以下が `label/normalize.py` の **標準パイプライン**です。
“この順番でやる”ことを推奨します（理由：失敗原因が分かりやすい）。

---

## Step 0：対象ラベル配列の選択（配列名の揺れ吸収）

**入力**：RawVtiImage.arrays
**出力**：選択された `label_array_name`

### 仕様

* `config.label_array_candidates` を上から順に探す
* 見つからない場合：

  * RawVtiImage内の配列一覧を添えて `MissingLabelArrayError` を投げる
* Cell/Point の両方に同名がある場合：

  * `prefer_cell_data` に従い、cell優先で選ぶ

### 実装ポイント（レビューしやすい）

* ここは副作用なし（純粋関数）
* エラー時に「候補名」「実際の配列名」「cell/point」を明確に出す

---

## Step 1：VTKメタ（spacing/origin/dims）の確定と上書き（units含む）

**目的**：後工程の座標系が崩れないように最初に固定する

### 仕様

* `spacing_xyz` と `origin_xyz` を読み込み
* `spacing_override_xyz` / `origin_override_xyz` があれば上書き
* `units` は `force_units` をデフォルト採用（vtiに確実なunitsが入っていないことが多い前提）

### 注意点（セル中心 origin の扱い）

VTKの origin は **格子点（point）**の位置を表すのが通常です。
しかし、ラベルが **CellData（voxel）** の場合、配列のサンプル位置はセル中心です。

**よって canonical GridSpec の origin は「サンプル点の物理座標」に統一**します：

* PointDataの場合：
  `origin_sample_xyz = origin_xyz`
* CellDataの場合：
  `origin_sample_xyz = origin_xyz + 0.5 * spacing_xyz`

この補正をしないと、後で SEM との座標合わせや observer のROIがズレます。

---

## Step 2：ndarrayの軸順・shape 正規化（XYZ → ZYX）

**目的**：下流が一切迷わず `material_id[z,y,x]` を扱えるようにする

### VTIの並びの基本

VTK ImageDataは概念として (x,y,z) を持ち、NumPyへ変換したときのshapeは実装依存になりがちです。
そこで **vti_reader では “rawのまま”**持ち、normalizeで必ず揃えます。

### 仕様

* `material_id` は必ず shape `(Z,Y,X)` に変換
* dimsは RawVtiImage.dims_xyz = (nx, ny, nz) を参照し、

  * point array: expected size = nx*ny*nz
  * cell array: expected size = (nx-1)*(ny-1)*(nz-1) があり得る
    （VTKのextentの取り方で差が出るため、ここは reader で “実際の配列shape”も返してよい）

### 実装のコツ（レビューしやすくする）

* `to_zyx(raw_array, dims_xyz, location)` を1関数に閉じ込める
* 変換の最後に assert（shapeチェック）し、失敗時は詳細エラー

---

## Step 3：PointData → CellData 変換（原則必須）

**目的**：voxelとして意味が一致しない入力を、voxel（セル）へ揃える

### なぜ原則 cell_center に統一するのか

* voxelシミュレーションの “材料ID” は通常セル（voxel）に定義される
* SDF生成、メッシュ化、界面推定、厚み計算は **セル中心**を前提にした方が一貫する

### 仕様（point_to_cell_policy）

PointData入力のとき、セル値に落とします：

* `"majority"`（推奨）

  * 1セルを構成する8点（2Dなら4点）のラベルの多数決
  * 引き分けは `MaterialSpec.priority` で決める（deterministic）
* `"nearest"`

  * セル中心に最も近い点の値を使う（高速）
* `"error"`

  * 変換せず例外（入力側に修正を要求）

> **注意**：PointData→CellData変換は “情報を失う可能性がある” ため、
> QAレポートに `converted_from_point=true` を必ず記録し、運用側で追えるようにします。

---

## Step 4：IDリマップ（工具差・工程差・データ生成差の吸収）

**目的**：異なるデータ生成系で material_id の値がズレても統一できる

### 仕様

* `config.remap_ids`（raw_id→canonical_id）があれば適用
* 例えば toolA: oxide=2, toolB: oxide=7 のようなズレを吸収

**実装ポイント**

* 変換前後でユニークID集合をログ化
* remap漏れ（存在しないraw_id）を検知しやすくする

---

## Step 5：未知ラベル・欠損ラベルの処理（unknown_label_policy）

**目的**：後段が落ちるのを防ぎ、運用方針に合わせて挙動を決める

### 仕様

* `MaterialSpec.ids` に存在しない値が混じっていたら：

  * `"error"`：`UnknownMaterialIdError`（推奨：学習用のGT生成時）
  * `"map_to_void"`：強制的に void_id へ（推奨：現場運用のロバスト性優先時）

**必須**

* いずれにしても `unknown_id_count`, `unknown_id_values` を QA に記録

---

## Step 6：dtype 正規化（メモリとI/O効率の安定化）

**仕様**

* 材料≤5なら、内部表現は `uint8` が十分

  * ただし原IDを保持したい場合は `uint16`
* 設計としては

  * `LabelVolume.material_id` は **canonical id（0..4）** を保存（uint8）
  * もし raw id 追跡が必要なら `meta.extra["raw_id_mapping"]` に保存（配列を二重に持たない）

---

## Step 7：最終的な GridSpec / Meta / MaterialSpec の組み立て

**GridSpec（canonical）**

* `dim=3`
* `axis_order="ZYX"`
* `spacing=(sz,sy,sx)` へ変換（Rawがxyzなら並べ替え）
* `origin=(oz,oy,ox)` へ変換（セル中心補正済み）
* `sample_location="cell_center"` に統一（原則）
* `units="nm"`（強制or設定）

**Meta**

* schema_version/profile_id/config_hash/generator_version/git_commit/input_hash/created_at
* `extra` に

  * `source_path`
  * `label_array_name`
  * `label_was_point_data`
  * `remap_applied`
  * `unknown_policy`
  * `is_2d`（Z==1など）
    を記録

---

# 7. Label QA（label/qa.py）：この層で出すべき“最低限の診断”

この層のQAは「後工程のSDFが変だから…」となる前に、入口で原因を見つける目的です。
実装は **純粋関数**として分離し、テストしやすくします。

## 7.1 LabelQA の設計（例）

```python
@dataclass(frozen=True)
class LabelQA:
    material_counts: dict[int, int]     # id -> voxel count
    material_volume_nm3: dict[int, float]
    void_fraction: float
    unknown_count: int
    connected_components: dict[int, int]  # id -> num CC (optional but useful)
    largest_component_ratio: dict[int, float]
    adjacency_matrix: list[list[int]]    # MxM counts of neighboring faces (optional)
    notes: list[str]
```

## 7.2 QAとして最低限入れるべき項目

1. **材料別ボクセル数**（counts）
2. **材料別体積**（counts × spacing積）
3. **void割合**（ROI切り出しミスやデータ破損検知に効く）
4. **連結成分数**（飛び地・ゴミ・分断検知）
5. **最大成分比**（「ほぼゴミ」や「分散しすぎ」を検知）
6. （推奨）**隣接（界面）統計**

   * 6近傍（面共有）で `i-j` の接触回数を数える
   * multi-material ≤5 なら「想定外界面」がすぐ見える（例：resist-metal接触が本来無いなど）

## 7.3 QAの実装ポイント

* CCは `scipy.ndimage.label` を使う想定（依存はcore/extraで調整）
* adjacencyは「隣接面の差分カウント」で軽量に計算可能

  * x方向：`material_id[:,:,1:] != material_id[:,:,:-1]` で界面抽出
  * 同様に y/z
  * i-j を行列に加算

> **運用上の強いメリット**：
> 「SEM同化が合わない」原因が **観測の問題**なのか、**入力ラベルの破綻**なのかを切り分けできます。

---

# 8. Artifact化（LabelArtifact）— 入口成果物として保存する内容

## 8.1 LabelArtifactの中身（推奨）

* `material_id`（Z,Y,X）uint8
* `grid.json`（GridSpec）
* `materials.json`（MaterialSpec）
* `meta.json`
* `qa.json`

## 8.2 書き出し規約

* 形式は Zarr/HDF5 どちらでも良いが、

  * chunk + 圧縮（zstd等）
  * メタを必ず属性or別JSONで保存
* artifact_id は input_hash + config_hash + generator_version で決まる
  → ingestは重いのでキャッシュが効く

---

# 9. 例外設計（errors.py）：レビューワーが把握しやすい“失敗の分類”

### 推奨例外型

* `MissingLabelArrayError`：候補配列が見つからない
* `InvalidArrayShapeError`：配列shapeが想定と一致しない（dimsと矛盾）
* `UnknownMaterialIdError`：MaterialSpecに存在しないIDが出現
* `PointToCellConversionError`：PointData→CellDataが許容できない/不可能
* `InvalidGridMetaError`：spacing/originが不正（0やNaNなど）

### 例外メッセージに必ず含める情報

* file path
* candidate arrays / available arrays
* selected array name and location
* expected shape / actual shape
* unknown ids list（上位数個＋count）

> エラーの質がそのまま運用コストに直結します。

---

# 10. CLIワークフロー（この層の実行単位を固定する）

例：
`wafergeo ingest-vti --vti path/to/a.vti --materials configs/materials.yaml --profile surrogate_sdf_profile_v2.yaml`

このコマンドは内部で

1. vti_readerで Raw を作る
2. normalizeで LabelVolume + LabelQA を作る
3. ArtifactStoreへ LabelArtifact を保存する（キャッシュがあればスキップ）

までを **1つの“原子的”ステップ**として扱います。

---

# 11. テスト設計（この層は“最優先で固める”べき）

この層が壊れると以降すべてが破綻します。
よって **ユニットテスト＋統合テスト**を厚くします。

## 11.1 ユニットテスト（純粋関数中心）

* `select_label_array`（候補選択）
* `to_zyx`（軸変換・shape検証）
* `point_to_cell_majority`（多数決＋priority tie-break）
* `remap_ids`（辞書適用）
* `validate_known_ids`（unknown policy）
* `compute_label_qa`（counts/cc/adjacency）

## 11.2 統合テスト（小さなvti）

* 小さい3D（例：10×10×10）で materials=3
* Z=1 の “2D相当” vti
* PointDataラベルのvti（変換結果の一致）
* unknown id混入 vti（policyで挙動差を確認）

---

# 12. “第三者が追加しやすい”拡張点（この層で増えがちな要求）

## 12.1 vti以外の入力を増やしたい（例：nrrd、npz）

* `io/*_reader.py` を追加して `RawVtiImage` 相当の Raw 型を返す
* normalize.py は Raw 型が同じならそのまま流用できる

## 12.2 ラベル正規化の追加ルールを入れたい

例：特定材料を常にvoidに落とす、微小島を除く等
→ それは “正規化”というより “後処理” になり得ます。

推奨：

* `label/postprocess.py` を別に作り、profileでオンにできるようにする
* その場合も **必ず差分とパラメータをArtifactに保存**する

---

# 13. まとめ（Ingest/Label正規化層の設計要点）

* この層は **入力揺れの吸収＋canonical化＋QA＋Artifact化**が責務
* 出力の契約は

  * `(Z,Y,X)`
  * `axis_order="ZYX"`
  * `sample_location="cell_center"`（原則）
  * `material_id ∈ MaterialSpec.ids`
    を必ず守る
* VTK origin の意味（point基準）と cell_center のズレは **ここで補正**する
* QAは「材料数≤5」を活かして **界面統計まで**出すと運用が強くなる
* 正規化処理は **純粋関数中心**にして、レビュー・テスト・拡張を容易にする


