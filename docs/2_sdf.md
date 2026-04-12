# SDF

SDF は Signed Distance Field の略で、境界からの距離を持つ特徴量です。
このパッケージでは、特徴量化と比較評価の両方で SDF を使います。

## 公開用途

| 場所 | 内容 |
|---|---|
| `transform` の `sdf` | 2D view の軽量 SDF を `features/simulation_sdf.npz` に出力 |
| `transform` の `sdf_raw` | non-void union の raw 3D signed distance を `features/sdf_raw.npz` に出力 |
| `transform` の `tsdf_views` | `sdf_raw` から固定 clip 幅の 3D TSDF views を `features/tsdf_views.npz` に出力 |
| `transform` の `udf` | non-void boundary への 3D unsigned distance を `features/udf.npz` に出力 |
| `transform` の `sdf3d` | material ごとの 3D TSDF stack を `features/sdf.npz` に出力 |
| `compare` の `sdf` | simulation と target の 2D SDF 差分 |
| `compare` の `sdf_material` | material ごとの SDF 差分 |
| `compare` の `sdf_band` | boundary 近傍 10 nm に絞った SDF 差分 |

通常の比較ではまず `sdf` を使います。material ごとの原因分析が必要になった場合だけ
`sdf_material` や `sdf_band` を追加します。

```yaml
metrics:
  use: [cd, sdf, iou]
```

## `sdf`

`sdf` は 2D view 上の SDF 差分を評価します。
label volume target の場合は、material label field だけでなく projected
material boundary の SDF も考慮します。

外形が同じで内部 material boundary だけが違うケースでも、差分を拾いやすくしています。

## `sdf_raw`

`sdf_raw` は `transform` 専用の 3D feature です。non-void union を interior として、
raw signed distance を `features/sdf_raw.npz` に保存します。

```yaml
features:
  use: [sdf_raw]
```

出力 field:

| field | 内容 |
|---|---|
| `sdf_nm` | 3D signed distance、shape は `[Z,Y,X]`、単位 nm |
| `mask` | non-void union mask、shape は `[Z,Y,X]` |
| `spacing_zyx_nm` | 内部 grid spacing |
| `origin_zyx_nm` | 内部 grid origin |
| `material_ids` | 入力に存在する material id |
| `void_id` | void material id |

符号は inside が負、outside が正です。`feature_summary.json` には shape、dtype、
spacing、units、min/max/mean/std、NaN/inf 数を出します。

`sdf_raw` は metric ではありません。`compare` の score には直接影響せず、
サロゲート学習や後段解析に渡す 3D field feature として使います。

## `sdf_material`

`sdf_material` は、non-void material を自動検出して material ごとの SDF loss を出します。
ユーザーが material id を YAML に列挙する必要はありません。

総合 loss は、各 material の projected union area による重み付き平均です。
小さいノイズ material が ranking を過剰に支配しにくくなります。

material ごとの詳細は次に出力されます。

```text
metric_details.json
per_material_sdf.csv
```

## `sdf_band`

`sdf_band` は boundary 近傍だけを見る SDF loss です。
デフォルトでは 10 nm band を使います。

遠い背景領域より、界面や edge 近傍のずれを重視したい場合に使います。

## 内部契約

- 3D label volume は内部で `[Z,Y,X]`。
- 2D 比較 map は `[Y,X]`。
- 単位は loader で変換し、内部では nm として扱います。
- 3D raw SDF の符号は inside negative、outside positive です。

## 拡張ルール

- 新しい SDF 実装は小さい関数または backend として追加する。
- scoring logic は `compare.metric_*` 側に置く。
- YAML の設定項目は、実運用で必要になったものだけ増やす。

## `tsdf_views`

`tsdf_views` は `transform` 専用の任意 feature です。`sdf_raw` と同じ non-void union の raw 3D SDF から、
外部解析や学習に使いやすい固定 clip 幅の TSDF views を `features/tsdf_views.npz` にまとめます。

```yaml
features:
  use: [tsdf_views]
```

出力 field:

| field | 内容 |
|---|---|
| `sdf_nm` | 元の 3D raw SDF、shape は `[Z,Y,X]`、単位 nm |
| `tsdf_10nm` | `sdf_nm / 10` を `[-1, 1]` に clip した TSDF |
| `tsdf_30nm` | `sdf_nm / 30` を `[-1, 1]` に clip した TSDF |
| `tsdf_100nm` | `sdf_nm / 100` を `[-1, 1]` に clip した TSDF |
| `log_abs_sdf` | `log1p(abs(sdf_nm))` |
| `mask` | non-void union mask |
| `clip_nm` | clip 幅 `[10, 30, 100]` |
| `spacing_zyx_nm` | 内部 grid spacing |
| `origin_zyx_nm` | 内部 grid origin |
| `material_ids` | 入力に存在する material id |
| `void_id` | void material id |

`tsdf_views` は metric ではありません。`compare` の score には影響せず、`transform` で特徴量を出したい場合だけ指定します。

## `udf`

`udf` は `transform` 専用の 3D feature です。`sdf_raw` と同じ non-void union から、
boundary への unsigned distance を `features/udf.npz` に保存します。

```yaml
features:
  use: [udf]
```

出力 field:

| field | 内容 |
|---|---|
| `udf_nm` | 3D unsigned distance、shape は `[Z,Y,X]`、単位 nm |
| `mask` | non-void union mask |
| `spacing_zyx_nm` | 内部 grid spacing |
| `origin_zyx_nm` | 内部 grid origin |
| `material_ids` | 入力に存在する material id |
| `void_id` | void material id |

初期実装の `udf` は label volume 由来です。`contour_json` から直接 UDF を作る route は、
grid 範囲と解像度の contract を決めてから追加します。

## 内部 helper

2D view の SDF 計算は `wafergeo.compare.sdf_helpers` に集約しています。

| helper | 用途 |
|---|---|
| `signed_distance_from_mask_2d` | mask の signed distance。inside は負、outside は正 |
| `unsigned_distance_from_mask_2d` | open contour など線/境界からの unsigned distance |
| `clipped_signed_distance_from_mask_2d` | material ごとの SDF loss で使う capped distance |
| `tsdf_from_sdf_nm` | `tsdf_views` の TSDF 派生表現 |

helper は 2D `[Y,X]` mask と正の `spacing_yx` だけを受け付けます。
3D full SDF は domain 層、2D view SDF は compare 層という境界を守るためです。

3D full-material TSDF は `wafergeo.sdf.full_material` の責務です。2D view helper と 3D SDF backend は混ぜず、用途ごとに分けます。
