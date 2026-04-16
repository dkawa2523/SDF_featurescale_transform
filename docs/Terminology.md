# 用語

transform / transform-eval の特徴量は、次の言葉で整理します。

```text
target_shape x method, plus derived relations
```

`target_shape` は、特徴量化の対象となる geometry です。

`method` は、その geometry に適用する distance-field 変換です。

`code_name` は、現在の実装で `features.use` に指定できる名前です。

`execution_label` は、eval CSV の行ラベルとして生成される名前です。

`relation` は、SDF stack から派生する compact field です。新しい SDF 手法ではありません。

## Target Shapes

| target_shape | 意味 |
| --- | --- |
| `full_shape` | final geometry の non-void 領域 |
| `material_shape` | material id ごとの shape |
| `process_delta_shape` | reference から final への変化領域 |

## Methods

| method | 意味 |
| --- | --- |
| `sdf` | Signed distance field |
| `multi_scale_tsdf` | 同じ SDF から作る複数の clipped signed-distance channel。near-boundary と広い文脈を両方モデルに見せたい場合に有用 |
| `udf` | Unsigned distance field。inside/outside の符号ではなく、boundary への近さを使いたい場合に有用 |

これらは method 名であり、target shape ではありません。同じ method を
`full_shape`, `material_shape`, `process_delta_shape` に適用できます。

## Relations

| relation | source | 意味 |
| --- | --- | --- |
| `material_interface_relation` | `material_shape` SDF stack | 最近接 material interface distance と material pair |
| `process_transition_relation` | `process_delta_shape` SDF | reference-to-final material transition code と transition distance |

## 現在の Code Names

| code_name | target_shape | method or relation |
| --- | --- | --- |
| `sdf_raw` | `full_shape` | `sdf` |
| `tsdf_views` | `full_shape` | `multi_scale_tsdf` |
| `udf` | `full_shape` | `udf` |
| `material_sdf` | `material_shape` | `sdf` |
| `material_tsdf_views` | `material_shape` | `multi_scale_tsdf` |
| `material_udf` | `material_shape` | `udf` |
| `material_interface_relation` | `material_shape` | relation |
| `process_delta_sdf` | `process_delta_shape` | `sdf` |
| `process_delta_tsdf_views` | `process_delta_shape` | `multi_scale_tsdf` |
| `process_delta_udf` | `process_delta_shape` | `udf` |
| `process_transition_relation` | `process_delta_shape` | relation |

組み合わせ済みの code name を method 名として扱わないでください。
例えば `material_sdf` は次を意味します。

```text
target_shape=material_shape
method=sdf
```

## Compare Terms

compare workflow では別の 3 つの言葉を使います。

| term | 意味 |
| --- | --- |
| compare feature | simulation と target から作る中間表現。例: `sdf`, `contour` |
| metric | compare feature を使って loss を計算する手法。例: `sdf`, `cd`, `iou` |
| evaluation axis | `eval.metric_sets` 配下の名前付き metric group。例: `height_cd`, `shape_distance` |

feature `sdf` と metric `sdf` は同じ名前ですが、属する layer が違います。

```text
features.use: [sdf]  -> SDF field を作る
metrics.use: [sdf]   -> SDF field 同士を比較する
```
