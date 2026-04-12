# SDF

SDF は Signed Distance Field の略で、境界からの距離を持つ特徴量です。
このパッケージでは、特徴量化と比較評価の両方で SDF を使います。

## 公開用途

| 場所 | 内容 |
|---|---|
| `transform` の `sdf` | 2D view の軽量 SDF を `features/simulation_sdf.npz` に出力 |
| `transform` の `sdf3d` | full 3D SDF を `features/sdf.npz` に出力 |
| `compare` の `sdf` | simulation と target の 2D SDF 差分 |
| `compare` の `sdf_material` | material ごとの SDF 差分 |
| `compare` の `sdf_band` | boundary 近傍 10 nm に絞った SDF 差分 |

通常の比較では次の metric を使います。

```yaml
metrics:
  use: [cd, chamfer, sdf, sdf_material, sdf_band, iou]
```

## `sdf`

`sdf` は 2D view 上の SDF 差分を評価します。
label volume target の場合は、material label field だけでなく projected
material boundary の SDF も考慮します。

外形が同じで内部 material boundary だけが違うケースでも、差分を拾いやすくしています。

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

## 拡張ルール

- 新しい SDF 実装は小さい関数または backend として追加する。
- scoring logic は `compare.metric_*` 側に置く。
- YAML の設定項目は、実運用で必要になったものだけ増やす。
