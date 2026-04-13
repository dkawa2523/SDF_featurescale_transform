# Scoring

`compare` 系 workflow は、simulation と target を同じ 2D view にそろえてから
metric を計算します。

結果は metric ごとの値と、重み付き合算の `objective` として出力されます。
外部 optimizer や sampler は、この `objective.json` を読む想定です。

## metric の分類

| 分類 | metric | 何を見るか | 向いている用途 |
| --- | --- | --- | --- |
| Primary | `cd` | 高さごとの幅、edge 位置 | 製造 CD 管理、説明しやすい比較 |
| Primary | `sdf` | 形状全体の距離場差 | 外形と内部形状の総合 loss |
| Primary | `iou` | 領域の重なり | 一致率、欠損/過剰領域の確認 |
| Boundary | `sdf_band` | 境界近傍の SDF 差 | mask 丸み、微細 edge 差分 |
| Material | `sdf_material` | material ごとの距離場差 | 成膜、エッチング、material 差分 |
| Contour | `chamfer` | contour 点群距離 | 輪郭点列同士の比較 |
| Local | `corner` | corner 周辺の形状差 | 局所形状を強く合わせたい場合 |
| Profile | `profile` | profile 量の差 | 高さ方向や material 量の集計比較 |
| Topology | `topology` | 連結性、穴、分離 | 大域的な形状崩れの検出 |

## 最初に使う組み合わせ

| 目的 | 推奨 metric |
| --- | --- |
| まず全体一致を見たい | `[cd, sdf, iou]` |
| 境界の丸みや微細差分を見たい | `[sdf, sdf_band, iou]` |
| material 差分を見たい | `[sdf, sdf_material, iou]` |
| 説明しやすい CD を重視したい | `[cd, sdf]` |
| metric set を比較したい | `compare-eval` で候補を並べる |

## objective の作り方

```yaml
metrics:
  use: [sdf, sdf_band, iou]
  weights:
    sdf: 1.0
    sdf_band: 1.0
    iou: 1.0
```

`metrics.use` に複数 metric を入れると、各 metric の値を別々に出したうえで、
正規化 loss と `weights` から 1 つの `objective` も出します。

つまり、最適化には `objective.json` を使い、原因分析には `metrics.csv` を使います。

## CD gauge

`cd` は、断面形状で中心から左右の edge までの幅を高さごとに評価する metric です。
必要に応じて gauge を指定できます。

```yaml
metrics:
  use: [cd, sdf, iou]
  cd:
    gauge:
      axis: x
      height_axis: z
      center: 0.0
      height_range: [-200.0, 100.0]
```

未指定の場合は、view と座標から妥当な既定値を使います。
まずは未指定で動かし、製造上の測定定義が決まっている場合だけ明示してください。

## 半導体製造エンジニア向けの見方

| 見たいこと | 注目する metric |
| --- | --- |
| CD が target に近いか | `cd` |
| trench / hole の形状全体が近いか | `sdf`, `iou` |
| mask 丸み、肩、底部の微細差が近いか | `sdf_band`, `corner` |
| material ごとの加工差分が近いか | `sdf_material`, `profile` |
| 形が分断・連結していないか | `topology` |

## データサイエンティスト向けの見方

| 見たいこと | 注目する出力 |
| --- | --- |
| 最適化で使う 1 つの loss | `objective.json` |
| metric ごとの寄与 | `metrics.csv` |
| 複数 metric set の安定性 | `compare-eval/candidate_summary.csv` |
| case ごとの順位変化 | `compare-eval/case_scores.csv`, `ranking_consistency.csv` |

`cd` は説明性が高い一方、局所的な内部差分を拾いにくい場合があります。
`sdf`, `sdf_band`, `sdf_material`, `iou` を組み合わせると、
最適化で形状全体と境界近傍の両方を扱いやすくなります。

## 参考

- H. Blum, "A Transformation for Extracting New Descriptors of Shape", 1967.
- H. Zhao, "A Fast Sweeping Method for Eikonal Equations", 2005.
- P. Jaccard, "The Distribution of the Flora in the Alpine Zone", 1912.
- H. Edelsbrunner and J. Harer, "Computational Topology", 2010.
