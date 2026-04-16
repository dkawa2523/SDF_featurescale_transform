# 拡張ガイド

拡張は、小さな capability を 1 つずつ追加します。

## Loader を追加する

1. label volume 用なら `label_loaders.py`、contour-like target 用なら
   `contour_loaders.py` に reader を追加する。
2. 対応する loader map と dispatch path に登録する。
3. `schema_types.py` に YAML kind を追加する。
4. 小さい synthetic file を使った focused test を追加する。

loader は `LabelVolume` や `ContourData` などの domain object を返します。
data normalization は loader 側に閉じ込め、feature code と metric code が安定した domain object を使えるようにします。

## Transform Feature を追加する

1. feature math を feature module に実装する。
2. `feature_outputs.py` に writer を追加する。
3. schema validation に feature name を追加する。
4. `feature_taxonomy.py` に feature name を追加する。
5. `features/<feature_name>.*` 配下に出力する。
6. `feature_summary.json` に metadata を記録する。
7. focused synthetic tests を追加する。

[用語](Terminology.md) に従ってください。すべての transform-eval feature name は
`target_shape x method` に mapping できる必要があります。

process-aware feature では `process.enabled: true` と `input.reference` を必須にします。

## Metric を追加する

1. `wafergeo/compare/metric_<name>.py` を追加する。
2. `compute_<name>(context)` を実装する。
3. `metric_defs.py` に登録する。
4. required features を宣言する。
5. focused tests を追加し、`Scoring.md` を更新する。

metrics は raw files ではなく、準備済み feature objects を消費してください。
`required_features` で、metric 実行前にどの compare features が必要かを明示します。
例えば SDF-family metric は feature `sdf` を要求し、contour/profile 系 metric は
feature `contour` を要求します。

## Compare-Eval Axis を追加する

ほとんどの compare-eval 拡張では新しい code は不要です。
`eval.metric_sets` に新しい key を追加し、既存 metrics を組み合わせます。

```yaml
eval:
  metric_sets:
    shape_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou]
```

key 名は、どの形状評価の問いに答えるかで付けます。
例: `height_cd`, `shape_distance`。
既存 metric では問いを表現できない場合だけ、新しい metric を追加します。

axis 名は user-facing に保ちます。どの実装ファイルを使うかではなく、
なぜその metric group が必要かを表す名前にしてください。

## Output を追加する

正本は CSV/JSON/NPZ を優先します。PNG files は診断用です。
output writers は小さく保ち、core data を書いた後に runner から呼びます。
