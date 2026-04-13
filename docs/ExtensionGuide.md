# Extension Guide

Add one small capability at a time.

## Add A Loader

1. Add the reader in `wafergeo/compare/loader.py`.
2. Register it in the loader map.
3. Add the YAML kind in `schema_types.py`.
4. Add focused tests with a tiny synthetic file.

Loaders should return domain objects such as `LabelVolume` or `ContourData`.

## Add A Transform Feature

1. Implement feature math in a feature module.
2. Add a writer in `feature_outputs.py`.
3. Add the feature name to schema validation.
4. Add the feature name to `feature_taxonomy.py`.
5. Write outputs under `features/<feature_name>.*`.
6. Record metadata in `feature_summary.json`.
7. Add focused synthetic tests.

Use [Terminology](Terminology.md). Every transform-eval feature name must map
to `target_shape x method`.

For process-aware features, require `process.enabled: true` and
`input.reference`.

## Add A Metric

1. Add `wafergeo/compare/metric_<name>.py`.
2. Implement `compute_<name>(context)`.
3. Register it in `metric_defs.py`.
4. Declare required features.
5. Add focused tests and update `Scoring.md`.

Metrics should consume prepared feature objects, not raw files.

## Add A Compare-Eval Axis

Most compare-eval extensions do not need new code. Add a new key under
`eval.metric_sets` and combine existing metrics:

```yaml
eval:
  metric_sets:
    shape_distance:
      features:
        use: [sdf, contour]
      metrics:
        use: [sdf, iou]
```

Name the key by the geometry question it answers, such as `height_cd` or
`shape_distance`. Add a new metric only when no existing metric can express the
question.

## Add An Output

Prefer CSV/JSON/NPZ for authoritative data. PNG files are diagnostics.
Keep output writers small and called from runners after core data has been
written.
