# Eval Visualization

Eval figures are diagnostics. CSV/JSON/NPZ files remain the authoritative data.

## Transform-Eval

Read transform-eval as:

```text
target_shape x method, plus relation outputs derived from SDF stacks
```

Transform-eval YAML uses `eval.features`. Each entry explicitly names one
`target_shape` and one `method`; `code_name` is resolved by the package.
See [Terminology](Terminology.md) for the naming rules.

Current targets:

| target_shape | meaning |
| --- | --- |
| `full_shape` | final non-void shape |
| `material_shape` | material-id-specific shapes |
| `process_delta_shape` | reference-to-final changed shape |

Current feature mapping:

| feature | target_shape | method or relation |
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

Recommended outputs:

| output | purpose |
| --- | --- |
| `input_shape_sections.png` | Original material geometry for each case. |
| `by_target_shape/<target_shape>/<method>/field.png` | Field report for one explicit target shape and method. |
| `by_target_shape/<target_shape>/<method>/scores.png` | Scores for that target shape and method. |
| `by_target_shape/<target_shape>/<method>/case_distance.png` | Case distance in that feature space. |
| `by_target_shape/<target_shape>/relations/<relation>/field.png` | Relation report derived from SDF fields. |
| `feature_scores.csv` | Scores with separate `role`, `target_shape`, `method`, `relation`, and `code_name` columns. |
| `case_distance.csv` | Case distances with separate `role`, `target_shape`, `method`, and `relation` columns. |
| `distance_correlation.csv` | Machine-readable redundancy check. |

Avoid mixed aggregate scores across different targets. Use the separate score
columns instead:

| score | meaning |
| --- | --- |
| `shape_match` | SDF/TSDF can recover the source shape. |
| `boundary_match` | UDF captures the source boundary neighborhood. |
| `interface_match` | Material-interface relation matches material boundaries. |
| `transition_match` | Process-transition relation matches changed voxels. |
| `case_sensitivity` | The feature changes across cases. |
| `data_cost` | Higher means smaller feature output. |

## Compare-Eval

`compare-eval` compares evaluation axes. Each axis is still configured under
`eval.metric_sets`, but the names should describe the comparison purpose:
`height_cd`, `shape_distance`, `material_distance`, or
`boundary_band_distance`.

In output CSV files, the legacy column name `metric_set` means the evaluation
axis name. Treat it as a grouping column, not as a separate product concept.

It should show:

- which cases each evaluation axis scores differently;
- which metrics dominate the comparison loss;
- whether changing metrics changes ranking;
- whether skipped metrics affect the result.

Recommended outputs:

| output | purpose |
| --- | --- |
| `comparison_loss_heatmap.png` | Comparison loss by evaluation axis and case. Lower is better. |
| `ranking_shift_heatmap.png` | Ranking movement relative to baseline. |
| `metric_loss_breakdown.png` | Which metrics dominate. |
| `cd_vs_sdf_scatter.png` | Whether the height-CD axis and SDF shape-distance axis judge cases differently. |
| `evaluation_axis_summary.png` | Coverage, case separation, and ranking shift diagnostics. |
| `representative_differences/*.png` | Shape-level inspection for selected cases. |
| `axis_agreement.csv` | Pairwise loss correlation and rank agreement between evaluation axes. |

Use `comparison_loss` for ranking or optimization. Use
`evaluation_axis_summary` only as a diagnostic; it is not an objective.

`cd_vs_sdf_scatter.png` plots axis-level `comparison_loss`, not raw `cd_loss`
against raw `sdf_loss`. Inspect `case_scores.csv` and `metric_summary.csv` when
you need the per-metric raw losses.

Do not add a `process_delta` compare axis unless the compare input has an
explicit reference geometry. Without that reference, a process-delta score would
hide an assumption instead of measuring the process change.

Use `[x,z]` or `[y,z]` views when comparing against `height_cd`. A top-view
`[x,y]` comparison can still be useful for SDF shape overlap, but it cannot
answer whether SDF is better than height-wise CD.

## Code Shape

- Runners resolve config, run workflows, collect rows, and call writers.
- Feature math stays in feature modules.
- Figure code stays in `transform_eval_figures.py` and
  `compare_eval_figures.py`.
- Figure generation requires Matplotlib. Missing dependencies or missing
  feature fields should fail with a clear error instead of producing incomplete
  images.
