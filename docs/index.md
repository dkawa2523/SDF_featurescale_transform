# wafergeo

`wafergeo` converts simulation or measurement-derived geometry into feature
files and compares geometry against targets. The package writes data that
external analysis or optimization code can consume.

## Workflows

| goal | workflow | main outputs |
| --- | --- | --- |
| Convert one case to features | `transform` | `features/`, `feature_summary.json` |
| Convert many cases | `batch-transform` | `dataset_index.csv`, `features_summary.csv` |
| Evaluate feature methods | `transform-eval` | CSV summaries, `figures/` |
| Compare one simulation/target pair | `compare` | `objective.json`, `metrics.csv`, `difference.png` |
| Compare many pairs | `batch-compare` | `objectives.csv`, `ranking.csv` |
| Evaluate comparison axes | `compare-eval` | `metric_set_summary.csv`, `case_scores.csv`, `figures/` |

## Feature Evaluation

For `transform-eval`, use the wording in [Terminology](Terminology.md):

```text
target_shape x method
```

Examples:

- `sdf_raw`: `target_shape=full_shape`, `method=sdf`
- `material_sdf`: `target_shape=material_shape`, `method=sdf`
- `process_delta_sdf`: `target_shape=process_delta_shape`, `method=sdf`
- `material_interface_relation`: relation derived from material SDF channels
- `process_transition_relation`: relation derived from reference/final changes

Transform-eval YAML uses `eval.features`; compare-eval YAML uses
`eval.metric_sets`. In compare-eval, each `metric_sets` key is an evaluation
axis name.

## Reading Order

| need | page |
| --- | --- |
| Run the first example | [Quickstart](Quickstart.md) |
| Check naming rules | [Terminology](Terminology.md) |
| Understand YAML and outputs | [User Manual](UserManual.md) |
| Interpret eval figures | [Eval Visualization](EvalVisualization.md) |
| Understand views | [View](View.md) |
| Understand metrics | [Scoring](Scoring.md) |
| Modify code | [Developer Manual](DeveloperManual.md) |
| Add loaders/features/metrics | [Extension Guide](ExtensionGuide.md) |
