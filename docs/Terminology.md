# Terminology

Use these words for transform and transform-eval feature wording.

```text
target_shape x method, plus derived relations
```

`target_shape` is the geometry being converted.

`method` is the distance-field conversion applied to that geometry.

`code_name` is the current implementation name accepted by `features.use`.

`execution_label` is the generated row label in eval CSV files.

`relation` is a compact field derived from SDF stacks. It is not a new SDF
method.

## Target Shapes

| target_shape | meaning |
| --- | --- |
| `full_shape` | Non-void final geometry. |
| `material_shape` | One material-id shape at a time. |
| `process_delta_shape` | Changed geometry from reference to final. |

## Methods

| method | meaning |
| --- | --- |
| `sdf` | Signed distance field. |
| `multi_scale_tsdf` | Multiple clipped signed-distance channels. |
| `udf` | Unsigned distance field. |

## Relations

| relation | source | meaning |
| --- | --- | --- |
| `material_interface_relation` | `material_shape` SDF stack | Nearest material-interface distance and material pair. |
| `process_transition_relation` | `process_delta_shape` SDF | Reference-to-final material transition code and transition distance. |

## Current Code Names

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

Do not use a combined code name as the method name. For example,
`material_sdf` means:

```text
target_shape=material_shape
method=sdf
```
