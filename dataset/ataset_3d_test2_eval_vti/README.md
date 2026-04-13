# atset_3d_test2 evaluation dataset

This dataset is regenerated from `dataset/ataset_3d_test2` for feature and comparison evaluation.

Material semantics:

- `ID0`: mask material
- `ID2`: void / no material
- `ID5`, `ID6`, `ID9`: small process/material labels preserved from source
- `ID10`: main material; source `ID7`, `ID8`, and `ID10` are merged here
- source `ID11` is merged into `ID2` because it also means no material

NPZ contract:

- `labels`: `[X,Y,Z]`
- `spacing`: `[X,Y,Z]`, nm
- `origin`: `[X,Y,Z]`, nm
- `void_id`: `2`

VTI contract:

- `MaterialIds` is written as `CellData`.
- The label values match the NPZ dataset exactly.
- When using VTI through YAML, set `void_id: 2` because generic VTI readers may not preserve the void-id metadata.
