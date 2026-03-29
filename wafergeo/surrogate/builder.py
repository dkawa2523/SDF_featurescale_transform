from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from wafergeo.io.artifact_store import ArtifactStore
from wafergeo.observe.spec import ObserverSpecV2, observer_spec_hash
from wafergeo.surrogate.adapters import (
    load_mesh_from_artifact,
    load_obs2d_from_artifact,
    load_tsdf_from_artifact,
)
from wafergeo.surrogate.export.json_npy_export import pack_sample_json_npy
from wafergeo.surrogate.manifest_io import (
    SampleInputRow,
    load_input_manifest_json,
    write_dataset_manifest,
)
from wafergeo.surrogate.qa import (
    compute_dataset_qa_summary,
    compute_dataset_stats,
    compute_sample_qa,
)
from wafergeo.surrogate.schema import DatasetManifest, SampleRecord
from wafergeo.surrogate.spec import DatasetBuildSpecV1, dataset_build_spec_hash


def _qa_messages(qa: dict[str, object]) -> list[str]:
    raw = qa.get("messages", [])
    if not isinstance(raw, list):
        return []
    return [str(v) for v in raw]


def _require_obs_ids(
    row: SampleInputRow,
    expected_observers: list[str],
    *,
    include_obs2d_pack: bool,
) -> tuple[dict[str, str], list[str], str]:
    obs_ids = dict(row.obs2d_sim_ids)
    missing = [name for name in expected_observers if name not in obs_ids]
    if not missing:
        return obs_ids, [], "OK"

    messages = [f"missing obs2d_sim_ids for observers: {missing}"]
    if include_obs2d_pack:
        return obs_ids, messages, "FAIL"
    return obs_ids, messages, "WARN"


def _material_payload_from_tsdf(tsdf) -> dict[str, object]:
    return {
        "ids": list(tsdf.material.ids),
        "names": list(tsdf.material.names),
        "void_id": int(tsdf.material.void_id),
        "priority": list(tsdf.material.priority),
        "ignore_in_exposure": list(tsdf.material.ignore_in_exposure),
    }


def _material_payload_from_mesh(mesh) -> dict[str, object]:
    return {
        "ids": list(mesh.material.ids),
        "names": list(mesh.material.names),
        "void_id": int(mesh.material.void_id),
        "priority": list(mesh.material.priority),
        "ignore_in_exposure": list(mesh.material.ignore_in_exposure),
    }


def _grid_payload_from_tsdf(tsdf) -> dict[str, object]:
    return {
        "dim": int(tsdf.grid.dim),
        "spacing": list(tsdf.grid.spacing),
        "origin": list(tsdf.grid.origin),
        "axis_order": tsdf.grid.axis_order,
        "sample_location": tsdf.grid.sample_location,
        "units": tsdf.grid.units,
    }


def _grid_payload_from_mesh(mesh) -> dict[str, object]:
    return {
        "dim": int(mesh.grid.dim),
        "spacing": list(mesh.grid.spacing),
        "origin": list(mesh.grid.origin),
        "axis_order": mesh.grid.axis_order,
        "sample_location": mesh.grid.sample_location,
        "units": mesh.grid.units,
    }


def _is_sample_valid(qa: dict[str, object]) -> bool:
    return str(qa.get("status", "OK")) != "FAIL"


def _required_teacher_available(spec: DatasetBuildSpecV1, sample: SampleRecord) -> bool:
    if spec.task_kind == "sdf":
        return bool(sample.tsdf_artifact_id)
    if spec.task_kind == "mesh":
        return bool(sample.mesh_artifact_id)
    return bool(sample.tsdf_artifact_id) and bool(sample.mesh_artifact_id)


def build_dataset_package(
    spec: DatasetBuildSpecV1,
    store: ArtifactStore,
    *,
    output_dir: str | Path,
) -> DatasetManifest:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_input_manifest_json(spec.input_manifest_path)
    if not rows:
        raise ValueError("input manifest is empty")

    build_hash = dataset_build_spec_hash(spec)
    observers = sorted(spec.obs_targets.keys())

    sample_records: list[SampleRecord] = []
    records_qa: list[dict[str, object]] = []
    recipe_params_list: list[dict[str, object]] = []

    materials_payload: dict[str, object] | None = None
    grid3d_payload: dict[str, object] | None = None

    for row in rows:
        sample_messages: list[str] = []
        sample_meta: dict[str, object] = {}

        tsdf = None
        if row.tsdf_artifact_id:
            try:
                tsdf = load_tsdf_from_artifact(store, row.tsdf_artifact_id)
                materials_payload = materials_payload or _material_payload_from_tsdf(tsdf)
                grid3d_payload = grid3d_payload or _grid_payload_from_tsdf(tsdf)
            except Exception as exc:
                sample_messages.append(f"failed to load tsdf artifact: {exc}")

        _mesh = None
        point_cloud = None
        if row.mesh_artifact_id:
            try:
                _mesh, point_cloud = load_mesh_from_artifact(store, row.mesh_artifact_id)
                if _mesh is not None:
                    materials_payload = materials_payload or _material_payload_from_mesh(_mesh)
                    grid3d_payload = grid3d_payload or _grid_payload_from_mesh(_mesh)
            except Exception as exc:
                sample_messages.append(f"failed to load mesh artifact: {exc}")

        obs_ids, obs_messages, obs_status = _require_obs_ids(
            row,
            observers,
            include_obs2d_pack=spec.storage_mode == "packed" and spec.include_obs2d_pack,
        )
        sample_messages.extend(obs_messages)

        obs_targets = {}
        for observer_name, artifact_id in obs_ids.items():
            try:
                obs_targets[observer_name] = load_obs2d_from_artifact(store, artifact_id)
            except Exception as exc:
                sample_messages.append(
                    f"failed to load obs2d artifact observer={observer_name}: {exc}"
                )

        expected_material_ids: list[int]
        if tsdf is not None:
            selected = None if tsdf.meta is None else tsdf.meta.extra.get("selected_material_ids")
            if selected:
                expected_material_ids = [int(v) for v in selected.split(",") if v.strip()]
            else:
                expected_material_ids = list(tsdf.material.ids[: tsdf.tsdf.shape[0]])
        else:
            expected_material_ids = []

        sample_qa = compute_sample_qa(
            tsdf=tsdf,
            point_cloud=point_cloud,
            obs_targets=obs_targets,
            expected_material_ids=expected_material_ids,
        )

        if sample_messages and sample_qa.get("status") == "OK":
            sample_qa["status"] = "WARN"
        if obs_status == "FAIL":
            sample_qa["status"] = "FAIL"
        sample_qa_messages = _qa_messages(sample_qa) + sample_messages
        sample_qa["messages"] = sample_qa_messages

        packed_paths: dict[str, str] = {}
        if spec.storage_mode == "packed":
            try:
                packed_paths = pack_sample_json_npy(
                    base_dir=out_dir,
                    sample_id=row.sample_id,
                    tsdf=tsdf,
                    point_cloud=point_cloud,
                    obs_targets=obs_targets,
                    include_sdf_features=spec.include_sdf_features,
                    include_obs2d_pack=spec.include_obs2d_pack,
                )
            except Exception as exc:
                sample_qa["status"] = "FAIL"
                sample_qa_messages = _qa_messages(sample_qa)
                sample_qa_messages.append(f"packing failed: {exc}")
                sample_qa["messages"] = sample_qa_messages

        sample_meta.update(
            {
                "build_spec_hash": build_hash,
                "storage_mode": spec.storage_mode,
                "task_kind": spec.task_kind,
            }
        )

        record = SampleRecord(
            sample_id=row.sample_id,
            group_id=row.group_id,
            recipe_params=dict(row.recipe_params),
            param_vector=row.param_vector,
            label_artifact_id=row.label_artifact_id,
            tsdf_artifact_id=row.tsdf_artifact_id,
            mesh_artifact_id=row.mesh_artifact_id,
            obs2d_sim_ids=obs_ids,
            packed_paths=packed_paths,
            qa=sample_qa,
            meta=sample_meta,
        )
        sample_records.append(record)
        records_qa.append(sample_qa)
        recipe_params_list.append(dict(row.recipe_params))

    valid_rows = [
        row
        for row, rec in zip(rows, sample_records, strict=True)
        if _is_sample_valid(rec.qa)
    ]
    if not valid_rows:
        raise ValueError("all samples failed during dataset build")

    required_teacher_any = any(
        _required_teacher_available(spec, rec) and _is_sample_valid(rec.qa)
        for rec in sample_records
    )
    if not required_teacher_any:
        raise ValueError("required teacher artifacts are missing for all valid samples")

    from wafergeo.surrogate.splits import make_group_split

    splits = make_group_split(valid_rows, spec.split)

    stats = compute_dataset_stats(records_qa, recipe_params_list)
    qa_summary = compute_dataset_qa_summary(stats=stats, records_qa=records_qa, qa_spec=spec.qa)
    if spec.fail_on_qa_status and str(qa_summary.get("status", "OK")) == "FAIL":
        raise ValueError("dataset qa summary is FAIL and fail_on_qa_status=true")

    now_utc = datetime.now(timezone.utc)  # noqa: UP017
    dataset_id = f"{spec.dataset_id_prefix}_{now_utc.strftime('%Y%m%d_%H%M%S')}"
    observer_hashes = {}
    for name, observer_raw in spec.obs_targets.items():
        try:
            if isinstance(observer_raw, ObserverSpecV2):
                observer_hashes[name] = observer_spec_hash(observer_raw)
            elif isinstance(observer_raw, str):
                observer_hashes[name] = observer_raw
            elif isinstance(observer_raw, dict) and "hash" in observer_raw:
                observer_hashes[name] = str(cast(dict[str, object], observer_raw)["hash"])
            else:
                observer_hashes[name] = "unknown"
        except Exception:
            observer_hashes[name] = "unknown"

    param_spec_hash = spec.param_spec_hash

    if materials_payload is None:
        materials_payload = {
            "ids": [],
            "names": [],
            "void_id": 0,
            "priority": [],
            "ignore_in_exposure": [],
        }
    if grid3d_payload is None:
        grid3d_payload = {
            "dim": 3,
            "spacing": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "axis_order": "ZYX",
            "sample_location": "cell_center",
            "units": "nm",
        }

    manifest = DatasetManifest(
        schema_version="surrogate_dataset/v3",
        dataset_id=dataset_id,
        profile_id=spec.profile_id,
        created_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        generator_version="0.1.0",
        git_commit="unknown",
        storage_mode=spec.storage_mode,
        task_kind=spec.task_kind,
        materials=materials_payload,
        grid3d=grid3d_payload,
        observers=observers,
        param_spec_hash=param_spec_hash,
        observer_spec_hashes=observer_hashes,
        build_spec_hash=build_hash,
        samples=sample_records,
        splits=splits,
        stats=stats,
        qa_summary=qa_summary,
    )

    write_dataset_manifest(out_dir / "dataset_manifest.json", manifest)
    hash_payload = spec.to_hash_payload()
    (out_dir / "build_spec.resolved.json").write_text(
        json.dumps(hash_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "materials.snapshot.json").write_text(
        json.dumps(materials_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "grid3d.snapshot.json").write_text(
        json.dumps(grid3d_payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
