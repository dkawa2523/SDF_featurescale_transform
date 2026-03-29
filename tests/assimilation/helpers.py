from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from tests.observe.helpers import build_material_spec, build_observer_spec
from tests.sdf.helpers import register_bruteforce_engine
from wafergeo.assimilation.policies import FailurePolicy, LoggingPolicy, TransformPolicy
from wafergeo.assimilation.types import CaseSpec, ModelPackageSpec, ParamAxis, ParamSpec
from wafergeo.core.grid import GridSpec
from wafergeo.core.meta import Meta
from wafergeo.core.types import LabelVolume
from wafergeo.io.artifact_store import LocalDiskArtifactStore
from wafergeo.metrics.spec import MetricEntrySpec, MetricSpecV2
from wafergeo.observe.factory import create_observer
from wafergeo.sem.artifact import write_sem_obs_artifact
from wafergeo.sem.qa import SEMQA


def _grid3d(shape_zyx: tuple[int, int, int]) -> GridSpec:
    _ = shape_zyx
    return GridSpec(
        dim=3,
        spacing=(10.0, 10.0, 10.0),
        origin=(0.0, 0.0, 0.0),
        axis_order="ZYX",
        sample_location="cell_center",
        units="nm",
    )


def make_label_volume(
    *,
    shape_zyx: tuple[int, int, int] = (3, 40, 40),
    shift_x: int = 0,
) -> LabelVolume:
    z_size, y_size, x_size = shape_zyx
    labels = np.zeros(shape_zyx, dtype=np.uint8)
    x0 = max(2, 12 + int(shift_x))
    x1 = min(x_size - 2, x0 + 12)
    y0 = 12
    y1 = min(y_size - 2, y0 + 12)
    labels[:, y0:y1, x0:x1] = 1
    meta = Meta(
        schema_version="label/v1",
        profile_id="assim_test_label",
        config_hash="cfg",
        generator_version="0.1.0",
        git_commit="deadbeef",
        input_hash=f"shape_{shape_zyx}",
        created_at=datetime.now(UTC).isoformat(),
        extra={},
    )
    return LabelVolume(
        grid=_grid3d(shape_zyx),
        material=build_material_spec(),
        material_id=labels,
        meta=meta,
    )


class ShiftSurrogate:
    def __init__(self, *, shape_zyx: tuple[int, int, int] = (3, 40, 40)) -> None:
        self.name = "shift_surrogate"
        self.shape_zyx = shape_zyx

    def predict(self, params: dict[str, object]) -> LabelVolume:
        if bool(params.get("raise_error", False)):
            raise RuntimeError("forced surrogate error")
        shift_x = int(params.get("shift_x", 0))
        return make_label_volume(shape_zyx=self.shape_zyx, shift_x=shift_x)


def make_case_and_store(
    tmp_path,
    *,
    logging_mode: str = "none",
    logging_period: int = 2,
    oob_policy: str = "clamp",
    observer_kind: str = "topdown_exposed",
    model_shape_zyx: tuple[int, int, int] = (3, 40, 40),
    model_obj: object | None = None,
    param_spec_override: ParamSpec | None = None,
    on_surrogate_exception: str = "penalty",
    on_observer_exception: str = "penalty",
) -> tuple[CaseSpec, LocalDiskArtifactStore, str]:
    backend = register_bruteforce_engine("brute_assim")
    observer_spec = build_observer_spec(
        kind=observer_kind,
        backend=backend,
        contour_resample_points=0,
        allow_missing_backend=True,
        params={},
    )
    observer = create_observer(observer_spec.kind)
    baseline_label = make_label_volume(shape_zyx=(3, 40, 40), shift_x=0)
    sem_obs = observer.observe(baseline_label, observer_spec)

    store = LocalDiskArtifactStore(root=tmp_path / "artifacts")
    sem_obs_id = write_sem_obs_artifact(
        store,
        sem_obs,
        SEMQA(status="OK"),
        extra_payload={},
    )

    metric_spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="assim_test_metrics",
        fail_penalty=1e6,
        observer_weights={"topdown": 1.0},
        metrics=(
            MetricEntrySpec(
                name="tsdf_band_robust_weight",
                weight=1.0,
                observers=("topdown",),
                params={
                    "band": "obs_band",
                    "robust": {"type": "l2"},
                    "weight_mode": "uniform",
                },
            ),
        ),
    )

    param_spec = param_spec_override or ParamSpec(
        axes=[
            ParamAxis(
                name="shift_x",
                kind="int",
                bounds=(-5.0, 5.0),
                transform="identity",
                default=0,
            )
        ],
        vector_order=["shift_x"],
    )
    model_value = model_obj or ShiftSurrogate(shape_zyx=model_shape_zyx)

    case = CaseSpec(
        case_id="assim_case_test",
        sem_obs_ids={"topdown": sem_obs_id},
        observer_specs={"topdown": observer_spec},
        metric_spec=metric_spec,
        measurement_specs_by_ref={},
        model_package=ModelPackageSpec(
            model_package_id="model_pkg_test",
            loader_key="in_memory",
            model_ref={"model": model_value},
        ),
        param_spec=param_spec,
        transform_policy=TransformPolicy(mode="strict_sim_grid"),
        failure_policy=FailurePolicy(
            out_of_bounds=oob_policy,
            penalty=1e6,
            on_surrogate_exception=on_surrogate_exception,
            on_observer_exception=on_observer_exception,
        ),
        logging_policy=LoggingPolicy(mode=logging_mode, period=logging_period, save_pred_obs=False),
    )
    return case, store, sem_obs_id


def count_assim_trial_dirs(store: LocalDiskArtifactStore) -> int:
    root = store.root / "assim_trial"
    if not root.exists():
        return 0
    return len([d for d in root.iterdir() if d.is_dir()])
