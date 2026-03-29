from __future__ import annotations

import pytest

from tests.metrics.helpers import build_obs_pair_shifted
from wafergeo.metrics.aggregate import build_metric_context
from wafergeo.metrics.spec import (
    MeasurementLineSpec,
    MeasurementSpecV1,
    MetricEntrySpec,
    MetricSpecV2,
)

pytest.importorskip("scipy")


def _measurement() -> MeasurementSpecV1:
    return MeasurementSpecV1(
        schema_version="measurement/v1",
        name="cd_test",
        lines=(
            MeasurementLineSpec(
                id="center",
                axis="x",
                coord_nm=160.0,
                range_nm=(40.0, 280.0),
                expected_edges=2,
                edge_pair="outer",
                method="tsdf_zero_cross",
            ),
        ),
    )


def test_kdtree_and_line_cache_built_once() -> None:
    _, obs, _ = build_obs_pair_shifted(shift_x=0)
    spec = MetricSpecV2(
        schema_version="metric/v2",
        metric_set_id="ctx_test",
        metrics=(
            MetricEntrySpec(
                name="contour_chamfer",
                weight=1.0,
                observers=("topdown",),
                params={"use_holes": False},
            ),
            MetricEntrySpec(
                name="cd_linescan",
                weight=1.0,
                observers=("topdown",),
                params={},
                measurement_ref="cd_ref",
            ),
        ),
    )

    ctx = build_metric_context(
        {"topdown": obs},
        spec,
        {"cd_ref": _measurement()},
    )

    contour_ctx = ctx.precomputed[(0, "topdown")]
    cd_ctx = ctx.precomputed[(1, "topdown")]

    assert isinstance(contour_ctx, dict)
    assert contour_ctx.get("contour_kdtree") is not None
    assert isinstance(cd_ctx, dict)
    assert "line_scan_cache" in cd_ctx
    assert "center" in cd_ctx["line_scan_cache"]
