from __future__ import annotations

from typing import cast

from wafergeo.core.registry import Registry
from wafergeo.metrics.base import MetricProtocol
from wafergeo.metrics.cd_metrics import CDLineScanMetric
from wafergeo.metrics.contour_metrics import ContourChamferMetric
from wafergeo.metrics.tsdf_loss import TsdfBandRobustWeightMetric

_metric_registry: Registry[object] = Registry("metric_impl")


def register_metric(metric: MetricProtocol, *, aliases: tuple[str, ...] = ()) -> None:
    _metric_registry.register(metric.name, metric, override=True)
    for alias in aliases:
        _metric_registry.register(alias, metric, override=True)


def get_metric(name: str) -> MetricProtocol:
    register_default_metrics()
    return cast(MetricProtocol, _metric_registry.get(name))


def list_metrics() -> tuple[str, ...]:
    register_default_metrics()
    return _metric_registry.list()


def register_default_metrics() -> None:
    if "tsdf_band_robust_weight" not in _metric_registry.list():
        register_metric(TsdfBandRobustWeightMetric())
    if "contour_chamfer" not in _metric_registry.list():
        register_metric(ContourChamferMetric())
    if "cd_linescan" not in _metric_registry.list():
        register_metric(CDLineScanMetric())
