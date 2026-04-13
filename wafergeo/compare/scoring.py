from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from wafergeo.compare.features import ViewFeature
from wafergeo.compare.metric_defs import METRIC_DEFINITIONS, MetricContext
from wafergeo.compare.schema import MetricSpec


@dataclass(frozen=True)
class MetricRow:
    name: str
    loss: float
    value: float
    weight: float
    loss_scale: float
    normalized_loss: float
    status: str = "OK"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "loss": self.loss,
            "value": self.value,
            "weight": self.weight,
            "loss_scale": self.loss_scale,
            "normalized_loss": self.normalized_loss,
            "status": self.status,
        }


@dataclass(frozen=True)
class ScoreResult:
    total_score: float
    normalized_total_score: float
    metrics: list[MetricRow]
    metric_details: list[dict[str, object]] = field(default_factory=list)
    cd_profile: list[dict[str, float]] = field(default_factory=list)
    cd_profile_summary: dict[str, object] | None = None
    profile_rows: list[dict[str, float]] = field(default_factory=list)
    profile_summary: dict[str, object] | None = None
    corner_summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "total_score": self.total_score,
            "normalized_total_score": self.normalized_total_score,
            "metrics": [row.to_dict() for row in self.metrics],
        }
        skipped = [row.name for row in self.metrics if row.status == "SKIPPED"]
        if skipped:
            payload["skipped_metrics"] = skipped
        if self.metric_details:
            payload["metric_details"] = self.metric_details
        if self.cd_profile_summary is not None:
            payload["cd_profile_summary"] = self.cd_profile_summary
        if self.profile_summary is not None:
            payload["profile_summary"] = self.profile_summary
        if self.corner_summary is not None:
            payload["corner_summary"] = self.corner_summary
        return payload


def objective_payload(score: ScoreResult) -> dict[str, object]:
    skipped = [row.name for row in score.metrics if row.status == "SKIPPED"]
    return {
        "schema_version": "objective/v1",
        "status": "OK" if not skipped else "PARTIAL",
        "direction": "minimize",
        "objective_name": "normalized_total_score",
        "objective": score.normalized_total_score,
        "total_score": score.total_score,
        "metrics": {
            row.name: {
                "loss": row.loss,
                "normalized_loss": row.normalized_loss,
                "value": row.value,
                "status": row.status,
            }
            for row in score.metrics
        },
        "skipped_metrics": skipped,
        "failed_metrics": [],
    }


def objective_csv_row(case_id: str, score: ScoreResult) -> dict[str, object]:
    skipped = [row.name for row in score.metrics if row.status == "SKIPPED"]
    return {
        "case_id": case_id,
        "status": "OK" if not skipped else "PARTIAL",
        "objective": score.normalized_total_score,
        "objective_name": "normalized_total_score",
        "direction": "minimize",
        "total_score": score.total_score,
        "skipped_metrics": "|".join(skipped),
    }


def _compact_detail_summary(row: MetricRow, detail: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {
        "metric": row.name,
        "status": row.status,
        "loss": row.loss,
        "value": row.value,
    }
    for key in (
        "mode",
        "selected_loss_source",
        "selected_value_source",
        "distance_semantics",
        "skipped_reason",
    ):
        if key in detail:
            summary[key] = detail[key]
    return summary


def _metric_details_payload(score: ScoreResult) -> dict[str, object]:
    details_by_metric = {
        str(detail.get("metric", "")): detail
        for detail in score.metric_details
        if detail.get("metric")
    }
    summary_rows = [
        _compact_detail_summary(row, details_by_metric[row.name])
        for row in score.metrics
        if row.name in details_by_metric
    ]
    return {
        "_summary": {
            "metrics_with_details": [row["metric"] for row in summary_rows],
            "rows": summary_rows,
        },
        "details": score.metric_details,
    }


def score_features(sim: ViewFeature, target: ViewFeature, metric_spec: MetricSpec) -> ScoreResult:
    rows: list[MetricRow] = []
    cd_profile: list[dict[str, float]] = []
    cd_profile_summary: dict[str, object] | None = None
    profile_rows: list[dict[str, float]] = []
    profile_summary: dict[str, object] | None = None
    corner_summary: dict[str, object] | None = None
    metric_details: list[dict[str, object]] = []
    cd_gauge = metric_spec.cd_gauge
    context = MetricContext(
        cd_material_ids=metric_spec.cd_material_ids,
        cd_gauge_axis=None if cd_gauge is None else cd_gauge.axis,
        cd_gauge_height_axis=None if cd_gauge is None else cd_gauge.height_axis,
        cd_gauge_center_nm=None if cd_gauge is None else cd_gauge.center,
        cd_gauge_height_range_nm=None if cd_gauge is None else cd_gauge.height_range,
    )

    for name in metric_spec.use:
        definition = METRIC_DEFINITIONS[name]
        computed = definition.compute(sim, target, context)
        if computed.cd_profile:
            cd_profile = computed.cd_profile
        if computed.cd_profile_summary is not None:
            cd_profile_summary = computed.cd_profile_summary
        if computed.profile_rows:
            profile_rows = computed.profile_rows
        if computed.profile_summary is not None:
            profile_summary = computed.profile_summary
        if computed.corner_summary is not None:
            corner_summary = computed.corner_summary
        if computed.details is not None:
            metric_details.append(computed.details)
        rows.append(
            MetricRow(
                name=computed.name,
                loss=computed.loss,
                value=computed.value,
                weight=metric_spec.weight_for(name),
                loss_scale=definition.loss_scale,
                normalized_loss=float(computed.loss / definition.loss_scale),
                status=computed.status,
            )
        )

    total = float(sum(row.loss * row.weight for row in rows))
    normalized_total = float(sum(row.normalized_loss * row.weight for row in rows))
    return ScoreResult(
        total_score=total,
        normalized_total_score=normalized_total,
        metrics=rows,
        metric_details=metric_details,
        cd_profile=cd_profile,
        cd_profile_summary=cd_profile_summary,
        profile_rows=profile_rows,
        profile_summary=profile_summary,
        corner_summary=corner_summary,
    )


def write_score_outputs(score: ScoreResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score.json").write_text(
        json.dumps(score.to_dict(), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "objective.json").write_text(
        json.dumps(objective_payload(score), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "loss",
                "value",
                "weight",
                "loss_scale",
                "normalized_loss",
                "status",
            ],
        )
        writer.writeheader()
        for row in score.metrics:
            writer.writerow(row.to_dict())
    if score.metric_details:
        (output_dir / "metric_details.json").write_text(
            json.dumps(_metric_details_payload(score), ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if score.cd_profile:
        with (output_dir / "cd_profile.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "z_nm",
                    "sim_width_nm",
                    "target_width_nm",
                    "diff_nm",
                    "abs_diff_nm",
                    "left_diff_nm",
                    "right_diff_nm",
                    "left_abs_diff_nm",
                    "right_abs_diff_nm",
                    "edge_loss_nm",
                    "sim_left_nm",
                    "sim_right_nm",
                    "target_left_nm",
                    "target_right_nm",
                ],
            )
            writer.writeheader()
            for profile_row in score.cd_profile:
                writer.writerow(profile_row)
    if score.cd_profile_summary is not None:
        (output_dir / "cd_profile_summary.json").write_text(
            json.dumps(score.cd_profile_summary, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if score.profile_rows:
        with (output_dir / "profile.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "height_nm",
                    "sim_width_nm",
                    "target_width_nm",
                    "width_diff_nm",
                    "width_abs_diff_nm",
                    "sim_center_nm",
                    "target_center_nm",
                    "center_diff_nm",
                    "center_abs_diff_nm",
                    "sim_left_nm",
                    "sim_right_nm",
                    "target_left_nm",
                    "target_right_nm",
                    "left_diff_nm",
                    "right_diff_nm",
                    "edge_loss_nm",
                ],
            )
            writer.writeheader()
            for profile_row in score.profile_rows:
                writer.writerow(profile_row)
    if score.profile_summary is not None:
        (output_dir / "profile_summary.json").write_text(
            json.dumps(score.profile_summary, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if score.corner_summary is not None:
        (output_dir / "corner_summary.json").write_text(
            json.dumps(score.corner_summary, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
