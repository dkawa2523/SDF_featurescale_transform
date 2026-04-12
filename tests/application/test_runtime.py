from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from wafergeo.application.runtime.cli import main
from wafergeo.application.runtime.runner import PUBLIC_PIPELINES, run_pipeline_from_config
from wafergeo.compare.schema import (
    load_batch_compare_spec_yaml,
    load_compare_spec_yaml,
    load_transform_spec_yaml,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_runtime_cli_prints_result_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "wafergeo.application.runtime.cli.run_pipeline_from_config",
        lambda pipeline, config: {
            "pipeline": pipeline,
            "config": config,
            "status": "OK",
        },
    )

    exit_code = main(["run", "compare", "--config", "compare.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"pipeline": "compare"' in captured.out
    assert '"status": "OK"' in captured.out


def test_runtime_dispatch_rejects_old_pipelines() -> None:
    with pytest.raises(ValueError, match="unsupported pipeline"):
        run_pipeline_from_config("old-task", "old.yaml")


def test_public_pipelines_are_only_simple_tasks() -> None:
    assert PUBLIC_PIPELINES == ("transform", "compare", "batch-compare")


def test_bundled_configs_and_examples_load() -> None:
    root = _repo_root()

    transform_example = load_transform_spec_yaml(
        root / "configs" / "examples" / "transform.simple.yaml"
    )
    compare_example = load_compare_spec_yaml(root / "configs" / "examples" / "compare.simple.yaml")
    batch_example = load_batch_compare_spec_yaml(
        root / "configs" / "examples" / "batch-compare.simple.yaml"
    )
    realdata_eval = load_batch_compare_spec_yaml(
        root / "configs" / "runs" / "dataset_t08_vs_run0010.yaml"
    )

    assert transform_example.task == "transform"
    assert compare_example.task == "compare"
    assert batch_example.task == "batch-compare"
    assert realdata_eval.task == "batch-compare"


def test_bundled_example_input_paths_exist() -> None:
    root = _repo_root()
    config_paths = [
        root / "configs" / "examples" / "transform.simple.yaml",
        root / "configs" / "examples" / "compare.simple.yaml",
        root / "configs" / "examples" / "batch-compare.simple.yaml",
    ]

    for config_path in config_paths:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        input_raw = raw["input"]
        if "index" in input_raw:
            assert (config_path.parent / input_raw["index"]).resolve().exists()
            continue
        assert (config_path.parent / input_raw["simulation"]["path"]).resolve().exists()
        if "target" in input_raw:
            assert (config_path.parent / input_raw["target"]["path"]).resolve().exists()


def test_realdata_eval_config_paths_exist() -> None:
    root = _repo_root()
    config_path = root / "configs" / "runs" / "dataset_t08_vs_run0010.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    index_path = (config_path.parent / raw["input"]["index"]).resolve()

    assert index_path.exists()
    with index_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    for row in rows:
        assert (index_path.parent / row["simulation_path"]).resolve().exists()
        assert (index_path.parent / row["target_path"]).resolve().exists()


def test_examples_dir_contains_only_public_examples() -> None:
    root = _repo_root()
    names = sorted(path.name for path in (root / "configs" / "examples").iterdir())
    assert names == [
        "README.md",
        "batch-compare.simple.yaml",
        "compare.simple.yaml",
        "transform.simple.yaml",
    ]


def test_configs_dir_contains_only_public_config_groups() -> None:
    root = _repo_root()
    names = sorted(path.name for path in (root / "configs").iterdir())
    assert names == ["examples", "runs"]


def test_runs_dir_contains_only_realdata_eval_inputs() -> None:
    root = _repo_root()
    names = sorted(path.name for path in (root / "configs" / "runs").iterdir())
    assert names == [
        "README.md",
        "dataset_t08_vs_run0010.yaml",
        "dataset_t08_vs_run0010_pairs.csv",
    ]
