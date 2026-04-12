from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import wafergeo
from wafergeo.core.hashing import sha256_file


def resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (base_dir / value).resolve()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_run_info(
    *,
    config_path: Path,
    output_dir: Path,
    task: str,
    inputs: dict[str, str],
) -> None:
    run_dir = output_dir / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "used_config.yaml")
    input_hashes: dict[str, str] = {}
    for name, value in inputs.items():
        path = Path(value)
        if path.exists() and path.is_file():
            input_hashes[name] = sha256_file(path)
    write_json(
        run_dir / "run_info.json",
        {
            "task": task,
            "created_at": datetime.now(UTC).isoformat(),
            "wafergeo_version": wafergeo.__version__,
            "config_path": str(config_path),
            "inputs": dict(inputs),
            "input_hashes": input_hashes,
        },
    )
