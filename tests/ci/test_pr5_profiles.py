from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_profile_paths_exposes_core_and_full_commands() -> None:
    script = Path("scripts/pr5_profile_paths.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert set(payload) == {"core", "full"}
    assert "python" in payload["core"]["install"]
    assert "pytest" in payload["core"]["test"]
    assert "pip install" in payload["full"]["install"]
    assert "pytest -q" in payload["full"]["test"]


def test_profile_runner_dry_run_prints_selected_profile() -> None:
    script = Path("scripts/run_pr5_profile.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--profile", "core", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "profile=core" in proc.stdout
    assert "install=" in proc.stdout
    assert "test=" in proc.stdout

