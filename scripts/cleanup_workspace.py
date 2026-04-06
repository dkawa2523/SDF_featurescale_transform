#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SAFE_ROOT_DIRS = (
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "wafergeo.egg-info",
)
SAFE_ROOT_GLOBS = ("tmp_*",)


def _collect_targets(root: Path) -> list[Path]:
    targets: set[Path] = set()

    for name in SAFE_ROOT_DIRS:
        path = root / name
        if path.exists():
            targets.add(path)

    for pattern in SAFE_ROOT_GLOBS:
        for path in root.glob(pattern):
            targets.add(path)

    for path in root.rglob("__pycache__"):
        targets.add(path)
    for path in root.rglob("*.pyc"):
        targets.add(path)

    return sorted(targets)


def _is_within_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean safe, reproducible local artifacts from this workspace."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete targets. Default is dry-run.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = _collect_targets(root)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {len(targets)} target(s)")

    for path in targets:
        if not _is_within_root(root, path):
            raise RuntimeError(f"Refusing out-of-root path: {path}")
        rel = path.resolve().relative_to(root.resolve())
        print(rel)
        if args.apply:
            _remove_path(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
