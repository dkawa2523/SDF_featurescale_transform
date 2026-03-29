#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PR5 evaluation profile.")
    parser.add_argument("--profile", choices=("core", "full"), required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.pr5_profiles import get_profile_spec, render_command

    args = parse_args()
    spec = get_profile_spec(args.profile)
    print(f"profile={spec.name}")
    print(f"install={render_command(spec.install_argv)}")
    print(f"test={render_command(spec.test_argv)}")

    if args.dry_run:
        return 0

    subprocess.run(spec.install_argv, check=True)
    subprocess.run(spec.test_argv, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

