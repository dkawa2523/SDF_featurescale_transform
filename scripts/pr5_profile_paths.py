#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the core/full install and test paths used by PR5 evaluation automation."
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full", "all"),
        default="all",
        help="Profile to render",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format",
    )
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.pr5_profiles import PROFILE_SPECS, ProfileSpec, render_command

    args = parse_args()
    if args.profile == "all":
        selected: dict[str, ProfileSpec] = dict(PROFILE_SPECS)
    else:
        selected = {args.profile: PROFILE_SPECS[args.profile]}

    payload = {
        name: {
            "install": render_command(spec.install_argv),
            "test": render_command(spec.test_argv),
        }
        for name, spec in selected.items()
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, commands in payload.items():
            print(f"[{name}]")
            print(f"install: {commands['install']}")
            print(f"test: {commands['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

