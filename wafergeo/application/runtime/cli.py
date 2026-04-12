from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from wafergeo.application.runtime.runner import PUBLIC_PIPELINES, run_pipeline_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wafergeo",
        description="Run wafergeo feature transform and shape comparison tasks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a pipeline from a YAML config.")
    run_parser.add_argument(
        "pipeline",
        choices=PUBLIC_PIPELINES,
        help="Pipeline name.",
    )
    run_parser.add_argument("--config", required=True, help="Run spec YAML path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run":
        result = run_pipeline_from_config(
            args.pipeline,
            args.config,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
