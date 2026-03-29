#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run correspondence benchmark for VTI <-> SDF/Mesh mismatch diagnosis."
    )
    parser.add_argument("--spec", required=True, help="Benchmark YAML spec path")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from wafergeo.bench.correspondence import load_benchmark_spec_yaml, run_correspondence_benchmark

    args = parse_args()
    spec = load_benchmark_spec_yaml(args.spec)
    manifest = run_correspondence_benchmark(spec, args.out)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
