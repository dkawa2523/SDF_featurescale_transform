#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VTI correspondence audit (raw vs converted).")
    parser.add_argument("--vti", required=True, help="Input VTI path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--outside-id",
        type=int,
        default=2,
        help="Outside material id",
    )
    return parser.parse_args()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from wafergeo.pipelines import run_vti_correspondence_audit

    args = parse_args()
    manifest = run_vti_correspondence_audit(
        vti_path=Path(args.vti),
        output_dir=Path(args.out),
        outside_material_id=args.outside_id,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
