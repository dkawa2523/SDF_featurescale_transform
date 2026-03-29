#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT_REQUIRED_KEYS = {"schema_version", "status", "messages"}
SCHEMA_REQUIRED_KEYS = {
    "correspondence_bench_manifest/v1": {
        "schema_version",
        "case_id",
        "spec_hash",
        "code_version",
        "status",
        "messages",
        "thresholds",
        "summary",
        "diagnosis",
        "rows",
        "outputs",
    },
    "vti_audit/v2": {
        "schema_version",
        "profile_id",
        "profile_hash",
        "status",
        "messages",
        "input_path",
        "input_hash",
        "outputs",
        "metrics",
        "postprocess",
    },
    "vti_preview/v2": {
        "schema_version",
        "profile_id",
        "profile_hash",
        "status",
        "messages",
        "input_path",
        "input_hash",
        "created_at",
        "generator_version",
        "outputs",
        "audit_manifest_path",
        "sdf",
        "postprocess",
    },
    "report_manifest/v1": {
        "schema_version",
        "report_id",
        "created_at",
        "spec_hash",
        "index_hash",
        "code_version",
        "tables",
        "figures",
        "status",
        "messages",
        "extra",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PR5 machine-check manifests for status/message consistency."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a manifest JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expected-fallback-warning-count",
        type=int,
        default=None,
        help="Expected total number of fallback warning messages across all checked manifests.",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def _status_messages_consistent(
    node: dict[str, Any],
    *,
    require_messages: bool,
) -> tuple[bool, str | None]:
    if "status" not in node:
        return False, "missing status"
    status = node["status"]
    if status not in {"OK", "WARN", "FAIL"}:
        return False, f"invalid status: {status!r}"

    if "messages" not in node:
        if require_messages:
            return False, "missing messages"
        return True, None

    messages = node["messages"]
    if not isinstance(messages, list) or any(not isinstance(message, str) for message in messages):
        return False, "messages must be a list[str]"
    if status == "OK" and messages:
        return False, "status OK must not carry messages"
    if status in {"WARN", "FAIL"} and not messages:
        return False, f"status {status} must carry at least one message"
    return True, None


def _iter_string_messages(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, str):
                    yield message
        for child in value.values():
            yield from _iter_string_messages(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_messages(item)


def _count_fallback_warnings(manifest: dict[str, Any]) -> int:
    return sum(1 for message in _iter_string_messages(manifest) if "fallback" in message.lower())


def _validate_nested_status_blocks(node: Any, *, path: str, errors: list[str]) -> None:
    if isinstance(node, dict):
        has_status = "status" in node
        has_messages = "messages" in node
        if has_status or has_messages:
            ok, error = _status_messages_consistent(node, require_messages=False)
            if not ok and error is not None:
                errors.append(f"{path}: {error}")
        for key, child in node.items():
            if isinstance(child, (dict, list)):
                _validate_nested_status_blocks(child, path=f"{path}.{key}", errors=errors)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _validate_nested_status_blocks(child, path=f"{path}[{index}]", errors=errors)


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_manifest(path)
    schema_version = str(manifest.get("schema_version", ""))
    required_keys = set(ROOT_REQUIRED_KEYS)
    required_keys.update(SCHEMA_REQUIRED_KEYS.get(schema_version, set()))
    missing_keys = sorted(key for key in required_keys if key not in manifest)
    errors: list[str] = []
    if missing_keys:
        errors.append(f"missing required keys: {', '.join(missing_keys)}")

    ok, error = _status_messages_consistent(manifest, require_messages=True)
    if not ok and error is not None:
        errors.append(error)

    _validate_nested_status_blocks(manifest, path="manifest", errors=errors)

    fallback_warning_count = _count_fallback_warnings(manifest)
    return {
        "path": str(path),
        "schema_version": schema_version,
        "profile_id": str(manifest.get("profile_id", "")),
        "status": str(manifest.get("status", "")),
        "missing_keys": missing_keys,
        "fallback_warning_count": fallback_warning_count,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    results = [validate_manifest(Path(path)) for path in args.manifest]
    total_fallback_warning_count = sum(int(result["fallback_warning_count"]) for result in results)
    errors: list[str] = []
    for result in results:
        errors.extend(str(error) for error in result["errors"])

    if (
        args.expected_fallback_warning_count is not None
        and total_fallback_warning_count != args.expected_fallback_warning_count
    ):
        errors.append(
            "fallback warning count mismatch: "
            f"expected={args.expected_fallback_warning_count} actual={total_fallback_warning_count}"
        )

    status = "OK" if not errors else "FAIL"
    payload = {
        "status": status,
        "fallback_warning_count": total_fallback_warning_count,
        "manifests": results,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
