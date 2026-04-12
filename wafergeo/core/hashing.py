from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _normalize_for_json(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for canonical JSON hashing: {type(value)!r}")


def canonical_json_dumps(obj: Any) -> str:
    normalized = _normalize_for_json(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def hash_config(obj: Any) -> str:
    return sha256_bytes(canonical_json_dumps(obj).encode("utf-8"))
