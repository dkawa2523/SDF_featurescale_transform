from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from wafergeo.core.hashing import make_artifact_id
from wafergeo.core.meta import Meta


class ArtifactStore(Protocol):
    def exists(self, artifact_id: str) -> bool:
        ...

    def read_meta(self, artifact_id: str) -> Meta:
        ...

    def write(self, artifact_type: str, payload: object, meta: Meta) -> str:
        ...

    def load(self, artifact_id: str) -> object:
        ...


class LocalDiskArtifactStore:
    """Local artifact store with JSON metadata and NPY arrays.

    Layout:
    artifacts/<artifact_type>/<artifact_id>/{manifest.json,meta.json,payload.json,payload/*.npy}
    """

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, artifact_id: str) -> bool:
        return self._resolve_artifact_dir(artifact_id) is not None

    def read_meta(self, artifact_id: str) -> Meta:
        artifact_dir = self._require_artifact_dir(artifact_id)
        meta_path = artifact_dir / "meta.json"
        raw = self._read_json(meta_path)
        return Meta.from_dict(raw)

    def write(self, artifact_type: str, payload: object, meta: Meta) -> str:
        if not artifact_type:
            raise ValueError("artifact_type must be non-empty")

        artifact_id = make_artifact_id(
            input_hash=meta.input_hash,
            profile_id=meta.profile_id,
            config_hash=meta.config_hash,
            generator_version=meta.generator_version,
        )
        artifact_type_dir = self.root / artifact_type
        artifact_type_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = artifact_type_dir / artifact_id
        lock_path = artifact_type_dir / f".{artifact_id}.lock"

        with self._artifact_lock(lock_path):
            if (artifact_dir / "manifest.json").exists():
                return artifact_id
            if artifact_dir.exists():
                raise FileExistsError(
                    f"artifact directory already exists without manifest: {artifact_dir}"
                )

            staging_dir = artifact_type_dir / (
                f".tmp_{artifact_id}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            )
            payload_dir = staging_dir / "payload"
            payload_dir.mkdir(parents=True, exist_ok=False)
            try:
                counter = [0]
                encoded_payload = self._encode_payload(payload, payload_dir, counter)

                self._write_json(staging_dir / "payload.json", encoded_payload)
                self._write_json(staging_dir / "meta.json", meta.to_dict())
                self._write_json(
                    staging_dir / "manifest.json",
                    {
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type,
                        "payload_format": "json+npy",
                        "payload_format_version": 1,
                        "schema_version": meta.schema_version,
                        "profile_id": meta.profile_id,
                        "config_hash": meta.config_hash,
                        "generator_version": meta.generator_version,
                        "input_hash": meta.input_hash,
                    },
                )
                os.replace(staging_dir, artifact_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
        return artifact_id

    def load(self, artifact_id: str) -> object:
        artifact_dir = self._require_artifact_dir(artifact_id)
        payload = self._read_json(artifact_dir / "payload.json")
        return self._decode_payload(payload, artifact_dir)

    def _resolve_artifact_dir(self, artifact_id: str) -> Path | None:
        if not self.root.exists():
            return None
        candidates: list[Path] = []
        for artifact_type_dir in self.root.iterdir():
            if not artifact_type_dir.is_dir():
                continue
            candidate = artifact_type_dir / artifact_id
            if candidate.is_dir() and (candidate / "manifest.json").exists():
                candidates.append(candidate)
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(f"artifact_id is ambiguous across artifact types: {artifact_id}")
        return candidates[0]

    def _require_artifact_dir(self, artifact_id: str) -> Path:
        artifact_dir = self._resolve_artifact_dir(artifact_id)
        if artifact_dir is None:
            raise FileNotFoundError(f"artifact_id not found: {artifact_id}")
        return artifact_dir

    def _encode_payload(self, value: Any, payload_dir: Path, counter: list[int]) -> Any:
        if is_dataclass(value):
            return {
                field.name: self._encode_payload(getattr(value, field.name), payload_dir, counter)
                for field in fields(value)
            }
        if isinstance(value, np.ndarray):
            filename = f"arr_{counter[0]:06d}.npy"
            counter[0] += 1
            np.save(payload_dir / filename, value, allow_pickle=False)
            return {"__kind__": "ndarray", "path": f"payload/{filename}"}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._encode_payload(v, payload_dir, counter) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._encode_payload(v, payload_dir, counter) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(f"Unsupported payload type: {type(value)!r}")

    def _decode_payload(self, value: Any, artifact_dir: Path) -> Any:
        if isinstance(value, dict):
            kind = value.get("__kind__")
            if kind == "ndarray":
                rel_path = value["path"]
                return np.load(artifact_dir / rel_path, allow_pickle=False)
            return {k: self._decode_payload(v, artifact_dir) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decode_payload(v, artifact_dir) for v in value]
        return value

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @contextmanager
    def _artifact_lock(
        self,
        lock_path: Path,
        *,
        timeout_sec: float = 10.0,
        poll_sec: float = 0.05,
    ):
        deadline = time.monotonic() + timeout_sec
        lock_fd: int | None = None
        while True:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, b"lock")
                break
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timeout acquiring artifact lock: {lock_path}"
                    ) from exc
                time.sleep(poll_sec)
        try:
            yield
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
