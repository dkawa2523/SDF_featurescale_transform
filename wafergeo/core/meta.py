from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Meta:
    """Common metadata carried by derived geometry objects."""

    schema_version: str
    profile_id: str
    config_hash: str
    generator_version: str
    git_commit: str
    input_hash: str
    created_at: str
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "config_hash": self.config_hash,
            "generator_version": self.generator_version,
            "git_commit": self.git_commit,
            "input_hash": self.input_hash,
            "created_at": self.created_at,
        }
        for key, value in required.items():
            if not value:
                raise ValueError(f"{key} must be non-empty")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"created_at must be ISO-8601, got {self.created_at}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "config_hash": self.config_hash,
            "generator_version": self.generator_version,
            "git_commit": self.git_commit,
            "input_hash": self.input_hash,
            "created_at": self.created_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Meta:
        return cls(
            schema_version=str(raw["schema_version"]),
            profile_id=str(raw["profile_id"]),
            config_hash=str(raw["config_hash"]),
            generator_version=str(raw["generator_version"]),
            git_commit=str(raw["git_commit"]),
            input_hash=str(raw["input_hash"]),
            created_at=str(raw["created_at"]),
            extra={str(k): str(v) for k, v in dict(raw.get("extra", {})).items()},
        )
