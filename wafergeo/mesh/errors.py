from __future__ import annotations


class MeshBackendUnavailableError(RuntimeError):
    """Raised when requested mesh backend is unavailable."""


class MeshOptionalDependencyError(MeshBackendUnavailableError):
    """Raised when optional dependency for mesh backend is missing."""


class ChannelMaterialMappingError(ValueError):
    """Raised when channel->material mapping cannot be resolved."""
