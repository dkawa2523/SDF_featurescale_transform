from __future__ import annotations


class InvalidMuError(ValueError):
    """Raised when TSDF truncation width is invalid."""


class InvalidSpacingError(ValueError):
    """Raised when grid spacing is invalid for EDT."""


class EDTBackendUnavailableError(RuntimeError):
    """Raised when the requested EDT backend is unavailable."""


class OptionalDependencyUnavailableError(EDTBackendUnavailableError):
    """Raised when an optional backend dependency is missing."""

    def __init__(self, engine_name: str, dependency: str, install_hint: str) -> None:
        message = (
            f"Engine '{engine_name}' requires optional dependency '{dependency}'. "
            f"Install with: {install_hint}"
        )
        super().__init__(message)


class EDTComputationError(RuntimeError):
    """Raised when the EDT backend fails during distance computation."""


class ShapeMismatchError(ValueError):
    """Raised when ROI placement or channel shapes are inconsistent."""
