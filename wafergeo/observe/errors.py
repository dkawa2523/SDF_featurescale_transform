from __future__ import annotations


class ObserverError(RuntimeError):
    """Base observer-layer error."""


class ObserverOptionalDependencyError(ObserverError):
    """Raised when an optional dependency required by observer code is missing."""


class ObserverSpecValidationError(ValueError):
    """Raised when observer specification data is invalid."""
