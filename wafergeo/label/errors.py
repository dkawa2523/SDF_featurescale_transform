from __future__ import annotations


class LabelNormalizeError(ValueError):
    """Base error for ingest/label normalization."""


class MissingLabelArrayError(LabelNormalizeError):
    """Raised when no candidate label source can be selected."""


class InvalidArrayShapeError(LabelNormalizeError):
    """Raised when array shape is inconsistent with grid dimensions."""


class UnknownMaterialIdError(LabelNormalizeError):
    """Raised when unknown material ids are detected and policy is strict."""


class PointToCellConversionError(LabelNormalizeError):
    """Raised when point-to-cell conversion is required but not allowed."""


class InvalidGridMetaError(LabelNormalizeError):
    """Raised when spacing/origin metadata is invalid."""
