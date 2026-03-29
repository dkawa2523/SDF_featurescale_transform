from __future__ import annotations


class ReportError(Exception):
    """Base error for reports subsystem."""


class ReportDependencyError(ReportError):
    """Raised when optional runtime dependency is missing."""


class ReportRegistryError(ReportError):
    """Raised when extractor/plot plugin lookup fails."""
