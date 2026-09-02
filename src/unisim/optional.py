"""Normalized diagnostics for optional engine runtimes."""

from __future__ import annotations

from .contract import BackendError


class OptionalDependencyError(BackendError, ImportError):
    """Raised when an adapter's optional runtime cannot be loaded."""


__all__ = ["OptionalDependencyError"]
