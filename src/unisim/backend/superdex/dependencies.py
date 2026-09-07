"""Optional, process-local loading of the supported SuperDex Python runtime."""

from __future__ import annotations

import importlib
import sys
from importlib import metadata
from typing import Any

from unisim.optional import OptionalDependencyError

_DISTRIBUTIONS = ("superdex-physics", "superdex-robotics")
_HINT = "Use Python 3.12 and install unisim-core[superdex] (SuperDex 1.0.0)."


class SuperDexDependencyError(OptionalDependencyError):
    """The optional SuperDex ABI or distribution is unavailable."""


def superdex_dependencies_available() -> bool:
    """Check package metadata without importing the native runtime."""
    if sys.version_info[:2] != (3, 12):
        return False
    try:
        return all(metadata.version(name) == "1.0.0" for name in _DISTRIBUTIONS)
    except metadata.PackageNotFoundError:
        return False


def load_superdex_dependencies() -> tuple[Any, Any]:
    """Load the precision-consistent Physics and Robotics public facades lazily."""
    if sys.version_info[:2] != (3, 12):
        raise SuperDexDependencyError(f"superdex requires CPython 3.12. {_HINT}")
    for name in _DISTRIBUTIONS:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SuperDexDependencyError(f"Missing {name}==1.0.0. {_HINT}") from exc
        if installed != "1.0.0":
            raise SuperDexDependencyError(
                f"superdex requires {name}==1.0.0; found {installed}. {_HINT}"
            )
    try:
        return (
            importlib.import_module("superdex.physics"),
            importlib.import_module("superdex.robotics"),
        )
    except (ImportError, OSError) as exc:
        raise SuperDexDependencyError(f"Could not load SuperDex: {exc}. {_HINT}") from exc
