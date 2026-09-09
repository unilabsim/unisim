"""Optional, process-local loading of the supported SuperDex Python runtime."""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from unisim.optional import OptionalDependencyError

# Temporary: the unilabsim superdex-uni wheels carry the native batch executor
# until the upstream project_superdex PR merges and publishes equivalent
# superdex-physics/superdex-robotics wheels; switch these names back then.
_DISTRIBUTIONS = ("superdex-physics-uni", "superdex-robotics-uni")
_SUPPORTED_PYTHON = ((3, 12), (3, 13))
_SUPPORTED_PYTHON_TEXT = "3.12 or 3.13"
_HINT = (
    f"Use Python {_SUPPORTED_PYTHON_TEXT} and install unisim-core[superdex] "
    "(SuperDex 1.0.0, superdex-uni build)."
)


class SuperDexDependencyError(OptionalDependencyError):
    """The optional SuperDex ABI or distribution is unavailable."""


def _prioritize_local_native_extension() -> None:
    """Restore a local pybind directory after a spawned facade mutated sys.path."""
    for entry in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if not entry:
            continue
        path = Path(entry)
        if path.is_dir() and any(path.glob("mochi_physics*.so")):
            entry_text = str(path)
            if entry_text in sys.path:
                sys.path.remove(entry_text)
            sys.path.insert(0, entry_text)


def superdex_dependencies_available() -> bool:
    """Check package metadata without importing the native runtime."""
    if sys.version_info[:2] not in _SUPPORTED_PYTHON:
        return False
    try:
        return all(metadata.version(name) == "1.0.0" for name in _DISTRIBUTIONS)
    except metadata.PackageNotFoundError:
        return False


def load_superdex_dependencies() -> tuple[Any, Any]:
    """Load the precision-consistent Physics and Robotics public facades lazily."""
    if sys.version_info[:2] not in _SUPPORTED_PYTHON:
        raise SuperDexDependencyError(
            f"superdex requires CPython {_SUPPORTED_PYTHON_TEXT}. {_HINT}"
        )
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
        # A source-built SceneBatchExecutor is supplied through PYTHONPATH for
        # local integration. Preload it before the public facade inserts its
        # packaged `_native` directory ahead of Python's normal search path;
        # spawn collectors then inherit the same selected extension.
        try:
            _prioritize_local_native_extension()
            importlib.import_module("mochi_physics")
        except ImportError:
            pass
        return (
            importlib.import_module("superdex.physics"),
            importlib.import_module("superdex.robotics"),
        )
    except (ImportError, OSError) as exc:
        raise SuperDexDependencyError(f"Could not load SuperDex: {exc}. {_HINT}") from exc
