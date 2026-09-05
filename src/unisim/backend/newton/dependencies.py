"""Lazy optional-runtime boundary for the Newton backend."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Any

from unisim.optional import OptionalDependencyError


class NewtonDependencyError(OptionalDependencyError):
    """Raised when the pinned Newton runtime cannot be loaded."""


@dataclass(frozen=True)
class NewtonDependencies:
    """Public modules used by the Newton adapter."""

    newton: Any
    warp: Any
    mujoco: Any
    mujoco_warp: Any


PINNED_DISTRIBUTIONS: dict[str, str] = {
    "newton": "1.5.1",
    "mujoco-warp": "3.11.0",
    "mujoco": "3.11.0",
    "warp-lang": "1.16.0",
}
_MODULES = {
    "newton": "newton",
    "mujoco-warp": "mujoco_warp",
    "mujoco": "mujoco",
    "warp-lang": "warp",
}
_INSTALL_HINT = "Install the isolated runtime with `uv sync --extra newton`."


def newton_dependencies_available() -> bool:
    """Probe the Newton stack without importing any engine module."""
    return all(find_spec(module_name) is not None for module_name in _MODULES.values())


def load_newton_dependencies() -> NewtonDependencies:
    """Validate exact versions, then import the public Newton runtime modules."""
    for distribution, expected in PINNED_DISTRIBUTIONS.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise NewtonDependencyError(
                f"newton backend requires {distribution}=={expected}. {_INSTALL_HINT}"
            ) from exc
        if installed != expected:
            raise NewtonDependencyError(
                f"newton backend requires {distribution}=={expected}, found {installed}. "
                f"{_INSTALL_HINT} Do not combine the newton extra with mujoco or mjwarp."
            )
    modules: dict[str, Any] = {}
    try:
        for distribution, module_name in _MODULES.items():
            modules[distribution] = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise NewtonDependencyError(
            f"newton backend could not import its optional runtime: {exc}. {_INSTALL_HINT}"
        ) from exc
    return NewtonDependencies(
        newton=modules["newton"],
        warp=modules["warp-lang"],
        mujoco=modules["mujoco"],
        mujoco_warp=modules["mujoco-warp"],
    )


__all__ = [
    "NewtonDependencies",
    "NewtonDependencyError",
    "PINNED_DISTRIBUTIONS",
    "load_newton_dependencies",
    "newton_dependencies_available",
]
