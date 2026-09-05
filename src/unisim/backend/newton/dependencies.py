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

# Native ViewerGL rendering needs GUI libraries that must stay out of the
# pure-training environment; they ship in the separate ``newton-render``
# extra (Newton pins the same floors under its own ``examples`` extra).
_RENDER_REQUIREMENTS: dict[str, tuple[tuple[int, ...], tuple[int, ...] | None]] = {
    "pyglet": ((2, 1, 6), (3, 0, 0)),
    "imgui-bundle": ((1, 92, 0), None),
}
_RENDER_MODULES = {"pyglet": "pyglet", "imgui-bundle": "imgui_bundle"}
_RENDER_INSTALL_HINT = (
    "Install them with `uv sync --extra newton --extra newton-render`."
)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.replace("-", ".").split("."):
        if not token.isdigit():
            break
        parts.append(int(token))
    return tuple(parts)


def _render_dependency_problem(distribution: str) -> str | None:
    module_name = _RENDER_MODULES[distribution]
    if find_spec(module_name) is None:
        return f"{distribution} is not installed"
    minimum, maximum = _RENDER_REQUIREMENTS[distribution]
    try:
        installed = _version_tuple(metadata.version(distribution))
    except metadata.PackageNotFoundError:
        return f"{distribution} is not installed"
    if installed < minimum:
        return f"{distribution}>={'.'.join(str(p) for p in minimum)} is required, found {installed}"
    if maximum is not None and installed >= maximum:
        return f"{distribution}<{'.'.join(str(p) for p in maximum)} is required, found {installed}"
    return None


def newton_render_dependencies_available() -> bool:
    """Probe the native viewer stack without importing any GUI module."""
    return all(
        _render_dependency_problem(distribution) is None
        for distribution in _RENDER_REQUIREMENTS
    )


def require_newton_render_dependencies() -> None:
    """Fail closed unless Newton's native viewer stack is importable."""
    problems = [
        problem
        for distribution in _RENDER_REQUIREMENTS
        if (problem := _render_dependency_problem(distribution)) is not None
    ]
    if problems:
        raise NewtonDependencyError(
            "newton native rendering requires the viewer dependencies "
            "(pyglet>=2.1.6,<3, imgui-bundle>=1.92.0): "
            + "; ".join(problems)
            + f". {_RENDER_INSTALL_HINT}"
        )


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
    "newton_render_dependencies_available",
    "require_newton_render_dependencies",
]
