"""Lazy dependency boundary for the CUDA-resident ``motrix_gpu`` backend."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from unisim.optional import OptionalDependencyError


class MotrixGpuDependencyError(OptionalDependencyError):
    """Raised with an actionable install hint when the GPU runtime is absent."""


@dataclass(frozen=True)
class MotrixGpuDependencies:
    """Modules required by the device-resident ``motrix_gpu`` implementation."""

    mgpu: Any
    warp: Any


_PINNED_DISTRIBUTIONS = {
    "motrixsim-core": "1.0.0",
    "motrixsim-gpu": "1.0.0",
    "warp-lang": "1.17.0",
}
_INSTALL_HINT = "Install `unisim-core[motrix-gpu]` and a CUDA-enabled PyTorch build."
_LABEL = "motrix_gpu backend"


def motrix_gpu_available() -> bool:
    """Check the supported platform and package metadata without importing an SDK."""
    return _dependency_problem() is None


def _dependency_problem() -> str | None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return "requires Linux x86-64 with NVIDIA CUDA"
    for distribution, expected in _PINNED_DISTRIBUTIONS.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return f"requires {distribution}=={expected}"
        if installed.split("+", 1)[0] != expected:
            return f"requires {distribution}=={expected}, found {installed}"
    return None


def _import_module(module_name: str, package: str) -> Any:
    """Import one runtime module, normalizing every failure to one diagnostic."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MotrixGpuDependencyError(
            f"{_LABEL} requires optional dependency '{package}'. {_INSTALL_HINT}"
        ) from exc
    except ImportError as exc:
        # CUDA/Warp initialization and dependency conflicts surface here.
        raise MotrixGpuDependencyError(
            f"{_LABEL} could not import its optional runtime: {exc}. {_INSTALL_HINT}"
        ) from exc


def load_motrix_gpu_dependencies() -> MotrixGpuDependencies:
    """Import the optional GPU runtime only when a backend instance is built."""
    problem = _dependency_problem()
    if problem is not None:
        raise MotrixGpuDependencyError(f"{_LABEL} {problem}. {_INSTALL_HINT}")
    mgpu = _import_module("motrixsim.gpu", "motrixsim-gpu")
    warp = _import_module("warp", "warp-lang")
    return MotrixGpuDependencies(mgpu=mgpu, warp=warp)


__all__ = [
    "MotrixGpuDependencies",
    "MotrixGpuDependencyError",
    "load_motrix_gpu_dependencies",
    "motrix_gpu_available",
]
