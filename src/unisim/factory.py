"""Lazy backend factory for optional engine adapters."""

from __future__ import annotations

from typing import Any

from .adapters import adapter_spec
from .contract import BackendError, SimBackend


def create_backend(backend_type: str, **kwargs: Any) -> SimBackend:
    """Construct an optional backend without importing engine SDKs eagerly."""
    if backend_type == "fake":
        from .fake import FakeBackend

        return FakeBackend(**kwargs)
    if backend_type == "mujoco":
        from .mujoco import MuJoCoBackend

        return MuJoCoBackend(**kwargs)
    if backend_type == "motrix":
        from .motrix import MotrixBackend

        return MotrixBackend(**kwargs)
    try:
        spec = adapter_spec(backend_type)
    except KeyError:
        raise ValueError(f"unknown UniSim backend: {backend_type!r}") from None
    if spec.status == "planned":
        raise BackendError(
            f"backend '{backend_type}' is declared in the UniSim roadmap but its adapter "
            "has not been migrated yet"
        )
    raise ValueError(f"unknown UniSim backend: {backend_type!r}")
