"""Lazy backend factory for optional engine adapters."""

from __future__ import annotations

from typing import Any

from .contract import SimBackend


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
    raise ValueError(f"unknown UniSim backend: {backend_type!r}")
