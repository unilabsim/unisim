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
    if backend_type == "drake":
        from .drake import DrakeBackend

        return DrakeBackend(**kwargs)
    if backend_type == "mjwarp":
        from .mjwarp import MJWarpBackend

        return MJWarpBackend(**kwargs)
    if backend_type == "genesis":
        from .genesis import GenesisBackend

        return GenesisBackend(**kwargs)
    if backend_type == "isaacgym":
        from .isaacgym import IsaacGymBackend

        return IsaacGymBackend(**kwargs)
    if backend_type == "isaacsim":
        from .isaacsim import IsaacSimBackend

        return IsaacSimBackend(**kwargs)
    try:
        spec = adapter_spec(backend_type)
    except KeyError:
        raise ValueError(f"unknown UniSim backend: {backend_type!r}") from None
    # Every backend in the manifest has a concrete public adapter.  Optional
    # SDK/worker availability is diagnosed by that adapter at construction;
    # this branch is retained only as a guard for future manifest mistakes.
    if spec.status != "available":
        raise BackendError(f"backend '{backend_type}' is not currently available")
    raise ValueError(f"unknown UniSim backend: {backend_type!r}")
