"""Process-global device binding owned by the Newton adapter."""

from __future__ import annotations

from unisim.backend.process_device import bind_warp_process_device

from .dependencies import load_newton_dependencies

_BOUND_DEVICE: str | None = None


def bind_newton_process_device(device: str) -> str:
    """Select Newton's Warp device explicitly for the current process."""
    global _BOUND_DEVICE
    _BOUND_DEVICE = bind_warp_process_device(
        load_newton_dependencies, backend_label="newton", device=device
    )
    return _BOUND_DEVICE


def get_bound_newton_process_device() -> str | None:
    """Return the explicitly selected device without probing Warp defaults."""
    return _BOUND_DEVICE


__all__ = ["bind_newton_process_device", "get_bound_newton_process_device"]
