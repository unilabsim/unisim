"""Process-global device binding owned by the Newton adapter."""

from __future__ import annotations

from .dependencies import load_newton_dependencies

_BOUND_DEVICE: str | None = None


def bind_newton_process_device(device: str) -> str:
    """Select Newton's Warp device explicitly for the current process."""
    global _BOUND_DEVICE
    dependencies = load_newton_dependencies()
    dependencies.warp.set_device(device)
    selected = dependencies.warp.get_device()
    if not bool(selected.is_cuda):
        raise RuntimeError(
            "newton backend requires an active CUDA Warp device; "
            f"resolved {selected!s} from {device!r}"
        )
    _BOUND_DEVICE = str(selected)
    return _BOUND_DEVICE


def get_bound_newton_process_device() -> str | None:
    """Return the explicitly selected device without probing Warp defaults."""
    return _BOUND_DEVICE


__all__ = ["bind_newton_process_device", "get_bound_newton_process_device"]
