"""Process-global runtime setup owned by the ``mjwarp`` backend."""

from __future__ import annotations

from unisim.backend.process_device import bind_warp_process_device

from .dependencies import load_mjwarp_dependencies


def bind_mjwarp_process_device(device: str) -> str:
    """Make one CUDA device Warp's default/current device for this process."""
    return bind_warp_process_device(
        load_mjwarp_dependencies, backend_label="mjwarp", device=device
    )
