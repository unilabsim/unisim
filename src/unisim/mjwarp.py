"""MJWarp adapter with an explicit CUDA/runtime dependency boundary."""
from __future__ import annotations

from .optional import RuntimeBackend


class MJWarpBackend(RuntimeBackend):
    backend_type = "mjwarp"
    module_names = ("mujoco_warp", "warp")
    install_hint = "Install `unisim-core[mjwarp]` on a CUDA-capable host."


MjwarpBackend = MJWarpBackend
__all__ = ["MJWarpBackend", "MjwarpBackend"]
