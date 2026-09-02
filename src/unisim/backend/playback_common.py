"""Shared playback helper utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


class _ImageIOProxy:
    """Lazy imageio namespace kept out of the NumPy-only import path."""

    def __getattr__(self, name: str) -> Any:
        import imageio.v2 as imageio_v2

        return getattr(imageio_v2, name)


imageio = _ImageIOProxy()


def env_cfg_value(env: Any, name: str, default: Any) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def write_playback_video(path: str, frames: list[np.ndarray], *, fps: int) -> None:
    """Write playback frames with the repository-managed imageio stack."""
    imageio.mimsave(path, frames, fps=fps)
