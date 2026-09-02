"""Shared playback helper utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def env_cfg_value(env: Any, name: str, default: Any) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def write_playback_video(path: str, frames: list[np.ndarray], *, fps: int) -> None:
    """Write playback frames with the repository-managed imageio stack."""
    # Keep imageio on the cold playback path.  Importing a backend contract or
    # probing optional adapters must work in the minimal NumPy-only wheel.
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)
