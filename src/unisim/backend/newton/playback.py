"""Cold-path native ViewerGL playback bridge for the Newton adapter.

Rendering happens only on the play/eval path; the training hot path never
touches this module.  Headless offscreen rendering still needs a GL context:
on Linux hosts without a display server use EGL (``PYOPENGL_PLATFORM=egl``);
under Wayland Newton forces GLX for its pyglet window.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from os import PathLike
from typing import Any, TypeVar

import numpy as np

from unisim.backend.playback_common import env_cfg_value, write_playback_video

ObsT = TypeVar("ObsT")

NEWTON_NATIVE_RENDERER = "newton-viewer-gl"
MUJOCO_SNAPSHOT_RENDERER = "mujoco-snapshot"

# Native playback renders a bounded number of env worlds; the offline MuJoCo
# snapshot path stays the way to visualize larger batches.
MAX_RENDER_WORLDS = 16


def display_available() -> bool:
    """Return whether a display is reachable for the interactive viewer."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_newton_native_playback(
    *,
    backend: Any,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    headless: bool,
    record_video: bool,
    camera_kwargs: dict[str, Any] | None,
) -> str | None:
    """Drive native ViewerGL playback for an env wrapper (genesis semantics)."""
    if record_video and not headless:
        raise ValueError("newton native video recording requires headless=true.")

    if headless or record_video:
        if num_steps is None:
            raise ValueError("newton captured playback requires a finite num_steps value.")
        if record_video and output_video is None:
            raise ValueError("newton native video recording requires an output_video path.")

        backend.init_renderer(
            spacing=float(render_spacing) if render_spacing is not None else 1.0,
            headless=headless,
            capture=True,
            width=1280,
            height=720,
            camera_kwargs=dict(camera_kwargs or {}),
        )

        obs = initialize()
        frames: list[np.ndarray] | None = [] if record_video else None
        for _ in range(num_steps):
            obs = step(obs)
            frame = np.asarray(backend.capture_video_frame(), dtype=np.uint8)
            if frames is not None:
                frames.append(frame.copy())

        if not record_video:
            return None

        assert output_video is not None
        assert frames is not None
        ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
        write_playback_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
        return str(output_video)

    backend.init_renderer(
        spacing=float(render_spacing) if render_spacing is not None else 1.0,
        headless=False,
        camera_kwargs=dict(camera_kwargs or {}),
    )
    obs = initialize()
    last_render_time = time.perf_counter()
    render_dt = 1.0 / 60.0
    steps_run = 0

    while num_steps is None or steps_run < num_steps:
        obs = step(obs)
        current_time = time.perf_counter()
        elapsed = current_time - last_render_time
        if elapsed < render_dt:
            time.sleep(render_dt - elapsed)
        last_render_time = time.perf_counter()
        backend.render()
        steps_run += 1
    return None


__all__ = [
    "MUJOCO_SNAPSHOT_RENDERER",
    "MAX_RENDER_WORLDS",
    "NEWTON_NATIVE_RENDERER",
    "display_available",
    "run_newton_native_playback",
]
