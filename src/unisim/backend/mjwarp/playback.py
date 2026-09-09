"""Cold-path MuJoCo offline playback bridge for ``mjwarp``.

The implementation lives in :mod:`unisim.backend.playback_common` so other
snapshot-based adapters (Newton) share one offline MuJoCo pipeline; the
wrappers below keep the historical mjwarp names, signatures, and messages.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from os import PathLike
from typing import Any, TypeVar

import numpy as np

from unisim.backend.playback_common import (
    run_offline_snapshot_playback,
    validate_offline_visual_model,
)

ObsT = TypeVar("ObsT")


def validate_mjwarp_visual_model(
    *,
    mujoco: Any,
    physics_model: Any,
    model_file: str | PathLike[str],
) -> str:
    """Validate the detached MuJoCo visual twin used for offline playback."""
    return validate_offline_visual_model(
        mujoco=mujoco,
        physics_model=physics_model,
        model_file=model_file,
        backend_label="mjwarp",
    )


def run_mjwarp_playback(
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
    snapshot_shape: tuple[int, int],
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
    extra_data_getter: Callable[[], np.ndarray | None] | None = None,
) -> str | None:
    """Render detached mjwarp host snapshots with the existing MuJoCo pipeline."""
    if not headless:
        if record_video:
            raise ValueError("mjwarp interactive playback cannot record video simultaneously.")
        return _run_interactive(
            backend=backend, env=env, initialize=initialize, step=step,
            num_steps=num_steps, snapshot_shape=snapshot_shape,
            frame_state_getter=frame_state_getter, camera_kwargs=camera_kwargs,
        )
    return run_offline_snapshot_playback(
        backend=backend,
        env=env,
        initialize=initialize,
        step=step,
        num_steps=num_steps,
        output_video=output_video,
        render_spacing=render_spacing,
        headless=headless,
        record_video=record_video,
        snapshot_shape=snapshot_shape,
        frame_state_getter=frame_state_getter,
        camera_kwargs=camera_kwargs,
        backend_label="mjwarp",
        extra_data_getter=extra_data_getter,
    )


def _run_interactive(
    *, backend: Any, env: Any, initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT], num_steps: int | None,
    snapshot_shape: tuple[int, int], frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: dict[str, Any] | None,
) -> None:
    """Display one selected Warp world; MuJoCo only computes visual kinematics.

    The passive viewer owns detached model/data, so mouse perturbations cannot
    mutate the physics state. Closing its window ends playback.
    """
    if num_steps is not None and (isinstance(num_steps, bool) or num_steps <= 0):
        raise ValueError("mjwarp interactive playback requires positive num_steps or None.")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("mjwarp interactive playback requires a desktop DISPLAY (GLFW/X11).")
    if os.environ.get("MUJOCO_GL", "glfw").lower() not in ("glfw", ""):
        raise RuntimeError("mjwarp interactive playback requires MUJOCO_GL=glfw.")
    import mujoco
    import mujoco.viewer

    camera = dict(camera_kwargs or {})
    world = int(camera.get("cam_tracking_env_idx", 0))
    if not 0 <= world < snapshot_shape[0]:
        raise ValueError("mjwarp interactive camera environment index is out of range.")
    model = mujoco.MjModel.from_xml_path(backend.get_playback_model(world))
    data = mujoco.MjData(model)
    if model.nmocap != backend._mocap_pos.shape[1]:
        raise ValueError("mjwarp interactive visual model mocap layout is incompatible.")
    getter = frame_state_getter or env.get_physics_state_snapshot
    from unisim.backend.playback_common import env_cfg_value

    ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1 / 60))

    def update() -> None:
        state = np.asarray(getter())
        if state.shape != snapshot_shape:
            raise ValueError(f"mjwarp interactive snapshot must have shape {snapshot_shape}.")
        data.time = float(state[world, 0])
        data.qpos[:] = state[world, 1:1 + model.nq]
        data.qvel[:] = state[world, 1 + model.nq:]
        mocap_pos, mocap_quat = backend.get_playback_mocap_state(world)
        data.mocap_pos[:] = mocap_pos
        data.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(model, data)

    obs = initialize()
    update()
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        raise RuntimeError(
            "mjwarp could not open the MuJoCo viewer; check GLFW/display access "
            "(on macOS use mjpython)."
        ) from exc
    with viewer:
        with viewer.lock():
            for key, attribute in (("cam_distance", "distance"),
                                   ("cam_elevation", "elevation"),
                                   ("cam_azimuth", "azimuth")):
                if key in camera:
                    setattr(viewer.cam, attribute, float(camera[key]))
        viewer.sync()
        count = 0
        while viewer.is_running() and (num_steps is None or count < num_steps):
            start = time.monotonic()
            obs = step(obs)
            with viewer.lock():
                update()
            viewer.sync()
            count += 1
            time.sleep(max(0, ctrl_dt - (time.monotonic() - start)))
