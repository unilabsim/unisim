"""Cold-path MuJoCo offline playback bridge for ``mjwarp``.

The implementation lives in :mod:`unisim.backend.playback_common` so other
snapshot-based adapters (Newton) share one offline MuJoCo pipeline; the
wrappers below keep the historical mjwarp names, signatures, and messages.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, TypeVar

import numpy as np

from unisim.backend.base import CameraCfg, DebugOverlayGetter, validate_debug_overlays
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
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
    debug_overlay_getter: DebugOverlayGetter | None = None,
    on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
) -> str | None:
    """Render detached mjwarp host snapshots with the existing MuJoCo pipeline."""
    if not headless:
        if record_video:
            raise ValueError("mjwarp interactive playback cannot record video simultaneously.")
        if on_frame is not None:
            raise NotImplementedError(
                "mjwarp interactive playback does not support on_frame callbacks; "
                "use play_render_mode=record (offline MuJoCo snapshot renderer)"
            )
        return _run_interactive(
            backend=backend, env=env, initialize=initialize, step=step,
            num_steps=num_steps, snapshot_shape=snapshot_shape,
            frame_state_getter=frame_state_getter, camera_kwargs=camera_kwargs,
            debug_overlay_getter=debug_overlay_getter,
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
        debug_overlay_getter=debug_overlay_getter,
        on_frame=on_frame,
    )


def _inject_interactive_debug_overlays(
    *,
    user_scn: Any,
    overlays: Any,
    world: int,
    num_envs: int,
    model: Any,
    mesh_id_cache: dict[str, int],
) -> int:
    """Inject this frame's debug primitives into the passive viewer scene.

    Returns the number of scene geoms written.  The caller owns
    ``viewer.lock()``; the scene is reset (``ngeom = 0``) before injection
    because ``viewer.sync()`` does not clear user geoms.
    """
    import mujoco

    from unisim.visualization.render_many import append_debug_primitives

    validated = validate_debug_overlays(overlays, num_envs)
    user_scn.ngeom = 0
    if validated is None:
        return 0
    env_primitives = validated[world]
    if not env_primitives:
        return 0
    for primitive in env_primitives:
        if primitive.kind == "ghost_geom" and primitive.mesh_asset not in mesh_id_cache:
            mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, primitive.mesh_asset)
            if mesh_id < 0:
                raise ValueError(
                    f"mjwarp interactive ghost_geom mesh {primitive.mesh_asset!r} is not "
                    "registered in the playback model; interactive overlay meshes cannot "
                    "be injected from files (see append_debug_primitives)"
                )
            mesh_id_cache[primitive.mesh_asset] = int(mesh_id)
    return append_debug_primitives(
        user_scn, [env_primitives], offsets=None, mesh_ids=mesh_id_cache
    )


def _run_interactive(
    *, backend: Any, env: Any, initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT], num_steps: int | None,
    snapshot_shape: tuple[int, int], frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
    debug_overlay_getter: DebugOverlayGetter | None = None,
) -> None:
    """Display one selected Warp world; MuJoCo only computes visual kinematics.

    The passive viewer owns detached model/data, so mouse perturbations cannot
    mutate the physics state. Closing its window ends playback.  When
    ``debug_overlay_getter`` is provided, the current frame's primitives for
    the displayed world are injected into ``viewer.user_scn`` before each
    ``viewer.sync()``.
    """
    if num_steps is not None and (isinstance(num_steps, bool) or num_steps <= 0):
        raise ValueError("mjwarp interactive playback requires positive num_steps or None.")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("mjwarp interactive playback requires a desktop DISPLAY (GLFW/X11).")
    if os.environ.get("MUJOCO_GL", "glfw").lower() not in ("glfw", ""):
        raise RuntimeError("mjwarp interactive playback requires MUJOCO_GL=glfw.")
    import mujoco
    import mujoco.viewer

    camera = CameraCfg.from_kwargs(camera_kwargs)
    world = camera.cam_tracking_env_idx
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
        mesh_id_cache: dict[str, int] = {}
        if debug_overlay_getter is not None and viewer.user_scn is None:
            raise RuntimeError(
                "mjwarp interactive debug overlays require viewer.user_scn support."
            )
        with viewer.lock():
            viewer.cam.distance = camera.cam_distance
            viewer.cam.elevation = camera.cam_elevation
            viewer.cam.azimuth = camera.cam_azimuth
        viewer.sync()
        count = 0
        while viewer.is_running() and (num_steps is None or count < num_steps):
            start = time.monotonic()
            obs = step(obs)
            with viewer.lock():
                update()
                if debug_overlay_getter is not None:
                    _inject_interactive_debug_overlays(
                        user_scn=viewer.user_scn,
                        overlays=debug_overlay_getter(),
                        world=world,
                        num_envs=snapshot_shape[0],
                        model=model,
                        mesh_id_cache=mesh_id_cache,
                    )
            viewer.sync()
            count += 1
            time.sleep(max(0, ctrl_dt - (time.monotonic() - start)))
