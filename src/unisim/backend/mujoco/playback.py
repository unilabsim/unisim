"""MuJoCo-owned playback execution helpers."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from unisim.backend.base import (
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    validate_debug_overlays,
)
from unisim.backend.playback_common import (
    apply_on_frame_callback,
    env_cfg_value,
    write_playback_video,
)
from unisim.scene import SceneCfg

ObsT = TypeVar("ObsT")


def run_mujoco_playback(
    *,
    env: Any,
    initialize: Callable[[], ObsT],
    step: Callable[[ObsT], ObsT],
    num_steps: int | None,
    output_video: str | PathLike[str] | None,
    render_spacing: float | None,
    headless: bool,
    record_video: bool,
    frame_state_getter: Callable[[], np.ndarray] | None,
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
    debug_overlay_getter: DebugOverlayGetter | None = None,
    on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
) -> str | None:
    if not headless:
        raise NotImplementedError("MuJoCo play mode does not support interactive rendering here.")
    if not record_video:
        raise ValueError("MuJoCo play rendering requires record_video=true.")
    if num_steps is None:
        raise ValueError("MuJoCo play rendering requires a finite num_steps value.")
    if output_video is None:
        raise ValueError("MuJoCo play rendering requires an output_video path.")
    camera = CameraCfg.from_kwargs(camera_kwargs)
    if frame_state_getter is None:
        frame_state_getter = env.get_physics_state_snapshot
    assert frame_state_getter is not None

    obs = initialize()
    state_list = []
    overlay_list: list[Sequence[Sequence[DebugPrimitive] | None] | None] = []
    for _ in range(num_steps):
        obs = step(obs)
        state_list.append(np.asarray(frame_state_getter(), dtype=np.float32).copy())
        overlay_list.append(debug_overlay_getter() if debug_overlay_getter is not None else None)

    num_envs = int(state_list[0].shape[0])
    validated_overlays = [
        validate_debug_overlays(overlays, num_envs) for overlays in overlay_list
    ]
    debug_overlays_list = (
        validated_overlays if any(overlays is not None for overlays in validated_overlays) else None
    )

    from unisim.visualization import render_many

    effective_spacing = (
        float(render_spacing)
        if render_spacing is not None
        else float(env_cfg_value(env, "render_spacing", 1.0))
    )
    with tempfile.TemporaryDirectory(prefix="unilab-playback-models-") as tmp_dir:
        model_files = resolve_render_play_model_files(
            env,
            num_envs=num_envs,
            tmp_dir=tmp_dir,
        )

        if camera.cam_tracking:
            frames = render_many.render_states_get_frames_tracking(
                state_list,
                model_files,
                width=1280,
                height=720,
                tracking_env_idx=camera.cam_tracking_env_idx,
                max_extra_envs=camera.cam_tracking_extra_envs,
                cam_distance=camera.cam_distance,
                cam_elevation=camera.cam_elevation,
                cam_azimuth=camera.cam_azimuth,
                cam_fov=camera.cam_fov,
                render_spacing=effective_spacing,
                debug_overlays_list=debug_overlays_list,
            )
        else:
            frames = render_many.render_states_get_frames(
                state_list,
                model_files,
                width=1280,
                height=720,
                camera_id=-1,
                cam_distance=camera.cam_distance,
                cam_elevation=camera.cam_elevation,
                cam_azimuth=camera.cam_azimuth,
                cam_lookat=camera.cam_lookat,
                cam_fov=camera.cam_fov,
                render_spacing=effective_spacing,
                debug_overlays_list=debug_overlays_list,
            )

    if not frames:
        # Rendering was skipped (e.g. no usable off-screen GL backend on a
        # headless host). render_many already warned with actionable guidance;
        # don't fail the eval/play run — just skip the video export.
        print(f"[playback] No frames rendered; skipping video export to {output_video}.")
        return None

    frames = apply_on_frame_callback(frames, on_frame, backend_label="mujoco")
    ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
    write_playback_video(str(output_video), frames, fps=int(1.0 / ctrl_dt))
    return str(output_video)


def _configured_model_file(env: Any) -> str | None:
    cfg = getattr(env, "cfg", None)
    scene = getattr(cfg, "scene", None) if cfg is not None else None
    if scene is None:
        return None
    if not isinstance(scene, SceneCfg):
        raise TypeError("env.cfg.scene must be a SceneCfg")
    return scene.model_file


def _visual_model_file(env: Any) -> str | None:
    backend = getattr(env, "_backend", None)
    backend_visual_model_file = getattr(backend, "scene_visual_model_file", None)
    if backend_visual_model_file:
        return str(backend_visual_model_file)
    return _configured_model_file(env)


def resolve_render_play_model_files(
    env: Any,
    *,
    num_envs: int,
    tmp_dir: str | Path,
) -> str | list[str]:
    """Resolve the visual MuJoCo model file for offline play/video export.

    The mjbatch executor has no per-env model variants, so one model file
    serves every rendered env: the scene's visual model when configured, else
    a single mjb saved from the backend's playback model.
    """
    del num_envs  # one model file serves every env; no per-env variants exist
    visual_model_file = _visual_model_file(env)
    if visual_model_file is not None:
        return visual_model_file
    if not hasattr(env, "get_playback_model"):
        raise ValueError("MuJoCo playback requires either cfg.scene or get_playback_model().")

    playback_model = env.get_playback_model(0)
    if isinstance(playback_model, (str, Path)):
        return str(playback_model)

    import mujoco as _mujoco

    mujoco: Any = _mujoco

    output_path = Path(tmp_dir) / "playback_model.mjb"
    mujoco.mj_saveModel(playback_model, str(output_path))
    return str(output_path)
