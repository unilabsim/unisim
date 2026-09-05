"""Cold-path MuJoCo offline playback bridge for ``mjwarp``.

The implementation lives in :mod:`unisim.backend.playback_common` so other
snapshot-based adapters (Newton) share one offline MuJoCo pipeline; the
wrappers below keep the historical mjwarp names, signatures, and messages.
"""

from __future__ import annotations

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
) -> str:
    """Render detached mjwarp host snapshots with the existing MuJoCo pipeline."""
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
