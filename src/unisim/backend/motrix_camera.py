from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.backend.base import CameraCfg


@dataclass(frozen=True)
class MotrixTrackingCamera:
    env_idx: int
    distance: float
    elevation: float
    azimuth: float


@dataclass(frozen=True)
class MotrixCameraView:
    lookat: list[float]
    distance: float
    elevation: float
    azimuth: float
    tracking: MotrixTrackingCamera | None = None


def render_offsets(num_envs: int, spacing: float, offset_mode: str = "grid") -> list[list[float]]:
    if offset_mode == "zero":
        return [[0.0, 0.0, 0.0] for _ in range(num_envs)]
    if offset_mode != "grid":
        raise ValueError(f"Unsupported Motrix render_offset_mode: {offset_mode!r}")
    cols = int(np.ceil(np.sqrt(num_envs)))
    offsets = []
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        offsets.append([col * spacing, row * spacing, 0.0])
    return offsets


def tracking_camera_lookat(
    base_positions: np.ndarray,
    tracking_camera: MotrixTrackingCamera,
    offsets: np.ndarray,
) -> list[float]:
    base_pos = np.asarray(base_positions[tracking_camera.env_idx], dtype=np.float64)
    render_offset = np.asarray(offsets[tracking_camera.env_idx], dtype=np.float64)
    lookat = base_pos + render_offset
    return [float(lookat[0]), float(lookat[1]), float(lookat[2])]


def resolve_system_camera_view(
    num_envs: int,
    base_positions: np.ndarray | None,
    offsets: Sequence[Sequence[float]],
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
) -> MotrixCameraView:
    camera = CameraCfg.from_kwargs(camera_kwargs)
    if camera.cam_tracking:
        if base_positions is None:
            raise ValueError("base_positions is required when cam_tracking=true")
        env_idx = max(0, min(camera.cam_tracking_env_idx, num_envs - 1))
        tracking_camera = MotrixTrackingCamera(
            env_idx=env_idx,
            distance=camera.cam_distance,
            elevation=camera.cam_elevation,
            azimuth=camera.cam_azimuth,
        )
        lookat = tracking_camera_lookat(
            base_positions,
            tracking_camera,
            np.asarray(offsets, dtype=np.float64),
        )
        return MotrixCameraView(
            lookat=lookat,
            distance=tracking_camera.distance,
            elevation=tracking_camera.elevation,
            azimuth=tracking_camera.azimuth,
            tracking=tracking_camera,
        )

    if camera.cam_lookat is None:
        offsets_np = np.asarray(offsets, dtype=np.float64)
        lookat = [
            float(np.mean(offsets_np[:, 0])),
            float(np.mean(offsets_np[:, 1])),
            0.75,
        ]
    else:
        lookat = [float(v) for v in camera.cam_lookat]

    return MotrixCameraView(
        lookat=lookat,
        distance=camera.cam_distance,
        elevation=camera.cam_elevation,
        azimuth=camera.cam_azimuth,
    )
