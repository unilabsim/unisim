"""MotrixSim GPU playback rendering helpers.

MotrixSim's GPU package exposes the scene renderer but no headless capture
entry point, while the CPU renderer (`motrixsim.render`) provides a proven
headless system-camera capture path.  This module therefore mirrors the GPU
link poses into that renderer: it loads the same whole-MJCF file as a render
model and synchronizes one ``(num_worlds, num_links, 7)`` pose block per frame,
so the physics stays GPU-resident and only the rendered kinematics cross into
the renderer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from unisim.backend.base import CameraCfg
from unisim.backend.motrix.camera import render_offsets, resolve_system_camera_view

# The render app returns pre-load placeholder frames until the scene meshes and
# materials finish streaming in; the observed latency is a handful of frames for
# the G1 asset, so steady-state capture starts after this many discarded
# round-trips.
_CAPTURE_WARMUP_FRAMES = 12


class MotrixGpuPlaybackRenderer:
    """Pose-mirroring CPU renderer driven by MotrixSim GPU link poses."""

    def __init__(
        self,
        *,
        model_file: str,
        scene_world: Any,
        num_envs: int,
        link_names: tuple[str, ...],
        root_link_index: int,
        pose_getter: Callable[[], np.ndarray],
    ) -> None:
        self._model_file = model_file
        self._scene_world = scene_world
        self._num_envs = int(num_envs)
        self._link_names = link_names
        self._root_link_index = int(root_link_index)
        self._pose_getter = pose_getter
        self._app: Any = None
        self._headless: bool | None = None
        self._capture = False

    @property
    def active(self) -> bool:
        return self._app is not None

    def init(
        self,
        *,
        spacing: float,
        offset_mode: str,
        headless: bool,
        capture: bool,
        width: int,
        height: int,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None,
    ) -> None:
        """Launch the render app, optionally arming the system-camera capture."""
        headless = bool(headless)
        capture = bool(capture)
        if self._app is not None:
            if self._headless != headless:
                raise ValueError(
                    "MotrixSim GPU renderer is already initialized with "
                    f"headless={self._headless!r}; cannot reuse it with headless={headless!r}"
                )
            if capture and not self._capture:
                raise ValueError(
                    "MotrixSim GPU renderer is already initialized without video capture; "
                    "cannot enable capture on the existing renderer"
                )
            return

        import motrixsim.msd as msd
        from motrixsim.render import RenderApp, RenderSettings

        camera = CameraCfg.from_kwargs(camera_kwargs)
        world = (
            self._scene_world if self._scene_world is not None else msd.from_file(self._model_file)
        )
        if capture:
            world.system_camera.render_target = msd.RenderTarget.Image
            world.system_camera.resolution = [int(width), int(height)]
        model = msd.build(world)
        render_link_names = tuple("" if name is None else str(name) for name in model.link_names)
        if render_link_names != self._link_names:
            raise RuntimeError(
                "MotrixSim GPU playback requires the render model's link layout to match "
                "the simulation model; the renderer exposes "
                f"{len(render_link_names)} links and the simulation {len(self._link_names)}"
            )
        offsets = render_offsets(self._num_envs, float(spacing), str(offset_mode))
        settings = RenderSettings.performance()
        settings.enable_shadow = True
        app = RenderApp(headless=headless)
        app.launch(
            model,
            batch=self._num_envs,
            render_offset=offsets,
            render_settings=settings,
        )
        if capture or camera_kwargs is not None:
            base_positions = (
                np.asarray(self._pose_getter())[:, self._root_link_index, :3]
                if camera.cam_tracking
                else None
            )
            camera_view = resolve_system_camera_view(
                self._num_envs, base_positions, offsets, camera
            )
            app.system_camera.set_view(
                camera_view.lookat,
                camera_view.distance,
                camera_view.elevation,
                camera_view.azimuth,
            )
        self._app = app
        self._headless = headless
        self._capture = capture
        if capture:
            # Discard the pre-load placeholder frames so the first delivered
            # capture is a real render.
            poses = self._pose_getter()
            for _ in range(_CAPTURE_WARMUP_FRAMES):
                warmup = app.system_camera.capture()
                app.sync(data=poses, wait=True)
                warmup.take_image()

    def render(self) -> None:
        """Synchronize the mirrored poses for the interactive renderer."""
        if self._app is None:
            self.init(
                spacing=1.0,
                offset_mode="grid",
                headless=False,
                capture=False,
                width=1280,
                height=720,
                camera_kwargs=None,
            )
        assert self._app is not None
        self._app.sync(data=self._pose_getter(), wait=False)

    def capture_frame(self) -> np.ndarray:
        """Capture one RGB frame from the system camera."""
        if self._app is None:
            self.init(
                spacing=1.0,
                offset_mode="grid",
                headless=True,
                capture=True,
                width=1280,
                height=720,
                camera_kwargs=None,
            )
        if not self._capture:
            raise RuntimeError("MotrixSim GPU renderer is not initialized for video capture")
        assert self._app is not None
        task = self._app.system_camera.capture()
        self._app.sync(data=self._pose_getter(), wait=True)
        image = task.take_image()
        if image is None:
            raise RuntimeError("MotrixSim GPU system camera capture did not return an image")
        pixels = np.asarray(image.pixels)
        if pixels.ndim != 3:
            raise RuntimeError(
                "MotrixSim GPU system camera capture must return an HWC image, "
                f"got shape {pixels.shape}"
            )
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        if pixels.shape[-1] != 3:
            raise RuntimeError(
                "MotrixSim GPU system camera capture must return RGB/RGBA pixels, "
                f"got shape {pixels.shape}"
            )
        return np.ascontiguousarray(pixels, dtype=np.uint8)

    def close(self) -> None:
        app, self._app = self._app, None
        self._headless = None
        self._capture = False
        if app is not None:
            app.close()


__all__ = ["MotrixGpuPlaybackRenderer"]
