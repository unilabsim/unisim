"""Host-side IsaacSim/IsaacLab subprocess backend.

IsaacSim is deliberately kept out of the UniLab interpreter: the supported
IsaacSim 5.1 wheels require Python 3.11, while the main project supports a
different Python range.  The NumPy-facing contract and lifecycle come from the
backend-neutral ``subprocess_ipc`` owner; this module supplies IsaacSim runtime
discovery, the worker entrypoint, clone-origin validation, and the eval-owned
Kit/viewer/camera capability boundary.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    CameraCfg,
    normalize_play_render_mode,
)
from unisim.backend.isaacgym.backend import IsaacGymWorkerError
from unisim.backend.playback_common import display_available
from unisim.backend.subprocess_ipc.backend import (
    MjcfSubprocessBackend,
    SubprocessModelInfo,
)
from unisim.backend.subprocess_ipc.scene_materialization import PreparedWorkerScene
from unisim.backend.subprocess_ipc.sensors import (
    KIND_CONTACT_FORCE,
    KIND_CONTACT_FOUND,
    SceneSensorSpec,
    UnsupportedSensorSpec,
)

from .dependencies import build_worker_env, resolve_isaacsim_runtime

_MODULE_DIR = Path(__file__).resolve().parent
_WORKER_PATH = _MODULE_DIR / "worker.py"


class IsaacSimRenderError(RuntimeError):
    """Raised when the requested IsaacSim render mode cannot be initialized."""


class IsaacSimWorkerError(IsaacGymWorkerError):
    """Raised when the external IsaacSim/IsaacLab worker fails.

    ``IsaacGymWorkerError`` is retained as a compatibility ancestor because
    early subprocess adapters exposed that exception as the public worker
    failure type.  The concrete class remains distinct, so callers can still
    distinguish IsaacSim failures with an exact type check while existing
    ``except IsaacGymWorkerError`` handlers continue to work.
    """


@dataclass(frozen=True)
class IsaacSimModelInfo(SubprocessModelInfo):
    """Opaque metadata returned by the IsaacSim worker handshake."""


class IsaacSimBackend(MjcfSubprocessBackend):
    """Thin host client for the IsaacLab/PhysX worker.

    The shared client owns pipe framing, shared-memory slot allocation,
    timeout/crash diagnostics, XML cold-path metadata, and all NumPy state
    views.  Physics remains the default (``render_mode=None``). Eval/play
    profiles pass a concrete render intent through the cold ``INIT`` payload
    so Kit selects the correct experience before simulation materialization.
    The worker owns all camera/viewport operations; this class validates the
    NumPy-facing frame contract.
    """

    _BACKEND_TYPE = "isaacsim"
    _BACKEND_LABEL = "isaacsim"
    _WORKER_ERROR_CLS = IsaacSimWorkerError
    _MODEL_INFO_CLS = IsaacSimModelInfo

    def __init__(
        self,
        scene: Any,
        num_envs: int,
        sim_dt: float,
        *,
        render_mode: str | None = None,
        render_width: int = 1280,
        render_height: int = 720,
        **kwargs: Any,
    ) -> None:
        mode = None if render_mode is None else normalize_play_render_mode(render_mode)
        for name, value in (("render_width", render_width), ("render_height", render_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._requested_render_mode = mode
        self._resolved_render_mode: str | None = None
        super().__init__(scene, num_envs, sim_dt, **kwargs)
        self._render_width = int(render_width)
        self._render_height = int(render_height)

    def _resolve_render_mode(self) -> str:
        """Resolve eval intent before Kit is launched."""
        requested = self._requested_render_mode
        if requested is None or requested == "none":
            return "none"
        if requested == "auto":
            return "interactive" if display_available() else "record"
        if requested == "interactive" and not display_available():
            raise IsaacSimRenderError(
                "IsaacSim interactive rendering was requested but no local display was found; "
                "set DISPLAY/WAYLAND_DISPLAY or use training.play_render_mode=record for "
                "headless RGB capture."
            )
        return requested

    def _worker_init_payload(self) -> dict[str, Any]:
        mode = self._resolve_render_mode()
        self._resolved_render_mode = mode
        return {
            "render_mode": mode,
            "render_width": self._render_width,
            "render_height": self._render_height,
            "contact_force_sensors": self._contact_force_sensor_payload(),
        }

    def _worker_entrypoint(self) -> Path:
        return _WORKER_PATH

    def _resolve_worker_runtime(self) -> Any:
        return resolve_isaacsim_runtime()

    def _build_worker_environment(self, runtime: Any) -> dict[str, str]:
        return build_worker_env(runtime)

    def _runtime_payload(self, runtime: Any) -> dict[str, str]:
        # The worker receives the resolved interpreter path as a diagnostic
        # and a stable contract field.  It does not import the host package.
        return {
            "isaacsim_python": str(runtime.python),
            "isaaclab_source": (
                "" if runtime.isaaclab_source is None else str(runtime.isaaclab_source)
            ),
        }

    def _mapped_contact_force_sensor_count(self) -> int:
        if self._entity_scene is None:
            return 0
        return len(self._contact_force_sensor_payload())

    def _contact_force_sensor_payload(self) -> list[dict[str, str]]:
        """Translate canonical pair sensors into worker body declarations."""
        if self._entity_scene is None:
            return []
        metadata = self._get_scene_metadata()
        body_owners = {
            entity.name + "/" + body_name: (entity.name, body_name)
            for entity in self._entity_scene.layout.entities
            for body_name in entity.body_names
        }
        specs = [
            spec for spec in metadata.sensors.values() if spec.kind == KIND_CONTACT_FORCE
        ]
        records: list[dict[str, str]] = []
        for spec in specs:
            if spec.target_body_name is None or spec.sensor_index is None:
                raise self._worker_error(
                    "malformed IsaacSim contact-force sensor declaration: " + spec.name
                )
            source = body_owners.get(spec.body_name)
            target = body_owners.get(spec.target_body_name)
            if source is None or target is None:
                raise self._worker_error(
                    "IsaacSim collision-pair contact sensors require both bodies to belong "
                    f"to materialized entities (sensor {spec.name!r})"
                )
            records.append(
                {
                    "name": spec.name,
                    "source_entity": source[0],
                    "source_body": source[1],
                    "target_entity": target[0],
                    "target_body": target[1],
                }
            )
        if [spec.sensor_index for spec in specs] != list(range(len(specs))):
            raise self._worker_error("IsaacSim contact-force sensor row indexes are malformed")
        return records

    def _resolve_sensor_map(self) -> dict[str, tuple[Any, int]]:
        """Resolve only sensors backed by a real IsaacSim state quantity.

        Legacy workers reserve a body-net contact slot without a PhysX reporter,
        while mapped workers do not implement MuJoCo's body-net ``found``
        reduction. Both fail closed. Mapped collision-pair ``force`` sensors use
        the dedicated PhysX reporter slot.
        """
        resolved = super()._resolve_sensor_map()
        metadata = self._get_scene_metadata()
        for name, (spec, _body_id) in tuple(resolved.items()):
            if spec.kind == KIND_CONTACT_FORCE and self._entity_scene is not None:
                continue
            if spec.kind not in (KIND_CONTACT_FOUND, KIND_CONTACT_FORCE):
                continue
            metadata.unsupported_sensors[name] = UnsupportedSensorSpec(
                name=name,
                reason=(
                    "IsaacSim serves this contact declaration only on mapped scenes "
                    "with the dedicated PhysX collision-pair force reporter"
                ),
            )
            del resolved[name]
        return resolved

    def get_sensor_data(self, name: str) -> np.ndarray:
        mapped = self._sensor_map.get(name)
        if mapped is not None and mapped[0].kind == KIND_CONTACT_FORCE:
            self._require_state("get_sensor_data")
            spec: SceneSensorSpec = mapped[0]
            if spec.sensor_index is None:
                raise self._worker_error("IsaacSim contact-force sensor row is unresolved")
            return self._slots["contact_sensor_force"][:, spec.sensor_index, :].copy()
        return super().get_sensor_data(name)

    def _bind_model_metadata(self, meta: dict[str, Any]) -> None:
        """Validate the worker's private clone, collision, and render contract."""
        expected_render_mode = self._resolved_render_mode
        if expected_render_mode is None:
            raise self._worker_error(
                "isaacsim worker metadata arrived before the host resolved its render mode"
            )
        required_render_fields = {
            "graphics_enabled",
            "render_mode",
            "render_width",
            "render_height",
        }
        missing_render_fields = sorted(required_render_fields.difference(meta))
        if missing_render_fields:
            raise self._worker_error(
                "isaacsim worker metadata is missing render startup fields: "
                + ", ".join(missing_render_fields)
            )

        reported_render_mode = meta["render_mode"]
        if reported_render_mode != expected_render_mode:
            raise self._worker_error(
                "isaacsim worker render_mode does not match the host INIT request: "
                f"worker={reported_render_mode!r}, host={expected_render_mode!r}"
            )
        raw_width = meta["render_width"]
        raw_height = meta["render_height"]
        if (
            isinstance(raw_width, bool)
            or not isinstance(raw_width, (int, np.integer))
            or isinstance(raw_height, bool)
            or not isinstance(raw_height, (int, np.integer))
            or (int(raw_width), int(raw_height)) != (self._render_width, self._render_height)
        ):
            raise self._worker_error(
                "isaacsim worker render dimensions do not match the host INIT request: "
                f"worker={raw_width!r}x{raw_height!r}, "
                f"host={self._render_width}x{self._render_height}"
            )
        reported_graphics = meta["graphics_enabled"]
        expected_graphics = expected_render_mode != "none"
        if (
            not isinstance(reported_graphics, (bool, np.bool_))
            or bool(reported_graphics) != expected_graphics
        ):
            raise self._worker_error(
                "isaacsim worker graphics_enabled does not match its startup render mode: "
                f"render_mode={expected_render_mode!r}, "
                f"graphics_enabled={reported_graphics!r}, expected={expected_graphics!r}"
            )

        raw_origins = meta.get("env_origins")
        if raw_origins is None:
            raise self._worker_error(
                "isaacsim worker did not report environment origins; refusing to expose "
                "world-space clone state through the local-frame SimBackend contract"
            )
        origins = np.asarray(raw_origins, dtype=np.float32)
        expected = (self._num_envs, 3)
        if origins.shape != expected or not np.isfinite(origins).all():
            raise self._worker_error(
                f"isaacsim worker environment origins have invalid shape or values: "
                f"got shape {origins.shape}, expected {expected}"
            )
        if self._num_envs > 1:
            unique = np.unique(origins, axis=0)
            if unique.shape[0] != self._num_envs:
                raise self._worker_error(
                    "isaacsim worker returned duplicate environment origins; cloned actors "
                    "would overlap in world space"
                )
            if not bool(meta.get("collision_filtering_applied", False)):
                raise self._worker_error(
                    "isaacsim worker did not apply PhysX collision filtering between environments"
                )
        super()._bind_model_metadata(meta)
        if self._entity_scene is not None:
            self._native_entity_table("body_mass")
            self._native_entity_table("body_com", width=3)

    def _require_mapped_entity_scene(self) -> PreparedWorkerScene:
        if self._entity_scene is None:
            raise NotImplementedError(
                "IsaacSim body-property readback requires an explicit entity scene"
            )
        return self._entity_scene

    def _canonical_body_table(self, field: str) -> np.ndarray:
        scene = self._require_mapped_entity_scene()
        vector = field in ("body_com", "body_ipos")
        model_field = "body_ipos" if field == "body_com" else field
        expected = (scene.layout.nbody, 3) if vector else (scene.layout.nbody,)
        try:
            canonical = np.asarray(
                getattr(scene.owner.model, model_field), dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            raise self._worker_error(
                f"compiled canonical {field} is malformed: expected shape {expected}"
            ) from exc
        if canonical.shape != expected or not np.isfinite(canonical).all():
            raise self._worker_error(
                f"compiled canonical {field} is malformed: got shape {canonical.shape}, "
                f"expected {expected}"
            )
        return canonical.copy()

    def _native_entity_table(
        self, field: str, width: int | None = None
    ) -> np.ndarray:
        scene = self._require_mapped_entity_scene()
        self._require_materialized()
        layout = scene.layout
        shape = (
            (self._num_envs, layout.nbody)
            if width is None
            else (self._num_envs, layout.nbody, width)
        )
        canonical = self._canonical_body_table(field)
        values = np.broadcast_to(canonical, shape).copy()
        for entity in layout.entities:
            record = self._native_entity_records.get(entity.name)
            if record is None:
                raise self._worker_error(
                    "worker omitted native entity properties: " + entity.name
                )
            entity_shape = (
                (self._num_envs, len(entity.body_ids))
                if width is None
                else (self._num_envs, len(entity.body_ids), width)
            )
            try:
                raw = np.asarray(record.get(field), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise self._worker_error(
                    f"worker native {field} is malformed for entity {entity.name}: "
                    f"expected shape {entity_shape}"
                ) from exc
            if raw.shape != entity_shape or not np.isfinite(raw).all():
                raise self._worker_error(
                    f"worker native {field} is malformed for entity {entity.name}: "
                    f"got shape {raw.shape}, expected {entity_shape}"
                )
            if width is None:
                values[:, entity.body_ids] = raw
            else:
                values[:, entity.body_ids, :] = raw
        return values

    def get_body_mass(self) -> np.ndarray:
        """Return native per-environment body masses in public body-id order."""
        return self._native_entity_table("body_mass")

    def get_body_ipos(
        self, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> np.ndarray:
        """Return canonical defaults or native selected materialization COM offsets."""
        self._require_mapped_entity_scene()
        if env_ids is None:
            return self._canonical_body_table("body_ipos")

        ids = self._validate_env_ids(env_ids)
        return self._native_entity_table("body_com", width=3)[ids]

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return the native Kit viewer and RGB camera capabilities."""
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_physics_state_playback=False,
            supports_native_video_capture=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        requested_mode = normalize_play_render_mode(play_render_mode)
        if requested_mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )

        # Kit's experience is immutable after AppLauncher starts. Resolve
        # ``auto`` from the same cold-start decision and fail before playback
        # if a direct caller created a no-rendering or differently configured
        # worker. The eval adapters inject matching intent before env creation.
        startup_mode = self._resolved_render_mode
        if startup_mode is None:
            startup_mode = self._resolve_render_mode()
        mode = startup_mode if requested_mode == "auto" else requested_mode
        if startup_mode == "none" or startup_mode != mode:
            raise IsaacSimRenderError(
                f"IsaacSim worker started in render_mode={startup_mode!r}, but playback requested "
                f"{requested_mode!r}; select the matching training.play_render_mode "
                "before env creation."
            )
        if mode == "interactive":
            return BackendPlayRenderPlan(
                mode=mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if isinstance(play_steps, bool) or play_steps is None or int(play_steps) <= 0:
            raise ValueError(
                "isaacsim record playback requires a positive finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("isaacsim record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode="record",
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def init_renderer(
        self,
        spacing: float = 1.0,
        *,
        offset_mode: str = "grid",
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
    ) -> None:
        mode = self._resolved_render_mode
        if mode is None:
            mode = self._resolve_render_mode()
        requested = "record" if (headless or capture) else "interactive"
        if mode != requested:
            raise IsaacSimRenderError(
                f"IsaacSim worker started in render_mode={mode!r}, but renderer requested "
                f"{requested!r}; select the matching training.play_render_mode before env creation."
            )
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, np.integer))
            or isinstance(height, bool)
            or not isinstance(height, (int, np.integer))
            or int(width) <= 0
            or int(height) <= 0
        ):
            raise IsaacSimRenderError(
                f"IsaacSim render dimensions must be positive integers; got {width!r}x{height!r}."
            )
        if int(width) != self._render_width or int(height) != self._render_height:
            raise IsaacSimRenderError(
                "IsaacSim render dimensions are fixed at worker startup: "
                f"configured {self._render_width}x{self._render_height}, "
                f"requested {width}x{height}."
            )
        # ``super`` owns the protocol/lifecycle and is intentionally called
        # only after the mode/dimension checks above.
        super().init_renderer(
            spacing=spacing,
            offset_mode=offset_mode,
            headless=headless,
            capture=capture,
            width=width,
            height=height,
            camera_kwargs=camera_kwargs,
        )

    def capture_video_frame(self) -> np.ndarray:
        frame = np.asarray(super().capture_video_frame())
        expected = (self._render_height, self._render_width, 3)
        if frame.dtype != np.uint8 or frame.shape != expected:
            raise IsaacSimRenderError(
                "IsaacSim camera returned an invalid RGB frame: "
                f"shape={frame.shape}, dtype={frame.dtype}, expected shape={expected}, dtype=uint8"
            )
        if frame.size == 0 or int(np.ptp(frame)) == 0:
            raise IsaacSimRenderError(
                "IsaacSim camera returned an empty/uniform RGB frame; refusing to write a "
                "placeholder video. Check camera pose, lighting, and RTX camera support."
            )
        return np.ascontiguousarray(frame)


# The model metadata shape is backend-neutral; retaining the alias avoids a
# second, structurally identical dataclass while the error class above remains
# intentionally distinct.
__all__ = [
    "IsaacSimBackend",
    "IsaacSimModelInfo",
    "IsaacSimRenderError",
    "IsaacSimWorkerError",
]
