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
import time
from collections.abc import Callable, Mapping, Sequence
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
from unisim.backend.subprocess_ipc import protocol
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
from unisim.dr.interval import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalTermOp,
)
from unisim.dr.types import (
    RESET_TERM_GEOM_FRICTION,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unisim.entities import SceneResetRequest

from .dependencies import build_worker_env, resolve_isaacsim_runtime
from .physx_solver import PhysxSolverConfig, solver_value_matches
from .raw_usd_cache import resolve_raw_usd_cache_root, resolve_role_usd_cache_root

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
        solver_position_iteration_count: int | None = None,
        solver_velocity_iteration_count: int | None = None,
        bounce_threshold_velocity: float | None = None,
        contact_offset: float | None = None,
        **kwargs: Any,
    ) -> None:
        mode = None if render_mode is None else normalize_play_render_mode(render_mode)
        for name, value in (("render_width", render_width), ("render_height", render_height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        # Validation is fail-closed at construction, before any worker spawn.
        self._physx_solver = PhysxSolverConfig(
            solver_position_iteration_count=solver_position_iteration_count,
            solver_velocity_iteration_count=solver_velocity_iteration_count,
            bounce_threshold_velocity=bounce_threshold_velocity,
            contact_offset=contact_offset,
        )
        self._requested_render_mode = mode
        self._resolved_render_mode: str | None = None
        super().__init__(scene, num_envs, sim_dt, **kwargs)
        self._render_width = int(render_width)
        self._render_height = int(render_height)
        self._staged_body_wrench = (
            None
            if self._entity_scene is None
            else np.zeros((self._num_envs, self._entity_scene.layout.nbody, 6), dtype=np.float32)
        )
        self._body_wrench_pending = False

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
        raw_usd_cache_root = resolve_raw_usd_cache_root()
        role_usd_cache_root = resolve_role_usd_cache_root()
        return {
            "render_mode": mode,
            "render_width": self._render_width,
            "render_height": self._render_height,
            "contact_force_sensors": self._contact_force_sensor_payload(),
            "physx_solver": self._physx_solver.to_payload(),
            "raw_usd_cache_dir": (
                None if raw_usd_cache_root is None else str(raw_usd_cache_root)
            ),
            "role_usd_cache_dir": (
                None if role_usd_cache_root is None else str(role_usd_cache_root)
            ),
        }

    def _worker_configuration_requested(self) -> dict[str, Any]:
        return self._physx_solver.to_payload()

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise only reset and wrench terms implemented by mapped scenes."""
        if self._entity_scene is None:
            return super().get_dr_capabilities()
        terms = frozenset({INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE})
        return DomainRandomizationCapabilities(
            supports_interval_body_force=True,
            supports_interval_body_torque=True,
            supported_interval_terms=terms,
            supported_reset_terms=frozenset({RESET_TERM_GEOM_FRICTION}),
        )

    @staticmethod
    def _coerce_mapped_reset_field(
        values: Any, name: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"isaacsim {name} must be a floating NumPy array")
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"isaacsim {name} must have shape {shape}, got {array.shape}")
        if (
            not np.isfinite(array).all()
            or np.any(np.abs(array.astype(np.float64)) > np.finfo(np.float32).max)
        ):
            raise ValueError(f"isaacsim {name} must contain finite float32 values")
        return array.copy()

    def _validated_mapped_reset_randomization(
        self, randomization: ResetRandomizationPayload | None, rows: np.ndarray
    ) -> ResetRandomizationPayload | None:
        if randomization is None or randomization.is_empty():
            return None
        scene = self._require_mapped_entity_scene()
        unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
            randomization.requested_terms()
        )
        if unsupported:
            requested = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"isaacsim does not support reset domain randomization terms: {requested}."
            )

        geom_friction: np.ndarray | None = None
        if randomization.geom_friction is not None:
            geom_friction = self._coerce_mapped_reset_field(
                randomization.geom_friction,
                "geom_friction",
                (rows.size, scene.layout.ngeom, 3),
            )
            if np.any(geom_friction < 0.0):
                raise ValueError("isaacsim geom_friction values must be nonnegative")
            if (
                np.any(geom_friction[..., 0] != geom_friction[..., 1])
                or np.any(geom_friction[..., 2] != 0.0)
            ):
                raise ValueError(
                    "isaacsim geom_friction requires static == dynamic and a zero third column"
                )

        return ResetRandomizationPayload(geom_friction=geom_friction)

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        if self._entity_scene is not None:
            self._reject_wrench_write_inside_pre_step_control("apply_interval_randomization")
            assert self._staged_body_wrench is not None
            self._staged_body_wrench.fill(0.0)
            self._body_wrench_pending = False
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        if self._entity_scene is None:
            return super()._interval_term_handlers()
        return {
            INTERVAL_TERM_BODY_FORCE: self._apply_interval_body_force,
            INTERVAL_TERM_BODY_TORQUE: self._apply_interval_body_torque,
        }

    def _apply_interval_body_force(self, op: IntervalTermOp) -> None:
        if op.body_ids is None:
            raise ValueError("interval body force requires body_ids")
        self.apply_body_force(op.body_ids, op.payload)

    def _apply_interval_body_torque(self, op: IntervalTermOp) -> None:
        if op.body_ids is None:
            raise ValueError("interval body torque requires body_ids")
        self._apply_body_torque(op.body_ids, op.payload)

    def _apply_body_torque(self, body_ids: np.ndarray, torque: np.ndarray) -> None:
        zero_force = np.zeros((self._num_envs, len(body_ids), 3), dtype=np.float32)
        self.apply_body_force(body_ids, zero_force, torque=torque)

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Stage a world-frame wrench for the next mapped-scene step.

        The force acts at each target body's center of mass and the optional
        torque is about that center. Repeated submissions accumulate until the
        next successful step consumes them.
        """
        if self._entity_scene is None:
            super().apply_body_force(body_ids, force, torque)
            return
        self._reject_wrench_write_inside_pre_step_control("apply_body_force")
        self._require_state("interval body force perturbation")
        assert self._staged_body_wrench is not None
        body_ids_np = np.asarray(body_ids)
        if (
            body_ids_np.ndim != 1
            or body_ids_np.dtype.kind not in "iu"
            or np.issubdtype(body_ids_np.dtype, np.bool_)
        ):
            raise ValueError("body_ids must be a one-dimensional integer array")
        if np.any(body_ids_np < 0) or np.any(body_ids_np >= self._staged_body_wrench.shape[1]):
            raise ValueError(
                f"body ids must be in [0, {self._staged_body_wrench.shape[1]}), "
                f"got range [{body_ids_np.min(initial=0)}, {body_ids_np.max(initial=0)}]"
            )
        body_ids_np = body_ids_np.astype(np.intp, copy=False)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        force_np = np.asarray(force, dtype=np.float32)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        if not np.isfinite(force_np).all():
            raise ValueError("body force contains NaN or Inf")
        torque_np = None
        if torque is not None:
            torque_np = np.asarray(torque, dtype=np.float32)
            if torque_np.shape != expected_shape:
                raise ValueError(
                    f"body torque must have shape {expected_shape}, got {torque_np.shape}"
                )
            if not np.isfinite(torque_np).all():
                raise ValueError("body torque contains NaN or Inf")
        for body_offset, body_id in enumerate(body_ids_np):
            self._staged_body_wrench[:, int(body_id), 0:3] += force_np[:, body_offset, :]
            if torque_np is not None:
                self._staged_body_wrench[:, int(body_id), 3:6] += torque_np[:, body_offset, :]
        self._body_wrench_pending = bool(
            np.any(self._staged_body_wrench) or self._body_wrench_pending
        )

    def _step_payload(
        self,
        nsteps: int,
        *,
        body_wrench: np.ndarray | None = None,
    ) -> dict[str, Any]:
        payload = super()._step_payload(nsteps)
        if body_wrench is not None:
            payload["body_wrench"] = body_wrench.tobytes(order="C")
        elif self._entity_scene is not None and self._body_wrench_pending:
            assert self._staged_body_wrench is not None
            # NumPy pickle internals are not stable across the host and Isaac
            # worker interpreter versions; raw C-order bytes are.
            payload["body_wrench"] = self._staged_body_wrench.tobytes(order="C")
        return payload

    def set_pre_step_control(self, fn: Any | None) -> None:
        """Register a callback for mapped scenes; legacy callbacks stay rejected."""
        if fn is not None and self._entity_scene is None:
            super().set_pre_step_control(fn)
            return
        self._pre_step_control_fn = fn

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        if self._entity_scene is None or self._pre_step_control_fn is None:
            return super().step(ctrl, nsteps)
        self._require_state("step")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        self._require_materialized()
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self.num_actuators)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")
        if not np.isfinite(ctrl_array).all():
            raise ValueError("control must contain finite target values")
        ctrl_array = np.clip(
            ctrl_array, self._entity_scene.control_lower, self._entity_scene.control_upper
        )
        return self._step_with_pre_step_control(ctrl_array, int(nsteps))

    def _step_with_pre_step_control(
        self, ctrl: np.ndarray, nsteps: int
    ) -> dict[str, dict[str, float]]:
        """Run one mapped worker STEP per public physics substep.

        The worker owns each native substep, while this host loop preserves the
        backend-neutral callback boundary: shared state is refreshed after every
        one-step command before the next callback invocation.  This is a
        correctness path, not the device-resident controller investigated by
        #152.
        """
        assert self._entity_scene is not None
        assert self._staged_body_wrench is not None
        fixed_wrench = self._staged_body_wrench.copy()
        composed_wrench = np.zeros_like(fixed_wrench)
        control_ms = 0.0
        physics_ms = 0.0
        started = time.perf_counter()
        self._pre_step_control_active = True
        try:
            for substep in range(nsteps):
                callback_started = time.perf_counter()
                output = self._convert_pre_step_control(ctrl)
                converted_ctrl = np.clip(
                    output.ctrl,
                    self._entity_scene.control_lower,
                    self._entity_scene.control_upper,
                )
                if not np.isfinite(converted_ctrl).all():
                    raise ValueError("pre-step control must contain finite target values")
                composed_wrench[:] = fixed_wrench
                if output.force is not None or output.torque is not None:
                    body_ids = np.asarray(output.body_ids, dtype=np.intp)
                    if np.any(body_ids < 0) or np.any(body_ids >= composed_wrench.shape[1]):
                        raise ValueError(
                            f"pre-step control body ids must be in "
                            f"[0, {composed_wrench.shape[1]}), got range "
                            f"[{body_ids.min(initial=0)}, {body_ids.max(initial=0)}]"
                        )
                    for body_offset, body_id in enumerate(body_ids):
                        if output.force is not None:
                            composed_wrench[:, int(body_id), 0:3] += output.force[
                                :, body_offset, :
                            ].astype(np.float32, copy=False)
                        if output.torque is not None:
                            composed_wrench[:, int(body_id), 3:6] += output.torque[
                                :, body_offset, :
                            ].astype(np.float32, copy=False)
                np.copyto(self._slots["ctrl"], converted_ctrl)
                control_ms += time.perf_counter() - callback_started

                step_payload = self._step_payload(1, body_wrench=composed_wrench)
                payload = self._request(protocol.CMD_STEP, step_payload, expect=protocol.CMD_READY)
                if isinstance(payload, dict):
                    physics_ms += float(payload.get("timing", {}).get("physics_ms", 0.0))
                if substep == nsteps - 1:
                    self._after_step(step_payload)
                self._stale_body_ids.clear()
        finally:
            self._pre_step_control_active = False
            # A callback step consumes staged interval wrench even if a later
            # callback or worker substep fails; completed native substeps cannot
            # be rolled back and a partially applied wrench must not leak.
            self._staged_body_wrench.fill(0.0)
            self._body_wrench_pending = False

        return {
            "timing": {
                "control_upload_ms": control_ms * 1000.0,
                "physics_ms": physics_ms,
                "worker_ipc_total_ms": (time.perf_counter() - started) * 1000.0,
            }
        }

    def _after_step(self, payload: dict[str, Any]) -> None:
        if "body_wrench" in payload:
            assert self._staged_body_wrench is not None
            self._staged_body_wrench.fill(0.0)
            self._body_wrench_pending = False

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        super().reset(env_ids)
        if self._entity_scene is None:
            return
        rows = np.arange(self._num_envs) if env_ids is None else np.asarray(env_ids)
        if rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise ValueError("reset env_ids must be one-dimensional integers")
        if not len(rows):
            return
        if (
            np.any(rows < 0)
            or np.any(rows >= self._num_envs)
            or len(set(rows.tolist())) != len(rows)
        ):
            raise ValueError("reset env_ids must be distinct and in range")
        assert self._staged_body_wrench is not None
        # A full default-state reset covers every public body, including
        # entities with no writable state patch and unowned world rows.
        self._staged_body_wrench[rows.astype(np.intp, copy=False)] = 0.0
        self._body_wrench_pending = bool(np.any(self._staged_body_wrench))

    def _commit_entity_reset(
        self,
        request: SceneResetRequest,
        control_values: np.ndarray | None = None,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        super()._commit_entity_reset(request, control_values, randomization)
        if self._entity_scene is None:
            return
        assert self._staged_body_wrench is not None
        rows = np.asarray(request.env_ids, dtype=np.intp)
        body_id_groups = tuple(
            np.asarray(self.get_scene_layout().get_entity(patch.entity).body_ids, dtype=np.intp)
            for patch in request.patches
        )
        body_ids = (
            np.concatenate(body_id_groups)
            if body_id_groups
            else np.empty(0, dtype=np.intp)
        )
        self._staged_body_wrench[np.ix_(rows, body_ids)] = 0.0
        self._body_wrench_pending = bool(np.any(self._staged_body_wrench))

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
        self._validate_solver_readback(meta)
        super()._bind_model_metadata(meta)
        if self._entity_scene is not None:
            self._native_entity_table("body_mass")
            self._native_entity_table("body_com", width=3)
            self._validated_native_geometry_records()

    def _validate_solver_readback(self, meta: dict[str, Any]) -> None:
        """Fail closed when the worker's engine readback misses the INIT request."""
        requested = self._physx_solver.to_payload()
        if not requested:
            return
        envelope = meta.get("configuration_report")
        if not isinstance(envelope, dict) or not isinstance(envelope.get("effective"), dict):
            raise self._worker_error(
                "isaacsim worker omitted its configuration report although PhysX solver "
                "overrides were requested"
            )
        effective = envelope["effective"]
        raw_readback = envelope.get("engine_readback")
        readback_fields = (
            set(raw_readback) if isinstance(raw_readback, (list, tuple)) else set()
        )
        for field, value in requested.items():
            if field not in readback_fields:
                raise self._worker_error(
                    f"isaacsim worker did not read back the requested PhysX setting "
                    f"{field!r} from the engine"
                )
            if field not in effective:
                raise self._worker_error(
                    "isaacsim worker configuration report is missing the requested "
                    f"PhysX setting {field!r}"
                )
            if not solver_value_matches(field, value, effective[field]):
                raise self._worker_error(
                    f"isaacsim worker PhysX {field} does not match the host INIT request: "
                    f"worker={effective[field]!r}, host={value!r}"
                )

    def _require_mapped_entity_scene(self) -> PreparedWorkerScene:
        if self._entity_scene is None:
            raise NotImplementedError(
                "IsaacSim body-property readback requires an explicit entity scene"
            )
        return self._entity_scene

    def _consume_entity_reset_randomization(self, response: Any) -> None:
        """Replace current native property records from the reset barrier."""
        scene = self._require_mapped_entity_scene()
        raw_records = response.get("native_entity_records") if isinstance(response, dict) else None
        if not isinstance(raw_records, list):
            raise self._worker_error(
                "isaacsim worker omitted native records after property mutation"
            )
        records = {record.get("name"): record for record in raw_records if isinstance(record, dict)}
        if len(records) != len(raw_records) or set(records) != set(self.get_entity_names()):
            raise self._worker_error(
                "isaacsim worker property-mutation records do not match the frozen entity layout"
            )
        for entity in scene.layout.entities:
            current = self._native_entity_records.get(entity.name)
            if current is None:
                raise self._worker_error(
                    "worker omitted native entity properties: " + entity.name
                )
            reported = records[entity.name]
            for field in ("body_mass", "geom_friction"):
                if field not in reported:
                    raise self._worker_error(
                        f"worker omitted native {field} after property mutation: {entity.name}"
                    )
                current[field] = reported[field]
        # Validate the accepted records before returning from the reset barrier.
        self._native_entity_table("body_mass")
        self._validated_native_geometry_records()

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

    def _validated_native_geometry_records(
        self,
    ) -> tuple[tuple[Any, int, np.ndarray, np.ndarray], ...]:
        """Validate and return native geometry records in public entity order.

        The returned tuple contains ``(entity, public offset, masks, friction)``.
        Identity is checked against the frozen layout while values remain native
        worker readback; contact state is role-immutable and friction may vary by
        immutable variant assignment.
        """
        scene = self._require_mapped_entity_scene()
        self._require_materialized()
        result = []
        public_offset = 0
        for entity in scene.layout.entities:
            record = self._native_entity_records.get(entity.name)
            if record is None:
                raise self._worker_error(
                    "worker omitted native entity geometry: " + entity.name
                )
            names = record.get("geom_names")
            body_names = record.get("geom_body_names")
            expected_names = [geom.name for geom in entity.geoms]
            expected_bodies = [geom.body_name for geom in entity.geoms]
            name_rows = [expected_names] * self._num_envs
            body_rows = [expected_bodies] * self._num_envs
            if names != name_rows:
                raise self._worker_error(
                    f"worker native geom_names differ from the frozen layout for entity "
                    f"{entity.name}: expected {expected_names!r}, got {names!r}"
                )
            if body_names != body_rows:
                raise self._worker_error(
                    f"worker native geom_body_names differ from the frozen layout for entity "
                    f"{entity.name}: expected {expected_bodies!r}, got {body_names!r}"
                )
            if not entity.geoms:
                empty_rows: list[list[int]] = [[] for _ in range(self._num_envs)]
                if (
                    record.get("geom_contact_masks") != empty_rows
                    or record.get("geom_friction") != empty_rows
                ):
                    raise self._worker_error(
                        "worker native geometry values are malformed for entity " + entity.name
                    )
                masks = np.empty((self._num_envs, 0, 2), dtype=np.int32)
                friction = np.empty((self._num_envs, 0, 3), dtype=np.float32)
            else:
                try:
                    masks = np.asarray(record.get("geom_contact_masks"), dtype=np.int32)
                    friction = np.asarray(record.get("geom_friction"), dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise self._worker_error(
                        f"worker native geom_contact_masks or geom_friction is malformed "
                        f"for entity {entity.name}"
                    ) from exc
                expected_mask_shape = (self._num_envs, len(entity.geoms), 2)
                expected_friction_shape = (self._num_envs, len(entity.geoms), 3)
                if masks.shape != expected_mask_shape or not np.isin(masks, (0, 1)).all():
                    raise self._worker_error(
                        f"worker native geom_contact_masks are malformed for entity "
                        f"{entity.name}: got shape {masks.shape}, expected "
                        f"{expected_mask_shape}"
                    )
                if (
                    friction.shape != expected_friction_shape
                    or not np.isfinite(friction).all()
                    or np.any(friction < 0.0)
                ):
                    raise self._worker_error(
                        f"worker native geom_friction is malformed for entity {entity.name}: "
                        f"got shape {friction.shape}, expected {expected_friction_shape}"
                    )
                if not np.all(masks == masks[0]):
                    raise self._worker_error(
                        "worker native geom_contact_masks vary across environments for entity "
                        + entity.name
                    )
            result.append((entity, public_offset, masks, friction))
            public_offset += len(entity.geoms)
        if public_offset != scene.layout.ngeom:
            raise self._worker_error("worker native geometry does not cover the frozen layout")
        return tuple(result)

    def get_geom_names(self) -> tuple[str, ...]:
        """Return qualified geometry names in frozen public geometry order."""
        if self._entity_scene is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not expose geom names")
        return tuple(
            entity.name + "/" + geom.name
            for entity in self._entity_scene.layout.entities
            for geom in entity.geoms
        )

    def get_geom_body_ids(self) -> np.ndarray:
        """Return owning public body IDs in frozen geometry order."""
        if self._entity_scene is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not expose geom body ids")
        body_ids = np.empty(self._entity_scene.layout.ngeom, dtype=np.int32)
        offset = 0
        for entity in self._entity_scene.layout.entities:
            owners = dict(zip(entity.body_names, entity.body_ids, strict=True))
            for geom in entity.geoms:
                body_ids[offset] = owners[geom.body_name]
                offset += 1
        return body_ids

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized native collision state in public geometry order."""
        if self._entity_scene is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not expose geom contact masks"
            )
        scene = self._require_mapped_entity_scene()
        records = self._validated_native_geometry_records()
        contype = np.empty(scene.layout.ngeom, dtype=np.int32)
        conaffinity = np.empty(scene.layout.ngeom, dtype=np.int32)
        for entity, offset, masks, _friction in records:
            rows = masks[0] if len(entity.geoms) else np.empty((0, 2), dtype=np.int32)
            contype[offset : offset + len(entity.geoms)] = rows[:, 0]
            conaffinity[offset : offset + len(entity.geoms)] = rows[:, 1]
        return contype, conaffinity

    def get_geom_friction(self) -> np.ndarray:
        """Return native effective per-environment friction in public order.

        Columns are IsaacSim ``[static_friction, dynamic_friction, 0]``. PhysX
        has no native equivalent for MuJoCo's torsional/rolling coefficients.
        """
        if self._entity_scene is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not expose geom friction")
        scene = self._require_mapped_entity_scene()
        records = self._validated_native_geometry_records()
        result = np.empty(
            (self._num_envs, scene.layout.ngeom, 3), dtype=np.float32
        )
        for entity, offset, _masks, friction in records:
            result[:, offset : offset + len(entity.geoms), :] = friction
        return result

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
