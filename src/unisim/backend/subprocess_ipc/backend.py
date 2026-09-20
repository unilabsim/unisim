"""Shared host adapter for MJCF-backed subprocess physics workers.

This owner layer contains the pipe lifecycle, shared-memory slot allocation,
cold-path MJCF metadata, NumPy state views, selected reset transaction, and
fail-closed default capability surface used by both IsaacGym and IsaacSim.
Runtime discovery, worker physics, and optional rendering stay in the sibling
backend adapters.

Bulk state crosses the process boundary only through shared memory. Control
messages use the canonical Python-3.8-compatible protocol, and hot-path getters
read the cache published by the last STEP/SET_STATE barrier without touching
assets or probing runtime-private objects.
"""

from __future__ import annotations

import atexit
import logging
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from unisim.backend.base import (
    _NATIVE_RENDERER_PLAY_CAPABILITIES,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    CameraCfg,
    RenderClosedError,
    SimBackend,
    normalize_play_render_mode,
    unsupported_debug_overlay_error,
)
from unisim.backend.subprocess_ipc.scene_materialization import (
    PreparedWorkerScene,
    body_sphere_radii_close,
    full_state_reset_patches,
    prepare_worker_scene,
    validate_body_sphere_radii,
)
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    ResetRandomizationPayload,
)
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset, row_columns
from unisim.inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ConfigurationScope,
    Difference,
    ImportReport,
    compare_configuration,
)
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_layout import CompiledSceneLayout, EntityLayout
from unisim.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_mul_batched,
)

from . import protocol
from .playback import run_subprocess_playback
from .sensors import (
    KIND_CONTACT_FOUND,
    KIND_FRAMEPOS,
    KIND_FRAMEQUAT,
    KIND_FRAMEZAXIS,
    KIND_GYRO,
    KIND_LOCAL_LINVEL,
    SceneMetadata,
    SceneSensorSpec,
    UnsupportedSensorSpec,
    scan_scene_kinematics,
    scan_scene_metadata,
    scan_scene_metadata_with_kinematics,
)

logger = logging.getLogger(__name__)

# The protocol is owned by the shared subprocess layer.  Keep the path
# explicit because external workers cannot import the host package.
_PROTOCOL_PATH = Path(protocol.__file__).resolve()

_DEFAULT_WORKER_TIMEOUT_S = 120.0
_SHUTDOWN_TIMEOUT_S = 5.0
_STDERR_TAIL_BYTES = 4096

_ROOT_QPOS_DIM = 7
_ROOT_QVEL_DIM = 6

# PhysX clamps |force| <= dof effort; MJCF forcerange "0 0" (or absent) means
# unlimited, mapped to a finite stand-in (float32-safe) for the dof property.
_UNLIMITED_DOF_EFFORT = 1e20


def _display_available() -> bool:
    """Return whether a display is reachable for an interactive worker viewer."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _single_precision_equal(reported: Any, requested: Any) -> bool:
    """Match values that only differ by the engine's float32 storage rounding."""
    try:
        return float(np.float32(float(reported))) == float(np.float32(float(requested)))
    except (TypeError, ValueError):
        return False


def _normalize_camera_kwargs(
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
) -> dict[str, float]:
    """Validate the native camera profile before mapping its spherical offset."""
    camera = CameraCfg.from_kwargs(camera_kwargs)
    defaults = CameraCfg()
    unsupported = [
        name
        for name in (
            "cam_lookat",
            "cam_tracking",
            "cam_tracking_env_idx",
            "cam_tracking_extra_envs",
            "cam_fov",
        )
        if getattr(camera, name) != getattr(defaults, name)
    ]
    if unsupported:
        raise NotImplementedError(
            "Isaac native renderers do not support camera overrides: "
            + ", ".join(unsupported)
            + ". Capture follows the first entity in environment 0; only "
            "cam_distance, cam_elevation and cam_azimuth can be configured."
        )
    return {
        "distance": camera.cam_distance,
        "elevation_deg": -camera.cam_elevation,
        "azimuth_deg": camera.cam_azimuth,
    }


class SubprocessWorkerError(RuntimeError):
    """Raised when a physics worker fails, exits, or stops responding.

    Carries the worker-side traceback and/or the captured stderr tail so
    crashes inside an external runtime remain diagnosable from the host.
    """

    def __init__(
        self,
        message: str,
        *,
        worker_traceback: str | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.worker_traceback = worker_traceback
        self.stderr_tail = stderr_tail


@dataclass(frozen=True)
class SubprocessModelInfo:
    """Opaque backend-owned model metadata returned by the worker handshake."""

    num_dof: int
    num_bodies: int
    dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    gravity: tuple[float, float, float]
    use_gpu_pipeline: bool


def _release_worker(
    proc: "subprocess.Popen[bytes] | None",
    shm_handles: list[shared_memory.SharedMemory],
    stderr_file: Any,
) -> None:
    """Best-effort worker shutdown: SHUTDOWN handshake, then terminate, then kill.

    Called from ``close()``/``__del__``/``atexit`` so the worker never survives
    the host process or leaks shared-memory segments.
    """
    if proc is not None and proc.poll() is None:
        try:
            if proc.stdin is not None:
                protocol.send_message(cast(BinaryIO, proc.stdin), protocol.CMD_SHUTDOWN)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except Exception:
                    pass
    for handle in shm_handles:
        try:
            handle.close()
            handle.unlink()
        except Exception:
            pass
    if stderr_file is not None:
        try:
            stderr_file.close()
        except Exception:
            pass


def _read_exactly_with_deadline(stream: Any, size: int, deadline: float) -> bytes:
    """Read exactly ``size`` bytes from an unbuffered stream before ``deadline``.

    ``stream`` must be unbuffered (``Popen(..., bufsize=0)``) so ``select`` on
    the underlying fd cannot be desynchronized by read-ahead buffering.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise TimeoutError(f"timed out waiting for {size} bytes from worker")
        readable, _, _ = select.select([stream], [], [], wait)
        if not readable:
            raise TimeoutError(f"timed out waiting for {size} bytes from worker")
        chunk = stream.read(remaining)
        if not chunk:
            raise protocol.WorkerDisconnectedError(
                f"worker pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class MjcfSubprocessBackend(SimBackend):
    """Backend-neutral host adapter for one MJCF subprocess worker."""

    _BACKEND_TYPE = "subprocess"
    _BACKEND_LABEL = "subprocess"
    _WORKER_ERROR_CLS: type[SubprocessWorkerError] = SubprocessWorkerError
    _MODEL_INFO_CLS: type[SubprocessModelInfo] = SubprocessModelInfo
    _play_capabilities = _NATIVE_RENDERER_PLAY_CAPABILITIES

    def _worker_error(self, message: str, **kwargs: Any) -> SubprocessWorkerError:
        """Construct the concrete adapter's public worker error type."""
        return self._WORKER_ERROR_CLS(message, **kwargs)

    def _worker_entrypoint(self) -> Path:
        raise NotImplementedError(f"{self.__class__.__name__} must declare a worker entrypoint")

    def _protocol_entrypoint(self) -> Path:
        return _PROTOCOL_PATH

    def _resolve_worker_runtime(self) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} must resolve its worker runtime")

    def _build_worker_environment(self, runtime: Any) -> dict[str, str]:
        raise NotImplementedError(f"{self.__class__.__name__} must build its worker environment")

    def _runtime_payload(self, runtime: Any) -> dict[str, str]:
        del runtime
        return {}

    def _supports_fixed_variant_plans(self) -> bool:
        """Return whether this adapter's worker realizes fixed variant plans."""
        return False

    def _mapped_contact_force_sensor_count(self) -> int:
        """Return dedicated IsaacSim collision-pair force rows, if supported."""
        return 0

    def _worker_init_payload(self) -> dict[str, Any]:
        """Return backend-owned cold-path INIT options.

        Subprocess workers must receive any runtime mode that changes Kit
        startup before the first ``INIT`` handshake.  The shared adapter keeps
        this hook empty so IsaacGym and other workers retain their existing
        startup contract; IsaacSim overrides it for eval rendering.
        """
        return {}

    def _worker_configuration_requested(self) -> dict[str, Any]:
        """Return backend-owned INIT settings surfaced in the import report.

        The shared adapter has no extra worker settings; IsaacSim overrides
        this hook to publish its bounded PhysX solver request.
        """
        return {}

    def _worker_configuration_fields(
        self, effective: Mapping[str, Any], readback_fields: Container[str]
    ) -> list[ConfigurationField]:
        """Compare backend-owned INIT requests against the worker's report."""
        fields = []
        for key, requested in self._worker_configuration_requested().items():
            value = effective.get(key)
            difference: Difference = "unknown"
            reason = "Requested value or engine adoption could not be verified."
            if requested is not None and value is not None:
                if value == requested:
                    difference, reason = "exact", ""
                elif _single_precision_equal(value, requested):
                    difference = "approximate"
                    reason = (
                        "The engine stores this setting in single precision; the "
                        "readback matches the float32 rounding of the request."
                    )
                else:
                    difference = "overridden"
                    reason = "Engine readback differs from the host INIT request."
            verified = key in readback_fields and value is not None
            fields.append(
                ConfigurationField(
                    key,
                    requested,
                    value,
                    difference,
                    (
                        ConfigurationProvenance(
                            "adapter_setting", "Host INIT worker configuration request"
                        ),
                        ConfigurationProvenance(
                            "engine_readback" if verified else "unverified",
                            (
                                "Worker native runtime readback"
                                if verified
                                else "Worker omitted an engine readback"
                            ),
                        ),
                    ),
                    reason=reason,
                )
            )
        return fields

    # Same column-stability contract as the other backends: non-applicable
    # sub-steps report 0.0.
    _SET_STATE_TIMING_ZERO_KEYS = (
        "set_state_mask_ms",
        "set_state_data_slice_ms",
        "set_state_data_reset_ms",
        "set_state_clear_forces_ms",
        "set_state_geom_overrides_ms",
        "set_state_reset_rand_ms",
        "set_state_set_dof_vel_ms",
        "set_state_set_dof_pos_ms",
        "set_state_actuator_ctrl_ms",
        "set_state_forward_kinematic_ms",
        "set_state_refresh_pose_cache_ms",
        "set_state_invalidate_velocity_ms",
        "set_state_qpos_convert_ms",
        "set_state_pool_reset_ms",
        "set_state_state_scatter_ms",
    )

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        device_id: int | None = None,
        worker_timeout_s: float | None = None,
        worker_command: list[str] | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        require_scene_composition_support(scene, self._BACKEND_TYPE)
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"{self.__class__.__name__} does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if worker_command is not None and (
            not isinstance(worker_command, list)
            or not worker_command
            or not all(isinstance(part, str) for part in worker_command)
        ):
            raise TypeError(
                "worker_command must be a non-empty list of strings or None, "
                f"got {worker_command!r}"
            )
        if scene.fragment_files and not scene.entity_assets:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} backend does not compose MuJoCo scene fragments; provide a "
                "self-contained MJCF scene through scene.model_file"
            )
        if scene.terrain is not None:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} backend does not support generated terrain scenes yet"
            )
        if scene.fixed_variant_plan is not None:
            scene.fixed_variant_plan.validate(int(num_envs))
            if not self._supports_fixed_variant_plans():
                raise NotImplementedError(
                    f"{self._BACKEND_LABEL} backend does not support fixed variant plans"
                )

        self._entity_scene: PreparedWorkerScene | None = None
        self._entity_source_scene = scene
        if scene.entity_assets:
            self._entity_scene = prepare_worker_scene(scene, int(num_envs), float(sim_dt))
            scene = replace(
                scene,
                model_file=self._entity_scene.owner.model_file,
                entity_assets=(),
                entity_variant=None,
                fragment_files=[],
            )
        self._stale_body_ids: set[int] = set()
        self._scene = scene
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        self._device_id = 0 if device_id is None else int(device_id)
        self._worker_timeout_s = (
            _DEFAULT_WORKER_TIMEOUT_S if worker_timeout_s is None else float(worker_timeout_s)
        )
        if self._worker_timeout_s <= 0.0:
            raise ValueError(f"worker_timeout_s must be positive, got {self._worker_timeout_s!r}")
        self._worker_command = list(worker_command) if worker_command is not None else None
        self.backend_type = self._BACKEND_TYPE
        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None

        # Everything below is materialized lazily in materialize().
        self._proc: subprocess.Popen[bytes] | None = None
        self._shm_handles: dict[str, shared_memory.SharedMemory] = {}
        self._slots: dict[str, np.ndarray] = {}
        self._stderr_file: Any = None
        self._worker_dead_error: SubprocessWorkerError | None = None
        self._model_info: SubprocessModelInfo | None = None
        self._scene_metadata: SceneMetadata | None = None
        self._scene_kinematics: dict[str, Any] | None = None
        self._initial_qpos: np.ndarray | None = None
        self._initial_qpos_resolved = False
        self._fixed_variant_plan = scene.fixed_variant_plan
        self._fixed_variant_metadata: tuple[SceneMetadata, ...] | None = None
        self._fixed_variant_kinematics: tuple[dict[str, Any], ...] | None = None
        self._variant_initial_qpos: tuple[np.ndarray | None, ...] | None = None
        self._variant_initial_qpos_resolved = False
        self._sensor_map: dict[str, tuple[SceneSensorSpec, int]] = {}
        self._body_id_by_name: dict[str, int] = {}
        self._dof_id_by_name: dict[str, int] = {}
        self._base_body_id = 0
        self._closed = False
        # Native rendering state (worker-owned viewer/camera; see the play
        # contract section below).  ``_render_config`` pins the first
        # init_renderer(headless, capture) pair like the Motrix backend.
        self._graphics_enabled = False
        self._render_config: tuple[bool, bool] | None = None
        self._viewer_open = False
        self._capture_ready = False
        # Native playback dimensions are fixed for one worker lifetime.  The
        # IsaacSim specialization overwrites these from EnvCfg before INIT;
        # IsaacGym keeps the historical 1280x720 defaults.
        self._render_width = 1280
        self._render_height = 720
        self._bind_entity_query_maps()

    def _bind_entity_query_maps(self) -> None:
        """Freeze owner names and packed columns before worker construction."""
        self._entity_query_map: dict[str, tuple[EntityLayout, int]] = {}
        self._entity_query_names: tuple[str, ...] = ()
        self._entity_dof_qpos_columns = np.empty(0, dtype=np.intp)
        self._entity_dof_qvel_columns = np.empty(0, dtype=np.intp)
        self._entity_primary_index = 0
        if self._entity_scene is None:
            return
        entities = self._entity_scene.layout.entities
        self._entity_query_map = {
            entity.name: (entity, index) for index, entity in enumerate(entities)
        }
        self._entity_query_names = tuple(self._entity_query_map)
        self._entity_dof_qpos_columns = np.asarray(
            [i for entity in entities for joint in entity.joints for i in joint.qpos_indices],
            dtype=np.intp,
        )
        self._entity_dof_qvel_columns = np.asarray(
            [i for entity in entities for joint in entity.joints for i in joint.qvel_indices],
            dtype=np.intp,
        )
        if self._base_name is None:
            self._entity_primary_index = next(
                (index for index, entity in enumerate(entities) if entity.actuator_indices), 0
            )
        else:
            matched = [
                index
                for index, entity in enumerate(entities)
                if self._base_name in (entity.name, entity.name + "/" + entity.root_body)
            ]
            if not matched:
                raise ValueError(f"base_name {self._base_name!r} does not name an entity root")
            self._entity_primary_index = matched[0]

    def _entity_query(self, name: str) -> tuple[EntityLayout, int]:
        if self._entity_scene is None:
            self.get_scene_layout()  # Retain unsupported-interface diagnostics.
        try:
            return self._entity_query_map[name]
        except KeyError:
            raise ValueError(f"unknown scene entity {name!r}") from None

    # ------------------------------------------------------------------ #
    # Worker lifecycle (cold path)
    # ------------------------------------------------------------------ #

    def get_scene_layout(self) -> CompiledSceneLayout:
        if self._entity_scene is None:
            return super().get_scene_layout()
        return self._entity_scene.layout

    def get_entity_names(self) -> tuple[str, ...]:
        if self._entity_scene is None:
            self.get_scene_layout()
        return self._entity_query_names

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> Mapping[str, np.ndarray]:
        from unisim.entity_state import selected_state_rows

        owner, index = self._entity_query(entity)
        ids = selected_state_rows(env_ids, self._num_envs)
        assert self._entity_scene is not None
        return entity_state_snapshot(
            owner,
            self._entity_scene.qpos[ids],
            self._entity_scene.qvel[ids],
            self._entity_scene.roots[ids, index],
        )

    def _primary_entity_index(self) -> int:
        return self._entity_primary_index

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        owner, index = self._entity_query(entity)
        self._require_state("entity state read")
        return entity_state_snapshot(
            owner,
            self._slots["qpos"],
            self._slots["qvel"],
            self._slots["entity_root_state"][:, index],
        )

    def get_state(self, fields: tuple[str, ...] | str | None = None) -> Mapping[str, np.ndarray]:
        if self._entity_scene is None:
            requested = (
                ("qpos", "qvel")
                if fields is None
                else ((fields,) if isinstance(fields, str) else tuple(fields))
            )
            result = dict(super().get_state(tuple(name for name in requested if name != "ctrl")))
            if "ctrl" in requested:
                self._require_state("control snapshot")
                result["ctrl"] = self._slots["ctrl"].copy()
            return result
        self._require_state("generalized state read")
        requested = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else fields)
        )
        if set(requested) - {"qpos", "qvel", "ctrl"}:
            raise KeyError("unknown generalized state field")
        return {name: self._slots[name].copy() for name in requested}

    def _entity_joint_indices(
        self, names: Sequence[str], *, velocity: bool, packed: bool
    ) -> np.ndarray:
        layout = self.get_scene_layout()
        selected = layout.get_joint_layouts(names)
        attr = "qvel_indices" if velocity else "qpos_indices"
        result = [i for joint in selected for i in getattr(joint, attr)]
        if packed:
            columns = [
                i
                for entity in layout.entities
                for joint in entity.joints
                for i in getattr(joint, attr)
            ]
            result = [columns.index(i) for i in result]
        return np.asarray(result, dtype=np.int32)

    def reset_entities(self, request: SceneResetRequest) -> None:
        controls = None
        if request.restore_default_controls:
            layout = self.get_scene_layout()
            bound = layout.validate_reset(request, num_envs=self._num_envs)
            self._require_state("entity default controls")
            assert self._entity_scene is not None
            rows = np.asarray(request.env_ids, dtype=np.intp)
            controls = self._slots["ctrl"][rows].copy()
            defaults = np.asarray(self._entity_scene.payload["initial_ctrl"], dtype=controls.dtype)
            for item in bound.patches:
                root_changed = (
                    item.patch.root_pose is not None or item.patch.root_velocity is not None
                )
                joints = {joint.name for joint in item.joints}
                columns = [
                    index
                    for index, joint in zip(
                        item.entity.actuator_indices, item.entity.actuator_joint_names
                    )
                    if root_changed or joint in joints
                ]
                controls[:, columns] = defaults[row_columns(rows, columns)]
        self._commit_entity_reset(request, controls)

    def _commit_entity_reset(
        self,
        request: SceneResetRequest,
        control_values: np.ndarray | None = None,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        layout = self.get_scene_layout()
        # Do not even materialize a native worker for a malformed request.
        layout.validate_reset(request, num_envs=self._num_envs)
        self._require_state("entity reset")
        prepared = prepare_scene_reset(
            layout,
            request,
            self._slots["qpos"],
            self._slots["qvel"],
            self._slots["entity_root_state"],
        )
        count = len(prepared.env_ids)
        for slot, values in (
            ("reset_env_ids", prepared.env_ids),
            ("reset_qpos", prepared.qpos),
            ("reset_qvel", prepared.qvel),
            ("reset_entity_root_state", prepared.roots),
        ):
            np.copyto(self._slots[slot][:count], values)
        for slot, values in (
            ("reset_qpos_mask", prepared.qpos_mask),
            ("reset_qvel_mask", prepared.qvel_mask),
            ("reset_root_mask", prepared.root_mask),
        ):
            np.copyto(self._slots[slot], values)
        randomization_wire: dict[str, Any] = {}
        if randomization is not None:
            if randomization.body_mass is not None:
                randomization_wire["body_mass"] = randomization.body_mass.tolist()
            if randomization.geom_friction is not None:
                randomization_wire["geom_friction"] = randomization.geom_friction.tolist()
        response = self._request(
            protocol.CMD_RESET_ENTITIES,
            {
                "count": count,
                "entity_names": list(prepared.entity_names),
                **(
                    {"control_values": control_values.tolist()}
                    if control_values is not None
                    else {}
                ),
                **(
                    {"randomization": randomization_wire}
                    if randomization_wire
                    else {}
                ),
            },
            expect=protocol.CMD_READY,
        )
        if randomization is not None:
            self._consume_entity_reset_randomization(response)
        if self._BACKEND_TYPE == "isaacgym":
            for patch in request.patches:
                if any(
                    value is not None
                    for value in (
                        patch.joint_positions,
                        patch.joint_velocities,
                        patch.root_pose,
                        patch.root_velocity,
                    )
                ):
                    entity = layout.get_entity(patch.entity)
                    self._stale_body_ids.update(
                        bid
                        for name, bid in zip(entity.body_names, entity.body_ids)
                        if name != entity.root_body
                    )

    def _set_mapped_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        layout = self.get_scene_layout()
        ids = np.asarray(env_indices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("env_indices must contain one-dimensional integer IDs")
        if (
            ids.dtype.kind == "i"
            and (np.any(ids < 0) or np.any(ids >= self._num_envs))
        ) or (
            ids.dtype.kind == "u"
            and (np.any(ids >= self._num_envs))
        ):
            raise ValueError("env_indices must be in environment range")
        if np.unique(ids).size != ids.size:
            raise ValueError("env_indices must not contain duplicate rows")
        if qpos.shape != (len(ids), layout.nq) or qvel.shape != (len(ids), layout.nv):
            raise ValueError("full state shape differs from scene layout")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("full state requires finite values")
        if not len(ids):
            if randomization is not None and not randomization.is_empty():
                raise ValueError("selected property mutation requires at least one environment")
            return {"timing": {}}
        validated_randomization = self._validated_mapped_reset_randomization(randomization, ids)
        patches = full_state_reset_patches(layout, qpos, qvel)
        if patches or validated_randomization is not None:
            self._commit_entity_reset(
                SceneResetRequest(tuple(int(i) for i in ids), patches),
                randomization=validated_randomization,
            )
        return {"timing": {}}

    def _validated_mapped_reset_randomization(
        self, randomization: ResetRandomizationPayload | None, rows: np.ndarray
    ) -> ResetRandomizationPayload | None:
        if randomization is None or randomization.is_empty():
            return None
        requested = ", ".join(sorted(randomization.requested_terms()))
        raise NotImplementedError(
            f"{self._BACKEND_LABEL} does not support reset domain randomization terms: "
            f"{requested}."
        )

    def _consume_entity_reset_randomization(self, response: Any) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reset property mutation readback"
        )

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if self._entity_scene is None:
            return super().reset(env_ids)
        ids = np.arange(self._num_envs) if env_ids is None else np.asarray(env_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("reset env_ids must be one-dimensional integers")
        if not len(ids):
            return
        if np.any(ids >= self._num_envs) or len(set(ids)) != len(ids):
            raise ValueError("reset env_ids must be distinct and in range")
        prepared = self._entity_scene
        patches = list(
            full_state_reset_patches(prepared.layout, prepared.qpos[ids], prepared.qvel[ids])
        )
        for index, entity in enumerate(prepared.layout.entities):
            if entity.root_mode == "kinematic":
                patches.append(
                    EntityStatePatch(entity.name, root_pose=prepared.roots[ids, index, :7])
                )
        if patches:
            self._commit_entity_reset(
                SceneResetRequest(tuple(int(i) for i in ids), tuple(patches)),
                np.asarray(prepared.payload["initial_ctrl"], dtype=np.float32)[ids],
            )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        if self._entity_scene is None:
            return super().get_playback_model(env_index)
        plan = self._entity_scene.owner.variant_plan
        if env_index is None:
            if plan is not None:
                raise ValueError("entity-variant playback requires env_index")
            return self._entity_scene.owner.model_file
        if isinstance(env_index, bool) or not isinstance(env_index, int):
            raise TypeError("env_index must be an integer")
        if not 0 <= env_index < self._num_envs:
            raise IndexError("playback environment is out of range")
        return (
            self._entity_scene.owner.model_file
            if plan is None
            else plan.variants[int(plan.assignment[env_index])].model_file
        )

    def get_physics_state(self) -> np.ndarray:
        if self._entity_scene is None:
            return super().get_physics_state()
        self._require_state("physics snapshot")
        # Native rendering uses the worker scene. Offline reconstruction requires
        # an explicit time and mocap channel, which this profile does not yet expose.
        raise NotImplementedError(
            "mapped worker physics snapshots are not yet available; use native playback"
        )

    def _bind_scene_metadata(self, meta: dict[str, Any]) -> None:
        assert self._entity_scene is not None
        layout = self._entity_scene.layout
        if self._BACKEND_TYPE == "isaacsim":
            # Preserve the rendering/clone checks in the subclass. Its final
            # super call dispatches here only after setting this reentry flag.
            if not getattr(self, "_binding_scene_metadata", False):
                self._binding_scene_metadata = True
                try:
                    self._bind_model_metadata(meta)
                finally:
                    self._binding_scene_metadata = False
                return
        layout.require_same_layout(CompiledSceneLayout.from_dict(meta.get("scene_layout")))
        actual = meta.get("scene_entities_actual")
        if not isinstance(actual, list) or len(actual) != len(layout.entities):
            raise self._worker_error("worker omitted actual entity materialization records")
        records = {record["name"]: record for record in actual}
        if len(records) != len(actual) or set(records) != set(self.get_entity_names()):
            raise self._worker_error("worker entity identity names do not match declaration")
        for entry in self._entity_scene.payload["scene_entities"]:
            record = records[entry["name"]]
            assignment = np.asarray(record.get("assignment"))
            if assignment.dtype.kind not in "iu" or not np.array_equal(
                assignment, entry["assignment"]
            ):
                raise self._worker_error("worker changed entity assignment: " + entry["name"])
            expected = np.asarray([entry["variants"][i]["body_mass"] for i in entry["assignment"]])
            masses = np.asarray(record.get("body_mass"), dtype=float)
            if masses.shape != expected.shape or not np.isfinite(masses).all():
                raise self._worker_error(
                    "worker entity body masses are malformed: " + entry["name"]
                )
            if entry["root_mode"] == "floating" and not np.allclose(
                masses, expected, rtol=1e-4, atol=1e-6
            ):
                raise self._worker_error(
                    "native entity body masses differ from compiled source: " + entry["name"]
                )
            reported_radii = record.get("body_sphere_radii")
            if not isinstance(reported_radii, list) or len(reported_radii) != self._num_envs:
                raise self._worker_error(
                    "worker entity sphere radii are malformed: " + entry["name"]
                )
            for row in reported_radii:
                try:
                    validate_body_sphere_radii(row, len(entry["variants"][0]["body_names"]))
                except ValueError as exc:
                    raise self._worker_error(
                        "worker entity sphere radii are malformed: " + entry["name"]
                    ) from exc
            expected_radii = [
                entry["variants"][i]["body_sphere_radii"] for i in entry["assignment"]
            ]
            if not all(
                body_sphere_radii_close(actual, expected, rtol=2e-6, atol=1e-8)
                for actual, expected in zip(reported_radii, expected_radii)
            ):
                raise self._worker_error(
                    "native entity sphere radii differ from compiled source: " + entry["name"]
                )
        body_names = [""] * layout.nbody
        for entity in layout.entities:
            for name, index in zip(entity.body_names, entity.body_ids, strict=True):
                body_names[index] = entity.name + "/" + name
        joints = tuple(
            entity.name + "/" + joint.name for entity in layout.entities for joint in entity.joints
        )
        gravity = tuple(float(i) for i in meta["gravity"])
        if len(gravity) != 3 or not np.isfinite(gravity).all():
            raise self._worker_error("malformed worker gravity")
        self._model_info = self._MODEL_INFO_CLS(
            len(joints),
            layout.nbody,
            joints,
            tuple(body_names),
            gravity,
            bool(meta.get("use_gpu_pipeline", False)),
        )
        self._body_id_by_name = {name: i for i, name in enumerate(body_names) if name}
        self._dof_id_by_name = {name: i for i, name in enumerate(joints)}
        primary = layout.entities[self._primary_entity_index()]
        self._base_body_id = primary.body_ids[primary.body_names.index(primary.root_body)]
        self._native_entity_records = records

    def _capture_entity_report(self, meta: dict[str, Any]) -> None:
        assert self._entity_scene is not None
        fields = []
        envelope = meta.get("configuration_report", {})
        if (
            not isinstance(envelope, dict)
            or type(envelope.get("schema_version")) is not int
            or envelope["schema_version"] != 1
            or not isinstance(envelope.get("effective"), dict)
        ):
            raise self._worker_error("unsupported or malformed entity configuration report schema")
        effective = envelope.get("effective", {})
        for entry in self._entity_scene.payload["scene_entities"]:
            record = self._native_entity_records[entry["name"]]
            fields.append(
                ConfigurationField(
                    "entity.identity",
                    entry["assignment"],
                    record["assignment"],
                    "exact",
                    (
                        ConfigurationProvenance("source", "Immutable entity assignment"),
                        ConfigurationProvenance(
                            "engine_readback", "Worker instance materialization audit"
                        ),
                    ),
                    ConfigurationScope(entity=entry["name"]),
                    reason="Assignment is separately checked against native instance parameters.",
                )
            )
            for index, variant in enumerate(entry["variants"]):
                rows = [
                    row for row, selected in enumerate(entry["assignment"]) if selected == index
                ]
                fields.append(
                    ConfigurationField(
                        "body_mass",
                        [variant["body_mass"] for _ in rows],
                        [record["body_mass"][row] for row in rows],
                        "unknown",
                        (
                            ConfigurationProvenance("source", entry["sources"][index]),
                            ConfigurationProvenance(
                                "engine_readback", "Worker per-instance body properties"
                            ),
                        ),
                        ConfigurationScope(
                            entity=entry["name"], env_ids=tuple(rows), variant=str(index)
                        ),
                        unit="kg",
                        reason=(
                            "Compiled source and per-instance native readback; "
                            "fixed roots may have infinite mass."
                        ),
                    )
                )
        for key in ("dt", "gravity", "solver", "integrator", "collision_filter"):
            requested = (
                self._sim_dt
                if key == "dt"
                else self._entity_scene.payload.get(key)
            )
            value = effective.get(key, meta.get(key))
            fields.append(
                ConfigurationField(
                    key,
                    requested,
                    value,
                    "unknown",
                    (ConfigurationProvenance("adapter_setting", "Native worker scene profile"),),
                )
            )
        entity_readback = set(envelope.get("engine_readback") or ())
        fields.extend(self._worker_configuration_fields(effective, entity_readback))
        self._import_report = ImportReport(
            self.backend_type, tuple(fields), lifecycle="materialization"
        )

    def materialize(self) -> None:
        """Spawn the worker, run the handshake, and bind shared-memory slots.

        Idempotent. Called lazily by the first state/metadata access, so env
        constructors that read shapes before the explicit lifecycle point work
        like they do on the MuJoCo backend. A closed backend cannot be
        materialized again.
        """
        if self._proc is not None:
            return
        if self._closed:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} backend is closed and cannot be materialized again"
            )
        # Parent-side MJCF metadata and the #141 FK tree share one parse here.
        # Metadata-only access before materialization remains permissive.
        if self._entity_scene is None:
            self._get_scene_metadata_with_kinematics()
            if self._fixed_variant_plan is not None:
                self._get_fixed_variant_metadata_with_kinematics()
            self._resolve_initial_qpos()
        runtime: Any = None
        if self._worker_command is None:
            runtime = self._resolve_worker_runtime()
            command = [str(runtime.python), str(self._worker_entrypoint())]
            env = self._build_worker_environment(runtime)
        else:
            command = list(self._worker_command)
            env = None
        runtime_payload = self._runtime_payload(runtime) if runtime is not None else {}
        worker_init_payload = self._worker_init_payload()
        if not isinstance(worker_init_payload, dict):
            raise TypeError(
                f"{self._BACKEND_LABEL} _worker_init_payload() must return a dict, "
                f"got {type(worker_init_payload).__name__}"
            )

        self._stderr_file = tempfile.TemporaryFile(
            mode="w+b", prefix=f"{self._BACKEND_LABEL}_worker_stderr_"
        )
        try:
            self._proc = subprocess.Popen(
                [*command, "--protocol", str(self._protocol_entrypoint())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                bufsize=0,
                env=env,
            )
        except OSError as exc:
            raise self._worker_error(
                f"failed to spawn {self._BACKEND_LABEL} worker {command}: {exc}"
            ) from exc

        try:
            fixed_variant_payload = self._fixed_variant_init_payload()
            scene_payload = None if self._entity_scene is None else self._entity_scene.payload
            meta = self._request(
                protocol.CMD_INIT,
                {
                    "configuration_report_version": 1,
                    "model_file": str(Path(self._scene.model_file).expanduser()),
                    "num_envs": self._num_envs,
                    "sim_dt": self._sim_dt,
                    "device_id": self._device_id,
                    **runtime_payload,
                    **worker_init_payload,
                    **(
                        scene_payload
                        if scene_payload is not None
                        else {
                            "root_body_name": self._base_name
                            or self._get_scene_metadata().freejoint_body_name,
                            # Some importers do not preserve MJCF traversal order.
                            # Send the cold-path body contract explicitly so a worker
                            # can remap native link indices before publishing state.
                            "mjcf_body_names": list(self._get_scene_metadata().body_names),
                            "mjcf_joint_names": list(self._get_scene_metadata().joint_names),
                            # Kinematic tree for worker-side FK: PhysX cannot
                            # refresh link poses without stepping, so post-reset
                            # body state is overlaid with exact FK (#141).
                            "mjcf_kinematics": self._get_scene_kinematics(),
                            # Fixed variants carry their own per-source actuation and
                            # keyframe tables; the legacy single-model fields are omitted
                            # rather than duplicated (or allowed to conflict).
                            **(
                                {}
                                if fixed_variant_payload
                                else {
                                    "keyframe_qpos": (
                                        None
                                        if self._initial_qpos is None
                                        else [float(value) for value in self._initial_qpos]
                                    ),
                                    **self._position_actuation_payload(),
                                }
                            ),
                            **fixed_variant_payload,
                        }
                    ),
                },
                expect=protocol.CMD_META,
            )
            if self._entity_scene is None:
                self._bind_model_metadata(meta)
            else:
                self._bind_scene_metadata(meta)
            self._graphics_enabled = bool(meta.get("graphics_enabled", False))
            if self._entity_scene is None:
                self._validate_initial_keyframe()
            self._allocate_slots()
            self._request(
                protocol.CMD_ATTACH, {"slots": self._slot_specs()}, expect=protocol.CMD_READY
            )
            if self._entity_scene is not None:
                self._slots["ctrl"][:] = self._entity_scene.payload["initial_ctrl"]
                if self._BACKEND_TYPE == "isaacgym":
                    self._stale_body_ids.update(
                        bid
                        for entity in self._entity_scene.layout.entities
                        for name, bid in zip(entity.body_names, entity.body_ids)
                        if entity.joints and name != entity.root_body
                    )
            self._sensor_map = self._resolve_sensor_map()
            if self._entity_scene is None:
                self._capture_import_report(meta)
            else:
                self._capture_entity_report(meta)
            if self._base_name is not None and self._entity_scene is None:
                try:
                    self._base_body_id = self._body_id_by_name[self._base_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Base body {self._base_name!r} not found in {self._BACKEND_LABEL} model"
                    ) from exc
        except Exception:
            self.close()
            raise
        # Subprocess liveness cannot rely on __del__ alone during interpreter
        # teardown; the atexit hook is unregistered by close().
        atexit.register(self.close)

    def _capture_import_report(self, meta: dict[str, Any]) -> None:
        """Accept only versioned worker observations, never host XML as actual."""
        envelope = meta.get("configuration_report")
        if envelope is not None and (
            not isinstance(envelope, dict)
            or type(envelope.get("schema_version")) is not int
            or envelope.get("schema_version") != 1
        ):
            raise self._worker_error("unsupported worker configuration report schema version")
        effective = {} if envelope is None else envelope.get("effective", {})
        reports: list[ConfigurationField] = []
        variants = self._get_fixed_variant_metadata() or (self._get_scene_metadata(),)
        variant_env_ids: list[list[int]] = [[] for _ in variants]
        if self._fixed_variant_plan is not None:
            for env, variant in enumerate(self._fixed_variant_plan.assignment):
                variant_env_ids[int(variant)].append(env)
        for index, metadata in enumerate(variants):
            options = metadata.source_options
            requested: dict[str, Any] = {
                "solver": options.get("solver"),
                "integrator": options.get("integrator"),
                "dt": float(options["timestep"]) if "timestep" in options else None,
                "gravity": (
                    [float(v) for v in options["gravity"].split()] if "gravity" in options else None
                ),
                "body_mass": (
                    {"authored_inertials": list(metadata.source_inertials)}
                    if metadata.source_inertials
                    else None
                ),
                "body_inertia": (
                    {"authored_inertials": list(metadata.source_inertials)}
                    if metadata.source_inertials
                    else None
                ),
                "collision_filter": metadata.source_collision,
                "actuator_mapping": [
                    {"name": spec.name, "joint": spec.joint_name, "kp": spec.kp, "kv": spec.kv}
                    for spec in metadata.actuators
                ],
                "sensors": [
                    {"name": spec.name, "kind": spec.kind, "body": spec.body_name}
                    for spec in metadata.sensors.values()
                ],
            }
            scope = ConfigurationScope()
            if self._fixed_variant_plan is not None:
                scope = ConfigurationScope(
                    env_ids=tuple(variant_env_ids[index]),
                    variant=str(index),
                )
            scoped_effective = dict(effective)
            rows = tuple(range(self._num_envs)) if scope.env_ids is None else scope.env_ids
            for name, value in effective.items():
                if isinstance(value, dict) and any(k.startswith("per_env_") for k in value):
                    scoped_effective[name] = {
                        key: [data[row] for row in rows] if key.startswith("per_env_") else data
                        for key, data in value.items()
                    }
                    scoped_effective[name]["env_ids"] = list(rows)
            reports.extend(
                compare_configuration(
                    self.backend_type,
                    requested,
                    scoped_effective,
                    source=f"MJCF declarations resolved by cached scanner: {metadata.model_file}",
                    effective_source=(
                        "Worker construction settings (not native readback)"
                        if envelope is not None
                        else "Legacy worker supplied no report"
                    ),
                    effective_kind="adapter_setting",
                    scope=scope,
                    lifecycle="materialization",
                ).fields
            )
        readback = set() if envelope is None else set(envelope.get("engine_readback", ()))
        normalized = []
        for item in reports:
            changes: dict[str, Any] = {}
            if item.field in readback and item.effective is not None:
                changes["provenance"] = (
                    item.provenance[0],
                    ConfigurationProvenance("engine_readback", "Worker native runtime readback"),
                )
            if item.field == "dt" and item.difference == "overridden":
                changes.update(
                    reason="Explicit sim_dt constructor argument replaces the source timestep."
                )
            elif item.field in {"solver", "integrator"} and item.effective is not None:
                changes.update(
                    difference="unknown",
                    reason="MJCF and PhysX names denote different engine concepts; "
                    "equivalence has not been established.",
                )
            elif item.field == "gravity" and item.difference == "overridden":
                changes.update(
                    difference="approximate",
                    reason="Fixed worker gravity replaces the authored gravity value.",
                )
            elif item.field == "collision_filter" and item.effective is not None:
                changes.update(
                    difference="approximate",
                    reason="Worker disables self-collision globally; MJCF pair and "
                    "geom filtering semantics are not preserved.",
                )
            elif item.field in {"body_mass", "body_inertia"}:
                changes.update(
                    difference="unknown",
                    frame=(
                        "body-local center-of-mass inertia tensor"
                        if item.field == "body_inertia"
                        else item.frame
                    ),
                    reason="Authored inertial attributes are retained literally; importer "
                    "inference, defaults and native tensor equivalence are not resolved.",
                )
            elif item.field == "sensors" and envelope is not None:
                changes.update(
                    effective=[
                        {
                            "name": name,
                            "quantity": spec.kind,
                            "body": spec.body_name,
                            "dim": spec.dim,
                            "cache_offset": offset,
                        }
                        for name, (spec, offset) in self._sensor_map.items()
                    ],
                    difference="unknown",
                    provenance=(
                        item.provenance[0],
                        ConfigurationProvenance(
                            "adapter_setting", "Resolved host sensor map over worker state cache"
                        ),
                    ),
                    reason="Host quantities are recorded; equivalence to native MJCF sensor "
                    "filtering and contact semantics is not asserted.",
                )
            elif item.field == "actuator_mapping" and item.effective is not None:
                changes.update(
                    difference="unknown",
                    reason="Source actuator and native drive tables use different "
                    "representations; native drive adoption is recorded.",
                )
            normalized.append(replace(item, **changes) if changes else item)
        normalized.extend(self._worker_configuration_fields(effective, readback))
        self._import_report = ImportReport(
            self.backend_type, tuple(normalized), lifecycle="materialization"
        )

    def _get_scene_metadata(self) -> SceneMetadata:
        """Return the parent-side MJCF scan, scanning lazily on first access.

        This is pure XML metadata — no worker handshake is required, matching
        the MuJoCo backend where the model (and thus keyframes) is available
        right after construction.  ``materialize()`` reuses this cache.
        """
        if self._scene_metadata is None:
            self._scene_metadata = scan_scene_metadata(
                str(Path(self._scene.model_file).expanduser()),
                backend_label=self._BACKEND_LABEL,
                resolve_actuators=self._entity_scene is None,
            )
        return self._scene_metadata

    def _get_scene_metadata_with_kinematics(self) -> tuple[SceneMetadata, dict[str, Any]]:
        """Parse the legacy source once for both public metadata and FK tables."""
        if self._scene_metadata is None or self._scene_kinematics is None:
            if self._scene_metadata is None and self._scene_kinematics is None:
                metadata, kinematics = scan_scene_metadata_with_kinematics(
                    str(Path(self._scene.model_file).expanduser()),
                    backend_label=self._BACKEND_LABEL,
                )
            else:
                metadata = self._get_scene_metadata()
                kinematics = scan_scene_kinematics(
                    str(Path(self._scene.model_file).expanduser()),
                    backend_label=self._BACKEND_LABEL,
                )
            self._validate_kinematics(kinematics, metadata)
            self._scene_metadata = metadata
            self._scene_kinematics = kinematics
        assert self._scene_metadata is not None
        assert self._scene_kinematics is not None
        return self._scene_metadata, self._scene_kinematics

    def _get_fixed_variant_metadata(self) -> tuple[SceneMetadata, ...]:
        """Return the per-variant MJCF scans required by the public layout."""
        if self._fixed_variant_plan is None:
            return ()
        if self._fixed_variant_metadata is None:
            assert self._fixed_variant_plan is not None
            metadata = tuple(
                scan_scene_metadata(
                    str(Path(variant.model_file).expanduser()),
                    backend_label=self._BACKEND_LABEL,
                    resolve_actuators=self._entity_scene is None,
                )
                for variant in self._fixed_variant_plan.variants
            )
            self._validate_fixed_variant_metadata(metadata)
            self._fixed_variant_metadata = metadata
        return self._fixed_variant_metadata

    def _get_fixed_variant_metadata_with_kinematics(self) -> tuple[SceneMetadata, ...]:
        """Parse every legacy variant once for metadata and FK tables."""
        if self._fixed_variant_metadata is None or self._fixed_variant_kinematics is None:
            assert self._fixed_variant_plan is not None
            if self._fixed_variant_metadata is None and self._fixed_variant_kinematics is None:
                pairs = tuple(
                    scan_scene_metadata_with_kinematics(
                        str(Path(variant.model_file).expanduser()),
                        backend_label=self._BACKEND_LABEL,
                    )
                    for variant in self._fixed_variant_plan.variants
                )
                metadata = tuple(item[0] for item in pairs)
                kinematics = tuple(item[1] for item in pairs)
            else:
                metadata = self._get_fixed_variant_metadata()
                kinematics = tuple(
                    scan_scene_kinematics(
                        str(Path(variant.model_file).expanduser()),
                        backend_label=self._BACKEND_LABEL,
                    )
                    for variant in self._fixed_variant_plan.variants
                )
            self._validate_fixed_variant_metadata(metadata)
            for item, tables in zip(metadata, kinematics):
                self._validate_kinematics(tables, item)
            self._fixed_variant_metadata = metadata
            self._fixed_variant_kinematics = kinematics
        assert self._fixed_variant_metadata is not None
        return self._fixed_variant_metadata

    def _validate_fixed_variant_metadata(
        self,
        variants: tuple[SceneMetadata, ...],
    ) -> None:
        """Require every variant to preserve the canonical public layout."""
        assert self._fixed_variant_plan is not None
        canonical = self._get_scene_metadata()
        expected_fields = (
            ("joint_names", canonical.joint_names),
            ("body_names", canonical.body_names),
            ("sensors", canonical.sensors),
            (
                "actuated joint names",
                tuple(spec.joint_name for spec in canonical.actuators),
            ),
        )
        for index, metadata in enumerate(variants):
            source = self._fixed_variant_plan.variants[index].model_file
            for field, expected in expected_fields:
                actual = (
                    tuple(spec.joint_name for spec in metadata.actuators)
                    if field == "actuated joint names"
                    else getattr(metadata, field)
                )
                if actual != expected:
                    raise ValueError(
                        f"fixed variant {source!r} changes public {field}: "
                        f"canonical={expected}, variant={actual}; IsaacGym requires "
                        "one actor slot with identical dof/body name order per environment"
                    )

    def _get_scene_kinematics(self) -> dict[str, Any]:
        """Return the parent-side kinematic-tree scan, scanned lazily once."""
        if self._scene_kinematics is None:
            self._get_scene_metadata_with_kinematics()
        assert self._scene_kinematics is not None
        return self._scene_kinematics

    def _get_fixed_variant_kinematics(self) -> tuple[dict[str, Any], ...]:
        """Return per-variant kinematic trees aligned with the variant plan."""
        if self._fixed_variant_plan is None:
            return ()
        if self._fixed_variant_kinematics is None:
            self._get_fixed_variant_metadata_with_kinematics()
        assert self._fixed_variant_kinematics is not None
        return self._fixed_variant_kinematics

    def _validate_kinematics(self, kinematics: dict[str, Any], metadata: SceneMetadata) -> None:
        """Pin the FK payload to the same public column contract as the metadata scan."""
        if kinematics["body_names"] != list(metadata.body_names) or kinematics[
            "joint_names"
        ] != list(metadata.joint_names):
            raise self._worker_error(
                f"{self._BACKEND_LABEL} kinematics scan disagrees with the metadata scan:\n"
                f"  metadata bodies: {metadata.body_names}\n"
                f"  kinematics bodies: {kinematics['body_names']}"
            )

    def _resolve_initial_qpos(self) -> np.ndarray | None:
        """Lazily select the scene keyframe used as the backend default state."""
        if not self._initial_qpos_resolved:
            self._initial_qpos = self._select_initial_keyframe(self._get_scene_metadata())
            self._initial_qpos_resolved = True
        return self._initial_qpos

    def _resolve_variant_initial_qpos(self) -> tuple[np.ndarray | None, ...]:
        """Select the task-initial qpos independently for every variant."""
        if self._fixed_variant_plan is None:
            return ()
        if not self._variant_initial_qpos_resolved:
            variants = self._get_fixed_variant_metadata()
            canonical_has_initial = self._resolve_initial_qpos() is not None
            values: list[np.ndarray | None] = []
            for index, metadata in enumerate(variants):
                source = self._fixed_variant_plan.variants[index].model_file
                selected = self._select_initial_keyframe(metadata)
                if canonical_has_initial and selected is None:
                    raise ValueError(
                        f"fixed variant {source!r} has no unique initial keyframe while "
                        "the canonical scene selects one"
                    )
                if not canonical_has_initial and selected is not None:
                    raise ValueError(
                        f"fixed variant {source!r} introduces a unique initial keyframe "
                        "while the canonical scene selects none"
                    )
                values.append(selected)
            self._variant_initial_qpos = tuple(values)
            self._variant_initial_qpos_resolved = True
        assert self._variant_initial_qpos is not None
        return self._variant_initial_qpos

    def _effective_initial_qpos(self) -> np.ndarray | None:
        """Return the default qpos of the first environment's assigned variant."""
        if self._fixed_variant_plan is None:
            return self._resolve_initial_qpos()
        values = self._resolve_variant_initial_qpos()
        variant = int(self._fixed_variant_plan.assignment[0])
        return values[variant]

    def _select_initial_keyframe(self, metadata: SceneMetadata) -> np.ndarray | None:
        """Pick the scene's task-initial keyframe (cold path).

        ``SceneCfg.default_keyframe_name`` wins when set; a scene with exactly
        one keyframe uses it implicitly (AGENTS.md: the keyframe is the task
        initial pose); a scene without (or with ambiguous) keyframes falls back
        to the all-zero qpos convention.  The selected qpos becomes the
        backend's default state, so ``get_default_qpos``/``get_default_dof_pos``
        match the post-INIT worker state.
        """
        name = self._scene.default_keyframe_name
        if name is not None:
            if name not in metadata.keyframes:
                available = ", ".join(sorted(metadata.keyframes))
                raise ValueError(
                    f"scene default_keyframe_name {name!r} not found in MJCF keyframes; "
                    f"available: {available}"
                )
            qpos = metadata.keyframes[name]
        elif len(metadata.keyframes) == 1:
            qpos = next(iter(metadata.keyframes.values()))
        else:
            return None
        if qpos.size != _ROOT_QPOS_DIM + len(metadata.joint_names):
            raise ValueError(
                f"keyframe qpos has {qpos.size} entries; expected "
                f"{_ROOT_QPOS_DIM + len(metadata.joint_names)} (7 root + "
                f"{len(metadata.joint_names)} joints in document order)"
            )
        return qpos.astype(np.float32, copy=True)

    def _validate_initial_keyframe(self) -> None:
        """Check the selected keyframe against the worker's actual dof count."""
        info = self._require_materialized()
        selected = (
            (self._initial_qpos,)
            if self._fixed_variant_plan is None
            else self._resolve_variant_initial_qpos()
        )
        expected = _ROOT_QPOS_DIM + info.num_dof
        for qpos in selected:
            if qpos is not None and qpos.size != expected:
                raise ValueError(
                    f"scene keyframe qpos has {qpos.size} entries; "
                    f"the {self._BACKEND_LABEL} "
                    f"asset exposes {info.num_dof} dofs, expected {expected}"
                )

    def _bind_model_metadata(self, meta: dict[str, Any]) -> None:
        if self._entity_scene is not None:
            self._bind_scene_metadata(meta)
            return
        num_dof = int(meta["num_dof"])
        num_bodies = int(meta["num_bodies"])
        dof_names = tuple(str(name) for name in meta["dof_names"])
        body_names = tuple(str(name) for name in meta["body_names"])
        if len(dof_names) != num_dof or len(body_names) != num_bodies:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker metadata is inconsistent: "
                f"num_dof={num_dof} vs {len(dof_names)} names, "
                f"num_bodies={num_bodies} vs {len(body_names)} names"
            )
        gravity = tuple(float(value) for value in meta["gravity"])
        self._model_info = self._MODEL_INFO_CLS(
            num_dof=num_dof,
            num_bodies=num_bodies,
            dof_names=dof_names,
            body_names=body_names,
            gravity=(gravity[0], gravity[1], gravity[2]),
            use_gpu_pipeline=bool(meta.get("use_gpu_pipeline", False)),
        )
        raw_origins = meta.get("env_origins")
        if raw_origins is not None:
            origins = np.asarray(raw_origins, dtype=np.float32)
            expected_origins = (self._num_envs, 3)
            if origins.shape != expected_origins or not np.isfinite(origins).all():
                raise self._worker_error(
                    f"{self._BACKEND_LABEL} worker environment origins have invalid "
                    "shape or values: "
                    f"got shape {origins.shape}, expected {expected_origins}"
                )
        self._body_id_by_name = {name: index for index, name in enumerate(body_names)}
        self._dof_id_by_name = {name: index for index, name in enumerate(dof_names)}
        self._validate_fixed_variant_handshake(meta)
        self._validate_xml_metadata_against_worker()

    def _validate_fixed_variant_handshake(self, meta: dict[str, Any]) -> None:
        """Require the worker to echo the immutable variant assignment."""
        if self._fixed_variant_plan is None:
            return
        expected_count = len(self._fixed_variant_plan.variants)
        actual_count = int(meta.get("fixed_variant_count", -1))
        if actual_count != expected_count:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} fixed-variant handshake reported "
                f"{actual_count} variants, expected {expected_count}"
            )
        raw_assignment = meta.get("fixed_variant_assignment")
        assignment = (
            np.asarray(raw_assignment, dtype=np.int32) if raw_assignment is not None else None
        )
        if (
            assignment is None
            or assignment.shape != self._fixed_variant_plan.assignment.shape
            or not np.array_equal(assignment, self._fixed_variant_plan.assignment)
        ):
            raise self._worker_error(
                f"{self._BACKEND_LABEL} fixed-variant handshake changed or omitted the "
                "immutable per-env assignment"
            )

    def _fixed_variant_init_payload(self) -> dict[str, Any]:
        """Serialize the complete construction-time variant identity to the worker."""
        if self._fixed_variant_plan is None:
            return {}
        metadata = self._get_fixed_variant_metadata()
        initial_qpos = self._resolve_variant_initial_qpos()
        return {
            "variant_model_files": [
                str(Path(variant.model_file).expanduser())
                for variant in self._fixed_variant_plan.variants
            ],
            "variant_assignment": [int(value) for value in self._fixed_variant_plan.assignment],
            "variant_dof_fields": [
                self._position_actuation_payload_for(variant) for variant in metadata
            ],
            # Only the legacy raw-MJCF worker consumes the FK kinematic tree
            # (#141); the mapped scene path keeps its own host-side contract.
            **(
                {}
                if self._entity_scene is not None
                else {"variant_mjcf_kinematics": list(self._get_fixed_variant_kinematics())}
            ),
            "variant_keyframe_qpos": [
                None if value is None else [float(item) for item in value] for value in initial_qpos
            ],
        }

    def _position_actuation_payload(self) -> dict[str, list[float]]:
        """Per-dof PD/limit/dynamics arrays in MJCF joint document order.

        The worker maps them onto the asset's dof order by name.  Joints with
        no ``<position>`` actuator are passive: zero gains and zero effort.
        """
        fields = self._position_actuation_payload_for(self._get_scene_metadata())
        return {
            "dof_stiffness": fields["stiffness"],
            "dof_damping": fields["damping"],
            "dof_effort": fields["effort"],
            "dof_armature": fields["armature"],
            "dof_friction": fields["friction"],
        }

    def _position_actuation_payload_for(
        self,
        metadata: SceneMetadata,
    ) -> dict[str, list[float]]:
        """Build per-dof actuation arrays for one already-scanned MJCF source."""
        by_joint = {spec.joint_name: spec for spec in metadata.actuators}
        stiffness: list[float] = []
        damping: list[float] = []
        effort: list[float] = []
        for joint in metadata.joint_names:
            spec = by_joint.get(joint)
            if spec is None:
                stiffness.append(0.0)
                damping.append(0.0)
                effort.append(0.0)
            else:
                stiffness.append(spec.kp)
                damping.append(spec.kv)
                # PhysX clamps |force| <= effort; the scan guarantees symmetry.
                effort.append(
                    _UNLIMITED_DOF_EFFORT if spec.forcerange is None else spec.forcerange[1]
                )
        return {
            "stiffness": stiffness,
            "damping": damping,
            "effort": effort,
            "armature": [float(value) for value in metadata.joint_armature],
            "friction": [float(value) for value in metadata.joint_frictionloss],
        }

    def _validate_xml_metadata_against_worker(self) -> None:
        """Fail closed when the MJCF importer changed names or ordering.

        Pre-materialize metadata answers (body/joint ids, keyframe pose) come
        from the parent-side XML scan; this handshake check makes those
        answers contractual by verifying the worker's asset preserves them.
        """
        assert self._model_info is not None
        metadata = self._get_scene_metadata()
        # Variant XML sources were already checked against this canonical XML
        # scan before INIT, and the worker checks every imported asset against its
        # canonical asset. Repeating either comparison here would add a third
        # redundant validation layer without observable behavior.
        if metadata.body_names and tuple(self._model_info.body_names) != metadata.body_names:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} importer changed the rigid-body name order:\n"
                f"  xml:    {metadata.body_names}\n"
                f"  worker: {self._model_info.body_names}\n"
                "Body ids resolved before materialize() would be wrong; fix the scene "
                "or extend the backend to remap by name."
            )
        if metadata.joint_names and tuple(self._model_info.dof_names) != metadata.joint_names:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} importer changed the dof name order:\n"
                f"  xml:    {metadata.joint_names}\n"
                f"  worker: {self._model_info.dof_names}\n"
                "Joint indices resolved before materialize() would be wrong; fix the "
                "scene or extend the backend to remap by name."
            )

    def _allocate_slots(self) -> None:
        assert self._model_info is not None
        shapes = (
            protocol.slot_shapes(
                self._num_envs, self._model_info.num_dof, self._model_info.num_bodies
            )
            if self._entity_scene is None
            else protocol.scene_slot_shapes(
                self._num_envs,
                self._entity_scene.layout,
                num_contact_force_sensors=self._mapped_contact_force_sensor_count(),
            )
        )
        for name in shapes:
            shape = shapes[name]
            handle = shared_memory.SharedMemory(
                create=True, size=protocol.slot_allocation_nbytes(name, shape)
            )
            self._shm_handles[name] = handle
            self._slots[name] = np.ndarray(
                shape, dtype=protocol.slot_dtype(name), buffer=handle.buf
            )

    def _slot_specs(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shm": handle.name,
                "shape": list(self._slots[name].shape),
                "dtype": str(self._slots[name].dtype),
            }
            for name, handle in self._shm_handles.items()
        }

    def _resolve_sensor_map(self) -> dict[str, tuple[SceneSensorSpec, int]]:
        """Resolve supported scene sensors to (spec, body_id) on the cold path."""
        metadata = self._get_scene_metadata()
        resolved: dict[str, tuple[SceneSensorSpec, int]] = {}
        for name, spec in metadata.sensors.items():
            body_id = self._body_id_by_name.get(spec.body_name)
            if spec.target_body_name is not None:
                if self._body_id_by_name.get(spec.target_body_name) is None:
                    metadata.unsupported_sensors[name] = _unsupported_spec(
                        spec,
                        f"sensor target body {spec.target_body_name!r} is not present in "
                        f"the {self._BACKEND_LABEL} asset rigid-body list",
                    )
                    continue
            if body_id is None:
                # The MJCF importer may drop or rename bodies; record as
                # unsupported so access fails closed with context.
                metadata.unsupported_sensors[name] = _unsupported_spec(
                    spec,
                    f"sensor body {spec.body_name!r} is not present in the {self._BACKEND_LABEL} "
                    "asset rigid-body list",
                )
                continue
            resolved[name] = (spec, body_id)
        return resolved

    # ------------------------------------------------------------------ #
    # Request/response plumbing
    # ------------------------------------------------------------------ #

    def _stderr_tail(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - _STDERR_TAIL_BYTES))
            return str(self._stderr_file.read().decode("utf-8", errors="replace"))
        except Exception:
            return ""

    def _request(self, cmd: str, payload: Any, *, expect: str) -> Any:
        if self._worker_dead_error is not None:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker is unavailable from an earlier failure; "
                f"refusing {cmd}"
            ) from self._worker_dead_error
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} backend is not materialized; call materialize() first"
            )
        if proc.poll() is not None:
            error = self._worker_error(
                f"{self._BACKEND_LABEL} worker exited with code {proc.returncode} before {cmd}; "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            )
            self._worker_dead_error = error
            raise error
        try:
            protocol.send_message(cast(BinaryIO, proc.stdin), cmd, payload)
            message = self._recv_with_timeout(proc.stdout, self._worker_timeout_s, cmd)
        except SubprocessWorkerError as exc:
            self._worker_dead_error = exc
            self._kill_worker()
            raise
        if message["cmd"] == protocol.CMD_ERROR:
            error = self._worker_error(
                protocol.format_worker_error(message["payload"], self._BACKEND_LABEL),
                worker_traceback=message["payload"].get("traceback"),
            )
            if message["payload"].get("faulted", False):
                self._worker_dead_error = error
                self._kill_worker()
            raise error
        if message["cmd"] != expect:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker replied {message['cmd']!r} to {cmd}, "
                f"expected {expect!r}"
            )
        return message.get("payload")

    def _recv_with_timeout(self, stream: Any, timeout_s: float, cmd: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        try:
            header = _read_exactly_with_deadline(stream, protocol.HEADER_SIZE, deadline)
            body = _read_exactly_with_deadline(stream, protocol.unpack_header(header), deadline)
        except TimeoutError as exc:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker did not answer {cmd} within {timeout_s}s; "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            ) from exc
        except protocol.WorkerDisconnectedError as exc:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker closed its pipe during {cmd} "
                f"(exit code {self._proc.poll() if self._proc else '?'}); "
                f"stderr tail:\n{self._stderr_tail()}",
                stderr_tail=self._stderr_tail(),
            ) from exc
        return protocol.decode_message(body)

    def _kill_worker(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except Exception:
                pass

    def close(self) -> None:
        """Shut down the worker and release shared memory. Idempotent."""
        self._closed = True
        proc, self._proc = self._proc, None
        shm_handles, self._shm_handles = list(self._shm_handles.values()), {}
        self._slots = {}
        stderr_file, self._stderr_file = self._stderr_file, None
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        _release_worker(proc, shm_handles, stderr_file)
        if self._entity_scene is not None:
            self._entity_scene.close()

    def cleanup_scene_assets(self) -> None:
        """Release all backend-owned resources, including the worker process.

        ``NpEnv.close()`` deliberately depends on the base
        ``cleanup_scene_assets`` hook rather than a backend-specific ``close``
        method.  A subprocess backend must therefore bridge that lifecycle
        hook explicitly; otherwise closing a manager environment would leave
        the Python 3.8/3.11 worker and its shared-memory segments alive until
        interpreter teardown.
        """
        self.close()
        super().cleanup_scene_assets()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        try:
            super().__del__()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Materialized-state guards and views
    # ------------------------------------------------------------------ #

    def _require_materialized(self) -> SubprocessModelInfo:
        if self._model_info is None:
            # Lazy cold path: the first state/metadata access materializes the
            # worker, matching the MuJoCo backend whose constructor leaves the
            # model fully queryable. Env construction reads state shapes
            # (e.g. Entity._validate_joint_state) before the explicit
            # materialize() lifecycle point.
            self.materialize()
        assert self._model_info is not None
        return self._model_info

    def _require_state(self, operation: str) -> None:
        self._require_materialized()
        if self._worker_dead_error is not None:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker is unavailable from an earlier failure; "
                f"refusing {operation}"
            ) from self._worker_dead_error
        if not self._slots:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} backend is closed or not materialized; refusing {operation}"
            )

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> SubprocessModelInfo:
        """Return backend-owned model metadata; never a live physics object."""
        return self._require_materialized()

    def _num_dof(self) -> int:
        """DoF count from the worker handshake, falling back to the XML scan."""
        if self._entity_scene is not None:
            return sum(
                len(j.qvel_indices) for e in self._entity_scene.layout.entities for j in e.joints
            )
        if self._model_info is not None:
            return self._model_info.num_dof
        return len(self._get_scene_metadata().joint_names)

    def _body_name_map(self) -> dict[str, int]:
        """Body name→id map; worker-authoritative post-INIT, XML pre-INIT."""
        if self._model_info is not None:
            return self._body_id_by_name
        metadata = self._get_scene_metadata()
        return {name: index for index, name in enumerate(metadata.body_names) if name}

    def _dof_name_map(self) -> dict[str, int]:
        """Joint name→dof map; worker-authoritative post-INIT, XML pre-INIT."""
        if self._model_info is not None:
            return self._dof_id_by_name
        metadata = self._get_scene_metadata()
        return {name: index for index, name in enumerate(metadata.joint_names)}

    @property
    def num_actuators(self) -> int:
        if self._entity_scene is not None:
            return self._entity_scene.layout.nu
        return self._num_dof()

    @property
    def num_dof_vel(self) -> int:
        return self._num_dof()

    def get_actuator_ctrl_range(self) -> np.ndarray:
        """Position-target clamp per dof, from the MJCF ``ctrlrange`` attributes.

        Pure XML metadata (available pre-materialize).  Undeclared ctrlranges
        report ``(0, 0)``, matching the MuJoCo backend which returns the raw
        ``actuator_ctrlrange`` (``ctrllimited=false`` → ``0 0``).
        """
        if self._entity_scene is not None:
            return self._entity_scene.ctrl_ranges.copy()
        metadata = self._get_scene_metadata()
        by_joint = {spec.joint_name: spec for spec in metadata.actuators}
        rows = [
            (0.0, 0.0)
            if (spec := by_joint.get(joint)) is None or spec.ctrlrange is None
            else spec.ctrlrange
            for joint in metadata.joint_names
        ]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 2)

    def get_actuator_names(self) -> tuple[str, ...]:
        if self._entity_scene is not None:
            values = sorted(
                (i, e.name + "/" + name)
                for e in self._entity_scene.layout.entities
                for i, name in zip(e.actuator_indices, e.actuator_names, strict=True)
            )
            return tuple(name for _, name in values)
        if self._model_info is not None:
            return self._model_info.dof_names
        return self._get_scene_metadata().joint_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        """The shared position-control profile drives one actuator per DoF."""
        if self._entity_scene is not None:
            values = sorted(
                (i, e.name + "/" + name)
                for e in self._entity_scene.layout.entities
                for i, name in zip(e.actuator_indices, e.actuator_joint_names, strict=True)
            )
            return tuple(name for _, name in values)
        return self.get_actuator_names()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-dof (kp, kd) from the MJCF ``<position>`` actuators (pure XML).

        Passive joints (no actuator) report zero gains.  Returned in MJCF
        joint document order, which the INIT handshake pins to the worker's
        dof order.
        """
        if self._entity_scene is not None:
            return self._entity_scene.kp.copy(), self._entity_scene.kd.copy()
        metadata = self._get_scene_metadata()
        by_joint = {spec.joint_name: spec for spec in metadata.actuators}
        kp = np.asarray(
            [
                spec.kp if (spec := by_joint.get(joint)) is not None else 0.0
                for joint in metadata.joint_names
            ],
            dtype=np.float64,
        )
        kd = np.asarray(
            [
                spec.kv if (spec := by_joint.get(joint)) is not None else 0.0
                for joint in metadata.joint_names
            ],
            dtype=np.float64,
        )
        return kp, kd

    def get_scene_model_file(self) -> str | None:
        return str(self._scene.model_file)

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if self._entity_scene is not None:
            model = self._entity_scene.owner.model
            try:
                return model.key(name).qpos.copy()
            except KeyError as exc:
                raise ValueError(f"unknown composed keyframe {name!r}") from exc
        # Pure parent-side XML metadata: available before materialize(),
        # matching the MuJoCo backend (whose model loads in the constructor).
        metadata = self._get_scene_metadata()
        try:
            qpos = metadata.keyframes[name]
        except KeyError as exc:
            available = ", ".join(sorted(metadata.keyframes))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc
        nq = _ROOT_QPOS_DIM + len(metadata.joint_names)
        if qpos.size != nq:
            raise ValueError(
                f"Keyframe {name!r} qpos has {qpos.size} entries; expected {nq} "
                f"(7 root + {nq - 7} dofs) for the {self._BACKEND_LABEL} layout"
            )
        if self._model_info is not None and qpos.size != _ROOT_QPOS_DIM + self._model_info.num_dof:
            raise ValueError(
                f"Keyframe {name!r} qpos does not match the {self._BACKEND_LABEL} asset: "
                f"{qpos.size} entries vs 7 root + {self._model_info.num_dof} dofs"
            )
        return qpos.copy()

    def get_default_qpos(self) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_scene.qpos[0].copy()
        initial_qpos = self._effective_initial_qpos()
        if initial_qpos is not None:
            # The selected scene keyframe is the backend default state (and the
            # post-INIT worker state).
            return initial_qpos.copy()
        qpos = np.zeros((_ROOT_QPOS_DIM + self._num_dof(),), dtype=np.float32)
        qpos[3] = 1.0
        return qpos

    def get_default_dof_pos(self) -> np.ndarray:
        if self._entity_scene is not None:
            ids = [
                i
                for e in self._entity_scene.layout.entities
                for j in e.joints
                for i in j.qpos_indices
            ]
            return self._entity_scene.qpos[0, ids].copy()
        initial_qpos = self._effective_initial_qpos()
        if initial_qpos is not None:
            return initial_qpos[_ROOT_QPOS_DIM:].copy()
        return np.zeros((self._num_dof(),), dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_scene.qvel[0].copy()
        return np.zeros((_ROOT_QVEL_DIM + self._num_dof(),), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self._entity_scene is not None:
            for entity in self._entity_scene.layout.entities:
                if root_body_name == entity.name + "/" + entity.root_body:
                    if entity.root_mode != "floating":
                        raise NotImplementedError("entity root has no generalized coordinates")
                    return BackendRootStateLayout(
                        entity.root_qpos_indices, entity.root_qvel_indices
                    )
            raise ValueError(f"unknown entity root body {root_body_name!r}")
        if self._model_info is not None:
            root_name = self._model_info.body_names[0]
        else:
            metadata = self._get_scene_metadata()
            if metadata.freejoint_body_name is None:
                raise NotImplementedError(
                    f"backend '{self._BACKEND_LABEL}' capability 'root-state layout' "
                    "requires the scene "
                    "to declare a free joint (floating base)"
                )
            root_name = metadata.freejoint_body_name
        if root_body_name != root_name:
            raise NotImplementedError(
                f"backend '{self._BACKEND_LABEL}' capability 'root-state layout' requires "
                f"{root_body_name!r} to be the actor root body {root_name!r}"
            )
        return BackendRootStateLayout(
            qpos_indices=tuple(range(_ROOT_QPOS_DIM)),
            qvel_indices=tuple(range(_ROOT_QVEL_DIM)),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_scene is not None:
            return np.asarray(self._entity_scene.layout.get_body_ids(names), dtype=np.int32)
        body_map = self._body_name_map()
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(body_map[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in {self._BACKEND_LABEL} model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        # Motion datasets use MuJoCo-style body ids, where worldbody is id 0;
        # both the XML scan and the worker body table exclude worldbody, so
        # get_body_ids is off by one (same offset as the motrix adapter).
        return self.get_body_ids(names) + 1

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        """Per-joint ``range`` from the MJCF (pure XML, available pre-materialize).

        The XML is the cross-runtime source of truth for this contract.
        Joints without a ``range`` attribute report ``(-inf, inf)``.
        """
        if self._entity_scene is not None:
            if names is None:
                return self._entity_scene.joint_ranges.copy()
            ordered = [
                e.name + "/" + j.name for e in self._entity_scene.layout.entities for j in e.joints
            ]
            self._entity_scene.layout.get_joint_layouts(names)
            return self._entity_scene.joint_ranges[[ordered.index(name) for name in names]].copy()
        self._reject_named_joint_ranges(names, "joint ranges")
        metadata = self._get_scene_metadata()
        if not metadata.joint_ranges:
            return None
        return np.asarray(metadata.joint_ranges, dtype=np.float32)

    def get_gravity(self) -> np.ndarray:
        info = self._require_materialized()
        return np.asarray(info.gravity, dtype=np.float32).copy()

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to absolute qvel indices (root 6 columns first)."""
        if self._entity_scene is not None:
            return self.get_joint_state_qvel_indices(names)
        return self._resolve_dof_ids(names) + _ROOT_QVEL_DIM

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_joint_indices(names, velocity=False, packed=True)
        return self._resolve_dof_ids(names)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_joint_indices(names, velocity=True, packed=True)
        return self._resolve_dof_ids(names)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_joint_indices(names, velocity=False, packed=False)
        return self._resolve_dof_ids(names) + _ROOT_QPOS_DIM

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_scene is not None:
            return self._entity_joint_indices(names, velocity=True, packed=False)
        return self._resolve_dof_ids(names) + _ROOT_QVEL_DIM

    def _resolve_dof_ids(self, names: Sequence[str]) -> np.ndarray:
        dof_map = self._dof_name_map()
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(dof_map[str(name)])
            except KeyError as exc:
                raise ValueError(
                    f"Joint {name!r} not found in {self._BACKEND_LABEL} model"
                ) from exc
        return np.asarray(resolved, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Simulation control
    # ------------------------------------------------------------------ #

    def set_pre_step_control(self, fn: Any | None) -> None:
        if fn is not None:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} rejects host pre-step callbacks; a per-substep callback "
                "cannot cross the worker process boundary inside one physics substep."
            )
        self._pre_step_control_fn = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
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
        if self._entity_scene is not None:
            ctrl_array = np.clip(
                ctrl_array, self._entity_scene.control_lower, self._entity_scene.control_upper
            )

        t0 = time.perf_counter()
        np.copyto(self._slots["ctrl"], ctrl_array)
        step_payload = self._step_payload(int(nsteps))
        payload = self._request(protocol.CMD_STEP, step_payload, expect=protocol.CMD_READY)
        self._after_step(step_payload)
        self._stale_body_ids.clear()
        ipc_ms = (time.perf_counter() - t0) * 1000.0
        timing = dict(payload.get("timing", {})) if isinstance(payload, dict) else {}
        timing["worker_ipc_total_ms"] = ipc_ms
        return {"timing": timing}

    def _step_payload(self, nsteps: int) -> dict[str, Any]:
        """Build the engine-owned STEP command payload.

        Adapter subclasses may extend this payload with validated command-only
        data.  They must not use it to move a host callback across the worker
        boundary.
        """
        return {"nsteps": nsteps}

    def _after_step(self, payload: dict[str, Any]) -> None:
        """Consume adapter-owned data after a successful worker STEP."""

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_state("set_state")
        if self._entity_scene is not None:
            return self._set_mapped_state(env_indices, qpos, qvel, randomization)
        if randomization is not None and not randomization.is_empty():
            requested = ", ".join(sorted(randomization.requested_terms()))
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} does not support reset domain randomization terms: "
                f"{requested}."
            )
        info = self._require_materialized()
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        nq = _ROOT_QPOS_DIM + info.num_dof
        nv = _ROOT_QVEL_DIM + info.num_dof
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        if qpos_array.shape != (rows.size, nq):
            raise ValueError(f"qpos must have shape ({rows.size}, {nq}), got {qpos_array.shape}")
        if qvel_array.shape != (rows.size, nv):
            raise ValueError(f"qvel must have shape ({rows.size}, {nv}), got {qvel_array.shape}")

        timing: dict[str, float] = {key: 0.0 for key in self._SET_STATE_TIMING_ZERO_KEYS}
        timing.update(
            {
                "set_state_reset_upload_ms": 0.0,
                "set_state_reset_forward_ms": 0.0,
                "set_state_host_cache_refresh_ms": 0.0,
                "set_state_internal_gap_ms": 0.0,
            }
        )
        if rows.size == 0:
            return {"timing": timing}

        t0 = time.perf_counter()
        count = int(rows.size)
        np.copyto(self._slots["reset_env_ids"][:count], rows.astype(np.int32))
        np.copyto(self._slots["reset_qpos"][:count], qpos_array)
        np.copyto(self._slots["reset_qvel"][:count], qvel_array)
        payload = self._request(protocol.CMD_SET_STATE, {"count": count}, expect=protocol.CMD_READY)
        ipc_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(payload, dict):
            worker_timing = payload.get("timing", {})
            timing["set_state_reset_upload_ms"] = float(
                worker_timing.get("set_state_reset_upload_ms", 0.0)
            )
            timing["set_state_host_cache_refresh_ms"] = float(
                worker_timing.get("set_state_host_cache_refresh_ms", 0.0)
            )
        timing["set_state_internal_gap_ms"] = (
            ipc_ms - timing["set_state_reset_upload_ms"] - timing["set_state_host_cache_refresh_ms"]
        )
        return {"timing": timing}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise no DR until per-env model mutation is effect-tested."""
        return DomainRandomizationCapabilities()

    # ------------------------------------------------------------------ #
    # Native rendering / playback (worker-owned viewer and camera sensor)
    # ------------------------------------------------------------------ #

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "auto":
            # The interactive viewer needs a reachable display; headless hosts
            # fall back to offscreen camera-sensor recording.
            mode = "interactive" if _display_available() else "record"
        if mode == "none":
            return BackendPlayRenderPlan(
                mode=mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if mode == "interactive":
            return BackendPlayRenderPlan(
                mode=mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        assert mode == "record"
        if play_steps is None:
            raise ValueError(
                f"{self._BACKEND_LABEL} record playback requires a finite "
                "training.play_steps value."
            )
        if output_video is None:
            raise ValueError(
                f"{self._BACKEND_LABEL} record playback requires an output video path."
            )
        return BackendPlayRenderPlan(
            mode=mode,
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
        """Initialize the worker-side viewer and/or capture camera.

        ``spacing``/``offset_mode`` are accepted for contract parity and
        ignored: envs are already laid out on the worker sim's grid.
        """
        del spacing, offset_mode
        camera = _normalize_camera_kwargs(camera_kwargs)
        if not capture and camera != _normalize_camera_kwargs(None):
            raise NotImplementedError(
                "Isaac native interactive viewers do not apply camera_kwargs; "
                "configure the view through the native viewer."
            )
        config = (bool(headless), bool(capture))
        if self._render_config is not None:
            if self._render_config != config:
                raise RuntimeError(
                    f"{self._BACKEND_LABEL} renderer is already initialized with "
                    f"headless={self._render_config[0]}, capture={self._render_config[1]}; "
                    f"cannot reinitialize it with headless={config[0]}, capture={config[1]}"
                )
            return
        self._require_materialized()
        if not self._graphics_enabled:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} rendering requires a GPU sim (device_id >= 0); "
                "this backend runs on the CPU pipeline without a graphics context"
            )
        reply = self._request(
            protocol.CMD_INIT_RENDERER,
            {
                "headless": config[0],
                "capture": config[1],
                "width": int(width),
                "height": int(height),
                "camera": camera,
            },
            expect=protocol.CMD_META,
        )
        self._render_config = config
        self._viewer_open = bool(reply.get("viewer"))
        self._capture_ready = bool(reply.get("capture"))

    def render(self) -> None:
        """Draw one interactive viewer frame through the worker."""
        if not self._viewer_open:
            self.init_renderer(
                headless=False,
                width=self._render_width,
                height=self._render_height,
            )
        reply = self._request(protocol.CMD_RENDER_FRAME, None, expect=protocol.CMD_META)
        if bool(reply.get("closed")):
            self._viewer_open = False
            raise RenderClosedError(f"{self._BACKEND_LABEL} viewer window was closed")

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame from the worker's camera sensor."""
        if not self._capture_ready:
            self.init_renderer(
                headless=True,
                capture=True,
                width=self._render_width,
                height=self._render_height,
            )
        reply = self._request(protocol.CMD_CAPTURE_FRAME, None, expect=protocol.CMD_META)
        # Preserve the worker's dtype so backend specializations can validate
        # the frame contract instead of silently coercing malformed output.
        return np.asarray(reply["frame"])

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Any,
        step: Any,
        num_steps: int | None,
        output_video: str | os.PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Any = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: Any = None,
        on_frame: Any = None,
    ) -> str | None:
        del frame_state_getter
        if debug_overlay_getter is not None:
            raise unsupported_debug_overlay_error(self.__class__.__name__)
        if on_frame is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} renders through a native renderer and "
                "does not support on_frame callbacks"
            )
        camera = CameraCfg.from_kwargs(camera_kwargs)
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        try:
            return run_subprocess_playback(
                backend=self,
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                render_offset_mode=render_offset_mode,
                headless=should_run_headless,
                record_video=should_record_video,
                camera_kwargs=camera,
                width=self._render_width,
                height=self._render_height,
            )
        except RenderClosedError:
            if not should_run_headless and not should_record_video:
                logger.info("Render window closed.")
                return None
            raise

    # ------------------------------------------------------------------ #
    # Cached state getters (shm views; no worker round trip)
    # ------------------------------------------------------------------ #

    def _root_slot(self) -> np.ndarray:
        self._require_state("state read")
        if self._entity_scene is not None:
            index = self._primary_entity_index()
            return self._slots["entity_root_state"][:, index]
        return self._slots["root_state"]

    def _body_slot(self) -> np.ndarray:
        self._require_state("state read")
        return self._slots["body_state"]

    def get_base_pos(self) -> np.ndarray:
        return self._root_slot()[:, 0:3]

    def get_base_quat(self) -> np.ndarray:
        return self._root_slot()[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        return self._root_slot()[:, 7:10]

    def get_base_ang_vel(self) -> np.ndarray:
        return self._root_slot()[:, 10:13]

    def get_dof_pos(self) -> np.ndarray:
        self._require_state("get_dof_pos")
        if self._entity_scene is not None:
            return self._slots["qpos"][:, self._entity_dof_qpos_columns]
        return self._slots["dof_state"][:, :, 0]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
        if self._entity_scene is not None:
            return self._slots["qvel"][:, self._entity_dof_qvel_columns]
        return self._slots["dof_state"][:, :, 1]

    def _selected_body_state(self, body_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(body_ids, dtype=np.intp)
        info = self._require_materialized()
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= info.num_bodies):
            raise ValueError(
                f"body_ids must be a 1-D array in [0, {info.num_bodies}), got {body_ids!r}"
            )
        if self._stale_body_ids.intersection(int(i) for i in ids):
            raise NotImplementedError(
                "IsaacGym descendant body state is unavailable after joint reset until step"
            )
        return self._body_slot()[:, ids, :]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 0:3]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 3:7]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 7:10]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._selected_body_state(body_ids)[:, :, 10:13]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        base_state = self._body_slot()[:, self._base_body_id, :]
        rel = self.get_body_pos_w(body_ids) - base_state[:, 0:3][:, None, :]
        return np_quat_apply_inverse_batched(base_state[:, 3:7][:, None, :], rel)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        inverse = self._body_slot()[:, self._base_body_id, 3:7].copy()
        inverse[:, 1:4] = -inverse[:, 1:4]
        quat_w = self.get_body_quat_w(body_ids)
        return np_quat_mul_batched(inverse[:, None, :], quat_w)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Contract definition (#1254): world velocity rotated into each body's
        # own frame, identical to the MuJoCo/mjwarp backends.
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(body_ids), self.get_body_lin_vel_w(body_ids)
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(body_ids), self.get_body_ang_vel_w(body_ids)
        )

    # ------------------------------------------------------------------ #
    # Sensors
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        """Compute one mapped scene sensor from the shm state caches.

        See ``sensors.py`` for the MJCF-sensor → tensor-quantity mapping table.
        Names that are not declared in the scene raise ``ValueError``; declared
        but unmappable sensors fail closed with ``NotImplementedError``.
        """
        self._require_state("get_sensor_data")
        metadata = self._get_scene_metadata()
        mapped = self._sensor_map.get(name)
        if mapped is None:
            unsupported = metadata.unsupported_sensors.get(name)
            if unsupported is not None:
                raise NotImplementedError(
                    f"{self._BACKEND_LABEL} cannot serve sensor {name!r}: {unsupported.reason}"
                )
            available = ", ".join(sorted(self._sensor_map))
            raise ValueError(f"Sensor {name!r} not found; available: {available}")
        spec, body_id = mapped
        state = self._selected_body_state(np.asarray([body_id]))[:, 0, :]
        kind = spec.kind
        local_quat = np.asarray(spec.local_quat, dtype=np.float32)[None, :]
        local_pos = np.asarray(spec.local_pos, dtype=np.float32)[None, :]
        if kind == KIND_GYRO:
            body_frame = np_quat_apply_inverse_batched(state[:, 3:7], state[:, 10:13])
            return np_quat_apply_inverse_batched(local_quat, body_frame).astype(np.float32)
        if kind == KIND_LOCAL_LINVEL:
            body_frame = np_quat_apply_inverse_batched(state[:, 3:7], state[:, 7:10])
            return np_quat_apply_inverse_batched(local_quat, body_frame).astype(np.float32)
        if kind == KIND_FRAMEQUAT:
            return np_quat_mul_batched(state[:, 3:7], local_quat).astype(np.float32)
        if kind == KIND_FRAMEPOS:
            offset = np_quat_apply_batched(state[:, 3:7], local_pos)
            return (state[:, 0:3] + offset).astype(np.float32)
        if kind == KIND_FRAMEZAXIS:
            frame_quat = np_quat_mul_batched(state[:, 3:7], local_quat)
            z_axis = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            return np_quat_apply_batched(frame_quat, z_axis).astype(np.float32)
        if kind == KIND_CONTACT_FOUND:
            force = self._slots["contact_force"][:, body_id, :]
            return (np.linalg.norm(force, axis=-1, keepdims=True) > 0.0).astype(np.float32)
        raise NotImplementedError(f"{self._BACKEND_LABEL} sensor kind {kind!r} is not implemented")


def _unsupported_spec(spec: SceneSensorSpec, reason: str) -> UnsupportedSensorSpec:
    return UnsupportedSensorSpec(name=spec.name, reason=reason)


__all__ = [
    "MjcfSubprocessBackend",
    "SubprocessModelInfo",
    "SubprocessWorkerError",
    "_normalize_camera_kwargs",
]
