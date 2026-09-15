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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    CameraCfg,
    RenderClosedError,
    SimBackend,
    normalize_play_render_mode,
    unsupported_debug_overlay_error,
)
from unisim.dr.interval import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    IntervalTermOp,
)
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantMetadata,
    FixedVariantPlan,
    IntervalRandomizationPlan,
    ResetRandomizationPayload,
)
from unisim.scene import (
    ENTITY_MATERIALIZATION_RIGID,
    ENTITY_ROOT_FLOATING,
    MODEL_FORMAT_URDF,
    ActuatorGainOverride,
    SceneCfg,
    SceneEntitySpec,
    validate_scene_composition_support,
)
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
    scan_scene_entities,
    scan_scene_metadata,
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


def build_init_variant_pool_payload(
    variant_plan: FixedVariantPlan,
    *,
    num_envs: int,
    entity_assets: tuple[SceneEntitySpec, ...],
    backend_label: str,
) -> dict[str, Any]:
    """Validate a fixed variant plan into a worker INIT payload entry.

    Pure cold-path validation shared by every subprocess backend that
    materializes whole-file model variant pools.  The plan's per-environment
    assignment maps to one ``MultiUsdFileCfg``-style pool on the worker, so
    exactly one declared scene entity must bind itself to the pool with
    ``consumes_fixed_variant_pool=True``; that entity must be a floating
    rigid object whose declared format matches the variant files.  Every
    check fails closed before any worker is spawned: undeclared or ambiguous
    targets, non-URDF sources, missing files, malformed assignments, and —
    through the host-side URDF scan — unparseable documents, multiple root
    links, unsupported joint types, movable-joint (articulated) variants,
    and root/body layout drift across the catalog or against the target's
    bootstrap asset.  The payload never carries a ``masses`` key: variant
    mass is backend-authoritative and measured from the materialized asset.
    """
    declared = [spec for spec in entity_assets if spec.consumes_fixed_variant_pool]
    if not declared:
        raise ValueError(
            f"{backend_label} fixed variant pool requires exactly one scene entity with "
            "consumes_fixed_variant_pool=True, got 0; the plan has no target"
        )
    if len(declared) > 1:
        names = sorted(spec.name for spec in declared)
        raise ValueError(
            f"{backend_label} fixed variant pool requires exactly one scene entity with "
            f"consumes_fixed_variant_pool=True, got {len(declared)}: {names}; one pool maps "
            "to one worker-side asset pool"
        )
    spec = declared[0]
    target = spec.name
    if spec.materialization != ENTITY_MATERIALIZATION_RIGID:
        raise NotImplementedError(
            f"{backend_label} variant target {target!r} has materialization "
            f"{spec.materialization!r}; only rigid-object pools are supported"
        )
    if spec.root_mode != ENTITY_ROOT_FLOATING:
        raise NotImplementedError(
            f"{backend_label} variant target {target!r} has root_mode {spec.root_mode!r}; "
            "only floating rigid-object pools are supported"
        )
    if spec.asset_format != MODEL_FORMAT_URDF:
        raise ValueError(
            f"{backend_label} variant target {target!r} declares asset_format "
            f"{spec.asset_format!r}, which does not match the URDF variant files"
        )

    variants = variant_plan.variants
    if not variants:
        # FixedVariantPlan already rejects empty catalogs; keep the explicit
        # guard so the pool semantics cannot drift.
        raise ValueError(f"{backend_label} init variant pool requires at least one variant")
    # The rigid-object pool realization guarantees one identical public body
    # layout per variant; it cannot provide UNIFORM_PUBLIC_LAYOUT optional
    # slots, so such a plan fails closed at staging rather than being
    # silently narrowed.  Capability negotiation rejects it through the same
    # declared-layout set, but direct materialization must not depend on the
    # caller consulting the capability report first.
    if variant_plan.layout is not FixedVariantLayout.SAME_LAYOUT:
        raise NotImplementedError(
            f"{backend_label} fixed variant pool supports SAME_LAYOUT plans only; "
            f"got {variant_plan.layout.value!r}"
        )
    # Host-side source validation, fail-closed before any worker process is
    # spawned: every variant must be a parseable single-root URDF whose
    # movable-joint-free tree merges to exactly one rigid body, and the
    # public root/body layout must be identical across the catalog and the
    # target entity's bootstrap asset.  Sources that only fail inside the
    # URDF converter (wrong extension, malformed XML, unsupported joint
    # types, or a drifted layout) would otherwise surface after the Kit
    # worker has already paid its startup cost.
    bootstrap = scan_scene_metadata(
        str(Path(spec.model_file).expanduser()),
        backend_label=backend_label,
        urdf_fixed_base=spec.fixed_base,
    )
    expected_layout = (bootstrap.urdf_root_link_name, bootstrap.body_names)
    source_files: list[str] = []
    for index, variant in enumerate(variants):
        path = Path(variant.model_file).expanduser()
        if path.suffix.lower() != ".urdf":
            raise ValueError(
                f"{backend_label} variant {index} source must be a URDF file: {path}"
            )
        if not path.is_file():
            raise ValueError(
                f"{backend_label} variant {index} source file does not exist: {path}"
            )
        # Pool variants are always floating rigid objects (enforced above).
        scanned = scan_scene_metadata(
            str(path), backend_label=backend_label, urdf_fixed_base=False
        )
        if scanned.joint_names:
            raise NotImplementedError(
                f"{backend_label} variant {index} ({path.name}) declares movable joints "
                f"{list(scanned.joint_names)}; a pool variant must be one rigid body "
                "because the target entity materializes as a floating rigid object"
            )
        layout = (scanned.urdf_root_link_name, scanned.body_names)
        if layout != expected_layout:
            raise ValueError(
                f"{backend_label} variant {index} ({path.name}) changes the public "
                f"root/body layout: bootstrap={expected_layout}, variant={layout}"
            )
        source_files.append(str(path.resolve()))

    variant_plan.validate(num_envs)
    assignments = [int(value) for value in variant_plan.assignment]
    # Performance contract: only the deterministic round-robin assignment is
    # supported.  It maps onto K unique prototypes (``MultiUsdFileCfg`` with
    # ``random_choice=False`` materializes environment i from source
    # ``i % K``), so the worker builds K assets regardless of the environment
    # count.  An arbitrary assignment would need one prototype reference per
    # environment (O(num_envs) stage authoring — 12,288 redundant prototypes
    # at production smoke sizes) and fails closed here instead of silently
    # degrading; the exact per-env spawner is a planned follow-up.
    expected_round_robin = [index % len(source_files) for index in range(num_envs)]
    if assignments != expected_round_robin:
        raise NotImplementedError(
            f"{backend_label} fixed variant pool supports round-robin assignments only "
            f"(assignment[i] == i % {len(source_files)}); an arbitrary per-env "
            "assignment would expand to one prototype per environment — the exact "
            "K-prototype spawner is a planned follow-up"
        )
    return {
        "target_entity": target,
        "source_files": source_files,
        "assignments": assignments,
    }


def _display_available() -> bool:
    """Return whether a display is reachable for an interactive worker viewer."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _normalize_camera_kwargs(
    camera_kwargs: CameraCfg | Mapping[str, Any] | None,
) -> dict[str, float]:
    """Map the repository camera configuration onto the worker's spherical offset."""
    camera = CameraCfg.from_kwargs(camera_kwargs)
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

    def _supports_entity_assets(self) -> bool:
        """Return whether this adapter's worker materializes declared entity assets."""
        return False

    def _supports_ground_plane(self) -> bool:
        """Return whether this adapter's worker consumes ``SceneCfg.ground_plane``."""
        return False

    def _supports_scene_physx(self) -> bool:
        """Return whether this adapter's worker consumes ``SceneCfg.physx``."""
        return False

    def _supports_env_grid_spacing(self) -> bool:
        """Return whether this adapter's worker consumes ``SceneCfg.env_grid_spacing``."""
        return False

    def _worker_init_payload(self) -> dict[str, Any]:
        """Return backend-owned cold-path INIT options.

        Subprocess workers must receive any runtime mode that changes Kit
        startup before the first ``INIT`` handshake.  The shared adapter keeps
        this hook empty so IsaacGym and other workers retain their existing
        startup contract; IsaacSim overrides it for eval rendering.
        """
        return {}

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
        if scene.fragment_files:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} backend does not compose MuJoCo scene fragments; provide a "
                "self-contained MJCF scene through scene.model_file"
            )
        if scene.terrain is not None:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} backend does not support generated terrain scenes yet"
            )
        # Composition declarations are scene content: an adapter whose worker
        # does not materialize them fails closed here instead of silently
        # degrading to single-asset behavior.
        validate_scene_composition_support(
            scene,
            self._BACKEND_LABEL,
            supports_entity_assets=self._supports_entity_assets(),
            supports_ground_plane=self._supports_ground_plane(),
            supports_scene_physx=self._supports_scene_physx(),
            supports_env_grid_spacing=self._supports_env_grid_spacing(),
        )
        if scene.env_grid_spacing is not None:
            spacing = float(scene.env_grid_spacing)
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(
                    f"{self._BACKEND_LABEL} env_grid_spacing must be a finite positive "
                    f"number, got {scene.env_grid_spacing!r}"
                )
        if scene.fixed_variant_plan is not None:
            scene.fixed_variant_plan.validate(int(num_envs))
            if not self._supports_fixed_variant_plans():
                raise NotImplementedError(
                    f"{self._BACKEND_LABEL} backend does not support fixed variant plans"
                )

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
        # Full INIT metadata for diagnostics/probes (variant assignment
        # forensics, actuator gain readback); ``None`` before materialize().
        self._worker_init_meta: dict[str, Any] | None = None
        # Optional diagnostics supplied by workers that maintain private
        # world-space environment origins. Workers that do not send this
        # metadata leave the value as ``None``.
        self._worker_env_origins: np.ndarray | None = None
        self._collision_filtering_applied = False
        self._scene_metadata: SceneMetadata | None = None
        self._entity_metadata: dict[str, SceneMetadata] | None = None
        # Variant pool staged by the adapter that owns the entity-bound pool
        # realization (IsaacSimBackend.materialize); ``None`` keeps the INIT
        # payload without the ``variant_pool`` key.
        self._init_variant_pool: dict[str, Any] | None = None
        self._initial_qpos: np.ndarray | None = None
        self._initial_qpos_resolved = False
        self._fixed_variant_plan = scene.fixed_variant_plan
        self._fixed_variant_metadata: tuple[SceneMetadata, ...] | None = None
        self._variant_initial_qpos: tuple[np.ndarray | None, ...] | None = None
        self._variant_initial_qpos_resolved = False
        self._sensor_map: dict[str, tuple[SceneSensorSpec, int]] = {}
        self._body_id_by_name: dict[str, int] = {}
        self._dof_id_by_name: dict[str, int] = {}
        # Rigid scene entities (SimToolReal step 1.3c), in declaration order.
        # Each owns one ``entity_root_state__<name>``/``entity_reset_state__<name>``
        # shm slot pair; its scanned root body name maps to the extended body id
        # ``num_bodies + k`` so the public body getters route to the new slot.
        # Empty for single-articulation scenes.
        self._rigid_root_entities: tuple[str, ...] = ()
        self._rigid_root_body_names: dict[str, str] = {}
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

    # ------------------------------------------------------------------ #
    # Worker lifecycle (cold path)
    # ------------------------------------------------------------------ #

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
        # Parent-side MJCF metadata (sensors, keyframes, joint document order)
        # is resolved lazily on first access and reused here so the INIT
        # payload can carry the keyframe pose.
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
            meta = self._request(
                protocol.CMD_INIT,
                {
                    "model_file": str(Path(self._scene.model_file).expanduser()),
                    "num_envs": self._num_envs,
                    "sim_dt": self._sim_dt,
                    "device_id": self._device_id,
                    # Fixed-base scenes (no free joint in the metadata scan)
                    # must not write root pose/velocity.  For URDF inputs the
                    # fix/float choice is a converter flag: it comes from the
                    # matching SceneEntitySpec root_mode when the scene
                    # declares entity_assets, else the step-0 default (fixed).
                    "fixed_base": self._get_scene_metadata().freejoint_body_name is None,
                    **runtime_payload,
                    **worker_init_payload,
                    "root_body_name": self._base_name
                    or self._get_scene_metadata().freejoint_body_name,
                    # Some importers do not preserve MJCF traversal order.
                    # Send the cold-path body contract explicitly so a worker
                    # can remap native link indices before publishing state.
                    "mjcf_body_names": list(self._get_scene_metadata().body_names),
                    "mjcf_joint_names": list(self._get_scene_metadata().joint_names),
                    # Fixed variants carry their own per-source actuation and
                    # keyframe tables; the single-model fields are omitted
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
                    # Multi-asset scenes: per-role asset, fixed_base, and
                    # gain-resolved actuation arrays.  Empty list for
                    # single-asset scenes.
                    "entities": self._entity_payloads(),
                    # Declarative world-level ground plane: ``None`` keeps
                    # the backend's native ground behavior.
                    "ground_plane": self._ground_plane_payload(),
                    # Scene-level PhysX solver declaration (``None`` keeps
                    # the backend's own defaults) and the environment clone
                    # grid spacing (``None`` keeps the backend's native
                    # layout; the IsaacSim worker's GridCloner default is
                    # 2.0 m).
                    "scene_physx": self._scene_physx_payload(),
                    "env_grid_spacing": self._env_grid_spacing_payload(),
                    **self._init_randomization_payload(),
                },
                expect=protocol.CMD_META,
            )
            self._worker_init_meta = dict(meta)
            self._bind_model_metadata(meta)
            self._graphics_enabled = bool(meta.get("graphics_enabled", False))
            self._validate_initial_keyframe()
            self._allocate_slots()
            self._request(
                protocol.CMD_ATTACH, {"slots": self._slot_specs()}, expect=protocol.CMD_READY
            )
            self._sensor_map = self._resolve_sensor_map()
            if self._base_name is not None:
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

    def _primary_entity_spec(self) -> SceneEntitySpec | None:
        """Return the declared entity whose asset is the scene's primary model file."""
        if not self._scene.entity_assets:
            return None
        model_path = Path(self._scene.model_file).expanduser().resolve()
        for spec in self._scene.entity_assets:
            if Path(spec.model_file).expanduser().resolve() == model_path:
                return spec
        return None

    def _get_scene_metadata(self) -> SceneMetadata:
        """Return the parent-side MJCF scan, scanning lazily on first access.

        This is pure XML metadata — no worker handshake is required, matching
        the MuJoCo backend where the model (and thus keyframes) is available
        right after construction.  ``materialize()`` reuses this cache.  When
        the primary asset is a URDF declared in ``SceneCfg.entity_assets``,
        the entity's ``root_mode`` supplies the converter fixed/floating
        choice; otherwise the step-0 default (fixed) applies.
        """
        if self._scene_metadata is None:
            model_path = Path(self._scene.model_file).expanduser()
            spec = self._primary_entity_spec()
            urdf_fixed_base = (
                spec.fixed_base
                if spec is not None and model_path.suffix.lower() == ".urdf"
                else None
            )
            self._scene_metadata = scan_scene_metadata(
                str(model_path),
                backend_label=self._BACKEND_LABEL,
                urdf_fixed_base=urdf_fixed_base,
            )
        return self._scene_metadata

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
                )
                for variant in self._fixed_variant_plan.variants
            )
            self._validate_fixed_variant_metadata(metadata)
            self._fixed_variant_metadata = metadata
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

    def _get_entity_metadata(self) -> dict[str, SceneMetadata]:
        """Return per-role metadata for ``SceneCfg.entity_assets`` (cold path).

        Scans lazily on first access, validating per-role fixed_base and
        actuator gain overrides fail-closed before any worker is spawned.
        Empty when the scene declares no entity assets.
        """
        if self._entity_metadata is None:
            self._entity_metadata = scan_scene_entities(
                self._scene.entity_assets, backend_label=self._BACKEND_LABEL
            )
        return self._entity_metadata

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
        if raw_origins is None:
            self._worker_env_origins = None
            self._collision_filtering_applied = False
        else:
            origins = np.asarray(raw_origins, dtype=np.float32)
            expected_origins = (self._num_envs, 3)
            if origins.shape != expected_origins or not np.isfinite(origins).all():
                raise self._worker_error(
                    f"{self._BACKEND_LABEL} worker environment origins have invalid "
                    "shape or values: "
                    f"got shape {origins.shape}, expected {expected_origins}"
                )
            self._worker_env_origins = origins.copy()
            self._collision_filtering_applied = bool(meta.get("collision_filtering_applied", False))
        self._body_id_by_name = {name: index for index, name in enumerate(body_names)}
        self._dof_id_by_name = {name: index for index, name in enumerate(dof_names)}
        self._validate_fixed_variant_handshake(meta)
        self._bind_rigid_root_entities(meta)
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
            "variant_keyframe_qpos": [
                None if value is None else [float(item) for item in value] for value in initial_qpos
            ],
        }

    def _bind_rigid_root_entities(self, meta: dict[str, Any]) -> None:
        """Bind declared rigid entities to shm slots and extended body ids.

        Runs at INIT metadata binding (cold path).  The worker's ``entities``
        INIT meta is cross-checked against the declared specs fail-closed, then
        each rigid entity's scanned root body name is mapped to the extended
        body id ``num_bodies + k`` (declaration order), so the public
        ``get_body_*_w`` getters route those names to the per-entity
        ``entity_root_state__`` slot while robot bodies keep the existing
        ``body_state`` slot.  Legacy scenes bind nothing.
        """
        rigid_specs = tuple(
            spec
            for spec in self._scene.entity_assets
            if spec.materialization == ENTITY_MATERIALIZATION_RIGID
        )
        if not rigid_specs:
            if meta.get("entities"):
                raise self._worker_error(
                    f"{self._BACKEND_LABEL} worker reported scene entities but the scene "
                    "declares no entity_assets; refusing to guess a mapping"
                )
            return
        worker_entities = meta.get("entities")
        if not isinstance(worker_entities, list):
            raise self._worker_error(
                f"{self._BACKEND_LABEL} scene declares entity_assets but the worker INIT "
                "meta carries no entities list; the worker is out of sync with the host"
            )
        worker_rigid = [
            str(entry.get("name"))
            for entry in worker_entities
            if str(entry.get("materialization")) == ENTITY_MATERIALIZATION_RIGID
        ]
        declared = [spec.name for spec in rigid_specs]
        if worker_rigid != declared:
            raise self._worker_error(
                f"{self._BACKEND_LABEL} worker rigid entities {worker_rigid} do not match "
                f"the declared scene entities {declared}; refusing to bind root slots"
            )
        entity_metadata = self._get_entity_metadata()
        self._rigid_root_entities = tuple(declared)
        self._rigid_root_body_names = {}
        for index, spec in enumerate(rigid_specs):
            scanned = entity_metadata[spec.name]
            root_body = scanned.freejoint_body_name or scanned.urdf_root_link_name
            if not root_body:
                raise self._worker_error(
                    f"{self._BACKEND_LABEL} rigid entity {spec.name!r} has no scanned root "
                    "body name; cannot bind a root state slot"
                )
            if (
                root_body in self._body_id_by_name
                or root_body in self._rigid_root_body_names.values()
            ):
                raise self._worker_error(
                    f"{self._BACKEND_LABEL} rigid entity {spec.name!r} root body "
                    f"{root_body!r} collides with an existing body name; refusing to "
                    "overload the body map"
                )
            self._rigid_root_body_names[spec.name] = root_body
            self._body_id_by_name[root_body] = self._model_info.num_bodies + index

    def _position_actuation_payload(
        self,
        metadata: SceneMetadata | None = None,
        gain_overrides: tuple[ActuatorGainOverride, ...] | None = None,
    ) -> dict[str, list[float]]:
        """Per-dof PD/limit/dynamics arrays in MJCF joint document order.

        The worker maps them onto the asset's dof order by name.  Joints with
        no ``<position>`` actuator are passive: zero gains and zero effort.

        Owner-supplied ``gain_overrides`` (by joint name) replace the scanned
        stiffness/damping and, when set, armature/frictionloss; unknown joint
        names fail closed.  With both arguments left as ``None`` the primary
        scene metadata is used, picking up the matching entity spec's gain
        overrides when the scene declares one for ``model_file``.
        """
        if metadata is None:
            metadata = self._get_scene_metadata()
            spec = self._primary_entity_spec()
            gain_overrides = () if spec is None else spec.actuator_gain_overrides
        overrides = {override.joint_name: override for override in gain_overrides or ()}
        unknown = sorted(set(overrides) - set(metadata.joint_names))
        if unknown:
            raise ValueError(
                f"{self._BACKEND_LABEL} actuator gain overrides reference joints not in "
                f"the scanned asset: {unknown}; scanned joints: {list(metadata.joint_names)}"
            )
        by_joint = {spec.joint_name: spec for spec in metadata.actuators}
        stiffness: list[float] = []
        damping: list[float] = []
        effort: list[float] = []
        armature: list[float] = []
        friction: list[float] = []
        for index, joint in enumerate(metadata.joint_names):
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
            armature.append(float(metadata.joint_armature[index]))
            friction.append(float(metadata.joint_frictionloss[index]))
            override = overrides.get(joint)
            if override is not None:
                stiffness[-1] = float(override.stiffness)
                damping[-1] = float(override.damping)
                if override.armature is not None:
                    armature[-1] = float(override.armature)
                if override.frictionloss is not None:
                    friction[-1] = float(override.frictionloss)
        return {
            "dof_stiffness": stiffness,
            "dof_damping": damping,
            "dof_effort": effort,
            "dof_armature": armature,
            "dof_friction": friction,
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

    def _entity_payloads(self) -> list[dict[str, Any]]:
        """Per-entity INIT entries for multi-asset scenes (SimToolReal step 1).

        Each entry mirrors the top-level single-asset INIT contract (actuation
        arrays in joint document order, ``fixed_base``/``root_body_name`` from
        the per-role scan) plus the typed declaration fields, so the worker
        can materialize one articulation/rigid object per role.  Contact
        material declarations (``friction``/``friction_by_body``) are appended
        only when the spec declares them.  Empty when the scene declares no
        ``entity_assets``; the single-asset top-level keys are unchanged.
        """
        specs = tuple(self._scene.entity_assets)
        if not specs:
            return []
        metadata_by_name = self._get_entity_metadata()
        payloads: list[dict[str, Any]] = []
        for spec in specs:
            metadata = metadata_by_name[spec.name]
            entry: dict[str, Any] = {
                "name": spec.name,
                "model_file": str(Path(spec.model_file).expanduser()),
                "asset_format": spec.asset_format,
                "materialization": spec.materialization,
                "root_mode": spec.root_mode,
                "fixed_base": metadata.freejoint_body_name is None,
                # Floating assets name their free-joint body; fixed-base
                # URDF assets fall back to the scanned root link so the
                # worker can still locate the articulation root prim.
                "root_body_name": (metadata.freejoint_body_name or metadata.urdf_root_link_name),
                "joint_names": list(metadata.joint_names),
                "body_names": list(metadata.body_names),
                **self._position_actuation_payload(
                    metadata, gain_overrides=spec.actuator_gain_overrides
                ),
            }
            # Contact materials are opt-in: undeclared entities carry no
            # friction keys at all.
            if spec.contact_friction is not None:
                entry["friction"] = [float(value) for value in spec.contact_friction]
                if spec.contact_friction_by_body:
                    entry["friction_by_body"] = {
                        override.body_name: [float(value) for value in override.friction]
                        for override in spec.contact_friction_by_body
                    }
            # Composition is declaration-driven: spawn pose, USD-bake
            # collision flag, converter capsule flag, and the pool mirror
            # binding ride the entity entry only when declared.
            if spec.init_state is not None:
                entry["init_state"] = {
                    "pos": [float(value) for value in spec.init_state.pos],
                    "rot_wxyz": [float(value) for value in spec.init_state.rot_wxyz],
                }
            if spec.collision_enabled is not None:
                entry["collision_enabled"] = bool(spec.collision_enabled)
            if spec.replace_cylinders_with_capsules is not None:
                entry["replace_cylinders_with_capsules"] = bool(
                    spec.replace_cylinders_with_capsules
                )
            if spec.mirrors_fixed_variant_pool:
                entry["mirrors_fixed_variant_pool"] = True
            payloads.append(entry)
        return payloads

    def _init_randomization_payload(self) -> dict[str, Any]:
        """INIT entry for the validated variant pool; absent for unpolled scenes."""
        if self._init_variant_pool is None:
            return {}
        return {"variant_pool": dict(self._init_variant_pool)}

    def _ground_plane_payload(self) -> dict[str, Any] | None:
        """Serialize ``SceneCfg.ground_plane`` into the INIT payload.

        ``None`` keeps the backend's native ground behavior; a declaration
        carries the PhysX triple, the restitution scalar, and the extent in
        meters.  The key is always present so the wire contract is explicit.
        Only adapters whose worker consumes the declaration reach this
        serialization; every other backend rejected the scene at
        construction (:func:`validate_scene_composition_support`).
        """
        declaration = self._scene.ground_plane
        if declaration is None:
            return None
        return {
            "friction": [float(value) for value in declaration.friction],
            "restitution": float(declaration.restitution),
            "size_m": float(declaration.size_m),
        }

    def _scene_physx_payload(self) -> dict[str, Any] | None:
        """Serialize ``SceneCfg.physx`` into the INIT payload.

        ``None`` keeps the worker's own PhysX defaults; a declaration carries
        the validated :class:`ScenePhysxCfg` fields verbatim.  The key is
        always present so the wire contract is explicit, and only adapters
        whose worker applies scene-level PhysX settings reach this
        serialization.
        """
        declaration = self._scene.physx
        if declaration is None:
            return None
        return dict(declaration.as_kwargs())

    def _env_grid_spacing_payload(self) -> float | None:
        """Serialize ``SceneCfg.env_grid_spacing`` into the INIT payload.

        ``None`` keeps the worker's native environment layout; the IsaacSim
        worker's ``GridCloner`` default is 2.0 m.  The key is always present
        so the wire contract is explicit.
        """
        spacing = self._scene.env_grid_spacing
        return None if spacing is None else float(spacing)

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
        shapes = protocol.slot_shapes(
            self._num_envs,
            self._model_info.num_dof,
            self._model_info.num_bodies,
            rigid_root_entities=self._rigid_root_entities,
        )
        # Scenes without rigid entities allocate exactly protocol.SLOT_NAMES.
        for name, shape in shapes.items():
            handle = shared_memory.SharedMemory(create=True, size=protocol.slot_nbytes(name, shape))
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
        if self._model_info is not None:
            return self._model_info.dof_names
        return self._get_scene_metadata().joint_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        """The shared position-control profile drives one actuator per DoF."""
        return self.get_actuator_names()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-dof (kp, kd) from the MJCF ``<position>`` actuators (pure XML).

        Passive joints (no actuator) report zero gains.  Returned in MJCF
        joint document order, which the INIT handshake pins to the worker's
        dof order.
        """
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
        initial_qpos = self._effective_initial_qpos()
        if initial_qpos is not None:
            # The selected scene keyframe is the backend default state (and the
            # post-INIT worker state).
            return initial_qpos.copy()
        qpos = np.zeros((_ROOT_QPOS_DIM + self._num_dof(),), dtype=np.float32)
        qpos[3] = 1.0
        return qpos

    def get_default_dof_pos(self) -> np.ndarray:
        initial_qpos = self._effective_initial_qpos()
        if initial_qpos is not None:
            return initial_qpos[_ROOT_QPOS_DIM:].copy()
        return np.zeros((self._num_dof(),), dtype=np.float32)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((_ROOT_QVEL_DIM + self._num_dof(),), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self._model_info is not None:
            root_name = self._model_info.body_names[0]
            if root_body_name in self._rigid_root_body_names.values():
                # Multi-asset rigid roots use the same 7+6 public layout as
                # the primary actor, but their rows live in the entity root
                # slots rather than the robot qpos/qvel slots.
                return BackendRootStateLayout(
                    qpos_indices=tuple(range(_ROOT_QPOS_DIM)),
                    qvel_indices=tuple(range(_ROOT_QVEL_DIM)),
                )
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

    def get_rigid_root_entities(self) -> tuple[str, ...]:
        """Return materialized rigid scene entities with independent root slots."""
        return tuple(self._rigid_root_entities)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        body_map = self._body_name_map()
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(body_map[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in {self._BACKEND_LABEL} model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_joint_range(self) -> np.ndarray | None:
        """Per-joint ``range`` from the MJCF (pure XML, available pre-materialize).

        The XML is the cross-runtime source of truth for this contract.
        Joints without a ``range`` attribute report ``(-inf, inf)``.
        """
        metadata = self._get_scene_metadata()
        if not metadata.joint_ranges:
            return None
        return np.asarray(metadata.joint_ranges, dtype=np.float32)

    def get_gravity(self) -> np.ndarray:
        info = self._require_materialized()
        return np.asarray(info.gravity, dtype=np.float32).copy()

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to absolute qvel indices (root 6 columns first)."""
        return self._resolve_dof_ids(names) + _ROOT_QVEL_DIM

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self._resolve_dof_ids(names) + _ROOT_QPOS_DIM

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
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
        # Fail closed on registration: every physics substep is integrated
        # inside the worker process, so storing a host callback would silently
        # drop it (declared gap).  Clearing with
        # ``None`` keeps the base unregister contract because "no callback" is
        # this family's real state.
        if fn is not None:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} pre-step control callbacks require per-substep host "
                "control inside the physics worker; not implemented for the subprocess family "
                "(declared gap)"
            )
        self._pre_step_control_fn = None

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_state("step")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        info = self._require_materialized()
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, info.num_dof)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")

        profile_detail = os.environ.get("UNISIM_PROFILE_DETAIL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        t0 = time.perf_counter()
        np.copyto(self._slots["ctrl"], ctrl_array)
        control_slot_copy_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        payload = self._request(
            protocol.CMD_STEP, {"nsteps": int(nsteps)}, expect=protocol.CMD_READY
        )
        worker_request_ms = (time.perf_counter() - t0) * 1000.0
        ipc_ms = control_slot_copy_ms + worker_request_ms
        timing = dict(payload.get("timing", {})) if isinstance(payload, dict) else {}
        timing["worker_ipc_total_ms"] = ipc_ms
        if profile_detail:
            timing["host_control_slot_copy_ms"] = control_slot_copy_ms
            timing["host_worker_request_ms"] = worker_request_ms
        return {"timing": timing}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        randomization: ResetRandomizationPayload | None = None,
        *,
        entity_root_states: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_state("set_state")
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
        # ``qpos``/``qvel`` are paired: both None means the robot's generalized
        # state is not touched by this transaction (e.g. a goalviz-only reset,
        # which must not perturb object or robot state — DESIGN.md §4).
        if (qpos is None) != (qvel is None):
            raise ValueError("set_state qpos and qvel must be provided together or both omitted")
        robot_write = qpos is not None
        qpos_array = None
        qvel_array = None
        if robot_write:
            qpos_array = np.asarray(qpos, dtype=np.float32)
            qvel_array = np.asarray(qvel, dtype=np.float32)
            if qpos_array.shape != (rows.size, nq):
                raise ValueError(
                    f"qpos must have shape ({rows.size}, {nq}), got {qpos_array.shape}"
                )
            if qvel_array.shape != (rows.size, nv):
                raise ValueError(
                    f"qvel must have shape ({rows.size}, {nv}), got {qvel_array.shape}"
                )
            if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
                raise ValueError("set_state qpos/qvel must be finite (no NaN or Inf)")
        entity_states: dict[str, np.ndarray] = {}
        for entity, value in (entity_root_states or {}).items():
            name = str(entity)
            if name not in self._rigid_root_entities:
                raise ValueError(
                    f"set_state entity_root_states must name declared rigid scene "
                    f"entities {list(self._rigid_root_entities)}, got {name!r}"
                )
            # Layout matches the read slot: pos xyz, quat wxyz, world linear
            # velocity, world angular velocity (batch-first).
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (rows.size, 13):
                raise ValueError(
                    f"set_state entity_root_states[{name!r}] must have shape "
                    f"({rows.size}, 13), got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"set_state entity_root_states[{name!r}] must be finite (no NaN or Inf)"
                )
            entity_states[name] = array
        if not robot_write and not entity_states and rows.size > 0:
            raise ValueError(
                "set_state requires qpos/qvel or at least one entity root state; "
                "an empty write would be a silent no-op"
            )

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

        profile_detail = os.environ.get("UNISIM_PROFILE_DETAIL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        upload_t0 = time.perf_counter()
        count = int(rows.size)
        # A reset cancels any wrench staged for the selected rows by an
        # interval event earlier in the same control step.
        # The original clears its wrench buffers inside the task reset
        # (reset_utils.py:405-406), so a freshly reset row must not receive a
        # pre-reset impulse; the worker applies these slots at the next
        # CMD_STEP and only clears them afterwards.  Legacy single-asset
        # scenes have no wrench slots.
        for slot_name in (protocol.WRENCH_FORCE_SLOT, protocol.WRENCH_TORQUE_SLOT):
            slot = self._slots.get(slot_name)
            if slot is not None:
                slot[rows] = 0.0
        t0 = time.perf_counter()
        np.copyto(self._slots["reset_env_ids"][:count], rows.astype(np.int32))
        reset_ids_copy_ms = (time.perf_counter() - t0) * 1000.0
        # Legacy scenes (no rigid entities, robot state always written) send
        # exactly {"count": count}; the extra keys exist only for multi-asset
        # scenes so older mujoco-family workers never see them.
        payload: dict[str, Any] = {"count": count}
        if robot_write:
            assert qpos_array is not None and qvel_array is not None
            t0 = time.perf_counter()
            np.copyto(self._slots["reset_qpos"][:count], qpos_array)
            np.copyto(self._slots["reset_qvel"][:count], qvel_array)
            reset_robot_copy_ms = (time.perf_counter() - t0) * 1000.0
        else:
            reset_robot_copy_ms = 0.0
        reset_entity_copy_ms = 0.0
        t0 = time.perf_counter()
        for name, array in entity_states.items():
            np.copyto(self._slots[protocol.entity_reset_state_slot(name)][:count], array)
        reset_entity_copy_ms = (time.perf_counter() - t0) * 1000.0
        if self._rigid_root_entities:
            payload["robot"] = robot_write
            payload["entity_roots"] = sorted(entity_states)
        host_upload_ms = (time.perf_counter() - upload_t0) * 1000.0
        request_t0 = time.perf_counter()
        payload = self._request(protocol.CMD_SET_STATE, payload, expect=protocol.CMD_READY)
        request_ms = (time.perf_counter() - request_t0) * 1000.0
        ipc_ms = host_upload_ms + request_ms
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
        if profile_detail:
            timing["set_state_host_env_ids_copy_ms"] = reset_ids_copy_ms
            timing["set_state_host_robot_copy_ms"] = reset_robot_copy_ms
            timing["set_state_host_entity_copy_ms"] = reset_entity_copy_ms
            timing["set_state_host_upload_total_ms"] = host_upload_ms
            timing["set_state_host_worker_request_ms"] = request_ms
        return {"timing": timing}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise interval wrench support for declared rigid entities.

        Scenes carrying a ``fixed_variant_plan`` (the variant-pool channel
        staged by :meth:`materialize`) additionally declare fixed-variant
        support under the ``SAME_LAYOUT`` guarantee together with
        per-environment playback — ``get_playback_model(env_index)`` resolves
        each environment to its assigned variant source — whether or not
        rigid root entities are bound yet.  ``UNIFORM_PUBLIC_LAYOUT`` stays
        undeclared: the rigid-object pool realizes one identical public body
        layout per variant and cannot provide optional public slots.
        """
        pooled = self._scene.fixed_variant_plan is not None
        if not self._rigid_root_entities and not pooled:
            return DomainRandomizationCapabilities()
        fields: dict[str, Any] = {}
        if self._rigid_root_entities:
            fields.update(
                supports_interval_body_force=True,
                supports_interval_body_torque=True,
                supported_interval_terms=frozenset(
                    {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
                ),
            )
        if pooled:
            fields.update(
                supports_fixed_variants=True,
                supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
                supports_per_env_playback=True,
            )
        return DomainRandomizationCapabilities(**fields)

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the assigned variant source for one environment.

        Pooled scenes — the IsaacGym model-level plan and the IsaacSim
        entity-bound pool both carry ``SceneCfg.fixed_variant_plan`` —
        resolve each environment to its assigned variant's model file; the
        request must name one environment explicitly.  Scenes without a
        plan keep the base behavior and return the backend model.
        """
        plan = self._scene.fixed_variant_plan
        if plan is None:
            return super().get_playback_model(env_index)
        if env_index is None:
            raise ValueError(
                f"{self._BACKEND_LABEL} fixed-variant playback requires an explicit "
                "env_index"
            )
        if isinstance(env_index, bool) or not isinstance(env_index, int):
            raise TypeError("env_index must be an integer or None")
        if env_index < 0 or env_index >= self._num_envs:
            raise IndexError(f"env_index must be in [0, {self._num_envs - 1}]")
        variant_index = int(plan.assignment[env_index])
        return plan.variants[variant_index].model_file

    def get_entity_variant_metadata(self, entity: str) -> FixedVariantMetadata:
        """Expand the worker-measured variant masses per environment.

        The worker measures every pool variant's mass from the baked USD at
        INIT and reports the table through the INIT metadata
        (backend-authoritative measurement, not a payload echo);
        this readback expands it to ``(num_envs,)`` with the pool assignment
        already validated at payload assembly.  Every direction fails
        closed: no pool, an entity the pool is not bound to, or missing
        worker measurements raise instead of returning defaults.
        """
        if self._scene.fixed_variant_plan is None:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} scene carries no fixed variant pool"
            )
        self._require_materialized()
        pool = self._init_variant_pool
        assert pool is not None  # materialize() assembles it from the plan
        meta = self._worker_init_meta
        assert meta is not None  # set together with _model_info by materialize()
        if entity != pool["target_entity"]:
            raise ValueError(
                f"{self._BACKEND_LABEL} fixed variant metadata requested for entity "
                f"{entity!r} but the pool targets {pool['target_entity']!r}"
            )
        assignment_meta = meta.get("variant_assignment")
        masses = None if assignment_meta is None else assignment_meta.get("masses")
        if masses is None:
            raise RuntimeError(
                f"{self._BACKEND_LABEL} worker did not report measured variant masses"
            )
        measured = np.asarray([float(value) for value in masses], dtype=np.float64)
        if measured.size != len(pool["source_files"]):
            raise RuntimeError(
                f"{self._BACKEND_LABEL} worker reported {measured.size} measured variant "
                f"masses for {len(pool['source_files'])} pool sources"
            )
        return FixedVariantMetadata(
            mass=measured[np.asarray(pool["assignments"], dtype=np.intp)],
            variant_files=tuple(pool["source_files"]),
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Apply a world-frame force (and optional torque) to rigid roots.

        Base-contract staging entry: ``body_ids`` select rigid entity roots,
        ``force`` has shape ``(num_envs, len(body_ids), 3)`` in the world
        frame, and ``torque`` is optional with the same shape.  Values
        accumulate into the dense ``WRENCH_FORCE_SLOT``/``WRENCH_TORQUE_SLOT``
        shm rows **within one interval plan**: each non-empty plan starts
        from cleared staging (the prologue in
        :meth:`apply_interval_randomization`), so submissions from separate
        plans replace rather than add to each other.  The worker applies the
        staged rows at the next control step and clears the slots afterwards,
        and ``set_state`` zeroes the selected rows so a freshly reset row
        never receives a pre-reset impulse.
        """
        if not self._rigid_root_entities:
            raise NotImplementedError(
                f"{self._BACKEND_LABEL} worker has no rigid-entity wrench slots"
            )
        ids = np.asarray(body_ids, dtype=np.intp)
        info = self._require_materialized()
        extended = info.num_bodies + len(self._rigid_root_entities)
        if (
            ids.ndim != 1
            or ids.size == 0
            or np.any(ids < info.num_bodies)
            or np.any(ids >= extended)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError(
                f"{self._BACKEND_LABEL} wrench body_ids must select unique rigid roots in "
                f"[{info.num_bodies}, {extended}), got {ids.tolist()}"
            )
        values = np.asarray(force, dtype=np.float32)
        expected = (self._num_envs, ids.size, 3)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"wrench force must have finite shape {expected}, got {values.shape}")
        if torque is None:
            torque_values = np.zeros(expected, dtype=np.float32)
        else:
            torque_values = np.asarray(torque, dtype=np.float32)
            if torque_values.shape != expected or not np.isfinite(torque_values).all():
                raise ValueError(
                    f"wrench torque must have finite shape {expected}, got {torque_values.shape}"
                )
        self._slots[protocol.WRENCH_FORCE_SLOT][:, ids, :] += values
        self._slots[protocol.WRENCH_TORQUE_SLOT][:, ids, :] += torque_values

    _interval_handlers: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        """Return the wrench staging handler table, built lazily exactly once.

        Both wrench terms route through the public :meth:`apply_body_force`
        staging entry; every other term fails closed in the base dispatch.
        The empty-table case is deliberately not cached: rigid roots bind at
        INIT metadata binding (cold path), so a pre-INIT ``{}`` must not
        shadow the real table a rigid scene acquires afterwards.
        """
        if not self._rigid_root_entities:
            return {}
        if self._interval_handlers is None:
            self._interval_handlers = {
                INTERVAL_TERM_BODY_FORCE: (
                    lambda op: self.apply_body_force(op.body_ids, op.payload, None)
                ),
                INTERVAL_TERM_BODY_TORQUE: (
                    lambda op: self.apply_body_force(
                        op.body_ids,
                        np.zeros_like(op.payload, dtype=np.float32),
                        op.payload,
                    )
                ),
            }
        return self._interval_handlers

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Stage interval force/torque rows for the next worker control step.

        Thin prologue override per the base contract: a non-empty plan on a
        rigid scene starts from cleared wrench slots, then the base handler
        table dispatch accumulates the ops through the staging entry.  The
        per-plan prologue is exactly the form the base contract sanctions
        ("Backends that need per-plan prologue/epilogue semantics (for
        example clearing staged external forces before the ops accumulate)
        keep a thin override that calls this base implementation"), so
        :meth:`apply_body_force` accumulation is bounded by one plan.
        Caveat: the prologue clears the whole staging, so if multiple wrench
        terms each arrive as their own plan, the later plan's prologue drops
        the earlier plan's staged rows — under the current contract a plan
        carries at most one wrench term (SimToolReal's plans do), so this
        cannot fire today.
        """
        if plan.is_empty():
            return
        if not self._rigid_root_entities:
            return super().apply_interval_randomization(plan)
        self._require_state("apply_interval_randomization")
        self._slots[protocol.WRENCH_FORCE_SLOT].fill(0.0)
        self._slots[protocol.WRENCH_TORQUE_SLOT].fill(0.0)
        super().apply_interval_randomization(plan)

    # ------------------------------------------------------------------ #
    # Native rendering / playback (worker-owned viewer and camera sensor)
    # ------------------------------------------------------------------ #

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_native_video_capture=True,
        )

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
                "camera": _normalize_camera_kwargs(camera_kwargs),
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
        return self._slots["dof_state"][:, :, 0]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
        return self._slots["dof_state"][:, :, 1]

    def _selected_body_state(self, body_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(body_ids, dtype=np.intp)
        info = self._require_materialized()
        num_robot_bodies = info.num_bodies
        num_extended = num_robot_bodies + len(self._rigid_root_entities)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= num_extended):
            raise ValueError(
                f"body_ids must be a 1-D array in [0, {num_extended}), got {body_ids!r}"
            )
        if not self._rigid_root_entities or not np.any(ids >= num_robot_bodies):
            # Legacy path (robot-only ids): unchanged fancy-index into the
            # body_state slot.
            return self._body_slot()[:, ids, :]
        # Mixed selection: robot bodies come from body_state; rigid entity
        # roots (extended ids, declaration order) come from their per-entity
        # root state slots.
        result = np.empty((self._num_envs, ids.size, 13), dtype=np.float32)
        robot_mask = ids < num_robot_bodies
        if np.any(robot_mask):
            result[:, robot_mask, :] = self._body_slot()[:, ids[robot_mask], :]
        for index, entity in enumerate(self._rigid_root_entities):
            entity_mask = ids == num_robot_bodies + index
            if np.any(entity_mask):
                result[:, entity_mask, :] = self._slots[protocol.entity_root_state_slot(entity)][
                    :, None, :
                ]
        return result

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
        state = self._body_slot()[:, body_id, :]
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
    "build_init_variant_pool_payload",
]
