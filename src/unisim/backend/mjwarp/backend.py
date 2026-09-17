"""Host-compatibility implementation of the independent ``mjwarp`` backend.

``mjwarp`` is not a MuJoCo backend mode.  It uploads a CPU MuJoCo model to
``mujoco_warp`` and owns its own device data and host cache.  The cache is
refreshed exactly at explicit step/reset barriers; legacy getters only return
views into that cache and therefore never trigger an implicit Warp ``.numpy``
transfer.
"""

from __future__ import annotations

import gc
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from os import PathLike
from typing import Any, NoReturn

import numpy as np

from unisim.backend.base import (
    BackendMocapPoseBinding,
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    CameraCfg,
    DebugOverlayGetter,
    SimBackend,
    normalize_play_render_mode,
)
from unisim.backend.model_index import CompiledModelIndex
from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_IQUAT,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GEOM_SIZE,
    RESET_TERM_GEOM_SOLIMP,
    RESET_TERM_GEOM_SOLREF,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    FixedVariantPlan,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ResetRandomizationPayload,
    _validate_reset_term,
    require_op_body_ids,
)
from unisim.entities import SceneResetRequest
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset, row_columns
from unisim.inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ConfigurationScope,
    ImportReport,
    compare_configuration,
    mujoco_model_configuration,
)
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_layout import CompiledSceneLayout
from unisim.utils.rotation import np_quat_apply_inverse_batched

from ..body_state import copy_selected_body_state
from ..reset_impact import bind_reset_impacts
from .dependencies import load_mjwarp_dependencies
from .materialization import materialize_mjwarp_scene
from .playback import run_mjwarp_playback, validate_mjwarp_visual_model
from .randomization import PrimitiveGeomBounds, expand_model_fields
from .state_commit import ModelUpdates, StateCommitPlan, float_values
from .variants import FixedVariantRealization, install_fixed_variant_fields, prepare_fixed_variants

_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)
# Reset scratch storage is deliberately bounded.  The original 128-world
# allocation covers the smaller G1 owners, while the 8192-world motion
# tracking owner routinely resets about 250 rows per vector step.  Scale the
# cold-path allocation with the world count and cap it at 512 so that this
# workload stays on the sparse route without making an unbounded per-task
# allocation.  Keeping the minimum at 128 preserves the #1288 route for the
# 1024/2048-world owners.
_RESET_SCRATCH_CAPACITY = 512
_RESET_SCRATCH_MIN_CAPACITY = 128
_RESET_SCRATCH_MIN_BATCH_SIZE = 8 * _RESET_SCRATCH_MIN_CAPACITY
_RESET_SCRATCH_WORLD_FRACTION = 16


def _reset_scratch_capacity_for_batch(num_envs: int) -> int:
    """Choose a bounded, power-of-two scratch capacity on the cold path.

    A fixed shape is required by the captured reset graphs.  The capacity is
    rounded down to a power of two so graph/data shapes stay predictable, and
    is never allowed below the proven 128-world route or above the 512-world
    memory bound.  Small batches retain the eager/full-forward fallback rather
    than paying for a second ``mujoco_warp.Data`` instance.
    """
    if num_envs < _RESET_SCRATCH_MIN_BATCH_SIZE:
        return 0
    target = max(_RESET_SCRATCH_MIN_CAPACITY, num_envs // _RESET_SCRATCH_WORLD_FRACTION)
    power_of_two = 1 << (target.bit_length() - 1)
    return min(_RESET_SCRATCH_CAPACITY, power_of_two)


@contextmanager
def _suspend_gc() -> Iterator[None]:
    """Keep graph finalizers from running inside a new Warp capture."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _cuda_graph_eligibility(warp: Any, device: Any) -> tuple[bool, str | None]:
    """Return the cold-path CUDA graph decision and a fallback diagnostic."""
    if not bool(device.is_cuda):
        return False, "active Warp device is not CUDA"

    try:
        driver_version = warp.get_cuda_driver_version()
    except Exception as exc:
        return False, f"CUDA driver query failed: {type(exc).__name__}: {exc}"
    if driver_version is None:
        return False, "CUDA driver version is unavailable"

    try:
        mempool_enabled = bool(warp.is_mempool_enabled(device))
    except Exception as exc:
        return False, f"CUDA mempool query failed: {type(exc).__name__}: {exc}"

    reasons: list[str] = []
    if tuple(driver_version) < _GRAPH_CAPTURE_MIN_DRIVER:
        reasons.append(f"CUDA driver {driver_version[0]}.{driver_version[1]} is older than 12.4")
    if not mempool_enabled:
        reasons.append("CUDA mempool is disabled")
    if reasons:
        return False, "; ".join(reasons)
    return True, None


class MjwarpBackend(SimBackend):
    """Independent CUDA backend exposed through the host NumPy profile.

    State and control cross the host/device boundary only at explicit
    step/reset barriers with bounded, statically declared transfers.  Reset DR
    writes per-world rows of cold-path-expanded model arrays in place and
    recomputes derived constants with the graded ``set_const*`` family;
    interval DR stages ``xfrc_applied`` pushes/body forces for the next step
    barrier or kicks root velocity through the reset upload+forward path.
    A registered pre-step control converter runs on the host before every
    physics substep (see ``set_pre_step_control``) and may additionally return
    a per-substep body wrench that composes with the staged interval wrenches;
    native rendering remains fail-closed. Detached host snapshots support
    finite MuJoCo-based offline recording.
    """

    _fixed_variant_plan: FixedVariantPlan | None = None
    _fixed_variant_realization: FixedVariantRealization | None = None

    # Host DR mirrors bound by ``_bind_dr_host_mirrors`` on the cold path
    # (declared here because that helper assigns them by name in a loop).
    _dr_geom_size: np.ndarray
    _dr_geom_rbound: np.ndarray
    _dr_geom_aabb: np.ndarray
    _dr_geom_solref: np.ndarray
    _dr_geom_solimp: np.ndarray
    _dr_dof_damping: np.ndarray
    _dr_dof_frictionloss: np.ndarray
    _tracked_sensor_slots: dict[str, tuple[int, int]]

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        add_body_sensors: bool = False,
        **unexpected_kwargs: Any,
    ) -> None:
        self._composed_scene = None
        self._entity_layout: CompiledSceneLayout | None = None
        self._entity_faulted = False
        self._entity_closed = False
        self._entity_constructing = True
        original = scene
        try:
            require_scene_composition_support(scene, "mjwarp")
            if scene.entity_assets:
                from unisim.backend.mujoco.composition import compose_scene

                self._composed_scene = compose_scene(scene, num_envs, sim_dt)
                scene = replace(
                    scene,
                    model_file=self._composed_scene.model_file,
                    fixed_variant_plan=self._composed_scene.variant_plan,
                    entity_assets=(),
                    entity_variant=None,
                )
            self._initialize(
                scene,
                num_envs,
                sim_dt,
                base_name=base_name,
                push_body_name=push_body_name,
                nconmax=nconmax,
                njmax=njmax,
                add_body_sensors=add_body_sensors,
                **unexpected_kwargs,
            )
            if self._composed_scene is not None:
                self._initialize_entities(original)
        except BaseException:
            self._entity_constructing = False
            self.cleanup_scene_assets()
            raise
        self._entity_constructing = False

    def _initialize(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        add_body_sensors: bool = False,
        **unexpected_kwargs: Any,
    ) -> None:
        require_scene_composition_support(scene, "mjwarp")
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"MjwarpBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if not isinstance(add_body_sensors, bool):
            raise TypeError(
                "MjwarpBackend add_body_sensors must be bool, got "
                f"{type(add_body_sensors).__name__}"
            )
        nconmax = self._require_capacity(nconmax, name="nconmax", default=512)
        njmax = self._require_capacity(njmax, name="njmax", default=512)
        if push_body_name is not None and not isinstance(push_body_name, str):
            raise TypeError(
                "MjwarpBackend push_body_name must be str or None, got "
                f"{type(push_body_name).__name__}"
            )

        deps = load_mjwarp_dependencies()
        device = deps.warp.get_device()
        if not bool(device.is_cuda):
            raise RuntimeError(
                "mjwarp backend requires an active CUDA Warp device; choose a CUDA-capable "
                "host or select the mujoco backend."
            )

        scene_context = materialize_mjwarp_scene(scene, add_body_sensors=add_body_sensors)
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self.scene_model_file = scene_context.diagnostic_model_file
        self.scene_visual_model_file = str(scene.visual_model_file or scene.model_file)
        self._playback_model_validated = False
        self._tracked_body_state_dirty = False
        self.backend_type = "mjwarp"
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        self._push_body_name = push_body_name
        self._nconmax = nconmax
        self._njmax = njmax
        self._add_body_sensors = add_body_sensors
        self._tracked_body_names = scene_context.tracked_body_names
        self._tracked_sensor_slots: dict[str, tuple[int, int]] = {}

        self._mujoco = deps.mujoco
        self._mujoco_warp = deps.mujoco_warp
        self._warp = deps.warp
        self._fixed_variant_plan: FixedVariantPlan | None = scene.fixed_variant_plan
        self._fixed_variant_realization: FixedVariantRealization | None = None
        if self._fixed_variant_plan is not None:
            self._fixed_variant_plan.validate(self._num_envs)
            self._fixed_variant_realization = prepare_fixed_variants(
                self._fixed_variant_plan,
                sim_dt=self._sim_dt,
                sensor_body_names=scene_context.tracked_body_names,
            )
        if self._fixed_variant_realization is None:
            try:
                self._cpu_model = deps.mujoco.MjModel.from_xml_path(scene_context.source_model_file)
            finally:
                # The materialized source (fragment merge and/or injected tracking
                # sensors) is only needed to compile the model; release the
                # temporary files immediately like the MuJoCo backend does.
                self.cleanup_scene_assets()
        else:
            self._cpu_model = self._fixed_variant_realization.canonical_model
            self.cleanup_scene_assets()
        report_requested = mujoco_model_configuration(self._cpu_model, self._mujoco)
        self._cpu_model.opt.timestep = self._sim_dt
        self._device_model = deps.mujoco_warp.put_model(self._cpu_model)
        self._device_data = deps.mujoco_warp.make_data(
            self._cpu_model,
            nworld=self._num_envs,
            # These capacities are owner-configured cold-path physical storage
            # limits.  They are intentionally not inferred or changed during a
            # rollout: a task must select and validate its own safe budget.
            nconmax=nconmax,
            njmax=njmax,
        )

        self._nq = int(self._cpu_model.nq)
        self._nv = int(self._cpu_model.nv)
        self._nu = int(self._cpu_model.nu)
        self._nbody = int(self._cpu_model.nbody)
        self._root_qpos_dim, self._root_qvel_dim = self._root_state_dims()
        self._num_dof_pos = self._nq - self._root_qpos_dim
        self._num_dof_vel = self._nv - self._root_qvel_dim

        self._sensor_slots = self._bind_sensor_slots()
        self._keyframe_qpos = self._bind_keyframes()
        self._body_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_BODY, self._nbody)
        self._joint_ids = self._bind_names(
            deps.mujoco.mjtObj.mjOBJ_JOINT,
            int(self._cpu_model.njnt),
        )
        if self._base_name is None:
            self._base_body_id: int | None = None
        else:
            try:
                self._base_body_id = self._body_ids[self._base_name]
            except KeyError as exc:
                raise ValueError(
                    f"Base body {self._base_name!r} not found in mjwarp model"
                ) from exc
        self._geom_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_GEOM, int(self._cpu_model.ngeom))
        self._site_ids = self._bind_names(deps.mujoco.mjtObj.mjOBJ_SITE, int(self._cpu_model.nsite))
        self._push_body_id = self._resolve_push_body_id()
        self._compiled_index = CompiledModelIndex.from_model(self._cpu_model)
        self._interval_root_velocity_qvel_ids = self._resolve_interval_root_velocity_qvel_ids()
        # Per-world DR expansion replaces model arrays, so it must run on the
        # cold path before the first forward and before CUDA graph capture
        # (captured pointers would otherwise keep reading the shared arrays).
        # Later DR writes are in-place ``assign`` uploads into the expanded
        # arrays and therefore stay graph-safe.
        expand_model_fields(self._warp, self._device_model, self._num_envs)
        if self._fixed_variant_realization is not None:
            assert self._fixed_variant_plan is not None
            install_fixed_variant_fields(
                self._warp,
                self._device_model,
                self._fixed_variant_realization,
                self._fixed_variant_plan.assignment,
            )
            # Variant sources carry independent mass/inertial differences, but
            # compiler-derived invweight/acc0 tables are recomputed by Warp.
            # Refresh them before any host mirror or CUDA graph captures these
            # fixed per-world allocations.
            self._mujoco_warp.set_const(self._device_model, self._device_data)
        self._bind_dr_host_mirrors()
        self._geom_bounds = PrimitiveGeomBounds(self._cpu_model.geom_type, deps.mujoco.mjtGeom)
        mocap_bodies = np.flatnonzero(self._cpu_model.body_mocapid >= 0)
        mocap_bodies = mocap_bodies[np.argsort(self._cpu_model.body_mocapid[mocap_bodies])]
        self._nmocap = len(mocap_bodies)
        self._default_mocap_pos = self._cpu_model.body_pos[mocap_bodies].astype(np.float32)
        self._default_mocap_quat = self._cpu_model.body_quat[mocap_bodies].astype(np.float32)
        self._mocap_pos = np.broadcast_to(
            self._default_mocap_pos, (self._num_envs, len(mocap_bodies), 3)
        ).copy()
        self._mocap_quat = np.broadcast_to(
            self._default_mocap_quat, (self._num_envs, len(mocap_bodies), 4)
        ).copy()
        self._xfrc_staging = np.zeros((self._num_envs, self._nbody, 6), dtype=np.float32)
        self._xfrc_pending = False
        self._actuator_names = tuple(
            deps.mujoco.mj_id2name(
                self._cpu_model,
                deps.mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            or ""
            for actuator_id in range(self._nu)
        )
        self._actuator_ctrl_range = np.asarray(
            self._cpu_model.actuator_ctrlrange,
            dtype=np.float32,
        ).copy()
        self._joint_range = self._bind_joint_range()

        # All legacy getters below return views into these stable pinned host
        # buffers. They are refreshed only by _refresh_host_cache(), called
        # after a device step or a reset/forward lifecycle barrier. Keeping the
        # Warp storage alive lets D2H copies target the public NumPy cache
        # directly instead of allocating a temporary array on every refresh.
        self._qpos_cache_storage, self._qpos_cache = self._allocate_pinned_host_cache(
            self._device_data.qpos
        )
        self._qvel_cache_storage, self._qvel_cache = self._allocate_pinned_host_cache(
            self._device_data.qvel
        )
        self._time_cache = np.zeros((self._num_envs,), dtype=np.float32)
        self._sensor_cache_storage, self._sensor_cache = self._allocate_pinned_host_cache(
            self._device_data.sensordata
        )
        self._ctrl_staging = np.zeros((self._num_envs, self._nu), dtype=np.float32)
        self._reset_mask_host = np.zeros((self._num_envs,), dtype=np.bool_)
        self._reset_mask_device = deps.warp.zeros(self._num_envs, dtype=bool)
        # A bounded secondary Data avoids running reset-time forward over every
        # production world when only a small row set terminated.  It is built
        # only for batches large enough to amortize the extra graph and copies.
        self._reset_scratch_capacity = _reset_scratch_capacity_for_batch(self._num_envs)
        self._reset_scratch_data: Any | None = None
        self._reset_scratch_mask_device: Any | None = None
        self._reset_scratch_qpos_staging: np.ndarray | None = None
        self._reset_scratch_qvel_staging: np.ndarray | None = None
        self._reset_scratch_sensor_storage: Any | None = None
        self._reset_scratch_sensor_cache: np.ndarray | None = None
        # Tracked-body views are zero-copy slices of _sensor_cache; they must be
        # bound before the first forward barrier refreshes the cache below.
        self._body_id_to_tracked_idx: np.ndarray | None = None
        if self._add_body_sensors:
            self._bind_tracked_body_state()
        # Begin from explicit model defaults, run a forward barrier, and cache
        # the resulting sensors/kinematics.  This avoids an uninitialized host
        # cache before NpEnv's first selected-row reset.
        defaults = np.broadcast_to(
            np.asarray(self._cpu_model.qpos0, dtype=np.float32),
            (self._num_envs, self._nq),
        )
        np.copyto(self._qpos_cache, defaults)
        self._qvel_cache.fill(0.0)
        self._upload(self._device_data.qpos, self._qpos_cache)
        self._upload(self._device_data.qvel, self._qvel_cache)
        self._mujoco_warp.forward(self._device_model, self._device_data)
        self._synchronize()
        self._refresh_host_cache()
        self._initialize_cuda_graphs(device)
        self._capture_import_report(report_requested)
        if self._fixed_variant_realization is not None:
            self._fixed_variant_realization = replace(
                self._fixed_variant_realization, report_requested=()
            )

    def _initialize_entities(self, scene: SceneCfg) -> None:
        from unisim.backend.mujoco.composition import compile_scene_layout

        assert self._composed_scene is not None
        layout = compile_scene_layout(self._cpu_model, scene.entity_assets)
        self._composed_scene.layout.require_same_layout(layout)
        self._entity_layout = layout
        self._compiled_index.validate_entity_layout(layout)
        self._entity_reset_impacts = bind_reset_impacts(
            layout, self._cpu_model.actuator_actadr, self._cpu_model.actuator_actnum)
        self._entity_root_ids = tuple(
            entity.body_ids[entity.body_names.index(entity.root_body)] for entity in layout.entities
        )
        self._entity_mocap_ids = tuple(
            int(self._cpu_model.body_mocapid[i]) for i in self._entity_root_ids
        )
        plan = self._composed_scene.variant_plan
        files = (
            [self._composed_scene.model_file]
            if plan is None
            else [v.model_file for v in plan.variants]
        )
        values: dict[str, list[np.ndarray]] = {
            name: [] for name in ("qpos", "qvel", "ctrl", "act", "time", "mocap_pos", "mocap_quat")
        }
        for file in files:
            model = self._mujoco.MjModel.from_xml_path(file)
            layout.require_same_layout(compile_scene_layout(model, scene.entity_assets))
            data = self._mujoco.MjData(model)
            if scene.default_keyframe_name is not None:
                key = self._mujoco.mj_name2id(
                    model, self._mujoco.mjtObj.mjOBJ_KEY, scene.default_keyframe_name
                )
                self._mujoco.mj_resetDataKeyframe(model, data, key)
            for name in values:
                values[name].append(np.asarray(getattr(data, name), dtype=np.float32).copy())
        assignment = np.zeros(self._num_envs, dtype=int) if plan is None else plan.assignment
        self._entity_defaults = {
            name: np.stack(items)[assignment] for name, items in values.items()
        }
        for name, array in self._entity_defaults.items():
            self._upload(getattr(self._device_data, name), array)
        self._ctrl_staging[:] = self._entity_defaults["ctrl"]
        self._mocap_pos[:] = self._entity_defaults["mocap_pos"]
        self._mocap_quat[:] = self._entity_defaults["mocap_quat"]
        self._time_cache[:] = self._entity_defaults["time"]
        self._execute_device_forward()
        self._synchronize()
        self._refresh_host_cache()
        fields = []
        actual = {name: getattr(self._device_data, name).numpy() for name in self._entity_defaults}
        groups = (
            ((None, np.arange(self._num_envs)),)
            if plan is None
            else tuple(
                (str(variant), np.flatnonzero(plan.assignment == variant))
                for variant in range(len(plan.variants))
            )
        )
        for variant, rows in groups:
            scope = ConfigurationScope(env_ids=tuple(int(row) for row in rows), variant=variant)
            requested = {
                name: values[rows].tolist() for name, values in self._entity_defaults.items()
            }
            effective = {name: values[rows].tolist() for name, values in actual.items()}
            fields.append(
                ConfigurationField(
                    "entity.initial_defaults",
                    requested,
                    effective,
                    difference="exact" if requested == effective else "approximate",
                    provenance=(
                        ConfigurationProvenance(
                            "adapter_setting",
                            "Per-environment defaults from composed sources and selected named key",
                        ),
                        ConfigurationProvenance(
                            "engine_readback",
                            "Warp main Data after initialization uploads and forward barrier",
                        ),
                    ),
                    scope=scope,
                    reason="Initial construction snapshot, not current state. Root pose uses "
                    "EntityInitialState and root velocity zero, overriding source/key roots. "
                    "Joint/control/activation defaults come from compiled source keys or qpos0.",
                )
            )
        self._import_report = replace(
            self._import_report, fields=self._import_report.fields + tuple(fields)
        )

    def _require_entity_healthy(self) -> None:
        if getattr(self, "_entity_faulted", False):
            raise RuntimeError("MJWarp backend is faulted after native submission; reconstruct it")
        if getattr(self, "_entity_closed", False):
            raise RuntimeError("MJWarp backend is closed")

    def get_scene_layout(self) -> CompiledSceneLayout:
        self._require_entity_healthy()
        if self._entity_layout is None:
            return super().get_scene_layout()
        return self._entity_layout

    def get_entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.get_scene_layout().entities)

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> Mapping[str, np.ndarray]:
        from unisim.entity_state import selected_state_rows

        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        ids = selected_state_rows(env_ids, self._num_envs)
        index = layout.entities.index(owner)
        roots = np.zeros((len(ids), 13), dtype=np.float32)
        if owner.root_mode == "kinematic":
            mocap = self._entity_mocap_ids[index]
            roots[:, :3] = self._entity_defaults["mocap_pos"][ids, mocap]
            roots[:, 3:7] = self._entity_defaults["mocap_quat"][ids, mocap]
        elif owner.root_mode == "fixed":
            roots[:, :3] = self._cpu_model.body_pos[self._entity_root_ids[index]]
            roots[:, 3:7] = self._cpu_model.body_quat[self._entity_root_ids[index]]
        return entity_state_snapshot(
            owner, self._entity_defaults["qpos"][ids], self._entity_defaults["qvel"][ids], roots
        )

    def _entity_roots(self) -> np.ndarray:
        layout = self.get_scene_layout()
        result = np.zeros((self._num_envs, len(layout.entities), 13), dtype=np.float32)
        for index, entity in enumerate(layout.entities):
            if entity.root_mode == "floating":
                state = entity_state_snapshot(entity, self._qpos_cache, self._qvel_cache)
                result[:, index, :7] = state["root_pose"]
                result[:, index, 7:] = state["root_velocity"]
            elif entity.root_mode == "kinematic":
                mocap = self._entity_mocap_ids[index]
                result[:, index, :3] = self._mocap_pos[:, mocap]
                result[:, index, 3:7] = self._mocap_quat[:, mocap]
            else:
                body = self._entity_root_ids[index]
                result[:, index, :3] = self._cpu_model.body_pos[body]
                result[:, index, 3:7] = self._cpu_model.body_quat[body]
        return result

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        layout = self.get_scene_layout()
        item = layout.get_entity(entity)
        if item.root_mode == "floating":
            return entity_state_snapshot(item, self._qpos_cache, self._qvel_cache)
        index = layout.entities.index(item)
        root = np.zeros((self._num_envs, 13), dtype=np.float32)
        if item.root_mode == "kinematic":
            mocap = self._entity_mocap_ids[index]
            root[:, :3] = self._mocap_pos[:, mocap]
            root[:, 3:7] = self._mocap_quat[:, mocap]
        else:
            body = self._entity_root_ids[index]
            root[:, :3] = self._cpu_model.body_pos[body]
            root[:, 3:7] = self._cpu_model.body_quat[body]
        return entity_state_snapshot(item, self._qpos_cache, self._qvel_cache, root)

    def _entity_persistent_channels(self) -> dict[str, np.ndarray]:
        """One explicit reset-barrier download, never a query hot-path transfer."""
        return {
            name: getattr(self._device_data, name).numpy().copy()
            for name in ("ctrl", "act", "qfrc_applied", "xfrc_applied", "qacc_warmstart")
        }

    def _commit_state(self, plan: StateCommitPlan) -> dict[str, float]:
        """Submit every reset intent through one main Data/native barrier.

        Scratch forward remains a derived-cache optimization for homogeneous
        legacy full resets only; it never selects per-world variant model rows.
        """
        self._require_entity_healthy()
        rows = plan.rows
        timing = {
            name: 0.0
            for name in (
                "reset_upload_ms",
                "reset_forward_ms",
                "host_cache_refresh_ms",
                "model_update_ms",
            )
        }
        if not rows.size:
            return timing
        complement = np.ones(self._num_envs, dtype=bool)
        complement[rows] = False
        sensors = self._sensor_cache.copy()
        dirty = self._tracked_body_state_dirty
        updates = plan.model_updates
        use_scratch = (
            plan.allow_scratch
            and self._can_use_reset_scratch(len(rows))
            and self._fixed_variant_plan is None
            and not self._nmocap
            and not updates.fields
            and not updates.actuator_fields
        )
        try:
            start = time.perf_counter()
            for name, values in updates.fields.items():
                mirror = getattr(self, f"_dr_{name}")
                mirror[:] = values
                target = (
                    self._device_model.opt.gravity
                    if name == "gravity"
                    else getattr(self._device_model, name)
                )
                self._upload(target, mirror)
            if updates.refresh == 2:
                self._mujoco_warp.set_const(self._device_model, self._device_data)
            elif updates.refresh == 1:
                self._mujoco_warp.set_const_0(self._device_model, self._device_data)
            for name, values in updates.actuator_fields.items():
                mirror = getattr(self, f"_dr_{name}")
                mirror[:] = values
                self._upload(getattr(self._device_model, name), mirror)
            timing["model_update_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            if plan.reset_world:
                self._reset_mask_host.fill(False)
                self._reset_mask_host[rows] = True
                self._upload(self._reset_mask_device, self._reset_mask_host)
                self._execute_device_reset()
            self._upload(self._device_data.qpos, plan.qpos)
            self._upload(self._device_data.qvel, plan.qvel)
            self._upload(self._device_data.mocap_pos, plan.mocap_pos)
            self._upload(self._device_data.mocap_quat, plan.mocap_quat)
            self._upload(self._device_data.time, plan.time)
            for name, values in plan.channels.items():
                self._upload(getattr(self._device_data, name), values)
            timing["reset_upload_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            if use_scratch:
                self._execute_reset_scratch_forward(plan.qpos[rows], plan.qvel[rows])
            else:
                self._execute_device_forward()
            # Forward may update warmstart; retain unselected channels exactly
            # and the explicitly cleared selected warmstart for the next step.
            self._upload(self._device_data.qacc_warmstart, plan.channels["qacc_warmstart"])
            self._synchronize()
            timing["reset_forward_ms"] = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            if use_scratch:
                self._qpos_cache[:] = plan.qpos
                self._qvel_cache[:] = plan.qvel
                self._refresh_reset_scratch_cache(rows)
            else:
                self._refresh_host_cache()
            self._sensor_cache[complement] = sensors[complement]
            if plan.entity_names is not None:
                # Authoritative authored force/contact snapshots on untouched
                # entities must not advance merely because another root moved.
                prefixes = tuple(name + "/" for name in plan.entity_names)
                for name, (start, width) in self._sensor_slots.items():
                    if not name.startswith(prefixes):
                        self._sensor_cache[rows, start : start + width] = sensors[
                            rows, start : start + width
                        ]
            self._ctrl_staging[:] = plan.channels["ctrl"]
            self._mocap_pos[:] = plan.mocap_pos
            self._mocap_quat[:] = plan.mocap_quat
            self._time_cache[:] = plan.time
            self._xfrc_staging[:] = plan.staged_wrenches
            self._xfrc_pending = bool(np.any(plan.staged_wrenches))
            self._tracked_body_state_dirty = dirty
            timing["host_cache_refresh_ms"] = (time.perf_counter() - start) * 1000.0
        except BaseException:
            self._entity_faulted = True
            raise
        return timing

    def reset_entities(self, request: SceneResetRequest) -> None:
        layout = self.get_scene_layout()
        prepared = prepare_scene_reset(
            layout, request, self._qpos_cache, self._qvel_cache, self._entity_roots()
        )
        bound = prepared.binding
        rows = prepared.env_ids
        qpos, qvel = self._qpos_cache.copy(), self._qvel_cache.copy()
        qcols, vcols = np.flatnonzero(prepared.qpos_mask), np.flatnonzero(prepared.qvel_mask)
        qpos[np.ix_(rows, qcols)] = prepared.qpos[:, qcols]
        qvel[np.ix_(rows, vcols)] = prepared.qvel[:, vcols]
        mpos, mquat = self._mocap_pos.copy(), self._mocap_quat.copy()
        for index, entity in enumerate(layout.entities):
            if entity.root_mode == "kinematic" and prepared.root_mask[index, 0]:
                mocap = self._entity_mocap_ids[index]
                mpos[rows, mocap] = prepared.roots[:, index, :3]
                mquat[rows, mocap] = prepared.roots[:, index, 3:7]
        impact = self._entity_reset_impacts.select(bound)
        bodies, dofs, controls, act = (
            impact.bodies, impact.dofs, impact.controls, impact.activations)
        channels = self._entity_persistent_channels()
        channels["ctrl"][row_columns(rows, controls)] = 0
        channels["act"][row_columns(rows, act)] = 0
        if request.restore_default_controls:
            channels["ctrl"][row_columns(rows, sorted(controls))] = self._entity_defaults["ctrl"][
                row_columns(rows, sorted(controls))
            ]
            channels["act"][row_columns(rows, act)] = self._entity_defaults["act"][
                row_columns(rows, act)
            ]
        for name in ("qfrc_applied", "qacc_warmstart"):
            channels[name][row_columns(rows, sorted(dofs))] = 0
        channels["xfrc_applied"][row_columns(rows, sorted(bodies))] = 0
        staging = self._xfrc_staging.copy()
        staging[row_columns(rows, sorted(bodies))] = 0
        self._commit_state(
            StateCommitPlan(
                rows,
                qpos,
                qvel,
                mpos,
                mquat,
                channels,
                staging,
                self._time_cache.copy(),
                entity_names=prepared.entity_names,
            )
        )

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        self._require_entity_healthy()
        if self._entity_layout is None:
            super().reset(env_ids)
            return
        rows = (
            np.arange(self._num_envs, dtype=np.int32)
            if env_ids is None
            else self._validate_rows(env_ids)
        )
        channels = self._entity_persistent_channels()
        qpos, qvel = self._qpos_cache.copy(), self._qvel_cache.copy()
        mpos, mquat = self._mocap_pos.copy(), self._mocap_quat.copy()
        for name, values in (
            ("qpos", qpos),
            ("qvel", qvel),
            ("mocap_pos", mpos),
            ("mocap_quat", mquat),
        ):
            values[rows] = self._entity_defaults[name][rows]
        for name, values in channels.items():
            values[rows] = self._entity_defaults[name][rows] if name in ("ctrl", "act") else 0
        time_values = self._time_cache.copy()
        time_values[rows] = self._entity_defaults["time"][rows]
        staging = self._xfrc_staging.copy()
        staging[rows] = 0
        self._commit_state(
            StateCommitPlan(
                rows, qpos, qvel, mpos, mquat, channels, staging, time_values, reset_world=True
            )
        )

    def cleanup_scene_assets(self) -> None:
        super().cleanup_scene_assets()
        if not getattr(self, "_entity_constructing", False):
            composed = getattr(self, "_composed_scene", None)
            if composed is not None:
                composed.close()
                self._composed_scene = None

    def close(self) -> None:
        self._entity_closed = True
        self.cleanup_scene_assets()

    # ------------------------------------------------------------------ #
    # Cold-path model binding                                             #
    # ------------------------------------------------------------------ #

    def _capture_import_report(self, report_requested: dict[str, Any]) -> None:
        """Read device adoption once; CPU source tables never stand in for readback."""
        model = self._device_model
        groups: dict[int | None, list[int]] = {}
        for env in range(self._num_envs):
            group_key = (
                None
                if self._fixed_variant_plan is None
                else int(self._fixed_variant_plan.assignment[env])
            )
            groups.setdefault(group_key, []).append(env)
        shared_readback: dict[str, Any] = {}
        representatives = tuple(env_ids[0] for env_ids in groups.values())
        representative_offsets = {env: index for index, env in enumerate(representatives)}
        representative_indices = None

        def readback(name: str, array: Any, env: int) -> Any:
            nonlocal representative_indices
            # A report already uses one representative per fixed-variant group.
            # Transfer only those rows instead of copying discarded worlds to CPU.
            if (
                array.shape[0] == self._num_envs
                and len(groups) < self._num_envs
                and all(array.shape)
            ):
                if len(groups) == 1:
                    return array[env : env + 1].numpy()
                if representative_indices is None:
                    representative_indices = self._warp.array(
                        representatives, dtype=self._warp.int32, device=array.device
                    )
                if name not in shared_readback:
                    shared_readback[name] = (
                        self._warp.indexedarray(array, indices=representative_indices)
                        .contiguous()
                        .numpy()
                    )
                offset = representative_offsets[env]
                return shared_readback[name][offset : offset + 1]
            if name not in shared_readback:
                shared_readback[name] = array.numpy()
            return shared_readback[name]

        effective: dict[str, Any] = {}
        option_arrays: dict[str, Any] = {}
        for name in ("dt", "gravity", "solver", "integrator"):
            attr = "timestep" if name == "dt" else name
            value = getattr(model.opt, attr, None)
            if value is not None:
                if hasattr(value, "numpy"):
                    option_arrays[name] = value
                    continue
                elif name in {"solver", "integrator"}:
                    enum = (
                        self._mujoco.mjtSolver if name == "solver" else self._mujoco.mjtIntegrator
                    )
                    value = str(enum(int(value)).name)
                elif name == "dt":
                    value = float(value)
                else:
                    value = list(value)
                effective[name] = value
        # Device fields can be expanded per world. Preserve that axis and
        # explicit env scope instead of claiming canonical source values apply.
        reports: list[ConfigurationField] = []
        device_fields = {
            name: getattr(model, name)
            for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat")
            if getattr(model, name, None) is not None
        }
        shared_fields = {
            name: getattr(model, name).numpy().tolist()
            for name in (
                "actuator_trntype",
                "actuator_trnid",
                "geom_contype",
                "geom_conaffinity",
                "exclude_signature",
                "pair_geom1",
                "pair_geom2",
                "sensor_type",
                "sensor_objtype",
                "sensor_objid",
                "sensor_dim",
            )
        }
        actuator_fields = {
            name: getattr(model, name)
            for name in ("actuator_gear", "actuator_gainprm", "actuator_biasprm")
        }
        for env_ids in groups.values():
            env = env_ids[0]
            requested = dict(report_requested)
            if self._fixed_variant_realization is not None and self._fixed_variant_plan is not None:
                variant = int(self._fixed_variant_plan.assignment[env])
                requested = dict(self._fixed_variant_realization.report_requested[variant])
            else:
                variant = None
            adopted = dict(effective)
            adopted.update(
                {name: readback(name, value, env).tolist() for name, value in option_arrays.items()}
            )
            adopted["actuator_mapping"] = {
                "names": requested["actuator_mapping"]["names"],
                "trntype": shared_fields["actuator_trntype"],
                "trnid": shared_fields["actuator_trnid"],
            }
            for attr, array in actuator_fields.items():
                values = readback(attr, array, env)
                if values.ndim == 3:
                    values = values[env if values.shape[0] == self._num_envs else 0]
                adopted["actuator_mapping"][attr.removeprefix("actuator_")] = values.tolist()
            adopted["collision_filter"] = {
                "geom_names": report_requested["collision_filter"]["geom_names"],
                "contype": shared_fields["geom_contype"],
                "conaffinity": shared_fields["geom_conaffinity"],
                "exclude_signature": shared_fields["exclude_signature"],
                "pair_geom1": shared_fields["pair_geom1"],
                "pair_geom2": shared_fields["pair_geom2"],
                "disableflags": int(model.opt.disableflags),
            }
            adopted["sensors"] = {
                "names": requested["sensors"]["names"],
                **{
                    name.removeprefix("sensor_"): shared_fields[name]
                    for name in ("sensor_type", "sensor_objtype", "sensor_objid", "sensor_dim")
                },
            }

            for name in ("dt", "gravity"):
                value = adopted.get(name)
                if isinstance(value, list) and (name == "dt" or isinstance(value[0], list)):
                    adopted[name] = value[env if len(value) == self._num_envs else 0]
            for name in ("body_mass", "body_inertia"):
                if name not in device_fields:
                    continue
                values = readback(name, device_fields[name], env)
                expected_ndim = 1 if name == "body_mass" else 2
                if values.ndim > expected_ndim:
                    values = values[env if values.shape[0] == self._num_envs else 0]
                adopted[name] = {"names": requested[name]["names"], "values": values.tolist()}
                if name == "body_inertia":
                    for key, attr in (("ipos", "body_ipos"), ("iquat_wxyz", "body_iquat")):
                        extra = readback(attr, device_fields[attr], env)
                        if extra.ndim == 3:
                            extra = extra[env if extra.shape[0] == self._num_envs else 0]
                        adopted[name][key] = extra.tolist()
            reports.extend(
                compare_configuration(
                    "mjwarp",
                    requested,
                    adopted,
                    source="Compiled composed MuJoCo input before Warp upload",
                    effective_source="mujoco_warp device Model cold-path readback",
                    scope=ConfigurationScope(
                        env_ids=tuple(env_ids), variant=None if variant is None else str(variant)
                    ),
                    lifecycle="materialization",
                ).fields
            )
        self._import_report = ImportReport("mjwarp", tuple(reports), lifecycle="materialization")

    @staticmethod
    def _require_capacity(value: int | None, *, name: str, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"mjwarp {name} must be a positive integer, got {value!r}")
        return value

    def _root_state_dims(self) -> tuple[int, int]:
        if int(self._cpu_model.njnt) == 0:
            return 0, 0
        free_joint = int(self._mujoco.mjtJoint.mjJNT_FREE)
        if int(self._cpu_model.jnt_type[0]) == free_joint:
            return 7, 6
        return 0, 0

    def _bind_names(self, object_type: Any, count: int) -> dict[str, int]:
        names: dict[str, int] = {}
        for object_id in range(count):
            name = self._mujoco.mj_id2name(self._cpu_model, object_type, object_id)
            if name is not None:
                names[str(name)] = object_id
        return names

    def _bind_sensor_slots(self) -> dict[str, tuple[int, int]]:
        slots: dict[str, tuple[int, int]] = {}
        sensor_type = self._mujoco.mjtObj.mjOBJ_SENSOR
        for sensor_id in range(int(self._cpu_model.nsensor)):
            name = self._mujoco.mj_id2name(self._cpu_model, sensor_type, sensor_id)
            if name is None:
                continue
            slots[str(name)] = (
                int(self._cpu_model.sensor_adr[sensor_id]),
                int(self._cpu_model.sensor_dim[sensor_id]),
            )
        return slots

    def _bind_keyframes(self) -> dict[str, np.ndarray]:
        keyframes: dict[str, np.ndarray] = {}
        key_type = self._mujoco.mjtObj.mjOBJ_KEY
        for key_id in range(int(self._cpu_model.nkey)):
            name = self._mujoco.mj_id2name(self._cpu_model, key_type, key_id)
            if name is not None:
                keyframes[str(name)] = np.asarray(
                    self._cpu_model.key_qpos[key_id],
                    dtype=np.float32,
                ).copy()
        return keyframes

    def _bind_joint_range(self) -> np.ndarray | None:
        free_joint = int(self._mujoco.mjtJoint.mjJNT_FREE)
        mask = np.asarray(self._cpu_model.jnt_type, dtype=np.int32) != free_joint
        joint_range = np.asarray(self._cpu_model.jnt_range, dtype=np.float32)[mask]
        return None if joint_range.size == 0 else joint_range.copy()

    def _resolve_push_body_id(self) -> int | None:
        """Resolve the interval-push target body on the cold path."""
        body_name = self._push_body_name if self._push_body_name is not None else self._base_name
        if body_name is None:
            return None
        try:
            return self._body_ids[body_name]
        except KeyError as exc:
            raise ValueError(f"Push body {body_name!r} not found in mjwarp model") from exc

    def _resolve_interval_root_velocity_qvel_ids(self) -> tuple[int, int, int] | None:
        """Bind the configured free root's world-linear qvel columns on the cold path."""
        if self._base_name is None or self._base_body_id is None:
            return None
        try:
            layout = self.get_root_state_layout(self._base_name)
        except (NotImplementedError, ValueError):
            return None
        linear_ids = layout.qvel_indices[:3]
        if linear_ids != tuple(range(linear_ids[0], linear_ids[0] + 3)):
            return None
        return int(linear_ids[0]), int(linear_ids[1]), int(linear_ids[2])

    def _bind_dr_host_mirrors(self) -> None:
        """Allocate per-world host mirrors of the DR-writable model fields.

        Reset randomization stages rows into these mirrors and uploads them
        with in-place ``assign`` into the expanded device arrays, so the
        mirrors are also the source of truth for the rows a reset did not
        touch.  Immutable model defaults are kept separately for the delta
        payload terms (``base_mass_delta`` / ``base_com_offset``).
        """
        cpu = self._cpu_model
        num_envs = self._num_envs
        self._dr_body_mass = np.broadcast_to(
            np.asarray(cpu.body_mass, dtype=np.float32), (num_envs, self._nbody)
        ).copy()
        self._dr_gravity = np.broadcast_to(
            np.asarray(cpu.opt.gravity, dtype=np.float32), (num_envs, 3)
        ).copy()
        self._dr_body_ipos = np.broadcast_to(
            np.asarray(cpu.body_ipos, dtype=np.float32), (num_envs, self._nbody, 3)
        ).copy()
        self._dr_body_iquat = np.broadcast_to(
            np.asarray(cpu.body_iquat, dtype=np.float32), (num_envs, self._nbody, 4)
        ).copy()
        self._dr_body_inertia = np.broadcast_to(
            np.asarray(cpu.body_inertia, dtype=np.float32), (num_envs, self._nbody, 3)
        ).copy()
        self._dr_dof_armature = np.broadcast_to(
            np.asarray(cpu.dof_armature, dtype=np.float32), (num_envs, self._nv)
        ).copy()
        self._dr_geom_friction = np.broadcast_to(
            np.asarray(cpu.geom_friction, dtype=np.float32), (num_envs, int(cpu.ngeom), 3)
        ).copy()
        self._dr_actuator_gainprm = np.broadcast_to(
            np.asarray(cpu.actuator_gainprm, dtype=np.float32), (num_envs, self._nu, 10)
        ).copy()
        self._dr_actuator_biasprm = np.broadcast_to(
            np.asarray(cpu.actuator_biasprm, dtype=np.float32), (num_envs, self._nu, 10)
        ).copy()
        for name in (
            "geom_size",
            "geom_rbound",
            "geom_aabb",
            "geom_solref",
            "geom_solimp",
            "dof_damping",
            "dof_frictionloss",
        ):
            default = np.asarray(getattr(cpu, name), dtype=np.float32)
            if name == "geom_aabb":
                default = default.reshape(int(cpu.ngeom), 2, 3)
            setattr(
                self, f"_dr_{name}", np.broadcast_to(default, (num_envs, *default.shape)).copy()
            )
        plan = self._fixed_variant_plan
        realization = self._fixed_variant_realization
        # Keep one immutable row for shared fields and reuse the compiler's
        # K variant rows. Only mutable mirrors need N environment rows.
        variant_count = len(plan.variants) if realization is not None and plan is not None else 1
        self._reset_default_assignment = (
            plan.assignment
            if realization is not None and plan is not None
            else np.zeros(num_envs, dtype=np.intp)
        )
        self._reset_field_defaults: dict[str, np.ndarray] = {}
        for name in (
            "gravity",
            "body_mass",
            "body_ipos",
            "body_iquat",
            "body_inertia",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "geom_friction",
            "geom_size",
            "geom_solref",
            "geom_solimp",
            "actuator_gainprm",
            "actuator_biasprm",
        ):
            if realization is not None and name in realization.fields:
                self._reset_field_defaults[name] = realization.fields[name]
            else:
                default = getattr(self, f"_dr_{name}")[0].copy()
                self._reset_field_defaults[name] = np.broadcast_to(
                    default, (variant_count, *default.shape)
                )
        if self._fixed_variant_realization is not None and plan is not None:
            assignment = plan.assignment
            realization = self._fixed_variant_realization
            for name in (
                "geom_size",
                "geom_rbound",
                "geom_aabb",
                "body_mass",
                "body_ipos",
                "body_iquat",
                "body_inertia",
            ):
                if name in realization.fields:
                    mirror = getattr(self, f"_dr_{name}")
                    mirror[...] = realization.fields[name][assignment]

    def _bind_tracked_body_state(self) -> None:
        """Bind zero-copy tracked-body views into the per-step sensor cache.

        Sensor columns follow the ``tracked_body_names`` insertion order from
        the cold-path injection; body ids are rebuilt from the compiled model
        because MjSpec compilation can reorder bodies (same reasoning as the
        MuJoCo backend).
        """
        names = self._tracked_body_names
        if not names:
            raise ValueError(
                "mjwarp add_body_sensors requires at least one named body in the model"
            )
        body_type = self._mujoco.mjtObj.mjOBJ_BODY
        tracked_ids = [self._mujoco.mj_name2id(self._cpu_model, body_type, name) for name in names]
        missing = [name for name, body_id in zip(names, tracked_ids, strict=True) if body_id < 0]
        if missing:
            raise ValueError(
                "Injected mjwarp body tracking sensors reference bodies missing from "
                f"the compiled model: {missing}"
            )
        mapping = np.full(self._nbody, -1, dtype=np.intp)
        for index, body_id in enumerate(tracked_ids):
            mapping[body_id] = index
        self._body_id_to_tracked_idx = mapping
        self._tracked_pos_w_all = self._tracked_sensor_view("track_pos_w", 3)
        self._tracked_quat_w_all = self._tracked_sensor_view("track_quat_w", 4)
        self._tracked_linvel_w_all = self._tracked_sensor_view("track_linvel_w", 3)
        self._tracked_angvel_w_all = self._tracked_sensor_view("track_angvel_w", 3)
        slots = sorted(self._tracked_sensor_slots.values())
        first = slots[0][0]
        stop = first
        for start, width in slots:
            if start != stop:
                raise ValueError("Injected mjwarp tracking sensor blocks must be contiguous")
            stop += width
        self._tracked_refresh_source = self._device_data.sensordata[:, first:stop]
        # Cross-device copies of strided Warp arrays allocate temporary device
        # storage. Pack into one stable allocation before the single D2H copy.
        self._tracked_refresh_device = self._warp.empty(
            self._tracked_refresh_source.shape,
            dtype=self._tracked_refresh_source.dtype,
            device=self._tracked_refresh_source.device,
        )
        self._tracked_refresh_storage, self._tracked_refresh_cache = (
            self._allocate_pinned_host_cache(self._tracked_refresh_device)
        )
        self._tracked_refresh_public = self._sensor_cache[:, first:stop]

    def _tracked_sensor_view(self, prefix: str, dim: int) -> np.ndarray:
        count = len(self._tracked_body_names)
        addresses = []
        for name in self._tracked_body_names:
            sensor_name = f"{prefix}_{name}"
            try:
                address, sensor_dim = self._sensor_slots[sensor_name]
            except KeyError as exc:
                raise ValueError(
                    f"Injected mjwarp tracking sensor {sensor_name!r} is missing from the "
                    "compiled model"
                ) from exc
            if sensor_dim != dim:
                raise ValueError(
                    f"Injected mjwarp tracking sensor {sensor_name!r} has dim {sensor_dim}; "
                    f"expected {dim}"
                )
            addresses.append(address)
        first = addresses[0]
        if addresses != [first + index * dim for index in range(count)]:
            raise ValueError(
                f"Injected mjwarp tracking sensors {prefix}_* are not one contiguous "
                "sensor block in tracked-body order"
            )
        self._tracked_sensor_slots[prefix] = (first, count * dim)
        return self._sensor_cache[:, first : first + count * dim].reshape(
            self._num_envs, count, dim
        )

    def _mapped_tracked_ids(self, operation: str, body_ids: np.ndarray) -> np.ndarray:
        mapping = self._body_id_to_tracked_idx
        if mapping is None:
            self._unsupported_body_kinematics(operation)
        mapped = mapping[np.asarray(body_ids, dtype=np.intp)]
        if np.any(mapped < 0):
            raise ValueError(
                f"mjwarp {operation} received body ids without injected tracking sensors: "
                f"{np.asarray(body_ids)[mapped < 0].tolist()}"
            )
        return mapped

    # ------------------------------------------------------------------ #
    # Explicit host-cache barriers                                        #
    # ------------------------------------------------------------------ #

    def _allocate_pinned_host_cache(self, device_array: Any) -> tuple[Any, np.ndarray]:
        """Allocate stable CPU storage for one fixed-shape device-state cache."""
        storage = self._warp.empty(
            device_array.shape,
            dtype=device_array.dtype,
            device="cpu",
            pinned=True,
        )
        return storage, storage.numpy()

    def _refresh_host_cache(self) -> None:
        """Copy all legacy-visible device state at one explicit lifecycle barrier."""
        self._download(self._device_data.qpos, self._qpos_cache_storage)
        self._download(self._device_data.qvel, self._qvel_cache_storage)
        self._download(self._device_data.sensordata, self._sensor_cache_storage)
        self._synchronize()

    def _upload(self, device_array: Any, host_array: np.ndarray) -> None:
        device_array.assign(host_array)

    def _download(self, device_array: Any, host_array: Any) -> None:
        self._warp.copy(host_array, device_array)

    def _synchronize(self) -> None:
        self._warp.synchronize_device()

    def _refresh_tracked_body_state_device(self) -> None:
        """Refresh only tracked body sensors for the final generalized state.

        ``step`` leaves MuJoCo phase data at the last substep boundary.  These
        kinematics and frame-sensor kernels use the live per-world model and
        final qpos/qvel without re-running constraint solving.  Only the four
        tracked frame-sensor blocks are copied back into the public host cache;
        unrelated authored sensors retain their completed-substep values.
        """
        if not self._tracked_body_names:
            return
        self._mujoco_warp.kinematics(self._device_model, self._device_data)
        self._mujoco_warp.com_pos(self._device_model, self._device_data)
        self._mujoco_warp.com_vel(self._device_model, self._device_data)
        self._mujoco_warp.sensor_pos(self._device_model, self._device_data)
        self._mujoco_warp.sensor_vel(self._device_model, self._device_data)
        self._warp.copy(self._tracked_refresh_device, self._tracked_refresh_source)
        self._download(self._tracked_refresh_device, self._tracked_refresh_storage)
        self._synchronize()
        np.copyto(self._tracked_refresh_public, self._tracked_refresh_cache)
        self._tracked_body_state_dirty = False

    def _disable_cuda_graphs(self, reason: str) -> None:
        """Atomically select the eager path and release any captured graphs."""
        self._cuda_graph_enabled = False
        self._step_graph = None
        self._forward_graph = None
        self._reset_graph = None
        self._reset_scratch_reset_graph = None
        self._reset_scratch_forward_graph = None
        self._cuda_graph_disable_reason: str | None = reason

    def _prepare_reset_scratch(self) -> None:
        """Materialize and warm the bounded reset-forward data on the cold path."""
        if self._reset_scratch_capacity == 0 or self._reset_scratch_data is not None:
            return

        capacity = self._reset_scratch_capacity
        data = self._mujoco_warp.make_data(
            self._cpu_model,
            nworld=capacity,
            nconmax=self._nconmax,
            njmax=self._njmax,
        )
        qpos = np.broadcast_to(
            np.asarray(self._cpu_model.qpos0, dtype=np.float32),
            (capacity, self._nq),
        ).copy()
        qvel = np.zeros((capacity, self._nv), dtype=np.float32)
        reset_mask = self._warp.ones(capacity, dtype=bool)
        sensor_storage, sensor_cache = self._allocate_pinned_host_cache(data.sensordata)

        self._reset_scratch_data = data
        self._reset_scratch_mask_device = reset_mask
        self._reset_scratch_qpos_staging = qpos
        self._reset_scratch_qvel_staging = qvel
        self._reset_scratch_sensor_storage = sensor_storage
        self._reset_scratch_sensor_cache = sensor_cache

        # Warm dynamically specialized reset/forward kernels before capture;
        # compiling or allocating from inside a CUDA capture is unsupported.
        self._mujoco_warp.reset_data(self._device_model, data, reset=reset_mask)
        self._upload(data.qpos, qpos)
        self._upload(data.qvel, qvel)
        self._mujoco_warp.forward(self._device_model, data)
        self._synchronize()

    def _initialize_cuda_graphs(self, device: Any) -> None:
        """Capture fixed-address device operations or retain the eager fallback.

        Current uploads mutate existing Warp arrays with ``assign``. Any future
        owner-layer operation that replaces a model or data array must call this
        method afterward so captured pointers cannot become stale.
        """
        self._disable_cuda_graphs("CUDA graph capture has not been initialized")
        eligible, reason = _cuda_graph_eligibility(self._warp, device)
        if not eligible:
            assert reason is not None
            self._cuda_graph_disable_reason = reason
            warnings.warn(
                f"mjwarp CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            self._prepare_reset_scratch()
            # Assign only after all captures succeed. This keeps step/reset on
            # one execution mode if any MJWarp operation is not capturable.
            with _suspend_gc(), self._warp.ScopedDevice(device):
                with self._warp.ScopedCapture() as step_capture:
                    self._mujoco_warp.step(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as forward_capture:
                    self._mujoco_warp.forward(self._device_model, self._device_data)
                with self._warp.ScopedCapture() as reset_capture:
                    self._mujoco_warp.reset_data(
                        self._device_model,
                        self._device_data,
                        reset=self._reset_mask_device,
                    )
                reset_scratch_reset_capture = None
                reset_scratch_forward_capture = None
                if self._reset_scratch_data is not None:
                    assert self._reset_scratch_mask_device is not None
                    with self._warp.ScopedCapture() as reset_scratch_reset_capture:
                        self._mujoco_warp.reset_data(
                            self._device_model,
                            self._reset_scratch_data,
                            reset=self._reset_scratch_mask_device,
                        )
                    with self._warp.ScopedCapture() as reset_scratch_forward_capture:
                        self._mujoco_warp.forward(
                            self._device_model,
                            self._reset_scratch_data,
                        )
            step_graph = step_capture.graph
            forward_graph = forward_capture.graph
            reset_graph = reset_capture.graph
            reset_scratch_reset_graph = (
                None if reset_scratch_reset_capture is None else reset_scratch_reset_capture.graph
            )
            reset_scratch_forward_graph = (
                None
                if reset_scratch_forward_capture is None
                else reset_scratch_forward_capture.graph
            )
        except Exception as exc:
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            self._disable_cuda_graphs(reason)
            warnings.warn(
                f"mjwarp CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._step_graph = step_graph
        self._forward_graph = forward_graph
        self._reset_graph = reset_graph
        self._reset_scratch_reset_graph = reset_scratch_reset_graph
        self._reset_scratch_forward_graph = reset_scratch_forward_graph
        self._cuda_graph_enabled = True
        self._cuda_graph_disable_reason = None

    def _execute_device_steps(self, nsteps: int) -> None:
        """Advance fixed-shape device state through graph replay or eager calls."""
        if self._cuda_graph_enabled:
            assert self._step_graph is not None
            for _ in range(nsteps):
                self._warp.capture_launch(self._step_graph)
            return
        for _ in range(nsteps):
            self._mujoco_warp.step(self._device_model, self._device_data)

    def _execute_device_reset(self) -> None:
        """Clear selected device rows before the host state upload."""
        if self._cuda_graph_enabled:
            assert self._reset_graph is not None
            self._warp.capture_launch(self._reset_graph)
            return
        self._mujoco_warp.reset_data(
            self._device_model,
            self._device_data,
            reset=self._reset_mask_device,
        )

    def _execute_device_forward(self) -> None:
        """Refresh kinematics after the host state upload."""
        if self._cuda_graph_enabled:
            assert self._forward_graph is not None
            self._warp.capture_launch(self._forward_graph)
            return
        self._mujoco_warp.forward(self._device_model, self._device_data)

    def _can_use_reset_scratch(self, num_rows: int) -> bool:
        return (
            self._cuda_graph_enabled
            and 0 < num_rows <= self._reset_scratch_capacity
            and self._reset_scratch_data is not None
            and self._reset_scratch_reset_graph is not None
            and self._reset_scratch_forward_graph is not None
        )

    def _execute_reset_scratch_forward(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        """Forward reset rows in bounded scratch storage without touching main rows."""
        data = self._reset_scratch_data
        qpos_staging = self._reset_scratch_qpos_staging
        qvel_staging = self._reset_scratch_qvel_staging
        assert data is not None
        assert qpos_staging is not None and qvel_staging is not None
        assert self._reset_scratch_reset_graph is not None
        assert self._reset_scratch_forward_graph is not None

        num_rows = len(qpos)
        np.copyto(qpos_staging[:num_rows], qpos)
        np.copyto(qvel_staging[:num_rows], qvel)
        self._warp.capture_launch(self._reset_scratch_reset_graph)
        self._upload(data.qpos, qpos_staging)
        self._upload(data.qvel, qvel_staging)
        self._warp.capture_launch(self._reset_scratch_forward_graph)

    def _refresh_reset_scratch_cache(self, row_ids: np.ndarray) -> None:
        """Publish scratch sensor rows while retaining complement host-cache rows."""
        data = self._reset_scratch_data
        storage = self._reset_scratch_sensor_storage
        cache = self._reset_scratch_sensor_cache
        assert data is not None and storage is not None and cache is not None
        self._download(data.sensordata, storage)
        self._synchronize()
        self._sensor_cache[row_ids] = cache[: len(row_ids)]

    def _validate_rows(self, env_indices: np.ndarray) -> np.ndarray:
        raw = np.asarray(env_indices)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("env_indices must contain integer row IDs")
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        return rows.copy()

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata                             #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        """Return the backend-owned device model, never a MuJoCo backend model."""
        return self._device_model

    @property
    def num_actuators(self) -> int:
        return self._nu

    @property
    def num_dof_vel(self) -> int:
        return self._num_dof_vel

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._actuator_ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        supported_transmissions = {
            int(self._mujoco.mjtTrn.mjTRN_JOINT),
            int(self._mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        supported_joint_types = {
            int(self._mujoco.mjtJoint.mjJNT_HINGE),
            int(self._mujoco.mjtJoint.mjJNT_SLIDE),
        }
        names: list[str] = []
        for actuator_id, actuator_name in enumerate(self._actuator_names):
            transmission = int(self._cpu_model.actuator_trntype[actuator_id])
            joint_id = int(self._cpu_model.actuator_trnid[actuator_id, 0])
            if transmission not in supported_transmissions or joint_id < 0:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires a "
                    f"joint transmission; actuator '{actuator_name}' uses "
                    f"transmission type {transmission}"
                )
            if int(self._cpu_model.jnt_type[joint_id]) not in supported_joint_types:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires a "
                    f"single-DoF joint; actuator '{actuator_name}' targets joint id {joint_id}"
                )
            joint_name = self._mujoco.mj_id2name(
                self._cpu_model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name:
                raise NotImplementedError(
                    "backend 'mjwarp' capability 'actuator target joint' requires named "
                    f"joints; actuator '{actuator_name}' targets unnamed joint id {joint_id}"
                )
            names.append(str(joint_name))
        return tuple(names)

    def get_scene_model_file(self) -> str | None:
        return self.scene_model_file

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        try:
            return self._keyframe_qpos[name].copy()
        except KeyError as exc:
            available = ", ".join(sorted(self._keyframe_qpos))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc

    def get_default_qpos(self) -> np.ndarray:
        return np.asarray(self._cpu_model.qpos0, dtype=np.float32).copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return np.asarray(self._cpu_model.qpos0[self._root_qpos_dim :], dtype=np.float32).copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._nv,), dtype=np.float32)

    def get_state(self, fields: tuple[str, ...] | str | None = None) -> Mapping[str, np.ndarray]:
        """Return detached full generalized-state snapshots.

        The host caches use the complete MuJoCo qpos/qvel layouts.  Copying
        them directly keeps snapshots compatible with ``set_state`` and named
        state indices even when the first model joint is not a free root.
        """
        self._require_entity_healthy()
        requested = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else tuple(fields))
        )
        result: dict[str, np.ndarray] = {}
        if "qpos" in requested:
            result["qpos"] = self._qpos_cache.copy()
        if "qvel" in requested:
            result["qvel"] = self._qvel_cache.copy()
        if "ctrl" in requested:
            result["ctrl"] = self._ctrl_staging.copy()
        unknown = set(requested) - {"qpos", "qvel", "ctrl"}
        if unknown:
            raise KeyError(f"unknown {self.backend_type} state field(s): {sorted(unknown)}")
        return result

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        qpos, qvel = self._compiled_index.free_root_layout(root_body_name)
        return BackendRootStateLayout(qpos_indices=qpos, qvel_indices=qvel)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._compiled_index.body_id(str(name)))
            except ValueError as exc:
                raise ValueError(f"Body {name!r} not found in mjwarp model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_geom_id(self, name: str) -> int:
        try:
            return int(self._geom_ids[name])
        except KeyError as exc:
            raise ValueError(f"Geom {name!r} not found in mjwarp model") from exc

    def get_geom_size(self, name: str) -> np.ndarray:
        geom_id = self.get_geom_id(name)
        if self._fixed_variant_realization is not None:
            return np.asarray(self._dr_geom_size[:, geom_id], dtype=np.float32).copy()
        return np.asarray(self._cpu_model.geom_size[geom_id], dtype=np.float32).copy()

    def get_geom_sizes(self) -> np.ndarray:
        if self._fixed_variant_realization is not None:
            return self._dr_geom_size.copy()
        return np.asarray(self._cpu_model.geom_size, dtype=np.float32).copy()

    def get_geom_solref(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_solref, dtype=np.float32).copy()

    def get_geom_solimp(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_solimp, dtype=np.float32).copy()

    def get_dof_damping(self) -> np.ndarray:
        return np.asarray(self._cpu_model.dof_damping, dtype=np.float32).copy()

    def get_dof_frictionloss(self) -> np.ndarray:
        return np.asarray(self._cpu_model.dof_frictionloss, dtype=np.float32).copy()

    def bind_mocap_pose(self, body_name: str) -> BackendMocapPoseBinding:
        if not isinstance(body_name, str) or not body_name:
            raise TypeError("mjwarp mocap body_name must be a non-empty string")
        if body_name not in self._body_ids:
            raise KeyError(f"mjwarp mocap body {body_name!r} does not exist")
        mocap_id = int(self._cpu_model.body_mocapid[self._body_ids[body_name]])
        if mocap_id < 0:
            raise NotImplementedError(f"mjwarp body {body_name!r} is not a mocap body")
        return BackendMocapPoseBinding(
            backend_type=self.backend_type,
            body_name=body_name,
            num_envs=self.num_envs,
            default_pose=np.concatenate(
                (self._default_mocap_pos[mocap_id], self._default_mocap_quat[mocap_id])
            ),
            _reader=partial(self._read_mocap_pose, mocap_id),
            _writer=partial(self._write_mocap_pose, mocap_id),
        )

    def _read_mocap_pose(self, mocap_id: int) -> np.ndarray:
        self._require_entity_healthy()
        return np.concatenate((self._mocap_pos[:, mocap_id], self._mocap_quat[:, mocap_id]), axis=1)

    def _write_mocap_pose(self, mocap_id: int, rows: np.ndarray, poses: np.ndarray) -> None:
        self._require_entity_healthy()
        with np.errstate(over="ignore", invalid="ignore"):
            poses = np.asarray(poses, dtype=np.float32)
        if not np.isfinite(poses).all():
            raise ValueError("mjwarp mocap poses must be finite float32 values")
        self._mocap_pos[rows, mocap_id] = poses[:, :3]
        self._mocap_quat[rows, mocap_id] = poses[:, 3:]
        self._upload(self._device_data.mocap_pos, self._mocap_pos)
        self._upload(self._device_data.mocap_quat, self._mocap_quat)
        self._execute_device_forward()
        self._synchronize()
        self._refresh_host_cache()

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        root = int(root_body_id)
        if root < 0 or root >= self._nbody:
            raise ValueError(f"root_body_id must be in [0, {self._nbody}), got {root}")
        descendants = {root}
        changed = True
        parent_ids = np.asarray(self._cpu_model.body_parentid, dtype=np.int32)
        while changed:
            changed = False
            for body_id, parent_id in enumerate(parent_ids):
                if body_id not in descendants and int(parent_id) in descendants:
                    descendants.add(body_id)
                    changed = True
        return np.asarray(sorted(descendants), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        names = [""] * int(self._cpu_model.ngeom)
        for name, geom_id in self._geom_ids.items():
            names[geom_id] = name
        return tuple(names)

    def get_geom_body_ids(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_bodyid, dtype=np.int32).copy()

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._cpu_model.geom_contype, dtype=np.int32).copy(),
            np.asarray(self._cpu_model.geom_conaffinity, dtype=np.int32).copy(),
        )

    def get_geom_friction(self) -> np.ndarray:
        return np.asarray(self._cpu_model.geom_friction, dtype=np.float32).copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._cpu_model.opt.gravity, dtype=np.float32).copy()

    def get_body_mass(self) -> np.ndarray:
        if self._fixed_variant_realization is not None:
            return self._dr_body_mass.copy()
        return np.asarray(self._cpu_model.body_mass, dtype=np.float32).copy()

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if env_ids is None:
            # Always the canonical default table, in every mode; per-env
            # variant defaults are exposed via get_reset_term_default().
            return np.asarray(self._cpu_model.body_ipos, dtype=np.float32).copy()
        ids = self._validate_env_ids(env_ids)
        return self._dr_body_ipos[ids]

    def get_dof_armature(self) -> np.ndarray:
        return np.asarray(self._cpu_model.dof_armature, dtype=np.float32).copy()

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        if names is None:
            return None if self._joint_range is None else self._joint_range.copy()

        single_dof_types = {
            int(self._mujoco.mjtJoint.mjJNT_HINGE),
            int(self._mujoco.mjtJoint.mjJNT_SLIDE),
        }
        ranges: list[tuple[float, float]] = []
        for name in names:
            try:
                jid = self._joint_ids[str(name)]
            except KeyError as exc:
                raise ValueError(f"Joint '{name}' not found in mjwarp model") from exc
            joint_type = int(self._cpu_model.jnt_type[jid])
            if joint_type not in single_dof_types:
                raise ValueError(f"Joint '{name}' is not a hinge or slide joint")
            if not bool(self._cpu_model.jnt_limited[jid]):
                ranges.append((-np.inf, np.inf))
            else:
                ranges.append(
                    (
                        float(self._cpu_model.jnt_range[jid, 0]),
                        float(self._cpu_model.jnt_range[jid, 1]),
                    )
                )
        return np.asarray(ranges, dtype=np.float32).reshape(len(ranges), 2)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._site_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Site {name!r} not found in mjwarp model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joint qvel coordinates on the cold metadata path."""

        resolved: list[int] = []
        for name in names:
            try:
                joint_id = self._joint_ids[str(name)]
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in mjwarp model") from exc
            resolved.append(int(self._cpu_model.jnt_dofadr[joint_id]))
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named single-DoF qpos coordinates excluding the free root."""

        single_dof_types = {
            int(self._mujoco.mjtJoint.mjJNT_HINGE),
            int(self._mujoco.mjtJoint.mjJNT_SLIDE),
        }
        resolved: list[int] = []
        for name in names:
            try:
                joint_id = self._joint_ids[str(name)]
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in mjwarp model") from exc
            if int(self._cpu_model.jnt_type[joint_id]) not in single_dof_types:
                raise ValueError(f"Joint {name!r} is not a single-DoF joint")
            resolved.append(int(self._cpu_model.jnt_qposadr[joint_id]) - self._root_qpos_dim)
        return np.asarray(resolved, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joint qvel coordinates excluding the free root."""

        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to full reset qpos columns."""
        self.get_joint_dof_pos_indices(names)
        return np.asarray(self._compiled_index.joint_qpos_indices(names), dtype=np.int32)

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve named joints to full reset qvel columns."""
        self.get_joint_dof_vel_indices(names)
        return np.asarray(self._compiled_index.joint_qvel_indices(names), dtype=np.int32)

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose immutable model defaults; this does not advertise gain DR support."""
        kp = np.asarray(self._cpu_model.actuator_gainprm[:, 0], dtype=np.float32).copy()
        kd = np.asarray(-self._cpu_model.actuator_biasprm[:, 2], dtype=np.float32).copy()
        return kp, kd

    def _execute_host_step(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> dict[str, float]:
        """Execute the owner-layer host-cache barrier for one legacy step."""
        t0 = time.perf_counter()
        np.copyto(self._ctrl_staging, ctrl)
        self._upload(self._device_data.ctrl, self._ctrl_staging)
        if self._xfrc_pending:
            # Staged interval push/body forces apply for the whole upcoming
            # step (all substeps), matching the MuJoCo backend's pending
            # ``xfrc_applied`` semantics.
            self._upload(self._device_data.xfrc_applied, self._xfrc_staging)
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._execute_device_steps(nsteps)
        if self._xfrc_pending:
            self._xfrc_staging.fill(0.0)
            self._upload(self._device_data.xfrc_applied, self._xfrc_staging)
            self._xfrc_pending = False
        self._synchronize()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache += np.float32(nsteps * self._sim_dt)
        self._tracked_body_state_dirty = bool(self._tracked_body_names)
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "control_upload_ms": control_upload_ms,
            "physics_ms": physics_ms,
            "host_cache_refresh_ms": host_cache_ms,
        }

    def _execute_host_step_with_pre_step_control(
        self,
        ctrl: np.ndarray,
        nsteps: int,
    ) -> dict[str, float]:
        """Advance one step through the registered per-substep converter.

        Mirrors the MuJoCo backend's ``_step_with_pre_step_control`` substep
        boundary: before every substep the host qpos/qvel cache holds the
        substep-start state, the owner callback converts the policy control,
        and the result is uploaded as that substep's device ctrl.  A body-state
        getter lazily refreshes tracked frame sensors on the live device state
        because ``step`` leaves MuJoCo phase sensors one substep behind qpos/qvel.
        ``xfrc_applied`` is recomposed and uploaded absolutely before every
        substep as the sum of the staged interval wrench and the callback's
        dynamic wrench; both channels are cleared when the call finishes.
        """
        control_upload_ms = 0.0
        host_cache_ms = 0.0
        composed_xfrc = np.zeros_like(self._xfrc_staging)

        t0 = time.perf_counter()
        self._pre_step_control_active = True
        completed_steps = 0
        try:
            for substep in range(nsteps):
                if substep > 0:
                    t1 = time.perf_counter()
                    self._download(self._device_data.qpos, self._qpos_cache_storage)
                    self._download(self._device_data.qvel, self._qvel_cache_storage)
                    self._synchronize()
                    host_cache_ms += (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                output = self._convert_pre_step_control(ctrl)
                np.copyto(self._ctrl_staging, output.ctrl)
                self._upload(self._device_data.ctrl, self._ctrl_staging)
                # Absolute per-substep wrench write: fixed interval wrench +
                # this substep's dynamic callback wrench.  A substep whose
                # callback returns no wrench applies the fixed part alone, so
                # dynamic wrenches never leak across substeps.
                composed_xfrc[:] = self._xfrc_staging
                if output.force is not None or output.torque is not None:
                    for body_offset, body_id in enumerate(np.asarray(output.body_ids)):
                        if output.force is not None:
                            composed_xfrc[:, int(body_id), 0:3] += output.force[:, body_offset, :]
                        if output.torque is not None:
                            composed_xfrc[:, int(body_id), 3:6] += output.torque[:, body_offset, :]
                self._upload(self._device_data.xfrc_applied, composed_xfrc)
                control_upload_ms += (time.perf_counter() - t1) * 1000.0
                # Eager launch: a captured step graph cannot observe the
                # per-substep host ctrl/xfrc uploads between kernel boundaries.
                try:
                    self._mujoco_warp.step(self._device_model, self._device_data)
                except BaseException:
                    self._entity_faulted = True
                    raise
                completed_steps += 1
                self._tracked_body_state_dirty = bool(self._tracked_body_names)
        finally:
            self._pre_step_control_active = False
            # The loop wrote absolute wrenches every substep, so the device
            # channel must be returned to zero unconditionally (not only when a
            # staged interval wrench existed); otherwise the final dynamic
            # wrench would leak into the next step, including when a callback
            # interrupts the control cycle.
            self._xfrc_staging.fill(0.0)
            self._upload(self._device_data.xfrc_applied, self._xfrc_staging)
            self._xfrc_pending = False
            # Publish any completed substeps even when a callback interrupts
            # the cycle, so public state and time remain usable for recovery.
            t1 = time.perf_counter()
            self._refresh_host_cache()
            self._time_cache += np.float32(completed_steps * self._sim_dt)
            self._tracked_body_state_dirty = bool(self._tracked_body_names)
            final_cache_ms = (time.perf_counter() - t1) * 1000.0
            host_cache_ms += final_cache_ms
        physics_ms = (time.perf_counter() - t0) * 1000.0 - final_cache_ms
        return {
            "control_upload_ms": control_upload_ms,
            "physics_ms": physics_ms,
            "host_cache_refresh_ms": host_cache_ms,
        }

    def _sync_tracked_body_state(self) -> None:
        self._require_entity_healthy()
        """Refresh the tracked-body views on the first read of a substep."""
        if not self._tracked_body_state_dirty:
            return
        self._refresh_tracked_body_state_device()

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_entity_healthy()
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self._nu)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")
        if not np.isfinite(ctrl_array).all():
            raise ValueError("ctrl must contain finite values")
        try:
            if self._pre_step_control_fn is not None:
                timings = self._execute_host_step_with_pre_step_control(ctrl_array, int(nsteps))
            else:
                timings = self._execute_host_step(ctrl_array, int(nsteps))
        except BaseException:
            if self._pre_step_control_fn is None:
                self._entity_faulted = True
            raise
        return {"timing": timings}

    # All backends report the same set_state key set for column stability;
    # sub-keys that don't apply to the mjwarp host profile report 0.0.
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

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_entity_healthy()
        rows = self._validate_rows(env_indices)
        qpos_array = float_values("qpos", qpos, (rows.size, self._nq))
        qvel_array = float_values("qvel", qvel, (rows.size, self._nv))
        updates = self._prepare_reset_randomization(rows, randomization)
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

        outer_t0 = time.perf_counter()
        positions, velocities = self._qpos_cache.copy(), self._qvel_cache.copy()
        positions[rows], velocities[rows] = qpos_array, qvel_array
        mocap_pos, mocap_quat = self._mocap_pos.copy(), self._mocap_quat.copy()
        mocap_pos[rows], mocap_quat[rows] = self._default_mocap_pos, self._default_mocap_quat
        channels = self._entity_persistent_channels()
        for values in channels.values():
            values[rows] = 0
        staging = self._xfrc_staging.copy()
        staging[rows] = 0
        times = self._time_cache.copy()
        times[rows] = 0
        timings = self._commit_state(
            StateCommitPlan(
                rows,
                positions,
                velocities,
                mocap_pos,
                mocap_quat,
                channels,
                staging,
                times,
                reset_world=True,
                model_updates=updates,
                allow_scratch=True,
            )
        )
        timing["set_state_reset_rand_ms"] = timings["model_update_ms"]
        timing["set_state_reset_upload_ms"] = timings["reset_upload_ms"]
        timing["set_state_reset_forward_ms"] = timings["reset_forward_ms"]
        timing["set_state_host_cache_refresh_ms"] = timings["host_cache_refresh_ms"]
        outer_total_ms = (time.perf_counter() - outer_t0) * 1000.0
        measured_ms = (
            timing["set_state_reset_upload_ms"]
            + timing["set_state_reset_forward_ms"]
            + timing["set_state_host_cache_refresh_ms"]
        )
        timing["set_state_internal_gap_ms"] = outer_total_ms - measured_ms
        return {"timing": timing}

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return canonical or fixed-variant authoritative reset defaults."""

        _validate_reset_term(term)
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(f"MjwarpBackend does not support reset term '{term}'")
        per_env = self._fixed_variant_realization is not None
        if term == RESET_TERM_BASE_MASS:
            values = np.zeros((self._num_envs if per_env else 0), dtype=np.float32)
        elif term == RESET_TERM_BASE_COM:
            values = np.zeros(
                (self._num_envs, 3) if per_env else (3,),
                dtype=np.float32,
            )
        elif term in (RESET_TERM_KP, RESET_TERM_KD):
            values = (
                self._reset_field_defaults["actuator_gainprm"][:, :, 0]
                if term == RESET_TERM_KP
                else -self._reset_field_defaults["actuator_biasprm"][:, :, 2]
            )
            values = values[self._reset_default_assignment] if per_env else values[0]
        else:
            contract_name = {
                RESET_TERM_GRAVITY: "gravity",
                RESET_TERM_BODY_IPOS: "body_ipos",
                RESET_TERM_BODY_IQUAT: "body_iquat",
                RESET_TERM_BODY_INERTIA: "body_inertia",
                RESET_TERM_BODY_MASS: "body_mass",
                RESET_TERM_DOF_ARMATURE: "dof_armature",
                RESET_TERM_DOF_DAMPING: "dof_damping",
                RESET_TERM_DOF_FRICTIONLOSS: "dof_frictionloss",
                RESET_TERM_GEOM_FRICTION: "geom_friction",
                RESET_TERM_GEOM_SIZE: "geom_size",
                RESET_TERM_GEOM_SOLREF: "geom_solref",
                RESET_TERM_GEOM_SOLIMP: "geom_solimp",
            }[term]
            values = self._reset_field_defaults[contract_name]
            values = values[self._reset_default_assignment] if per_env else values[0]

        # Variant assignment indexing already produced a detached array.
        result = np.array(values, dtype=np.float32, copy=not per_env)
        result.setflags(write=False)
        return result

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise the per-world model mutation set validated by effect tests."""
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {
                    RESET_TERM_BASE_MASS,
                    RESET_TERM_BASE_COM,
                    RESET_TERM_GRAVITY,
                    RESET_TERM_BODY_IQUAT,
                    RESET_TERM_BODY_INERTIA,
                    RESET_TERM_BODY_IPOS,
                    RESET_TERM_BODY_MASS,
                    RESET_TERM_DOF_ARMATURE,
                    RESET_TERM_DOF_DAMPING,
                    RESET_TERM_DOF_FRICTIONLOSS,
                    RESET_TERM_GEOM_FRICTION,
                    RESET_TERM_GEOM_SIZE,
                    RESET_TERM_GEOM_SOLREF,
                    RESET_TERM_GEOM_SOLIMP,
                    RESET_TERM_KP,
                    RESET_TERM_KD,
                }
            ),
            supports_interval_push=self._push_body_id is not None,
            supports_interval_body_velocity_delta=(
                self._interval_root_velocity_qvel_ids is not None
            ),
            supports_interval_body_force=True,
            supports_interval_body_torque=True,
            supported_interval_terms=frozenset(
                {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
                | ({INTERVAL_TERM_PUSH} if self._push_body_id is not None else set())
                | (
                    {INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA}
                    if self._interval_root_velocity_qvel_ids is not None
                    else set()
                )
            ),
            supports_fixed_variants=True,
            supported_fixed_variant_layouts=frozenset(
                {FixedVariantLayout.SAME_LAYOUT, FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT}
            ),
            supports_per_env_playback=True,
        )

    # ------------------------------------------------------------------ #
    # Reset domain randomization: per-world model row writes              #
    # ------------------------------------------------------------------ #

    def _validate_extended_randomization(
        self, rows: np.ndarray, payload: ResetRandomizationPayload
    ) -> dict[str, np.ndarray]:
        """Validate every new field before mutating state or device buffers."""
        staged: dict[str, np.ndarray] = {}
        for name in ("geom_size", "geom_solref", "geom_solimp", "dof_damping", "dof_frictionloss"):
            values = getattr(payload, name)
            if values is None:
                continue
            if not isinstance(values, np.ndarray) or not np.issubdtype(values.dtype, np.floating):
                raise TypeError(f"mjwarp {name} must be a floating NumPy array")
            mirror = getattr(self, f"_dr_{name}")
            expected = (rows.size, *mirror.shape[1:])
            if values.shape != expected:
                raise ValueError(f"mjwarp {name} must have shape {expected}, got {values.shape}")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(values, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"mjwarp {name} must be finite float32 values")
            if name in ("dof_damping", "dof_frictionloss") and np.any(array < 0):
                raise ValueError(f"mjwarp {name} must be non-negative")
            if name == "geom_solref":
                valid = np.all(array > 0, axis=-1) | np.all(array <= 0, axis=-1)
                if not np.all(valid):
                    raise ValueError(
                        "mjwarp geom_solref requires two positive or two non-positive values"
                    )
            if name == "geom_solimp" and (
                np.any(array[..., :2] < 0)
                or np.any(array[..., :2] > 1)
                or np.any(array[..., 2] <= 0)
                or np.any(array[..., 3] <= 0)
                or np.any(array[..., 3] >= 1)
                or np.any(array[..., 4] < 1)
            ):
                raise ValueError(
                    "mjwarp geom_solimp requires impedance [0,1], width>0, midpoint (0,1), power>=1"
                )
            staged[name] = array
        if "geom_size" in staged:
            rbound, aabb = self._geom_bounds.compute(
                staged["geom_size"],
                self._dr_geom_size[rows],
                self._dr_geom_rbound[rows],
                self._dr_geom_aabb[rows],
            )
            staged["geom_rbound"] = rbound
            staged["geom_aabb"] = aabb
        return staged

    @staticmethod
    def _coerce_dr_field(
        name: str,
        values: np.ndarray,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        """Accept the flat or shaped per-row layout, matching the MuJoCo backend."""
        array = np.asarray(values)
        flat_tail = int(np.prod(shaped_tail)) if shaped_tail else 1
        shaped = (num_reset, *shaped_tail)
        if array.shape == shaped:
            return float_values(name, array, shaped)
        if array.shape == (num_reset, flat_tail):
            return float_values(name, array.reshape(shaped), shaped)
        raise ValueError(
            f"{name} must have shape {shaped} or {(num_reset, flat_tail)}, got {array.shape}"
        )

    def _require_dr_base_body(self, term: str) -> int:
        if self._base_body_id is None:
            raise ValueError(
                f"mjwarp reset randomization term {term!r} requires base_name to "
                "identify the base body"
            )
        return self._base_body_id

    def _prepare_reset_randomization(
        self,
        rows: np.ndarray,
        randomization: ResetRandomizationPayload | None,
    ) -> ModelUpdates:
        """Own and validate all model writes without touching host/device state."""
        if randomization is None:
            return ModelUpdates()
        if not isinstance(randomization, ResetRandomizationPayload):
            raise TypeError("randomization must be ResetRandomizationPayload or None")
        unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
            randomization.requested_terms()
        )
        if unsupported:
            raise NotImplementedError(f"mjwarp does not support reset terms: {sorted(unsupported)}")
        num_reset = rows.size
        staged: dict[str, np.ndarray] = {}
        for name, tail in (
            ("gravity", (3,)),
            ("body_mass", (self._nbody,)),
            ("body_ipos", (self._nbody, 3)),
            ("body_iquat", (self._nbody, 4)),
            ("body_inertia", (self._nbody, 3)),
            ("dof_armature", (self._nv,)),
            ("geom_friction", (int(self._cpu_model.ngeom), 3)),
        ):
            value = getattr(randomization, name)
            if value is not None:
                staged[name] = self._coerce_dr_field(name, value, num_reset, tail)
        if randomization.base_mass_delta is not None:
            base_id = self._require_dr_base_body("base_mass_delta")
            delta = float_values("base_mass_delta", randomization.base_mass_delta, (num_reset,))
            if "body_mass" not in staged:
                # Only the base component uses immutable defaults; preserve
                # previously randomized values of other bodies in selected rows.
                staged["body_mass"] = self._dr_body_mass[rows].copy()
                staged["body_mass"][:, base_id] = self._reset_field_defaults["body_mass"][
                    self._reset_default_assignment[rows], base_id
                ]
            with np.errstate(over="ignore", invalid="ignore"):
                staged["body_mass"][:, base_id] += delta
        if randomization.base_com_offset is not None:
            base_id = self._require_dr_base_body("base_com_offset")
            offset = float_values("base_com_offset", randomization.base_com_offset, (num_reset, 3))
            if "body_ipos" not in staged:
                staged["body_ipos"] = self._dr_body_ipos[rows].copy()
                staged["body_ipos"][:, base_id] = self._reset_field_defaults["body_ipos"][
                    self._reset_default_assignment[rows], base_id
                ]
            with np.errstate(over="ignore", invalid="ignore"):
                staged["body_ipos"][:, base_id] += offset
        staged.update(self._validate_extended_randomization(rows, randomization))
        for name, values in staged.items():
            if not np.isfinite(values).all():
                raise ValueError(f"computed {name} must be finite")
            if name in {
                "body_mass",
                "body_inertia",
                "dof_armature",
                "geom_friction",
                "geom_size",
                "dof_damping",
                "dof_frictionloss",
            } and np.any(values < 0):
                raise ValueError(f"computed {name} must be non-negative")
            if name == "body_iquat" and not np.allclose(
                np.linalg.norm(values, axis=-1), 1.0, rtol=0.0, atol=1e-5
            ):
                raise ValueError("body_iquat requires unit wxyz quaternions")
        fields = {}
        for name, values in staged.items():
            fields[name] = getattr(self, f"_dr_{name}").copy()
            fields[name][rows] = values
        refresh = (
            2
            if set(staged) & {"body_mass", "body_ipos", "body_iquat"}
            else (1 if set(staged) & {"body_inertia", "dof_armature"} else 0)
        )
        actuator_fields = {}
        if randomization.kp is not None:
            kp = self._coerce_dr_field("kp", randomization.kp, num_reset, (self._nu,))
            actuator_fields["actuator_gainprm"] = self._dr_actuator_gainprm.copy()
            actuator_fields["actuator_biasprm"] = self._dr_actuator_biasprm.copy()
            actuator_fields["actuator_gainprm"][rows, :, 0] = kp
            actuator_fields["actuator_biasprm"][rows, :, 1] = -kp
        if randomization.kd is not None:
            kd = self._coerce_dr_field("kd", randomization.kd, num_reset, (self._nu,))
            if "actuator_biasprm" not in actuator_fields:
                actuator_fields["actuator_biasprm"] = self._dr_actuator_biasprm.copy()
            actuator_fields["actuator_biasprm"][rows, :, 2] = -kd
        return ModelUpdates(fields, refresh, actuator_fields)

    # ------------------------------------------------------------------ #
    # Interval domain randomization                                       #
    # ------------------------------------------------------------------ #

    _interval_term_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        self._reject_wrench_write_inside_pre_step_control("apply_interval_randomization")
        # A non-empty plan starts from cleared external wrenches, matching the
        # MuJoCo backend: ops within one plan accumulate, while a later plan
        # replaces anything a previous plan staged before it was consumed.
        self._xfrc_staging.fill(0.0)
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        # Built lazily once.  The angular-velocity term intentionally has no
        # handler and fails closed in the base dispatch (previously it was
        # silently dropped).
        if self._interval_term_handler_cache is None:
            self._interval_term_handler_cache = {
                INTERVAL_TERM_PUSH: lambda op: self.push_robots(op.payload),
                INTERVAL_TERM_BODY_FORCE: lambda op: self.apply_body_force(
                    require_op_body_ids(op), op.payload
                ),
                INTERVAL_TERM_BODY_TORQUE: lambda op: self._apply_body_torque(
                    require_op_body_ids(op), op.payload
                ),
                INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA: (
                    lambda op: self._apply_body_linear_velocity_delta(
                        require_op_body_ids(op), op.payload
                    )
                ),
            }
        return self._interval_term_handler_cache

    def push_robots(self, force_range: Sequence[float] | np.ndarray) -> None:
        """Sample one world-frame push force per env and stage it for the next step."""
        self._reject_wrench_write_inside_pre_step_control("push_robots")
        if self._push_body_id is None:
            raise NotImplementedError(
                "mjwarp interval push requires base_name or push_body_name to identify "
                "a push target body"
            )
        limit = np.asarray(force_range, dtype=np.float32)
        if limit.shape != (3,) or not np.isfinite(limit).all():
            raise ValueError(f"push force_range must be a finite (3,) array, got {limit.shape}")
        self._xfrc_staging.fill(0.0)
        sampled = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3)).astype(np.float32)
        self._xfrc_staging[:, self._push_body_id, 0:3] = sampled * limit
        self._xfrc_pending = True

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Accumulate a world-frame wrench on the staged ``xfrc_applied`` rows.

        The force acts at each target body's center of mass and the torque is
        about that center, matching MuJoCo ``xfrc_applied`` semantics.  The
        staged wrench applies to every substep of the next ``step()`` call and
        is cleared after it; repeated calls accumulate until consumed.
        """
        self._reject_wrench_write_inside_pre_step_control("apply_body_force")
        body_ids_np = np.asarray(body_ids, dtype=np.intp).reshape(-1)
        if np.any(body_ids_np < 0) or np.any(body_ids_np >= self._nbody):
            raise ValueError(f"body_ids must be in [0, {self._nbody}), got {body_ids_np}")
        force_np = np.asarray(force, dtype=np.float32)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
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
            self._xfrc_staging[:, int(body_id), 0:3] += force_np[:, body_offset, :]
            if torque_np is not None:
                self._xfrc_staging[:, int(body_id), 3:6] += torque_np[:, body_offset, :]
        self._xfrc_pending = True

    def _apply_body_torque(self, body_ids: np.ndarray, torque: np.ndarray) -> None:
        """Accumulate a torque-only wrench through the shared staging buffer."""
        zero_force = np.zeros((self._num_envs, len(body_ids), 3), dtype=np.float32)
        self.apply_body_force(body_ids, zero_force, torque=torque)

    def _apply_body_linear_velocity_delta(
        self,
        body_ids: np.ndarray,
        velocity_delta: np.ndarray,
    ) -> None:
        """Apply a row-selective world-frame velocity kick to the configured free root."""
        qvel_ids = self._interval_root_velocity_qvel_ids
        if qvel_ids is None:
            raise NotImplementedError(
                "mjwarp interval body velocity perturbation requires base_name to identify "
                "a body with exactly one free joint"
            )

        raw_body_ids = np.asarray(body_ids)
        if (
            raw_body_ids.ndim != 1
            or not np.issubdtype(raw_body_ids.dtype, np.integer)
            or np.issubdtype(raw_body_ids.dtype, np.bool_)
        ):
            raise TypeError(
                "mjwarp interval body velocity perturbation body_ids must be a 1-D "
                f"integer array, got shape={raw_body_ids.shape}, dtype={raw_body_ids.dtype}"
            )
        resolved_body_ids = np.asarray(raw_body_ids, dtype=np.int32)
        expected_body_ids = np.asarray([self._base_body_id], dtype=np.int32)
        if not np.array_equal(resolved_body_ids, expected_body_ids):
            raise NotImplementedError(
                "mjwarp interval body velocity perturbation only supports the configured "
                f"free root body {self._base_name!r} (id={self._base_body_id}); "
                f"received body_ids={resolved_body_ids.tolist()}"
            )

        if not isinstance(velocity_delta, np.ndarray):
            raise TypeError(
                "mjwarp interval body velocity perturbation must be an np.ndarray, "
                f"got {type(velocity_delta).__name__}"
            )
        expected_shape = (self._num_envs, 1, 3)
        if velocity_delta.shape != expected_shape:
            raise ValueError(
                "mjwarp interval body velocity perturbation has shape "
                f"{velocity_delta.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(velocity_delta.dtype, np.floating):
            raise TypeError(
                "mjwarp interval body velocity perturbation must have floating dtype, "
                f"got {velocity_delta.dtype}"
            )
        if not np.isfinite(velocity_delta).all():
            raise ValueError("mjwarp interval body velocity perturbation contains NaN or Inf")

        active_rows = np.flatnonzero(np.any(velocity_delta[:, 0, :] != 0.0, axis=1)).astype(
            np.intp,
            copy=False,
        )
        if active_rows.size == 0:
            return

        # Kick the host-cache rows and re-commit them through the existing
        # upload + forward barrier so sensors/kinematics stay coherent.
        qvel_columns = np.asarray(qvel_ids, dtype=np.intp)
        self._qvel_cache[active_rows[:, None], qvel_columns[None, :]] += velocity_delta[
            active_rows, 0, :
        ].astype(np.float32)
        self._upload(self._device_data.qvel, self._qvel_cache)
        self._execute_device_forward()
        self._synchronize()
        self._refresh_host_cache()

    def materialize(self) -> None:
        self._require_entity_healthy()
        """Resources are fully materialized during the constructor cold path."""

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_physics_state_playback=True,
            supports_debug_overlay=True,
            supports_interactive_debug_overlay=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none",
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if mode == "auto":
            raise NotImplementedError(
                "mjwarp playback does not support auto mode; select record or none explicitly."
            )
        if mode == "interactive":
            if play_steps is not None and (isinstance(play_steps, bool) or play_steps <= 0):
                raise ValueError(
                    "mjwarp interactive playback requires positive play_steps or None."
                )
            return BackendPlayRenderPlan(
                mode="interactive",
                headless=False,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        if isinstance(play_steps, bool) or play_steps is None or int(play_steps) <= 0:
            raise ValueError(
                "mjwarp record playback requires a positive finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("mjwarp record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode="record",
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Any,
        step: Any,
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Any = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: DebugOverlayGetter | None = None,
        on_frame: Any = None,
    ) -> str | None:
        del render_offset_mode
        camera = CameraCfg.from_kwargs(camera_kwargs)
        should_record = bool(record_video) if record_video is not None else output_video is not None
        should_run_headless = bool(headless) if headless is not None else should_record
        return run_mjwarp_playback(
            backend=self,
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=should_run_headless,
            record_video=should_record,
            snapshot_shape=(self._num_envs, 1 + self._nq + self._nv + 7 * self._nmocap),
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera,
            debug_overlay_getter=debug_overlay_getter,
            on_frame=on_frame,
        )

    def get_physics_state(self) -> np.ndarray:
        self._require_entity_healthy()
        # Layout: [time, qpos, qvel] plus, when the model has mocap bodies,
        # [mocap_pos(nmocap*3), mocap_quat(nmocap*4)] so offline playback can
        # replay mocap-driven geometry (e.g. a mocap palm) at its recorded
        # pose instead of the model defaults.
        nstate = 1 + self._nq + self._nv + 7 * self._nmocap
        state = np.empty((self._num_envs, nstate), dtype=np.float32)
        state[:, 0] = self._time_cache
        state[:, 1 : 1 + self._nq] = self._qpos_cache
        state[:, 1 + self._nq : 1 + self._nq + self._nv] = self._qvel_cache
        if self._nmocap:
            base = 1 + self._nq + self._nv
            state[:, base : base + 3 * self._nmocap] = self._mocap_pos.reshape(self._num_envs, -1)
            state[:, base + 3 * self._nmocap :] = self._mocap_quat.reshape(self._num_envs, -1)
        return state

    def get_playback_mocap_state(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return copied mocap pose arrays for detached visual playback."""
        self._require_entity_healthy()
        if env_index < 0 or env_index >= self._num_envs:
            raise IndexError("mjwarp playback environment index is out of range")
        return self._mocap_pos[env_index].copy(), self._mocap_quat[env_index].copy()

    def get_playback_model(self, env_index: int | None = None) -> str:
        self._require_entity_healthy()
        if self._fixed_variant_realization is not None:
            if env_index is None:
                raise ValueError("fixed-variant playback requires an explicit env_index")
            else:
                if isinstance(env_index, bool) or not isinstance(env_index, int):
                    raise TypeError("env_index must be an integer or None")
                if env_index < 0 or env_index >= self._num_envs:
                    raise IndexError(f"env_index must be in [0, {self._num_envs - 1}]")
                index = env_index
            assert self._fixed_variant_plan is not None
            variant_index = int(self._fixed_variant_plan.assignment[index])
            return self._fixed_variant_realization.playback_model_files[variant_index]
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        if not self._playback_model_validated:
            self.scene_visual_model_file = validate_mjwarp_visual_model(
                mujoco=self._mujoco,
                physics_model=self._cpu_model,
                model_file=self.scene_visual_model_file,
            )
            self._playback_model_validated = True
        return self.scene_visual_model_file

    # ------------------------------------------------------------------ #
    # Legacy getters: cache views only, never direct Warp transfers       #
    # ------------------------------------------------------------------ #

    def _require_free_root(self, operation: str) -> None:
        self._require_entity_healthy()
        if self._root_qpos_dim != 7 or self._root_qvel_dim != 6:
            raise NotImplementedError(
                f"{operation} requires a first free joint; mjwarp host_numpy profile is "
                "currently validated only for floating-base G1 layouts."
            )

    def get_base_pos(self) -> np.ndarray:
        self._require_free_root("get_base_pos")
        return self._qpos_cache[:, 0:3]

    def get_base_quat(self) -> np.ndarray:
        self._require_free_root("get_base_quat")
        return self._qpos_cache[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_free_root("get_base_lin_vel")
        return self._qvel_cache[:, 0:3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_free_root("get_base_ang_vel")
        return self._qvel_cache[:, 3:6]

    def get_dof_pos(self) -> np.ndarray:
        self._require_entity_healthy()
        return self._qpos_cache[:, self._root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        self._require_entity_healthy()
        return self._qvel_cache[:, self._root_qvel_dim :]

    def _unsupported_body_kinematics(self, operation: str) -> NoReturn:
        self._require_entity_healthy()
        raise NotImplementedError(
            f"mjwarp host_numpy profile does not expose {operation}; the G1 host adapter "
            "supports base, dof, and configured sensor cache reads, plus tracked body "
            "kinematics when constructed with body_state_required/add_body_sensors."
        )

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("world-frame body positions", body_ids)
        return self._tracked_pos_w_all[:, mapped, :]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("world-frame body orientations", body_ids)
        return self._tracked_quat_w_all[:, mapped, :]

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._sync_tracked_body_state()
        """Gather world-frame body pose for selected environments only."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body poses", body_ids)
        return self._tracked_pos_w_all[rows[:, None], mapped], self._tracked_quat_w_all[
            rows[:, None], mapped
        ]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("world-frame body linear velocities", body_ids)
        return self._tracked_linvel_w_all[:, mapped, :]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("world-frame body angular velocities", body_ids)
        return self._tracked_angvel_w_all[:, mapped, :]

    def get_body_lin_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        """Gather world-frame body linear velocity for selected rows."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body linear velocities", body_ids)
        return self._tracked_linvel_w_all[rows[:, None], mapped]

    def get_body_ang_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        """Gather world-frame body angular velocity for selected rows."""
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._mapped_tracked_ids("world-frame body angular velocities", body_ids)
        return self._tracked_angvel_w_all[rows[:, None], mapped]

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("world-frame body state", body_ids)
        copy_selected_body_state(
            self._tracked_pos_w_all,
            self._tracked_quat_w_all,
            self._tracked_linvel_w_all,
            self._tracked_angvel_w_all,
            mapped,
            out_pos,
            out_quat,
            out_lin_vel,
            out_ang_vel,
        )
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body positions")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        del body_ids
        self._unsupported_body_kinematics("base-frame body orientations")

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        # Analytical per the SimBackend contract (#1254): world-frame velocity
        # rotated into each body's own frame, matching MuJoCoBackend.
        mapped = self._mapped_tracked_ids("base-frame body linear velocities", body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_linvel_w_all[:, mapped, :],
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._sync_tracked_body_state()
        mapped = self._mapped_tracked_ids("base-frame body angular velocities", body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_angvel_w_all[:, mapped, :],
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._require_entity_healthy()
        try:
            address, dimension = self._sensor_slots[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._sensor_slots))
            raise ValueError(f"Sensor {name!r} not found; available: {available}") from exc
        return self._sensor_cache[:, address : address + dimension]

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Capture numeric host-cache slots for a zero-metadata hot-path view."""
        slots = tuple(self._sensor_slots[name] for name in names)

        def read() -> np.ndarray:
            self._require_entity_healthy()
            values = [
                self._sensor_cache[:, address : address + dimension] for address, dimension in slots
            ]
            return np.concatenate(values, axis=1)

        return read
