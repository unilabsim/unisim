import os
import tempfile
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, cast

import mjbatch
import mujoco
import numpy as np
from mjbatch._bindings import Batch as _RawMjBatch
from mjbatch.variants import VariantPack

from unisim.dr.types import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
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
    ModelSourceDescriptor,
    ResetRandomizationPayload,
    _validate_reset_term,
    require_op_body_ids,
)
from unisim.dtype import get_global_dtype
from unisim.scene import SceneCfg
from unisim.utils.rotation import np_quat_apply_inverse_batched

from ..base import (
    BackendHeightScanner,
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    BackendTerrainSpawnData,
    CameraCfg,
    DebugOverlayGetter,
    SimBackend,
    normalize_play_render_mode,
)
from ..body_state import copy_selected_body_state
from .playback import run_mujoco_playback

_MUJOCO_RESET_TERMS = frozenset(
    (
        RESET_TERM_BASE_COM,
        RESET_TERM_BASE_MASS,
        RESET_TERM_BODY_IQUAT,
        RESET_TERM_BODY_INERTIA,
        RESET_TERM_BODY_IPOS,
        RESET_TERM_BODY_MASS,
        RESET_TERM_DOF_ARMATURE,
        RESET_TERM_DOF_DAMPING,
        RESET_TERM_DOF_FRICTIONLOSS,
        RESET_TERM_GEOM_FRICTION,
        RESET_TERM_GEOM_SIZE,
        RESET_TERM_GEOM_SOLIMP,
        RESET_TERM_GEOM_SOLREF,
        RESET_TERM_GRAVITY,
        RESET_TERM_KP,
        RESET_TERM_KD,
    )
)


def _effective_cpu_count() -> int:
    """CPUs usable by this process for pool worker sizing.

    ``os.sched_getaffinity`` respects taskset/cgroup affinity masks, unlike
    ``os.cpu_count`` which reports machine-wide CPUs. Falls back to
    ``os.cpu_count`` where the syscall is unavailable (e.g. macOS).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, cpu_count() or 1)


def _root_state_dims(model) -> tuple[int, int]:
    if model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    return 0, 0


@dataclass(frozen=True)
class _StateLayout:
    """Column slices of one ``mjSTATE_INTEGRATION`` row (``bind("state")``).

    Offsets are accumulated from ``mujoco.mj_stateSize`` per state component in
    emission (bit) order, so models with delay buffers (``history``) or plugin
    state never shift the qpos/qvel columns. Only the components the adapter
    consumes are exposed; ``nstate`` pins the full row length.
    """

    time: slice
    qpos: slice
    qvel: slice
    act: slice
    ctrl: slice
    xfrc_applied: slice
    qacc_warmstart: slice
    nstate: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "_StateLayout":
        state = mujoco.mjtState
        ordered = (
            ("time", state.mjSTATE_TIME),
            ("qpos", state.mjSTATE_QPOS),
            ("qvel", state.mjSTATE_QVEL),
            ("act", state.mjSTATE_ACT),
            ("history", state.mjSTATE_HISTORY),
            ("qacc_warmstart", state.mjSTATE_WARMSTART),
            ("ctrl", state.mjSTATE_CTRL),
            ("qfrc_applied", state.mjSTATE_QFRC_APPLIED),
            ("xfrc_applied", state.mjSTATE_XFRC_APPLIED),
            ("eq_active", state.mjSTATE_EQ_ACTIVE),
            ("mocap_pos", state.mjSTATE_MOCAP_POS),
            ("mocap_quat", state.mjSTATE_MOCAP_QUAT),
            ("userdata", state.mjSTATE_USERDATA),
            ("plugin_state", state.mjSTATE_PLUGIN),
        )
        slices: dict[str, slice] = {}
        offset = 0
        for name, component in ordered:
            size = int(mujoco.mj_stateSize(model, component))
            slices[name] = slice(offset, offset + size)
            offset += size
        nstate = int(mujoco.mj_stateSize(model, state.mjSTATE_INTEGRATION))
        if offset != nstate:
            raise RuntimeError(
                f"mjSTATE_INTEGRATION layout mismatch: components sum to {offset}, "
                f"mj_stateSize reports {nstate}"
            )
        return cls(
            time=slices["time"],
            qpos=slices["qpos"],
            qvel=slices["qvel"],
            act=slices["act"],
            ctrl=slices["ctrl"],
            xfrc_applied=slices["xfrc_applied"],
            qacc_warmstart=slices["qacc_warmstart"],
            nstate=nstate,
        )


@dataclass
class _MuJoCoHeightScanner(BackendHeightScanner):
    backend: "MuJoCoBackend"
    hfield_geom_id: int
    offsets: np.ndarray
    frame_body_id: int
    alignment: str

    def scan(self) -> np.ndarray:
        pool = self.backend._pool
        transient_pool = pool is None
        if transient_pool:
            # ObservationManager evaluates terms once to infer their dimensions
            # before startup randomization and the backend's formal materialize
            # phase.  Use a real, short-lived pool for that cold-path read; the
            # scanner automatically switches to the final pool afterwards.
            pool = self.backend._build_pool()
        assert pool is not None
        backend = self.backend
        # mjbatch query ops skip the bound-field CopyOut, so the bound views
        # are untouched by this call.
        out = np.zeros((backend._num_envs, self.offsets.shape[0]), dtype=np.float64)
        # Id-based raw binding: geom/body ids are resolved once on the cold
        # path and may refer to unnamed model elements.
        _RawMjBatch.sample_hfield(
            pool,
            self.hfield_geom_id,
            self.frame_body_id,
            self.offsets,
            out,
            None,
            self.alignment,
        )
        return np.asarray(out, dtype=backend._np_dtype)


def _actuator_ids_from_selector(model, actuator_ids) -> np.ndarray:
    ids = np.arange(model.nu)[actuator_ids]
    return np.atleast_1d(np.asarray(ids, dtype=np.int32))


def _assert_position_actuator_targets(model, actuator_ids=slice(None)) -> None:
    ids = _actuator_ids_from_selector(model, actuator_ids)
    if ids.size == 0:
        return
    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    invalid = ids[np.asarray(model.actuator_biastype[ids], dtype=np.int32) != affine_bias]
    if invalid.size == 0:
        return
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(idx)) or str(int(idx))
        for idx in invalid[:8]
    ]
    suffix = "" if invalid.size <= 8 else f", ... ({invalid.size} total)"
    raise ValueError(
        "position_actuator_gains can only target MuJoCo position actuators; "
        f"non-position actuator ids/names: {', '.join(names)}{suffix}"
    )


def _apply_position_actuator_gains_to_mj_model(
    model,
    *,
    kp: float | np.ndarray,
    kd: float | np.ndarray,
    actuator_ids=slice(None),
) -> None:
    _assert_position_actuator_targets(model, actuator_ids)
    kp_arr = np.asarray(kp, dtype=np.float64)
    kd_arr = np.asarray(kd, dtype=np.float64)
    model.actuator_gainprm[actuator_ids, 0] = kp_arr
    model.actuator_biasprm[actuator_ids, 1] = -kp_arr
    model.actuator_biasprm[actuator_ids, 2] = -kd_arr


def _remove_temp_xml(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


class _TempXmlCleanup:
    def __init__(self, path: str) -> None:
        self.path = path
        self._finalizer = weakref.finalize(self, _remove_temp_xml, path)

    def cleanup(self) -> None:
        self._finalizer()

    def __del__(self) -> None:
        self.cleanup()


@dataclass(frozen=True)
class _FixedVariantBuild:
    """Cold-path artifacts backing one immutable fixed variant plan."""

    plan: FixedVariantPlan
    pack: VariantPack
    default_tables: Mapping[str, tuple[np.ndarray, ...]]
    geom_names: tuple[tuple[str, ...], ...]
    default_qpos: np.ndarray


def _model_names(model: mujoco.MjModel, kind: str, count: int) -> tuple[str, ...]:
    accessor = getattr(model, kind)
    return tuple(str(accessor(i).name) for i in range(count))


def _public_layout(
    model: mujoco.MjModel,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Return the layout consumed by UniSim's uniform public arrays."""
    sizes = tuple(
        int(value)
        for value in (
            model.nq,
            model.nv,
            model.na,
            model.nu,
            model.nbody,
            model.njnt,
            model.nsite,
            model.nsensor,
            model.nsensordata,
        )
    )
    names = (
        *_model_names(model, "body", model.nbody),
        *_model_names(model, "joint", model.njnt),
        *_model_names(model, "site", model.nsite),
        *_model_names(model, "actuator", model.nu),
        *_model_names(model, "sensor", model.nsensor),
    )
    topology = (
        tuple(int(value) for value in model.body_parentid),
        tuple(int(value) for value in model.jnt_type),
        tuple(int(value) for value in model.jnt_qposadr),
        tuple(int(value) for value in model.jnt_dofadr),
        tuple(int(value) for value in model.sensor_adr),
        tuple(int(value) for value in model.sensor_dim),
    )
    return sizes, names, topology


def _validate_fixed_variant_layout(
    models: Sequence[mujoco.MjModel], requested: FixedVariantLayout
) -> None:
    """Fail closed on topology changes that a canonical batch cannot hide."""
    reference_layout = _public_layout(models[0])
    geom_names = [_model_names(model, "geom", model.ngeom) for model in models]
    same_layout = all(
        _public_layout(model) == reference_layout and names == geom_names[0]
        for model, names in zip(models, geom_names, strict=True)
    )
    if same_layout:
        return

    # The only canonicalization supported by the executor is optional named
    # mesh-geom slots: the union model disables a missing slot per world. All
    # state/control/sensor topology must still be name-for-name identical.
    canonical_geoms = set().union(*(set(names) for names in geom_names))
    geom_type_by_name = {
        name: int(model.geom_type[geom_id])
        for model, names in zip(models, geom_names, strict=True)
        for geom_id, name in enumerate(names)
    }
    uniform_public_layout = all(
        _public_layout(model) == reference_layout
        and len(names) == len(set(names))
        and "" not in names
        and set(names) <= canonical_geoms
        for model, names in zip(models, geom_names, strict=True)
    )
    if not uniform_public_layout:
        raise ValueError(
            "fixed variants must have the same public topology; heterogeneous "
            "topology groups are not supported by this MuJoCo batch"
        )
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for names in geom_names:
        if any(
            geom_type_by_name[name] != mesh_type for name in canonical_geoms if name not in names
        ):
            raise ValueError("only optional mesh-geom slots may be absent from a variant")
    if requested is FixedVariantLayout.SAME_LAYOUT:
        raise ValueError(
            "fixed variants have optional mesh-geom slots; declare "
            "uniform_public_layout instead of same_layout"
        )


def _configured_variant_spec(
    spec: mujoco.MjSpec,
    *,
    sim_dt: float,
    iterations: int | None,
    position_actuator_gains: dict | None,
) -> mujoco.MjSpec:
    """Apply backend model configuration before an independent compile."""
    spec.option.timestep = sim_dt
    if iterations is not None:
        spec.option.iterations = iterations
    if position_actuator_gains is None:
        return spec

    probe = spec.compile()
    _apply_position_actuator_gains_to_mj_model(probe, **position_actuator_gains)
    for actuator, gainprm, biasprm in zip(
        spec.actuators,
        probe.actuator_gainprm,
        probe.actuator_biasprm,
        strict=True,
    ):
        actuator.gainprm = list(np.asarray(gainprm, dtype=np.float64))
        actuator.biasprm = list(np.asarray(biasprm, dtype=np.float64))
    return spec


@dataclass
class _MuJoCoSceneContext:
    model_source: str | mujoco.MjModel
    model_file: str
    visual_model_file: str | None = None
    artifacts_dir: str | None = None
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: Any | None = None
    cleanup_handle: Any | None = None


def _build_mujoco_scene_context(scene: SceneCfg) -> _MuJoCoSceneContext:
    from unisim.backend.mujoco.xml import (
        materialize_mujoco_hfield_attached_scene,
        materialize_scene_fragments,
    )

    if scene is None:
        raise ValueError("SceneCfg must be provided")
    if not scene.model_file:
        raise ValueError("SceneCfg.model_file must be provided")

    if scene.terrain is None:
        if not scene.fragment_files:
            return _MuJoCoSceneContext(
                model_source=scene.model_file,
                model_file=scene.model_file,
                visual_model_file=scene.visual_model_file or scene.model_file,
            )
        model_source = materialize_scene_fragments(
            scene.model_file,
            fragment_files=scene.fragment_files,
        )
        return _MuJoCoSceneContext(
            model_source=model_source,
            model_file=scene.model_file,
            visual_model_file=model_source,
            cleanup_handle=_TempXmlCleanup(model_source),
        )

    if scene.terrain.generator is None:
        raise ValueError("SceneCfg.terrain.generator must be configured for terrain scenes")

    output_dir = tempfile.TemporaryDirectory(prefix="unilab_scene_")
    try:
        _, terrain_origins, terrain_surface_sampler = materialize_mujoco_hfield_attached_scene(
            model_file=scene.model_file,
            terrain_cfg=scene.terrain.generator,
            output_dir=output_dir.name,
            fragment_files=scene.fragment_files,
            hfield_name=scene.terrain.hfield_name,
            geom_name=scene.terrain.geom_name or "floor",
            return_surface_sampler=True,
        )
    except Exception:
        output_dir.cleanup()
        raise

    return _MuJoCoSceneContext(
        # The materializer already writes the complete composed scene.  Keep XML
        # as the physics source so manager-requested body sensors can be injected
        # before compilation just like they are for static scenes.
        model_source=os.path.join(output_dir.name, "scene.xml"),
        model_file=scene.model_file,
        visual_model_file=os.path.join(output_dir.name, "scene.xml"),
        artifacts_dir=output_dir.name,
        terrain_origins=terrain_origins,
        terrain_surface_sampler=terrain_surface_sampler,
        cleanup_handle=output_dir,
    )


class MuJoCoBackend(SimBackend):
    """MuJoCo backend implementation.

    The native batch executor is ``mjbatch.Batch`` (unilabsim fork). Canonical
    state storage is the batch's bound per-field views (see ``_bind_views``);
    the adapter never ships full state rows to the pool. Behavioral contract:

    1. ``xfrc_applied`` is written absolutely before every dispatch (zero when
       nothing is staged): mjbatch persists the channel across steps, so staged
       wrenches must not be left to decay on their own.
    2. Warmstart is structurally zeroed on state upload: ``Batch.reset`` runs
       ``mj_resetData`` (zeroing ``qacc_warmstart``/act/ctrl/xfrc/time) and the
       velocity-delta path zeroes the bound warmstart view explicitly.
    3. State layout is derived per component from ``mujoco.mj_stateSize``; no
       hardcoded FULLPHYSICS offsets exist anywhere in the adapter.
    """

    _fixed_variant_build: _FixedVariantBuild | None = None

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        base_name: Optional[str] = None,
        np_dtype=None,
        add_body_sensors: bool = False,
        refresh_pre_step_body_state: bool = True,
        position_actuator_gains: dict | None = None,
        iterations: int | None = None,
        push_body_name: Optional[str] = None,
        cpu_ids: Optional[Sequence[int]] = None,
    ):
        if not isinstance(refresh_pre_step_body_state, bool):
            raise TypeError("refresh_pre_step_body_state must be bool")
        scene_context = _build_mujoco_scene_context(scene)
        self.scene_model_file = scene_context.model_file
        self.scene_visual_model_file = scene_context.visual_model_file
        self.scene_artifacts_dir = scene_context.artifacts_dir
        self.terrain_origins = scene_context.terrain_origins
        self.terrain_surface_sampler = scene_context.terrain_surface_sampler
        self._terrain_spawn_data = (
            None
            if self.terrain_origins is None
            else BackendTerrainSpawnData(
                terrain_origins=self.terrain_origins,
                sample_height=(
                    None
                    if self.terrain_surface_sampler is None
                    else self.terrain_surface_sampler.sample_height
                ),
            )
        )
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self.add_body_sensors = add_body_sensors
        self._refresh_pre_step_body_state = refresh_pre_step_body_state
        self._base_name = base_name
        self._push_body_name = push_body_name
        self._model_file = scene_context.model_source
        self._sim_dt = float(sim_dt)
        self._iterations = None if iterations is None else int(iterations)
        self._cpu_ids = self._validate_cpu_ids(cpu_ids)
        self._position_actuator_gains = (
            None if position_actuator_gains is None else dict(position_actuator_gains)
        )
        self._tracked_sensor_copyout_range: tuple[int, int] | None = None
        self._fixed_variant_build: _FixedVariantBuild | None = None
        self._static_playback_model: mujoco.MjModel | None = None
        self._num_envs = num_envs
        self._np_dtype = np_dtype if np_dtype is not None else get_global_dtype()
        self.backend_type = "mujoco"

        # Thread configuration. An explicit ``cpu_ids`` affinity pins one worker
        # per CPU, so it also fixes the pool worker count. Otherwise size the
        # pool to the CPUs actually usable by this process: oversubscribing
        # (e.g. 2x) only adds contention once the physics phase saturates
        # memory bandwidth (#1328).
        self._n_threads = (
            min(num_envs, _effective_cpu_count()) if self._cpu_ids is None else len(self._cpu_ids)
        )

        self._pool: mjbatch.Batch | None = None
        if scene.fixed_variant_plan is not None:
            # Complete variant sources are the authoritative cold-path inputs;
            # do not compile and initialize a canonical scene merely to discard it.
            self._install_fixed_variant_plan(scene.fixed_variant_plan)
            return

        self._model = self._load_base_model()
        self._base_body_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_name is not None
            else -1
        )
        self._push_body_id = self._resolve_push_body_id(self._model)
        self._push_body_force_slice = self._resolve_push_body_force_slice(self._push_body_id)
        self._pending_xfrc_applied = np.zeros((num_envs, 6 * self._model.nbody), dtype=np.float64)
        # State indices and integration-row layout.
        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nact = int(self._model.na)
        self._state_layout = _StateLayout.from_model(self._model)
        self._root_qpos_dim, self._root_qvel_dim = _root_state_dims(self._model)
        self._num_dof_pos = self.nq - self._root_qpos_dim
        self._num_dof_vel = self.nv - self._root_qvel_dim
        self._interval_root_velocity_qvel_ids = self._resolve_interval_root_velocity_qvel_ids()

        # Host-side state storage, replaced by the batch's bound views at
        # materialize().  Until then these arrays answer getter calls (cold
        # path: observation-dimension inference before materialize).
        self._time_view = np.zeros((num_envs,), dtype=self._np_dtype)
        self._qpos_view = np.broadcast_to(
            np.asarray(self._model.qpos0, dtype=self._np_dtype), (num_envs, self.nq)
        ).copy()
        self._qvel_view = np.zeros((num_envs, self.nv), dtype=self._np_dtype)
        self._sensor_data = np.zeros((num_envs, self._model.nsensordata), dtype=self._np_dtype)
        self._rebuild_derived_views()
        self._base_body_mass = np.broadcast_to(
            np.asarray(self._model.body_mass), (num_envs, self._model.nbody)
        ).copy()
        self._base_body_ipos = np.broadcast_to(
            np.asarray(self._model.body_ipos), (num_envs, self._model.nbody, 3)
        ).copy()
        if scene.fixed_variant_plan is not None:
            self._install_fixed_variant_plan(scene.fixed_variant_plan)

    # ------------------------------------------------------------------ #
    # Views                                                              #
    # ------------------------------------------------------------------ #

    def _rebuild_derived_views(self) -> None:
        """(Re)derive dof/base/sensor views rooted at the current state arrays.

        Called with the host arrays at construction and again at materialize
        once the canonical storage is re-pointed at the batch's bound views.
        """
        num_envs = self._num_envs
        self._dof_pos_view = self._qpos_view[:, self._root_qpos_dim : self.nq]
        self._dof_vel_view = self._qvel_view[:, self._root_qvel_dim : self.nv]
        if self._root_qpos_dim == 7:
            self._base_pos_view = self._qpos_view[:, 0:3]
            self._base_quat_view = self._qpos_view[:, 3:7]
            self._base_lin_vel_view = self._qvel_view[:, 0:3]
            self._base_ang_vel_view = self._qvel_view[:, 3:6]
        else:
            if self._base_body_id >= 0:
                data0 = mujoco.MjData(self._model)
                mujoco.mj_forward(self._model, data0)
                base_pos = np.asarray(data0.xpos[self._base_body_id], dtype=self._np_dtype).copy()
                base_quat = np.asarray(data0.xquat[self._base_body_id], dtype=self._np_dtype).copy()
            else:
                base_pos = np.zeros((3,), dtype=self._np_dtype)
                base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=self._np_dtype)
            self._base_pos_view = np.broadcast_to(base_pos, (num_envs, 3)).copy()
            self._base_quat_view = np.broadcast_to(base_quat, (num_envs, 4)).copy()
            self._base_lin_vel_view = np.zeros((num_envs, 3), dtype=self._np_dtype)
            self._base_ang_vel_view = np.zeros((num_envs, 3), dtype=self._np_dtype)
        self._build_sensor_views()

    def _build_sensor_views(self) -> None:
        """(Re)build named sensor slices rooted at the current ``_sensor_data``."""
        num_envs = self._num_envs
        self._sensor_indices = {}
        self._sensor_views = {}
        for i in range(self._model.nsensor):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                adr = self._model.sensor_adr[i]
                dim = self._model.sensor_dim[i]
                self._sensor_indices[name] = list(range(adr, adr + dim))
                self._sensor_views[name] = self._sensor_data[:, adr : adr + dim]

        # Zero-copy view mapping for tracked-body sensors.
        if self.add_body_sensors and self._valid_bnames:

            def _get_sensor_view(prefix, dim):
                adrs = [
                    self._model.sensor_adr[
                        mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_{nb}")
                    ]
                    for nb in self._valid_bnames
                ]
                return self._sensor_data[:, adrs[0] : adrs[-1] + dim].reshape(
                    num_envs, len(self._valid_bnames), dim
                )

            # Global (world) sensors
            self._tracked_pos_w_all = _get_sensor_view("track_pos_w", 3)
            self._tracked_quat_w_all = _get_sensor_view("track_quat_w", 4)
            self._tracked_linvel_w_all = _get_sensor_view("track_linvel_w", 3)
            self._tracked_angvel_w_all = _get_sensor_view("track_angvel_w", 3)
            # One contiguous sensordata column superset covering the four
            # world-frame blocks (they may interleave unrelated sensors only
            # between blocks, which the superset copies harmlessly).  Used as
            # the opt-in mjbatch split-substep copyout range.
            starts: list[int] = []
            stops: list[int] = []
            for prefix, dim in (
                ("track_pos_w", 3),
                ("track_quat_w", 4),
                ("track_linvel_w", 3),
                ("track_angvel_w", 3),
            ):
                first = self._model.sensor_adr[
                    mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}_{self._valid_bnames[0]}"
                    )
                ]
                last = self._model.sensor_adr[
                    mujoco.mj_name2id(
                        self._model,
                        mujoco.mjtObj.mjOBJ_SENSOR,
                        f"{prefix}_{self._valid_bnames[-1]}",
                    )
                ]
                starts.append(int(first))
                stops.append(int(last) + dim)
            self._tracked_sensor_copyout_range = (min(starts), max(stops))

            # Local (baselink) sensors
            self._tracked_pos_b_all = _get_sensor_view("track_pos_b", 3)
            self._tracked_quat_b_all = _get_sensor_view("track_quat_b", 4)

    def _bind_views(self, batch: mjbatch.Batch) -> None:
        """Re-point canonical storage at the batch's bound per-field views.

        Upload order matters: each bound field is written from the current
        host array before the attribute is re-pointed, so no pre-materialize
        state is lost.  The batch is constructed with ``forward=False``: after
        ``step()`` the sensordata view is one substep behind qpos/qvel, matching
        mj_step itself; ``reset``/``forward`` always leave it current.  The
        closing ``forward()`` makes sensordata (and every derived field) valid
        immediately after materialize, before the first step or reset — a
        derived field bound between calls is only filled by the next call.
        """
        dtype = self._np_dtype
        self._time_view = batch.bind("time", dtype)
        self._time_view[:] = 0.0
        qpos = batch.bind("qpos", dtype)
        qpos[:] = self._qpos_view
        self._qpos_view = qpos
        qvel = batch.bind("qvel", dtype)
        qvel[:] = self._qvel_view
        self._qvel_view = qvel
        # Fresh batches start with zero act; no host upload exists for it.
        self._act_view = batch.bind("act", dtype)
        self._ctrl_view = batch.bind("ctrl", dtype)
        # Staging parity: the wrench and warmstart reach the sim unrounded.
        self._xfrc_view = batch.bind("xfrc_applied")
        self._warm_view = batch.bind("qacc_warmstart")
        sensordata = batch.bind("sensordata", dtype)
        sensordata[:] = self._sensor_data
        self._sensor_data = sensordata
        self._rebuild_derived_views()
        batch.forward()

    def _load_base_model(self) -> mujoco.MjModel:
        if isinstance(self._model_file, mujoco.MjModel):
            if self.add_body_sensors:
                raise ValueError("add_body_sensors is not supported for precompiled MuJoCo models")
            self._tracked_body_ids = []
            self._valid_bnames = []
            model = self._model_file
            self._configure_model(model)
            return model

        model_path, tmp_paths, tracked_body_ids, valid_bnames = self._prepare_model_xml()
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
        finally:
            for tmp_path in reversed(tmp_paths):
                os.remove(tmp_path)

        if self.add_body_sensors:
            # MjSpec compilation can reorder bodies expanded from <replicate>.
            # Sensor columns follow ``valid_bnames`` insertion order, so rebuild
            # the name-to-column map from the final compiled model instead of
            # retaining IDs from the pre-injection source model.
            self._tracked_body_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in valid_bnames
            ]
            missing = [
                name
                for name, body_id in zip(valid_bnames, self._tracked_body_ids, strict=True)
                if body_id < 0
            ]
            if missing:
                raise ValueError(
                    "Injected MuJoCo body tracking sensors reference bodies missing from "
                    f"the compiled model: {missing}"
                )
            self._body_id_to_tracked_idx = np.full(model.nbody, -1, dtype=int)
            for idx, bid in enumerate(self._tracked_body_ids):
                self._body_id_to_tracked_idx[bid] = idx
        else:
            self._tracked_body_ids = tracked_body_ids
        self._valid_bnames = valid_bnames
        self._configure_model(model)
        return model

    def _prepare_model_xml(self) -> tuple[str, list[str], list[int], list[str]]:
        from unisim.backend.mujoco.xml import (
            create_discardvisual_xml,
            inject_mujoco_tracking_sensors,
        )

        model_path = create_discardvisual_xml(str(self._model_file))
        tmp_paths = [model_path]
        if self.add_body_sensors:
            model_path, tracked_body_ids, valid_bnames = inject_mujoco_tracking_sensors(
                model_path,
                baselink_name=self._base_name,
            )
            tmp_paths.append(model_path)
        else:
            tracked_body_ids = []
            valid_bnames = []
        return model_path, tmp_paths, tracked_body_ids, valid_bnames

    def _configure_model(self, model: mujoco.MjModel) -> None:
        model.opt.timestep = self._sim_dt
        if self._iterations is not None:
            model.opt.iterations = self._iterations
        if self._position_actuator_gains is not None:
            self._apply_position_actuator_gains_to_model(model, **self._position_actuator_gains)

    def _load_fixed_variant_build(
        self, plan: FixedVariantPlan
    ) -> tuple[_FixedVariantBuild, list[str]]:
        """Independently compile every source, then build one canonical batch."""
        from unisim.backend.mujoco.xml import (
            create_discardvisual_xml,
            inject_mujoco_tracking_sensors,
        )

        physics_specs: list[mujoco.MjSpec] = []
        temp_paths: list[str] = []
        valid_bnames: list[str] | None = None
        try:
            for descriptor in plan.variants:
                source = self._variant_source(descriptor)
                physics_path = create_discardvisual_xml(source)
                temp_paths.append(physics_path)
                if self.add_body_sensors:
                    physics_path, _, names = inject_mujoco_tracking_sensors(
                        physics_path, baselink_name=self._base_name
                    )
                    temp_paths.append(physics_path)
                    if valid_bnames is None:
                        valid_bnames = list(names)
                    elif valid_bnames != list(names):
                        raise ValueError("fixed variants must have the same named body layout")

                physics_spec = _configured_variant_spec(
                    mujoco.MjSpec.from_file(physics_path),
                    sim_dt=self._sim_dt,
                    iterations=self._iterations,
                    position_actuator_gains=self._position_actuator_gains,
                )
                physics_specs.append(physics_spec)

            physics_models = tuple(spec.compile() for spec in physics_specs)
            _validate_fixed_variant_layout(physics_models, plan.layout)
            pack = VariantPack.from_specs(physics_specs)
            default_qpos = np.stack([np.asarray(model.qpos0) for model in physics_models])
            default_tables = {
                term: tuple(
                    self._reset_term_default_from_model(term, model) for model in physics_models
                )
                for term in _MUJOCO_RESET_TERMS
            }
            default_tables.update(
                {
                    "actuator_gainprm": tuple(
                        np.asarray(model.actuator_gainprm, dtype=np.float64).copy()
                        for model in physics_models
                    ),
                    "actuator_biasprm": tuple(
                        np.asarray(model.actuator_biasprm, dtype=np.float64).copy()
                        for model in physics_models
                    ),
                }
            )
            build = _FixedVariantBuild(
                plan=plan,
                pack=pack,
                default_tables=MappingProxyType(default_tables),
                geom_names=tuple(
                    _model_names(model, "geom", model.ngeom) for model in physics_models
                ),
                default_qpos=default_qpos,
            )
            return build, list(valid_bnames or [])
        finally:
            for path in reversed(temp_paths):
                _remove_temp_xml(path)

    @staticmethod
    def _variant_source(descriptor: ModelSourceDescriptor) -> str:
        path = Path(descriptor.model_file)
        if not path.is_file():
            raise ValueError(f"fixed variant MJCF source does not exist: {descriptor.model_file}")
        return str(path.resolve())

    def _install_fixed_variant_plan(self, plan: FixedVariantPlan) -> None:
        """Install immutable compiler-coherent identities during construction."""
        plan.validate(self._num_envs)
        if not self._supports_fixed_variant_executor():
            raise NotImplementedError(
                "MuJoCoBackend cannot realize the fixed variant plan: the mjbatch "
                "VariantPack API is unavailable"
            )

        build, valid_bnames = self._load_fixed_variant_build(plan)
        assignment = np.asarray(plan.assignment, dtype=np.int32)
        self._model = build.pack.model
        self._base_body_id = (
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._base_name)
            if self._base_name is not None
            else -1
        )
        self._push_body_id = self._resolve_push_body_id(self._model)
        self._push_body_force_slice = self._resolve_push_body_force_slice(self._push_body_id)
        self._pending_xfrc_applied = np.zeros(
            (self._num_envs, 6 * self._model.nbody), dtype=np.float64
        )

        if self.add_body_sensors:
            self._tracked_body_ids = [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in valid_bnames
            ]
            missing = [
                name
                for name, body_id in zip(valid_bnames, self._tracked_body_ids, strict=True)
                if body_id < 0
            ]
            if missing:
                raise ValueError(
                    "Injected fixed-variant body sensors reference bodies missing "
                    f"from the canonical model: {missing}"
                )
            self._body_id_to_tracked_idx = np.full(self._model.nbody, -1, dtype=int)
            for idx, body_id in enumerate(self._tracked_body_ids):
                self._body_id_to_tracked_idx[body_id] = idx
        self._valid_bnames = valid_bnames

        self.nq = self._model.nq
        self.nv = self._model.nv
        self.nact = int(self._model.na)
        self._state_layout = _StateLayout.from_model(self._model)
        self._root_qpos_dim, self._root_qvel_dim = _root_state_dims(self._model)
        self._num_dof_pos = self.nq - self._root_qpos_dim
        self._num_dof_vel = self.nv - self._root_qvel_dim
        self._interval_root_velocity_qvel_ids = self._resolve_interval_root_velocity_qvel_ids()

        self._time_view = np.zeros((self._num_envs,), dtype=self._np_dtype)
        self._qpos_view = np.asarray(build.default_qpos[assignment], dtype=self._np_dtype).copy()
        self._qvel_view = np.zeros((self._num_envs, self.nv), dtype=self._np_dtype)
        self._sensor_data = np.zeros(
            (self._num_envs, self._model.nsensordata), dtype=self._np_dtype
        )
        self._rebuild_derived_views()
        self._base_body_mass = np.stack(build.default_tables[RESET_TERM_BODY_MASS])[
            assignment
        ].copy()
        self._base_body_ipos = np.stack(build.default_tables[RESET_TERM_BODY_IPOS])[
            assignment
        ].copy()
        self._fixed_variant_build = build

    def _resolve_push_body_id(self, model: mujoco.MjModel) -> int:
        body_name = self._push_body_name if self._push_body_name is not None else self._base_name
        if body_name is None:
            return -1
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Push body '{body_name}' not found in MuJoCo model")
        return int(body_id)

    def _resolve_push_body_force_slice(self, body_id: int) -> slice:
        if body_id < 0:
            return slice(0, 0)
        start = 6 * body_id
        return slice(start, start + 3)

    def _resolve_push_body_torque_slice(self, body_id: int) -> slice:
        if body_id < 0:
            return slice(0, 0)
        start = 6 * body_id
        return slice(start + 3, start + 6)

    def _sample_push_force(self, force_range: Sequence[float] | np.ndarray) -> np.ndarray:
        """Sample one world-frame push force vector per environment.

        Args:
            force_range: Per-axis push-force magnitude range.

        Returns:
            Array with shape ``(num_envs, 3)`` containing sampled forces.
        """
        ex_force = np.random.uniform(-1.0, 1.0, size=(self._num_envs, 3))
        ex_force *= np.asarray(force_range, dtype=np.float64)
        return ex_force.astype(np.float64, copy=False)

    @staticmethod
    def _validate_cpu_ids(cpu_ids: Optional[Sequence[int]]) -> tuple[int, ...] | None:
        """Cold-path structural validation for the optional worker CPU affinity.

        ``cpu_ids[i]`` pins pool worker thread ``i`` to one CPU, so its length
        also fixes ``nthread``. Platform/availability checks happen in mjbatch
        when the pool is created.
        """
        if cpu_ids is None:
            return None
        if isinstance(cpu_ids, (str, bytes)):
            raise TypeError("cpu_ids must be a sequence of integer CPU ids")
        entries = list(cpu_ids)
        if not entries:
            raise ValueError("cpu_ids must be non-empty")
        for cpu_id in entries:
            if isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0:
                raise ValueError(f"cpu_ids entries must be non-negative integers, got {cpu_id!r}")
        ids = tuple(entries)
        if len(set(ids)) != len(ids):
            raise ValueError(f"cpu_ids entries must be unique, got {list(ids)!r}")
        return ids

    def _build_pool(self) -> mjbatch.Batch:
        """Construct a batch seeded with the current host state.

        Also used for the scanner's transient pre-materialize pool; only
        ``materialize`` re-points the backend's canonical views at the result.
        """
        kwargs: dict[str, Any] = {"num_threads": self._n_threads}
        if self._cpu_ids is not None:
            kwargs["cpu_ids"] = list(self._cpu_ids)
        if self._fixed_variant_build is None:
            batch = mjbatch.Batch(self._model, self._num_envs, forward=False, **kwargs)
        else:
            batch = mjbatch.Batch.from_variant_pack(
                self._fixed_variant_build.pack,
                self._num_envs,
                self._fixed_variant_build.plan.assignment,
                forward=False,
                **kwargs,
            )
            self._scatter_fixed_variant_defaults(batch)
        batch.bind("qpos", self._np_dtype)[:] = self._qpos_view
        batch.bind("qvel", self._np_dtype)[:] = self._qvel_view
        batch.forward()
        return batch

    def _scatter_fixed_variant_defaults(self, batch: mjbatch.Batch) -> None:
        """Seed every advertised reset field from the independent oracles."""
        build = self._fixed_variant_build
        assert build is not None
        assignment = np.asarray(build.plan.assignment, dtype=np.int64)
        direct_fields = {
            RESET_TERM_GRAVITY: "gravity",
            RESET_TERM_BODY_IQUAT: "body_iquat",
            RESET_TERM_BODY_INERTIA: "body_inertia",
            RESET_TERM_BODY_IPOS: "body_ipos",
            RESET_TERM_BODY_MASS: "body_mass",
            RESET_TERM_DOF_ARMATURE: "dof_armature",
            RESET_TERM_DOF_DAMPING: "dof_damping",
            RESET_TERM_DOF_FRICTIONLOSS: "dof_frictionloss",
            RESET_TERM_GEOM_FRICTION: "geom_friction",
            RESET_TERM_GEOM_SIZE: "geom_size",
            RESET_TERM_GEOM_SOLIMP: "geom_solimp",
            RESET_TERM_GEOM_SOLREF: "geom_solref",
        }
        pack_seeded_fields = {
            RESET_TERM_BODY_IPOS,
            RESET_TERM_BODY_IQUAT,
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_BODY_MASS,
            RESET_TERM_GEOM_SIZE,
        }
        for term, field in direct_fields.items():
            if term in pack_seeded_fields:
                continue
            batch.expand(field)[:] = self.get_reset_term_default(term)
        batch.expand("actuator_gainprm")[:] = np.stack(build.default_tables["actuator_gainprm"])[
            assignment
        ]
        batch.expand("actuator_biasprm")[:] = np.stack(build.default_tables["actuator_biasprm"])[
            assignment
        ]
        batch.set_const()

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    # ------------------------------------------------------------------ #
    # Model properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_actuators(self) -> int:
        return int(self._model.nu)

    @property
    def num_dof_vel(self) -> int:
        return int(self._num_dof_vel)

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.array(self._model.actuator_ctrlrange, dtype=self._np_dtype)

    def get_actuator_names(self) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            or f"#{actuator_id}"
            for actuator_id in range(int(self._model.nu))
        )

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        supported_transmissions = {
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        supported_joint_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        names: list[str] = []
        for actuator_id in range(int(self._model.nu)):
            transmission = int(self._model.actuator_trntype[actuator_id])
            joint_id = int(self._model.actuator_trnid[actuator_id, 0])
            if transmission not in supported_transmissions or joint_id < 0:
                actuator_name = self.get_actuator_names()[actuator_id]
                raise NotImplementedError(
                    "backend 'mujoco' capability 'actuator target joint' requires "
                    f"a joint transmission; actuator '{actuator_name}' uses "
                    f"transmission type {transmission}"
                )
            if int(self._model.jnt_type[joint_id]) not in supported_joint_types:
                actuator_name = self.get_actuator_names()[actuator_id]
                raise NotImplementedError(
                    "backend 'mujoco' capability 'actuator target joint' requires "
                    f"a single-DoF joint; actuator '{actuator_name}' targets joint id {joint_id}"
                )
            joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not joint_name:
                raise NotImplementedError(
                    "backend 'mujoco' capability 'actuator target joint' requires named joints; "
                    f"actuator id {actuator_id} targets unnamed joint id {joint_id}"
                )
            names.append(joint_name)
        return tuple(names)

    def get_scene_model_file(self) -> str | None:
        return str(self.scene_model_file) if self.scene_model_file else None

    def get_scene_visual_model_file(self) -> str | None:
        return str(self.scene_visual_model_file) if self.scene_visual_model_file else None

    def get_terrain_spawn_data(self) -> BackendTerrainSpawnData | None:
        return self._terrain_spawn_data

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found in MuJoCo model")
        return np.array(self._model.key_qpos[key_id].copy(), dtype=self._np_dtype)

    def get_default_qpos(self) -> np.ndarray:
        if self._fixed_variant_build is not None:
            variant = int(self._fixed_variant_build.plan.assignment[0])
            return np.asarray(
                self._fixed_variant_build.default_qpos[variant], dtype=np.float64
            ).copy()
        return np.asarray(self._model.qpos0, dtype=np.float64).copy()

    def get_default_dof_pos(self) -> np.ndarray:
        if self._fixed_variant_build is not None:
            variant = int(self._fixed_variant_build.plan.assignment[0])
            return np.asarray(
                self._fixed_variant_build.default_qpos[variant][self._root_qpos_dim :],
                dtype=self._np_dtype,
            ).copy()
        return np.asarray(self._model.qpos0[self._root_qpos_dim :], dtype=self._np_dtype).copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self.nv,), dtype=self._np_dtype)

    def get_state(
        self, fields: tuple[str, ...] | str | None = None
    ) -> Mapping[str, np.ndarray]:
        """Return detached full generalized-state snapshots.

        Complete MuJoCo qpos/qvel layouts already include every root and
        non-root joint.  Rebuilding them from base/dof getters would insert a
        virtual root when the first joint is not free and break ``set_state``
        and named-state index compatibility.
        """
        requested = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else tuple(fields))
        )
        result: dict[str, np.ndarray] = {}
        if "qpos" in requested:
            result["qpos"] = self._qpos_view.copy()
        if "qvel" in requested:
            result["qvel"] = self._qvel_view.copy()
        if "ctrl" in requested:
            raise NotImplementedError(f"{self.__class__.__name__} does not expose control state")
        unknown = set(requested) - {"qpos", "qvel", "ctrl"}
        if unknown:
            raise KeyError(f"unknown {self.backend_type} state field(s): {sorted(unknown)}")
        return result

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset each world to its own fixed variant's compiler default."""
        if self._fixed_variant_build is None:
            super().reset(env_ids)
            return
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int32)
        )
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids must be a one-dimensional in-range index array")
        variants = np.asarray(self._fixed_variant_build.plan.assignment, dtype=np.int32)[ids]
        qpos = np.asarray(
            self._fixed_variant_build.default_qpos[variants], dtype=self._np_dtype
        ).copy()
        qvel = np.zeros((ids.size, self.nv), dtype=self._np_dtype)
        self.set_state(ids, qpos, qvel)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if body_id < 0:
            raise ValueError(f"Body '{root_body_name}' not found in MuJoCo model")
        joint_count = int(self._model.body_jntnum[body_id])
        joint_id = int(self._model.body_jntadr[body_id])
        free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
        if joint_count != 1 or joint_id < 0 or int(self._model.jnt_type[joint_id]) != free_joint:
            raise NotImplementedError(
                "backend 'mujoco' capability 'root-state layout' requires body "
                f"'{root_body_name}' to own exactly one free joint"
            )
        qpos_start = int(self._model.jnt_qposadr[joint_id])
        qvel_start = int(self._model.jnt_dofadr[joint_id])
        return BackendRootStateLayout(
            qpos_indices=tuple(range(qpos_start, qpos_start + 7)),
            qvel_indices=tuple(range(qvel_start, qvel_start + 6)),
        )

    def _resolve_interval_root_velocity_qvel_ids(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """Bind the configured free root's qvel columns and quat qpos columns."""
        if self._base_name is None or self._base_body_id < 0:
            return None
        try:
            layout = self.get_root_state_layout(self._base_name)
        except (NotImplementedError, ValueError):
            return None
        qvel_ids = tuple(int(index) for index in layout.qvel_indices)
        quat_ids = tuple(int(index) for index in layout.qpos_indices[3:7])
        if len(qvel_ids) != 6 or qvel_ids != tuple(range(qvel_ids[0], qvel_ids[0] + 6)):
            return None
        if len(quat_ids) != 4 or quat_ids != tuple(range(quat_ids[0], quat_ids[0] + 4)):
            return None
        return qvel_ids, quat_ids

    def get_body_ids(self, names: "Sequence[str]") -> np.ndarray:
        ids: list[int] = []
        for name in names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in MuJoCo model")
            ids.append(bid)
        return np.array(ids, dtype=np.int32)

    def get_geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in MuJoCo model")
        return int(geom_id)

    def get_geom_size(self, name: str) -> np.ndarray:
        return np.asarray(self._model.geom_size[self.get_geom_id(name)], dtype=np.float64).copy()

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        """Create a reusable height-field scanner on the init/cold path.

        The MuJoCo backend supports ``alignment="world"`` and
        ``alignment="yaw"`` (mjbatch rejects anything else) and only
        ``output="height"``: the sampled world z of the hfield surface.
        """
        offsets_np = np.ascontiguousarray(np.asarray(offsets, dtype=np.float64))
        if offsets_np.ndim != 2 or offsets_np.shape[1] != 2:
            raise ValueError(f"offsets must have shape (num_points, 2), got {offsets_np.shape}")
        if output != "height":
            raise ValueError(f"MuJoCoBackend only supports output='height', got {output!r}")

        return _MuJoCoHeightScanner(
            backend=self,
            hfield_geom_id=int(hfield_geom_id),
            offsets=offsets_np,
            frame_body_id=int(frame_body_id),
            alignment=alignment,
        )

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        subtree_ids = {int(root_body_id)}
        changed = True
        while changed:
            changed = False
            for body_id in range(self._model.nbody):
                parent_id = int(self._model.body_parentid[body_id])
                if body_id not in subtree_ids and parent_id in subtree_ids:
                    subtree_ids.add(body_id)
                    changed = True
        return np.asarray(sorted(subtree_ids), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        return tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(self._model.ngeom)
        )

    def get_geom_body_ids(self) -> np.ndarray:
        return np.asarray(self._model.geom_bodyid, dtype=np.int32).copy()

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._model.geom_contype, dtype=np.int32).copy(),
            np.asarray(self._model.geom_conaffinity, dtype=np.int32).copy(),
        )

    def get_geom_friction(self) -> np.ndarray:
        return np.asarray(self._model.geom_friction, dtype=np.float64).copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._model.opt.gravity, dtype=np.float64).copy()

    def get_body_mass(self) -> np.ndarray:
        return np.asarray(self._model.body_mass, dtype=np.float64).copy()

    def get_body_ipos(self) -> np.ndarray:
        return np.asarray(self._model.body_ipos, dtype=np.float64).copy()

    def get_dof_armature(self) -> np.ndarray:
        return np.asarray(self._model.dof_armature, dtype=np.float64).copy()

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"Site '{name}' not found in MuJoCo model")
            ids.append(sid)
        return np.array(ids, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            indices.append(int(self._model.jnt_dofadr[jid]))
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        single_dof_types = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        for name in names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            if int(self._model.jnt_type[jid]) not in single_dof_types:
                raise ValueError(f"Joint '{name}' is not a single-DoF joint")
            indices.append(int(self._model.jnt_qposadr[jid]) - self._root_qpos_dim)
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names) + self._root_qpos_dim

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_vel_indices(names) + self._root_qvel_dim

    def get_joint_range(self) -> np.ndarray | None:
        jnt_range = self._model.jnt_range
        mask = self._model.jnt_type != int(mujoco.mjtJoint.mjJNT_FREE)
        return np.array(jnt_range[mask], dtype=self._np_dtype)

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        """Step all envs with a constant control (zero-order hold).

        ``ctrl`` and ``xfrc_applied`` persist across the ``nsteps`` substeps
        inside the batch (mj_step never clears them), which is numerically
        identical to broadcasting the same trajectory the whole way through.
        """
        if self._pre_step_control_fn is not None:
            return self._step_with_pre_step_control(ctrl, nsteps)

        t0 = time.perf_counter()
        self._ctrl_view[:] = ctrl
        # Obligation 1: xfrc_applied is batch-persistent state, so the staged
        # wrench is written absolutely every step — an idle step writes zeros.
        self._xfrc_view.reshape(self._num_envs, -1)[:] = self._pending_xfrc_applied
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._pool.step(nstep=nsteps)  # type: ignore[union-attr]
        physics_ms = (time.perf_counter() - t0) * 1000.0
        self._pending_xfrc_applied.fill(0.0)

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": 0.0,
            }
        }

    def _step_with_pre_step_control(
        self, ctrl: np.ndarray, nsteps: int
    ) -> dict[str, dict[str, float]]:
        # Single batch dispatch for all substeps (#1259 M1b): the upstream
        # pre-step control hook recomputes the Manager-Based action before
        # every substep through mjbatch's native per-substep callback.  Joint
        # state comes from the callback's state rows, and the tracked
        # world-frame sensor views are refreshed by the executor's split-
        # substep copyout at every substep boundary (including substep 0) at
        # memcpy cost.  xfrc_applied is recomposed and written absolutely
        # before every substep as the sum of the staged interval wrench and the
        # callback's dynamic wrench.
        set_ctrl_ms = 0.0
        refresh_cache_ms = 0.0
        layout = self._state_layout
        sensor_copyout = (
            self._tracked_sensor_copyout_range if self._refresh_pre_step_body_state else None
        )

        def _callback(k, state_view, ctrl_view, sensor_view=None) -> None:
            nonlocal set_ctrl_ms, refresh_cache_ms
            if k > 0:
                # k == 0 receives the state from before the call, which is
                # exactly what the bound views already hold.  At k > 0 the
                # callback receives full-width (num_sims, ...) rows even for
                # an ids subset, so the views can be refreshed wholesale.
                t0 = time.perf_counter()
                self._qpos_view[:] = state_view[:, layout.qpos]
                self._qvel_view[:] = state_view[:, layout.qvel]
                if self.nact:
                    self._act_view[:] = state_view[:, layout.act]
                refresh_cache_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            output = self._convert_pre_step_control(ctrl)
            ctrl_view[:] = output.ctrl
            # Absolute per-substep wrench write: fixed interval wrench + this
            # substep's dynamic callback wrench.  A substep whose callback
            # returns no wrench applies the fixed part alone, so dynamic
            # wrenches never leak across substeps of one call.
            xfrc_view = self._xfrc_view.reshape(self._num_envs, -1)
            xfrc_view[:] = self._pending_xfrc_applied
            if output.force is not None or output.torque is not None:
                for body_offset, body_id in enumerate(np.asarray(output.body_ids)):
                    if output.force is not None:
                        xfrc_view[:, self._resolve_push_body_force_slice(int(body_id))] += (
                            output.force[:, body_offset, :]
                        )
                    if output.torque is not None:
                        xfrc_view[
                            :, self._resolve_push_body_torque_slice(int(body_id))
                        ] += output.torque[:, body_offset, :]
            set_ctrl_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._pre_step_control_active = True
        try:
            self._pool.step(  # type: ignore[union-attr]
                nstep=nsteps,
                callback=_callback,
                substep_sensor_copyout=sensor_copyout,
            )
        finally:
            self._pre_step_control_active = False
        physics_ms = (time.perf_counter() - t0) * 1000.0 - set_ctrl_ms - refresh_cache_ms
        self._pending_xfrc_applied.fill(0.0)
        # The last substep's composed wrench stays in the persistent batch
        # channel; return it to zero so nothing leaks into the next call.
        self._xfrc_view.reshape(self._num_envs, -1)[:] = 0.0

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": refresh_cache_ms,
            }
        }


    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict | None:
        if randomization is not None:
            unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
                randomization.requested_terms()
            )
            if unsupported:
                raise NotImplementedError(
                    f"MuJoCo reset randomization does not support terms: {sorted(unsupported)}"
                )
        timing: dict[str, float] = {
            "set_state_mask_ms": 0.0,
            "set_state_data_slice_ms": 0.0,
            "set_state_data_reset_ms": 0.0,
            "set_state_clear_forces_ms": 0.0,
            "set_state_geom_overrides_ms": 0.0,
            "set_state_reset_rand_ms": 0.0,
            "set_state_set_dof_vel_ms": 0.0,
            "set_state_set_dof_pos_ms": 0.0,
            "set_state_actuator_ctrl_ms": 0.0,
            "set_state_forward_kinematic_ms": 0.0,
            "set_state_refresh_pose_cache_ms": 0.0,
            "set_state_invalidate_velocity_ms": 0.0,
            "set_state_qpos_convert_ms": 0.0,
            "set_state_pool_reset_ms": 0.0,
            "set_state_state_scatter_ms": 0.0,
            "set_state_reset_upload_ms": 0.0,
            "set_state_reset_forward_ms": 0.0,
            "set_state_host_cache_refresh_ms": 0.0,
            "set_state_internal_gap_ms": 0.0,
        }
        if len(env_indices) == 0:
            return {"timing": timing}

        outer_t0 = time.perf_counter()

        # Payload rows are indexed with the caller's env_indices (fancy
        # per-row writes); the batch itself requires sorted unique ids.
        env_indices = np.asarray(env_indices)
        ids = np.unique(np.asarray(env_indices, dtype=np.int32))

        t0 = time.perf_counter()
        # Order matters: reset() FIRST, then the per-field view writes, then
        # forward() applies them as pending writes on top of the reset state.
        # reset() itself only applies view writes it can detect by mirror
        # diff, so a re-upload of values identical to the current state (a
        # same-seed reset with no intervening step) would be silently dropped
        # and mj_resetData's qpos0 would win.  mj_resetData also zeroes
        # qacc_warmstart (and act/ctrl/xfrc/time) before the qpos/qvel
        # overlay — that is obligation 2, structural.
        self._pool.reset(ids)  # type: ignore[union-attr]
        if randomization is not None and not randomization.is_empty():
            self._apply_reset_randomization(randomization, env_indices)
            self._pool.set_const(ids)  # type: ignore[union-attr]
        t_q0 = time.perf_counter()
        self._qpos_view[env_indices] = qpos
        self._qvel_view[env_indices] = qvel
        timing["set_state_qpos_convert_ms"] = (time.perf_counter() - t_q0) * 1000.0
        self._pool.forward(ids)  # type: ignore[union-attr]
        timing["set_state_pool_reset_ms"] = (time.perf_counter() - t0) * 1000.0

        timing["set_state_state_scatter_ms"] = 0.0  # views are live; no scatter

        outer_total_ms = (time.perf_counter() - outer_t0) * 1000.0
        measured_ms = (
            timing["set_state_qpos_convert_ms"]
            + timing["set_state_pool_reset_ms"]
            + timing["set_state_state_scatter_ms"]
        )
        timing["set_state_internal_gap_ms"] = outer_total_ms - measured_ms
        return {"timing": timing}

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=_MUJOCO_RESET_TERMS,
            supports_fixed_variants=self._supports_fixed_variant_executor(),
            supported_fixed_variant_layouts=(
                frozenset(
                    {FixedVariantLayout.SAME_LAYOUT, FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT}
                )
                if self._supports_fixed_variant_executor()
                else frozenset()
            ),
            supports_per_env_playback=self._fixed_variant_build is not None,
            supports_interval_push=self._push_body_id >= 0,
            supports_interval_body_velocity_delta=(
                self._interval_root_velocity_qvel_ids is not None
            ),
            supports_interval_body_angular_velocity_delta=(
                self._interval_root_velocity_qvel_ids is not None
            ),
            supports_interval_body_force=True,
            supports_interval_body_torque=True,
            supported_interval_terms=frozenset(
                {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
                | ({INTERVAL_TERM_PUSH} if self._push_body_id >= 0 else set())
                | (
                    {
                        INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
                        INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
                    }
                    if self._interval_root_velocity_qvel_ids is not None
                    else set()
                )
            ),
        )

    @staticmethod
    def _supports_fixed_variant_executor() -> bool:
        return hasattr(mjbatch, "VariantPack") and hasattr(mjbatch.Batch, "from_variant_pack")

    @staticmethod
    def _reset_term_default_from_model(term: str, model: mujoco.MjModel) -> np.ndarray:
        if term == RESET_TERM_BASE_COM:
            return np.zeros((3,), dtype=np.float64)
        if term == RESET_TERM_BASE_MASS:
            return np.zeros((0,), dtype=np.float64)
        if term == RESET_TERM_GRAVITY:
            return np.asarray(model.opt.gravity, dtype=np.float64).copy()
        if term == RESET_TERM_BODY_IQUAT:
            return np.asarray(model.body_iquat, dtype=np.float64).copy()
        if term == RESET_TERM_BODY_INERTIA:
            return np.asarray(model.body_inertia, dtype=np.float64).copy()
        if term == RESET_TERM_BODY_IPOS:
            return np.asarray(model.body_ipos, dtype=np.float64).copy()
        if term == RESET_TERM_BODY_MASS:
            return np.asarray(model.body_mass, dtype=np.float64).copy()
        if term == RESET_TERM_DOF_ARMATURE:
            return np.asarray(model.dof_armature, dtype=np.float64).copy()
        if term == RESET_TERM_DOF_DAMPING:
            return np.asarray(model.dof_damping, dtype=np.float64).copy()
        if term == RESET_TERM_DOF_FRICTIONLOSS:
            return np.asarray(model.dof_frictionloss, dtype=np.float64).copy()
        if term == RESET_TERM_GEOM_FRICTION:
            return np.asarray(model.geom_friction, dtype=np.float64).copy()
        if term == RESET_TERM_GEOM_SIZE:
            return np.asarray(model.geom_size, dtype=np.float64).copy()
        if term == RESET_TERM_GEOM_SOLIMP:
            return np.asarray(model.geom_solimp, dtype=np.float64).copy()
        if term == RESET_TERM_GEOM_SOLREF:
            return np.asarray(model.geom_solref, dtype=np.float64).copy()
        if term == RESET_TERM_KP:
            return np.asarray(model.actuator_gainprm[:, 0], dtype=np.float64).copy()
        if term == RESET_TERM_KD:
            return np.asarray(-model.actuator_biasprm[:, 2], dtype=np.float64).copy()
        raise AssertionError(f"missing default mapping for reset term {term!r}")

    def get_reset_term_default(self, term: str) -> np.ndarray:
        _validate_reset_term(term)
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(f"MuJoCoBackend does not support reset term '{term}'")
        if self._fixed_variant_build is None:
            value = self._reset_term_default_from_model(term, self._model)
        else:
            build = self._fixed_variant_build
            if term in {
                RESET_TERM_GEOM_FRICTION,
                RESET_TERM_GEOM_SIZE,
                RESET_TERM_GEOM_SOLIMP,
                RESET_TERM_GEOM_SOLREF,
            }:
                variant_values = self._canonical_variant_geom_default(term, build)
            else:
                variant_values = np.stack(build.default_tables[term])
            value = np.asarray(variant_values[np.asarray(build.plan.assignment)])
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _canonical_variant_geom_default(self, term: str, build: _FixedVariantBuild) -> np.ndarray:
        canonical_value = self._reset_term_default_from_model(term, self._model)
        if term == RESET_TERM_GEOM_SIZE:
            return np.asarray(build.pack.fields[term], dtype=np.float64).copy()

        values = np.broadcast_to(
            canonical_value, (len(build.default_tables[term]), *canonical_value.shape)
        ).copy()
        canonical_names = _model_names(self._model, "geom", self._model.ngeom)
        canonical_ids = {name: geom_id for geom_id, name in enumerate(canonical_names)}
        for variant, (names, source_values) in enumerate(
            zip(build.geom_names, build.default_tables[term], strict=True)
        ):
            for source_id, name in enumerate(names):
                values[variant, canonical_ids[name]] = source_values[source_id]
        return values

    def materialize(self) -> None:
        if self._pool is not None:
            raise RuntimeError("MuJoCo backend pool is already materialized")
        self._pool = self._build_pool()
        self._bind_views(self._pool)

    _interval_term_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        self._reject_wrench_write_inside_pre_step_control("apply_interval_randomization")
        # A non-empty plan starts from cleared external wrenches; the force and
        # torque handlers then accumulate into ``_pending_xfrc_applied``.
        self._pending_xfrc_applied.fill(0.0)
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        # Built lazily once; the table only binds methods, so it is stable for
        # the backend lifetime and is never rebuilt per plan.
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
                    lambda op: self._apply_body_velocity_delta(
                        require_op_body_ids(op), op.payload, None
                    )
                ),
                INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA: (
                    lambda op: self._apply_body_velocity_delta(
                        require_op_body_ids(op), None, op.payload
                    )
                ),
            }
        return self._interval_term_handler_cache

    def _apply_body_torque(self, body_ids: np.ndarray, torque: np.ndarray) -> None:
        """Accumulate a torque-only wrench through the shared xfrc staging.

        ``apply_body_force`` accumulates force and torque channels
        independently, so a zero force keeps a plan carrying separate force
        and torque ops numerically identical to the legacy single-call form.
        """
        zero_force = np.zeros((self._num_envs, len(body_ids), 3), dtype=np.float64)
        self.apply_body_force(body_ids, zero_force, torque=torque)

    def _validate_body_velocity_delta(
        self,
        velocity_delta: np.ndarray | None,
        *,
        label: str,
    ) -> np.ndarray | None:
        if velocity_delta is None:
            return None
        if not isinstance(velocity_delta, np.ndarray):
            raise TypeError(
                f"MuJoCo interval body {label} velocity perturbation must be an np.ndarray, "
                f"got {type(velocity_delta).__name__}"
            )
        expected_shape = (self._num_envs, 1, 3)
        if velocity_delta.shape != expected_shape:
            raise ValueError(
                f"MuJoCo interval body {label} velocity perturbation has shape "
                f"{velocity_delta.shape}; expected {expected_shape}"
            )
        if not np.issubdtype(velocity_delta.dtype, np.floating):
            raise TypeError(
                f"MuJoCo interval body {label} velocity perturbation must have floating dtype, "
                f"got {velocity_delta.dtype}"
            )
        if not np.isfinite(velocity_delta).all():
            raise ValueError(
                f"MuJoCo interval body {label} velocity perturbation contains NaN or Inf"
            )
        return velocity_delta

    def _apply_body_velocity_delta(
        self,
        body_ids: np.ndarray,
        linear_delta: np.ndarray | None,
        angular_delta: np.ndarray | None,
    ) -> None:
        """Apply a row-selective world-frame velocity kick to the configured free root.

        Linear deltas are world-frame (matching the free-root qvel layout).
        Angular deltas are sampled in the world frame (community push contract)
        and converted into the root-body frame expected by the qvel columns
        using the root orientation already present in the physics state.
        """
        resolved = self._interval_root_velocity_qvel_ids
        if resolved is None:
            raise NotImplementedError(
                "MuJoCo interval body velocity perturbation requires base_name to identify "
                "a body with exactly one free joint"
            )
        qvel_ids, quat_qpos_ids = resolved

        raw_body_ids = np.asarray(body_ids)
        if (
            raw_body_ids.ndim != 1
            or not np.issubdtype(raw_body_ids.dtype, np.integer)
            or np.issubdtype(raw_body_ids.dtype, np.bool_)
        ):
            raise TypeError(
                "MuJoCo interval body velocity perturbation body_ids must be a 1-D "
                f"integer array, got shape={raw_body_ids.shape}, dtype={raw_body_ids.dtype}"
            )
        resolved_body_ids = np.asarray(raw_body_ids, dtype=np.int32)
        expected_body_ids = np.asarray([self._base_body_id], dtype=np.int32)
        if not np.array_equal(resolved_body_ids, expected_body_ids):
            raise NotImplementedError(
                "MuJoCo interval body velocity perturbation only supports the configured "
                f"free root body '{self._base_name}' (id={self._base_body_id}); "
                f"received body_ids={resolved_body_ids.tolist()}"
            )

        linear_delta = self._validate_body_velocity_delta(linear_delta, label="linear")
        angular_delta = self._validate_body_velocity_delta(angular_delta, label="angular")
        if self._pool is None:
            raise RuntimeError(
                "MuJoCo interval body velocity perturbation requires a materialized backend"
            )

        active_mask = np.zeros(self._num_envs, dtype=np.bool_)
        if linear_delta is not None:
            active_mask |= np.any(linear_delta[:, 0, :] != 0.0, axis=1)
        if angular_delta is not None:
            active_mask |= np.any(angular_delta[:, 0, :] != 0.0, axis=1)
        active_rows = np.flatnonzero(active_mask).astype(np.int32, copy=False)
        if active_rows.size == 0:
            return

        if linear_delta is not None:
            linear_columns = np.asarray(qvel_ids[:3], dtype=np.intp)
            self._qvel_view[active_rows[:, None], linear_columns] += linear_delta[active_rows, 0, :]
        if angular_delta is not None:
            quat_columns = np.asarray(quat_qpos_ids, dtype=np.intp)
            angular_columns = np.asarray(qvel_ids[3:6], dtype=np.intp)
            self._qvel_view[active_rows[:, None], angular_columns] += np_quat_apply_inverse_batched(
                self._qpos_view[active_rows[:, None], quat_columns],
                angular_delta[active_rows, 0, :],
            )
        # Obligation 2, explicit on the re-upload path: the kick invalidates
        # the solver's warmstart guess for the touched rows.
        self._warm_view[active_rows] = 0.0
        # active_rows comes from flatnonzero: sorted and unique, as required.
        self._pool.forward(ids=active_rows)

    def push_robots(self, force_range: Sequence[float] | np.ndarray) -> None:
        self._reject_wrench_write_inside_pre_step_control("push_robots")
        self._pending_xfrc_applied.fill(0.0)
        self._pending_xfrc_applied[:, self._push_body_force_slice] = self._sample_push_force(
            force_range
        )

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Accumulate one external world-frame wrench per target body.

        Args:
            body_ids: Body ids to perturb.
            force: Force tensor with shape ``(num_envs, len(body_ids), 3)``.
            torque: Optional world-frame torque tensor with the same shape,
                staged in the ``xfrc_applied`` torque channel.

        Returns:
            None. The wrench is staged in ``xfrc_applied`` for the next step.
        """
        self._reject_wrench_write_inside_pre_step_control("apply_body_force")
        body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_np = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        torque_np = None
        if torque is not None:
            torque_np = np.asarray(torque, dtype=np.float64)
            if torque_np.shape != expected_shape:
                raise ValueError(
                    f"body torque must have shape {expected_shape}, got {torque_np.shape}"
                )
        for body_offset, body_id in enumerate(body_ids_np):
            self._pending_xfrc_applied[:, self._resolve_push_body_force_slice(int(body_id))] += (
                force_np[:, body_offset, :]
            )
            if torque_np is not None:
                self._pending_xfrc_applied[
                    :, self._resolve_push_body_torque_slice(int(body_id))
                ] += torque_np[:, body_offset, :]

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_physics_state_playback=True,
            supports_debug_overlay=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "record" if mode == "auto" else mode
        if effective_mode == "none":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if effective_mode == "interactive":
            raise NotImplementedError("MuJoCo playback does not support interactive rendering.")
        assert effective_mode == "record"
        if play_steps is None:
            raise ValueError("MuJoCo record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("MuJoCo record playback requires an output video path.")
        return BackendPlayRenderPlan(
            mode=effective_mode,
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize,
        step,
        num_steps: int | None,
        output_video: str | os.PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter=None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: DebugOverlayGetter | None = None,
        on_frame=None,
    ) -> str | None:
        del render_offset_mode
        camera = CameraCfg.from_kwargs(camera_kwargs)
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        return run_mujoco_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=should_run_headless,
            record_video=should_record_video,
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera,
            debug_overlay_getter=debug_overlay_getter,
            on_frame=on_frame,
        )

    # ------------------------------------------------------------------ #
    # Base kinematics                                                    #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        return self._base_pos_view

    def get_base_quat(self) -> np.ndarray:
        return self._base_quat_view

    def get_base_lin_vel(self) -> np.ndarray:
        return self._base_lin_vel_view

    def get_base_ang_vel(self) -> np.ndarray:
        return self._base_ang_vel_view

    # ------------------------------------------------------------------ #
    # DOF state                                                          #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        return self._dof_pos_view

    def get_dof_vel(self) -> np.ndarray:
        return self._dof_vel_view

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                      #
    # ------------------------------------------------------------------ #

    def _get_mapped_indices(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_id_to_tracked_idx[body_ids]  # type: ignore[no-any-return]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_pos_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_quat_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        mapped = self._get_mapped_indices(body_ids)
        return self._tracked_pos_w_all[rows[:, None], mapped], self._tracked_quat_w_all[
            rows[:, None], mapped
        ]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_linvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_angvel_w_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_lin_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        return self._tracked_linvel_w_all[rows[:, None], self._get_mapped_indices(body_ids)]  # type: ignore[no-any-return]

    def get_body_ang_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        return self._tracked_angvel_w_all[rows[:, None], self._get_mapped_indices(body_ids)]  # type: ignore[no-any-return]

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mapped = self._get_mapped_indices(body_ids)
        return (
            self._tracked_pos_w_all[:, mapped, :],
            self._tracked_quat_w_all[:, mapped, :],
            self._tracked_linvel_w_all[:, mapped, :],
            self._tracked_angvel_w_all[:, mapped, :],
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mapped = self._get_mapped_indices(body_ids)
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

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_pos_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._tracked_quat_b_all[:, self._get_mapped_indices(body_ids), :]  # type: ignore[no-any-return]

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Analytical per the SimBackend contract: world-frame velocity rotated
        # into each body's own frame. MuJoCo framelinvel sensors with a baselink
        # reference report relative motion and degenerate to zero for the root.
        idx = self._get_mapped_indices(body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, idx, :], self._tracked_linvel_w_all[:, idx, :]
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        idx = self._get_mapped_indices(body_ids)
        return np_quat_apply_inverse_batched(
            self._tracked_quat_w_all[:, idx, :], self._tracked_angvel_w_all[:, idx, :]
        )

    # ------------------------------------------------------------------ #
    # Sensors                                                            #
    # ------------------------------------------------------------------ #

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._sensor_views[name]

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        return self._sensor_views[name][np.asarray(env_ids, dtype=np.intp)]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        sensor_names = tuple(names)
        if not sensor_names:
            return np.empty((self._num_envs, 0), dtype=self._np_dtype)
        values = [self._sensor_views[name].reshape(self._num_envs, -1) for name in sensor_names]
        return np.concatenate(values, axis=1)

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Resolve MuJoCo's sensor slices at read time, not bind time.

        ``_bind_views`` re-points ``_sensor_data`` (and rebuilds
        ``_sensor_views``) at ``materialize()``; manager terms bind on the cold
        path before the pool exists and must follow the re-pointed storage
        instead of capturing the pre-materialize host arrays.
        """
        sensor_names = tuple(names)

        def read() -> np.ndarray:
            views = tuple(
                self._sensor_views[name].reshape(self._num_envs, -1) for name in sensor_names
            )
            return np.concatenate(views, axis=1)

        return read

    def get_site_jacobian_w(
        self,
        site_id: int,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return batched Jacobians with shape ``(num_envs, 3, len(dof_indices))``.

        This uses mjbatch's native live-state ``jac_site`` op, so it does not
        allocate one ``MjData`` per env.  Query ops skip the bound-field
        CopyOut, so the bound views are untouched by the call.
        """
        site_id_int = int(site_id)
        if site_id_int < 0 or site_id_int >= int(self._model.nsite):
            raise ValueError(
                f"Invalid site_id {site_id_int}; expected 0 <= site_id < {self._model.nsite}"
            )
        dof_indices = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
        if np.any(dof_indices < 0) or np.any(dof_indices >= self.nv):
            raise ValueError(f"dof_indices must be within [0, {self.nv})")
        pool = self._pool
        if pool is None:
            raise RuntimeError("MuJoCo site Jacobians require a materialized backend")
        jacp = np.zeros((self._num_envs, 3, self.nv))
        jacr = np.zeros_like(jacp)
        # Id-based raw binding: the backend holds site ids and sites may be
        # unnamed, which the name-based Python wrapper cannot resolve.
        _RawMjBatch.jac_site(pool, site_id_int, jacp, jacr, None)
        return (
            jacp[:, :, dof_indices].astype(self._np_dtype),
            jacr[:, :, dof_indices].astype(self._np_dtype),
        )

    # ------------------------------------------------------------------ #
    # Mujoco-specific                                                    #
    # ------------------------------------------------------------------ #

    def get_physics_state(self) -> np.ndarray:
        """Assemble the contract ``[time, qpos, qvel]`` snapshot rows."""
        out = np.empty((self._num_envs, 1 + self.nq + self.nv), dtype=self._np_dtype)
        out[:, 0] = self._time_view
        out[:, 1 : 1 + self.nq] = self._qpos_view
        out[:, 1 + self.nq :] = self._qvel_view
        return out

    def get_playback_model(self, env_index: int | None = None):
        """Return the MuJoCo model used by playback.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The renderable backend model. With a fixed variant plan, this is
            that world's independently compiled visual model; runtime
            reset-randomization field snapshots are not copied into the
            playback oracle.
        """
        if env_index is None:
            if self._fixed_variant_build is not None:
                raise ValueError("fixed-variant playback requires an explicit env_index")
            return self._get_static_playback_model()
        idx = int(env_index)
        if idx < 0 or idx >= self._num_envs:
            raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        if self._fixed_variant_build is None:
            return self._get_static_playback_model()
        variant = int(self._fixed_variant_build.plan.assignment[env_index])
        descriptor = self._fixed_variant_build.plan.variants[variant]
        return self._compile_playback_model(self._variant_source(descriptor))

    def _get_static_playback_model(self) -> mujoco.MjModel:
        if self._static_playback_model is None:
            if self.scene_visual_model_file is None:
                return self._model
            self._static_playback_model = self._compile_playback_model(self.scene_visual_model_file)
        return self._static_playback_model

    def _compile_playback_model(self, model_file: str) -> mujoco.MjModel:
        """Cold-compile a detached visual oracle without discarding visual assets."""
        spec = _configured_variant_spec(
            mujoco.MjSpec.from_file(model_file),
            sim_dt=self._sim_dt,
            iterations=self._iterations,
            position_actuator_gains=self._position_actuator_gains,
        )
        return spec.compile()

    def _coerce_reset_field(
        self,
        value: np.ndarray,
        *,
        name: str,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        arr = cast(np.ndarray, np.asarray(value, dtype=np.float64))
        flat_tail = int(np.prod(shaped_tail))
        flat_shape = (num_reset, flat_tail)
        shaped = (num_reset, *shaped_tail)
        if arr.shape == flat_shape:
            return cast(np.ndarray, arr.copy())
        if arr.shape == shaped:
            return cast(np.ndarray, arr.reshape(num_reset, flat_tail).copy())
        raise ValueError(f"{name} must have shape {flat_shape} or {shaped}, got {arr.shape}")

    def _apply_reset_randomization(
        self,
        randomization: ResetRandomizationPayload | None,
        env_indices: np.ndarray,
    ) -> None:
        """Write a reset randomization payload into per-sim expanded model fields.

        Fields are expanded lazily on first use (cold path: the first
        expansion allocates one model copy per worker thread, so DR tasks pay
        ``nthread x model`` memory instead of ``num_envs x model``).  Values
        persist per sim until rewritten, matching the old per-env model patch
        semantics; ``set_const`` refreshes the derived constants afterwards.
        """
        if randomization is None or randomization.is_empty():
            return
        if (
            randomization.base_mass_delta is not None or randomization.base_com_offset is not None
        ) and self._base_body_id < 0:
            raise ValueError(f"Body '{self._base_name}' not found in MuJoCo model")

        pool = self._pool
        if pool is None:
            raise RuntimeError("MuJoCo reset randomization requires a materialized backend")
        num_reset = len(env_indices)
        nbody = self._model.nbody

        body_mass = None
        if randomization.body_mass is not None:
            body_mass = self._coerce_reset_field(
                randomization.body_mass,
                name="body_mass",
                num_reset=num_reset,
                shaped_tail=(nbody,),
            )
        if randomization.base_mass_delta is not None:
            if body_mass is None:
                body_mass = np.array(self._base_body_mass[env_indices], copy=True)
            body_mass[:, self._base_body_id] += np.asarray(randomization.base_mass_delta)
        if body_mass is not None:
            pool.expand("body_mass")[env_indices] = body_mass

        body_ipos = None
        if randomization.body_ipos is not None:
            body_ipos = self._coerce_reset_field(
                randomization.body_ipos,
                name="body_ipos",
                num_reset=num_reset,
                shaped_tail=(nbody, 3),
            )
        if randomization.base_com_offset is not None:
            if body_ipos is None:
                body_ipos = np.array(self._base_body_ipos[env_indices], copy=True)
            body_ipos[:, self._base_body_id, :] += np.asarray(randomization.base_com_offset)
        if body_ipos is not None:
            pool.expand("body_ipos")[env_indices] = body_ipos.reshape(num_reset, nbody, 3)

        if randomization.gravity is not None:
            gravity = self._coerce_reset_field(
                randomization.gravity,
                name="gravity",
                num_reset=num_reset,
                shaped_tail=(3,),
            )
            pool.expand("gravity")[env_indices] = gravity

        if randomization.body_iquat is not None:
            body_iquat = self._coerce_reset_field(
                randomization.body_iquat,
                name="body_iquat",
                num_reset=num_reset,
                shaped_tail=(nbody, 4),
            ).reshape(num_reset, nbody, 4)
            pool.expand("body_iquat")[env_indices] = body_iquat

        if randomization.body_inertia is not None:
            body_inertia = self._coerce_reset_field(
                randomization.body_inertia,
                name="body_inertia",
                num_reset=num_reset,
                shaped_tail=(nbody, 3),
            ).reshape(num_reset, nbody, 3)
            pool.expand("body_inertia")[env_indices] = body_inertia

        if randomization.geom_friction is not None:
            geom_friction = self._coerce_reset_field(
                randomization.geom_friction,
                name="geom_friction",
                num_reset=num_reset,
                shaped_tail=(self._model.ngeom, 3),
            ).reshape(num_reset, self._model.ngeom, 3)
            pool.expand("geom_friction")[env_indices] = geom_friction

        if randomization.geom_size is not None:
            geom_size = self._coerce_reset_field(
                randomization.geom_size,
                name="geom_size",
                num_reset=num_reset,
                shaped_tail=(self._model.ngeom, 3),
            ).reshape(num_reset, self._model.ngeom, 3)
            pool.expand("geom_size")[env_indices] = geom_size

        if randomization.geom_solref is not None:
            geom_solref = self._coerce_reset_field(
                randomization.geom_solref,
                name="geom_solref",
                num_reset=num_reset,
                shaped_tail=(self._model.ngeom, mujoco.mjNREF),
            ).reshape(num_reset, self._model.ngeom, mujoco.mjNREF)
            pool.expand("geom_solref")[env_indices] = geom_solref

        if randomization.geom_solimp is not None:
            geom_solimp = self._coerce_reset_field(
                randomization.geom_solimp,
                name="geom_solimp",
                num_reset=num_reset,
                shaped_tail=(self._model.ngeom, mujoco.mjNIMP),
            ).reshape(num_reset, self._model.ngeom, mujoco.mjNIMP)
            pool.expand("geom_solimp")[env_indices] = geom_solimp

        if randomization.dof_armature is not None:
            dof_armature = self._coerce_reset_field(
                randomization.dof_armature,
                name="dof_armature",
                num_reset=num_reset,
                shaped_tail=(self.nv,),
            )
            pool.expand("dof_armature")[env_indices] = dof_armature

        if randomization.dof_damping is not None:
            dof_damping = self._coerce_reset_field(
                randomization.dof_damping,
                name="dof_damping",
                num_reset=num_reset,
                shaped_tail=(self.nv,),
            )
            pool.expand("dof_damping")[env_indices] = dof_damping

        if randomization.dof_frictionloss is not None:
            dof_frictionloss = self._coerce_reset_field(
                randomization.dof_frictionloss,
                name="dof_frictionloss",
                num_reset=num_reset,
                shaped_tail=(self.nv,),
            )
            pool.expand("dof_frictionloss")[env_indices] = dof_frictionloss

        if randomization.kp is not None:
            kp = self._coerce_reset_field(
                randomization.kp,
                name="kp",
                num_reset=num_reset,
                shaped_tail=(self._model.nu,),
            )
            # Mirrors _apply_position_actuator_gains_to_mj_model.
            gain = pool.expand("actuator_gainprm")
            gain[env_indices, :, 0] = kp
            bias = pool.expand("actuator_biasprm")
            bias[env_indices, :, 1] = -kp

        if randomization.kd is not None:
            kd = self._coerce_reset_field(
                randomization.kd,
                name="kd",
                num_reset=num_reset,
                shaped_tail=(self._model.nu,),
            )
            bias = pool.expand("actuator_biasprm")
            bias[env_indices, :, 2] = -kd

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays read from the current model state."""
        kp = np.asarray(self._model.actuator_gainprm[:, 0], dtype=np.float64).copy()
        kd = np.asarray(-self._model.actuator_biasprm[:, 2], dtype=np.float64).copy()
        return kp, kd

    def _apply_position_actuator_gains_to_model(
        self,
        model,
        *,
        kp: float | np.ndarray,
        kd: float | np.ndarray,
        actuator_ids=slice(None),
    ) -> None:
        _apply_position_actuator_gains_to_mj_model(
            model,
            kp=kp,
            kd=kd,
            actuator_ids=actuator_ids,
        )
