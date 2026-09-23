"""UniSim adapter for the ``drake-uni`` batch runtime.

UniLab owns task logic, reset sampling, named sensor views, and training flow.
DrakeUni owns Drake model construction, batched stepping, and raw sensor
evaluation. This module translates the ``SimBackend`` contract into DrakeUni
runtime calls and keeps UniLab's cached state/sensor views synchronized.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.util import find_spec
from multiprocessing import cpu_count
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    CameraCfg,
    DebugOverlayGetter,
    PhysicsStateLayout,
    SimBackend,
    normalize_play_render_mode,
)
from unisim.backend.drake.playback import run_drake_playback
from unisim.backend.drake.properties import (
    scan_expected_native_properties,
    validate_native_model_properties,
)
from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ResetRandomizationPayload,
    require_op_body_ids,
)
from unisim.entities import SceneResetRequest
from unisim.entity_state import (
    entity_state_snapshot,
    prepare_scene_reset,
    selected_state_rows,
)
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_compiler import compile_portable_scene
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


# ``drake-uni`` availability globals. These are cheap import-time probes so callers
# can ask whether Drake support exists without constructing a backend.
def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


DRAKE_AVAILABLE = _module_available("drake_uni")
DRAKE_IMPORT_ERROR: ImportError | None = None
DRAKE_BATCH_AVAILABLE = _module_available("drake_uni")
DRAKE_BATCH_IMPORT_ERROR: ImportError | None = None
DrakeBatchConfig = None
create_drake_runtime = None

_DRAKE_UNI_SYMBOLS_LOADED = False


# Lazy import and pydrake guard helpers.
def _pydrake_loaded() -> bool:
    # DrakeUni's batch extension owns Drake symbol loading; mixing it with an
    # already-imported pydrake module has produced unstable process state.
    return any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)


def _load_drake_uni_symbols() -> None:
    """Load ``drake-uni`` only when a Drake backend is actually constructed."""

    global DRAKE_AVAILABLE
    global DRAKE_BATCH_AVAILABLE
    global DRAKE_BATCH_IMPORT_ERROR
    global DrakeBatchConfig
    global create_drake_runtime
    global _DRAKE_UNI_SYMBOLS_LOADED

    if _DRAKE_UNI_SYMBOLS_LOADED:
        return
    try:
        from drake_uni.runtime import DrakeBatchConfig as ImportedDrakeBatchConfig
        from drake_uni.runtime import batch_diagnostics
        from drake_uni.runtime import create_runtime as imported_create_runtime
    except ImportError as exc:  # pragma: no cover - optional local package.
        DRAKE_AVAILABLE = False
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = exc
        raise ImportError("DrakeUni batch runtime is not installed.") from exc

    diagnostics = batch_diagnostics()
    if not diagnostics.batch_available:
        detail = diagnostics.batch_import_error
        import_error = ImportError(detail or "DrakeEnvPool batch extension has not been built.")
        DRAKE_AVAILABLE = False
        DRAKE_BATCH_AVAILABLE = False
        DRAKE_BATCH_IMPORT_ERROR = import_error
        raise ImportError("DrakeEnvPool batch extension has not been built.") from import_error

    DrakeBatchConfig = ImportedDrakeBatchConfig
    create_drake_runtime = imported_create_runtime
    DRAKE_AVAILABLE = True
    DRAKE_BATCH_AVAILABLE = True
    DRAKE_BATCH_IMPORT_ERROR = None
    _DRAKE_UNI_SYMBOLS_LOADED = True


def ensure_drake_batch_available() -> tuple[bool, ImportError | None]:
    """Report whether the DrakeUni batch extension can be used."""

    try:
        _load_drake_uni_symbols()
    except ImportError as exc:
        return False, exc
    return True, None


# Floating-base compact state starts with xyz + quaternion in qpos and
# 3 linear + 3 angular components in qvel. UniLab usually wants only the
# actuated joint slices behind those root coordinates.
ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6


# Small helper types.
@dataclass(frozen=True)
class _DrakeUniModelView:
    """Read-only model-shape facade for UniLab's ``backend.model`` API.

    DrakeUni exposes model dimensions through ``model_info``. UniLab's backend
    contract expects a model-like object with dimension methods, so this facade
    carries those shape queries without exposing Drake internals.
    """

    nq: int
    nv: int
    nu: int

    def num_actuators(self) -> int:
        return self.nu


@dataclass(frozen=True)
class _DrakeRuntimeGroup:
    """One immutable variant runtime and its public environment rows."""

    variant: int
    public_ids: tuple[int, ...]
    runtime: Any

    @property
    def count(self) -> int:
        return len(self.public_ids)


# Path and thread helpers.
def _resolve_batch_nthread(num_envs: int, requested: int) -> int:
    """Resolve a worker count without creating idle workers above num_envs."""

    env_count = max(1, int(num_envs))
    requested_count = int(requested)
    if requested_count > 0:
        return min(env_count, requested_count)
    return min(env_count, max(1, cpu_count() * 2))


def _resolve_scene_path(scene: SceneCfg) -> Path:
    """Convert UniLab's scene pointer into an absolute model path."""

    if not scene.model_file:
        raise ValueError("DrakeBackend requires SceneCfg.model_file")
    path = Path(scene.model_file)
    return path if path.is_absolute() else Path.cwd() / path


class DrakeBackend(SimBackend):
    """UniLab ``SimBackend`` implementation backed by DrakeUni batch runtime.

    The backend keeps the public UniLab API stable while delegating model
    construction, integration, and raw sensor evaluation to DrakeUni.
    """

    backend_type = "drake"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        drake_backend_mode: str = "batch",
        nthread: int = 0,
    ) -> None:
        if scene.entity_assets:
            if any(entity.root_mode == "kinematic" for entity in scene.entity_assets):
                raise NotImplementedError(
                    "Drake portable entity scenes do not support kinematic mirrors yet"
                )
            binding = scene.entity_variant
            if binding is not None and (
                binding.plan.layout is not FixedVariantLayout.SAME_LAYOUT
            ):
                raise NotImplementedError(
                    "Drake portable entity scenes support only same_layout fixed variants"
                )
        elif scene.fixed_variant_plan is not None:
            raise NotImplementedError(
                "Drake fixed variants require explicit entity_assets"
            )
        require_scene_composition_support(scene, "drake")
        # Validate the backend mode at construction so Hydra/config mistakes
        # fail at the backend boundary.
        mode = str(drake_backend_mode or "batch").strip().lower()
        if mode != "batch":
            raise ValueError(
                "UniLab DrakeBackend requires drake_backend_mode='batch'. "
                f"Got {drake_backend_mode!r}."
            )
        if _pydrake_loaded():
            raise ImportError(
                "Drake batch backend cannot be loaded after pydrake has already "
                "been imported in this process. Start a fresh process before "
                "constructing DrakeBackend."
            )
        if int(num_envs) < 1:
            raise ValueError(f"DrakeUni batch backend requires num_envs >= 1, got {num_envs}")
        self._entity_layout: CompiledSceneLayout | None = None
        self._composed_scene = None
        self._entity_faulted = False
        self._entity_closed = False
        self._entity_root_ids: tuple[int, ...] = ()
        self._entity_joint_qpos_indices = np.empty(0, dtype=np.intp)
        self._entity_joint_qvel_indices = np.empty(0, dtype=np.intp)
        self._entity_joint_dof_pos_indices: dict[str, int] = {}
        self._entity_joint_dof_vel_indices: dict[str, int] = {}
        self._entity_default_qpos = np.empty(0, dtype=np.float64)
        self._entity_default_qvel = np.empty(0, dtype=np.float64)
        self._entity_default_roots = np.empty(0, dtype=np.float64)
        self._runtime_groups: tuple[_DrakeRuntimeGroup, ...] = ()
        self._variant_assignment: np.ndarray | None = None
        self._variant_model_files: tuple[str, ...] = ()
        self._fixed_variant_realization = False
        self._runtime: Any | None = None
        _load_drake_uni_symbols()
        if DrakeBatchConfig is None or create_drake_runtime is None:
            detail = DRAKE_BATCH_IMPORT_ERROR
            message = "DrakeUni runtime is not available."
            if detail is not None:
                message = f"{message} Import error: {detail}"
            raise ImportError(message) from detail

        self._pre_step_control_fn = None
        self._scene_cleanup_handle = None
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)

        composed = None
        if scene.entity_assets:
            composed = compile_portable_scene(scene, self._num_envs, self._sim_dt)
            scene = replace(
                scene,
                model_file=composed.model_file,
                entity_assets=(),
                entity_variant=None,
                fragment_files=[],
            )
        self._composed_scene = composed
        try:
            if composed is None:
                assignment = np.zeros((self._num_envs,), dtype=np.int32)
                model_files: tuple[str, ...] = (str(_resolve_scene_path(scene)),)
            else:
                plan = composed.variant_plan
                if plan is None:
                    assignment = np.zeros((self._num_envs,), dtype=np.int32)
                    model_files = (composed.model_file,)
                else:
                    assignment = np.asarray(plan.assignment, dtype=np.int32).reshape(-1)
                    model_files = tuple(
                        str(variant.model_file) for variant in plan.variants
                    )
                    if assignment.size != self._num_envs:
                        raise ValueError(
                            "Drake fixed-variant assignment length "
                            f"{assignment.size} does not match num_envs {self._num_envs}"
                        )
                    if np.any(assignment < 0) or np.any(assignment >= len(model_files)):
                        raise ValueError(
                            "Drake fixed-variant assignment refers to an absent source"
                        )
            groups: list[_DrakeRuntimeGroup] = []
            for variant in np.unique(assignment).tolist():
                public_ids = tuple(int(index) for index in np.flatnonzero(assignment == variant))
                # DrakeUni receives only generic batch facts. Task concepts such as
                # base bodies, push targets, and observation semantics stay in UniLab.
                config = DrakeBatchConfig(
                    model_file=model_files[int(variant)],
                    num_envs=len(public_ids),
                    sim_dt=self._sim_dt,
                    nthread=int(nthread),
                )
                group = _DrakeRuntimeGroup(int(variant), public_ids, create_drake_runtime(config))
                groups.append(group)
                # Keep successfully created runtimes reachable so a later group failure
                # cannot leak Drake resources during constructor cleanup.
                self._runtime_groups = tuple(groups)
            self._runtime_groups = tuple(groups)
            self._runtime = self._runtime_group(int(assignment[0])).runtime
            self._variant_assignment = (
                assignment
                if composed is not None and composed.variant_plan is not None
                else None
            )
            self._variant_model_files = model_files
            self._fixed_variant_realization = composed is not None and (
                composed.variant_plan is not None
            )
            self._scene_model_file = model_files[int(assignment[0])]
        except BaseException:
            self.close()
            raise
        try:
            default_group = self._runtime_group(int(assignment[0]))
            model_info = default_group.runtime.model_info()
            # Cache static model metadata once and expose copies through the
            # UniLab backend contract.
            self._home_qpos_mujoco = model_info.home_qpos.copy()
            self._home_qvel_mujoco = model_info.home_qvel.copy()
            self._ctrl_limits = model_info.ctrl_limits.copy()
            self._joint_ranges = model_info.joint_ranges.copy()
            self._actuator_stiffness = model_info.actuator_stiffness.copy()
            self._actuator_damping = model_info.actuator_damping.copy()
            self._actuator_qpos_adr = model_info.actuator_qpos_adr.astype(np.intp, copy=True)
            self._actuator_qvel_adr = model_info.actuator_qvel_adr.astype(np.intp, copy=True)
            raw_actuator_names = getattr(model_info, "actuator_names", None)
            self._actuator_names = (
                None
                if raw_actuator_names is None
                else tuple(str(name) for name in raw_actuator_names)
            )
            self._sensor_names = tuple(model_info.sensor_names)
            self._sensor_adr = model_info.sensor_adr.copy()
            self._sensor_dim = model_info.sensor_dim.copy()
            self._site_name_to_id = {
                str(name): index
                for index, name in enumerate(getattr(model_info, "site_names", ()))
            }
            self._joint_qpos_adr_by_name = {
                str(name): int(adr)
                for name, adr in zip(
                    getattr(model_info, "joint_names", ()),
                    getattr(model_info, "joint_qpos_adr", ()),
                    strict=True,
                )
            }
            self._joint_qvel_adr_by_name = {
                str(name): int(adr)
                for name, adr in zip(
                    getattr(model_info, "joint_names", ()),
                    getattr(model_info, "joint_qvel_adr", ()),
                    strict=True,
                )
            }
            self._joint_dims_by_name = {
                str(name): (int(qpos_dim), int(qvel_dim))
                for name, qpos_dim, qvel_dim in zip(
                    getattr(model_info, "joint_names", ()),
                    getattr(model_info, "joint_qpos_dim", ()),
                    getattr(model_info, "joint_qvel_dim", ()),
                    strict=True,
                )
            }
            joint_name_by_qpos_adr = {
                int(adr): str(name)
                for name, adr, dim in zip(
                    getattr(model_info, "joint_names", ()),
                    getattr(model_info, "joint_qpos_adr", ()),
                    getattr(model_info, "joint_qpos_dim", ()),
                    strict=True,
                )
                if int(dim) == 1
            }
            self._actuator_joint_names = tuple(
                joint_name_by_qpos_adr.get(int(adr), "") for adr in self._actuator_qpos_adr
            )
            self._root_qpos_dim = (
                int(np.min(self._actuator_qpos_adr)) if self._actuator_qpos_adr.size else 0
            )
            self._root_qvel_dim = (
                int(np.min(self._actuator_qvel_adr)) if self._actuator_qvel_adr.size else 0
            )
            self._num_bodies = int(model_info.num_bodies)
            self._pending_body_forces = np.zeros(
                (self._num_envs, self._num_bodies, 3), dtype=np.float64
            )
            self._model = _DrakeUniModelView(
                nq=int(model_info.nq),
                nv=int(model_info.nv),
                nu=int(model_info.nu),
            )
            self._nthread = int(getattr(self._active_runtime(), "nthread", int(nthread)))
            # Runtime state and raw sensor views are refreshed after reset/step.
            initial_state = np.asarray(self._active_runtime().physics_state(), dtype=np.float64)
            self._physics_state = np.zeros(
                (self._num_envs, initial_state.shape[1]), dtype=np.float64
            )
            self._sensor_data = np.zeros(
                (self._num_envs, int(model_info.nsensordata)),
                dtype=np.float64,
            )
            self._sensor_views: dict[str, np.ndarray] = {}
            if composed is not None:
                self._bind_entity_layout(
                    model_info, composed.layout, default_group.runtime
                )
                for group in self._runtime_groups[1:]:
                    group_info = group.runtime.model_info()
                    self._bind_entity_layout(group_info, composed.layout, group.runtime)
                if composed.variant_plan is not None:
                    self._require_shared_model_metadata()
                    self._audit_native_variant_properties(composed)
                self._gather_runtime_state()
                self._capture_entity_defaults()
            else:
                self._gather_runtime_state()
        except BaseException:
            self.close()
            raise

    # Static model contract.
    #
    # These accessors expose stable dimensions, limits, and reset defaults from
    # the cached model metadata.
    @property
    def scene_model_file(self) -> str:
        return self._scene_model_file

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def nthread(self) -> int:
        return self._nthread

    @property
    def model(self) -> _DrakeUniModelView:
        return self._model

    @property
    def num_actuators(self) -> int:
        return self._model.nu

    @property
    def num_dof_vel(self) -> int:
        if self._entity_layout is not None:
            return int(self._entity_joint_qvel_indices.size)
        return int(self._actuator_qvel_adr.size)

    # Return copies for arrays that UniLab may clamp, concatenate, or normalize.
    # The backend cache should not be mutated by task-side code.
    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._ctrl_limits.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        names = self._actuator_names
        if names is None:
            raise NotImplementedError(
                "backend 'drake' capability 'actuator names' is unavailable: "
                "DrakeUni model_info does not expose actuator_names"
            )
        if len(names) != self.num_actuators or any(not name for name in names):
            raise NotImplementedError(
                "backend 'drake' capability 'actuator names' requires one non-empty name "
                f"per control column; received {names}"
            )
        if len(set(names)) != len(names):
            raise NotImplementedError(
                "backend 'drake' capability 'actuator names' requires unique names; "
                f"received {names}"
            )
        return names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        names = self._actuator_joint_names
        if len(names) != self.num_actuators or any(not name for name in names):
            raise NotImplementedError(
                "backend 'drake' capability 'actuator target joint' requires every "
                "actuator_qpos_adr to resolve to one named single-DoF joint; "
                f"received {names}"
            )
        return names

    def get_scene_model_file(self) -> str | None:
        return self._scene_model_file

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
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        ids = selected_state_rows(env_ids, self._num_envs)
        index = layout.entities.index(owner)
        return entity_state_snapshot(
            owner,
            self._entity_default_qpos[ids],
            self._entity_default_qvel[ids],
            self._entity_default_roots[ids, index],
        )

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        index = layout.entities.index(owner)
        qpos, qvel = self._state_qpos(), self._state_qvel()
        return entity_state_snapshot(
            owner,
            qpos,
            qvel,
            self._entity_roots(qpos, qvel)[:, index],
        )

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        self._reject_named_joint_ranges(names, "joint ranges")
        return self._joint_ranges.copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name == "home":
            return self._home_qpos_mujoco.copy()
        return self._active_runtime().keyframe_qpos(str(name))

    def get_default_qpos(self) -> np.ndarray:
        return self._home_qpos_mujoco.copy()

    def get_default_dof_pos(self) -> np.ndarray:
        if self._entity_layout is not None:
            return self._entity_default_qpos[self._entity_joint_qpos_indices].copy()
        return np.asarray(self._home_qpos_mujoco[self._actuator_qpos_adr], dtype=np.float64).copy()

    def get_init_qvel(self) -> np.ndarray:
        return self._home_qvel_mujoco.copy()

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._actuator_stiffness.copy(), self._actuator_damping.copy())

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        # Body IDs are owned by DrakeUni because they depend on the materialized
        # Drake model, not on UniLab's scene pointer.
        return self._active_runtime().body_ids(tuple(str(name) for name in names))

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return self.get_body_ids(names)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            key = str(name)
            try:
                ids.append(self._site_name_to_id[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain MJCF site {key!r}") from exc
        return np.asarray(ids, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qvel_adr_by_name[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_layout is not None:
            return self._entity_joint_indices(
                names, self._entity_joint_dof_pos_indices, "joint position"
            )
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qpos_adr_by_name[key] - self._root_qpos_dim)
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._entity_layout is not None:
            return self._entity_joint_indices(
                names, self._entity_joint_dof_vel_indices, "joint velocity"
            )
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qvel_adr_by_name[key] - self._root_qvel_dim)
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qpos_adr_by_name[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            self._require_single_dof_joint(key)
            try:
                indices.append(self._joint_qvel_adr_by_name[key])
            except KeyError as exc:
                raise ValueError(f"Drake model does not contain joint {key!r}") from exc
        return np.asarray(indices, dtype=np.int32)

    # Stepping and reset.
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        self._require_entity_healthy()
        # UniLab passes one actuator command per env. An optional pre-step hook
        # can convert policy actions into backend-native position targets.
        step_count = int(nsteps)
        if step_count < 1:
            raise ValueError(f"nsteps must be >= 1, got {nsteps}")
        values = np.asarray(ctrl, dtype=np.float64)
        if values.shape != (self._num_envs, self.num_actuators):
            raise ValueError(
                "DrakeUni batch backend step expected ctrl shape "
                f"({self._num_envs}, {self.num_actuators}), got {values.shape}"
            )
        start = time.perf_counter()
        try:
            timing_totals: dict[str, float] = {}
            if self._pre_step_control_fn is None:
                for group in self._runtime_groups:
                    public_ids = np.asarray(group.public_ids, dtype=np.intp)
                    runtime = group.runtime
                    native_ctrl = values[public_ids]
                    output = runtime.step(
                        native_ctrl,
                        step_count,
                        self._pending_body_forces_or_none(group),
                    )
                    self._apply_runtime_output(group, output)
                    self._accumulate_timing(timing_totals, output)
            else:
                for _ in range(step_count):
                    # Convert one complete public control row before dispatching it
                    # to local runtimes so callback order cannot affect physics.
                    native_ctrl = self._apply_pre_step_control(values)
                    for group in self._runtime_groups:
                        public_ids = np.asarray(group.public_ids, dtype=np.intp)
                        output = group.runtime.step(
                            native_ctrl[public_ids],
                            1,
                            self._pending_body_forces_or_none(group),
                        )
                        self._apply_runtime_output(group, output)
                        self._accumulate_timing(timing_totals, output)
        except BaseException:
            self._entity_faulted = True
            raise
        finally:
            self._pending_body_forces.fill(0.0)
        timing = {key: float(value) for key, value in timing_totals.items()}
        timing.setdefault("step_ms", (time.perf_counter() - start) * 1000.0)
        return {"timing": timing}

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        self._require_entity_healthy()
        # Reset is the handoff from UniLab's sampled state tensors into
        # DrakeUni's per-env runtime contexts.
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError(
                "DrakeUni batch backend does not apply reset randomization yet"
            )
        indices = np.asarray(env_indices, dtype=np.int32)
        qpos_rows = np.asarray(qpos, dtype=np.float64)
        qvel_rows = np.asarray(qvel, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self._num_envs):
            raise IndexError(
                f"env_indices must be in [0, {self._num_envs - 1}], got {indices.tolist()}"
            )
        if qpos_rows.shape != (indices.size, self._model.nq):
            raise ValueError(f"qpos must have shape ({indices.size}, {self._model.nq})")
        if qvel_rows.shape != (indices.size, self._model.nv):
            raise ValueError(f"qvel must have shape ({indices.size}, {self._model.nv})")
        self._scatter_reset(indices, qpos_rows, qvel_rows)

    def reset_entities(self, request: SceneResetRequest) -> None:
        layout = self.get_scene_layout()
        if request.restore_default_controls:
            raise NotImplementedError(
                "Drake portable entity reset does not support restore_default_controls"
            )
        qpos, qvel = self._state_qpos(), self._state_qvel()
        prepared = prepare_scene_reset(layout, request, qpos, qvel, self._entity_roots(qpos, qvel))
        try:
            self._scatter_reset(prepared.env_ids, prepared.qpos, prepared.qvel)
        except BaseException:
            self._entity_faulted = True
            raise

    # Playback and domain randomization.
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        # Unsupported randomization knobs fail explicitly instead of silently
        # becoming no-ops.
        return DomainRandomizationCapabilities(
            supports_interval_body_force=True,
            supported_interval_terms=frozenset({INTERVAL_TERM_BODY_FORCE}),
            supports_fixed_variants=True,
            supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
            supports_per_env_playback=self._fixed_variant_realization,
        )

    _interval_term_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if plan.is_empty():
            return
        # A non-empty plan starts from cleared pending forces; the force
        # handler then accumulates into ``_pending_body_forces``.
        self._pending_body_forces.fill(0.0)
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        # Built lazily once; only body force has a handler.  Push, torque and
        # velocity terms fail closed in the base dispatch.
        if self._interval_term_handler_cache is None:
            self._interval_term_handler_cache = {
                INTERVAL_TERM_BODY_FORCE: lambda op: self.apply_body_force(
                    require_op_body_ids(op), op.payload
                ),
            }
        return self._interval_term_handler_cache

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        # Drake advances playback physics, while the shared playback helper
        # handles recording. There is no native interactive Drake viewer path.
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=False,
            supports_physics_state_playback=True,
            supports_native_video_capture=False,
            supports_debug_overlay=True,
        )

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode in {"none", "auto"}:
            return BackendPlayRenderPlan(
                mode=mode,
                headless=True,
                record_video=False,
                num_steps=play_steps,
                output_video=None,
            )
        if mode == "interactive":
            raise NotImplementedError(
                "DrakeUni batch backend does not support interactive rendering"
            )
        if play_steps is None:
            raise ValueError("DrakeUni record playback requires a finite play_steps value.")
        if output_video is None:
            raise ValueError("DrakeUni record playback requires an output video path.")
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
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: DebugOverlayGetter | None = None,
        on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    ) -> str | None:
        # Playback keeps Drake as the physics backend. The helper owns rendering
        # and video capture so training code can use one playback contract.
        return run_drake_playback(
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=bool(headless),
            record_video=bool(record_video),
            frame_state_getter=frame_state_getter,
            camera_kwargs=CameraCfg.from_kwargs(camera_kwargs),
            debug_overlay_getter=debug_overlay_getter,
            on_frame=on_frame,
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
        del spacing, offset_mode, headless, capture, width, height, camera_kwargs
        raise NotImplementedError("DrakeUni batch backend records through run_playback")

    def render(self) -> None:
        raise NotImplementedError("DrakeUni batch backend does not support interactive rendering")

    def capture_video_frame(self) -> np.ndarray:
        raise NotImplementedError("DrakeUni batch backend records through run_playback")

    # Runtime state getters.
    #
    # ``physics_state`` is DrakeUni's compact per-env packet used by playback
    # and debugging. Sensor-specific getters below expose named slices/packets.
    def get_physics_state_layout(self) -> PhysicsStateLayout:
        """Return the ``[time, qpos, qvel]`` packet layout (no mocap bodies)."""
        self._require_entity_healthy()
        return PhysicsStateLayout(nq=int(self._model.nq), nv=int(self._model.nv))

    def get_physics_state(self) -> np.ndarray:
        self._require_entity_healthy()
        return self._physics_state.copy()

    def get_playback_model(self, env_index: int | None = None) -> str:
        self._require_entity_healthy()
        if self._variant_assignment is not None and len(self._variant_model_files) > 1:
            if env_index is None:
                raise ValueError("Drake fixed-variant playback requires an explicit env_index")
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
            return self._variant_model_files[int(self._variant_assignment[idx])]
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        return self._scene_model_file

    def diagnostics(self) -> Any:
        self._require_entity_healthy()
        return self._active_runtime().diagnostics()

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        self._require_entity_healthy()
        if torque is not None:
            raise NotImplementedError(
                "DrakeUni batch backend does not support interval body torque perturbation"
            )
        ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        values = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, ids.size, 3)
        if values.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {values.shape}")
        for offset, body_id in enumerate(ids):
            if body_id < 0 or body_id >= self._num_bodies:
                raise IndexError(f"body id {int(body_id)} is outside [0, {self._num_bodies - 1}]")
            self._pending_body_forces[:, int(body_id), :] += values[:, offset, :]

    # Sensor access.
    #
    # DrakeUni returns one flat sensor array; this class owns the MuJoCo-style
    # named views over that array.
    def get_base_pos(self) -> np.ndarray:
        self._reject_entity_mode("single-root base state")
        self._require_floating_root()
        return self._physics_state[:, 1:4].copy()

    def get_base_quat(self) -> np.ndarray:
        self._reject_entity_mode("single-root base state")
        self._require_floating_root()
        return self._physics_state[:, 4:8].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        self._reject_entity_mode("single-root base state")
        self._require_floating_root()
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start : qvel_start + 3].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        self._reject_entity_mode("single-root base state")
        self._require_floating_root()
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start + 3 : qvel_start + 6].copy()

    def get_dof_pos(self) -> np.ndarray:
        if self._entity_layout is not None:
            return self._physics_state[:, 1 + self._entity_joint_qpos_indices].copy()
        return self._physics_state[:, 1 + self._actuator_qpos_adr].copy()

    def get_dof_vel(self) -> np.ndarray:
        if self._entity_layout is not None:
            return self._physics_state[
                :,
                1 + self._model.nq + self._entity_joint_qvel_indices,
            ].copy()
        qvel_start = 1 + self._model.nq
        return self._physics_state[:, qvel_start + self._actuator_qvel_adr].copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["pos"]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["quat"]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["linvel"]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._body_state(body_ids)["angvel"]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._reject_entity_mode("single-root body-frame state")
        body_state = self._body_state(body_ids)
        base_pos = self.get_base_pos()
        base_rot = _quat_to_rotation_matrix(self.get_base_quat())
        delta = body_state["pos"] - base_pos[:, None, :]
        return np.einsum("nij,nkj->nki", np.swapaxes(base_rot, 1, 2), delta)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._reject_entity_mode("single-root body-frame state")
        body_quat = self._body_state(body_ids)["quat"]
        base_inv = _quat_conjugate(self.get_base_quat())
        return _quat_multiply(base_inv[:, None, :], body_quat)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._reject_entity_mode("single-root body-frame state")
        # Analytical per the SimBackend contract: world-frame velocity
        # expressed in each body's own frame.
        body_state = self._body_state(body_ids)
        body_rot = _quat_to_rotation_matrix(body_state["quat"])
        return np.einsum(
            "nkij,nkj->nki",
            np.swapaxes(body_rot, -1, -2),
            body_state["linvel"],
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        body_state = self._body_state(body_ids)
        body_rot = _quat_to_rotation_matrix(body_state["quat"])
        return np.einsum(
            "nkij,nkj->nki",
            np.swapaxes(body_rot, -1, -2),
            body_state["angvel"],
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._require_entity_healthy()
        if name in self._sensor_views:
            return self._sensor_views[name].copy()
        raise KeyError(f"Unknown DrakeUni sensor: {name}")

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Capture DrakeUni sensor addresses; read only the refreshed host cache."""
        name_to_index = {name: index for index, name in enumerate(self._sensor_names)}
        slots = tuple(
            (
                int(self._sensor_adr[name_to_index[name]]),
                int(self._sensor_dim[name_to_index[name]]),
            )
            for name in names
        )

        def read() -> np.ndarray:
            values = [
                self._sensor_data[:, address : address + dimension] for address, dimension in slots
            ]
            return np.concatenate(values, axis=1)

        return read

    # Internal helpers.
    def _bind_entity_layout(
        self,
        model_info: Any,
        layout: CompiledSceneLayout,
        runtime: Any,
    ) -> None:
        """Bind and validate every public layout mapping on the cold path."""
        for field, expected, actual in (
            ("nq", layout.nq, int(model_info.nq)),
            ("nv", layout.nv, int(model_info.nv)),
            ("nu", layout.nu, int(model_info.nu)),
            ("nbody", layout.nbody, int(model_info.num_bodies)),
        ):
            if expected != actual:
                raise ValueError(
                    f"Drake runtime scene {field} is {actual}; portable layout requires {expected}"
                )

        global_body_names = tuple(
            f"{entity.name}/{body_name}"
            for entity in layout.entities
            for body_name in entity.body_names
        )
        actual_body_ids = runtime.body_ids(global_body_names)
        expected_body_ids = np.asarray(
            [body_id for entity in layout.entities for body_id in entity.body_ids],
            dtype=np.int32,
        )
        if not np.array_equal(actual_body_ids, expected_body_ids):
            raise ValueError(
                "Drake runtime body IDs differ from the portable scene layout: "
                f"expected {expected_body_ids.tolist()}, got {actual_body_ids.tolist()}"
            )

        joint_names = tuple(str(name) for name in model_info.joint_names)
        joint_bodies = tuple(str(name) for name in getattr(model_info, "joint_body_names", ()))
        if len(joint_names) != len(joint_bodies):
            raise ValueError("Drake model_info joint names and body names are not aligned")
        metadata = []
        for index, (name, body_name) in enumerate(zip(joint_names, joint_bodies, strict=True)):
            qpos_start = int(model_info.joint_qpos_adr[index])
            qvel_start = int(model_info.joint_qvel_adr[index])
            qpos_width = int(model_info.joint_qpos_dim[index])
            qvel_width = int(model_info.joint_qvel_dim[index])
            metadata.append(
                (
                    name,
                    body_name,
                    tuple(range(qpos_start, qpos_start + qpos_width)),
                    tuple(range(qvel_start, qvel_start + qvel_width)),
                )
            )

        consumed: set[int] = set()

        def require_joint(
            entity: EntityLayout, joint: JointLayout | None
        ) -> None:
            body_name = entity.root_body if joint is None else joint.body_name
            expected_qpos = entity.root_qpos_indices if joint is None else joint.qpos_indices
            expected_qvel = entity.root_qvel_indices if joint is None else joint.qvel_indices
            matches = [
                index
                for index, (_, body, qpos, qvel) in enumerate(metadata)
                if index not in consumed
                and body == f"{entity.name}/{body_name}"
                and qpos == expected_qpos
                and qvel == expected_qvel
            ]
            if len(matches) != 1:
                label = "root" if joint is None else f"joint {entity.name}/{joint.name}"
                raise ValueError(f"Drake runtime cannot uniquely bind portable {label}")
            consumed.add(matches[0])

        for entity in layout.entities:
            if entity.root_mode == "floating":
                require_joint(entity, None)
            for joint in entity.joints:
                require_joint(entity, joint)
        if consumed != set(range(len(metadata))):
            raise ValueError("Drake runtime contains joints outside the portable scene layout")

        expected_actuators = tuple(
            f"{entity.name}/{name}"
            for entity in layout.entities
            for name in entity.actuator_names
        )
        actual_actuators = tuple(str(name) for name in model_info.actuator_names)
        if actual_actuators != expected_actuators:
            raise ValueError(
                "Drake runtime actuator order differs from the portable scene layout: "
                f"expected {expected_actuators}, got {actual_actuators}"
            )
        for entity in layout.entities:
            joints = {joint.name: joint for joint in entity.joints}
            for local_name, control in zip(
                entity.actuator_names, entity.actuator_indices, strict=True
            ):
                target_name = entity.actuator_joint_names[entity.actuator_names.index(local_name)]
                target = joints[target_name]
                if int(model_info.actuator_qpos_adr[control]) != target.qpos_indices[0]:
                    raise ValueError(
                        f"Drake actuator {entity.name}/{local_name} targets the wrong qpos column"
                    )
                if int(model_info.actuator_qvel_adr[control]) != target.qvel_indices[0]:
                    raise ValueError(
                        f"Drake actuator {entity.name}/{local_name} targets the wrong qvel column"
                    )

        qpos_columns: list[int] = []
        qvel_columns: list[int] = []
        for entity in layout.entities:
            for joint in entity.joints:
                qpos_columns.extend(joint.qpos_indices)
                qvel_columns.extend(joint.qvel_indices)
        self._entity_layout = layout
        self._entity_root_ids = tuple(
            entity.body_ids[entity.body_names.index(entity.root_body)]
            for entity in layout.entities
        )
        self._entity_joint_qpos_indices = np.asarray(qpos_columns, dtype=np.intp)
        self._entity_joint_qvel_indices = np.asarray(qvel_columns, dtype=np.intp)
        self._entity_joint_dof_pos_indices = {
            f"{entity.name}/{joint.name}": offset
            for offset, (entity, joint) in enumerate(
                (entity, joint)
                for entity in layout.entities
                for joint in entity.joints
            )
        }
        self._entity_joint_dof_vel_indices = dict(self._entity_joint_dof_pos_indices)

    def _capture_entity_defaults(self) -> None:
        self._entity_default_qpos = self._state_qpos()
        self._entity_default_qvel = self._state_qvel()
        self._entity_default_roots = self._entity_roots(
            self._entity_default_qpos, self._entity_default_qvel
        )

    def _state_qpos(self) -> np.ndarray:
        return self._physics_state[:, 1 : 1 + self._model.nq].copy()

    def _state_qvel(self) -> np.ndarray:
        return self._physics_state[:, 1 + self._model.nq :].copy()

    def _entity_roots(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        layout = self.get_scene_layout()
        roots = np.zeros((self._num_envs, len(layout.entities), 13), dtype=np.float64)
        fixed_entities = [
            self._entity_root_ids[index]
            for index, entity in enumerate(layout.entities)
            if entity.root_mode == "fixed"
        ]
        fixed_states = (
            self._body_state(np.asarray(fixed_entities, dtype=np.int32))
            if fixed_entities
            else {}
        )
        for index, entity in enumerate(layout.entities):
            if entity.root_mode == "floating":
                state = entity_state_snapshot(entity, qpos, qvel)
                roots[:, index, :7] = state["root_pose"]
                roots[:, index, 7:] = state["root_velocity"]
                continue
            body_offset = fixed_entities.index(self._entity_root_ids[index])
            roots[:, index, :7] = np.concatenate(
                (fixed_states["pos"][:, body_offset], fixed_states["quat"][:, body_offset]),
                axis=1,
            )
            roots[:, index, 7:10] = fixed_states["linvel"][:, body_offset]
            roots[:, index, 10:] = fixed_states["angvel"][:, body_offset]
        return roots

    def _entity_joint_indices(
        self, names: Sequence[str], mapping: Mapping[str, int], label: str
    ) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            key = str(name)
            try:
                indices.append(mapping[key])
            except KeyError as exc:
                raise ValueError(
                    f"Drake portable scene does not contain {label} joint {key!r}"
                ) from exc
        return np.asarray(indices, dtype=np.int32)

    def _reject_entity_mode(self, capability: str) -> None:
        if self._entity_layout is not None:
            raise NotImplementedError(
                f"Drake portable entity scenes do not expose implicit {capability}"
            )

    def _require_entity_healthy(self) -> None:
        if self._entity_faulted:
            raise RuntimeError("Drake backend is faulted after a native reset; reconstruct it")
        if self._entity_closed or self._runtime is None:
            raise RuntimeError("Drake backend is closed")

    def _active_runtime(self) -> Any:
        self._require_entity_healthy()
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Drake backend is closed")
        return runtime

    def _runtime_group(self, variant: int) -> _DrakeRuntimeGroup:
        for group in self._runtime_groups:
            if group.variant == int(variant):
                return group
        raise ValueError(f"Drake fixed-variant runtime {int(variant)} is not assigned")

    def cleanup_scene_assets(self) -> None:
        self.close()

    def close(self) -> None:
        if self._entity_closed:
            return
        self._entity_closed = True
        groups = self._runtime_groups
        composed = self._composed_scene
        self._runtime_groups = ()
        self._runtime = None
        self._composed_scene = None
        first_close_error: BaseException | None = None
        try:
            for group in groups:
                try:
                    group.runtime.close()
                except BaseException as error:
                    if first_close_error is None:
                        first_close_error = error
        finally:
            if composed is not None:
                composed.close()
        if first_close_error is not None:
            raise first_close_error

    def _gather_runtime_state(self) -> None:
        """Gather every local runtime cache into public environment row order."""

        rows: list[tuple[_DrakeRuntimeGroup, np.ndarray, np.ndarray]] = []
        for group in self._runtime_groups:
            state = np.asarray(group.runtime.physics_state(), dtype=np.float64)
            sensor = np.asarray(group.runtime.sensor_data(), dtype=np.float64)
            expected_state_shape = (group.count, self._physics_state.shape[1])
            expected_sensor_shape = self._sensor_data.shape[1:]
            if state.shape != expected_state_shape or sensor.shape != (
                (group.count,) + expected_sensor_shape
            ):
                raise ValueError(
                    f"Drake runtime variant {group.variant} returned an invalid "
                    "state or sensor shape"
                )
            rows.append((group, state, sensor))
        for group, state, sensor in rows:
            public_ids = np.asarray(group.public_ids, dtype=np.intp)
            self._physics_state[public_ids] = state
            self._sensor_data[public_ids] = sensor
        self._rebuild_sensor_views()

    def _apply_runtime_output(self, group: _DrakeRuntimeGroup, output: dict[str, Any]) -> None:
        state = np.asarray(output["state"], dtype=np.float64)
        sensor = np.asarray(output["sensor_data"], dtype=np.float64)
        if "env_ids" in output:
            local_ids = np.asarray(output["env_ids"], dtype=np.intp).reshape(-1)
            public_ids = np.asarray(group.public_ids, dtype=np.intp)
            if np.any(local_ids < 0) or np.any(local_ids >= group.count):
                raise ValueError(
                    f"Drake runtime variant {group.variant} returned an invalid "
                    "local environment id"
                )
            selected_public_ids = public_ids[local_ids]
            expected_state_shape = (local_ids.size, self._physics_state.shape[1])
            expected_sensor_shape = (local_ids.size, self._sensor_data.shape[1])
            if state.shape != expected_state_shape or sensor.shape != expected_sensor_shape:
                raise ValueError(
                    f"Drake runtime variant {group.variant} returned invalid reset output"
                )
            self._physics_state[selected_public_ids] = state
            self._sensor_data[selected_public_ids] = sensor
        else:
            public_ids = np.asarray(group.public_ids, dtype=np.intp)
            expected_state_shape = (group.count, self._physics_state.shape[1])
            expected_sensor_shape = (group.count, self._sensor_data.shape[1])
            if state.shape != expected_state_shape or sensor.shape != expected_sensor_shape:
                raise ValueError(
                    f"Drake runtime variant {group.variant} returned an incomplete output"
                )
            self._physics_state[public_ids] = state
            self._sensor_data[public_ids] = sensor
        self._rebuild_sensor_views()

    def _scatter_reset(
        self, indices: np.ndarray, qpos: np.ndarray, qvel: np.ndarray
    ) -> None:
        requested = np.asarray(indices, dtype=np.intp)
        for group in self._runtime_groups:
            group_public = np.asarray(group.public_ids, dtype=np.intp)
            mask = np.isin(group_public, requested)
            if not np.any(mask):
                continue
            local_ids = np.flatnonzero(mask)
            selected_public = group_public[mask]
            request_row_by_public = {
                int(public_id): row for row, public_id in enumerate(requested)
            }
            request_rows = np.asarray(
                [request_row_by_public[int(public_id)] for public_id in selected_public],
                dtype=np.intp,
            )
            output = group.runtime.reset(
                local_ids.astype(np.int32),
                qpos[request_rows],
                qvel[request_rows],
            )
            self._apply_runtime_output(group, output)

    def _audit_native_variant_properties(self, composed: Any) -> None:
        plan = composed.variant_plan
        for group in self._runtime_groups:
            read_native = getattr(group.runtime, "native_model_properties", None)
            if not callable(read_native):
                raise RuntimeError(
                    "DrakeUni runtime does not expose public native_model_properties(); "
                    "fixed variants cannot be accepted"
                )
            expected = scan_expected_native_properties(
                str(plan.variants[group.variant].model_file), composed.layout
            )
            validate_native_model_properties(expected, read_native(), group.variant)

    def _require_shared_model_metadata(self) -> None:
        reference_info = self._runtime_groups[0].runtime.model_info()
        reference = self._model_metadata_signature(reference_info)
        for group in self._runtime_groups[1:]:
            actual = self._model_metadata_signature(group.runtime.model_info())
            if actual != reference:
                raise ValueError(
                    "Drake fixed variants differ in public control or sensor metadata; "
                    "only same-layout native property variants are supported"
                )

    @staticmethod
    def _model_metadata_signature(model_info: Any) -> tuple[Any, ...]:
        def strings(name: str) -> tuple[str, ...]:
            return tuple(str(value) for value in getattr(model_info, name))

        def integers(name: str) -> tuple[int, ...]:
            return tuple(int(value) for value in getattr(model_info, name))

        def arrays(name: str) -> list[float]:
            return [
                float(value)
                for value in np.asarray(
                    getattr(model_info, name), dtype=np.float64
                ).reshape(-1)
            ]

        return (
            int(model_info.nq),
            int(model_info.nv),
            int(model_info.nu),
            int(model_info.num_bodies),
            int(model_info.nsensordata),
            strings("joint_names"),
            integers("joint_qpos_adr"),
            integers("joint_qvel_adr"),
            integers("joint_qpos_dim"),
            integers("joint_qvel_dim"),
            strings("actuator_names"),
            integers("actuator_qpos_adr"),
            integers("actuator_qvel_adr"),
            arrays("ctrl_limits"),
            arrays("joint_ranges"),
            arrays("actuator_stiffness"),
            arrays("actuator_damping"),
            strings("sensor_names"),
            integers("sensor_adr"),
            integers("sensor_dim"),
        )

    def _rebuild_sensor_views(self) -> None:
        self._sensor_views = {}
        for index, name in enumerate(self._sensor_names):
            adr = int(self._sensor_adr[index])
            dim = int(self._sensor_dim[index])
            self._sensor_views[name] = self._sensor_data[:, adr : adr + dim]

    def _body_state(self, body_ids: np.ndarray) -> dict[str, np.ndarray]:
        self._require_entity_healthy()
        ids = np.asarray(body_ids, dtype=np.int32)
        if ids.ndim != 1:
            raise ValueError(f"body_ids must be one-dimensional, got {ids.shape}")
        outputs: list[tuple[_DrakeRuntimeGroup, dict[str, np.ndarray]]] = []
        for group in self._runtime_groups:
            local = group.runtime.compute_body_state(ids)
            outputs.append(
                (
                    group,
                    {
                        name: np.asarray(value, dtype=np.float64)
                        for name, value in local.items()
                    },
                )
            )
        if not outputs:
            raise RuntimeError("Drake backend has no active runtimes")
        public = {}
        reference_group, reference_state = outputs[0]
        reference_ids = np.asarray(reference_group.public_ids, dtype=np.intp)
        for name, reference_values in reference_state.items():
            values = np.zeros(
                (self._num_envs, *reference_values.shape[1:]), dtype=np.float64
            )
            values[reference_ids] = reference_values
            for group, group_state in outputs[1:]:
                if tuple(group_state) != tuple(reference_state):
                    raise ValueError("Drake runtime body-state fields differ between variants")
                group_values = np.asarray(group_state[name], dtype=np.float64)
                expected_group_shape = (group.count, *reference_values.shape[1:])
                if group_values.shape != expected_group_shape:
                    raise ValueError(
                        "Drake runtime body-state shapes differ between variants"
                    )
                values[np.asarray(group.public_ids, dtype=np.intp)] = group_values
            public[name] = values
        return cast(dict[str, np.ndarray], public)

    def _pending_body_forces_or_none(
        self, group: _DrakeRuntimeGroup
    ) -> np.ndarray | None:
        public_ids = np.asarray(group.public_ids, dtype=np.intp)
        values = self._pending_body_forces[public_ids]
        if np.any(values):
            return values
        return None

    @staticmethod
    def _accumulate_timing(totals: dict[str, float], output: dict[str, Any]) -> None:
        for key, value in dict(output.get("timing", {})).items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                totals[key] = totals.get(key, 0.0) + float(value)

    def _require_single_dof_joint(self, name: str) -> None:
        dims = self._joint_dims_by_name.get(name)
        if dims is None:
            raise ValueError(f"Drake model does not contain joint {name!r}")
        if dims != (1, 1):
            raise ValueError(f"Drake joint {name!r} is not a single-DoF joint")

    def _require_floating_root(self) -> None:
        if self._model.nq < ROOT_QPOS_DIM or self._model.nv < ROOT_QVEL_DIM:
            raise NotImplementedError(
                "DrakeBackend root-state helpers require a floating-root compact state"
            )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64).copy()
    values[..., 1:] *= -1.0
    return values


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    q = np.divide(values, np.maximum(norm, 1.0e-12))
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


__all__ = [
    "DRAKE_AVAILABLE",
    "DRAKE_IMPORT_ERROR",
    "DRAKE_BATCH_AVAILABLE",
    "DRAKE_BATCH_IMPORT_ERROR",
    "DrakeBackend",
    "_resolve_batch_nthread",
    "ensure_drake_batch_available",
]
