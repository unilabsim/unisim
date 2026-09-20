import logging
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import numpy as np

from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    INTERVAL_TERM_PUSH,
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    FixedVariantLayout,
    IntervalTermOp,
    ResetRandomizationPayload,
    require_op_body_ids,
)
from unisim.entity_state import (
    entity_state_snapshot,
    prepare_scene_reset,
    selected_state_rows,
)
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_layout import BoundSceneReset, CompiledSceneLayout
from unisim.utils.rotation import np_quat_apply_inverse_batched

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings
    from motrixsim.render import RenderClosedError as _MotrixRenderClosedError

    MOTRIX_AVAILABLE = True
except ImportError:
    MOTRIX_AVAILABLE = False
    # No motrixsim in this process: the placeholders stay None and every
    # consumer is gated behind MOTRIX_AVAILABLE (the factory raises before
    # construction), while the ``except _MotrixRenderClosedError`` clauses
    # below never match, which is correct because the renderer cannot exist
    # without the package.  The Any cast keeps the names bound for type
    # checkers without changing runtime behavior.
    mtx = cast(Any, None)
    RenderApp = cast(Any, None)
    RenderSettings = cast(Any, None)
    _MotrixRenderClosedError = ()

from ..base import (
    _NATIVE_RENDERER_PLAY_CAPABILITIES,
    BackendHeightScanner,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    BackendTerrainSpawnData,
    CameraCfg,
    RenderClosedError,
    SimBackend,
    normalize_play_render_mode,
    unsupported_debug_overlay_error,
)
from ..motrix_camera import (
    MotrixTrackingCamera,
    render_offsets,
    resolve_system_camera_view,
    tracking_camera_lookat,
)
from ..reset_impact import ResetImpactIndex, bind_reset_impacts
from .playback import run_motrix_playback

logger = logging.getLogger(__name__)

T = TypeVar("T")
DEFAULT_MOTRIX_MAX_ITERATIONS = 3


def _require_not_none(value: T | None, error_message: str) -> T:
    if value is None:
        raise ValueError(error_message)
    return value


def _first_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32)
    return float(arr.reshape(-1)[0])


def _validate_motrix_cpu_ids(cpu_ids: Any) -> tuple[int, ...]:
    """Validate an explicit Motrix worker CPU block on the cold path.

    Mirrors the MuJoCo ``BatchEnvPool`` affinity contract: ``cpu_ids[i]``
    pins MotrixSim worker thread ``i`` (modulo the worker count), so entries
    must be non-empty, unique, non-negative integers, and available to this
    process.
    """
    if cpu_ids is None:
        return ()
    if isinstance(cpu_ids, (str, bytes)):
        raise TypeError("cpu_ids must be a sequence of integer CPU ids")
    entries = list(cpu_ids)
    if not entries:
        raise ValueError("cpu_ids must be non-empty")
    ids: list[int] = []
    for cpu_id in entries:
        if isinstance(cpu_id, bool) or not isinstance(cpu_id, (int, np.integer)):
            raise ValueError(f"cpu_ids entries must be non-negative integers, got {cpu_id!r}")
        cpu_id = int(cpu_id)
        if cpu_id < 0:
            raise ValueError(f"cpu_ids entries must be non-negative integers, got {cpu_id!r}")
        ids.append(cpu_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"cpu_ids entries must be unique, got {ids!r}")
    available = getattr(os, "sched_getaffinity", None)
    if available is not None:
        missing = sorted(set(ids) - available(0))
        if missing:
            raise ValueError(
                f"cpu_ids entries {missing} are not available to this process "
                f"(sched_getaffinity={sorted(available(0))})"
            )
    return tuple(ids)


def _configure_motrix_worker_affinity(cpu_ids: Sequence[int] | None) -> tuple[int, ...] | None:
    """Initialize MotrixSim's shared worker pool with explicit core pinning.

    Cold path only: must run before the first MotrixSim model load, because
    the shared pool is created once per process (the first step otherwise
    lazy-creates it with the default one-worker-per-CPU policy). ``None``
    leaves the default policy untouched. A pool that another backend in this
    process already initialized keeps its mapping; the conflict degrades to a
    warning instead of failing env construction.
    """
    ids = _validate_motrix_cpu_ids(cpu_ids)
    if not ids:
        return None
    try:
        mtx.init_thread_pool(core_ids=list(ids))
    except RuntimeError as exc:
        warnings.warn(
            f"MotrixSim shared worker pool was already initialized in this process; "
            f"cpu_ids={list(ids)} was not applied: {exc}",
            stacklevel=2,
        )
    return ids


def _contiguous_slice(indices: np.ndarray) -> slice | None:
    if indices.size == 0:
        return None
    start = int(indices[0])
    stop = start + int(indices.size)
    if np.array_equal(indices, np.arange(start, stop, dtype=indices.dtype)):
        return slice(start, stop)
    return None


def _resolve_portable_native_name(model: Any, local_name: str) -> str:
    """Resolve a legacy local name against one uniquely namespaced Motrix link."""
    if "/" in local_name:
        return local_name
    matches = [str(link.name) for link in model.links if str(link.name).endswith("/" + local_name)]
    if len(matches) != 1:
        raise ValueError(f"portable Motrix body {local_name!r} matched {len(matches)} native links")
    return matches[0]


def _resolve_portable_layout_name(layout: CompiledSceneLayout, local_name: str) -> str:
    """Resolve a configured local body against one public entity body."""
    if "/" in local_name:
        for entity in layout.entities:
            if local_name in {
                f"{entity.name}/{body_name}" for body_name in entity.body_names
            }:
                return local_name
        raise ValueError(f"portable Motrix body {local_name!r} is not in the public layout")
    matches = [
        f"{entity.name}/{body_name}"
        for entity in layout.entities
        for body_name in entity.body_names
        if body_name == local_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"portable Motrix body {local_name!r} matched {len(matches)} public bodies"
        )
    return matches[0]


@dataclass
class _MotrixSceneContext:
    model: Any
    sensor_names: tuple[str, ...]
    terrain_origins: np.ndarray | None = None
    terrain_surface_sampler: object | None = None
    cleanup_handle: object | None = None


@dataclass
class _MotrixPortableBinding:
    """Audited native mappings and construction defaults for one source."""

    links_by_id: dict[int, Any]
    geoms_by_id: dict[int, Any]
    joints_by_public_dof: dict[int, Any]
    kinematic_mocaps: dict[int, Any]
    public_to_native_body: np.ndarray
    public_to_native_geom: np.ndarray
    default_body_mass: np.ndarray
    default_body_ipos: np.ndarray
    default_dof_armature: np.ndarray
    default_dof_frictionloss: np.ndarray
    default_geom_sizes: np.ndarray
    default_actuator_kp: np.ndarray
    default_actuator_kd: np.ndarray
    default_geom_friction: np.ndarray


@dataclass
class _MotrixPortableRuntime:
    """One immutable variant context and its assigned public rows."""

    variant: int
    rows: np.ndarray
    model: Any
    data: Any
    sensor_names: tuple[str, ...]
    binding: _MotrixPortableBinding
    default_controls: np.ndarray
    found_contact_geom_pairs: dict[str, tuple[int, int]] = field(default_factory=dict)
    default_qpos: np.ndarray | None = None
    default_qvel: np.ndarray | None = None
    default_roots: np.ndarray | None = None

    def local_rows(self, public_rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.rows, public_rows)


@dataclass(frozen=True)
class _MotrixPortableResetRandomization:
    """Prevalidated public reset values ready for per-variant submission."""

    body_mass: np.ndarray | None
    body_ipos: np.ndarray | None
    dof_armature: np.ndarray | None
    dof_frictionloss: np.ndarray | None


@dataclass
class _MotrixTerrainScanner(BackendHeightScanner):
    scanner: Any
    data: Any
    out: np.ndarray

    def scan(self) -> np.ndarray:
        heights = np.asarray(self.scanner.scan(self.data, out=self.out))
        if heights.shape != self.out.shape:
            raise ValueError(
                f"Motrix TerrainScanner.scan returned shape {heights.shape}, "
                f"expected {self.out.shape}"
            )
        return heights


@dataclass(frozen=True)
class _MotrixSourceSensorContract:
    """One common-compiled source sensor and its expected Motrix identity."""

    identity: tuple[Any, ...]
    dimension: int
    sensor_kind: str
    site_name: str | None = None
    site_identity: (
        tuple[str, tuple[float, float, float], tuple[float, float, float, float]] | None
    ) = None


def _build_motrix_scene_context(
    scene: SceneCfg,
    *,
    add_body_sensors: bool,
    base_name: str,
) -> _MotrixSceneContext:
    from unisim.backend.motrix.scene import (
        _materialize_motrix_hfield_attached_scene_with_sensor_names,
        _materialize_motrix_scene_with_sensor_names,
    )

    if scene is None:
        raise ValueError("SceneCfg must be provided")
    if not scene.model_file:
        raise ValueError("SceneCfg.model_file must be provided")

    if scene.terrain is None:
        model, sensor_names = _materialize_motrix_scene_with_sensor_names(
            model_file=scene.model_file,
            fragment_files=scene.fragment_files,
            add_body_sensors=add_body_sensors,
            base_name=base_name,
        )
        return _MotrixSceneContext(model=model, sensor_names=sensor_names)

    if scene.terrain.generator is None:
        raise ValueError("SceneCfg.terrain.generator must be configured for terrain scenes")

    model, terrain_origins, terrain_surface_sampler, sensor_names = (
        _materialize_motrix_hfield_attached_scene_with_sensor_names(
            model_file=scene.model_file,
            terrain_cfg=scene.terrain.generator,
            fragment_files=scene.fragment_files,
            hfield_name=scene.terrain.hfield_name,
            geom_name=scene.terrain.geom_name or "floor",
            add_body_sensors=add_body_sensors,
            base_name=base_name,
            return_surface_sampler=True,
        )
    )
    return _MotrixSceneContext(
        model=model,
        sensor_names=sensor_names,
        terrain_origins=terrain_origins,
        terrain_surface_sampler=terrain_surface_sampler,
    )


class MotrixBackend(SimBackend):
    """MotrixSim backend implementation."""

    _play_capabilities = _NATIVE_RENDERER_PLAY_CAPABILITIES
    _composed_scene: Any
    _data: Any
    _entity_layout: CompiledSceneLayout | None
    _model: Any
    _portable_default_body_ipos: np.ndarray
    _portable_default_body_mass: np.ndarray
    _portable_default_dof_armature: np.ndarray
    _portable_default_dof_frictionloss: np.ndarray
    _portable_default_qpos: np.ndarray
    _portable_default_qvel: np.ndarray
    _portable_default_roots: np.ndarray
    _portable_faulted: bool
    _portable_mode: bool
    _portable_public_to_native_body: np.ndarray
    _portable_public_to_native_geom: np.ndarray
    _portable_pending_body_forces: dict[int, np.ndarray]
    _portable_pending_body_torques: dict[int, np.ndarray]
    _portable_reset_impacts: ResetImpactIndex
    _portable_runtimes: tuple[_MotrixPortableRuntime, ...]
    _portable_variant_assignment: np.ndarray | None
    _portable_variant_geom_sizes: np.ndarray | None
    _supports_link_mass_override: bool
    _supports_link_com_override: bool
    _supports_joint_armature_override: bool
    _supports_joint_frictionloss_override: bool
    _closed: bool
    _cpu_ids: tuple[int, ...] | None

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        base_name: str = "base",
        np_dtype=np.float32,
        add_body_sensors: bool = False,
        max_iterations: int | None = DEFAULT_MOTRIX_MAX_ITERATIONS,
        push_body_name: str | None = None,
        cpu_ids: Sequence[int] | None = None,
    ):
        portable_mode = bool(scene.entity_assets)
        if portable_mode:
            if scene.entity_variant is not None and (
                scene.entity_variant.plan.layout is not FixedVariantLayout.SAME_LAYOUT
            ):
                raise NotImplementedError(
                    "Motrix portable entity scenes support only same_layout fixed variants"
                )
            if any(
                entity.root_mode == "kinematic" and entity.mirror_of is None
                for entity in scene.entity_assets
            ):
                raise NotImplementedError(
                    "Motrix portable entity scenes do not support physical kinematic entities"
                )
            if scene.terrain is not None:
                raise NotImplementedError(
                    "Motrix portable entity scenes do not support generated terrain"
                )
        require_scene_composition_support(scene, "motrix")
        if not MOTRIX_AVAILABLE:
            raise ImportError("motrixsim not available")

        # Must precede every model load: MotrixSim's shared worker pool is
        # created once per process and the first load/step lazy-creates it.
        self._cpu_ids = _configure_motrix_worker_affinity(cpu_ids)

        self._portable_mode = portable_mode
        self._composed_scene: Any = None
        self._entity_layout: CompiledSceneLayout | None = None
        self._portable_runtimes: tuple[_MotrixPortableRuntime, ...] = ()
        self._portable_variant_assignment: np.ndarray | None = None
        self._portable_variant_geom_sizes: np.ndarray | None = None
        self._supports_link_mass_override = False
        self._supports_link_com_override = False
        self._supports_joint_armature_override = False
        self._supports_joint_frictionloss_override = False
        self._portable_pending_body_forces: dict[int, np.ndarray] = {}
        self._portable_pending_body_torques: dict[int, np.ndarray] = {}
        self._portable_faulted = False
        self._closed = False
        self._num_envs = int(num_envs)
        self._np_dtype = np_dtype
        self._pre_step_control_fn = None
        runtimes: list[_MotrixPortableRuntime] = []
        if portable_mode:
            from unisim.scene_compiler import compile_portable_scene

            from .scene import _materialize_motrix_expanded_scene_with_sensor_inventory

            composed = compile_portable_scene(scene, int(num_envs), float(sim_dt))
            self._composed_scene = composed
            try:
                if int(composed.model.na) != 0:
                    raise NotImplementedError(
                        "Motrix portable entity scenes do not support actuator activation "
                        "state because Motrix does not expose native activation state"
                    )
                source_sensor_contracts = self._audit_portable_source_sensor_contract(
                    composed.model, composed.layout
                )
                sources: tuple[str, ...]
                if composed.variant_plan is None:
                    assignment = np.zeros((self._num_envs,), dtype=np.int32)
                    sources = (composed.model_file,)
                else:
                    assignment = np.asarray(composed.variant_plan.assignment, dtype=np.int32)
                    sources = tuple(
                        variant.model_file for variant in composed.variant_plan.variants
                    )
                if assignment.shape != (self._num_envs,):
                    raise ValueError(
                        "Motrix fixed-variant assignment shape "
                        f"{assignment.shape} differs from ({self._num_envs},)"
                    )
                if np.any(assignment < 0) or np.any(assignment >= len(sources)):
                    raise ValueError("Motrix fixed-variant assignment refers to an absent source")
                if max_iterations is None:
                    max_iterations = DEFAULT_MOTRIX_MAX_ITERATIONS
                portable_base_name = (
                    _resolve_portable_layout_name(composed.layout, base_name)
                    if add_body_sensors
                    else base_name
                )
                for variant in np.unique(assignment).tolist():
                    model, sensor_inventory = (
                        _materialize_motrix_expanded_scene_with_sensor_inventory(
                            model_file=sources[int(variant)],
                            add_body_sensors=add_body_sensors,
                            base_name=portable_base_name,
                        )
                    )
                    sensor_names = sensor_inventory.names
                    if add_body_sensors:
                        generated_sensors = {
                            f"{prefix}_{entity.name}/{body_name}"
                            for prefix in ("track_pos_b", "track_quat_b")
                            for entity in composed.layout.entities
                            for body_name in entity.body_names
                        }
                    else:
                        generated_sensors = set()
                    expected_sensors = set(source_sensor_contracts) | generated_sensors
                    if len(sensor_names) != len(set(sensor_names)) or set(sensor_names) != (
                        expected_sensors
                    ):
                        raise RuntimeError(
                            "Motrix portable sensors differ from the compiled public "
                            "sensor layout"
                        )
                    native_frame_identities = {
                        identity.name: (
                            identity.sensor_type,
                            identity.object_type,
                            identity.reference_frame,
                        )
                        for identity in sensor_inventory.frame_identities
                    }
                    native_contact_identities = {
                        identity.name: (
                            identity.reduce_mode,
                            identity.reports_force,
                            identity.reports_found,
                            identity.geom1,
                            identity.geom2,
                        )
                        for identity in sensor_inventory.contact_identities
                    }
                    expected_frame_contracts = {
                        name: contract
                        for name, contract in source_sensor_contracts.items()
                        if contract.sensor_kind == "frame"
                    }
                    expected_contact_contracts = {
                        name: contract
                        for name, contract in source_sensor_contracts.items()
                        if contract.sensor_kind == "contact"
                    }
                    if (
                        set(native_frame_identities) != set(expected_frame_contracts)
                        or set(native_contact_identities) != set(expected_contact_contracts)
                    ):
                        raise RuntimeError(
                            "Motrix source sensor identities differ from the compiled "
                            "public sensor layout"
                        )
                    native_source_identities = native_frame_identities | native_contact_identities
                    for name, contract in source_sensor_contracts.items():
                        if native_source_identities[name] != contract.identity:
                            raise RuntimeError(
                                f"Motrix source sensor {name!r} native identity differs "
                                "from the compiled public sensor layout"
                            )
                        self._audit_portable_site_identity(model, contract)
                    model.options.timestep = float(sim_dt)
                    model.options.max_iterations = int(max_iterations)
                    rows = np.flatnonzero(assignment == variant).astype(np.intp, copy=False)
                    data = mtx.SceneData(model, batch=[int(rows.size)])  # pyright: ignore[reportPossiblyUnbound]
                    expected_dimensions = {
                        name: contract.dimension
                        for name, contract in source_sensor_contracts.items()
                    }
                    for sensor_name in sensor_names:
                        if sensor_name in generated_sensors:
                            expected_dimension = (
                                3 if sensor_name.startswith("track_pos_b_") else 4
                            )
                        else:
                            expected_dimension = expected_dimensions[sensor_name]
                        native_values = np.asarray(model.get_sensor_value(sensor_name, data))
                        if native_values.shape != (rows.size, expected_dimension):
                            raise RuntimeError(
                                f"Motrix sensor {sensor_name!r} has shape "
                                f"{native_values.shape}, expected "
                                f"{(rows.size, expected_dimension)}"
                            )
                    binding = self._bind_portable_layout(
                        model, data, composed.layout, np_dtype=self._np_dtype
                    )
                    found_contact_geom_pairs: dict[str, tuple[int, int]] = {}
                    for identity in sensor_inventory.contact_identities:
                        if not identity.reports_found:
                            continue
                        geom1 = model.get_geom_index(identity.geom1)
                        geom2 = model.get_geom_index(identity.geom2)
                        if geom1 is None or geom2 is None:
                            raise RuntimeError(
                                f"Motrix contact sensor {identity.name!r} refers to an "
                                "absent native geom"
                            )
                        found_contact_geom_pairs[identity.name] = (int(geom1), int(geom2))
                    keyframe = (
                        None
                        if scene.default_keyframe_name is None
                        else self._portable_native_default_keyframe(
                            model, scene.default_keyframe_name
                        )
                    )
                    default_controls = self._portable_native_default_controls(
                        model,
                        data,
                        keyframe=keyframe,
                        np_dtype=self._np_dtype,
                    )
                    if keyframe is not None:
                        self._portable_apply_native_default_keyframe(
                            keyframe,
                            data,
                            default_controls=default_controls,
                            np_dtype=self._np_dtype,
                        )
                    runtimes.append(
                        _MotrixPortableRuntime(
                            variant=int(variant),
                            rows=rows,
                            model=model,
                            data=data,
                            sensor_names=sensor_names,
                            binding=binding,
                            default_controls=default_controls,
                            found_contact_geom_pairs=found_contact_geom_pairs,
                        )
                    )
                    if len(runtimes) > 1 and sensor_names != runtimes[0].sensor_names:
                        raise RuntimeError(
                            "Motrix portable sensors differ across fixed variants"
                        )
                    if (
                        len(runtimes) > 1
                        and found_contact_geom_pairs != runtimes[0].found_contact_geom_pairs
                    ):
                        raise RuntimeError(
                            "Motrix portable contact sensors differ across fixed variants"
                        )
                    if len(runtimes) > 1:
                        self._audit_portable_variant_identity(runtimes[0], runtimes[-1])
                self._portable_runtimes = tuple(runtimes)
                self._portable_variant_assignment = (
                    None if composed.variant_plan is None else assignment.copy()
                )
                primary_runtime = runtimes[0]
                scene_context = _MotrixSceneContext(
                    model=primary_runtime.model,
                    sensor_names=primary_runtime.sensor_names,
                )
            except BaseException:
                self._portable_runtimes = ()
                self._composed_scene = None
                composed.close()
                raise
        else:
            scene_context = _build_motrix_scene_context(
                scene,
                add_body_sensors=add_body_sensors,
                base_name=base_name,
            )
        self._scene = scene
        self.scene_artifacts_dir = None
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
                    else cast(Any, self.terrain_surface_sampler).sample_height
                ),
            )
        )
        self._scene_cleanup_handle = scene_context.cleanup_handle
        self._base_name = base_name

        self._model = scene_context.model
        self._sensor_names = frozenset(scene_context.sensor_names)
        self._body_id_to_name = {  # type: ignore[assignment]
            link.index: link.name for link in self._model.links if link.name
        }

        self._model.options.timestep = sim_dt
        if max_iterations is None:
            max_iterations = DEFAULT_MOTRIX_MAX_ITERATIONS
        self._model.options.max_iterations = int(max_iterations)
        if not portable_mode:
            self._num_envs = int(num_envs)
            self._np_dtype = np_dtype
            self._data = mtx.SceneData(self._model, batch=[num_envs])  # pyright: ignore[reportPossiblyUnbound]
        if portable_mode:
            primary_runtime = runtimes[0]
            self._data = primary_runtime.data
            self._portable_public_to_native_body = (
                primary_runtime.binding.public_to_native_body
            )
            self._portable_public_to_native_geom = (
                primary_runtime.binding.public_to_native_geom
            )
            self._portable_variant_geom_sizes = np.stack(
                [runtime.binding.default_geom_sizes for runtime in runtimes]
            )
            self._links_by_id = primary_runtime.binding.links_by_id
            self._geoms_by_id = primary_runtime.binding.geoms_by_id
            base_name = _resolve_portable_native_name(self._model, base_name)
        self._body: Any = _require_not_none(
            self._model.get_body(base_name), f"Body '{base_name}' not found in Motrix model"
        )
        self._body_link: Any = _require_not_none(
            self._model.get_link(base_name), f"Link '{base_name}' not found in Motrix model"
        )
        push_body = push_body_name if push_body_name is not None else base_name
        self._push_body_link: Any = _require_not_none(
            self._model.get_link(push_body), f"Push link '{push_body}' not found in Motrix model"
        )
        self._body_floatingbase = self._body.floatingbase
        self._joint_dof_pos_indices = np.asarray(self._model.joint_dof_pos_indices, dtype=np.intp)
        self._joint_dof_vel_indices = np.asarray(self._model.joint_dof_vel_indices, dtype=np.intp)
        self._joint_dof_pos_slice = _contiguous_slice(self._joint_dof_pos_indices)
        position_actuators: list[Any] = []
        for actuator in self._model.actuators:
            if actuator.typ == "position":
                position_actuators.append(actuator)
        self._position_actuators = position_actuators
        self._supports_position_actuator_gains = len(self._position_actuators) == int(
            self._model.num_actuators
        )
        # qpos index of each position actuator's own target joint, in actuator
        # order. Used to reset position actuators to "hold current pose" without
        # assuming a fully-actuated model: parallel / under-actuated mechanisms
        # (e.g. a Stewart platform) have passive joints, so the model-wide
        # ``joint_dof_pos_indices`` is wider than ``num_actuators``.
        self._actuator_joint_pos_indices: np.ndarray | None = None
        self._actuator_joint_vel_indices: np.ndarray | None = None
        if self._supports_position_actuator_gains:
            joint_pos_idx: list[int] = []
            joint_vel_idx: list[int] = []
            for actuator in sorted(self._position_actuators, key=lambda a: int(a.index)):
                if actuator.target_type != "joint":
                    joint_pos_idx = []
                    joint_vel_idx = []
                    break
                joint = self._model.get_joint(actuator.target_name)
                if joint is None or int(joint.num_dof_pos) != 1:
                    joint_pos_idx = []
                    joint_vel_idx = []
                    break
                joint_pos_idx.append(int(joint.dof_pos_index))
                joint_vel_idx.append(int(joint.dof_vel_index))
            if len(joint_pos_idx) == int(self._model.num_actuators):
                self._actuator_joint_pos_indices = np.asarray(joint_pos_idx, dtype=np.intp)
                self._actuator_joint_vel_indices = np.asarray(joint_vel_idx, dtype=np.intp)
        self._actuator_joint_pos_slice = (
            _contiguous_slice(self._actuator_joint_pos_indices)
            if self._actuator_joint_pos_indices is not None
            else None
        )
        self._default_actuator_kp = np.zeros((self.num_actuators,), dtype=np.float32)
        self._default_actuator_kd = np.zeros((self.num_actuators,), dtype=np.float32)
        for actuator in self._position_actuators:
            idx = int(actuator.index)
            # TODO: switch to motrixsim model-level actuator gain API once available.
            self._default_actuator_kp[idx] = _first_scalar(actuator.get_kp_override(self._data))
            self._default_actuator_kd[idx] = _first_scalar(actuator.get_kd_override(self._data))
        self._floating_base_quat_indices: tuple[np.ndarray, ...] = tuple(
            np.asarray(floating_base.dof_pos_indices[3:7], dtype=np.intp)
            for floating_base in getattr(self._model, "floating_bases", [])
            if len(floating_base.dof_pos_indices) >= 7
        )
        if not portable_mode:
            self._links_by_id = {
                int(link.index): link for link in self._model.links
            }
        self._supports_link_mass_override = all(
            callable(getattr(link, "set_mass_override", None))
            for link in (
                (
                    link
                    for runtime in self._portable_runtimes
                    for link in runtime.binding.links_by_id.values()
                )
                if portable_mode
                else self._links_by_id.values()
            )
        )
        self._supports_link_com_override = all(
            callable(getattr(link, "set_center_of_mass_override", None))
            for link in (
                (
                    link
                    for runtime in self._portable_runtimes
                    for link in runtime.binding.links_by_id.values()
                )
                if portable_mode
                else self._links_by_id.values()
            )
        )
        self._supports_joint_armature_override = bool(self._portable_runtimes) and all(
            bool(runtime.binding.joints_by_public_dof)
            and all(
                callable(getattr(joint, "set_armature_override", None))
                for joint in runtime.binding.joints_by_public_dof.values()
            )
            for runtime in self._portable_runtimes
        )
        self._supports_joint_frictionloss_override = bool(self._portable_runtimes) and all(
            bool(runtime.binding.joints_by_public_dof)
            and all(
                callable(getattr(joint, "set_frictionloss_override", None))
                for joint in runtime.binding.joints_by_public_dof.values()
            )
            for runtime in self._portable_runtimes
        )
        self._supports_external_force = all(
            callable(getattr(link, "add_external_force", None))
            for link in self._links_by_id.values()
        )
        self._supports_external_torque = all(
            callable(getattr(link, "add_external_torque", None))
            for link in self._links_by_id.values()
        )
        if portable_mode:
            self._supports_external_force = all(
                callable(getattr(link, "add_external_force", None))
                for runtime in runtimes
                for link in runtime.binding.links_by_id.values()
            )
            self._supports_external_torque = all(
                callable(getattr(link, "add_external_torque", None))
                for runtime in runtimes
                for link in runtime.binding.links_by_id.values()
            )
        self._applied_body_forces: dict[int, np.ndarray] = {}
        if not portable_mode:
            self._geoms_by_id = {
                int(geom.index): geom for geom in self._model.geoms
            }
        # TODO(motrixsim): once pure visual geoms either stop exposing friction
        # override methods or safely no-op them, drop this collision-mask filter.
        self._geom_friction_override_ids = tuple(
            geom_id
            for geom_id, geom in self._geoms_by_id.items()
            if (
                int(getattr(geom, "collision_group", 0)) != 0
                or int(getattr(geom, "collision_affinity", 0)) != 0
            )
        )
        self._supports_geom_friction_override = all(
            callable(getattr(geom, "get_friction_override", None))
            and callable(getattr(geom, "set_friction_override", None))
            for geom_id, geom in self._geoms_by_id.items()
            if geom_id in self._geom_friction_override_ids
        )
        self._supports_gravity_override = callable(
            getattr(self._model, "get_gravity_override", None)
        ) and callable(getattr(self._model, "set_gravity_override", None))
        if portable_mode:
            primary_runtime = runtimes[0]
            self._default_geom_friction = (
                primary_runtime.binding.default_geom_friction.copy()
            )
            assert self._composed_scene is not None
            entity_layout = self._composed_scene.layout
            self._entity_layout = entity_layout
            activation_zeros = tuple(0 for _ in range(entity_layout.nu))
            self._portable_reset_impacts = bind_reset_impacts(
                entity_layout,
                activation_zeros,
                activation_zeros,
            )
        else:
            self._default_body_mass = np.zeros((int(self._model.num_links),), dtype=np.float32)
            self._default_body_ipos = np.zeros((int(self._model.num_links), 3), dtype=np.float32)
            for link_id, link in self._links_by_id.items():
                self._default_body_mass[link_id] = _first_scalar(
                    link.get_mass_override(self._data)
                )
                self._default_body_ipos[link_id] = np.asarray(
                    link.get_center_of_mass_override(self._data),
                    dtype=np.float32,
                ).reshape(self._num_envs, 3)[0]
            self._default_geom_friction = np.zeros(
                (int(self._model.num_geoms), 3), dtype=np.float32
            )
            if self._supports_geom_friction_override:
                for geom_id in self._geom_friction_override_ids:
                    geom = self._geoms_by_id[geom_id]
                    self._default_geom_friction[geom_id] = np.asarray(
                        geom.get_friction_override(self._data),
                        dtype=np.float32,
                    ).reshape(self._num_envs, 3)[0]
        self._render_app: Any | None = None
        self._render_headless: bool | None = None
        self._render_capture_enabled = False
        self._render_offsets_np: np.ndarray | None = None
        self._render_tracking_camera: MotrixTrackingCamera | None = None
        self.backend_type = "motrix"
        self._link_velocity_cache: np.ndarray | None = None

        # Pre-cache link objects to avoid repeated get_link() lookups.
        self._link_cache: dict[int, Any] = {}
        for link in self._model.links:
            if link.name:
                self._link_cache[link.index] = link

        # Run forward kinematics once so initial link poses and sensor data are valid.
        self._link_velocities: np.ndarray | None = None
        self._link_velocity_cache_valid = False
        if portable_mode:
            for runtime in self._portable_runtimes:
                runtime.model.forward_kinematic(runtime.data)
        else:
            self._model.forward_kinematic(self._data)
        self._refresh_link_pose_cache()
        if portable_mode:
            for runtime in self._portable_runtimes:
                runtime.default_qpos = self._motrix_qpos_to_mujoco(
                    np.asarray(runtime.data.dof_pos, dtype=self._np_dtype)
                ).copy()
                runtime.default_qvel = np.asarray(
                    runtime.data.dof_vel, dtype=self._np_dtype
                ).copy()
            self._portable_default_qpos = self._portable_state_qpos()
            self._portable_default_qvel = self._portable_state_qvel()
            self._portable_default_roots = self._portable_entity_roots().copy()
            for runtime in self._portable_runtimes:
                runtime.default_roots = self._portable_default_roots[runtime.rows].copy()
            assert self._entity_layout is not None
            layout = self._entity_layout
            assignment = self._portable_variant_assignment
            if assignment is None:
                self._portable_default_body_mass = (
                    runtimes[0].binding.default_body_mass[0].copy()
                )
            else:
                self._portable_default_body_mass = np.stack(
                    [
                        next(
                            runtime
                            for runtime in self._portable_runtimes
                            if runtime.variant == int(variant)
                        ).binding.default_body_mass[0]
                        for variant in assignment
                    ]
                )
            self._portable_default_body_ipos = np.zeros(
                (self._num_envs, layout.nbody, 3), dtype=self._np_dtype
            )
            for runtime in self._portable_runtimes:
                self._portable_default_body_ipos[runtime.rows] = (
                    runtime.binding.default_body_ipos
                )
            self._portable_default_dof_armature = np.zeros(
                (self._num_envs, layout.nv), dtype=self._np_dtype
            )
            self._portable_default_dof_frictionloss = np.zeros(
                (self._num_envs, layout.nv), dtype=self._np_dtype
            )
            for runtime in self._portable_runtimes:
                runtime_armature = np.zeros((layout.nv,), dtype=self._np_dtype)
                runtime_frictionloss = np.zeros((layout.nv,), dtype=self._np_dtype)
                for public_dof, joint in runtime.binding.joints_by_public_dof.items():
                    runtime_armature[public_dof] = runtime.binding.default_dof_armature[
                        public_dof
                    ]
                    runtime_frictionloss[public_dof] = (
                        runtime.binding.default_dof_frictionloss[public_dof]
                    )
                self._portable_default_dof_armature[runtime.rows] = runtime_armature
                self._portable_default_dof_frictionloss[runtime.rows] = runtime_frictionloss

        # Scratch buffers reused by set_state() to avoid per-reset allocations.
        # Sized to the full env count and rewritten in place each call.
        self._set_state_mask_scratch: np.ndarray = np.zeros(self._num_envs, dtype=bool)
        self._set_state_qpos_motrix_scratch: np.ndarray | None = None

    @staticmethod
    def _audit_portable_source_sensor_contract(
        model: Any, layout: CompiledSceneLayout
    ) -> dict[str, _MotrixSourceSensorContract]:
        """Freeze the reviewed common-compiled source-sensor subset."""

        import mujoco

        msd = mtx.msd
        generated_names = {
            f"{prefix}_{entity.name}/{body_name}"
            for prefix in ("track_pos_b", "track_quat_b")
            for entity in layout.entities
            for body_name in entity.body_names
        }
        supported_types = {
            int(mujoco.mjtSensor.mjSENS_FRAMEPOS): (
                msd.FrameSensorType.FramePos,
                3,
                msd.FrameSensorRef.world(),
            ),
            int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): (
                msd.FrameSensorType.FrameQuat,
                4,
                msd.FrameSensorRef.world(),
            ),
            int(mujoco.mjtSensor.mjSENS_FRAMELINVEL): (
                msd.FrameSensorType.FrameLinVel,
                3,
                msd.FrameSensorRef.world(),
            ),
            int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL): (
                msd.FrameSensorType.FrameAngVel,
                3,
                msd.FrameSensorRef.world(),
            ),
            int(mujoco.mjtSensor.mjSENS_VELOCIMETER): (
                msd.FrameSensorType.FrameLinVel,
                3,
                msd.FrameSensorRef.local(),
            ),
            int(mujoco.mjtSensor.mjSENS_GYRO): (
                msd.FrameSensorType.FrameAngVel,
                3,
                msd.FrameSensorRef.local(),
            ),
        }
        contact_forms = {
            (2, 3, 1): (
                msd.ContactSensorReduce.NetForce,
                True,
                False,
                3,
            ),
            (1, 0, 1): (
                msd.ContactSensorReduce.None_,
                False,
                True,
                1,
            ),
        }
        public_geoms = {
            f"{entity.name}/{geom.name}"
            for entity in layout.entities
            for geom in entity.geoms
        }
        site_names = tuple(
            str(model.site(site_id).name) for site_id in range(int(model.nsite))
        )
        if any(not site_name for site_name in site_names) or len(set(site_names)) != len(
            site_names
        ):
            raise RuntimeError(
                "portable Motrix site sensors require unique non-empty site names"
            )
        public_site_names = set(site_names)
        contracts: dict[str, _MotrixSourceSensorContract] = {}
        for sensor_id in range(int(model.nsensor)):
            name = str(model.sensor(sensor_id).name)
            owners = [entity for entity in layout.entities if name.startswith(entity.name + "/")]
            if len(owners) > 1:
                raise NotImplementedError(
                    "Motrix portable entity source sensors must retain their owning "
                    "entity prefix"
                )
            owner = owners[0] if owners else None
            sensor_type = int(model.sensor_type[sensor_id])
            object_type = int(model.sensor_objtype[sensor_id])
            reference_type = int(model.sensor_reftype[sensor_id])
            reference_id = int(model.sensor_refid[sensor_id])
            dimension = int(model.sensor_dim[sensor_id])
            if name in generated_names:
                raise ValueError(
                    "Motrix source sensor names collide with generated tracking sensor names"
                )
            if sensor_type == int(mujoco.mjtSensor.mjSENS_CONTACT):
                intprm = (
                    int(model.sensor_intprm[sensor_id, 0]),
                    int(model.sensor_intprm[sensor_id, 1]),
                    int(model.sensor_intprm[sensor_id, 2]),
                )
                if (
                    object_type != int(mujoco.mjtObj.mjOBJ_GEOM)
                    or reference_type != int(mujoco.mjtObj.mjOBJ_GEOM)
                    or intprm not in contact_forms
                ):
                    raise NotImplementedError(
                        "Motrix portable contact sensors support only geom-pair "
                        "netforce and found forms"
                    )
                reduce_mode, reports_force, reports_found, expected_dimension = contact_forms[
                    intprm
                ]
                if dimension != expected_dimension:
                    raise RuntimeError(
                        "common portable contact sensor dimension disagrees with its form"
                    )
                geom1 = str(model.geom(int(model.sensor_objid[sensor_id])).name)
                geom2 = str(model.geom(reference_id).name)
                if geom1 not in public_geoms or geom2 not in public_geoms:
                    raise RuntimeError("common portable contact sensor target is not a public geom")
                contracts[name] = _MotrixSourceSensorContract(
                    identity=(reduce_mode, reports_force, reports_found, geom1, geom2),
                    dimension=dimension,
                    sensor_kind="contact",
                )
                continue
            if sensor_type not in supported_types or reference_type != int(
                mujoco.mjtObj.mjOBJ_UNKNOWN
            ) or reference_id != -1:
                raise NotImplementedError(
                    "Motrix portable sensors support only world-referenced "
                    "body/site FramePos/FrameQuat sensors, scene-level qualified-body/site "
                    "FrameLinVel/FrameAngVel fragments and entity-owned site "
                    "Velocimeter/Gyro sensors"
                )
            native_type, expected_dimension, reference_frame = supported_types[sensor_type]
            if dimension != expected_dimension:
                raise RuntimeError("common portable sensor dimension disagrees with its type")
            if (
                sensor_type
                in {
                    int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
                    int(mujoco.mjtSensor.mjSENS_GYRO),
                }
                and object_type != int(mujoco.mjtObj.mjOBJ_SITE)
            ):
                raise NotImplementedError(
                    "Motrix portable site motion sensors support only site targets"
                )
            if sensor_type in {
                int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
                int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
            }:
                if object_type not in {
                    int(mujoco.mjtObj.mjOBJ_BODY),
                    int(mujoco.mjtObj.mjOBJ_SITE),
                }:
                    raise NotImplementedError(
                        "Motrix portable frame-motion fragments support only body/site targets"
                    )
                if owner is not None:
                    raise NotImplementedError(
                        "Motrix portable frame-motion sensors support scene-level "
                        "fragments only"
                    )
            if object_type == int(mujoco.mjtObj.mjOBJ_SITE):
                site_name = str(model.site(int(model.sensor_objid[sensor_id])).name)
                site_owners = [
                    entity
                    for entity in layout.entities
                    if site_name.startswith(entity.name + "/")
                ]
                if len(site_owners) != 1 or (
                    owner is not None and site_owners[0].name != owner.name
                ):
                    raise NotImplementedError(
                        "Motrix portable site sensors must resolve to exactly one "
                        "public entity site; entity-owned sensors must retain owner "
                        "identity"
                    )
                if site_name not in public_site_names:
                    raise RuntimeError("common portable site sensor target is not a public site")
                site_id = int(model.sensor_objid[sensor_id])
                site_parent = str(model.body(int(model.site_bodyid[site_id])).name)
                site_pos = np.asarray(model.site_pos[site_id], dtype=np.float64).reshape(3)
                site_quat_xyzw = np.roll(
                    np.asarray(model.site_quat[site_id], dtype=np.float64).reshape(4), -1
                )
                site_pos_identity = (
                    float(site_pos[0]),
                    float(site_pos[1]),
                    float(site_pos[2]),
                )
                site_quat_identity = (
                    float(site_quat_xyzw[0]),
                    float(site_quat_xyzw[1]),
                    float(site_quat_xyzw[2]),
                    float(site_quat_xyzw[3]),
                )
                contracts[name] = _MotrixSourceSensorContract(
                    identity=(
                        native_type,
                        msd.ObjectType.site(site_name),
                        str(reference_frame),
                    ),
                    dimension=dimension,
                    sensor_kind="frame",
                    site_name=site_name,
                    site_identity=(
                        site_parent,
                        site_pos_identity,
                        site_quat_identity,
                    ),
                )
                continue

            if object_type != int(mujoco.mjtObj.mjOBJ_BODY):
                raise NotImplementedError(
                    "Motrix portable body sensors support only world-referenced "
                    "FramePos/FrameQuat forms and scene-level body "
                    "FrameLinVel/FrameAngVel fragments"
                )
            body_name = str(model.body(int(model.sensor_objid[sensor_id])).name)
            target_owner = next(
                (
                    entity
                    for entity in layout.entities
                    if body_name in {f"{entity.name}/{item}" for item in entity.body_names}
                ),
                None,
            )
            if owner is not None and target_owner is not None and target_owner.name != owner.name:
                raise NotImplementedError(
                    "Motrix portable entity source sensors must reference their owning "
                    "entity's bodies"
                )
            if target_owner is None:
                raise RuntimeError("common portable source sensor target is not a public body")
            if name in generated_names:
                raise ValueError(
                    "Motrix source sensor names collide with generated tracking sensor names"
                )
            contracts[name] = _MotrixSourceSensorContract(
                identity=(
                    native_type,
                    msd.ObjectType.link_inertia(body_name),
                    str(msd.FrameSensorRef.world()),
                ),
                dimension=dimension,
                sensor_kind="frame",
            )
        return contracts

    @staticmethod
    def _audit_portable_site_identity(
        model: Any, contract: _MotrixSourceSensorContract
    ) -> None:
        """Require a referenced native site to retain its complete public identity."""
        if contract.site_name is None or contract.site_identity is None:
            return
        site = model.get_site(contract.site_name)
        parent = site.parent_link if site is not None else None
        parent_name = None if parent is None else str(parent.name)
        if site is None or parent_name is None:
            raise RuntimeError(
                f"Motrix source sensor target site {contract.site_name!r} is missing "
                "or world-attached"
            )
        actual_identity = (
            parent_name,
            tuple(
                float(value)
                for value in np.asarray(site.local_pos, dtype=np.float64).reshape(3)
            ),
            tuple(
                float(value)
                for value in np.asarray(site.local_quat, dtype=np.float64).reshape(4)
            ),
        )
        expected_identity = contract.site_identity
        if actual_identity[0] != expected_identity[0] or not np.allclose(
            np.asarray(actual_identity[1], dtype=np.float64),
            np.asarray(expected_identity[1], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ) or not np.allclose(
            np.asarray(actual_identity[2], dtype=np.float64),
            np.asarray(expected_identity[2], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Motrix source sensor target site {contract.site_name!r} identity "
                "differs from the compiled public site layout"
            )

    @staticmethod
    def _bind_portable_layout(
        model: Any,
        data: Any,
        layout: CompiledSceneLayout,
        *,
        np_dtype=np.float32,
    ) -> _MotrixPortableBinding:
        """Audit actual Motrix metadata and bind it to frozen public addresses."""
        if int(model.num_dof_pos) != layout.nq:
            raise RuntimeError(
                f"Motrix scene qpos dimension {model.num_dof_pos} differs from "
                f"portable layout {layout.nq}"
            )
        if int(model.num_dof_vel) != layout.nv:
            raise RuntimeError(
                f"Motrix scene qvel dimension {model.num_dof_vel} differs from "
                f"portable layout {layout.nv}"
            )
        if int(model.num_actuators) != layout.nu:
            raise RuntimeError(
                f"Motrix scene actuator dimension {model.num_actuators} differs from "
                f"portable layout {layout.nu}"
            )

        links_by_id = {int(link.index): link for link in model.links}
        native_links = {str(link.name): int(link.index) for link in model.links}
        public_bodies = {
            f"{entity.name}/{body_name}": body_id
            for entity in layout.entities
            for body_name, body_id in zip(entity.body_names, entity.body_ids, strict=True)
        }
        if set(native_links) != set(public_bodies) or len(native_links) != len(public_bodies):
            missing = sorted(set(public_bodies) - set(native_links))
            extra = sorted(set(native_links) - set(public_bodies))
            raise RuntimeError(
                "Motrix native link names differ from the portable layout; "
                f"missing={missing}, extra={extra}"
            )
        public_to_native_body = np.full((layout.nbody,), -1, dtype=np.intp)
        for name, public_id in public_bodies.items():
            public_to_native_body[public_id] = native_links[name]

        native_qpos_by_public: dict[int, int] = {}
        native_qvel_by_public: dict[int, int] = {}
        joints_by_public_dof: dict[int, Any] = {}
        kinematic_mocaps: dict[int, Any] = {}
        for owner in layout.entities:
            body = model.get_body(f"{owner.name}/{owner.root_body}")
            if body is None:
                raise RuntimeError(
                    f"Motrix is missing portable root body {owner.name}/{owner.root_body}"
                )
            if owner.root_mode == "floating":
                floating_base = body.floatingbase
                if floating_base is None:
                    raise RuntimeError(
                        f"Motrix portable entity {owner.name!r} has no native floating root"
                    )
                for public, native in zip(
                    owner.root_qpos_indices,
                    floating_base.dof_pos_indices,
                    strict=True,
                ):
                    native_qpos_by_public[public] = int(native)
                for public, native in zip(
                    owner.root_qvel_indices,
                    floating_base.dof_vel_indices,
                    strict=True,
                ):
                    native_qvel_by_public[public] = int(native)
            elif body.floatingbase is not None:
                raise RuntimeError(
                    f"Motrix fixed entity {owner.name!r} unexpectedly owns a floating root"
                )
            elif owner.root_mode == "kinematic":
                if not bool(body.is_mocap) or body.mocap is None:
                    raise RuntimeError(
                        f"Motrix kinematic entity {owner.name!r} has no native mocap root"
                    )
                kinematic_mocaps[layout.entities.index(owner)] = body.mocap
                for geom in owner.geoms:
                    native_geom = model.get_geom(f"{owner.name}/{geom.name}")
                    if native_geom is None:
                        raise RuntimeError(
                            f"Motrix is missing portable mirror geom "
                            f"{owner.name}/{geom.name}"
                        )
                    if (
                        int(getattr(native_geom, "collision_group", -1)) != 0
                        or int(getattr(native_geom, "collision_affinity", -1)) != 0
                    ):
                        raise RuntimeError(
                            f"Motrix portable mirror {owner.name!r} retained collision masks"
                        )

            for joint in owner.joints:
                if joint.kind not in ("hinge", "slide"):
                    raise NotImplementedError(
                        f"Motrix portable entity {owner.name!r} supports only hinge/slide joints"
                    )
                native_joint = model.get_joint(f"{owner.name}/{joint.name}")
                if native_joint is None:
                    raise RuntimeError(
                        f"Motrix is missing portable joint {owner.name}/{joint.name}"
                    )
                if int(native_joint.num_dof_pos) != 1 or int(native_joint.num_dof_vel) != 1:
                    raise RuntimeError(
                        f"Motrix portable joint {owner.name}/{joint.name} is not scalar"
                    )
                native_qpos_by_public[joint.qpos_indices[0]] = int(native_joint.dof_pos_index)
                native_qvel_by_public[joint.qvel_indices[0]] = int(native_joint.dof_vel_index)
                joints_by_public_dof[int(joint.qvel_indices[0])] = native_joint

        public_qpos = np.arange(layout.nq, dtype=np.intp)
        public_qvel = np.arange(layout.nv, dtype=np.intp)
        native_qpos = np.asarray(
            [native_qpos_by_public[int(index)] for index in public_qpos], dtype=np.intp
        )
        native_qvel = np.asarray(
            [native_qvel_by_public[int(index)] for index in public_qvel], dtype=np.intp
        )
        if not np.array_equal(native_qpos, public_qpos) or not np.array_equal(
            native_qvel, public_qvel
        ):
            raise RuntimeError(
                "Motrix native generalized-state order differs from the portable layout"
            )

        expected_actuators = tuple(
            (
                f"{entity.name}/{actuator_name}",
                f"{entity.name}/{actuator_name}",
                f"{entity.name}/{joint_name}",
            )
            for entity in layout.entities
            for actuator_name, joint_name in zip(
                entity.actuator_names, entity.actuator_joint_names, strict=True
            )
        )
        actual_actuators = tuple(
            (
                str(actuator.name),
                str(actuator.name),
                str(actuator.target_name),
            )
            for actuator in sorted(model.actuators, key=lambda item: int(item.index))
        )
        if actual_actuators != expected_actuators:
            raise RuntimeError("Motrix native actuator order or targets differ from source")

        geoms_by_id = {int(geom.index): geom for geom in model.geoms}
        native_geoms = {str(geom.name): int(geom.index) for geom in model.geoms}
        public_geoms: dict[str, int] = {}
        public_geom_id = 0
        for entity in layout.entities:
            for geom in entity.geoms:
                public_geoms[f"{entity.name}/{geom.name}"] = public_geom_id
                public_geom_id += 1
        if set(native_geoms) != set(public_geoms):
            missing = sorted(set(public_geoms) - set(native_geoms))
            extra = sorted(set(native_geoms) - set(public_geoms))
            raise RuntimeError(
                f"Motrix native geom names differ from portable layout; missing={missing}, "
                f"extra={extra}"
            )
        public_to_native_geom = np.full((layout.ngeom,), -1, dtype=np.intp)
        for name, public_id in public_geoms.items():
            public_to_native_geom[public_id] = native_geoms[name]

        row_count = int(np.asarray(data.dof_pos).shape[0])
        default_body_mass = np.zeros((row_count, layout.nbody), dtype=np.float32)
        default_body_ipos = np.zeros((row_count, layout.nbody, 3), dtype=np.float32)
        default_dof_armature = np.zeros((layout.nv,), dtype=np_dtype)
        default_dof_frictionloss = np.zeros((layout.nv,), dtype=np_dtype)
        for public_dof, joint in joints_by_public_dof.items():
            default_dof_armature[public_dof] = float(joint.armature)
            default_dof_frictionloss[public_dof] = float(joint.frictionloss)
        for public_id, native_id in enumerate(public_to_native_body):
            if native_id < 0:
                continue
            link = links_by_id[int(native_id)]
            default_body_mass[:, public_id] = np.asarray(
                link.get_mass_override(data), dtype=np_dtype
            ).reshape(-1)
            default_body_ipos[:, public_id] = np.asarray(
                link.get_center_of_mass_override(data), dtype=np_dtype
            ).reshape(row_count, 3)

        default_geom_sizes = np.zeros((layout.ngeom, 3), dtype=np_dtype)
        for public_id, native_id in enumerate(public_to_native_geom):
            default_geom_sizes[public_id] = np.asarray(
                geoms_by_id[int(native_id)].size, dtype=np_dtype
            ).reshape(3)

        default_actuator_kp = np.zeros((layout.nu,), dtype=np.float32)
        default_actuator_kd = np.zeros((layout.nu,), dtype=np.float32)
        for actuator in model.actuators:
            if actuator.typ != "position":
                continue
            index = int(actuator.index)
            default_actuator_kp[index] = _first_scalar(actuator.get_kp_override(data))
            default_actuator_kd[index] = _first_scalar(actuator.get_kd_override(data))

        default_geom_friction = np.zeros((int(model.num_geoms), 3), dtype=np.float32)
        for geom_id, geom in geoms_by_id.items():
            if not callable(getattr(geom, "get_friction_override", None)):
                continue
            default_geom_friction[geom_id] = np.asarray(
                geom.get_friction_override(data), dtype=np_dtype
            ).reshape(row_count, 3)[0]

        return _MotrixPortableBinding(
            links_by_id=links_by_id,
            geoms_by_id=geoms_by_id,
            joints_by_public_dof=joints_by_public_dof,
            kinematic_mocaps=kinematic_mocaps,
            public_to_native_body=public_to_native_body,
            public_to_native_geom=public_to_native_geom,
            default_body_mass=default_body_mass,
            default_body_ipos=default_body_ipos,
            default_dof_armature=default_dof_armature,
            default_dof_frictionloss=default_dof_frictionloss,
            default_geom_sizes=default_geom_sizes,
            default_actuator_kp=default_actuator_kp,
            default_actuator_kd=default_actuator_kd,
            default_geom_friction=default_geom_friction,
        )

    @staticmethod
    def _audit_portable_variant_identity(
        primary: _MotrixPortableRuntime, runtime: _MotrixPortableRuntime
    ) -> None:
        """Require variants to differ only where per-env identity is exposed."""
        model = runtime.model
        if (
            int(model.num_dof_pos) != int(primary.model.num_dof_pos)
            or int(model.num_dof_vel) != int(primary.model.num_dof_vel)
            or int(model.num_actuators) != int(primary.model.num_actuators)
            or int(model.num_links) != int(primary.model.num_links)
            or int(model.num_geoms) != int(primary.model.num_geoms)
        ):
            raise RuntimeError("Motrix variant dimensions differ from the public layout")
        if not np.array_equal(
            runtime.binding.public_to_native_body,
            primary.binding.public_to_native_body,
        ) or not np.array_equal(
            runtime.binding.public_to_native_geom,
            primary.binding.public_to_native_geom,
        ):
            raise RuntimeError(
                "Motrix variant native body/geom order differs from the public layout"
            )
        if set(runtime.binding.joints_by_public_dof) != set(
            primary.binding.joints_by_public_dof
        ):
            raise RuntimeError("Motrix variant native scalar-joint mapping differs")
        if set(runtime.binding.kinematic_mocaps) != set(
            primary.binding.kinematic_mocaps
        ):
            raise RuntimeError("Motrix variant native mirror identity differs")
        for index, mocap in runtime.binding.kinematic_mocaps.items():
            primary_mocap = primary.binding.kinematic_mocaps[index]
            if (str(mocap.body.name), int(mocap.body.index)) != (
                str(primary_mocap.body.name),
                int(primary_mocap.body.index),
            ):
                raise RuntimeError("Motrix variant native mirror identity differs")
        if not np.allclose(
            np.asarray(model.actuator_ctrl_limits, dtype=np.float64),
            np.asarray(primary.model.actuator_ctrl_limits, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ) or not np.allclose(
            np.asarray(model.joint_limits, dtype=np.float64),
            np.asarray(primary.model.joint_limits, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ):
            raise NotImplementedError(
                "Motrix fixed variants do not support differing public control/joint limits"
            )
        if not np.allclose(
            runtime.binding.default_actuator_kp,
            primary.binding.default_actuator_kp,
            rtol=0.0,
            atol=0.0,
        ) or not np.allclose(
            runtime.binding.default_actuator_kd,
            primary.binding.default_actuator_kd,
            rtol=0.0,
            atol=0.0,
        ):
            raise NotImplementedError(
                "Motrix fixed variants do not support differing actuator gains"
            )
        if not np.allclose(
            runtime.binding.default_geom_friction,
            primary.binding.default_geom_friction,
            rtol=0.0,
            atol=0.0,
        ):
            raise NotImplementedError(
                "Motrix fixed variants do not support differing geom friction"
            )

    @staticmethod
    def _portable_native_default_keyframe(model: Any, keyframe_name: str) -> Any:
        """Return the uniquely named native construction keyframe."""
        keys = [key for key in model.keyframes if str(key.name) == keyframe_name]
        if len(keys) != 1:
            raise RuntimeError(
                f"Motrix native keyframe {keyframe_name!r} matched {len(keys)} records"
            )
        return keys[0]

    @staticmethod
    def _portable_native_default_controls(
        model: Any, data: Any, *, keyframe: Any | None, np_dtype: Any
    ) -> np.ndarray:
        """Capture native construction/default-key controls before runtime mutation."""
        row_count = int(np.asarray(data.dof_pos).shape[0])
        control_count = int(model.num_actuators)
        values = np.asarray(data.actuator_ctrls, dtype=np_dtype)
        if values.shape != (row_count, control_count):
            raise RuntimeError(
                "Motrix native controls have shape "
                f"{values.shape}, expected {(row_count, control_count)}"
            )
        if keyframe is not None:
            key_controls = np.asarray(keyframe.ctrl, dtype=np_dtype)
            if key_controls.shape != (control_count,):
                raise RuntimeError(
                    "Motrix native keyframe controls have shape "
                    f"{key_controls.shape}, expected {(control_count,)}"
                )
            values = np.broadcast_to(key_controls, (row_count, control_count)).copy()
        limits = np.asarray(model.actuator_ctrl_limits, dtype=np.float64)
        if limits.shape != (2, control_count):
            raise RuntimeError(
                f"Motrix native control limits have shape {limits.shape}, "
                f"expected {(2, control_count)}"
            )
        lower, upper = limits
        valid_lower = (lower == -np.inf) | np.isfinite(lower)
        valid_upper = (upper == np.inf) | np.isfinite(upper)
        if (
            not np.isfinite(values).all()
            or np.any(np.isnan(limits))
            or np.any(lower > upper)
            or not valid_lower.all()
            or not valid_upper.all()
        ):
            raise RuntimeError("Motrix native default controls or control limits are invalid")
        return np.clip(values, lower, upper).astype(np_dtype, copy=True)

    @staticmethod
    def _portable_apply_native_default_keyframe(
        keyframe: Any, data: Any, *, default_controls: np.ndarray, np_dtype: Any
    ) -> None:
        """Apply the selected native key state before construction defaults are captured."""
        row_count = int(np.asarray(data.dof_pos).shape[0])
        qpos_count = int(np.asarray(data.dof_pos).shape[1])
        qvel_count = int(np.asarray(data.dof_vel).shape[1])
        control_count = int(default_controls.shape[1])
        qpos = np.asarray(keyframe.dof_pos, dtype=np_dtype)
        qvel = np.asarray(keyframe.dof_vel, dtype=np_dtype)
        controls = np.asarray(keyframe.ctrl, dtype=np_dtype)
        if qpos.shape != (qpos_count,) or qvel.shape != (qvel_count,):
            raise RuntimeError(
                "Motrix native keyframe generalized state has shape "
                f"({qpos.shape}, {qvel.shape}), expected "
                f"({(qpos_count,)}, {(qvel_count,)})"
            )
        if controls.shape != (control_count,):
            raise RuntimeError(
                f"Motrix native keyframe controls have shape {controls.shape}, "
                f"expected {(control_count,)}"
            )
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise RuntimeError("Motrix native keyframe generalized state is invalid")
        keyframe.apply(data)
        data.actuator_ctrls = np.ascontiguousarray(default_controls, dtype=np_dtype)
        if not np.allclose(
            np.asarray(data.dof_pos, dtype=np_dtype),
            np.broadcast_to(qpos, (row_count, qpos_count)),
            rtol=0.0,
            atol=0.0,
        ) or not np.allclose(
            np.asarray(data.dof_vel, dtype=np_dtype),
            np.broadcast_to(qvel, (row_count, qvel_count)),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("Motrix failed to apply the selected native default keyframe")

    def _require_portable_healthy(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"Motrix backend is closed; cannot run {operation}")
        if self._portable_faulted:
            raise RuntimeError(
                f"Motrix backend state is faulted; cannot run {operation} after a partial "
                "native submission"
            )

    def _portable_state_qpos(self) -> np.ndarray:
        layout = self.get_scene_layout()
        values = np.empty((self._num_envs, layout.nq), dtype=self._np_dtype)
        for runtime in self._portable_runtimes:
            values[runtime.rows] = self._motrix_qpos_to_mujoco(
                np.asarray(runtime.data.dof_pos, dtype=self._np_dtype)
            )
        return values

    def _portable_state_qvel(self) -> np.ndarray:
        layout = self.get_scene_layout()
        values = np.empty((self._num_envs, layout.nv), dtype=self._np_dtype)
        for runtime in self._portable_runtimes:
            values[runtime.rows] = np.asarray(
                runtime.data.dof_vel, dtype=self._np_dtype
            )
        return values

    def _portable_sensor_value(self, name: str) -> np.ndarray:
        values: np.ndarray | None = None
        for runtime in self._portable_runtimes:
            geom_pair = runtime.found_contact_geom_pairs.get(name)
            if geom_pair is None:
                native_values = np.asarray(
                    runtime.model.get_sensor_value(name, runtime.data), dtype=self._np_dtype
                )
            else:
                # MotrixSim 0.8.2 broadcasts a found sensor's batch-wide any-contact
                # bit to every row; its public ContactQuery remains row-local.
                native_values = np.asarray(
                    runtime.model.get_contact_query(runtime.data).is_colliding(
                        np.asarray((geom_pair,), dtype=np.uint32)
                    ),
                    dtype=self._np_dtype,
                )
            if native_values.ndim < 1:
                raise RuntimeError(
                    f"Motrix sensor {name!r} returned scalar values with shape "
                    f"{native_values.shape}"
                )
            if native_values.shape[0] != runtime.rows.size:
                raise RuntimeError(
                    f"Motrix sensor {name!r} returned {native_values.shape[0]} rows for "
                    f"{runtime.rows.size} native environments"
                )
            if values is None:
                values = np.empty((self._num_envs, *native_values.shape[1:]), dtype=self._np_dtype)
            elif values.shape[1:] != native_values.shape[1:]:
                raise RuntimeError(
                    f"Motrix sensor {name!r} dimensions differ across fixed variants: "
                    f"{values.shape[1:]} and {native_values.shape[1:]}"
                )
            values[runtime.rows] = native_values
        if values is None:
            raise RuntimeError(f"Motrix sensor {name!r} has no native runtime")
        return values

    def _portable_sensor_values(self, names: tuple[str, ...]) -> np.ndarray:
        if not names:
            return np.empty((self._num_envs, 0), dtype=self._np_dtype)
        values = [
            self._portable_sensor_value(name).reshape(self._num_envs, -1) for name in names
        ]
        return np.concatenate(values, axis=1)

    def _portable_entity_roots(self) -> np.ndarray:
        layout = self.get_scene_layout()
        roots = np.zeros((self._num_envs, len(layout.entities), 13), dtype=np.float32)
        velocities = self._ensure_link_velocity_cache()
        for entity_index, entity in enumerate(layout.entities):
            root_public_id = entity.body_ids[0]
            native_id = int(self._portable_public_to_native_body[root_public_id])
            roots[:, entity_index, :3] = self._link_poses[:, native_id, :3]
            roots[:, entity_index, 3:7] = self._xyzw_to_wxyz(self._link_poses[:, native_id, 3:])
            roots[:, entity_index, 7:10] = velocities[:, native_id, :3]
            roots[:, entity_index, 10:] = velocities[:, native_id, 3:]
        return roots

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return self.get_body_ids(names)
        ids: list[int] = []
        for name in names:
            link_id = self._model.get_link_index(name)
            if link_id is None or link_id < 0:
                raise ValueError(f"Motion body '{name}' not found in Motrix model")
            # Motion datasets use MuJoCo-style body ids, where worldbody is id 0.
            ids.append(int(link_id) + 1)
        return np.array(ids, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def cpu_ids(self) -> tuple[int, ...] | None:
        """CPU block requested for MotrixSim workers, or ``None`` for the default policy.

        Worker ``i`` is pinned to ``cpu_ids[i % len(cpu_ids)]``. The value is
        the validated construction request; a pool already initialized by an
        earlier MotrixSim user in this process keeps its own mapping (a
        construction-time warning reports that conflict).
        """
        return self._cpu_ids

    # ------------------------------------------------------------------ #
    # Model properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def num_actuators(self) -> int:
        return int(self._model.num_actuators)

    @property
    def num_dof_vel(self) -> int:
        return int(len(self._joint_dof_vel_indices))

    def get_actuator_ctrl_range(self) -> np.ndarray:
        arr: np.ndarray = np.array(self._model.actuator_ctrl_limits, dtype=self._np_dtype)
        result: np.ndarray = arr.T.copy()
        return result

    def get_actuator_names(self) -> tuple[str, ...]:
        actuators = sorted(self._model.actuators, key=lambda actuator: int(actuator.index))
        names = tuple(str(actuator.name) for actuator in actuators)
        if len(names) != self.num_actuators or any(not name for name in names):
            raise NotImplementedError(
                "backend 'motrix' capability 'actuator names' requires one non-empty name "
                f"per control column; received {names}"
            )
        if len(set(names)) != len(names):
            raise NotImplementedError(
                "backend 'motrix' capability 'actuator names' requires unique names; "
                f"received {names}"
            )
        return names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        actuators = sorted(self._model.actuators, key=lambda actuator: int(actuator.index))
        names: list[str] = []
        for actuator in actuators:
            if actuator.target_type != "joint":
                raise NotImplementedError(
                    "backend 'motrix' capability 'actuator target joint' requires a joint "
                    f"transmission; actuator '{actuator.name}' targets '{actuator.target_type}'"
                )
            joint = self._model.get_joint(actuator.target_name)
            if joint is None or int(joint.num_dof_pos) != 1 or int(joint.num_dof_vel) != 1:
                raise NotImplementedError(
                    "backend 'motrix' capability 'actuator target joint' requires a "
                    f"single-DoF joint; actuator '{actuator.name}' targets "
                    f"'{actuator.target_name}'"
                )
            names.append(str(actuator.target_name))
        if len(names) != self.num_actuators:
            raise NotImplementedError(
                "backend 'motrix' capability 'actuator target joint' returned "
                f"{len(names)} targets for {self.num_actuators} actuators"
            )
        return tuple(names)

    def get_terrain_spawn_data(self) -> BackendTerrainSpawnData | None:
        return self._terrain_spawn_data

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if self._portable_mode:
            if name != "home":
                raise NotImplementedError(
                    "portable Motrix scenes expose only the construction default keyframe"
                )
            return self._portable_default_qpos.copy()
        if hasattr(self._model, "keyframes") and self._model.num_keyframes > 0:
            qpos = np.array(self._model.keyframes[0].dof_pos, dtype=self._np_dtype)
        else:
            qpos = np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)
        return self._motrix_qpos_to_mujoco(qpos)

    def get_default_qpos(self) -> np.ndarray:
        if self._portable_mode:
            return self._portable_default_qpos.copy()
        qpos = np.array(self._model.compute_init_dof_pos(), dtype=self._np_dtype)
        return self._motrix_qpos_to_mujoco(qpos)

    def get_default_dof_pos(self) -> np.ndarray:
        if self._portable_mode:
            indices = (
                self._actuator_joint_pos_indices
                if self._actuator_joint_pos_indices is not None
                else self._joint_dof_pos_indices
            )
            return np.asarray(self._portable_default_qpos[0, indices], dtype=self._np_dtype).copy()
        qpos = np.asarray(self._model.compute_init_dof_pos(), dtype=self._np_dtype)
        indices = (
            self._actuator_joint_pos_indices
            if self._actuator_joint_pos_indices is not None
            else self._joint_dof_pos_indices
        )
        return np.asarray(qpos[indices], dtype=self._np_dtype).copy()

    def get_init_qvel(self) -> np.ndarray:
        if self._portable_mode:
            return self._portable_default_qvel[0].copy()
        return np.zeros((self._model.num_dof_vel,), dtype=self._np_dtype)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self._portable_mode:
            layout = self.get_scene_layout()
            entity_name, separator, local_name = str(root_body_name).partition("/")
            if not separator:
                matches = [
                    entity for entity in layout.entities if entity.root_body == root_body_name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"portable Motrix root {root_body_name!r} matched {len(matches)} entities"
                    )
                entity = matches[0]
            else:
                entity = layout.get_entity(entity_name)
            if entity.root_mode != "floating" or local_name != entity.root_body:
                raise ValueError(
                    f"portable Motrix root {root_body_name!r} does not own a floating base"
                )
            return BackendRootStateLayout(
                qpos_indices=entity.root_qpos_indices,
                qvel_indices=entity.root_qvel_indices,
            )
        body = self._model.get_body(root_body_name)
        if body is None:
            raise ValueError(f"Body '{root_body_name}' not found in Motrix model")
        floating_base = body.floatingbase
        if floating_base is None:
            raise NotImplementedError(
                "backend 'motrix' capability 'root-state layout' requires body "
                f"'{root_body_name}' to own a floating base"
            )
        return BackendRootStateLayout(
            qpos_indices=tuple(int(index) for index in floating_base.dof_pos_indices),
            qvel_indices=tuple(int(index) for index in floating_base.dof_vel_indices),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            layout = self.get_scene_layout()
            return np.asarray(
                layout.get_body_ids(tuple(str(name) for name in names)), dtype=np.int32
            )
        ids: list[int] = []
        for name in names:
            bid = self._model.get_link_index(name)
            if bid is None or bid < 0:
                raise ValueError(f"Body '{name}' not found in Motrix model")
            ids.append(int(bid))
        return np.array(ids, dtype=np.int32)

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids: list[int] = []
        for name in names:
            sid = self._model.get_site_index(name)
            if sid is None or sid < 0:
                raise ValueError(f"Site '{name}' not found in Motrix model")
            ids.append(int(sid))
        return np.array(ids, dtype=np.int32)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            joint = self._resolve_single_dof_joint(name)
            indices.append(int(joint.dof_vel_index))
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            joint = self._resolve_single_dof_joint(name)
            indices.append(self._joint_dof_local_index(name, int(joint.dof_pos_index), pos=True))
        return np.array(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for name in names:
            joint = self._resolve_single_dof_joint(name)
            indices.append(self._joint_dof_local_index(name, int(joint.dof_vel_index), pos=False))
        return np.array(indices, dtype=np.int32)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        indices = [int(self._resolve_single_dof_joint(name).dof_pos_index) for name in names]
        return np.asarray(indices, dtype=np.int32)

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        indices = [int(self._resolve_single_dof_joint(name).dof_vel_index) for name in names]
        return np.asarray(indices, dtype=np.int32)

    @staticmethod
    def _portable_site_names(model: Any) -> tuple[str, ...]:
        names = tuple(
            str(site.name) if site.name is not None else "" for site in model.sites
        )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise RuntimeError(
                "portable Motrix site Jacobians require unique non-empty native site names"
            )
        return names

    @staticmethod
    def _select_site_jacobian(
        site: Any,
        jacobian: np.ndarray,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_dof_indices = np.asarray(site.dof_vel_indices, dtype=np.int64).reshape(-1)
        if len(np.unique(site_dof_indices)) != len(site_dof_indices):
            raise ValueError("Motrix site Jacobian contains duplicate DoF indices")
        columns_by_dof = {
            int(dof_index): column
            for column, dof_index in enumerate(site_dof_indices)
        }
        columns: list[int] = []
        for dof_index in dof_indices:
            key = int(dof_index)
            if key not in columns_by_dof:
                raise ValueError(f"DoF index {key} is not present in site Jacobian")
            columns.append(columns_by_dof[key])

        selected = jacobian[:, :, np.asarray(columns, dtype=np.intp)]
        # Motrix returns angular rows first and linear rows second.
        return selected[:, 3:6, :], selected[:, 0:3, :]

    def get_site_jacobian_w(
        self,
        site_id: int,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sid = int(site_id)
        if sid < 0 or sid >= int(self._model.num_sites):
            raise ValueError(f"site_id out of range: {sid}")

        requested = np.asarray(dof_indices, dtype=np.int64)
        dof_count = (
            int(self._entity_layout.nv)
            if self._portable_mode and self._entity_layout is not None
            else int(self._model.num_dof_vel)
        )
        if requested.ndim != 1 or np.any(requested < 0) or np.any(
            requested >= dof_count
        ):
            raise ValueError("site Jacobian DoF indices must be a one-dimensional in-range array")

        if not self._portable_mode:
            site = self._model.sites[sid]
            jac = np.asarray(site.get_jacobian(self._data), dtype=self._np_dtype)
            if jac.ndim != 3 or jac.shape[0] != self._num_envs or jac.shape[1] != 6:
                raise ValueError(
                    f"Motrix site Jacobian for site {sid} must have shape "
                    f"({self._num_envs}, 6, n), got {jac.shape}"
                )
            jacp, jacr = self._select_site_jacobian(site, jac, requested)
            if jacp.shape != (self._num_envs, 3, requested.size):
                raise ValueError("Motrix site Jacobian has an invalid selected shape")
            if not np.isfinite(jacp).all() or not np.isfinite(jacr).all():
                raise ValueError("Motrix site Jacobian contains NaN or Inf")
            return jacp, jacr

        primary_site = self._model.sites[sid]
        primary_names = self._portable_site_names(self._model)
        site_name = primary_names[sid]
        primary_parent = primary_site.parent_link
        primary_parent_name = None if primary_parent is None else str(primary_parent.name)
        primary_dofs = np.asarray(primary_site.dof_vel_indices, dtype=np.int64).reshape(-1)
        jacp = np.empty((self._num_envs, 3, requested.size), dtype=self._np_dtype)
        jacr = np.empty((self._num_envs, 3, requested.size), dtype=self._np_dtype)

        for runtime in self._portable_runtimes:
            native_names = self._portable_site_names(runtime.model)
            if set(native_names) != set(primary_names):
                raise RuntimeError(
                    "Motrix fixed-variant site names differ from the public layout"
                )
            native_sid = runtime.model.get_site_index(site_name)
            if native_sid is None or int(native_sid) < 0:
                raise RuntimeError(
                    f"Motrix fixed variant {runtime.variant} is missing site {site_name!r}"
                )
            site = runtime.model.sites[int(native_sid)]
            native_parent = site.parent_link
            native_parent_name = None if native_parent is None else str(native_parent.name)
            native_dofs = np.asarray(site.dof_vel_indices, dtype=np.int64).reshape(-1)
            if native_parent_name != primary_parent_name or not np.array_equal(
                native_dofs, primary_dofs
            ):
                raise RuntimeError(
                    f"Motrix fixed variant {runtime.variant} site {site_name!r} identity differs"
                )

            jac = np.asarray(site.get_jacobian(runtime.data), dtype=self._np_dtype)
            if (
                jac.ndim != 3
                or jac.shape[0] != runtime.rows.size
                or jac.shape[1] != 6
                or jac.shape[2] != native_dofs.size
            ):
                raise ValueError(
                    f"Motrix site Jacobian for site {site_name!r} must have shape "
                    f"({runtime.rows.size}, 6, {native_dofs.size}), got {jac.shape}"
                )
            runtime_jacp, runtime_jacr = self._select_site_jacobian(
                site, jac, requested
            )
            if (
                runtime_jacp.shape != (runtime.rows.size, 3, requested.size)
                or not np.isfinite(runtime_jacp).all()
                or not np.isfinite(runtime_jacr).all()
            ):
                raise ValueError("Motrix portable site Jacobian has invalid shape or values")
            jacp[runtime.rows] = runtime_jacp
            jacr[runtime.rows] = runtime_jacr

        return jacp, jacr

    def _resolve_single_dof_joint(self, name: str):
        jid = self._model.get_joint_index(name)
        if jid is None or jid < 0:
            raise ValueError(f"Joint '{name}' not found in Motrix model")
        joint = self._model.joints[int(jid)]
        if int(getattr(joint, "num_dof_vel", 1)) != 1:
            raise ValueError(f"Joint '{name}' is not a single-DoF joint")
        return joint

    def _joint_dof_local_index(self, name: str, model_index: int, *, pos: bool) -> int:
        all_indices = self._joint_dof_pos_indices if pos else self._joint_dof_vel_indices
        matches = np.flatnonzero(all_indices == int(model_index))
        if matches.size != 1:
            space = "qpos" if pos else "qvel"
            raise ValueError(f"Joint '{name}' {space} index {model_index} is not in joint DoFs")
        return int(matches[0])

    def get_geom_id(self, name: str) -> int:
        if self._portable_mode:
            layout = self.get_scene_layout()
            return int(layout.get_geom_ids((str(name),))[0])
        geom_id = self._model.get_geom_index(name)
        if geom_id is None or geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in Motrix model")
        return int(geom_id)

    def get_geom_size(self, name: str) -> np.ndarray:
        if self._portable_mode and self._portable_variant_assignment is not None:
            geom_id = self.get_geom_id(name)
            variant_sizes = self._portable_variant_geom_sizes
            assert variant_sizes is not None
            sizes = variant_sizes[:, geom_id]
            if not np.all(sizes == sizes[0]):
                raise NotImplementedError(
                    "Motrix fixed variants do not expose non-uniform public geometry sizes"
                )
        geom = _require_not_none(
            self._model.get_geom(name),
            f"Geom '{name}' not found in Motrix model",
        )
        return np.asarray(geom.size, dtype=np.float64).copy()

    def get_body_mass(self) -> np.ndarray:
        if self._portable_mode:
            return self._portable_default_body_mass.copy()
        return self._default_body_mass.copy()

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if self._portable_mode:
            if env_ids is None:
                return self._portable_default_body_ipos[0].copy()
            rows = selected_state_rows(env_ids, self._num_envs)
            return self._portable_default_body_ipos[rows].copy()
        if env_ids is not None:
            raise NotImplementedError("MotrixBackend does not expose per-environment body ipos")
        return self._default_body_ipos.copy()

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        if self._portable_mode:
            root_id = int(root_body_id)
            for entity in self.get_scene_layout().entities:
                body_id_by_name = dict(zip(entity.body_names, entity.body_ids, strict=True))
                if root_id not in body_id_by_name.values():
                    continue
                root_name = next(
                    name for name, body_id in body_id_by_name.items() if body_id == root_id
                )
                descendants = {root_name}
                pending = [root_name]
                while pending:
                    parent = pending.pop()
                    for body, body_parent in zip(
                        entity.body_names, entity.body_parent_names, strict=True
                    ):
                        if body_parent == parent and body not in descendants:
                            descendants.add(body)
                            pending.append(body)
                return np.asarray(
                    sorted(body_id_by_name[name] for name in descendants), dtype=np.int32
                )
            raise ValueError(f"portable Motrix body id {root_id} is not owned by an entity")
        root_id = int(root_body_id)
        if root_id < 0 or root_id >= int(self._model.num_links):
            raise ValueError(f"root_body_id out of range: {root_id}")
        if root_id != int(self._body_link.index):
            raise NotImplementedError(
                "MotrixBackend only exposes the configured base articulation subtree"
            )

        subtree_ids = {root_id}
        for link in self._model.links:
            link_id = int(link.index)
            joint_indices = getattr(link, "joint_indices", ())
            if link_id != root_id and len(joint_indices) > 0:
                subtree_ids.add(link_id)
        return np.asarray(sorted(subtree_ids), dtype=np.int32)

    def get_geom_names(self) -> tuple[str, ...]:
        if self._portable_mode:
            return tuple(
                f"{entity.name}/{geom.name}"
                for entity in self.get_scene_layout().entities
                for geom in entity.geoms
            )
        return tuple(
            str(getattr(self._geoms_by_id[geom_id], "name", "") or "")
            for geom_id in range(int(self._model.num_geoms))
        )

    def get_geom_body_ids(self) -> np.ndarray:
        if self._portable_mode:
            layout = self.get_scene_layout()
            values = np.zeros((layout.ngeom,), dtype=np.int32)
            offset = 0
            for entity in layout.entities:
                for geom in entity.geoms:
                    values[offset] = entity.body_ids[entity.body_names.index(geom.body_name)]
                    offset += 1
            return values
        body_ids = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        for geom_id in range(int(self._model.num_geoms)):
            link = getattr(self._geoms_by_id[geom_id], "link", None)
            if link is None:
                body_ids[geom_id] = -1
            else:
                body_ids[geom_id] = int(link.index)
        return body_ids

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        if self._portable_mode:
            mapping = self._portable_public_to_native_geom
            contype, conaffinity = self._native_geom_contact_masks()
            return contype[mapping], conaffinity[mapping]
        contype = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        conaffinity = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        for geom_id in range(int(self._model.num_geoms)):
            geom = self._geoms_by_id[geom_id]
            if not hasattr(geom, "collision_group") or not hasattr(geom, "collision_affinity"):
                raise NotImplementedError("Motrix geom objects do not expose contact masks")
            contype[geom_id] = int(geom.collision_group)
            conaffinity[geom_id] = int(geom.collision_affinity)
        return contype, conaffinity

    def _native_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        contype = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        conaffinity = np.zeros((int(self._model.num_geoms),), dtype=np.int32)
        for geom_id, geom in self._geoms_by_id.items():
            if not hasattr(geom, "collision_group") or not hasattr(geom, "collision_affinity"):
                raise NotImplementedError("Motrix geom objects do not expose contact masks")
            contype[geom_id] = int(geom.collision_group)
            conaffinity[geom_id] = int(geom.collision_affinity)
        return contype, conaffinity

    def get_geom_friction(self) -> np.ndarray:
        if self._portable_mode:
            if not self._supports_geom_friction_override:
                raise NotImplementedError("Motrix geom friction override is not available")
            return self._default_geom_friction[self._portable_public_to_native_geom].copy()
        if not self._supports_geom_friction_override:
            raise NotImplementedError("Motrix geom friction override is not available")
        return self._default_geom_friction.copy()

    def get_gravity(self) -> np.ndarray:
        return np.asarray(self._model.options.gravity, dtype=np.float64).copy()

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        """Return single-DoF joint limits in backend DOF order.

        Motrix stores the model-wide limits as a ``(2, num_dof)`` table,
        whereas the UniLab backend contract exposes the MuJoCo-shaped
        ``(num_dof, 2)`` table.  This is materialized once by ``Entity`` and
        never queried from a task hot path.
        """
        self._reject_named_joint_ranges(names, "joint ranges")
        raw_limits = np.asarray(self._model.joint_limits, dtype=self._np_dtype)
        if raw_limits.ndim != 2 or raw_limits.shape != (2, self.num_dof_vel):
            raise ValueError(
                "Motrix joint limits must have shape (2, num_dof); "
                f"received {raw_limits.shape} for {self.num_dof_vel} DOFs"
            )
        return np.array(raw_limits.T, copy=True)

    def get_scene_layout(self) -> CompiledSceneLayout:
        if self._entity_layout is None:
            return super().get_scene_layout()
        self._require_portable_healthy("get_scene_layout")
        return self._entity_layout

    def get_entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.get_scene_layout().entities)

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> Mapping[str, np.ndarray]:
        self._require_portable_healthy("get_entity_default_state")
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        rows = selected_state_rows(env_ids, self._num_envs)
        index = layout.entities.index(owner)
        return entity_state_snapshot(
            owner,
            self._portable_default_qpos[rows],
            self._portable_default_qvel[rows],
            self._portable_default_roots[rows, index],
        )

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        self._require_portable_healthy("get_entity_state")
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        index = layout.entities.index(owner)
        return entity_state_snapshot(
            owner,
            self._portable_state_qpos(),
            self._portable_state_qvel(),
            self._portable_entity_roots()[:, index],
        )

    def get_physics_state(self) -> np.ndarray:
        if not self._portable_mode:
            return super().get_physics_state()
        self._require_portable_healthy("get_physics_state")
        return np.concatenate((self._portable_state_qpos(), self._portable_state_qvel()), axis=1)

    def get_scene_model_file(self) -> str | None:
        if self._composed_scene is None:
            return None
        return str(self._composed_scene.model_file)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        render_app = getattr(self, "_render_app", None)
        if render_app is not None and callable(getattr(render_app, "close", None)):
            render_app.close()
        self._render_app = None
        self._portable_runtimes = ()
        self._data = None
        self._model = None
        composed = self._composed_scene
        self._composed_scene = None
        if composed is not None:
            composed.close()

    # ------------------------------------------------------------------ #
    # Simulation control                                                 #
    # ------------------------------------------------------------------ #

    def _portable_submit_controls(self, ctrl: np.ndarray) -> None:
        if self.num_actuators:
            for runtime in self._portable_runtimes:
                runtime.data.actuator_ctrls = np.ascontiguousarray(
                    ctrl[runtime.rows], dtype=self._np_dtype
                )

    def _portable_step_runtimes(self, nsteps: int) -> None:
        try:
            for runtime in self._portable_runtimes:
                if nsteps == 1:
                    runtime.model.step(runtime.data)
                else:
                    runtime.model.step_n(runtime.data, nsteps)
            # Motrix external-force submissions are additive in SceneData and
            # consumed by the first native step.  A public step therefore ends
            # the staged-wrench interval without resubmitting it for later
            # substeps.
            for pending in self._portable_pending_body_forces.values():
                pending.fill(0.0)
            for pending in self._portable_pending_body_torques.values():
                pending.fill(0.0)
        except BaseException:
            self._portable_faulted = True
            raise

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        if self._portable_mode:
            self._require_portable_healthy("step")
            ctrl_array = np.asarray(ctrl)
            if ctrl_array.shape != (self._num_envs, self.num_actuators):
                raise ValueError(
                    f"ctrl must have shape ({self._num_envs}, {self.num_actuators}), "
                    f"got {ctrl_array.shape}"
                )
            if type(nsteps) is not int or nsteps < 1:
                raise ValueError("nsteps must be a positive integer")
            if self._pre_step_control_fn is not None:
                return self._step_with_pre_step_control(ctrl_array, nsteps)
            t0 = time.perf_counter()
            self._portable_submit_controls(ctrl_array)
            set_ctrl_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            self._portable_step_runtimes(nsteps)
            physics_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            self._refresh_link_pose_cache()
            self._invalidate_link_velocity_cache()
            refresh_cache_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "timing": {
                    "set_ctrl_ms": set_ctrl_ms,
                    "physics_ms": physics_ms,
                    "refresh_cache_ms": refresh_cache_ms,
                }
            }
        if self._pre_step_control_fn is not None:
            return self._step_with_pre_step_control(ctrl, nsteps)

        t0 = time.perf_counter()
        self._data.actuator_ctrls = np.ascontiguousarray(ctrl)
        set_ctrl_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if nsteps == 1:
            self._model.step(self._data)
        else:
            self._model.step_n(self._data, nsteps)
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_link_pose_cache()
        self._invalidate_link_velocity_cache()
        refresh_cache_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timing": {
                "set_ctrl_ms": set_ctrl_ms,
                "physics_ms": physics_ms,
                "refresh_cache_ms": refresh_cache_ms,
            }
        }

    def _step_with_pre_step_control(
        self, ctrl: np.ndarray, nsteps: int
    ) -> dict[str, dict[str, float]]:
        set_ctrl_ms = 0.0
        physics_ms = 0.0
        refresh_cache_ms = 0.0

        for _ in range(nsteps):
            t0 = time.perf_counter()
            if self._portable_mode:
                native_ctrl = self._apply_pre_step_control(ctrl)
                self._portable_submit_controls(np.asarray(native_ctrl))
            else:
                native_ctrl = self._apply_pre_step_control(ctrl)
                self._data.actuator_ctrls = np.ascontiguousarray(native_ctrl)
            set_ctrl_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            if self._portable_mode:
                self._portable_step_runtimes(1)
            else:
                self._model.step(self._data)
            physics_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            self._refresh_link_pose_cache()
            self._invalidate_link_velocity_cache()
            refresh_cache_ms += (time.perf_counter() - t0) * 1000.0

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
                    f"Motrix reset randomization does not support terms: {sorted(unsupported)}"
                )
        if self._portable_mode:
            return self._portable_set_state(env_indices, qpos, qvel, randomization=randomization)
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
        outer_t0 = time.perf_counter()

        # Pre-convert env_indices once; every downstream helper reuses this.
        env_ids_intp = np.asarray(env_indices, dtype=np.intp)

        t0 = time.perf_counter()
        # Reuse the scratch qpos buffer when its shape matches; the reset path
        # feeds a fixed-shape (num_envs, qpos_dim) array so this holds after
        # the first call.
        scratch = self._set_state_qpos_motrix_scratch
        if scratch is None or scratch.shape != qpos.shape or scratch.dtype != qpos.dtype:
            scratch = np.empty_like(qpos)
            self._set_state_qpos_motrix_scratch = scratch
        qpos_motrix = self._mujoco_qpos_to_motrix_into(scratch, qpos)
        timing["set_state_qpos_convert_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        # Reuse the persistent bool-mask scratch; only the touched entries need
        # rewriting each call.
        mask = self._set_state_mask_scratch
        if mask.shape[0] != self._num_envs:
            mask = np.zeros(self._num_envs, dtype=bool)
            self._set_state_mask_scratch = mask
        mask.fill(False)
        mask[env_ids_intp] = True
        timing["set_state_mask_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        data_slice = self._data[mask]
        timing["set_state_data_slice_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        data_slice.reset(self._model)
        timing["set_state_data_reset_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._clear_applied_body_forces(env_indices, env_ids_intp=env_ids_intp)
        timing["set_state_clear_forces_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._apply_reset_randomization(
            data_slice, env_indices, randomization, env_ids_intp=env_ids_intp
        )
        timing["set_state_reset_rand_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        data_slice.set_dof_vel(qvel)
        timing["set_state_set_dof_vel_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        data_slice.set_dof_pos(qpos_motrix, self._model)
        timing["set_state_set_dof_pos_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self._supports_position_actuator_gains and len(self._joint_dof_pos_indices) == int(
            self.num_actuators
        ):
            # Fully-actuated model: hold every joint at its reset position (unchanged).
            if self._joint_dof_pos_slice is not None:
                ctrl = qpos_motrix[:, self._joint_dof_pos_slice]
            else:
                ctrl = qpos_motrix[:, self._joint_dof_pos_indices]
        elif self._actuator_joint_pos_indices is not None:
            # Under-actuated / parallel model: hold only the actuated joints.
            if self._actuator_joint_pos_slice is not None:
                ctrl = qpos_motrix[:, self._actuator_joint_pos_slice]
            else:
                ctrl = qpos_motrix[:, self._actuator_joint_pos_indices]
        else:
            ctrl = np.zeros((len(env_indices), self.num_actuators), dtype=self._np_dtype)
        # Only pay the copy when the underlying slice is non-contiguous; view
        # slices of a contiguous scratch buffer already satisfy the motrixsim
        # contiguous requirement.
        if not ctrl.flags.c_contiguous:
            ctrl = np.ascontiguousarray(ctrl)
        data_slice.actuator_ctrls = ctrl
        timing["set_state_actuator_ctrl_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._model.forward_kinematic(data_slice)
        timing["set_state_forward_kinematic_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_link_pose_cache(env_indices, data_slice=data_slice, env_ids_intp=env_ids_intp)
        timing["set_state_refresh_pose_cache_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._invalidate_link_velocity_cache()
        timing["set_state_invalidate_velocity_ms"] = (time.perf_counter() - t0) * 1000.0

        outer_total_ms = (time.perf_counter() - outer_t0) * 1000.0
        measured_ms = (
            timing["set_state_qpos_convert_ms"]
            + timing["set_state_mask_ms"]
            + timing["set_state_data_slice_ms"]
            + timing["set_state_data_reset_ms"]
            + timing["set_state_clear_forces_ms"]
            + timing["set_state_geom_overrides_ms"]
            + timing["set_state_reset_rand_ms"]
            + timing["set_state_set_dof_vel_ms"]
            + timing["set_state_set_dof_pos_ms"]
            + timing["set_state_actuator_ctrl_ms"]
            + timing["set_state_forward_kinematic_ms"]
            + timing["set_state_refresh_pose_cache_ms"]
            + timing["set_state_invalidate_velocity_ms"]
        )
        timing["set_state_internal_gap_ms"] = outer_total_ms - measured_ms
        return {"timing": timing}

    def _portable_control_hold(self, qpos_motrix: np.ndarray) -> np.ndarray:
        indices = self._actuator_joint_pos_indices
        if indices is not None:
            values = qpos_motrix[:, indices]
            return values if values.flags.c_contiguous else np.ascontiguousarray(values)
        return np.zeros((qpos_motrix.shape[0], self.num_actuators), dtype=self._np_dtype)

    def _portable_commit_rows(
        self,
        rows: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        controls: np.ndarray | None,
        roots: np.ndarray | None = None,
        root_mask: np.ndarray | None = None,
    ) -> None:
        layout = self.get_scene_layout()
        if (roots is None) != (root_mask is None):
            raise ValueError("kinematic roots and root mask must be supplied together")
        if roots is not None and (
            roots.ndim != 3
            or roots.shape != (rows.size, len(layout.entities), 13)
            or root_mask is None
            or root_mask.shape != (len(layout.entities), 2)
        ):
            raise ValueError("kinematic root write plan differs from the portable layout")

        qpos_motrix = self._mujoco_qpos_to_motrix(qpos)
        try:
            assignment = self._portable_variant_assignment
            row_variants = (
                np.zeros(rows.shape, dtype=np.int32)
                if assignment is None
                else assignment[rows]
            )
            for runtime in self._portable_runtimes:
                selected = np.flatnonzero(row_variants == runtime.variant)
                if selected.size == 0:
                    continue
                public_rows = rows[selected]
                local_rows = runtime.local_rows(public_rows)
                data_slice = runtime.data[mtx.DisjointIndices(local_rows)]
                if controls is not None and controls.shape[1]:
                    data_slice.actuator_ctrls = np.ascontiguousarray(
                        controls[selected], dtype=self._np_dtype
                    )
                if roots is not None and root_mask is not None:
                    for entity_index, mocap in runtime.binding.kinematic_mocaps.items():
                        if not root_mask[entity_index, 0]:
                            continue
                        native_pose = np.asarray(
                            roots[selected, entity_index, :7], dtype=np.float32
                        ).copy()
                        # Public roots use wxyz; Motrix mocap pose storage is xyzw.
                        native_pose[:, 3:] = native_pose[:, 3:][:, [1, 2, 3, 0]]
                        mocap.set_pose(data_slice, np.ascontiguousarray(native_pose))
                data_slice.set_dof_pos(qpos_motrix[selected], runtime.model)
                data_slice.set_dof_vel(
                    np.ascontiguousarray(qvel[selected], dtype=self._np_dtype)
                )
                if runtime.sensor_names:
                    # Frame-sensor storage belongs to the full SceneData context
                    # and is not refreshed by forwarding only a disjoint view.
                    # Refresh the owning context after the selected write;
                    # generalized state outside the slice remains untouched.
                    runtime.model.forward_kinematic(runtime.data)
                else:
                    runtime.model.forward_kinematic(data_slice)
                self._link_poses[public_rows] = runtime.model.get_link_poses(data_slice)
            self._invalidate_link_velocity_cache()
        except BaseException:
            self._portable_faulted = True
            raise

    def _portable_set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        randomization: ResetRandomizationPayload | None,
    ) -> dict | None:
        self._require_portable_healthy("set_state")
        portable_randomization = (
            self._prepare_portable_reset_randomization(
                randomization,
                np.asarray(env_indices, dtype=np.intp),
            )
            if randomization is not None and not randomization.is_empty()
            else None
        )
        rows = np.asarray(env_indices, dtype=np.intp)
        qpos_rows = np.asarray(qpos, dtype=self._np_dtype)
        qvel_rows = np.asarray(qvel, dtype=self._np_dtype)
        layout = self.get_scene_layout()
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be one-dimensional and in [0, {self._num_envs})")
        if qpos_rows.shape != (rows.size, layout.nq):
            raise ValueError(f"qpos must have shape ({rows.size}, {layout.nq})")
        if qvel_rows.shape != (rows.size, layout.nv):
            raise ValueError(f"qvel must have shape ({rows.size}, {layout.nv})")
        controls = self._portable_control_hold(self._mujoco_qpos_to_motrix(qpos_rows))
        self._clear_applied_body_forces(rows)
        self._portable_commit_rows(rows, qpos_rows, qvel_rows, controls=controls)
        if portable_randomization is not None:
            self._apply_portable_reset_randomization(portable_randomization, rows)
        return {"timing": {}}

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if not self._portable_mode:
            super().reset(env_ids)
            return
        self._require_portable_healthy("reset")
        rows = (
            np.arange(self._num_envs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp)
        )
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError("env_ids must be a one-dimensional in-range index array")
        self._clear_applied_body_forces(rows)
        self._portable_commit_rows(
            rows,
            self._portable_default_qpos[rows],
            self._portable_default_qvel[rows],
            controls=self._portable_default_controls()[rows],
            roots=self._portable_default_roots[rows],
            root_mask=self._portable_default_root_mask(),
        )

    def _portable_current_controls(self) -> np.ndarray:
        controls = np.empty((self._num_envs, self.num_actuators), dtype=self._np_dtype)
        for runtime in self._portable_runtimes:
            controls[runtime.rows] = np.asarray(
                runtime.data.actuator_ctrls, dtype=self._np_dtype
            )
        return controls

    def _portable_default_controls(self) -> np.ndarray:
        controls = np.empty((self._num_envs, self.num_actuators), dtype=self._np_dtype)
        for runtime in self._portable_runtimes:
            controls[runtime.rows] = runtime.default_controls
        return controls

    @staticmethod
    def _portable_reset_control_columns(binding: BoundSceneReset) -> tuple[int, ...]:
        columns: set[int] = set()
        for item in binding.patches:
            root_changed = (
                item.patch.root_pose is not None or item.patch.root_velocity is not None
            )
            joint_names = {joint.name for joint in item.joints}
            columns.update(
                control
                for control, target in zip(
                    item.entity.actuator_indices,
                    item.entity.actuator_joint_names,
                    strict=True,
                )
                if root_changed or target in joint_names
            )
        return tuple(sorted(columns))

    def reset_entities(self, request: Any) -> None:
        if not self._portable_mode:
            super().reset_entities(request)
            return
        self._require_portable_healthy("reset_entities")
        layout = self.get_scene_layout()
        prepared = prepare_scene_reset(
            layout,
            request,
            self._portable_state_qpos(),
            self._portable_state_qvel(),
            self._portable_entity_roots(),
        )
        controls = None
        if request.restore_default_controls:
            columns = self._portable_reset_control_columns(prepared.binding)
            if columns:
                rows = prepared.env_ids.astype(np.intp, copy=False)
                column_array = np.asarray(columns, dtype=np.intp)
                controls = self._portable_current_controls()[rows]
                controls[:, column_array] = self._portable_default_controls()[rows][
                    :, column_array
                ]
        impact = self._portable_reset_impacts.select(prepared.binding)
        rows = prepared.env_ids.astype(np.intp, copy=False)
        self._clear_applied_body_forces(rows, env_ids_intp=rows, body_ids=impact.bodies)
        self._portable_commit_rows(
            rows,
            prepared.qpos,
            prepared.qvel,
            controls=controls,
            roots=prepared.roots,
            root_mask=prepared.root_mask,
        )

    def _portable_default_root_mask(self) -> np.ndarray:
        layout = self.get_scene_layout()
        mask = np.zeros((len(layout.entities), 2), dtype=np.uint8)
        for index, entity in enumerate(layout.entities):
            if entity.root_mode == "kinematic":
                mask[index, 0] = 1
        return mask

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        if self._portable_mode:
            supported_interval_terms: frozenset[str] = frozenset()
            supported_reset_terms: set[str] = set()
            if self._supports_link_mass_override:
                supported_reset_terms |= {RESET_TERM_BODY_MASS, RESET_TERM_BASE_MASS}
            if self._supports_link_com_override:
                supported_reset_terms |= {RESET_TERM_BODY_IPOS, RESET_TERM_BASE_COM}
            if self._supports_joint_armature_override:
                supported_reset_terms.add(RESET_TERM_DOF_ARMATURE)
            if self._supports_joint_frictionloss_override:
                supported_reset_terms.add(RESET_TERM_DOF_FRICTIONLOSS)
            if self._supports_external_force:
                supported_interval_terms |= {INTERVAL_TERM_BODY_FORCE}
            if self._supports_external_force and self._supports_external_torque:
                supported_interval_terms |= {INTERVAL_TERM_BODY_TORQUE}
            return DomainRandomizationCapabilities(
                supports_interval_push=False,
                supports_interval_body_velocity_delta=False,
                supports_interval_body_force=self._supports_external_force,
                supports_interval_body_torque=(
                    self._supports_external_force and self._supports_external_torque
                ),
                supports_fixed_variants=self._portable_variant_assignment is not None,
                supported_fixed_variant_layouts=(
                    frozenset({FixedVariantLayout.SAME_LAYOUT})
                    if self._portable_variant_assignment is not None
                    else frozenset()
                ),
                supported_interval_terms=supported_interval_terms,
                supported_reset_terms=frozenset(supported_reset_terms),
            )
        supported_reset_terms = {
            RESET_TERM_BASE_MASS,
            RESET_TERM_BASE_COM,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
        }
        if getattr(self, "_supports_position_actuator_gains", False):
            supported_reset_terms.update({RESET_TERM_KP, RESET_TERM_KD})
        if getattr(self, "_supports_geom_friction_override", False):
            supported_reset_terms.add(RESET_TERM_GEOM_FRICTION)
        if getattr(self, "_supports_gravity_override", False):
            supported_reset_terms.add(RESET_TERM_GRAVITY)
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(supported_reset_terms),
            supports_interval_push=True,
            supports_interval_body_velocity_delta=False,
            supports_interval_body_force=getattr(self, "_supports_external_force", False),
            supported_interval_terms=frozenset(
                {INTERVAL_TERM_PUSH}
                | (
                    {INTERVAL_TERM_BODY_FORCE}
                    if getattr(self, "_supports_external_force", False)
                    else set()
                )
            ),
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return the Motrix default table for a curated reset term."""
        portable_reset_terms = {
            RESET_TERM_BASE_MASS,
            RESET_TERM_BASE_COM,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_DOF_FRICTIONLOSS,
        }
        if self._portable_mode and term not in portable_reset_terms:
            raise NotImplementedError(f"MotrixBackend does not support reset term {term!r}")
        if self._portable_mode and not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(f"MotrixBackend does not support reset term {term!r}")
        if self._portable_mode:
            if term == RESET_TERM_BASE_MASS:
                value = np.zeros((self._num_envs,), dtype=np.float64)
            elif term == RESET_TERM_BODY_MASS:
                value = self._portable_default_body_mass
            elif term == RESET_TERM_BASE_COM:
                value = np.zeros((self._num_envs, 3), dtype=np.float64)
            elif term == RESET_TERM_BODY_IPOS:
                value = self._portable_default_body_ipos
            elif term == RESET_TERM_DOF_ARMATURE:
                value = self._portable_default_dof_armature
            elif term == RESET_TERM_DOF_FRICTIONLOSS:
                value = self._portable_default_dof_frictionloss
            else:
                raise NotImplementedError(f"MotrixBackend does not support reset term {term!r}")
            result = np.array(value, dtype=np.float64, copy=True)
            result.setflags(write=False)
            return result
        if term in (RESET_TERM_BASE_MASS, RESET_TERM_BASE_COM):
            value = np.zeros(() if term == RESET_TERM_BASE_MASS else (3,), dtype=np.float64)
        elif term == RESET_TERM_BODY_MASS:
            value = self.get_body_mass().astype(np.float64, copy=False)
        elif term == RESET_TERM_BODY_IPOS:
            value = self.get_body_ipos().astype(np.float64, copy=False)
        elif term == RESET_TERM_KP:
            value = self.get_actuator_gains()[0].astype(np.float64, copy=False)
        elif term == RESET_TERM_KD:
            value = self.get_actuator_gains()[1].astype(np.float64, copy=False)
        elif term == RESET_TERM_GEOM_FRICTION:
            value = self.get_geom_friction().astype(np.float64, copy=False)
        elif term == RESET_TERM_GRAVITY:
            value = self.get_gravity().astype(np.float64, copy=False)
        else:
            raise NotImplementedError(f"MotrixBackend does not expose reset term {term!r}")
        result = np.array(value, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def _portable_base_body_public_id(self) -> int:
        matches = np.flatnonzero(
            self._portable_public_to_native_body == int(self._body_link.index)
        )
        if matches.size != 1:
            raise RuntimeError(
                f"portable Motrix base link {self._body_link.name!r} matched "
                f"{matches.size} public bodies"
            )
        return int(matches[0])

    def _prepare_portable_reset_randomization(
        self,
        randomization: ResetRandomizationPayload,
        rows: np.ndarray,
    ) -> _MotrixPortableResetRandomization:
        body_mass: np.ndarray | None = None
        body_ipos: np.ndarray | None = None
        dof_armature: np.ndarray | None = None
        dof_frictionloss: np.ndarray | None = None

        if randomization.body_mass is not None or randomization.base_mass_delta is not None:
            if randomization.body_mass is None:
                mass_values = self._portable_default_body_mass[rows].copy()
            else:
                mass_values = np.asarray(randomization.body_mass, dtype=np.float32)
                mass_expected = (rows.size, self.get_scene_layout().nbody)
                if mass_values.shape != mass_expected:
                    raise ValueError(
                        f"body_mass must have shape {mass_expected}, got {mass_values.shape}"
                    )
                mass_values = mass_values.copy()
            if randomization.base_mass_delta is not None:
                delta = np.asarray(randomization.base_mass_delta, dtype=np.float32).reshape(-1)
                delta_expected = (rows.size,)
                if delta.shape != delta_expected:
                    raise ValueError(
                        f"base_mass_delta must have shape {delta_expected}, got {delta.shape}"
                    )
                if not np.isfinite(delta).all():
                    raise ValueError("base_mass_delta must contain only finite values")
                mass_values[:, self._portable_base_body_public_id()] += delta
            if not np.isfinite(mass_values).all():
                raise ValueError("body_mass must contain only finite values")
            unmapped_bodies = np.flatnonzero(self._portable_public_to_native_body < 0)
            if unmapped_bodies.size and not np.array_equal(
                mass_values[:, unmapped_bodies],
                self._portable_default_body_mass[rows][:, unmapped_bodies],
            ):
                raise ValueError(
                    "body_mass cannot randomize public columns without native Motrix links"
                )
            body_mass = mass_values

        if randomization.body_ipos is not None or randomization.base_com_offset is not None:
            if randomization.body_ipos is None:
                ipos_values = self._portable_default_body_ipos[rows].copy()
            else:
                ipos_values = np.asarray(randomization.body_ipos, dtype=np.float32)
                ipos_expected = (rows.size, self.get_scene_layout().nbody, 3)
                if ipos_values.shape != ipos_expected:
                    raise ValueError(
                        f"body_ipos must have shape {ipos_expected}, got {ipos_values.shape}"
                    )
                ipos_values = ipos_values.copy()
            if randomization.base_com_offset is not None:
                delta = np.asarray(randomization.base_com_offset, dtype=np.float32)
                com_delta_expected = (rows.size, 3)
                if delta.shape != com_delta_expected:
                    raise ValueError(
                        f"base_com_offset must have shape {com_delta_expected}, got {delta.shape}"
                    )
                if not np.isfinite(delta).all():
                    raise ValueError("base_com_offset must contain only finite values")
                ipos_values[:, self._portable_base_body_public_id(), :] += delta
            if not np.isfinite(ipos_values).all():
                raise ValueError("body_ipos must contain only finite values")
            unmapped_bodies = np.flatnonzero(self._portable_public_to_native_body < 0)
            if unmapped_bodies.size and not np.array_equal(
                ipos_values[:, unmapped_bodies, :],
                self._portable_default_body_ipos[rows][:, unmapped_bodies, :],
            ):
                raise ValueError(
                    "body_ipos cannot randomize public columns without native Motrix links"
                )
            body_ipos = ipos_values

        portable_dof_terms = (
            (RESET_TERM_DOF_ARMATURE, "dof_armature"),
            (RESET_TERM_DOF_FRICTIONLOSS, "dof_frictionloss"),
        )
        for term, attribute in portable_dof_terms:
            value = getattr(randomization, term)
            if value is None:
                continue
            values = np.asarray(value, dtype=np.float32)
            expected = (rows.size, self.get_scene_layout().nv)
            if values.shape != expected:
                raise ValueError(f"{term} must have shape {expected}, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{term} must contain only finite values")
            if np.any(values < 0.0):
                raise ValueError(f"{term} must contain only non-negative values")
            defaults = getattr(self, f"_portable_default_{term}")
            unmapped_dofs = np.ones(expected[1], dtype=bool)
            for runtime in self._portable_runtimes:
                unmapped_dofs[list(runtime.binding.joints_by_public_dof)] = False
            if unmapped_dofs.any() and not np.array_equal(
                values[:, unmapped_dofs],
                defaults[rows][:, unmapped_dofs],
            ):
                raise ValueError(
                    f"{term} cannot randomize public columns without native Motrix scalar joints"
                )
            if attribute == "dof_armature":
                dof_armature = values.copy()
            else:
                dof_frictionloss = values.copy()

        if (
            body_mass is None
            and body_ipos is None
            and dof_armature is None
            and dof_frictionloss is None
        ):
            raise ValueError("Motrix portable reset randomization contains no supported values")

        return _MotrixPortableResetRandomization(
            body_mass=body_mass,
            body_ipos=body_ipos,
            dof_armature=dof_armature,
            dof_frictionloss=dof_frictionloss,
        )

    def _apply_portable_reset_randomization(
        self,
        values: _MotrixPortableResetRandomization,
        rows: np.ndarray,
    ) -> None:
        body_mass = values.body_mass
        body_ipos = values.body_ipos
        dof_armature = values.dof_armature
        dof_frictionloss = values.dof_frictionloss
        try:
            assignment = self._portable_variant_assignment
            row_variants = (
                np.zeros(rows.shape, dtype=np.int32)
                if assignment is None
                else assignment[rows]
            )
            for runtime in self._portable_runtimes:
                selected = np.flatnonzero(row_variants == runtime.variant)
                if selected.size == 0:
                    continue
                data_slice = runtime.data[
                    mtx.DisjointIndices(runtime.local_rows(rows[selected]))
                ]
                for public_body_id, native_body_id in enumerate(
                    runtime.binding.public_to_native_body
                ):
                    if native_body_id < 0:
                        continue
                    link = runtime.binding.links_by_id[int(native_body_id)]
                    if body_mass is not None:
                        link.set_mass_override(
                            data_slice,
                            np.ascontiguousarray(
                                body_mass[selected, public_body_id],
                                dtype=np.float32,
                            ),
                        )
                    if body_ipos is not None:
                        link.set_center_of_mass_override(
                            data_slice,
                            np.ascontiguousarray(
                                body_ipos[selected, public_body_id, :],
                                dtype=np.float32,
                            ),
                        )
                    if dof_armature is not None or dof_frictionloss is not None:
                        for public_dof, joint in runtime.binding.joints_by_public_dof.items():
                            if dof_armature is not None:
                                joint.set_armature_override(
                                    data_slice,
                                    np.ascontiguousarray(
                                        dof_armature[selected, public_dof],
                                        dtype=np.float32,
                                    ),
                                )
                            if dof_frictionloss is not None:
                                joint.set_frictionloss_override(
                                    data_slice,
                                    np.ascontiguousarray(
                                        dof_frictionloss[selected, public_dof],
                                        dtype=np.float32,
                                    ),
                                )
        except BaseException:
            self._portable_faulted = True
            raise

    _interval_term_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        # Built lazily once.  Angular-velocity and linear-velocity terms have
        # no handler and fail closed in the base dispatch.  Wrench terms stay
        # gated inside ``apply_body_force`` on the runtime native API probes.
        if self._interval_term_handler_cache is None:
            self._interval_term_handler_cache = {
                INTERVAL_TERM_PUSH: lambda op: self.push_robots(op.payload),
                INTERVAL_TERM_BODY_FORCE: lambda op: self.apply_body_force(
                    require_op_body_ids(op), op.payload
                ),
            }
            if (
                self._portable_mode
                and self._supports_external_force
                and self._supports_external_torque
            ):

                def apply_body_torque(op: IntervalTermOp) -> None:
                    ids = require_op_body_ids(op)
                    self.apply_body_force(
                        ids,
                        np.zeros((self._num_envs, len(ids), 3), dtype=np.float32),
                        torque=op.payload,
                    )

                self._interval_term_handler_cache[INTERVAL_TERM_BODY_TORQUE] = apply_body_torque
        return self._interval_term_handler_cache

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "interactive" if mode == "auto" else mode
        if effective_mode == "none":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if effective_mode == "interactive":
            return BackendPlayRenderPlan(
                mode=effective_mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        assert effective_mode == "record"
        if play_steps is None:
            raise ValueError("Motrix record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("Motrix record playback requires an output video path.")
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
        debug_overlay_getter=None,
        on_frame=None,
    ) -> str | None:
        del frame_state_getter
        if debug_overlay_getter is not None:
            raise unsupported_debug_overlay_error(self.__class__.__name__)
        if on_frame is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} renders through a native renderer and "
                "does not support on_frame callbacks"
            )
        if self._portable_mode and self._portable_variant_assignment is not None:
            raise NotImplementedError(
                "Motrix native playback does not support fixed-variant scenes yet"
            )
        camera = CameraCfg.from_kwargs(camera_kwargs)
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        try:
            return run_motrix_playback(
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
            )
        except RenderClosedError:
            if not should_run_headless and not should_record_video:
                logger.info("Render window closed.")
                return None
            raise

    # ------------------------------------------------------------------ #
    # Base kinematics                                                    #
    # ------------------------------------------------------------------ #

    def get_base_pos(self) -> np.ndarray:
        if self._portable_mode:
            native_id = int(self._body_link.index)
            return self._link_poses[:, native_id, :3].copy()
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_translation(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_pose(self._data)[:, :3]  # type: ignore[no-any-return]

    def get_base_quat(self) -> np.ndarray:
        if self._portable_mode:
            native_id = int(self._body_link.index)
            return self._xyzw_to_wxyz(self._link_poses[:, native_id, 3:]).copy()
        if self._body_floatingbase is not None:
            quat = self._body_floatingbase.get_rotation(self._data)
        else:
            quat = self._body_link.get_rotation(self._data)
        return self._xyzw_to_wxyz(quat)

    def get_base_lin_vel(self) -> np.ndarray:
        if self._portable_mode:
            native_id = int(self._body_link.index)
            return self._ensure_link_velocity_cache()[:, native_id, :3].copy()
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_global_linear_velocity(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_linear_velocity(self._data)  # type: ignore[no-any-return]

    def get_base_ang_vel(self) -> np.ndarray:
        if self._portable_mode:
            native_id = int(self._body_link.index)
            return self._ensure_link_velocity_cache()[:, native_id, 3:].copy()
        if self._body_floatingbase is not None:
            return self._body_floatingbase.get_global_angular_velocity(self._data)  # type: ignore[no-any-return]
        return self._body_link.get_angular_velocity(self._data)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # DOF state                                                          #
    # ------------------------------------------------------------------ #

    def get_dof_pos(self) -> np.ndarray:
        indices = (
            self._actuator_joint_pos_indices
            if self._actuator_joint_pos_indices is not None
            else self._joint_dof_pos_indices
        )
        if self._portable_mode:
            return self._portable_state_qpos()[:, indices].copy()
        result = self._data.dof_pos[..., indices]  # type: ignore[no-any-return]
        return result

    def get_dof_vel(self) -> np.ndarray:
        indices = (
            self._actuator_joint_vel_indices
            if self._actuator_joint_vel_indices is not None
            else self._joint_dof_vel_indices
        )
        if self._portable_mode:
            return self._portable_state_qvel()[:, indices].copy()
        return self._data.dof_vel[..., indices]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                      #
    # ------------------------------------------------------------------ #

    def _as_body_ids(self, body_ids: np.ndarray) -> np.ndarray:
        values = np.asarray(body_ids, dtype=np.int32)
        if self._portable_mode:
            native_values = self._portable_public_to_native_body[values]
            if np.any(native_values < 0):
                raise ValueError(
                    f"public body ids contain unowned rows: {values[native_values < 0].tolist()}"
                )
            return native_values.astype(np.int32, copy=False)
        return values

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_poses_w(body_ids)[:, :, :3]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._xyzw_to_wxyz(self._get_link_poses_w(body_ids)[:, :, 3:])

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.intp)
        poses_w = self._link_poses[rows[:, None], self._as_body_ids(body_ids), :]
        return poses_w[:, :, :3], self._xyzw_to_wxyz(poses_w[:, :, 3:])

    def get_body_pose_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = self._get_link_poses_w(body_ids)
        return poses[:, :, :3], self._xyzw_to_wxyz(poses[:, :, 3:])

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_lin_vel_w(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_link_ang_vel_w(body_ids)

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        poses_w = self._get_link_poses_w(body_ids)
        lin_vel_w, ang_vel_w = self.get_body_vel_w(body_ids)
        return (
            poses_w[:, :, :3],
            self._xyzw_to_wxyz(poses_w[:, :, 3:]),
            lin_vel_w,
            ang_vel_w,
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids = self._as_body_ids(body_ids)
        poses_w = self._get_link_poses_w(ids)
        out_pos[..., 0] = poses_w[..., 0]
        out_pos[..., 1] = poses_w[..., 1]
        out_pos[..., 2] = poses_w[..., 2]
        out_quat[..., 0] = poses_w[..., 6]
        out_quat[..., 1] = poses_w[..., 3]
        out_quat[..., 2] = poses_w[..., 4]
        out_quat[..., 3] = poses_w[..., 5]

        link_velocity_cache = self._ensure_link_velocity_cache()
        if self._link_velocity_cache is None or self._link_velocity_cache.shape != (
            self._num_envs,
            len(ids),
            6,
        ):
            self._link_velocity_cache = np.empty(
                (self._num_envs, len(ids), 6), dtype=self._np_dtype
            )
        np.take(link_velocity_cache, ids, axis=1, out=self._link_velocity_cache)
        out_lin_vel[..., 0] = self._link_velocity_cache[..., 0]
        out_lin_vel[..., 1] = self._link_velocity_cache[..., 1]
        out_lin_vel[..., 2] = self._link_velocity_cache[..., 2]
        out_ang_vel[..., 0] = self._link_velocity_cache[..., 3]
        out_ang_vel[..., 1] = self._link_velocity_cache[..., 4]
        out_ang_vel[..., 2] = self._link_velocity_cache[..., 5]
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_vel_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = self._as_body_ids(body_ids)
        velocities = np.ascontiguousarray(self._ensure_link_velocity_cache()[:, ids, :])
        return velocities[:, :, :3], velocities[:, :, 3:]

    def get_body_lin_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        return self._ensure_link_velocity_cache()[rows[:, None], self._as_body_ids(body_ids), :3]  # type: ignore[no-any-return]

    def get_body_ang_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_ids, dtype=np.intp)
        return self._ensure_link_velocity_cache()[rows[:, None], self._as_body_ids(body_ids), 3:]  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                   #
    # ------------------------------------------------------------------ #

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._get_body_sensor_values(body_ids, "track_pos_b")

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        # MotrixSim framequat sensors output xyzw; convert to the wxyz contract.
        return self._xyzw_to_wxyz(self._get_body_sensor_values(body_ids, "track_quat_b"))

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Analytical per the SimBackend contract: world-frame velocity rotated
        # into each body's own frame. MotrixSim frame sensors report motion
        # relative to the baselink and degenerate to zero for the root body.
        ids = self._as_body_ids(body_ids)
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(ids), self._get_link_lin_vel_w(ids)
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np_quat_apply_inverse_batched(
            self.get_body_quat_w(ids), self._get_link_ang_vel_w(ids)
        )

    # ------------------------------------------------------------------ #
    # Sensors                                                            #
    # ------------------------------------------------------------------ #

    def _validate_sensor_names(self, names: Sequence[str]) -> tuple[str, ...]:
        sensor_names = tuple(names)
        missing = tuple(name for name in sensor_names if name not in self._sensor_names)
        if missing:
            available = ", ".join(sorted(self._sensor_names))
            raise KeyError(f"Unknown Motrix sensor(s) {missing}; available sensors: {available}")
        return sensor_names

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._validate_sensor_names((name,))
        if self._portable_mode:
            self._require_portable_healthy("get_sensor_data")
            return self._portable_sensor_value(name)
        return self._model.get_sensor_value(name, self._data)  # type: ignore[no-any-return]

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        self._validate_sensor_names((name,))
        rows = np.asarray(env_ids, dtype=np.intp)
        if self._portable_mode:
            self._require_portable_healthy("get_sensor_data_rows")
            if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self._num_envs):
                raise IndexError(f"env_ids must be one-dimensional and in [0, {self._num_envs})")
            return self._portable_sensor_value(name)[rows]
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[rows] = True
        selected_rows = np.flatnonzero(mask)
        selected_values = self._model.get_sensor_value(name, self._data[mask])
        return selected_values[np.searchsorted(selected_rows, rows)]  # type: ignore[no-any-return]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        sensor_names = self._validate_sensor_names(names)
        if not sensor_names:
            return np.empty((self._num_envs, 0), dtype=self._np_dtype)
        if self._portable_mode:
            self._require_portable_healthy("get_sensor_data_batch")
            return self._portable_sensor_values(sensor_names)
        values = self._model.get_sensor_values(sensor_names, self._data)
        return np.asarray(values, dtype=self._np_dtype)

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Retain Motrix's opaque native reader after cold-path name validation.

        MotrixSim exposes named sensor access but no public numeric sensor ID;
        the bound callable is therefore the narrowest backend-owned reader.
        It does not inspect XML or model metadata on the manager hot path.
        """
        if self._portable_mode:
            def portable_read() -> np.ndarray:
                self._require_portable_healthy("sensor data reader")
                return self._portable_sensor_values(names)

            return portable_read

        native_reader = self._model.get_sensor_values

        def read() -> np.ndarray:
            return np.asarray(native_reader(names, self._data), dtype=self._np_dtype)

        return read

    # ------------------------------------------------------------------ #
    # MotrixSim-specific                                                 #
    # ------------------------------------------------------------------ #

    def _get_body_names(self, body_ids: np.ndarray) -> list[str]:
        return [self._body_id_to_name[int(bid)] for bid in self._as_body_ids(body_ids)]

    def _get_link_poses_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.ascontiguousarray(self._link_poses[:, ids, :])

    def _get_link_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.ascontiguousarray(self._ensure_link_velocity_cache()[:, ids, :3])

    def _get_link_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._as_body_ids(body_ids)
        return np.ascontiguousarray(self._ensure_link_velocity_cache()[:, ids, 3:])

    def _get_body_sensor_values(self, body_ids: np.ndarray, prefix: str) -> np.ndarray:
        if self._portable_mode:
            self._require_portable_healthy("get body sensor values")
            names = self._get_body_names(body_ids)
            return np.stack(
                [self._portable_sensor_value(f"{prefix}_{name}") for name in names],
                axis=1,
            )
        names = self._get_body_names(body_ids)
        return np.stack(
            [
                self._model.get_sensor_value(f"{prefix}_{name}", self._data)
                for name in names
            ],
            axis=1,
        )

    def _xyzw_to_wxyz(self, q: np.ndarray) -> np.ndarray:
        """motrix xyzw → wxyz"""
        return q[..., [3, 0, 1, 2]]

    def _mujoco_qpos_to_motrix(self, qpos: np.ndarray) -> np.ndarray:
        """Convert every MuJoCo freejoint quaternion slice from wxyz to xyzw."""
        qpos_motrix = np.array(qpos, copy=True)
        for quat_indices in self._floating_base_quat_indices:
            qpos_motrix[..., quat_indices] = qpos[..., quat_indices[[1, 2, 3, 0]]]
        return qpos_motrix

    def _mujoco_qpos_to_motrix_into(self, dst: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Hot-path variant that writes into ``dst`` in place.

        Same conversion as :meth:`_mujoco_qpos_to_motrix` but avoids the per-call
        ``np.array(qpos, copy=True)`` allocation. Returns ``dst`` so callers can
        chain: ``qpos_motrix = self._mujoco_qpos_to_motrix_into(scratch, qpos)``.
        ``dst`` must have the same shape as ``qpos``; caller is responsible for
        sizing the scratch buffer.
        """
        np.copyto(dst, qpos, casting="same_kind")
        for quat_indices in self._floating_base_quat_indices:
            dst[..., quat_indices] = qpos[..., quat_indices[[1, 2, 3, 0]]]
        return dst

    def _motrix_qpos_to_mujoco(self, qpos: np.ndarray) -> np.ndarray:
        """Convert every Motrix freejoint quaternion slice from xyzw to wxyz."""
        qpos_mujoco = np.array(qpos, copy=True)
        for quat_indices in self._floating_base_quat_indices:
            qpos_mujoco[..., quat_indices] = qpos[..., quat_indices[[3, 0, 1, 2]]]
        return qpos_mujoco

    def _refresh_link_pose_cache(
        self,
        env_indices: np.ndarray | None = None,
        data_slice: Any | None = None,
        env_ids_intp: np.ndarray | None = None,
    ) -> None:
        if self._portable_mode:
            self._link_poses = np.empty(
                (self._num_envs, int(self._model.num_links), 7), dtype=self._np_dtype
            )
            for runtime in self._portable_runtimes:
                self._link_poses[runtime.rows] = runtime.model.get_link_poses(runtime.data)
            return
        if env_indices is None:
            self._link_poses = self._model.get_link_poses(self._data)
        else:
            if data_slice is None:
                mask = np.zeros(self._num_envs, dtype=bool)
                mask[env_indices] = True
                data_slice = self._data[mask]
            # Indexing with intp is a hair faster than the arbitrary-dtype
            # ndarray path; use the pre-converted array when set_state hands
            # it down.
            idx = env_indices if env_ids_intp is None else env_ids_intp
            self._link_poses[idx] = self._model.get_link_poses(data_slice)

    def _refresh_link_velocity_cache(self, env_indices: np.ndarray | None = None) -> None:
        if self._portable_mode:
            self._link_velocities = np.empty(
                (self._num_envs, int(self._model.num_links), 6), dtype=self._np_dtype
            )
            for runtime in self._portable_runtimes:
                self._link_velocities[runtime.rows] = runtime.model.get_link_velocities(
                    runtime.data
                )
            self._link_velocity_cache_valid = True
            return
        if env_indices is None:
            self._link_velocities = self._model.get_link_velocities(self._data)
        else:
            mask = np.zeros(self._num_envs, dtype=bool)
            mask[env_indices] = True
            if self._link_velocities is None:
                self._link_velocities = self._model.get_link_velocities(self._data)
                self._link_velocity_cache_valid = True
                return
            self._link_velocities[env_indices] = self._model.get_link_velocities(self._data[mask])
        self._link_velocity_cache_valid = True

    def _invalidate_link_velocity_cache(self) -> None:
        self._link_velocity_cache_valid = False

    def _ensure_link_velocity_cache(self) -> np.ndarray:
        if self._link_velocities is None or not self._link_velocity_cache_valid:
            self._refresh_link_velocity_cache()
        assert self._link_velocities is not None
        return self._link_velocities

    def _coerce_reset_field(
        self,
        value: np.ndarray,
        *,
        name: str,
        num_reset: int,
        shaped_tail: tuple[int, ...],
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        shaped = (num_reset, *shaped_tail)
        flat_shape = (num_reset, int(np.prod(shaped_tail)))
        if arr.shape == shaped:
            return arr.copy()
        if arr.shape == flat_shape:
            return arr.reshape(shaped).copy()
        raise ValueError(f"{name} must have shape {shaped} or {flat_shape}, got {arr.shape}")

    def _set_link_mass_overrides(self, data_slice, body_mass: np.ndarray) -> None:
        for link_id, link in self._links_by_id.items():
            link.set_mass_override(
                data_slice,
                np.ascontiguousarray(np.asarray(body_mass[:, link_id], dtype=np.float32)),
            )

    def _set_link_ipos_overrides(self, data_slice, body_ipos: np.ndarray) -> None:
        for link_id, link in self._links_by_id.items():
            link.set_center_of_mass_override(
                data_slice,
                np.ascontiguousarray(np.asarray(body_ipos[:, link_id, :], dtype=np.float32)),
            )

    def _set_geom_friction_overrides(self, data_slice, geom_friction: np.ndarray) -> None:
        if not self._supports_geom_friction_override:
            raise NotImplementedError("Motrix geom friction override is not available")
        override_ids = getattr(self, "_geom_friction_override_ids", tuple(self._geoms_by_id))
        unsupported_ids = sorted(set(self._geoms_by_id) - set(override_ids))
        if unsupported_ids:
            unsupported_values = geom_friction[:, unsupported_ids, :]
            default_values = self._default_geom_friction[None, unsupported_ids, :]
            if not np.allclose(unsupported_values, default_values):
                raise ValueError(
                    "Motrix geom friction override only supports collision geoms; "
                    f"non-collision geom ids were modified: {unsupported_ids}"
                )
        for geom_id in override_ids:
            geom = self._geoms_by_id[int(geom_id)]
            geom.set_friction_override(
                data_slice,
                np.ascontiguousarray(np.asarray(geom_friction[:, geom_id, :], dtype=np.float32)),
            )

    def _clear_applied_body_forces(
        self,
        env_indices: np.ndarray,
        env_ids_intp: np.ndarray | None = None,
        body_ids: Sequence[int] | None = None,
    ) -> None:
        if self._portable_mode:
            rows = (
                np.asarray(env_indices, dtype=np.intp)
                if env_ids_intp is None
                else np.asarray(env_ids_intp, dtype=np.intp)
            )
            self._clear_portable_pending_body_forces(rows, body_ids=body_ids)
            return
        if not self._applied_body_forces:
            return
        env_ids = (
            env_ids_intp if env_ids_intp is not None else np.asarray(env_indices, dtype=np.intp)
        )
        for applied_force in self._applied_body_forces.values():
            applied_force[env_ids, :] = 0.0

    def _clear_portable_pending_body_forces(
        self,
        rows: np.ndarray,
        *,
        body_ids: Sequence[int] | None,
    ) -> None:
        """Cancel only unconsumed native wrenches in the selected reset scope."""
        if rows.size == 0:
            return
        selected_bodies = None if body_ids is None else {int(body_id) for body_id in body_ids}
        pending_channels = (
            (self._portable_pending_body_forces, False),
            (self._portable_pending_body_torques, True),
        )
        native_writes: list[tuple[Any, Any, np.ndarray, bool]] = []
        cleared: list[tuple[np.ndarray, np.ndarray]] = []
        for pending_by_body, is_torque in pending_channels:
            for public_body_id in sorted(pending_by_body):
                if selected_bodies is not None and public_body_id not in selected_bodies:
                    continue
                pending = pending_by_body[public_body_id]
                selected = pending[rows]
                if not np.any(selected):
                    cleared.append((pending, rows))
                    continue
                cancellation = np.ascontiguousarray(-selected.astype(np.float32))
                for runtime in self._portable_runtimes:
                    common_rows, runtime_positions, row_positions = np.intersect1d(
                        runtime.rows, rows, return_indices=True
                    )
                    if common_rows.size == 0:
                        continue
                    native_id = int(runtime.binding.public_to_native_body[public_body_id])
                    link = runtime.binding.links_by_id.get(native_id)
                    if link is None:
                        raise RuntimeError(
                            f"Motrix portable body {public_body_id} is missing native link "
                            f"{native_id} in variant {runtime.variant}"
                        )
                    data_slice = runtime.data[
                        mtx.DisjointIndices(runtime.local_rows(common_rows))
                    ]
                    native_writes.append(
                        (
                            link,
                            data_slice,
                            cancellation[row_positions],
                            is_torque,
                        )
                    )
                cleared.append((pending, rows))
        try:
            for link, data_slice, cancellation, is_torque in native_writes:
                if is_torque:
                    link.add_external_torque(data_slice, cancellation, local=False)
                else:
                    link.add_external_force(data_slice, cancellation, local=False)
            for pending, selected_rows in cleared:
                pending[selected_rows] = 0.0
        except BaseException:
            self._portable_faulted = True
            raise

    def push_robots(self, force_range):
        if self._portable_mode:
            raise NotImplementedError("portable Motrix scenes do not support interval pushes")
        ex_force = np.random.rand(self.num_envs, 3) * 2 - 1  # [x_force, y_force, z_force]
        ex_force[:, 0] *= force_range[0]
        ex_force[:, 1] *= force_range[1]
        ex_force[:, 2] *= force_range[2]
        self._push_body_link.add_external_force(self._data, ex_force, local=True)

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Apply absolute world-frame external wrenches through Motrix Link APIs."""
        self._reject_wrench_write_inside_pre_step_control("apply_body_force")
        if torque is not None and not self._portable_mode:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support interval body torque perturbation"
            )
        if self._portable_mode:
            self._require_portable_healthy("apply_body_force")
            if not self._supports_external_force:
                raise NotImplementedError("Motrix link external-force API is not available")
            if torque is not None and not self._supports_external_torque:
                raise NotImplementedError("Motrix link external-torque API is not available")
            layout = self.get_scene_layout()
            body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
            force_np = np.asarray(force, dtype=np.float32)
            torque_np = None if torque is None else np.asarray(torque, dtype=np.float32)
            expected_shape = (self._num_envs, body_ids_np.size, 3)
            if force_np.shape != expected_shape:
                raise ValueError(
                    f"body force must have shape {expected_shape}, got {force_np.shape}"
                )
            if not np.isfinite(force_np).all():
                raise ValueError("body force contains NaN or Inf")
            if torque_np is not None:
                if torque_np.shape != expected_shape:
                    raise ValueError(
                        f"body torque must have shape {expected_shape}, got {torque_np.shape}"
                    )
                if not np.isfinite(torque_np).all():
                    raise ValueError("body torque contains NaN or Inf")
            if np.any(body_ids_np < 0) or np.any(body_ids_np >= layout.nbody):
                raise ValueError(f"body_ids must be in [0, {layout.nbody})")
            if np.any(self._portable_public_to_native_body[body_ids_np] < 0):
                raise ValueError("portable Motrix body ids must reference owned physical bodies")

            native_writes: list[tuple[Any, Any, np.ndarray, bool]] = []
            for body_offset, public_body_id_value in enumerate(body_ids_np):
                public_body_id = int(public_body_id_value)
                target = np.ascontiguousarray(force_np[:, body_offset, :])
                for runtime in self._portable_runtimes:
                    native_id = int(runtime.binding.public_to_native_body[public_body_id])
                    link = runtime.binding.links_by_id.get(native_id)
                    if link is None:
                        raise RuntimeError(
                            f"Motrix portable body {public_body_id} is missing native link "
                            f"{native_id} in variant {runtime.variant}"
                        )
                    native_writes.append((link, runtime.data, target[runtime.rows], False))
                    if torque_np is not None:
                        torque_target = np.ascontiguousarray(torque_np[:, body_offset, :])
                        native_writes.append(
                            (link, runtime.data, torque_target[runtime.rows], True)
                        )
            try:
                for link, data, target, is_torque in native_writes:
                    if is_torque:
                        link.add_external_torque(data, target, local=False)
                    else:
                        link.add_external_force(data, target, local=False)
                for body_offset, public_body_id_value in enumerate(body_ids_np):
                    public_body_id = int(public_body_id_value)
                    pending = self._portable_pending_body_forces.setdefault(
                        public_body_id,
                        np.zeros((self._num_envs, 3), dtype=np.float32),
                    )
                    pending += force_np[:, body_offset, :]
                    if torque_np is not None:
                        pending_torque = self._portable_pending_body_torques.setdefault(
                            public_body_id,
                            np.zeros((self._num_envs, 3), dtype=np.float32),
                        )
                        pending_torque += torque_np[:, body_offset, :]
            except BaseException:
                self._portable_faulted = True
                raise
            return
        if not getattr(self, "_supports_external_force", False):
            raise NotImplementedError("Motrix link external-force API is not available")
        body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_np = np.asarray(force, dtype=np.float64)
        expected_shape = (self._num_envs, body_ids_np.size, 3)
        if force_np.shape != expected_shape:
            raise ValueError(f"body force must have shape {expected_shape}, got {force_np.shape}")
        for body_offset, body_id in enumerate(body_ids_np):
            link_id = int(body_id)
            link = self._links_by_id.get(link_id)
            if link is None:
                raise ValueError(f"Body id {link_id} not found in Motrix model")
            target_force = np.asarray(force_np[:, body_offset, :], dtype=np.float64)
            applied_force = self._applied_body_forces.setdefault(
                link_id,
                np.zeros((self._num_envs, 3), dtype=np.float64),
            )
            delta_force = target_force - applied_force
            if np.any(delta_force):
                link.add_external_force(
                    self._data,
                    np.ascontiguousarray(delta_force.astype(np.float32)),
                    local=False,
                )
                applied_force[:] = target_force

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
        offsets_np = np.ascontiguousarray(np.asarray(offsets, dtype=np.float32))
        if offsets_np.ndim != 2 or offsets_np.shape[1] != 2:
            raise ValueError(f"offsets must have shape (num_points, 2), got {offsets_np.shape}")

        if alignment != "yaw":
            raise ValueError(f"MotrixBackend only supports alignment='yaw', got {alignment!r}")
        if output not in {"height", "clearance"}:
            raise ValueError(f"Unsupported hfield sampling output: {output!r}")

        geom_id = int(hfield_geom_id)
        if geom_id < 0 or geom_id >= int(self._model.num_geoms):
            raise ValueError(f"hfield_geom_id out of range: {geom_id}")

        body_id = int(frame_body_id)
        if body_id < 0 or body_id >= int(self._model.num_links):
            raise ValueError(f"frame_body_id out of range: {body_id}")

        terrain = self._model.get_geom(geom_id)
        if terrain is None:
            raise ValueError(f"Geom id {geom_id} not found in Motrix model")
        if not isinstance(terrain, mtx.GeomHField):
            raise ValueError(f"Geom id {geom_id} is not backed by a Motrix hfield")
        frame = self._link_cache[body_id]
        scanner = mtx.TerrainScanner(
            terrain,
            frame,
            offsets_np,
            alignment=alignment,
            output=output,
        )
        return _MotrixTerrainScanner(
            scanner=scanner,
            data=self._data,
            out=np.empty((self._num_envs, offsets_np.shape[0]), dtype=self._np_dtype),
        )

    def _update_tracking_camera_view(self) -> None:
        if (
            self._render_app is None
            or self._render_tracking_camera is None
            or self._render_offsets_np is None
        ):
            return
        lookat = tracking_camera_lookat(
            self.get_base_pos(),
            self._render_tracking_camera,
            self._render_offsets_np,
        )
        self._render_app.system_camera.set_view(
            lookat,
            self._render_tracking_camera.distance,
            self._render_tracking_camera.elevation,
            self._render_tracking_camera.azimuth,
        )

    def _assert_render_context_available(self, *, headless: bool, capture: bool) -> None:
        if self._render_app is None:
            return
        if self._render_headless != headless:
            raise RuntimeError(
                "Motrix renderer is already initialized with "
                f"headless={self._render_headless!r}; cannot reuse it with headless={headless!r}"
            )
        if capture and not self._render_capture_enabled:
            raise RuntimeError(
                "Motrix renderer is already initialized without video capture; "
                "cannot enable capture on the existing renderer"
            )
        return

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
        """Initialize a Motrix renderer, optionally enabling system-camera capture."""
        if self._portable_mode and self._portable_variant_assignment is not None:
            raise NotImplementedError(
                "Motrix native rendering does not support fixed-variant scenes yet"
            )
        headless = bool(headless)
        capture = bool(capture)
        self._assert_render_context_available(headless=headless, capture=capture)
        if self._render_app is not None:
            return

        camera = CameraCfg.from_kwargs(camera_kwargs)
        settings = RenderSettings.performance()
        settings.enable_shadow = True
        offsets = render_offsets(
            self._num_envs,
            float(spacing),
            offset_mode=str(offset_mode),
        )
        offsets_np = np.asarray(offsets, dtype=np.float64)
        self._render_offsets_np = offsets_np
        use_configured_camera = capture or camera_kwargs is not None
        camera_view = None
        if use_configured_camera:
            base_positions = self.get_base_pos() if camera.cam_tracking else None
            camera_view = resolve_system_camera_view(
                self._num_envs,
                base_positions,
                offsets,
                camera,
            )
            tracking_camera = camera_view.tracking
        else:
            tracking_camera = None
        if capture:
            self._model.cameras.set_system_render_target("image", int(width), int(height))
        render_app = RenderApp(headless=headless)
        try:
            render_app.launch(
                self._model,
                batch=self._num_envs,
                render_offset=offsets,
                render_settings=settings,
            )
        except _MotrixRenderClosedError as e:
            # Normalize the motrixsim-private window-closed error to the
            # interface-level signal declared on SimBackend.
            raise RenderClosedError(str(e)) from e
        if camera_view is not None:
            render_app.system_camera.set_view(
                camera_view.lookat,
                camera_view.distance,
                camera_view.elevation,
                camera_view.azimuth,
            )
            if not capture:
                render_app.set_main_camera(None)
        self._render_app = render_app
        self._render_headless = headless
        self._render_capture_enabled = capture
        self._render_tracking_camera = tracking_camera

    def render(self):
        """Render current state (interactive visualization)"""
        if self._render_app is None:
            self.init_renderer()
        self._assert_render_context_available(headless=False, capture=False)
        assert self._render_app is not None
        self._update_tracking_camera_view()
        try:
            self._render_app.sync(data=self._data)
        except _MotrixRenderClosedError as e:
            # Normalize the motrixsim-private window-closed error to the
            # interface-level signal declared on SimBackend.
            raise RenderClosedError(str(e)) from e

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame from Motrix's system camera."""
        if self._render_app is None:
            self.init_renderer(headless=True, capture=True)
        if not self._render_capture_enabled:
            raise RuntimeError("Motrix renderer is not initialized for video capture")
        assert self._render_app is not None

        self._update_tracking_camera_view()
        try:
            task = self._render_app.system_camera.capture()
            self._render_app.sync(data=self._data, wait=True)
            image = task.take_image()
        except _MotrixRenderClosedError as e:
            # Normalize the motrixsim-private window-closed error to the
            # interface-level signal declared on SimBackend.
            raise RenderClosedError(str(e)) from e
        if image is None:
            raise RuntimeError("Motrix system camera capture did not return an image")

        pixels = np.asarray(image.pixels)
        if pixels.ndim != 3:
            raise RuntimeError(
                f"Motrix system camera capture must return an HWC image, got shape {pixels.shape}"
            )
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        if pixels.shape[-1] != 3:
            raise RuntimeError(
                "Motrix system camera capture must return RGB/RGBA pixels, "
                f"got shape {pixels.shape}"
            )
        return np.ascontiguousarray(pixels, dtype=np.uint8)

    def _apply_reset_randomization(
        self,
        data_slice,
        env_indices: np.ndarray,
        randomization: ResetRandomizationPayload | None,
        env_ids_intp: np.ndarray | None = None,
    ) -> None:
        if randomization is None or randomization.is_empty():
            return
        unsupported = (
            randomization.requested_terms() - self.get_dr_capabilities().supported_reset_terms
        )
        if unsupported:
            terms = ", ".join(sorted(unsupported))
            raise NotImplementedError(
                f"{self.backend_type} backend does not support reset randomization terms: {terms}"
            )

        env_ids = (
            env_ids_intp if env_ids_intp is not None else np.asarray(env_indices, dtype=np.intp)
        )
        num_reset = len(env_ids)
        body_mass = None
        if randomization.body_mass is not None:
            body_mass = self._coerce_reset_field(
                randomization.body_mass,
                name="body_mass",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_links),),
            )
        if randomization.base_mass_delta is not None:
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._default_body_mass,
                    (num_reset, int(self._model.num_links)),
                ).copy()
            body_mass[:, int(self._body_link.index)] += np.asarray(
                randomization.base_mass_delta, dtype=np.float32
            )
        if body_mass is not None:
            self._set_link_mass_overrides(data_slice, body_mass)

        body_ipos = None
        if randomization.body_ipos is not None:
            body_ipos = self._coerce_reset_field(
                randomization.body_ipos,
                name="body_ipos",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_links), 3),
            )
        if randomization.base_com_offset is not None:
            if body_ipos is None:
                body_ipos = np.broadcast_to(
                    self._default_body_ipos,
                    (num_reset, int(self._model.num_links), 3),
                ).copy()
            body_ipos[:, int(self._body_link.index), :] += np.asarray(
                randomization.base_com_offset, dtype=np.float32
            )
        if body_ipos is not None:
            self._set_link_ipos_overrides(data_slice, body_ipos)

        if randomization.geom_friction is not None:
            geom_friction = self._coerce_reset_field(
                randomization.geom_friction,
                name="geom_friction",
                num_reset=num_reset,
                shaped_tail=(int(self._model.num_geoms), 3),
            )
            self._set_geom_friction_overrides(data_slice, geom_friction)

        if randomization.gravity is not None:
            gravity = self._coerce_reset_field(
                randomization.gravity,
                name="gravity",
                num_reset=num_reset,
                shaped_tail=(3,),
            )
            self._set_gravity_override(data_slice, gravity)

        if randomization.kp is not None:
            kp = np.asarray(randomization.kp, dtype=np.float32)
            expected_shape = (num_reset, self.num_actuators)
            if kp.shape != expected_shape:
                raise ValueError(f"kp must have shape {expected_shape}, got {kp.shape}")
            self._set_position_actuator_kp_override(data_slice, kp)

        if randomization.kd is not None:
            kd = np.asarray(randomization.kd, dtype=np.float32)
            expected_shape = (num_reset, self.num_actuators)
            if kd.shape != expected_shape:
                raise ValueError(f"kd must have shape {expected_shape}, got {kd.shape}")
            self._set_position_actuator_kd_override(data_slice, kd)

    def _set_gravity_override(self, data_slice, gravity: np.ndarray) -> None:
        if not getattr(self, "_supports_gravity_override", False):
            raise NotImplementedError("Motrix gravity override is not available")
        self._model.set_gravity_override(
            data_slice,
            np.ascontiguousarray(np.asarray(gravity, dtype=np.float32)),
        )

    def _set_position_actuator_kp_override(self, data_slice, kp: np.ndarray) -> None:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator kp override is only available for all-position-actuator models"
            )
        for actuator in self._position_actuators:
            # TODO(motrixsim#1384): drop the copy once strided NumPy views are accepted.
            actuator.set_kp_override(data_slice, np.ascontiguousarray(kp[:, int(actuator.index)]))

    def _set_position_actuator_kd_override(self, data_slice, kd: np.ndarray) -> None:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator kd override is only available for all-position-actuator models"
            )
        for actuator in self._position_actuators:
            # TODO(motrixsim#1384): drop the copy once strided NumPy views are accepted.
            actuator.set_damping_override(
                data_slice,
                np.ascontiguousarray(kd[:, int(actuator.index)]),
            )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_position_actuator_gains:
            raise NotImplementedError(
                "Motrix actuator gains are only exposed for all-position-actuator models"
            )
        return self._default_actuator_kp.copy(), self._default_actuator_kd.copy()
