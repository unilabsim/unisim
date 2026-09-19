"""Host-compatibility implementation of the independent ``genesis`` backend.

The adapter serves the ``SimBackend`` NumPy contract on top of Genesis 1.3.3,
following its measured feasibility mappings (#1372): link-addressed root state
(never entity-level getters, REPORT §5.5), ``control_dofs_position`` inside an
adapter-owned nsteps loop
honoring ``set_pre_step_control`` (§5.4), host caches refreshed once per
step/reset barrier (§5.9), MJCF-named sensor equivalents from link state plus
one IMUSensor per accelerometer site with clean (noise-free) data (§3.4/§5.3),
genesis-native recoded geom contact masks that must not be compared against
MuJoCo tables (§5.10), and a DR capability set restricted to the per-env
round-trip-measured items (§3.5 [8] / §5.7).
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any

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
from unisim.dr.types import (
    INTERVAL_TERM_BODY_FORCE,
    INTERVAL_TERM_BODY_TORQUE,
    RESET_TERM_BASE_COM,
    RESET_TERM_BASE_MASS,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ResetRandomizationPayload,
    _validate_reset_term,
    require_op_body_ids,
)
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_layout import CompiledSceneLayout
from unisim.utils.rotation import (
    np_matrix_from_quat,
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

from . import dependencies, materialization, playback

logger = logging.getLogger(__name__)

_WORLD_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class _GenesisEntityRuntime:
    """One public entity bound to its independent native Genesis entity."""

    entity: Any
    is_visual_mirror: bool
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    native_qpos_indices: np.ndarray
    native_qvel_indices: np.ndarray
    actuator_indices: np.ndarray
    native_actuated_dofs: np.ndarray
    body_ids: np.ndarray
    native_body_indices: np.ndarray
    source_metadata: tuple[materialization.GenesisModelMetadata, ...]
    collision_geom_indices: np.ndarray
    geom_sizes: np.ndarray
    geom_sizes_nonuniform: bool
    contact_masks: tuple[np.ndarray, np.ndarray] | None
    contact_masks_nonuniform: bool
    geom_frictions: np.ndarray | None
    geom_frictions_nonuniform: bool
    geom_solver_params: np.ndarray | None
    geom_solver_params_nonuniform: bool
    dof_damping: np.ndarray
    dof_damping_nonuniform: bool
    dof_frictionloss: np.ndarray
    dof_frictionloss_nonuniform: bool
    dof_armature: np.ndarray
    dof_armature_nonuniform: bool


@dataclass(frozen=True)
class _GenesisContactSensorBinding:
    """One exact public geom pair bound to native Genesis collision IDs."""

    name: str
    entity1: Any
    entity2: Any
    geom1_ids: np.ndarray
    geom2_ids: np.ndarray
    netforce: bool


@dataclass(frozen=True)
class _GenesisSensorLinkBinding:
    """One audited public link and its per-variant inertial-frame metadata."""

    runtime: _GenesisEntityRuntime
    native_body: int
    body_ipos: np.ndarray | None
    body_iquat: np.ndarray | None


@dataclass(frozen=True)
class _GenesisPortableResetRandomization:
    """Prevalidated public reset values ready for per-entity submission."""

    body_mass: np.ndarray | None
    body_ipos: np.ndarray | None
    dof_damping: np.ndarray | None
    dof_frictionloss: np.ndarray | None
    dof_armature: np.ndarray | None
    kp: np.ndarray | None
    kd: np.ndarray | None


def _make_device_cache(torch: Any, shape: tuple[int, ...]) -> tuple[Any, np.ndarray]:
    """One fixed-shape host cache: (pinned storage, zero-copy NumPy view)."""
    pinned = torch.empty(shape, dtype=torch.float32, pin_memory=torch.cuda.is_available())
    return pinned, pinned.numpy()


class GenesisBackend(SimBackend):
    """Independent Genesis backend exposed through the host NumPy profile.

    Construction performs dependency loading, the MJCF cold-path scan, the
    process-wide ``gs.init`` (once), and scene/entity/sensor creation;
    ``materialize()`` builds the batched solver state and binds all runtime
    caches.  Terrain, geom-name contracts, and site Jacobians fail closed.
    Native interactive/offscreen rendering attaches lazily post-build (see
    the play contract section).  Call ``close()`` to end the process-wide
    Genesis session; re-initialization afterwards fails closed by design.
    """

    _play_capabilities = _NATIVE_RENDERER_PLAY_CAPABILITIES
    _metadata: materialization.GenesisModelMetadata
    _composed_scene: Any | None = None
    _portable_sources: materialization.GenesisPortableSources | None = None
    _scene_cleanup_handle: Any | None = None
    _portable_default_body_ipos: np.ndarray
    _portable_default_roots: np.ndarray

    def _expected_geometry_bounds(
        self, geom_type: int, geom_size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mujoco = self._deps.mujoco
        size = np.asarray(geom_size, dtype=np.float64)
        if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            if not np.all(np.isfinite(size)) or size[0] <= 0.0:
                raise RuntimeError("genesis portable sphere geometry has an invalid native radius")
            half_extent = np.full((3,), float(size[0]), dtype=np.float64)
        elif int(geom_type) == int(mujoco.mjtGeom.mjGEOM_BOX):
            if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
                raise RuntimeError("genesis portable box geometry has invalid native half extents")
            half_extent = size.copy()
        else:
            raise NotImplementedError(
                "genesis portable geometry identity is reviewed only for named sphere "
                "and box geoms; unsupported geom types fail closed"
            )
        return -half_extent, half_extent

    @staticmethod
    def _native_vgeom_env_ids(vgeom: Any, num_envs: int) -> np.ndarray:
        if vgeom.active_envs_idx is None:
            return np.arange(num_envs, dtype=np.intp)
        return np.asarray(vgeom.active_envs_idx, dtype=np.intp)

    def _geometry_size_from_native_bounds(
        self, geom_type: int, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Derive a primitive size from an audited Genesis visual AABB."""

        mujoco = self._deps.mujoco
        half_extent = (np.asarray(upper, dtype=np.float64) - lower) * 0.5
        if not np.all(np.isfinite(half_extent)) or np.any(half_extent <= 0.0):
            raise RuntimeError("genesis portable native geometry has invalid bounds")
        if not np.allclose(lower, -half_extent, rtol=2e-5, atol=2e-6) or not np.allclose(
            upper, half_extent, rtol=2e-5, atol=2e-6
        ):
            raise RuntimeError("genesis portable native geometry bounds are not symmetric")
        if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            if not np.allclose(half_extent, half_extent[0], rtol=2e-5, atol=2e-6):
                raise RuntimeError("genesis portable native sphere bounds are not spherical")
            radius = float(np.mean(half_extent))
            return np.asarray((radius, 0.0, 0.0), dtype=np.float64)
        if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_BOX):
            return half_extent.copy()
        raise NotImplementedError(
            "genesis portable geometry sizes are reviewed only for named sphere "
            "and box geoms; unsupported geom types fail closed"
        )

    @staticmethod
    def _bind_portable_collision_properties(
        native_entity: Any,
        owner: Any,
        is_visual_mirror: bool,
        source_metadata: tuple[materialization.GenesisModelMetadata, ...],
        num_envs: int,
        variant_assignment: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray] | None,
        np.ndarray | None,
        np.ndarray | None,
        bool,
        bool,
        bool,
    ]:
        """Bind actual Genesis collision properties in frozen public geom order.

        Genesis may omit a collision instance for a collision-disabled source
        geom. That is not a native property readback, so getters must fail
        closed rather than synthesizing the authored source values.
        """

        if is_visual_mirror:
            if list(getattr(native_entity, "geoms", ())):
                raise RuntimeError(
                    f"genesis visual mirror {owner.name!r} unexpectedly owns native "
                    "collision geometry"
                )
            return (
                (
                    np.zeros(len(owner.geoms), dtype=np.int32),
                    np.zeros(len(owner.geoms), dtype=np.int32),
                ),
                None,
                None,
                False,
                False,
                False,
            )
        native_geoms = list(native_entity.geoms)
        expected_count = len(owner.geoms) * len(source_metadata)
        if len(native_geoms) != expected_count:
            return None, None, None, False, False, False

        masks = np.empty((len(source_metadata), len(owner.geoms), 2), dtype=np.int32)
        frictions = np.empty((len(source_metadata), len(owner.geoms), 3), dtype=np.float64)
        solver_params = np.empty((len(source_metadata), len(owner.geoms), 7), dtype=np.float64)
        used: set[int] = set()
        for variant, metadata in enumerate(source_metadata):
            expected_rows = (
                np.arange(num_envs, dtype=np.intp)
                if len(source_metadata) == 1
                else np.flatnonzero(variant_assignment == variant)
            )
            for geom_index, geom in enumerate(owner.geoms):
                matches: list[int] = []
                for native_index, native_geom in enumerate(native_geoms):
                    if native_index in used:
                        continue
                    native_metadata = native_geom.metadata
                    if str(native_metadata.get("name", "")) != geom.name:
                        continue
                    if str(native_geom.link.name) != geom.body_name:
                        continue
                    active_rows = (
                        np.arange(num_envs, dtype=np.intp)
                        if native_geom.active_envs_idx is None
                        else np.asarray(native_geom.active_envs_idx, dtype=np.intp)
                    )
                    if not np.array_equal(active_rows, expected_rows):
                        continue
                    matches.append(native_index)
                if len(matches) != 1:
                    return None, None, None, False, False, False
                native_geom = native_geoms[matches[0]]
                used.add(matches[0])
                try:
                    friction = np.asarray(
                        (
                            native_geom.friction,
                            native_geom.friction_torsional,
                            native_geom.friction_rolling,
                        ),
                        dtype=np.float64,
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} geom {geom.name!r} does not "
                        "expose valid native friction properties"
                    ) from exc
                if (
                    friction.shape != (3,)
                    or not np.isfinite(friction).all()
                    or np.any(friction < 0.0)
                ):
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} geom {geom.name!r} has invalid "
                        "native friction properties"
                    )
                try:
                    raw_solver_params = native_geom.sol_params
                    if hasattr(raw_solver_params, "detach"):
                        raw_solver_params = raw_solver_params.detach()
                    if hasattr(raw_solver_params, "cpu"):
                        raw_solver_params = raw_solver_params.cpu()
                    native_solver_params = np.asarray(raw_solver_params, dtype=np.float64)
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} geom {geom.name!r} does not "
                        "expose valid native solver parameters"
                    ) from exc
                if (
                    native_solver_params.shape != (7,)
                    or not np.isfinite(native_solver_params).all()
                ):
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} geom {geom.name!r} has invalid "
                        "native solver parameters"
                    )
                masks[variant, geom_index, 0] = int(native_geom.contype)
                masks[variant, geom_index, 1] = int(native_geom.conaffinity)
                frictions[variant, geom_index] = friction
                solver_params[variant, geom_index] = native_solver_params

        masks_nonuniform = len(source_metadata) > 1 and not np.all(masks == masks[0])
        frictions_nonuniform = len(source_metadata) > 1 and not np.all(frictions == frictions[0])
        solver_params_nonuniform = len(source_metadata) > 1 and not np.all(
            solver_params == solver_params[0]
        )
        native_masks = None if masks_nonuniform else (masks[0, :, 0].copy(), masks[0, :, 1].copy())
        native_frictions = None if frictions_nonuniform else frictions[0].copy()
        native_solver_params = None if solver_params_nonuniform else solver_params[0].copy()
        return (
            native_masks,
            native_frictions,
            native_solver_params,
            masks_nonuniform,
            frictions_nonuniform,
            solver_params_nonuniform,
        )

    @staticmethod
    def _bind_portable_dof_properties(
        native_entity: Any,
        owner: Any,
        is_visual_mirror: bool,
        source_metadata: tuple[materialization.GenesisModelMetadata, ...],
        num_envs: int,
        variant_assignment: np.ndarray,
        native_qvel_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
        """Capture native DOF properties in frozen public qvel order."""

        empty = np.empty((len(source_metadata), 0), dtype=np.float64)
        if is_visual_mirror:
            return empty, empty.copy(), empty.copy(), False, False, False
        native_values: dict[str, np.ndarray] = {}
        for getter_name, property_name in (
            ("get_dofs_damping", "damping"),
            ("get_dofs_frictionloss", "frictionloss"),
            ("get_dofs_armature", "armature"),
        ):
            try:
                raw_values = getattr(native_entity, getter_name)()
                if hasattr(raw_values, "detach"):
                    raw_values = raw_values.detach()
                if hasattr(raw_values, "cpu"):
                    raw_values = raw_values.cpu()
                values = np.asarray(raw_values, dtype=np.float64)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"genesis entity {owner.name!r} does not expose valid native "
                    f"DOF {property_name}"
                ) from exc
            expected_native_shape = (num_envs, int(native_entity.n_dofs))
            if (
                values.shape != expected_native_shape
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                raise RuntimeError(
                    f"genesis entity {owner.name!r} has invalid native DOF "
                    f"{property_name} shape or values"
                )
            native_values[property_name] = values

        result: dict[str, np.ndarray] = {}
        nonuniform: dict[str, bool] = {}
        for property_name, values in native_values.items():
            variant_values = np.empty(
                (len(source_metadata), len(native_qvel_indices)), dtype=np.float64
            )
            for variant in range(len(source_metadata)):
                rows = (
                    np.arange(num_envs, dtype=np.intp)
                    if len(source_metadata) == 1
                    else np.flatnonzero(variant_assignment == variant)
                )
                selected = values[np.ix_(rows, native_qvel_indices)]
                if not np.all(selected == selected[0]):
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} native DOF {property_name} "
                        f"is non-uniform within variant {variant}"
                    )
                variant_values[variant] = selected[0]
            nonuniform[property_name] = len(source_metadata) > 1 and not np.all(
                variant_values == variant_values[0]
            )
            result[property_name] = variant_values

        return (
            result["damping"],
            result["frictionloss"],
            result["armature"],
            nonuniform["damping"],
            nonuniform["frictionloss"],
            nonuniform["armature"],
        )

    @staticmethod
    def _bind_portable_collision_geom_indices(
        native_entity: Any,
        owner: Any,
        is_visual_mirror: bool,
        source_metadata: tuple[materialization.GenesisModelMetadata, ...],
        num_envs: int,
        variant_assignment: np.ndarray,
    ) -> np.ndarray:
        """Bind exact native collision geom IDs for each variant/public geom."""

        native_ids = np.full((len(source_metadata), len(owner.geoms)), -1, dtype=np.int64)
        if is_visual_mirror:
            return native_ids
        native_geoms = list(native_entity.geoms)
        for variant in range(len(source_metadata)):
            expected_rows = (
                np.arange(num_envs, dtype=np.intp)
                if len(source_metadata) == 1
                else np.flatnonzero(variant_assignment == variant)
            )
            used: set[int] = set()
            for geom_index, geom in enumerate(owner.geoms):
                matches: list[int] = []
                for native_index, native_geom in enumerate(native_geoms):
                    if native_index in used:
                        continue
                    if str(native_geom.metadata.get("name", "")) != geom.name:
                        continue
                    if str(native_geom.link.name) != geom.body_name:
                        continue
                    active_rows = (
                        np.arange(num_envs, dtype=np.intp)
                        if native_geom.active_envs_idx is None
                        else np.asarray(native_geom.active_envs_idx, dtype=np.intp)
                    )
                    if not np.array_equal(active_rows, expected_rows):
                        continue
                    matches.append(native_index)
                if len(matches) == 1:
                    used.add(matches[0])
                    native_ids[variant, geom_index] = int(native_geoms[matches[0]].idx)
        return native_ids

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        device_id: int | None = None,
        integrator: str | None = None,
        constraint_solver: str | None = None,
        friction_cone: str | None = None,
        solver_iterations: int | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        try:
            self._initialize(
                scene,
                num_envs,
                sim_dt,
                base_name=base_name,
                push_body_name=push_body_name,
                device_id=device_id,
                integrator=integrator,
                constraint_solver=constraint_solver,
                friction_cone=friction_cone,
                solver_iterations=solver_iterations,
                **unexpected_kwargs,
            )
        except BaseException:
            self._cleanup_materialization_resources()
            raise

    def _initialize(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        push_body_name: str | None = None,
        device_id: int | None = None,
        integrator: str | None = None,
        constraint_solver: str | None = None,
        friction_cone: str | None = None,
        solver_iterations: int | None = None,
        **unexpected_kwargs: Any,
    ) -> None:
        require_scene_composition_support(scene, "genesis")
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"GenesisBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if solver_iterations is not None and (
            isinstance(solver_iterations, bool)
            or not isinstance(solver_iterations, int)
            or solver_iterations <= 0
        ):
            raise ValueError(
                f"genesis solver_iterations must be a positive integer or None, "
                f"got {solver_iterations!r}"
            )
        if device_id is not None and (
            isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0
        ):
            raise ValueError(
                f"genesis device_id must be a non-negative integer or None, got {device_id!r}"
            )
        if push_body_name is not None:
            raise NotImplementedError(
                "genesis backend does not support interval push or external wrench "
                "randomization; remove domain_rand.push_body_name and disable push_robots."
            )

        deps = dependencies.load_genesis_dependencies()
        self._deps = deps
        self._torch = deps.torch
        self._gs = deps.genesis
        self._device_id = None if device_id is None else int(device_id)
        self._portable_mode = bool(scene.entity_assets)
        self._entity_layout: CompiledSceneLayout | None = None
        self._variant_assignment: np.ndarray | None = None
        self._entity_runtimes: dict[str, _GenesisEntityRuntime] = {}
        self._sensor_link_bindings: tuple[_GenesisSensorLinkBinding, ...] = ()
        self._sensor_contact_bindings: dict[str, _GenesisContactSensorBinding] = {}
        self._contact_sensor_rows_valid = np.zeros((int(num_envs),), dtype=np.bool_)
        self._sensor_link_pos_cache: np.ndarray | None = None
        self._sensor_link_quat_cache: np.ndarray | None = None
        self._entity_faulted = False
        self._portable_body_link_ids: np.ndarray | None = None
        self._portable_force_body_ids: np.ndarray | None = None
        self._portable_pending_body_forces: np.ndarray | None = None
        self._portable_pending_body_torques: np.ndarray | None = None
        if self._portable_mode:
            from unisim.mjcf_compiler import compose_scene

            composed = compose_scene(scene, num_envs, float(sim_dt))
            self._composed_scene = composed
            variant_plan = composed.variant_plan
            variant_count = 1 if variant_plan is None else len(variant_plan.variants)
            assignment = (
                materialization.validate_genesis_variant_assignment(
                    variant_plan.assignment, variant_count
                )
                if variant_plan is not None
                else np.zeros((num_envs,), dtype=np.int32)
            )
            portable_sources = materialization.prepare_genesis_portable_sources(
                deps.mujoco, scene, composed.layout
            )
            self._portable_sources = portable_sources
            self._metadata = materialization.scan_genesis_portable_composed_metadata(
                deps.mujoco, composed
            )
            self._sensor_plans = materialization.validate_genesis_portable_sensor_plans(
                deps.mujoco, portable_sources, composed.layout, self._metadata.sensor_plans
            )
            self._entity_layout = composed.layout
            self._variant_assignment = assignment
            self._scene_model_file = composed.model_file
        else:
            self._metadata = materialization.scan_genesis_model_metadata(deps.mujoco, scene)
            self._sensor_plans = self._metadata.sensor_plans
            self._scene_cleanup_handle = self._metadata.cleanup_handle
            self._scene_model_file = str(scene.model_file)

        # One gs.init per process; re-init after destroy fails closed here.
        materialization.init_genesis_session(deps, device_id=self._device_id)
        self._device = (
            self._torch.device(
                "cuda",
                self._device_id
                if self._device_id is not None
                else self._torch.cuda.current_device(),
            )
            if (self._torch.cuda.is_available())
            else self._torch.device("cpu")
        )

        self._scene = materialization.build_genesis_scene(
            deps,
            sim_dt=float(sim_dt),
            gravity=self._metadata.gravity,
            integrator=integrator,
            constraint_solver=constraint_solver,
            friction_cone=friction_cone,
            solver_iterations=solver_iterations,
        )
        self._entity: Any
        self._entity_runtimes = {}
        native_entities: list[Any] = []
        physical_native_entities: list[Any] = []
        if self._portable_mode:
            assert self._portable_sources is not None
            for source in self._portable_sources.entities:
                entity_spec = next(item for item in scene.entity_assets if item.name == source.name)
                morph_kwargs: dict[str, Any] = {}
                # Genesis consumes the declared root pose through its public morph
                # API. Passing it only for fixed roots leaves a floating root's
                # normalized free-joint pose at the origin.
                morph_kwargs.update(
                    pos=entity_spec.initial_state.position,
                    quat=entity_spec.initial_state.quaternion,
                )
                morphs = [
                    self._gs.morphs.MJCF(file=path, **morph_kwargs) for path in source.model_files
                ]
                material = (
                    self._gs.materials.Kinematic()
                    if entity_spec.mirror_of is not None
                    else None
                )
                native_entity = self._scene.add_entity(
                    morphs[0] if len(morphs) == 1 else morphs,
                    material=material,
                    name=source.name,
                )
                native_entities.append(native_entity)
                if entity_spec.mirror_of is None:
                    physical_native_entities.append(native_entity)
            self._entity = physical_native_entities[0]
        else:
            self._entity = self._scene.add_entity(
                self._gs.morphs.MJCF(file=self._metadata.source_model_file)
            )
        # One IMUSensor per accelerometer site (REPORT §3.4 equivalent); the
        # link index resolves pre-build from the cold-path entity structure.
        self._imu_sensors: dict[str, Any] = {}
        portable_entities = (
            {str(entity.name): entity for entity in native_entities} if self._portable_mode else {}
        )
        for plan in self._sensor_plans:
            if plan.kind != "accelerometer" or plan.name in self._imu_sensors:
                continue
            assert plan.site_pos is not None  # guaranteed by the cold-path scan
            entity = self._entity
            body_name = plan.body_name
            if self._portable_mode:
                entity_name, separator, local_body_name = plan.body_name.partition("/")
                if not separator:
                    raise RuntimeError(
                        f"genesis portable accelerometer {plan.name!r} has no entity owner"
                    )
                entity = portable_entities[entity_name]
                body_name = local_body_name
            link = entity.get_link(body_name)
            self._imu_sensors[plan.name] = self._scene.add_sensor(
                self._gs.sensors.IMU(
                    entity_idx=entity.idx,
                    link_idx_local=int(link.idx_local),
                    pos_offset=tuple(plan.site_pos),
                )
            )

        self._pre_step_control_fn = None
        self.backend_type = "genesis"
        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._base_name = base_name
        # Root dims and name->index maps come from the cold-path MJCF scan
        # (mjwarp-style), so cold metadata like get_default_dof_pos and the
        # joint index getters are correct before materialize(); the
        # materialize-time binding validates the live import against them.
        if self._portable_mode:
            assert self._entity_layout is not None
            self._root_qpos_dim = 7
            self._root_qvel_dim = 6
            self._body_ids = {
                f"{entity.name}/{body_name}": body_id
                for entity in self._entity_layout.entities
                for body_name, body_id in zip(entity.body_names, entity.body_ids, strict=True)
            }
            self._joint_dof_ids = {
                f"{entity.name}/{joint.name}": joint.qvel_indices[0]
                for entity in self._entity_layout.entities
                for joint in entity.joints
            }
            self._joint_qpos_ids = {
                f"{entity.name}/{joint.name}": joint.qpos_indices[0]
                for entity in self._entity_layout.entities
                for joint in entity.joints
            }
        else:
            self._root_qpos_dim = self._metadata.root_qpos_dim
            self._root_qvel_dim = self._metadata.root_qvel_dim
            self._body_ids = {name: idx for idx, name in enumerate(self._metadata.body_names)}
            self._joint_dof_ids = dict(
                zip(self._metadata.joint_names, self._metadata.joint_dof_adrs, strict=True)
            )
            self._joint_qpos_ids = dict(
                zip(self._metadata.joint_names, self._metadata.joint_qpos_adrs, strict=True)
            )
        self._materialized = False
        self._closed = False
        # Native rendering state (post-build lazy viewer/camera; see the play
        # contract section below). ``_render_config`` pins the first
        # init_renderer(headless, capture) pair like the isaacgym backend.
        self._render_config: tuple[bool, bool] | None = None
        self._viewer: Any | None = None
        self._render_camera: Any | None = None
        self._camera_cfg: CameraCfg = CameraCfg()
        self._camera_tracking_env_idx: int | None = None

    # ------------------------------------------------------------------ #
    # Materialization-time binding                                        #
    # ------------------------------------------------------------------ #

    def materialize(self) -> None:
        """Build the batched scene, cross-check the import, and bind caches.

        Idempotent, and called lazily by the first state access: env
        constructors that validate state shapes before the explicit
        lifecycle point (ManagerBasedRlEnv builds its EntityScene before
        calling ``materialize()``, #1382) work like they do on the MuJoCo
        backend — the same pattern as the isaacgym backend's lazy
        materialize.  A closed backend cannot be materialized again.
        """
        if self._materialized:
            return
        if self._closed:
            raise RuntimeError("genesis backend is closed and cannot be materialized again")
        self._scene.build(n_envs=self._num_envs)
        self._materialized = True
        self._bind_materialized_metadata()

    def _bind_materialized_metadata(self) -> None:
        metadata = self._metadata
        if self._portable_mode:
            self._bind_portable_metadata()
            torch = self._torch
            n = self._num_envs
            self._sensor_slots, sensor_constants, total_dim = self._bind_sensor_slots()
            self._sensor_constants = sensor_constants
            self._bind_portable_sensor_link_frames()
            self._bind_portable_contact_sensors()
            self._contact_sensor_rows_valid = np.zeros((n,), dtype=np.bool_)
            self._sensor_link_pos_cache = np.zeros(
                (n, len(self._sensor_plans), 3), dtype=np.float32
            )
            self._sensor_link_quat_cache = np.zeros(
                (n, len(self._sensor_plans), 4), dtype=np.float32
            )
            self._qpos_cache = _make_device_cache(torch, (n, metadata.nq))
            self._qvel_cache = _make_device_cache(torch, (n, metadata.nv))
            self._links_pos_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
            self._links_quat_cache = _make_device_cache(torch, (n, metadata.nbody, 4))
            self._links_vel_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
            self._links_ang_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
            self._contact_force_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
            self._sensor_cache = np.zeros((n, total_dim), dtype=np.float32)
            self._imu_caches = {
                name: _make_device_cache(torch, (n, 3)) for name in self._imu_sensors
            }
            self._time_cache = np.zeros((n,), dtype=np.float32)
            self._refresh_host_cache()
            return
        entity = self._entity
        if int(entity.n_dofs) != metadata.nv or int(entity.n_qs) != metadata.nq:
            raise RuntimeError(
                f"genesis MJCF import mismatch: n_dofs/n_qs {entity.n_dofs}/{entity.n_qs} "
                f"!= MJCF nv/nq {metadata.nv}/{metadata.nq}"
            )
        if int(entity.n_links) != metadata.nbody:
            raise RuntimeError(
                f"genesis MJCF import mismatch: n_links {entity.n_links} != MJCF nbody "
                f"{metadata.nbody}"
            )
        link_names = tuple(str(link.name) for link in entity.links)
        if link_names != metadata.body_names:
            raise RuntimeError(
                f"genesis MJCF import mismatch: link names/order {link_names} != MJCF body "
                f"names {metadata.body_names}"
            )
        one_dof_joints = [joint for joint in entity.joints if int(joint.n_dofs) == 1]
        joint_names = tuple(str(joint.name) for joint in one_dof_joints)
        if joint_names != metadata.joint_names:
            raise RuntimeError(
                f"genesis MJCF import mismatch: joint names/order {joint_names} != MJCF "
                f"single-DoF joints {metadata.joint_names}"
            )
        entity_dof_ids = {
            name: int(joint.dofs_idx_local[0])
            for name, joint in zip(metadata.joint_names, one_dof_joints, strict=True)
        }
        entity_qpos_ids = {
            name: int(joint.qs_idx_local[0])
            for name, joint in zip(metadata.joint_names, one_dof_joints, strict=True)
        }
        if entity_dof_ids != self._joint_dof_ids or entity_qpos_ids != self._joint_qpos_ids:
            raise RuntimeError(
                "genesis MJCF import mismatch: joint dof/qpos indices do not match the "
                "scanned MJCF layout"
            )
        # Actuator order: MJCF actuator-target joints map 1:1 onto actuated
        # dofs (REPORT §3.1 [1b]); gains are cross-checked against the import.
        self._actuated_dofs = [
            self._joint_dof_ids[joint_name] for joint_name in metadata.actuator_joint_names
        ]
        imported_kp = entity.get_dofs_kp().cpu().numpy()[0, self._actuated_dofs]
        imported_kv = entity.get_dofs_kv().cpu().numpy()[0, self._actuated_dofs]
        if not np.allclose(imported_kp, metadata.actuator_kp, atol=1e-4) or not np.allclose(
            imported_kv, metadata.actuator_kv, atol=1e-4
        ):
            raise RuntimeError(
                "genesis MJCF import mismatch: imported dof kp/kv do not match the MJCF "
                "position-actuator gains (REPORT §3.1 [3a] expects PD-reducible gains)."
            )

        self._base_link_idx: int | None = None
        if self._base_name is not None:
            try:
                self._base_link_idx = self._body_ids[self._base_name]
            except KeyError as exc:
                raise ValueError(
                    f"Base body {self._base_name!r} not found in genesis model"
                ) from exc
            root_layout = self.get_root_state_layout(self._base_name)
            if (
                len(root_layout.qpos_indices) != self._root_qpos_dim
                or len(root_layout.qvel_indices) != self._root_qvel_dim
            ):
                raise RuntimeError(
                    f"genesis MJCF import mismatch: base link {self._base_name!r} root "
                    "layout does not match the scanned free-root block"
                )
        self._link_start = int(entity.link_start)

        self._sensor_slots, sensor_constants, total_dim = self._bind_sensor_slots()
        self._sensor_constants = sensor_constants

        torch = self._torch
        n = self._num_envs
        self._qpos_cache = _make_device_cache(torch, (n, metadata.nq))
        self._qvel_cache = _make_device_cache(torch, (n, metadata.nv))
        self._links_pos_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._links_quat_cache = _make_device_cache(torch, (n, metadata.nbody, 4))
        self._links_vel_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._links_ang_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._contact_force_cache = _make_device_cache(torch, (n, metadata.nbody, 3))
        self._sensor_cache = np.zeros((n, total_dim), dtype=np.float32)
        self._imu_caches = {name: _make_device_cache(torch, (n, 3)) for name in self._imu_sensors}
        self._time_cache = np.zeros((n,), dtype=np.float32)
        self._refresh_host_cache()

    def _bind_portable_sensor_link_frames(self) -> None:
        """Bind site and body sensors to Genesis user link frames."""

        bindings: list[_GenesisSensorLinkBinding] = []
        for plan in self._sensor_plans:
            entity_name = plan.body_name.partition("/")[0]
            runtime = self._entity_runtimes[entity_name]
            public_body_id = self._body_ids[plan.body_name]
            local_matches = np.flatnonzero(runtime.body_ids == public_body_id)
            if local_matches.size != 1:
                raise RuntimeError(
                    f"genesis site/body sensor {plan.name!r} does not bind to exactly one "
                    f"native body in entity {entity_name!r}"
                )
            native_body = runtime.native_body_indices[int(local_matches[0])]
            body_ipos: np.ndarray | None = None
            body_iquat: np.ndarray | None = None
            if plan.object_kind == "body":
                _, _, local_body_name = plan.body_name.partition("/")
                source_body_indices = [
                    metadata.body_names.index(local_body_name)
                    for metadata in runtime.source_metadata
                ]
                body_ipos = np.asarray(
                    [
                        metadata.body_ipos[source_body_index]
                        for metadata, source_body_index in zip(
                            runtime.source_metadata, source_body_indices, strict=True
                        )
                    ],
                    dtype=np.float64,
                )
                body_iquat = np.asarray(
                    [
                        metadata.body_iquat[source_body_index]
                        for metadata, source_body_index in zip(
                            runtime.source_metadata, source_body_indices, strict=True
                        )
                    ],
                    dtype=np.float64,
                )
                if body_ipos.shape != (len(runtime.source_metadata), 3) or (
                    body_iquat.shape != (len(runtime.source_metadata), 4)
                ):
                    raise RuntimeError(
                        f"genesis portable body sensor {plan.name!r} has malformed "
                        "variant inertial identity"
                    )
            bindings.append(
                _GenesisSensorLinkBinding(
                    runtime=runtime,
                    native_body=int(native_body),
                    body_ipos=body_ipos,
                    body_iquat=body_iquat,
                )
            )
        self._sensor_link_bindings = tuple(bindings)

    def _bind_portable_contact_sensors(self) -> None:
        """Bind exact cross-entity contact fragments to native collision IDs."""

        assert self._entity_layout is not None
        assert self._variant_assignment is not None
        public_geoms = {
            f"{entity.name}/{geom.name}": (entity.name, index)
            for entity in self._entity_layout.entities
            for index, geom in enumerate(entity.geoms)
        }
        bindings: dict[str, _GenesisContactSensorBinding] = {}
        for plan in self._sensor_plans:
            if plan.kind != "contact":
                continue
            assert plan.contact_geom1_name is not None
            assert plan.contact_geom2_name is not None
            geom1 = public_geoms.get(plan.contact_geom1_name)
            geom2 = public_geoms.get(plan.contact_geom2_name)
            if geom1 is None or geom2 is None or geom1[0] == geom2[0]:
                raise RuntimeError(
                    f"genesis contact sensor {plan.name!r} does not resolve to two "
                    "distinct public entities"
                )
            runtime1 = self._entity_runtimes[geom1[0]]
            runtime2 = self._entity_runtimes[geom2[0]]
            geom1_ids = np.empty((self._num_envs,), dtype=np.int64)
            geom2_ids = np.empty((self._num_envs,), dtype=np.int64)
            for row, variant in enumerate(self._variant_assignment):
                variant1 = variant if len(runtime1.source_metadata) > 1 else 0
                variant2 = variant if len(runtime2.source_metadata) > 1 else 0
                geom1_id = int(runtime1.collision_geom_indices[variant1, geom1[1]])
                geom2_id = int(runtime2.collision_geom_indices[variant2, geom2[1]])
                if geom1_id < 0 or geom2_id < 0 or geom1_id == geom2_id:
                    raise RuntimeError(
                        f"genesis contact sensor {plan.name!r} has absent or ambiguous "
                        "native collision geom identity"
                    )
                geom1_ids[row] = geom1_id
                geom2_ids[row] = geom2_id
            bindings[plan.name] = _GenesisContactSensorBinding(
                name=plan.name,
                entity1=runtime1.entity,
                entity2=runtime2.entity,
                geom1_ids=geom1_ids,
                geom2_ids=geom2_ids,
                netforce=plan.contact_netforce,
            )
        self._sensor_contact_bindings = bindings

    def _bind_portable_metadata(self) -> None:
        """Audit each native entity and bind it to the frozen public layout."""

        assert self._entity_layout is not None
        assert self._portable_sources is not None
        assert self._variant_assignment is not None
        sources = {item.name: item for item in self._portable_sources.entities}
        runtimes: dict[str, _GenesisEntityRuntime] = {}
        for owner in self._entity_layout.entities:
            source = sources[owner.name]
            candidates = [
                native_entity
                for native_entity in self._scene.entities
                if str(native_entity.name) == owner.name
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"genesis entity name binding found {len(candidates)} native entities "
                    f"named {owner.name!r}"
                )
            native_entity = candidates[0]
            is_visual_mirror = owner.root_mode == "kinematic"
            metadata = source.metadata[0]
            source_dof_ids = dict(zip(metadata.joint_names, metadata.joint_dof_adrs, strict=True))
            source_qpos_ids = dict(zip(metadata.joint_names, metadata.joint_qpos_adrs, strict=True))
            expected_qpos = len(owner.qpos_indices)
            expected_qvel = len(owner.qvel_indices)
            if int(native_entity.n_qs) != expected_qpos or int(native_entity.n_dofs) != (
                expected_qvel
            ):
                raise RuntimeError(
                    f"genesis entity {owner.name!r} native state dimensions "
                    f"{native_entity.n_qs}/{native_entity.n_dofs} differ from public "
                    f"{expected_qpos}/{expected_qvel}"
                )

            native_links = {str(link.name): link for link in native_entity.links}
            if len(native_links) != int(native_entity.n_links):
                raise RuntimeError(f"genesis entity {owner.name!r} has duplicate native links")
            missing = [name for name in owner.body_names if name not in native_links]
            if missing:
                raise RuntimeError(
                    f"genesis entity {owner.name!r} is missing native links {missing}"
                )
            native_bodies: list[int] = []
            for body_name in owner.body_names:
                native_bodies.append(int(native_links[body_name].idx_local))

            if not is_visual_mirror:
                for body_name in owner.body_names:
                    source_body = metadata.body_names.index(body_name)
                    source_rotation = np_matrix_from_quat(metadata.body_iquat[source_body])
                    expected_inertia = (
                        source_rotation
                        @ np.diag(metadata.body_inertia[source_body].astype(np.float64))
                        @ source_rotation.T
                    )
                    native_inertia = np.asarray(
                        native_links[body_name].inertial_i, dtype=np.float64
                    )
                    if native_inertia.shape != (3, 3) or not np.allclose(
                        native_inertia, expected_inertia, rtol=2e-6, atol=2e-7
                    ):
                        raise RuntimeError(
                            f"genesis entity {owner.name!r} body {body_name!r} native inertia "
                            "differs from its normalized source"
                        )

            expected_geom_names = tuple(geom.name for geom in owner.geoms)
            expected_geom_bodies = tuple(geom.body_name for geom in owner.geoms)
            if metadata.geom_names != expected_geom_names or (
                metadata.geom_body_names != expected_geom_bodies
            ):
                raise RuntimeError(
                    f"genesis entity {owner.name!r} source geom names/ownership differ "
                    "from the frozen public layout"
                )
            native_vgeoms = list(native_entity.vgeoms)
            if len(native_vgeoms) != len(owner.geoms) * len(source.metadata):
                raise RuntimeError(
                    f"genesis entity {owner.name!r} native visual geometry count "
                    f"{len(native_vgeoms)} differs from its source variants"
                )
            matched_native_vgeoms: set[int] = set()
            native_geom_sizes = np.empty(
                (len(source.metadata), len(owner.geoms), 3), dtype=np.float64
            )
            for variant, variant_metadata in enumerate(source.metadata):
                expected_rows = (
                    np.arange(self._num_envs, dtype=np.intp)
                    if len(source.metadata) == 1
                    else np.flatnonzero(self._variant_assignment == variant)
                )
                geom_records = zip(
                    variant_metadata.geom_names,
                    variant_metadata.geom_body_names,
                    variant_metadata.geom_types,
                    variant_metadata.geom_sizes,
                    strict=True,
                )
                for geom_index, (geom_name, geom_body, geom_type, geom_size) in enumerate(
                    geom_records
                ):
                    expected_lower, expected_upper = self._expected_geometry_bounds(
                        geom_type, geom_size
                    )
                    matches: list[int] = []
                    native_bounds: tuple[np.ndarray, np.ndarray] | None = None
                    for native_index, vgeom in enumerate(native_vgeoms):
                        if native_index in matched_native_vgeoms:
                            continue
                        native_metadata = vgeom.metadata
                        if str(native_metadata.get("name", "")) != geom_name or (
                            str(vgeom.link.name) != geom_body
                        ):
                            continue
                        if not np.array_equal(
                            self._native_vgeom_env_ids(vgeom, self._num_envs), expected_rows
                        ):
                            continue
                        vertices = np.asarray(vgeom.init_vverts, dtype=np.float64)
                        if vertices.ndim != 2 or vertices.shape[1] != 3:
                            continue
                        native_lower = np.min(vertices, axis=0)
                        native_upper = np.max(vertices, axis=0)
                        if np.allclose(
                            native_lower, expected_lower, rtol=2e-5, atol=2e-6
                        ) and np.allclose(native_upper, expected_upper, rtol=2e-5, atol=2e-6):
                            matches.append(native_index)
                            native_bounds = (native_lower, native_upper)
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"genesis entity {owner.name!r} geom {geom_name!r} variant "
                            f"{variant} native identity/active-environment binding is "
                            f"ambiguous or mismatched ({len(matches)} matches)"
                        )
                    assert native_bounds is not None
                    native_lower, native_upper = native_bounds
                    matched_native_vgeoms.add(matches[0])
                    native_geom_sizes[variant, geom_index] = self._geometry_size_from_native_bounds(
                        geom_type, native_lower, native_upper
                    )
            geom_sizes_nonuniform = len(source.metadata) > 1 and not np.array_equal(
                native_geom_sizes, native_geom_sizes[0]
            )

            (
                contact_masks,
                geom_frictions,
                geom_solver_params,
                contact_masks_nonuniform,
                geom_frictions_nonuniform,
                geom_solver_params_nonuniform,
            ) = self._bind_portable_collision_properties(
                native_entity,
                owner,
                is_visual_mirror,
                source.metadata,
                self._num_envs,
                self._variant_assignment,
            )
            collision_geom_indices = self._bind_portable_collision_geom_indices(
                native_entity,
                owner,
                is_visual_mirror,
                source.metadata,
                self._num_envs,
                self._variant_assignment,
            )

            one_dof_joints = [joint for joint in native_entity.joints if int(joint.n_dofs) == 1]
            joint_names = tuple(str(joint.name) for joint in one_dof_joints)
            if joint_names != metadata.joint_names:
                raise RuntimeError(
                    f"genesis entity {owner.name!r} native joint order {joint_names} "
                    f"differs from normalized source {metadata.joint_names}"
                )
            native_joint_dofs = {
                name: int(joint.dofs_idx_local[0])
                for name, joint in zip(joint_names, one_dof_joints, strict=True)
            }
            native_joint_qpos = {
                name: int(joint.qs_idx_local[0])
                for name, joint in zip(joint_names, one_dof_joints, strict=True)
            }
            if native_joint_dofs != source_dof_ids or (native_joint_qpos != source_qpos_ids):
                raise RuntimeError(
                    f"genesis entity {owner.name!r} native joint addresses differ from source"
                )

            native_qpos: list[int] = []
            native_qvel: list[int] = []
            if owner.root_mode == "floating":
                free_joints = [
                    joint
                    for joint in native_entity.joints
                    if int(joint.n_dofs) == 6 and int(joint.n_qs) == 7
                ]
                if len(free_joints) != 1:
                    raise RuntimeError(
                        f"genesis floating entity {owner.name!r} must have one native free root"
                    )
                native_qpos.extend(int(value) for value in free_joints[0].qs_idx_local)
                native_qvel.extend(int(value) for value in free_joints[0].dofs_idx_local)
            for joint in owner.joints:
                if joint.kind not in ("hinge", "slide"):
                    raise NotImplementedError(
                        f"genesis portable entity {owner.name!r} supports only hinge/slide joints"
                    )
                native_qpos.append(source_qpos_ids[joint.name])
                native_qvel.append(source_dof_ids[joint.name])

            public_qpos = np.asarray(owner.qpos_indices, dtype=np.intp)
            public_qvel = np.asarray(owner.qvel_indices, dtype=np.intp)
            if public_qpos.size != expected_qpos or public_qvel.size != expected_qvel:
                raise RuntimeError(f"genesis entity {owner.name!r} public state binding is partial")
            native_qvel_indices = np.asarray(native_qvel, dtype=np.intp)
            (
                dof_damping,
                dof_frictionloss,
                dof_armature,
                dof_damping_nonuniform,
                dof_frictionloss_nonuniform,
                dof_armature_nonuniform,
            ) = self._bind_portable_dof_properties(
                native_entity,
                owner,
                is_visual_mirror,
                source.metadata,
                self._num_envs,
                self._variant_assignment,
                native_qvel_indices,
            )
            native_actuated = np.asarray(
                [source_dof_ids[name] for name in owner.actuator_joint_names],
                dtype=np.intp,
            )
            if native_actuated.size:
                imported_kp = native_entity.get_dofs_kp().cpu().numpy()[0, native_actuated]
                imported_kv = native_entity.get_dofs_kv().cpu().numpy()[0, native_actuated]
                if not np.allclose(imported_kp, metadata.actuator_kp, atol=1e-4) or (
                    not np.allclose(imported_kv, metadata.actuator_kv, atol=1e-4)
                ):
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} native PD gains differ from source"
                    )

            if len(source.metadata) > 1 and not is_visual_mirror:
                native_mass = np.asarray(
                    native_entity.get_links_inertial_mass(native_bodies).cpu().numpy()
                )
                native_mass = native_mass.reshape(self._num_envs, len(native_bodies))
                expected_mass = np.empty_like(native_mass)
                for row, variant in enumerate(self._variant_assignment):
                    variant_metadata = source.metadata[int(variant)]
                    for column, body_name in enumerate(owner.body_names):
                        expected_mass[row, column] = variant_metadata.body_mass[
                            variant_metadata.body_names.index(body_name)
                        ]
                if not np.allclose(native_mass, expected_mass, rtol=2e-6, atol=1e-7):
                    raise RuntimeError(
                        f"genesis entity {owner.name!r} native variant masses differ from "
                        "their normalized sources"
                    )

            runtimes[owner.name] = _GenesisEntityRuntime(
                entity=native_entity,
                is_visual_mirror=is_visual_mirror,
                qpos_indices=public_qpos,
                qvel_indices=public_qvel,
                native_qpos_indices=np.asarray(native_qpos, dtype=np.intp),
                native_qvel_indices=native_qvel_indices,
                actuator_indices=np.asarray(owner.actuator_indices, dtype=np.intp),
                native_actuated_dofs=native_actuated,
                body_ids=np.asarray(owner.body_ids, dtype=np.intp),
                native_body_indices=np.asarray(native_bodies, dtype=np.intp),
                source_metadata=source.metadata,
                collision_geom_indices=collision_geom_indices,
                geom_sizes=native_geom_sizes,
                geom_sizes_nonuniform=geom_sizes_nonuniform,
                contact_masks=contact_masks,
                contact_masks_nonuniform=contact_masks_nonuniform,
                geom_frictions=geom_frictions,
                geom_frictions_nonuniform=geom_frictions_nonuniform,
                geom_solver_params=geom_solver_params,
                geom_solver_params_nonuniform=geom_solver_params_nonuniform,
                dof_damping=dof_damping,
                dof_damping_nonuniform=dof_damping_nonuniform,
                dof_frictionloss=dof_frictionloss,
                dof_frictionloss_nonuniform=dof_frictionloss_nonuniform,
                dof_armature=dof_armature,
                dof_armature_nonuniform=dof_armature_nonuniform,
            )
        self._entity_runtimes = runtimes
        body_link_ids = np.full((self._metadata.nbody,), -1, dtype=np.intp)
        force_body_ids = np.zeros((self._metadata.nbody,), dtype=np.bool_)
        public_bodies = np.concatenate([runtime.body_ids for runtime in runtimes.values()])
        if np.unique(public_bodies).size != public_bodies.size or not np.array_equal(
            np.sort(public_bodies), np.arange(1, self._metadata.nbody, dtype=np.intp)
        ):
            raise RuntimeError("genesis portable public-to-native body mapping is incomplete")
        for runtime in runtimes.values():
            if not runtime.is_visual_mirror:
                global_links = int(runtime.entity.link_start) + runtime.native_body_indices
                occupied = body_link_ids[runtime.body_ids]
                if np.any(occupied >= 0) or np.unique(global_links).size != global_links.size:
                    raise RuntimeError(
                        "genesis portable public-to-native physical body mapping is ambiguous"
                    )
                body_link_ids[runtime.body_ids] = global_links
                force_body_ids[runtime.body_ids] = True
        body_link_ids[0] = 0
        self._portable_body_link_ids = body_link_ids
        self._portable_force_body_ids = force_body_ids
        self._portable_pending_body_forces = np.zeros(
            (self._num_envs, self._metadata.nbody, 3), dtype=np.float32
        )
        self._portable_pending_body_torques = np.zeros(
            (self._num_envs, self._metadata.nbody, 3), dtype=np.float32
        )
        self._entity = next(
            runtime.entity for runtime in runtimes.values() if not runtime.is_visual_mirror
        )
        self._actuated_dofs = []
        self._capture_portable_default_roots()
        self._apply_portable_default_state()
        body_mass = np.broadcast_to(
            self._metadata.body_mass, (self._num_envs, self._metadata.nbody)
        ).copy()
        body_ipos = np.broadcast_to(
            self._metadata.body_ipos[None],
            (self._num_envs, self._metadata.nbody, 3),
        ).copy()
        for owner, runtime in zip(self._entity_layout.entities, runtimes.values(), strict=True):
            if runtime.is_visual_mirror or len(runtime.source_metadata) == 1:
                continue
            for row, variant in enumerate(self._variant_assignment):
                metadata = runtime.source_metadata[int(variant)]
                for body_name, public_body_id in zip(owner.body_names, owner.body_ids, strict=True):
                    source_id = metadata.body_names.index(body_name)
                    body_mass[row, public_body_id] = metadata.body_mass[source_id]
                    body_ipos[row, public_body_id] = metadata.body_ipos[source_id]
        self._body_mass_cache = body_mass
        self._portable_default_body_ipos = body_ipos.copy()
        self._body_ipos_cache = body_ipos

        self._base_link_idx = None
        if self._base_name is not None:
            try:
                self._base_link_idx = self._body_ids[self._base_name]
            except KeyError as exc:
                raise ValueError(
                    f"Base body {self._base_name!r} not found in genesis portable model"
                ) from exc
            self.get_root_state_layout(self._base_name)

    def _apply_portable_default_state(self) -> None:
        """Apply cold source defaults through public per-entity Genesis APIs."""

        assert self._variant_assignment is not None
        for runtime in self._entity_runtimes.values():
            metadata = runtime.source_metadata
            root_qpos_dim = metadata[0].root_qpos_dim
            root_qvel_dim = metadata[0].root_qvel_dim
            variants = (
                self._variant_assignment
                if len(metadata) > 1
                else np.zeros(self._num_envs, dtype=np.int32)
            )
            scalar_dofs = runtime.native_qvel_indices[root_qvel_dim:]
            if scalar_dofs.size:
                desired_qpos = np.stack(
                    [metadata[int(variant)].default_qpos[root_qpos_dim:] for variant in variants]
                )
                current_qpos = np.asarray(runtime.entity.get_qpos().cpu().numpy())[
                    :, runtime.native_qpos_indices[root_qpos_dim:]
                ]
                qpos_deltas = desired_qpos - current_qpos
                runtime.entity.set_dofs_position(
                    self._to_device(qpos_deltas),
                    dofs_idx_local=scalar_dofs.tolist(),
                    zero_velocity=False,
                )
                qvel = np.stack(
                    [metadata[int(variant)].default_qvel[root_qvel_dim:] for variant in variants]
                )
                runtime.entity.set_dofs_velocity(
                    self._to_device(qvel),
                    dofs_idx_local=scalar_dofs.tolist(),
                )
            if runtime.native_actuated_dofs.size:
                ctrl = np.stack([metadata[int(variant)].default_ctrl for variant in variants])
                runtime.entity.control_dofs_position(
                    self._to_device(ctrl),
                    dofs_idx_local=runtime.native_actuated_dofs.tolist(),
                )

    def _capture_portable_default_roots(self) -> None:
        """Freeze native construction root poses before any runtime mutation."""

        assert self._entity_layout is not None
        roots = np.zeros((self._num_envs, len(self._entity_layout.entities), 13), dtype=np.float32)
        for entity_index, runtime in enumerate(self._entity_runtimes.values()):
            roots[:, entity_index, :3] = np.asarray(
                runtime.entity.get_links_pos(relative=False).cpu().numpy()
            )[:, int(runtime.native_body_indices[0])]
            roots[:, entity_index, 3:7] = np.asarray(
                runtime.entity.get_links_quat(relative=False).cpu().numpy()
            )[:, int(runtime.native_body_indices[0])]
        self._portable_default_roots = roots

    def _bind_sensor_slots(self) -> tuple[dict[str, tuple[int, int]], dict[str, tuple], int]:
        slots: dict[str, tuple[int, int]] = {}
        constants: dict[str, tuple] = {}
        address = 0
        for plan in self._sensor_plans:
            if plan.body_name not in self._body_ids:
                raise RuntimeError(
                    f"genesis sensor {plan.name!r} references missing body {plan.body_name!r}"
                )
            link_idx = self._body_ids[plan.body_name]
            slots[plan.name] = (address, plan.dim)
            if plan.kind == "contact":
                constants[plan.name] = (link_idx,)
            elif plan.object_kind == "body":
                constants[plan.name] = (link_idx,)
            else:
                assert plan.site_pos is not None and plan.site_quat is not None
                constants[plan.name] = (
                    link_idx,
                    np.asarray(plan.site_pos, dtype=np.float64),
                    np.asarray(plan.site_quat, dtype=np.float64),
                )
            address += plan.dim
        return slots, constants, address

    # ------------------------------------------------------------------ #
    # Host-cache barriers                                                 #
    # ------------------------------------------------------------------ #

    def _require_state(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"genesis backend is closed; cannot run {operation}")
        if self._entity_faulted:
            raise RuntimeError(
                f"genesis backend state is faulted; cannot run {operation} after a partial "
                "native submission"
            )
        # Lazy, idempotent materialize: the first state read builds the scene.
        self.materialize()

    def _refresh_host_cache(self) -> None:
        """Refresh every legacy-visible cache at one explicit lifecycle barrier."""
        if self._portable_mode:
            self._qpos_cache[1].fill(0.0)
            self._qvel_cache[1].fill(0.0)
            self._links_pos_cache[1].fill(0.0)
            self._links_quat_cache[1].fill(0.0)
            self._links_vel_cache[1].fill(0.0)
            self._links_ang_cache[1].fill(0.0)
            self._contact_force_cache[1].fill(0.0)
            for runtime in self._entity_runtimes.values():
                native = runtime.entity
                if runtime.qpos_indices.size:
                    self._qpos_cache[1][:, runtime.qpos_indices] = (
                        native.get_qpos().cpu().numpy()[:, runtime.native_qpos_indices]
                    )
                if runtime.qvel_indices.size:
                    self._qvel_cache[1][:, runtime.qvel_indices] = (
                        native.get_dofs_velocity().cpu().numpy()[:, runtime.native_qvel_indices]
                    )
                links_pos = (
                    native.get_links_pos(relative=False)
                    .cpu()
                    .numpy()[:, runtime.native_body_indices]
                )
                links_quat = (
                    native.get_links_quat(relative=False)
                    .cpu()
                    .numpy()[:, runtime.native_body_indices]
                )
                links_vel = native.get_links_vel().cpu().numpy()[:, runtime.native_body_indices]
                links_ang = native.get_links_ang().cpu().numpy()[:, runtime.native_body_indices]
                contact = (
                    np.zeros((self._num_envs, runtime.native_body_indices.size, 3), np.float32)
                    if runtime.is_visual_mirror
                    else native.get_links_net_contact_force()
                    .cpu()
                    .numpy()[:, runtime.native_body_indices]
                )
                self._links_pos_cache[1][:, runtime.body_ids] = links_pos
                self._links_quat_cache[1][:, runtime.body_ids] = links_quat
                self._links_vel_cache[1][:, runtime.body_ids] = links_vel
                self._links_ang_cache[1][:, runtime.body_ids] = links_ang
                self._contact_force_cache[1][:, runtime.body_ids] = contact
            if self._sensor_link_pos_cache is None or self._sensor_link_quat_cache is None:
                raise RuntimeError("genesis portable site sensor frame caches are unbound")
            for sensor_index, binding in enumerate(self._sensor_link_bindings):
                native = binding.runtime.entity
                self._sensor_link_pos_cache[:, sensor_index] = (
                    native.get_links_pos(binding.native_body, relative=True)
                    .cpu()
                    .numpy()
                    .reshape(self._num_envs, -1, 3)[:, 0]
                )
                self._sensor_link_quat_cache[:, sensor_index] = (
                    native.get_links_quat(binding.native_body, relative=True)
                    .cpu()
                    .numpy()
                    .reshape(self._num_envs, -1, 4)[:, 0]
                )
            for name, sensor in self._imu_sensors.items():
                self._imu_caches[name][0].copy_(sensor.read().lin_acc)
            self._refresh_sensor_cache()
            return
        entity = self._entity
        self._qpos_cache[0].copy_(entity.get_qpos())
        self._qvel_cache[0].copy_(entity.get_dofs_velocity())
        self._links_pos_cache[0].copy_(entity.get_links_pos())
        self._links_quat_cache[0].copy_(entity.get_links_quat())
        self._links_vel_cache[0].copy_(entity.get_links_vel())
        self._links_ang_cache[0].copy_(entity.get_links_ang())
        self._contact_force_cache[0].copy_(entity.get_links_net_contact_force())
        for name, sensor in self._imu_sensors.items():
            self._imu_caches[name][0].copy_(sensor.read().lin_acc)
        self._refresh_sensor_cache()

    def _refresh_sensor_cache(self) -> None:
        """Compute MJCF-named sensors from link caches (REPORT §3.4 mappings)."""
        for sensor_index, plan in enumerate(self._sensor_plans):
            address, dim = self._sensor_slots[plan.name]
            out = self._sensor_cache[:, address : address + dim]
            if plan.kind == "contact":
                if not self._portable_mode:
                    (link_idx,) = self._sensor_constants[plan.name]
                    force = self._contact_force_cache[1][:, link_idx, :]
                    magnitude = np.linalg.norm(force, axis=-1, keepdims=True)
                    threshold = materialization.CONTACT_FOUND_FORCE_THRESHOLD_N
                    out[...] = (magnitude > threshold).astype(np.float32)
                else:
                    if plan.contact_netforce:
                        out[...] = self._read_portable_contact(plan.name, netforce=True)
                    else:
                        out[:, 0] = self._read_portable_contact(plan.name, netforce=False)
                continue
            if plan.kind == "accelerometer":
                out[...] = self._imu_caches[plan.name][1]
                continue
            if self._portable_mode and plan.object_kind == "body":
                assert plan.kind in (
                    "framepos",
                    "framequat",
                    "framelinvel",
                    "frameangvel",
                )
                if self._sensor_link_pos_cache is None or self._sensor_link_quat_cache is None:
                    raise RuntimeError("genesis portable body sensor frames are unbound")
                binding = self._sensor_link_bindings[sensor_index]
                assert self._variant_assignment is not None
                if binding.body_ipos is None or binding.body_iquat is None:
                    raise RuntimeError(
                        f"genesis portable body sensor {plan.name!r} lacks inertial identity"
                    )
                (link_idx,) = self._sensor_constants[plan.name]
                variants = self._variant_assignment
                inertial_pos = binding.body_ipos[variants]
                inertial_quat = binding.body_iquat[variants]
                if plan.kind == "framepos":
                    link_pos = self._sensor_link_pos_cache[:, sensor_index]
                    link_quat = self._sensor_link_quat_cache[:, sensor_index]
                    offset_w = np_quat_apply_batched(link_quat, inertial_pos)
                    out[...] = link_pos + offset_w
                elif plan.kind == "framequat":
                    link_quat = self._sensor_link_quat_cache[:, sensor_index]
                    out[...] = np_quat_mul_batched(link_quat, inertial_quat)
                elif plan.kind == "frameangvel":
                    out[...] = self._links_ang_cache[1][:, link_idx, :]
                else:
                    link_quat = self._sensor_link_quat_cache[:, sensor_index]
                    offset_w = np_quat_apply_batched(link_quat, inertial_pos)
                    out[...] = self._links_vel_cache[1][:, link_idx, :] + np.cross(
                        self._links_ang_cache[1][:, link_idx, :], offset_w
                    )
                continue
            link_idx, site_pos, site_quat = self._sensor_constants[plan.name]
            if self._portable_mode and plan.kind in (
                "framepos",
                "framequat",
                "framelinvel",
                "frameangvel",
                "gyro",
                "velocimeter",
            ):
                if self._sensor_link_pos_cache is None or self._sensor_link_quat_cache is None:
                    raise RuntimeError("genesis portable site sensor frame caches are unbound")
                link_pos = self._sensor_link_pos_cache[:, sensor_index]
                link_quat = self._sensor_link_quat_cache[:, sensor_index]
            else:
                link_pos = self._links_pos_cache[1][:, link_idx, :]
                link_quat = self._links_quat_cache[1][:, link_idx, :]
            batch3 = link_quat.shape[:-1] + (3,)
            site_quat_w = np_quat_mul_batched(
                link_quat, np.broadcast_to(site_quat, link_quat.shape)
            )
            if plan.kind == "gyro":
                out[...] = np_quat_apply_inverse_batched(
                    site_quat_w, self._links_ang_cache[1][:, link_idx, :]
                )
            elif plan.kind == "framequat":
                out[...] = site_quat_w
            elif plan.kind == "framezaxis":
                out[...] = np_quat_apply_batched(
                    site_quat_w, np.broadcast_to(_WORLD_Z_AXIS, batch3)
                )
            else:
                offset_w = np_quat_apply_batched(link_quat, np.broadcast_to(site_pos, batch3))
                if plan.kind == "velocimeter":
                    lin_vel_w = self._links_vel_cache[1][:, link_idx, :] + np.cross(
                        self._links_ang_cache[1][:, link_idx, :], offset_w
                    )
                    out[...] = np_quat_apply_inverse_batched(site_quat_w, lin_vel_w)
                elif plan.kind == "framepos":
                    out[...] = link_pos + offset_w
                elif plan.kind == "framelinvel":
                    out[...] = self._links_vel_cache[1][:, link_idx, :] + np.cross(
                        self._links_ang_cache[1][:, link_idx, :], offset_w
                    )
                elif plan.kind == "frameangvel":
                    out[...] = self._links_ang_cache[1][:, link_idx, :]

    @staticmethod
    def _public_contact_array(value: Any, field: str, sensor_name: str, *, ndim: int) -> np.ndarray:
        """Convert one public Genesis contact tensor without device-side mutation."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value)
        if array.ndim != ndim:
            raise RuntimeError(
                f"genesis contact sensor {sensor_name!r} native {field} is not batched"
            )
        return array

    def _read_portable_contact(self, name: str, *, netforce: bool) -> np.ndarray:
        """Read one exact geom-pair contact value from Genesis' public contacts.

        ``netforce=False`` returns a ``(num_envs,)`` found flag.  ``netforce=True``
        returns the ``(num_envs, 3)`` force applied to authored geom1 by geom2,
        summed over all exact valid native contact slots.
        """

        if name not in self._sensor_contact_bindings:
            return np.zeros(
                (self._num_envs, 3) if netforce else (self._num_envs,), dtype=np.float32
            )
        binding = self._sensor_contact_bindings[name]
        if binding.netforce != netforce:
            raise RuntimeError(
                f"genesis contact sensor {name!r} reader disagrees with its bound contact form"
            )
        force_a: np.ndarray = np.empty((0, 0, 3), dtype=np.float32)
        force_b: np.ndarray = np.empty((0, 0, 3), dtype=np.float32)
        try:
            contacts = binding.entity1.get_contacts(binding.entity2)
            geom_a = self._public_contact_array(contacts["geom_a"], "geom_a", name, ndim=2)
            geom_b = self._public_contact_array(contacts["geom_b"], "geom_b", name, ndim=2)
            valid_mask = self._public_contact_array(
                contacts["valid_mask"], "valid_mask", name, ndim=2
            )
            if netforce:
                force_a = self._public_contact_array(contacts["force_a"], "force_a", name, ndim=3)
                force_b = self._public_contact_array(contacts["force_b"], "force_b", name, ndim=3)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"genesis contact sensor {name!r} native public contact read failed"
            ) from exc
        expected = (self._num_envs, None)
        if (
            geom_a.shape[0] != expected[0]
            or geom_b.shape != geom_a.shape
            or valid_mask.shape != geom_a.shape
        ):
            raise RuntimeError(
                f"genesis contact sensor {name!r} native contact shapes are malformed"
            )
        if netforce and (force_a.shape != geom_a.shape + (3,) or force_b.shape != force_a.shape):
            raise RuntimeError(
                f"genesis contact sensor {name!r} native contact force shapes are malformed"
            )
        geom_a = np.asarray(geom_a, dtype=np.int64)
        geom_b = np.asarray(geom_b, dtype=np.int64)
        valid = np.asarray(valid_mask, dtype=bool)
        if np.any(valid & ((geom_a < 0) | (geom_b < 0))):
            raise RuntimeError(
                f"genesis contact sensor {name!r} native contact identity is malformed"
            )
        matched = (
            (geom_a == binding.geom1_ids[:, None]) & (geom_b == binding.geom2_ids[:, None])
        ) | ((geom_a == binding.geom2_ids[:, None]) & (geom_b == binding.geom1_ids[:, None]))
        selected = matched & valid
        if netforce:
            force_a_values = np.asarray(force_a, dtype=np.float64)
            force_b_values = np.asarray(force_b, dtype=np.float64)
            if not (np.all(np.isfinite(force_a_values)) and np.all(np.isfinite(force_b_values))):
                raise RuntimeError(
                    f"genesis contact sensor {name!r} native contact forces are nonfinite"
                )
            force_on_geom1 = np.where(
                (
                    (geom_a == binding.geom1_ids[:, None])
                    & (geom_b == binding.geom2_ids[:, None])
                    & selected
                )[:, :, None],
                force_a_values,
                0.0,
            ) + np.where(
                (
                    (geom_a == binding.geom2_ids[:, None])
                    & (geom_b == binding.geom1_ids[:, None])
                    & selected
                )[:, :, None],
                force_b_values,
                0.0,
            )
            force = force_on_geom1.sum(axis=1)
            force[~self._contact_sensor_rows_valid] = 0.0
            return force.astype(np.float32)
        found = np.any(selected, axis=1)
        found &= self._contact_sensor_rows_valid
        return found.astype(np.float32)

    def _to_device(self, array: np.ndarray) -> Any:
        host = np.ascontiguousarray(array, dtype=np.float32)
        return self._torch.from_numpy(host).to(self._device)

    def _validate_rows(self, env_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1:
            raise ValueError(f"env_indices must be one-dimensional, got shape {rows.shape}")
        if np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError(f"env_indices must be in [0, {self._num_envs}), got {rows}")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicate rows")
        return rows

    # ------------------------------------------------------------------ #
    # SimBackend properties and cold metadata                             #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        """Return the backend-owned Genesis rigid entity."""
        return self._entity

    @property
    def num_actuators(self) -> int:
        return len(self._metadata.actuator_names)

    @property
    def num_dof_vel(self) -> int:
        if self._portable_mode:
            assert self._entity_layout is not None
            return self._entity_layout.nv - sum(
                6 for entity in self._entity_layout.entities if entity.root_mode == "floating"
            )
        return self._metadata.nv - self._root_qvel_dim

    def get_actuator_ctrl_range(self) -> np.ndarray:
        """MJCF ``ctrlrange`` metadata; Genesis does not enforce it in-engine."""
        return self._metadata.actuator_ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_joint_names

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self._metadata.actuator_kp.copy(), self._metadata.actuator_kv.copy()

    def get_scene_model_file(self) -> str | None:
        return self._scene_model_file

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        keyframes = dict(self._metadata.keyframe_qpos)
        try:
            return keyframes[name].copy()
        except KeyError as exc:
            available = ", ".join(sorted(keyframes))
            raise ValueError(f"Keyframe {name!r} not found; available: {available}") from exc

    def get_default_qpos(self) -> np.ndarray:
        return self._metadata.default_qpos.copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return self._metadata.default_qpos[self._root_qpos_dim :].copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._metadata.nv,), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self._portable_mode:
            assert self._entity_layout is not None
            entity_name, separator, local_name = str(root_body_name).partition("/")
            if not separator:
                raise ValueError("portable genesis root names use entity/local_name")
            entity = self._entity_layout.get_entity(entity_name)
            if local_name != entity.root_body:
                raise ValueError(
                    f"root {root_body_name!r} is not entity {entity.name!r}'s root body"
                )
            if entity.root_mode != "floating":
                raise NotImplementedError(
                    f"portable genesis entity {entity.name!r} has no floating root state"
                )
            return BackendRootStateLayout(
                tuple(entity.root_qpos_indices), tuple(entity.root_qvel_indices)
            )
        if root_body_name not in self._body_ids:
            raise ValueError(f"Body {root_body_name!r} not found in genesis model")
        link = self._entity.get_link(root_body_name)
        free_joints = [
            joint for joint in link.joints if int(joint.n_dofs) == 6 and int(joint.n_qs) == 7
        ]
        if len(free_joints) != 1:
            raise NotImplementedError(
                "backend 'genesis' capability 'root-state layout' requires body "
                f"{root_body_name!r} to own exactly one free joint"
            )
        joint = free_joints[0]
        return BackendRootStateLayout(
            qpos_indices=tuple(int(v) for v in joint.qs_idx_local),
            qvel_indices=tuple(int(v) for v in joint.dofs_idx_local),
        )

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            assert self._entity_layout is not None
            return np.asarray(self._entity_layout.get_body_ids(names), dtype=np.int32)
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(self._body_ids[str(name)])
            except KeyError as exc:
                raise ValueError(f"Body {name!r} not found in genesis model") from exc
        return np.asarray(resolved, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return self.get_body_ids(names)
        # ``_body_ids`` follows the MJCF body scan, where worldbody is id 0.
        return self.get_body_ids(names)

    def _portable_geometry_sizes(self) -> np.ndarray:
        assert self._entity_layout is not None
        values: list[np.ndarray] = []
        for entity in self._entity_layout.entities:
            runtime = self._entity_runtimes[entity.name]
            if runtime.geom_sizes_nonuniform:
                raise NotImplementedError(
                    "portable genesis fixed variants do not expose non-uniform "
                    "public geometry sizes"
                )
            values.append(runtime.geom_sizes[0])
        return np.concatenate(values)

    def get_geom_size(self, name: str) -> np.ndarray:
        self._require_state("get_geom_size")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose geom sizes")
        geom_id = self.get_geom_id(name)
        assert self._entity_layout is not None
        entity_name = str(name).partition("/")[0]
        entity = self._entity_layout.get_entity(entity_name)
        entity_offset = 0
        for owner in self._entity_layout.entities:
            if owner.name == entity.name:
                break
            entity_offset += len(owner.geoms)
        local_geom_index = geom_id - entity_offset
        runtime = self._entity_runtimes[entity.name]
        if runtime.geom_sizes_nonuniform:
            raise NotImplementedError(
                "portable genesis fixed variants do not expose non-uniform public geometry sizes"
            )
        return runtime.geom_sizes[0, local_geom_index].copy()

    def get_geom_sizes(self) -> np.ndarray:
        self._require_state("get_geom_sizes")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose geom size defaults")
        return self._portable_geometry_sizes().copy()

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Genesis-native recoded contype/conaffinity of collision geoms.

        Genesis re-synthesizes contype/conaffinity at import: the collision
        matrix semantics are preserved, but the integer values must NOT be
        compared against MuJoCo tables (REPORT #1372 §5.10).
        """
        self._require_state("get_geom_contact_masks")
        if self._portable_mode:
            assert self._entity_layout is not None
            contypes: list[np.ndarray] = []
            conaffinities: list[np.ndarray] = []
            for entity in self._entity_layout.entities:
                runtime = self._entity_runtimes[entity.name]
                if runtime.contact_masks_nonuniform:
                    raise NotImplementedError(
                        "portable genesis fixed variants do not expose non-uniform "
                        "public geometry contact masks"
                    )
                if runtime.contact_masks is None:
                    raise NotImplementedError(
                        "portable genesis native collision identity is unavailable or "
                        "ambiguous for geometry contact masks"
                    )
                contypes.append(runtime.contact_masks[0])
                conaffinities.append(runtime.contact_masks[1])
            return np.concatenate(contypes), np.concatenate(conaffinities)
        return (
            np.asarray([geom.contype for geom in self._entity.geoms], dtype=np.int32),
            np.asarray([geom.conaffinity for geom in self._entity.geoms], dtype=np.int32),
        )

    def get_geom_friction(self) -> np.ndarray:
        """Return audited native Genesis geom-friction coefficients.

        Columns follow Genesis' public coefficient properties as
        ``[sliding, torsional, rolling]``. The values are captured from exact
        collision instances during construction; reset mutation and solver
        feature enablement are separate unsupported semantics.
        """
        self._require_state("get_geom_friction")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom friction")
        assert self._entity_layout is not None
        values: list[np.ndarray] = []
        for entity in self._entity_layout.entities:
            runtime = self._entity_runtimes[entity.name]
            if runtime.geom_frictions_nonuniform:
                raise NotImplementedError(
                    "portable genesis fixed variants do not expose non-uniform "
                    "public geometry friction"
                )
            if runtime.geom_frictions is None:
                raise NotImplementedError(
                    "portable genesis native collision identity is unavailable or "
                    "ambiguous for geometry friction"
                )
            values.append(runtime.geom_frictions)
        return np.concatenate(values, axis=0).copy()

    def get_geom_solref(self) -> np.ndarray:
        """Return audited native Genesis contact reference parameters."""
        self._require_state("get_geom_solref")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom solref")
        assert self._entity_layout is not None
        values: list[np.ndarray] = []
        for entity in self._entity_layout.entities:
            runtime = self._entity_runtimes[entity.name]
            if runtime.geom_solver_params_nonuniform:
                raise NotImplementedError(
                    "portable genesis fixed variants do not expose non-uniform "
                    "public geometry solver parameters"
                )
            if runtime.geom_solver_params is None:
                raise NotImplementedError(
                    "portable genesis native collision identity is unavailable or "
                    "ambiguous for geometry solref"
                )
            values.append(runtime.geom_solver_params[:, :2])
        return np.concatenate(values, axis=0).copy()

    def get_geom_solimp(self) -> np.ndarray:
        """Return audited native Genesis contact impedance parameters."""
        self._require_state("get_geom_solimp")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom solimp")
        assert self._entity_layout is not None
        values: list[np.ndarray] = []
        for entity in self._entity_layout.entities:
            runtime = self._entity_runtimes[entity.name]
            if runtime.geom_solver_params_nonuniform:
                raise NotImplementedError(
                    "portable genesis fixed variants do not expose non-uniform "
                    "public geometry solver parameters"
                )
            if runtime.geom_solver_params is None:
                raise NotImplementedError(
                    "portable genesis native collision identity is unavailable or "
                    "ambiguous for geometry solimp"
                )
            values.append(runtime.geom_solver_params[:, 2:])
        return np.concatenate(values, axis=0).copy()

    def get_geom_id(self, name: str) -> int:
        self._require_state("get_geom_id")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom ids")
        assert self._entity_layout is not None
        return int(self._entity_layout.get_geom_ids((name,))[0])

    def get_geom_names(self) -> tuple[str, ...]:
        self._require_state("get_geom_names")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom names")
        layout = self.get_scene_layout()
        return tuple(
            entity.name + "/" + geom.name for entity in layout.entities for geom in entity.geoms
        )

    def get_geom_body_ids(self) -> np.ndarray:
        self._require_state("get_geom_body_ids")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable geom body ids")
        layout = self.get_scene_layout()
        body_ids = np.empty(layout.ngeom, dtype=np.int32)
        offset = 0
        for entity in layout.entities:
            owners = dict(zip(entity.body_names, entity.body_ids, strict=True))
            for geom in entity.geoms:
                body_ids[offset] = owners[geom.body_name]
                offset += 1
        return body_ids

    def get_gravity(self) -> np.ndarray:
        return self._metadata.gravity.copy()

    def get_body_mass(self) -> np.ndarray:
        if self._portable_mode:
            return self._body_mass_cache.copy()
        return self._metadata.body_mass.copy()

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if self._portable_mode:
            if env_ids is None:
                return self._metadata.body_ipos.copy()
            rows = np.asarray(env_ids, dtype=np.intp)
            if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self._num_envs):
                raise ValueError("env_ids must be a one-dimensional in-range selection")
            return self._body_ipos_cache[rows].copy()
        if env_ids is not None:
            raise NotImplementedError("GenesisBackend does not expose per-environment body ipos")
        return self._metadata.body_ipos.copy()

    def _portable_dof_values(self, field: str) -> np.ndarray:
        assert self._entity_layout is not None
        values = np.full((self._metadata.nv,), np.nan, dtype=np.float64)
        for entity in self._entity_layout.entities:
            runtime = self._entity_runtimes[entity.name]
            if getattr(runtime, f"{field}_nonuniform"):
                raise NotImplementedError(
                    f"portable genesis fixed variants do not expose non-uniform public DOF {field}"
                )
            values[runtime.qvel_indices] = getattr(runtime, field)[0]
        if not np.isfinite(values).all():
            raise RuntimeError("portable genesis public DOF property binding is incomplete")
        return values

    def get_dof_damping(self) -> np.ndarray:
        """Return cold-captured native Genesis DOF damping in public order."""
        self._require_state("get_dof_damping")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable dof damping")
        return self._portable_dof_values("dof_damping")

    def get_dof_frictionloss(self) -> np.ndarray:
        """Return cold-captured native Genesis friction loss in public order."""
        self._require_state("get_dof_frictionloss")
        if not self._portable_mode:
            raise NotImplementedError("GenesisBackend does not expose portable dof frictionloss")
        return self._portable_dof_values("dof_frictionloss")

    def get_dof_armature(self) -> np.ndarray:
        if self._portable_mode:
            self._require_state("get_dof_armature")
            return self._portable_dof_values("dof_armature")
        return self._metadata.dof_armature.copy()

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        self._reject_named_joint_ranges(names, "joint ranges")
        joint_range = self._metadata.joint_range
        return None if joint_range is None else joint_range.copy()

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return np.asarray(self._resolve_joint_ids(names, self._joint_dof_ids), dtype=np.int32)
        return np.asarray(self._resolve_joint_ids(names, self._joint_dof_ids), dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return np.asarray(self._resolve_joint_ids(names, self._joint_qpos_ids), dtype=np.int32)
        qpos_ids = np.asarray(self._resolve_joint_ids(names, self._joint_qpos_ids))
        return (qpos_ids - self._root_qpos_dim).astype(np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return self.get_joint_dof_indices(names)
        return self.get_joint_dof_indices(names) - self._root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        if self._portable_mode:
            return self.get_joint_dof_pos_indices(names)
        return self.get_joint_dof_pos_indices(names) + self._root_qpos_dim

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_vel_indices(names) + self._root_qvel_dim

    def _resolve_joint_ids(self, names: Sequence[str], table: dict[str, int]) -> list[int]:
        resolved: list[int] = []
        for name in names:
            try:
                resolved.append(table[str(name)])
            except KeyError as exc:
                raise ValueError(f"Joint {name!r} not found in genesis model") from exc
        return resolved

    def get_scene_layout(self) -> CompiledSceneLayout:
        if self._entity_faulted:
            raise RuntimeError("genesis backend state is faulted")
        if self._entity_layout is None:
            raise NotImplementedError("genesis model-file scenes do not expose a scene layout")
        return self._entity_layout

    def get_entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.get_scene_layout().entities)

    def _entity_roots(self) -> np.ndarray:
        layout = self.get_scene_layout()
        roots = np.zeros((self._num_envs, len(layout.entities), 13), dtype=np.float32)
        for index, entity in enumerate(layout.entities):
            roots[:, index, :3] = self._links_pos_cache[1][:, entity.body_ids[0]]
            roots[:, index, 3:7] = self._links_quat_cache[1][:, entity.body_ids[0]]
            if entity.root_mode == "floating":
                state = entity_state_snapshot(entity, self._qpos_cache[1], self._qvel_cache[1])
                roots[:, index, 7:] = state["root_velocity"]
        return roots

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        owner = self.get_scene_layout().get_entity(entity)
        if owner.root_mode == "floating":
            return entity_state_snapshot(owner, self._qpos_cache[1], self._qvel_cache[1])
        root = np.zeros((self._num_envs, 13), dtype=np.float32)
        root[:, :7] = self._entity_roots()[:, self.get_scene_layout().entities.index(owner), :7]
        return entity_state_snapshot(owner, self._qpos_cache[1], self._qvel_cache[1], root)

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> Mapping[str, np.ndarray]:
        from unisim.entity_state import selected_state_rows

        self._require_state("get_entity_default_state")
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        rows = selected_state_rows(env_ids, self._num_envs)
        assert self._variant_assignment is not None
        qpos = np.zeros((rows.size, layout.nq), dtype=np.float32)
        qvel = np.zeros((rows.size, layout.nv), dtype=np.float32)
        roots = np.zeros((rows.size, 1, 13), dtype=np.float32)
        runtime = self._entity_runtimes[entity]
        variants = (
            self._variant_assignment[rows]
            if len(runtime.source_metadata) > 1
            else np.zeros(rows.size, dtype=np.int32)
        )
        local_qpos = np.stack(
            [runtime.source_metadata[int(variant)].default_qpos for variant in variants]
        )
        local_qvel = np.stack(
            [runtime.source_metadata[int(variant)].default_qvel for variant in variants]
        )
        qpos[:, runtime.qpos_indices] = local_qpos[:, runtime.native_qpos_indices]
        qvel[:, runtime.qvel_indices] = local_qvel[:, runtime.native_qvel_indices]
        entity_index = layout.entities.index(owner)
        roots[:, 0, :7] = self._portable_default_roots[rows, entity_index, :7]
        if owner.root_mode == "floating":
            qpos[:, owner.root_qpos_indices] = roots[:, 0, :7]
        return entity_state_snapshot(owner, qpos, qvel, roots[:, 0])

    def _commit_portable_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        rows: np.ndarray,
        entity_names: set[str] | None = None,
        roots: np.ndarray | None = None,
    ) -> None:
        self._contact_sensor_rows_valid[rows] = False
        selected_names = set(self._entity_runtimes) if entity_names is None else entity_names
        envs_idx = rows.tolist()
        for entity_name in selected_names:
            runtime = self._entity_runtimes[entity_name]
            if runtime.is_visual_mirror:
                if roots is None:
                    continue
                layout = self.get_scene_layout()
                entity_index = layout.entities.index(layout.get_entity(entity_name))
                pose = roots[:, entity_index, :7]
                runtime.entity.set_pos(
                    self._to_device(np.ascontiguousarray(pose[:, :3])),
                    envs_idx=envs_idx,
                    zero_velocity=False,
                    relative=False,
                )
                runtime.entity.set_quat(
                    self._to_device(np.ascontiguousarray(pose[:, 3:7])),
                    envs_idx=envs_idx,
                    zero_velocity=False,
                    relative=False,
                )
                continue
            if runtime.qpos_indices.size:
                local_qpos = qpos[:, runtime.qpos_indices][rows]
                runtime.entity.set_qpos(
                    self._to_device(local_qpos),
                    envs_idx=envs_idx,
                    zero_velocity=False,
                )
            if runtime.qvel_indices.size:
                local_qvel = qvel[:, runtime.qvel_indices][rows]
                runtime.entity.set_dofs_velocity(self._to_device(local_qvel), envs_idx=envs_idx)

    def reset_entities(self, request: Any) -> None:
        if not self._portable_mode:
            super().reset_entities(request)
            return
        self._require_state("reset_entities")
        layout = self.get_scene_layout()
        prepared = prepare_scene_reset(
            layout,
            request,
            self._qpos_cache[1],
            self._qvel_cache[1],
            self._entity_roots(),
        )
        qpos = self._qpos_cache[1].copy()
        qvel = self._qvel_cache[1].copy()
        qpos_columns = np.flatnonzero(prepared.qpos_mask)
        qvel_columns = np.flatnonzero(prepared.qvel_mask)
        qpos[np.ix_(prepared.env_ids, qpos_columns)] = prepared.qpos[:, qpos_columns]
        qvel[np.ix_(prepared.env_ids, qvel_columns)] = prepared.qvel[:, qvel_columns]
        impacted_bodies: list[int] = []
        for entity_name in prepared.entity_names:
            impacted_bodies.extend(layout.get_entity(entity_name).body_ids)
        try:
            self._cancel_portable_body_wrenches(prepared.env_ids, body_ids=impacted_bodies)
            default_controls = (
                self._portable_default_controls(prepared.env_ids)
                if request.restore_default_controls
                else None
            )
            for entity_name in prepared.entity_names:
                runtime = self._entity_runtimes[entity_name]
                if runtime.native_actuated_dofs.size:
                    controls = (
                        default_controls[:, runtime.actuator_indices]
                        if default_controls is not None
                        else np.zeros(
                            (prepared.env_ids.size, runtime.native_actuated_dofs.size),
                            dtype=np.float32,
                        )
                    )
                    runtime.entity.control_dofs_position(
                        self._to_device(controls),
                        dofs_idx_local=runtime.native_actuated_dofs.tolist(),
                        envs_idx=prepared.env_ids.tolist(),
                    )
            self._commit_portable_state(
                qpos,
                qvel,
                prepared.env_ids,
                set(prepared.entity_names),
                prepared.roots,
            )
            self._refresh_host_cache()
            self._time_cache[prepared.env_ids] = 0.0
        except BaseException:
            self._entity_faulted = True
            raise

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset public state, including state-only visual mirror roots."""

        super().reset(env_ids)
        if not self._portable_mode:
            return
        rows = (
            np.arange(self._num_envs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp)
        )
        if rows.size == 0:
            return
        from unisim.entities import EntityStatePatch, SceneResetRequest

        patches = tuple(
            EntityStatePatch(
                name,
                root_pose=np.asarray(
                    self.get_entity_default_state(name, rows)["root_pose"], dtype=np.float32
                ),
            )
            for name, runtime in self._entity_runtimes.items()
            if runtime.is_visual_mirror
        )
        if patches:
            self.reset_entities(SceneResetRequest(tuple(rows.tolist()), patches))

    def _portable_default_controls(self, rows: np.ndarray) -> np.ndarray:
        """Build variant-assigned public default controls for selected rows."""

        assert self._variant_assignment is not None
        controls = np.zeros((rows.size, self.num_actuators), dtype=np.float32)
        for runtime in self._entity_runtimes.values():
            if not runtime.actuator_indices.size:
                continue
            variants = (
                self._variant_assignment[rows]
                if len(runtime.source_metadata) > 1
                else np.zeros(rows.size, dtype=np.int32)
            )
            controls[:, runtime.actuator_indices] = np.stack(
                [runtime.source_metadata[int(variant)].default_ctrl for variant in variants]
            )
        return controls

    # ------------------------------------------------------------------ #
    # Simulation control                                                  #
    # ------------------------------------------------------------------ #

    def _push_control(self, ctrl: np.ndarray) -> None:
        if self._portable_mode:
            for runtime in self._entity_runtimes.values():
                if not runtime.native_actuated_dofs.size:
                    continue
                runtime.entity.control_dofs_position(
                    self._to_device(ctrl[:, runtime.actuator_indices]),
                    dofs_idx_local=runtime.native_actuated_dofs.tolist(),
                )
            return
        self._entity.control_dofs_position(
            self._to_device(ctrl), dofs_idx_local=self._actuated_dofs
        )

    def _raise_if_viewer_closed(self, exc: BaseException | None = None) -> None:
        """Translate a closed genesis viewer into the contract's RenderClosedError.

        Genesis 1.3.3 raises its private exception from ``visualizer.update``,
        which fires both from our ``render()`` and from ``scene.step()`` itself
        while a viewer is attached; the contract surface is the same either
        way. Any other exception while the viewer is still alive is re-raised.
        The dead viewer is also detached from the visualizer so later physics
        steps are not poisoned by it (the renderer is gone for good).
        """
        if self._viewer is not None and not self._viewer.is_alive():
            visualizer = self._scene.visualizer
            if getattr(visualizer, "_viewer", None) is self._viewer:
                # Only drop the viewer reference; ``viewer_lock`` stays: the
                # rasterizer uses it as a context manager during destroy.
                visualizer._viewer = None
            self._viewer = None
            raise RenderClosedError("genesis viewer window was closed") from exc

    def _physics_substep(self) -> None:
        try:
            self._scene.step()
            if self._portable_pending_body_forces is not None:
                self._portable_pending_body_forces.fill(0.0)
            if self._portable_pending_body_torques is not None:
                self._portable_pending_body_torques.fill(0.0)
        except Exception as exc:
            self._raise_if_viewer_closed(exc)
            raise

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_state("step")
        if isinstance(nsteps, bool) or int(nsteps) <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self.num_actuators)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")

        t0 = time.perf_counter()
        if self._pre_step_control_fn is None:
            # control_dofs_position holds the target across scene.step() calls,
            # matching MuJoCo ctrl-broadcast semantics (REPORT §3.3 [3b]).
            self._push_control(ctrl_array)
            for _ in range(int(nsteps)):
                self._physics_substep()
                self._contact_sensor_rows_valid.fill(True)
        else:
            # Adapter-side per-substep hook: the conversion reads the freshly
            # refreshed sensor contract before every physics substep (REPORT
            # §5.4); no new SimBackend surface is introduced.
            for _ in range(int(nsteps)):
                self._pre_step_control_active = True
                try:
                    converted = self._apply_pre_step_control(ctrl_array)
                finally:
                    self._pre_step_control_active = False
                self._push_control(converted)
                self._physics_substep()
                self._contact_sensor_rows_valid.fill(True)
                self._refresh_host_cache()
        physics_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache += np.float32(int(nsteps) * self._sim_dt)
        host_cache_ms = (time.perf_counter() - t0) * 1000.0
        return {"timing": {"physics_ms": physics_ms, "host_cache_refresh_ms": host_cache_ms}}

    # All backends report the same set_state key set for column stability;
    # sub-keys that don't apply to the genesis host profile report 0.0.
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
    _SET_STATE_TIMING_OWN_KEYS = (
        "set_state_reset_upload_ms",
        "set_state_reset_forward_ms",
        "set_state_host_cache_refresh_ms",
        "set_state_internal_gap_ms",
    )

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        if randomization is not None:
            unsupported = self.get_dr_capabilities().get_unsupported_reset_terms(
                randomization.requested_terms()
            )
            if unsupported:
                raise NotImplementedError(
                    f"Genesis reset randomization does not support terms: {sorted(unsupported)}"
                )
        self._require_state("set_state")
        rows = self._validate_rows(env_indices)
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (rows.size, self._metadata.nq)
        expected_qvel = (rows.size, self._metadata.nv)
        if qpos_array.shape != expected_qpos:
            raise ValueError(f"qpos must have shape {expected_qpos}, got {qpos_array.shape}")
        if qvel_array.shape != expected_qvel:
            raise ValueError(f"qvel must have shape {expected_qvel}, got {qvel_array.shape}")
        timing: dict[str, float] = {
            key: 0.0 for key in self._SET_STATE_TIMING_ZERO_KEYS + self._SET_STATE_TIMING_OWN_KEYS
        }
        if rows.size == 0:
            return {"timing": timing}

        outer_t0 = time.perf_counter()
        envs_idx = rows.tolist()
        t0 = time.perf_counter()
        if self._portable_mode:
            portable_randomization = (
                self._prepare_portable_reset_randomization(randomization, rows)
                if randomization is not None and not randomization.is_empty()
                else None
            )
            full_qpos = self._qpos_cache[1].copy()
            full_qvel = self._qvel_cache[1].copy()
            full_qpos[rows] = qpos_array
            full_qvel[rows] = qvel_array
            try:
                self._cancel_portable_body_wrenches(rows)
                self._commit_portable_state(full_qpos, full_qvel, rows)
                if portable_randomization is not None:
                    self._apply_portable_reset_randomization(portable_randomization, rows)
            except BaseException:
                self._entity_faulted = True
                raise
            timing["set_state_reset_upload_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            self._refresh_host_cache()
            self._time_cache[rows] = 0.0
            timing["set_state_host_cache_refresh_ms"] = (time.perf_counter() - t0) * 1000.0
            measured_ms = (
                timing["set_state_reset_upload_ms"]
                + timing["set_state_reset_forward_ms"]
                + timing["set_state_host_cache_refresh_ms"]
            )
            total_ms = (time.perf_counter() - outer_t0) * 1000.0
            timing["set_state_internal_gap_ms"] = total_ms - measured_ms
            return {"timing": timing}
        # set_qpos runs forward kinematics for the touched envs, so positions
        # are immediately readable afterwards (REPORT §5.6).
        self._entity.set_qpos(self._to_device(qpos_array), envs_idx=envs_idx, zero_velocity=False)
        self._entity.set_dofs_velocity(self._to_device(qvel_array), envs_idx=envs_idx)
        timing["set_state_reset_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if randomization is not None and not randomization.is_empty():
            self._apply_reset_randomization(randomization, rows)
        self._refresh_host_cache()
        self._time_cache[rows] = 0.0
        timing["set_state_host_cache_refresh_ms"] = (time.perf_counter() - t0) * 1000.0

        measured_ms = (
            timing["set_state_reset_upload_ms"]
            + timing["set_state_reset_forward_ms"]
            + timing["set_state_host_cache_refresh_ms"]
        )
        total_ms = (time.perf_counter() - outer_t0) * 1000.0
        timing["set_state_internal_gap_ms"] = total_ms - measured_ms
        return {"timing": timing}

    # ------------------------------------------------------------------ #
    # Domain randomization (REPORT §3.5 [8] measured items only)          #
    # ------------------------------------------------------------------ #

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Declare only the per-env round-trip-measured DR items (REPORT §5.7).

        Portable mode maps the measured mass, COM, DOF-property and PD-gain
        terms through audited independent entities. Non-portable mode also
        declares the solver-level external-force API. Absolute geom_friction
        randomization remains unsupported because Genesis exposes only a
        per-env ratio API.
        """
        if self._portable_mode:
            supported_reset_terms = {
                RESET_TERM_BODY_MASS,
                RESET_TERM_BASE_MASS,
                RESET_TERM_KP,
                RESET_TERM_KD,
                RESET_TERM_BODY_IPOS,
                RESET_TERM_BASE_COM,
                RESET_TERM_DOF_DAMPING,
                RESET_TERM_DOF_FRICTIONLOSS,
                RESET_TERM_DOF_ARMATURE,
            }
            return DomainRandomizationCapabilities(
                supported_reset_terms=frozenset(supported_reset_terms),
                supports_interval_body_force=True,
                supports_interval_body_torque=True,
                supported_interval_terms=frozenset(
                    {INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE}
                ),
            )
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                {RESET_TERM_BODY_MASS, RESET_TERM_BASE_MASS, RESET_TERM_KP, RESET_TERM_KD}
            ),
            supports_interval_body_force=True,
            supported_interval_terms=frozenset({INTERVAL_TERM_BODY_FORCE}),
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return canonical or variant-assigned Genesis reset defaults."""

        _validate_reset_term(term)
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(f"GenesisBackend does not support reset term {term!r}")
        if term == RESET_TERM_BASE_MASS:
            shape = (self._num_envs,) if self._portable_mode else ()
            value = np.zeros(shape, dtype=np.float32)
        elif term == RESET_TERM_BASE_COM and self._portable_mode:
            value = np.zeros((self._num_envs, 3), dtype=np.float32)
        elif term == RESET_TERM_BODY_MASS:
            value = (
                self._portable_default_body_mass(np.arange(self._num_envs, dtype=np.intp))
                if self._portable_mode
                else self._metadata.body_mass
            )
        elif term == RESET_TERM_BODY_IPOS and self._portable_mode:
            value = self._portable_default_body_ipos
        elif self._portable_mode and term in (
            RESET_TERM_DOF_DAMPING,
            RESET_TERM_DOF_FRICTIONLOSS,
            RESET_TERM_DOF_ARMATURE,
        ):
            value = self._portable_default_dof_values(
                self._RESET_TERM_DOF_FIELDS[term],
                np.arange(self._num_envs, dtype=np.intp),
            )
        else:
            canonical = (
                self._metadata.actuator_kp if term == RESET_TERM_KP else self._metadata.actuator_kv
            )
            if not self._portable_mode:
                value = canonical
            else:
                value = np.broadcast_to(
                    canonical,
                    (self._num_envs, self.num_actuators),
                ).copy()
                assert self._variant_assignment is not None
                for row, variant in enumerate(self._variant_assignment):
                    for runtime in self._entity_runtimes.values():
                        if not runtime.actuator_indices.size:
                            continue
                        metadata = runtime.source_metadata[
                            int(variant if len(runtime.source_metadata) > 1 else 0)
                        ]
                        value[row, runtime.actuator_indices] = (
                            metadata.actuator_kp if term == RESET_TERM_KP else metadata.actuator_kv
                        )
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _portable_default_body_mass(self, rows: np.ndarray) -> np.ndarray:
        """Build variant-assigned public body-mass defaults for selected rows."""

        layout = self.get_scene_layout()
        assert self._variant_assignment is not None
        mass = np.broadcast_to(
            self._metadata.body_mass,
            (rows.size, layout.nbody),
        ).copy()
        for owner, runtime in zip(
            layout.entities,
            self._entity_runtimes.values(),
            strict=True,
        ):
            if runtime.is_visual_mirror:
                continue
            variants = (
                self._variant_assignment[rows]
                if len(runtime.source_metadata) > 1
                else np.zeros(rows.size, dtype=np.int32)
            )
            for row_index, variant in enumerate(variants):
                metadata = runtime.source_metadata[int(variant)]
                for body_name, public_body_id in zip(
                    owner.body_names,
                    runtime.body_ids,
                    strict=True,
                ):
                    source_id = metadata.body_names.index(body_name)
                    mass[row_index, int(public_body_id)] = metadata.body_mass[source_id]
        return mass

    _RESET_TERM_DOF_FIELDS = {
        RESET_TERM_DOF_DAMPING: "dof_damping",
        RESET_TERM_DOF_FRICTIONLOSS: "dof_frictionloss",
        RESET_TERM_DOF_ARMATURE: "dof_armature",
    }

    def _portable_default_dof_values(self, field: str, rows: np.ndarray) -> np.ndarray:
        """Build variant-assigned public DOF defaults for selected rows."""

        assert self._variant_assignment is not None
        values = np.full((rows.size, self._metadata.nv), np.nan, dtype=np.float32)
        mapped = np.zeros(self._metadata.nv, dtype=bool)
        row_indices = np.arange(rows.size, dtype=np.intp)
        for runtime in self._entity_runtimes.values():
            if runtime.is_visual_mirror:
                continue
            variants = (
                self._variant_assignment[rows]
                if len(runtime.source_metadata) > 1
                else np.zeros(rows.size, dtype=np.int32)
            )
            values[np.ix_(row_indices, runtime.qvel_indices)] = getattr(runtime, field)[variants]
            mapped[runtime.qvel_indices] = True
        if not mapped.all() or not np.isfinite(values).all():
            raise RuntimeError("portable genesis public DOF property binding is incomplete")
        return values

    def _prepare_portable_reset_randomization(
        self,
        randomization: ResetRandomizationPayload,
        rows: np.ndarray,
    ) -> _GenesisPortableResetRandomization:
        unsupported = [
            term
            for term in self._PORTABLE_UNSUPPORTED_RESET_TERMS
            if getattr(randomization, term) is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "genesis backend does not support reset domain randomization terms: "
                + ", ".join(sorted(unsupported))
            )

        body_mass: np.ndarray | None = None
        if randomization.body_mass is not None:
            body_mass = np.asarray(randomization.body_mass, dtype=np.float32)
            expected = (rows.size, self._metadata.nbody)
            if body_mass.shape != expected:
                raise ValueError(f"body_mass must have shape {expected}, got {body_mass.shape}")
            body_mass = body_mass.copy()
        if randomization.base_mass_delta is not None:
            if body_mass is None:
                body_mass = self._portable_default_body_mass(rows)
            if self._base_link_idx is None:
                raise ValueError(
                    "genesis base_mass_delta randomization requires base_name to identify "
                    "the base link"
                )
            delta = np.asarray(
                randomization.base_mass_delta,
                dtype=np.float32,
            ).reshape(-1)
            if delta.shape != (rows.size,):
                raise ValueError(
                    f"base_mass_delta must have shape ({rows.size},), got {delta.shape}"
                )
            if not np.isfinite(delta).all():
                raise ValueError("base_mass_delta must contain only finite values")
            body_mass[:, self._base_link_idx] += delta
            if not np.isfinite(body_mass).all():
                raise ValueError("body_mass must contain only finite values")
        elif body_mass is not None and not np.isfinite(body_mass).all():
            raise ValueError("body_mass must contain only finite values")
        mapped_bodies = np.concatenate(
            [
                runtime.body_ids
                for runtime in self._entity_runtimes.values()
                if not runtime.is_visual_mirror
            ]
        )
        unmapped_bodies = np.ones(self._metadata.nbody, dtype=bool)
        unmapped_bodies[mapped_bodies] = False
        if unmapped_bodies.any() and body_mass is not None and not np.array_equal(
            body_mass[:, unmapped_bodies],
            self._portable_default_body_mass(rows)[:, unmapped_bodies],
        ):
            raise ValueError(
                "body_mass cannot randomize public columns without native Genesis links"
            )

        body_ipos: np.ndarray | None = None
        if randomization.body_ipos is not None or randomization.base_com_offset is not None:
            if randomization.body_ipos is None:
                ipos_values = self._portable_default_body_ipos[rows].copy()
            else:
                ipos_values = np.asarray(randomization.body_ipos, dtype=np.float32)
                ipos_expected = (rows.size, self._metadata.nbody, 3)
                if ipos_values.shape != ipos_expected:
                    raise ValueError(
                        f"body_ipos must have shape {ipos_expected}, got {ipos_values.shape}"
                    )
                ipos_values = ipos_values.copy()
            if randomization.base_com_offset is not None:
                if self._base_link_idx is None:
                    raise ValueError(
                        "genesis base_com_offset randomization requires base_name to "
                        "identify the base link"
                    )
                delta = np.asarray(randomization.base_com_offset, dtype=np.float32)
                if delta.shape != (rows.size, 3):
                    raise ValueError(
                        f"base_com_offset must have shape ({rows.size}, 3), got {delta.shape}"
                    )
                if not np.isfinite(delta).all():
                    raise ValueError("base_com_offset must contain only finite values")
                ipos_values[:, self._base_link_idx] += delta
            if not np.isfinite(ipos_values).all():
                raise ValueError("body_ipos must contain only finite values")
            body_ipos = ipos_values
        if body_ipos is not None:
            mapped_bodies = np.concatenate(
                [
                    runtime.body_ids
                    for runtime in self._entity_runtimes.values()
                    if not runtime.is_visual_mirror
                ]
            )
            unmapped_bodies = np.ones(self._metadata.nbody, dtype=bool)
            unmapped_bodies[mapped_bodies] = False
            if unmapped_bodies.any() and not np.array_equal(
                body_ipos[:, unmapped_bodies, :],
                np.asarray(
                    self._portable_default_body_ipos[rows][:, unmapped_bodies, :],
                    dtype=np.float32,
                ),
            ):
                raise ValueError(
                    "body_ipos cannot randomize public columns without native Genesis links"
                )

        prepared_gains: dict[str, np.ndarray | None] = {}
        prepared_dof_values: dict[str, np.ndarray | None] = {}
        mapped_dofs = np.zeros(self._metadata.nv, dtype=bool)
        for runtime in self._entity_runtimes.values():
            mapped_dofs[runtime.qvel_indices] = True
        if not mapped_dofs.all():
            raise RuntimeError("portable genesis public DOF property binding is incomplete")
        for term, field in self._RESET_TERM_DOF_FIELDS.items():
            value = getattr(randomization, term)
            if value is None:
                prepared_dof_values[field] = None
                continue
            values = np.asarray(value, dtype=np.float32)
            expected = (rows.size, self._metadata.nv)
            if values.shape != expected:
                raise ValueError(f"{term} must have shape {expected}, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{term} must contain only finite values")
            if np.any(values < 0.0):
                raise ValueError(f"{term} must contain only non-negative values")
            prepared_dof_values[field] = values.copy()
        for value, name in (
            (randomization.kp, "kp"),
            (randomization.kd, "kd"),
        ):
            if value is None:
                prepared_gains[name] = None
                continue
            gains = np.asarray(value, dtype=np.float32)
            expected = (rows.size, self.num_actuators)
            if gains.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {gains.shape}")
            if not np.isfinite(gains).all():
                raise ValueError(f"{name} must contain only finite values")
            prepared_gains[name] = gains.copy()
        return _GenesisPortableResetRandomization(
            body_mass=body_mass,
            body_ipos=body_ipos,
            dof_damping=prepared_dof_values["dof_damping"],
            dof_frictionloss=prepared_dof_values["dof_frictionloss"],
            dof_armature=prepared_dof_values["dof_armature"],
            kp=prepared_gains["kp"],
            kd=prepared_gains["kd"],
        )

    def _apply_portable_reset_randomization(
        self,
        values: _GenesisPortableResetRandomization,
        rows: np.ndarray,
    ) -> None:
        envs_idx = rows.tolist()
        for runtime in self._entity_runtimes.values():
            if runtime.is_visual_mirror:
                continue
            if values.body_mass is not None:
                local_mass = np.ascontiguousarray(
                    values.body_mass[:, runtime.body_ids],
                    dtype=np.float32,
                )
                runtime.entity.set_links_inertial_mass(
                    self._to_device(local_mass),
                    links_idx_local=runtime.native_body_indices.tolist(),
                    envs_idx=envs_idx,
                )
            if values.body_ipos is not None:
                default_ipos = self._portable_default_body_ipos[rows][:, runtime.body_ids, :]
                local_shift = np.ascontiguousarray(
                    values.body_ipos[:, runtime.body_ids, :] - default_ipos,
                    dtype=np.float32,
                )
                runtime.entity.set_COM_shift(
                    self._to_device(local_shift),
                    links_idx_local=runtime.native_body_indices.tolist(),
                    envs_idx=envs_idx,
                )
        for field, setter_name in (
            ("dof_damping", "set_dofs_damping"),
            ("dof_frictionloss", "set_dofs_frictionloss"),
            ("dof_armature", "set_dofs_armature"),
        ):
            dof_values = getattr(values, field)
            if dof_values is None:
                continue
            for runtime in self._entity_runtimes.values():
                if not runtime.native_qvel_indices.size:
                    continue
                local_values = np.ascontiguousarray(
                    dof_values[:, runtime.qvel_indices],
                    dtype=np.float32,
                )
                getattr(runtime.entity, setter_name)(
                    self._to_device(local_values),
                    dofs_idx_local=runtime.native_qvel_indices.tolist(),
                    envs_idx=envs_idx,
                )
        for gains, setter_name in (
            (values.kp, "set_dofs_kp"),
            (values.kd, "set_dofs_kv"),
        ):
            if gains is None:
                continue
            for runtime in self._entity_runtimes.values():
                if not runtime.native_actuated_dofs.size:
                    continue
                local_gains = np.ascontiguousarray(
                    gains[:, runtime.actuator_indices],
                    dtype=np.float32,
                )
                getattr(runtime.entity, setter_name)(
                    self._to_device(local_gains),
                    dofs_idx_local=runtime.native_actuated_dofs.tolist(),
                    envs_idx=envs_idx,
                )
        if values.body_mass is not None:
            self._body_mass_cache[rows] = values.body_mass
        if values.body_ipos is not None:
            self._body_ipos_cache[rows] = values.body_ipos

    _PORTABLE_UNSUPPORTED_RESET_TERMS = (
        "gravity",
        "body_iquat",
        "body_inertia",
        "geom_friction",
    )

    _UNSUPPORTED_RESET_TERMS = _PORTABLE_UNSUPPORTED_RESET_TERMS + (
        "body_ipos",
        "base_com_offset",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
    )

    def _apply_reset_randomization(
        self, randomization: ResetRandomizationPayload, rows: np.ndarray
    ) -> None:
        if self._portable_mode:
            raise NotImplementedError(
                "portable genesis entity reset randomization is not supported"
            )
        unsupported = [
            term
            for term in self._UNSUPPORTED_RESET_TERMS
            if getattr(randomization, term) is not None
        ]
        if unsupported:
            raise NotImplementedError(
                "genesis backend does not support reset domain randomization terms: "
                f"{', '.join(sorted(unsupported))} (REPORT #1372 §5.7 declares only the "
                "measured items)."
            )
        envs_idx = rows.tolist()
        body_mass = randomization.body_mass
        if randomization.base_mass_delta is not None:
            if self._base_link_idx is None:
                raise ValueError(
                    "genesis base_mass_delta randomization requires base_name to identify "
                    "the base link"
                )
            delta = np.asarray(randomization.base_mass_delta, dtype=np.float32).reshape(-1)
            if delta.shape != (rows.size,):
                raise ValueError(
                    f"base_mass_delta must have shape ({rows.size},), got {delta.shape}"
                )
            if body_mass is None:
                body_mass = np.broadcast_to(
                    self._metadata.body_mass, (rows.size, self._metadata.nbody)
                ).copy()
            else:
                body_mass = np.asarray(body_mass, dtype=np.float32).copy()
            body_mass[:, self._base_link_idx] += delta
        if body_mass is not None:
            mass = np.asarray(body_mass, dtype=np.float32)
            expected = (rows.size, self._metadata.nbody)
            if mass.shape != expected:
                raise ValueError(f"body_mass must have shape {expected}, got {mass.shape}")
            self._entity.set_links_inertial_mass(self._to_device(mass), envs_idx=envs_idx)
        for value, name, setter in (
            (randomization.kp, "kp", self._entity.set_dofs_kp),
            (randomization.kd, "kd", self._entity.set_dofs_kv),
        ):
            if value is None:
                continue
            gains = np.asarray(value, dtype=np.float32)
            expected = (rows.size, self.num_actuators)
            if gains.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {gains.shape}")
            setter(
                self._to_device(gains),
                dofs_idx_local=self._actuated_dofs,
                envs_idx=envs_idx,
            )

    _interval_term_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        self._require_state("apply_interval_randomization")
        if self._portable_mode and not plan.is_empty():
            self._cancel_portable_body_wrenches(np.arange(self._num_envs, dtype=np.intp))
        super().apply_interval_randomization(plan)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        # Built lazily once; only body wrenches have handlers.  Push and
        # velocity terms fail closed in the base dispatch.
        if self._interval_term_handler_cache is None:
            self._interval_term_handler_cache = {
                INTERVAL_TERM_BODY_FORCE: lambda op: self.apply_body_force(
                    require_op_body_ids(op), op.payload
                ),
            }
            if self._portable_mode:

                def apply_body_torque(op: IntervalTermOp) -> None:
                    ids = require_op_body_ids(op)
                    self.apply_body_force(
                        ids,
                        np.zeros((self._num_envs, len(ids), 3), dtype=np.float32),
                        torque=op.payload,
                    )

                self._interval_term_handler_cache[INTERVAL_TERM_BODY_TORQUE] = apply_body_torque
        return self._interval_term_handler_cache

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Apply a world-frame wrench at each body COM for the upcoming step."""
        self._reject_wrench_write_inside_pre_step_control("apply_body_force")
        if torque is not None and not self._portable_mode:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support interval body torque perturbation"
            )
        self._require_state("apply_body_force")
        ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
        force_array = np.asarray(force, dtype=np.float32)
        torque_array = None if torque is None else np.asarray(torque, dtype=np.float32)
        expected = (self._num_envs, ids.size, 3)
        if force_array.shape != expected:
            raise ValueError(f"body force must have shape {expected}, got {force_array.shape}")
        if not np.isfinite(force_array).all():
            raise ValueError("body force contains NaN or Inf")
        if torque_array is not None:
            if torque_array.shape != expected:
                raise ValueError(
                    f"body torque must have shape {expected}, got {torque_array.shape}"
                )
            if not np.isfinite(torque_array).all():
                raise ValueError("body torque contains NaN or Inf")
        if self._portable_mode:
            assert self._portable_body_link_ids is not None
            assert self._portable_pending_body_forces is not None
            assert self._portable_pending_body_torques is not None
            assert self._portable_force_body_ids is not None
            if np.any(ids < 0) or np.any(ids >= self._metadata.nbody):
                raise ValueError(f"body_ids must be in [0, {self._metadata.nbody}), got {ids}")
            if not self._portable_force_body_ids[ids].all():
                raise ValueError("genesis body ids must reference owned physical bodies")
            native_links = self._portable_body_link_ids[ids]
            force_device = self._to_device(force_array)
            solver = self._scene.sim.rigid_solver
            try:
                solver.apply_links_external_force(
                    force_device,
                    links_idx=native_links.tolist(),
                    ref="link_com",
                    local=False,
                )
                if torque_array is not None:
                    solver.apply_links_external_torque(
                        self._to_device(torque_array),
                        links_idx=native_links.tolist(),
                        ref="link_com",
                        local=False,
                    )
                for offset, body_id in enumerate(ids):
                    self._portable_pending_body_forces[:, int(body_id), :] += force_array[
                        :, offset, :
                    ]
                    if torque_array is not None:
                        self._portable_pending_body_torques[:, int(body_id), :] += (
                            torque_array[:, offset, :]
                        )
            except BaseException:
                self._entity_faulted = True
                raise
            return
        if np.any(ids < 0) or np.any(ids >= self._metadata.nbody):
            raise ValueError(f"body_ids must be in [0, {self._metadata.nbody}), got {ids}")
        solver = self._scene.sim.rigid_solver
        force_device = self._to_device(force_array)
        for offset, body_id in enumerate(ids):
            # Solver-level API uses global link indices (REPORT §3.5 [8]);
            # single-entity scenes keep global == link_start + local.
            solver.apply_links_external_force(
                force_device[:, offset, :],
                links_idx=[self._link_start + int(body_id)],
            )

    def _cancel_portable_body_wrenches(
        self,
        rows: np.ndarray,
        *,
        body_ids: Sequence[int] | None = None,
    ) -> None:
        """Cancel unconsumed portable wrenches in the selected entity/body scope."""

        pending_forces = self._portable_pending_body_forces
        pending_torques = self._portable_pending_body_torques
        if pending_forces is None or pending_torques is None or rows.size == 0:
            return
        candidate_bodies = (
            np.arange(pending_forces.shape[1], dtype=np.intp)
            if body_ids is None
            else np.asarray(body_ids, dtype=np.intp).reshape(-1)
        )
        active_forces = np.any(pending_forces[np.ix_(rows, candidate_bodies)], axis=(0, 2))
        active_torques = np.any(pending_torques[np.ix_(rows, candidate_bodies)], axis=(0, 2))
        active = active_forces | active_torques
        if active.size == 0:
            return
        selected_bodies = candidate_bodies[active]
        assert self._portable_body_link_ids is not None
        native_links = self._portable_body_link_ids[selected_bodies]
        force_cancellation = np.ascontiguousarray(-pending_forces[np.ix_(rows, selected_bodies)])
        torque_cancellation = np.ascontiguousarray(-pending_torques[np.ix_(rows, selected_bodies)])
        solver = self._scene.sim.rigid_solver
        try:
            solver.apply_links_external_force(
                self._to_device(force_cancellation),
                links_idx=native_links.tolist(),
                envs_idx=rows.tolist(),
                ref="link_com",
                local=False,
            )
            solver.apply_links_external_torque(
                self._to_device(torque_cancellation),
                links_idx=native_links.tolist(),
                envs_idx=rows.tolist(),
                ref="link_com",
                local=False,
            )
            pending_forces[np.ix_(rows, selected_bodies)] = 0.0
            pending_torques[np.ix_(rows, selected_bodies)] = 0.0
        except BaseException:
            self._entity_faulted = True
            raise

    # ------------------------------------------------------------------ #
    # Native rendering / playback (post-build lazy viewer and camera)      #
    # ------------------------------------------------------------------ #

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "auto":
            # The interactive viewer needs a reachable display; headless hosts
            # fall back to offscreen camera recording (isaacgym semantics).
            mode = "interactive" if playback.display_available() else "record"
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
            raise ValueError("genesis record playback requires a finite training.play_steps value.")
        if output_video is None:
            raise ValueError("genesis record playback requires an output video path.")
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
        """Lazily attach the Genesis viewer and/or an offscreen camera.

        Both are post-build attachments (verified on 1.3.3): the interactive
        viewer is a ``genesis.vis.viewer.Viewer`` built on the visualizer's
        shared context; capture uses a visualizer camera built on demand.
        ``spacing``/``offset_mode`` are accepted for contract parity and
        ignored: envs are laid out on the Genesis scene's own grid.  The first
        (headless, capture) pair is pinned, like the isaacgym backend.
        """
        del spacing, offset_mode
        config = (bool(headless), bool(capture))
        if self._render_config is not None:
            if self._render_config != config:
                raise RuntimeError(
                    "genesis renderer is already initialized with "
                    f"headless={self._render_config[0]}, capture={self._render_config[1]}; "
                    f"cannot reinitialize it with headless={config[0]}, capture={config[1]}"
                )
            return
        self._require_state("init_renderer")
        self._render_config = config
        self._camera_cfg = CameraCfg.from_kwargs(camera_kwargs)
        self._camera_tracking_env_idx = (
            self._camera_cfg.cam_tracking_env_idx if self._camera_cfg.cam_tracking else None
        )
        visualizer = self._scene.visualizer
        if not headless:
            if not playback.display_available():
                raise RuntimeError(
                    "genesis interactive viewer requires a reachable display "
                    "(DISPLAY or WAYLAND_DISPLAY); select play_render_mode=record on "
                    "headless hosts."
                )
            viewer_module = importlib.import_module("genesis.vis.viewer")
            options = self._gs.options.ViewerOptions(
                res=(int(width), int(height)), run_in_thread=False
            )
            try:
                viewer = viewer_module.Viewer(options, visualizer.context)
                viewer.build(self._scene)
            except Exception as exc:
                raise RuntimeError(
                    f"genesis failed to create the interactive viewer: {type(exc).__name__}: {exc}"
                ) from exc
            # The visualizer owns no public post-build viewer setter on 1.3.3;
            # attach through its documented internals (cold render path only).
            visualizer._viewer = viewer
            visualizer.viewer_lock = viewer.lock
            pos, lookat = playback.camera_pose_from_kwargs(self._camera_cfg, self._camera_lookat())
            # #1396: the viewer's pos/lookat branch reuses its polluted
            # default _camera_up; pass the full Z-up pose matrix instead.
            viewer.set_camera_pose(pose=playback.camera_pose_matrix_z_up(pos, lookat))
            if self._camera_tracking_env_idx is not None:
                viewer.follow_entity(self._entity)
            self._viewer = viewer
        if capture:
            pos, lookat = playback.camera_pose_from_kwargs(self._camera_cfg, self._camera_lookat())
            camera = visualizer.add_camera(
                res=(int(width), int(height)),
                pos=tuple(pos),
                lookat=tuple(lookat),
                up=(0.0, 0.0, 1.0),
                model="pinhole",
                fov=self._camera_cfg.cam_fov if self._camera_cfg.cam_fov is not None else 30.0,
                aperture=2.0,
                focus_dist=None,
                spp=256,
                denoise=None,
                near=0.1,
                far=20.0,
                env_idx=None,
                debug=False,
                GUI=False,
            )
            camera.build()
            self._render_camera = camera

    def _camera_lookat(self) -> np.ndarray:
        """Static camera lookat: the env-0 root position from the host cache."""
        if self._base_link_idx is not None:
            return np.asarray(self._links_pos_cache[1][0, self._base_link_idx], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def render(self) -> None:
        """Draw one interactive viewer frame (self-initializes interactive)."""
        if self._viewer is None:
            self.init_renderer(headless=False, camera_kwargs=self._camera_cfg)
        assert self._viewer is not None
        try:
            self._scene.visualizer.update(force=False)
        except Exception as exc:
            # Genesis 1.3.3 raises its private error when the window is gone;
            # translate it at the interface boundary per the contract.
            self._raise_if_viewer_closed(exc)
            raise
        self._raise_if_viewer_closed()

    def capture_video_frame(self) -> np.ndarray:
        """Capture one offscreen RGB frame (self-initializes headless+capture)."""
        if self._render_camera is None:
            self.init_renderer(headless=True, capture=True, camera_kwargs=self._camera_cfg)
        assert self._render_camera is not None
        if self._camera_tracking_env_idx is not None and self._base_link_idx is not None:
            lookat = np.asarray(
                self._links_pos_cache[1][self._camera_tracking_env_idx, self._base_link_idx],
                dtype=np.float64,
            )
            pos, lookat = playback.camera_pose_from_kwargs(self._camera_cfg, lookat)
            self._render_camera.set_pose(pos=tuple(pos), lookat=tuple(lookat))
        frame = self._render_camera.render()[0]
        if frame.ndim == 4:
            # Batched renderer: take env 0's frame (unbatched path returns
            # (H, W, 3) directly, verified on 1.3.3).
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"genesis camera returned an unexpected frame shape {frame.shape}; "
                "expected (H, W, 3) RGB"
            )
        return np.asarray(frame, dtype=np.uint8)

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
        debug_overlay_getter: Any = None,
        on_frame: Any = None,
    ) -> str | None:
        # Native live-scene playback: no state snapshots are needed.
        del render_spacing, render_offset_mode, frame_state_getter
        if debug_overlay_getter is not None:
            raise unsupported_debug_overlay_error(self.__class__.__name__)
        if on_frame is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} renders through a native renderer and "
                "does not support on_frame callbacks"
            )
        camera_cfg = CameraCfg.from_kwargs(camera_kwargs)
        should_record_video = (
            bool(record_video) if record_video is not None else output_video is not None
        )
        should_run_headless = bool(headless) if headless is not None else should_record_video
        try:
            return playback.run_genesis_playback(
                backend=self,
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                headless=should_run_headless,
                record_video=should_record_video,
                camera_kwargs=camera_cfg,
            )
        except RenderClosedError:
            if not should_run_headless and not should_record_video:
                logger.info("Render window closed.")
                return None
            raise

    # ------------------------------------------------------------------ #
    # Legacy getters: cache views only, never direct device transfers     #
    # ------------------------------------------------------------------ #

    def _require_free_root(self, operation: str) -> None:
        self._require_state(operation)
        if self._root_qpos_dim != 7 or self._root_qvel_dim != 6:
            raise NotImplementedError(
                f"{operation} requires a free root joint; genesis host profile is "
                "currently validated only for floating-base layouts."
            )

    def get_base_pos(self) -> np.ndarray:
        self._require_free_root("get_base_pos")
        return self._links_pos_cache[1][:, self._base_link_idx, :]

    def get_base_quat(self) -> np.ndarray:
        self._require_free_root("get_base_quat")
        return self._links_quat_cache[1][:, self._base_link_idx, :]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_free_root("get_base_lin_vel")
        if self._portable_mode:
            layout = self.get_root_state_layout(self._base_name or "")
            return self._qvel_cache[1][:, list(layout.qvel_indices[:3])]
        # qvel[0:3] is the root linear velocity in world coordinates and stays
        # valid immediately after set_state (REPORT §5.6).
        return self._qvel_cache[1][:, 0:3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_free_root("get_base_ang_vel")
        if self._portable_mode:
            layout = self.get_root_state_layout(self._base_name or "")
            return np_quat_apply_batched(
                self.get_base_quat(),
                self._qvel_cache[1][:, list(layout.qvel_indices[3:6])],
            )
        # qvel[3:6] is body-frame angular velocity; the contract wants world
        # frame.  Deriving it from qvel keeps the value fresh after reset
        # (genesis link velocity getters only refresh across a step barrier,
        # REPORT §5.6); it equals get_links_ang(root) after a step.
        return np_quat_apply_batched(self.get_base_quat(), self._qvel_cache[1][:, 3:6])

    def get_dof_pos(self) -> np.ndarray:
        self._require_state("get_dof_pos")
        if self._portable_mode:
            raise NotImplementedError(
                "portable genesis scenes expose entity-local joint state, not a single "
                "legacy primary-entity dof view"
            )
        return self._qpos_cache[1][:, self._root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
        if self._portable_mode:
            raise NotImplementedError(
                "portable genesis scenes expose entity-local joint state, not a single "
                "legacy primary-entity dof view"
            )
        return self._qvel_cache[1][:, self._root_qvel_dim :]

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_pos_w")
        return self._links_pos_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_quat_w")
        return self._links_quat_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_lin_vel_w")
        return self._links_vel_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_ang_vel_w")
        return self._links_ang_cache[1][:, np.asarray(body_ids, dtype=np.intp), :]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Position relative to the baselink frame: R_base^-1 (pos_w - base_pos_w).
        self._require_free_root("get_body_pos_b")
        base_quat = self._links_quat_cache[1][:, self._base_link_idx, :]
        relative = (
            self.get_body_pos_w(body_ids)
            - self._links_pos_cache[1][:, self._base_link_idx, :][:, None, :]
        )
        return np_quat_apply_inverse_batched(base_quat[:, None, :], relative)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Orientation relative to the baselink frame: quat_base^-1 * quat_w.
        self._require_free_root("get_body_quat_b")
        base_quat = self._links_quat_cache[1][:, self._base_link_idx, :]
        return np_quat_mul_batched(
            np_quat_conjugate_batched(base_quat[:, None, :]), self.get_body_quat_w(body_ids)
        )

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        # Analytical per the SimBackend contract (#1254): world-frame velocity
        # rotated into each body's own frame.
        self._require_state("get_body_lin_vel_b")
        ids = np.asarray(body_ids, dtype=np.intp)
        return np_quat_apply_inverse_batched(
            self._links_quat_cache[1][:, ids, :], self._links_vel_cache[1][:, ids, :]
        )

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_ang_vel_b")
        ids = np.asarray(body_ids, dtype=np.intp)
        return np_quat_apply_inverse_batched(
            self._links_quat_cache[1][:, ids, :], self._links_ang_cache[1][:, ids, :]
        )

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._require_state(f"get_sensor_data({name!r})")
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
            values = [
                self._sensor_cache[:, address : address + dimension] for address, dimension in slots
            ]
            return np.concatenate(values, axis=1)

        return read

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _cleanup_materialization_resources(self) -> None:
        """Release all cold-path materialization resources owned by this backend."""

        composed = self._composed_scene
        self._composed_scene = None
        if composed is not None:
            try:
                composed.close()
            except Exception:
                logger.warning(
                    "failed to clean up composed Genesis scene resources",
                    exc_info=True,
                )

        portable_sources = self._portable_sources
        self._portable_sources = None
        if portable_sources is not None:
            try:
                portable_sources.cleanup_handle.cleanup()
            except Exception:
                logger.warning(
                    "failed to clean up portable Genesis entity sources",
                    exc_info=True,
                )

        scene_cleanup = self._scene_cleanup_handle
        self._scene_cleanup_handle = None
        if scene_cleanup is not None:
            try:
                scene_cleanup.cleanup()
            except Exception:
                logger.warning(
                    "failed to clean up legacy Genesis scene resources",
                    exc_info=True,
                )

    def close(self) -> None:
        """End the process-wide Genesis session; re-init afterwards fails closed."""
        if self._closed:
            return
        self._closed = True
        if self._viewer is not None:
            try:
                self._viewer.stop()
            except Exception:  # viewer teardown must not mask session cleanup
                logger.debug("genesis viewer stop failed during close", exc_info=True)
            self._viewer = None
        try:
            materialization.destroy_genesis_session(self._deps)
        finally:
            self._cleanup_materialization_resources()
