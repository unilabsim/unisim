"""Newton 1.5.1 adapter implementing the host-NumPy SimBackend profile.

All Newton/Warp transfers happen at explicit materialize, step, or set_state
barriers. Public getters return stable NumPy cache views and never inspect
MuJoCo's CPU data object. GPU execution is not bitwise deterministic; callers
and tests must use numerical tolerances.
"""

from __future__ import annotations

import gc
import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    CameraCfg,
    DebugOverlayGetter,
    RenderClosedError,
    SimBackend,
    normalize_play_render_mode,
)
from unisim.backend.playback_common import (
    run_offline_snapshot_playback,
    validate_offline_visual_model,
)
from unisim.dr.types import DomainRandomizationCapabilities, ResetRandomizationPayload
from unisim.entities import SceneResetRequest
from unisim.entity_state import (
    entity_state_snapshot,
    prepare_scene_reset,
    selected_state_rows,
)
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.scene_layout import CompiledSceneLayout
from unisim.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

from .capacity import calibrate_capacity, validate_capacity_limits
from .dependencies import (
    load_newton_dependencies,
    newton_render_dependencies_available,
    require_newton_render_dependencies,
)
from .materialization import (
    NewtonSensorPlan,
    audit_newton_model,
    audit_newton_variant_model,
    build_newton_assigned_world_builder,
    build_newton_source_builder,
    compute_contact_found_flags,
    scan_newton_model_metadata,
    validate_newton_portable_metadata,
    validate_newton_variant_sources,
)
from .playback import (
    MAX_RENDER_WORLDS,
    MUJOCO_SNAPSHOT_RENDERER,
    NEWTON_NATIVE_RENDERER,
    display_available,
    run_newton_native_playback,
)
from .runtime import get_bound_newton_process_device

_WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_NEWTON_DEFAULT_GROUND_COLOR = (0.125, 0.125, 0.15)
_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)


@dataclass(frozen=True)
class _NewtonEntityRuntime:
    """Cold-path binding from one public entity to a Newton articulation view."""

    view: Any
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    view_body_indices: np.ndarray


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

logger = logging.getLogger(__name__)


def _prepare_newton_render_floor(builder: Any, newton: Any) -> bool:
    """Make authored world planes visible and report whether one exists.

    Newton's MJCF importer treats unclassified geoms as collision shapes.  If
    the imported model also contains visual meshes, those planes intentionally
    omit ``ShapeFlags.VISIBLE`` and therefore disappear in ``ViewerGL``.  The
    floor is a scene-level visual concern, so expose existing static planes
    without changing their collision behavior.  A separate, non-colliding
    floor is added by :meth:`NewtonBackend.materialize` when the scene has no
    static plane at all.
    """
    shape_types = getattr(builder, "shape_type", None)
    shape_bodies = getattr(builder, "shape_body", None)
    shape_flags = getattr(builder, "shape_flags", None)
    shape_colors = getattr(builder, "shape_color", None)
    if shape_types is None or shape_bodies is None or shape_flags is None:
        return False
    plane_type = getattr(getattr(newton, "GeoType", None), "PLANE", None)
    if plane_type is None:
        return False
    visible_flag = int(getattr(getattr(newton, "ShapeFlags", None), "VISIBLE", 1))
    plane_value = int(plane_type)
    has_plane = False
    for index, (shape_type, body) in enumerate(zip(shape_types, shape_bodies, strict=True)):
        if int(shape_type) != plane_value or int(body) != -1:
            continue
        has_plane = True
        if index < len(shape_flags) and not (int(shape_flags[index]) & visible_flag):
            shape_flags[index] = int(shape_flags[index]) | visible_flag
            # Collision-only planes have no visual material in Newton's
            # importer.  Use the same dark color as ``add_ground_plane`` so
            # an authored MJCF floor gets Newton's default appearance.
            if shape_colors is not None and index < len(shape_colors):
                shape_colors[index] = _NEWTON_DEFAULT_GROUND_COLOR
    return has_plane


def _add_newton_render_floor(builder: Any) -> None:
    """Add Newton's default-colored floor as a visual-only global shape."""
    cfg = builder.default_shape_cfg.copy()
    cfg.is_visible = True
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    builder.add_ground_plane(cfg=cfg)


class NewtonBackend(SimBackend):
    """In-process Newton/SolverMuJoCo adapter with explicit device placement."""

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        device: str | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        capacity_check_steps: int = 1,
        use_cuda_graph: bool = False,
        **unexpected_kwargs: Any,
    ) -> None:
        require_scene_composition_support(scene, "newton")
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"NewtonBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        if not isinstance(use_cuda_graph, bool):
            raise TypeError(
                "NewtonBackend use_cuda_graph must be bool, got "
                f"{type(use_cuda_graph).__name__}"
            )
        self._nconmax = self._capacity(nconmax, "nconmax", 512)
        self._njmax = self._capacity(njmax, "njmax", 512)
        self._capacity_check_steps = self._capacity(
            capacity_check_steps, "capacity_check_steps", 1
        )
        self._use_cuda_graph = use_cuda_graph
        self._cuda_graphs: tuple[Any, Any] | None = None
        self._cuda_graph_input_states: tuple[Any, Any] | None = None
        self._cuda_graph_enabled = False
        self._cuda_graph_disable_reason: str | None = None

        self._deps = load_newton_dependencies()
        selected_device = device or get_bound_newton_process_device()
        if selected_device is None:
            raise ValueError(
                "newton requires explicit device placement; pass newton_device='cuda:N' "
                "or call configure_backend_process_device before construction"
            )
        self._deps.warp.set_device(str(selected_device))
        resolved = self._deps.warp.get_device()
        if not bool(resolved.is_cuda):
            raise RuntimeError(f"newton backend requires a CUDA Warp device, got {resolved!s}")

        self.backend_type = "newton"
        self._pre_step_control_fn = None
        self._num_envs = num_envs
        self._sim_dt = float(sim_dt)
        self._device = str(resolved)
        self._composed_scene: Any = None
        self._entity_layout: CompiledSceneLayout | None = None
        self._variant_metadata: tuple[Any, ...] | None = None
        self._variant_assignment: np.ndarray | None = None
        self._source_builders: tuple[Any, ...] | None = None
        self._entity_runtimes: dict[str, _NewtonEntityRuntime] = {}
        self._entity_faulted = False
        self._portable_mode = bool(scene.entity_assets)
        if self._portable_mode:
            from unisim.mjcf_compiler import compose_scene

            composed = compose_scene(scene, num_envs, sim_dt)
            try:
                variant_plan = composed.variant_plan
                variant_files = (
                    tuple(item.model_file for item in variant_plan.variants)
                    if variant_plan is not None
                    else (composed.model_file,)
                )
                metadata = tuple(
                    scan_newton_model_metadata(
                        self._deps.mujoco, SceneCfg(model_file=str(model_file))
                    )
                    for model_file in variant_files
                )
                validate_newton_portable_metadata(metadata, composed.layout)
            except BaseException:
                composed.close()
                raise
            self._composed_scene = composed
            self._entity_layout = composed.layout
            self._variant_metadata = metadata
            self._variant_assignment = (
                np.asarray(variant_plan.assignment, dtype=np.int32).copy()
                if variant_plan is not None
                else np.zeros((num_envs,), dtype=np.int32)
            )
            self._metadata = metadata[0]
            self._scene_visual_model_file = variant_files[0]
            self._scene_cleanup_handle = None
            primary = next(
                (
                    index
                    for index, entity in enumerate(composed.layout.entities)
                    if entity.root_mode == "floating"
                ),
                0,
            )
            primary_entity = composed.layout.entities[primary]
            self._primary_entity_name = primary_entity.name
            authored_root_name = f"{primary_entity.name}/{primary_entity.root_body}"
        else:
            self._scene_visual_model_file = str(scene.visual_model_file or scene.model_file)
            self._metadata = scan_newton_model_metadata(self._deps.mujoco, scene)
            if not self._metadata.root_qpos_dim:
                raise NotImplementedError(
                    "newton backend currently requires a free root joint for the "
                    "SimBackend state contract"
                )
            self._scene_cleanup_handle = self._metadata.cleanup_handle
            self._variant_metadata = (self._metadata,)
            self._variant_assignment = np.zeros((num_envs,), dtype=np.int32)
            authored_root_name = str(next(
                (name for name in self._metadata.body_names[1:] if name), None
            ))
            self._primary_entity_name = str(authored_root_name)
        if base_name is not None and base_name != authored_root_name:
            raise ValueError(
                f"newton base_name must identify the authored free root body {authored_root_name!r}"
            )
        self._base_name = base_name or authored_root_name
        self._body_names = self._metadata.body_names[1:]
        self._body_ids = {name: index for index, name in enumerate(self._body_names) if name}
        if self._base_name not in self._body_ids:
            raise ValueError(f"Base body {self._base_name!r} not found in newton model")
        self._base_body_id = self._body_ids[self._base_name]
        self._joint_ids = dict(
            zip(self._metadata.joint_names, range(len(self._metadata.joint_names)))
        )
        self._keyframes = dict(self._metadata.keyframes)
        self._sensor_slots: dict[str, tuple[int, int]] = {}
        address = 0
        for sensor_plan in self._metadata.sensor_plans:
            self._sensor_slots[sensor_plan.name] = (address, sensor_plan.dim)
            address += sensor_plan.dim

        # Engine objects are created by materialize() on the first state
        # access (``_require_state``); they stay None until then, so they are
        # annotated Any rather than carrying an uncheckable Optional engine
        # type.
        self._model: Any = None
        self._solver: Any = None
        self._state: Any = None
        self._state_out: Any = None
        self._control: Any = None
        self._contacts: Any = None
        self._view: Any = None
        self._shape_world: np.ndarray | None = None
        self._contact_sensor_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._playback_model_validated = False
        self._portable_playback_models_validated: set[int] = set()
        self._viewer: Any | None = None
        self._render_config: tuple[bool, bool] | None = None
        self._closed = False

        nbody = len(self._body_names)
        self._qpos_cache = np.zeros((num_envs, self._metadata.nq), dtype=np.float32)
        self._qvel_cache = np.zeros((num_envs, self._metadata.nv), dtype=np.float32)
        self._body_pos_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
        self._body_quat_cache = np.zeros((num_envs, nbody, 4), dtype=np.float32)
        self._body_lin_vel_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
        self._body_ang_vel_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
        assert self._variant_metadata is not None
        assert self._variant_assignment is not None
        selected_metadata = tuple(
            self._variant_metadata[int(variant)] for variant in self._variant_assignment
        )
        self._body_ipos_cache = np.stack(
            [item.body_ipos[1:] for item in selected_metadata]
        ).astype(np.float32, copy=False)
        self._body_mass_cache = np.stack(
            [item.body_mass[1:] for item in selected_metadata]
        ).astype(np.float32, copy=False)
        self._control_cache = np.zeros((num_envs, self._metadata.nu), dtype=np.float32)
        self._control_q_cache = np.zeros((num_envs, self._metadata.nq), dtype=np.float32)
        self._control_qd_cache = np.zeros((num_envs, self._metadata.nv), dtype=np.float32)
        self._sensor_cache = np.zeros((num_envs, address), dtype=np.float32)
        self._previous_site_velocity: dict[str, np.ndarray] = {}
        self._time_cache = np.zeros((num_envs,), dtype=np.float32)

    @staticmethod
    def _capacity(value: int | None, name: str, default: int) -> int:
        resolved = default if value is None else value
        validate_capacity_limits(
            nconmax=resolved if name == "nconmax" else 1,
            njmax=resolved if name != "nconmax" else 1,
        )
        return int(resolved)

    def cleanup_scene_assets(self) -> None:
        composed = self._composed_scene
        # A materialized portable model no longer reads the generated MJCF, but
        # ``get_playback_model`` returns that path throughout the backend
        # lifetime. Failed construction has no native model and must release
        # immediately; ``close()`` marks the backend closed before cleanup.
        model = getattr(self, "_model", None)
        closed = getattr(self, "_closed", False)
        if composed is not None and (model is None or closed):
            composed.close()
            self._composed_scene = None
        super().cleanup_scene_assets()

    def _require_state(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"newton backend is closed; cannot run {operation}")
        if self._entity_faulted:
            raise RuntimeError(
                "newton backend is faulted after native submission; reconstruct it"
            )
        self.materialize()

    def materialize(self) -> None:
        if self._model is not None:
            return
        newton = self._deps.newton
        previous_layout = bool(newton.use_coord_layout_targets)
        newton.use_coord_layout_targets = True
        try:
            if self._portable_mode:
                assert self._variant_metadata is not None
                assert self._variant_assignment is not None
                source_builders = tuple(
                    build_newton_source_builder(
                        newton, item.source_model_file, item.gravity
                    )
                    for item in self._variant_metadata
                )
                validate_newton_variant_sources(source_builders, self._variant_metadata)
                has_authored_floor = all(
                    _prepare_newton_render_floor(item, newton) for item in source_builders
                )
                builder = build_newton_assigned_world_builder(
                    newton,
                    source_builders,
                    self._variant_metadata,
                    self._variant_assignment,
                )
                self._source_builders = source_builders
            else:
                template = newton.ModelBuilder()
                newton.solvers.SolverMuJoCo.register_custom_attributes(template)
                template.add_mjcf(self._metadata.source_model_file, ctrl_direct=False)
                has_authored_floor = _prepare_newton_render_floor(template, newton)
                builder = newton.ModelBuilder()
                builder.replicate(template, self._num_envs)
            if not has_authored_floor:
                _add_newton_render_floor(builder)
            self._model = builder.finalize(device=self._device)
        finally:
            newton.use_coord_layout_targets = previous_layout
            self.cleanup_scene_assets()

        if self._portable_mode:
            assert self._variant_metadata is not None
            assert self._variant_assignment is not None
            assert self._source_builders is not None
            assert self._entity_layout is not None
            audit_newton_variant_model(
                self._model,
                self._variant_metadata,
                self._source_builders,
                self._variant_assignment,
                self._entity_layout,
            )
        else:
            audit_newton_model(self._model, self._metadata, self._num_envs)
        if self._portable_mode:
            self._assign_portable_model_defaults()
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            separate_worlds=True,
            nconmax=self._nconmax,
            njmax=self._njmax,
            solver="newton",
            integrator="implicitfast",
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
        )
        actual_nconmax = int(self._solver.mjw_data.naconmax)
        actual_njmax = int(self._solver.mjw_data.njmax)
        if actual_nconmax < self._nconmax or actual_njmax < self._njmax:
            raise RuntimeError(
                "Newton compiled solver capacity below the requested explicit limit: "
                f"requested nconmax/njmax={self._nconmax}/{self._njmax}, "
                f"compiled {actual_nconmax}/{actual_njmax}"
            )
        # SolverMuJoCo may raise a requested capacity to the initial-state
        # requirement. Track the effective limits so every later check covers
        # the actual allocation and cannot miss a runtime truncation.
        self._nconmax = actual_nconmax
        self._njmax = actual_njmax
        if self._portable_mode:
            assert self._entity_layout is not None
            self._bind_entity_runtimes(self._entity_layout)
        else:
            self._view = newton.selection.ArticulationView(self._model, "*")
            if int(self._view.count_per_world) != 1:
                raise NotImplementedError(
                    "newton backend requires exactly one articulation per replicated world"
                )
        self._state = self._model.state()
        self._state_out = self._model.state()
        self._control = self._model.control()
        self._contacts = newton.Contacts(
            self._solver.get_max_contact_count(), 0, device=self._device
        )
        self._shape_world = np.asarray(self._model.shape_world.numpy(), dtype=np.int64)
        self._contact_sensor_pairs = self._resolve_contact_sensor_pairs()
        newton.eval_fk(
            self._model, self._state.joint_q, self._state.joint_qd, self._state
        )
        self._refresh_host_cache()
        calibrate_capacity(
            self._advance_capacity_probe,
            nconmax=self._nconmax,
            njmax=self._njmax,
            sample_steps=self._capacity_check_steps,
            context="Newton representative materialization rollout",
        )
        self._state = self._model.state()
        self._state_out = self._model.state()
        self._solver.reset(self._state)
        newton.eval_fk(
            self._model, self._state.joint_q, self._state.joint_qd, self._state
        )
        self._refresh_host_cache()
        if self._use_cuda_graph:
            self._initialize_cuda_graphs()

    def _bind_entity_runtimes(self, layout: CompiledSceneLayout) -> None:
        """Bind one public entity to one native articulation view on the cold path."""
        runtimes: dict[str, _NewtonEntityRuntime] = {}
        for entity in layout.entities:
            view = self._deps.newton.selection.ArticulationView(
                self._model, f"*/{entity.name}/{entity.root_body}"
            )
            if int(view.count) != self._num_envs or int(view.count_per_world) != 1:
                raise RuntimeError(
                    f"newton entity {entity.name!r} does not own one articulation per world"
                )
            qpos_indices = np.asarray(entity.qpos_indices, dtype=np.intp)
            qvel_indices = np.asarray(entity.qvel_indices, dtype=np.intp)
            if int(view.joint_coord_count) != qpos_indices.size:
                raise RuntimeError(
                    f"newton entity {entity.name!r} qpos width differs from the public layout"
                )
            if int(view.joint_dof_count) != qvel_indices.size:
                raise RuntimeError(
                    f"newton entity {entity.name!r} qvel width differs from the public layout"
                )
            labels = tuple(str(label) for label in view.body_labels)
            native_slots: list[int] = []
            for local_name, public_body_id in zip(
                entity.body_names, entity.body_ids, strict=True
            ):
                suffix = f"/{entity.name}/{local_name}"
                matches = [index for index, label in enumerate(labels) if label.endswith(suffix)]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"newton entity {entity.name!r} body {local_name!r} is not uniquely "
                        f"represented by its articulation view"
                    )
                native_slots.append(matches[0])
            if len(native_slots) != int(view.link_count):
                raise RuntimeError(
                    f"newton entity {entity.name!r} body count differs from its view"
                )
            runtimes[entity.name] = _NewtonEntityRuntime(
                view=view,
                qpos_indices=qpos_indices,
                qvel_indices=qvel_indices,
                view_body_indices=np.asarray(native_slots, dtype=np.intp),
            )
        if not runtimes:
            raise RuntimeError("newton portable scene contains no physical entities")
        self._entity_runtimes = runtimes
        self._view = next(iter(runtimes.values())).view

    def _assign_portable_model_defaults(self) -> None:
        """Materialize per-world authored defaults into the native model."""
        assert self._entity_layout is not None
        assert self._variant_metadata is not None
        assert self._variant_assignment is not None
        selected_defaults = np.stack(
            [item.default_qpos for item in self._variant_metadata]
        )[self._variant_assignment]
        raw_qpos = np.zeros((self._num_envs, self._entity_layout.nq), dtype=np.float32)
        for entity in self._entity_layout.entities:
            columns = np.asarray(entity.qpos_indices, dtype=np.intp)
            if columns.size == 0:
                continue
            values = np.ascontiguousarray(selected_defaults[:, columns])
            if entity.root_mode == "floating":
                values[:, 3:7] = self._wxyz_to_xyzw(values[:, 3:7])
            raw_qpos[:, columns] = values
        self._model.joint_q.assign(np.ascontiguousarray(raw_qpos.reshape(-1)))

    def _disable_cuda_graphs(self, reason: str) -> None:
        """Select eager execution and release captured graph references."""
        self._cuda_graph_enabled = False
        self._cuda_graphs = None
        self._cuda_graph_input_states = None
        self._cuda_graph_disable_reason = reason

    def _initialize_cuda_graphs(self) -> None:
        """Capture both Newton state-buffer transitions, or retain eager steps."""
        self._disable_cuda_graphs("CUDA graph capture has not been initialized")
        device = self._deps.warp.get_device()
        eligible, reason = _cuda_graph_eligibility(self._deps.warp, device)
        if not eligible:
            assert reason is not None
            self._cuda_graph_disable_reason = reason
            warnings.warn(
                f"newton CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            # Compile both transition directions before capture. Lazy Newton and
            # MJWarp kernels or allocations cannot be introduced inside a graph.
            warmup_ctrl = np.zeros((self._num_envs, self.num_actuators), dtype=np.float32)
            for _ in range(2):
                self._physics_substep(warmup_ctrl)
            self._deps.warp.synchronize_device(self._device)

            state_a = self._state
            state_b = self._state_out
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                with self._deps.warp.ScopedDevice(device):
                    with self._deps.warp.ScopedCapture() as capture_ab:
                        state_a.clear_forces()
                        self._solver.step(
                            state_a,
                            state_b,
                            self._control,
                            self._contacts,
                            self._sim_dt,
                        )
                    with self._deps.warp.ScopedCapture() as capture_ba:
                        state_b.clear_forces()
                        self._solver.step(
                            state_b,
                            state_a,
                            self._control,
                            self._contacts,
                            self._sim_dt,
                        )
            finally:
                if gc_enabled:
                    gc.enable()
            graphs = (capture_ab.graph, capture_ba.graph)
        except Exception as exc:
            reason = f"capture failed: {type(exc).__name__}: {exc}"
            self._disable_cuda_graphs(reason)
            warnings.warn(
                f"newton CUDA graphs disabled; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._cuda_graphs = graphs
        self._cuda_graph_input_states = (state_a, state_b)
        self._cuda_graph_enabled = True
        self._cuda_graph_disable_reason = None

        # Warmup advanced the simulation. Restore the explicit initial state
        # without replacing either State object, so captured pointers stay valid.
        self._solver.reset(self._state)
        self._deps.newton.eval_fk(
            self._model, self._state.joint_q, self._state.joint_qd, self._state
        )
        self._deps.warp.synchronize_device(self._device)
        self._refresh_host_cache()

    def _advance_capacity_probe(self) -> tuple[int, int]:
        self._physics_substep(np.zeros((self._num_envs, self.num_actuators), np.float32))
        return self._read_solver_counts()

    def _read_solver_counts(self) -> tuple[int, int]:
        self._deps.warp.synchronize_device(self._device)
        ncon = int(np.max(np.asarray(self._solver.mjw_data.nacon.numpy())))
        nefc = int(np.max(np.asarray(self._solver.mjw_data.nefc.numpy())))
        return ncon, nefc

    def _set_control(self, ctrl: np.ndarray) -> None:
        ctrl_array = np.asarray(ctrl, dtype=np.float32).reshape(self._num_envs, -1)
        if ctrl_array.shape[1] != self._metadata.nu:
            raise ValueError(
                f"newton control width differs from the compiled model: {ctrl_array.shape[1]}"
            )
        self._control_cache[...] = ctrl_array
        self._control_q_cache.fill(0.0)
        self._control_qd_cache.fill(0.0)
        for actuator_id, kind in enumerate(self._metadata.actuator_target_kinds):
            if kind == "position":
                self._control_q_cache[
                    :, self._metadata.actuator_target_qpos_adrs[actuator_id]
                ] = ctrl_array[:, actuator_id]
            elif kind == "velocity":
                self._control_qd_cache[
                    :, self._metadata.actuator_target_qvel_adrs[actuator_id]
                ] = ctrl_array[:, actuator_id]
        self._upload_control_state()

    def _upload_control_state(self) -> None:
        namespace = getattr(self._control, "mujoco", None)
        target = getattr(namespace, "ctrl", None) if namespace is not None else None
        if target is not None:
            target.assign(np.ascontiguousarray(self._control_cache.reshape(-1)))
        target_q = getattr(self._control, "joint_target_q", None)
        target_qd = getattr(self._control, "joint_target_qd", None)
        if target_q is not None:
            target_q.assign(np.ascontiguousarray(self._control_q_cache.reshape(-1)))
        if target_qd is not None:
            target_qd.assign(np.ascontiguousarray(self._control_qd_cache.reshape(-1)))

    def _replay_cuda_graph_substep(self) -> None:
        assert self._cuda_graphs is not None
        assert self._cuda_graph_input_states is not None
        if self._state is self._cuda_graph_input_states[0]:
            graph = self._cuda_graphs[0]
        elif self._state is self._cuda_graph_input_states[1]:
            graph = self._cuda_graphs[1]
        else:
            raise RuntimeError("newton CUDA graph state pointers are stale")
        self._deps.warp.capture_launch(graph)
        self._state, self._state_out = self._state_out, self._state

    def _physics_substep(self, ctrl: np.ndarray) -> None:
        self._set_control(ctrl)
        self._state.clear_forces()
        self._solver.step(
            self._state, self._state_out, self._control, self._contacts, self._sim_dt
        )
        self._state, self._state_out = self._state_out, self._state

    def _view_array(self, value: Any, width: int) -> np.ndarray:
        self._deps.warp.synchronize_device(self._device)
        return np.asarray(value.numpy(), dtype=np.float32).reshape(self._num_envs, width)

    @staticmethod
    def _xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
        return value[..., [3, 0, 1, 2]]

    @staticmethod
    def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
        return value[..., [1, 2, 3, 0]]

    def _refresh_host_cache(self, *, sensor_dt: float | None = None) -> None:
        if self._portable_mode:
            self._refresh_portable_host_cache(sensor_dt=sensor_dt)
            return
        qpos_raw = self._view_array(self._view.get_dof_positions(self._state), self._metadata.nq)
        qvel_raw = self._view_array(self._view.get_dof_velocities(self._state), self._metadata.nv)
        link_q = self._view_array(
            self._view.get_link_transforms(self._state), len(self._body_names) * 7
        ).reshape(self._num_envs, len(self._body_names), 7)
        link_qd = self._view_array(
            self._view.get_link_velocities(self._state), len(self._body_names) * 6
        ).reshape(self._num_envs, len(self._body_names), 6)

        self._qpos_cache[...] = qpos_raw
        if self._metadata.root_qpos_dim:
            self._qpos_cache[:, 3:7] = self._xyzw_to_wxyz(qpos_raw[:, 3:7])
        self._body_pos_cache[...] = link_q[..., :3]
        self._body_quat_cache[...] = self._xyzw_to_wxyz(link_q[..., 3:7])
        self._body_ang_vel_cache[...] = link_qd[..., 3:6]
        ipos = np.broadcast_to(
            self._metadata.body_ipos[1:], self._body_pos_cache.shape
        )
        offset_w = np_quat_apply_batched(self._body_quat_cache, ipos)
        self._body_lin_vel_cache[...] = link_qd[..., :3] - np.cross(
            self._body_ang_vel_cache, offset_w
        )
        self._qvel_cache[...] = qvel_raw
        if self._metadata.root_qvel_dim:
            root_quat = self._body_quat_cache[:, self._base_body_id]
            root_omega = self._body_ang_vel_cache[:, self._base_body_id]
            root_offset = offset_w[:, self._base_body_id]
            self._qvel_cache[:, :3] = qvel_raw[:, :3] - np.cross(root_omega, root_offset)
            self._qvel_cache[:, 3:6] = np_quat_apply_inverse_batched(root_quat, root_omega)
        self._refresh_sensor_cache(sensor_dt=sensor_dt)

    def _refresh_portable_host_cache(self, *, sensor_dt: float | None) -> None:
        assert self._entity_layout is not None
        self._qpos_cache.fill(0.0)
        self._qvel_cache.fill(0.0)
        self._body_pos_cache.fill(0.0)
        self._body_quat_cache.fill(0.0)
        self._body_lin_vel_cache.fill(0.0)
        self._body_ang_vel_cache.fill(0.0)
        link_velocities: dict[str, np.ndarray] = {}

        for entity_name, runtime in self._entity_runtimes.items():
            view = runtime.view
            qpos_width = runtime.qpos_indices.size
            if qpos_width:
                qpos_raw = self._view_array(view.get_dof_positions(self._state), qpos_width)
                self._qpos_cache[:, runtime.qpos_indices] = qpos_raw
                if int(view.root_joint_type) == int(self._deps.newton.JointType.FREE):
                    columns = runtime.qpos_indices
                    self._qpos_cache[:, columns[3:7]] = self._xyzw_to_wxyz(qpos_raw[:, 3:7])

            link_q = self._view_array(
                view.get_link_transforms(self._state),
                runtime.view_body_indices.size * 7,
            ).reshape(self._num_envs, runtime.view_body_indices.size, 7)
            link_qd = self._view_array(
                view.get_link_velocities(self._state),
                runtime.view_body_indices.size * 6,
            ).reshape(self._num_envs, runtime.view_body_indices.size, 6)
            entity = self._entity_layout.get_entity(entity_name)
            body_ids = entity.body_ids
            body_rows = np.asarray(body_ids, dtype=np.intp) - 1
            native_rows = runtime.view_body_indices
            self._body_pos_cache[:, body_rows] = link_q[:, native_rows, :3]
            self._body_quat_cache[:, body_rows] = self._xyzw_to_wxyz(
                link_q[:, native_rows, 3:7]
            )
            self._body_ang_vel_cache[:, body_rows] = link_qd[:, native_rows, 3:6]
            link_velocities[entity_name] = link_qd

            qvel_width = runtime.qvel_indices.size
            if qvel_width:
                qvel_raw = self._view_array(
                    view.get_dof_velocities(self._state), qvel_width
                )
                self._qvel_cache[:, runtime.qvel_indices] = qvel_raw

        offset_w = np_quat_apply_batched(self._body_quat_cache, self._body_ipos_cache)
        self._body_lin_vel_cache[...] = self._body_ang_vel_cache.copy()
        for entity_name, runtime in self._entity_runtimes.items():
            entity = self._entity_layout.get_entity(entity_name)
            body_rows = np.asarray(entity.body_ids, dtype=np.intp) - 1
            native_rows = runtime.view_body_indices
            link_qd = link_velocities[entity_name]
            self._body_lin_vel_cache[:, body_rows] = link_qd[:, native_rows, :3] - np.cross(
                self._body_ang_vel_cache[:, body_rows], offset_w[:, body_rows]
            )
            if entity.root_mode != "floating":
                continue
            root_row = body_rows[0]
            root_quat = self._body_quat_cache[:, root_row]
            root_omega = self._body_ang_vel_cache[:, root_row]
            root_offset = offset_w[:, root_row]
            raw_velocity = self._qvel_cache[:, runtime.qvel_indices].copy()
            raw_velocity[:, :3] -= np.cross(root_omega, root_offset)
            raw_velocity[:, 3:6] = np_quat_apply_inverse_batched(root_quat, root_omega)
            self._qvel_cache[:, runtime.qvel_indices] = raw_velocity

        self._refresh_sensor_cache(sensor_dt=sensor_dt)

    def _resolve_contact_sensor_pairs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        plans = [plan for plan in self._metadata.sensor_plans if plan.kind == "contact"]
        if not plans:
            return {}
        mapping = getattr(self._solver, "mjc_geom_to_newton_shape", None)
        if mapping is None:
            raise RuntimeError(
                "newton SolverMuJoCo did not publish mjc_geom_to_newton_shape; "
                "contact sensors cannot be resolved against the compiled model"
            )
        mapping_np = np.asarray(mapping.numpy(), dtype=np.int64)
        collision_columns = np.flatnonzero(
            (self._metadata.geom_contype != 0) | (self._metadata.geom_conaffinity != 0)
        )
        if mapping_np.ndim != 2 or mapping_np.shape[0] != self._num_envs:
            raise RuntimeError(
                "newton compiled geom mapping does not match the requested worlds: "
                f"shape {mapping_np.shape}, num_envs {self._num_envs}"
            )
        if mapping_np.shape[1] != collision_columns.size:
            raise RuntimeError(
                "newton compiled collision-geom mapping differs from the authored MJCF: "
                f"compiled {mapping_np.shape[1]}, authored {collision_columns.size}"
            )
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for plan in plans:
            shape_columns: list[int] = []
            for geom_id in (plan.geom1_id, plan.geom2_id):
                location = np.searchsorted(collision_columns, geom_id)
                if (
                    location >= collision_columns.size
                    or int(collision_columns[location]) != geom_id
                ):
                    raise RuntimeError(
                        f"newton compiled model dropped geom id {geom_id} required by "
                        f"contact sensor {plan.name!r}"
                    )
                shape_columns.append(int(location))
            shape_a = mapping_np[:, shape_columns[0]].copy()
            shape_b = mapping_np[:, shape_columns[1]].copy()
            if np.any(shape_a < 0) or np.any(shape_b < 0):
                raise RuntimeError(
                    f"newton compiled model dropped geoms {plan.geom1_name!r}/"
                    f"{plan.geom2_name!r} required by contact sensor {plan.name!r}"
                )
            pairs[plan.name] = (shape_a, shape_b)
        return pairs

    def _refresh_contact_sensors(self, plans: list[NewtonSensorPlan]) -> None:
        if self._shape_world is None:
            raise RuntimeError("newton backend is not materialized; cannot read contacts")
        self._solver.update_contacts(self._contacts)
        self._deps.warp.synchronize_device(self._device)
        count = int(np.asarray(self._contacts.rigid_contact_count.numpy())[0])
        shape0 = np.asarray(
            self._contacts.rigid_contact_shape0.numpy()[:count], dtype=np.int64
        )
        shape1 = np.asarray(
            self._contacts.rigid_contact_shape1.numpy()[:count], dtype=np.int64
        )
        for plan in plans:
            shape_a, shape_b = self._contact_sensor_pairs[plan.name]
            found = compute_contact_found_flags(
                self._shape_world, shape0, shape1, shape_a, shape_b
            )
            address, dim = self._sensor_slots[plan.name]
            self._sensor_cache[:, address : address + dim] = found[:, None]

    def _refresh_sensor_cache(self, *, sensor_dt: float | None) -> None:
        contact_plans = [
            plan for plan in self._metadata.sensor_plans if plan.kind == "contact"
        ]
        if contact_plans:
            self._refresh_contact_sensors(contact_plans)
        for plan in self._metadata.sensor_plans:
            if plan.kind == "contact":
                continue
            body_id = plan.body_id - 1
            if body_id < 0:
                raise RuntimeError(f"sensor {plan.name!r} is attached to the world body")
            address, dim = self._sensor_slots[plan.name]
            out = self._sensor_cache[:, address : address + dim]
            body_quat = self._body_quat_cache[:, body_id]
            site_quat = np_quat_mul_batched(
                body_quat, np.broadcast_to(plan.site_quat, body_quat.shape)
            )
            offset_w = np_quat_apply_batched(
                body_quat, np.broadcast_to(plan.site_pos, (self._num_envs, 3))
            )
            site_velocity = self._body_lin_vel_cache[:, body_id] + np.cross(
                self._body_ang_vel_cache[:, body_id], offset_w
            )
            if plan.kind == "gyro":
                out[...] = np_quat_apply_inverse_batched(
                    site_quat, self._body_ang_vel_cache[:, body_id]
                )
            elif plan.kind == "velocimeter":
                out[...] = np_quat_apply_inverse_batched(site_quat, site_velocity)
            elif plan.kind == "framepos":
                out[...] = self._body_pos_cache[:, body_id] + offset_w
            elif plan.kind == "framequat":
                out[...] = site_quat
            elif plan.kind == "framezaxis":
                out[...] = np_quat_apply_batched(
                    site_quat, np.broadcast_to(_WORLD_Z, (self._num_envs, 3))
                )
            elif plan.kind == "accelerometer":
                previous = self._previous_site_velocity.get(plan.name)
                acceleration = np.zeros_like(site_velocity)
                if previous is not None and sensor_dt is not None:
                    acceleration = (site_velocity - previous) / sensor_dt
                acceleration -= self._metadata.gravity
                out[...] = np_quat_apply_inverse_batched(site_quat, acceleration)
            self._previous_site_velocity[plan.name] = site_velocity.copy()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        self._require_state("model")
        return self._model

    @property
    def num_actuators(self) -> int:
        return len(self._metadata.actuator_names)

    @property
    def num_dof_vel(self) -> int:
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            return sum(len(joint.qvel_indices) for joint in entity.joints)
        return self._metadata.nv - self._metadata.root_qvel_dim

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._metadata.actuator_ctrl_range.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self._metadata.actuator_joint_names

    def get_scene_model_file(self) -> str | None:
        return self._metadata.diagnostic_model_file

    def get_scene_visual_model_file(self) -> str | None:
        return self._scene_visual_model_file

    def get_scene_layout(self) -> CompiledSceneLayout:
        if self._entity_faulted:
            raise RuntimeError("newton backend is faulted after native submission")
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
        assert self._variant_assignment is not None
        assert self._variant_metadata is not None
        variants = self._variant_assignment[ids]
        qpos = np.stack(
            [self._variant_metadata[int(variant)].default_qpos for variant in variants]
        )
        qvel = np.zeros((ids.size, layout.nv), dtype=np.float32)
        roots = np.zeros((ids.size, 13), dtype=np.float32)
        for row, variant in enumerate(variants):
            metadata = self._variant_metadata[int(variant)]
            root_body_id = owner.body_ids[0]
            roots[row, :3] = metadata.body_pos[root_body_id]
            roots[row, 3:7] = metadata.body_quat[root_body_id]
        return entity_state_snapshot(owner, qpos, qvel, roots)

    def _entity_roots(self) -> np.ndarray:
        layout = self.get_scene_layout()
        roots = np.zeros((self._num_envs, len(layout.entities), 13), dtype=np.float32)
        for index, entity in enumerate(layout.entities):
            root_body_id = entity.body_ids[0] - 1
            roots[:, index, :3] = self._body_pos_cache[:, root_body_id]
            roots[:, index, 3:7] = self._body_quat_cache[:, root_body_id]
            if entity.root_mode == "floating":
                state = entity_state_snapshot(entity, self._qpos_cache, self._qvel_cache)
                roots[:, index, 7:] = state["root_velocity"]
        return roots

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        if owner.root_mode == "floating":
            return entity_state_snapshot(owner, self._qpos_cache, self._qvel_cache)
        root = np.zeros((self._num_envs, 13), dtype=np.float32)
        root_body_id = owner.body_ids[0] - 1
        root[:, :3] = self._body_pos_cache[:, root_body_id]
        root[:, 3:7] = self._body_quat_cache[:, root_body_id]
        return entity_state_snapshot(owner, self._qpos_cache, self._qvel_cache, root)

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        try:
            return self._keyframes[name].copy()
        except KeyError as exc:
            raise ValueError(f"Keyframe {name!r} not found") from exc

    def get_default_qpos(self) -> np.ndarray:
        return self._metadata.default_qpos.copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return self._metadata.default_qpos[self._metadata.root_qpos_dim :].copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._metadata.nv,), dtype=np.float32)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self._portable_mode:
            assert self._entity_layout is not None
            entity_name, separator, local_name = str(root_body_name).partition("/")
            if not separator:
                raise ValueError("portable Newton root names use entity/local_name")
            entity = self._entity_layout.get_entity(entity_name)
            if local_name != entity.root_body:
                raise ValueError(
                    f"root {root_body_name!r} is not entity {entity.name!r}'s root body"
                )
            if entity.root_mode != "floating":
                raise NotImplementedError(
                    f"portable Newton entity {entity.name!r} has no floating root state"
                )
            return BackendRootStateLayout(
                tuple(entity.root_qpos_indices), tuple(entity.root_qvel_indices)
            )
        if root_body_name != self._base_name or not self._metadata.root_qpos_dim:
            raise NotImplementedError(
                f"backend 'newton' requires {self._base_name!r} as its free root body"
            )
        return BackendRootStateLayout(tuple(range(7)), tuple(range(6)))

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([self._body_ids[str(name)] for name in names], dtype=np.int32)
        except KeyError as exc:
            raise ValueError(f"Body {exc.args[0]!r} not found in newton model") from exc

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        # Motion datasets use MuJoCo-style body ids, where worldbody is id 0;
        # ``_body_names`` drops worldbody, so get_body_ids is off by one.
        return self.get_body_ids(names) + 1

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        root = int(root_body_id)
        if root < 0 or root >= len(self._body_names):
            raise ValueError(f"root_body_id out of range: {root}")
        parents = self._metadata.body_parent_ids[1:] - 1
        descendants = {root}
        for body_id in range(root + 1, len(parents)):
            if int(parents[body_id]) in descendants:
                descendants.add(body_id)
        return np.asarray(sorted(descendants), dtype=np.int32)

    def get_gravity(self) -> np.ndarray:
        return self._metadata.gravity.copy()

    def get_body_mass(self) -> np.ndarray:
        if self._portable_mode:
            return self._body_mass_cache.copy()
        return self._metadata.body_mass.copy()

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if env_ids is not None and not self._portable_mode:
            raise NotImplementedError(
                "NewtonBackend does not expose per-environment body ipos"
            )
        if env_ids is not None:
            ids = selected_state_rows(env_ids, self._num_envs)
            return self._body_ipos_cache[ids].copy()
        return self._metadata.body_ipos.copy()

    def get_dof_armature(self) -> np.ndarray:
        return self._metadata.dof_armature.copy()

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        self._reject_named_joint_ranges(names, "joint ranges")
        value = self._metadata.joint_range
        return None if value is None else value.copy()

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self._metadata.joint_dof_adrs[self._joint_ids[str(name)]] for name in names],
            dtype=np.int32,
        )

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        full = [self._metadata.joint_qpos_adrs[self._joint_ids[str(name)]] for name in names]
        return np.asarray(full, dtype=np.int32) - self._metadata.root_qpos_dim

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_indices(names) - self._metadata.root_qvel_dim

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names) + self._metadata.root_qpos_dim

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_vel_indices(names) + self._metadata.root_qvel_dim

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self._metadata.actuator_kp.copy(), self._metadata.actuator_kd.copy()

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict[str, dict[str, float]]:
        self._require_state("step")
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise ValueError(f"nsteps must be a positive integer, got {nsteps!r}")
        ctrl_array = np.asarray(ctrl, dtype=np.float32)
        expected = (self._num_envs, self.num_actuators)
        if ctrl_array.shape != expected:
            raise ValueError(f"ctrl must have shape {expected}, got {ctrl_array.shape}")
        t0 = time.perf_counter()
        if self._cuda_graph_enabled and self._pre_step_control_fn is None:
            self._set_control(ctrl_array)
            for _ in range(nsteps):
                self._replay_cuda_graph_substep()
        else:
            for _ in range(nsteps):
                native_ctrl = self._apply_pre_step_control(ctrl_array)
                self._physics_substep(native_ctrl)
        self._deps.warp.synchronize_device(self._device)
        physics_ms = (time.perf_counter() - t0) * 1000.0
        ncon, nefc = self._read_solver_counts()
        validate_capacity_limits(
            nconmax=self._nconmax,
            njmax=self._njmax,
            peak_ncon=ncon,
            peak_nefc=nefc,
            context="Newton runtime step",
        )
        t0 = time.perf_counter()
        self._refresh_host_cache(sensor_dt=nsteps * self._sim_dt)
        self._time_cache += np.float32(nsteps * self._sim_dt)
        cache_ms = (time.perf_counter() - t0) * 1000.0
        return {"timing": {"physics_ms": physics_ms, "host_cache_refresh_ms": cache_ms}}

    def _raw_state_from_public(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_qpos = np.zeros_like(qpos, dtype=np.float32)
        raw_qvel = np.zeros_like(qvel, dtype=np.float32)
        assert self._entity_layout is not None
        for entity_name, runtime in self._entity_runtimes.items():
            entity = self._entity_layout.get_entity(entity_name)
            q_columns = runtime.qpos_indices
            v_columns = runtime.qvel_indices
            raw_qpos[:, q_columns] = qpos[:, q_columns]
            raw_qvel[:, v_columns] = qvel[:, v_columns]
            if entity.root_mode != "floating":
                continue
            raw_qpos[:, q_columns[3:7]] = self._wxyz_to_xyzw(qpos[:, q_columns[3:7]])
            pose_quat = qpos[:, q_columns[3:7]]
            public_velocity = qvel[:, v_columns]
            # ``_qvel_cache`` carries body-frame angular velocity, while the
            # Newton free-root qvel uses world-frame angular velocity and a
            # COM-origin linear component.
            omega_world = np_quat_apply_batched(pose_quat, public_velocity[:, 3:6])
            root_body_id = entity.body_ids[0] - 1
            offset_w = np_quat_apply_batched(
                pose_quat, self._body_ipos_cache[:, root_body_id]
            )
            raw_qvel[:, v_columns[:3]] = public_velocity[:, :3] + np.cross(
                omega_world, offset_w
            )
            raw_qvel[:, v_columns[3:6]] = omega_world
        return raw_qpos, raw_qvel

    def _commit_portable_state(
        self,
        raw_qpos: np.ndarray,
        raw_qvel: np.ndarray,
        rows: np.ndarray,
        entity_names: set[str] | None = None,
    ) -> None:
        selected_names = set(self._entity_runtimes) if entity_names is None else entity_names
        mask = np.zeros((self._num_envs,), dtype=np.bool_)
        mask[rows] = True
        for entity_name in selected_names:
            runtime = self._entity_runtimes[entity_name]
            q_columns = runtime.qpos_indices
            v_columns = runtime.qvel_indices
            # Newton's public selection views index zero-width slices as 0:0,
            # which Warp rejects. A no-DoF static articulation has no state to
            # submit or forward, so avoid the no-op calls at the owner boundary.
            if q_columns.size:
                runtime.view.set_dof_positions(
                    self._state,
                    np.ascontiguousarray(raw_qpos[:, None, q_columns]),
                    mask=mask,
                )
            if v_columns.size:
                runtime.view.set_dof_velocities(
                    self._state,
                    np.ascontiguousarray(raw_qvel[:, None, v_columns]),
                    mask=mask,
                )
            if q_columns.size or v_columns.size:
                runtime.view.eval_fk(self._state, mask=mask)
        warp_mask = self._deps.warp.array(
            np.concatenate((mask, (False,))),
            dtype=self._deps.warp.bool,
            device=self._device,
        )
        self._solver.reset(self._state, warp_mask, flags=0)
        self._upload_control_state()

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict[str, dict[str, float]]:
        self._require_state("set_state")
        rows = np.asarray(env_indices, dtype=np.intp)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self._num_envs):
            raise ValueError("env_indices must be a one-dimensional in-range index array")
        if np.unique(rows).size != rows.size:
            raise ValueError("env_indices must not contain duplicates")
        qpos_array = np.asarray(qpos, dtype=np.float32)
        qvel_array = np.asarray(qvel, dtype=np.float32)
        if qpos_array.shape != (rows.size, self._metadata.nq):
            raise ValueError("qpos shape does not match selected Newton worlds")
        if qvel_array.shape != (rows.size, self._metadata.nv):
            raise ValueError("qvel shape does not match selected Newton worlds")
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError("newton backend does not yet support reset randomization")
        if not rows.size:
            return {"timing": {"set_state_upload_ms": 0.0, "set_state_cache_ms": 0.0}}

        t0 = time.perf_counter()
        full_qpos = self._qpos_cache.copy()
        full_qvel = self._qvel_cache.copy()
        full_qpos[rows] = qpos_array
        full_qvel[rows] = qvel_array
        if self._portable_mode:
            raw_qpos, raw_qvel = self._raw_state_from_public(full_qpos, full_qvel)
            try:
                self._commit_portable_state(raw_qpos, raw_qvel, rows)
            except BaseException:
                self._entity_faulted = True
                raise
            upload_ms = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            self._refresh_host_cache()
            self._time_cache[rows] = 0.0
            cache_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "timing": {
                    "set_state_upload_ms": upload_ms,
                    "set_state_cache_ms": cache_ms,
                }
            }
        if self._metadata.root_qpos_dim:
            full_qpos[:, 3:7] = self._wxyz_to_xyzw(full_qpos[:, 3:7])
            root_quat = qpos_array[:, 3:7]
            omega_world = np_quat_apply_batched(root_quat, qvel_array[:, 3:6])
            ipos = np.broadcast_to(
                self._metadata.body_ipos[self._base_body_id + 1], (rows.size, 3)
            )
            offset_w = np_quat_apply_batched(root_quat, ipos)
            full_qvel[rows, :3] = qvel_array[:, :3] + np.cross(omega_world, offset_w)
            full_qvel[rows, 3:6] = omega_world
        mask = np.zeros((self._num_envs,), dtype=np.bool_)
        mask[rows] = True
        warp_mask = self._deps.warp.array(
            np.concatenate((mask, (False,))),
            dtype=self._deps.warp.bool,
            device=self._device,
        )
        qpos_device = self._deps.warp.array(
            full_qpos[:, None, :], dtype=self._deps.warp.float32, device=self._device
        )
        qvel_device = self._deps.warp.array(
            full_qvel[:, None, :], dtype=self._deps.warp.float32, device=self._device
        )
        self._view.set_dof_positions(self._state, qpos_device, mask=mask)
        self._view.set_dof_velocities(self._state, qvel_device, mask=mask)
        self._deps.newton.eval_fk(
            self._model, self._state.joint_q, self._state.joint_qd, self._state
        )
        # flags=0 clears MuJoCo's persistent solver buffers without reverting
        # joint_q/joint_qd to model defaults. update_data_interval=1 performs
        # the subsequent public state-to-MJWarp synchronization on step.
        self._solver.reset(self._state, warp_mask, flags=0)
        upload_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache[rows] = 0.0
        cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timing": {"set_state_upload_ms": upload_ms, "set_state_cache_ms": cache_ms}
        }

    def reset_entities(self, request: SceneResetRequest) -> None:
        if not self._portable_mode:
            super().reset_entities(request)
            return
        self._require_state("reset_entities")
        if request.restore_default_controls:
            raise NotImplementedError(
                "portable Newton entity reset does not support "
                "restore_default_controls; selected controls are cleared while "
                "unselected controls persist"
            )
        layout = self.get_scene_layout()
        prepared = prepare_scene_reset(
            layout,
            request,
            self._qpos_cache,
            self._qvel_cache,
            self._entity_roots(),
        )
        rows = prepared.env_ids
        qpos = self._qpos_cache.copy()
        qvel = self._qvel_cache.copy()
        qcols = np.flatnonzero(prepared.qpos_mask)
        vcols = np.flatnonzero(prepared.qvel_mask)
        qpos[np.ix_(rows, qcols)] = prepared.qpos[:, qcols]
        qvel[np.ix_(rows, vcols)] = prepared.qvel[:, vcols]

        # A selected reset clears only selected entity controls. Controls for
        # untouched entities and environments are uploaded back after the
        # solver's selected-world reset barrier.
        for entity_name in prepared.entity_names:
            entity = layout.get_entity(entity_name)
            self._control_cache[np.ix_(rows, entity.actuator_indices)] = 0.0
            for actuator_id in entity.actuator_indices:
                kind = self._metadata.actuator_target_kinds[actuator_id]
                if kind == "position":
                    self._control_q_cache[
                        rows, self._metadata.actuator_target_qpos_adrs[actuator_id]
                    ] = 0.0
                elif kind == "velocity":
                    self._control_qd_cache[
                        rows, self._metadata.actuator_target_qvel_adrs[actuator_id]
                    ] = 0.0

        raw_qpos, raw_qvel = self._raw_state_from_public(qpos, qvel)
        try:
            self._commit_portable_state(raw_qpos, raw_qvel, rows, set(prepared.entity_names))
            self._refresh_host_cache()
            self._time_cache[rows] = 0.0
        except BaseException:
            self._entity_faulted = True
            raise

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities()

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        native = newton_render_dependencies_available()
        return BackendPlayCapabilities(
            supports_native_interactive_renderer=native,
            supports_physics_state_playback=True,
            supports_native_video_capture=native,
            supports_debug_overlay=True,
        )

    # Static so the plan resolves without a backend instance (class-level
    # calls in tests); instance calls keep working.  The base declares an
    # instance method, hence the override ignores.
    @staticmethod
    def resolve_play_render_plan(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve playback modes, selecting the concrete renderer.

        ``record`` uses the native ViewerGL offscreen renderer (the normal
        ``newton`` install includes its viewer dependencies) and falls back to
        the offline MuJoCo snapshot pipeline only for an incomplete runtime.
        ``interactive`` requires both the viewer dependencies and a reachable
        display and fails closed otherwise.
        ``auto`` resolves to ``interactive`` with a display and ``record``
        without one.  A staticmethod so the semantics stay testable without a
        CUDA runtime.
        """
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
            mode = "interactive" if display_available() else "record"
        if mode == "interactive":
            if not newton_render_dependencies_available():
                raise NotImplementedError(
                    "newton interactive playback requires the native viewer dependencies "
                    "(pyglet>=2.1.6,<3, imgui-bundle>=1.92.0); install them with "
                    "`uv sync --extra newton`, or select "
                    "training.play_render_mode=record or none."
                )
            if not display_available():
                raise NotImplementedError(
                    "newton interactive playback requires a reachable display "
                    "(DISPLAY or WAYLAND_DISPLAY); on headless hosts select "
                    "training.play_render_mode=record (offscreen GL needs EGL via "
                    "PYOPENGL_PLATFORM=egl, or GLX under Wayland)."
                )
            return BackendPlayRenderPlan(
                mode="interactive",
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
                renderer=NEWTON_NATIVE_RENDERER,
            )
        if isinstance(play_steps, bool) or play_steps is None or int(play_steps) <= 0:
            raise ValueError(
                "newton record playback requires a positive finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("newton record playback requires an output video path.")
        renderer = (
            NEWTON_NATIVE_RENDERER
            if newton_render_dependencies_available()
            else MUJOCO_SNAPSHOT_RENDERER
        )
        return BackendPlayRenderPlan(
            mode="record",
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
            renderer=renderer,
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
        if not should_run_headless and not should_record:
            if debug_overlay_getter is not None or on_frame is not None:
                raise NotImplementedError(
                    "newton interactive playback supports neither debug overlay primitives "
                    "nor on_frame callbacks; use play_render_mode=record"
                )
            try:
                return run_newton_native_playback(
                    backend=self,
                    env=env,
                    initialize=initialize,
                    step=step,
                    num_steps=num_steps,
                    output_video=None,
                    render_spacing=render_spacing,
                    headless=False,
                    record_video=False,
                    camera_kwargs=camera,
                )
            except RenderClosedError:
                logger.info("Render window closed.")
                return None
        if debug_overlay_getter is not None or on_frame is not None:
            # The native ViewerGL renderer can neither inject user geoms nor
            # post-process frames; overlays and on_frame route to the offline
            # MuJoCo snapshot pipeline even when the native viewer
            # dependencies are installed.
            return run_offline_snapshot_playback(
                backend=self,
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                headless=should_run_headless,
                record_video=should_record,
                snapshot_shape=(self._num_envs, 1 + self._metadata.nq + self._metadata.nv),
                frame_state_getter=frame_state_getter,
                camera_kwargs=camera,
                backend_label="newton",
                debug_overlay_getter=debug_overlay_getter,
                on_frame=on_frame,
            )
        if newton_render_dependencies_available():
            return run_newton_native_playback(
                backend=self,
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                output_video=output_video,
                render_spacing=render_spacing,
                headless=should_run_headless,
                record_video=should_record,
                camera_kwargs=camera,
            )
        return run_offline_snapshot_playback(
            backend=self,
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=should_run_headless,
            record_video=should_record,
            snapshot_shape=(self._num_envs, 1 + self._metadata.nq + self._metadata.nv),
            frame_state_getter=frame_state_getter,
            camera_kwargs=camera,
            backend_label="newton",
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
        """Attach a native ViewerGL renderer on the cold playback path.

        ``offset_mode`` is accepted for contract parity and ignored: envs are
        laid out with ViewerGL's grid world offsets.  ``camera_kwargs`` is
        validated (normalized to :class:`CameraCfg`) but the native viewer
        keeps its default camera (the offline MuJoCo snapshot path honors the
        camera configuration).  The first (headless, capture) pair is pinned.
        """
        del offset_mode
        CameraCfg.from_kwargs(camera_kwargs)
        config = (bool(headless), bool(capture))
        if self._viewer is not None:
            if self._render_config != config:
                raise RuntimeError(
                    "newton renderer is already initialized with "
                    f"(headless, capture)={self._render_config}, requested {config}"
                )
            return
        require_newton_render_dependencies()
        self._require_state("init_renderer")
        try:
            from newton.viewer import ViewerGL
        except ImportError as exc:
            raise RuntimeError(
                "newton native renderer could not import newton.viewer.ViewerGL: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            viewer = ViewerGL(width=int(width), height=int(height), headless=bool(headless))
        except Exception as exc:
            raise RuntimeError(
                "newton native viewer could not create an OpenGL context "
                f"({type(exc).__name__}: {exc}); interactive mode needs a reachable "
                "display (DISPLAY or WAYLAND_DISPLAY), headless offscreen mode needs "
                "EGL (PYOPENGL_PLATFORM=egl) or GLX under Wayland."
            ) from exc
        viewer.set_model(self._model)
        if self._num_envs > MAX_RENDER_WORLDS:
            viewer.set_visible_worlds(list(range(MAX_RENDER_WORLDS)))
        viewer.set_world_offsets((float(spacing), float(spacing), 0.0))
        self._viewer = viewer
        self._render_config = config

    def _require_viewer(self, operation: str) -> Any:
        if self._viewer is None:
            raise RuntimeError(f"newton {operation} requires init_renderer first")
        return self._viewer

    def _render_viewer_frame(self, viewer: Any) -> None:
        self._require_state("render")
        viewer.begin_frame(float(self._time_cache[0]))
        viewer.log_state(self._state)
        viewer.end_frame()

    def render(self) -> None:
        viewer = self._require_viewer("render")
        self._render_viewer_frame(viewer)
        if not viewer.is_running():
            raise RenderClosedError("newton render window was closed")

    def capture_video_frame(self) -> np.ndarray:
        viewer = self._require_viewer("capture_video_frame")
        self._render_viewer_frame(viewer)
        return np.asarray(viewer.get_frame().numpy(), dtype=np.uint8)

    def get_physics_state(self) -> np.ndarray:
        self._require_state("get_physics_state")
        nq = self._metadata.nq
        nv = self._metadata.nv
        state = np.empty((self._num_envs, 1 + nq + nv), dtype=np.float32)
        state[:, 0] = self._time_cache
        state[:, 1 : 1 + nq] = self._qpos_cache
        state[:, 1 + nq :] = self._qvel_cache
        return state

    def set_physics_state(self, state: np.ndarray) -> None:
        """Restore a ``get_physics_state`` snapshot through the set_state path."""
        self._require_state("set_physics_state")
        nq = self._metadata.nq
        nv = self._metadata.nv
        state_array = np.asarray(state, dtype=np.float32)
        expected = (self._num_envs, 1 + nq + nv)
        if state_array.shape != expected:
            raise ValueError(
                f"newton physics snapshot must use [time, qpos, qvel] layout with shape "
                f"{expected}, got {state_array.shape}"
            )
        self.set_state(
            np.arange(self._num_envs, dtype=np.intp),
            state_array[:, 1 : 1 + nq],
            state_array[:, 1 + nq :],
        )
        self._time_cache[...] = state_array[:, 0]

    def get_playback_model(self, env_index: int | None = None) -> str:
        if env_index is not None:
            idx = int(env_index)
            if idx < 0 or idx >= self._num_envs:
                raise IndexError(f"env_index must be in [0, {self._num_envs - 1}], got {idx}")
        if self._portable_mode:
            assert self._variant_assignment is not None
            assert self._variant_metadata is not None
            index = 0 if env_index is None else int(env_index)
            variant = int(self._variant_assignment[index])
            if variant not in self._portable_playback_models_validated:
                validate_offline_visual_model(
                    mujoco=self._deps.mujoco,
                    physics_model=self._variant_metadata[variant].playback_model,
                    model_file=self._variant_metadata[variant].source_model_file,
                    backend_label="newton",
                )
                self._portable_playback_models_validated.add(variant)
            return str(self._variant_metadata[variant].source_model_file)
        if not self._playback_model_validated:
            self._scene_visual_model_file = validate_offline_visual_model(
                mujoco=self._deps.mujoco,
                physics_model=self._metadata.playback_model,
                model_file=self._scene_visual_model_file,
                backend_label="newton",
            )
            self._playback_model_validated = True
        return self._scene_visual_model_file

    def get_base_pos(self) -> np.ndarray:
        self._require_state("get_base_pos")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            if entity.root_mode != "floating":
                raise NotImplementedError("newton fixed-root scenes use entity state APIs")
            return self._qpos_cache[:, entity.root_qpos_indices[:3]]
        return self._qpos_cache[:, :3]

    def get_base_quat(self) -> np.ndarray:
        self._require_state("get_base_quat")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            if entity.root_mode != "floating":
                raise NotImplementedError("newton fixed-root scenes use entity state APIs")
            return self._qpos_cache[:, entity.root_qpos_indices[3:7]]
        return self._qpos_cache[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_state("get_base_lin_vel")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            if entity.root_mode != "floating":
                raise NotImplementedError("newton fixed-root scenes use entity state APIs")
            return self._qvel_cache[:, entity.root_qvel_indices[:3]]
        return self._qvel_cache[:, :3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_state("get_base_ang_vel")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            if entity.root_mode != "floating":
                raise NotImplementedError("newton fixed-root scenes use entity state APIs")
            return np_quat_apply_batched(
                self._qpos_cache[:, entity.root_qpos_indices[3:7]],
                self._qvel_cache[:, entity.root_qvel_indices[3:6]],
            )
        quat = self._qpos_cache[:, 3:7]
        return np_quat_apply_batched(quat, self._qvel_cache[:, 3:6])

    def get_dof_pos(self) -> np.ndarray:
        self._require_state("get_dof_pos")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            return self._qpos_cache[
                :, tuple(i for joint in entity.joints for i in joint.qpos_indices)
            ]
        return self._qpos_cache[:, self._metadata.root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
        if self._portable_mode:
            assert self._entity_layout is not None
            entity = self._entity_layout.get_entity(self._primary_entity_name)
            return self._qvel_cache[
                :, tuple(i for joint in entity.joints for i in joint.qvel_indices)
            ]
        return self._qvel_cache[:, self._metadata.root_qvel_dim :]

    def _ids(self, body_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(body_ids, dtype=np.intp)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self._body_names)):
            raise ValueError("body_ids must be one-dimensional and in range")
        return ids

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_pos_w")
        return self._body_pos_cache[:, self._ids(body_ids)]

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_quat_w")
        return self._body_quat_cache[:, self._ids(body_ids)]

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_lin_vel_w")
        return self._body_lin_vel_cache[:, self._ids(body_ids)]

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        self._require_state("get_body_ang_vel_w")
        return self._body_ang_vel_cache[:, self._ids(body_ids)]

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._ids(body_ids)
        delta = self.get_body_pos_w(ids) - self._body_pos_cache[:, self._base_body_id, None]
        root = self._body_quat_cache[:, self._base_body_id, None]
        return np_quat_apply_inverse_batched(root, delta)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        ids = self._ids(body_ids)
        root = self._body_quat_cache[:, self._base_body_id, None]
        return np_quat_mul_batched(np_quat_conjugate_batched(root), self.get_body_quat_w(ids))

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        quat = self.get_body_quat_w(body_ids)
        return np_quat_apply_inverse_batched(quat, self.get_body_lin_vel_w(body_ids))

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        quat = self.get_body_quat_w(body_ids)
        return np_quat_apply_inverse_batched(quat, self.get_body_ang_vel_w(body_ids))

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._require_state("get_sensor_data")
        try:
            address, dim = self._sensor_slots[name]
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} not found in newton model") from exc
        return self._sensor_cache[:, address : address + dim]

    def _bind_sensor_data_reader(self, names: tuple[str, ...]):
        slices = [self._sensor_slots[name] for name in names]
        contiguous = all(
            slices[index][0] + slices[index][1] == slices[index + 1][0]
            for index in range(len(slices) - 1)
        )
        if contiguous:
            start = slices[0][0]
            width = sum(dim for _, dim in slices)
            return lambda: self._sensor_cache[:, start : start + width]
        return lambda: np.concatenate(
            [self._sensor_cache[:, address : address + dim] for address, dim in slices], axis=1
        )

    def close(self) -> None:
        self._disable_cuda_graphs("newton backend is closed")
        if self._viewer is not None:
            viewer = self._viewer
            self._viewer = None
            try:
                viewer.close()
            except Exception:  # teardown must not mask the primary lifecycle
                logger.debug("newton viewer close failed", exc_info=True)
        self._closed = True
        self.cleanup_scene_assets()


__all__ = ["NewtonBackend"]
