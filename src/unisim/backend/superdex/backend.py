"""CPU SuperDex adapter with NumPy state barriers and independent native scenes."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import replace
from itertools import chain
from typing import Any

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    CameraCfg,
    SimBackend,
    normalize_play_render_mode,
)
from unisim.dr.types import DomainRandomizationCapabilities, ResetRandomizationPayload
from unisim.entities import SceneResetRequest
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset, selected_state_rows
from unisim.inspection import ConfigurationField, ConfigurationProvenance
from unisim.scene import SceneCfg, require_scene_composition_support
from unisim.utils.rotation import (
    np_quat_apply_batched as rotate,
)
from unisim.utils.rotation import (
    np_quat_apply_inverse_batched as unrotate,
)
from unisim.utils.rotation import (
    np_quat_conjugate_batched as conjugate,
)
from unisim.utils.rotation import (
    np_quat_mul_batched as multiply,
)

from .cpu_topology import physical_cpu_count
from .dependencies import load_superdex_dependencies
from .plans import ModelPlan, SensorPlan
from .runtime import acquire_runtime, release_runtime


class SuperDexBackend(SimBackend):
    """One independent CPU scene per environment, initialized in its owning process.

    Native engine state is translated at materialize/set_state/step barriers.
    Public getters read detached NumPy caches; they never parse assets or query
    native metadata. A source-built SceneBatchExecutor owns the hot-path CPU
    barrier for independent scenes in the default "batch" execution mode;
    "serial" mode steps every scene on the environment thread so the native
    SuperDex debugger can attach without violating DebugDraw thread affinity.
    Reset and sensor ownership remain here.
    """

    backend_type = "superdex"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int,
        sim_dt: float,
        *,
        base_name: str | None = None,
        num_workers: int = 0,
        execution_mode: str = "batch",
        effort_limits: Sequence[float] | None = None,
        allow_contact_approximation: bool = False,
        **unexpected: Any,
    ) -> None:
        require_scene_composition_support(scene, "superdex")
        if unexpected:
            raise TypeError(f"SuperDexBackend does not accept options: {sorted(unexpected)}")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not np.isfinite(sim_dt) or sim_dt <= 0:
            raise ValueError("sim_dt must be finite and positive")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer (0 is automatic)")
        if not isinstance(execution_mode, str):
            raise TypeError("execution_mode must be a string")
        if execution_mode not in ("batch", "serial"):
            raise ValueError("execution_mode must be 'batch' or 'serial'")
        if execution_mode == "serial" and num_workers:
            raise ValueError("num_workers has no effect in serial execution mode")
        if not isinstance(allow_contact_approximation, bool):
            raise TypeError("allow_contact_approximation must be bool")
        self._num_envs = num_envs
        self._dt = float(sim_dt)
        self.scene_visual_model_file = scene.visual_model_file or (
            scene.model_file if scene.model_file.lower().endswith(".xml") else None
        )
        self._pid = os.getpid()
        self._pre_step_control_fn = None
        self._closed = False
        self._acquired = False
        self._entity_faulted = False
        self._batch_executor = None
        self._execution_mode = execution_mode
        self._batch_num_workers = (
            self._resolve_num_workers(num_workers) if execution_mode == "batch" else 0
        )
        self._worlds: list[Any] = []
        self._actors: list[Any] = []
        self._links: list[list[Any]] = []
        self._native_actor_links: list[tuple[tuple[Any, ...], ...]] = []
        self._actor_cleanups: list[Any] = []
        self._snapshots: list[Any] = []
        self._sensor_sources: list[dict[str, tuple[Any, Any]]] = []
        self._plan: ModelPlan | None = None
        self._p, self._r = load_superdex_dependencies()
        self._dtype = np.float64 if self._p.uses_double_precision() else np.float32
        try:
            acquire_runtime(self._p)
            self._acquired = True
            from .materialization import materialize_model

            self._plan = materialize_model(
                self._p,
                self._r,
                scene,
                effort_limits=effort_limits,
                allow_contact_approximation=allow_contact_approximation,
                sim_dt=self._dt,
            )
            self._body_lookup = {name: i for i, name in enumerate(self._plan.body_names)}
            self._joint_lookup = {name: i for i, name in enumerate(self._plan.joint_names)}
            self._base_id = (
                self._body_lookup[base_name] if base_name is not None else self._plan.root_body_id
            )
            self._allocate_caches()
            self.materialize()
            report = self.get_import_report()
            self._import_report = replace(
                report,
                fields=report.fields + (
                    ConfigurationField(
                        "superdex_allow_contact_approximation",
                        requested=allow_contact_approximation,
                        effective=allow_contact_approximation,
                        difference="exact",
                        provenance=(
                            ConfigurationProvenance(
                                "adapter_setting",
                                "SuperDex materialization contact approximation opt-in",
                            ),
                        ),
                    ),
                ),
            )
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> ModelPlan:
        assert self._plan is not None
        return self._plan

    @property
    def num_actuators(self) -> int:
        return len(self.model.actuator_names)

    @property
    def num_dof_vel(self) -> int:
        return len(self.model.joint_names)

    def _check_open(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError(
                "SuperDex backend cannot cross processes; pass an EnvFactory to spawn"
            )
        if self._closed:
            raise RuntimeError("SuperDex backend is closed")
        if self._entity_faulted:
            raise RuntimeError("SuperDex backend is faulted after a partial native entity write")

    def _resolve_num_workers(self, requested: int) -> int:
        """Resolve native scene workers from the current process affinity."""
        if requested:
            return min(self.num_envs, requested)
        return min(self.num_envs, physical_cpu_count())

    def _allocate_caches(self) -> None:
        n, m, dtype = self.num_envs, self.model, self._dtype
        default_ctrl = (
            np.zeros(self.num_actuators, dtype=dtype)
            if m.default_ctrl is None
            else np.asarray(m.default_ctrl, dtype=dtype)
        )
        if default_ctrl.shape != (self.num_actuators,) or not np.isfinite(default_ctrl).all():
            raise ValueError("SuperDex default controls have an invalid shape or values")
        self._default_ctrl = default_ctrl.copy()
        self._qpos = np.zeros((n, m.nq), dtype=dtype)
        self._qvel = np.zeros((n, m.nv), dtype=dtype)
        self._ctrl = np.zeros((n, self.num_actuators), dtype=dtype)
        self._batch_ctrl = np.zeros_like(self._ctrl)
        self._native_q = np.zeros((n, m.nv), dtype=dtype)
        self._native_v = np.zeros_like(self._native_q)
        self._batch_forces = np.zeros_like(self._native_q)
        self._env_ids = np.arange(n, dtype=np.intp)
        self._pending_wrench = np.zeros((n, len(m.body_names), 6), dtype=dtype)
        shape = (n, len(m.body_names), 3)
        self._pos = np.zeros(shape, dtype=dtype)
        self._quat = np.zeros((*shape[:2], 4), dtype=dtype)
        self._quat[..., 0] = 1
        self._lin = np.zeros(shape, dtype=dtype)
        self._ang = np.zeros(shape, dtype=dtype)
        self._com = np.zeros(shape, dtype=dtype)
        self._native_link_state = np.zeros((n, 0, 16), dtype=dtype)
        self._body_sources = tuple(
            (body, int(link)) for body, link in enumerate(m.body_link_indices) if link >= 0
        )
        self._unsupported_sensors = {
            s.name: "SuperDex does not expose instantaneous point acceleration"
            for s in m.sensors
            if s.kind == "accelerometer"
        }
        self._sensor_values = {
            s.name: np.zeros((n, s.dim), dtype=dtype)
            for s in m.sensors
            if s.name not in self._unsupported_sensors
        }
        self._contact_sensors: list[SensorPlan] = []
        groups: dict[str, list[SensorPlan]] = {}
        for sensor in m.sensors:
            if sensor.name not in self._sensor_values:
                continue
            if sensor.kind in {"contact", "contact_found", "contact_force", "contact_torque"}:
                self._contact_sensors.append(sensor)
            else:
                groups.setdefault(sensor.kind, []).append(sensor)
        self._sensor_batches = []
        for kind, sensors in groups.items():
            if kind == "jointpos":
                indices = [m.joint_qpos_indices[s.joint_index] for s in sensors]
            elif kind == "jointvel":
                indices = [m.joint_qvel_indices[s.joint_index] for s in sensors]
            else:
                indices = [s.body_id for s in sensors]
            axis = {
                "framexaxis": (1.0, 0.0, 0.0),
                "frameyaxis": (0.0, 1.0, 0.0),
                "framezaxis": (0.0, 0.0, 1.0),
            }.get(kind)
            self._sensor_batches.append(
                (
                    kind,
                    tuple(self._sensor_values[s.name] for s in sensors),
                    np.asarray(indices, dtype=np.intp),
                    # Match the scalar path's np.asarray(tuple) precision. Results
                    # are still copied into the original native-precision caches.
                    np.asarray([s.local_pos for s in sensors], dtype=np.float64),
                    np.asarray([s.local_quat for s in sensors], dtype=np.float64),
                    None if axis is None else np.asarray(axis),
                )
            )
        self._all_dofs = np.arange(m.nv, dtype=np.int32)
        self._contact_sensor_index = {
            sensor.name: i for i, sensor in enumerate(self._contact_sensors)
        }
        self._native_contact = np.zeros((n, len(self._contact_sensors), 3), dtype=dtype)
        self._native_diverged = np.zeros(n, dtype=np.uint8)
        if m.actor_plans:
            body_actors = np.full(len(m.body_names), -1, dtype=np.intp)
            body_local_links = np.full(len(m.body_names), -1, dtype=np.intp)
            for slot, actor in enumerate(m.actor_plans):
                bodies = np.asarray(actor.body_ids, dtype=np.intp)
                body_actors[bodies] = slot
                if actor.local_body_link_indices.size:
                    body_local_links[bodies] = actor.local_body_link_indices
            self._body_actor_indices = body_actors
            self._body_local_link_indices = body_local_links
            self._actuator_slot_indices = np.asarray(m.actuator_slot_indices, dtype=np.intp)
            native_qpos = np.empty(self.num_actuators, dtype=np.int32)
            native_qvel = np.empty(self.num_actuators, dtype=np.int32)
            for actor in m.actor_plans:
                native_qpos[actor.actuator_indices] = actor.native_actuator_qpos_indices
                native_qvel[actor.actuator_indices] = actor.native_actuator_qvel_indices
            self._native_actuator_qpos_indices = native_qpos
            self._native_actuator_qvel_indices = native_qvel
        else:
            self._actuator_slot_indices = np.arange(self.num_actuators, dtype=np.intp)
            self._native_actuator_qpos_indices = np.asarray(
                m.actuator_qpos_indices, dtype=np.int32
            ).copy()
            if m.floating:
                self._native_actuator_qpos_indices -= 1
            self._native_actuator_qvel_indices = np.asarray(
                m.actuator_qvel_indices, dtype=np.int32
            ).copy()
        self._native_actuator_kp = np.asarray(m.actuator_kp, dtype=dtype)
        self._native_actuator_kd = np.asarray(m.actuator_kd, dtype=dtype)
        self._native_actuator_gear = np.asarray(m.actuator_gear, dtype=dtype)
        # The native step_control contract requires finite force ranges; map
        # unlimited actuators to the dtype's representable bounds.
        finite_limit = np.finfo(dtype).max
        if m.actuator_force_ranges is None:
            self._native_actuator_force_ranges = np.full(
                (self.num_actuators, 2), [-finite_limit, finite_limit], dtype=dtype
            )
        else:
            self._native_actuator_force_ranges = np.clip(
                np.asarray(m.actuator_force_ranges, dtype=dtype),
                -finite_limit,
                finite_limit,
            )
        # ABI-2 actuator vectors are actor-slot-major even though public control
        # rows remain the composed MJCF actuator order.
        self._native_slot_actuator_qpos_indices = self._native_actuator_qpos_indices[
            self._actuator_slot_indices
        ].copy()
        self._native_slot_actuator_qvel_indices = self._native_actuator_qvel_indices[
            self._actuator_slot_indices
        ].copy()
        self._native_slot_actuator_kp = self._native_actuator_kp[
            self._actuator_slot_indices
        ].copy()
        self._native_slot_actuator_kd = self._native_actuator_kd[
            self._actuator_slot_indices
        ].copy()
        self._native_slot_actuator_gear = self._native_actuator_gear[
            self._actuator_slot_indices
        ].copy()
        self._native_slot_actuator_force_ranges = self._native_actuator_force_ranges[
            self._actuator_slot_indices
        ].copy()

    def materialize(self) -> None:
        self._check_open()
        if self._actors:
            return
        m = self.model
        for i in range(self.num_envs):
            world = self._p.create_scene(f"UniSim SuperDex {i}")
            self._worlds.append(world)
            world.set_gravity(self.model.gravity)
            if m.spawn_scene is not None:
                scene_actors = m.spawn_scene(world)
                actors = scene_actors.actors
                actor_links = scene_actors.actor_links

                def cleanup_scene(
                    cleanups: tuple[Any, ...] = scene_actors.cleanups,
                ) -> None:
                    for cleanup in cleanups:
                        cleanup()

                cleanup: Any = cleanup_scene
            else:
                actor, cleanup = m.spawn_actor(world)
                actors = (actor,)
                actor_links = (
                    tuple(world.get_actor(handle) for handle in actor.get_nested_link_actors()),
                )
            self._actors.append(tuple(actors) if m.actor_plans else actors[0])
            self._native_actor_links.append(actor_links)
            self._links.append(list(chain.from_iterable(actor_links)))
            self._actor_cleanups.append(cleanup)
            if m.actor_plans:
                for actor, plan in zip(actors, m.actor_plans, strict=True):
                    if int(actor.get_num_dofs()) != plan.native_qvel_indices.size:
                        raise ValueError(
                            "SuperDex compiled actor DoF count differs from audited plan"
                        )
            elif int(actors[0].get_num_dofs()) != m.nv:
                raise ValueError("SuperDex compiled DoF count differs from audited authoring plan")
            links = self._links[i]
            actor_names: dict[str, Any] = {}
            world.for_each_actor(lambda item: actor_names.__setitem__(item.get_name(), item))
            sources: dict[str, tuple[Any, Any]] = {}
            for sensor in self.model.sensors:
                if sensor.kind not in {
                    "contact",
                    "contact_found",
                    "contact_force",
                    "contact_torque",
                }:
                    continue
                index = sensor.native_link_index
                if index < 0:
                    index = int(self.model.body_link_indices[sensor.body_id])
                if m.actor_plans:
                    source_slot = sensor.source_actor_index
                    source_link = sensor.source_link_index
                    if source_slot < 0:
                        raise ValueError("superdex contact sensor lost its source actor")
                    source = (
                        actor_links[source_slot][source_link]
                        if source_link >= 0
                        else actors[source_slot]
                    )
                    if sensor.other_actor_index < 0:
                        raise ValueError("superdex contact sensor lost its other actor")
                    other_slot = sensor.other_actor_index
                    other_link = sensor.other_link_index
                    other = (
                        actor_links[other_slot][other_link]
                        if other_link >= 0
                        else actors[other_slot]
                    )
                else:
                    source = links[index]
                    other = (
                        actor_names[sensor.other_actor_name]
                        if sensor.other_actor_name
                        else None
                    )
                query = (
                    self._p.QueryType.CONTACT_POINTS
                    if sensor.kind in {"contact", "contact_found"}
                    else self._p.QueryType.TOTAL_CONTACT_FORCE
                )
                source.register_query(query)
                sources[sensor.name] = source, other
            self._sensor_sources.append(sources)
            if m.actor_plans:
                for plan in m.actor_plans:
                    if plan.native_qvel_indices.size:
                        continue
                    root = plan.root_body_id
                    self._pos[:, root] = m.body_pos[root]
                    self._quat[:, root] = m.body_quat[root]
            self._snapshots.append(world.capture_state())
        if self._links:
            link_count = len(self._links[0])
            if any(len(links) != link_count for links in self._links):
                raise ValueError("SuperDex articulated actors must have equal link counts")
            self._native_link_state = np.zeros((self.num_envs, link_count, 16), dtype=self._dtype)
            if m.actor_plans and any(
                tuple(actor.get_num_dofs() for actor in actors) != first_dofs
                for actors, first_dofs in (
                    (self._actors[i], tuple(actor.get_num_dofs() for actor in self._actors[0]))
                    for i in range(self.num_envs)
                )
            ):
                raise ValueError("SuperDex scenes must use the same actor slot DoF layout")
        if self._execution_mode == "serial":
            self.reset()
            return
        self._reject_if_debugger_attached()
        contact_sources = [
            [self._sensor_sources[i][sensor.name][0] for sensor in self._contact_sensors]
            for i in range(self.num_envs)
        ]
        contact_others = [
            [self._sensor_sources[i][sensor.name][1] for sensor in self._contact_sensors]
            for i in range(self.num_envs)
        ]
        contact_kind_codes = {
            "contact_found": 0,
            "contact": 0,
            "contact_force": 1,
            "contact_torque": 2,
        }
        contact_kinds = [contact_kind_codes[sensor.kind] for sensor in self._contact_sensors]
        contact_distances = [sensor.contact_distance for sensor in self._contact_sensors]
        if m.actor_plans:
            executor_cls = getattr(self._p, "SceneBatchExecutorV2", None)
            if (
                executor_cls is None
                or int(getattr(self._p, "SCENE_BATCH_EXECUTOR_ABI_VERSION", 0)) != 2
            ):
                raise RuntimeError(
                    "superdex portable scenes require SceneBatchExecutorV2 ABI 2 "
                    "(superdex-uni 1.1.0)"
                )
            actuator_counts = [len(actor.actuator_indices) for actor in m.actor_plans]
            self._batch_executor = executor_cls(
                self._worlds,
                self._actors,
                self._native_actor_links,
                actuator_counts,
                contact_sources,
                contact_others,
                contact_kinds,
                contact_distances,
                num_workers=self._batch_num_workers,
            )
        else:
            executor_cls = getattr(self._p, "SceneBatchExecutor", None)
            if executor_cls is None or not hasattr(executor_cls, "num_links"):
                raise RuntimeError(
                    "superdex requires the local project_superdex build with the extended "
                    "SceneBatchExecutor state/contact API"
                )
            self._batch_executor = executor_cls(
                self._worlds,
                self._actors,
                self._links,
                contact_sources,
                contact_others,
                contact_kinds,
                contact_distances,
                num_workers=self._batch_num_workers,
            )
        self.reset()

    def _ids(self, value: np.ndarray) -> np.ndarray:
        ids = np.asarray(value)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("env_indices must be a one-dimensional integer array")
        if np.any(ids < 0) or np.any(ids >= self.num_envs) or np.unique(ids).size != ids.size:
            raise ValueError("env_indices must contain unique in-range indices")
        return ids.astype(np.intp, copy=False)

    def _validate_portable_quaternions(self, qpos: np.ndarray) -> None:
        m = self.model
        assert m.layout is not None
        for actor in m.actor_plans:
            if not actor.floating:
                continue
            entity = m.layout.get_entity(actor.entity_name or "")
            columns = np.asarray(entity.root_qpos_indices, dtype=np.intp)[3:7]
            quaternion = qpos[:, columns]
            if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5):
                raise ValueError("floating entity quaternions must be normalized wxyz")

    def _public_to_native_state(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        assert m.layout is not None
        native_q = np.zeros((qpos.shape[0], m.nv), dtype=self._dtype)
        native_v = np.zeros_like(native_q)
        for actor in m.actor_plans:
            native_qcols = actor.native_qpos_indices
            native_vcols = actor.native_qvel_indices
            if actor.floating:
                qcols = np.asarray(actor.native_order_qpos_indices, dtype=np.intp)
                vcols = np.asarray(actor.native_order_qvel_indices, dtype=np.intp)
                native_q[:, native_qcols[:3]] = qpos[:, qcols[:3]]
                native_q[:, native_qcols[6:]] = qpos[:, qcols[7:]]
                native_v[:, native_vcols[:3]] = qvel[:, vcols[:3]]
                native_v[:, native_vcols[6:]] = qvel[:, vcols[6:]]
                for row in range(qpos.shape[0]):
                    native_q[row, native_qcols[3:6]] = np.asarray(
                        self._p.Quaternion(qpos[row, qcols[[4, 5, 6, 3]]]).to_rotation_vector()
                    )
                    quaternion = qpos[row, qcols[3:7]]
                    native_v[row, native_vcols[3:6]] = rotate(
                        quaternion[None, :], qvel[row, vcols[3:6]][None, :]
                    )[0]
            else:
                native_q[:, native_qcols] = qpos[:, actor.native_order_qpos_indices]
                native_v[:, native_vcols] = qvel[:, actor.native_order_qvel_indices]
        return native_q, native_v

    def _native_to_public_state(self, rows: np.ndarray) -> None:
        m = self.model
        assert m.layout is not None
        for actor in m.actor_plans:
            entity = m.layout.get_entity(actor.entity_name or "")
            native_q = self._native_q[rows][:, actor.native_qpos_indices]
            native_v = self._native_v[rows][:, actor.native_qvel_indices]
            if actor.floating:
                root_q = np.asarray(entity.root_qpos_indices, dtype=np.intp)
                root_v = np.asarray(entity.root_qvel_indices, dtype=np.intp)
                self._qpos[np.ix_(rows, root_q[:3])] = native_q[:, :3]
                angle = np.linalg.norm(native_q[:, 3:6], axis=1)
                half_angle = 0.5 * angle
                scale = np.empty_like(angle)
                small = angle < 1e-6
                scale[small] = 0.5 - angle[small] ** 2 / 48.0
                scale[~small] = np.sin(half_angle[~small]) / angle[~small]
                self._qpos[rows, root_q[3]] = np.cos(half_angle)
                self._qpos[np.ix_(rows, root_q[4:7])] = native_q[:, 3:6] * scale[:, None]
                self._qpos[np.ix_(rows, actor.native_order_qpos_indices[7:])] = native_q[:, 6:]
                self._qvel[np.ix_(rows, root_v[:3])] = native_v[:, :3]
                self._qvel[np.ix_(rows, actor.native_order_qvel_indices[6:])] = native_v[:, 6:]
                world_angular = native_v[:, 3:6]
                quaternion = self._qpos[np.ix_(rows, root_q[3:7])]
                self._qvel[np.ix_(rows, root_v[3:6])] = unrotate(quaternion, world_angular)
            else:
                qcols = np.asarray(actor.native_order_qpos_indices, dtype=np.intp)
                vcols = np.asarray(actor.native_order_qvel_indices, dtype=np.intp)
                self._qpos[np.ix_(rows, qcols)] = native_q
                self._qvel[np.ix_(rows, vcols)] = native_v

    def _write_serial_native_state(
        self, i: int, native_q: np.ndarray, native_v: np.ndarray
    ) -> None:
        for actor, plan in zip(self._actors[i], self.model.actor_plans, strict=True):
            if not plan.native_qvel_indices.size:
                continue
            columns = plan.native_qvel_indices
            actor.set_articulated_pose_from_joints(native_q[columns])
            actor.set_articulated_joint_velocities(native_v[columns])
            actor.set_external_forces_on_dofs(
                np.arange(columns.size, dtype=np.int32), np.zeros(columns.size, native_q.dtype)
            )

    def _commit_portable_state(
        self,
        rows: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        control_indices: np.ndarray | None = None,
        control_values: np.ndarray | None = None,
    ) -> None:
        """Restore selected worlds, then submit their complete public state."""
        native_q, native_v = self._public_to_native_state(qpos, qvel)
        try:
            for row, i in enumerate(rows):
                self._worlds[i].restore_state(self._snapshots[i], release_immediately=False)
                if self._batch_executor is None:
                    self._write_serial_native_state(i, native_q[row], native_v[row])
            if self._batch_executor is not None:
                qpos_mask = np.zeros(native_q.shape, dtype=np.uint8)
                qvel_mask = np.zeros_like(qpos_mask)
                qpos_mask[rows] = 1
                qvel_mask[rows] = 1
                self._batch_executor.write_state(native_q, native_v, qpos_mask, qvel_mask)
            for row, i in enumerate(rows):
                if self._batch_executor is None:
                    self._worlds[i].step(0)
                self._native_q[i] = native_q[row]
                self._native_v[i] = native_v[row]
                self._qpos[i] = qpos[row]
                self._qvel[i] = qvel[row]
                self._pending_wrench[i] = 0
            if control_indices is None:
                self._ctrl[rows] = 0
            elif control_values is None:
                self._ctrl[np.ix_(rows, control_indices)] = 0
            else:
                self._ctrl[np.ix_(rows, control_indices)] = control_values
        except BaseException:
            self._entity_faulted = True
            raise

    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> None:
        self._check_open()
        ids = self._ids(env_indices)
        q = np.asarray(qpos, dtype=self._dtype)
        v = np.asarray(qvel, dtype=self._dtype)
        if q.shape != (len(ids), self.model.nq) or v.shape != (len(ids), self.model.nv):
            raise ValueError(
                "set_state qpos/qvel shapes must match selected rows and model dimensions"
            )
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("set_state requires finite qpos/qvel")
        if randomization is not None and randomization.requested_terms():
            raise NotImplementedError("superdex does not support reset model randomization")
        if self.model.actor_plans:
            self._validate_portable_quaternions(q)
            full_q = self._qpos.copy()
            full_v = self._qvel.copy()
            full_q[ids] = q
            full_v[ids] = v
            self._commit_portable_state(ids, full_q, full_v)
            self._refresh(ids, refresh_contacts=False)
            return
        if self.model.floating and not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5):
            raise ValueError("free-root quaternion must be normalized wxyz")
        for row, i in enumerate(ids):
            world, actor = self._worlds[i], self._actors[i]
            world.restore_state(self._snapshots[i], release_immediately=False)
            native_q, native_v = self._native_q[i], self._native_v[i]
            if self.model.floating:
                native_q[:3] = q[row, :3]
                native_q[3:6] = np.asarray(
                    self._p.Quaternion(q[row, [4, 5, 6, 3]]).to_rotation_vector()
                )
                native_q[6:] = q[row, 7:]
                native_v[:3] = v[row, :3]
                native_v[3:6] = rotate(q[row, 3:7], v[row, 3:6])
                native_v[6:] = v[row, 6:]
            else:
                native_q[:] = q[row]
                native_v[:] = v[row]
            actor.set_articulated_pose_from_joints(native_q)
            actor.set_articulated_joint_velocities(native_v)
            actor.set_external_forces_on_dofs(self._all_dofs, np.zeros_like(native_v))
            self._ctrl[i] = 0
            self._pending_wrench[i] = 0
            world.step(0)
        self._refresh(ids)

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        self._check_open()
        values = np.asarray(ctrl, dtype=self._dtype)
        if values.shape != self._ctrl.shape or not np.isfinite(values).all():
            raise ValueError(f"ctrl must be finite with shape {self._ctrl.shape}")
        if isinstance(nsteps, bool) or not isinstance(nsteps, (int, np.integer)) or nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        if self._execution_mode == "serial":
            self._step_serial(values, nsteps)
            return
        self._reject_if_debugger_attached()
        m = self.model
        if nsteps > 1 and self._pre_step_control_fn is None and not self._pending_wrench.any():
            self._ctrl[:] = np.clip(
                values, m.actuator_ctrl_ranges[:, 0], m.actuator_ctrl_ranges[:, 1]
            )
            assert self._batch_executor is not None
            if m.actor_plans:
                self._batch_ctrl[:] = self._ctrl[:, self._actuator_slot_indices]
                controls = self._batch_ctrl
                qpos_indices = self._native_slot_actuator_qpos_indices
                qvel_indices = self._native_slot_actuator_qvel_indices
                kp = self._native_slot_actuator_kp
                kd = self._native_slot_actuator_kd
                gear = self._native_slot_actuator_gear
                force_ranges = self._native_slot_actuator_force_ranges
            else:
                controls = self._ctrl
                qpos_indices = self._native_actuator_qpos_indices
                qvel_indices = self._native_actuator_qvel_indices
                kp = self._native_actuator_kp
                kd = self._native_actuator_kd
                gear = self._native_actuator_gear
                force_ranges = self._native_actuator_force_ranges
            self._batch_executor.step_control(
                self._dt,
                controls,
                qpos_indices,
                qvel_indices,
                kp,
                kd,
                gear,
                force_ranges,
                int(nsteps),
                self._native_q,
                self._native_v,
                self._native_link_state,
                self._native_contact,
                self._native_diverged,
                31,
            )
            diverged = np.flatnonzero(self._native_diverged)
            if diverged.size:
                raise RuntimeError(f"SuperDex solver diverged in environment {int(diverged[0])}")
            self._refresh(self._env_ids, native_state_ready=True)
            return
        for substep in range(nsteps):
            full_readback = substep == nsteps - 1
            converted = self._apply_pre_step_control(values)
            if not np.isfinite(converted).all():
                raise ValueError("pre-step control returned non-finite values")
            self._ctrl[:] = np.clip(
                converted, m.actuator_ctrl_ranges[:, 0], m.actuator_ctrl_ranges[:, 1]
            )
            q = self._qpos[:, m.actuator_qpos_indices]
            v = self._qvel[:, m.actuator_qvel_indices]
            # kp==0 denotes a direct motor; otherwise ctrl is a position target.
            force = np.where(
                m.actuator_kp > 0,
                m.actuator_kp * (self._ctrl - q * m.actuator_gear)
                - m.actuator_kd * v * m.actuator_gear,
                self._ctrl,
            )
            if m.actuator_force_ranges is not None:
                force = np.clip(
                    force, m.actuator_force_ranges[:, 0], m.actuator_force_ranges[:, 1]
                )
            force = force * m.actuator_gear
            self._batch_forces.fill(0)
            if self._pending_wrench.any():
                active_envs = np.flatnonzero(np.any(self._pending_wrench != 0, axis=(1, 2)))
                for i in active_envs:
                    generalized = self._batch_forces[i]
                    for body in np.flatnonzero(np.any(self._pending_wrench[i] != 0, axis=1)):
                        link = self._links[i][m.body_link_indices[body]]
                        if m.actor_plans:
                            plan = m.actor_plans[self._body_actor_indices[body]]
                            local_columns = plan.native_qvel_indices
                        else:
                            local_columns = self._all_dofs
                        jacobian = np.asarray(link.get_articulated_jacobian()).reshape(
                            6, local_columns.size
                        )
                        generalized[local_columns] += (
                            jacobian.T @ self._pending_wrench[i, body]
                        )
            if m.actor_plans:
                np.add.at(
                    self._batch_forces,
                    (self._env_ids[:, None], self._native_actuator_qvel_indices[None, :]),
                    force,
                )
            else:
                np.add.at(
                    self._batch_forces,
                    (self._env_ids[:, None], m.actuator_qvel_indices[None, :]),
                    force,
                )
            assert self._batch_executor is not None
            self._batch_executor.step(
                self._dt,
                self._batch_forces,
                self._native_q,
                self._native_v,
                self._native_link_state if full_readback else None,
                self._native_contact if full_readback else None,
                self._native_diverged,
                31 if full_readback else 19,
            )
            diverged = np.flatnonzero(self._native_diverged)
            if diverged.size:
                raise RuntimeError(f"SuperDex solver diverged in environment {int(diverged[0])}")
            self._refresh(
                self._env_ids,
                native_state_ready=True,
                full_state_ready=full_readback,
            )
        self._pending_wrench.fill(0)

    def _reject_if_debugger_attached(self) -> None:
        """Fail closed when a native debugger session meets executor-owned scenes.

        The batch executor steps scenes on persistent worker threads, while a
        connected SuperDex debugger gathers DebugDraw data from the scene's step
        thread; DebugDraw is thread-affine, so the combination traps natively.
        """
        get_server = getattr(self._p, "get_debug_server", None)
        if get_server is None or not get_server().has_connection():
            return
        raise RuntimeError(
            "superdex execution_mode='batch' is incompatible with an attached native "
            "debugger: SceneBatchExecutor steps scenes on worker threads, which "
            "violates the scene's DebugDraw thread affinity. Re-create the backend "
            "with execution_mode='serial' (UniLab: "
            "env.superdex_execution_mode=serial) before attaching the debugger."
        )

    def _step_serial(self, values: np.ndarray, nsteps: int) -> None:
        """Advance every scene on the environment thread (native-debugger safe)."""
        m = self.model
        for _ in range(nsteps):
            converted = self._apply_pre_step_control(values)
            if not np.isfinite(converted).all():
                raise ValueError("pre-step control returned non-finite values")
            self._ctrl[:] = np.clip(
                converted, m.actuator_ctrl_ranges[:, 0], m.actuator_ctrl_ranges[:, 1]
            )
            q = self._qpos[:, m.actuator_qpos_indices]
            v = self._qvel[:, m.actuator_qvel_indices]
            # kp==0 denotes a direct motor; otherwise ctrl is a position target.
            force = np.where(
                m.actuator_kp > 0,
                m.actuator_kp * (self._ctrl - q * m.actuator_gear)
                - m.actuator_kd * v * m.actuator_gear,
                self._ctrl,
            )
            if m.actuator_force_ranges is not None:
                force = np.clip(force, m.actuator_force_ranges[:, 0], m.actuator_force_ranges[:, 1])
            force = force * m.actuator_gear
            for i, world in enumerate(self._worlds):
                generalized: np.ndarray = np.zeros(m.nv, dtype=self._dtype)
                for body in np.flatnonzero(np.any(self._pending_wrench[i] != 0, axis=1)):
                    link = self._links[i][m.body_link_indices[body]]
                    if m.actor_plans:
                        plan = m.actor_plans[self._body_actor_indices[body]]
                        local_columns = plan.native_qvel_indices
                    else:
                        local_columns = self._all_dofs
                    jacobian = np.asarray(link.get_articulated_jacobian()).reshape(
                        6, local_columns.size
                    )
                    generalized[local_columns] += (
                        jacobian.T @ self._pending_wrench[i, body]
                    )
                if m.actor_plans:
                    np.add.at(
                        generalized,
                        self._native_actuator_qvel_indices,
                        force[i],
                    )
                    for actor, plan in zip(self._actors[i], m.actor_plans, strict=True):
                        columns = plan.native_qvel_indices
                        if not columns.size:
                            continue
                        actor.set_external_forces_on_dofs(
                            np.arange(columns.size, dtype=np.int32), generalized[columns]
                        )
                else:
                    np.add.at(generalized, m.actuator_qvel_indices, force[i])
                    self._actors[i].set_external_forces_on_dofs(self._all_dofs, generalized)
                world.step(self._dt)
                if (
                    world.get_solver_stats().convergence_status
                    == self._p.ConvergenceStatus.DIVERGED
                ):
                    raise RuntimeError(f"SuperDex solver diverged in environment {i}")
            self._refresh(self._env_ids)
        self._pending_wrench.fill(0)

    def _refresh(
        self,
        ids: np.ndarray,
        *,
        native_state_ready: bool = False,
        full_state_ready: bool = True,
        refresh_contacts: bool = True,
    ) -> None:
        m = self.model
        if not ids.size:
            return
        if native_state_ready:
            if m.actor_plans:
                self._native_to_public_state(ids)
            elif m.floating:
                native_rot = self._native_q[ids, 3:6]
                angle = np.linalg.norm(native_rot, axis=1)
                half_angle = 0.5 * angle
                scale = np.empty_like(angle)
                small = angle < 1e-6
                scale[small] = 0.5 - angle[small] ** 2 / 48.0
                scale[~small] = np.sin(half_angle[~small]) / angle[~small]
                self._qpos[ids, :3] = self._native_q[ids, :3]
                self._qpos[ids, 3] = np.cos(half_angle)
                self._qpos[ids, 4:7] = native_rot * scale[:, None]
                self._qpos[ids, 7:] = self._native_q[ids, 6:]
                self._qvel[ids, :3] = self._native_v[ids, :3]
                self._qvel[ids, 6:] = self._native_v[ids, 6:]
            else:
                self._qpos[ids] = self._native_q[ids]
                self._qvel[ids] = self._native_v[ids]
            if full_state_ready:
                for body_id, link_index in self._body_sources:
                    state = self._native_link_state[ids, link_index]
                    self._pos[ids, body_id] = state[:, :3]
                    self._quat[ids, body_id] = state[:, 3:7]
                    self._com[ids, body_id] = state[:, 7:10]
                    self._lin[ids, body_id] = state[:, 10:13]
                    self._ang[ids, body_id] = state[:, 13:16]
        else:
            for i in ids:
                if m.actor_plans:
                    for actor, plan in zip(self._actors[i], m.actor_plans, strict=True):
                        columns = plan.native_qvel_indices
                        if not columns.size:
                            continue
                        local_q = np.empty(columns.size, dtype=self._dtype)
                        local_v = np.empty_like(local_q)
                        actor.get_articulated_pose(local_q)
                        actor.get_articulated_joint_velocities(local_v)
                        self._native_q[i, columns] = local_q
                        self._native_v[i, columns] = local_v
                    self._native_to_public_state(np.asarray([i], dtype=np.intp))
                else:
                    actor = self._actors[i]
                    actor.get_articulated_pose(self._native_q[i])
                    actor.get_articulated_joint_velocities(self._native_v[i])
                    if m.floating:
                        self._qpos[i, :3] = self._native_q[i, :3]
                        quat = self._p.Quaternion.from_rotation_vector(self._native_q[i, 3:6])
                        self._qpos[i, 3:7] = np.asarray(quat)[[3, 0, 1, 2]]
                        self._qpos[i, 7:] = self._native_q[i, 6:]
                        self._qvel[i, :3] = self._native_v[i, :3]
                        self._qvel[i, 6:] = self._native_v[i, 6:]
                    else:
                        self._qpos[i] = self._native_q[i]
                        self._qvel[i] = self._native_v[i]
                if full_state_ready:
                    for body_id, link_index in self._body_sources:
                        link = self._links[i][link_index]
                        pose = link.get_root_transform()
                        self._pos[i, body_id] = np.asarray(pose.translation)
                        self._quat[i, body_id] = np.asarray(pose.rotation)[[3, 0, 1, 2]]
                        self._ang[i, body_id] = np.asarray(link.get_angular_velocity())
                        self._com[i, body_id] = np.asarray(
                            link.get_center_of_mass_transform().translation
                        )
                        self._lin[i, body_id] = np.asarray(link.get_linear_velocity())
        if m.floating and not m.actor_plans:
            self._qvel[ids, 3:6] = unrotate(self._qpos[ids, 3:7], self._native_v[ids, 3:6])
        if full_state_ready:
            self._lin[ids] -= np.cross(self._ang[ids], self._com[ids] - self._pos[ids])
            self._refresh_sensor_batches(ids)
            if native_state_ready:
                for sensor in self._contact_sensors:
                    index = self._contact_sensor_index[sensor.name]
                    self._sensor_values[sensor.name][ids] = self._native_contact[
                        ids, index, : sensor.dim
                    ]
            elif refresh_contacts:
                for i in ids:
                    for sensor in self._contact_sensors:
                        self._sensor_values[sensor.name][i] = self._read_sensor(i, sensor)
            else:
                for sensor in self._contact_sensors:
                    self._sensor_values[sensor.name][ids] = 0
        arrays = (self._qpos[ids], self._qvel[ids], self._pos[ids], self._lin[ids], self._ang[ids])
        if any(not np.isfinite(a).all() for a in arrays):
            raise RuntimeError("SuperDex returned non-finite physics state")

    def _refresh_sensor_batches(self, ids: np.ndarray) -> None:
        """Transform each sensor kind across selected rows in a single NumPy batch."""
        rows = ids[:, None]
        for kind, destinations, indices, local_pos, local_quat, axis in self._sensor_batches:
            if kind in {"jointpos", "jointvel"}:
                source = self._qpos if kind == "jointpos" else self._qvel
                values = source[rows, indices, None]
            else:
                body_quat = self._quat[rows, indices]
                if kind == "frameangvel":
                    values = self._ang[rows, indices]
                elif kind == "framequat":
                    values = multiply(body_quat, local_quat)
                elif axis is not None:
                    values = rotate(multiply(body_quat, local_quat), axis)
                elif kind == "gyro":
                    values = unrotate(multiply(body_quat, local_quat), self._ang[rows, indices])
                elif kind in {"framepos", "velocimeter", "framelinvel"}:
                    offset = rotate(body_quat, local_pos)
                    if kind == "framepos":
                        values = self._pos[rows, indices] + offset
                    else:
                        values = self._lin[rows, indices] + np.cross(
                            self._ang[rows, indices], offset
                        )
                        if kind == "velocimeter":
                            values = unrotate(multiply(body_quat, local_quat), values)
                else:
                    raise NotImplementedError(f"superdex sensor kind is unsupported: {kind}")
            for column, destination in enumerate(destinations):
                destination[ids] = values[:, column]

    def _read_sensor(self, i: int, sensor: SensorPlan) -> np.ndarray:
        """Read a native contact query without unnecessary frame transformations."""
        kind = sensor.kind
        if kind in {"contact", "contact_found", "contact_force", "contact_torque"}:
            source, other = self._sensor_sources[i][sensor.name]
            if kind == "contact_force":
                return np.asarray(source.get_contact_force_world())
            if kind == "contact_torque":
                return np.asarray(source.get_contact_torque_world())
            points = source.get_contact_points_world()
            if other is not None:
                # Pair filtering uses captured native actor handles, never name lookup.
                handle = other.get_handle()
                own = source.get_handle()
                found = any(
                    point.distance <= sensor.contact_distance
                    and (
                        (point.actor_a == own and point.actor_b == handle)
                        or (point.actor_a == handle and point.actor_b == own)
                    )
                    for point in points
                )
            else:
                found = any(point.distance <= sensor.contact_distance for point in points)
            return np.array([found], dtype=self._dtype)
        raise NotImplementedError(f"superdex sensor kind is unsupported: {kind}")

    def get_state(self, fields: Any = None) -> dict[str, np.ndarray]:
        self._check_open()
        names = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else fields)
        )
        values = {"qpos": self._qpos, "qvel": self._qvel, "ctrl": self._ctrl}
        return {name: values[name].copy() for name in names}

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self.model.actuator_ctrl_ranges.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self.model.actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        return self.model.actuator_joint_names

    def get_scene_model_file(self) -> str:
        return self.model.source_file

    def get_default_qpos(self) -> np.ndarray:
        return self.model.default_qpos.copy()

    def get_default_dof_pos(self) -> np.ndarray:
        return self.model.default_qpos[self.model.joint_qpos_indices].copy()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        return self.model.keyframes[name].copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(self.model.nv, dtype=self._dtype)

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray:
        self._reject_named_joint_ranges(names, "joint ranges")
        return self.model.joint_ranges.copy()

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self._body_lookup[name] for name in names], dtype=np.int32)
        except KeyError as exc:
            raise ValueError(f"superdex unknown body: {exc.args[0]}") from exc

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        found = {int(root_body_id)}
        for body, parent in enumerate(self.model.body_parent_ids):
            if body != parent and parent in found:
                found.add(body)
        return np.array(sorted(found), dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self._joint_lookup[name] for name in names], dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_dof_pos_indices(names)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.get_joint_state_qvel_indices(names)

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.model.joint_qpos_indices[self.get_joint_dof_pos_indices(names)].copy()

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        return self.model.joint_qvel_indices[self.get_joint_dof_vel_indices(names)].copy()

    def get_scene_layout(self):
        self._check_open()
        if self.model.layout is None:
            return super().get_scene_layout()
        return self.model.layout

    def get_entity_names(self) -> tuple[str, ...]:
        return tuple(entity.name for entity in self.get_scene_layout().entities)

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ):
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        ids = selected_state_rows(env_ids, self.num_envs)
        qpos = np.broadcast_to(self.model.default_qpos, (ids.size, self.model.nq))
        qvel = np.zeros((ids.size, self.model.nv), dtype=self._dtype)
        roots = np.zeros((ids.size, 13), dtype=self._dtype)
        root_body = owner.body_ids[0]
        roots[:, :3] = self.model.body_pos[root_body]
        roots[:, 3:7] = self.model.body_quat[root_body]
        return entity_state_snapshot(owner, qpos, qvel, roots)

    def _entity_roots(self) -> np.ndarray:
        layout = self.get_scene_layout()
        roots = np.zeros((self.num_envs, len(layout.entities), 13), dtype=self._dtype)
        for index, entity in enumerate(layout.entities):
            if entity.root_mode == "floating":
                state = entity_state_snapshot(entity, self._qpos, self._qvel)
                roots[:, index, :7] = state["root_pose"]
                roots[:, index, 7:] = state["root_velocity"]
            else:
                body = entity.body_ids[0]
                roots[:, index, :3] = self._pos[:, body]
                roots[:, index, 3:7] = self._quat[:, body]
        return roots

    def get_entity_state(self, entity: str):
        layout = self.get_scene_layout()
        owner = layout.get_entity(entity)
        if owner.root_mode == "floating":
            return entity_state_snapshot(owner, self._qpos, self._qvel)
        root = np.zeros((self.num_envs, 13), dtype=self._dtype)
        body = owner.body_ids[0]
        root[:, :3] = self._pos[:, body]
        root[:, 3:7] = self._quat[:, body]
        return entity_state_snapshot(owner, self._qpos, self._qvel, root)

    @staticmethod
    def _portable_reset_control_columns(binding: Any) -> tuple[int, ...]:
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

    def reset_entities(self, request: SceneResetRequest) -> None:
        if not self.model.actor_plans:
            super().reset_entities(request)
            return
        self._check_open()
        layout = self.get_scene_layout()
        prepared = prepare_scene_reset(
            layout, request, self._qpos, self._qvel, self._entity_roots()
        )
        rows = prepared.env_ids
        qpos = self._qpos.copy()
        qvel = self._qvel.copy()
        qcols = np.flatnonzero(prepared.qpos_mask)
        vcols = np.flatnonzero(prepared.qvel_mask)
        qpos[np.ix_(rows, qcols)] = prepared.qpos[:, qcols]
        qvel[np.ix_(rows, vcols)] = prepared.qvel[:, vcols]
        columns = np.asarray(
            self._portable_reset_control_columns(prepared.binding), dtype=np.intp
        )
        control_values = None
        if columns.size:
            target = (
                self._default_ctrl
                if request.restore_default_controls
                else np.zeros_like(self._default_ctrl)
            )
            control_values = np.broadcast_to(target[columns], (rows.size, columns.size)).copy()
        self._commit_portable_state(
            rows, qpos, qvel, control_indices=columns, control_values=control_values
        )
        self._refresh(rows, refresh_contacts=False)

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
        if self.model.actor_plans:
            assert self.model.layout is not None
            entity_name, separator, local_name = str(root_body_name).partition("/")
            if not separator:
                raise ValueError("portable SuperDex root names use entity/local_name")
            entity = self.model.layout.get_entity(entity_name)
            if local_name != entity.root_body:
                raise ValueError(
                    f"root {root_body_name!r} is not entity {entity.name!r}'s root body"
                )
            if entity.root_mode != "floating":
                raise NotImplementedError(
                    f"portable SuperDex entity {entity.name!r} has no floating root state"
                )
            return BackendRootStateLayout(
                tuple(entity.root_qpos_indices), tuple(entity.root_qvel_indices)
            )
        body = self.get_body_id(root_body_name)
        if not self.model.floating or body != self.model.root_body_id:
            raise NotImplementedError(
                f"superdex body {root_body_name!r} does not own a floating root"
            )
        return BackendRootStateLayout(
            qpos_indices=(0, 1, 2, 3, 4, 5, 6),
            qvel_indices=(0, 1, 2, 3, 4, 5),
        )

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities()

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        return BackendPlayCapabilities(
            supports_physics_state_playback=True,
            supports_debug_overlay=True,
            # The native Polyscope viewer reads the scene on the calling
            # thread, so interactive rendering only exists when serial mode
            # keeps stepping on that same thread. getattr keeps capability
            # queries working on skeleton instances that never ran __init__.
            supports_native_interactive_renderer=getattr(self, "_execution_mode", "batch")
            == "serial",
        )

    # Static so the plan resolves without a backend instance (class-level
    # calls in tests); instance calls keep working.  The base declares an
    # instance method, hence the override ignores.
    @staticmethod
    def resolve_play_render_plan(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Resolve playback: native interactive viewer or shared MuJoCo recorder."""
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none", headless=True, record_video=False, num_steps=None, output_video=None
            )
        if mode == "interactive":
            # Serial-mode enforcement happens in run_playback, where the
            # instance (and its execution mode) is available.
            return BackendPlayRenderPlan(
                mode="interactive",
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
            )
        if play_steps is None or isinstance(play_steps, bool) or int(play_steps) <= 0:
            raise ValueError("superdex MuJoCo playback requires positive training.play_steps")
        if output_video is None:
            raise ValueError("superdex MuJoCo playback requires an output video path")
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
        env,
        initialize,
        step,
        num_steps,
        output_video=None,
        render_spacing=None,
        render_offset_mode=None,
        headless=None,
        record_video=None,
        frame_state_getter=None,
        camera_kwargs=None,
        debug_overlay_getter=None,
        on_frame=None,
    ):
        should_record = bool(record_video) if record_video is not None else output_video is not None
        if not should_record:
            return self._run_interactive_playback(
                env=env,
                initialize=initialize,
                step=step,
                num_steps=num_steps,
                offscreen=bool(headless),
                debug_overlay_getter=debug_overlay_getter,
                on_frame=on_frame,
            )
        from unisim.backend.playback_common import run_offline_snapshot_playback

        if self.scene_visual_model_file is None:
            raise RuntimeError(
                "superdex MuJoCo playback requires scene.visual_model_file for .superdex_bot assets"
            )
        return run_offline_snapshot_playback(
            backend=self,
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            headless=True if headless is None else bool(headless),
            record_video=should_record,
            snapshot_shape=(self.num_envs, 1 + self.model.nq + self.model.nv),
            frame_state_getter=frame_state_getter,
            camera_kwargs=CameraCfg.from_kwargs(camera_kwargs),
            backend_label="superdex",
            debug_overlay_getter=debug_overlay_getter,
            on_frame=on_frame,
        )

    def _run_interactive_playback(
        self, *, env, initialize, step, num_steps, offscreen, debug_overlay_getter, on_frame
    ):
        """Drive the native Polyscope viewer on the single serial-mode scene."""
        if debug_overlay_getter is not None:
            raise NotImplementedError(
                "superdex native interactive rendering does not support debug overlays"
            )
        if on_frame is not None:
            raise NotImplementedError(
                "superdex native interactive rendering does not support on_frame callbacks"
            )
        if self._execution_mode != "serial":
            raise RuntimeError(
                "superdex native interactive rendering requires execution_mode='serial' "
                "(UniLab: env.superdex_execution_mode=serial, injected automatically for "
                "interactive eval): the viewer shares the scene's thread with stepping, "
                "which batch mode runs on SceneBatchExecutor workers"
            )
        if self.num_envs != 1:
            raise ValueError(
                "superdex native interactive rendering requires num_envs=1 "
                "(UniLab interactive eval forces training.play_env_num=1)"
            )
        from superdex.physics.viewer import VIEWER_AVAILABLE, Viewer, ViewerCfg

        if not VIEWER_AVAILABLE:
            raise RuntimeError(
                "superdex native interactive rendering requires Polyscope >= 2.5.0"
            )
        from unisim.backend.playback_common import env_cfg_value

        viewer_cfg = ViewerCfg()
        viewer_cfg.offscreen = offscreen
        viewer = Viewer(viewer_cfg)
        viewer.set_scene(self._worlds[0])
        # Polyscope's camera view matrix is uninitialized (NaN) until the first
        # explicit camera placement, and the viewer's navigation gizmo reads it
        # while building the first ImGui frame. Frame the scene up front so the
        # first frame_tick sees a finite camera.
        viewer.frame_scene()
        ctrl_dt = float(env_cfg_value(env, "ctrl_dt", 1.0 / 60.0))
        obs = initialize()
        steps = 0
        try:
            while num_steps is None or steps < num_steps:
                started = time.perf_counter()
                obs = step(obs)
                viewer.render()
                if viewer.user_requested_close():
                    break
                elapsed = time.perf_counter() - started
                if elapsed < ctrl_dt:
                    time.sleep(ctrl_dt - elapsed)
                steps += 1
        finally:
            viewer.close()
        return None

    def get_physics_state(self) -> np.ndarray:
        state = np.empty((self.num_envs, 1 + self.model.nq + self.model.nv), dtype=self._dtype)
        state[:, 0] = 0.0
        state[:, 1 : 1 + self.model.nq] = self._qpos
        state[:, 1 + self.model.nq :] = self._qvel
        return state

    def get_playback_model(self, env_index: int | None = None):
        del env_index
        if self.scene_visual_model_file is None:
            raise RuntimeError(
                "superdex MuJoCo playback requires scene.visual_model_file for .superdex_bot assets"
            )
        return self.scene_visual_model_file

    def get_scene_visual_model_file(self) -> str | None:
        return self.scene_visual_model_file

    def get_gravity(self) -> np.ndarray:
        return self.model.gravity.copy()

    def get_body_mass(self) -> np.ndarray:
        return self.model.body_mass.copy()

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if env_ids is not None:
            raise NotImplementedError(
                "SuperDexBackend does not expose per-environment body ipos"
            )
        return self.model.body_ipos.copy()

    def get_dof_armature(self) -> np.ndarray:
        if self.model.dof_armature is None:
            return np.zeros(self.model.nv, dtype=self._dtype)
        return self.model.dof_armature.copy()

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._check_open()
        if name in self._unsupported_sensors:
            raise NotImplementedError(
                f"superdex sensor {name!r}: {self._unsupported_sensors[name]}"
            )
        return self._sensor_values[name].copy()

    def get_base_pos(self) -> np.ndarray:
        return self._pos[:, self._base_id].copy()

    def get_base_quat(self) -> np.ndarray:
        return self._quat[:, self._base_id].copy()

    def get_base_lin_vel(self) -> np.ndarray:
        return self._lin[:, self._base_id].copy()

    def get_base_ang_vel(self) -> np.ndarray:
        return self._ang[:, self._base_id].copy()

    def get_dof_pos(self) -> np.ndarray:
        return self._qpos[:, self.model.joint_qpos_indices].copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._qvel[:, self.model.joint_qvel_indices].copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._pos[:, body_ids].copy()

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._quat[:, body_ids].copy()

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._lin[:, body_ids].copy()

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._ang[:, body_ids].copy()

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return unrotate(
            self.get_base_quat()[:, None], self._pos[:, body_ids] - self.get_base_pos()[:, None]
        )

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return multiply(conjugate(self.get_base_quat())[:, None], self._quat[:, body_ids])

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return unrotate(self._quat[:, body_ids], self._lin[:, body_ids])

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return unrotate(self._quat[:, body_ids], self._ang[:, body_ids])

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        self._check_open()
        body_ids = np.asarray(body_ids)
        expected = (self.num_envs, len(body_ids), 3)
        f = np.asarray(force, dtype=self._dtype)
        t = np.zeros_like(f) if torque is None else np.asarray(torque, dtype=self._dtype)
        if (
            f.shape != expected
            or t.shape != expected
            or not np.isfinite(f).all()
            or not np.isfinite(t).all()
        ):
            raise ValueError(f"body force and torque must be finite with shape {expected}")
        if body_ids.ndim != 1 or not np.issubdtype(body_ids.dtype, np.integer):
            raise ValueError("body_ids must be an integer vector")
        if np.any(body_ids <= 0) or np.any(body_ids >= len(self.model.body_names)):
            raise ValueError("body force must target an articulated body")
        for column, body_id in enumerate(body_ids):
            self._pending_wrench[:, body_id, :3] += f[:, column]
            self._pending_wrench[:, body_id, 3:] += t[:, column]

    def close(self) -> None:
        if self._pid != os.getpid() or (self._closed and not self._acquired):
            return
        self._closed = True
        # Cleanup must keep trying after an error so the remaining scenes do not leak.
        errors: list[Exception] = []
        if self._batch_executor is not None:
            try:
                self._batch_executor.close()
                self._batch_executor = None
            except Exception as exc:
                errors.append(exc)
        for index, world in reversed(list(enumerate(self._worlds))):
            if world is None:
                continue
            try:
                if index < len(self._snapshots) and self._snapshots[index] is not None:
                    world.release_state(self._snapshots[index])
                    self._snapshots[index] = None
            except Exception as exc:
                errors.append(exc)
            try:
                if index < len(self._actor_cleanups):
                    self._actor_cleanups[index]()
            except Exception as exc:
                # A live Bot/controller may still reference this scene. Keep
                # ownership and allow a later close() to retry its cleanup.
                errors.append(exc)
                continue
            try:
                self._p.destroy_scene(world)
                self._worlds[index] = None
            except Exception as exc:
                errors.append(exc)
        if any(world is not None for world in self._worlds):
            raise RuntimeError(f"SuperDex cleanup is incomplete; close may be retried: {errors[0]}")
        self._worlds.clear()
        self._actors.clear()
        self._links.clear()
        self._native_actor_links.clear()
        self._actor_cleanups.clear()
        self._snapshots.clear()
        self._sensor_sources.clear()
        if self._plan is not None:
            self._plan.cleanup()
        if self._acquired:
            release_runtime(self._p)
            self._acquired = False
        if errors:
            raise RuntimeError(f"SuperDex cleanup failed: {errors[0]}") from errors[0]

    def cleanup_scene_assets(self) -> None:
        """Honor the public cleanup hook used by UniLab's environment lifecycle."""
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            try:
                self.close()
            except Exception:
                pass
