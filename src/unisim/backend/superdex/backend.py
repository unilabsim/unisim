"""CPU SuperDex adapter with NumPy state barriers and independent native scenes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from unisim.backend.base import (
    BackendPlayCapabilities,
    BackendPlayRenderPlan,
    BackendRootStateLayout,
    SimBackend,
    normalize_play_render_mode,
)
from unisim.dr.types import DomainRandomizationCapabilities, ResetRandomizationPayload
from unisim.scene import SceneCfg
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
    barrier for independent scenes; reset and sensor ownership remain here.
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
        effort_limits: Sequence[float] | None = None,
        allow_contact_approximation: bool = False,
        **unexpected: Any,
    ) -> None:
        if unexpected:
            raise TypeError(f"SuperDexBackend does not accept options: {sorted(unexpected)}")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not np.isfinite(sim_dt) or sim_dt <= 0:
            raise ValueError("sim_dt must be finite and positive")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer (0 is automatic)")
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
        self._batch_executor = None
        self._batch_num_workers = self._resolve_num_workers(num_workers)
        self._worlds: list[Any] = []
        self._actors: list[Any] = []
        self._links: list[list[Any]] = []
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
            )
            self._body_lookup = {name: i for i, name in enumerate(self._plan.body_names)}
            self._joint_lookup = {name: i for i, name in enumerate(self._plan.joint_names)}
            self._base_id = (
                self._body_lookup[base_name] if base_name is not None else self._plan.root_body_id
            )
            self._allocate_caches()
            self.materialize()
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

    def _resolve_num_workers(self, requested: int) -> int:
        """Resolve native scene workers from the current process affinity."""
        if requested:
            return min(self.num_envs, requested)
        return min(self.num_envs, physical_cpu_count())

    def _allocate_caches(self) -> None:
        n, m, dtype = self.num_envs, self.model, self._dtype
        self._qpos = np.zeros((n, m.nq), dtype=dtype)
        self._qvel = np.zeros((n, m.nv), dtype=dtype)
        self._ctrl = np.zeros((n, self.num_actuators), dtype=dtype)
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
        self._native_actuator_qpos_indices = np.asarray(m.actuator_qpos_indices, dtype=np.int32).copy()
        if m.floating:
            self._native_actuator_qpos_indices -= 1
        self._native_actuator_qvel_indices = np.asarray(m.actuator_qvel_indices, dtype=np.int32).copy()
        self._native_actuator_kp = np.asarray(m.actuator_kp, dtype=dtype)
        self._native_actuator_kd = np.asarray(m.actuator_kd, dtype=dtype)
        self._native_actuator_gear = np.asarray(m.actuator_gear, dtype=dtype)
        self._native_actuator_force_ranges = (
            np.asarray(m.actuator_force_ranges, dtype=dtype)
            if m.actuator_force_ranges is not None
            else np.repeat(
                np.asarray([[-np.finfo(dtype).max, np.finfo(dtype).max]], dtype=dtype),
                self.num_actuators,
                axis=0,
            )
        )

    def materialize(self) -> None:
        self._check_open()
        if self._actors:
            return
        for i in range(self.num_envs):
            world = self._p.create_scene(f"UniSim SuperDex {i}")
            self._worlds.append(world)
            world.set_gravity(self.model.gravity)
            actor, cleanup = self.model.spawn_actor(world)
            self._actors.append(actor)
            self._actor_cleanups.append(cleanup)
            if actor.get_num_dofs() != self.model.nv:
                raise ValueError("SuperDex compiled DoF count differs from audited authoring plan")
            links = [world.get_actor(h) for h in actor.get_nested_link_actors()]
            self._links.append(links)
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
                source = links[index]
                query = (
                    self._p.QueryType.CONTACT_POINTS
                    if sensor.kind in {"contact", "contact_found"}
                    else self._p.QueryType.TOTAL_CONTACT_FORCE
                )
                source.register_query(query)
                other = actor_names[sensor.other_actor_name] if sensor.other_actor_name else None
                sources[sensor.name] = source, other
            self._sensor_sources.append(sources)
            self._snapshots.append(world.capture_state())
        if self._links:
            link_count = len(self._links[0])
            if any(len(links) != link_count for links in self._links):
                raise ValueError("SuperDex articulated actors must have equal link counts")
            self._native_link_state = np.zeros((self.num_envs, link_count, 16), dtype=self._dtype)
        executor_cls = getattr(self._p, "SceneBatchExecutor", None)
        if executor_cls is None or not hasattr(executor_cls, "num_links"):
            raise RuntimeError(
                "superdex requires the local project_superdex build with the extended "
                "SceneBatchExecutor state/contact API"
            )
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
        m = self.model
        if nsteps > 1 and self._pre_step_control_fn is None and not self._pending_wrench.any():
            self._ctrl[:] = np.clip(values, m.actuator_ctrl_ranges[:, 0], m.actuator_ctrl_ranges[:, 1])
            assert self._batch_executor is not None
            self._batch_executor.step_control(
                self._dt,
                self._ctrl,
                self._native_actuator_qpos_indices,
                self._native_actuator_qvel_indices,
                self._native_actuator_kp,
                self._native_actuator_kd,
                self._native_actuator_gear,
                self._native_actuator_force_ranges,
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
                force = np.clip(force, m.actuator_force_ranges[:, 0], m.actuator_force_ranges[:, 1])
            force = force * m.actuator_gear
            self._batch_forces.fill(0)
            if self._pending_wrench.any():
                active_envs = np.flatnonzero(np.any(self._pending_wrench != 0, axis=(1, 2)))
                for i in active_envs:
                    generalized = self._batch_forces[i]
                    for body in np.flatnonzero(np.any(self._pending_wrench[i] != 0, axis=1)):
                        link = self._links[i][m.body_link_indices[body]]
                        jacobian = np.asarray(link.get_articulated_jacobian()).reshape(6, m.nv)
                        generalized += jacobian.T @ self._pending_wrench[i, body]
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

    def _refresh(
        self,
        ids: np.ndarray,
        *,
        native_state_ready: bool = False,
        full_state_ready: bool = True,
    ) -> None:
        m = self.model
        if not ids.size:
            return
        if native_state_ready:
            if m.floating:
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
        if m.floating:
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
            else:
                for i in ids:
                    for sensor in self._contact_sensors:
                        self._sensor_values[sensor.name][i] = self._read_sensor(i, sensor)
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

    def get_joint_range(self) -> np.ndarray:
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

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
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
        return BackendPlayCapabilities(supports_physics_state_playback=True)

    @staticmethod
    def resolve_play_render_plan(
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | os.PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
        """Use the shared MuJoCo renderer for SuperDex state playback."""
        mode = normalize_play_render_mode(play_render_mode)
        if mode == "none":
            return BackendPlayRenderPlan(
                mode="none", headless=True, record_video=False, num_steps=None, output_video=None
            )
        if mode == "interactive":
            raise NotImplementedError(
                "superdex uses the MuJoCo offline renderer and does not support "
                "interactive rendering"
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

    def run_playback(self, *, env, initialize, step, num_steps, output_video=None,
                     render_spacing=None, render_offset_mode=None, headless=None,
                     record_video=None, frame_state_getter=None, camera_kwargs=None,
                     extra_data_getter=None):
        from unisim.backend.playback_common import run_offline_snapshot_playback

        if self.scene_visual_model_file is None:
            raise RuntimeError(
                "superdex MuJoCo playback requires scene.visual_model_file for .superdex_bot assets"
            )
        should_record = bool(record_video) if record_video is not None else output_video is not None
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
            camera_kwargs=camera_kwargs,
            backend_label="superdex",
            extra_data_getter=extra_data_getter,
        )

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

    def get_body_ipos(self) -> np.ndarray:
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
                raise RuntimeError(f"SuperDex batch executor cleanup failed: {exc}") from exc
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
