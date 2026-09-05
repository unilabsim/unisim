"""Newton 1.5.1 adapter implementing the host-NumPy SimBackend profile.

All Newton/Warp transfers happen at explicit materialize, step, or set_state
barriers. Public getters return stable NumPy cache views and never inspect
MuJoCo's CPU data object. GPU execution is not bitwise deterministic; callers
and tests must use numerical tolerances.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from unisim.backend.base import BackendRootStateLayout, SimBackend
from unisim.dr.types import DomainRandomizationCapabilities, ResetRandomizationPayload
from unisim.scene import SceneCfg
from unisim.utils.rotation import (
    np_quat_apply_batched,
    np_quat_apply_inverse_batched,
    np_quat_conjugate_batched,
    np_quat_mul_batched,
)

from .capacity import NewtonCapacityReport, calibrate_capacity, validate_capacity_limits
from .dependencies import load_newton_dependencies
from .materialization import (
    NewtonModelAudit,
    audit_newton_model,
    scan_newton_model_metadata,
)
from .runtime import get_bound_newton_process_device

_WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)


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
        **unexpected_kwargs: Any,
    ) -> None:
        if unexpected_kwargs:
            names = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"NewtonBackend does not accept backend options: {names}")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        if float(sim_dt) <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {sim_dt!r}")
        self._nconmax = self._capacity(nconmax, "nconmax", 512)
        self._njmax = self._capacity(njmax, "njmax", 512)
        self._capacity_check_steps = self._capacity(
            capacity_check_steps, "capacity_check_steps", 1
        )

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
        self._scene_visual_model_file = str(scene.visual_model_file or scene.model_file)
        self._metadata = scan_newton_model_metadata(self._deps.mujoco, scene)
        if not self._metadata.root_qpos_dim:
            raise NotImplementedError(
                "newton backend currently requires a free root joint for the "
                "SimBackend state contract"
            )
        self._scene_cleanup_handle = self._metadata.cleanup_handle
        authored_root_name = next((name for name in self._metadata.body_names[1:] if name), None)
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
        for plan in self._metadata.sensor_plans:
            self._sensor_slots[plan.name] = (address, plan.dim)
            address += plan.dim

        self._model: Any | None = None
        self._solver: Any | None = None
        self._state: Any | None = None
        self._state_out: Any | None = None
        self._control: Any | None = None
        self._contacts: Any | None = None
        self._view: Any | None = None
        self._audit: NewtonModelAudit | None = None
        self._capacity_report: NewtonCapacityReport | None = None
        self._closed = False

        nbody = len(self._body_names)
        self._qpos_cache = np.zeros((num_envs, self._metadata.nq), dtype=np.float32)
        self._qvel_cache = np.zeros((num_envs, self._metadata.nv), dtype=np.float32)
        self._body_pos_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
        self._body_quat_cache = np.zeros((num_envs, nbody, 4), dtype=np.float32)
        self._body_lin_vel_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
        self._body_ang_vel_cache = np.zeros((num_envs, nbody, 3), dtype=np.float32)
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

    def _require_state(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"newton backend is closed; cannot run {operation}")
        self.materialize()

    def materialize(self) -> None:
        if self._model is not None:
            return
        newton = self._deps.newton
        previous_layout = bool(newton.use_coord_layout_targets)
        newton.use_coord_layout_targets = True
        try:
            template = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(template)
            template.add_mjcf(self._metadata.source_model_file, ctrl_direct=False)
            builder = newton.ModelBuilder()
            builder.replicate(template, self._num_envs)
            self._model = builder.finalize(device=self._device)
        finally:
            newton.use_coord_layout_targets = previous_layout
            self.cleanup_scene_assets()

        self._audit = audit_newton_model(self._model, self._metadata, self._num_envs)
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            separate_worlds=True,
            nconmax=self._nconmax,
            njmax=self._njmax,
            solver="newton",
            integrator="implicitfast",
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=0,
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
        newton.eval_fk(
            self._model, self._state.joint_q, self._state.joint_qd, self._state
        )
        self._refresh_host_cache()
        self._capacity_report = calibrate_capacity(
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

    def _advance_capacity_probe(self) -> tuple[int, int]:
        self._physics_substep(np.zeros((self._num_envs, self.num_actuators), np.float32))
        return self._read_solver_counts()

    def _read_solver_counts(self) -> tuple[int, int]:
        self._deps.warp.synchronize_device(self._device)
        ncon = int(np.max(np.asarray(self._solver.mjw_data.nacon.numpy())))
        nefc = int(np.max(np.asarray(self._solver.mjw_data.nefc.numpy())))
        return ncon, nefc

    def _set_control(self, ctrl: np.ndarray) -> None:
        namespace = getattr(self._control, "mujoco", None)
        target = getattr(namespace, "ctrl", None) if namespace is not None else None
        if target is not None:
            target.assign(np.ascontiguousarray(ctrl.reshape(-1), dtype=np.float32))
        target_q = getattr(self._control, "joint_target_q", None)
        target_qd = getattr(self._control, "joint_target_qd", None)
        if target_q is not None:
            q_targets = np.zeros((self._num_envs, self._metadata.nq), dtype=np.float32)
            for actuator_id, kind in enumerate(self._metadata.actuator_target_kinds):
                if kind == "position":
                    q_targets[:, self._metadata.actuator_target_qpos_adrs[actuator_id]] = ctrl[
                        :, actuator_id
                    ]
            target_q.assign(np.ascontiguousarray(q_targets.reshape(-1)))
        if target_qd is not None:
            qd_targets = np.zeros((self._num_envs, self._metadata.nv), dtype=np.float32)
            for actuator_id, kind in enumerate(self._metadata.actuator_target_kinds):
                if kind == "velocity":
                    qd_targets[:, self._metadata.actuator_target_qvel_adrs[actuator_id]] = ctrl[
                        :, actuator_id
                    ]
            target_qd.assign(np.ascontiguousarray(qd_targets.reshape(-1)))

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

    def _refresh_sensor_cache(self, *, sensor_dt: float | None) -> None:
        for plan in self._metadata.sensor_plans:
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
        return self._metadata.body_mass.copy()

    def get_body_ipos(self) -> np.ndarray:
        return self._metadata.body_ipos.copy()

    def get_dof_armature(self) -> np.ndarray:
        return self._metadata.dof_armature.copy()

    def get_joint_range(self) -> np.ndarray | None:
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
        warp_mask = self._deps.warp.array(mask, dtype=self._deps.warp.bool, device=self._device)
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
        self._solver.reset(self._state, warp_mask)
        upload_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self._refresh_host_cache()
        self._time_cache[rows] = 0.0
        cache_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timing": {"set_state_upload_ms": upload_ms, "set_state_cache_ms": cache_ms}
        }

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities()

    def get_base_pos(self) -> np.ndarray:
        self._require_state("get_base_pos")
        return self._qpos_cache[:, :3]

    def get_base_quat(self) -> np.ndarray:
        self._require_state("get_base_quat")
        return self._qpos_cache[:, 3:7]

    def get_base_lin_vel(self) -> np.ndarray:
        self._require_state("get_base_lin_vel")
        return self._qvel_cache[:, :3]

    def get_base_ang_vel(self) -> np.ndarray:
        self._require_state("get_base_ang_vel")
        quat = self._qpos_cache[:, 3:7]
        return np_quat_apply_batched(quat, self._qvel_cache[:, 3:6])

    def get_dof_pos(self) -> np.ndarray:
        self._require_state("get_dof_pos")
        return self._qpos_cache[:, self._metadata.root_qpos_dim :]

    def get_dof_vel(self) -> np.ndarray:
        self._require_state("get_dof_vel")
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
        self._closed = True
        self.cleanup_scene_assets()


__all__ = ["NewtonBackend"]
