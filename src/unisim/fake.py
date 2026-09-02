"""Small deterministic backend used by contract tests and examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .contract import BackendCapability, SimBackend
from .dr.types import DomainRandomizationCapabilities, IntervalRandomizationPlan


class FakeBackend(SimBackend):
    """A dependency-free vectorized backend with deterministic state updates."""

    backend_type = "fake"

    def __init__(self, num_envs: int = 2, num_actuators: int = 1) -> None:
        if num_envs <= 0 or num_actuators <= 0:
            raise ValueError("num_envs and num_actuators must be positive")
        self._num_envs = num_envs
        self._num_actuators = num_actuators
        self._qpos = np.zeros((num_envs, num_actuators), dtype=np.float64)
        self._qvel = np.zeros_like(self._qpos)
        self._ctrl = np.zeros_like(self._qpos)
        self._step_count = 0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_actuators(self) -> int:
        return self._num_actuators

    @property
    def model(self):
        return self

    @property
    def num_dof_vel(self) -> int:
        return self._num_actuators

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return np.full((self._num_actuators, 2), (-1.0, 1.0), dtype=np.float64)

    def get_joint_range(self) -> np.ndarray:
        return self.get_actuator_ctrl_range()

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        if name not in {"home", "stand"}:
            raise KeyError(name)
        return np.zeros(self._num_actuators, dtype=np.float64)

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros(self._num_actuators, dtype=np.float64)

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        return np.arange(len(tuple(names)), dtype=np.int32)

    def _zeros_body(self, body_ids: np.ndarray) -> np.ndarray:
        return np.zeros(
            (self._num_envs, len(np.asarray(body_ids).reshape(-1)), 3), dtype=np.float64
        )

    def get_base_pos(self) -> np.ndarray:
        return np.zeros((self._num_envs, 3))

    def get_base_quat(self) -> np.ndarray:
        out = np.zeros((self._num_envs, 4))
        out[:, 0] = 1.0
        return out

    def get_base_lin_vel(self) -> np.ndarray:
        return np.zeros((self._num_envs, 3))

    def get_base_ang_vel(self) -> np.ndarray:
        return np.zeros((self._num_envs, 3))

    def get_dof_pos(self) -> np.ndarray:
        return self._qpos.copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._qvel.copy()

    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        out = np.zeros((*self._zeros_body(body_ids).shape[:2], 4))
        out[..., 0] = 1.0
        return out

    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self.get_body_quat_w(body_ids)

    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        return self._zeros_body(body_ids)

    def get_sensor_data(self, name: str) -> np.ndarray:
        raise KeyError(name)

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities()

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        if not plan.is_empty():
            raise NotImplementedError("fake backend has no randomization")

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        return frozenset(
            {
                BackendCapability.RESET,
                BackendCapability.SELECTED_RESET,
                BackendCapability.STATE_READ,
                BackendCapability.STATE_WRITE,
            }
        )

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.num_envs, self.num_actuators):
            raise ValueError(
                f"ctrl shape {ctrl.shape} does not match ({self.num_envs}, {self.num_actuators})"
            )
        if nsteps < 1:
            raise ValueError("nsteps must be positive")
        self._ctrl[...] = ctrl
        self._qpos += ctrl * nsteps
        self._qvel[...] = ctrl
        self._step_count += nsteps

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self._qpos.fill(0.0)
            self._qvel.fill(0.0)
            self._ctrl.fill(0.0)
            self._step_count = 0
            return
        ids = np.asarray(env_ids, dtype=np.intp)
        self._qpos[ids] = 0.0
        self._qvel[ids] = 0.0
        self._ctrl[ids] = 0.0

    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        requested = (
            ("qpos", "qvel", "ctrl", "step_count")
            if fields is None
            else ((fields,) if isinstance(fields, str) else fields)
        )
        result: dict[str, np.ndarray] = {}
        for field in requested:
            if field == "qpos":
                result[field] = self._qpos.copy()
            elif field == "step_count":
                result[field] = np.asarray(self._step_count, dtype=np.int64)
            elif field == "qvel":
                result[field] = self._qvel.copy()
            elif field == "ctrl":
                result[field] = self._ctrl.copy()
            else:
                raise KeyError(f"unknown fake state field: {field}")
        return result

    def set_state(self, env_indices, qpos=None, qvel=None, randomization=None) -> None:
        # Accept the full SimBackend transaction and the historical mapping
        # spelling for downstream callers during the migration window.
        if isinstance(env_indices, Mapping):
            state = env_indices
            if "qpos" in state:
                qpos = state["qpos"]
            if "qvel" in state:
                qvel = state["qvel"]
            if "ctrl" in state:
                self._ctrl[...] = np.asarray(state["ctrl"], dtype=np.float64)
            if "step_count" in state:
                self._step_count = int(np.asarray(state["step_count"]).item())
            env_indices = np.arange(self._num_envs, dtype=np.intp)
        if randomization is not None and not randomization.is_empty():
            raise NotImplementedError("fake backend has no randomization")
        ids = np.asarray(env_indices, dtype=np.intp)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self._num_envs):
            raise ValueError("env_indices must be a one-dimensional in-range index array")
        if qpos is not None:
            values = np.asarray(qpos, dtype=np.float64)
            if values.shape == (ids.size, self._num_actuators):
                self._qpos[ids] = values
            elif values.shape == (ids.size, 7 + self._num_actuators):
                self._qpos[ids] = values[:, 7:]
            else:
                raise ValueError(f"qpos shape {values.shape} does not match selected state")
        if qvel is not None:
            values = np.asarray(qvel, dtype=np.float64)
            if values.shape == (ids.size, self._num_actuators):
                self._qvel[ids] = values
            elif values.shape == (ids.size, 6 + self._num_actuators):
                self._qvel[ids] = values[:, 6:]
            else:
                raise ValueError(f"qvel shape {values.shape} does not match selected state")
