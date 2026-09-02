"""Small deterministic backend used by contract tests and examples."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .contract import BackendCapability, SimBackend


class FakeBackend(SimBackend):
    """A dependency-free vectorized backend with deterministic state updates."""

    backend_type = "fake"

    def __init__(self, num_envs: int = 2, num_actuators: int = 1) -> None:
        if num_envs <= 0 or num_actuators <= 0:
            raise ValueError("num_envs and num_actuators must be positive")
        self._num_envs = num_envs
        self._num_actuators = num_actuators
        self._qpos = np.zeros((num_envs, num_actuators), dtype=np.float64)
        self._step_count = 0

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_actuators(self) -> int:
        return self._num_actuators

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        return frozenset(
            {
                BackendCapability.RESET,
                BackendCapability.STATE_READ,
                BackendCapability.STATE_WRITE,
            }
        )

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        if ctrl.shape != (self.num_envs, self.num_actuators):
            raise ValueError(
                f"ctrl shape {ctrl.shape} does not match "
                f"({self.num_envs}, {self.num_actuators})"
            )
        if nsteps < 1:
            raise ValueError("nsteps must be positive")
        self._qpos += ctrl * nsteps
        self._step_count += nsteps

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self._qpos.fill(0.0)
            self._step_count = 0
            return
        ids = np.asarray(env_ids, dtype=np.intp)
        self._qpos[ids] = 0.0

    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        requested = ("qpos", "step_count") if fields is None else fields
        result: dict[str, np.ndarray] = {}
        for field in requested:
            if field == "qpos":
                result[field] = self._qpos.copy()
            elif field == "step_count":
                result[field] = np.asarray(self._step_count, dtype=np.int64)
            else:
                raise KeyError(f"unknown fake state field: {field}")
        return result

    def set_state(self, state: Mapping[str, np.ndarray]) -> None:
        if "qpos" in state:
            qpos = np.asarray(state["qpos"], dtype=np.float64)
            if qpos.shape != self._qpos.shape:
                raise ValueError(f"qpos shape {qpos.shape} does not match {self._qpos.shape}")
            self._qpos[...] = qpos
        if "step_count" in state:
            self._step_count = int(np.asarray(state["step_count"]).item())

