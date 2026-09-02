"""MotrixSim adapter for the public :mod:`unisim` contract.

MotrixSim is optional and imported only when the adapter is constructed. The
adapter keeps native ``SceneModel``/``SceneData`` objects private and exposes
NumPy state arrays through the same contract as MuJoCo.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .contract import BackendCapability, BackendError, SimBackend


def _load_motrix():
    try:
        import motrixsim
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise BackendError(
            "Motrix adapter requires the optional dependency; install "
            "unisim-core[motrix]"
        ) from exc
    return motrixsim


class MotrixBackend(SimBackend):
    """A batched MotrixSim model implementing the common backend lifecycle."""

    backend_type = "motrix"

    def __init__(self, model_path: str | Path, *, num_envs: int = 1, frame_skip: int = 1) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        self._motrix = _load_motrix()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Motrix model does not exist: {path}")
        try:
            self._model = self._motrix.load_model(str(path))
            self._data = self._motrix.SceneData(self._model, batch=[num_envs])
        except Exception as exc:  # noqa: BLE001 - normalize SDK diagnostics
            raise BackendError(f"failed to materialize Motrix model {path}: {exc}") from exc
        self._num_envs = num_envs
        self._frame_skip = frame_skip
        self._num_actuators = int(self._model.num_actuators)
        self._num_dof_pos = int(self._model.num_dof_pos)
        self._num_dof_vel = int(self._model.num_dof_vel)
        self.reset()

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
                BackendCapability.SELECTED_RESET,
                BackendCapability.STATE_READ,
                BackendCapability.STATE_WRITE,
            }
        )

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        controls = np.asarray(ctrl, dtype=np.float64)
        expected = (self.num_envs, self.num_actuators)
        if controls.shape != expected:
            raise ValueError(f"ctrl shape {controls.shape} does not match {expected}")
        if not np.isfinite(controls).all():
            raise ValueError("ctrl contains NaN or Inf")
        if nsteps < 1:
            raise ValueError("nsteps must be positive")
        self._data.actuator_ctrls = np.ascontiguousarray(controls)
        self._model.step_n(self._data, int(nsteps * self._frame_skip))

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self._data.reset(self._model)
            return
        ids = np.asarray(env_ids, dtype=np.intp)
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise IndexError("env_ids contains an out-of-range index")
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[ids] = True
        self._data[mask].reset(self._model)

    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        requested = ("qpos", "qvel", "ctrl") if fields is None else fields
        result: dict[str, np.ndarray] = {}
        for field in requested:
            if field == "qpos":
                result[field] = np.asarray(self._data.dof_pos).copy()
            elif field == "qvel":
                result[field] = np.asarray(self._data.dof_vel).copy()
            elif field == "ctrl":
                result[field] = np.asarray(self._data.actuator_ctrls).copy()
            else:
                raise KeyError(f"unknown Motrix state field: {field}")
        return result

    def set_state(self, state: Mapping[str, np.ndarray]) -> None:
        if "qpos" in state:
            qpos = np.asarray(state["qpos"], dtype=np.float64)
            expected = (self.num_envs, self._num_dof_pos)
            if qpos.shape != expected:
                raise ValueError(f"qpos shape {qpos.shape} does not match {expected}")
            self._data.set_dof_pos(qpos, self._model)
        if "qvel" in state:
            qvel = np.asarray(state["qvel"], dtype=np.float64)
            expected = (self.num_envs, self._num_dof_vel)
            if qvel.shape != expected:
                raise ValueError(f"qvel shape {qvel.shape} does not match {expected}")
            self._data.set_dof_vel(qvel)


__all__ = ["MotrixBackend"]
