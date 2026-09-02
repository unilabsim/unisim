"""MuJoCo adapter for the backend-neutral :mod:`unisim` contract.

The adapter is intentionally dependency-lazy: importing :mod:`unisim` does
not import MuJoCo. XML/model parsing happens only during construction, which
is the materialization cold path; ``step`` and ``reset`` operate on cached
model/data objects and numeric arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .contract import BackendCapability, BackendError, SimBackend


def _load_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise BackendError(
            "MuJoCo adapter requires the optional dependency; install "
            "unisim-core[mujoco]"
        ) from exc
    return mujoco


class MuJoCoBackend(SimBackend):
    """A vectorized collection of independent MuJoCo ``MjData`` instances."""

    backend_type = "mujoco"

    def __init__(
        self,
        model_path: str | Path,
        *,
        num_envs: int = 1,
        frame_skip: int = 1,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        self._mujoco = _load_mujoco()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        try:
            self._model = self._mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:  # noqa: BLE001 - normalize SDK diagnostics
            raise BackendError(f"failed to materialize MuJoCo model {path}: {exc}") from exc
        self._data = [self._mujoco.MjData(self._model) for _ in range(num_envs)]
        self._frame_skip = frame_skip
        self._num_envs = num_envs
        self._ctrl_width = int(self._model.nu)
        self._qpos_shape = (num_envs, int(self._model.nq))
        self._qvel_shape = (num_envs, int(self._model.nv))
        self._ctrl_shape = (num_envs, self._ctrl_width)
        self.reset()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_actuators(self) -> int:
        return self._ctrl_width

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
        controls = np.asarray(ctrl, dtype=np.float64)
        if controls.shape != self._ctrl_shape:
            raise ValueError(f"ctrl shape {controls.shape} does not match {self._ctrl_shape}")
        if not np.isfinite(controls).all():
            raise ValueError("ctrl contains NaN or Inf")
        if nsteps < 1:
            raise ValueError("nsteps must be positive")
        for index, data in enumerate(self._data):
            data.ctrl[...] = controls[index]
            for _ in range(nsteps * self._frame_skip):
                self._mujoco.mj_step(self._model, data)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            ids = np.arange(self._num_envs, dtype=np.intp)
        else:
            ids = np.asarray(env_ids, dtype=np.intp)
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if np.any(ids < 0) or np.any(ids >= self._num_envs):
                raise IndexError("env_ids contains an out-of-range index")
        for env_id in ids:
            data = self._data[int(env_id)]
            self._mujoco.mj_resetData(self._model, data)
            self._mujoco.mj_forward(self._model, data)

    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        requested = ("qpos", "qvel", "ctrl") if fields is None else fields
        result: dict[str, np.ndarray] = {}
        for field in requested:
            if field == "qpos":
                result[field] = np.stack([data.qpos.copy() for data in self._data])
            elif field == "qvel":
                result[field] = np.stack([data.qvel.copy() for data in self._data])
            elif field == "ctrl":
                result[field] = np.stack([data.ctrl.copy() for data in self._data])
            else:
                raise KeyError(f"unknown MuJoCo state field: {field}")
        return result

    def set_state(self, state: Mapping[str, np.ndarray]) -> None:
        if "qpos" in state:
            qpos = np.asarray(state["qpos"], dtype=np.float64)
            if qpos.shape != self._qpos_shape:
                raise ValueError(f"qpos shape {qpos.shape} does not match {self._qpos_shape}")
        else:
            qpos = None
        if "qvel" in state:
            qvel = np.asarray(state["qvel"], dtype=np.float64)
            if qvel.shape != self._qvel_shape:
                raise ValueError(f"qvel shape {qvel.shape} does not match {self._qvel_shape}")
        else:
            qvel = None
        for index, data in enumerate(self._data):
            if qpos is not None:
                data.qpos[...] = qpos[index]
            if qvel is not None:
                data.qvel[...] = qvel[index]
            self._mujoco.mj_forward(self._model, data)


__all__ = ["MuJoCoBackend"]
