"""Shared lazy runtime bridge for optional engine adapters.

The core wheel deliberately does not import any simulator SDK.  Adapters use
``RuntimeBackend`` to normalize dependency diagnostics and to provide a small
public contract around an injected runtime object.  The latter is useful for
conformance tests and for downstream integrations that already own engine
materialization (for example an Isaac worker process).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, Iterable

import numpy as np

from .contract import BackendCapability, BackendError


class OptionalDependencyError(BackendError, ImportError):
    """Raised when an adapter's optional runtime cannot be loaded."""


def load_optional_runtime(module_names: Iterable[str], *, backend: str, install_hint: str) -> Any:
    """Import the first available module and normalize missing-SDK errors."""
    names = tuple(module_names)
    for name in names:
        try:
            if find_spec(name) is None:
                continue
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise OptionalDependencyError(
                f"{backend} adapter could not import optional runtime {name!r}: {exc}. "
                f"{install_hint}"
            ) from exc
    joined = ", ".join(names)
    raise OptionalDependencyError(
        f"{backend} adapter requires one of [{joined}] to be installed. {install_hint}"
    )


def _runtime_value(runtime: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(runtime, Mapping) and name in runtime:
            return runtime[name]
        value = getattr(runtime, name, None)
        if value is not None:
            return value() if callable(value) and name.startswith("num_") else value
    return default


class RuntimeBackend:
    """Engine-neutral adapter around an already materialized runtime.

    Concrete adapters only provide identity and dependency discovery.  A
    runtime may expose ``step(ctrl, nsteps=...)``, ``reset(env_ids=...)``,
    ``get_state(fields=...)`` and ``set_state(state)``.  Numeric arrays are
    cached at lifecycle barriers, so getters never inspect SDK objects on the
    hot path.  If no bridge is supplied, the adapter attempts to load the
    optional SDK and raises an actionable error rather than pretending that
    physics is available.
    """

    backend_type = "runtime"
    module_names: tuple[str, ...] = ()
    install_hint = "Install the adapter extra listed in the UniSim support matrix."

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        num_envs: int = 1,
        num_actuators: int | None = None,
        frame_skip: int = 1,
        **_: Any,
    ) -> None:
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(frame_skip, bool) or int(frame_skip) <= 0:
            raise ValueError("frame_skip must be a positive integer")
        self._runtime = (
            runtime
            if runtime is not None
            else load_optional_runtime(
                self.module_names, backend=self.backend_type, install_hint=self.install_hint
            )
        )
        self._num_envs = int(_runtime_value(self._runtime, "num_envs", default=num_envs))
        if self._num_envs <= 0:
            raise ValueError("runtime num_envs must be positive")
        inferred = _runtime_value(self._runtime, "num_actuators", "nu", default=num_actuators)
        self._num_actuators = int(1 if inferred is None else inferred)
        if self._num_actuators <= 0:
            raise ValueError("runtime num_actuators must be positive")
        self._frame_skip = int(frame_skip)
        self._state: dict[str, np.ndarray] = {
            "qpos": np.zeros((self._num_envs, self._num_actuators), dtype=np.float64),
            "qvel": np.zeros((self._num_envs, self._num_actuators), dtype=np.float64),
            "ctrl": np.zeros((self._num_envs, self._num_actuators), dtype=np.float64),
        }
        self.reset()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_actuators(self) -> int:
        return self._num_actuators

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        caps = {BackendCapability.RESET, BackendCapability.STATE_READ}
        if callable(getattr(self._runtime, "reset", None)):
            caps.add(BackendCapability.SELECTED_RESET)
        if callable(getattr(self._runtime, "set_state", None)):
            caps.add(BackendCapability.STATE_WRITE)
        return frozenset(caps)

    def _sync_state(self) -> None:
        getter = getattr(self._runtime, "get_state", None)
        if not callable(getter):
            return
        try:
            value = getter()
        except TypeError:
            value = getter(None)
        if isinstance(value, Mapping):
            for key, array in value.items():
                self._state[str(key)] = np.asarray(array).copy()

    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        controls = np.asarray(ctrl, dtype=np.float64)
        expected = (self.num_envs, self.num_actuators)
        if controls.shape != expected:
            raise ValueError(f"ctrl shape {controls.shape} does not match {expected}")
        if not np.isfinite(controls).all():
            raise ValueError("ctrl contains NaN or Inf")
        if isinstance(nsteps, bool) or int(nsteps) < 1:
            raise ValueError("nsteps must be positive")
        self._state["ctrl"] = controls.copy()
        step_fn = getattr(self._runtime, "step", None)
        if callable(step_fn):
            try:
                step_fn(controls, nsteps=int(nsteps) * self._frame_skip)
            except TypeError:
                step_fn(controls, int(nsteps) * self._frame_skip)
            self._sync_state()
            return
        raise BackendError(
            f"{self.backend_type} runtime does not expose step(ctrl, nsteps); "
            "materialize it through the adapter-owned runtime bridge"
        )

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        reset_fn = getattr(self._runtime, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn(env_ids=env_ids)
            except TypeError:
                reset_fn(env_ids)
            self._sync_state()
            return
        if env_ids is None:
            self._state["qpos"].fill(0)
            self._state["qvel"].fill(0)
            self._state["ctrl"].fill(0)
            return
        ids = np.asarray(env_ids, dtype=np.intp)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise IndexError("env_ids must be a one-dimensional in-range index array")
        for value in self._state.values():
            if value.ndim and value.shape[0] == self.num_envs:
                value[ids] = 0

    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        if fields is None:
            requested = tuple(self._state)
        elif isinstance(fields, str):
            requested = (fields,)
        else:
            requested = fields
        result: dict[str, np.ndarray] = {}
        for field in requested:
            if field not in self._state:
                raise KeyError(f"unknown {self.backend_type} state field: {field}")
            result[field] = self._state[field].copy()
        return result

    def set_state(self, state: Mapping[str, np.ndarray]) -> None:
        setter = getattr(self._runtime, "set_state", None)
        converted = {str(key): np.asarray(value, dtype=np.float64) for key, value in state.items()}
        if callable(setter):
            setter(converted)
            self._sync_state()
            return
        for key, value in converted.items():
            if key not in self._state:
                raise KeyError(f"unknown {self.backend_type} state field: {key}")
            if value.shape != self._state[key].shape:
                raise ValueError(
                    f"{key} shape {value.shape} does not match {self._state[key].shape}"
                )
            self._state[key][...] = value

    def get_dr_capabilities(self):
        from .dr.types import DomainRandomizationCapabilities

        return DomainRandomizationCapabilities()

    def apply_interval_randomization(self, plan):
        if not plan.is_empty():
            raise NotImplementedError(f"{self.backend_type} runtime has no randomization bridge")


__all__ = ["OptionalDependencyError", "RuntimeBackend", "load_optional_runtime"]
