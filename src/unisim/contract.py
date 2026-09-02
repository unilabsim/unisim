"""Stable, engine-neutral simulation backend contract.

This module intentionally has no dependency on UniLab, Hydra, Torch, or an
engine SDK. Concrete adapters own materialization and engine-native resources.
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import Mapping

import numpy as np


class BackendError(RuntimeError):
    """Base error raised by a UniSim backend."""


class UnsupportedCapabilityError(BackendError):
    """Raised when a requested optional backend capability is unavailable."""


class BackendCapability(str, Enum):
    """Capabilities that an adapter may explicitly advertise."""

    RESET = "reset"
    SELECTED_RESET = "selected_reset"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    MUTATION = "mutation"


class SimBackend(abc.ABC):
    """Minimal lifecycle shared by benchmark and UniLab consumers."""

    backend_type: str

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Return the number of vectorized environments."""

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """Return the width of the control vector."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> frozenset[BackendCapability]:
        """Return capabilities resolved during cold-path materialization."""

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> None:
        """Advance physics by ``nsteps`` using a validated control batch."""

    @abc.abstractmethod
    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset all or selected environments."""

    @abc.abstractmethod
    def get_state(self, fields: tuple[str, ...] | None = None) -> Mapping[str, np.ndarray]:
        """Read named backend-neutral state fields."""

    def set_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Write state when the adapter advertises :attr:`STATE_WRITE`."""
        if BackendCapability.STATE_WRITE not in self.capabilities:
            raise UnsupportedCapabilityError(
                f"backend '{self.backend_type}' does not support state_write"
            )
        raise NotImplementedError
