"""Prepared host transactions for the single MJWarp native state barrier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelUpdates:
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    # 2: set_const; 1: set_const_0; 0: no derived-constant refresh.
    refresh: int = 0
    actuator_fields: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class StateCommitPlan:
    rows: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    channels: dict[str, np.ndarray]
    staged_wrenches: np.ndarray
    time: np.ndarray
    reset_world: bool = False
    entity_names: tuple[str, ...] | None = None
    model_updates: ModelUpdates = field(default_factory=ModelUpdates)
    allow_scratch: bool = False


def float_values(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Own finite representable float32 state/model rows before native writes."""
    raw = np.asarray(values)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind not in "fi" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real values")
    limit = np.finfo(np.float32).max
    if np.any(raw > limit) or np.any(raw < -limit):
        raise ValueError(f"{name} must contain finite float32 values")
    return raw.astype(np.float32, copy=True)
