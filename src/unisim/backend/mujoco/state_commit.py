"""Prepared writes consumed by the single MuJoCo native state submitter.

These adapter-local values describe an already validated transaction. They
do not inspect assets, call the engine, or impose an entity layout on legacy
models with arbitrary valid MuJoCo topology and transmissions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelWrite:
    field: str
    values: np.ndarray
    column: int | None = None


@dataclass(frozen=True)
class MocapWrite:
    index: int
    poses: np.ndarray


@dataclass(frozen=True)
class StateCommitPlan:
    """Owned prepared rows; public callers never submit this internal value.

    ``reset_world`` preserves the historical full-reset lifecycle; false
    preserves every unspecified entity/environment channel. ``defaults`` is
    used only by the explicit reset-to-initial-state entry point.
    """

    rows: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    qpos_columns: tuple[int, ...]
    qvel_columns: tuple[int, ...]
    reset_world: bool = False
    mocap: tuple[MocapWrite, ...] = ()
    control_clear: tuple[int, ...] = ()
    activation_clear: tuple[int, ...] = ()
    force_dof_clear: tuple[int, ...] = ()
    force_body_clear: tuple[int, ...] = ()
    model_writes: tuple[ModelWrite, ...] = ()
    defaults: tuple[tuple[str, np.ndarray], ...] = ()


def selected_rows(value: np.ndarray, num_envs: int) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("env_indices must be a one-dimensional integer array")
    if np.any(rows < 0) or np.any(rows >= num_envs):
        raise ValueError("env_indices must be in range")
    if len(np.unique(rows)) != rows.size:
        raise ValueError("env_indices must be unique")
    return rows.astype(np.int32, copy=True)


def state_values(value: np.ndarray, shape: tuple[int, ...], dtype, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if values.dtype.kind not in "fi" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real values")
    limit = np.finfo(dtype).max
    if np.any(values > limit) or np.any(values < -limit):
        raise ValueError(f"{name} values exceed {np.dtype(dtype)} range")
    return values.astype(dtype, copy=True)
