"""Shared host-copy kernel for backend-owned body-state caches."""

from __future__ import annotations

import numpy as np


def copy_selected_body_state(
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    source_lin_vel: np.ndarray,
    source_ang_vel: np.ndarray,
    selected_ids: np.ndarray,
    out_pos: np.ndarray,
    out_quat: np.ndarray,
    out_lin_vel: np.ndarray,
    out_ang_vel: np.ndarray,
) -> None:
    """Copy selected cache columns into four caller-owned state buffers.

    This is intentionally a NumPy-only helper.  The operation is a small
    indexed buffer copy and should not make the base wheel depend on Numba.
    Backends that need a compiled kernel can provide one in their optional
    extra without changing the public contract.
    """
    ids = np.asarray(selected_ids, dtype=np.intp)
    out_pos[...] = np.asarray(source_pos)[:, ids]
    out_quat[...] = np.asarray(source_quat)[:, ids]
    out_lin_vel[...] = np.asarray(source_lin_vel)[:, ids]
    out_ang_vel[...] = np.asarray(source_ang_vel)[:, ids]


__all__ = ["copy_selected_body_state"]
