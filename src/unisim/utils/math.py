"""Unified NumPy/Torch rotation, quaternion, and SE(3) primitives.

One module owns both backends.  Conventions are shared: quaternions are
``wxyz``, Euler angles are XYZ, and batched inputs broadcast with trailing
dimensions (``(..., 4)`` for quaternions, ``(..., 3)`` for vectors).
Scalar (non-batched) inputs follow the same broadcast rules, so ``(4,)``
quaternions and ``(3,)`` vectors return non-batched results naturally.

Naming rule: unprefixed functions are the Torch primitives
(``@torch.jit.script`` for hot-path use), NumPy primitives carry the
``np_`` prefix.  The NumPy primitives are always available; the Torch
primitives are compiled lazily on first access, and accessing them without
PyTorch raises a targeted :class:`ImportError`.  The base wheel therefore
stays NumPy-only: importing ``unisim.utils.math`` never imports or even
probes Torch, so the package import boundary stays Torch-free.

Isaac worker runtimes may load this module by file path inside external
Python 3.8 interpreters, so keep it self-contained and compatible with
Python 3.8 at runtime: standard library plus NumPy at import time, Torch
optional, no package imports, and TorchScript-compatible syntax in the
tensor section (static error messages, no star-args).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# NumPy primitives (``np_`` prefix)
# --------------------------------------------------------------------------- #


def np_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply broadcast-compatible wxyz quaternions with shape (..., 4)."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q1.shape} and {q2.shape}")

    lead_shape = np.broadcast_shapes(q1.shape[:-1], q2.shape[:-1])
    q1 = np.broadcast_to(q1, (*lead_shape, 4))
    q2 = np.broadcast_to(q2, (*lead_shape, 4))

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def np_quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternions (..., 4), w-first."""
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")
    conj = np.array(q, copy=True)
    conj[..., 1:] *= -1
    return conj


def np_quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate broadcast-compatible vectors (..., 3) by wxyz quaternions (..., 4)."""
    q = np.asarray(q)
    v = np.asarray(v)
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError(f"Expected q (..., 4) and v (..., 3), got {q.shape} and {v.shape}")

    lead_shape = np.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    q = np.broadcast_to(q, (*lead_shape, 4))
    v = np.broadcast_to(v, (*lead_shape, 3))

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)

    return v + np.stack(
        [
            w * tx + y * tz - z * ty,
            w * ty + z * tx - x * tz,
            w * tz + x * ty - y * tx,
        ],
        axis=-1,
    )


def np_quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate broadcast-compatible vectors into wxyz-quaternion frames."""
    return np_quat_apply(np_quat_conjugate(q), v)


def np_quat_canonicalize(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so the real part is non-negative."""
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")
    sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
    result: np.ndarray = q * sign
    return result


def np_quat_ensure_continuity(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs in a time sequence to keep adjacent dots non-negative."""
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {q.shape}")

    result = np.array(q, copy=True)
    for i in range(1, result.shape[0]):
        if float(np.dot(result[i - 1], result[i])) < 0.0:
            result[i] *= -1.0
    return result


def np_quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (..., 4), w-first, to axis-angle vectors (..., 3).

    Adapted from PyTorch3D. Uses atan2 + Taylor expansion for numerical
    stability near zero rotation.
    """
    q = np_quat_canonicalize(q)
    xyz = q[..., 1:]
    w = q[..., 0:1]
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half_angle = np.arctan2(norms, w)
    angle = 2.0 * half_angle
    small = np.abs(angle) < 1e-6
    safe_angle = np.where(small, 1.0, angle)
    sin_half_over_angle = np.where(
        small,
        0.5 - angle**2 / 48.0,
        np.sin(half_angle) / safe_angle,
    )
    axis_angle: np.ndarray = xyz / sin_half_over_angle
    return axis_angle


def np_quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Convert axis-angle pairs to unit quaternions (..., 4), w-first.

    ``angle`` has shape (...,) and ``axis`` has shape (..., 3); leading
    dimensions broadcast and axes are normalized internally.  Inverse of
    :func:`np_quat_to_axis_angle`.
    """
    angle = np.asarray(angle)
    axis = np.asarray(axis)
    if axis.shape[-1] != 3:
        raise ValueError(f"Expected axis last dimension 3, got {axis.shape}")
    lead_shape = np.broadcast_shapes(angle.shape, axis.shape[:-1])
    angle = np.broadcast_to(angle, lead_shape)
    axis = np.broadcast_to(axis, (*lead_shape, 3))
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * angle
    w = np.cos(half)[..., None]
    xyz = axis * np.sin(half)[..., None]
    return np.concatenate([w, xyz], axis=-1)


def np_quat_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity from a quaternion time sequence using shortest-arc diffs."""
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {q.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    rotations = np_quat_ensure_continuity(q)
    num_frames = rotations.shape[0]
    omega = np.zeros((num_frames, 3), dtype=rotations.dtype)
    if num_frames <= 1:
        return omega

    if num_frames == 2:
        q_rel = np_quat_mul(rotations[1], np_quat_conjugate(rotations[0]))
        q_rel = np_quat_canonicalize(q_rel)
        angvel = np_quat_to_axis_angle(q_rel) / dt
        omega[:] = angvel
        return omega

    q_prev = rotations[:-2]
    q_next = rotations[2:]
    q_rel = np_quat_mul(q_next, np_quat_conjugate(q_prev))
    q_rel = np_quat_canonicalize(q_rel)
    omega[1:-1] = np_quat_to_axis_angle(q_rel) / (2.0 * dt)
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    return omega


def np_quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert broadcast-compatible XYZ Euler angles to wxyz quaternions (..., 4)."""
    roll = np.asarray(roll)
    pitch = np.asarray(pitch)
    yaw = np.asarray(yaw)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.stack([w, x, y, z], axis=-1)


def np_quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular error magnitude between broadcast-compatible quaternions (..., 4)."""
    q_rel = np_quat_mul(q2, np_quat_conjugate(q1))
    q_rel = np_quat_canonicalize(q_rel)
    xyz_norm = np.linalg.norm(q_rel[..., 1:], axis=-1)
    w = np.clip(q_rel[..., 0], -1.0, 1.0)
    return 2.0 * np.arctan2(xyz_norm, w)


def np_quat_error_magnitude_squared(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Squared angular error for broadcast-compatible quaternions (..., 4)."""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q1.shape} and {q2.shape}")

    lead_shape = np.broadcast_shapes(q1.shape[:-1], q2.shape[:-1])
    q1 = np.broadcast_to(q1, (*lead_shape, 4))
    q2 = np.broadcast_to(q2, (*lead_shape, 4))

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Relative rotation q2 * conj(q1), without materializing the full quaternion.
    rel_w = np.abs(w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1)
    rel_x = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
    rel_y = -w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
    rel_z = -w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
    xyz_norm = np.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
    angle = 2.0 * np.arctan2(xyz_norm, np.clip(rel_w, -1.0, 1.0))
    return angle * angle


def np_yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    """Convert yaw angles (...,) to wxyz quaternions (..., 4)."""
    yaw = np.asarray(yaw)
    half = 0.5 * yaw
    return np.stack(
        [
            np.cos(half),
            np.zeros_like(half),
            np.zeros_like(half),
            np.sin(half),
        ],
        axis=-1,
    )


def np_yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    """Yaw angles (...,) from wxyz quaternions (..., 4)."""
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {quat.shape}")
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def np_yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternions from full wxyz quaternions (..., 4)."""
    return np_yaw_to_quat(np_yaw_from_quat(q))


def np_wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) into [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def np_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions (..., 4) to rotation matrices (..., 3, 3)."""
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1)
    row1 = np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1)
    row2 = np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def np_matrix_first_two_cols_from_quat(q: np.ndarray) -> np.ndarray:
    """Return flattened first two rotation-matrix columns for quaternions (..., 4).

    The output order matches ``np_matrix_from_quat(q)[..., :, :2].reshape(..., 6)``.
    """
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (xz - wy),
            2 * (yz + wx),
        ],
        axis=-1,
    )


def np_subtract_frame_transforms(
    pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative transform from frame 1 to frame 2 in frame-1 coordinates."""
    rel_pos = np_quat_apply_inverse(quat1, pos2 - pos1)
    rel_quat = np_quat_mul(np_quat_conjugate(quat1), quat2)
    return rel_pos, rel_quat


def np_subtract_anchor_frame_transforms(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body transforms relative to per-env anchors.

    ``anchor_pos``/``anchor_quat`` are shaped (N, 3)/(N, 4), while
    ``body_pos``/``body_quat`` are shaped (N, B, 3)/(N, B, 4).
    """
    aw = anchor_quat[:, None, 0]
    ax = anchor_quat[:, None, 1]
    ay = anchor_quat[:, None, 2]
    az = anchor_quat[:, None, 3]

    vx = body_pos[..., 0] - anchor_pos[:, None, 0]
    vy = body_pos[..., 1] - anchor_pos[:, None, 1]
    vz = body_pos[..., 2] - anchor_pos[:, None, 2]

    # Rotate by conj(anchor_quat) without materializing the conjugate quaternion.
    qx = -ax
    qy = -ay
    qz = -az
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    rel_pos = np.stack(
        [
            vx + aw * tx + qy * tz - qz * ty,
            vy + aw * ty + qz * tx - qx * tz,
            vz + aw * tz + qx * ty - qy * tx,
        ],
        axis=-1,
    )

    bw = body_quat[..., 0]
    bx = body_quat[..., 1]
    by = body_quat[..., 2]
    bz = body_quat[..., 3]
    rel_quat = np.stack(
        [
            aw * bw + ax * bx + ay * by + az * bz,
            aw * bx - ax * bw - ay * bz + az * by,
            aw * by + ax * bz - ay * bw - az * bx,
            aw * bz - ax * by + ay * bx - az * bw,
        ],
        axis=-1,
    )
    return rel_pos, rel_quat


# --------------------------------------------------------------------------- #
# Torch primitives (unprefixed, TorchScript-compiled lazily on first access)
# --------------------------------------------------------------------------- #

if TYPE_CHECKING:
    import torch

_TORCH_PRIMITIVE_IMPLS = {
    "axis_angle_from_quat": "_axis_angle_from_quat_impl",
    "matrix_first_two_cols_from_quat": "_matrix_first_two_cols_from_quat_impl",
    "matrix_from_quat": "_matrix_from_quat_impl",
    "quat_angular_velocity": "_quat_angular_velocity_impl",
    "quat_apply": "_quat_apply_impl",
    "quat_apply_inverse": "_quat_apply_inverse_impl",
    "quat_canonicalize": "_quat_canonicalize_impl",
    "quat_conjugate": "_quat_conjugate_impl",
    "quat_ensure_continuity": "_quat_ensure_continuity_impl",
    "quat_error_magnitude": "_quat_error_magnitude_impl",
    "quat_from_angle_axis": "_quat_from_angle_axis_impl",
    "quat_from_euler_xyz": "_quat_from_euler_xyz_impl",
    "quat_mul": "_quat_mul_impl",
    "quat_roll_pitch": "_quat_roll_pitch_impl",
    "quat_to_rot6": "_quat_to_rot6_impl",
    "wrap_to_pi": "_wrap_to_pi_impl",
    "yaw_from_quat": "_yaw_from_quat_impl",
    "yaw_quat": "_yaw_quat_impl",
    "yaw_to_quat": "_yaw_to_quat_impl",
}

def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=eps)

def _quat_mul_impl(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible wxyz quaternions with shape (..., 4)."""
    if left_wxyz.shape[-1] != 4 or right_wxyz.shape[-1] != 4:
        raise ValueError("quaternions must have final dimension 4")
    lw = left_wxyz[..., 0]
    lx = left_wxyz[..., 1]
    ly = left_wxyz[..., 2]
    lz = left_wxyz[..., 3]
    rw = right_wxyz[..., 0]
    rx = right_wxyz[..., 1]
    ry = right_wxyz[..., 2]
    rz = right_wxyz[..., 3]
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )

def _quat_conjugate_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Conjugate of unit quaternions (..., 4), w-first."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    return torch.cat((quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]), dim=-1)

def _quat_apply_impl(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate broadcast-compatible vectors (..., 3) by wxyz quaternions (..., 4).

    Uses the mjlab/IsaacLab cross-product form, extended with broadcasting.
    """
    if quat_wxyz.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("expected quaternion (..., 4) and vector (..., 3)")
    # Component-wise cross product so leading dims broadcast even when the
    # quaternion and vector carry different ranks (torch.cross cannot).
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    vx = vector[..., 0]
    vy = vector[..., 1]
    vz = vector[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack(
        (
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ),
        dim=-1,
    )

def _quat_apply_inverse_impl(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate broadcast-compatible vectors into wxyz-quaternion frames.

    Uses the mjlab/IsaacLab cross-product form, extended with broadcasting.
    """
    if quat_wxyz.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("expected quaternion (..., 4) and vector (..., 3)")
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    vx = vector[..., 0]
    vy = vector[..., 1]
    vz = vector[..., 2]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack(
        (
            vx - w * tx + y * tz - z * ty,
            vy - w * ty + z * tx - x * tz,
            vz - w * tz + x * ty - y * tx,
        ),
        dim=-1,
    )

def _quat_canonicalize_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Flip quaternion signs so the real part is non-negative."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    sign = torch.where(
        quat_wxyz[..., 0:1] < 0.0,
        -torch.ones_like(quat_wxyz),
        torch.ones_like(quat_wxyz),
    )
    return quat_wxyz * sign

def _quat_ensure_continuity_impl(quat: torch.Tensor) -> torch.Tensor:
    """Flip quaternion signs in a time sequence to keep adjacent dots non-negative."""
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("expected quaternion sequence with shape (T, 4)")
    if quat.shape[0] <= 1:
        return quat.clone()
    dots = (quat[1:] * quat[:-1]).sum(dim=-1)
    step_sign = torch.where(dots < 0.0, -torch.ones_like(dots), torch.ones_like(dots))
    one = torch.ones((1,), dtype=quat.dtype, device=quat.device)
    flip = torch.cat((one, torch.cumprod(step_sign, dim=0)), dim=0)
    return quat * flip.unsqueeze(-1)

def _axis_angle_from_quat_impl(quat_wxyz: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert unit wxyz quaternions (..., 4) to axis-angle vectors (..., 3).

    Follows the mjlab/IsaacLab canonicalized atan2/Taylor form so tensor
    observations preserve the established numerical semantics near zero angle.
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    quat = quat_wxyz * (1.0 - 2.0 * (quat_wxyz[..., 0:1] < 0.0))
    mag = torch.linalg.vector_norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # When the angle is too small, sin(half_angle)/angle is undefined;
    # fall back to the Taylor approximation 1/2 - angle**2/48.
    sin_half_over_angle = torch.where(
        torch.abs(angle) > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / sin_half_over_angle.unsqueeze(-1)

def _quat_from_angle_axis_impl(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle pairs to unit wxyz quaternions (..., 4).

    ``angle`` has shape (...,) and ``axis`` has shape (..., 3); leading
    dimensions broadcast and axes are normalized internally, matching the
    mjlab/IsaacLab form.  Inverse of :func:`axis_angle_from_quat`.
    """
    if axis.shape[-1] != 3:
        raise ValueError("axis must have final dimension 3")
    theta = (0.5 * angle).unsqueeze(-1)
    xyz = _normalize(axis) * torch.sin(theta)
    w = torch.cos(theta)
    return _normalize(torch.cat((w, xyz), dim=-1))

def _quat_angular_velocity_impl(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Estimate angular velocity from a quaternion time sequence (shortest-arc diffs)."""
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError("expected quaternion sequence with shape (T, 4)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    rotations = _quat_ensure_continuity_impl(quat)
    num_frames = rotations.shape[0]
    if num_frames <= 1:
        return torch.zeros((num_frames, 3), dtype=rotations.dtype, device=rotations.device)
    if num_frames == 2:
        q_rel = _quat_mul_impl(rotations[1], _quat_conjugate_impl(rotations[0]))
        q_rel = _quat_canonicalize_impl(q_rel)
        angvel = _axis_angle_from_quat_impl(q_rel) / dt
        return angvel.unsqueeze(0).expand(num_frames, 3).contiguous()
    q_rel = _quat_mul_impl(rotations[2:], _quat_conjugate_impl(rotations[:-2]))
    q_rel = _quat_canonicalize_impl(q_rel)
    inner = _axis_angle_from_quat_impl(q_rel) / (2.0 * dt)
    return torch.cat((inner[0:1], inner, inner[-1:]), dim=0)

def _quat_from_euler_xyz_impl(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Convert broadcast-compatible XYZ Euler angles to wxyz quaternions (..., 4)."""
    cos_roll = torch.cos(0.5 * roll)
    sin_roll = torch.sin(0.5 * roll)
    cos_pitch = torch.cos(0.5 * pitch)
    sin_pitch = torch.sin(0.5 * pitch)
    cos_yaw = torch.cos(0.5 * yaw)
    sin_yaw = torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        ),
        dim=-1,
    )

def _quat_roll_pitch_impl(quat_wxyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return roll and pitch in radians from broadcast-compatible wxyz quaternions."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(r21, r22)
    pitch = torch.atan2(-r20, torch.sqrt(torch.clamp(r21 * r21 + r22 * r22, min=0.0)))
    return roll, pitch

def _matrix_from_quat_impl(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions (..., 4) to rotation matrices (..., 3, 3).

    Uses the mjlab/PyTorch3D form, which normalizes internally so
    non-unit quaternions still produce proper rotation matrices.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    flat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return flat.reshape(quaternions.shape[:-1] + (3, 3))

def _matrix_first_two_cols_from_quat_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return flattened first two rotation-matrix columns for quaternions (..., 4).

    The output order matches ``_matrix_from_quat_impl(q)[..., :, :2].reshape(..., 6)``,
    including its internal normalization of non-unit quaternions.
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    r = quat_wxyz[..., 0]
    i = quat_wxyz[..., 1]
    j = quat_wxyz[..., 2]
    k = quat_wxyz[..., 3]
    two_s = 2.0 / (quat_wxyz * quat_wxyz).sum(dim=-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
        ),
        dim=-1,
    )

def _quat_to_rot6_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Flattened first two rotation-matrix columns (6-D rotation features)."""
    return _matrix_first_two_cols_from_quat_impl(quat_wxyz)

def _quat_error_magnitude_impl(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular error magnitude between broadcast-compatible quaternions (..., 4).

    Uses the mjlab/IsaacLab box-minus form: the norm of the axis-angle of
    the relative rotation.
    """
    axis_angle_error = _axis_angle_from_quat_impl(_quat_mul_impl(q1, _quat_conjugate_impl(q2)))
    return torch.linalg.vector_norm(axis_angle_error, dim=-1)

def _yaw_to_quat_impl(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles (...,) to wxyz quaternions (..., 4)."""
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)

def _yaw_from_quat_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw angles (...,) from wxyz quaternions (..., 4)."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quaternion must have final dimension 4")
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def _yaw_quat_impl(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternions from full wxyz quaternions (..., 4)."""
    return _yaw_to_quat_impl(_yaw_from_quat_impl(quat_wxyz))

def _wrap_to_pi_impl(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) into [-pi, pi)."""
    pi = 3.141592653589793
    return torch.remainder(angle + pi, 2.0 * pi) - pi




def _load_torch_primitives() -> None:
    """Compile and publish the Torch primitives on first access.

    Runs only when an unprefixed (Torch) name is requested, so importing this
    module never probes Torch and the package import boundary stays Torch-free.
    """
    global torch
    try:
        import torch
    except ImportError:
        raise ImportError(
            "the unprefixed primitives in unisim.utils.math require PyTorch; "
            "install torch or use the np_-prefixed NumPy counterparts"
        ) from None
    for public_name, impl_name in _TORCH_PRIMITIVE_IMPLS.items():
        globals()[public_name] = torch.jit.script(globals()[impl_name])


def __getattr__(name: str) -> Any:
    if name in _TORCH_PRIMITIVE_IMPLS:
        _load_torch_primitives()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "np_matrix_first_two_cols_from_quat",
    "np_matrix_from_quat",
    "np_quat_angular_velocity",
    "np_quat_apply",
    "np_quat_apply_inverse",
    "np_quat_canonicalize",
    "np_quat_conjugate",
    "np_quat_ensure_continuity",
    "np_quat_error_magnitude",
    "np_quat_error_magnitude_squared",
    "np_quat_from_angle_axis",
    "np_quat_from_euler_xyz",
    "np_quat_mul",
    "np_quat_to_axis_angle",
    "np_subtract_anchor_frame_transforms",
    "np_subtract_frame_transforms",
    "np_wrap_to_pi",
    "np_yaw_from_quat",
    "np_yaw_quat",
    "np_yaw_to_quat",
    *_TORCH_PRIMITIVE_IMPLS,
]
