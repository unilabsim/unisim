"""Pure-NumPy forward kinematics over a scanned MJCF kinematic tree.

PhysX-based workers (IsaacGym Preview 4) cannot refresh rigid-body link poses
without advancing physics: after ``set_actor_root_state_tensor_indexed`` /
``set_dof_state_tensor_indexed`` the rigid-body state tensor keeps the
pre-write poses until the first ``simulate``.  Publishing that buffer as body
state serves stale rows after every reset (issue #141).  The worker instead
overlays body state with exact MuJoCo-semantics forward kinematics computed
from the reset generalized state, until the first physics step makes the
native tensor authoritative again.

The kinematic tables are scanned host-side (``scan_scene_kinematics`` in
``sensors.py``) and cross the INIT payload as plain lists.  This module is
loaded by file path inside external Python 3.8 workers, so keep it
self-contained: standard library plus NumPy only, no package imports.

MJCF FK conventions (validated numerically against ``mj_forward``):

- Bodies are ordered depth-first pre-order under ``worldbody`` (parents
  before children), matching the host ``body_names`` scan.
- A body's ``pos``/``quat`` offset is applied in the parent frame; the joint
  rotation acts in the body reference frame, i.e.
  ``q_body = q_parent * body_quat * rot(axis, q)`` and the body origin
  ``x_body = x_parent + R_parent @ body_pos`` does not move with the body's
  own joint.
- The joint axis is expressed in the body reference frame.
- The free root reads ``qpos[:7]``/``qvel[:6]`` directly; the free-joint
  angular velocity is body-local (MuJoCo convention) and the linear velocity
  is the world-frame velocity of the body frame origin.
- Output rows follow the canonical 13-wide state layout: position, wxyz
  quaternion, world linear velocity of the body frame origin, world angular
  velocity.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

SCHEMA_VERSION = 1

JOINT_NONE = 0
JOINT_HINGE = 1
JOINT_SLIDE = 2


def _as_arrays(tables: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the wire payload into NumPy arrays, failing closed on drift."""
    if not isinstance(tables, dict) or tables.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported MJCF kinematics payload schema")
    body_names = [str(name) for name in tables.get("body_names") or ()]
    joint_names = [str(name) for name in tables.get("joint_names") or ()]
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("MJCF kinematics body names must be nonempty and unique")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("MJCF kinematics joint names must be unique")
    nbody = len(body_names)
    arrays = {
        "body_names": body_names,
        "joint_names": joint_names,
        "body_parent": np.asarray(tables.get("body_parent"), dtype=np.intp),
        "body_pos": np.asarray(tables.get("body_pos"), dtype=np.float64),
        "body_quat": np.asarray(tables.get("body_quat"), dtype=np.float64),
        "body_joint_kind": np.asarray(tables.get("body_joint_kind"), dtype=np.intp),
        "body_joint_axis": np.asarray(tables.get("body_joint_axis"), dtype=np.float64),
        "body_joint_column": np.asarray(tables.get("body_joint_column"), dtype=np.intp),
    }
    expected = {
        "body_parent": (nbody,),
        "body_pos": (nbody, 3),
        "body_quat": (nbody, 4),
        "body_joint_kind": (nbody,),
        "body_joint_axis": (nbody, 3),
        "body_joint_column": (nbody,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"MJCF kinematics field {key!r} must be finite with shape {shape}")
    free_root = tables.get("free_root", -1)
    if isinstance(free_root, bool) or not isinstance(free_root, int) or not -1 <= free_root < nbody:
        raise ValueError("MJCF kinematics free_root must be -1 or a valid body index")
    arrays["free_root"] = free_root
    for index in range(nbody):
        parent = int(arrays["body_parent"][index])
        if parent >= index:
            raise ValueError("MJCF kinematics parents must precede children (document order)")
        kind = int(arrays["body_joint_kind"][index])
        column = int(arrays["body_joint_column"][index])
        if kind not in (JOINT_NONE, JOINT_HINGE, JOINT_SLIDE):
            raise ValueError("MJCF kinematics joint kinds are limited to none/hinge/slide")
        if index == free_root:
            if kind != JOINT_NONE or parent != -1:
                raise ValueError("MJCF kinematics free root must be a top-level jointless body")
        elif kind == JOINT_NONE:
            if column != -1:
                raise ValueError("MJCF kinematics jointless bodies must not carry a qpos column")
        elif not 7 <= column < 7 + len(joint_names):
            raise ValueError("MJCF kinematics joint column out of range")
    return arrays


def prepare_kinematics(tables: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and convert one wire payload into the reusable NumPy layout."""
    return _as_arrays(tables)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = quat_wxyz[..., 0:1]
    u = quat_wxyz[..., 1:4]
    uv = np.cross(u, vec)
    uuv = np.cross(u, uv)
    return vec + 2.0 * (w * uv + uuv)


def forward_prepared_kinematics(
    kin: Dict[str, Any], qpos: np.ndarray, qvel: np.ndarray
) -> np.ndarray:
    """Return ``(num_envs, nbody, 13)`` world body states for generalized states.

    ``qpos``/``qvel`` follow the legacy MJCF layout: 7 free-root columns
    (xyz + wxyz) plus one column per single-DoF joint in document order, and
    6 root velocity columns (world linear, body-local angular) plus one per
    joint.  Rows are computed in float64; callers cast to the slot dtype.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    num_joints = len(kin["joint_names"])
    if qpos.ndim != 2 or qpos.shape[1] != 7 + num_joints:
        raise ValueError(
            f"kinematics qpos must have shape (num_envs, {7 + num_joints}), got {qpos.shape}"
        )
    if qvel.shape != (qpos.shape[0], 6 + num_joints):
        raise ValueError(
            f"kinematics qvel must have shape ({qpos.shape[0]}, {6 + num_joints}), "
            f"got {qvel.shape}"
        )
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("kinematics generalized state must be finite")
    free_root = kin["free_root"]
    if free_root < 0:
        raise ValueError("kinematics FK requires a free root (floating base)")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("kinematics qpos root quaternions must be unit wxyz")

    num_envs, nbody = qpos.shape[0], len(kin["body_names"])
    pos = np.zeros((num_envs, nbody, 3))
    quat = np.zeros((num_envs, nbody, 4))
    lin = np.zeros((num_envs, nbody, 3))
    ang = np.zeros((num_envs, nbody, 3))
    quat[:, :, 0] = 1.0
    parents = kin["body_parent"]
    for index in range(nbody):
        if index == free_root:
            pos[:, index] = qpos[:, 0:3]
            quat[:, index] = qpos[:, 3:7]
            lin[:, index] = qvel[:, 0:3]
            ang[:, index] = _quat_rotate(qpos[:, 3:7], qvel[:, 3:6])
            continue
        parent = int(parents[index])
        offset = _quat_rotate(quat[:, parent], kin["body_pos"][index])
        pos[:, index] = pos[:, parent] + offset
        # The joint rotation acts in the body reference frame: offset first,
        # then the rotation about the (offset-frame) axis.
        reference = _quat_mul(quat[:, parent], kin["body_quat"][index])
        kind = int(kin["body_joint_kind"][index])
        if kind == JOINT_NONE:
            quat[:, index] = reference
            ang[:, index] = ang[:, parent]
            lin[:, index] = lin[:, parent] + np.cross(ang[:, parent], offset)
            continue
        column = int(kin["body_joint_column"][index])
        value = qpos[:, column]
        rate = qvel[:, column - 1]
        axis_world = _quat_rotate(reference, kin["body_joint_axis"][index])
        if kind == JOINT_HINGE:
            sine = np.sin(0.5 * value)
            joint_quat = np.concatenate(
                (
                    np.cos(0.5 * value)[:, None],
                    sine[:, None] * kin["body_joint_axis"][index][None, :],
                ),
                axis=-1,
            )
            quat[:, index] = _quat_mul(reference, joint_quat)
            ang[:, index] = ang[:, parent] + axis_world * rate[:, None]
            lin[:, index] = lin[:, parent] + np.cross(ang[:, parent], offset)
        else:
            # A slide joint translates the body origin along the reference axis.
            quat[:, index] = reference
            ang[:, index] = ang[:, parent]
            lever = offset + axis_world * value[:, None]
            pos[:, index] = pos[:, parent] + lever
            lin[:, index] = lin[:, parent] + np.cross(ang[:, parent], lever) + (
                axis_world * rate[:, None]
            )

    state = np.zeros((num_envs, nbody, 13), dtype=np.float64)
    state[:, :, 0:3] = pos
    state[:, :, 3:7] = quat
    state[:, :, 7:10] = lin
    state[:, :, 10:13] = ang
    return state


def forward_kinematics(tables: Dict[str, Any], qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Run FK after validating one wire payload; hot owners pre-prepare instead."""
    return forward_prepared_kinematics(prepare_kinematics(tables), qpos, qvel)


__all__: List[str] = [
    "JOINT_HINGE",
    "JOINT_NONE",
    "JOINT_SLIDE",
    "SCHEMA_VERSION",
    "forward_prepared_kinematics",
    "forward_kinematics",
    "prepare_kinematics",
]
