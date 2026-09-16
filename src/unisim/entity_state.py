"""Array-only mapping between public entity patches and generalized state.

Adapters supply coherent snapshots and perform the native commit. This module
does not write backend state or infer native actor/DoF addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unisim.entities import SceneResetRequest
    from unisim.scene_layout import CompiledSceneLayout, EntityLayout


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world batch of vectors by unit wxyz quaternions."""
    xyz = quaternion[..., 1:]
    tangent = 2 * np.cross(xyz, vector)
    return vector + quaternion[..., :1] * tangent + np.cross(xyz, tangent)


def inverse_rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    inverse = quaternion.copy()
    inverse[..., 1:] *= -1
    return rotate_vector(inverse, vector)


def entity_state_snapshot(
    entity: EntityLayout,
    qpos: np.ndarray,
    qvel: np.ndarray,
    root_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Detached world-frame root state plus entity-local packed joint state."""
    if entity.root_mode == "floating":
        pose = qpos[:, entity.root_qpos_indices].copy()
        velocity = qvel[:, entity.root_qvel_indices].copy()
        velocity[:, 3:] = rotate_vector(pose[:, 3:], velocity[:, 3:])
    else:
        if root_state is None:
            raise NotImplementedError("fixed/kinematic entity state requires native root readback")
        pose = root_state[:, :7].copy()
        velocity = root_state[:, 7:13].copy()
    joint_qpos = tuple(i for joint in entity.joints for i in joint.qpos_indices)
    joint_qvel = tuple(i for joint in entity.joints for i in joint.qvel_indices)
    return {
        "root_pose": pose,
        "root_velocity": velocity,
        "joint_positions": qpos[:, joint_qpos].copy(),
        "joint_velocities": qvel[:, joint_qvel].copy(),
    }


@dataclass(frozen=True)
class PreparedSceneReset:
    """Owned scratch rows and masks, complete before any native submission."""

    env_ids: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    roots: np.ndarray
    qpos_mask: np.ndarray
    qvel_mask: np.ndarray
    root_mask: np.ndarray
    entity_names: tuple[str, ...]


def prepare_scene_reset(
    layout: CompiledSceneLayout,
    request: SceneResetRequest,
    qpos: np.ndarray,
    qvel: np.ndarray,
    roots: np.ndarray,
) -> PreparedSceneReset:
    """Validate all writes and build a complete selected-row native-independent plan.

    Inputs are coherent full-batch snapshots. A root pose-only patch preserves
    world angular velocity, so it may update generalized body-frame angular
    velocity columns. Nothing mutates input arrays, including on failure.
    """
    if qpos.ndim != 2 or qpos.shape[1] != layout.nq:
        raise ValueError("qpos snapshot differs from the scene layout")
    num_envs = qpos.shape[0]
    if qvel.shape != (num_envs, layout.nv):
        raise ValueError("qvel snapshot differs from the scene layout")
    if roots.shape != (num_envs, len(layout.entities), 13):
        raise ValueError("root snapshot differs from the scene layout")
    if any(
        array.dtype.kind != "f" or not np.isfinite(array).all() for array in (qpos, qvel, roots)
    ):
        raise ValueError("scene snapshots must contain finite floating-point values")
    bound = layout.validate_reset(request, num_envs=num_envs)
    ids = np.asarray(bound.env_ids, dtype=np.int32)
    positions = qpos[ids].copy()
    velocities = qvel[ids].copy()
    root_rows = roots[ids].copy()
    qpos_mask = np.zeros(layout.nq, dtype=np.uint8)
    qvel_mask = np.zeros(layout.nv, dtype=np.uint8)
    root_mask = np.zeros((len(layout.entities), 2), dtype=np.uint8)
    entity_ids = {entity.name: i for i, entity in enumerate(layout.entities)}
    for item in bound.patches:
        entity, patch = item.entity, item.patch
        index = entity_ids[entity.name]
        pose, velocity = patch.root_pose, patch.root_velocity
        if pose is not None or velocity is not None:
            if entity.root_mode == "floating":
                # Trust generalized state for floating roots, not a potentially
                # lagging native body cache (the M0 state boundary still applies).
                current = entity_state_snapshot(entity, positions, velocities)
                if pose is None:
                    pose = current["root_pose"]
                if velocity is None:
                    velocity = current["root_velocity"]
                positions[:, entity.root_qpos_indices] = pose
                local_velocity = np.array(velocity, dtype=velocities.dtype, copy=True)
                local_velocity[:, 3:] = inverse_rotate_vector(pose[:, 3:], velocity[:, 3:])
                velocities[:, entity.root_qvel_indices] = local_velocity
                root_rows[:, index, :7] = pose
                root_rows[:, index, 7:] = velocity
                if patch.root_pose is not None:
                    qpos_mask[list(entity.root_qpos_indices)] = 1
                    root_mask[index, 0] = 1
                qvel_mask[list(entity.root_qvel_indices)] = 1
                root_mask[index, 1] = 1
            else:
                # Layout validation only permits kinematic pose writes here.
                assert pose is not None
                root_rows[:, index, :7] = pose
                root_mask[index, 0] = 1
        if patch.joint_positions is not None:
            positions[:, item.joint_qpos_indices] = patch.joint_positions
            qpos_mask[list(item.joint_qpos_indices)] = 1
        if patch.joint_velocities is not None:
            velocities[:, item.joint_qvel_indices] = patch.joint_velocities
            qvel_mask[list(item.joint_qvel_indices)] = 1
    return PreparedSceneReset(
        ids,
        positions,
        velocities,
        root_rows,
        qpos_mask,
        qvel_mask,
        root_mask,
        tuple(item.entity.name for item in bound.patches),
    )
