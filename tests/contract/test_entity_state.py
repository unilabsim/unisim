"""Independent frame and isolation oracles for entity reset array preparation."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout

_S = np.sqrt(0.5)
_IDENTITY = [1.0, 0.0, 0.0, 0.0]
_Z90 = [_S, 0.0, 0.0, _S]
_X90 = [_S, _S, 0.0, 0.0]


def _scene() -> CompiledSceneLayout:
    robot = EntityLayout(
        "robot",
        "articulation",
        "floating",
        "base",
        ("base", "finger"),
        (1, 2),
        (None, "base"),
        (JointLayout("hinge", "hinge", (14,), (12,), "finger"),),
        ("drive",),
        ("hinge",),
        (0,),
        (0, 2, 4, 6, 8, 10, 12),
        (0, 2, 4, 6, 8, 10),
    )
    obj = EntityLayout(
        "object",
        "articulation",
        "floating",
        "base",
        ("base", "lid"),
        (3, 4),
        (None, "base"),
        (
            JointLayout("hinge", "hinge", (15,), (13,), "lid"),
            JointLayout("ball", "ball", (19, 16, 18, 17), (16, 14, 15), "lid"),
        ),
        (),
        (),
        (),
        (1, 3, 5, 7, 9, 11, 13),
        (1, 3, 5, 7, 9, 11),
    )
    mirror = EntityLayout(
        "target",
        "rigid",
        "kinematic",
        "base",
        ("base",),
        (5,),
        (None,),
        (),
        (),
        (),
        (),
    )
    return CompiledSceneLayout((robot, obj, mirror), nq=20, nv=17, nu=1, nbody=6)


def _snapshots(layout):
    qpos = np.zeros((5, 20))
    qvel = np.zeros((5, 17))
    roots = np.zeros((5, 3, 13))
    for env in range(5):
        robot_pose = [env + 1.0, env + 2.0, env + 3.0, *_Z90]
        object_pose = [env + 11.0, env + 12.0, env + 13.0, *_X90]
        qpos[env, [0, 2, 4, 6, 8, 10, 12]] = robot_pose
        qpos[env, [1, 3, 5, 7, 9, 11, 13]] = object_pose
        qpos[env, [14, 15]] = [env + 0.1, env + 0.2]
        qpos[env, [19, 16, 18, 17]] = _IDENTITY
        qvel[env, [0, 2, 4, 6, 8, 10]] = [env + 1, env + 2, env + 3, 1, 2, 3]
        qvel[env, [1, 3, 5, 7, 9, 11]] = [env + 4, env + 5, env + 6, 4, 5, 6]
        qvel[env, [12, 13, 16, 14, 15]] = [env + 0.3, env + 0.4, 7, 8, 9]
        # Hand rotations: Rz(90)*(1,2,3)=(-2,1,3); Rx(90)*(4,5,6)=(4,-6,5).
        roots[env, 0] = [*robot_pose, env + 1, env + 2, env + 3, -2, 1, 3]
        roots[env, 1] = [*object_pose, env + 4, env + 5, env + 6, 4, -6, 5]
        roots[env, 2, :7] = [env + 21, env + 22, env + 23, *_IDENTITY]
    assert qpos.shape[1] == layout.nq and qvel.shape[1] == layout.nv
    return qpos, qvel, roots


def _unchanged(actual, original) -> None:
    for current, before in zip(actual, original, strict=True):
        np.testing.assert_array_equal(current, before)


def test_pose_only_reset_preserves_world_omega_and_other_root_on_unsorted_rows() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    poses = np.array([[101, 102, 103, *_IDENTITY], [201, 202, 203, *_X90]])
    request = SceneResetRequest((4, 1), (EntityStatePatch("robot", root_pose=poses),))
    prepared = prepare_scene_reset(layout, request, *arrays)
    np.testing.assert_array_equal(prepared.env_ids, [4, 1])
    np.testing.assert_array_equal(prepared.qpos[:, [0, 2, 4, 6, 8, 10, 12]], poses)
    # Ridentity^-1*(-2,1,3)=(-2,1,3), Rx(90)^-1*(-2,1,3)=(-2,3,-1).
    np.testing.assert_allclose(prepared.qvel[:, [6, 8, 10]], [[-2, 1, 3], [-2, 3, -1]], atol=1e-14)
    np.testing.assert_allclose(prepared.roots[:, 0, 10:13], [[-2, 1, 3], [-2, 1, 3]], atol=1e-14)
    np.testing.assert_array_equal(prepared.qvel[:, [0, 2, 4]], [[5, 6, 7], [2, 3, 4]])
    np.testing.assert_array_equal(
        prepared.qpos[:, [1, 3, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19]],
        before[0][[4, 1]][:, [1, 3, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19]],
    )
    np.testing.assert_array_equal(
        prepared.qvel[:, [1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16]],
        before[1][[4, 1]][:, [1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16]],
    )
    np.testing.assert_array_equal(prepared.roots[:, 1:], before[2][[4, 1]][:, 1:])
    np.testing.assert_array_equal(np.flatnonzero(prepared.qpos_mask), [0, 2, 4, 6, 8, 10, 12])
    np.testing.assert_array_equal(np.flatnonzero(prepared.qvel_mask), [0, 2, 4, 6, 8, 10])
    np.testing.assert_array_equal(prepared.root_mask, [[1, 1], [0, 0], [0, 0]])
    _unchanged(arrays, before)


def test_velocity_only_reset_preserves_pose_and_converts_world_to_local() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    velocities = np.array([[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], [4.0, 5.0, 6.0, -7.0, -8.0, -9.0]])
    request = SceneResetRequest((3, 0), (EntityStatePatch("object", root_velocity=velocities),))
    prepared = prepare_scene_reset(layout, request, *arrays)
    np.testing.assert_array_equal(prepared.qpos, before[0][[3, 0]])
    np.testing.assert_allclose(prepared.qvel[:, [7, 9, 11]], [[7, 9, -8], [-7, -9, 8]], atol=1e-14)
    np.testing.assert_array_equal(prepared.qvel[:, [1, 3, 5]], velocities[:, :3])
    np.testing.assert_array_equal(prepared.roots[:, 1, :7], before[2][[3, 0], 1, :7])
    np.testing.assert_array_equal(prepared.roots[:, 1, 7:], velocities)
    np.testing.assert_array_equal(prepared.roots[:, 0], before[2][[3, 0], 0])
    assert not prepared.qpos_mask.any()
    np.testing.assert_array_equal(prepared.root_mask, [[0, 0], [0, 1], [0, 0]])
    _unchanged(arrays, before)


def test_floating_reset_uses_generalized_state_when_native_root_cache_is_old() -> None:
    layout = _scene()
    qpos, qvel, roots = _snapshots(layout)
    roots[:, 0, 10:] = 999.0
    request = SceneResetRequest(
        (2,), (EntityStatePatch("robot", root_pose=np.array([[1, 2, 3, *_IDENTITY]])),)
    )
    prepared = prepare_scene_reset(layout, request, qpos, qvel, roots)
    np.testing.assert_allclose(prepared.roots[0, 0, 10:], [-2, 1, 3], atol=1e-14)
    np.testing.assert_allclose(prepared.qvel[0, [6, 8, 10]], [-2, 1, 3], atol=1e-14)


def test_integer_velocity_input_does_not_truncate_rotated_generalized_velocity() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    # Rz(45)^-1 * world x = (sqrt(1/2), -sqrt(1/2), 0), never (0, 0, 0).
    pose = np.array([[1.0, 2.0, 3.0, np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]])
    velocity = np.array([[0, 0, 0, 1, 0, 0]], dtype=np.int64)
    request = SceneResetRequest(
        (4,), (EntityStatePatch("robot", root_pose=pose, root_velocity=velocity),)
    )
    prepared = prepare_scene_reset(layout, request, *arrays)
    np.testing.assert_allclose(prepared.qvel[0, [6, 8, 10]], [_S, -_S, 0], atol=1e-14)


def test_reordered_joint_patch_maps_ball_and_passive_hinge_without_touching_roots() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    positions = np.array([[*_Z90, 1.2], [*_X90, -0.3]])
    velocities = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    request = SceneResetRequest(
        (4, 1),
        (
            EntityStatePatch(
                "object",
                joint_names=("ball", "hinge"),
                joint_positions=positions,
                joint_velocities=velocities,
            ),
        ),
    )
    prepared = prepare_scene_reset(layout, request, *arrays)
    np.testing.assert_array_equal(prepared.qpos[:, [19, 16, 18, 17, 15]], positions)
    np.testing.assert_array_equal(prepared.qvel[:, [16, 14, 15, 13]], velocities)
    np.testing.assert_array_equal(prepared.qpos[:, :15], before[0][[4, 1], :15])
    np.testing.assert_array_equal(prepared.qvel[:, :13], before[1][[4, 1], :13])
    np.testing.assert_array_equal(prepared.roots, before[2][[4, 1]])
    assert not prepared.root_mask.any()
    assert layout.nu == 1
    _unchanged(arrays, before)


def test_kinematic_pose_patch_only_changes_the_selected_mirror_rows() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    poses = np.array([[31, 32, 33, *_Z90], [41, 42, 43, *_X90]])
    request = SceneResetRequest((4, 1), (EntityStatePatch("target", root_pose=poses),))
    prepared = prepare_scene_reset(layout, request, *arrays)
    np.testing.assert_array_equal(prepared.qpos, before[0][[4, 1]])
    np.testing.assert_array_equal(prepared.qvel, before[1][[4, 1]])
    np.testing.assert_array_equal(prepared.roots[:, :2], before[2][[4, 1]][:, :2])
    np.testing.assert_array_equal(prepared.roots[:, 2, :7], poses)
    np.testing.assert_array_equal(prepared.roots[:, 2, 7:], before[2][[4, 1], 2, 7:])
    np.testing.assert_array_equal(prepared.root_mask, [[0, 0], [0, 0], [1, 0]])
    assert not prepared.qpos_mask.any() and not prepared.qvel_mask.any()
    _unchanged(arrays, before)


@pytest.mark.parametrize("bad", ["width", "ball_quaternion", "unknown_joint"])
def test_late_bad_patch_leaves_all_input_state_unchanged(bad) -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    good = EntityStatePatch("robot", root_pose=np.array([[1, 2, 3, *_IDENTITY]]))
    if bad == "width":
        invalid = EntityStatePatch("object", joint_positions=np.zeros((1, 2)))
    elif bad == "ball_quaternion":
        invalid = EntityStatePatch("object", joint_positions=np.zeros((1, 5)))
    else:
        invalid = EntityStatePatch(
            "object", joint_names=("missing",), joint_positions=np.zeros((1, 1))
        )
    with pytest.raises(ValueError):
        prepare_scene_reset(layout, SceneResetRequest((4,), (good, invalid)), *arrays)
    _unchanged(arrays, before)


@pytest.mark.parametrize("entity_name", ["robot", "object", "target"])
def test_entity_snapshot_is_detached_and_uses_world_frame_hand_rotation(entity_name) -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    index = {"robot": 0, "object": 1, "target": 2}[entity_name]
    state = entity_state_snapshot(
        layout.get_entity(entity_name), arrays[0], arrays[1], arrays[2][:, index]
    )
    expected_omega = {"robot": [-2, 1, 3], "object": [4, -6, 5], "target": [0, 0, 0]}
    np.testing.assert_allclose(
        state["root_velocity"][:, 3:], np.tile(expected_omega[entity_name], (5, 1)), atol=1e-14
    )
    if entity_name == "object":
        # Snapshot order is declared hinge then ball, unlike the reordered write above.
        np.testing.assert_array_equal(state["joint_positions"][0], [0.2, *_IDENTITY])
        np.testing.assert_array_equal(state["joint_velocities"][0], [0.4, 7, 8, 9])
    for values in state.values():
        assert not any(np.shares_memory(values, array) for array in arrays)
        values[...] = -123.0
    _unchanged(arrays, before)


def test_prepared_reset_scratch_can_be_changed_without_mutating_source_snapshots() -> None:
    layout = _scene()
    arrays = _snapshots(layout)
    before = tuple(array.copy() for array in arrays)
    request = SceneResetRequest(
        (2,), (EntityStatePatch("robot", joint_positions=np.array([[3.0]])),)
    )
    prepared = prepare_scene_reset(layout, request, *arrays)
    for scratch in (prepared.qpos, prepared.qvel, prepared.roots):
        assert not any(np.shares_memory(scratch, array) for array in arrays)
        scratch[...] = -999.0
    _unchanged(arrays, before)


@pytest.mark.parametrize("field", ["joint_positions", "joint_velocities", "root_velocity"])
def test_target_dtype_overflow_is_rejected_before_native_submission(field) -> None:
    layout = _scene()
    arrays = tuple(array.astype(np.float32) for array in _snapshots(layout))
    before = tuple(array.copy() for array in arrays)
    width = 6 if field == "root_velocity" else 1
    patch = EntityStatePatch("robot", **{field: np.full((1, width), 1e100)})
    with pytest.raises(ValueError, match="range"):
        prepare_scene_reset(layout, SceneResetRequest((2,), (patch,)), *arrays)
    _unchanged(arrays, before)
