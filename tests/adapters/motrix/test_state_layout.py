"""Single-MJCF native generalized-state layout regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def _mixed_model(*, root_first: bool) -> str:
    object_body = (
        "<body name='object' pos='1 2 3'>"
        "<freejoint name='object_free'/>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<geom type='sphere' size='.05' contype='0' conaffinity='0'/>"
        "</body>"
    )
    hand_body = (
        "<body name='hand' pos='0 0 .5'>"
        "<joint name='joint0' type='hinge' axis='1 0 0'/>"
        "<joint name='joint1' type='hinge' axis='0 1 0'/>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<geom type='sphere' size='.05' contype='0' conaffinity='0'/>"
        "</body>"
    )
    bodies = (object_body, hand_body) if root_first else (hand_body, object_body)
    root_qpos = [4.0, 5.0, 6.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
    hand_qpos = [0.75, -0.5]
    key_qpos = root_qpos + hand_qpos if root_first else hand_qpos + root_qpos
    return (
        "<mujoco model='motrix-state-layout'>"
        "<option timestep='.002' gravity='0 0 0'/>"
        f"<worldbody>{''.join(bodies)}</worldbody>"
        "<actuator>"
        "<motor name='joint0' joint='joint0'/>"
        "<motor name='joint1' joint='joint1'/>"
        "</actuator>"
        f"<keyframe><key name='home' qpos='{' '.join(str(v) for v in key_qpos)}'/></keyframe>"
        "</mujoco>"
    )


@pytest.mark.parametrize("root_first", [False, True])
def test_single_mjcf_public_state_uses_native_mjcf_order(
    tmp_path: Path, root_first: bool
) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(_mixed_model(root_first=root_first), encoding="utf-8")
    backend = MotrixBackend(
        SceneCfg(str(model_path)), num_envs=2, sim_dt=0.002, base_name="object"
    )
    try:
        joint_names = ("joint0", "joint1")
        widths = backend.get_public_state_widths()
        assert (widths.nq, widths.nv) == (9, 8)
        joint_qpos = backend.get_joint_state_qpos_indices(joint_names)
        joint_qvel = backend.get_joint_state_qvel_indices(joint_names)
        root_layout = backend.get_root_state_layout("object")
        expected_joint_qpos = [7, 8] if root_first else [0, 1]
        expected_joint_qvel = [6, 7] if root_first else [0, 1]
        expected_root_qpos = tuple(range(7)) if root_first else tuple(range(2, 9))
        expected_root_qvel = tuple(range(6)) if root_first else tuple(range(2, 8))
        np.testing.assert_array_equal(joint_qpos, expected_joint_qpos)
        np.testing.assert_array_equal(joint_qvel, expected_joint_qvel)
        assert root_layout.qpos_indices == expected_root_qpos
        assert root_layout.qvel_indices == expected_root_qvel

        expected_default = np.zeros(9, dtype=np.float32)
        expected_default[np.asarray(expected_root_qpos)] = [1, 2, 3, 1, 0, 0, 0]
        np.testing.assert_allclose(backend.get_default_qpos(), expected_default, atol=1e-6)
        np.testing.assert_allclose(backend.get_default_dof_pos(), [0.0, 0.0], atol=1e-6)
        expected_keyframe = np.zeros(9, dtype=np.float32)
        expected_keyframe[np.asarray(expected_root_qpos)] = [
            4.0,
            5.0,
            6.0,
            0.7071067811865476,
            0.7071067811865476,
            0.0,
            0.0,
        ]
        expected_keyframe[joint_qpos] = [0.75, -0.5]
        keyframe = backend.get_keyframe_qpos("home")
        np.testing.assert_allclose(keyframe, expected_keyframe, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(keyframe[list(expected_root_qpos)[3:7]]), 1.0)

        row = np.asarray([1], dtype=np.intp)
        qpos = np.zeros((1, 9), dtype=np.float32)
        qpos[0, joint_qpos] = [1.25, -1.5]
        qpos[0, list(expected_root_qpos)] = [4, 5, 6, 0.5, 0.5, 0.5, -0.5]
        qvel = np.zeros((1, 8), dtype=np.float32)
        qvel[0, joint_qvel] = [0.7, -0.8]
        qvel[0, list(expected_root_qvel)] = [1, 2, 3, 0.1, 0.2, 0.3]
        backend._host_set_state(row, qpos, qvel)

        state = backend._host_get_state()
        np.testing.assert_allclose(state["qpos"][row], qpos, atol=1e-6)
        np.testing.assert_allclose(state["qvel"][row], qvel, atol=1e-6)

        native_qpos = np.asarray(backend.data.dof_pos, dtype=np.float32)
        native_qvel = np.asarray(backend.data.dof_vel, dtype=np.float32)
        np.testing.assert_allclose(native_qpos[row][:, joint_qpos], qpos[:, joint_qpos])
        np.testing.assert_allclose(native_qvel[row][:, joint_qvel], qvel[:, joint_qvel])
        np.testing.assert_allclose(
            native_qpos[row][:, list(expected_root_qpos)],
            [[4.0, 5.0, 6.0, 0.5, 0.5, -0.5, 0.5]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            native_qvel[row][:, list(expected_root_qvel)], qvel[:, list(expected_root_qvel)]
        )

        backend._host_step(np.zeros((2, 2), dtype=np.float32))
        after_step = backend._host_get_state()
        np.testing.assert_allclose(
            after_step["qpos"][row][:, joint_qpos],
            np.asarray(backend.data.dof_pos)[row][:, joint_qpos],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            after_step["qvel"][row][:, joint_qvel],
            np.asarray(backend.data.dof_vel)[row][:, joint_qvel],
            atol=1e-6,
        )

        snapshot = backend.get_physics_state().copy()
        replacement = np.full_like(qpos, -3.0)
        replacement[:, list(expected_root_qpos)[3:7]] = [1.0, 0.0, 0.0, 0.0]
        backend._host_set_state(row, replacement, np.full_like(qvel, 0.5))
        backend.set_physics_state(snapshot)
        np.testing.assert_allclose(backend.get_physics_state(), snapshot, atol=1e-6)
    finally:
        backend.close()


def test_fixed_base_with_passive_joint_and_object_uses_native_layout(tmp_path: Path) -> None:
    model_path = tmp_path / "fixed.xml"
    model_path.write_text(
        "<mujoco><option timestep='.002' gravity='0 0 0'/>"
        "<worldbody><body name='hand'>"
        "<joint name='driven' type='hinge' axis='1 0 0'/>"
        "<joint name='passive' type='hinge' axis='0 1 0'/>"
        "<geom type='sphere' size='.05'/></body>"
        "<body name='object' pos='1 2 3'><freejoint name='object_free'/>"
        "<geom type='sphere' size='.05'/></body></worldbody>"
        "<actuator><motor name='drive' joint='driven'/></actuator></mujoco>",
        encoding="utf-8",
    )
    backend = MotrixBackend(
        SceneCfg(str(model_path)), num_envs=2, sim_dt=0.002, base_name="hand"
    )
    try:
        widths = backend.get_public_state_widths()
        assert (widths.nq, widths.nv) == (9, 8)
        np.testing.assert_array_equal(
            backend.get_joint_state_qpos_indices(("driven", "passive")), [0, 1]
        )
        np.testing.assert_array_equal(
            backend.get_joint_state_qvel_indices(("driven", "passive")), [0, 1]
        )
        assert backend.get_root_state_layout("object").qpos_indices == tuple(range(2, 9))
        assert backend.get_root_state_layout("object").qvel_indices == tuple(range(2, 8))
        with pytest.raises(NotImplementedError, match="own a floating base"):
            backend.get_root_state_layout("hand")
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.packed_reset
        assert capabilities.device_reset_randomization
        np.testing.assert_allclose(
            backend.get_default_qpos(), [0, 0, 1, 2, 3, 1, 0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(backend.get_default_dof_pos(), [0, 0])
        assert backend.get_init_qvel().shape == (8,)

        qpos = np.asarray([[0.2, -0.3, 4, 5, 6, 0.5, 0.5, 0.5, -0.5]], dtype=np.float32)
        qvel = np.asarray([[0.7, -0.8, 1, 2, 3, 0.1, 0.2, 0.3]], dtype=np.float32)
        backend._host_set_state(np.asarray([1], dtype=np.intp), qpos, qvel)
        state = backend._host_get_state()
        np.testing.assert_allclose(state["qpos"][1], qpos[0], atol=1e-6)
        np.testing.assert_allclose(state["qvel"][1], qvel[0], atol=1e-6)
        np.testing.assert_allclose(
            backend.data.dof_pos[1], [0.2, -0.3, 4, 5, 6, 0.5, 0.5, -0.5, 0.5], atol=1e-6
        )
    finally:
        backend.close()
