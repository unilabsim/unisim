"""Real-CUDA acceptance for Newton replay-facing public APIs."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("newton")
warp = pytest.importorskip("warp")

from unisim.backend.newton.backend import NewtonBackend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("Newton replay public APIs require real CUDA")


def _write_model(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _backend(scene: SceneCfg, num_envs: int) -> NewtonBackend:
    backend = NewtonBackend(
        scene,
        num_envs=num_envs,
        sim_dt=0.002,
        device=str(warp.get_device()),
        nconmax=16,
        njmax=32,
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


def _single_scene(tmp_path: Path) -> SceneCfg:
    source = _write_model(
        tmp_path,
        "single.xml",
        """
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="base" pos="0 0 1">
            <freejoint name="root"/>
            <inertial pos=".01 .02 .03" mass="1.2" fullinertia=".11 .2 .3 .015 -.02 .005"/>
            <geom name="base_geom" size=".08" contype="0" conaffinity="0"/>
            <body name="link" pos="0 0 .2">
              <joint name="hinge" axis="0 1 0"/>
              <inertial pos="0 0 .01" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="link_geom" size=".03" contype="0" conaffinity="0"/>
            </body>
            <body name="other_link" pos="-.1 0 0">
              <joint name="other" axis="0 1 0"/>
              <inertial pos="0 0 0" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="other_geom" size=".03" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><actuator>
          <motor name="other_motor" joint="other"/>
          <position name="selected_pos" joint="hinge" kp="20"/>
          <velocity name="selected_vel" joint="hinge" kv="2"/>
        </actuator></mujoco>
        """,
    )
    return SceneCfg(model_file=source.model_file)


def _portable_scene(tmp_path: Path) -> tuple[SceneCfg, int]:
    robot = _write_model(
        tmp_path,
        "robot.xml",
        """
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="base">
            <inertial pos="0 0 0" mass="1" diaginertia=".11 .2 .3"/>
            <geom name="base_geom" size=".08" contype="0" conaffinity="0"/>
            <body name="link" pos="0 0 .2">
              <joint name="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="link_geom" size=".03" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><actuator>
          <position name="robot_pos" joint="hinge" kp="40"/>
          <velocity name="robot_vel" joint="hinge" kv="4"/>
        </actuator></mujoco>
        """,
    )
    object_a = _write_model(
        tmp_path,
        "object_a.xml",
        """
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="base">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="1.5" fullinertia=".11 .2 .3 .015 -.02 .005"/>
            <geom name="object_geom" size=".1" contype="0" conaffinity="0"/>
            <body name="link">
              <joint name="drive" axis="0 1 0"/>
              <inertial pos="0 0 0" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="drive_geom" size=".03" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><actuator>
          <position name="object_pos" joint="drive" kp="20"/>
          <velocity name="object_vel" joint="drive" kv="2"/>
        </actuator></mujoco>
        """,
    )
    object_b = _write_model(
        tmp_path,
        "object_b.xml",
        """
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="base">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="2" diaginertia=".4 .5 .6"/>
            <geom name="object_geom" size=".1" contype="0" conaffinity="0"/>
            <body name="link">
              <joint name="drive" axis="0 1 0"/>
              <inertial pos="0 0 0" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="drive_geom" size=".03" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody><actuator>
          <position name="object_pos" joint="drive" kp="30"/>
          <velocity name="object_vel" joint="drive" kv="3"/>
        </actuator></mujoco>
        """,
    )
    assignment = (1, 0, 1)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", robot, root_mode="fixed"),
            SceneEntitySpec("object", object_a),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.asarray(assignment, dtype=np.int32), (object_a, object_b))
        ),
    )
    return scene, len(assignment)


def _assert_readonly_detached(backend: NewtonBackend) -> np.ndarray:
    value = backend.get_reset_term_default("body_inertia")
    assert not value.flags.writeable
    snapshot = value.copy()
    value.setflags(write=True)
    value[...] = -1.0
    np.testing.assert_array_equal(backend.get_reset_term_default("body_inertia"), snapshot)
    return snapshot


def test_nonportable_body_inertia_defaults_use_native_public_body_order(tmp_path: Path):
    backend = _backend(_single_scene(tmp_path), num_envs=3)
    try:
        value = _assert_readonly_detached(backend)
        model = mujoco.MjModel.from_xml_path(_single_scene(tmp_path).model_file)
        expected = np.sort(np.asarray(model.body_inertia[1:], dtype=np.float32), axis=1)
        np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-7)
        native = np.asarray(backend.model.body_inertia.numpy()).reshape(3, 3, 3, 3)
        np.testing.assert_allclose(
            np.trace(native, axis1=-2, axis2=-1) / 3.0,
            np.broadcast_to(value.sum(axis=1) / 3.0, (3, 3)),
            rtol=2e-6,
            atol=2e-7,
        )
        capabilities = backend.get_dr_capabilities()
        assert not capabilities.supports_reset_term("body_inertia")
        assert not capabilities.supports_reset_term("body_inertia", plane="device")
    finally:
        backend.close()


def test_portable_body_inertia_defaults_follow_variants_and_public_body_order(tmp_path: Path):
    scene, num_envs = _portable_scene(tmp_path)
    backend = _backend(scene, num_envs)
    try:
        value = _assert_readonly_detached(backend)
        layout = backend.get_scene_layout()
        expected = np.empty_like(value)
        for env in range(num_envs):
            model = mujoco.MjModel.from_xml_path(backend.get_playback_model(env))
            for entity in layout.entities:
                for local_row, body_id in enumerate(entity.body_ids):
                    public_row = body_id - 1
                    expected[env, public_row] = np.sort(model.body_inertia[body_id])
                    assert layout.nbody - 1 > public_row >= 0
                    assert entity.body_names[local_row]
        np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-7)
        native = np.asarray(backend.model.body_inertia.numpy()).reshape(
            num_envs, layout.nbody - 1, 3, 3
        )
        np.testing.assert_allclose(
            np.trace(native, axis1=-2, axis2=-1) / 3.0,
            value.sum(axis=2) / 3.0,
            rtol=2e-6,
            atol=2e-7,
        )
        assert not backend.get_dr_capabilities().supports_reset_term("body_inertia")
    finally:
        backend.close()


def _native_gain_tables(backend: NewtonBackend) -> tuple[np.ndarray, np.ndarray]:
    warp.synchronize_device(str(warp.get_device()))
    return (
        np.asarray(backend._solver.mjw_model.actuator_gainprm.numpy()).copy(),
        np.asarray(backend._solver.mjw_model.actuator_biasprm.numpy()).copy(),
    )


def _assert_scaled_native_gains(
    backend: NewtonBackend,
    baseline: tuple[np.ndarray, np.ndarray],
    actuators: tuple[int, ...],
    env_ids: tuple[int, ...],
    scale: float,
) -> None:
    gain, bias = _native_gain_tables(backend)
    expected_gain, expected_bias = (value.copy() for value in baseline)
    kp, kd = backend.get_actuator_gains()
    native = backend._native_actuator_rows()
    for env in env_ids:
        for actuator in actuators:
            row = native[actuator]
            if backend._metadata.actuator_target_kinds[actuator] == "position":
                expected_gain[env, row, 0] = kp[actuator] * scale
                expected_bias[env, row, 1] = -kp[actuator] * scale
            else:
                expected_gain[env, row, 0] = kd[actuator] * scale
            expected_bias[env, row, 2] = -kd[actuator] * scale
    np.testing.assert_allclose(gain, expected_gain, rtol=0, atol=0)
    np.testing.assert_allclose(bias, expected_bias, rtol=0, atol=0)


def test_nonportable_paired_gain_scaling_uses_native_permutation_and_survives_reset(
    tmp_path: Path,
):
    torch = pytest.importorskip("torch")
    backend = _backend(_single_scene(tmp_path), num_envs=3)
    try:
        assert backend.get_tensor_capabilities().actuator_gain_scaling
        native = backend._native_actuator_rows()
        assert not np.array_equal(native, np.arange(backend.num_actuators))
        baseline = _native_gain_tables(backend)
        pair = (1, 2)
        backend.set_actuator_gain_scale(pair, 0.0, env_ids=(1,))
        _assert_scaled_native_gains(backend, baseline, pair, (1,), 0.0)
        with pytest.raises(NotImplementedError, match="position/velocity"):
            backend.set_actuator_gain_scale((0,), 0.0)
        with pytest.raises(ValueError, match="complete position/velocity pair"):
            backend.set_actuator_gain_scale((1,), 0.0, env_ids=(1,))
        _assert_scaled_native_gains(backend, baseline, pair, (1,), 0.0)

        views = backend.get_state_views()
        ctrl = torch.zeros((3, backend.num_actuators), dtype=torch.float32, device="cuda")
        ctrl[:, 1:] = 1.0
        backend.step_tensor(ctrl, nsteps=8)
        hinge_qd = backend.get_joint_dof_indices(("hinge",))[0]
        assert abs(views["qvel"][1, hinge_qd].item()) == 0.0
        assert abs(views["qvel"][0, hinge_qd].item()) > 0.0
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            views["qpos"][1:2].contiguous(),
            views["qvel"][1:2].contiguous(),
        )
        _assert_scaled_native_gains(backend, baseline, pair, (1,), 0.0)
        backend.set_actuator_gain_scale(pair, 1.0, env_ids=(1,))
        _assert_scaled_native_gains(backend, baseline, (), (), 1.0)
        backend.step_tensor(ctrl, nsteps=8)
        assert abs(views["qvel"][1, hinge_qd].item()) > 0.0
    finally:
        backend.close()


def test_portable_paired_gain_scaling_is_entity_scoped_and_restores_variant_defaults(
    tmp_path: Path,
):
    torch = pytest.importorskip("torch")
    scene, num_envs = _portable_scene(tmp_path)
    backend = _backend(scene, num_envs)
    try:
        layout = backend.get_scene_layout()
        object_actuators = layout.get_entity("object").actuator_indices
        robot_actuators = layout.get_entity("robot").actuator_indices
        assert len(object_actuators) == len(robot_actuators) == 2
        baseline = _native_gain_tables(backend)
        backend.set_actuator_gain_scale(object_actuators, 0.0, env_ids=(0, 2))
        gain, bias = _native_gain_tables(backend)
        native = backend._native_actuator_rows()
        robot_rows = np.asarray(robot_actuators, dtype=np.intp)
        object_rows = np.asarray(object_actuators, dtype=np.intp)
        np.testing.assert_array_equal(
            gain[:, native[robot_rows]], baseline[0][:, native[robot_rows]]
        )
        np.testing.assert_array_equal(
            bias[:, native[robot_rows]], baseline[1][:, native[robot_rows]]
        )
        np.testing.assert_array_equal(
            gain[1, native[object_rows]], baseline[0][1, native[object_rows]]
        )
        np.testing.assert_array_equal(
            bias[1, native[object_rows]], baseline[1][1, native[object_rows]]
        )
        position_row = native[object_actuators[0]]
        np.testing.assert_array_equal(gain[(0, 2), position_row, 0], 0.0)
        np.testing.assert_array_equal(bias[(0, 2), position_row, 1], 0.0)
        for row in native[object_rows]:
            np.testing.assert_array_equal(bias[(0, 2), row, 2], 0.0)

        views = backend.get_state_views()
        backend.reset_entities(
            SceneResetRequest(
                (0, 2),
                (
                    EntityStatePatch(
                        "object",
                        joint_positions=np.zeros((2, 1), dtype=np.float32),
                        joint_velocities=np.zeros((2, 1), dtype=np.float32),
                    ),
                ),
            )
        )
        after_reset = _native_gain_tables(backend)
        np.testing.assert_array_equal(after_reset[0], gain)
        np.testing.assert_array_equal(after_reset[1], bias)
        ctrl = torch.ones(
            (num_envs, backend.num_actuators), dtype=torch.float32, device="cuda"
        )
        backend.step_tensor(ctrl, nsteps=8)
        object_joint = layout.get_entity("object").joints[0]
        object_qd = object_joint.qvel_indices[0]
        assert abs(views["qvel"][1, object_qd].item()) > 0.0
        torch.testing.assert_close(views["qvel"][(0, 2), object_qd], torch.zeros(2, device="cuda"))

        backend.set_actuator_gain_scale(object_actuators, 1.0, env_ids=(0, 2))
        restored_gain, restored_bias = _native_gain_tables(backend)
        np.testing.assert_array_equal(restored_gain, baseline[0])
        np.testing.assert_array_equal(restored_bias, baseline[1])
    finally:
        backend.close()
