"""Numerical SuperDex conformance using small, explicitly authored rigid models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim import assert_backend_conformance, create_backend
from unisim.scene import SceneCfg
from unisim.utils.rotation import np_quat_apply_batched

if sys.version_info[:2] != (3, 12):
    pytest.skip("SuperDex wheels require Python 3.12", allow_module_level=True)
pytest.importorskip("superdex.physics")
mujoco = pytest.importorskip("mujoco")


def _model(tmp_path, floating=False):
    free = '<freejoint name="root"/>' if floating else ""
    path = tmp_path / ("free.xml" if floating else "fixed.xml")
    path.write_text(f"""<mujoco model="superdex_test">
      <compiler angle="radian"/>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="base" pos="0 0 1">
          {free}
          <inertial mass="2" pos=".03 .02 0" diaginertia=".03 .04 .05"/>
          <geom name="base_geom" type="box" size=".1 .1 .1"/>
          <site name="imu" pos=".01 .02 .03"/>
          <body name="arm" pos="0 0 .2">
            <joint name="hinge" axis="0 1 0" range="-1 1" damping=".1"/>
            <inertial mass="1" pos="0 0 .1" diaginertia=".02 .025 .01"/>
            <geom name="arm_geom" type="box" pos="0 0 .1" size=".025 .025 .1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="motor" joint="hinge" gear="2" ctrlrange="-2 2"/></actuator>
      <sensor>
        <jointpos name="angle" joint="hinge"/>
        <jointvel name="speed" joint="hinge"/>
        <framequat name="orientation" objtype="body" objname="base"/>
        <gyro name="gyro" site="imu"/>
        <velocimeter name="local_linvel" site="imu"/>
      </sensor>
    </mujoco>""")
    return path


@pytest.fixture
def fixed(tmp_path):
    backend = create_backend("superdex", SceneCfg(str(_model(tmp_path))), 2, 0.002)
    try:
        yield backend
    finally:
        backend.close()


def test_fixed_conformance_state_width_and_selected_reset(fixed):
    assert_backend_conformance(fixed)
    assert fixed.get_state()["qpos"].shape == (2, 1)
    fixed.set_state(np.array([0, 1]), np.array([[0.2], [-0.3]]), np.array([[0.1], [0.2]]))
    before = fixed.get_state()
    fixed.reset(np.array([0]))
    np.testing.assert_array_equal(fixed.get_state()["qpos"][1], before["qpos"][1])
    np.testing.assert_array_equal(fixed.get_state()["qvel"][1], before["qvel"][1])
    np.testing.assert_allclose(fixed.get_sensor_data("angle")[:, 0], [0, -0.3], atol=1e-6)
    with pytest.raises(NotImplementedError, match="floating"):
        fixed.get_root_state_layout("base")


def test_state_reads_are_detached_and_do_not_parse_assets(fixed, monkeypatch):
    from unisim.backend.superdex import materialization

    def forbidden(*args, **kwargs):
        raise AssertionError("cold model parsing entered hot path")

    monkeypatch.setattr(materialization, "materialize_model", forbidden)
    view = fixed.bind_sensor_data(["angle", "speed"])
    state = fixed.get_state()
    state["qpos"][:] = 100
    assert np.max(abs(fixed.get_dof_pos())) < 1
    fixed.step(np.ones((2, 1)), 2)
    assert view.read().shape == (2, 2)
    fixed.reset(np.array([1]))
    assert np.isfinite(view.read()).all()


def test_mujoco_playback_state_uses_the_authored_xml(fixed):
    snapshot = fixed.get_physics_state()
    assert fixed.get_play_capabilities().supports_physics_state_playback
    assert snapshot.shape == (2, 1 + fixed.model.nq + fixed.model.nv)
    assert np.all(snapshot[:, 0] == 0)
    assert fixed.get_playback_model() == fixed.scene_visual_model_file


def test_pre_step_feedback_and_control_limits(fixed):
    seen = []

    def convert(backend, action):
        seen.append(backend.get_dof_vel().copy())
        return action * 100

    fixed.set_pre_step_control(convert)
    fixed.step(np.ones((2, 1)), 3)
    assert len(seen) == 3
    assert np.max(abs(seen[1] - seen[0])) > 0
    np.testing.assert_array_equal(fixed.get_state("ctrl")["ctrl"], 2)
    assert np.all(fixed.get_dof_vel() > 0)


def test_invalid_selected_state_is_rejected_without_mutation(fixed):
    before = fixed.get_state()
    with pytest.raises(ValueError, match="unique"):
        fixed.set_state(np.array([0, 0]), np.ones((2, 1)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        fixed.set_state(np.array([0]), np.array([[np.nan]]), np.zeros((1, 1)))
    for field, value in before.items():
        np.testing.assert_array_equal(fixed.get_state()[field], value)


def test_world_force_is_reprojected_after_every_substep(fixed, tmp_path):
    other = create_backend("superdex", SceneCfg(str(_model(tmp_path))), 2, 0.002)
    try:
        q, v = np.full((2, 1), 0.4), np.full((2, 1), 1.5)
        for backend in (fixed, other):
            backend.set_state(np.arange(2), q, v)
        body = fixed.get_body_ids(["arm"])
        force = np.zeros((2, 1, 3))
        force[:, 0, 0] = 10.0
        fixed.apply_body_force(body, force)
        fixed.step(np.zeros((2, 1)), nsteps=20)
        for _ in range(20):
            other.apply_body_force(body, force)
            other.step(np.zeros((2, 1)))
        for field in ("qpos", "qvel"):
            np.testing.assert_allclose(
                fixed.get_state()[field], other.get_state()[field], atol=2e-6
            )
    finally:
        other.close()


def test_body_torques_accumulate_with_control_instead_of_replacing_it(fixed):
    body = fixed.get_body_ids(["arm"])
    force = np.zeros((2, 1, 3))
    torque = np.zeros_like(force)
    torque[0, 0, 1] = 1.0
    fixed.apply_body_force(body, force, torque)
    fixed.apply_body_force(body, force, torque)
    # Native motor gear=2: row 0 gets 1*2 plus two unit body torques;
    # row 1 gets 2*2 directly. A second native external-force write would
    # incorrectly erase row 0's disturbance or its control contribution.
    fixed.step(np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(fixed.get_dof_vel()[0], fixed.get_dof_vel()[1], atol=2e-6)
    fixed.step(np.zeros((2, 1)))
    np.testing.assert_allclose(fixed.get_dof_vel()[0], fixed.get_dof_vel()[1], atol=2e-6)


def test_free_root_frames_match_authored_mujoco_kinematics(tmp_path):
    path = _model(tmp_path, floating=True)
    backend = create_backend("superdex", SceneCfg(str(path)), 2, 0.002)
    try:
        q = np.tile(backend.get_default_qpos(), (2, 1))
        q[:, :3] = [[0.2, -0.4, 1.1], [0.1, 0.3, 0.9]]
        q[:, 3:7] = [[0.8, 0.2, -0.4, 0.4], [0.5, 0.5, 0.5, 0.5]]
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1)[:, None]
        q[:, 7] = [0.3, -0.2]
        v = np.array(
            [[0.3, -0.2, 0.1, 0.2, 0.4, -0.6, 0.7], [-0.4, 0.1, 0.3, -0.2, 0.6, 0.1, -0.5]]
        )
        backend.set_state(np.arange(2), q, v)
        np.testing.assert_allclose(backend.get_state()["qpos"], q, atol=3e-6)
        np.testing.assert_allclose(backend.get_state()["qvel"], v, atol=3e-6)
        np.testing.assert_allclose(backend.get_base_lin_vel(), v[:, :3], atol=3e-6)
        expected_omega = np_quat_apply_batched(q[:, 3:7], v[:, 3:6])
        np.testing.assert_allclose(backend.get_base_ang_vel(), expected_omega, atol=3e-6)
        np.testing.assert_allclose(backend.get_sensor_data("gyro"), v[:, 3:6], atol=3e-6)
        model = mujoco.MjModel.from_xml_path(str(path))
        body_ids = backend.get_body_ids(["base", "arm"])
        for row in range(2):
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = q[row], v[row]
            mujoco.mj_forward(model, data)
            for column, name in enumerate(["base", "arm"]):
                mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                np.testing.assert_allclose(
                    backend.get_body_pos_w(body_ids)[row, column], data.xpos[mid], atol=4e-6
                )
                jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, mid)
                np.testing.assert_allclose(
                    backend.get_body_lin_vel_w(body_ids)[row, column], jacp @ v[row], atol=5e-6
                )
                np.testing.assert_allclose(
                    backend.get_body_ang_vel_w(body_ids)[row, column], jacr @ v[row], atol=5e-6
                )
        root = backend.get_root_state_layout("base")
        assert root.qpos_indices == tuple(range(7))
        assert root.qvel_indices == tuple(range(6))
        before = backend.get_state()
        backend.reset(np.array([0]))
        np.testing.assert_array_equal(backend.get_state()["qpos"][1], before["qpos"][1])
    finally:
        backend.close()


def test_closing_one_backend_preserves_another(fixed, tmp_path):
    other = create_backend("superdex", SceneCfg(str(_model(tmp_path))), 1, 0.002)
    other.cleanup_scene_assets()
    other.close()
    fixed.step(np.ones((2, 1)))
    assert np.isfinite(fixed.get_dof_vel()).all()
    with pytest.raises(RuntimeError, match="closed"):
        other.step(np.zeros((1, 1)))


def test_unlimited_joint_does_not_acquire_a_zero_width_manager_limit(tmp_path):
    path = _model(tmp_path)
    path.write_text(path.read_text().replace('range="-1 1"', 'limited="false"'))
    backend = create_backend("superdex", SceneCfg(str(path)), 1, 0.002)
    try:
        np.testing.assert_array_equal(backend.get_joint_range(), [[-np.inf, np.inf]])
        backend.set_state(np.array([0]), np.array([[7.0]]), np.zeros((1, 1)))
        np.testing.assert_allclose(backend.get_dof_pos(), [[7.0]], atol=2e-6)
    finally:
        backend.close()


def test_native_fr3_when_registered_assets_are_available(monkeypatch):
    assets = os.environ.get("SUPERDEX_ASSETS_PATH")
    if not assets:
        pytest.skip("Set SUPERDEX_ASSETS_PATH to validate the external FR3 fixture")
    path = Path(assets) / "bots/arms/fr3_v2/fr3_v2.superdex_bot"
    import superdex.robotics as robotics

    load = robotics.load_bot_prefab_from_file

    def load_with_derived_com(path):
        prefab = load(path)
        prefab.links[2].center_of_mass = None
        prefab.links[2].moment_of_inertia = None
        return prefab

    monkeypatch.setattr(robotics, "load_bot_prefab_from_file", load_with_derived_com)
    backend = create_backend(
        "superdex",
        SceneCfg(str(path)),
        2,
        0.002,
        base_name="fr3_link0",
        superdex_effort_limits=[20, 20, 20, 20, 5, 5, 5],
    )
    try:
        assert_backend_conformance(backend)
        assert backend.get_state()["qpos"].shape == (2, 7)
        assert backend.get_actuator_names() == tuple(f"fr3_joint{i}" for i in range(1, 8))
        # The SDK infers a nonzero COM from this asymmetric link's mesh;
        # omitted authoring metadata must not silently turn it into zero.
        com = backend.get_body_ipos()[backend.get_body_id("fr3_link1")]
        np.testing.assert_allclose(com, [0.0000043, -0.03405, -0.06730], atol=1e-5)
        backend.step(np.ones((2, 7)), 5)
        assert np.isfinite(backend.get_dof_pos()).all()
    finally:
        backend.close()
