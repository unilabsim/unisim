"""Batched cache refresh preserves sensor frames and selected environment rows."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from unisim import create_backend
from unisim.scene import SceneCfg

if sys.version_info[:2] != (3, 12):
    pytest.skip("SuperDex wheels require Python 3.12", allow_module_level=True)
pytest.importorskip("superdex.physics")
mujoco = pytest.importorskip("mujoco")


@pytest.fixture
def sensor_backend(tmp_path):
    path = tmp_path / "sensor_frames.xml"
    path.write_text("""<mujoco>
      <compiler angle="radian"/>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="root" pos="0 0 .6">
          <freejoint name="free"/>
          <inertial mass="2" pos=".02 -.03 .01" diaginertia=".04 .05 .03"/>
          <geom name="root_geom" type="box" size=".1 .1 .1"/>
          <site name="root_site" pos=".04 .02 -.03" quat=".9238795 0 .3826834 0"/>
          <body name="arm" pos="0 0 .2" quat=".9238795 0 0 .3826834">
            <joint name="hinge" axis="0 1 0" damping=".1"/>
            <inertial mass=".5" pos="0 0 .06" diaginertia=".01 .01 .005"/>
            <geom type="box" pos="0 0 .06" size=".02 .02 .06"/>
            <site name="arm_site" pos=".01 -.02 .08" quat=".9238795 .3826834 0 0"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="motor" joint="hinge" ctrlrange="-2 2"/></actuator>
      <sensor>
        <jointpos name="angle" joint="hinge"/>
        <jointvel name="speed" joint="hinge"/>
        <framepos name="root_pos" objtype="site" objname="root_site"/>
        <framepos name="arm_pos" objtype="site" objname="arm_site"/>
        <framequat name="root_quat" objtype="site" objname="root_site"/>
        <framequat name="arm_quat" objtype="site" objname="arm_site"/>
        <framezaxis name="root_up" objtype="site" objname="root_site"/>
        <framezaxis name="arm_up" objtype="site" objname="arm_site"/>
        <gyro name="root_gyro" site="root_site"/>
        <gyro name="arm_gyro" site="arm_site"/>
        <velocimeter name="root_local_vel" site="root_site"/>
        <velocimeter name="arm_local_vel" site="arm_site"/>
        <framelinvel name="root_world_vel" objtype="site" objname="root_site"/>
        <framelinvel name="arm_world_vel" objtype="site" objname="arm_site"/>
        <frameangvel name="root_world_omega" objtype="site" objname="root_site"/>
        <frameangvel name="arm_world_omega" objtype="site" objname="arm_site"/>
        <contact name="ground_contact" geom1="floor" geom2="root_geom" data="found" num="1"/>
      </sensor>
    </mujoco>""")
    backend = create_backend("superdex", SceneCfg(str(path)), 3, 0.002)
    model = mujoco.MjModel.from_xml_path(str(path))
    try:
        yield backend, model
    finally:
        backend.close()


def _set_distinct_states(backend):
    q = np.tile(backend.get_default_qpos(), (3, 1))
    q[:, :3] = [[0.1, -0.2, 0.6], [0.2, 0.1, 0.7], [-0.3, 0.2, 0.5]]
    q[:, 3:7] = [[0.8, 0.2, -0.4, 0.4], [0.5, 0.5, 0.5, 0.5], [0.7, -0.1, 0.2, 0.3]]
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1)[:, None]
    q[:, 7] = [0.2, -0.4, 0.6]
    v = np.array(
        [
            [0.1, -0.2, 0.3, 0.2, 0.4, -0.1, 0.5],
            [-0.3, 0.4, 0.1, -0.5, 0.3, 0.2, -0.4],
            [0.2, 0.1, -0.1, 0.6, -0.2, 0.3, 0.1],
        ]
    )
    backend.set_state(np.arange(3), q, v)


def _assert_kinematic_sensors_match(backend, model):
    state = backend.get_state()
    body_ids = np.arange(model.nbody)
    body_quat = backend.get_body_quat_w(body_ids)
    body_lin = backend.get_body_lin_vel_w(body_ids)
    body_ang = backend.get_body_ang_vel_w(body_ids)
    velocity_kinds = {
        int(mujoco.mjtSensor.mjSENS_GYRO),
        int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
        int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
        int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
    }
    for row in range(backend.num_envs):
        data = mujoco.MjData(model)
        data.qpos[:] = state["qpos"][row]
        data.qvel[:] = state["qvel"][row]
        mujoco.mj_forward(model, data)
        for i in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name == "ground_contact":
                continue  # Contact geometry belongs to each engine's positive solver step.
            start, width = int(model.sensor_adr[i]), int(model.sensor_dim[i])
            expected = data.sensordata[start : start + width]
            kind = int(model.sensor_type[i])
            if kind in velocity_kinds:
                # Body twists are the backend contract's authoritative post-step
                # velocity. Its integrator need not reconstruct them identically
                # from joint-coordinate velocities at the new configuration.
                site = int(model.sensor_objid[i])
                body = int(model.site_bodyid[site])
                body_rotation, local_rotation = np.empty(9), np.empty(9)
                mujoco.mju_quat2Mat(body_rotation, body_quat[row, body].astype(float))
                mujoco.mju_quat2Mat(local_rotation, model.site_quat[site])
                body_rotation = body_rotation.reshape(3, 3)
                frame_rotation = body_rotation @ local_rotation.reshape(3, 3)
                if kind in (
                    int(mujoco.mjtSensor.mjSENS_GYRO),
                    int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
                ):
                    expected = body_ang[row, body].astype(float)
                else:
                    expected = body_lin[row, body] + np.cross(
                        body_ang[row, body], body_rotation @ model.site_pos[site]
                    )
                if kind in (
                    int(mujoco.mjtSensor.mjSENS_GYRO),
                    int(mujoco.mjtSensor.mjSENS_VELOCIMETER),
                ):
                    expected = frame_rotation.T @ expected
            actual = backend.get_sensor_data(name)[row]
            if width == 4 and np.dot(actual, expected) < 0:
                expected = -expected
            np.testing.assert_allclose(actual, expected, atol=3e-6, err_msg=f"row={row} {name}")


def test_batched_site_frames_match_mujoco_at_reset_and_after_substeps(sensor_backend):
    backend, model = sensor_backend
    _set_distinct_states(backend)
    _assert_kinematic_sensors_match(backend, model)
    backend.step(np.array([[0.1], [-0.2], [0.3]]), nsteps=3)
    _assert_kinematic_sensors_match(backend, model)


def test_selected_rows_preserve_other_caches_and_bound_readers(sensor_backend, monkeypatch):
    backend, model = sensor_backend
    _set_distinct_states(backend)
    names = ("arm_pos", "root_gyro", "angle", "speed", "ground_contact")
    view = backend.bind_sensor_data(names)
    before = {s.name: backend.get_sensor_data(s.name) for s in backend.model.sensors}
    state = backend.get_state()
    ids = np.array([2, 0])  # Non-monotonic indexing must retain input-to-row correspondence.
    q, v = state["qpos"][ids].copy(), state["qvel"][ids].copy()
    q[:, 7] = [-0.1, 0.7]
    v[:, :3] *= -0.5
    backend.set_state(ids, q, v)
    _assert_kinematic_sensors_match(backend, model)
    for name, values in before.items():
        np.testing.assert_array_equal(backend.get_sensor_data(name)[1], values[1])
    expected = np.concatenate([backend.get_sensor_data(name) for name in names], axis=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached reads must not refresh native state or rebuild bindings")

    monkeypatch.setattr(backend, "_refresh", forbidden)
    monkeypatch.setattr(backend, "_refresh_sensor_batches", forbidden)
    np.testing.assert_array_equal(view.read(), expected)
    for name in names:
        value = backend.get_sensor_data(name)
        value[:] = 1000
    np.testing.assert_array_equal(view.read(), expected)
