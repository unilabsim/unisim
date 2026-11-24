"""Final-state complete sensor refresh for the MuJoCo adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")

import mujoco

from unisim import MuJoCoBackend, SupportLevel
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model='mujoco-sensor-refresh'>
  <option timestep='0.005' iterations='50'/>
  <worldbody>
    <geom name='floor' type='plane' size='3 3 .1'/>
    <body name='ball' pos='0 0 .051'>
      <freejoint name='root'/>
      <geom name='ball_geom' type='sphere' size='.05' mass='1'/>
      <site name='ball_site'/>
    </body>
  </worldbody>
  <sensor>
    <framepos name='ball_pos' objtype='body' objname='ball'/>
    <framelinvel name='ball_vel' objtype='body' objname='ball'/>
    <accelerometer name='ball_acc' site='ball_site'/>
    <contact name='ball_floor' geom1='ball_geom' geom2='floor'
             data='found force pos' num='1' reduce='none'/>
  </sensor>
</mujoco>
"""


def _backend(tmp_path: Path) -> MuJoCoBackend:
    model_file = tmp_path / "refresh.xml"
    model_file.write_text(_MODEL)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=3,
        sim_dt=0.005,
        base_name="ball",
        add_body_sensors=True,
    )
    backend.materialize()
    return backend


def _independent_forward(
    backend: MuJoCoBackend, rows: np.ndarray, warmstart: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = backend._model
    sensors = []
    warm = []
    xfrc = backend._xfrc_view.reshape(backend.num_envs, model.nbody, 6)
    for row in rows:
        data = mujoco.MjData(model)
        data.qpos[:] = backend._qpos_view[row]
        data.qvel[:] = backend._qvel_view[row]
        data.ctrl[:] = backend._ctrl_view[row]
        data.xfrc_applied[:] = xfrc[row]
        data.qacc_warmstart[:] = warmstart[row]
        mujoco.mj_forward(model, data)
        sensors.append(data.sensordata.copy())
        warm.append(data.qacc_warmstart.copy())
    return np.asarray(sensors), np.asarray(warm)


def test_refresh_sensors_matches_independent_forward_and_preserves_other_rows(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    try:
        rows = np.arange(backend.num_envs, dtype=np.int32)
        qpos = np.broadcast_to(backend.get_default_qpos(), (rows.size, backend.nq)).copy()
        qpos[:, 0] = np.linspace(-0.2, 0.2, rows.size)
        backend._host_set_state(rows, qpos, np.zeros((rows.size, backend.nv)))
        backend._host_step(np.zeros((rows.size, 0)), nsteps=8)

        selected = np.array([1], dtype=np.int32)
        unselected = np.array([0, 2], dtype=np.int32)
        state_before = backend._host_get_state(("qpos", "qvel", "ctrl"))
        sensors_before = backend._sensor_data.copy()
        warm_before = backend._warm_view.copy()
        time_before = backend._time_view.copy()
        dirty_before = backend._tracked_body_state_dirty.copy()
        expected_sensors, expected_warm = _independent_forward(backend, selected, warm_before)

        ball_id = backend.get_body_ids(["ball"])
        pending_force = np.zeros((backend.num_envs, 1, 3))
        pending_force[:, 0, 0] = [4.0, -2.0, 1.0]
        backend.apply_body_force(ball_id, pending_force)
        pending_before = backend._pending_xfrc_applied.copy()
        native_xfrc_before = backend._xfrc_view.copy()

        assert (
            backend.get_capabilities().get("state.sensor_refresh").support is SupportLevel.EXACT
        )
        backend.refresh_sensors(selected)

        np.testing.assert_allclose(
            backend._sensor_data[selected], expected_sensors, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_array_equal(backend._sensor_data[unselected], sensors_before[unselected])
        np.testing.assert_allclose(
            backend._warm_view[selected], expected_warm, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_array_equal(backend._warm_view[unselected], warm_before[unselected])
        for name, before in state_before.items():
            np.testing.assert_array_equal(backend._host_get_state(name)[name], before)
        np.testing.assert_array_equal(backend._time_view, time_before)
        np.testing.assert_array_equal(backend._pending_xfrc_applied, pending_before)
        np.testing.assert_array_equal(backend._xfrc_view, native_xfrc_before)
        np.testing.assert_array_equal(
            backend._tracked_body_state_dirty[selected], np.zeros(selected.size, dtype=bool)
        )
        np.testing.assert_array_equal(
            backend._tracked_body_state_dirty[unselected], dirty_before[unselected]
        )

        current_warm = backend._warm_view.copy()
        expected_all_sensors, expected_all_warm = _independent_forward(
            backend, rows, current_warm
        )
        backend.refresh_sensors()
        np.testing.assert_allclose(
            backend._sensor_data, expected_all_sensors, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(backend._warm_view, expected_all_warm, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(backend._pending_xfrc_applied, pending_before)
    finally:
        backend.close()


def test_refresh_sensors_validates_rows_and_empty_selection_is_noop(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    try:
        sensors_before = backend._sensor_data.copy()
        backend.refresh_sensors([])
        np.testing.assert_array_equal(backend._sensor_data, sensors_before)
        with pytest.raises(ValueError, match="distinct"):
            backend.refresh_sensors([0, 0])
        with pytest.raises(ValueError, match="integer"):
            backend.refresh_sensors([0.0])
        with pytest.raises(ValueError, match="in range"):
            backend.refresh_sensors([backend.num_envs])
        np.testing.assert_array_equal(backend._sensor_data, sensors_before)
    finally:
        backend.close()
