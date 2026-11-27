"""Selected-row complete sensor refresh for MotrixSim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402


def test_refresh_sensors_selected_rows_preserve_state_and_unselected_sensors(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "refresh.xml"
    model_path.write_text(
        """<mujoco><option timestep='.01' gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='floor' type='plane' size='2 2 .1'/>
    <body name='base' pos='0 0 1'><joint name='j1' type='hinge'/>
      <geom type='sphere' size='.05'/></body>
    <body name='object'><freejoint name='root'/>
      <geom name='sphere' type='sphere' size='.1' mass='1'/><site name='site'/></body>
  </worldbody>
  <actuator><motor name='motor' joint='j1'/></actuator>
  <sensor><jointpos name='jp' joint='j1'/><actuatorfrc name='af' actuator='motor'/>
    <framepos name='pos' objtype='body' objname='object'/>
    <contact name='contact' geom1='sphere' geom2='floor' data='found force' num='1'/>
    <touch name='touch' site='site'/></sensor>
</mujoco>""",
        encoding="utf-8",
    )
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        qpos = np.zeros((2, 8), dtype=np.float32)
        qpos[:, 4] = 1.0
        qpos[0, 3] = 1.0
        qpos[1, 3] = 0.05
        qvel = np.zeros((2, 7), dtype=np.float32)
        backend._host_set_state(np.asarray([0, 1]), qpos, qvel)
        backend.data.actuator_ctrls = np.full((2, 1), 0.5, dtype=np.float32)
        before_qpos = backend.data.dof_pos.copy()
        before_qvel = backend.data.dof_vel.copy()
        before_ctrl = backend.data.actuator_ctrls.copy()
        before_time = backend._time_view.copy()
        names = ("jp", "af", "pos", "contact", "touch")
        before_unselected = backend._host_read_sensor_batch(names)[0].copy()

        backend.refresh_sensors([1])

        np.testing.assert_array_equal(backend.data.dof_pos, before_qpos)
        np.testing.assert_array_equal(backend.data.dof_vel, before_qvel)
        np.testing.assert_array_equal(backend.data.actuator_ctrls, before_ctrl)
        np.testing.assert_array_equal(backend._time_view, before_time)
        np.testing.assert_array_equal(
            backend._host_read_sensor_batch(names)[0], before_unselected
        )
        np.testing.assert_allclose(backend._host_read_sensor("jp")[1], [0.0])
        np.testing.assert_allclose(backend._host_read_sensor("af")[1], [0.5])
        np.testing.assert_allclose(backend._host_read_sensor("pos")[1], [0.0, 0.0, 0.05])
        contact = backend._host_read_sensor("contact")[1]
        assert contact[0] == 1.0
        assert contact[1] > 0.0
        backend.refresh_sensors([])
        with pytest.raises(ValueError, match="distinct"):
            backend.refresh_sensors([1, 1])
    finally:
        backend.close()


def test_refresh_sensors_preserves_pending_body_wrench(tmp_path: Path) -> None:
    model_path = tmp_path / "wrench.xml"
    model_path.write_text(
        "<mujoco><option timestep='.01' gravity='0 0 0'/>"
        "<worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>"
        "<geom type='box' size='.05 .05 .05'/></body></worldbody>"
        "<actuator><motor name='motor' joint='slide'/></actuator>"
        "<sensor><jointvel name='speed' joint='slide'/></sensor></mujoco>",
        encoding="utf-8",
    )
    backend = MotrixBackend(SceneCfg(str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        force = np.zeros((2, 1, 3), dtype=np.float64)
        force[0, 0, 0] = 10.0
        backend.apply_body_force(np.asarray([0]), force)
        backend.refresh_sensors([0])
        np.testing.assert_allclose(backend._host_read_sensor("speed"), [[0.0], [0.0]])
        backend._host_step(np.zeros((2, 1), dtype=np.float64))
        np.testing.assert_allclose(backend.data.dof_vel[:, 0], [0.1, 0.0], atol=1e-4)
    finally:
        backend.close()
