"""MJWarp/MuJoCo named-sensor metadata parity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")
pytest.importorskip("mjbatch")

import warp

from unisim import MjwarpBackend, MuJoCoBackend
from unisim.scene import SceneCfg

SENSOR_METADATA_MODEL = """<mujoco model='sensor-metadata'>
  <worldbody>
    <body name='moving' pos='0 0 0.2'>
      <joint name='slide' type='slide' axis='1 0 0'/>
      <geom name='moving_geom' type='sphere' size='0.1'/>
      <site name='moving_site'/>
    </body>
    <body name='target' pos='0.4 0 0.2'>
      <geom name='target_geom' type='sphere' size='0.1'/>
    </body>
  </worldbody>
  <sensor>
    <framepos name='z_position' objtype='body' objname='moving'/>
    <contact name='m_contact' geom1='moving_geom' geom2='target_geom' data='found'/>
    <gyro name='a_gyro' site='moving_site'/>
    <force name='b_force' site='moving_site'/>
    <accelerometer name='c_acceleration' site='moving_site'/>
  </sensor>
</mujoco>"""


def test_mjwarp_inventory_matches_mujoco_names_widths_and_categories(tmp_path: Path) -> None:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("MJWarp inventory parity requires an active CUDA Warp device")

    model_path = tmp_path / "sensor-metadata.xml"
    model_path.write_text(SENSOR_METADATA_MODEL, encoding="utf-8")
    scene = SceneCfg(model_file=str(model_path))
    mjwarp = MjwarpBackend(scene, num_envs=2, sim_dt=0.01)
    mujoco = MuJoCoBackend(scene, num_envs=2, sim_dt=0.01)

    assert mjwarp.get_sensor_names() == mujoco.get_sensor_names()
    assert mjwarp.get_sensor_inventory() == mujoco.get_sensor_inventory()
