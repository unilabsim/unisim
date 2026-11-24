"""MuJoCo named-sensor metadata tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from unisim import MuJoCoBackend, SimBackend
from unisim.backend.mujoco.sensor_metadata import classify_mujoco_sensor
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")

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


def test_mujoco_inventory_uses_base_contract_with_sorted_categories_and_widths(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sensor-metadata.xml"
    model_path.write_text(SENSOR_METADATA_MODEL, encoding="utf-8")
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)

    assert MuJoCoBackend.get_sensor_inventory is SimBackend.get_sensor_inventory
    assert backend.get_sensor_names() == (
        "a_gyro",
        "b_force",
        "c_acceleration",
        "m_contact",
        "z_position",
    )
    inventory = backend.get_sensor_inventory()
    assert {descriptor.name: descriptor.category for descriptor in inventory} == {
        "a_gyro": "velocity",
        "b_force": "force",
        "c_acceleration": "acceleration",
        "m_contact": "contact",
        "z_position": "position",
    }
    for descriptor in inventory:
        view_shape = backend._host_read_sensor(descriptor.name).shape
        assert descriptor.width == view_shape[1]
        assert view_shape == (backend.num_envs, descriptor.width)


def test_mujoco_inventory_widths_match_public_sensor_views(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    model_path = tmp_path / "sensor-metadata.xml"
    model_path.write_text(SENSOR_METADATA_MODEL, encoding="utf-8")
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)

    for descriptor in backend.get_sensor_inventory():
        assert backend.get_sensor_view(descriptor.name).shape == (
            backend.num_envs,
            descriptor.width,
        )


@pytest.mark.parametrize(
    ("sensor_type_name", "expected"),
    (
        ("mjSENS_FRAMEPOS", "position"),
        ("mjSENS_GYRO", "velocity"),
        ("mjSENS_ACCELEROMETER", "acceleration"),
        ("mjSENS_CONTACT", "contact"),
        ("mjSENS_TOUCH", "contact"),
        ("mjSENS_FORCE", "force"),
        ("mjSENS_ACTUATORFRC", "force"),
        ("mjSENS_USER", "other"),
        ("mjSENS_FUTURE", "unknown"),
    ),
)
def test_mujoco_sensor_type_categories(sensor_type_name: str, expected: str) -> None:
    assert classify_mujoco_sensor(sensor_type_name) == expected


def test_mujoco_sensor_type_mapping_covers_current_enum() -> None:
    categorized = {
        sensor_type.name
        for sensor_type in mujoco.mjtSensor.__members__.values()
        if classify_mujoco_sensor(sensor_type.name) != "unknown"
    }
    assert categorized == set(mujoco.mjtSensor.__members__)
