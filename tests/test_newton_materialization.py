"""Cold-path MJCF compensation and fail-closed asset checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.newton.materialization import scan_newton_model_metadata
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")


def _write_model(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(
        f"<mujoco model='metadata-test'><worldbody>{body}</worldbody></mujoco>",
        encoding="utf-8",
    )
    return path


def test_metadata_scans_keyframes_and_supported_sensors(tmp_path: Path) -> None:
    path = tmp_path / "scene.xml"
    path.write_text(
        """
        <mujoco model="metadata-test">
          <worldbody>
            <body name="base"><joint name="hinge" type="hinge"/>
              <site name="imu" pos="0 0 0"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <sensor><gyro name="gyro" site="imu"/></sensor>
          <keyframe><key name="home" qpos="0.25"/></keyframe>
        </mujoco>
        """,
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    assert metadata.joint_names == ("hinge",)
    assert metadata.keyframes[0][0] == "home"
    np.testing.assert_allclose(metadata.keyframes[0][1], [0.25])
    assert metadata.sensor_plans[0].name == "gyro"
    assert metadata.sensor_plans[0].dim == 3


def test_metadata_rejects_cone_geometry(tmp_path: Path) -> None:
    if not hasattr(mujoco.mjtGeom, "mjGEOM_CONE"):
        pytest.skip("installed MuJoCo does not expose a cone geom type")
    path = _write_model(
        tmp_path,
        "<body name='base'><joint name='hinge' type='hinge'/><geom type='cone' "
        "size='0.1 0.2'/></body>",
    )
    with pytest.raises(NotImplementedError, match="CONE"):
        scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


_CONTACT_SCENE = """
<mujoco model="contact-scan-test">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 1">
      <joint name="hinge" type="hinge"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
      <site name="box_site" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>{sensor}</sensor>
</mujoco>
"""


def _scan_contact_sensor(tmp_path: Path, sensor: str):
    path = tmp_path / "scene.xml"
    path.write_text(_CONTACT_SCENE.format(sensor=sensor), encoding="utf-8")
    return scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


@pytest.mark.parametrize("reduce_mode", ["none", "mindist", "netforce"])
def test_metadata_scans_contact_found_sensor(tmp_path: Path, reduce_mode: str) -> None:
    metadata = _scan_contact_sensor(
        tmp_path,
        f'<contact name="foot" geom1="floor" geom2="box_geom" data="found" num="1" '
        f'reduce="{reduce_mode}"/>',
    )
    assert len(metadata.sensor_plans) == 1
    plan = metadata.sensor_plans[0]
    assert plan.name == "foot"
    assert plan.kind == "contact"
    assert plan.dim == 1
    assert plan.geom1_name == "floor"
    assert plan.geom2_name == "box_geom"
    assert plan.geom1_id == 0
    assert plan.geom2_id == 1


def test_metadata_rejects_contact_sensor_data_channels(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="found"):
        _scan_contact_sensor(
            tmp_path,
            '<contact name="c" geom1="floor" geom2="box_geom" data="force" num="1"/>',
        )


def test_metadata_rejects_contact_sensor_num_above_one(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="num=1"):
        _scan_contact_sensor(
            tmp_path,
            '<contact name="c" geom1="floor" geom2="box_geom" data="found" num="2"/>',
        )


@pytest.mark.parametrize(
    "sensor",
    [
        '<contact name="c" body1="box" data="found" num="1"/>',
        '<contact name="c" site="box_site" data="found" num="1"/>',
    ],
)
def test_metadata_rejects_contact_sensor_non_geom_pairs(tmp_path: Path, sensor: str) -> None:
    with pytest.raises(NotImplementedError, match="geom1/geom2"):
        _scan_contact_sensor(tmp_path, sensor)


def test_newton_warp_path_has_no_mj_data_access() -> None:
    for path in Path(__file__).parents[1].joinpath("src/unisim/backend/newton").glob("*.py"):
        assert "mj_data" not in path.read_text(encoding="utf-8")
