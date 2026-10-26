"""Cold-path MJCF compensation and fail-closed asset checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.newton.materialization import (
    build_source_builder,
    scan_newton_model_metadata,
)
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")
newton = pytest.importorskip("newton")


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


def test_metadata_scans_principal_inertia_frame(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        '<body name="base"><joint name="hinge" type="hinge"/>'
        '<inertial pos="0 0 0" mass="1" fullinertia=".11 .2 .3 .015 -.02 .005"/>'
        "</body>",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    w, x, y, z = np.asarray(metadata.body_iquat[1], dtype=np.float64)
    assert not np.allclose([w, x, y, z], [1, 0, 0, 0], atol=1e-6)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    authored = np.array(
        [[0.11, 0.015, -0.02], [0.015, 0.2, 0.005], [-0.02, 0.005, 0.3]]
    )
    rebuilt = rotation @ np.diag(metadata.body_inertia[1].astype(np.float64)) @ rotation.T
    np.testing.assert_allclose(rebuilt, authored, atol=1e-6)


def test_metadata_preserves_nonleading_passive_and_actuator_addresses(
    tmp_path: Path,
) -> None:
    path = _write_model(
        tmp_path,
        "<body name='base'><freejoint name='root'/>"
        "<geom type='sphere' size='0.1'/>"
        "<body name='passive_link'><joint name='passive' type='hinge' axis='0 0 1'/>"
        "<geom type='sphere' size='0.05'/></body>"
        "<body name='driven_link'><joint name='driven' type='hinge' axis='0 1 0'/>"
        "<geom type='sphere' size='0.05'/></body></body>",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "</worldbody>",
            "</worldbody><actuator><motor name='drive' joint='driven'/></actuator>",
        ),
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    assert metadata.root_qpos_dim == 7
    assert metadata.root_qvel_dim == 6
    assert metadata.joint_names == ("passive", "driven")
    assert metadata.joint_qpos_adrs == (7, 8)
    assert metadata.joint_dof_adrs == (6, 7)
    assert metadata.actuator_names == ("drive",)
    assert metadata.nu == 1
    assert metadata.actuator_target_qpos_adrs == (8,)
    assert metadata.actuator_target_qvel_adrs == (7,)


def test_source_builder_keeps_mujoco_default_rigid_gap(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        "<body name='a' pos='0 0 0'>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<freejoint name='a_root'/>"
        "<geom type='sphere' size='.05'/>"
        "</body>"
        "<body name='b' pos='0 0 .15'>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<freejoint name='b_root'/>"
        "<geom type='sphere' size='.05'/>"
        "</body>",
    )
    builder = build_source_builder(newton, str(path))
    assert builder.rigid_gap == 0.0
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, nconmax=4, njmax=4)
    contacts = newton.Contacts(
        rigid_contact_max=solver.get_max_contact_count(),
        soft_contact_max=0,
        device="cpu",
    )
    state = model.state()
    state_out = model.state()
    control = model.control()
    state.clear_forces()
    solver.step(state, state_out, control, contacts, 0.01)
    assert int(np.asarray(solver.mjw_data.nacon.numpy()).reshape(-1)[0]) == 0
def test_source_builder_preserves_authored_free_joint_armature(tmp_path: Path) -> None:
    newton = pytest.importorskip("newton")
    path = _write_model(
        tmp_path,
        "<body name='base'>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<joint name='root' type='free' armature='2'/>"
        "<geom type='sphere' size='.1'/>"
        "</body>",
    )
    direct_model = mujoco.MjModel.from_xml_path(str(path))
    expected = np.full((6,), 2.0, dtype=np.float32)
    np.testing.assert_array_equal(direct_model.dof_armature, expected)

    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    builder = build_source_builder(newton, str(path), metadata=metadata)
    np.testing.assert_array_equal(builder.joint_armature, expected)

    model = builder.finalize(device="cpu")
    np.testing.assert_array_equal(model.joint_armature.numpy(), expected)

    solver = newton.solvers.SolverMuJoCo(model, nconmax=4, njmax=4)
    solver_armature = solver.mjw_model.dof_armature.numpy().reshape(-1)
    np.testing.assert_array_equal(solver_armature, expected)


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
