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


def test_source_builder_preserves_authored_free_joint_fields(tmp_path: Path) -> None:
    """Free-root passive fields must survive import, finalization, and solver conversion."""
    path = _write_model(
        tmp_path,
        "<body name='base'>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<joint name='root' type='free' armature='2' frictionloss='.3' "
        "actuatorfrcrange='-8 8'/>"
        "<geom type='sphere' size='.1'/>"
        "</body>",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    expected = np.full((6,), 2.0, dtype=np.float32)
    np.testing.assert_array_equal(metadata.dof_armature, expected)
    np.testing.assert_array_equal(metadata.dof_friction, np.full((6,), 0.3, dtype=np.float32))
    np.testing.assert_array_equal(metadata.joint_effort_limit, np.full((6,), 1e6, dtype=np.float32))

    builder = build_source_builder(newton, str(path), metadata=metadata)
    np.testing.assert_array_equal(
        np.asarray(builder.joint_armature, dtype=np.float32)[:6], expected
    )
    model = builder.finalize(device="cpu")
    np.testing.assert_array_equal(
        np.asarray(model.joint_armature.numpy(), dtype=np.float32)[:6], expected
    )
    np.testing.assert_array_equal(
        model.joint_friction.numpy(), np.full((6,), 0.3, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        model.joint_effort_limit.numpy(), np.full((6,), 1e6, dtype=np.float32)
    )
    solver = newton.solvers.SolverMuJoCo(model, nconmax=4, njmax=4)
    np.testing.assert_array_equal(
        np.asarray(solver.mjw_model.dof_armature.numpy(), dtype=np.float32).reshape(-1)[
            :6
        ],
        expected,
    )
    np.testing.assert_allclose(
        np.asarray(solver.mjw_model.dof_frictionloss.numpy()).reshape(-1)[:6], 0.3
    )
    assert not solver.mjw_model.jnt_actfrclimited.numpy().reshape(-1)[0]


def test_source_builder_restores_canonical_general_target_actuators(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        "<body name='positioned'><joint name='positioned_joint' type='hinge' "
        "actuatorfrcrange='-45 45'/>"
        "<geom type='sphere' size='.05'/></body>"
        "<body name='driven'><joint name='driven_joint' type='hinge'/>"
        "<geom type='sphere' size='.05'/></body>",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "</worldbody>",
            "</worldbody><actuator>"
            "<general name='position' joint='positioned_joint' biastype='affine' "
            "gainprm='25' biasprm='0 -25 0'/>"
            "<general name='velocity' joint='positioned_joint' biastype='affine' "
            "gainprm='6' biasprm='0 0 -6'/>"
            "<motor name='motor' joint='driven_joint'/>"
            "</actuator>",
        ),
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    assert metadata.actuator_joint_target_compatible == (True, True, False)
    np.testing.assert_array_equal(metadata.joint_effort_limit, [45, 1e6])
    builder = build_source_builder(newton, str(path), metadata=metadata)
    assert builder.custom_attributes["mujoco:ctrl_source"].values == [0, 0, 1]
    assert builder.custom_attributes["mujoco:ctrl_type"].values == [0, 1, 2]
    assert builder.joint_target_mode == [int(newton.JointTargetMode.POSITION_VELOCITY), 0]
    assert builder.joint_target_ke == [25.0, 0.0]
    assert builder.joint_target_kd == [6.0, 0.0]

    solver = newton.solvers.SolverMuJoCo(builder.finalize(device="cpu"), nconmax=4, njmax=4)
    sources = solver.mjc_actuator_ctrl_source.numpy()
    mappings = solver.mjc_actuator_to_newton_idx.numpy()
    assert sources.tolist() == [0, 0, 1]
    np.testing.assert_allclose(solver.mj_model.actuator_gainprm[:, 0], [25, 6, 1])
    np.testing.assert_allclose(solver.mj_model.actuator_biasprm[:, 1], [-25, 0, 0])
    np.testing.assert_allclose(solver.mj_model.actuator_biasprm[:, 2], [0, -6, 0])
    assert mappings.tolist() == [0, -2, 2]


def test_source_builder_rejects_third_same_kind_target_on_same_dof(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        "<body name='base'><joint name='hinge' type='hinge'/>"
        "<geom type='sphere' size='.05'/></body>",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "</worldbody>",
            "</worldbody><actuator>"
            "<general name='position_a' joint='hinge' biastype='affine' "
            "gainprm='25' biasprm='0 -25 0'/>"
            "<general name='velocity' joint='hinge' biastype='affine' "
            "gainprm='6' biasprm='0 0 -6'/>"
            "<general name='position_b' joint='hinge' biastype='affine' "
            "gainprm='30' biasprm='0 -30 0'/>"
            "</actuator>",
        ),
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    with pytest.raises(RuntimeError, match="multiple position target actuators on same DOF"):
        build_source_builder(newton, str(path), metadata=metadata)


@pytest.mark.parametrize(
    "actuator",
    [
        "<general name='position' joint='hinge' biastype='affine' "
        "gainprm='25' biasprm='0 -25 0'/>",
        "<position name='position' joint='hinge' kp='25'/>",
    ],
)
def test_source_builder_pure_position_has_no_velocity_damping(
    tmp_path: Path, actuator: str
) -> None:
    path = _write_model(
        tmp_path,
        "<body name='base'><joint name='hinge' type='hinge'/>"
        "<geom type='sphere' size='.05'/></body>",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "</worldbody>", f"</worldbody><actuator>{actuator}</actuator>"
        ),
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    np.testing.assert_array_equal(metadata.actuator_kd, [0.0])
    builder = build_source_builder(newton, str(path), metadata=metadata)
    assert builder.joint_target_mode == [int(newton.JointTargetMode.POSITION)]
    assert builder.joint_target_kd == [0.0]
    solver = newton.solvers.SolverMuJoCo(builder.finalize(device="cpu"), nconmax=4, njmax=4)
    np.testing.assert_array_equal(solver.mj_model.actuator_biasprm[:, 2], [0.0])


def test_metadata_rejects_asymmetric_actuator_force_range(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        "<body name='base'><joint name='hinge' type='hinge' "
        "actuatorfrcrange='-5 45'/><geom type='sphere' size='.05'/></body>",
    )
    with pytest.raises(NotImplementedError, match="asymmetric actuator force range"):
        scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


def test_source_builder_keeps_dynamic_general_actuator_direct(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        "<body name='base'><joint name='hinge' type='hinge'/>"
        "<geom type='sphere' size='.05'/></body>",
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "</worldbody>",
            "</worldbody><actuator><general name='filtered' joint='hinge' "
            "dyntype='filter' biastype='affine' gainprm='25' biasprm='0 -25 0'/>"
            "</actuator>",
        ),
        encoding="utf-8",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    assert metadata.actuator_joint_target_compatible == (False,)
    builder = build_source_builder(newton, str(path), metadata=metadata)
    assert builder.custom_attributes["mujoco:ctrl_source"].values == [1]
    assert builder.custom_attributes["mujoco:ctrl_type"].values == [2]
    assert builder.joint_target_mode == [0]


def test_source_builder_keeps_mujoco_default_rigid_gap(tmp_path: Path) -> None:
    """MuJoCo's implicit ``gap=0`` must not become Newton's 0.1 fallback."""
    # The spheres are 0.15 m apart and have combined radius 0.10 m. MuJoCo
    # therefore reports no contact, while Newton's legacy ``rigid_gap=0.1``
    # fallback would widen each candidate into a false contact.
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
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    builder = build_source_builder(newton, str(path), metadata=metadata)
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


_WEDGE_VERTICES = "0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"
_WEDGE_FACES = "0 2 1  0 1 3  0 3 2  1 2 3"
_DENTED_VERTICES = "0 0 0  0.1 0 0  0 0.1 0  0 0 0.1  0.0167 0.0167 0.0167"
_DENTED_FACES = "0 2 1  0 1 3  0 3 2  1 2 4  2 3 4  3 1 4"
_OPEN_FACES = "0 2 1  0 1 3  0 3 2"


def _sphere_mesh(rings: int = 8, sectors: int = 12) -> tuple[str, str, int]:
    vertices = [(0.0, 0.0, 0.1)]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            vertices.append(
                (
                    0.1 * np.sin(phi) * np.cos(theta),
                    0.1 * np.sin(phi) * np.sin(theta),
                    0.1 * np.cos(phi),
                )
            )
    vertices.append((0.0, 0.0, -0.1))
    bottom = len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for sector in range(sectors):
        faces.append((0, 1 + sector % sectors, 1 + (sector + 1) % sectors))
    for ring in range(rings - 2):
        upper = 1 + ring * sectors
        lower = upper + sectors
        for sector in range(sectors):
            a, b = upper + sector % sectors, upper + (sector + 1) % sectors
            c, d = lower + sector % sectors, lower + (sector + 1) % sectors
            faces.extend(((a, c, d), (a, d, b)))
    last = 1 + (rings - 2) * sectors
    for sector in range(sectors):
        faces.append((last + sector % sectors, bottom, last + (sector + 1) % sectors))
    vertex_text = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face_text = "  ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex_text, face_text, len(vertices)


def _scan_mesh_scene(
    tmp_path: Path,
    vertices: str,
    faces: str,
    *,
    asset_attributes: str = "",
    geom_attributes: str = "",
):
    path = _write_model(tmp_path, "")
    path.write_text(
        "<mujoco model='mesh-test'>"
        f"<asset><mesh name='piece' {asset_attributes} "
        f"vertex='{vertices}' face='{faces}'/></asset>"
        "<worldbody><geom name='floor' type='plane' size='1 1 0.1'/>"
        "<body name='mesh_body' pos='0 0 0.5'><joint name='root' type='free'/>"
        f"<geom name='mesh_geom' type='mesh' mesh='piece' mass='0.5' {geom_attributes}/>"
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


@pytest.mark.parametrize(
    ("vertices", "faces"),
    [
        (_WEDGE_VERTICES, _WEDGE_FACES),
        (_DENTED_VERTICES, _DENTED_FACES),
        (_WEDGE_VERTICES, _OPEN_FACES),
    ],
)
def test_metadata_accepts_collision_mesh_for_newton_hull_construction(
    tmp_path: Path, vertices: str, faces: str
) -> None:
    metadata = _scan_mesh_scene(tmp_path, vertices, faces)
    assert metadata.nq == 7
    assert metadata.geom_contype.tolist() == [1, 1]


@pytest.mark.parametrize("asset_attributes", ["", 'maxhullvert="128"'])
def test_metadata_preserves_newton_mesh_hull_budget_ownership(
    tmp_path: Path, asset_attributes: str
) -> None:
    vertices, faces, count = _sphere_mesh()
    metadata = _scan_mesh_scene(
        tmp_path, vertices, faces, asset_attributes=asset_attributes
    )
    assert metadata.nq == 7
    assert count > 64


def test_metadata_preserves_visual_only_mesh_geom(tmp_path: Path) -> None:
    metadata = _scan_mesh_scene(
        tmp_path,
        _DENTED_VERTICES,
        _DENTED_FACES,
        geom_attributes='contype="0" conaffinity="0"',
    )
    assert metadata.geom_contype.tolist() == [1, 0]
    assert metadata.geom_conaffinity.tolist() == [1, 0]


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


_BALL_SCENE = """
<mujoco model="ball-scan-test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hand" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="palm" type="sphere" size="0.03" mass="0.5"/>
      <body name="finger" pos="0 0 0.05">
        {ball_joint}
        <geom name="tip" type="sphere" size="0.02" mass="0.1"/>
        <body name="nail" pos="0 0 0.03">
          <joint name="flex" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="nail_geom" type="sphere" size="0.01" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>{actuator}</actuator>
</mujoco>
"""


def _scan_ball_scene(
    tmp_path: Path,
    *,
    ball_joint: str = '<joint name="wrist_ball" type="ball"/>',
    actuator: str = '<motor name="flex_motor" joint="flex" ctrlrange="-1 1"/>',
):
    path = _write_model(tmp_path, "")
    path.write_text(
        _BALL_SCENE.format(ball_joint=ball_joint, actuator=actuator), encoding="utf-8"
    )
    return scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


def test_metadata_accepts_ball_joints(tmp_path: Path) -> None:
    metadata = _scan_ball_scene(tmp_path)
    assert metadata.nq == 12
    assert metadata.nv == 10
    assert metadata.ball_joint_names == ("wrist_ball",)
    assert metadata.ball_joint_qpos_adrs == (7,)
    assert metadata.ball_joint_qvel_adrs == (6,)
    assert metadata.ball_joint_addresses == ((7, 6),)
    # Ball joints stay passive multi-DoF joints: the name-addressable
    # single-DoF joint API and its aligned joint-range rows are unchanged.
    assert metadata.joint_names == ("flex",)
    assert metadata.joint_qpos_adrs == (11,)
    assert metadata.joint_dof_adrs == (9,)
    assert metadata.joint_range is not None
    np.testing.assert_allclose(metadata.joint_range, [[-1.5, 1.5]])
    np.testing.assert_allclose(metadata.default_qpos[7:11], [1.0, 0.0, 0.0, 0.0])


def test_metadata_expands_ball_joint_passive_fields(tmp_path: Path) -> None:
    metadata = _scan_ball_scene(
        tmp_path,
        ball_joint='<joint name="wrist_ball" type="ball" frictionloss=".2" '
        'armature=".1" actuatorfrcrange="-45 45"/>',
    )
    np.testing.assert_allclose(metadata.dof_friction[6:9], 0.2)
    np.testing.assert_allclose(metadata.dof_armature[6:9], 0.1)
    # MuJoCo does not enable actuator force limits on a ball joint; do not
    # reinterpret an ignored range as a native Newton limit.
    np.testing.assert_array_equal(metadata.joint_effort_limit[6:9], [1e6, 1e6, 1e6])


def test_metadata_rejects_actuators_targeting_ball_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="single-DoF"):
        _scan_ball_scene(
            tmp_path, actuator='<motor name="wrist_motor" joint="wrist_ball"/>'
        )


def test_metadata_rejects_unnamed_ball_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="named ball joints"):
        _scan_ball_scene(tmp_path, ball_joint='<joint type="ball"/>')
