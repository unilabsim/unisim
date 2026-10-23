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
    """#441: body_iquat is scanned so the variant audit can rotate the
    principal-frame diagonal into Newton's body-frame inertia tensor."""
    path = _write_model(
        tmp_path,
        '<body name="base"><joint name="hinge" type="hinge"/>'
        '<inertial pos="0 0 0" mass="1" fullinertia=".11 .2 .3 .015 -.02 .005"/>'
        "</body>",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    iquat = np.asarray(metadata.body_iquat[1], dtype=np.float64)
    assert not np.allclose(iquat, [1, 0, 0, 0], atol=1e-6)
    w, x, y, z = iquat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    authored = np.array([[0.11, 0.015, -0.02], [0.015, 0.2, 0.005], [-0.02, 0.005, 0.3]])
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


def test_source_builder_preserves_authored_free_joint_armature(
    tmp_path: Path,
) -> None:
    """A direct-MJCF free-root armature must survive Newton import (#445)."""
    path = _write_model(
        tmp_path,
        "<body name='base'>"
        "<inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>"
        "<joint name='root' type='free' armature='2'/>"
        "<geom type='sphere' size='.1'/>"
        "</body>",
    )
    metadata = scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))
    expected = np.full((6,), 2.0, dtype=np.float32)
    np.testing.assert_array_equal(metadata.dof_armature, expected)

    builder = build_source_builder(newton, str(path), metadata=metadata)
    np.testing.assert_array_equal(
        np.asarray(builder.joint_armature, dtype=np.float32)[:6], expected
    )
    model = builder.finalize(device="cpu")
    np.testing.assert_array_equal(
        np.asarray(model.joint_armature.numpy(), dtype=np.float32)[:6], expected
    )
    solver = newton.solvers.SolverMuJoCo(model, nconmax=4, njmax=4)
    np.testing.assert_array_equal(
        np.asarray(solver.mjw_model.dof_armature.numpy(), dtype=np.float32).reshape(-1)[
            :6
        ],
        expected,
    )


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
    state_a = model.state()
    state_b = model.state()
    control = model.control()
    state_a.clear_forces()
    solver.step(state_a, state_b, control, contacts, 0.01)
    assert int(np.asarray(solver.mjw_data.nacon.numpy()).reshape(-1)[0]) == 0


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


_WEDGE_VERTICES = "0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"
_WEDGE_FACES = "0 2 1  0 1 3  0 3 2  1 2 3"
# The wedge with one hull-face centroid pushed halfway toward the opposite
# vertex: a closed but non-convex surface.
_DENTED_VERTICES = "0 0 0  0.1 0 0  0 0.1 0  0 0 0.1  0.0167 0.0167 0.0167"
_DENTED_FACES = "0 2 1  0 1 3  0 3 2  1 2 4  2 3 4  3 1 4"
# The wedge missing its (1, 2, 3) hull face: an open surface.
_OPEN_FACES = "0 2 1  0 1 3  0 3 2"


def _sphere_mesh(rings: int = 8, sectors: int = 12) -> tuple[str, str, int]:
    """Build a closed UV-sphere whose vertices all lie on its convex hull."""
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
            faces.append((a, c, d))
            faces.append((a, d, b))
    last = 1 + (rings - 2) * sectors
    for sector in range(sectors):
        faces.append((last + sector % sectors, bottom, last + (sector + 1) % sectors))
    vertex_text = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face_text = "  ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex_text, face_text, len(vertices)


def _mesh_scene(mesh_asset: str, geom_attributes: str = "") -> str:
    return (
        "<mujoco model='mesh-audit-test'>"
        f"<asset>{mesh_asset}</asset>"
        "<worldbody><geom name='floor' type='plane' size='1 1 0.1'/>"
        "<body name='mesh_body' pos='0 0 0.5'><joint name='root' type='free'/>"
        f"<geom name='mesh_geom' type='mesh' mesh='piece' mass='0.5' {geom_attributes}/>"
        "</body></worldbody></mujoco>"
    )


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
        _mesh_scene(
            f"<mesh name='piece' {asset_attributes} vertex='{vertices}' face='{faces}'/>",
            geom_attributes,
        ),
        encoding="utf-8",
    )
    return scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


def test_metadata_accepts_convex_collision_mesh_geom(tmp_path: Path) -> None:
    metadata = _scan_mesh_scene(tmp_path, _WEDGE_VERTICES, _WEDGE_FACES)
    assert metadata.nq == 7
    assert metadata.geom_contype.tolist() == [1, 1]


def test_metadata_rejects_nonconvex_collision_mesh_geom(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="not convex"):
        _scan_mesh_scene(tmp_path, _DENTED_VERTICES, _DENTED_FACES)


def test_metadata_rejects_open_collision_mesh_geom(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="not closed"):
        _scan_mesh_scene(tmp_path, _WEDGE_VERTICES, _OPEN_FACES)


def test_metadata_rejects_mesh_hull_above_default_maxhullvert(tmp_path: Path) -> None:
    vertices, faces, count = _sphere_mesh()
    assert count > 64
    with pytest.raises(NotImplementedError, match="maxhullvert"):
        _scan_mesh_scene(tmp_path, vertices, faces)


def test_metadata_accepts_mesh_hull_with_authored_maxhullvert(tmp_path: Path) -> None:
    vertices, faces, count = _sphere_mesh()
    metadata = _scan_mesh_scene(tmp_path, vertices, faces, asset_attributes='maxhullvert="128"')
    assert metadata.nq == 7
    assert count > 64


def test_metadata_skips_visual_only_mesh_geoms(tmp_path: Path) -> None:
    metadata = _scan_mesh_scene(
        tmp_path,
        _DENTED_VERTICES,
        _DENTED_FACES,
        geom_attributes='contype="0" conaffinity="0"',
    )
    assert metadata.geom_contype.tolist() == [1, 0]


_CUBE_VERTICES = "0 0 0  1 0 0  0 1 0  0 0 1  1 1 0  1 0 1  0 1 1  1 1 1"
_CUBE_FACES = (
    "0 2 1  1 2 4  0 1 5  0 5 3  1 4 7  1 7 5  2 6 4  4 6 7  3 5 7"
    "  3 7 6  0 3 6  0 6 2"
)


def _scan_two_mesh_scene(tmp_path: Path, second_vertices: str, second_faces: str):
    path = _write_model(tmp_path, "")
    path.write_text(
        "<mujoco model='two-mesh-audit-test'>"
        "<asset>"
        f"<mesh name='piece_a' vertex='{_WEDGE_VERTICES}' face='{_WEDGE_FACES}'/>"
        f"<mesh name='piece_b' vertex='{second_vertices}' face='{second_faces}'/>"
        "</asset>"
        "<worldbody>"
        "<body name='body_a' pos='0 0 0.5'><joint name='root_a' type='free'/>"
        "<geom name='geom_a' type='mesh' mesh='piece_a' mass='0.5'/></body>"
        "<body name='body_b' pos='0 0 2'><joint name='root_b' type='free'/>"
        "<geom name='geom_b' type='mesh' mesh='piece_b' mass='0.5'/></body>"
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    return scan_newton_model_metadata(mujoco, SceneCfg(model_file=str(path)))


def test_metadata_audits_mesh_at_nonzero_vertex_offset(tmp_path: Path) -> None:
    # mesh_polyvert indexes the mesh's own vertex block: the cube's compiled
    # hull must be read at its mesh_vertadr, not at the model vertex origin.
    metadata = _scan_two_mesh_scene(tmp_path, _CUBE_VERTICES, _CUBE_FACES)
    assert metadata.nq == 14


def test_metadata_rejects_nonconvex_mesh_at_nonzero_vertex_offset(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="piece_b"):
        _scan_two_mesh_scene(tmp_path, _DENTED_VERTICES, _DENTED_FACES)


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


def test_metadata_rejects_actuators_targeting_ball_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="single-DoF"):
        _scan_ball_scene(
            tmp_path, actuator='<motor name="wrist_motor" joint="wrist_ball"/>'
        )


def test_metadata_rejects_unnamed_ball_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="named ball joints"):
        _scan_ball_scene(tmp_path, ball_joint='<joint type="ball"/>')
