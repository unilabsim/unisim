"""Tests for the URDF branch of ``scan_scene_metadata`` (SimToolReal step 0)."""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend
from unisim.backend.subprocess_ipc.sensors import scan_scene_entities, scan_scene_metadata
from unisim.scene import ActuatorGainOverride, SceneCfg, SceneEntitySpec


class _EntityAssetHarnessBackend(MjcfSubprocessBackend):
    """Family harness opting into composition consumption for host-side tests.

    The constructor gate rejects declared entity assets/ground planes unless
    the adapter opts in; these tests exercise the family's serialization and
    binding machinery that the IsaacSim realization stands on.
    """

    def _supports_entity_assets(self) -> bool:
        return True

    def _supports_ground_plane(self) -> bool:
        return True


URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="mount" type="fixed">
    <parent link="base_link"/>
    <child link="arm_mid"/>
  </joint>
  <link name="arm_mid"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="arm"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
  </joint>
  <joint name="free_spin" type="continuous">
    <parent link="arm"/>
    <child link="arm_mid"/>
    <limit effort="5" velocity="11.6"/>
  </joint>
</robot>
"""

OBJECT_URDF = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""

TABLE_MJCF = """<mujoco model='table'>
  <worldbody><body name='table_body'><freejoint/>
    <geom type='box' size='0.3 0.3 0.02'/></body></worldbody>
</mujoco>"""


@pytest.fixture()
def urdf_file(tmp_path):
    path = tmp_path / "two_link.urdf"
    path.write_text(URDF)
    return path


@pytest.fixture()
def object_urdf_file(tmp_path):
    path = tmp_path / "cube.urdf"
    path.write_text(OBJECT_URDF)
    return path


@pytest.fixture()
def table_mjcf_file(tmp_path):
    path = tmp_path / "table.xml"
    path.write_text(TABLE_MJCF)
    return path


def test_urdf_branch_reports_links_and_movable_joints(urdf_file):
    meta = scan_scene_metadata(str(urdf_file), backend_label="isaacsim")
    # merge_fixed_joints semantics: fixed-joint children (arm_mid) are absorbed
    # into their parent and disappear from the contract body list.
    assert meta.body_names == ("base_link", "arm")
    # Fixed joints are dropped; revolute/continuous keep document order.
    assert meta.joint_names == ("shoulder", "free_spin")
    assert meta.freejoint_body_name is None
    assert meta.keyframes == {}
    assert meta.sensors == {}
    assert meta.joint_ranges[0] == (-1.57, 1.57)
    # Continuous joints have no lower/upper bounds.
    assert meta.joint_ranges[1] == (-np.inf, np.inf)


def test_urdf_branch_synthesizes_zero_gain_position_actuators(urdf_file):
    meta = scan_scene_metadata(str(urdf_file), backend_label="isaacsim")
    assert len(meta.actuators) == 2
    shoulder, free_spin = meta.actuators
    assert shoulder.joint_name == "shoulder"
    assert shoulder.kp == 0.0 and shoulder.kv == 0.0
    assert shoulder.forcerange == (-300.0, 300.0)
    assert shoulder.ctrlrange == (-1.57, 1.57)
    assert free_spin.forcerange == (-5.0, 5.0)
    assert free_spin.ctrlrange is None


def test_urdf_branch_rejects_unsupported_joint_types(tmp_path):
    path = tmp_path / "bad.urdf"
    path.write_text(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j" type="planar"><parent link="a"/><child link="b"/></joint></robot>'
    )
    with pytest.raises(NotImplementedError, match="planar"):
        scan_scene_metadata(str(path), backend_label="isaacsim")


def test_urdf_floating_root_reports_root_link(object_urdf_file):
    meta = scan_scene_metadata(str(object_urdf_file), urdf_fixed_base=False)
    assert meta.freejoint_body_name == "cube_link"
    fixed = scan_scene_metadata(str(object_urdf_file), urdf_fixed_base=True)
    assert fixed.freejoint_body_name is None


def test_urdf_branch_rejects_multi_root_documents(tmp_path):
    path = tmp_path / "two_roots.urdf"
    path.write_text('<robot name="x"><link name="a"/><link name="b"/></robot>')
    with pytest.raises(ValueError, match="exactly one root link"):
        scan_scene_metadata(str(path))


def test_urdf_fixed_base_flag_rejected_for_mjcf(table_mjcf_file):
    with pytest.raises(ValueError, match="URDF scenes only"):
        scan_scene_metadata(str(table_mjcf_file), urdf_fixed_base=False)


def _simtoolreal_specs(urdf_file, object_urdf_file, table_mjcf_file):
    return (
        SceneEntitySpec(
            name="robot",
            model_file=str(urdf_file),
            asset_format="urdf",
            materialization="articulation",
            root_mode="fixed",
            actuator_gain_overrides=(
                ActuatorGainOverride(
                    joint_name="shoulder", stiffness=100.0, damping=5.0, armature=0.1
                ),
            ),
        ),
        SceneEntitySpec(
            name="table",
            model_file=str(table_mjcf_file),
            asset_format="mjcf",
            materialization="rigid",
            root_mode="floating",
        ),
        SceneEntitySpec(
            name="object",
            model_file=str(object_urdf_file),
            asset_format="urdf",
            materialization="rigid",
            root_mode="floating",
        ),
        SceneEntitySpec(
            name="goalviz",
            model_file=str(object_urdf_file),
            asset_format="urdf",
            materialization="rigid",
            root_mode="kinematic",
        ),
    )


def test_scan_scene_entities_multi_asset_per_role_fixed_base(
    urdf_file, object_urdf_file, table_mjcf_file
):
    specs = _simtoolreal_specs(urdf_file, object_urdf_file, table_mjcf_file)
    metadata = scan_scene_entities(specs, backend_label="isaacsim")
    assert list(metadata) == ["robot", "table", "object", "goalviz"]
    # Fixed-base articulation: no free root.
    assert metadata["robot"].freejoint_body_name is None
    assert metadata["robot"].joint_names == ("shoulder", "free_spin")
    # Floating/kinematic URDF roots report the root link as the free body.
    assert metadata["object"].freejoint_body_name == "cube_link"
    assert metadata["goalviz"].freejoint_body_name == "cube_link"
    # MJCF root motion comes from content.
    assert metadata["table"].freejoint_body_name == "table_body"


def test_scan_scene_entities_validates_gain_overrides(urdf_file):
    spec = SceneEntitySpec(
        name="robot",
        model_file=str(urdf_file),
        asset_format="urdf",
        materialization="articulation",
        root_mode="fixed",
        actuator_gain_overrides=(
            ActuatorGainOverride(joint_name="elbow", stiffness=1.0, damping=0.1),
        ),
    )
    with pytest.raises(ValueError, match="elbow"):
        scan_scene_entities((spec,), backend_label="isaacsim")


def test_scan_scene_entities_rejects_duplicate_names(urdf_file):
    spec = SceneEntitySpec(
        name="object",
        model_file=str(urdf_file),
        asset_format="urdf",
        materialization="rigid",
        root_mode="floating",
    )
    with pytest.raises(ValueError, match="unique"):
        scan_scene_entities((spec, spec), backend_label="isaacsim")


def test_scan_scene_entities_mjcf_root_mode_cross_check(table_mjcf_file, urdf_file):
    # Declared fixed but the MJCF has a freejoint.
    fixed = SceneEntitySpec(
        name="table",
        model_file=str(table_mjcf_file),
        asset_format="mjcf",
        materialization="rigid",
        root_mode="fixed",
    )
    with pytest.raises(ValueError, match="freejoint"):
        scan_scene_entities((fixed,), backend_label="isaacsim")
    # Declared floating but the MJCF has no freejoint.
    fixed_mjcf = urdf_file.with_name("fixed.xml")
    fixed_mjcf.write_text(
        "<mujoco model='arm'><worldbody><body name='base'>"
        "<joint name='j' type='slide'/><geom type='box' size='0.1 0.1 0.1'/>"
        "</body></worldbody></mujoco>"
    )
    floating = SceneEntitySpec(
        name="robot",
        model_file=str(fixed_mjcf),
        asset_format="mjcf",
        materialization="articulation",
        root_mode="floating",
    )
    with pytest.raises(ValueError, match="no freejoint"):
        scan_scene_entities((floating,), backend_label="isaacsim")


def _host_backend(urdf_file, object_urdf_file, table_mjcf_file):
    scene = SceneCfg(
        model_file=str(urdf_file),
        entity_assets=_simtoolreal_specs(urdf_file, object_urdf_file, table_mjcf_file),
    )
    # Host-only construction: no worker is spawned before materialize().
    return _EntityAssetHarnessBackend(scene, num_envs=2, sim_dt=0.01)


def test_host_primary_payload_applies_entity_gain_overrides(
    urdf_file, object_urdf_file, table_mjcf_file
):
    backend = _host_backend(urdf_file, object_urdf_file, table_mjcf_file)
    payload = backend._position_actuation_payload()
    assert payload["dof_stiffness"] == [100.0, 0.0]
    assert payload["dof_damping"] == [5.0, 0.0]
    assert payload["dof_armature"] == [0.1, 0.0]
    assert payload["dof_friction"] == [0.0, 0.0]
    # Scanned effort/limits still come from the URDF scan.
    assert payload["dof_effort"] == [300.0, 5.0]


def test_host_entity_payloads_shape(urdf_file, object_urdf_file, table_mjcf_file):
    backend = _host_backend(urdf_file, object_urdf_file, table_mjcf_file)
    payloads = backend._entity_payloads()
    assert [entry["name"] for entry in payloads] == ["robot", "table", "object", "goalviz"]
    robot, table, obj, goalviz = payloads
    assert robot["asset_format"] == "urdf"
    assert robot["materialization"] == "articulation"
    assert robot["fixed_base"] is True
    # Fixed-base URDF: no free-joint body, so the payload names the scanned
    # root link (the worker needs it for the ArticulationRootAPI patch).
    assert robot["root_body_name"] == "base_link"
    assert robot["joint_names"] == ["shoulder", "free_spin"]
    assert robot["dof_stiffness"] == [100.0, 0.0]
    assert table["asset_format"] == "mjcf"
    assert table["fixed_base"] is False
    assert table["root_body_name"] == "table_body"
    assert obj["fixed_base"] is False and obj["root_body_name"] == "cube_link"
    assert goalviz["root_mode"] == "kinematic"
    assert goalviz["fixed_base"] is False and goalviz["root_body_name"] == "cube_link"
    for entry in payloads:
        for key in ("dof_stiffness", "dof_damping", "dof_effort", "dof_armature", "dof_friction"):
            assert len(entry[key]) == len(entry["joint_names"])


def test_host_without_entity_assets_keeps_legacy_payload(urdf_file):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=str(urdf_file)), num_envs=2, sim_dt=0.01
    )
    assert backend._entity_payloads() == []
    payload = backend._position_actuation_payload()
    assert payload["dof_stiffness"] == [0.0, 0.0]
    # Step-0 default: undeclared URDF primary stays fixed-base.
    assert backend._get_scene_metadata().freejoint_body_name is None


def test_host_primary_urdf_honors_declared_root_mode(object_urdf_file):
    scene = SceneCfg(
        model_file=str(object_urdf_file),
        entity_assets=(
            SceneEntitySpec(
                name="object",
                model_file=str(object_urdf_file),
                asset_format="urdf",
                materialization="rigid",
                root_mode="floating",
            ),
        ),
    )
    backend = _EntityAssetHarnessBackend(scene, num_envs=1, sim_dt=0.01)
    assert backend._get_scene_metadata().freejoint_body_name == "cube_link"


def test_host_gain_override_unknown_joint_fails_closed(urdf_file):
    backend = _EntityAssetHarnessBackend(
        SceneCfg(model_file=str(urdf_file)), num_envs=1, sim_dt=0.01
    )
    metadata = backend._get_scene_metadata()
    with pytest.raises(ValueError, match="not in the scanned asset"):
        backend._position_actuation_payload(
            metadata,
            gain_overrides=(ActuatorGainOverride(joint_name="nope", stiffness=1.0, damping=1.0),),
        )
