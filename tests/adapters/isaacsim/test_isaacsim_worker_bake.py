"""Tests for the IsaacSim worker bake family (SimToolReal step 1.3b-1).

Pure-Python coverage only: the role bake matrix, URDF adjacency computation
(FilteredPairs source), and SDF collision-marker parsing.  The pxr write side
(``_bake_usd_in_place``/``_apply_self_collision_filters``) is exercised by the
Kit readback probe (step 1.3b-3).
"""

from __future__ import annotations

import pytest

from unisim.backend.isaacsim.worker import (
    _parse_urdf_sdf_collision_markers,
    _usd_safe_identifier,
    bake_plan_for_entity,
    compute_adjacent_link_pairs,
)

# Miniature of the Sharpa/KUKA structure: an arm chain, a fixed-joint mount
# merge (base+mount+hand collapse into one body), and one finger whose two
# revolute joints are split by a "_VL" virtual spacer link.
SHARPA_LIKE_URDF = """<?xml version="1.0"?>
<robot name="mini">
  <link name="link_0"/><link name="link_1"/>
  <link name="mount"/><link name="hand"/>
  <link name="finger_MCP_VL"/><link name="finger_PP"/>
  <link name="finger_DP"/><link name="finger_tip"/>
  <joint name="j1" type="revolute"><parent link="link_0"/><child link="link_1"/></joint>
  <joint name="ee" type="fixed"><parent link="link_1"/><child link="mount"/></joint>
  <joint name="mount_joint" type="fixed"><parent link="mount"/><child link="hand"/></joint>
  <joint name="finger_FE" type="revolute">
    <parent link="hand"/><child link="finger_MCP_VL"/>
  </joint>
  <joint name="finger_AA" type="revolute">
    <parent link="finger_MCP_VL"/><child link="finger_PP"/>
  </joint>
  <joint name="finger_PIP" type="revolute">
    <parent link="finger_PP"/><child link="finger_DP"/>
  </joint>
  <joint name="tip_fix" type="fixed"><parent link="finger_DP"/><child link="finger_tip"/></joint>
</robot>
"""

PLAIN_CHAIN_URDF = """<?xml version="1.0"?>
<robot name="chain">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j_ab" type="revolute"><parent link="a"/><child link="b"/></joint>
  <joint name="j_bc" type="revolute"><parent link="b"/><child link="c"/></joint>
</robot>
"""

SDF_URDF = """<?xml version="1.0"?>
<robot name="sdf_tool">
  <link name="object_root">
    <collision>
      <geometry><mesh filename="meshes/7_hole_patch.obj"/></geometry>
      <sdf resolution="64" margin="0.001" narrowBandThickness="0.02" subgrid_resolution="4"/>
    </collision>
    <collision>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </collision>
  </link>
</robot>
"""


@pytest.fixture()
def sharpa_like_file(tmp_path):
    path = tmp_path / "mini.urdf"
    path.write_text(SHARPA_LIKE_URDF)
    return path


def test_bake_plan_robot_articulation():
    # scene_utils.py:1730-1739.
    plan = bake_plan_for_entity("articulation", "fixed")
    assert plan.props == {
        "disable_gravity": True,
        "max_depenetration_velocity": 1000.0,
        "enabled_self_collisions": True,
        "solver_position_iterations": 8,
        "solver_velocity_iterations": 0,
    }
    assert plan.apply_physx_articulation is True
    assert plan.collision_enabled is None
    # Floating articulation roots take the same robot bake.
    assert bake_plan_for_entity("articulation", "floating") == plan


def test_bake_plan_object_variant_target():
    # scene_utils.py:1707-1713.
    plan = bake_plan_for_entity("rigid", "floating")
    assert plan.props == {
        "kinematic_enabled": False,
        "disable_gravity": False,
        "max_depenetration_velocity": 1000.0,
        "articulation_enabled": False,
    }
    assert plan.apply_physx_articulation is False
    assert plan.collision_enabled is None


def test_bake_plan_kinematic_collision_is_declared():
    # The kinematic bake authors kinematic/gravity; the collision flag is the
    # scene's declaration, never an entity-name guess.  The table contract
    # keeps the converted-USD collision state (scene_utils.py:1752-1758
    # passes no collision flag — the support surface must contact the
    # object), the goalviz contract disables it (scene_utils.py:1714-1719).
    table = bake_plan_for_entity("rigid", "kinematic", collision_enabled=None)
    assert table.props == {
        "kinematic_enabled": True,
        "disable_gravity": True,
        "articulation_enabled": False,
    }
    assert table.apply_physx_articulation is False
    assert table.collision_enabled is None
    goalviz = bake_plan_for_entity("rigid", "kinematic", collision_enabled=False)
    assert goalviz.props == table.props
    assert goalviz.collision_enabled is False
    explicit = bake_plan_for_entity("rigid", "kinematic", collision_enabled=True)
    assert explicit.collision_enabled is True
    # Any kinematic entity name may carry either contract.
    assert bake_plan_for_entity("rigid", "kinematic", collision_enabled=False) == goalviz


def test_bake_plan_rejects_unknown_materializations():
    with pytest.raises(ValueError, match="materialization"):
        bake_plan_for_entity("deformable", "floating")
    with pytest.raises(ValueError, match="root_mode"):
        bake_plan_for_entity("rigid", "fixed")
    # Floating rigids are always the dynamic object contract.
    with pytest.raises(ValueError, match="floating rigid"):
        bake_plan_for_entity("rigid", "floating", is_variant_target=False)


def test_adjacency_plain_chain_has_no_distance_two_pairs(tmp_path):
    path = tmp_path / "chain.urdf"
    path.write_text(PLAIN_CHAIN_URDF)
    adjacency = compute_adjacent_link_pairs(str(path))
    assert adjacency == {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def test_adjacency_merges_fixed_joints_and_vl_spacers(sharpa_like_file):
    adjacency = compute_adjacent_link_pairs(str(sharpa_like_file))
    pairs = {frozenset((link, nb)) for link, nbs in adjacency.items() for nb in nbs}
    assert pairs == {
        # Arm chain (movable-joint neighbors in the merged-body graph).
        frozenset(("link_0", "link_1")),
        # Fixed joints merge mount+hand into the link_1 body.
        frozenset(("link_1", "finger_MCP_VL")),
        frozenset(("finger_MCP_VL", "finger_PP")),
        frozenset(("finger_PP", "finger_DP")),
        # Distance-2 pair through the "_VL" virtual spacer only.
        frozenset(("link_1", "finger_PP")),
    }
    # finger_tip is merged into finger_DP (fixed joint) and never appears.
    assert "finger_tip" not in adjacency
    assert "mount" not in adjacency and "hand" not in adjacency
    # No distance-2 pair through the plain link finger_PP.
    assert frozenset(("finger_MCP_VL", "finger_DP")) not in pairs


def test_adjacency_merged_group_without_unique_root_fails_closed(tmp_path):
    # A fixed-joint cycle has no member that is "not a fixed child".
    path = tmp_path / "cycle.urdf"
    path.write_text(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>'
        '<joint name="j2" type="fixed"><parent link="b"/><child link="a"/></joint></robot>'
    )
    with pytest.raises(ValueError, match="no unique"):
        compute_adjacent_link_pairs(str(path))


def test_sdf_marker_parsing(tmp_path):
    path = tmp_path / "sdf_tool.urdf"
    path.write_text(SDF_URDF)
    markers = _parse_urdf_sdf_collision_markers(str(path))
    assert len(markers) == 1
    marker = markers[0]
    # Digit-leading mesh stems get the mesh_ prefix (USD identifier rules).
    assert marker.mesh_stem == "mesh_7_hole_patch"
    assert marker.mesh_filename == "meshes/7_hole_patch.obj"
    assert marker.resolution == 64
    assert marker.margin == 0.001
    # camelCase attribute spellings are accepted (scene_utils.py:1191-1198).
    assert marker.narrow_band_thickness == 0.02
    assert marker.subgrid_resolution == 4


def test_sdf_marker_absent_returns_empty(sharpa_like_file):
    assert _parse_urdf_sdf_collision_markers(str(sharpa_like_file)) == []


def test_usd_safe_identifier():
    assert _usd_safe_identifier("6_hole_patch") == "mesh_6_hole_patch"
    assert _usd_safe_identifier("finger-tip") == "finger_tip"
    assert _usd_safe_identifier("plain") == "plain"
