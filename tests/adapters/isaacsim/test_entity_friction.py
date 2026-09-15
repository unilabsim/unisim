"""Tests for the entity contact-friction channel (SimToolReal step 1.3b-2).

Pure-Python coverage only: contract validation (scene.py), host scan
cross-checks and INIT payload serialization (subprocess_ipc), and the
worker's pure table builder / wire re-validation.  The PhysX view write and
USD bake readback are exercised by the Kit readback probe (step 1.3b-3,
probes/probe_b2_full_scene_readback.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from unisim.backend.isaacsim.worker import build_friction_shape_table, parse_entity_friction
from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend
from unisim.backend.subprocess_ipc.sensors import scan_scene_entities
from unisim.scene import BodyFrictionOverride, SceneCfg, SceneEntitySpec


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


ROBOT_URDF = """<?xml version="1.0"?>
<robot name="mini_hand">
  <link name="base_link"/>
  <link name="finger_DP"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="finger_DP"/>
    <limit lower="-1.57" upper="1.57" effort="300" velocity="10"/>
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


@pytest.fixture()
def robot_urdf_file(tmp_path):
    path = tmp_path / "mini_hand.urdf"
    path.write_text(ROBOT_URDF)
    return path


@pytest.fixture()
def object_urdf_file(tmp_path):
    path = tmp_path / "cube.urdf"
    path.write_text(OBJECT_URDF)
    return path


def _robot_spec(model_file, **overrides) -> SceneEntitySpec:
    kwargs = {
        "name": "robot",
        "model_file": str(model_file),
        "asset_format": "urdf",
        "materialization": "articulation",
        "root_mode": "fixed",
    }
    kwargs.update(overrides)
    return SceneEntitySpec(**kwargs)


def test_body_friction_override_validation():
    override = BodyFrictionOverride(body_name="finger_DP", friction=[1.5, 1.5, 0])
    assert override.friction == (1.5, 1.5, 0.0)
    with pytest.raises(ValueError, match="body_name"):
        BodyFrictionOverride(body_name="", friction=(0.5, 0.5, 0.0))
    for bad in ((0.5, 0.5), (-0.1, 0.5, 0.0), (0.5, float("nan"), 0.0), (0.5, 0.5, "x")):
        with pytest.raises((TypeError, ValueError)):
            BodyFrictionOverride(body_name="finger_DP", friction=bad)
    with pytest.raises(TypeError, match="triple"):
        BodyFrictionOverride(body_name="finger_DP", friction="0.5")


def test_scene_entity_contact_friction_defaults_and_normalization(robot_urdf_file):
    spec = _robot_spec(robot_urdf_file)
    assert spec.contact_friction is None
    assert spec.contact_friction_by_body == ()
    spec = _robot_spec(robot_urdf_file, contact_friction=[0.5, 0.5, 0])
    assert spec.contact_friction == (0.5, 0.5, 0.0)
    for bad in ((0.5, 0.5), (-1.0, 0.5, 0.0), (0.5, float("inf"), 0.0)):
        with pytest.raises(ValueError, match="contact_friction"):
            _robot_spec(robot_urdf_file, contact_friction=bad)


def test_scene_entity_friction_overrides_fail_closed(robot_urdf_file, object_urdf_file):
    fingertip = BodyFrictionOverride(body_name="finger_DP", friction=(1.5, 1.5, 0.0))
    # Rigid entities cannot carry per-body overrides.
    with pytest.raises(ValueError, match="articulation"):
        SceneEntitySpec(
            name="object",
            model_file=str(object_urdf_file),
            asset_format="urdf",
            materialization="rigid",
            root_mode="floating",
            contact_friction=(0.5, 0.5, 0.0),
            contact_friction_by_body=(fingertip,),
        )
    # Overrides require the default triple.
    with pytest.raises(ValueError, match="without a contact_friction default"):
        _robot_spec(robot_urdf_file, contact_friction_by_body=(fingertip,))
    # Duplicate body overrides fail closed.
    with pytest.raises(ValueError, match="duplicate contact friction overrides"):
        _robot_spec(
            robot_urdf_file,
            contact_friction=(0.5, 0.5, 0.0),
            contact_friction_by_body=(fingertip, fingertip),
        )
    # Valid combination constructs cleanly.
    spec = _robot_spec(
        robot_urdf_file,
        contact_friction=(0.5, 0.5, 0.0),
        contact_friction_by_body=(fingertip,),
    )
    assert spec.contact_friction_by_body == (fingertip,)


def test_scan_scene_entities_validates_friction_override_bodies(robot_urdf_file):
    spec = _robot_spec(
        robot_urdf_file,
        contact_friction=(0.5, 0.5, 0.0),
        contact_friction_by_body=(
            BodyFrictionOverride(body_name="missing_link", friction=(1.5, 1.5, 0.0)),
        ),
    )
    with pytest.raises(ValueError, match="missing_link"):
        scan_scene_entities((spec,), backend_label="isaacsim")
    # A scanned body name passes the cross-check.
    valid = _robot_spec(
        robot_urdf_file,
        contact_friction=(0.5, 0.5, 0.0),
        contact_friction_by_body=(
            BodyFrictionOverride(body_name="finger_DP", friction=(1.5, 1.5, 0.0)),
        ),
    )
    metadata = scan_scene_entities((valid,), backend_label="isaacsim")
    assert metadata["robot"].body_names == ("base_link", "finger_DP")


def test_host_entity_payloads_serialize_friction(robot_urdf_file, object_urdf_file):
    scene = SceneCfg(
        model_file=str(robot_urdf_file),
        entity_assets=(
            _robot_spec(
                robot_urdf_file,
                contact_friction=(0.5, 0.5, 0.0),
                contact_friction_by_body=(
                    BodyFrictionOverride(body_name="finger_DP", friction=(1.5, 1.5, 0.0)),
                ),
            ),
            SceneEntitySpec(
                name="object",
                model_file=str(object_urdf_file),
                asset_format="urdf",
                materialization="rigid",
                root_mode="floating",
                contact_friction=(0.5, 0.5, 0.0),
            ),
            SceneEntitySpec(
                name="goalviz",
                model_file=str(object_urdf_file),
                asset_format="urdf",
                materialization="rigid",
                root_mode="kinematic",
            ),
        ),
    )
    # Host-only construction: no worker is spawned before materialize().
    backend = _EntityAssetHarnessBackend(scene, num_envs=2, sim_dt=0.01)
    payloads = {entry["name"]: entry for entry in backend._entity_payloads()}
    assert payloads["robot"]["friction"] == [0.5, 0.5, 0.0]
    assert payloads["robot"]["friction_by_body"] == {"finger_DP": [1.5, 1.5, 0.0]}
    assert payloads["object"]["friction"] == [0.5, 0.5, 0.0]
    # Default-only entities carry no override map.
    assert "friction_by_body" not in payloads["object"]
    # Undeclared entities carry no friction keys at all: legacy payloads stay
    # byte-identical.
    assert "friction" not in payloads["goalviz"]
    assert "friction_by_body" not in payloads["goalviz"]


def test_parse_entity_friction_wire_validation():
    assert parse_entity_friction({"name": "goalviz"}) is None
    default, overrides = parse_entity_friction(
        {
            "name": "robot",
            "friction": [0.5, 0.5, 0.0],
            "friction_by_body": {"finger_DP": [1.5, 1.5, 0.0]},
        }
    )
    assert default == (0.5, 0.5, 0.0)
    assert overrides == {"finger_DP": (1.5, 1.5, 0.0)}
    # Overrides without a default fail closed.
    with pytest.raises(ValueError, match="without a friction default"):
        parse_entity_friction(
            {"name": "robot", "friction_by_body": {"finger_DP": [1.5, 1.5, 0.0]}}
        )
    with pytest.raises(TypeError, match="must be a dict"):
        parse_entity_friction(
            {"name": "robot", "friction": [0.5, 0.5, 0.0], "friction_by_body": [1.5]}
        )
    with pytest.raises(ValueError, match="exactly 3 components"):
        parse_entity_friction({"name": "robot", "friction": [0.5, 0.5]})
    with pytest.raises(ValueError, match="finite non-negative"):
        parse_entity_friction({"name": "robot", "friction": [0.5, -0.5, 0.0]})


def test_build_friction_shape_table_tiles_and_overrides():
    table = build_friction_shape_table(
        ["base", "finger_DP", "tip"],
        [2, 3, 1],
        (0.5, 0.5, 0.0),
        {"finger_DP": (1.5, 1.5, 0.0)},
    )
    assert table.shape == (6, 3)
    expected = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(table, expected)


def test_build_friction_shape_table_fails_closed():
    with pytest.raises(ValueError, match="not present in the PhysX view"):
        build_friction_shape_table(["base"], [1], (0.5, 0.5, 0.0), {"nope": (1.5, 1.5, 0.0)})
    with pytest.raises(ValueError, match="length mismatch"):
        build_friction_shape_table(["base", "tip"], [1], (0.5, 0.5, 0.0), {})
    with pytest.raises(ValueError, match="non-negative integers"):
        build_friction_shape_table(["base"], [-1], (0.5, 0.5, 0.0), {})
    with pytest.raises(ValueError, match="finite non-negative"):
        build_friction_shape_table(["base"], [1], (0.5, 0.5, -1.0), {})
