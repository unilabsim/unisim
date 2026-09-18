"""Real cold-source compilation for native worker payloads, not native physics."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from unisim.backend.subprocess_ipc.scene_materialization import prepare_worker_scene
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg


def scene(tmp_path: Path, *, damping: float = 0) -> SceneCfg:
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base"><geom size=".1" mass="1"/>'
        f'<body name="tip"><joint name="hinge" damping="{damping}"/>'
        '<geom size=".1" mass="1"/></body></body></worldbody>'
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator></mujoco>'
    )
    objects = []
    for index, mass in enumerate((1, 3)):
        source = tmp_path / f"object-{index}.xml"
        source.write_text(
            '<mujoco><worldbody><body name="base"><freejoint/>'
            f'<geom size=".1" mass="{mass}"/></body></worldbody></mujoco>'
        )
        objects.append(ModelSourceDescriptor(str(source)))
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
            SceneEntitySpec(
                "object",
                objects[0],
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                mirror_of="object",
                collision_enabled=False,
                initial_state=EntityInitialState(position=(2.0, 0.0, 1.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), tuple(objects))
        ),
    )


def test_worker_sources_have_explicit_inertia_and_no_unsupported_canonical_actuators(tmp_path):
    prepared = prepare_worker_scene(scene(tmp_path), 5, 0.002)
    directory = Path(prepared.owner.model_file).parent
    try:
        assert (prepared.layout.nq, prepared.layout.nv, prepared.layout.nu) == (8, 7, 1)
        entries = prepared.payload["scene_entities"]
        assert entries[1]["assignment"] == entries[2]["assignment"] == [1, 1, 0, 1, 0]
        assert entries[1]["variants"][1]["body_mass"] == [3.0]
        assert entries[0]["variants"][0]["dof_stiffness"] == [20.0]
        assert entries[0]["variants"][0]["dof_damping"] == [2.0]
        for entry in entries:
            for source in entry["sources"]:
                root = ET.parse(source).getroot()
                assert root.find(".//inertial") is not None
                assert root.find("actuator") is None
                assert not root.findall('.//body[@mocap="true"]')
        np.testing.assert_allclose(prepared.roots[:, 1, :3], np.tile([0, 0, 1], (5, 1)))
        np.testing.assert_allclose(prepared.roots[:, 2, :3], np.tile([2, 0, 1], (5, 1)))
    finally:
        prepared.close()
    assert not directory.exists()


def test_mirror_receives_the_same_role_neutral_expanded_source_as_its_source_entity(tmp_path):
    config = scene(tmp_path)
    config.entity_assets = (
        config.entity_assets[0],
        config.entity_assets[2],
        config.entity_assets[1],
    )
    prepared = prepare_worker_scene(config, 5, 0.002)
    try:
        entries = prepared.payload["scene_entities"]
        mirror_entry, object_entry = entries[1], entries[2]
        assert len(object_entry["sources"]) == len(mirror_entry["sources"]) == 2
        for object_source, mirror_source in zip(object_entry["sources"], mirror_entry["sources"]):
            assert Path(object_source).read_bytes() == Path(mirror_source).read_bytes()
    finally:
        prepared.close()


def test_source_passive_damping_is_not_silently_replaced_by_drive_damping(tmp_path):
    with pytest.raises(NotImplementedError, match="passive joint damping"):
        prepare_worker_scene(scene(tmp_path, damping=0.1), 5, 0.002)


@pytest.mark.parametrize("actuated", [True, False])
def test_source_joint_spring_is_not_silently_dropped_from_native_drive_table(tmp_path, actuated):
    config = scene(tmp_path)
    robot = Path(config.entity_assets[0].source.model_file)
    xml = robot.read_text().replace(
        'name="hinge" damping=', 'name="hinge" stiffness="50" springref=".3" damping='
    )
    if not actuated:
        start, end = xml.index("<actuator>"), xml.index("</actuator>") + len("</actuator>")
        xml = xml[:start] + xml[end:]
    robot.write_text(xml)
    with pytest.raises(NotImplementedError, match="passive joint springs"):
        prepare_worker_scene(config, 5, 0.002)
