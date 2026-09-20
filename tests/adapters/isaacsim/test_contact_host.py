"""SDK-free host checks for IsaacSim collision-pair and body-net contact sensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim import IsaacGymBackend, IsaacSimBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
from unisim.scene import SceneCfg


def scene(tmp_path: Path) -> SceneCfg:
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="base_geom" size=".1" mass="1"/>'
        '<body name="tip"><joint name="hinge"/><geom name="tip_geom" size=".1" mass="1"/>'
        "</body></body></worldbody>"
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator>'
        '<sensor><contact name="tip_base" geom1="tip_geom" geom2="base_geom" '
        'data="force" reduce="netforce"/></sensor></mujoco>',
        encoding="utf-8",
    )
    obj = tmp_path / "object.xml"
    obj.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/>'
        '<geom name="object_geom" size=".1" mass="1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(obj)),
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
        )
    )


def test_mapped_pair_sensor_reaches_worker_payload_and_sensor_map(tmp_path: Path):
    config = scene(tmp_path)
    backend = IsaacSimBackend(config, 2, 0.002)
    try:
        payload = backend._worker_init_payload()
        assert payload["contact_force_sensors"] == [{
            "name": "robot/tip_base",
            "source_entity": "robot",
            "source_body": "tip",
            "target_entity": "robot",
            "target_body": "base",
        }]
        backend._body_id_by_name = {
            entity.name + "/" + body_name: body_id
            for entity in backend._entity_scene.layout.entities
            for body_name, body_id in zip(entity.body_names, entity.body_ids)
        }
        sensor_map = backend._resolve_sensor_map()
        spec, _body_id = sensor_map["robot/tip_base"]
        assert spec.kind == "contact_force"
        assert spec.sensor_index == 0
        backend._sensor_map = sensor_map
        backend._slots = {
            "contact_sensor_force": np.array(
                [[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32
            )
        }
        backend._require_state = lambda operation: None
        result = backend.get_sensor_data("robot/tip_base")
        backend._slots["contact_sensor_force"][0, 0] = 9
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])
    finally:
        backend.close()


def test_portable_cross_entity_sensor_fragment_reaches_worker_payload(tmp_path: Path):
    config = scene(tmp_path)
    table = tmp_path / "table.xml"
    table.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="base_geom" size="1 1 .1" mass="5"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    fragment = tmp_path / "sensors.xml"
    fragment.write_text(
        '<mujoco><sensor><contact name="object_table" geom1="object/object_geom" '
        'geom2="table/base_geom" data="force" reduce="netforce"/></sensor></mujoco>',
        encoding="utf-8",
    )
    config.entity_assets = (
        *config.entity_assets,
        SceneEntitySpec(
            "table", ModelSourceDescriptor(str(table)), kind="rigid", root_mode="fixed"
        ),
    )
    config.fragment_files = [str(fragment)]
    backend = IsaacSimBackend(config, 2, 0.002)
    try:
        sensors = backend._worker_init_payload()["contact_force_sensors"]
        assert sensors == [
            {
                "name": "robot/tip_base",
                "source_entity": "robot",
                "source_body": "tip",
                "target_entity": "robot",
                "target_body": "base",
            },
            {
                "name": "object_table",
                "source_entity": "object",
                "source_body": "base",
                "target_entity": "table",
                "target_body": "base",
            },
        ]
    finally:
        backend.close()


def test_isaacgym_does_not_claim_collision_pair_force_reporting(tmp_path: Path):
    backend = IsaacGymBackend(scene(tmp_path), 2, 0.002)
    try:
        backend._body_id_by_name = {
            entity.name + "/" + body_name: body_id
            for entity in backend._entity_scene.layout.entities
            for body_name, body_id in zip(entity.body_names, entity.body_ids)
        }
        backend._sensor_map = backend._resolve_sensor_map()
        backend._require_state = lambda operation: None
        backend._stale_body_ids.clear()
        backend._selected_body_state = (
            lambda body_ids: np.zeros((2, len(body_ids), 13), dtype=np.float32)
        )
        with pytest.raises(NotImplementedError, match="sensor kind 'contact_force'"):
            backend.get_sensor_data("robot/tip_base")
    finally:
        backend.close()


def net_scene(tmp_path: Path) -> SceneCfg:
    """Mapped scene with a wildcard body-net force and a found declaration."""
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="base_geom" size=".1" mass="1"/>'
        '<body name="tip"><joint name="hinge"/><geom name="tip_geom" size=".1" mass="1"/>'
        "</body></body></worldbody>"
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator>'
        '<sensor><contact name="tip_base" geom1="tip_geom" geom2="base_geom" '
        'data="force" reduce="netforce"/>'
        '<contact name="tip_net" geom1="tip_geom" data="force" reduce="netforce"/>'
        '<contact name="tip_touch" geom1="tip_geom" data="found"/>'
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    obj = tmp_path / "object.xml"
    obj.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/>'
        '<geom name="object_geom" size=".1" mass="1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(obj)),
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
        )
    )


def _bind_public_bodies(backend) -> None:
    backend._body_id_by_name = {
        entity.name + "/" + body_name: body_id
        for entity in backend._entity_scene.layout.entities
        for body_name, body_id in zip(entity.body_names, entity.body_ids)
    }


def test_mapped_body_net_sensors_reach_worker_payload_and_sensor_map(tmp_path: Path):
    backend = IsaacSimBackend(net_scene(tmp_path), 2, 0.002)
    try:
        payload = backend._worker_init_payload()
        assert payload["body_net_contact_entities"] == ["robot"]
        # Only the declared geom pair consumes a dedicated reporter row.
        assert payload["contact_force_sensors"] == [{
            "name": "robot/tip_base",
            "source_entity": "robot",
            "source_body": "tip",
            "target_entity": "robot",
            "target_body": "base",
        }]
        _bind_public_bodies(backend)
        sensor_map = backend._resolve_sensor_map()
        net_spec, net_body = sensor_map["robot/tip_net"]
        assert net_spec.kind == "contact_force"
        assert net_spec.target_body_name is None
        assert net_spec.sensor_index is None
        touch_spec, touch_body = sensor_map["robot/tip_touch"]
        assert touch_spec.kind == "contact_found"
        assert touch_body == net_body
        assert sensor_map["robot/tip_base"][0].sensor_index == 0

        backend._sensor_map = sensor_map
        backend._require_state = lambda operation: None
        backend._stale_body_ids.clear()
        backend._selected_body_state = (
            lambda body_ids: np.zeros((2, len(body_ids), 13), dtype=np.float32)
        )
        layout = backend._entity_scene.layout
        nbody = layout.nbody
        tip_id = dict(zip(layout.entities[0].body_names, layout.entities[0].body_ids))["tip"]
        forces = np.zeros((2, nbody, 3), dtype=np.float32)
        forces[:, tip_id] = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
        backend._slots = {"contact_force": forces}
        net = backend.get_sensor_data("robot/tip_net")
        np.testing.assert_array_equal(net, [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        touch = backend.get_sensor_data("robot/tip_touch")
        np.testing.assert_array_equal(touch, [[1.0], [0.0]])
        # The wildcard view is detached from the shared slot.
        forces[:, tip_id] = 9.0
        np.testing.assert_array_equal(net, [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    finally:
        backend.close()


def test_mapped_body_net_fragment_sensor_reaches_worker_payload(tmp_path: Path):
    config = scene(tmp_path)
    fragment = tmp_path / "sensors.xml"
    fragment.write_text(
        '<mujoco><sensor>'
        '<contact name="object_net" geom1="object/object_geom" '
        'data="force" reduce="netforce"/>'
        '<contact name="object_touch" geom1="object/object_geom" '
        'data="found" num="1"/>'
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    config.fragment_files = [str(fragment)]
    backend = IsaacSimBackend(config, 2, 0.002)
    try:
        payload = backend._worker_init_payload()
        assert payload["body_net_contact_entities"] == ["object"]
        # Wildcard declarations never consume a dedicated pair reporter row.
        assert [record["name"] for record in payload["contact_force_sensors"]] == [
            "robot/tip_base"
        ]
        _bind_public_bodies(backend)
        sensor_map = backend._resolve_sensor_map()
        assert sensor_map["object_net"][0].target_body_name is None
        assert sensor_map["object_touch"][0].kind == "contact_found"
    finally:
        backend.close()


def test_legacy_contact_declarations_fail_closed(tmp_path: Path):
    model = tmp_path / "legacy.xml"
    model.write_text(
        '<mujoco><worldbody><body name="base"><geom name="base_geom" size=".1" mass="1"/>'
        '<body name="tip"><joint name="hinge"/>'
        '<geom name="tip_geom" size=".1" mass="1"/></body></body></worldbody>'
        '<sensor><contact name="tip_net" geom1="tip_geom" data="force" reduce="netforce"/>'
        '<contact name="tip_touch" geom1="tip_geom" data="found"/>'
        '<contact name="tip_base" geom1="tip_geom" geom2="base_geom" '
        'data="force" reduce="netforce"/></sensor></mujoco>',
        encoding="utf-8",
    )
    backend = IsaacSimBackend(SceneCfg(model_file=str(model)), 2, 0.002)
    try:
        payload = backend._worker_init_payload()
        assert payload["contact_force_sensors"] == []
        assert payload["body_net_contact_entities"] == []
        backend._body_id_by_name = {"base": 0, "tip": 1}
        sensor_map = backend._resolve_sensor_map()
        assert "tip_net" not in sensor_map
        assert "tip_touch" not in sensor_map
        assert "tip_base" not in sensor_map
        metadata = backend._get_scene_metadata()
        for name in ("tip_net", "tip_touch", "tip_base"):
            assert "no PhysX per-body or pair contact reporter" in (
                metadata.unsupported_sensors[name].reason
            )
    finally:
        backend.close()
