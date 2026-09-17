"""SDK-free host checks for IsaacSim collision-pair contact sensors."""

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
