"""Real Motrix acceptance for shared-scene entity interaction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from tests.adapters.motrix.test_portable_entities import _scene, _write
from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityStatePatch, SceneResetRequest


def _interaction_robot(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "interaction-robot",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="0 0 1">
          <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
          <geom name="base_geom" type="sphere" size=".08"/>
          <body name="link" pos="0 0 .3">
            <joint name="drive" axis="0 1 0" ref="0.1"/>
            <inertial pos=".15 0 0" mass=".2" diaginertia=".03 .03 .03"/>
            <geom name="link_geom" type="sphere" size=".05" pos=".3 0 0"/>
          </body>
        </body></worldbody>
        <actuator><motor name="drive" joint="drive"/></actuator></mujoco>
        """,
    )


def _robot_object_contact_fragment(tmp_path: Path) -> Path:
    target = tmp_path / "robot-object-contact.xml"
    target.write_text(
        "<mujoco><sensor>"
        "<contact name='robot_object_force' geom1='robot/link_geom' "
        "geom2='object/object_geom' data='force' reduce='netforce'/>"
        "<contact name='robot_object_found' geom1='robot/link_geom' "
        "geom2='object/object_geom' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return target


def test_shared_scene_robot_object_interaction_is_row_local(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    robot = next(entity for entity in scene.entity_assets if entity.name == "robot")
    robot = replace(robot, source=_interaction_robot(tmp_path / "robot"))
    scene.entity_assets = tuple(
        robot if entity.name == "robot" else entity for entity in scene.entity_assets
    )
    scene.fragment_files = (str(_robot_object_contact_fragment(tmp_path)),)
    backend = MotrixBackend(scene, 5, 0.002, base_name="robot/base")
    try:
        noninteracting_rows = np.asarray((0, 1, 3, 4), dtype=np.intp)
        object_pose = np.tile(
            np.asarray((2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (5, 1),
        )
        object_pose[2, :3] = (0.25, 0.0, 1.18)
        object_velocity = np.zeros((5, 6), dtype=np.float32)
        robot_joint_positions = np.full((5, 1), 0.4, dtype=np.float32)
        robot_joint_velocities = np.zeros((5, 1), dtype=np.float32)
        robot_joint_velocities[2, 0] = 8.0

        backend.reset_entities(
            SceneResetRequest(
                tuple(range(5)),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=robot_joint_positions,
                        joint_velocities=robot_joint_velocities,
                    ),
                    EntityStatePatch(
                        "object",
                        root_pose=object_pose,
                        root_velocity=object_velocity,
                    ),
                ),
            )
        )
        found_history = []
        force_history = []
        for _ in range(40):
            backend.step(np.zeros((5, 1), dtype=np.float32))
            found_history.append(np.asarray(backend.get_sensor_data("robot_object_found")).copy())
            force_history.append(np.asarray(backend.get_sensor_data("robot_object_force")).copy())

        found = np.stack(found_history)
        force = np.stack(force_history)
        assert found.shape == (40, 5, 1)
        assert force.shape == (40, 5, 3)
        assert np.any(found[:, 2] > 0.0)
        np.testing.assert_array_equal(found[:, noninteracting_rows], 0.0)
        assert np.all(np.isfinite(force))
        assert np.any(np.linalg.norm(force[:, 2], axis=1) > 0.0)
        assert np.all(np.linalg.norm(force[:, noninteracting_rows], axis=2) == 0.0)

        object_state = backend.get_entity_state("object")
        robot_state = backend.get_entity_state("robot")
        assert abs(object_state["root_pose"][2, 0] - 0.25) > 0.02
        assert abs(object_state["root_velocity"][2, 0]) > 0.02
        assert robot_state["joint_positions"][2, 0] > robot_state["joint_positions"][0, 0]

        for entity_name in backend.get_entity_names():
            entity_state = backend.get_entity_state(entity_name)
            for values in entity_state.values():
                array = np.asarray(values)
                reference = np.broadcast_to(array[0], array[noninteracting_rows].shape)
                np.testing.assert_allclose(
                    array[noninteracting_rows],
                    reference,
                    rtol=2e-6,
                    atol=2e-6,
                )
    finally:
        backend.close()
