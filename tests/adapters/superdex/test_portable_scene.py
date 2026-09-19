"""Native portable multi-actor coverage for the bounded SuperDex profile."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")
pytest.importorskip("mujoco")


def _sources(tmp_path: Path) -> tuple[ModelSourceDescriptor, ...]:
    robot = tmp_path / "robot.xml"
    robot.write_text(
        """<mujoco><worldbody><body name="base">
          <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <geom name="base_geom" type="box" size=".05 .05 .05"/>
          <body name="arm" pos=".08 0 0">
            <joint name="hinge" axis="0 0 1" damping=".05"/>
            <inertial mass=".2" pos=".04 0 0" diaginertia=".002 .002 .001"/>
            <geom name="arm_geom" type="box" size=".04 .015 .015" pos=".04 0 0"/>
            <body name="tool" pos=".08 0 0">
              <joint name="tool_hinge" axis="0 0 1" damping=".02"/>
              <inertial mass=".1" pos=".02 0 0" diaginertia=".0005 .0005 .0003"/>
              <geom name="tool_geom" type="box" size=".02 .01 .01" pos=".02 0 0"/>
            </body>
          </body>
        </body></worldbody>
        <actuator>
          <motor name="hinge_motor" joint="hinge" ctrlrange="-2 2"/>
          <motor name="tool_motor" joint="tool_hinge" ctrlrange="-2 2"/>
        </actuator>
        </mujoco>""",
        encoding="utf-8",
    )
    obj = tmp_path / "object.xml"
    obj.write_text(
        """<mujoco><worldbody><body name="body">
          <freejoint name="root"/>
          <inertial mass=".3" pos="0 0 0" diaginertia=".001 .001 .001"/>
          <geom name="shape" type="sphere" size=".035"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    table = tmp_path / "table.xml"
    table.write_text(
        """<mujoco><worldbody><body name="top">
          <geom name="surface" type="box" size=".15 .15 .015" mass="5"/>
        </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    return (
        ModelSourceDescriptor(str(robot)),
        ModelSourceDescriptor(str(obj)),
        ModelSourceDescriptor(str(table)),
    )


def _scene(tmp_path: Path) -> SceneCfg:
    robot, obj, table = _sources(tmp_path)
    sensors = tmp_path / "sensors.xml"
    sensors.write_text(
        """<mujoco><sensor>
          <contact name="object_table" geom1="object/shape" geom2="table/surface"
                   data="found" num="1"/>
        </sensor></mujoco>""",
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", robot, root_mode="fixed"),
            SceneEntitySpec(
                "object",
                obj,
                initial_state=EntityInitialState((0.02, 0.01, 0.12)),
            ),
            SceneEntitySpec("table", table, kind="rigid", root_mode="fixed"),
        ),
        fragment_files=[str(sensors)],
    )


@pytest.fixture
def scene(tmp_path: Path) -> SceneCfg:
    return _scene(tmp_path)


def test_portable_backend_steps_and_maps_entity_state(tmp_path: Path, scene: SceneCfg):
    backend = create_backend("superdex", scene, 2, 0.002)
    serial = create_backend(
        "superdex", _scene(tmp_path), 2, 0.002, superdex_execution_mode="serial"
    )
    try:
        assert backend.get_entity_names() == ("robot", "object", "table")
        assert backend.model.actor_plans[0].entity_name == "robot"
        assert backend.model.actor_plans[1].entity_name == "object"
        assert backend.model.actor_plans[2].native_qvel_indices.size == 0
        layout = backend.get_root_state_layout("object/body")
        assert tuple(layout.qpos_indices) == (2, 3, 4, 5, 6, 7, 8)
        assert tuple(layout.qvel_indices) == (2, 3, 4, 5, 6, 7)

        q = np.tile(backend.get_default_qpos(), (2, 1))
        v = np.zeros((2, backend.model.nv), dtype=q.dtype)
        q[:, 2:5] = [[0.02, 0.01, 0.12], [-0.01, 0.02, 0.10]]
        q[:, 0] = [0.2, -0.1]
        q[:, 1] = [0.1, -0.05]
        v[:, 5] = [0.3, -0.2]
        for item in (backend, serial):
            item.set_state(np.arange(2), q, v)
            item.step(np.array([[0.4, -0.2], [-0.4, 0.2]]), nsteps=2)
        np.testing.assert_allclose(
            backend.get_state()["qpos"], serial.get_state()["qpos"], atol=3e-6
        )
        np.testing.assert_allclose(
            backend.get_state()["qvel"], serial.get_state()["qvel"], atol=3e-6
        )
        object_state = backend.get_entity_state("object")
        np.testing.assert_allclose(object_state["root_pose"][0, :3], [0.02, 0.01, 0.12], atol=1e-3)
        object_body = backend.get_body_ids(["object/body"])[0]
        np.testing.assert_allclose(
            backend.get_body_pos_w(np.array([object_body]))[0, 0],
            [0.02, 0.01, 0.12],
            atol=1e-3,
        )
        assert backend.get_entity_state("table")["root_pose"][0, 2] == pytest.approx(0.0)
    finally:
        backend.close()
        serial.close()


def test_selected_entity_reset_preserves_other_entities_and_controls(scene: SceneCfg):
    backend = create_backend("superdex", scene, 2, 0.002)
    try:
        robot_body = backend.get_body_ids(["robot/arm"])[0]
        backend.step(np.array([[0.5, -0.5], [-0.5, 0.5]]))
        before = backend.get_state()
        entity_before = backend.get_entity_state("object")
        object_pose = entity_before["root_pose"].copy()
        object_velocity = entity_before["root_velocity"].copy()
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.array([[0.3, -0.2]]),
                        joint_velocities=np.array([[0.1, -0.05]]),
                    ),
                ),
            )
        )
        state = backend.get_state()
        np.testing.assert_allclose(state["qpos"][0], before["qpos"][0], atol=1e-6)
        np.testing.assert_allclose(state["qvel"][0], before["qvel"][0], atol=1e-6)
        np.testing.assert_allclose(state["qpos"][1, 0], 0.3, atol=1e-6)
        np.testing.assert_allclose(state["qvel"][1, 0], 0.1, atol=1e-6)
        np.testing.assert_allclose(state["qpos"][1, 1], -0.2, atol=1e-6)
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_pose"], object_pose, atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_velocity"],
            object_velocity,
            atol=1e-6,
        )
        np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"][1], np.zeros(2))
        assert np.all(backend.get_state("ctrl")["ctrl"][0] != 0)
        assert backend.get_body_pos_w(np.array([robot_body])).shape == (2, 1, 3)
    finally:
        backend.close()


def test_portable_body_force_on_second_actor_matches_serial(tmp_path: Path):
    batch = create_backend("superdex", _scene(tmp_path), 2, 0.002)
    serial = create_backend(
        "superdex", _scene(tmp_path), 2, 0.002, superdex_execution_mode="serial"
    )
    try:
        body = batch.get_body_ids(["object/body"])[0]
        force = np.zeros((2, 1, 3), dtype=batch.get_default_qpos().dtype)
        force[:, 0, 0] = 1.5
        for backend in (batch, serial):
            backend.apply_body_force(np.array([body]), force)
            backend.step(np.zeros((2, 2)))
        np.testing.assert_allclose(
            batch.get_state()["qvel"], serial.get_state()["qvel"], atol=3e-6
        )
    finally:
        batch.close()
        serial.close()


def test_portable_contact_sensor_refreshes_after_positive_step(scene: SceneCfg):
    backend = create_backend("superdex", scene, 1, 0.002)
    try:
        q = np.tile(backend.get_default_qpos(), (1, 1))
        v = np.zeros((1, backend.model.nv), dtype=q.dtype)
        q[0, 2:5] = [0.0, 0.0, 0.03]
        backend.set_state(np.arange(1), q, v)
        assert backend.get_sensor_data("object_table")[0, 0] == 0
        backend.step(np.zeros((1, 2)), nsteps=2)
        assert backend.get_sensor_data("object_table")[0, 0] == 1
        q[0, 4] = 0.12
        backend.set_state(np.arange(1), q, v)
        assert backend.get_sensor_data("object_table")[0, 0] == 0
    finally:
        backend.close()


def test_portable_control_restoration_fails_closed(scene: SceneCfg):
    backend = create_backend("superdex", scene, 1, 0.002)
    try:
        with pytest.raises(NotImplementedError, match="restore_default_controls"):
            backend.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch(
                            "robot",
                            joint_positions=np.zeros((1, 2)),
                            joint_names=("hinge", "tool_hinge"),
                        ),
                    ),
                    restore_default_controls=True,
                )
            )
    finally:
        backend.close()


def test_portable_profile_rejects_unsupported_authoring(tmp_path: Path):
    _, obj, _ = _sources(tmp_path)
    rejected_scenes = (
        SceneCfg(
            entity_assets=(SceneEntitySpec("object", obj, root_mode="kinematic"),)
        ),
        SceneCfg(
            entity_assets=(
                SceneEntitySpec("object", obj),
                SceneEntitySpec(
                    "mirror",
                    kind="rigid",
                    mirror_of="object",
                    root_mode="kinematic",
                    collision_enabled=False,
                ),
            )
        ),
    )
    variant_scene = SceneCfg(entity_assets=(SceneEntitySpec("object", obj),))
    variant_scene.entity_variant = EntityVariantBinding(
        "object", FixedVariantPlan(np.array([0, 1]), (obj, obj))
    )
    rejected_scenes += (variant_scene,)
    # Fixed variants are rejected by the capability gate before SuperDex's
    # defensive materialization check; kinematic roots and mirrors are distinct
    # backend fail-closed paths.
    matches = ("entity.multiple", "portable mirrors", "entity.multiple")
    for rejected, message in zip(rejected_scenes, matches, strict=True):
        with pytest.raises(NotImplementedError, match=message):
            create_backend(
                "superdex",
                rejected,
                len(rejected.entity_assets),
                0.002,
                superdex_execution_mode="serial",
            )
