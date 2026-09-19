"""Native portable multi-actor coverage for the bounded SuperDex profile."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
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


def test_portable_selected_control_restoration_is_scoped(tmp_path: Path, scene: SceneCfg):
    robot_source = next(entity for entity in scene.entity_assets if entity.name == "robot").source
    assert robot_source is not None
    robot_path = Path(robot_source.model_file)
    robot_path.write_text(
        robot_path.read_text(encoding="utf-8").replace(
            "</mujoco>",
            "<keyframe><key name=\"home\" qpos=\".15 -.2\" "
            "qvel=\".3 -.4\" ctrl=\".25 -.4\"/></keyframe></mujoco>",
            1,
        ),
        encoding="utf-8",
    )
    scene.default_keyframe_name = "home"
    backend = create_backend("superdex", scene, 2, 0.002)
    try:
        initial = np.array([[0.7, -0.8], [0.6, -0.5]], dtype=backend.get_default_qpos().dtype)
        backend.step(initial)

        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.array([[0.15]]),
                        joint_names=("hinge",),
                    ),
                ),
            )
        )
        controls = backend.get_state("ctrl")["ctrl"]
        np.testing.assert_allclose(controls[0], [0.0, -0.8], atol=0)
        np.testing.assert_allclose(controls[1], initial[1], atol=0)

        backend.step(initial)
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.array([[0.15]]),
                        joint_names=("hinge",),
                    ),
                ),
                restore_default_controls=True,
            )
        )
        controls = backend.get_state("ctrl")["ctrl"]
        np.testing.assert_allclose(controls[0], [0.25, -0.8], atol=1e-7)
        np.testing.assert_allclose(controls[1], initial[1], atol=0)

        object_before = backend.get_entity_state("object").copy()
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=np.array([[0.02, 0.01, 0.12, 1, 0, 0, 0]]),
                    ),
                ),
                restore_default_controls=True,
            )
        )
        np.testing.assert_allclose(backend.get_state("ctrl")["ctrl"], controls, atol=0)
        for name, values in backend.get_entity_state("object").items():
            if name == "root_pose":
                np.testing.assert_allclose(values[1, :3], [0.02, 0.01, 0.12], atol=1e-7)
            else:
                np.testing.assert_allclose(values[0], object_before[name][0], atol=1e-7)
    finally:
        backend.close()


def test_fixed_variants_preserve_layout_and_native_mass_identity(tmp_path: Path):
    scene = _scene(tmp_path)
    object_entity = next(entity for entity in scene.entity_assets if entity.name == "object")
    object_source = object_entity.source
    assert object_source is not None
    heavy_path = tmp_path / "heavy-object.xml"
    heavy_path.write_text(
        Path(object_source.model_file)
        .read_text(encoding="utf-8")
        .replace('mass=".3"', 'mass=".9"')
        .replace('diaginertia=".001 .001 .001"', 'diaginertia=".003 .003 .003"'),
        encoding="utf-8",
    )
    heavy_source = ModelSourceDescriptor(
        str(heavy_path),
    )
    scene.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(np.array([1, 1, 0, 1, 0], dtype=np.int32), (object_source, heavy_source)),
    )
    backend = create_backend("superdex", scene, 5, 0.002)
    serial = create_backend(
        "superdex", scene, 5, 0.002, superdex_execution_mode="serial"
    )
    try:
        assert tuple(backend._variant_assignment) == (1, 1, 0, 1, 0)
        assert backend.get_dr_capabilities().supports_fixed_variants
        assert backend.get_dr_capabilities().supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT}
        )
        body = backend.get_body_ids(["object/body"])[0]
        np.testing.assert_allclose(
            backend.get_body_mass()[:, body], [0.9, 0.9, 0.3, 0.9, 0.3], atol=0
        )
        force = np.zeros((5, 1, 3), dtype=backend.get_default_qpos().dtype)
        force[:, 0, 0] = 2.0
        torque = np.zeros_like(force)
        torque[:, 0, 2] = 0.05
        controls = np.tile(np.array([[0.4, -0.2]], dtype=backend.get_default_qpos().dtype), (5, 1))
        for item in (backend, serial):
            item.apply_body_force(np.array([body]), force)
            item.apply_body_force(np.array([body]), np.zeros_like(force), torque)
            item.step(controls, nsteps=2)
        light_rows = np.array((2, 4))
        heavy_rows = np.array((0, 1, 3))
        for item in (backend, serial):
            velocity = item.get_entity_state("object")["root_velocity"][:, 0]
            assert velocity[light_rows].min() > velocity[heavy_rows].max() * 2.5
            assert velocity[light_rows].min() - velocity[heavy_rows].max() > 0.015
            angular_velocity = item.get_entity_state("object")["root_velocity"][:, 5]
            assert angular_velocity[light_rows].min() > angular_velocity[heavy_rows].max() * 2.5
        np.testing.assert_allclose(
            backend.get_state()["qvel"], serial.get_state()["qvel"], atol=3e-6
        )
        object_before = backend.get_entity_state("object").copy()
        controls_before = backend.get_state("ctrl")["ctrl"].copy()
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=np.array([[0.02, 0.01, 0.12, 1, 0, 0, 0]]),
                    ),
                ),
            )
        )
        object_after = backend.get_entity_state("object")
        np.testing.assert_allclose(
            object_after["root_pose"][0, :3], [0.02, 0.01, 0.12], atol=1e-7
        )
        for field, values in object_after.items():
            np.testing.assert_array_equal(values[1:], object_before[field][1:])
        np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"], controls_before)
    finally:
        backend.close()
        serial.close()


def _mirror_scene(tmp_path: Path, mirror_position: tuple[float, float, float]) -> SceneCfg:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = _scene(tmp_path)
    scene.entity_assets = scene.entity_assets + (
        SceneEntitySpec(
            "mirror",
            kind="rigid",
            root_mode="kinematic",
            collision_enabled=False,
            mirror_of="object",
            initial_state=EntityInitialState(mirror_position),
        ),
    )
    return scene


def test_portable_mirrors_are_collision_free_and_pose_writes_are_row_local(tmp_path: Path):
    far = create_backend(
        "superdex", _mirror_scene(tmp_path / "far", (20.0, 0.0, 10.0)), 3, 0.002
    )
    overlap = create_backend(
        "superdex", _mirror_scene(tmp_path / "overlap", (0.02, 0.01, 0.12)), 3, 0.002
    )
    serial = create_backend(
        "superdex",
        _mirror_scene(tmp_path / "serial", (0.02, 0.01, 0.12)),
        3,
        0.002,
        superdex_execution_mode="serial",
    )
    try:
        mirror_layout = overlap.get_scene_layout().get_entity("mirror")
        assert mirror_layout.root_mode == "kinematic"
        assert mirror_layout.joints == ()
        assert mirror_layout.actuator_names == ()
        assert any(plan.kinematic_mirror for plan in overlap.model.actor_plans)

        mirror_body = overlap.get_body_ids(["mirror/body"])[0]
        with pytest.raises(
            NotImplementedError, match="kinematic entities do not own physical"
        ):
            overlap.apply_body_force(np.array([mirror_body]), np.ones((3, 1, 3)))

        default_pose = overlap.get_entity_state("mirror")["root_pose"].copy()
        selected_pose = np.asarray(
            (
                (3.0, 0.1, 2.5, 0.8, 0.6, 0.0, 0.0),
                (0.5, 0.0, 0.5, 0.8, -0.6, 0.0, 0.0),
            ),
            dtype=overlap.get_default_qpos().dtype,
        )
        selected_pose[:, 3:] /= np.linalg.norm(selected_pose[:, 3:], axis=1, keepdims=True)
        selected_rows = np.asarray((1, 2), dtype=np.intp)
        unrelated_before = {
            name: overlap.get_entity_state(name).copy()
            for name in overlap.get_entity_names()
            if name != "mirror"
        }
        overlap.reset_entities(
            SceneResetRequest(
                (1, 2), (EntityStatePatch("mirror", root_pose=selected_pose),)
            )
        )
        mirror_after = overlap.get_entity_state("mirror")
        np.testing.assert_allclose(
            mirror_after["root_pose"][selected_rows], selected_pose, rtol=0, atol=2e-7
        )
        np.testing.assert_array_equal(mirror_after["root_pose"][0], default_pose[0])
        np.testing.assert_array_equal(mirror_after["root_velocity"], 0.0)
        for name, state in unrelated_before.items():
            for field, values in state.items():
                np.testing.assert_array_equal(overlap.get_entity_state(name)[field], values)

        controls = np.zeros((3, overlap.num_actuators), dtype=overlap.get_default_qpos().dtype)
        for _ in range(20):
            for backend in (far, overlap, serial):
                backend.step(controls)
        mirror_pose = overlap.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(
            mirror_pose[selected_rows], selected_pose, rtol=0, atol=1e-7
        )
        far_object = far.get_entity_state("object")
        overlap_object = overlap.get_entity_state("object")
        for field in far_object:
            np.testing.assert_allclose(
                overlap_object[field], far_object[field], rtol=2e-6, atol=2e-6
            )
        np.testing.assert_allclose(
            overlap.get_state()["qpos"], serial.get_state()["qpos"], atol=3e-6
        )

        overlap.reset((1,))
        mirror_pose = overlap.get_entity_state("mirror")["root_pose"]
        np.testing.assert_allclose(mirror_pose[1], default_pose[1], rtol=0, atol=0)
        np.testing.assert_allclose(mirror_pose[2], selected_pose[1], rtol=0, atol=1e-7)
    finally:
        far.close()
        overlap.close()
        serial.close()


def test_fixed_variant_mirror_identity_follows_assignment_and_pose_routing(tmp_path: Path):
    base = _mirror_scene(tmp_path / "base", (4.0, 0.0, 3.0))
    light = next(entity for entity in base.entity_assets if entity.name == "object").source
    assert light is not None
    heavy_path = tmp_path / "heavy-object.xml"
    heavy_path.write_text(
        Path(light.model_file)
        .read_text(encoding="utf-8")
        .replace('mass=".3"', 'mass=".9"')
        .replace('size=".035"', 'size=".045"'),
        encoding="utf-8",
    )
    heavy = ModelSourceDescriptor(str(heavy_path))
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", light),
            base.entity_assets[-1],
        )
    )
    scene.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(np.array([1, 0], dtype=np.int32), (light, heavy)),
    )
    backend = create_backend("superdex", scene, 2, 0.002)
    try:
        object_body = backend.get_body_ids(["object/body"])[0]
        mirror_body = backend.get_body_ids(["mirror/body"])[0]
        np.testing.assert_allclose(backend.get_body_mass()[:, object_body], [0.9, 0.3], atol=0)
        np.testing.assert_allclose(backend.get_body_mass()[:, mirror_body], [0.9, 0.3], atol=0)
        poses = np.asarray(
            (
                (5.0, 0.2, 3.2, 0.8, 0.6, 0.0, 0.0),
                (6.0, -0.2, 3.4, 0.8, 0.0, 0.6, 0.0),
            ),
            dtype=backend.get_default_qpos().dtype,
        )
        poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
        backend.reset_entities(
            SceneResetRequest((0, 1), (EntityStatePatch("mirror", root_pose=poses),))
        )
        backend.step(np.zeros((2, backend.num_actuators), dtype=poses.dtype), nsteps=3)
        np.testing.assert_allclose(
            backend.get_entity_state("mirror")["root_pose"], poses, rtol=0, atol=1e-7
        )
    finally:
        backend.close()


def test_mirror_contact_sensors_fail_closed(tmp_path: Path):
    scene = _mirror_scene(tmp_path / "scene", (0.02, 0.01, 0.12))
    contact_fragment = tmp_path / "scene" / "mirror-contact.xml"
    contact_fragment.write_text(
        "<mujoco><sensor>"
        "<contact name='object_mirror_found' geom1='object/shape' "
        "geom2='mirror/shape' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    scene.fragment_files = (str(contact_fragment),)
    with pytest.raises(
        NotImplementedError, match="contact sensors cannot target kinematic entities"
    ):
        create_backend("superdex", scene, 1, 0.002)


def _physical_root_scene(
    tmp_path: Path,
    *,
    collision_enabled: bool = True,
    held_position: tuple[float, float, float] = (0.0, 0.0, 0.08),
) -> SceneCfg:
    tmp_path.mkdir(parents=True)
    scene = _scene(tmp_path)
    scene.entity_assets = tuple(
        replace(entity, initial_state=EntityInitialState((0.0, 0.0, 0.4)))
        if entity.name == "object"
        else entity
        for entity in scene.entity_assets
    )
    obj = next(entity for entity in scene.entity_assets if entity.name == "object").source
    assert obj is not None
    scene.entity_assets = scene.entity_assets + (
        SceneEntitySpec(
            "held",
            obj,
            root_mode="kinematic",
            collision_enabled=collision_enabled,
            initial_state=EntityInitialState(held_position),
        ),
    )
    return scene


def test_physical_kinematic_root_has_hidden_carrier_and_frozen_public_layout(
    tmp_path: Path,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "baseline").mkdir()
    baseline = create_backend("superdex", _scene(tmp_path / "baseline"), 1, 0.002)
    backend = create_backend("superdex", _physical_root_scene(tmp_path / "physical"), 1, 0.002)
    try:
        assert (backend.model.nq, backend.model.nv, backend.num_actuators) == (
            baseline.model.nq,
            baseline.model.nv,
            baseline.num_actuators,
        )
        layout = backend.get_scene_layout().get_entity("held")
        assert layout.root_mode == "kinematic"
        assert layout.joints == ()
        assert layout.actuator_names == ()
        plan = backend.model.actor_plans[3]
        assert plan.physical_kinematic
        assert not plan.floating
        assert not plan.kinematic_mirror
        assert plan.native_qpos_indices.size == 6
        assert plan.native_qvel_indices.size == 6
        for field in (
            "qpos_indices",
            "qvel_indices",
            "native_order_qpos_indices",
            "native_order_qvel_indices",
            "actuator_indices",
        ):
            assert getattr(plan, field).size == 0
        with pytest.raises(NotImplementedError, match="no floating root state"):
            backend.get_root_state_layout("held/body")
        assert backend.get_entity_state("held")["root_velocity"].shape == (1, 6)
        np.testing.assert_array_equal(backend.get_entity_state("held")["root_velocity"], 0)
    finally:
        baseline.close()
        backend.close()


def test_physical_kinematic_root_collision_holds_dynamic_object_in_batch_and_serial(
    tmp_path: Path,
):
    batch = create_backend("superdex", _physical_root_scene(tmp_path / "batch"), 2, 0.002)
    serial = create_backend(
        "superdex",
        _physical_root_scene(tmp_path / "serial"),
        2,
        0.002,
        superdex_execution_mode="serial",
    )
    try:
        controls = np.zeros((2, batch.num_actuators), dtype=batch.get_default_qpos().dtype)
        for backend in (batch, serial):
            backend.step(controls, nsteps=300)
        blocked = batch.get_entity_state("object")["root_pose"][:, 2]
        held_pose = batch.get_entity_state("held")["root_pose"]
        np.testing.assert_allclose(
            held_pose[:, :3], np.broadcast_to([0.0, 0.0, 0.08], (2, 3)), atol=2e-7
        )
        np.testing.assert_allclose(
            held_pose[:, 3:], np.broadcast_to([1.0, 0.0, 0.0, 0.0], (2, 4)), atol=2e-7
        )
        assert np.min(blocked) > 0.14
        np.testing.assert_allclose(
            batch.get_state()["qpos"], serial.get_state()["qpos"], atol=3e-6
        )
        np.testing.assert_allclose(
            batch.get_state()["qvel"], serial.get_state()["qvel"], atol=3e-6
        )
    finally:
        batch.close()
        serial.close()


def test_physical_kinematic_selected_reset_is_row_and_entity_local(tmp_path: Path):
    backend = create_backend("superdex", _physical_root_scene(tmp_path / "scene"), 3, 0.002)
    try:
        controls = np.array(
            [[0.4, -0.2], [-0.4, 0.2], [0.1, 0.3]], dtype=backend.get_default_qpos().dtype
        )
        backend.step(controls)
        before = backend.get_state()
        object_before = backend.get_entity_state("object").copy()
        default_pose = backend.get_entity_state("held")["root_pose"].copy()
        selected_pose = np.asarray(
            (
                (0.1, -0.02, 0.08, 0.8, 0.6, 0.0, 0.0),
                (-0.1, 0.02, 0.06, 0.8, -0.6, 0.0, 0.0),
            ),
            dtype=backend.get_default_qpos().dtype,
        )
        selected_pose[:, 3:] /= np.linalg.norm(
            selected_pose[:, 3:], axis=1, keepdims=True
        )
        backend.reset_entities(
            SceneResetRequest(
                (1, 2), (EntityStatePatch("held", root_pose=selected_pose),)
            )
        )
        held = backend.get_entity_state("held")
        np.testing.assert_allclose(
            held["root_pose"][[1, 2]], selected_pose, rtol=0, atol=2e-7
        )
        np.testing.assert_allclose(held["root_pose"][0], default_pose[0], atol=0)
        np.testing.assert_array_equal(held["root_velocity"], 0.0)
        for field, values in object_before.items():
            np.testing.assert_array_equal(backend.get_entity_state("object")[field], values)
        np.testing.assert_allclose(backend.get_state()["qpos"][0], before["qpos"][0], atol=0)
        np.testing.assert_allclose(backend.get_state()["qvel"][0], before["qvel"][0], atol=0)
        np.testing.assert_allclose(
            backend.get_state("ctrl")["ctrl"][0], controls[0], rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            backend.get_state("ctrl")["ctrl"][1:], controls[1:], rtol=0, atol=2e-8
        )

        backend.reset((1,))
        held = backend.get_entity_state("held")
        np.testing.assert_allclose(held["root_pose"][1], default_pose[1], atol=0)
        np.testing.assert_allclose(held["root_pose"][2], selected_pose[1], atol=2e-7)
    finally:
        backend.close()


def test_physical_kinematic_fixed_variant_identity_changes_support(tmp_path: Path):
    scene = _physical_root_scene(tmp_path / "scene")
    held = next(entity for entity in scene.entity_assets if entity.name == "held").source
    assert held is not None
    large_path = tmp_path / "large-held.xml"
    large_path.write_text(
        Path(held.model_file)
        .read_text(encoding="utf-8")
        .replace('mass=".3"', 'mass=".9"')
        .replace('size=".035"', 'size=".045"'),
        encoding="utf-8",
    )
    large = ModelSourceDescriptor(str(large_path))
    scene.entity_variant = EntityVariantBinding(
        "held",
        FixedVariantPlan(np.array([0, 1], dtype=np.int32), (held, large)),
    )
    backend = create_backend("superdex", scene, 2, 0.002)
    try:
        held_body = backend.get_body_ids(["held/body"])[0]
        np.testing.assert_allclose(backend.get_body_mass()[:, held_body], [0.3, 0.9], atol=0)
        topology = (backend.model.nq, backend.model.nv, backend.num_actuators)
        assert all(
            (plan.physical_kinematic, plan.native_qpos_indices.size)
            == (True, 6)
            for plan in backend.model.actor_plans
            if plan.entity_name == "held"
        )
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32), nsteps=300)
        height = backend.get_entity_state("object")["root_pose"][:, 2]
        assert height[1] > height[0] + 0.005
        assert topology == (backend.model.nq, backend.model.nv, backend.num_actuators)
    finally:
        backend.close()


def test_physical_kinematic_collision_declaration_is_enforced(tmp_path: Path):
    enabled = create_backend(
        "superdex", _physical_root_scene(tmp_path / "enabled"), 1, 0.002
    )
    disabled = create_backend(
        "superdex",
        _physical_root_scene(tmp_path / "disabled", collision_enabled=False),
        1,
        0.002,
    )
    try:
        controls = np.zeros((1, enabled.num_actuators), dtype=enabled.get_default_qpos().dtype)
        for backend in (enabled, disabled):
            backend.step(controls, nsteps=300)
        enabled_height = enabled.get_entity_state("object")["root_pose"][0, 2]
        disabled_height = disabled.get_entity_state("object")["root_pose"][0, 2]
        assert enabled_height > disabled_height + 0.05
    finally:
        enabled.close()
        disabled.close()


def test_physical_kinematic_wrench_and_contact_sensors_fail_closed(tmp_path: Path):
    scene = _physical_root_scene(tmp_path / "scene")
    backend = create_backend("superdex", scene, 1, 0.002)
    try:
        body = backend.get_body_ids(["held/body"])[0]
        with pytest.raises(
            NotImplementedError, match="kinematic entities do not own physical"
        ):
            backend.apply_body_force(np.array([body]), np.ones((1, 1, 3)))
    finally:
        backend.close()

    contact_fragment = tmp_path / "scene" / "held-contact.xml"
    contact_fragment.write_text(
        "<mujoco><sensor>"
        "<contact name='held_object_found' geom1='held/shape' "
        "geom2='object/shape' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    scene.fragment_files = (str(contact_fragment),)
    with pytest.raises(
        NotImplementedError, match="contact sensors cannot target kinematic entities"
    ):
        create_backend("superdex", scene, 1, 0.002)


def test_portable_executor_selection_uses_minimum_reviewed_abi():
    import superdex.physics

    from unisim.backend.superdex.backend import _select_portable_executor_class

    assert (
        _select_portable_executor_class(superdex.physics, physical_kinematic=False).__name__
        == "SceneBatchExecutorV2"
    )
    assert (
        _select_portable_executor_class(superdex.physics, physical_kinematic=True).__name__
        == "SceneBatchExecutorV3"
    )


def test_portable_profile_rejects_unsupported_authoring(tmp_path: Path):
    _, obj, _ = _sources(tmp_path)
    uniform_variant_scene = SceneCfg(entity_assets=(SceneEntitySpec("object", obj),))
    uniform_variant_scene.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(
            np.array([0, 1]),
            (obj, obj),
            layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
        ),
    )
    with pytest.raises(NotImplementedError, match="same_layout"):
        create_backend(
            "superdex",
            uniform_variant_scene,
            len(uniform_variant_scene.entity_variant.plan.assignment),
            0.002,
            superdex_execution_mode="serial",
        )
