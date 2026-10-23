"""Real Newton SolverMuJoCo acceptance for portable multi-entity scenes."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("newton")
warp = pytest.importorskip("warp")

from unisim.backend.newton.backend import NewtonBackend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.entity_state import inverse_rotate_vector, rotate_vector
from unisim.scene import SceneCfg


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    device = warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton multi-entity acceptance requires real CUDA")


def _robot(
    tmp_path: Path,
    *,
    gravity: str = "0 0 0",
    collide: bool = False,
    passive: bool = False,
) -> ModelSourceDescriptor:
    path = tmp_path / "robot.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    passive_joint = '<joint name="passive" axis="0 0 1"/>' if passive else ""
    collision = int(collide)
    path.write_text(
        f"""
        <mujoco><compiler angle="radian"/><option gravity="{gravity}" integrator="Euler"/>
        <worldbody>
          <body name="base">
            <inertial pos=".02 .01 .03" mass="1" diaginertia=".11 .2 .3"/>
            <geom name="base_geom" size=".08" contype="{collision}" conaffinity="{collision}"/>
            <body name="link" pos="0 0 .2">
              {passive_joint}
              <joint name="hinge" axis="0 1 0" ref=".4"/>
              <inertial pos="0 0 .01" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="link_geom" size=".03" contype="{collision}" conaffinity="{collision}"/>
            </body>
          </body>
        </worldbody>
        <actuator><motor name="drive" joint="hinge" ctrlrange="-1 1"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _object(
    tmp_path: Path,
    name: str,
    *,
    radius: float,
    mass: float,
    com: tuple[float, float, float],
    shape: str = "sphere",
    half_length: float = 0.2,
    gravity: str = "0 0 0",
    collide: bool = False,
    diaginertia: float = 0.1,
    actuated: bool = False,
) -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    geom_size = f"{radius}" if shape == "sphere" else f"{radius} {half_length}"
    driven_link = (
        f"""
        <body name="link" pos="0 0 0">
          <joint name="drive" axis="0 0 1"/>
          <inertial pos="0 0 0" mass="{mass}"
            diaginertia="{diaginertia} {diaginertia} {diaginertia}"/>
          <geom name="object_geom" type="{shape}" size="{geom_size}"
            contype="{int(collide)}" conaffinity="{int(collide)}"/>
        </body>
        """
        if actuated
        else ""
    )
    actuator_block = '<actuator><motor name="drive" joint="drive"/></actuator>' if actuated else ""
    path.write_text(
        f"""
        <mujoco><option gravity="{gravity}" integrator="Euler"/><worldbody>
          <body name="base">
            <freejoint name="root"/>
            <inertial pos="{com[0]} {com[1]} {com[2]}" mass="{0.1 if actuated else mass}"
              diaginertia="{diaginertia} {diaginertia} {diaginertia}"/>
            <geom name="{"base_geom" if actuated else "object_geom"}"
              type="{shape}" size="{geom_size}"
              contype="{int(collide)}" conaffinity="{int(collide)}"/>
            {driven_link}
          </body>
        </worldbody>{actuator_block}</mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _table(
    tmp_path: Path, *, gravity: str = "0 0 0", collide: bool = False
) -> ModelSourceDescriptor:
    path = tmp_path / "table.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <mujoco><option gravity="{gravity}" integrator="Euler"/><worldbody>
          <body name="base">
            <inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
            <geom name="table_geom" type="box" size="1 1 .1"
              contype="{int(collide)}" conaffinity="{int(collide)}"/>
          </body>
        </worldbody></mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(
    tmp_path: Path,
    *,
    object_a: ModelSourceDescriptor | None = None,
    object_b: ModelSourceDescriptor | None = None,
    assignment: tuple[int, ...] | np.ndarray = (1, 1, 0, 1, 0),
    object_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    table_position: tuple[float, float, float] = (0.0, 0.0, -3.0),
    gravity: str = "0 0 0",
    collide: bool = False,
    robot_passive: bool = False,
) -> SceneCfg:
    object_a = object_a or _object(
        tmp_path,
        "object_a",
        radius=0.1,
        mass=0.5,
        com=(0.01, 0.0, 0.0),
        gravity=gravity,
        collide=collide,
    )
    object_b = object_b or _object(
        tmp_path,
        "object_b",
        radius=0.15,
        mass=1.5,
        com=(0.03, 0.0, 0.0),
        gravity=gravity,
        collide=collide,
        diaginertia=0.2,
    )
    assignment_array = np.asarray(assignment, dtype=np.int32)
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _robot(
                    tmp_path,
                    gravity=gravity,
                    collide=collide,
                    passive=robot_passive,
                ),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "object",
                object_a,
                initial_state=EntityInitialState(object_position),
            ),
            SceneEntitySpec(
                "table",
                _table(tmp_path, gravity=gravity, collide=collide),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState(table_position),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(assignment_array, (object_a, object_b)),
        ),
    )


def _backend(scene: SceneCfg) -> NewtonBackend:
    num_envs = len(scene.entity_variant.plan.assignment) if scene.entity_variant else 5
    return NewtonBackend(
        scene,
        num_envs=num_envs,
        sim_dt=0.002,
        device=str(warp.get_device()),
        nconmax=16,
        njmax=32,
        capacity_check_steps=1,
    )


def test_fixed_passive_static_variant_and_selected_reset_survive_step(tmp_path: Path):
    backend = _backend(_scene(tmp_path))
    generated_files = tuple(item.source_model_file for item in backend._variant_metadata)
    try:
        layout = backend.get_scene_layout()
        assert backend.get_entity_names() == ("robot", "object", "table")
        assert backend.num_actuators == 1
        assert layout.get_entity("robot").root_mode == "fixed"
        assert layout.get_entity("table").root_mode == "fixed"
        assert layout.get_entity("object").root_mode == "floating"

        backend.model
        for name in ("robot", "object", "table"):
            runtime = backend._entity_runtimes[name]
            assert runtime.view.count == 5
            assert runtime.view.count_per_world == 1
        playback_paths = {backend.get_playback_model(env) for env in range(backend.num_envs)}
        assert playback_paths == set(generated_files)
        assert all(Path(path).exists() for path in playback_paths)
        assert all(mujoco.MjModel.from_xml_path(path).nbody == 5 for path in playback_paths)
        native_mass = np.asarray(backend.model.body_mass.numpy(), dtype=np.float32).copy()
        native_com = np.asarray(backend.model.body_com.numpy(), dtype=np.float32).copy()
        native_inertia = np.asarray(backend.model.body_inertia.numpy(), dtype=np.float32).copy()
        native_shape_type = np.asarray(backend.model.shape_type.numpy(), dtype=np.int32).copy()
        native_shape_scale = np.asarray(backend.model.shape_scale.numpy(), dtype=np.float32).copy()

        object_row = layout.get_entity("object").body_ids[0] - 1
        expected_assignment = np.array([1, 1, 0, 1, 0], dtype=np.int32)
        expected_mass = np.array([1.5, 1.5, 0.5, 1.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(backend.get_body_mass()[:, object_row], expected_mass, rtol=2e-6)
        np.testing.assert_allclose(
            backend.get_body_ipos(np.arange(5))[:, object_row, 0],
            expected_assignment * 0.02 + 0.01,
            rtol=2e-6,
            atol=1e-7,
        )
        shape_world = np.asarray(backend.model.shape_world.numpy(), dtype=np.int64)
        shape_scale = np.asarray(backend.model.shape_scale.numpy(), dtype=np.float32)
        object_scales = np.asarray(
            [
                shape_scale[shape_world == world_id, 0][object_row]
                for world_id in range(backend.num_envs)
            ]
        )
        np.testing.assert_allclose(
            object_scales,
            np.where(expected_assignment == 1, 0.15, 0.1),
            rtol=2e-6,
        )

        np.testing.assert_allclose(
            backend.get_entity_state("table")["root_pose"][:, :3],
            np.tile((0.0, 0.0, -3.0), (5, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"], 0.4, atol=1e-6
        )

        controls = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        backend._set_control(controls)
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=backend.get_entity_state("object")["root_pose"][[0]].copy(),
                    ),
                ),
            )
        )
        np.testing.assert_array_equal(backend._control_cache, controls)

        state_before = backend.get_physics_state().copy()
        with pytest.raises(ValueError, match="duplicate"):
            backend._host_set_state(
                np.asarray((0, 0), dtype=np.intp),
                backend._qpos_cache[[0, 0]].copy(),
                backend._qvel_cache[[0, 0]].copy(),
            )
        with pytest.raises(ValueError, match="exceed the backend environment count"):
            backend.reset_entities(
                SceneResetRequest(
                    (5,),
                    (
                        EntityStatePatch(
                            "robot", joint_positions=np.zeros((1, 1), dtype=np.float32)
                        ),
                    ),
                )
            )
        with pytest.raises(NotImplementedError, match="restore_default_controls"):
            backend.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch(
                            "robot",
                            joint_positions=np.zeros((1, 1), dtype=np.float32),
                        ),
                    ),
                    restore_default_controls=True,
                )
            )
        np.testing.assert_array_equal(backend.get_physics_state(), state_before)

        robot_before = {
            name: values.copy() for name, values in backend.get_entity_state("robot").items()
        }
        table_before = {
            name: values.copy() for name, values in backend.get_entity_state("table").items()
        }
        object_before = {
            name: values.copy() for name, values in backend.get_entity_state("object").items()
        }
        ids = (4, 1)
        pose = np.tile((4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5), (2, 1)).astype(np.float32)
        velocity = np.tile((0.2, 0.3, 0.4, 0.7, -0.8, 0.9), (2, 1)).astype(np.float32)
        backend.reset_entities(
            SceneResetRequest(
                ids,
                (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),),
            )
        )
        actual = backend.get_entity_state("object")
        np.testing.assert_allclose(actual["root_pose"][list(ids)], pose, atol=1e-6)
        np.testing.assert_allclose(actual["root_velocity"][list(ids)], velocity, atol=1e-6)
        for name, values in robot_before.items():
            np.testing.assert_array_equal(backend.get_entity_state("robot")[name], values)
        for name, values in table_before.items():
            np.testing.assert_array_equal(backend.get_entity_state("table")[name], values)
        untouched = np.asarray((0, 2, 3), dtype=np.intp)
        for name, values in object_before.items():
            if values.shape[0] == backend.num_envs:
                np.testing.assert_array_equal(actual[name][untouched], values[untouched])

        # Full selected snapshots include every entity, including the static
        # table's zero-width Newton articulation view.
        backend._host_set_state(
            np.asarray((4,), dtype=np.intp),
            backend._qpos_cache[[4]].copy(),
            backend._qvel_cache[[4]].copy(),
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_pose"][4], pose[0], atol=1e-6
        )

        robot_ids = (2, 3)
        robot_positions = np.array([[-0.2], [-0.3]], dtype=np.float32)
        robot_velocities = np.array([[0.4], [-0.5]], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                robot_ids,
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=robot_positions,
                        joint_velocities=robot_velocities,
                    ),
                ),
            )
        )
        robot_after = backend.get_entity_state("robot")
        np.testing.assert_allclose(
            robot_after["joint_positions"][list(robot_ids)], robot_positions, atol=1e-6
        )
        np.testing.assert_allclose(
            robot_after["joint_velocities"][list(robot_ids)], robot_velocities, atol=1e-6
        )
        robot_untouched = np.asarray((0, 1, 4), dtype=np.intp)
        np.testing.assert_array_equal(
            robot_after["joint_positions"][robot_untouched],
            robot_before["joint_positions"][robot_untouched],
        )
        np.testing.assert_allclose(
            backend._control_cache[:, 0],
            np.array([0.1, 0.2, 0.0, 0.0, 0.5], dtype=np.float32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            np.asarray(backend._control.mujoco.ctrl.numpy()).reshape(5, 1)[:, 0],
            np.array([0.1, 0.2, 0.0, 0.0, 0.5], dtype=np.float32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_pose"][list(ids)],
            pose,
            atol=1e-6,
        )

        backend._host_step(np.zeros((5, 1), dtype=np.float32))
        assert np.isfinite(backend.get_physics_state()).all()
        after_step = backend.get_entity_state("object")
        np.testing.assert_allclose(
            after_step["root_pose"][list(ids), :3],
            pose[:, :3] + velocity[:, :3] * 0.002,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(after_step["root_pose"][list(ids), 3:7], axis=1),
            1.0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            after_step["root_velocity"][list(ids), 3:], velocity[:, 3:], atol=2e-6
        )
        assert np.max(np.abs(after_step["root_velocity"][list(ids), :3] - velocity[:, :3])) < 2e-4
        np.testing.assert_array_equal(np.asarray(backend.model.body_mass.numpy()), native_mass)
        np.testing.assert_array_equal(np.asarray(backend.model.body_com.numpy()), native_com)
        np.testing.assert_array_equal(
            np.asarray(backend.model.body_inertia.numpy()), native_inertia
        )
        np.testing.assert_array_equal(
            np.asarray(backend.model.shape_type.numpy()), native_shape_type
        )
        np.testing.assert_array_equal(
            np.asarray(backend.model.shape_scale.numpy()), native_shape_scale
        )
    finally:
        backend.close()
    assert generated_files
    assert not any(Path(path).exists() for path in generated_files)


def test_real_portable_contact_is_attributed_only_to_its_world(tmp_path: Path):
    scene = _scene(
        tmp_path,
        assignment=(0, 1),
        object_position=(0.0, 0.0, 0.1501),
        table_position=(0.0, 0.0, 0.0),
        gravity="0 0 -9.81",
        collide=True,
    )
    fragment = tmp_path / "sensors.xml"
    fragment.write_text(
        '<mujoco><sensor><contact name="object_table" geom1="object/object_geom" '
        'geom2="table/table_geom" data="found" num="1"/></sensor></mujoco>',
        encoding="utf-8",
    )
    scene.fragment_files = [str(fragment)]
    scene.entity_assets = (
        scene.entity_assets[0],
        scene.entity_assets[1],
        SceneEntitySpec(
            "payload",
            _object(
                tmp_path,
                "payload",
                radius=0.07,
                mass=0.3,
                com=(0.0, 0.0, 0.0),
                gravity="0 0 -9.81",
            ),
            initial_state=EntityInitialState((0.0, 5.0, 2.0)),
        ),
        scene.entity_assets[2],
    )
    backend = _backend(scene)
    try:
        backend.model
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=np.array(
                            [[100.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                        ),
                    ),
                ),
            )
        )
        backend._host_step(np.zeros((2, 1), dtype=np.float32))
        np.testing.assert_array_equal(backend._host_read_sensor("object_table"), [[1.0], [0.0]])

        # Every native solver contact belongs to exactly one env world.  This
        # complements the public flag check above and catches a malformed
        # cross-world shape pair even when another sensor pair would match.
        shape_world = np.asarray(backend.model.shape_world.numpy(), dtype=np.int64)
        contact_count = int(np.asarray(backend._contacts.rigid_contact_count.numpy())[0])
        assert contact_count > 0
        shape0 = np.asarray(
            backend._contacts.rigid_contact_shape0.numpy()[:contact_count], dtype=np.int64
        )
        shape1 = np.asarray(
            backend._contacts.rigid_contact_shape1.numpy()[:contact_count], dtype=np.int64
        )
        contact_worlds0 = shape_world[shape0]
        contact_worlds1 = shape_world[shape1]
        assert np.all(contact_worlds0 == contact_worlds1)
        assert np.all((contact_worlds0 >= 0) & (contact_worlds0 < backend.num_envs))
        assert set(contact_worlds0.tolist()) == {0}

        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=np.array(
                            [[-100.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                        ),
                    ),
                ),
            )
        )
        backend._host_step(np.zeros((2, 1), dtype=np.float32))
        np.testing.assert_array_equal(backend._host_read_sensor("object_table"), [[0.0], [0.0]])
    finally:
        backend.close()


def test_nonidentity_orientation_offset_com_and_variant_response(tmp_path: Path):
    scene = _scene(
        tmp_path,
        assignment=(0, 0, 1, 1),
        object_position=(-10.0, 0.0, 5.0),
        table_position=(-100.0, 0.0, -100.0),
        gravity="0 0 -9.81",
    )
    backend = _backend(scene)
    try:
        backend.model
        object_entity = backend.get_scene_layout().get_entity("object")
        qcols = object_entity.root_qpos_indices
        vcols = object_entity.root_qvel_indices
        root_row = object_entity.body_ids[0] - 1
        half_root_2 = np.float32(np.sqrt(0.5))
        pose_quaternion = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [half_root_2, 0.0, 0.0, half_root_2],
                [1.0, 0.0, 0.0, 0.0],
                [half_root_2, 0.0, 0.0, half_root_2],
            ],
            dtype=np.float32,
        )
        pose = np.tile(np.array([-10.0, 0.0, 5.0], dtype=np.float32), (4, 1))
        pose = np.concatenate((pose, pose_quaternion), axis=1)
        public_velocity = np.tile(
            np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32), (4, 1)
        )
        backend.reset_entities(
            SceneResetRequest(
                (0, 1, 2, 3),
                (EntityStatePatch("object", root_pose=pose, root_velocity=public_velocity),),
            )
        )
        actual = backend.get_entity_state("object")
        np.testing.assert_allclose(actual["root_pose"], pose, atol=1e-6)
        np.testing.assert_allclose(actual["root_velocity"], public_velocity, atol=1e-6)

        raw_qpos = backend._qpos_cache.copy()
        raw_qvel = backend._qvel_cache.copy()
        converted_qpos, converted_qvel = backend._raw_state_from_public(raw_qpos, raw_qvel)
        np.testing.assert_allclose(
            converted_qpos[:, qcols[3:7]], pose[:, 3:7][:, [1, 2, 3, 0]], atol=1e-7
        )
        expected_body_omega = inverse_rotate_vector(pose[:, 3:7], public_velocity[:, 3:6])
        expected_world_omega = public_velocity[:, 3:6]
        expected_native_omega = expected_world_omega
        np.testing.assert_allclose(raw_qvel[:, vcols[3:6]], expected_body_omega, atol=1e-6)
        offset_world = rotate_vector(pose[:, 3:7], backend.get_body_ipos(np.arange(4))[:, root_row])
        expected_native_linear = public_velocity[:, :3] + np.cross(
            expected_native_omega, offset_world
        )
        np.testing.assert_allclose(
            converted_qvel[np.ix_(np.arange(4), vcols[:3])],
            expected_native_linear,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            converted_qvel[np.ix_(np.arange(4), vcols[3:6])],
            expected_native_omega,
            atol=1e-6,
        )

        velocity_before = backend.get_entity_state("object")["root_velocity"].copy()
        backend._host_step(np.zeros((4, 1), dtype=np.float32))
        state_after = backend.get_entity_state("object")
        velocity_after = state_after["root_velocity"]
        pose_after = state_after["root_pose"]
        np.testing.assert_allclose(
            pose_after[:, :2], pose[:, :2] + velocity_before[:, :2] * 0.002, atol=2e-6
        )
        np.testing.assert_allclose(
            pose_after[:, 2], pose[:, 2] + velocity_after[:, 2] * 0.002, atol=2e-5
        )
        assert np.linalg.norm(pose_after[:, 3:7] - pose[:, 3:7], axis=1).min() > 1e-4
        np.testing.assert_allclose(
            velocity_after[:, 2], velocity_before[:, 2] - 9.81 * 0.002, atol=2e-4
        )
    finally:
        backend.close()


def test_passive_joint_keeps_action_dimension_and_control_force_drives_response(
    tmp_path: Path,
):
    backend = _backend(
        _scene(
            tmp_path,
            assignment=(0, 1),
            object_position=(-20.0, 0.0, 5.0),
            robot_passive=True,
        )
    )
    try:
        backend.model
        robot = backend.get_scene_layout().get_entity("robot")
        assert tuple(joint.name for joint in robot.joints) == ("passive", "hinge")
        assert backend.num_actuators == 1
        assert robot.actuator_names == ("drive",)
        assert robot.actuator_indices == (0,)
        before = backend.get_entity_state("robot")["joint_velocities"].copy()
        backend._host_step(np.array([[0.0], [0.5]], dtype=np.float32))
        after = backend.get_entity_state("robot")["joint_velocities"]
        np.testing.assert_allclose(after[:, 0], before[:, 0], atol=1e-7)
        assert abs(float(after[1, 1] - before[1, 1])) > 1e-4
        assert abs(float(after[1, 1] - after[0, 1])) > 1e-4
    finally:
        backend.close()


def test_variant_mass_and_inertia_change_solver_force_response(tmp_path: Path):
    object_a = _object(
        tmp_path,
        "object_a",
        radius=0.1,
        mass=0.5,
        com=(0.0, 0.0, 0.0),
        diaginertia=0.1,
        actuated=True,
    )
    object_b = _object(
        tmp_path,
        "object_b",
        radius=0.1,
        mass=1.5,
        com=(0.0, 0.0, 0.0),
        diaginertia=0.2,
        actuated=True,
    )
    backend = _backend(
        _scene(
            tmp_path,
            object_a=object_a,
            object_b=object_b,
            assignment=(0, 1),
            object_position=(-30.0, 0.0, 5.0),
            table_position=(-100.0, 0.0, -100.0),
        )
    )
    try:
        backend.model
        object_entity = backend.get_scene_layout().get_entity("object")
        assert tuple(joint.name for joint in object_entity.joints) == ("drive",)
        assert object_entity.actuator_names == ("drive",)

        controls = np.zeros((2, 2), dtype=np.float32)
        controls[:, object_entity.actuator_indices] = 1.0
        backend._host_step(controls)
        velocity = backend.get_entity_state("object")["joint_velocities"]

        assert abs(float(velocity[0, 0]) - float(velocity[1, 0])) > 0.002
        assert abs(float(velocity[0, 0])) > abs(float(velocity[1, 0]))
        assert abs(float(velocity[0, 0]) - 2.0 * float(velocity[1, 0])) < 1e-5
    finally:
        backend.close()


def test_selected_world_matches_independent_single_world_solver(tmp_path: Path):
    batched = _backend(_scene(tmp_path / "batched", assignment=(1, 1, 1, 1, 1)))
    single = _backend(_scene(tmp_path / "single", assignment=(1,)))
    try:
        batched.model
        single.model
        pose = np.tile(np.array([3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (5, 1))
        velocity = np.tile(np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0], dtype=np.float32), (5, 1))
        batched.reset_entities(
            SceneResetRequest(
                tuple(range(5)),
                (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),),
            )
        )
        single.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=pose[[0]],
                        root_velocity=velocity[[0]],
                    ),
                ),
            )
        )
        control = np.full((5, 1), 0.3, dtype=np.float32)
        batched._host_step(control, nsteps=5)
        single._host_step(control[[0]], nsteps=5)
        for entity in ("robot", "object", "table"):
            batched_state = batched.get_entity_state(entity)
            single_state = single.get_entity_state(entity)
            for field, values in batched_state.items():
                np.testing.assert_allclose(
                    values,
                    np.repeat(single_state[field], 5, axis=0),
                    rtol=2e-6,
                    atol=2e-6,
                )
    finally:
        batched.close()
        single.close()


def test_source_order_permutation_keeps_entity_address_identity(tmp_path: Path):
    forward = _backend(_scene(tmp_path / "forward", assignment=(0, 1)))
    reverse_scene = _scene(tmp_path / "reverse", assignment=(0, 1))
    reverse_scene.entity_assets = tuple(reversed(reverse_scene.entity_assets))
    reverse = _backend(reverse_scene)
    try:
        forward.model
        reverse.model
        assert forward.get_entity_names() == ("robot", "object", "table")
        assert reverse.get_entity_names() == ("table", "object", "robot")
        forward_layout = forward.get_scene_layout()
        reverse_layout = reverse.get_scene_layout()
        for layout in (forward_layout, reverse_layout):
            qpos_spans = {
                entity.name: (
                    int(np.min(entity.qpos_indices)),
                    int(np.max(entity.qpos_indices)),
                )
                for entity in layout.entities
                if entity.qpos_indices
            }
            assert len(qpos_spans) == 2
            assert qpos_spans["robot"] != qpos_spans["object"]
            assert qpos_spans["robot"][1] < qpos_spans["object"][0] or (
                qpos_spans["object"][1] < qpos_spans["robot"][0]
            )

        object_pose = np.tile(
            np.array([2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1)
        )
        object_velocity = np.tile(
            np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1)
        )
        robot_positions = np.array([[0.1], [-0.2]], dtype=np.float32)
        robot_velocities = np.array([[0.3], [-0.4]], dtype=np.float32)
        for backend in (forward, reverse):
            backend.reset_entities(
                SceneResetRequest(
                    (0, 1),
                    (
                        EntityStatePatch(
                            "object", root_pose=object_pose, root_velocity=object_velocity
                        ),
                        EntityStatePatch(
                            "robot",
                            joint_positions=robot_positions,
                            joint_velocities=robot_velocities,
                        ),
                    ),
                )
            )
            backend._host_step(np.array([[0.2], [-0.2]], dtype=np.float32), nsteps=3)
        for entity in ("robot", "object", "table"):
            forward_state = forward.get_entity_state(entity)
            reverse_state = reverse.get_entity_state(entity)
            for field, values in forward_state.items():
                np.testing.assert_allclose(values, reverse_state[field], rtol=2e-6, atol=2e-6)
    finally:
        forward.close()
        reverse.close()


def test_kinematic_mirror_fails_closed_before_solver_construction(tmp_path: Path):
    scene = _scene(tmp_path, assignment=(0,))
    scene.entity_assets = (
        *scene.entity_assets,
        SceneEntitySpec(
            "mirror",
            kind="rigid",
            root_mode="kinematic",
            collision_enabled=False,
            mirror_of="object",
        ),
    )
    with pytest.raises(NotImplementedError, match="kinematic mirrors"):
        _backend(scene)


def _hand(tmp_path: Path, name: str, *, gravity: str = "0 0 0") -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <mujoco><compiler angle="radian"/><option gravity="{gravity}" integrator="Euler"/>
        <worldbody>
          <body name="base">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
            <geom name="palm_geom" size=".05" contype="0" conaffinity="0"/>
            <body name="wrist_link" pos="0 0 .1">
              <joint name="wrist" type="ball"/>
              <inertial pos="0 0 .01" mass=".2" diaginertia=".04 .05 .06"/>
              <geom name="wrist_geom" size=".03" contype="0" conaffinity="0"/>
              <body name="finger" pos="0 0 .05">
                <joint name="flex" type="hinge" axis="0 1 0"/>
                <inertial pos="0 0 .01" mass=".1" diaginertia=".01 .01 .01"/>
                <geom name="finger_geom" size=".02" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>
        </worldbody>
        <actuator><motor name="flex_motor" joint="flex" ctrlrange="-1 1"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _ball_scene(tmp_path: Path, *, gravity: str = "0 0 -9.81") -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "left_hand",
                _hand(tmp_path, "left_hand", gravity=gravity),
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "right_hand",
                _hand(tmp_path, "right_hand", gravity=gravity),
                initial_state=EntityInitialState((0.5, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "object",
                _object(
                    tmp_path,
                    "object",
                    radius=0.1,
                    mass=0.5,
                    com=(0.0, 0.0, 0.0),
                    gravity=gravity,
                ),
                initial_state=EntityInitialState((0.25, 0.5, 1.0)),
            ),
        ),
    )


def _axis_angle_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * unit))


def _canonical(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[..., 0] < 0.0:
        quat = -quat
    return quat


_ROOT_QUAT_A = _axis_angle_wxyz((0.0, 0.0, 1.0), 0.3)
_ROOT_QUAT_B = _axis_angle_wxyz((0.0, 1.0, 0.0), -0.25)
_BALL_QUAT_A = _axis_angle_wxyz((1.0, 2.0, 3.0), 0.4)
_BALL_QUAT_B = _axis_angle_wxyz((0.0, 1.0, 0.0), 0.7)


def _ball_payload(layout_nq: int, layout_nv: int) -> tuple[np.ndarray, np.ndarray]:
    """Two full-width public-layout rows covering every entity's coordinates."""
    qpos = np.zeros((2, layout_nq), dtype=np.float32)
    qvel = np.zeros((2, layout_nv), dtype=np.float32)
    for row, (root_quat, ball_quat) in enumerate(
        ((_ROOT_QUAT_A, _BALL_QUAT_A), (_ROOT_QUAT_B, _BALL_QUAT_B))
    ):
        qpos[row, 0:3] = (0.1, -0.05, 1.2) if row == 0 else (-0.08, 0.07, 0.9)
        qpos[row, 3:7] = root_quat
        qpos[row, 7:11] = ball_quat
        qpos[row, 11] = 0.4 if row == 0 else -0.2
        qpos[row, 12:15] = (0.6, 0.1, 1.1) if row == 0 else (0.45, -0.12, 1.05)
        qpos[row, 15:19] = (_ROOT_QUAT_B if row == 0 else _ROOT_QUAT_A)
        qpos[row, 19:23] = (_BALL_QUAT_B if row == 0 else _BALL_QUAT_A)
        qpos[row, 23] = -0.3 if row == 0 else 0.25
        qpos[row, 24:27] = (0.3, 0.55, 1.2) if row == 0 else (0.2, 0.45, 1.15)
        qpos[row, 27:31] = root_quat
        qvel[row, 0:6] = (
            (0.01, 0.02, -0.05, 0.1, 0.05, -0.2)
            if row == 0
            else (-0.02, 0.03, 0.04, -0.15, 0.1, 0.25)
        )
        qvel[row, 6:9] = (0.3, -0.2, 0.5) if row == 0 else (-0.4, 0.15, -0.35)
        qvel[row, 9] = -0.3 if row == 0 else 0.45
        qvel[row, 10:16] = (
            (0.05, -0.01, 0.02, 0.2, -0.1, 0.15)
            if row == 0
            else (-0.03, 0.02, -0.04, 0.1, 0.2, -0.1)
        )
        qvel[row, 16:19] = (-0.25, 0.35, -0.15) if row == 0 else (0.2, -0.3, 0.4)
        qvel[row, 19] = 0.5 if row == 0 else -0.4
        qvel[row, 20:26] = (
            (0.1, 0.2, -0.1, 0.3, -0.2, 0.1) if row == 0 else (-0.15, 0.1, 0.05, -0.2, 0.15, -0.25)
        )
    return qpos, qvel


def test_portable_ball_joints_materialize_round_trip_and_match_mujoco(tmp_path: Path):
    backend = _backend(_ball_scene(tmp_path))
    try:
        backend.model
        layout = backend.get_scene_layout()
        assert (layout.nq, layout.nv) == (31, 26)
        left = layout.get_entity("left_hand")
        assert tuple((joint.name, joint.kind) for joint in left.joints) == (
            ("wrist", "ball"),
            ("flex", "hinge"),
        )
        ball = left.joints[0]
        assert ball.qpos_indices == (7, 8, 9, 10)
        assert ball.qvel_indices == (6, 7, 8)
        assert backend._metadata.ball_joint_names == ("left_hand/wrist", "right_hand/wrist")
        assert backend._metadata.ball_joint_addresses == ((7, 6), (19, 16))

        # Public defaults keep the MJCF wxyz layout while the native model is
        # seeded with raw xyzw ball quaternions per world.
        default_qpos = backend.get_default_qpos()
        np.testing.assert_allclose(default_qpos[7:11], (1.0, 0.0, 0.0, 0.0), atol=1e-7)
        np.testing.assert_allclose(default_qpos[19:23], (1.0, 0.0, 0.0, 0.0), atol=1e-7)
        raw_default = np.asarray(backend.model.joint_q.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nq
        )
        for env in range(backend.num_envs):
            np.testing.assert_allclose(raw_default[env, 7:11], (0.0, 0.0, 0.0, 1.0), atol=1e-7)
            np.testing.assert_allclose(raw_default[env, 19:23], (0.0, 0.0, 0.0, 1.0), atol=1e-7)

        qpos, qvel = _ball_payload(layout.nq, layout.nv)
        backend._host_set_state(np.asarray((1, 3), dtype=np.intp), qpos, qvel)
        np.testing.assert_allclose(backend._qpos_cache[1], qpos[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(backend._qpos_cache[3], qpos[1], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(backend._qvel_cache[1], qvel[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(backend._qvel_cache[3], qvel[1], rtol=1e-6, atol=1e-6)
        for env in (0, 2, 4):
            np.testing.assert_allclose(
                backend._qpos_cache[env], default_qpos, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(backend._qvel_cache[env], 0.0, atol=1e-7)
        left_state = backend.get_entity_state("left_hand")
        np.testing.assert_allclose(left_state["root_pose"][1], qpos[0, 0:7], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            left_state["joint_positions"][1], qpos[0, 7:12], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            left_state["joint_velocities"][1], qvel[0, 6:10], rtol=1e-6, atol=1e-6
        )

        # The composed playback file is a plain MJCF model; CPU MuJoCo pins
        # the public-to-native translation of every entity's articulation view.
        reference_model = mujoco.MjModel.from_xml_path(backend.get_playback_model(1))
        reference = mujoco.MjData(reference_model)
        reference.qpos[:] = qpos[0]
        reference.qvel[:] = qvel[0]
        mujoco.mj_forward(reference_model, reference)
        for body_row in range(layout.nbody - 1):
            body_id = body_row + 1
            np.testing.assert_allclose(
                backend._body_pos_cache[1, body_row],
                reference.xpos[body_id],
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                _canonical(backend._body_quat_cache[1, body_row]),
                _canonical(np.asarray(reference.xquat[body_id])),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                backend._body_ang_vel_cache[1, body_row],
                np.asarray(reference.cvel[body_id])[:3],
                rtol=1e-4,
                atol=1e-5,
            )

        # Entity patches address ball joints as 4-wide wxyz positions and
        # 3-wide velocities, and fail closed on non-unit quaternions.
        patch_positions = np.concatenate((_BALL_QUAT_A, (0.2,))).astype(np.float32).reshape(1, -1)
        patch_velocities = np.array([[0.4, -0.1, 0.2, 0.6]], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                (2,),
                (
                    EntityStatePatch(
                        "left_hand",
                        joint_positions=patch_positions,
                        joint_velocities=patch_velocities,
                    ),
                ),
            )
        )
        patched = backend.get_entity_state("left_hand")
        np.testing.assert_allclose(
            patched["joint_positions"][2], patch_positions[0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            patched["joint_velocities"][2], patch_velocities[0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            patched["joint_positions"][1], qpos[0, 7:12], rtol=1e-6, atol=1e-6
        )
        right_state = backend.get_entity_state("right_hand")
        np.testing.assert_allclose(
            right_state["joint_positions"][2], (1.0, 0.0, 0.0, 0.0, 0.0), atol=1e-7
        )
        with pytest.raises(ValueError, match="unit wxyz"):
            backend.reset_entities(
                SceneResetRequest(
                    (2,),
                    (
                        EntityStatePatch(
                            "left_hand",
                            joint_positions=np.ones((1, 5), dtype=np.float32),
                        ),
                    ),
                )
            )

        backend._host_step(
            np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32), nsteps=3
        )
        assert np.isfinite(backend.get_physics_state()).all()
    finally:
        backend.close()


def test_portable_ball_joint_single_step_matches_native_newton(tmp_path: Path):
    backend = _backend(_ball_scene(tmp_path))
    try:
        backend.model
        newton = backend._deps.newton
        layout = backend.get_scene_layout()
        playback_file = backend.get_playback_model(1)
        previous_layout = bool(newton.use_coord_layout_targets)
        newton.use_coord_layout_targets = True
        try:
            template = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(template)
            template.add_mjcf(playback_file, ctrl_direct=False)
            builder = newton.ModelBuilder()
            builder.replicate(template, 1)
            native_model = builder.finalize(device=backend._device)
        finally:
            newton.use_coord_layout_targets = previous_layout
        native_solver = newton.solvers.SolverMuJoCo(
            native_model,
            separate_worlds=True,
            nconmax=16,
            njmax=32,
            solver="newton",
            integrator="implicitfast",
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
        )
        native_state = native_model.state()
        native_state_out = native_model.state()
        native_control = native_model.control()
        native_contacts = newton.Contacts(
            native_solver.get_max_contact_count(), 0, device=backend._device
        )

        qpos, qvel = _ball_payload(layout.nq, layout.nv)
        backend._host_set_state(np.asarray((1,), dtype=np.intp), qpos[:1], qvel[:1])
        raw_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nq
        )
        raw_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nv
        )
        # Seed the native single-world model with the adapter's raw env slice
        # so both pipelines integrate one step from identical conditions.
        native_state.joint_q.assign(raw_q[1])
        native_state.joint_qd.assign(raw_qd[1])
        newton.eval_fk(native_model, native_state.joint_q, native_state.joint_qd, native_state)
        mask = warp.array(
            np.array([True, False]),
            dtype=warp.bool,
            device=backend._device,
        )
        native_solver.reset(native_state, mask, flags=0)

        backend._host_step(
            np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32), nsteps=1
        )
        native_state.clear_forces()
        native_solver.step(
            native_state, native_state_out, native_control, native_contacts, 0.002
        )
        native_state, native_state_out = native_state_out, native_state

        stepped_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nq
        )
        stepped_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nv
        )
        np.testing.assert_allclose(
            np.asarray(native_state.joint_q.numpy(), dtype=np.float32),
            stepped_q[1],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(native_state.joint_qd.numpy(), dtype=np.float32),
            stepped_qd[1],
            rtol=1e-5,
            atol=1e-6,
        )
    finally:
        backend.close()


def test_mixed_shape_variants_fail_before_solver_construction(tmp_path: Path):
    capsule = _object(
        tmp_path,
        "object_capsule",
        radius=0.12,
        mass=1.5,
        com=(0.02, 0.0, 0.0),
        shape="capsule",
    )
    backend = _backend(_scene(tmp_path, object_b=capsule))
    try:
        with pytest.raises(ValueError, match="same shape-type sequence"):
            backend.materialize()
        assert backend._solver is None
    finally:
        backend.close()
