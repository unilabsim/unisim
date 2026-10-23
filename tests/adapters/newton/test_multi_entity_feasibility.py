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
from unisim.utils.rotation import np_quat_apply_batched


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


def test_rotated_principal_inertia_frame_survives_variant_audit(tmp_path: Path):
    """#441: off-diagonal fullinertia (rotated principal frame) materializes.

    Newton stores ``body_inertia`` as the full tensor in the body frame; the
    audit expectation is rotated out of the MuJoCo principal frame with
    ``body_iquat``, so a rotated principal frame no longer false-rejects.
    """
    path = tmp_path / "tilted.xml"
    path.write_text(
        """
        <mujoco><option gravity="0 0 0" integrator="Euler"/><worldbody>
          <body name="base">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="1.5" fullinertia=".11 .2 .3 .015 -.02 .005"/>
            <geom name="object_geom" type="sphere" size="0.1" contype="0" conaffinity="0"/>
          </body>
        </worldbody></mujoco>
        """,
        encoding="utf-8",
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", ModelSourceDescriptor(str(path))),
            SceneEntitySpec(
                "table",
                _table(tmp_path),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, -3.0)),
            ),
        )
    )
    backend = _backend(scene)
    try:
        backend.model  # materialize; the variant audit runs here
        reference = mujoco.MjModel.from_xml_path(str(path))
        native = np.asarray(backend.model.body_inertia.numpy(), dtype=np.float64)
        object_row = backend.get_scene_layout().get_entity("object").body_ids[0] - 1
        w, x, y, z = np.asarray(reference.body_iquat[1], dtype=np.float64)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        expected = (
            rotation @ np.diag(reference.body_inertia[1].astype(np.float64)) @ rotation.T
        )
        assert not np.allclose(expected, np.diag(np.diag(expected)), atol=1e-9)
        np.testing.assert_allclose(native[object_row], expected, rtol=2e-5, atol=2e-6)
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


def test_portable_joint_f_ball_torque_injection_converts_body_to_anchor_frame(tmp_path: Path):
    """Portable injection rotates public body-frame ball torques into the anchor frame."""
    torch = pytest.importorskip("torch")
    backend = _backend(_ball_scene(tmp_path))
    try:
        backend.model
        layout = backend.get_scene_layout()
        num_envs = backend.num_envs
        nv = layout.nv
        assert backend._metadata.ball_joint_addresses == ((7, 6), (19, 16))

        # Alternate the two non-identity wrist poses across envs.
        quats = np.stack(
            [_BALL_QUAT_A if env % 2 == 0 else _BALL_QUAT_B for env in range(num_envs)]
        )
        qpos = np.tile(backend.get_default_qpos(), (num_envs, 1))
        qvel = np.zeros((num_envs, nv), dtype=np.float32)
        for qpos_adr, _ in backend._metadata.ball_joint_addresses:
            qpos[:, qpos_adr : qpos_adr + 4] = quats
        backend._host_set_state(np.arange(num_envs, dtype=np.intp), qpos, qvel)

        device = torch.device(backend._device)
        joint_f = torch.zeros((num_envs, nv), dtype=torch.float32, device=device)
        variation = np.arange(num_envs, dtype=np.float32)[:, None] * 0.05
        torques_body = {
            6: np.tile(np.array([0.3, -0.2, 0.5], dtype=np.float32), (num_envs, 1)) + variation,
            16: np.tile(np.array([-0.4, 0.15, -0.35], dtype=np.float32), (num_envs, 1))
            - variation,
        }
        hinge_values = {9: np.linspace(0.7, -0.6, num_envs, dtype=np.float32), 19: 0.25}
        object_wrench = np.linspace(-0.4, 0.4, 6, dtype=np.float32)
        for qvel_adr, torque in torques_body.items():
            joint_f[:, qvel_adr : qvel_adr + 3] = torch.as_tensor(torque, device=device)
        for qvel_adr, value in hinge_values.items():
            joint_f[:, qvel_adr] = torch.as_tensor(value, device=device)
        joint_f[:, 20:26] = torch.as_tensor(
            np.tile(object_wrench, (num_envs, 1)), device=device
        )
        snapshot = joint_f.clone()
        backend.set_generalized_force_tensor(joint_f=joint_f)

        # Per-entity control views expose the native joint_f: ball columns at
        # local 6:9 carry the anchor-frame torque, everything else passes
        # through unconverted.
        for name, qvel_adr in (("left_hand", 6), ("right_hand", 16)):
            native = (
                backend._deps.warp.to_torch(
                    backend._entity_runtimes[name].view.get_dof_forces(backend._control)
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
            assert native.shape == (num_envs, 10)
            expected = np_quat_apply_batched(quats, torques_body[qvel_adr])
            np.testing.assert_allclose(native[:, 6:9], expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                native[:, 9], hinge_values[9 if name == "left_hand" else 19], atol=1e-7
            )
            np.testing.assert_allclose(native[:, :6], 0.0, atol=1e-7)
        # The free-root object entity owns no ball columns: its world-frame
        # root wrench passes through verbatim.
        object_native = (
            backend._deps.warp.to_torch(
                backend._entity_runtimes["object"].view.get_dof_forces(backend._control)
            )
            .squeeze(1)
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(
            object_native, np.tile(object_wrench, (num_envs, 1)), rtol=1e-6, atol=1e-7
        )
        # The conversion never mutates the caller's tensor.
        assert torch.equal(joint_f, snapshot)
    finally:
        backend.close()


def _composite_hand(tmp_path: Path, name: str, *, gravity: str = "0 0 0") -> ModelSourceDescriptor:
    """CHORD-style composite floating base: 3 canonical slides + terminal ball (#435)."""
    path = tmp_path / f"{name}.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <mujoco><compiler angle="radian"/><option gravity="{gravity}" integrator="Euler"/>
        <worldbody>
          <body name="mount_x">
            <inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>
            <joint name="px" type="slide" axis="1 0 0"/>
            <body name="mount_y">
              <inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>
              <joint name="py" type="slide" axis="0 1 0"/>
              <body name="mount_z">
                <inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>
                <joint name="pz" type="slide" axis="0 0 1"/>
                <body name="palm">
                  <inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
                  <joint name="palm_ball" type="ball"/>
                  <geom name="palm_geom" type="sphere" size=".05" contype="0" conaffinity="0"/>
                  <body name="finger" pos="0 0 .1">
                    <inertial mass=".05" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>
                    <joint name="finger_hinge" type="hinge" axis="0 1 0"/>
                    <geom name="finger_geom" type="capsule" size=".01 .05"
                      contype="0" conaffinity="0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </worldbody>
        <actuator>
          <position name="px_act" joint="px" kp="10"/>
          <position name="py_act" joint="py" kp="10"/>
          <position name="pz_act" joint="pz" kp="10"/>
          <position name="finger_act" joint="finger_hinge" kp="5"/>
        </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _free_hand(tmp_path: Path, name: str, *, gravity: str = "0 0 0") -> ModelSourceDescriptor:
    """The equivalent single-free-joint writing of the composite hand's palm subtree."""
    path = tmp_path / f"{name}.xml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <mujoco><compiler angle="radian"/><option gravity="{gravity}" integrator="Euler"/>
        <worldbody>
          <body name="palm">
            <freejoint name="root"/>
            <inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>
            <geom name="palm_geom" type="sphere" size=".05" contype="0" conaffinity="0"/>
            <body name="finger" pos="0 0 .1">
              <inertial mass=".05" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>
              <joint name="finger_hinge" type="hinge" axis="0 1 0"/>
              <geom name="finger_geom" type="capsule" size=".01 .05" contype="0" conaffinity="0"/>
            </body>
          </body>
        </worldbody>
        <actuator><position name="finger_act" joint="finger_hinge" kp="5"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _composite_scene(tmp_path: Path, *, gravity: str = "0 0 0") -> SceneCfg:
    """One composite-base hand, its single-free-joint twin, and a free object."""
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "composite_hand",
                _composite_hand(tmp_path, "composite_hand", gravity=gravity),
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "free_hand",
                _free_hand(tmp_path, "free_hand", gravity=gravity),
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


_COMPOSITE_QUAT_A = _axis_angle_wxyz((1.0, 2.0, 3.0), 0.4)
_COMPOSITE_QUAT_B = _axis_angle_wxyz((2.0, -1.0, 0.5), -0.7)


_COMPOSITE_MOUNT = np.array((0.0, 0.0, 1.0), dtype=np.float32)


def _composite_payload(layout_nq: int, layout_nv: int) -> tuple[np.ndarray, np.ndarray]:
    """Two full-width public-layout rows covering both hands and the object.

    Composite root columns are mount-origin-relative source values, so the
    free hand carries the composite hand's root position translated by the
    composite mount origin in every row; their palm body poses/velocities
    must then agree exactly after a commit.
    """
    qpos = np.zeros((2, layout_nq), dtype=np.float32)
    qvel = np.zeros((2, layout_nv), dtype=np.float32)
    for row, (hand_quat, object_quat) in enumerate(
        ((_COMPOSITE_QUAT_A, _COMPOSITE_QUAT_B), (_COMPOSITE_QUAT_B, _COMPOSITE_QUAT_A))
    ):
        qpos[row, 0:3] = (0.1, -0.05, 1.2) if row == 0 else (-0.08, 0.07, 0.9)
        qpos[row, 3:7] = hand_quat
        qpos[row, 7] = 0.4 if row == 0 else -0.2
        qpos[row, 8:11] = qpos[row, 0:3] + _COMPOSITE_MOUNT
        qpos[row, 11:15] = qpos[row, 3:7]
        qpos[row, 15] = qpos[row, 7]
        qpos[row, 16:19] = (0.3, 0.55, 1.2) if row == 0 else (0.2, 0.45, 1.15)
        qpos[row, 19:23] = object_quat
        qvel[row, 0:6] = (
            (0.01, 0.02, -0.05, 0.1, 0.05, -0.2)
            if row == 0
            else (-0.02, 0.03, 0.04, -0.15, 0.1, 0.25)
        )
        qvel[row, 6] = -0.3 if row == 0 else 0.45
        qvel[row, 7:13] = qvel[row, 0:6]
        qvel[row, 13] = qvel[row, 6]
        qvel[row, 14:20] = (
            (0.1, 0.2, -0.1, 0.3, -0.2, 0.1)
            if row == 0
            else (-0.15, 0.1, 0.05, -0.2, 0.15, -0.25)
        )
    return qpos, qvel


def _assert_composite_scene_matches_mujoco(
    backend: NewtonBackend,
    layout,
    composite_entity,
    env: int,
    qpos_row: np.ndarray,
    qvel_row: np.ndarray,
) -> mujoco.MjData:
    """Pin every body's kinematics against CPU MuJoCo on the playback file.

    The composite base's declared initial position locates only the mount
    origin on the chain root body; slide joint ``ref`` values stay
    source-owned (this fixture authors none, so they read back as zero).
    MuJoCo slide kinematics place bodies at ``axis * (qpos - ref)`` while
    Newton's importer keeps slide coordinates absolute, so whenever a source
    authors nonzero references, chain body positions line up only after
    adding back the references accumulated along the body's own chain path.
    Joint velocities, orientations and angular velocities are ``ref``-
    invariant.
    """
    reference_model = mujoco.MjModel.from_xml_path(backend.get_playback_model(env))
    reference = mujoco.MjData(reference_model)
    reference.qpos[:] = qpos_row
    reference.qvel[:] = qvel_row
    mujoco.mj_forward(reference_model, reference)
    ref_by_body: dict[str, np.ndarray] = {}
    for joint in composite_entity.root_joints:
        if joint.kind != "slide":
            continue
        joint_id = mujoco.mj_name2id(
            reference_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{composite_entity.name}/{joint.name}",
        )
        ref_by_body[joint.body_name] = (
            np.asarray(reference_model.jnt_axis[joint_id], dtype=np.float64)
            * float(reference_model.qpos0[joint.qpos_indices[0]])
        )
    parent_of = dict(zip(composite_entity.body_names, composite_entity.body_parent_names))
    name_by_row = {body_id - 1: name for body_id, name in zip(
        composite_entity.body_ids, composite_entity.body_names
    )}

    def _ref_offset(local_name: str) -> np.ndarray:
        total = np.zeros(3)
        current: str | None = local_name
        while current is not None:
            total = total + ref_by_body.get(current, np.zeros(3))
            current = parent_of[current]
        return total

    for body_row in range(layout.nbody - 1):
        body_id = body_row + 1
        expected_pos = np.asarray(reference.xpos[body_id], dtype=np.float64)
        if body_row in name_by_row:
            expected_pos = expected_pos + _ref_offset(name_by_row[body_row])
        np.testing.assert_allclose(
            backend._body_pos_cache[env, body_row],
            expected_pos,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _canonical(backend._body_quat_cache[env, body_row]),
            _canonical(np.asarray(reference.xquat[body_id])),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend._body_ang_vel_cache[env, body_row],
            np.asarray(reference.cvel[body_id])[:3],
            rtol=1e-4,
            atol=1e-5,
        )
    return reference


def test_portable_composite_root_state_matches_free_root_and_mujoco(tmp_path: Path):
    backend = _backend(_composite_scene(tmp_path))
    try:
        backend.model
        layout = backend.get_scene_layout()
        assert (layout.nq, layout.nv) == (23, 20)
        composite = layout.get_entity("composite_hand")
        assert tuple((joint.name, joint.kind) for joint in composite.root_joints) == (
            ("px", "slide"),
            ("py", "slide"),
            ("pz", "slide"),
            ("palm_ball", "ball"),
        )
        assert composite.root_qpos_indices == tuple(range(7))
        assert composite.root_qvel_indices == tuple(range(6))
        assert composite.joints[0].qpos_indices == (7,)
        free = layout.get_entity("free_hand")
        assert free.root_joints == ()
        assert free.root_qpos_indices == tuple(range(8, 15))
        assert free.root_qvel_indices == tuple(range(7, 13))
        assert backend._metadata.ball_joint_names == ("composite_hand/palm_ball",)
        assert backend._metadata.ball_joint_addresses == ((3, 3),)

        # Public defaults keep the MJCF wxyz layout with an identity root
        # orientation while the native model is seeded raw xyzw per world.
        # Composite root columns are mount-origin-relative source values: the
        # declared initial position locates only the mount origin on the chain
        # root body, so the public default root pose is zero-relative while
        # the palm spawns at the mount origin in world space.
        default_state = backend.get_entity_state("composite_hand")
        np.testing.assert_allclose(
            default_state["root_pose"][0], (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), atol=1e-7
        )
        np.testing.assert_allclose(default_state["root_velocity"], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            backend.get_entity_default_state("composite_hand")["root_pose"][0],
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            atol=1e-7,
        )
        default_palm_row = layout.get_body_ids(["palm"], entity="composite_hand")[0] - 1
        np.testing.assert_allclose(
            backend._body_pos_cache[0, default_palm_row], (0.0, 0.0, 1.0), atol=1e-6
        )
        raw_default = np.asarray(backend.model.joint_q.numpy(), dtype=np.float32).reshape(
            backend.num_envs, layout.nq
        )
        for env in range(backend.num_envs):
            np.testing.assert_allclose(raw_default[env, 0:3], (0.0, 0.0, 0.0), atol=1e-7)
            np.testing.assert_allclose(raw_default[env, 3:7], (0.0, 0.0, 0.0, 1.0), atol=1e-7)

        # Asymmetric-quaternion pose + displacement writes round-trip
        # value-for-value on both floating root forms.
        qpos, qvel = _composite_payload(layout.nq, layout.nv)
        backend._host_set_state(np.asarray((1, 3), dtype=np.intp), qpos, qvel)
        for env, row in ((1, 0), (3, 1)):
            np.testing.assert_allclose(backend._qpos_cache[env], qpos[row], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(backend._qvel_cache[env], qvel[row], rtol=1e-6, atol=1e-6)

        # The free hand carries the composite root state translated by the
        # composite mount origin, so both palms reach identical world poses
        # and angular velocities.
        composite_palm_row = layout.get_body_ids(["palm"], entity="composite_hand")[0] - 1
        free_palm_row = layout.get_body_ids(["palm"], entity="free_hand")[0] - 1
        for env in (1, 3):
            np.testing.assert_allclose(
                backend._body_pos_cache[env, composite_palm_row],
                backend._body_pos_cache[env, free_palm_row],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                _canonical(backend._body_quat_cache[env, composite_palm_row]),
                _canonical(backend._body_quat_cache[env, free_palm_row]),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                backend._body_ang_vel_cache[env, composite_palm_row],
                backend._body_ang_vel_cache[env, free_palm_row],
                rtol=1e-5,
                atol=1e-6,
            )

        # CPU MuJoCo mj_forward on the composed playback file pins the
        # public-to-native translation of every body's kinematics.
        reference = _assert_composite_scene_matches_mujoco(
            backend, layout, composite, 1, qpos[0], qvel[0]
        )
        # The public root velocity is the world-frame mount-origin linear
        # velocity plus the palm-body-frame angular velocity.
        palm_world_omega = np.asarray(reference.cvel[composite_palm_row + 1])[:3]
        np.testing.assert_allclose(
            backend._qvel_cache[1, 3:6],
            inverse_rotate_vector(qpos[0, 3:7][None, :], palm_world_omega[None, :])[0],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend._qvel_cache[1, 0:3],
            backend._body_lin_vel_cache[1, composite_palm_row],
            rtol=1e-5,
            atol=1e-6,
        )

        # Entity root patches address the composite root with free-joint
        # semantics: link-origin/wxyz pose and world-frame velocity.
        patch_pose = np.concatenate(((-0.2, 0.3, 1.1), _COMPOSITE_QUAT_A)).astype(
            np.float32
        ).reshape(1, -1)
        patch_velocity = np.array([[0.05, -0.04, 0.02, 0.2, -0.1, 0.3]], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                (2,),
                (
                    EntityStatePatch(
                        "composite_hand",
                        root_pose=patch_pose,
                        root_velocity=patch_velocity,
                    ),
                ),
            )
        )
        patched = backend.get_entity_state("composite_hand")
        np.testing.assert_allclose(patched["root_pose"][2], patch_pose[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            patched["root_velocity"][2], patch_velocity[0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(patched["root_pose"][1], qpos[0, 0:7], rtol=1e-6, atol=1e-6)
        free_state = backend.get_entity_state("free_hand")
        np.testing.assert_allclose(free_state["root_pose"][1], qpos[0, 8:15], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            free_state["joint_positions"][1], qpos[0, 15:16], rtol=1e-6, atol=1e-6
        )

        backend._host_step(
            np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32), nsteps=3
        )
        assert np.isfinite(backend.get_physics_state()).all()
    finally:
        backend.close()


def test_portable_composite_root_tensor_lifecycle_round_trip(tmp_path: Path):
    torch = pytest.importorskip("torch")

    backend = _backend(_composite_scene(tmp_path))
    ball_backend = _backend(_ball_scene(tmp_path))
    try:
        backend.model
        ball_backend.model
        device = torch.device(backend._device)
        layout = backend.get_scene_layout()

        views0 = backend.get_state_views()
        # Composite root defaults publish zero-relative source columns with an
        # identity wxyz orientation on the tensor plane (the declared position
        # locates only the mount origin, not the public root state).
        np.testing.assert_allclose(
            views0["qpos"][:, 0:7].cpu().numpy(),
            np.tile((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), (5, 1)),
            atol=1e-6,
        )

        qpos, qvel = _composite_payload(layout.nq, layout.nv)
        rows = torch.tensor([1, 3], dtype=torch.int64, device=device)
        backend.set_state_tensor(
            rows,
            torch.as_tensor(qpos, device=device),
            torch.as_tensor(qvel, device=device),
        )
        views1 = backend.get_state_views()
        torch.testing.assert_close(views1["qpos"][rows], torch.as_tensor(qpos, device=device))
        torch.testing.assert_close(views1["qvel"][rows], torch.as_tensor(qvel, device=device))
        torch.testing.assert_close(views1["qpos"][[0, 2, 4]], views0["qpos"][[0, 2, 4]])
        torch.testing.assert_close(views1["qvel"][[0, 2, 4]], views0["qvel"][[0, 2, 4]])

        # The host plane agrees with the tensor plane, and the committed raw
        # state reproduces the public kinematics under CPU MuJoCo mj_forward.
        backend._refresh_host_cache(sensor_dt=None)
        np.testing.assert_allclose(
            backend._qpos_cache, views1["qpos"].cpu().numpy(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            backend._qvel_cache, views1["qvel"].cpu().numpy(), rtol=1e-6, atol=1e-6
        )
        _assert_composite_scene_matches_mujoco(
            backend, layout, layout.get_entity("composite_hand"), 1, qpos[0], qvel[0]
        )

        backend.step_tensor(
            torch.zeros((5, backend.num_actuators), dtype=torch.float32, device=device), nsteps=2
        )
        views2 = backend.get_state_views()
        assert torch.isfinite(views2["qpos"]).all()
        assert torch.isfinite(views2["qvel"]).all()

        # Free-root entities with non-root ball joints round-trip through the
        # same portable tensor paths.
        ball_layout = ball_backend.get_scene_layout()
        ball_qpos, ball_qvel = _ball_payload(ball_layout.nq, ball_layout.nv)
        ball_backend.set_state_tensor(
            rows,
            torch.as_tensor(ball_qpos, device=device),
            torch.as_tensor(ball_qvel, device=device),
        )
        ball_views = ball_backend.get_state_views()
        torch.testing.assert_close(
            ball_views["qpos"][rows], torch.as_tensor(ball_qpos, device=device)
        )
        torch.testing.assert_close(
            ball_views["qvel"][rows], torch.as_tensor(ball_qvel, device=device)
        )
    finally:
        backend.close()
        ball_backend.close()


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


def _floating_actuated_scene(tmp_path: Path) -> SceneCfg:
    """Two floating actuated entities under gravity for tensor acceptance."""
    gravity = "0 0 -9.81"
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "obj_a",
                _object(
                    tmp_path,
                    "obj_a",
                    radius=0.1,
                    mass=0.5,
                    com=(0.01, 0.0, 0.0),
                    gravity=gravity,
                    actuated=True,
                ),
                initial_state=EntityInitialState((0.0, 0.0, 2.0)),
            ),
            SceneEntitySpec(
                "obj_b",
                _object(
                    tmp_path,
                    "obj_b",
                    radius=0.15,
                    mass=1.5,
                    com=(0.03, 0.0, 0.0),
                    gravity=gravity,
                    actuated=True,
                    diaginertia=0.2,
                ),
                initial_state=EntityInitialState((1.0, 0.0, 3.0)),
            ),
        ),
    )


def _quat_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    scalar = np.cos(angle / 2.0)
    vector = np.sin(angle / 2.0) * axis_array
    return np.concatenate(([scalar], vector)).astype(np.float32)


def test_portable_multi_entity_tensor_lifecycle_matches_per_entity_references(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from unisim.dr.types import TensorResetRandomizationPayload

    scene = _floating_actuated_scene(tmp_path)
    backend = _backend(scene)
    ref_a = _backend(SceneCfg(model_file=str(tmp_path / "obj_a.xml")))
    ref_b = _backend(SceneCfg(model_file=str(tmp_path / "obj_b.xml")))
    try:
        device = torch.device(backend._device)
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.execution.value == "device_resident"
        assert capabilities.state_views and capabilities.stepping
        assert capabilities.selected_reset and capabilities.mask_selected_reset
        # Device reset randomization and packed reset stay closed on portable
        # scenes; the training-scale follow-up owns that family.
        assert not capabilities.reset_randomization
        assert not capabilities.device_reset_randomization
        assert not capabilities.packed_reset
        assert capabilities.packed_reset_fields == frozenset()

        layout = backend.get_scene_layout()
        assert backend.num_actuators == 2
        entities = {name: layout.get_entity(name) for name in ("obj_a", "obj_b")}
        qcols = {name: list(entity.qpos_indices) for name, entity in entities.items()}
        vcols = {name: list(entity.qvel_indices) for name, entity in entities.items()}
        acols = {name: list(entity.actuator_indices) for name, entity in entities.items()}
        nq, nv = layout.nq, layout.nv

        views0 = backend.get_state_views()
        qpos0 = views0["qpos"].clone()
        qvel0 = views0["qvel"].clone()
        assert qpos0.shape == (5, nq) and qvel0.shape == (5, nv)
        np.testing.assert_allclose(
            qpos0[:, qcols["obj_a"][:3]].cpu().numpy(),
            np.tile((0.0, 0.0, 2.0), (5, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            qpos0[:, qcols["obj_b"][:3]].cpu().numpy(),
            np.tile((1.0, 0.0, 3.0), (5, 1)),
            atol=1e-6,
        )

        rows = torch.tensor([1, 3], dtype=torch.int64, device=device)
        per_entity_qpos = {
            "obj_a": np.array(
                [
                    [0.5, 0.2, 1.5, *_quat_wxyz((0, 1, 0), np.pi / 2), 0.3],
                    [-0.3, 0.7, 2.5, *_quat_wxyz((1, 0, 0), np.pi / 4), -0.5],
                ],
                dtype=np.float32,
            ),
            "obj_b": np.array(
                [
                    [1.5, -0.4, 3.5, *_quat_wxyz((0, 0, 1), np.pi / 3), 0.2],
                    [0.8, 0.9, 2.8, *_quat_wxyz((1, 1, 0), np.pi / 6), 0.6],
                ],
                dtype=np.float32,
            ),
        }
        per_entity_qvel = {
            "obj_a": np.array(
                [
                    [0.10, -0.20, 0.30, 0.50, -0.30, 0.20, 0.70],
                    [-0.40, 0.10, 0.20, 0.10, 0.40, -0.60, -0.80],
                ],
                dtype=np.float32,
            ),
            "obj_b": np.array(
                [
                    [0.05, 0.25, -0.15, -0.20, 0.35, 0.45, -0.30],
                    [0.30, -0.05, 0.40, 0.60, 0.15, -0.25, 0.50],
                ],
                dtype=np.float32,
            ),
        }
        payload_qpos = np.zeros((2, nq), dtype=np.float32)
        payload_qvel = np.zeros((2, nv), dtype=np.float32)
        for name in ("obj_a", "obj_b"):
            payload_qpos[:, qcols[name]] = per_entity_qpos[name]
            payload_qvel[:, vcols[name]] = per_entity_qvel[name]

        # A randomization payload fails closed on the portable device plane.
        with pytest.raises(NotImplementedError, match="device terms: kp"):
            backend.set_state_tensor(
                rows,
                torch.as_tensor(payload_qpos, device=device),
                torch.as_tensor(payload_qvel, device=device),
                TensorResetRandomizationPayload(
                    kp=torch.ones((2, backend.num_actuators), dtype=torch.float32, device=device)
                ),
            )

        result = backend.set_state_tensor(
            rows,
            torch.as_tensor(payload_qpos, device=device),
            torch.as_tensor(payload_qvel, device=device),
        )
        assert result is not None and "tensor_reset_ms" in result["timing"]
        views1 = backend.get_state_views()
        untouched = [0, 2, 4]
        torch.testing.assert_close(
            views1["qpos"][rows], torch.as_tensor(payload_qpos, device=device)
        )
        torch.testing.assert_close(
            views1["qvel"][rows], torch.as_tensor(payload_qvel, device=device)
        )
        torch.testing.assert_close(views1["qpos"][untouched], qpos0[untouched])
        torch.testing.assert_close(views1["qvel"][untouched], qvel0[untouched])

        # The single-entity reference backends start at the MJCF default root
        # pose, so align every row with the portable scene's authored initial
        # state before mirroring the selected reset; otherwise untouched rows
        # legitimately diverge.
        all_rows = np.arange(5, dtype=np.intp)
        for name, ref in (("obj_a", ref_a), ("obj_b", ref_b)):
            ref._host_set_state(
                all_rows,
                qpos0[:, qcols[name]].cpu().numpy(),
                qvel0[:, vcols[name]].cpu().numpy(),
            )
        host_rows = np.array([1, 3], dtype=np.intp)
        ref_a._host_set_state(host_rows, per_entity_qpos["obj_a"], per_entity_qvel["obj_a"])
        ref_b._host_set_state(host_rows, per_entity_qpos["obj_b"], per_entity_qvel["obj_b"])

        ctrl = np.zeros((5, backend.num_actuators), dtype=np.float32)
        ctrl[:, acols["obj_a"]] = 0.6
        ctrl[:, acols["obj_b"]] = -0.4
        backend.step_tensor(torch.as_tensor(ctrl, device=device), nsteps=3)
        ref_a._host_step(ctrl[:, acols["obj_a"]], nsteps=3)
        ref_b._host_step(ctrl[:, acols["obj_b"]], nsteps=3)

        views2 = backend.get_state_views()
        for name, ref in (("obj_a", ref_a), ("obj_b", ref_b)):
            # References read through their host plane on purpose: the legacy
            # single-articulation tensor refresh carries the known #427 root
            # angular-velocity bug, which is out of scope here.
            np.testing.assert_allclose(
                views2["qpos"][:, qcols[name]].cpu().numpy(),
                ref._qpos_cache,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                views2["qvel"][:, vcols[name]].cpu().numpy(),
                ref._qvel_cache,
                rtol=2e-4,
                atol=2e-4,
            )

        # Tensor views and the portable host cache agree after a host refresh.
        backend._refresh_host_cache(sensor_dt=None)
        np.testing.assert_allclose(
            backend._qpos_cache, views2["qpos"].cpu().numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            backend._qvel_cache, views2["qvel"].cpu().numpy(), rtol=1e-5, atol=1e-6
        )

        # Mask variant: return rows {0, 4} to the initial state, preserve the
        # stepped rows, and keep the empty-mask boundary a no-op.
        reset_mask = torch.zeros(5, dtype=torch.bool, device=device)
        reset_mask[[0, 4]] = True
        backend.set_state_tensor_mask(reset_mask, qpos0.contiguous(), qvel0.contiguous())
        views3 = backend.get_state_views()
        torch.testing.assert_close(views3["qpos"][[0, 4]], qpos0[[0, 4]])
        torch.testing.assert_close(views3["qvel"][[0, 4]], qvel0[[0, 4]])
        torch.testing.assert_close(views3["qpos"][[1, 2, 3]], views2["qpos"][[1, 2, 3]])
        torch.testing.assert_close(views3["qvel"][[1, 2, 3]], views2["qvel"][[1, 2, 3]])
        empty = backend.set_state_tensor_mask(
            torch.zeros(5, dtype=torch.bool, device=device),
            views3["qpos"].contiguous(),
            views3["qvel"].contiguous(),
        )
        assert empty == {"timing": {"set_state_tensor_mask_ms": 0.0}}

        # Stepping after mask resets still works (eval_fk mask interop held).
        backend.step_tensor(torch.as_tensor(ctrl, device=device), nsteps=2)
        views4 = backend.get_state_views()
        assert torch.isfinite(views4["qpos"]).all()
        assert torch.isfinite(views4["qvel"]).all()
    finally:
        backend.close()
        ref_a.close()
        ref_b.close()


def test_portable_multi_entity_generalized_force_injection_matches_references(
    tmp_path: Path,
):
    """Portable generalized-force injection matches per-entity references.

    body_f gravity compensation holds obj_a stationary across substeps, a
    joint_f-only PD servo drives obj_b's hinge to its target while the base
    free falls, tensor resets keep the injected forces, and a zero write
    restores gravity — each phase mirroring a non-portable single-entity
    reference backend.
    """
    torch = pytest.importorskip("torch")

    scene = _floating_actuated_scene(tmp_path)
    backend = _backend(scene)
    ref_a = _backend(SceneCfg(model_file=str(tmp_path / "obj_a.xml")))
    ref_b = _backend(SceneCfg(model_file=str(tmp_path / "obj_b.xml")))
    try:
        device = torch.device(backend._device)
        assert backend.get_tensor_capabilities().generalized_force_injection
        layout = backend.get_scene_layout()
        entities = {name: layout.get_entity(name) for name in ("obj_a", "obj_b")}
        qcols = {name: list(entity.qpos_indices) for name, entity in entities.items()}
        vcols = {name: list(entity.qvel_indices) for name, entity in entities.items()}
        brows = {
            name: [body_id - 1 for body_id in entity.body_ids] for name, entity in entities.items()
        }
        nv = layout.nv

        views0 = backend.get_state_views()
        qpos0 = views0["qpos"].clone()
        qvel0 = views0["qvel"].clone()

        # Align the single-entity references with the portable scene's
        # authored initial state (they start at the MJCF default root pose).
        all_rows = np.arange(5, dtype=np.intp)
        for name, ref in (("obj_a", ref_a), ("obj_b", ref_b)):
            ref._host_set_state(
                all_rows,
                qpos0[:, qcols[name]].cpu().numpy(),
                qvel0[:, vcols[name]].cpu().numpy(),
            )

        # Actuated objects fix the base mass at 0.1; the link mass is the
        # authored mass argument.
        masses = {"obj_a": (0.1, 0.5), "obj_b": (0.1, 1.5)}
        body_count = len(backend._body_names)
        ctrl = torch.zeros((5, backend.num_actuators), dtype=torch.float32, device=device)

        # obj_a is gravity-compensated through body_f; obj_b keeps falling.
        body_f = torch.zeros((5, body_count, 6), dtype=torch.float32, device=device)
        for local, mass in enumerate(masses["obj_a"]):
            body_f[:, brows["obj_a"][local], 2] = mass * 9.81
        backend.set_generalized_force_tensor(body_f=body_f)
        ref_body_f = torch.zeros((5, 2, 6), dtype=torch.float32, device=device)
        for local, mass in enumerate(masses["obj_a"]):
            ref_body_f[:, local, 2] = mass * 9.81
        ref_a.set_generalized_force_tensor(body_f=ref_body_f)

        backend.step_tensor(ctrl, nsteps=8)
        backend.step_tensor(ctrl, nsteps=8)
        ref_a._host_step(np.zeros((5, 1), dtype=np.float32), nsteps=16)
        # ref_b follows the same 16 substeps so the obj_b trajectories stay
        # substep-aligned with the portable scene.
        ref_b._host_step(np.zeros((5, 1), dtype=np.float32), nsteps=16)
        views1 = backend.get_state_views()
        np.testing.assert_allclose(
            views1["qpos"][:, qcols["obj_a"]].cpu().numpy(),
            qpos0[:, qcols["obj_a"]].cpu().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            views1["qvel"][:, vcols["obj_a"]].cpu().numpy(),
            qvel0[:, vcols["obj_a"]].cpu().numpy(),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            views1["qpos"][:, qcols["obj_a"]].cpu().numpy(),
            ref_a._qpos_cache,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            views1["qvel"][:, vcols["obj_a"]].cpu().numpy(),
            ref_a._qvel_cache,
            rtol=1e-5,
            atol=1e-5,
        )

        # A hinge-only joint_f PD servo on the free-falling obj_b: the base
        # is weightless, so the hinge has no gravity-torque steady-state
        # error. The reference mirrors the same controller.
        kp, kd, target = 8.0, 0.3, 0.6
        hinge_qd = vcols["obj_b"][6]
        hinge_q = qcols["obj_b"][7]
        ref_nv = ref_b.get_public_state_widths().nv
        for _ in range(2000):
            views = backend.get_state_views()
            joint_f = torch.zeros((5, nv), dtype=torch.float32, device=device)
            joint_f[:, hinge_qd] = (
                kp * (target - views["qpos"][:, hinge_q]) - kd * views["qvel"][:, hinge_qd]
            )
            backend.set_generalized_force_tensor(joint_f=joint_f)
            backend.step_tensor(ctrl)
            ref_views = ref_b.get_state_views()
            ref_joint_f = torch.zeros((5, ref_nv), dtype=torch.float32, device=device)
            ref_joint_f[:, 6] = (
                kp * (target - ref_views["qpos"][:, 7]) - kd * ref_views["qvel"][:, 6]
            )
            ref_b.set_generalized_force_tensor(joint_f=ref_joint_f)
            ref_b.step_tensor(torch.zeros((5, 1), dtype=torch.float32, device=device))
        views2 = backend.get_state_views()
        ref_views2 = ref_b.get_state_views()
        # The host cache only refreshes on demand; the PD loop stepped ref_b
        # through the tensor plane, so sync it before reading host caches.
        ref_b._refresh_host_cache(sensor_dt=None)
        # The reference's tensor qvel root angular columns carry the known
        # #427 bug, so compare the hinge columns live and everything else
        # through the reference host cache.
        np.testing.assert_allclose(
            views2["qpos"][:, hinge_q].cpu().numpy(),
            ref_views2["qpos"][:, 7].cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(ref_b._qpos_cache[:, 7], target, atol=0.02)
        np.testing.assert_allclose(
            views2["qvel"][:, hinge_qd].cpu().numpy(),
            ref_b._qvel_cache[:, 6],
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            views2["qpos"][:, qcols["obj_b"]].cpu().numpy(),
            ref_b._qpos_cache,
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            views2["qvel"][:, vcols["obj_b"]].cpu().numpy(),
            ref_b._qvel_cache,
            rtol=2e-3,
            atol=2e-3,
        )
        # obj_a stayed held while joint_f drove obj_b.
        np.testing.assert_allclose(
            views2["qpos"][:, qcols["obj_a"]].cpu().numpy(),
            qpos0[:, qcols["obj_a"]].cpu().numpy(),
            atol=1e-6,
        )

        # Freeze the residual PD torque so the remaining phases evolve both
        # obj_b copies as identical pure free falls (also covers the joint_f
        # clearing path).
        backend.set_generalized_force_tensor(
            joint_f=torch.zeros((5, nv), dtype=torch.float32, device=device)
        )
        ref_b.set_generalized_force_tensor(
            joint_f=torch.zeros((5, ref_nv), dtype=torch.float32, device=device)
        )

        # Portable tensor resets do not clear injected forces.
        rows = torch.arange(5, dtype=torch.int64, device=device)
        backend.set_state_tensor(rows, qpos0.contiguous(), qvel0.contiguous())
        backend.step_tensor(ctrl, nsteps=8)
        ref_b._host_set_state(
            all_rows,
            qpos0[:, qcols["obj_b"]].cpu().numpy(),
            qvel0[:, vcols["obj_b"]].cpu().numpy(),
        )
        ref_b._host_step(np.zeros((5, 1), dtype=np.float32), nsteps=8)
        views3 = backend.get_state_views()
        np.testing.assert_allclose(
            views3["qpos"][:, qcols["obj_a"]].cpu().numpy(),
            qpos0[:, qcols["obj_a"]].cpu().numpy(),
            atol=1e-6,
        )

        # A zero body_f write clears the channel: obj_a falls again, in sync
        # with the reference.
        backend.set_generalized_force_tensor(body_f=torch.zeros_like(body_f))
        ref_a.set_generalized_force_tensor(body_f=torch.zeros_like(ref_body_f))
        ref_a._host_set_state(
            all_rows,
            qpos0[:, qcols["obj_a"]].cpu().numpy(),
            qvel0[:, vcols["obj_a"]].cpu().numpy(),
        )
        backend.step_tensor(ctrl, nsteps=8)
        ref_a._host_step(np.zeros((5, 1), dtype=np.float32), nsteps=8)
        ref_b._host_step(np.zeros((5, 1), dtype=np.float32), nsteps=8)
        views4 = backend.get_state_views()
        fall = views4["qvel"][:, vcols["obj_a"][2]].cpu().numpy()
        assert bool((fall < -0.05).all())
        np.testing.assert_allclose(
            views4["qvel"][:, vcols["obj_a"]].cpu().numpy(),
            ref_a._qvel_cache,
            rtol=2e-4,
            atol=2e-4,
        )
    finally:
        backend.close()
        ref_a.close()
        ref_b.close()


def test_portable_multi_entity_tracked_body_views_match_references(tmp_path: Path):
    """Tracked-body sensor views are live and match per-entity references."""
    torch = pytest.importorskip("torch")

    scene = _floating_actuated_scene(tmp_path)
    backend = _backend(scene)
    ref_a = _backend(SceneCfg(model_file=str(tmp_path / "obj_a.xml")))
    ref_b = _backend(SceneCfg(model_file=str(tmp_path / "obj_b.xml")))
    try:
        device = torch.device(backend._device)
        layout = backend.get_scene_layout()
        entities = {name: layout.get_entity(name) for name in ("obj_a", "obj_b")}
        qcols = {name: list(entity.qpos_indices) for name, entity in entities.items()}
        vcols = {name: list(entity.qvel_indices) for name, entity in entities.items()}
        acols = {name: list(entity.actuator_indices) for name, entity in entities.items()}

        views0 = backend.get_state_views()
        all_rows = np.arange(5, dtype=np.intp)
        for name, ref in (("obj_a", ref_a), ("obj_b", ref_b)):
            ref._host_set_state(
                all_rows,
                views0["qpos"][:, qcols[name]].cpu().numpy(),
                views0["qvel"][:, vcols[name]].cpu().numpy(),
            )

        pos_view = backend.get_sensor_view("track_pos_w_obj_a/base")
        quat_view = backend.get_sensor_view("track_quat_w_obj_a/link")
        linvel_view = backend.get_sensor_view("track_linvel_w_obj_b/base")
        angvel_view = backend.get_sensor_view("track_angvel_w_obj_b/link")

        ctrl = np.zeros((5, backend.num_actuators), dtype=np.float32)
        ctrl[:, acols["obj_a"]] = 0.6
        ctrl[:, acols["obj_b"]] = -0.4
        backend.step_tensor(torch.as_tensor(ctrl, device=device), nsteps=4)
        ref_a._host_step(ctrl[:, acols["obj_a"]], nsteps=4)
        ref_b._host_step(ctrl[:, acols["obj_b"]], nsteps=4)
        # References read through their host plane on purpose: the legacy
        # single-articulation tensor body views carry the known #427 root
        # angular-velocity bug, which is out of scope here.
        backend._refresh_host_cache(sensor_dt=None)
        ref_a._refresh_host_cache(sensor_dt=None)
        ref_b._refresh_host_cache(sensor_dt=None)
        np.testing.assert_allclose(
            pos_view.cpu().numpy(), ref_a._body_pos_cache[:, 0], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            quat_view.cpu().numpy(), ref_a._body_quat_cache[:, 1], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            linvel_view.cpu().numpy(),
            ref_b._body_lin_vel_cache[:, 0],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            angvel_view.cpu().numpy(),
            ref_b._body_ang_vel_cache[:, 1],
            rtol=1e-4,
            atol=1e-5,
        )

        # The view object is live: stepping updates it in place.
        before = pos_view.cpu().numpy().copy()
        backend.step_tensor(torch.as_tensor(ctrl, device=device), nsteps=2)
        ref_a._host_step(ctrl[:, acols["obj_a"]], nsteps=2)
        ref_b._host_step(ctrl[:, acols["obj_b"]], nsteps=2)
        backend._refresh_host_cache(sensor_dt=None)
        ref_a._refresh_host_cache(sensor_dt=None)
        ref_b._refresh_host_cache(sensor_dt=None)
        after = pos_view.cpu().numpy()
        assert not np.allclose(before, after)
        np.testing.assert_allclose(after, ref_a._body_pos_cache[:, 0], rtol=1e-4, atol=1e-5)

        # Unknown bodies fail closed instead of silently binding a row.
        for bad in ("track_pos_w_obj_a/nope", "track_pos_w_base"):
            with pytest.raises(KeyError, match="unknown Newton tensor tracked body"):
                backend.get_sensor_view(bad)
    finally:
        backend.close()
        ref_a.close()
        ref_b.close()
