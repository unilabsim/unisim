"""Real mjbatch entity state, reset isolation, identity and playback acceptance."""

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

mujoco = pytest.importorskip("mujoco")


def _file(tmp_path, name, *, fixed=False, mass=1.0, controlled=False, key=False):
    path = tmp_path / f"{name}.xml"
    root = "" if fixed else '<freejoint name="free"/>'
    control = (
        ('<actuator><general name="drive" joint="hinge" dyntype="filter" dynprm=".1"/></actuator>')
        if controlled
        else ""
    )
    keys = (
        ('<keyframe><key name="home" qpos=".4" qvel=".2" ctrl=".3" act=".1"/></keyframe>')
        if key
        else ""
    )
    path.write_text(
        f'<mujoco><option gravity="0 0 0"/><worldbody><body name="base">{root}'
        f'<inertial pos=".03 .01 .02" mass="{mass}" diaginertia=".1 .2 .3"/>'
        '<geom name="root_geom" size=".05" contype="0" conaffinity="0"/>'
        '<body name="link" pos="0 0 .3"><joint name="hinge"/>'
        '<geom name="link_geom" size=".03" mass=".2" contype="0" '
        'conaffinity="0"/></body></body></worldbody>' + control + keys + "</mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path, *, n=5, variants=True, fixed_robot=False, mirror=True, key=False):
    robot = _file(tmp_path, "robot", fixed=fixed_robot, controlled=True, key=key)
    a, b = _file(tmp_path, "a", mass=1.0), _file(tmp_path, "b", mass=3.0)
    table = tmp_path / "table.xml"
    table.write_text(
        '<mujoco><option gravity="0 0 0"/><worldbody><body name="table">'
        '<geom name="table_geom" type="box" size="1 1 .1"/>'
        "</body></worldbody></mujoco>"
    )
    entities = [
        SceneEntitySpec(
            "robot",
            robot,
            root_mode="fixed" if fixed_robot else "floating",
            initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
        ),
        SceneEntitySpec("object", a, initial_state=EntityInitialState(position=(2.0, 0.0, 1.0))),
        SceneEntitySpec(
            "table",
            ModelSourceDescriptor(str(table)),
            kind="rigid",
            root_mode="fixed",
            initial_state=EntityInitialState(position=(0.0, 0.0, -3.0)),
        ),
    ]
    if mirror:
        entities.append(
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
                initial_state=EntityInitialState(position=(3.0, 0.0, 2.0)),
            )
        )
    assignment = np.array([1, 1, 0, 1, 0]) if n == 5 else np.array([1, 0])
    return SceneCfg(
        entity_assets=tuple(entities),
        entity_variant=EntityVariantBinding("object", FixedVariantPlan(assignment, (a, b)))
        if variants
        else None,
        default_keyframe_name="home" if key else None,
    )


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("fixed_robot", [False, True])
def test_real_variants_native_identity_and_entity_scoped_reset(tmp_path, n, fixed_robot):
    scene = _scene(tmp_path, n=n, fixed_robot=fixed_robot)
    backend = create_backend("mujoco", scene, num_envs=n, sim_dt=0.002, np_dtype=np.float64)
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        assert backend.num_actuators == 1
        assert len(layout.get_entity("object").joints) == 1  # Passive state, no extra action.
        object_body = layout.get_entity("object").body_ids[0]
        native_mass = backend._pool.expand("body_mass")[:, object_body]
        np.testing.assert_allclose(
            native_mass, np.array([1.0, 3.0])[scene.entity_variant.plan.assignment]
        )
        native_inertia = backend._pool.expand("body_inertia")[:, object_body]
        np.testing.assert_allclose(native_inertia, np.broadcast_to([0.1, 0.2, 0.3], (n, 3)))
        robot = layout.get_entity("robot")
        robot_body = robot.body_ids[0]
        backend._ctrl_view[:] = 0.8
        backend._act_view[:] = 0.3
        backend._xfrc_view[:, robot_body, :] = 0.15
        backend._pending_xfrc_applied.reshape(n, -1, 6)[:, robot_body, :] = 0.25
        backend._entity_qfrc_view[:] = 0.12
        backend._warm_view[:] = 0.21
        before = backend.get_state()
        robot_before = backend.get_entity_state("robot")
        ids = (n - 1, 0)
        pose = np.tile([4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5], (2, 1))
        velocity = np.tile([0.2, 0.3, 0.4, 0.7, -0.8, 0.9], (2, 1))
        backend.reset_entities(
            SceneResetRequest(
                ids, (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),)
            )
        )
        actual = backend.get_entity_state("object")
        np.testing.assert_allclose(actual["root_pose"][list(ids)], pose)
        np.testing.assert_allclose(actual["root_velocity"][list(ids)], velocity)
        for field in robot_before:
            np.testing.assert_array_equal(
                backend.get_entity_state("robot")[field], robot_before[field]
            )
        np.testing.assert_array_equal(backend._ctrl_view, 0.8)
        np.testing.assert_array_equal(backend._act_view, 0.3)
        robot_dofs = list(robot.root_qvel_indices) + [
            i for j in robot.joints for i in j.qvel_indices
        ]
        np.testing.assert_array_equal(backend._entity_qfrc_view[:, robot_dofs], 0.12)
        np.testing.assert_array_equal(backend._warm_view[:, robot_dofs], 0.21)
        np.testing.assert_array_equal(backend._xfrc_view[:, robot_body, :], 0.15)
        np.testing.assert_array_equal(
            backend._pending_xfrc_applied.reshape(n, -1, 6)[:, robot_body, :], 0.25
        )
        untouched = [i for i in range(n) if i not in ids]
        for field in before:
            np.testing.assert_array_equal(
                backend.get_state()[field][untouched], before[field][untouched]
            )
        # Independent single-model native readback verifies link/world velocity;
        # a public writer+reader using the same wrong mapping would not suffice.
        model = backend.get_playback_model(ids[0])
        data = mujoco.MjData(model)
        data.qpos[:] = backend._qpos_view[ids[0]]
        data.qvel[:] = backend._qvel_view[ids[0]]
        mujoco.mj_forward(model, data)
        native = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, object_body, native, 0)
        np.testing.assert_allclose(native[:3], velocity[0, 3:], atol=1e-12)
        np.testing.assert_allclose(native[3:], velocity[0, :3], atol=1e-12)
        # Pose-only reset preserves world angular velocity under a new orientation.
        other_pose = pose.copy()
        other_pose[:, 3:] = [1.0, 0.0, 0.0, 0.0]
        backend.reset_entities(
            SceneResetRequest(ids, (EntityStatePatch("object", root_pose=other_pose),))
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_velocity"][list(ids)], velocity
        )
    finally:
        backend.close()


def test_joint_only_mirror_and_full_playback_snapshot(tmp_path):
    backend = create_backend(
        "mujoco",
        _scene(tmp_path, n=2, variants=False),
        num_envs=2,
        sim_dt=0.002,
        np_dtype=np.float64,
    )
    try:
        backend.materialize()
        before = backend.get_state()
        backend._ctrl_view[:] = 0.8
        backend._act_view[:] = 0.4
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "object", joint_positions=np.array([[0.5]]), joint_names=("hinge",)
                    ),
                ),
            )
        )
        assert backend.get_entity_state("object")["joint_positions"][1, 0] == 0.5
        np.testing.assert_array_equal(backend._ctrl_view, 0.8)
        np.testing.assert_array_equal(backend._act_view, 0.4)
        np.testing.assert_array_equal(backend.get_state()["qvel"], before["qvel"])
        target = np.array([[7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0]])
        backend.reset_entities(
            SceneResetRequest((1,), (EntityStatePatch("target", root_pose=target),))
        )
        np.testing.assert_allclose(backend.get_entity_state("target")["root_pose"][1], target[0])
        model = backend.get_playback_model(1)
        assert all(
            model.body(name).id >= 0
            for name in ["robot/base", "object/base", "table/table", "target/base"]
        )
        state = backend.get_physics_state()[1]
        from unisim.visualization.render_many import _set_worker_state

        data = mujoco.MjData(model)
        _set_worker_state(model, data, state, None, data.mocap_pos.copy())
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[model.body("target/base").id], target[0, :3])
    finally:
        path = Path(backend.scene_model_file)
        backend.close()
        assert not path.exists()


def test_default_keyframe_real_batch_reset_keeps_per_variant_defaults(tmp_path):
    backend = create_backend(
        "mujoco",
        _scene(tmp_path, n=2, fixed_robot=True, key=True),
        num_envs=2,
        sim_dt=0.002,
        np_dtype=np.float64,
    )
    try:
        backend.materialize()
        np.testing.assert_allclose(backend.get_entity_state("robot")["joint_positions"], 0.4)
        np.testing.assert_allclose(backend._ctrl_view, 0.3)
        np.testing.assert_allclose(backend._act_view, 0.1)
        backend.step(np.ones((2, 1)), nsteps=2)
        other = backend.get_state()
        backend.reset(np.array([1]))
        np.testing.assert_allclose(backend.get_entity_state("robot")["joint_positions"][1], 0.4)
        np.testing.assert_allclose(backend._ctrl_view[1], 0.3)
        np.testing.assert_allclose(backend._act_view[1], 0.1)
        for name in other:
            np.testing.assert_array_equal(backend.get_state()[name][0], other[name][0])
    finally:
        backend.close()


def test_validation_zero_write_and_native_failure_faults_backend(tmp_path):
    backend = create_backend(
        "mujoco", _scene(tmp_path, n=2, variants=False), num_envs=2, sim_dt=0.002
    )
    backend.materialize()
    before = backend.get_state()
    with pytest.raises(ValueError):
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch("object", joint_positions=np.array([[0.8]])),
                    EntityStatePatch(
                        "table", root_pose=np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
                    ),
                ),
            )
        )
    for name in before:
        np.testing.assert_array_equal(backend.get_state()[name], before[name])

    class BrokenForward:
        def forward(self, ids):
            raise RuntimeError("native submission failure")

    pool = backend._pool
    backend._pool = BrokenForward()
    with pytest.raises(RuntimeError, match="native submission"):
        backend.reset_entities(
            SceneResetRequest(
                (0,), (EntityStatePatch("object", joint_positions=np.array([[0.8]])),)
            )
        )
    backend._pool = pool
    for call in [
        backend.get_state,
        lambda: backend.get_entity_state("robot"),
        lambda: backend.step(np.zeros((2, 1))),
        backend.get_physics_state,
        backend.get_base_pos,
        backend.get_base_quat,
        backend.get_base_lin_vel,
        backend.get_base_ang_vel,
        backend.get_dof_pos,
        backend.get_dof_vel,
        lambda: backend.get_body_pos_w(np.array([1])),
        lambda: backend.get_body_quat_w(np.array([1])),
        lambda: backend.get_body_lin_vel_w(np.array([1])),
        lambda: backend.get_body_ang_vel_w(np.array([1])),
        lambda: backend.get_body_pos_b(np.array([1])),
        lambda: backend.get_body_quat_b(np.array([1])),
        lambda: backend.get_body_lin_vel_b(np.array([1])),
        lambda: backend.get_body_ang_vel_b(np.array([1])),
        lambda: backend.get_sensor_data("missing"),
        lambda: backend.get_sensor_data_rows("missing", np.array([0])),
        lambda: backend.get_sensor_data_batch(()),
        backend._bind_sensor_data_reader(()),
        lambda: backend.get_site_jacobian_w(0, np.array([0])),
    ]:
        with pytest.raises(RuntimeError, match="faulted"):
            call()
    backend.close()


def test_batched_rollout_matches_native_single_world_and_mirror_has_no_physics(tmp_path):
    scene = _scene(tmp_path, n=5)
    without = _scene(tmp_path, n=5, mirror=False)
    a = create_backend("mujoco", scene, num_envs=5, sim_dt=0.002, np_dtype=np.float64)
    b = create_backend("mujoco", without, num_envs=5, sim_dt=0.002, np_dtype=np.float64)
    try:
        a.materialize()
        b.materialize()
        # Independent native data/model per selected env, not a second adapter.
        models = [a.get_playback_model(i) for i in range(5)]
        datas = [mujoco.MjData(model) for model in models]
        for i, data in enumerate(datas):
            data.qpos[:] = a._qpos_view[i]
            data.qvel[:] = a._qvel_view[i]
            mujoco.mj_forward(models[i], data)
        ctrl = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        for _ in range(10):
            a.step(ctrl)
            b.step(ctrl)
            for i, data in enumerate(datas):
                data.ctrl[:] = ctrl[i]
                mujoco.mj_step(models[i], data)
        for name in ["robot", "object"]:
            for field, values in a.get_entity_state(name).items():
                np.testing.assert_allclose(
                    values, b.get_entity_state(name)[field], rtol=1e-11, atol=1e-12
                )
        for i, data in enumerate(datas):
            np.testing.assert_allclose(a.get_state()["qpos"][i], data.qpos, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(a.get_state()["qvel"][i], data.qvel, rtol=1e-11, atol=1e-12)
        # Returned values must not alias batch state.
        detached = a.get_entity_state("object")
        detached["root_pose"][:] = 100
        assert not np.all(a.get_entity_state("object")["root_pose"] == 100)
    finally:
        a.close()
        b.close()


def test_portable_sensor_fragment_reaches_public_backend_sensor_view(tmp_path):
    scene = _scene(tmp_path, n=2, variants=False)
    fragment = tmp_path / "sensors.xml"
    fragment.write_text(
        '<mujoco><sensor><contact name="object_table" geom1="object/root_geom" '
        'geom2="table/table_geom" data="force" reduce="netforce"/></sensor></mujoco>',
        encoding="utf-8",
    )
    scene.fragment_files = [str(fragment)]
    backend = create_backend("mujoco", scene, num_envs=2, sim_dt=0.002, np_dtype=np.float64)
    try:
        backend.materialize()
        assert backend.get_sensor_data("object_table").shape == (2, 3)
    finally:
        backend.close()


def test_joint_reset_clears_only_selected_actuator_and_force_state(tmp_path):
    backend = create_backend(
        "mujoco",
        _scene(tmp_path, n=2, variants=False),
        num_envs=2,
        sim_dt=0.002,
        np_dtype=np.float64,
    )
    try:
        backend.materialize()
        robot = backend.get_scene_layout().get_entity("robot")
        backend._ctrl_view[:] = 0.4
        backend._act_view[:] = 0.2
        backend._xfrc_view[:] = 0.1
        backend._pending_xfrc_applied[:] = 0.3
        backend._entity_qfrc_view[:] = 0.5
        backend._warm_view[:] = 0.6
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "robot", joint_positions=np.array([[0.7]]), joint_names=("hinge",)
                    ),
                ),
            )
        )
        np.testing.assert_array_equal(backend._ctrl_view[:, 0], [0.4, 0])
        np.testing.assert_array_equal(backend._act_view[:, 0], [0.2, 0])
        joint_dof = robot.joints[0].qvel_indices[0]
        assert backend._entity_qfrc_view[1, joint_dof] == 0
        assert backend._warm_view[1, joint_dof] == 0
        object_ids = backend.get_scene_layout().get_entity("object").body_ids
        np.testing.assert_array_equal(backend._xfrc_view[:, object_ids, :], 0.1)
        np.testing.assert_array_equal(
            backend._pending_xfrc_applied.reshape(2, -1, 6)[:, object_ids, :], 0.3
        )
        np.testing.assert_array_equal(backend._xfrc_view[0], 0.1)
    finally:
        backend.close()


def test_native_entity_collision_and_world_isolation(tmp_path):
    path = tmp_path / "sphere.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0"/><worldbody><body name="sphere">'
        '<freejoint/><geom name="shape" type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    source = ModelSourceDescriptor(str(path))
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "left",
                source,
                kind="rigid",
                initial_state=EntityInitialState(position=(-0.15, 0.0, 0.0)),
            ),
            SceneEntitySpec(
                "right",
                source,
                kind="rigid",
                initial_state=EntityInitialState(position=(0.15, 0.0, 0.0)),
            ),
        )
    )
    backend = create_backend("mujoco", scene, num_envs=2, sim_dt=0.001, np_dtype=np.float64)
    try:
        backend.materialize()
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (
                    EntityStatePatch(
                        "left", root_velocity=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                    ),
                    EntityStatePatch(
                        "right", root_velocity=np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                    ),
                ),
            )
        )
        before = backend.get_state()
        backend.step(np.empty((2, 0)), nsteps=120)
        # Contact changes both independent entities; the other world remains at rest.
        assert backend.get_entity_state("left")["root_velocity"][0, 0] < 0
        assert backend.get_entity_state("right")["root_velocity"][0, 0] > 0
        np.testing.assert_allclose(backend.get_state()["qpos"][1], before["qpos"][1], atol=1e-12)
        np.testing.assert_allclose(backend.get_state()["qvel"][1], before["qvel"][1], atol=1e-12)
    finally:
        backend.close()


def test_entity_construction_report_distinguishes_source_keys_and_staged_defaults(
    tmp_path, monkeypatch
):
    from unisim.inspection import ImportReport

    scene = _scene(tmp_path, n=5, fixed_robot=True, key=True)
    for descriptor, joint_value in zip(scene.entity_variant.plan.variants, [0.6, 0.9], strict=True):
        path = Path(descriptor.model_file)
        path.write_text(
            path.read_text().replace(
                "</mujoco>",
                '<keyframe><key name="home" '
                f'qpos="9 8 7 .5 .5 .5 .5 {joint_value}" qvel="1 2 3 4 5 6 .7"/>'
                "</keyframe></mujoco>",
            )
        )
    backend = create_backend("mujoco", scene, num_envs=5, sim_dt=0.002, np_dtype=np.float64)
    try:
        report = backend.get_import_report()
        snapshot = report.to_dict()
        assert report.lifecycle == "construction"
        assert ImportReport.from_dict(snapshot).to_dict() == snapshot
        fields = {
            (field.field, field.scope.entity, field.scope.variant): field.to_dict()
            for field in report.fields
            if field.field.startswith("entity.")
        }
        for variant, ids, joint in [("0", [2, 4], 0.6), ("1", [0, 1, 3], 0.9)]:
            pose = fields[("entity.source_root_pose", "object", variant)]
            assert pose["scope"]["env_ids"] == ids
            assert pose["difference"] == "overridden"
            assert pose["requested"] == [0, 0, 0, 1, 0, 0, 0]
            assert pose["effective"] == [2, 0, 1, 1, 0, 0, 0]
            assert [p["kind"] for p in pose["provenance"]] == ["source", "engine_readback"]
            keys = fields[("entity.keyframe_roots", "object", variant)]
            assert keys["requested"]["home"]["root_pose"] == [9, 8, 7, 0.5, 0.5, 0.5, 0.5]
            np.testing.assert_allclose(
                keys["requested"]["home"]["root_velocity"], [1, 2, 3, 6, 4, 5]
            )
            assert keys["effective"]["home"]["root_pose"] == [2, 0, 1, 1, 0, 0, 0]
            assert keys["effective"]["home"]["root_velocity"] == [0] * 6
            initial = fields[("entity.initial_defaults", "object", variant)]
            assert [p["kind"] for p in initial["provenance"]] == ["source", "adapter_setting"]
            np.testing.assert_allclose(initial["effective"]["joint_positions"], [joint])
            mirror = fields[("entity.initial_defaults", "target", variant)]
            assert mirror["effective"]["root_pose"] == [3, 0, 2, 1, 0, 0, 0]
            assert mirror["effective"]["joint_positions"] == []
            assert mirror["effective"]["ctrl"] == mirror["effective"]["act"] == []
        backend.materialize()
        monkeypatch.setattr(
            mujoco.MjModel,
            "from_xml_path",
            lambda *a, **k: pytest.fail("report/reset/step reparsed an asset"),
        )
        backend.reset_entities(
            SceneResetRequest(
                (1,), (EntityStatePatch("object", joint_positions=np.array([[0.8]])),)
            )
        )
        backend.step(np.zeros((5, 1)))
        backend.reset(np.array([3]))
        assert backend.get_import_report().to_dict() == snapshot
    finally:
        backend.close()
