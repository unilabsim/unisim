"""Native CUDA acceptance for entity state/reset using one Warp runtime."""

# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")

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


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    if not warp.get_device().is_cuda:
        pytest.skip("MJWarp entity acceptance requires real CUDA")


def _source(tmp_path, name, *, mass=1.0, fixed=False, controlled=False, key=False):
    root = "" if fixed else '<freejoint name="root"/>'
    actuator = (
        ('<actuator><general name="drive" joint="hinge" dyntype="filter" dynprm=".1"/></actuator>')
        if controlled
        else ""
    )
    keys = (
        '<keyframe><key name="home" qpos=".4" qvel=".2" ctrl=".3" act=".1"/></keyframe>'
        if key
        else ""
    )
    path = tmp_path / f"{name}.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0" integrator="Euler"/>'
        f'<worldbody><body name="base">{root}<inertial pos=".03 .01 .02" '
        f'mass="{mass}" diaginertia=".1 .2 .3"/>'
        '<geom name="base_geom" size=".05" contype="0" conaffinity="0"/>'
        '<body name="link" pos="0 0 .3"><joint name="hinge"/>'
        '<geom name="link_geom" size=".03" mass=".2" contype="0" conaffinity="0"/>'
        '</body></body></worldbody><sensor><framepos name="position" objtype="body" '
        'objname="base"/></sensor>' + actuator + keys + "</mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path, *, mirror=True, fixed=False, key=False, n=5):
    robot = _source(tmp_path, "robot", controlled=True, fixed=fixed, key=key)
    a, b = _source(tmp_path, "a"), _source(tmp_path, "b", mass=3.0)
    table = tmp_path / "table.xml"
    table.write_text(
        '<mujoco><option gravity="0 0 0" integrator="Euler"/><worldbody>'
        '<body name="base"><geom name="shape" type="box" size="1 1 .1"/>'
        '</body></worldbody></mujoco>'
    )
    entities = [
        SceneEntitySpec(
            "robot",
            robot,
            root_mode="fixed" if fixed else "floating",
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
    return SceneCfg(
        entity_assets=tuple(entities),
        default_keyframe_name="home" if key else None,
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0] if n == 5 else [1, 0]), (a, b))
        ),
    )


def _backend(scene, n=5):
    return create_backend("mjwarp", scene, num_envs=n, sim_dt=0.002, nconmax=32, njmax=64)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("n", [2, 5])
def test_native_variant_fields_selected_reset_and_independent_link_velocity(tmp_path, fixed, n):
    scene = _scene(tmp_path, fixed=fixed, n=n)
    backend = _backend(scene, n)
    try:
        layout = backend.get_scene_layout()
        obj = layout.get_entity("object")
        robot = layout.get_entity("robot")
        assert backend.num_actuators == 1 and len(obj.joints) == 1
        oid, rid = obj.body_ids[0], robot.body_ids[0]
        np.testing.assert_allclose(
            backend._device_model.body_mass.numpy()[:, oid],
            np.array([1.0, 3.0])[scene.entity_variant.plan.assignment],
        )
        np.testing.assert_allclose(
            backend._device_model.body_inertia.numpy()[:, oid],
            np.broadcast_to([0.1, 0.2, 0.3], (n, 3)),
            rtol=1e-6,
        )
        channels = backend._entity_persistent_channels()
        for name in channels:
            channels[name][:] = 0.2
        for name, values in channels.items():
            backend._upload(getattr(backend._device_data, name), values)
        backend._xfrc_staging[:] = 0.3
        backend._xfrc_pending = True
        before = backend.get_state()
        robot_before = backend.get_entity_state("robot")
        sensor_before = backend.get_sensor_data("robot/position").copy()
        pose = np.tile([4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5], (2, 1))
        velocity = np.tile([0.2, 0.3, 0.4, 0.7, -0.8, 0.9], (2, 1))
        ids = (n - 1, 0)
        backend.reset_entities(
            SceneResetRequest(
                ids, (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),)
            )
        )
        state = backend.get_entity_state("object")
        np.testing.assert_allclose(state["root_pose"][list(ids)], pose, atol=1e-6)
        np.testing.assert_allclose(state["root_velocity"][list(ids)], velocity, atol=1e-6)
        for name in robot_before:
            np.testing.assert_array_equal(
                backend.get_entity_state("robot")[name], robot_before[name]
            )
        after = backend._entity_persistent_channels()
        for name in ("ctrl", "act"):
            np.testing.assert_array_equal(after[name], channels[name])
        rdofs = list(robot.root_qvel_indices) + [i for j in robot.joints for i in j.qvel_indices]
        for name in ("qfrc_applied", "qacc_warmstart"):
            np.testing.assert_array_equal(after[name][:, rdofs], channels[name][:, rdofs])
        np.testing.assert_array_equal(
            after["xfrc_applied"][:, rid], channels["xfrc_applied"][:, rid]
        )
        np.testing.assert_array_equal(backend._xfrc_staging[:, rid], np.float32(0.3))
        np.testing.assert_array_equal(backend.get_sensor_data("robot/position"), sensor_before)
        untouched = [i for i in range(n) if i not in ids]
        for name in before:
            np.testing.assert_array_equal(
                backend.get_state()[name][untouched], before[name][untouched]
            )
        model = mujoco.MjModel.from_xml_path(backend.get_playback_model(ids[0]))
        data = mujoco.MjData(model)
        data.qpos[:] = backend.get_state()["qpos"][ids[0]]
        data.qvel[:] = backend.get_state()["qvel"][ids[0]]
        mujoco.mj_forward(model, data)
        native = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, oid, native, 0)
        np.testing.assert_allclose(native, np.r_[velocity[0, 3:], velocity[0, :3]], atol=2e-6)
        pose[:, 3:] = [1.0, 0.0, 0.0, 0.0]
        backend.reset_entities(
            SceneResetRequest(ids, (EntityStatePatch("object", root_pose=pose),))
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["root_velocity"][list(ids)], velocity, atol=1e-6
        )
        backend.step(np.zeros((n, 1)))
        assert np.isfinite(backend.get_state()["qpos"]).all()
    finally:
        backend.close()


def test_key_defaults_joint_reset_mirror_and_full_playback(tmp_path):
    backend = _backend(_scene(tmp_path, fixed=True, key=True))
    try:
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"], 0.4, atol=1e-6
        )
        np.testing.assert_allclose(backend._device_data.act.numpy(), 0.1, atol=1e-6)
        backend.reset_entities(
            SceneResetRequest(
                (4, 1),
                (
                    EntityStatePatch(
                        "object", joint_positions=np.array([[0.7], [0.8]]), joint_names=("hinge",)
                    ),
                    EntityStatePatch(
                        "target", root_pose=np.tile([7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0], (2, 1))
                    ),
                ),
            )
        )
        np.testing.assert_allclose(
            backend.get_entity_state("object")["joint_positions"][[4, 1], 0], [0.7, 0.8]
        )
        np.testing.assert_allclose(backend._device_data.ctrl.numpy(), 0.3, atol=1e-6)
        np.testing.assert_allclose(backend._device_data.act.numpy(), 0.1, atol=1e-6)
        from unisim.visualization.render_many import _set_worker_state

        model = mujoco.MjModel.from_xml_path(backend.get_playback_model(4))
        data = mujoco.MjData(model)
        _set_worker_state(model, data, backend.get_physics_state()[4], None, data.mocap_pos.copy())
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[model.body("target/base").id], [7, 8, 9])
        assert all(model.body(name).id >= 0 for name in ("robot/base", "object/base", "table/base"))
        backend.step(np.ones((5, 1)), nsteps=2)
        before = backend.get_state()
        backend.reset(np.array([4]))
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"][4], 0.4, atol=1e-6
        )
        for name in before:
            np.testing.assert_array_equal(backend.get_state()[name][0], before[name][0])
    finally:
        path = Path(backend.scene_model_file)
        backend.close()
        assert not path.exists()


def test_validation_atomicity_and_native_failure_faults_reads(tmp_path, monkeypatch):
    backend = _backend(_scene(tmp_path))
    try:
        before = backend.get_state()
        with pytest.raises(ValueError):
            backend.reset_entities(
                SceneResetRequest(
                    (0,),
                    (
                        EntityStatePatch("object", joint_positions=np.array([[0.6]])),
                        EntityStatePatch("table", root_pose=np.array([[0, 0, 0, 1, 0, 0, 0.0]])),
                    ),
                )
            )
        for name in before:
            np.testing.assert_array_equal(backend.get_state()[name], before[name])

        def broken():
            raise RuntimeError("native forward failure")

        monkeypatch.setattr(backend, "_execute_device_forward", broken)
        with pytest.raises(RuntimeError, match="native forward"):
            backend.reset_entities(
                SceneResetRequest(
                    (0,), (EntityStatePatch("object", joint_positions=np.array([[0.7]])),)
                )
            )
        for call in (
            backend.get_state,
            backend.get_base_pos,
            backend.get_dof_pos,
            lambda: backend.get_entity_state("object"),
            lambda: backend.get_sensor_data("robot/position"),
            lambda: backend.step(np.zeros((5, 1))),
        ):
            with pytest.raises(RuntimeError, match="faulted"):
                call()
    finally:
        backend.close()


def test_gpu_rollout_matches_independent_cpu_and_mirror_is_nonphysical(tmp_path):
    a = _backend(_scene(tmp_path))
    b = _backend(_scene(tmp_path, mirror=False))
    try:
        models = [mujoco.MjModel.from_xml_path(a.get_playback_model(i)) for i in range(5)]
        datas = [mujoco.MjData(model) for model in models]
        for i, data in enumerate(datas):
            data.qpos[:] = a.get_state()["qpos"][i]
            mujoco.mj_forward(models[i], data)
        ctrl = np.arange(1, 6, dtype=np.float32).reshape(5, 1) * 0.1
        for _ in range(5):
            a.step(ctrl)
            b.step(ctrl)
            for i, data in enumerate(datas):
                data.ctrl[:] = ctrl[i]
                mujoco.mj_step(models[i], data)
        for name in ("robot", "object"):
            for field, values in a.get_entity_state(name).items():
                np.testing.assert_allclose(
                    values, b.get_entity_state(name)[field], rtol=1e-5, atol=1e-6
                )
        for i, data in enumerate(datas):
            np.testing.assert_allclose(a.get_state()["qpos"][i], data.qpos, rtol=2e-4, atol=2e-5)
            np.testing.assert_allclose(a.get_state()["qvel"][i], data.qvel, rtol=2e-4, atol=2e-5)
    finally:
        a.close()
        b.close()


def test_independent_single_world_warp_matches_batched_variant_rows(tmp_path):
    import mujoco_warp

    backend = _backend(_scene(tmp_path))
    try:
        oracle = []
        for env in (0, 2):
            model = mujoco.MjModel.from_xml_path(backend.get_playback_model(env))
            native_model = mujoco_warp.put_model(model)
            data = mujoco_warp.make_data(model, nworld=1, nconmax=32, njmax=64)
            data.qpos.assign(backend.get_state()["qpos"][env : env + 1])
            mujoco_warp.forward(native_model, data)
            oracle.append((env, native_model, data))
        controls = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        for _ in range(5):
            backend.step(controls)
            for env, model, data in oracle:
                data.ctrl.assign(controls[env : env + 1])
                mujoco_warp.step(model, data)
        warp.synchronize()
        for env, model, data in oracle:
            np.testing.assert_allclose(
                backend.get_state()["qpos"][env], data.qpos.numpy()[0], rtol=1e-5, atol=2e-6
            )
            np.testing.assert_allclose(
                backend.get_state()["qvel"][env], data.qvel.numpy()[0], rtol=1e-5, atol=2e-6
            )
    finally:
        backend.close()


def test_rigid_contact_zero_actions_and_environment_isolation(tmp_path):
    path = tmp_path / "sphere.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0" integrator="Euler"/><worldbody>'
        '<body name="base"><freejoint/><geom name="shape" type="sphere" '
        'size=".1" mass="1"/></body></worldbody></mujoco>'
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
    backend = _backend(scene, n=2)
    try:
        assert backend.num_actuators == 0
        before = backend.get_state()
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
        backend.step(np.empty((2, 0)), nsteps=80)
        assert backend.get_entity_state("left")["root_velocity"][0, 0] < 0
        assert backend.get_entity_state("right")["root_velocity"][0, 0] > 0
        np.testing.assert_array_equal(backend.get_state()["qpos"][1], before["qpos"][1])
        np.testing.assert_array_equal(backend.get_state()["qvel"][1], before["qvel"][1])
    finally:
        backend.close()


def test_selected_joint_clears_own_actuation_only_and_report_remains_initial(tmp_path):
    backend = _backend(_scene(tmp_path))
    try:
        report = backend.get_import_report().to_dict()
        records = [f for f in report["fields"] if f["field"] == "entity.initial_defaults"]
        assert len(records) == 2
        assert all(f["difference"] == "exact" for f in records)
        assert all(
            [p["kind"] for p in f["provenance"]] == ["adapter_setting", "engine_readback"]
            for f in records
        )
        channels = backend._entity_persistent_channels()
        for values in channels.values():
            values[:] = 0.4
        for name, values in channels.items():
            backend._upload(getattr(backend._device_data, name), values)
        backend._xfrc_staging[:] = 0.3
        backend._xfrc_pending = True
        backend.reset_entities(
            SceneResetRequest(
                (4, 1),
                (
                    EntityStatePatch(
                        "robot", joint_positions=np.array([[0.7], [0.8]]), joint_names=("hinge",)
                    ),
                ),
            )
        )
        after = backend._entity_persistent_channels()
        np.testing.assert_array_equal(
            after["ctrl"][:, 0], np.array([0.4, 0, 0.4, 0.4, 0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            after["act"][:, 0], np.array([0.4, 0, 0.4, 0.4, 0], dtype=np.float32)
        )
        object_ids = backend.get_scene_layout().get_entity("object").body_ids
        np.testing.assert_array_equal(after["xfrc_applied"][:, object_ids], np.float32(0.4))
        np.testing.assert_array_equal(backend._xfrc_staging[:, object_ids], np.float32(0.3))
        backend.step(np.zeros((5, 1)))
        assert backend.get_import_report().to_dict() == report
    finally:
        backend.close()


def test_variant_geometry_identity_fails_portable_layout_validation(tmp_path):
    a = _source(tmp_path, "a")
    b = _source(tmp_path, "b", mass=3.0)
    path = Path(b.model_file)
    path.write_text(path.read_text().replace('name="base_geom" ', ""))
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", a),),
        entity_variant=EntityVariantBinding("object", FixedVariantPlan(np.array([0, 1]), (a, b))),
    )
    with pytest.raises(
        ValueError, match="scene layouts differ in public names, topology, ordering or addresses"
    ):
        _backend(scene, n=2)


def _mesh_variant(tmp_path, name, *, headed):
    spec = mujoco.MjSpec()
    handle = spec.add_mesh(name="handle")
    handle.make_sphere(2)
    body = spec.worldbody.add_body(name="tool", pos=(0, 0, 0))
    body.add_freejoint(name="root")
    body.add_geom(name="handle", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="handle")
    if headed:
        head = spec.add_mesh(name="head")
        head.make_sphere(1)
        body.add_geom(
            name="head", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="head", pos=(0.056, 0, 0)
        )
    spec.compile()
    path = tmp_path / f"{name}.xml"
    spec.to_file(str(path))
    return ModelSourceDescriptor(str(path))


def test_uniform_public_entity_variant_allows_optional_mesh_slot(tmp_path):
    headed = _mesh_variant(tmp_path, "headed", headed=True)
    headless = _mesh_variant(tmp_path, "headless", headed=False)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "object",
                headed,
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.array([0, 1], dtype=np.int32),
                (headed, headless),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    backend = _backend(scene, n=2)
    try:
        headed_model = mujoco.MjModel.from_xml_path(backend.get_playback_model(0))
        headless_model = mujoco.MjModel.from_xml_path(backend.get_playback_model(1))
        assert headless_model.ngeom == headed_model.ngeom - 1
        backend.reset()
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32))
        assert np.isfinite(backend.get_physics_state()).all()
    finally:
        backend.close()
