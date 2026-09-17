"""Legacy wire adoption uses native IDs and the existing scene execution path."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.isaacgym.worker import _WorkerContext
from unisim.backend.subprocess_ipc import protocol


class _Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, key):
        return self.values[key]


def _kinematics_payload(joints=("active", "passive"), bodies=("base", "link")):
    """Minimal MJCF kinematics tables matching the synthetic adoption layout."""
    return {
        "schema_version": 1,
        "body_names": list(bodies),
        "joint_names": list(joints),
        "body_parent": [-1, 0],
        "body_pos": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        "body_quat": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        "body_joint_kind": [0, 1],
        "body_joint_axis": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "body_joint_column": [-1, 7],
        "free_root": 0,
    }


def _adopt(payload_extra=None):
    joints, bodies = ("active", "passive"), ("base", "link")
    actors = [4, 2]
    body_ids = [(7, 4), (0, 9)]
    dof_ids = [(6, 1), (8, 3)]
    root = np.zeros((10, 13), dtype=np.float32)
    root[:, 6] = 1
    root[actors, 0] = [11, 22]
    root[actors, 7:10] = [1, 2, 3]
    root[actors, 10:13] = [0, 0, 2]
    body = root.copy()
    body[:, 0] = np.arange(10)
    dof = np.stack((np.arange(10), np.arange(10) + 0.5), axis=1).astype(np.float32)
    gym = SimpleNamespace(
        get_actor_asset=lambda env, actor: env,
        get_asset_dof_names=lambda asset: joints,
        get_asset_rigid_body_names=lambda asset: bodies,
        get_actor_index=lambda env, actor, domain: actors[env],
        get_actor_dof_index=lambda env, actor, j, domain: dof_ids[env][j],
        get_actor_rigid_body_index=lambda env, actor, b, domain: body_ids[env][b],
        get_actor_rigid_body_properties=lambda env, actor: [
            SimpleNamespace(com=SimpleNamespace(x=0.1, y=0, z=0)),
            SimpleNamespace(com=SimpleNamespace(x=0.2, y=0, z=0)),
        ],
    )
    ctx = SimpleNamespace(
        protocol=protocol,
        num_envs=2,
        env_handles=[0, 1],
        actor_handles=[6, 8],
        gym=gym,
        gymapi=SimpleNamespace(DOMAIN_SIM=0),
        torch=SimpleNamespace(zeros_like=np.zeros_like),
        _root_state=_Tensor(root),
        _body_state=_Tensor(body),
        _dof_state=_Tensor(dof),
        _contact_force=_Tensor(np.arange(30, dtype=np.float32).reshape(10, 3)),
        _refresh_tensors=lambda: None,
    )
    meta = {"dof_names": list(joints), "body_names": list(bodies), "gravity": [0, 0, -9.81]}
    payload = {"model_file": "raw.xml", "mjcf_kinematics": _kinematics_payload(joints, bodies)}
    payload.update(payload_extra or {})
    runtime = SceneWorker.adopt_initialized_context(
        ctx, meta, payload, protocol.load_legacy_projection()
    )
    legacy = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.slot_shapes(2, 2, 2).items()
    }
    ctx.slots = runtime.projection.attach(legacy)
    return ctx, runtime, legacy


def test_adoption_maps_real_noncontiguous_native_ids_and_preserves_public_com_outputs():
    ctx, runtime, legacy = _adopt()
    runtime.refresh()
    np.testing.assert_array_equal(runtime.actor_ids[:, 0], [4, 2])
    np.testing.assert_array_equal(runtime.control_dofs, [[6, 1], [8, 3]])
    np.testing.assert_array_equal(legacy["root_state"][:, 0], [11, 22])
    np.testing.assert_allclose(legacy["root_state"][:, 7:10], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(legacy["dof_state"][:, :, 0], [[6, 1], [8, 3]])
    np.testing.assert_array_equal(legacy["body_state"][:, :, 0], [[7, 4], [0, 9]])
    np.testing.assert_array_equal(
        legacy["contact_force"], ctx._contact_force.values[[[7, 4], [0, 9]]]
    )
    assert runtime.metadata["dof_names"] == ["active", "passive"]
    assert runtime.layout.nu == 2 and runtime.initial_ctrl is None
    assert set(runtime.pending_roots) == {2, 4}
    assert set(runtime.pending_dofs) == {1, 3, 6, 8}


def test_legacy_reset_codec_delegates_to_same_runtime_and_preserves_control_targets():
    ctx, runtime, legacy = _adopt()
    runtime.refresh()
    ctx.scene_worker = runtime
    legacy["ctrl"][:] = [[0.7, 0.8], [0.9, 1.0]]
    legacy["reset_env_ids"][:] = [1, 0]
    legacy["reset_qpos"][:, :7] = [0, 0, 2, 1, 0, 0, 0]
    legacy["reset_qpos"][:, 7:] = [[0.1, 0.2], [0.3, 0.4]]
    calls = []
    runtime.reset = lambda payload: calls.append(payload) or {"timing": {}}
    _WorkerContext.set_state(ctx, {"count": 2})
    assert len(calls) == 1
    assert calls[0]["entity_names"] == ["legacy_model"]
    np.testing.assert_allclose(calls[0]["control_values"], [[0.9, 1.0], [0.7, 0.8]])
    np.testing.assert_array_equal(ctx.slots["reset_env_ids"], [1, 0])


def test_worker_hot_entrypoints_have_no_second_native_execution_loop():
    source = Path(
        __import__("unisim.backend.isaacgym.worker", fromlist=["__file__"]).__file__
    ).read_text()
    tree = ast.parse(source, feature_version=(3, 8))
    context = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_WorkerContext"
    )
    for name in ("step", "set_state", "refresh_state_slots"):
        method = next(
            node for node in context.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for node in ast.walk(method):
            assert not isinstance(node, (ast.For, ast.While))
            if isinstance(node, ast.Attribute):
                assert node.attr not in (
                    "simulate",
                    "fetch_results",
                    "set_actor_root_state_tensor_indexed",
                    "set_dof_state_tensor_indexed",
                    "refresh_actor_root_state_tensor",
                )


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_SCENE") != "1",
    reason="requires native IsaacGym GPU opt-in",
)
def test_native_post_reset_body_state_matches_mujoco_fk(tmp_path):
    """Issue #141: PhysX link poses stay stale until the first simulate, so the
    legacy path must publish exact MJCF FK rows after materialize/set_state."""
    mujoco = pytest.importorskip("mujoco")
    from unisim import create_backend
    from unisim.scene import SceneCfg

    path = tmp_path / "legacy.xml"
    path.write_text(
        '<mujoco><worldbody><body name="base" pos="0 0 1"><freejoint name="root"/>'
        '<inertial pos=".1 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        '<geom name="base_geom" type="box" size=".08 .08 .08" mass="1"/>'
        '<body name="active_link" pos="0 0 .2"><joint name="active" axis="0 1 0"/>'
        '<inertial pos=".05 0 0" mass=".3" diaginertia=".01 .01 .01"/>'
        '<geom name="active_geom" size=".05" mass=".3"/></body>'
        '<body name="passive_link" pos="0 0 .4"><joint name="passive" axis="0 0 1"/>'
        '<inertial pos="0 .05 0" mass=".3" diaginertia=".01 .01 .01"/>'
        '<geom name="passive_geom" size=".05" mass=".3"/></body>'
        '</body></worldbody><actuator><position name="drive" joint="active" kp="20" kv="2"/>'
        '</actuator><keyframe><key name="home" qpos="0 0 1 1 0 0 0 .15 -.2"/></keyframe></mujoco>'
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    bodies = ["base", "active_link", "passive_link"]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in bodies]

    def reference(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        lin_com, ang = [], []
        for b in body_ids:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
            ang.append(vel6[:3].copy())
            lin_com.append(vel6[3:].copy())
        return (
            np.asarray([data.xpos[b] for b in body_ids]),
            np.asarray([data.xquat[b] for b in body_ids]),
            # The legacy isaacgym output channel is the historical COM velocity.
            np.asarray(lin_com),
            np.asarray(ang),
        )

    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(path)),
        num_envs=2,
        sim_dt=0.002,
        base_name="base",
        worker_timeout_s=90,
    )
    try:
        backend.materialize()
        ids = backend.get_body_ids(bodies)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key)
        pos, quat, _, _ = reference(data.qpos.copy(), np.zeros(model.nv))
        np.testing.assert_allclose(backend.get_body_pos_w(ids), np.tile(pos, (2, 1, 1)), atol=1e-5)
        np.testing.assert_allclose(
            np.abs(np.sum(backend.get_body_quat_w(ids) * quat[None, :, :], axis=-1)),
            1.0,
            atol=1e-5,
        )

        # Arbitrary motion-frame state with nonzero root velocities; the offset
        # root COM distinguishes link-origin from COM velocity conventions.
        qpos = np.array([0.1, -0.05, 1.2, np.sqrt(0.5), 0, np.sqrt(0.5), 0, 0.35, -0.5])
        qvel = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.6, -0.7])
        backend.set_state(
            np.array([0, 1], dtype=np.int32),
            np.tile(qpos, (2, 1)).astype(np.float32),
            np.tile(qvel, (2, 1)).astype(np.float32),
        )
        pos, quat, lin_com, ang = reference(qpos, qvel)
        np.testing.assert_allclose(backend.get_body_pos_w(ids), np.tile(pos, (2, 1, 1)), atol=1e-5)
        np.testing.assert_allclose(
            np.abs(np.sum(backend.get_body_quat_w(ids) * quat[None, :, :], axis=-1)),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            backend.get_body_lin_vel_w(ids), np.tile(lin_com, (2, 1, 1)), atol=1e-5
        )
        np.testing.assert_allclose(
            backend.get_body_ang_vel_w(ids), np.tile(ang, (2, 1, 1)), atol=1e-5
        )

        # The first physics step replaces the overlay with native PhysX state;
        # the published pose must stay continuous with the pre-step FK rows.
        before = backend.get_body_pos_w(ids).copy()
        backend.step(np.zeros((2, 2), dtype=np.float32))
        after = backend.get_body_pos_w(ids)
        np.testing.assert_allclose(after, before, atol=0.02)
    finally:
        backend.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_SCENE") != "1",
    reason="requires native IsaacGym GPU opt-in",
)
@pytest.mark.parametrize("free", [False, True])
def test_native_legacy_shapes_names_passive_targets_and_repeated_reset(tmp_path, free):
    from unisim import create_backend
    from unisim.scene import SceneCfg

    path = tmp_path / "legacy.xml"
    root_joint = '<freejoint name="root"/>' if free else ""
    path.write_text(
        '<mujoco><worldbody><body name="base" pos="0 0 1">'
        + root_joint
        + '<inertial pos=".1 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        '<geom name="base_geom" type="box" size=".08 .08 .08" mass="1"/>'
        '<body name="active_link" pos="0 0 .2"><joint name="active" axis="0 1 0"/>'
        '<geom name="active_geom" size=".05" mass=".3"/></body>'
        '<body name="passive_link" pos="0 0 .4"><joint name="passive" axis="0 0 1"/>'
        '<geom name="passive_geom" size=".05" mass=".3"/></body>'
        '</body></worldbody><actuator><position name="drive" joint="active" kp="20" kv="2"/>'
        '</actuator><keyframe><key name="home" qpos="0 0 1 1 0 0 0 .15 -.2"/></keyframe></mujoco>'
    )
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(path)),
        num_envs=3,
        sim_dt=0.002,
        base_name="base",
        worker_timeout_s=90,
    )
    try:
        backend.materialize()
        assert backend.num_actuators == 2
        assert backend.get_actuator_names() == ("active", "passive")
        assert backend.get_body_ids(["base", "active_link", "passive_link"]).tolist() == [0, 1, 2]
        np.testing.assert_allclose(backend.get_dof_pos(), np.tile([0.15, -0.2], (3, 1)), atol=1e-6)
        controls = np.tile([0.2, 0.9], (3, 1)).astype(np.float32)
        backend.step(controls)
        qpos = np.tile([0, 0, 2, np.sqrt(0.5), 0, 0, np.sqrt(0.5), 0.3, -0.4], (2, 1)).astype(
            np.float32
        )
        qpos[:, 2] = [2.0, 3.0]
        qvel = np.tile([0.2, 0.3, 0.4, 1, 0, 0, 0.1, 0.2], (2, 1)).astype(np.float32)
        backend.set_state(np.array([2, 0], dtype=np.int32), qpos, qvel)
        np.testing.assert_allclose(backend.get_base_pos()[[2, 0]], qpos[:, :3], atol=1e-6)
        np.testing.assert_allclose(backend.get_base_lin_vel()[[2, 0]], qvel[:, :3], atol=1e-6)
        np.testing.assert_allclose(
            backend.get_base_ang_vel()[[2, 0]], [[0, 1, 0], [0, 1, 0]], atol=1e-6
        )
        np.testing.assert_allclose(backend.get_dof_pos()[[2, 0]], qpos[:, 7:], atol=1e-6)
        np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"], controls)
        qpos1 = qpos[[0]].copy()
        qpos1[:, 2] = 4
        backend.set_state(np.array([1]), qpos1, qvel[[0]])
        backend.step(controls)
        assert np.all(backend.get_base_pos()[:, 2] > [2.99, 3.99, 1.99])
        np.testing.assert_array_equal(backend.get_state("ctrl")["ctrl"], controls)
    finally:
        backend.close()
