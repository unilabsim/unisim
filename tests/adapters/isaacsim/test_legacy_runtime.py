"""Old wire and named scenes share the same native runtime after cold loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacsim.scene_worker import SceneWorkerContext
from unisim.backend.isaacsim.worker import _dispatch, _WorkerContext
from unisim.backend.subprocess_ipc import protocol


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, key):
        return _Tensor(self.values[key])


class _Asset:
    def __init__(self, log):
        self.log = log
        root = np.zeros((2, 13), dtype=np.float32)
        # Actual view order is env1 then env0, not public environment order.
        root[:, :3] = [[11, 2, 3], [4, 5, 6]]
        root[:, 3:7] = [[np.sqrt(.5), 0, 0, np.sqrt(.5)], [1, 0, 0, 0]]
        root[:, 7:] = [[.1, .2, .3, -2, 1, 3], [.4, .5, .6, 4, 5, 6]]
        bodies = np.stack((root.copy(), root.copy()), axis=1)
        bodies[:, 0, 2] += 1
        self.data = SimpleNamespace(
            root_link_state_w=_Tensor(root), body_link_state_w=_Tensor(bodies),
            joint_pos=_Tensor([[11, 12], [21, 22]]),
            joint_vel=_Tensor([[1, 2], [3, 4]]),
        )
        self.root_physx_view = SimpleNamespace(prim_paths=[
            "/World/envs/env_1/Robot/base/base", "/World/envs/env_0/Robot/base/base"])

    def set_joint_position_target(self, values, *, joint_ids, env_ids=None):
        self.log.append(("target", values.copy(), list(joint_ids), env_ids))

    def write_data_to_sim(self):
        self.log.append(("upload",))

    def write_root_pose_to_sim(self, values, *, env_ids):
        self.data.root_link_state_w.values[env_ids, :7] = values
        self.data.body_link_state_w.values[env_ids, 1, :7] = values
        self.log.append(("root_pose", env_ids.copy()))

    def write_root_link_velocity_to_sim(self, values, *, env_ids):
        self.data.root_link_state_w.values[env_ids, 7:] = values
        self.data.body_link_state_w.values[env_ids, 1, 7:] = values
        self.log.append(("root_velocity", env_ids.copy()))

    def write_joint_state_to_sim(self, positions, velocities, *, joint_ids, env_ids):
        self.data.joint_pos.values[np.ix_(env_ids, joint_ids)] = positions
        self.data.joint_vel.values[np.ix_(env_ids, joint_ids)] = velocities
        self.log.append(("joint_state", env_ids.copy()))

    def reset(self, env_ids):
        self.log.append(("reset", env_ids.copy()))

    def update(self, dt):
        self.log.append(("update", dt))


@pytest.fixture
def legacy_context():
    log = []
    asset = _Asset(log)
    renderer = SimpleNamespace(
        num_envs=2, sim_dt=.002, device="cpu", robot=asset,
        sim=SimpleNamespace(step=lambda **kwargs: log.append(("physics_step",))),
        torch=SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value),
                              float32=np.float32, long=np.int64),
        env_origins=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"],
        contract_joint_names=["drive", "passive"], contract_body_names=["base", "tip"],
        native_joint_for_contract=np.array([1, 0]), native_body_for_contract=np.array([1, 0]),
        shutdown=lambda: None,
    )
    metadata = {"num_dof": 2, "num_bodies": 2, "dof_names": ["drive", "passive"],
                "body_names": ["base", "tip"], "configuration_report": {"schema_version": 1}}

    def cold_init(payload):
        log.append(("cold_init",))
        return metadata

    renderer.init_sim = cold_init
    ctx = SceneWorkerContext(protocol, renderer)
    try:
        result = ctx.init_sim({"root_body_name": "base", "model_file": "existing.xml"})
        old = {name: np.zeros(shape, dtype=protocol.slot_dtype(name)) for name, shape in
               protocol.slot_shapes(2, 2, 2).items()}
        ctx.slots = ctx.legacy_projection.attach(old)
        ctx.refresh_state_slots()
        yield ctx, old, log, renderer, result
    finally:
        ctx.shutdown()


def test_legacy_cold_adoption_preserves_metadata_and_actual_view_maps(legacy_context):
    ctx, old, log, renderer, metadata = legacy_context
    assert ctx.sim is renderer.sim and ctx.assets == [renderer.robot]
    assert sum(entry[0] == "cold_init" for entry in log) == 1
    assert ctx.get_meta() == metadata and "scene_layout" not in ctx.get_meta()
    assert (ctx.layout.nq, ctx.layout.nv, ctx.layout.nu) == (9, 8, 2)
    assert ctx.layout.entities[0].actuator_joint_names == ("drive", "passive")
    np.testing.assert_allclose(old["root_state"][:, :3], [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_allclose(old["dof_state"][:, :, 0], [[22, 21], [12, 11]])
    # Canonical qvel is body angular velocity; old output remains world angular velocity.
    np.testing.assert_allclose(ctx.slots["qvel"][1, 3:6], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(old["root_state"][1, 10:], [-2, 1, 3], atol=1e-6)
    assert np.shares_memory(ctx.slots["ctrl"], old["ctrl"])


def test_legacy_step_uses_scene_runtime_and_preserves_historical_d_wide_actions(legacy_context):
    ctx, old, log, _, _ = legacy_context
    assert not hasattr(_WorkerContext, "step")
    assert not hasattr(_WorkerContext, "set_state")
    assert not hasattr(_WorkerContext, "refresh_state_slots")
    old["ctrl"][:] = [[.1, .2], [.3, .4]]
    log.clear()
    reply, _ = _dispatch(ctx, protocol, protocol.CMD_STEP, {"nsteps": 2})
    assert reply == protocol.CMD_READY
    target = next(entry for entry in log if entry[0] == "target")
    np.testing.assert_allclose(target[1], [[.3, .4], [.1, .2]])
    assert target[2] == [1, 0]
    assert sum(entry[0] == "physics_step" for entry in log) == 2
    assert sum(entry[0] == "upload" for entry in log) == 2


def test_legacy_reset_routes_one_native_commit_and_converts_nonidentity_root(legacy_context,
                                                                           monkeypatch):
    ctx, old, log, _, _ = legacy_context
    untouched = {name: value[0].copy() for name, value in old.items()}
    old["reset_env_ids"][0] = 1
    old["reset_qpos"][0] = [7, 8, 9, np.sqrt(.5), 0, 0, np.sqrt(.5), .5, .6]
    old["reset_qvel"][0] = [.1, .2, .3, 1, 2, 3, .7, .8]
    commits = []
    real = ctx._commit

    def record(*args, **kwargs):
        commits.append(args[0].copy())
        return real(*args, **kwargs)

    monkeypatch.setattr(ctx, "_commit", record)
    log.clear()
    reply, _ = _dispatch(ctx, protocol, protocol.CMD_SET_STATE, {"count": 1})
    assert reply == protocol.CMD_READY and len(commits) == 1
    np.testing.assert_array_equal(commits[0], [1])
    np.testing.assert_allclose(old["root_state"][1, :3], [7, 8, 9])
    np.testing.assert_allclose(old["root_state"][1, 7:], [.1, .2, .3, -2, 1, 3], atol=1e-6)
    np.testing.assert_allclose(old["dof_state"][1], [[.5, .7], [.6, .8]])
    for name in ("root_state", "dof_state", "body_state", "contact_force", "ctrl"):
        np.testing.assert_array_equal(old[name][0], untouched[name])
    np.testing.assert_allclose(ctx.assets[0].data.root_link_state_w.values[0, :3], [17, 8, 9])
    assert sum(entry[0] == "reset" for entry in log) == 1
    assert not any(entry[0] == "physics_step" for entry in log)


def test_invalid_old_reset_does_not_enter_native_commit(legacy_context):
    ctx, old, log, _, _ = legacy_context
    log.clear()
    old["reset_qpos"][0, 3:7] = 0
    with pytest.raises(ValueError, match="quaternion"):
        ctx.set_state({"count": 1})
    assert not log and not ctx.faulted
    assert ctx.set_state({"count": 0}) == {"timing": {}}
