"""Cold gathers preserve sparse, reordered native state and pending writes."""

from types import SimpleNamespace

import numpy as np

from tests.adapters.isaacgym.test_legacy_adoption import _Tensor
from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.subprocess_ipc import protocol
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def refresh_fixture(count=3, joints=2):
    entity = EntityLayout(
        "robot", "articulation", "floating", "root", ("root", "link"), (1, 3),
        (None, "root"), tuple(
            JointLayout("j" + str(i), "hinge", (7 + i,), (6 + i,), "link")
            for i in range(joints)
        ), (), (), (), tuple(range(7)), tuple(range(6)),
    )
    layout = CompiledSceneLayout((entity,), 7 + joints, 6 + joints, 0, 4)
    rng = np.random.default_rng(48)
    root = rng.normal(size=(count + 3, 13)).astype(np.float32)
    root[:, 3:7] /= np.linalg.norm(root[:, 3:7], axis=1, keepdims=True)
    bodies = np.repeat(root, 3, axis=0)
    dofs = rng.normal(size=(count * joints + 5, 2)).astype(np.float32)
    ctx = SimpleNamespace(
        _root_state=_Tensor(root), _dof_state=_Tensor(dofs), _body_state=_Tensor(bodies),
        _contact_force=_Tensor(rng.normal(size=(len(bodies), 3)).astype(np.float32)),
        _refresh_tensors=lambda: None,
        slots={name: np.zeros(shape, dtype=protocol.slot_dtype(name))
               for name, shape in protocol.scene_slot_shapes(count, layout).items()},
    )
    worker = SceneWorker.__new__(SceneWorker)
    worker.ctx, worker.protocol, worker.layout, worker.num_envs = ctx, protocol, layout, count
    worker.faulted = False
    worker.actor_ids = rng.permutation(count)[:, None]
    worker.body_ids = np.full((count, 4), -1, dtype=np.int64)
    worker.body_ids[:, [1, 3]] = rng.permutation(count * 2).reshape(count, 2)
    worker.body_com = rng.normal(size=(count, 4, 3)) * 0.1
    worker.root_com = rng.normal(size=(count, 1, 3)) * 0.1
    native_dofs = rng.permutation(count * joints).reshape(count, joints)
    worker.records = [[{"dof_ids": row.tolist()}] for row in native_dofs]
    worker.pending_roots = {int(worker.actor_ids[-1, 0]): root[-1].copy()}
    worker.pending_dofs = {int(native_dofs[-1, 0]): np.array([0.7, -0.2])} if joints else {}
    worker.publish_actor_roots_as_body = True
    worker._bind_refresh_indices()
    return worker


def test_refresh_gathers_native_ids_and_preserves_unowned_bodies():
    worker = refresh_fixture()
    worker.refresh()
    ctx, entity = worker.ctx, worker.layout.entities[0]
    for env, records in enumerate(worker.records):
        for joint, native_id in zip(entity.joints, records[0]["dof_ids"]):
            expected = worker.pending_dofs.get(native_id, ctx._dof_state.values[native_id])
            assert ctx.slots["qpos"][env, joint.qpos_indices[0]] == np.float32(expected[0])
            assert ctx.slots["qvel"][env, joint.qvel_indices[0]] == np.float32(expected[1])
        for public_id in (1, 3):
            native_id = worker.body_ids[env, public_id]
            np.testing.assert_array_equal(ctx.slots["contact_force"][env, public_id],
                                          ctx._contact_force.values[native_id])
    np.testing.assert_array_equal(ctx.slots["body_state"][:, [0, 2], 3], 1)
    np.testing.assert_array_equal(ctx.slots["body_state"][:, [0, 2], :3], 0)
    np.testing.assert_array_equal(ctx.slots["body_state"][:, 1],
                                  ctx.slots["entity_root_state"][:, 0])


def test_refresh_has_no_hot_record_or_body_mapping_scan():
    worker = refresh_fixture(joints=0)
    worker.records = None
    worker.body_ids = None
    worker.refresh()
    assert np.isfinite(worker.ctx.slots["body_state"]).all()
