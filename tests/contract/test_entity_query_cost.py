"""Entity reads gather only their own host state, without whole-scene snapshots."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tests.contract.test_entity_state import _scene, _snapshots


def query_owner(backend, count=5):
    module = importlib.import_module(f"unisim.backend.{backend}.backend")
    cls = module.MuJoCoBackend if backend == "mujoco" else module.MjwarpBackend
    layout = _scene()
    qpos, qvel, roots = _snapshots(layout)
    indices = np.arange(count) % 5
    qpos, qvel, roots = qpos[indices], qvel[indices], roots[indices]
    mocap_pos = roots[:, 2:3, :3].copy()
    mocap_quat = roots[:, 2:3, 3:7].copy()
    owner = SimpleNamespace(
        get_scene_layout=lambda: layout, _num_envs=count, _np_dtype=qpos.dtype,
        _qpos_view=qpos, _qvel_view=qvel, _qpos_cache=qpos, _qvel_cache=qvel,
        _entity_root_ids=(1, 3, 5), _entity_mocap_ids=(-1, -1, 0),
        _entity_mocap_pos=mocap_pos, _entity_mocap_quat=mocap_quat,
        _mocap_pos=mocap_pos, _mocap_quat=mocap_quat,
    )
    owner._entity_roots = lambda: cls._entity_roots(owner)
    return module, cls, owner, roots


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
@pytest.mark.parametrize("entity_name", ["robot", "object", "target"])
def test_one_entity_query_matches_independent_state_and_never_gathers_other_entities(
    backend, entity_name, monkeypatch
):
    if backend == "mujoco":
        pytest.importorskip("mujoco")
    module, cls, owner, roots = query_owner(backend)
    layout = owner.get_scene_layout()
    entity = layout.get_entity(entity_name)
    calls = []
    snapshot = module.entity_state_snapshot

    def track(item, *args, **kwargs):
        calls.append(item.name)
        return snapshot(item, *args, **kwargs)

    monkeypatch.setattr(module, "entity_state_snapshot", track)
    owner._entity_roots = lambda: pytest.fail("query gathered the whole scene")
    state = cls.get_entity_state(owner, entity_name)
    assert calls == [entity_name]
    index = {"robot": 0, "object": 1, "target": 2}[entity_name]
    np.testing.assert_allclose(state["root_pose"], roots[:, index, :7], atol=1e-6)
    np.testing.assert_allclose(state["root_velocity"], roots[:, index, 7:], atol=1e-6)
    qcols = tuple(i for joint in entity.joints for i in joint.qpos_indices)
    vcols = tuple(i for joint in entity.joints for i in joint.qvel_indices)
    np.testing.assert_array_equal(state["joint_positions"], owner._qpos_view[:, qcols])
    np.testing.assert_array_equal(state["joint_velocities"], owner._qvel_view[:, vcols])
    for values in state.values():
        assert not np.shares_memory(values, owner._qpos_view)
        assert not np.shares_memory(values, owner._qvel_view)
        assert not np.shares_memory(values, owner._mocap_pos)
        values[:] = -99
    assert not np.any(owner._qpos_view == -99)


@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
def test_fixed_entity_query_uses_only_declared_static_root(backend):
    from dataclasses import replace

    if backend == "mujoco":
        pytest.importorskip("mujoco")
    _, cls, owner, _ = query_owner(backend)
    layout = owner.get_scene_layout()
    fixed = replace(layout.entities[2], root_mode="fixed")
    owner.get_scene_layout = lambda: replace(layout, entities=(*layout.entities[:2], fixed))
    model = SimpleNamespace(body_pos=np.zeros((6, 3)), body_quat=np.zeros((6, 4)))
    model.body_pos[5] = [2, 3, 4]
    model.body_quat[5] = [0, 1, 0, 0]
    owner._model = owner._cpu_model = model
    owner._entity_roots = lambda: pytest.fail("query gathered the whole scene")
    state = cls.get_entity_state(owner, "target")
    np.testing.assert_array_equal(state["root_pose"], np.tile([2, 3, 4, 0, 1, 0, 0], (5, 1)))
    np.testing.assert_array_equal(state["root_velocity"], 0)
