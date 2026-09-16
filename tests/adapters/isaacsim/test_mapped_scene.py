"""Preflight and transaction checks independent of the optional Isaac SDK."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacsim.scene_worker import SceneWorkerContext, _rotate, validate_scene_payload
from unisim.backend.subprocess_ipc import protocol
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def _payload():
    robot = EntityLayout(
        "robot", "articulation", "fixed", "base", ("base", "tip"), (0, 1), (None, "base"),
        (JointLayout("passive", "hinge", (0,), (0,), "tip"),), (), (), (),
    )
    obj = EntityLayout(
        "object", "rigid", "floating", "box", ("box",), (2,), (None,), (), (), (), (),
        tuple(range(1, 8)), tuple(range(1, 7)),
    )
    layout = CompiledSceneLayout((robot, obj), 8, 7, 0, 3)
    entries = []
    for entity in layout.entities:
        n = len(entity.joints)
        record = {"joint_names": [joint.name for joint in entity.joints],
                  "body_names": list(entity.body_names), "actuator_names": [],
                  "actuator_joint_names": []}
        for field in ("dof_stiffness", "dof_damping", "dof_effort", "dof_armature",
                      "dof_friction", "dof_lower", "dof_upper"):
            record[field] = [0.0] * n
        entries.append({"name": entity.name, "kind": entity.kind,
                        "root_mode": entity.root_mode, "asset_format": "mjcf",
                        "sources": ["source.xml"], "variants": [record], "assignment": [0, 0]})
    return {"scene_layout": layout.to_dict(), "num_envs": 2, "scene_entities": entries,
            "initial_qpos": np.zeros((2, 8)).tolist(),
            "initial_qvel": np.zeros((2, 7)).tolist(),
            "initial_roots": np.zeros((2, 2, 13)).tolist()}


def test_passive_joint_has_state_but_no_control_and_unbounded_limits_are_valid():
    payload = _payload()
    record = payload["scene_entities"][0]["variants"][0]
    record["dof_lower"], record["dof_upper"] = [-np.inf], [np.inf]
    layout = validate_scene_payload(protocol, payload)
    assert layout.nv == 7 and layout.nu == 0


@pytest.mark.parametrize("bad", ["format", "assignment", "drive", "layout"])
def test_unimplemented_or_inconsistent_requests_fail_before_kit(bad):
    payload = _payload()
    entity = payload["scene_entities"][0]
    if bad == "format":
        entity["asset_format"] = "urdf"
    elif bad == "assignment":
        entity["sources"] *= 2
        entity["variants"] *= 2
        entity["assignment"] = [1, 0]
    elif bad == "drive":
        entity["variants"][0]["dof_stiffness"] = [1.0]
    else:
        entity["variants"][0]["joint_names"] = ["wrong"]
    with pytest.raises((NotImplementedError, ValueError)):
        validate_scene_payload(protocol, payload)


def _context():
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = validate_scene_payload(protocol, _payload())
    ctx.num_envs = 2
    ctx.slots = {name: np.zeros(shape, dtype=protocol.slot_dtype(name))
                 for name, shape in protocol.scene_slot_shapes(2, ctx.layout).items()}
    ctx.slots["reset_env_ids"][:] = [1, 0]
    ctx.slots["reset_entity_root_state"][:, :, 3] = 1
    ctx.slots["reset_qpos"][:, 4] = 1
    ctx.slots["reset_root_mask"][1] = 1
    ctx.slots["reset_qpos_mask"][1:] = 1
    ctx.slots["reset_qvel_mask"][1:] = 1
    return ctx


@pytest.mark.parametrize("bad", ["ids", "mask", "owner", "fixed", "quat", "nan", "root_mask"])
def test_entire_reset_is_rejected_before_first_native_write(bad):
    ctx = _context()
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)
    ctx.refresh_state_slots = lambda: None
    if bad == "ids":
        ctx.slots["reset_env_ids"][:] = 1
    elif bad == "mask":
        ctx.slots["reset_qpos_mask"][1] = 2
    elif bad == "owner":
        ctx.slots["reset_qpos_mask"][0] = 1
    elif bad == "fixed":
        ctx.slots["reset_root_mask"][0] = 1
    elif bad == "quat":
        ctx.slots["reset_entity_root_state"][0, 1, 3:7] = 0
    elif bad == "nan":
        ctx.slots["reset_qpos"][0, 0] = np.nan
    else:
        ctx.slots["reset_qvel_mask"][1] = 0
    with pytest.raises(ValueError):
        ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    assert writes == []


def test_unsorted_selected_rows_remain_unsorted_and_detached_at_commit():
    ctx = _context()
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)
    ctx.refresh_state_slots = lambda: None
    original = copy.deepcopy(ctx.slots)
    ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    np.testing.assert_array_equal(writes[0][0], [1, 0])
    for values in writes[0]:
        values[...] = 0
    for name in original:
        np.testing.assert_array_equal(ctx.slots[name], original[name])


def test_world_body_rotation_has_independent_ninety_degree_oracle():
    q = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    world = _rotate(q, np.array([[1., 2., 3.]]))
    np.testing.assert_allclose(world, [[-2, 1, 3]], atol=1e-6)
    np.testing.assert_allclose(_rotate(q, world, inverse=True), [[1, 2, 3]], atol=1e-6)


def test_readback_failure_after_native_reset_is_faulted():
    ctx = _context()
    ctx.faulted = False
    writes = []
    ctx._commit = lambda *args, **kwargs: writes.append(args)

    def broken_readback():
        raise RuntimeError("native state inaccessible after write")

    ctx.refresh_state_slots = broken_readback
    with pytest.raises(RuntimeError, match="inaccessible"):
        ctx.reset_entities({"count": 2, "entity_names": ["object"]})
    assert writes and ctx.faulted


def test_native_joint_commit_maps_reordered_envs_and_preserves_unselected_channel():
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    ctx = _context()
    ctx.device, ctx.sim_dt = "cpu", .01
    ctx.torch = SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value),
                                long=np.int64)
    ctx._tensor = lambda value: value
    writes, resets = [], []
    asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=Tensor([[10], [20]]),
                             joint_vel=Tensor([[1], [2]])),
        write_joint_state_to_sim=lambda p, v, **kw: writes.append((p.copy(), v.copy(), kw)),
        reset=lambda ids: resets.append(ids.copy()), update=lambda dt: None,
    )
    ctx.assets = [asset, asset]
    ctx.maps = [{"envs": np.array([1, 0]), "joints": np.array([0])},
                {"envs": np.array([0, 1]), "joints": np.array([], dtype=int)}]
    p = np.full((1, 8), 999, dtype=np.float32)
    v = np.zeros((1, 7), dtype=np.float32)
    v[0, 0] = 3
    pmask = np.zeros(8, dtype=np.uint8)
    vmask = np.zeros(7, dtype=np.uint8)
    vmask[0] = 1
    ctx._commit(np.array([1]), p, v, np.zeros((1, 2, 13)),
                pmask, vmask, np.zeros((2, 2), dtype=np.uint8))
    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], [[10]])
    np.testing.assert_array_equal(writes[0][1], [[3]])
    np.testing.assert_array_equal(writes[0][2]["env_ids"], [0])
    np.testing.assert_array_equal(resets, [[0]])


def test_initial_control_has_actuator_width_and_must_be_finite():
    payload = _payload()
    payload["initial_ctrl"] = [[], []]
    validate_scene_payload(protocol, payload)
    payload["initial_ctrl"] = [[1], [2]]
    with pytest.raises(ValueError, match="initial_ctrl"):
        validate_scene_payload(protocol, payload)


def test_keyframe_control_uses_control_columns_and_native_rows_not_joint_positions():
    payload = _payload()
    robot = payload["scene_layout"]["entities"][0]
    robot.update(actuator_names=["motor"], actuator_joint_names=["passive"],
                 actuator_indices=[0])
    payload["scene_layout"]["nu"] = 1
    record = payload["scene_entities"][0]["variants"][0]
    record.update(actuator_names=["motor"], actuator_joint_names=["passive"])
    payload["initial_ctrl"] = [[.25], [-.5]]
    layout = validate_scene_payload(protocol, payload)
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = layout
    writes = []
    ctx.assets = [SimpleNamespace(
        data=SimpleNamespace(joint_pos=np.array([[10], [20]])),
        set_joint_position_target=lambda value, **kw: writes.append((value.copy(), kw)),
    ), object()]
    ctx.maps = [{"public_for_native": np.array([1, 0]), "controls": np.array([0])}, {}]
    ctx._tensor = lambda value: value
    ctx._set_control_targets(np.asarray(payload["initial_ctrl"]))
    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0][0], [[-.5], [.25]])
    assert writes[0][1]["joint_ids"] == [0]
    payload["initial_ctrl"] = [[np.nan], [0]]
    with pytest.raises(ValueError, match="initial_ctrl"):
        validate_scene_payload(protocol, payload)
