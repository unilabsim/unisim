"""Preflight and transaction checks independent of the optional Isaac SDK."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacsim.backend import IsaacSimBackend, IsaacSimWorkerError
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


def _readback_backend(records):
    layout = validate_scene_payload(protocol, _payload())
    robot = replace(layout.entities[0], body_ids=(1, 2))
    obj = replace(layout.entities[1], body_ids=(0,))
    layout = replace(layout, entities=(robot, obj), nbody=4)
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._num_envs = 2
    backend._model_info = object()
    backend._entity_scene = SimpleNamespace(
        layout=layout,
        owner=SimpleNamespace(
            model=SimpleNamespace(
                body_mass=np.array([100, 101, 102, 103], dtype=np.float32),
                body_ipos=np.arange(12, dtype=np.float32).reshape(4, 3) / 7,
            )
        ),
    )
    backend._native_entity_records = records
    return backend


def test_mapped_native_body_mass_is_scattered_to_public_body_order_and_detached():
    records = {
        "robot": {"body_mass": [[10, 11], [20, 21]]},
        "object": {"body_mass": [[30], [40]]},
    }
    backend = _readback_backend(records)
    masses = backend.get_body_mass()
    np.testing.assert_array_equal(masses, [[30, 10, 11, 103], [40, 20, 21, 103]])
    masses[:] = 0
    np.testing.assert_array_equal(records["robot"]["body_mass"], [[10, 11], [20, 21]])


def test_mapped_native_body_ipos_selection_preserves_order_duplicates_and_empty_rows():
    entity_coms = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    public_coms = np.empty((2, 4, 3), dtype=np.float32)
    public_coms[:] = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
    public_coms[:, (1, 2)] = entity_coms[:, :2]
    public_coms[:, 0] = entity_coms[:, 2]
    records = {
        "robot": {"body_com": entity_coms[:, :2].tolist()},
        "object": {"body_com": entity_coms[:, 2:].tolist()},
    }
    backend = _readback_backend(records)
    selected = backend.get_body_ipos(env_ids=[1, 0, 1])
    np.testing.assert_array_equal(selected, public_coms[[1, 0, 1]])
    assert backend.get_body_ipos(env_ids=[]).shape == (0, 4, 3)
    selected[:] = 0
    np.testing.assert_array_equal(
        np.asarray(records["robot"]["body_com"]), entity_coms[:, :2]
    )


def test_mapped_canonical_body_ipos_is_a_detached_compiled_default_table():
    source = np.arange(12, dtype=np.float32).reshape(4, 3) / 7
    backend = _readback_backend({})
    defaults = backend.get_body_ipos()
    np.testing.assert_allclose(defaults, source)
    defaults[:] = -1
    np.testing.assert_allclose(backend._entity_scene.owner.model.body_ipos, source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("body_mass", None),
        ("body_mass", [[10, 11], [20]]),
        ("body_mass", [[10, np.nan], [20, 21]]),
        ("body_com", None),
        ("body_com", [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2]]]),
        ("body_com", [[[0, 0, np.inf], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]]),
    ],
)
def test_mapped_native_property_records_fail_closed(field, value):
    records = {
        "robot": {
            "body_mass": [[10, 11], [20, 21]],
            "body_com": [[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]],
        },
        "object": {"body_mass": [[30], [40]], "body_com": [[[4, 4, 4]], [[5, 5, 5]]]},
    }
    if value is None:
        del records["object"][field]
    else:
        records["object"][field] = value
    backend = _readback_backend(records)
    with pytest.raises(IsaacSimWorkerError, match=f"native {field}.*object"):
        backend.get_body_mass() if field == "body_mass" else backend.get_body_ipos(env_ids=[0])


def test_body_property_readback_requires_mapped_scene_and_keeps_other_properties_unsupported():
    backend = _readback_backend({})
    backend._entity_scene = None
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_mass()
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_ipos()
    with pytest.raises(NotImplementedError, match="explicit entity scene"):
        backend.get_body_ipos(env_ids=[0])
    with pytest.raises(NotImplementedError, match="does not expose geom names"):
        backend.get_geom_names()
    with pytest.raises(NotImplementedError, match="does not expose geom friction"):
        backend.get_geom_friction()
    with pytest.raises(NotImplementedError, match="does not expose geom contact masks"):
        backend.get_geom_contact_masks()


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

        def __getitem__(self, key):
            return Tensor(self.values[key])

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


@pytest.mark.parametrize("rows", [1,8,256])
def test_sparse_joint_commit_downloads_only_selected_rows_and_keeps_native_lifecycle(rows):
    from tests.adapters.isaacsim.reset_transfer_fixture import execute_case

    result = execute_case(SceneWorkerContext._commit, num_envs=1024, num_joints=32, rows=rows)
    assert result["d2h_calls"] == 2
    assert result["d2h_bytes_each"] == [rows*4, rows*4]
    # One selected env-ID upload, one selected joint-ID upload, pos/vel payloads.
    # The other three entities must not cause native-ID construction.
    assert result["h2d_calls"] == 4
    assert result["h2d_bytes"] == rows*16 + 8
    assert [op["operation"] for op in result["operations"]] == [
        "write_joint_state", "reset", "update"]
    assert all(op["entity"] == 1 for op in result["operations"])
    native_rows = list(range(1024-rows,1024))
    assert result["operations"][0]["rows"] == native_rows
    assert result["operations"][0]["joints"] == [28]
    np.testing.assert_array_equal(result["operations"][0]["position"],
                                  (np.arange(rows)+.75)[:,None])
    expected_velocity = -(np.asarray(native_rows)*32+28)-.25
    np.testing.assert_array_equal(result["operations"][0]["velocity"],expected_velocity[:,None])


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


def _controlled_context():
    from dataclasses import replace

    ctx = _context()
    robot = replace(ctx.layout.entities[0], actuator_names=("motor",),
                    actuator_joint_names=("passive",), actuator_indices=(0,))
    ctx.layout = replace(ctx.layout, entities=(robot, ctx.layout.entities[1]), nu=1)
    ctx.slots["ctrl"] = np.array([[.1], [.2]], dtype=np.float32)
    ctx.faulted, ctx.device = False, "cpu"
    ctx.torch = SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value), long=np.int64)
    ctx._tensor = lambda value: value
    writes = []
    ctx.assets = [SimpleNamespace(set_joint_position_target=lambda value, **kwargs:
                                 writes.append(("control", value.copy(), kwargs))), object()]
    ctx.maps = [{"envs": np.array([1, 0]), "controls": np.array([0])}, {}]
    ctx._commit = lambda *args, **kwargs: writes.append(("state",))
    ctx.refresh_state_slots = lambda: None
    return ctx, writes


def test_reset_keyframe_control_override_uses_selected_rows_and_independent_values():
    ctx, writes = _controlled_context()
    ctx.slots["reset_qpos"][0, 0] = 10
    ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                        "control_values": [[-.7]]})
    assert [write[0] for write in writes] == ["state", "control"]
    np.testing.assert_allclose(writes[1][1], [[-.7]])
    np.testing.assert_array_equal(writes[1][2]["env_ids"], [0])
    np.testing.assert_allclose(ctx.slots["ctrl"], [[.1], [-.7]])


@pytest.mark.parametrize("bad", [[[0, 1]], [[np.nan]], [[np.inf]], [[1e100]], [[True]]])
def test_reset_control_override_validation_precedes_native_state(bad):
    ctx, writes = _controlled_context()
    with pytest.raises(ValueError, match="control_values"):
        ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                            "control_values": bad})
    assert writes == [] and not ctx.faulted


def test_reset_control_override_cannot_change_an_unselected_entity():
    ctx, writes = _controlled_context()
    with pytest.raises(ValueError, match="unselected entity"):
        ctx.reset_entities({"count": 1, "entity_names": ["object"],
                            "control_values": [[-.7]]})
    assert writes == []


def test_normal_entity_patch_preserves_existing_hold_target_without_override():
    ctx, writes = _controlled_context()
    before = ctx.slots["ctrl"].copy()
    ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"]})
    assert writes == [("state",)]
    np.testing.assert_array_equal(ctx.slots["ctrl"], before)


def test_reset_native_control_failure_after_state_commit_faults_worker():
    ctx, writes = _controlled_context()

    def fail(*args, **kwargs):
        raise RuntimeError("native target failure")

    ctx.assets[0].set_joint_position_target = fail
    before = ctx.slots["ctrl"].copy()
    with pytest.raises(RuntimeError, match="target failure"):
        ctx.reset_entities({"count": 1, "entity_names": ["robot", "object"],
                            "control_values": [[-.7]]})
    assert writes == [("state",)] and ctx.faulted
    np.testing.assert_array_equal(ctx.slots["ctrl"], before)


def test_entity_prim_components_are_valid_stable_and_injective():
    import re

    from unisim.backend.isaacsim.scene_worker import _entity_prim_component

    public_names = ("robot-arm", "robot_arm", "robot", "robot0", "entity_726f626f74")
    encoded = [_entity_prim_component(name) for name in public_names]
    assert len(set(encoded)) == len(public_names)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in encoded)
    assert encoded == [_entity_prim_component(name) for name in public_names]
    assert [bytes.fromhex(name.removeprefix("entity_")).decode() for name in encoded] == list(
        public_names
    )


def test_native_environment_map_uses_exact_encoded_subtrees():
    from unisim.backend.isaacsim.scene_worker import (
        _entity_prim_component,
        _native_environment_order,
    )

    component = _entity_prim_component("robot-arm")
    roots = [f"/World/envs/env_{index}/{component}" for index in (1, 10)]
    actual = _native_environment_order([roots[1] + "/base", roots[0]], roots)
    np.testing.assert_array_equal(actual, [1, 0])
    wrong_entity = _entity_prim_component("robot_arm")
    for path in (roots[0] + "0/base", roots[0].replace(component, wrong_entity),
                 roots[0].replace("env_1/", "env_100/")):
        with pytest.raises(RuntimeError, match="unowned"):
            _native_environment_order([path, roots[1]], roots)
    with pytest.raises(RuntimeError, match="exactly one"):
        _native_environment_order([roots[0], roots[0] + "/base"], roots)
