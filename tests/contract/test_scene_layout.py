"""Public scene layout invariants independent of any physics SDK."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def _two_roots() -> CompiledSceneLayout:
    robot = EntityLayout(
        name="robot",
        kind="articulation",
        root_mode="floating",
        root_body="base",
        body_names=("finger", "base"),
        body_ids=(2, 1),
        body_parent_names=("base", None),
        joints=(JointLayout("hinge", "hinge", (14,), (12,), "finger"),),
        actuator_names=("drive",),
        actuator_joint_names=("hinge",),
        actuator_indices=(0,),
        root_qpos_indices=(0, 2, 4, 6, 8, 10, 12),
        root_qvel_indices=(0, 2, 4, 6, 8, 10),
    )
    object_entity = EntityLayout(
        name="object",
        kind="articulation",
        root_mode="floating",
        root_body="base",
        body_names=("base", "lid"),
        body_ids=(3, 4),
        body_parent_names=(None, "base"),
        joints=(JointLayout("hinge", "hinge", (15,), (13,), "lid"),),
        actuator_names=(),
        actuator_joint_names=(),
        actuator_indices=(),
        root_qpos_indices=(1, 3, 5, 7, 9, 11, 13),
        root_qvel_indices=(1, 3, 5, 7, 9, 11),
    )
    return CompiledSceneLayout((robot, object_entity), nq=16, nv=14, nu=1, nbody=5)


def _ball_scene() -> CompiledSceneLayout:
    entity = EntityLayout(
        name="object",
        kind="articulation",
        root_mode="fixed",
        root_body="base",
        body_names=("base", "lid"),
        body_ids=(0, 1),
        body_parent_names=(None, "base"),
        joints=(
            JointLayout("hinge", "hinge", (4,), (3,), "lid"),
            JointLayout("ball", "ball", (3, 0, 2, 1), (2, 0, 1), "lid"),
        ),
        actuator_names=(),
        actuator_joint_names=(),
        actuator_indices=(),
    )
    return CompiledSceneLayout((entity,), nq=5, nv=4, nu=0, nbody=2)


def _rigid(root_mode: str) -> CompiledSceneLayout:
    entity = EntityLayout(
        name="table",
        kind="rigid",
        root_mode=root_mode,
        root_body="base",
        body_names=("base",),
        body_ids=(0,),
        body_parent_names=(None,),
        joints=(),
        actuator_names=(),
        actuator_joint_names=(),
        actuator_indices=(),
    )
    return CompiledSceneLayout((entity,), nq=0, nv=0, nu=0, nbody=1)


def _pose() -> np.ndarray:
    return np.array([[1, 2, 3, 0.5, 0.5, 0.5, 0.5]], dtype=np.float64)


def test_two_roots_passive_joint_and_noncontiguous_public_addresses() -> None:
    layout = _two_roots()
    assert layout.nq == 16 and layout.nv == 14 and layout.nu == 1
    assert layout.get_body_ids(("robot/base", "object/base")) == (1, 3)
    assert layout.get_body_ids(("base", "finger"), entity="robot") == (1, 2)
    assert layout.get_actuator_ids(("robot/drive",)) == (0,)
    assert layout.get_actuator_ids(("drive",), entity="robot") == (0,)
    assert layout.get_joint_layouts(("object/hinge",))[0].qpos_indices == (15,)
    assert layout.get_entity("object").actuator_names == ()
    assert 0 not in tuple(i for entity in layout.entities for i in entity.body_ids)


@pytest.mark.parametrize("lookup", ["get_body_ids", "get_joint_layouts", "get_actuator_ids"])
def test_unqualified_or_missing_names_never_select_an_arbitrary_entity(lookup) -> None:
    method = getattr(_two_roots(), lookup)
    with pytest.raises(ValueError, match="entity/local_name"):
        method(("base",))
    with pytest.raises(ValueError, match="unknown scene entity"):
        method(("missing/base",))
    with pytest.raises(ValueError, match="unknown"):
        method(("robot/missing",))


@pytest.mark.parametrize("root_mode", ["fixed", "kinematic"])
def test_no_joint_no_actuator_static_layout_is_valid(root_mode) -> None:
    layout = _rigid(root_mode)
    assert layout.nq == layout.nv == layout.nu == 0
    assert layout.entities[0].qpos_indices == ()
    assert layout.entities[0].qvel_indices == ()


def test_json_roundtrip_preserves_full_semantics_and_detaches_wire_containers() -> None:
    layout = _two_roots()
    wire = layout.to_dict()
    restored = CompiledSceneLayout.from_dict(json.loads(json.dumps(wire)))
    layout.require_same_layout(restored)
    assert hash(restored) == hash(layout)
    wire["entities"][0]["body_ids"][0] = 99
    assert layout.entities[0].body_ids == (2, 1)
    assert restored.entities[0].body_ids == (2, 1)


@pytest.mark.parametrize("change", ["kind", "joint_order", "actuation", "body_parent"])
def test_equal_dimensions_do_not_prove_same_semantic_layout(change) -> None:
    original = _ball_scene()
    entity = original.entities[0]
    if change == "kind":
        altered = replace(
            entity, joints=(replace(entity.joints[0], kind="slide"), entity.joints[1])
        )
    elif change == "joint_order":
        altered = replace(entity, joints=tuple(reversed(entity.joints)))
    elif change == "actuation":
        # Same nu with a different actuator target must also fail comparison.
        original = replace(
            original,
            entities=(
                replace(
                    entity,
                    actuator_names=("drive",),
                    actuator_joint_names=("hinge",),
                    actuator_indices=(0,),
                ),
            ),
            nu=1,
        )
        altered = replace(original.entities[0], actuator_joint_names=("ball",))
    else:
        # Preserve all dimensions while moving a joint to another body.
        altered = replace(
            entity, joints=(replace(entity.joints[0], body_name="base"), entity.joints[1])
        )
    changed = replace(original, entities=(altered,))
    assert (original.nq, original.nv, original.nu) == (changed.nq, changed.nv, changed.nu)
    with pytest.raises(ValueError, match="layouts differ"):
        original.require_same_layout(changed)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"root_qpos_indices": (0,)}, "7 columns"),
        ({"root_mode": "fixed"}, "0 columns"),
        ({"body_names": ("base", "base")}, "unique"),
        ({"body_ids": (1, 1)}, "unique"),
        ({"body_parent_names": ("finger", None)}, "cycle"),
        ({"body_parent_names": ("missing", None)}, "entity-local parent"),
        ({"body_parent_names": (None, None)}, "entity-local parent"),
        ({"body_parent_names": ("base", "finger")}, "parent must be None"),
        ({"actuator_joint_names": ("missing",)}, "declared entity-local joints"),
        ({"actuator_indices": ()}, "equal lengths"),
        ({"kind": "rigid"}, "cannot contain"),
    ],
)
def test_entity_topology_and_root_contract_reject_inconsistent_metadata(changes, error) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        replace(_two_roots().entities[0], **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"qpos_indices": (1, 2)},
        {"qvel_indices": (1, 1, 2)},
        {"qpos_indices": (True, 1, 2, 3)},
        {"qvel_indices": (0.0, 1, 2)},
    ],
)
def test_ball_widths_and_indices_are_strict(changes) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(_ball_scene().entities[0].joints[1], **changes)


@pytest.mark.parametrize("dimension", ["nq", "nv", "nu", "nbody"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_layout_dimensions_are_nonnegative_integers(dimension, value) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(_two_roots(), **{dimension: value})


def test_layout_detects_state_holes_overlap_and_out_of_range_body_ids() -> None:
    layout = _two_roots()
    with pytest.raises(ValueError, match="completely cover"):
        replace(layout, nq=17)
    with pytest.raises(ValueError, match="exceeds"):
        replace(layout, nbody=4)
    with pytest.raises(ValueError, match="unique"):
        replace(layout, entities=(layout.entities[0], replace(layout.entities[1], body_ids=(1, 4))))
    overlapping = replace(layout.entities[1], root_qpos_indices=(0, 3, 5, 7, 9, 11, 13))
    with pytest.raises(ValueError, match="unique"):
        replace(layout, entities=(layout.entities[0], overlapping))


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_wire_schema_missing_or_unknown_cannot_be_assumed_compatible(version) -> None:
    wire = _two_roots().to_dict()
    if version is None:
        del wire["schema_version"]
    else:
        wire["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        CompiledSceneLayout.from_dict(wire)


@pytest.mark.parametrize("level", ["scene", "entity", "joint"])
def test_wire_unknown_fields_are_rejected_at_every_level(level) -> None:
    wire = _two_roots().to_dict()
    target = wire if level == "scene" else wire["entities"][0]
    if level == "joint":
        target = target["joints"][0]
    target["native_actor_id"] = 1
    with pytest.raises(ValueError, match="fields must be exactly"):
        CompiledSceneLayout.from_dict(wire)


def test_wire_tampered_mapping_and_non_json_containers_are_rejected() -> None:
    wire = _two_roots().to_dict()
    wire["entities"][1]["joints"][0]["qpos_indices"] = [14]
    with pytest.raises(ValueError, match="unique"):
        CompiledSceneLayout.from_dict(wire)
    wire = _two_roots().to_dict()
    wire["entities"] = tuple(wire["entities"])
    with pytest.raises(TypeError, match="JSON array"):
        CompiledSceneLayout.from_dict(wire)


def test_selected_reset_binds_noncontiguous_addresses_and_preserves_world_velocity() -> None:
    layout = _two_roots()
    patch = EntityStatePatch(
        "object",
        root_pose=_pose(),
        root_velocity=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        joint_positions=np.array([[0.25]]),
    )
    bound = layout.validate_reset(SceneResetRequest((4,), (patch,)), num_envs=5)
    assert bound.env_ids == (4,)
    item = bound.patches[0]
    assert item.entity.root_qpos_indices == (1, 3, 5, 7, 9, 11, 13)
    assert item.joint_qpos_indices == (15,)
    assert item.joint_qvel_indices == (13,)
    assert item.patch.joint_velocities is None
    np.testing.assert_array_equal(item.patch.root_velocity, [[1, 2, 3, 4, 5, 6]])


def test_ball_reset_uses_selector_order_and_distinct_qpos_qvel_widths() -> None:
    layout = _ball_scene()
    patch = EntityStatePatch(
        "object",
        joint_names=("ball", "hinge"),
        joint_positions=np.array([[0.5, 0.5, 0.5, 0.5, 0.3]]),
        joint_velocities=np.array([[1.0, 2.0, 3.0, 4.0]]),
    )
    bound = layout.validate_reset(SceneResetRequest((0,), (patch,)), num_envs=1)
    assert bound.patches[0].joint_qpos_indices == (3, 0, 2, 1, 4)
    assert bound.patches[0].joint_qvel_indices == (2, 0, 1, 3)
    assert tuple(j.name for j in bound.patches[0].joints) == ("ball", "hinge")


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"joint_names": ("missing",), "joint_positions": np.zeros((1, 1))}, "unknown joint"),
        ({"joint_velocities": np.zeros((1, 5))}, "4 columns"),
        ({"joint_positions": np.zeros((1, 4))}, "5 columns"),
        ({"joint_positions": np.zeros((1, 5))}, "unit wxyz"),
    ],
)
def test_reset_joint_widths_selectors_and_ball_orientation_are_checked(fields, error) -> None:
    request = SceneResetRequest((0,), (EntityStatePatch("object", **fields),))
    with pytest.raises(ValueError, match=error):
        _ball_scene().validate_reset(request, num_envs=1)


@pytest.mark.parametrize("field", ["root_pose", "root_velocity"])
def test_fixed_roots_cannot_be_written(field) -> None:
    values = _pose() if field == "root_pose" else np.zeros((1, 6))
    request = SceneResetRequest((0,), (EntityStatePatch("table", **{field: values}),))
    with pytest.raises(ValueError, match="fixed.*root writes"):
        _rigid("fixed").validate_reset(request, num_envs=1)


def test_kinematic_root_pose_is_writable_but_velocity_is_not() -> None:
    layout = _rigid("kinematic")
    request = SceneResetRequest((0,), (EntityStatePatch("table", root_pose=_pose()),))
    assert layout.validate_reset(request, num_envs=1).patches[0].entity.root_qpos_indices == ()
    invalid = SceneResetRequest((0,), (EntityStatePatch("table", root_velocity=np.zeros((1, 6))),))
    with pytest.raises(ValueError, match="kinematic.*root velocity"):
        layout.validate_reset(invalid, num_envs=1)


def test_entire_reset_validates_before_any_adapter_commit_can_run() -> None:
    layout = _two_roots()
    state = np.zeros((2, layout.nq))
    request = SceneResetRequest(
        (1,),
        (
            EntityStatePatch("robot", joint_positions=np.array([[9.0]])),
            EntityStatePatch("object", joint_positions=np.zeros((1, 2))),
        ),
    )

    def adapter_reset() -> None:
        bound = layout.validate_reset(request, num_envs=2)
        for patch in bound.patches:
            state[np.ix_(bound.env_ids, patch.joint_qpos_indices)] = patch.patch.joint_positions

    with pytest.raises(ValueError, match="1 columns"):
        adapter_reset()
    np.testing.assert_array_equal(state, np.zeros((2, layout.nq)))
    with pytest.raises(ValueError, match="environment count"):
        layout.validate_reset(SceneResetRequest((2,), request.patches), num_envs=2)


def test_worker_can_load_layout_by_path_without_importing_unisim() -> None:
    import unisim.scene_layout as module

    source = Path(module.__file__)
    ast.parse(source.read_text(), feature_version=(3, 8))
    script = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('worker_scene_layout', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'unisim' not in sys.modules
empty = module.CompiledSceneLayout((), nq=0, nv=0, nu=0, nbody=1)
assert module.CompiledSceneLayout.from_dict(empty.to_dict()) == empty
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
