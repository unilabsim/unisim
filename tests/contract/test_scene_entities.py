"""Entity authoring/reset values and fail-closed adapter boundaries.

These tests make no native multi-entity runtime support claims.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, replace

import numpy as np
import pytest

import unisim
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg


def _physical(name: str = "object", **kwargs) -> SceneEntitySpec:
    return SceneEntitySpec(name, ModelSourceDescriptor(f"{name}.xml"), **kwargs)


def _mirror(name: str = "target", target: str = "object", **kwargs) -> SceneEntitySpec:
    return SceneEntitySpec(
        name, kind="rigid", root_mode="kinematic", collision_enabled=False,
        mirror_of=target, **kwargs,
    )


def _plan(assignment=None) -> FixedVariantPlan:
    return FixedVariantPlan(
        np.array([1, 1, 0, 1, 0]) if assignment is None else assignment,
        (ModelSourceDescriptor("a.xml"), ModelSourceDescriptor("b.xml")),
    )


def _pose(rows: int = 2) -> np.ndarray:
    # A non-identity wxyz quaternion makes accidental xyzw interpretation visible.
    return np.tile([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5], (rows, 1))


def test_entity_catalog_supports_explicit_mirrors_with_independent_poses() -> None:
    object_entity = _physical(kind="rigid")
    target = _mirror(initial_state=EntityInitialState(position=(4.0, 5.0, 6.0)))
    scene = SceneCfg(
        entity_assets=(_physical("robot"), object_entity, target, _mirror("second_target")),
        entity_variant=EntityVariantBinding("object", _plan()),
        entities={"task_selector": object()},
    )
    scene.validate_composition(5)
    assert object_entity.initial_state.position == (0.0, 0.0, 0.0)
    assert target.initial_state.position == (4.0, 5.0, 6.0)
    assert target.source is None
    assert scene.entity_variant.target_entity == "object"
    assert tuple(scene.entities) == ("task_selector",)


def test_legacy_scene_and_unvaried_entity_need_no_variant_consumer() -> None:
    legacy = SceneCfg(model_file="robot.xml", entities={"robot": object()})
    legacy.validate_composition(2)
    SceneCfg(entity_assets=(_physical(),)).validate_composition(2)


@pytest.mark.parametrize("conflict", ["model_file", "fixed_variant_plan"])
def test_entity_sources_cannot_conflict_with_complete_scene_sources(conflict) -> None:
    value = "whole_scene.xml" if conflict == "model_file" else _plan()
    with pytest.raises(ValueError, match="model_file|fixed_variant_plan"):
        SceneCfg(entity_assets=(_physical(),), **{conflict: value})


@pytest.mark.parametrize("name", ["", "robot/base", "a:b", "0robot", "with space"])
def test_entity_names_cannot_ambiguously_encode_qualified_names(name) -> None:
    with pytest.raises(ValueError, match="entity name"):
        _physical(name)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"source": ModelSourceDescriptor("other.xml")}, "inherits its source"),
        ({"mirror_of": "target"}, "mirror itself"),
        ({"collision_enabled": True}, "collision-disabled"),
        ({"root_mode": "floating"}, "kinematic"),
        ({"kind": "articulation"}, "rigid"),
    ],
)
def test_mirror_cannot_acquire_independent_physics_or_identity(changes, error) -> None:
    with pytest.raises(ValueError, match=error):
        replace(_mirror(), **changes)


@pytest.mark.parametrize(
    ("entities", "error"),
    [
        ((_physical(), _physical()), "unique"),
        ((_mirror(),), "physical entity"),
        ((_physical(), _mirror(), _mirror("chained", "target")), "physical entity"),
        ((_physical(), _mirror(asset_format="urdf")), "asset format"),
    ],
)
def test_scene_rejects_ambiguous_entity_relationships(entities, error) -> None:
    with pytest.raises(ValueError, match=error):
        SceneCfg(entity_assets=entities)


@pytest.mark.parametrize("target", ["absent", "target"])
def test_variant_consumer_must_be_a_physical_entity(target) -> None:
    with pytest.raises(ValueError, match="physical entity"):
        SceneCfg(
            entity_assets=(_physical(), _mirror()),
            entity_variant=EntityVariantBinding(target, _plan()),
        )


def test_variant_binding_requires_sources_and_matches_environment_count() -> None:
    binding = EntityVariantBinding("object", _plan())
    with pytest.raises(ValueError):
        SceneCfg(entity_variant=binding)
    scene = SceneCfg(entity_assets=(_physical(),), entity_variant=binding)
    with pytest.raises(ValueError, match="shape"):
        scene.validate_composition(2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_entity_variant_identity_survives_alias_mutation_and_spawn_pickle(dtype) -> None:
    storage = np.array([1, 9, 1, 9, 0, 9, 1, 9, 0, 9], dtype=dtype)
    alias = storage[::2]
    binding = EntityVariantBinding("object", _plan(alias))
    fingerprint = hash(binding.plan)
    storage[::2] = 0
    assert alias.flags.writeable  # Constructing a plan must not freeze caller-owned arrays.
    np.testing.assert_array_equal(binding.plan.assignment, [1, 1, 0, 1, 0])
    assert hash(binding.plan) == fingerprint
    restored = pickle.loads(pickle.dumps(binding))
    assert restored == binding
    for plan in (binding.plan, restored.plan):
        with pytest.raises(ValueError):
            plan.assignment.setflags(write=True)
        with pytest.raises(ValueError):
            plan.assignment[0] = 0


@pytest.mark.parametrize("metadata", ["dtype", "shape"])
@pytest.mark.filterwarnings("ignore:Setting the (dtype|shape) on a NumPy array:DeprecationWarning")
def test_assignment_metadata_changes_cannot_rewrite_plan_identity(metadata) -> None:
    # Intentionally exercise the metadata writes deprecated, but still allowed, by NumPy 2.5.
    expected = np.array([1, 1, 0, 1, 0], dtype=np.int64)
    plan = _plan(expected)
    original_hash = hash(plan)
    catalog = {plan: "original identity"}
    exposed = plan.assignment
    if metadata == "dtype":
        exposed.dtype = np.uint8
        assert exposed.dtype != expected.dtype
    else:
        exposed.shape = (1, 5)
        assert exposed.shape != expected.shape

    plan.validate(num_envs=5)
    for value in (plan, pickle.loads(pickle.dumps(plan))):
        np.testing.assert_array_equal(value.assignment, expected)
        assert value.assignment.dtype == expected.dtype
        assert value.assignment.shape == (5,)
        assert hash(value) == original_hash
        assert catalog[value] == "original identity"
        value.validate(num_envs=5)


def test_equal_assignments_with_different_integer_dtypes_share_hash_and_mapping_key() -> None:
    plans = [_plan(np.array([1, 0, 1], dtype=dtype))
             for dtype in (np.int8, np.uint8, np.int32, np.int64, np.uint64)]
    reference = plans[0]
    mapping = {reference: "same logical plan"}
    for plan in plans[1:]:
        assert plan == reference
        assert hash(plan) == hash(reference)
        assert mapping[plan] == "same logical plan"
    assert len(set(plans)) == 1


def test_plan_dataclass_replacement_and_serialization_preserve_public_contract() -> None:
    plan = _plan()
    new_assignment = np.array([0, 0, 1, 0, 1], dtype=np.int32)
    reassigned = replace(plan, assignment=new_assignment)
    relaid = replace(plan, layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT)
    new_assignment[:] = 0
    np.testing.assert_array_equal(reassigned.assignment, [0, 0, 1, 0, 1])
    assert reassigned != plan
    assert relaid.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
    np.testing.assert_array_equal(relaid.assignment, [1, 1, 0, 1, 0])
    assert relaid.variants == plan.variants
    serialized = asdict(plan)
    assert set(serialized) == {"assignment", "variants", "layout"}
    np.testing.assert_array_equal(serialized["assignment"], [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(plan.assignment, [1, 1, 0, 1, 0])
    assert plan.layout is FixedVariantLayout.SAME_LAYOUT


@pytest.mark.parametrize(
    "assignment",
    [np.array([True, False]), np.array([0.0, 1.0]), np.array([-1, 0]), np.array([0, 2])],
)
def test_entity_variant_assignment_is_an_in_range_integer_identity(assignment) -> None:
    with pytest.raises((ValueError, TypeError)):
        EntityVariantBinding("object", _plan(assignment))


@pytest.mark.parametrize("quaternion", [(0.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)])
def test_initial_orientation_rejects_nonunit_quaternions(quaternion) -> None:
    with pytest.raises(ValueError, match="unit wxyz"):
        EntityInitialState(quaternion=quaternion)


def test_reset_preserves_world_frame_values_and_omitted_fields_through_pickle() -> None:
    pose = _pose()
    velocity = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-1, -2, -3, -4, -5, -6]])
    patch = EntityStatePatch("object", root_pose=pose, root_velocity=velocity)
    request = SceneResetRequest((np.int64(4), np.int32(1)), (patch,))
    pose[:] = 0
    velocity[:] = 0
    request.validate_env_count(5)
    restored = pickle.loads(pickle.dumps(request))
    for value in (request, restored):
        assert value.env_ids == (4, 1)
        assert all(type(index) is int for index in value.env_ids)
        state = value.patches[0]
        np.testing.assert_array_equal(state.root_pose, _pose())
        np.testing.assert_array_equal(state.root_velocity[0], [1, 2, 3, 4, 5, 6])
        assert state.joint_positions is None and state.joint_velocities is None
        for array in (state.root_pose, state.root_velocity):
            with pytest.raises(ValueError):
                array.setflags(write=True)


@pytest.mark.parametrize("metadata", ["dtype", "shape"])
@pytest.mark.filterwarnings("ignore:Setting the (dtype|shape) on a NumPy array:DeprecationWarning")
@pytest.mark.parametrize(
    "field", ["root_pose", "root_velocity", "joint_positions", "joint_velocities"]
)
def test_patch_array_metadata_cannot_change_a_prevalidated_reset(field, metadata) -> None:
    # A read-only data buffer alone does not prevent these legacy metadata writes.
    fields = {
        "root_pose": _pose(),
        "root_velocity": np.arange(12, dtype=np.float64).reshape(2, 6),
        "joint_positions": np.array([[0.2, 0.1], [0.4, 0.3]]),
        "joint_velocities": np.array([[2.0, 1.0], [4.0, 3.0]]),
    }
    patch = EntityStatePatch("object", joint_names=("hinge_b", "hinge_a"), **fields)
    request = SceneResetRequest((4, 1), (patch,))
    exposed = getattr(patch, field)
    if metadata == "dtype":
        exposed.dtype = np.uint8
        assert exposed.dtype != fields[field].dtype
    else:
        exposed.shape = (exposed.size,)
        assert exposed.shape != fields[field].shape

    for value in (request, pickle.loads(pickle.dumps(request))):
        value.validate_env_count(5)
        # Re-validating from the stored patch must retain the original selected rows.
        validated = SceneResetRequest(value.env_ids, value.patches)
        state = validated.patches[0]
        assert state.joint_names == ("hinge_b", "hinge_a")
        for name, expected in fields.items():
            actual = getattr(state, name)
            np.testing.assert_array_equal(actual, expected)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert not actual.flags.writeable


def test_joint_patch_keeps_declared_order_and_leaves_roots_unwritten() -> None:
    patch = EntityStatePatch(
        "object", joint_names=("hinge_b", "hinge_a"),
        joint_velocities=np.array([[2.0, 1.0], [4.0, 3.0]]),
    )
    request = SceneResetRequest((3, 0), (patch,))
    assert request.patches[0].joint_names == ("hinge_b", "hinge_a")
    assert patch.root_pose is None and patch.root_velocity is None
    assert patch.joint_positions is None
    np.testing.assert_array_equal(patch.joint_velocities, [[2, 1], [4, 3]])


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({}, "at least one field"),
        ({"root_pose": np.zeros((2, 6))}, "shape"),
        ({"root_velocity": np.zeros(6)}, "shape"),
        ({"root_velocity": np.zeros((2, 7))}, "shape"),
        ({"root_pose": np.zeros((2, 7))}, "unit wxyz"),
        ({"root_velocity": np.full((2, 6), np.nan)}, "finite"),
        ({"joint_positions": np.array([[np.inf]])}, "finite"),
        ({"joint_positions": np.array([[1j]])}, "finite real"),
        ({"joint_positions": np.array([[True]])}, "finite real"),
        ({"root_pose": _pose(), "root_velocity": np.zeros((1, 6))}, "row counts"),
        ({"root_pose": _pose(), "joint_names": ("hinge",)}, "joint state field"),
        ({"joint_positions": np.zeros((2, 2)), "joint_names": ("j", "j")}, "duplicates"),
    ],
)
def test_reset_patch_rejects_malformed_writes_before_binding(fields, error) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        EntityStatePatch("object", **fields)


@pytest.mark.parametrize("ids", [(), (0, 0), (-1,), (0.5,), (True,), (np.bool_(False),)])
def test_reset_rejects_invalid_environment_ids(ids) -> None:
    patch = EntityStatePatch("object", root_pose=_pose(max(len(ids), 1)))
    with pytest.raises((ValueError, TypeError), match="env_ids"):
        SceneResetRequest(ids, (patch,))


def test_reset_rejects_duplicate_entity_writes_and_wrong_row_count() -> None:
    patch = EntityStatePatch("object", root_pose=_pose())
    with pytest.raises(ValueError, match="single patch"):
        SceneResetRequest((0, 1), (patch, patch))
    with pytest.raises(ValueError, match="row count"):
        SceneResetRequest((0,), (patch,))
    with pytest.raises(TypeError, match="non-empty tuple"):
        SceneResetRequest((0, 1), ())


@pytest.mark.parametrize("count", [0, -1, 2.0, True])
def test_reset_environment_bound_requires_positive_integer(count) -> None:
    request = SceneResetRequest((0,), (EntityStatePatch("object", root_pose=_pose(1)),))
    with pytest.raises(ValueError, match="positive integer"):
        request.validate_env_count(count)


def test_reset_rejects_out_of_bounds_but_allows_unsorted_environment_ids() -> None:
    request = SceneResetRequest((4, 1), (EntityStatePatch("object", root_pose=_pose()),))
    request.validate_env_count(5)
    with pytest.raises(ValueError, match="environment count"):
        request.validate_env_count(4)


_ADAPTERS = (
    ("mujoco", "MuJoCoBackend"), ("mjwarp", "MjwarpBackend"),
    ("isaacgym", "IsaacGymBackend"), ("isaacsim", "IsaacSimBackend"),
    ("motrix", "MotrixBackend"), ("drake", "DrakeBackend"),
    ("genesis", "GenesisBackend"), ("newton", "NewtonBackend"),
    ("superdex", "SuperDexBackend"),
)


@pytest.mark.parametrize(("backend", "class_name"), _ADAPTERS)
def test_factory_rejects_unimplemented_entity_materialization_before_sdk_lookup(
    backend, class_name, monkeypatch,
) -> None:
    import unisim.factory as factory

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("unsupported composition reached adapter/SDK lookup")

    monkeypatch.setattr(factory, "adapter_spec", unexpected_lookup)
    scene = SceneCfg(entity_assets=(_physical(),))
    with pytest.raises(NotImplementedError, match=rf"{backend}.*entity_assets"):
        unisim.create_backend(backend, scene, num_envs=2, sim_dt=0.01)


@pytest.mark.parametrize(("backend", "class_name"), _ADAPTERS)
def test_direct_adapters_cannot_silently_discard_entity_declarations(backend, class_name) -> None:
    adapter = getattr(unisim, class_name)
    scene = SceneCfg(entity_assets=(_physical(),))
    with pytest.raises(NotImplementedError, match=rf"{backend}.*entity_assets"):
        adapter(scene=scene, num_envs=2, sim_dt=0.01)


def test_mutated_scene_is_revalidated_at_factory_boundary() -> None:
    scene = SceneCfg(model_file="legacy.xml")
    scene.entity_assets = (_physical(),)
    with pytest.raises(ValueError, match="model_file"):
        unisim.create_backend("isaacgym", scene, num_envs=2)


def test_unimplemented_entity_operations_preserve_existing_backend_state() -> None:
    backend = unisim.create_backend("fake", num_envs=2)
    backend.step(np.array([[0.25], [-0.5]]))
    before = backend.get_state()
    request = SceneResetRequest((1,), (EntityStatePatch("object", root_pose=_pose(1)),))
    with pytest.raises(NotImplementedError, match="fake.*physical entities"):
        backend.get_entity_names()
    with pytest.raises(NotImplementedError, match="fake.*entity state"):
        backend.get_entity_state("object")
    with pytest.raises(NotImplementedError, match="fake.*entity reset"):
        backend.reset_entities(request)
    after = backend.get_state()
    assert before.keys() == after.keys()
    for field in before:
        np.testing.assert_array_equal(after[field], before[field])
