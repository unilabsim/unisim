"""Contract tests for detached ray collision geometry and static filters."""

from __future__ import annotations

import numpy as np
import pytest

from unisim import (
    FakeRayCaster,
    RayCollisionDescription,
    RayGeomType,
    RayMeshDescription,
    RaySceneDescription,
    RayTraceFilter,
    UnsupportedCapabilityError,
)


def _scene(
    geom_types: tuple[RayGeomType, ...] = (RayGeomType.SPHERE, RayGeomType.BOX),
    geom_body_ids: np.ndarray | None = None,
) -> RaySceneDescription:
    count = len(geom_types)
    if geom_body_ids is None:
        geom_body_ids = np.arange(count, dtype=np.intp)
    return RaySceneDescription(
        num_bodies=3,
        geom_types=geom_types,
        geom_sizes=np.ones((count, 3), dtype=np.float64),
        geom_local_pos=np.zeros((count, 3), dtype=np.float64),
        geom_local_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1)),
        geom_body_ids=geom_body_ids,
        body_names=("world", "robot", "child"),
    )


def _mesh() -> RayMeshDescription:
    points = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    return RayMeshDescription(asset_id="mesh:triangle", points=points, indices=indices)


def test_mesh_descriptor_is_detached_and_validated() -> None:
    mesh = _mesh()
    assert mesh.points.dtype == np.float32
    assert mesh.indices.dtype == np.int32
    with pytest.raises(ValueError, match="positive area"):
        RayMeshDescription(
            asset_id="degenerate",
            points=np.zeros((3, 3), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        )


def test_collision_descriptor_normalizes_identities_tags_and_mesh_mapping() -> None:
    scene = _scene()
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("sensor", "target"),
        geom_tags=(["semantic", "robot"], ["semantic", "terrain"]),
        meshes=(_mesh(),),
        geom_mesh_ids=np.array([-1, -1], dtype=np.intp),
        body_parent_ids=np.array([-1, -1, 1], dtype=np.intp),
    )
    assert collision.geom_names == ("sensor", "target")
    assert collision.geom_tags == (("robot", "semantic"), ("semantic", "terrain"))
    assert collision.num_geoms == 2
    assert collision.num_meshes == 1
    assert not collision.geom_mesh_ids.flags.writeable
    assert not collision.body_parent_ids.flags.writeable


def test_collision_descriptor_rejects_invalid_identity_and_mesh_mapping() -> None:
    scene = _scene((RayGeomType.MESH,))
    with pytest.raises(ValueError, match="unique"):
        RayCollisionDescription(scene=_scene(), geom_names=("same", "same"))
    with pytest.raises(ValueError, match="mesh geoms"):
        RayCollisionDescription(scene=scene)


def test_filter_resolves_allow_tags_direct_body_and_ancestors() -> None:
    scene = _scene(geom_body_ids=np.array([1, 2], dtype=np.intp))
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("robot_sphere", "child_box"),
        geom_tags=(("robot",), ("terrain",)),
        body_parent_ids=np.array([-1, -1, 1], dtype=np.intp),
    )
    np.testing.assert_array_equal(
        RayTraceFilter(allowed_geom_ids=(1,)).geom_mask(collision), [False, True]
    )
    np.testing.assert_array_equal(
        RayTraceFilter(allowed_tags=("terrain",)).geom_mask(collision), [False, True]
    )
    np.testing.assert_array_equal(
        RayTraceFilter(excluded_body_ids=(1,)).geom_mask(collision), [False, True]
    )
    np.testing.assert_array_equal(
        RayTraceFilter(excluded_ancestor_body_ids=(1,)).geom_mask(collision),
        [False, False],
    )


def test_filter_fails_closed_on_unknown_identity_tags_and_missing_ancestry() -> None:
    scene = _scene()
    collision = RayCollisionDescription(
        scene=scene,
        geom_tags=(("robot",), ("terrain",)),
    )
    with pytest.raises(ValueError, match="allowed_geom_ids"):
        RayTraceFilter(allowed_geom_ids=(2,)).geom_mask(collision)
    with pytest.raises(ValueError, match="allowed_tags"):
        RayTraceFilter(allowed_tags=("missing",)).geom_mask(collision)
    with pytest.raises(ValueError, match="excluded_body_ids"):
        RayTraceFilter(excluded_body_ids=(3,)).geom_mask(collision)
    with pytest.raises(UnsupportedCapabilityError, match="ancestor"):
        RayTraceFilter(excluded_ancestor_body_ids=(1,)).geom_mask(collision)


def test_ancestor_exclusion_supports_reordered_and_omitted_axes() -> None:
    scene = _scene(geom_body_ids=np.array([1, 2], dtype=np.intp))
    reordered = RayCollisionDescription(
        scene=scene,
        geom_tags=(("robot",), ("robot",)),
        body_parent_ids=np.array([2, -1, 1], dtype=np.intp),
    )
    np.testing.assert_array_equal(
        RayTraceFilter(excluded_ancestor_body_ids=(1,)).geom_mask(reordered),
        [False, False],
    )

    omitted = RayCollisionDescription(
        scene=scene,
        geom_tags=(("robot",), ("robot",)),
        body_parent_ids=np.array([-1, -1, -1], dtype=np.intp),
    )
    np.testing.assert_array_equal(
        RayTraceFilter(excluded_ancestor_body_ids=(1,)).geom_mask(omitted),
        [False, True],
    )


def test_collision_descriptor_rejects_body_ancestry_cycles() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        RayCollisionDescription(
            scene=_scene(),
            body_parent_ids=np.array([1, 0, -1], dtype=np.intp),
        )


def test_fake_reference_caster_consumes_public_mesh_descriptor() -> None:
    mesh = _mesh()
    scene = _scene((RayGeomType.MESH,), np.zeros(1, dtype=np.intp))
    collision = RayCollisionDescription(
        scene=scene,
        geom_names=("ground",),
        meshes=(mesh,),
        geom_mesh_ids=np.zeros(1, dtype=np.intp),
    )
    caster = FakeRayCaster(num_envs=1, num_rays=1, collision=collision)
    caster.materialize(scene)
    result = caster.trace(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, -1.0]]),
        max_distance=2.0,
        outputs=None,
        min_distance=0.5,
        ray_filter=RayTraceFilter(allowed_geom_ids=(0,)),
    )
    assert result.hit.item()
    np.testing.assert_allclose(result.distance, 1.0)
