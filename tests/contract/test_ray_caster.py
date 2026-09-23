"""Contract tests for the backend-neutral ray-query plugin protocol."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from unisim import (
    BackendError,
    FakeRayCaster,
    RayCaster,
    RayCasterCapabilities,
    RayGeomType,
    RaySceneDescription,
    RayTraceOutputs,
    RayTraceResult,
    UnsupportedCapabilityError,
    assert_ray_caster_conformance,
    require_ray_trace_outputs,
)

DOWN = np.array([[0.0, 0.0, -1.0]])
UP = np.array([[0.0, 0.0, 1.0]])


def _plane_scene() -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=1,
        geom_types=(RayGeomType.PLANE,),
        geom_sizes=np.zeros((1, 3)),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.zeros(1, dtype=np.intp),
    )


def _identity_pose(num_envs: int, num_bodies: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((num_envs, num_bodies, 3))
    quat = np.zeros((num_envs, num_bodies, 4))
    quat[..., 0] = 1.0
    return pos, quat


def _materialized_caster(num_envs: int = 2, num_rays: int = 2) -> FakeRayCaster:
    caster = FakeRayCaster(num_envs=num_envs, num_rays=num_rays)
    caster.materialize(_plane_scene())
    return caster


class TestRaySceneDescription:
    def test_valid_construction_detaches_arrays(self) -> None:
        sizes = np.zeros((1, 3))
        scene = RaySceneDescription(
            num_bodies=1,
            geom_types=("plane",),
            geom_sizes=sizes,
            geom_local_pos=np.zeros((1, 3)),
            geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            geom_body_ids=np.zeros(1, dtype=np.int64),
        )
        assert scene.num_geoms == 1
        assert scene.geom_types == (RayGeomType.PLANE,)
        sizes[0, 0] = 1.0
        assert scene.geom_sizes[0, 0] == 0.0
        with pytest.raises(ValueError):
            scene.geom_sizes[0, 0] = 1.0

    def test_rejects_non_integer_num_bodies(self) -> None:
        with pytest.raises(TypeError, match="num_bodies must be an integer"):
            RaySceneDescription(
                num_bodies=True,
                geom_types=(),
                geom_sizes=np.zeros((0, 3)),
                geom_local_pos=np.zeros((0, 3)),
                geom_local_quat=np.zeros((0, 4)),
                geom_body_ids=np.zeros(0, dtype=np.intp),
            )

    def test_rejects_nonpositive_num_bodies(self) -> None:
        with pytest.raises(ValueError, match="num_bodies must be positive"):
            RaySceneDescription(
                num_bodies=0,
                geom_types=(),
                geom_sizes=np.zeros((0, 3)),
                geom_local_pos=np.zeros((0, 3)),
                geom_local_quat=np.zeros((0, 4)),
                geom_body_ids=np.zeros(0, dtype=np.intp),
            )

    def test_rejects_unknown_geom_type(self) -> None:
        with pytest.raises(ValueError, match="unknown ray scene geom type"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("hffield",),
                geom_sizes=np.zeros((1, 3)),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )

    def test_rejects_wrong_array_shape(self) -> None:
        with pytest.raises(ValueError, match="geom_sizes must have shape"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("sphere",),
                geom_sizes=np.zeros((2, 3)),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )

    def test_rejects_negative_sizes(self) -> None:
        with pytest.raises(ValueError, match="geom_sizes must be non-negative"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("sphere",),
                geom_sizes=np.array([[-1.0, 0.0, 0.0]]),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )

    def test_rejects_non_finite_pose(self) -> None:
        with pytest.raises(ValueError, match="geom_local_pos must be finite"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("sphere",),
                geom_sizes=np.array([[1.0, 0.0, 0.0]]),
                geom_local_pos=np.array([[np.nan, 0.0, 0.0]]),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )

    def test_rejects_non_unit_local_quat(self) -> None:
        with pytest.raises(ValueError, match="unit wxyz"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("sphere",),
                geom_sizes=np.array([[1.0, 0.0, 0.0]]),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[2.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )

    def test_rejects_out_of_range_body_ids(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            RaySceneDescription(
                num_bodies=1,
                geom_types=("sphere",),
                geom_sizes=np.array([[1.0, 0.0, 0.0]]),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.array([1]),
            )


class TestRayCasterCapabilities:
    def test_defaults_declare_nothing(self) -> None:
        capabilities = RayCasterCapabilities()
        assert not any(dataclasses.asdict(capabilities).values())

    def test_frozen(self) -> None:
        capabilities = RayCasterCapabilities()
        with pytest.raises(dataclasses.FrozenInstanceError):
            capabilities.supports_pose_sync = True  # type: ignore[misc]

    def test_fake_declares_host_primitive_profile(self) -> None:
        capabilities = FakeRayCaster().get_ray_capabilities()
        assert capabilities.supports_pose_sync
        assert capabilities.supports_per_env_rays
        assert capabilities.supports_host_readback
        assert not capabilities.supports_device_output
        assert capabilities.supports_hit_point
        assert capabilities.supports_normal
        assert capabilities.supports_geom_id
        assert capabilities.supports_body_id


class TestRequireRayTraceOutputs:
    def test_accepts_declared_outputs(self) -> None:
        capabilities = RayCasterCapabilities(supports_normal=True)
        require_ray_trace_outputs(capabilities, RayTraceOutputs(normal=True))

    def test_rejects_undeclared_outputs_fail_closed(self) -> None:
        capabilities = RayCasterCapabilities(supports_hit_point=True)
        with pytest.raises(UnsupportedCapabilityError, match="normal"):
            require_ray_trace_outputs(capabilities, RayTraceOutputs(normal=True))


class TestRayTraceResult:
    def _minimal(self) -> dict[str, np.ndarray]:
        return {
            "distance": np.ones((2, 3)),
            "hit": np.ones((2, 3), dtype=np.bool_),
        }

    def test_minimal_result(self) -> None:
        result = RayTraceResult(**self._minimal())
        assert result.hit_point is None
        assert result.normal is None
        assert result.geom_id is None
        assert result.body_id is None

    def test_rejects_non_finite_distance(self) -> None:
        fields = self._minimal()
        fields["distance"] = np.full((2, 3), np.nan)
        with pytest.raises(ValueError, match="distance must be finite"):
            RayTraceResult(**fields)

    def test_rejects_mismatched_hit_shape(self) -> None:
        fields = self._minimal()
        fields["hit"] = np.ones((2, 2), dtype=np.bool_)
        with pytest.raises(ValueError, match="hit must be a bool array"):
            RayTraceResult(**fields)

    def test_rejects_bad_hit_point_shape(self) -> None:
        with pytest.raises(ValueError, match="hit_point must be a floating array"):
            RayTraceResult(**self._minimal(), hit_point=np.zeros((2, 3)))

    def test_rejects_negative_geom_id_on_hit(self) -> None:
        with pytest.raises(ValueError, match="geom_id must be non-negative"):
            RayTraceResult(**self._minimal(), geom_id=np.full((2, 3), -1))

    def test_allows_negative_ids_on_missed_rays(self) -> None:
        fields = self._minimal()
        fields["hit"] = np.zeros((2, 3), dtype=np.bool_)
        result = RayTraceResult(**fields, geom_id=np.full((2, 3), -1))
        assert result.geom_id is not None


class TestFakeRayCasterLifecycle:
    def test_trace_before_materialize_fails(self) -> None:
        caster = FakeRayCaster(num_envs=1, num_rays=1)
        with pytest.raises(BackendError, match="materialized"):
            caster.trace(np.zeros((1, 3)), DOWN, 10.0)

    def test_double_materialize_fails(self) -> None:
        caster = _materialized_caster()
        with pytest.raises(BackendError, match="already materialized"):
            caster.materialize(_plane_scene())

    def test_close_is_idempotent_and_later_queries_fail(self) -> None:
        caster = _materialized_caster()
        caster.close()
        caster.close()
        with pytest.raises(BackendError, match="closed"):
            caster.trace(np.zeros((2, 3)), DOWN, 10.0)
        with pytest.raises(BackendError, match="closed"):
            caster.materialize(_plane_scene())
        with pytest.raises(BackendError, match="closed"):
            caster.update_pose(*_identity_pose(2, 1))

    def test_constructor_validates_batch_shape(self) -> None:
        with pytest.raises(TypeError, match="num_envs must be an integer"):
            FakeRayCaster(num_envs=True)
        with pytest.raises(ValueError, match="num_rays must be positive"):
            FakeRayCaster(num_rays=0)

    def test_rejects_non_scene_input(self) -> None:
        caster = FakeRayCaster()
        with pytest.raises(TypeError, match="RaySceneDescription"):
            caster.materialize(object())  # type: ignore[arg-type]


class _NoPoseSyncCaster(FakeRayCaster):
    """Fake caster narrowed to the minimal no-pose-sync capability set."""

    _ray_capabilities = RayCasterCapabilities(
        supports_host_readback=True,
        supports_per_env_rays=False,
        supports_hit_point=True,
        supports_geom_id=True,
    )

    def update_pose(self, body_pos, body_quat, env_ids=None) -> None:
        # Route through the fail-closed base default like a plugin that never
        # implemented pose sync would.
        RayCaster.update_pose(self, body_pos, body_quat, env_ids)


class TestCapabilityBoundaries:
    def test_mesh_geoms_fail_closed(self) -> None:
        caster = FakeRayCaster()
        scene = RaySceneDescription(
            num_bodies=1,
            geom_types=(RayGeomType.MESH,),
            geom_sizes=np.zeros((1, 3)),
            geom_local_pos=np.zeros((1, 3)),
            geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            geom_body_ids=np.zeros(1, dtype=np.intp),
        )
        with pytest.raises(UnsupportedCapabilityError, match="mesh"):
            caster.materialize(scene)

    def test_nonpositive_primitive_sizes_fail_closed(self) -> None:
        caster = FakeRayCaster()
        scene = RaySceneDescription(
            num_bodies=1,
            geom_types=(RayGeomType.SPHERE,),
            geom_sizes=np.zeros((1, 3)),
            geom_local_pos=np.zeros((1, 3)),
            geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            geom_body_ids=np.zeros(1, dtype=np.intp),
        )
        with pytest.raises(ValueError, match="positive sizes"):
            caster.materialize(scene)

    def test_pose_sync_rejected_when_undeclared(self) -> None:
        caster = _NoPoseSyncCaster(num_envs=1, num_rays=1)
        caster.materialize(_plane_scene())
        with pytest.raises(UnsupportedCapabilityError, match="supports_pose_sync"):
            caster.update_pose(*_identity_pose(1, 1))

    def test_per_env_rays_rejected_when_undeclared(self) -> None:
        caster = _NoPoseSyncCaster(num_envs=2, num_rays=1)
        caster.materialize(_plane_scene())
        with pytest.raises(UnsupportedCapabilityError, match="per-environment rays"):
            caster.trace(np.zeros((2, 1, 3)), np.broadcast_to(DOWN, (2, 1, 3)), 10.0)

    def test_undeclared_output_rejected_fail_closed(self) -> None:
        caster = _NoPoseSyncCaster(num_envs=1, num_rays=1)
        caster.materialize(_plane_scene())
        with pytest.raises(UnsupportedCapabilityError, match="normal"):
            caster.trace(np.zeros((1, 3)), DOWN, 10.0, outputs=RayTraceOutputs(normal=True))
        result = caster.trace(
            np.zeros((1, 3)),
            DOWN,
            10.0,
            outputs=RayTraceOutputs(hit_point=True, geom_id=True),
        )
        assert result.hit_point is not None
        assert result.geom_id is not None
        assert result.normal is None


class TestFakeRayCasterTrace:
    def test_plane_hit_and_miss_semantics(self) -> None:
        caster = _materialized_caster(num_envs=2, num_rays=2)
        origins = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
        directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        result = caster.trace(origins, directions, 10.0)
        assert result.hit.shape == (2, 2)
        np.testing.assert_array_equal(result.hit, [[True, False], [True, False]])
        np.testing.assert_allclose(result.distance, [[3.0, 10.0], [3.0, 10.0]])

    def test_max_distance_clips_hits(self) -> None:
        caster = _materialized_caster(num_envs=1, num_rays=1)
        result = caster.trace(np.array([[0.0, 0.0, 3.0]]), DOWN, 2.0)
        assert not result.hit[0, 0]
        assert result.distance[0, 0] == 2.0

    @pytest.mark.parametrize(
        ("geom_type", "size", "local_pos", "expected"),
        [
            ("sphere", (0.5, 0.0, 0.0), (0.0, 0.0, 0.0), 4.5),
            ("box", (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), 4.5),
            ("cylinder", (0.5, 1.0, 0.0), (0.0, 0.0, 0.0), 4.0),
            ("capsule", (0.5, 1.0, 0.0), (0.0, 0.0, 0.0), 3.5),
            ("ellipsoid", (1.0, 0.5, 0.25), (0.0, 0.0, 0.0), 4.75),
        ],
    )
    def test_primitive_intersection_distances(
        self,
        geom_type: str,
        size: tuple[float, float, float],
        local_pos: tuple[float, float, float],
        expected: float,
    ) -> None:
        caster = FakeRayCaster(num_envs=1, num_rays=1)
        caster.materialize(
            RaySceneDescription(
                num_bodies=1,
                geom_types=(geom_type,),
                geom_sizes=np.array([size]),
                geom_local_pos=np.array([local_pos]),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.zeros(1, dtype=np.intp),
            )
        )
        result = caster.trace(np.array([[0.0, 0.0, 5.0]]), DOWN, 100.0)
        assert result.hit[0, 0]
        np.testing.assert_allclose(result.distance[0, 0], expected, rtol=1e-9)

    def test_optional_outputs_are_consistent(self) -> None:
        caster = FakeRayCaster(num_envs=1, num_rays=1)
        caster.materialize(
            RaySceneDescription(
                num_bodies=2,
                geom_types=("sphere",),
                geom_sizes=np.array([[0.5, 0.0, 0.0]]),
                geom_local_pos=np.zeros((1, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.array([1]),
            )
        )
        result = caster.trace(
            np.array([[0.0, 0.0, 5.0]]),
            DOWN,
            100.0,
            outputs=RayTraceOutputs(hit_point=True, normal=True, geom_id=True, body_id=True),
        )
        np.testing.assert_allclose(result.hit_point[0, 0], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(result.normal[0, 0], [0.0, 0.0, 1.0])
        assert result.geom_id[0, 0] == 0
        assert result.body_id[0, 0] == 1

    def test_missed_rays_report_negative_ids(self) -> None:
        caster = _materialized_caster(num_envs=1, num_rays=1)
        result = caster.trace(
            np.array([[0.0, 0.0, 1.0]]),
            UP,
            10.0,
            outputs=RayTraceOutputs(geom_id=True, body_id=True),
        )
        assert result.geom_id[0, 0] == -1
        assert result.body_id[0, 0] == -1

    def test_update_pose_moves_geometry_per_environment(self) -> None:
        caster = FakeRayCaster(num_envs=2, num_rays=1)
        caster.materialize(
            RaySceneDescription(
                num_bodies=2,
                geom_types=("plane", "sphere"),
                geom_sizes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                geom_local_pos=np.zeros((2, 3)),
                geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                geom_body_ids=np.array([0, 1]),
            )
        )
        pos, quat = _identity_pose(1, 2)
        pos[0, 1, 2] = 2.0
        caster.update_pose(pos, quat, env_ids=np.array([1]))
        origins = np.zeros((2, 1, 3))
        origins[..., 2] = 5.0
        directions = np.zeros((2, 1, 3))
        directions[..., 2] = -1.0
        result = caster.trace(origins, directions, 100.0)
        np.testing.assert_allclose(result.distance[:, 0], [4.5, 2.5])

    def test_selected_rows_limit_the_result_batch(self) -> None:
        caster = _materialized_caster(num_envs=4, num_rays=2)
        origins = np.zeros((2, 2, 3))
        origins[..., 2] = 1.0
        directions = np.broadcast_to(DOWN, (2, 2, 3))
        result = caster.trace(origins, directions, 10.0, env_ids=[3, 1])
        assert result.distance.shape == (2, 2)
        assert result.hit.all()

    def test_shared_rays_broadcast_over_all_rows(self) -> None:
        caster = _materialized_caster(num_envs=3, num_rays=2)
        origins = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        result = caster.trace(origins, np.broadcast_to(DOWN, (2, 3)), 10.0)
        np.testing.assert_allclose(result.distance, [[1.0, 2.0]] * 3)

    def test_empty_scene_reports_all_misses(self) -> None:
        caster = FakeRayCaster(num_envs=2, num_rays=1)
        caster.materialize(
            RaySceneDescription(
                num_bodies=1,
                geom_types=(),
                geom_sizes=np.zeros((0, 3)),
                geom_local_pos=np.zeros((0, 3)),
                geom_local_quat=np.zeros((0, 4)),
                geom_body_ids=np.zeros(0, dtype=np.intp),
            )
        )
        result = caster.trace(np.zeros((2, 1, 3)), np.broadcast_to(DOWN, (2, 1, 3)), 7.0)
        assert not result.hit.any()
        np.testing.assert_allclose(result.distance, 7.0)

    @pytest.mark.parametrize(
        ("origins", "directions", "match"),
        [
            (np.zeros((2, 3)), np.zeros((3, 3)), "does not match directions"),
            (np.zeros((4, 3)), np.zeros((4, 3)), "must have shape"),
            (np.array([[0.0, 0.0, np.nan]]), DOWN, "must be finite"),
            (np.zeros((1, 3)), np.array([[0.0, 0.0, -2.0]]), "unit vectors"),
        ],
    )
    def test_ray_batch_validation(
        self, origins: np.ndarray, directions: np.ndarray, match: str
    ) -> None:
        caster = _materialized_caster(num_envs=2, num_rays=1)
        with pytest.raises(ValueError, match=match):
            caster.trace(origins, directions, 10.0)

    def test_max_distance_validation(self) -> None:
        caster = _materialized_caster(num_envs=1, num_rays=1)
        for bad in (0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError, match="max_distance"):
                caster.trace(np.zeros((1, 3)), DOWN, bad)

    def test_env_ids_validation(self) -> None:
        caster = _materialized_caster(num_envs=2, num_rays=1)
        with pytest.raises(ValueError, match="environment IDs"):
            caster.trace(np.zeros((1, 3)), DOWN, 10.0, env_ids=[2])
        with pytest.raises(ValueError, match="environment IDs"):
            caster.trace(
                np.zeros((2, 1, 3)),
                np.broadcast_to(DOWN, (2, 1, 3)),
                10.0,
                env_ids=[1, 1],
            )

    def test_update_pose_validation(self) -> None:
        caster = _materialized_caster(num_envs=2, num_rays=1)
        pos, quat = _identity_pose(2, 1)
        with pytest.raises(ValueError, match="body_pos must have shape"):
            caster.update_pose(np.zeros((3, 1, 3)), quat)
        with pytest.raises(ValueError, match="unit wxyz"):
            caster.update_pose(pos, np.zeros((2, 1, 4)))
        with pytest.raises(ValueError, match="environment IDs"):
            caster.update_pose(pos[:1], quat[:1], env_ids=[5])


class _BrokenShapeCaster(FakeRayCaster):
    def trace(self, ray_origins, ray_directions, max_distance, env_ids=None, outputs=None):
        result = super().trace(ray_origins, ray_directions, max_distance, env_ids, outputs)
        return RayTraceResult(distance=result.distance[:, :1], hit=result.hit[:, :1])


class TestRayCasterConformance:
    def test_conformance_accepts_fake_caster(self) -> None:
        assert_ray_caster_conformance(FakeRayCaster(num_envs=2, num_rays=3))

    def test_conformance_rejects_wrong_result_shape(self) -> None:
        caster = _BrokenShapeCaster(num_envs=2, num_rays=3)
        with pytest.raises(AssertionError):
            assert_ray_caster_conformance(caster)
