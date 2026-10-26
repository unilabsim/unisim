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
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceOutputs,
    RayTraceResult,
    RayTraceTensorResult,
    UnsupportedCapabilityError,
    assert_ray_caster_conformance,
    ray_geom_type,
    require_ray_trace_outputs,
)

DOWN = np.array([[0.0, 0.0, -1.0]])
UP = np.array([[0.0, 0.0, 1.0]])


def _plane_scene(num_bodies: int = 1) -> RaySceneDescription:
    return RaySceneDescription(
        num_bodies=num_bodies,
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


def _materialized_caster(
    num_envs: int = 2, num_rays: int = 2, num_bodies: int = 1
) -> FakeRayCaster:
    caster = FakeRayCaster(num_envs=num_envs, num_rays=num_rays)
    caster.materialize(_plane_scene(num_bodies=num_bodies))
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

    def test_rejects_hfield_native_code(self) -> None:
        with pytest.raises(ValueError, match="hfield"):
            ray_geom_type(1)

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
        assert capabilities.supports_device_output
        assert capabilities.supports_tensor_pose_sync
        assert capabilities.supports_tensor_rays
        assert capabilities.torch_devices == ("cpu",)
        assert capabilities.stream_event_ownership == "caller"
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


torch = pytest.importorskip("torch", reason="Torch is required for the ray tensor contract")


class TestFakeRayTensorProfile:
    def _tensor_caster(self, num_envs: int = 2, num_rays: int = 2) -> FakeRayCaster:
        caster = _materialized_caster(num_envs=num_envs, num_rays=num_rays)
        caster.bind_pose_layout(("body_0",))
        return caster

    def _pose(self, num_envs: int) -> RayPoseTensorView:
        pos = torch.zeros((num_envs, 1, 3), dtype=torch.float32)
        quat = torch.zeros((num_envs, 1, 4), dtype=torch.float32)
        quat[..., 0] = 1.0
        return RayPoseTensorView(body_names=("body_0",), pos_w=pos, quat_w=quat)

    def test_tensor_trace_parity_and_persistent_views(self) -> None:
        caster = self._tensor_caster()
        caster.update_pose_tensor(self._pose(2))
        origins = torch.zeros((2, 3), dtype=torch.float32)
        origins[:, 2] = 2.0
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        outputs = RayTraceOutputs(hit_point=True, geom_id=True, body_id=True)
        tensor_result = caster.trace_tensor(origins, directions, 3.0, outputs=outputs)
        host_result = caster.trace(origins.numpy(), directions.numpy(), 3.0, outputs=outputs)
        assert isinstance(tensor_result, RayTraceTensorResult)
        torch.testing.assert_close(
            tensor_result.distance, torch.as_tensor(host_result.distance, dtype=torch.float32)
        )
        torch.testing.assert_close(
            tensor_result.hit, torch.as_tensor(host_result.hit, dtype=torch.bool)
        )
        torch.testing.assert_close(
            tensor_result.hit_point, torch.as_tensor(host_result.hit_point, dtype=torch.float32)
        )
        torch.testing.assert_close(
            tensor_result.geom_id, torch.as_tensor(host_result.geom_id, dtype=torch.int32)
        )
        torch.testing.assert_close(
            tensor_result.body_id, torch.as_tensor(host_result.body_id, dtype=torch.int32)
        )

        selected_rows = torch.tensor([1], dtype=torch.int64)
        selected_origins = origins[1:].repeat(1, 2, 1)
        selected_directions = directions[1:].repeat(1, 2, 1)
        selected = caster.trace_tensor(
            selected_origins, selected_directions, 2.0, env_ids=selected_rows
        )
        first = selected.distance
        again = caster.trace_tensor(
            selected_origins, selected_directions, 1.0, env_ids=selected_rows
        )
        assert first is again.distance
        torch.testing.assert_close(again.distance, torch.full((1, 2), 1.0))

    def test_tensor_pose_accepts_packed_strided_packet_views(self) -> None:
        caster = self._tensor_caster()
        packet = torch.zeros((2, 10), dtype=torch.float32)
        packet[:, 0:3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=torch.float32
        )
        packet[:, 4:8] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        packed_pose = RayPoseTensorView(
            body_names=("body_0",),
            pos_w=packet[:, 0:3].reshape(2, 1, 3),
            quat_w=packet[:, 4:8].reshape(2, 1, 4),
        )
        assert not packed_pose.pos_w.is_contiguous()
        assert not packed_pose.quat_w.is_contiguous()
        caster.update_pose_tensor(packed_pose)
        assert caster._tensor_pos.data_ptr() == packed_pose.pos_w.data_ptr()
        assert caster._tensor_quat.data_ptr() == packed_pose.quat_w.data_ptr()

        origins = torch.zeros((2, 3), dtype=torch.float32)
        origins[:, 2] = 2.0
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        result = caster.trace_tensor(origins, directions, 3.0)
        torch.testing.assert_close(result.distance, torch.full((2, 2), 1.0))

    def test_tensor_pose_rejects_unsafe_storage_without_hidden_copy(self) -> None:
        caster = _materialized_caster(num_envs=2, num_rays=1, num_bodies=2)
        caster.bind_pose_layout(("body_0", "body_1"))
        original_pose = self._two_body_pose()
        caster.update_pose_tensor(original_pose)
        original = caster._tensor_pos
        contiguous_calls = 0
        original_contiguous = torch.Tensor.contiguous

        def reject_contiguous(self: torch.Tensor, *args: object, **kwargs: object):
            nonlocal contiguous_calls
            contiguous_calls += 1
            raise AssertionError("pose validation silently materialized storage")

        values = torch.zeros((2, 2, 3), dtype=torch.float32)
        invalid_cases = (
            ("overlapping", values.as_strided((2, 2, 3), (3, 2, 1))),
            ("innermost stride", values[:, :1, :].permute(1, 0, 2).expand(2, 2, 3)),
        )
        torch.Tensor.contiguous = reject_contiguous  # type: ignore[method-assign]
        try:
            for label, pos in invalid_cases:
                with pytest.raises(ValueError, match=label):
                    caster.update_pose_tensor(
                        RayPoseTensorView(
                            body_names=("body_0", "body_1"),
                            pos_w=pos,
                            quat_w=original_pose.quat_w,
                        )
                    )
        finally:
            torch.Tensor.contiguous = original_contiguous  # type: ignore[method-assign]
        assert contiguous_calls == 0
        assert caster._tensor_pos is original

    def _two_body_pose(self) -> RayPoseTensorView:
        pos = torch.zeros((2, 2, 3), dtype=torch.float32)
        quat = torch.zeros((2, 2, 4), dtype=torch.float32)
        quat[..., 0] = 1.0
        return RayPoseTensorView(body_names=("body_0", "body_1"), pos_w=pos, quat_w=quat)

    def test_tensor_layout_and_input_validation_fail_closed(self) -> None:
        caster = self._tensor_caster()
        with pytest.raises(BackendError, match="already bound"):
            caster.bind_pose_layout(("body_0",))
        with pytest.raises(ValueError, match="must equal the scene body layout"):
            fresh = _materialized_caster()
            fresh.bind_pose_layout(("body_0", "extra"))

        with pytest.raises(ValueError, match="does not match bound layout"):
            caster.update_pose_tensor(
                RayPoseTensorView(
                    body_names=("other",),
                    pos_w=torch.zeros((2, 1, 3)),
                    quat_w=torch.zeros((2, 1, 4)),
                )
            )
        with pytest.raises(TypeError, match="float32"):
            pose = self._pose(2)
            caster.update_pose_tensor(
                RayPoseTensorView(
                    body_names=pose.body_names,
                    pos_w=pose.pos_w.double(),
                    quat_w=pose.quat_w,
                )
            )

        origins = torch.zeros((2, 3), dtype=torch.float32)
        directions = torch.zeros_like(origins)
        directions[:, 2] = -1.0
        with pytest.raises(ValueError, match="contiguous"):
            caster.trace_tensor(origins.t(), directions.t(), 2.0)
        with pytest.raises(TypeError, match="int64"):
            caster.trace_tensor(
                origins, directions, 2.0, env_ids=torch.tensor([0], dtype=torch.int32)
            )
        with pytest.raises(UnsupportedCapabilityError, match="tensor path"):
            caster.trace_tensor(origins, directions, 2.0, outputs=RayTraceOutputs(normal=True))

    def test_tensor_path_requires_bound_layout_and_rejects_cuda(self) -> None:
        caster = _materialized_caster()
        origins = torch.zeros((2, 3), dtype=torch.float32)
        directions = torch.zeros_like(origins)
        with pytest.raises(BackendError, match="bind_pose_layout"):
            caster.trace_tensor(origins, directions, 2.0)
        caster.bind_pose_layout(("body_0",))
        pose = self._pose(2)
        if torch.cuda.is_available():
            cuda_pose = RayPoseTensorView(
                body_names=pose.body_names,
                pos_w=pose.pos_w.cuda(),
                quat_w=pose.quat_w.cuda(),
            )
            with pytest.raises(ValueError, match="cpu"):
                caster.update_pose_tensor(cuda_pose)


class TestRayTensorCapabilityValidation:
    def test_tensor_capabilities_require_complete_profile(self) -> None:
        with pytest.raises(ValueError, match="supports_tensor_pose_sync requires"):
            RayCasterCapabilities(supports_tensor_pose_sync=True)
        with pytest.raises(ValueError, match="supports_tensor_rays requires"):
            RayCasterCapabilities(
                supports_pose_sync=True,
                supports_per_env_rays=False,
                supports_tensor_rays=True,
            )
        with pytest.raises(ValueError, match="torch_devices"):
            RayCasterCapabilities(
                supports_pose_sync=True,
                supports_per_env_rays=True,
                supports_tensor_pose_sync=True,
                supports_tensor_rays=True,
                supports_device_output=True,
            )
        with pytest.raises(ValueError, match="caller"):
            RayCasterCapabilities(
                supports_pose_sync=True,
                supports_per_env_rays=True,
                supports_tensor_pose_sync=True,
                supports_tensor_rays=True,
                supports_device_output=True,
                torch_devices=("cpu",),
                stream_event_ownership="caster",
            )

    def test_rejects_invalid_torch_device_declaration(self) -> None:
        with pytest.raises(ValueError, match="invalid ray caster Torch device"):
            RayCasterCapabilities(
                supports_pose_sync=True,
                supports_per_env_rays=True,
                supports_tensor_pose_sync=True,
                supports_tensor_rays=True,
                supports_device_output=True,
                torch_devices=("gpu",),
                stream_event_ownership="caller",
            )


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
