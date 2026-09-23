"""Pure-NumPy reference ray caster used by contract tests and examples.

The fake caster implements analytic ray-vs-primitive intersection for the
contract's primitive set (plane, sphere, box, cylinder, capsule, ellipsoid)
with no optional SDK. Mesh geoms are reserved for acceleration plugins and
are rejected fail closed at materialization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .entity_state import inverse_rotate_vector, rotate_vector
from .errors import BackendError, UnsupportedCapabilityError
from .ray_query import (
    RayCaster,
    RayCasterCapabilities,
    RayGeomType,
    RaySceneDescription,
    RayTraceOutputs,
    RayTraceResult,
)

_EPS = 1e-12

_POSITIVE_SIZE_DIMS = {
    RayGeomType.SPHERE: (0,),
    RayGeomType.BOX: (0, 1, 2),
    RayGeomType.CYLINDER: (0, 1),
    RayGeomType.CAPSULE: (0, 1),
    RayGeomType.ELLIPSOID: (0, 1, 2),
}


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two unit wxyz quaternion batches."""
    w1, x1, y1, z1 = np.moveaxis(first, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _sphere_candidate(
    origin: np.ndarray, direction: np.ndarray, radius: float, center_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the entry distance and local normal of a sphere centered on the z axis."""
    oc = origin.copy()
    oc[..., 2] -= center_z
    half_b = np.sum(oc * direction, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = half_b * half_b - c
    root = np.sqrt(np.maximum(disc, 0.0))
    t = np.where(disc >= 0.0, -half_b - root, np.inf)
    t = np.where(t < 0.0, -half_b + root, t)
    t = np.where((disc >= 0.0) & (t >= 0.0), t, np.inf)
    point = oc + t[..., None] * direction
    normal = point / radius
    return t, normal


def _intersect_plane(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    del size  # the contract plane is the infinite local z = 0 plane
    dz = direction[..., 2]
    parallel = np.abs(dz) <= _EPS
    t = np.where(parallel, np.inf, -origin[..., 2] / np.where(parallel, 1.0, dz))
    t = np.where(t >= 0.0, t, np.inf)
    normal = np.broadcast_to(np.array([0.0, 0.0, 1.0]), origin.shape)
    return t, normal.copy()


def _intersect_sphere(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _sphere_candidate(origin, direction, float(size[0]))


def _intersect_box(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    near_t = np.full(origin.shape[:-1], -np.inf)
    far_t = np.full(origin.shape[:-1], np.inf)
    for axis in range(3):
        da = direction[..., axis]
        oa = origin[..., axis]
        parallel = np.abs(da) < _EPS
        safe_da = np.where(parallel, 1.0, da)
        low = (-size[axis] - oa) / safe_da
        high = (size[axis] - oa) / safe_da
        slab_near = np.where(parallel, -np.inf, np.minimum(low, high))
        slab_far = np.where(parallel, np.inf, np.maximum(low, high))
        miss = parallel & (np.abs(oa) > size[axis])
        slab_near = np.where(miss, np.inf, slab_near)
        slab_far = np.where(miss, -np.inf, slab_far)
        near_t = np.maximum(near_t, slab_near)
        far_t = np.minimum(far_t, slab_far)
    entry = near_t >= 0.0
    t = np.where(entry, near_t, far_t)
    t = np.where((near_t <= far_t) & (t >= 0.0), t, np.inf)
    point = origin + t[..., None] * direction
    axis = np.argmax(np.abs(point) / size[None, None, :], axis=-1)
    normal = np.zeros_like(origin)
    np.put_along_axis(
        normal, axis[..., None], np.sign(np.take_along_axis(point, axis[..., None], -1)), -1
    )
    return t, normal


def _cylinder_side_candidate(
    origin: np.ndarray, direction: np.ndarray, radius: float, half_length: float
) -> tuple[np.ndarray, np.ndarray]:
    a = direction[..., 0] ** 2 + direction[..., 1] ** 2
    safe_a = np.where(a > _EPS, a, 1.0)
    half_b = origin[..., 0] * direction[..., 0] + origin[..., 1] * direction[..., 1]
    c = origin[..., 0] ** 2 + origin[..., 1] ** 2 - radius * radius
    disc = half_b * half_b - safe_a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t = np.where(disc >= 0.0, (-half_b - root) / safe_a, np.inf)
    t = np.where(t < 0.0, (-half_b + root) / safe_a, t)
    z = origin[..., 2] + t * direction[..., 2]
    valid = (a > _EPS) & (disc >= 0.0) & (t >= 0.0) & (np.abs(z) <= half_length)
    t = np.where(valid, t, np.inf)
    point = origin + t[..., None] * direction
    normal = np.stack((point[..., 0], point[..., 1], np.zeros_like(point[..., 2])), axis=-1)
    return t, normal / radius


def _cylinder_cap_candidate(
    origin: np.ndarray, direction: np.ndarray, radius: float, cap_z: float
) -> tuple[np.ndarray, np.ndarray]:
    dz = direction[..., 2]
    parallel = np.abs(dz) <= _EPS
    t = np.where(parallel, np.inf, (cap_z - origin[..., 2]) / np.where(parallel, 1.0, dz))
    point = origin + t[..., None] * direction
    radial = point[..., 0] ** 2 + point[..., 1] ** 2
    t = np.where((t >= 0.0) & (radial <= radius * radius), t, np.inf)
    normal = np.zeros_like(origin)
    normal[..., 2] = np.sign(cap_z)
    return t, normal


def _nearest_candidate(
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(t, normal)`` candidates to the nearest valid intersection."""
    ts = np.stack([candidate[0] for candidate in candidates], axis=-1)
    normals = np.stack([candidate[1] for candidate in candidates], axis=-2)
    index = np.argmin(ts, axis=-1)
    t = np.take_along_axis(ts, index[..., None], axis=-1)[..., 0]
    normal = np.take_along_axis(normals, index[..., None, None], axis=-2)[..., 0, :]
    return t, normal


def _intersect_cylinder(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    radius, half_length = float(size[0]), float(size[1])
    return _nearest_candidate(
        (
            _cylinder_side_candidate(origin, direction, radius, half_length),
            _cylinder_cap_candidate(origin, direction, radius, half_length),
            _cylinder_cap_candidate(origin, direction, radius, -half_length),
        )
    )


def _intersect_capsule(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    radius, half_length = float(size[0]), float(size[1])
    return _nearest_candidate(
        (
            _cylinder_side_candidate(origin, direction, radius, half_length),
            _sphere_candidate(origin, direction, radius, center_z=half_length),
            _sphere_candidate(origin, direction, radius, center_z=-half_length),
        )
    )


def _intersect_ellipsoid(
    origin: np.ndarray, direction: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaled_origin = origin / size
    scaled_direction = direction / size
    norm = np.linalg.norm(scaled_direction, axis=-1, keepdims=True)
    unit_direction = scaled_direction / np.where(norm > _EPS, norm, 1.0)
    t, _ = _sphere_candidate(scaled_origin, unit_direction, 1.0)
    t = t / norm[..., 0]
    point = origin + t[..., None] * direction
    gradient = point / (size * size)
    normal = gradient / np.where(
        np.linalg.norm(gradient, axis=-1, keepdims=True) > _EPS,
        np.linalg.norm(gradient, axis=-1, keepdims=True),
        1.0,
    )
    return t, normal


_INTERSECT_FNS = {
    RayGeomType.PLANE: _intersect_plane,
    RayGeomType.SPHERE: _intersect_sphere,
    RayGeomType.BOX: _intersect_box,
    RayGeomType.CYLINDER: _intersect_cylinder,
    RayGeomType.CAPSULE: _intersect_capsule,
    RayGeomType.ELLIPSOID: _intersect_ellipsoid,
}


class FakeRayCaster(RayCaster):
    """A dependency-free analytic ray caster with deterministic results.

    Distances, hit flags, and the optional outputs are exact analytic
    intersections computed in float64. The caster declares host readback,
    pose sync, per-environment rays, and every optional output; mesh geoms
    and device-resident output remain unsupported and fail closed.
    """

    caster_type = "fake"

    _ray_capabilities = RayCasterCapabilities(
        supports_pose_sync=True,
        supports_per_env_rays=True,
        supports_host_readback=True,
        supports_hit_point=True,
        supports_normal=True,
        supports_geom_id=True,
        supports_body_id=True,
    )

    def __init__(self, num_envs: int = 1, num_rays: int = 1) -> None:
        for name, value in (("num_envs", num_envs), ("num_rays", num_rays)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._num_envs = num_envs
        self._num_rays = num_rays
        self._scene: RaySceneDescription | None = None
        self._body_pos = np.zeros((0, 0, 3), dtype=np.float64)
        self._body_quat = np.zeros((0, 0, 4), dtype=np.float64)
        self._closed = False

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_rays(self) -> int:
        return self._num_rays

    def materialize(self, scene: RaySceneDescription) -> None:
        self._require_open()
        if self._scene is not None:
            raise BackendError("fake ray caster is already materialized")
        if not isinstance(scene, RaySceneDescription):
            raise TypeError("scene must be a RaySceneDescription")
        if RayGeomType.MESH in scene.geom_types:
            raise UnsupportedCapabilityError(
                "fake ray caster does not support mesh geoms "
                "(geom type 'mesh' is reserved for acceleration plugins); "
                "use primitive geom types only"
            )
        for raw_type, size in zip(scene.geom_types, scene.geom_sizes):
            geom_type = RayGeomType(raw_type)
            required = _POSITIVE_SIZE_DIMS.get(geom_type, ())
            if any(float(size[dim]) <= 0.0 for dim in required):
                raise ValueError(
                    f"fake ray caster requires positive sizes for geom type "
                    f"'{geom_type.value}' dims {required}, got {tuple(size)}"
                )
        self._scene = scene
        self._body_pos = np.zeros((self._num_envs, scene.num_bodies, 3), dtype=np.float64)
        self._body_quat = np.zeros((self._num_envs, scene.num_bodies, 4), dtype=np.float64)
        self._body_quat[..., 0] = 1.0

    def update_pose(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self._require_open()
        scene = self._require_materialized()
        rows = self._resolve_selected_rows(env_ids)
        pos = np.asarray(body_pos, dtype=np.float64)
        quat = np.asarray(body_quat, dtype=np.float64)
        for name, value in (("body_pos", pos), ("body_quat", quat)):
            expected = (rows.size, scene.num_bodies, 3 if name == "body_pos" else 4)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("body_quat requires unit wxyz quaternions")
        self._body_pos[rows] = pos
        self._body_quat[rows] = quat

    def trace(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        max_distance: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
        outputs: RayTraceOutputs | None = None,
    ) -> RayTraceResult:
        self._require_open()
        scene = self._require_materialized()
        request = self._check_trace_outputs(outputs)
        rows = self._resolve_selected_rows(env_ids)
        limit = self._resolve_max_distance(max_distance)
        origins, directions = self._resolve_ray_batch(ray_origins, ray_directions, rows.size)

        count = rows.size
        best_t = np.full((count, self._num_rays), np.inf)
        best_geom = np.full((count, self._num_rays), -1, dtype=np.intp)
        best_normal = np.zeros((count, self._num_rays, 3), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            self._trace_scene(scene, rows, origins, directions, best_t, best_geom, best_normal)

        # Missed rays keep distance clipped to max_distance; normals oppose
        # the ray direction on both entry and exit hits.
        hit = best_t <= limit
        distance = np.where(hit, best_t, limit)
        flip = np.sum(best_normal * directions, axis=-1) > 0.0
        best_normal = np.where(flip[..., None], -best_normal, best_normal)

        hit_point = None
        normal = None
        geom_id = None
        body_id = None
        if request.hit_point:
            hit_point = origins + distance[..., None] * directions
        if request.normal:
            normal = best_normal
        if request.geom_id:
            geom_id = np.where(hit, best_geom, -1)
        if request.body_id:
            body_id = np.where(hit, scene.geom_body_ids[np.maximum(best_geom, 0)], -1)
        return RayTraceResult(
            distance=distance,
            hit=hit,
            hit_point=hit_point,
            normal=normal,
            geom_id=geom_id,
            body_id=body_id,
        )

    def _trace_scene(
        self,
        scene: RaySceneDescription,
        rows: np.ndarray,
        origins: np.ndarray,
        directions: np.ndarray,
        best_t: np.ndarray,
        best_geom: np.ndarray,
        best_normal: np.ndarray,
    ) -> None:
        """Intersect every ray with every geom, keeping the nearest hit.

        Intersection helpers intentionally produce inf/nan on misses; the
        caller suppresses the expected floating-point warnings.
        """
        count = rows.size
        for geom_index, raw_type in enumerate(scene.geom_types):
            geom_type = RayGeomType(raw_type)
            body_id = int(scene.geom_body_ids[geom_index])
            body_quat = self._body_quat[rows, body_id]
            geom_quat = _quat_multiply(body_quat, scene.geom_local_quat[geom_index])
            geom_pos = self._body_pos[rows, body_id] + rotate_vector(
                body_quat, scene.geom_local_pos[geom_index]
            )
            quat = np.broadcast_to(geom_quat[:, None, :], (count, self._num_rays, 4))
            local_origin = inverse_rotate_vector(quat, origins - geom_pos[:, None, :])
            local_direction = inverse_rotate_vector(quat, directions)
            t, local_normal = _INTERSECT_FNS[geom_type](
                local_origin, local_direction, scene.geom_sizes[geom_index]
            )
            better = t < best_t
            best_t[...] = np.where(better, t, best_t)
            best_geom[...] = np.where(better, geom_index, best_geom)
            world_normal = rotate_vector(quat, local_normal)
            best_normal[...] = np.where(better[..., None], world_normal, best_normal)

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise BackendError("fake ray caster is closed")

    def _require_materialized(self) -> RaySceneDescription:
        if self._scene is None:
            raise BackendError("fake ray caster must be materialized before querying")
        return self._scene


__all__ = ["FakeRayCaster"]
