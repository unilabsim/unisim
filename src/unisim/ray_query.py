"""Backend-neutral ray-query plugin contract.

This module defines the public surface implemented by ray caster plugins such
as ``uniray``: a scene description consumed on the cold path, a fixed-shape
batched ``trace`` operation, fine-grained capability declarations, and a
fail-closed output negotiation helper. The contract is deliberately array-only:
``mujoco.MjModel``/``mujoco.MjData``, Warp kernels or arrays, CUDA pointers,
and every other backend-private type must never cross this boundary. Real
adapters translate their native scene and pose state into these NumPy
containers on their own side of the boundary.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .entity_state import selected_state_rows
from .errors import UnsupportedCapabilityError


class RayGeomType(str, Enum):
    """Primitive geometry kinds addressable by the ray-query contract.

    ``MESH`` is reserved for mesh-acceleration plugins; the reference
    implementation rejects it fail closed.
    """

    PLANE = "plane"
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    ELLIPSOID = "ellipsoid"
    MESH = "mesh"


def ray_geom_type(native_type: int | str) -> RayGeomType:
    """Map a public MuJoCo-style geom code or contract name fail closed."""

    try:
        return RayGeomType(native_type)
    except ValueError:
        pass
    mujoco_types = {
        0: RayGeomType.PLANE,
        2: RayGeomType.SPHERE,
        3: RayGeomType.CAPSULE,
        4: RayGeomType.ELLIPSOID,
        5: RayGeomType.CYLINDER,
        6: RayGeomType.BOX,
        7: RayGeomType.MESH,
    }
    if native_type == 1:
        raise ValueError("MuJoCo hfield geoms (native type 1) are unsupported by ray_geom_type")
    if native_type not in mujoco_types:
        raise ValueError(f"unknown ray geom type: {native_type!r}")
    return mujoco_types[native_type]


@dataclass(frozen=True)
class RaySceneDescription:
    """Backend-neutral structure-of-arrays scene consumed by ``materialize``.

    Each geom is attached to one body and expressed in that body's local
    frame. ``geom_sizes`` follow the MuJoCo-style primitive conventions:
    sphere ``(radius, 0, 0)``; box half extents ``(x, y, z)``; cylinder and
    capsule ``(radius, half_length, 0)`` along the local ``+z`` axis;
    ellipsoid semi-axes ``(x, y, z)``. Planes are the infinite local ``z = 0``
    plane with normal ``+z`` and ignore their sizes. Quaternions are unit
    ``wxyz``. All arrays are detached and write-protected on construction.
    """

    num_bodies: int
    geom_types: tuple[RayGeomType | str, ...]
    geom_sizes: np.ndarray
    geom_local_pos: np.ndarray
    geom_local_quat: np.ndarray
    geom_body_ids: np.ndarray
    body_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.num_bodies, bool) or not isinstance(self.num_bodies, int):
            raise TypeError("ray scene num_bodies must be an integer")
        if self.num_bodies <= 0:
            raise ValueError("ray scene num_bodies must be positive")
        if self.body_names:
            if len(self.body_names) != self.num_bodies:
                raise ValueError(
                    "ray scene body_names must match num_bodies exactly; "
                    f"expected {self.num_bodies}, got {len(self.body_names)}"
                )
            if len(set(self.body_names)) != len(self.body_names):
                raise ValueError(f"ray scene body_names must be unique: {self.body_names}")
        else:
            object.__setattr__(
                self, "body_names", tuple(f"body_{index}" for index in range(self.num_bodies))
            )
        if any(not isinstance(name, str) or not name for name in self.body_names):
            raise TypeError("ray scene body_names must contain non-empty strings")
        if isinstance(self.geom_types, (str, bytes)) or not isinstance(self.geom_types, Sequence):
            raise TypeError("ray scene geom_types must be a sequence of geom type names")
        try:
            geom_types = tuple(ray_geom_type(value) for value in self.geom_types)
        except ValueError as error:
            raise ValueError(f"unknown ray scene geom type: {error}") from None
        object.__setattr__(self, "geom_types", geom_types)
        count = len(geom_types)
        sizes = self._float_array("geom_sizes", self.geom_sizes, (count, 3))
        if np.any(sizes < 0.0):
            raise ValueError("ray scene geom_sizes must be non-negative")
        object.__setattr__(self, "geom_sizes", sizes)
        object.__setattr__(
            self,
            "geom_local_pos",
            self._float_array("geom_local_pos", self.geom_local_pos, (count, 3)),
        )
        quat = self._float_array("geom_local_quat", self.geom_local_quat, (count, 4))
        if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("ray scene geom_local_quat requires unit wxyz quaternions")
        object.__setattr__(self, "geom_local_quat", quat)
        body_ids = np.array(self.geom_body_ids, copy=True)
        if body_ids.ndim != 1 or body_ids.shape[0] != count or body_ids.dtype.kind not in "iu":
            raise ValueError(
                f"ray scene geom_body_ids must be an integer array of shape ({count},)"
            )
        if count and (np.any(body_ids < 0) or np.any(body_ids >= self.num_bodies)):
            raise ValueError("ray scene geom_body_ids are outside [0, num_bodies)")
        body_ids = body_ids.astype(np.intp)
        body_ids.setflags(write=False)
        object.__setattr__(self, "geom_body_ids", body_ids)

    @staticmethod
    def _float_array(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        array = np.array(value, dtype=np.float64, copy=True)
        if array.shape != shape:
            raise ValueError(f"ray scene {name} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"ray scene {name} must be finite")
        array.setflags(write=False)
        return array

    @property
    def num_geoms(self) -> int:
        """Number of geoms in the scene description."""
        return len(self.geom_types)


@dataclass(frozen=True)
class RayMeshDescription:
    """One detached, deduplicated triangle-mesh asset in geom-local coordinates.

    ``asset_id`` identifies the mesh asset only. A geom instance is identified by
    its index and name in :class:`RayCollisionDescription`; multiple geom
    instances may reference one mesh asset. Identities are stable for the
    descriptor/materialization lifecycle, not across rebuilds or backends.
    Triangles must have positive area so face normals and ray hits are defined.
    """

    asset_id: str
    points: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise TypeError("ray mesh asset_id must be a non-empty string")
        points = np.array(self.points, dtype=np.float32, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError("ray mesh points must be a finite array of shape (V, 3) with V >= 3")
        if not np.isfinite(points).all():
            raise ValueError("ray mesh points must be finite")
        indices = np.array(self.indices, dtype=np.int32, copy=True)
        if indices.ndim != 1 or indices.size < 3 or indices.size % 3 != 0:
            raise ValueError("ray mesh indices must be a flat triangle array of shape (3F,)")
        if indices.min() < 0 or indices.max() >= points.shape[0]:
            raise ValueError("ray mesh indices are outside [0, V)")
        triangles = points[indices.reshape(-1, 3)]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        areas = np.linalg.norm(np.cross(edge1, edge2), axis=1)
        if np.any(areas <= 0.0):
            raise ValueError("ray mesh triangles must have positive area")
        points.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class RayCollisionDescription:
    """Detached collision geometry companion to :class:`RaySceneDescription`.

    The companion keeps the ray-caster scene contract compact while exposing the
    mesh topology, stable geom/mesh identities, semantic filter tags, and body
    ancestry needed by perception callers. ``body_parent_ids`` describes ancestry
    within the selected public body axis; an omitted native ancestor becomes a
    selected-axis root, so callers that need complete ancestor exclusion should
    select the relevant ancestor chain. No native model, mesh, Warp array,
    device pointer, USD prim, or other backend-private object may appear here.
    """

    scene: RaySceneDescription
    geom_names: tuple[str, ...] = ()
    geom_tags: tuple[tuple[str, ...], ...] = ()
    meshes: tuple[RayMeshDescription, ...] = ()
    geom_mesh_ids: np.ndarray | None = None
    body_parent_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scene, RaySceneDescription):
            raise TypeError("ray collision scene must be a RaySceneDescription")
        count = self.scene.num_geoms

        if self.geom_names:
            if len(self.geom_names) != count:
                raise ValueError(
                    f"ray collision geom_names must match num_geoms exactly; "
                    f"expected {count}, got {len(self.geom_names)}"
                )
            if len(set(self.geom_names)) != count:
                raise ValueError(f"ray collision geom_names must be unique: {self.geom_names}")
            names = tuple(self.geom_names)
        else:
            names = tuple(f"geom_{index}" for index in range(count))
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("ray collision geom_names must contain non-empty strings")
        object.__setattr__(self, "geom_names", names)

        if self.geom_tags:
            if len(self.geom_tags) != count:
                raise ValueError(
                    f"ray collision geom_tags must match num_geoms exactly; "
                    f"expected {count}, got {len(self.geom_tags)}"
                )
            normalized_tags = []
            for tags in self.geom_tags:
                if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
                    raise TypeError("ray collision geom_tags must be a sequence of tag sequences")
                values = tuple(tags)
                if any(not isinstance(tag, str) or not tag for tag in values):
                    raise TypeError("ray collision geom tags must be non-empty strings")
                if len(set(values)) != len(values):
                    raise ValueError(f"ray collision geom tags must be unique: {values}")
                normalized_tags.append(tuple(sorted(values)))
            object.__setattr__(self, "geom_tags", tuple(normalized_tags))
        else:
            object.__setattr__(self, "geom_tags", tuple(() for _ in range(count)))

        meshes = tuple(self.meshes)
        if any(not isinstance(mesh, RayMeshDescription) for mesh in meshes):
            raise TypeError("ray collision meshes must contain RayMeshDescription values")
        asset_ids = tuple(mesh.asset_id for mesh in meshes)
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError(f"ray mesh asset identities must be unique: {asset_ids}")
        object.__setattr__(self, "meshes", meshes)

        if self.geom_mesh_ids is None:
            mesh_ids = np.full(count, -1, dtype=np.intp)
        else:
            mesh_ids = np.array(self.geom_mesh_ids, copy=True)
            if mesh_ids.ndim != 1 or mesh_ids.shape[0] != count or mesh_ids.dtype.kind not in "iu":
                raise ValueError(
                    f"ray collision geom_mesh_ids must be an integer array of shape ({count},)"
                )
            mesh_ids = mesh_ids.astype(np.intp)
        if count and (mesh_ids.min() < -1 or mesh_ids.max() >= len(meshes)):
            raise ValueError("ray collision geom_mesh_ids are outside [-1, num_meshes)")
        for geom_index, geom_type in enumerate(self.scene.geom_types):
            mesh_index = int(mesh_ids[geom_index])
            if (geom_type == RayGeomType.MESH) != (mesh_index >= 0):
                raise ValueError(
                    "ray collision mesh geoms must reference a mesh and non-mesh geoms must use -1"
                )
        mesh_ids.setflags(write=False)
        object.__setattr__(self, "geom_mesh_ids", mesh_ids)

        if self.body_parent_ids is None:
            body_parents = None
        else:
            body_parents = np.array(self.body_parent_ids, copy=True)
            num_bodies = self.scene.num_bodies
            if (
                body_parents.ndim != 1
                or body_parents.shape[0] != num_bodies
                or body_parents.dtype.kind not in "iu"
            ):
                raise ValueError(
                    f"ray collision body_parent_ids must be an integer array of shape "
                    f"({num_bodies},)"
                )
            body_parents = body_parents.astype(np.intp)
            if body_parents.size and (body_parents.min() < -1 or body_parents.max() >= num_bodies):
                raise ValueError("ray collision body_parent_ids are outside [-1, num_bodies)")
            for body_index in range(body_parents.shape[0]):
                cursor = body_index
                visited: set[int] = set()
                while cursor >= 0:
                    if cursor in visited:
                        raise ValueError("ray collision body_parent_ids must be acyclic")
                    visited.add(cursor)
                    cursor = int(body_parents[cursor])
            body_parents.setflags(write=False)
        object.__setattr__(self, "body_parent_ids", body_parents)

    @property
    def num_geoms(self) -> int:
        """Number of geom instances in the companion descriptor."""
        return self.scene.num_geoms

    @property
    def num_meshes(self) -> int:
        """Number of deduplicated mesh assets in the companion descriptor."""
        return len(self.meshes)


@dataclass(frozen=True)
class RayTraceFilter:
    """Immutable, call-static geometry filter expressed with public identities.

    Constraints combine with AND semantics. ``allowed_geom_ids`` and
    ``allowed_tags`` are allowlists (an empty tuple allows no geom), while the
    body fields are exclusions. ``excluded_body_ids`` removes geoms directly
    owned by those bodies; ``excluded_ancestor_body_ids`` removes a geom when
    its owning body or any selected ancestor is listed and requires collision
    descriptor body ancestry. Filters are identical for every environment and
    ray in one trace; dynamic masks are outside this contract.
    """

    allowed_geom_ids: tuple[int, ...] | Sequence[int] | None = None
    allowed_tags: tuple[str, ...] | Sequence[str] | None = None
    excluded_body_ids: tuple[int, ...] | Sequence[int] | None = None
    excluded_ancestor_body_ids: tuple[int, ...] | Sequence[int] | None = None

    def __post_init__(self) -> None:
        for name in (
            "allowed_geom_ids",
            "excluded_body_ids",
            "excluded_ancestor_body_ids",
        ):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, None)
                continue
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"ray trace filter {name} must be an integer sequence or None")
            ids = tuple(value)
            if any(
                isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in ids
            ):
                raise TypeError(f"ray trace filter {name} must contain integers")
            object.__setattr__(self, name, tuple(sorted(int(item) for item in ids)))
        if self.allowed_tags is None:
            object.__setattr__(self, "allowed_tags", None)
        else:
            if isinstance(self.allowed_tags, (str, bytes)) or not isinstance(
                self.allowed_tags, Sequence
            ):
                raise TypeError("ray trace filter allowed_tags must be a string sequence or None")
            tags = tuple(self.allowed_tags)
            if any(not isinstance(tag, str) or not tag for tag in tags):
                raise TypeError("ray trace filter allowed_tags must contain non-empty strings")
            if len(set(tags)) != len(tags):
                raise ValueError(f"ray trace filter allowed_tags must be unique: {tags}")
            object.__setattr__(self, "allowed_tags", tuple(sorted(tags)))

    def has_constraints(self) -> bool:
        """Return whether any filter constraint is present."""
        return (
            self.allowed_geom_ids is not None
            or self.allowed_tags is not None
            or self.excluded_body_ids is not None
            or self.excluded_ancestor_body_ids is not None
        )

    def geom_mask(self, collision: RayCollisionDescription) -> np.ndarray:
        """Resolve this filter to a compact boolean geom mask.

        This is the cold-path bridge from public identities and tags to the mask
        consumed by caster kernels. Unknown identities or tags, and ancestor
        exclusion without body ancestry, fail closed rather than producing a
        silent partial scene.
        """

        if not isinstance(collision, RayCollisionDescription):
            raise TypeError("ray trace filter requires a RayCollisionDescription")
        scene = collision.scene
        mask = np.ones(scene.num_geoms, dtype=np.bool_)

        if self.allowed_geom_ids is not None:
            allowed = np.asarray(self.allowed_geom_ids, dtype=np.intp)
            if allowed.size and (allowed.min() < 0 or allowed.max() >= scene.num_geoms):
                raise ValueError("ray trace allowed_geom_ids are outside [0, num_geoms)")
            mask &= np.isin(np.arange(scene.num_geoms, dtype=np.intp), allowed)

        if self.allowed_tags is not None:
            known_tags = {tag for tags in collision.geom_tags for tag in tags}
            unknown = sorted(set(self.allowed_tags) - known_tags)
            if unknown:
                raise ValueError(f"ray trace allowed_tags are absent from the scene: {unknown}")
            allowed = set(self.allowed_tags)
            mask &= np.asarray(
                [bool(allowed.intersection(tags)) for tags in collision.geom_tags], dtype=np.bool_
            )

        if self.excluded_body_ids is not None:
            excluded = np.asarray(self.excluded_body_ids, dtype=np.intp)
            if excluded.size and (excluded.min() < 0 or excluded.max() >= scene.num_bodies):
                raise ValueError("ray trace excluded_body_ids are outside [0, num_bodies)")
            mask &= ~np.isin(scene.geom_body_ids, excluded)

        if self.excluded_ancestor_body_ids is not None:
            if collision.body_parent_ids is None:
                raise UnsupportedCapabilityError(
                    "ray trace ancestor exclusion requires RayCollisionDescription.body_parent_ids"
                )
            excluded = set(self.excluded_ancestor_body_ids)
            if any(body_id < 0 or body_id >= scene.num_bodies for body_id in excluded):
                raise ValueError("ray trace excluded_ancestor_body_ids are outside [0, num_bodies)")
            parents = collision.body_parent_ids
            excluded_owners = np.zeros(scene.num_bodies, dtype=np.bool_)
            for body_id in range(scene.num_bodies):
                cursor = body_id
                while cursor >= 0:
                    if cursor in excluded:
                        excluded_owners[body_id] = True
                        break
                    if excluded_owners[cursor]:
                        excluded_owners[body_id] = True
                        break
                    cursor = int(parents[cursor])
            mask &= ~excluded_owners[scene.geom_body_ids]
        return mask


@dataclass(frozen=True)
class RayCasterCapabilities:
    """Fine-grained ray caster capabilities surfaced through the contract.

    ``supports_pose_sync`` covers pose updates through :meth:`RayCaster.update_pose`
    after materialization. ``supports_per_env_rays`` covers the per-environment
    ray profile of :meth:`RayCaster.trace` (3-D ray batches); every caster
    accepts the shared 2-D profile. ``supports_host_readback`` covers the
    NumPy host result path of ``trace``. ``supports_tensor_pose_sync`` and
    ``supports_tensor_rays`` cover the Torch-native profile of
    :meth:`RayCaster.update_pose_tensor` and :meth:`RayCaster.trace_tensor`;
    ``supports_device_output`` declares its persistent caster-owned output
    buffers. ``torch_devices`` enumerates accepted Torch device families or
    exact CUDA indices using the same spelling as the backend tensor contract.
    ``stream_event_ownership`` names the ordering policy for the tensor path;
    the current profile is ``caller`` only. The remaining flags declare the
    optional result fields a caster can serve.
    """

    supports_pose_sync: bool = False
    supports_geometry_sync: bool = False
    supports_per_env_rays: bool = False
    supports_host_readback: bool = False
    supports_device_output: bool = False
    supports_tensor_pose_sync: bool = False
    supports_tensor_geometry_sync: bool = False
    supports_tensor_rays: bool = False
    torch_devices: tuple[str, ...] = ()
    stream_event_ownership: str | None = None
    supports_hit_point: bool = False
    supports_host_normal: bool = False
    supports_tensor_normal: bool = False
    supports_geom_id: bool = False
    supports_body_id: bool = False
    supports_static_filter: bool = False
    supports_min_distance: bool = False

    def __post_init__(self) -> None:
        if self.supports_tensor_pose_sync and not self.supports_pose_sync:
            raise ValueError(
                "supports_tensor_pose_sync requires supports_pose_sync; tensor "
                "pose sync is a transport profile of the same pose lifecycle"
            )
        if self.supports_geometry_sync and not self.supports_pose_sync:
            raise ValueError("supports_geometry_sync requires supports_pose_sync")
        if self.supports_tensor_geometry_sync and not (
            self.supports_geometry_sync and self.supports_tensor_pose_sync
        ):
            raise ValueError(
                "supports_tensor_geometry_sync requires both geometry and tensor pose sync"
            )
        if self.supports_tensor_rays and not self.supports_per_env_rays:
            raise ValueError(
                "supports_tensor_rays requires supports_per_env_rays; the tensor "
                "path accepts both shared and per-environment ray batches"
            )
        if self.supports_tensor_normal and not self.supports_tensor_rays:
            raise ValueError("supports_tensor_normal requires supports_tensor_rays")
        if self.supports_host_normal and not self.supports_host_readback:
            raise ValueError("supports_host_normal requires supports_host_readback")
        if self.supports_tensor_normal and not self.supports_device_output:
            raise ValueError("supports_tensor_normal requires supports_device_output")
        if self.supports_device_output and not (
            self.supports_tensor_pose_sync and self.supports_tensor_rays
        ):
            raise ValueError("supports_device_output requires the complete tensor ray profile")
        if bool(self.torch_devices) != self.supports_device_output:
            raise ValueError(
                "torch_devices must be declared if and only if device output is declared"
            )
        if self.stream_event_ownership not in (None, "caller"):
            raise ValueError("ray tensor stream_event_ownership must be 'caller' or None")
        if self.supports_device_output and self.stream_event_ownership is None:
            raise ValueError("device output requires caller stream_event_ownership")
        if len(set(self.torch_devices)) != len(self.torch_devices):
            raise ValueError(f"ray caster Torch devices must be unique: {self.torch_devices}")
        for device in self.torch_devices:
            family, separator, index = device.partition(":")
            if family not in {"cpu", "cuda"} or (separator and not index.isdigit()):
                raise ValueError(f"invalid ray caster Torch device declaration: {device!r}")


@dataclass(frozen=True)
class RayTraceOutputs:
    """Optional outputs requested from one :meth:`RayCaster.trace` call.

    The minimal result (``distance`` and ``hit``) is always returned; each
    flag here must be declared by :class:`RayCasterCapabilities` or the call
    fails closed with :class:`UnsupportedCapabilityError`.
    """

    hit_point: bool = False
    normal: bool = False
    geom_id: bool = False
    body_id: bool = False


def require_ray_trace_outputs(
    capabilities: RayCasterCapabilities,
    outputs: RayTraceOutputs,
    *,
    tensor: bool = False,
) -> None:
    """Fail closed when ``outputs`` requests an undeclared optional field."""
    normal_supported = (
        capabilities.supports_tensor_normal if tensor else capabilities.supports_host_normal
    )
    requested = (
        ("hit_point", outputs.hit_point, capabilities.supports_hit_point),
        ("normal", outputs.normal, normal_supported),
        ("geom_id", outputs.geom_id, capabilities.supports_geom_id),
        ("body_id", outputs.body_id, capabilities.supports_body_id),
    )
    unsupported = tuple(name for name, wanted, declared in requested if wanted and not declared)
    if unsupported:
        raise UnsupportedCapabilityError(
            f"ray trace outputs {unsupported} are not declared by this caster "
            f"({capabilities}); request only outputs declared in "
            "RayCasterCapabilities or pick a caster plugin that supports them"
        )


def require_ray_trace_query(
    capabilities: RayCasterCapabilities,
    min_distance: float,
    ray_filter: RayTraceFilter | None,
) -> None:
    """Fail closed on rich-query requests not declared by the caster.

    An empty :class:`RayTraceFilter` is an explicit no-op and does not require
    ``supports_static_filter``; every constraint field must be ``None``.
    """
    if ray_filter is not None and ray_filter.has_constraints():
        if not capabilities.supports_static_filter:
            raise UnsupportedCapabilityError(
                "static ray filters are not declared by this caster "
                f"({capabilities}); use a caster with supports_static_filter"
            )
    if min_distance != 0.0 and not capabilities.supports_min_distance:
        raise UnsupportedCapabilityError(
            "min_distance is not declared by this caster "
            f"({capabilities}); use a caster with supports_min_distance"
        )


@dataclass(frozen=True)
class RayTraceResult:
    """One batched ray query result.

    ``distance`` is the distance to the nearest hit along each unit ray,
    clipped to the request's ``max_distance``; missed rays report
    ``distance == max_distance`` with ``hit == False``. Optional fields are
    ``None`` unless requested and declared. ``hit_point`` and ``normal`` are
    world-space; normals are oriented against the ray direction. ``geom_id``
    indexes :class:`RaySceneDescription` geom order and ``body_id`` indexes
    its bodies; both are ``-1`` on missed rays. Optional fields are only
    meaningful where ``hit`` is ``True``.
    """

    distance: np.ndarray
    hit: np.ndarray
    hit_point: np.ndarray | None = None
    normal: np.ndarray | None = None
    geom_id: np.ndarray | None = None
    body_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance)
        if distance.ndim != 2 or not np.issubdtype(distance.dtype, np.floating):
            raise ValueError(
                f"ray trace distance must be a 2-D floating array, got {distance.shape}"
            )
        if not np.isfinite(distance).all():
            raise ValueError("ray trace distance must be finite")
        hit = np.asarray(self.hit)
        if hit.shape != distance.shape or hit.dtype != np.bool_:
            raise ValueError("ray trace hit must be a bool array matching the distance shape")
        rows, num_rays = distance.shape
        for name, value in (("hit_point", self.hit_point), ("normal", self.normal)):
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape != (rows, num_rays, 3) or not np.issubdtype(array.dtype, np.floating):
                raise ValueError(
                    f"ray trace {name} must be a floating array of shape "
                    f"({rows}, {num_rays}, 3), got {array.shape}"
                )
            if np.any(hit) and not np.isfinite(array[hit]).all():
                raise ValueError(f"ray trace {name} must be finite on hit rays")
        for name, value in (("geom_id", self.geom_id), ("body_id", self.body_id)):
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape != (rows, num_rays) or array.dtype.kind not in "iu":
                raise ValueError(
                    f"ray trace {name} must be an integer array of shape "
                    f"({rows}, {num_rays}), got {array.shape}"
                )
            if np.any(array[hit] < 0):
                raise ValueError(f"ray trace {name} must be non-negative on hit rays")


@dataclass(frozen=True)
class RayPoseTensorView:
    """Caller-owned Torch pose source bound by name and storage identity.

    ``pos_w`` and ``quat_w`` are the public full-batch views named by
    ``body_names`` with shapes ``(num_envs, num_bodies, 3)`` and
    ``(num_envs, num_bodies, 4)`` in ``wxyz`` order. The caster aliases these
    tensors in place; it never copies, reorders quaternions, or extends their
    lifetime. They must remain float32 Torch tensors on the bound device with
    positive non-overlapping strides and innermost stride one (packed packet
    column blocks need not be globally contiguous) until the next tensor pose
    update or ``close``.
    """

    body_names: tuple[str, ...]
    pos_w: Any
    quat_w: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.body_names, (str, bytes))
            or not isinstance(self.body_names, Sequence)
            or not self.body_names
        ):
            raise TypeError("RayPoseTensorView body_names must be a non-empty sequence")
        names = tuple(self.body_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("RayPoseTensorView body names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"RayPoseTensorView body names must be unique: {names}")
        object.__setattr__(self, "body_names", names)


@dataclass(frozen=True)
class RaySiteFrameDescription:
    """Detached named-site-to-body transform."""

    site_name: str
    body_name: str
    pos: np.ndarray
    quat: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("site_name", self.site_name), ("body_name", self.body_name)):
            if not isinstance(value, str) or not value:
                raise TypeError(f"RaySiteFrameDescription {name} must be non-empty")
        pos = np.asarray(self.pos, dtype=np.float64)
        quat = np.asarray(self.quat, dtype=np.float64)
        if pos.shape != (3,) or quat.shape != (4,):
            raise ValueError("RaySiteFrameDescription pos/quat must be vectors (3)/(4)")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError("RaySiteFrameDescription pos/quat must be finite")
        if not np.allclose(np.linalg.norm(quat), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("RaySiteFrameDescription quat must be unit wxyz")
        pos = pos.copy()
        quat = quat.copy()
        pos.setflags(write=False)
        quat.setflags(write=False)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "quat", quat)


@dataclass(frozen=True)
class RayGeometryState:
    """Read-only per-row geometry aligned to a collision descriptor."""

    num_envs: int
    geom_names: tuple[str, ...]
    geom_sizes: np.ndarray
    geom_local_pos: np.ndarray
    geom_local_quat: np.ndarray
    geom_enabled: np.ndarray
    geom_mesh_ids: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(self.geom_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("RayGeometryState geom_names must be non-empty and unique")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TypeError("RayGeometryState num_envs must be an integer")
        if self.num_envs <= 0:
            raise ValueError("RayGeometryState num_envs must be positive")
        specs = (
            ("geom_sizes", 3, np.float64),
            ("geom_local_pos", 3, np.float64),
            ("geom_local_quat", 4, np.float64),
            ("geom_enabled", 1, np.bool_),
            ("geom_mesh_ids", 1, np.int32),
        )
        for name, width, dtype in specs:
            raw = np.asarray(getattr(self, name))
            expected = (
                (self.num_envs, len(names)) if width == 1 else (self.num_envs, len(names), width)
            )
            if raw.shape != expected:
                raise ValueError(f"RayGeometryState {name} must have shape {expected}")
            value = raw.astype(dtype, copy=True)
            if dtype == np.float64 and not np.isfinite(value).all():
                raise ValueError(f"RayGeometryState {name} must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.geom_sizes < 0.0):
            raise ValueError("RayGeometryState geom_sizes must be non-negative")
        if not np.allclose(
            np.linalg.norm(self.geom_local_quat, axis=-1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("RayGeometryState geom_local_quat must be unit wxyz")
        if np.any(self.geom_mesh_ids < -1):
            raise ValueError("RayGeometryState geom_mesh_ids must be >= -1")


@dataclass(frozen=True)
class RayGeometryTensorView:
    """Caller-owned full-batch Torch geometry tensors."""

    geom_sizes: Any
    geom_local_pos: Any
    geom_local_quat: Any
    geom_enabled: Any
    geom_mesh_ids: Any


@dataclass(frozen=True)
class RayTraceTensorResult:
    """Persistent Torch views returned by the tensor-native ray profile.

    Fields are caster-owned views reused by the next :meth:`trace_tensor`
    call. Construction performs shape/type/device checks only; finiteness and
    output-id validation are intentionally deferred to device kernels or the
    caller's explicit debug gate.
    """

    distance: Any
    hit: Any
    hit_point: Any | None = None
    normal: Any | None = None
    geom_id: Any | None = None
    body_id: Any | None = None


class RayCaster(abc.ABC):
    """Batched ray query lifecycle: materialize, pose sync, trace, close.

    The batch shape ``(num_envs, num_rays)`` is fixed at construction. All
    public inputs and results are NumPy arrays; engine-native model, data,
    kernel, and device-pointer types must never appear on this interface.

    Result arrays may be views into implementation-owned buffers that the next
    :meth:`trace` call reuses; callers retaining results across calls must
    copy them. Implementations must not require unbounded per-call allocation.
    """

    caster_type: str
    _ray_capabilities = RayCasterCapabilities()

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Fixed number of environment rows in the batch."""

    @property
    @abc.abstractmethod
    def num_rays(self) -> int:
        """Fixed number of rays per environment row."""

    def get_ray_capabilities(self) -> RayCasterCapabilities:
        """Return the immutable capability declaration for this caster."""
        return self._ray_capabilities

    @abc.abstractmethod
    def materialize(self, scene: RaySceneDescription) -> None:
        """Bind the immutable scene geometry on the cold path.

        After materialization every body pose is the identity until
        :meth:`update_pose` writes it. Materializing twice must fail.
        """

    def update_pose(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Write world-space body poses for the selected environment rows.

        ``body_pos`` and ``body_quat`` are batch-shaped
        ``(rows, num_bodies, 3)`` and ``(rows, num_bodies, 4)`` with unit
        ``wxyz`` quaternions; ``rows`` is ``num_envs`` or ``len(env_ids)``.
        Casters without ``supports_pose_sync`` keep this fail-closed default.
        """
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support pose updates after materialize "
            "(get_ray_capabilities().supports_pose_sync is False); materialize the "
            "scene with final poses or use a caster plugin with pose sync"
        )

    def update_geometry(
        self,
        geometry: RayGeometryState,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Write selected per-row geometry state."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support geometry updates after materialize"
        )

    def bind_pose_layout(self, body_names: Sequence[str]) -> None:
        """Bind the public dynamic-body name order on the cold path.

        ``body_names`` must cover every body used by the materialized scene and
        become the caster's dynamic-body identity axis. The call only fixes the
        layout; Torch storage is attached later by
        :meth:`update_pose_tensor`. Casters without
        ``supports_tensor_pose_sync`` keep this fail-closed default.
        """
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support tensor pose layouts "
            "(get_ray_capabilities().supports_tensor_pose_sync is False); use a "
            "caster plugin with the tensor ray profile"
        )

    def update_pose_tensor(self, pose: RayPoseTensorView) -> None:
        """Alias one full-batch caller-owned Torch pose view without copying.

        The view must exactly match the layout bound by ``bind_pose_layout``,
        be float32 Torch tensors with the correct shapes on one accepted device,
        with positive non-overlapping strides and innermost stride one.
        Implementations must not silently materialize ``.contiguous()``, create
        a ray-specific pose transfer, or synchronize the caller stream.
        """
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support tensor pose sync "
            "(get_ray_capabilities().supports_tensor_pose_sync is False); use a "
            "caster plugin with the tensor ray profile"
        )

    def update_geometry_tensor(self, geometry: RayGeometryTensorView) -> None:
        """Alias one full-batch caller-owned Torch geometry view."""
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support tensor geometry sync"
        )

    @abc.abstractmethod
    def trace(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        max_distance: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
        outputs: RayTraceOutputs | None = None,
        *,
        min_distance: float = 0.0,
        ray_filter: RayTraceFilter | None = None,
    ) -> RayTraceResult:
        """Cast the fixed ray batch against the materialized scene.

        Rays are world-space with unit directions. The shared profile passes
        ``(num_rays, 3)`` arrays broadcast to every selected row; the
        per-environment profile passes ``(rows, num_rays, 3)`` arrays and
        requires ``supports_per_env_rays``. ``rows`` is ``num_envs`` or
        ``len(env_ids)``. ``max_distance`` is a positive finite scalar;
        distances are clipped to it and missed rays report ``hit == False``.
        ``outputs`` must only request fields declared by
        :meth:`get_ray_capabilities`; undeclared requests fail closed with
        :class:`UnsupportedCapabilityError`. Casters without
        ``supports_host_readback`` must raise the same error here. A nonzero
        ``min_distance`` accepts only hits at or beyond that distance, and
        ``ray_filter`` is one immutable filter shared by every selected row and
        ray; both require their declared capabilities.
        """

    def trace_tensor(
        self,
        ray_origins: Any,
        ray_directions: Any,
        max_distance: float,
        env_ids: Any | None = None,
        outputs: RayTraceOutputs | None = None,
        *,
        min_distance: float = 0.0,
        ray_filter: RayTraceFilter | None = None,
    ) -> RayTraceTensorResult:
        """Cast a Torch-native batch and return persistent Torch result views.

        Rays follow the host semantics, but both shared and per-environment
        inputs are already materialized on the bound device. ``env_ids`` is a
        contiguous one-dimensional int64 Torch tensor. The caster launches on
        the caller's current stream, performs no host readback or implicit
        synchronization, and returns views whose contents are valid only while
        the caller's subsequent stream operations preserve that order. Rich
        query semantics follow the host path and are capability-checked per
        profile.
        """
        raise UnsupportedCapabilityError(
            f"{type(self).__name__} does not support tensor ray tracing "
            "(get_ray_capabilities().supports_tensor_rays is False); use the host "
            "trace path or a tensor-capable caster plugin"
        )

    @abc.abstractmethod
    def close(self) -> None:
        """Release caster resources; idempotent. Later queries must fail."""

    def _resolve_selected_rows(self, env_ids: Sequence[int] | np.ndarray | None) -> np.ndarray:
        """Validate an environment selection against the fixed batch."""
        return selected_state_rows(env_ids, self.num_envs)

    def _resolve_ray_batch(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and broadcast one trace ray batch, enforcing the ray profile."""
        origins = np.asarray(ray_origins, dtype=np.float64)
        directions = np.asarray(ray_directions, dtype=np.float64)
        if origins.shape != directions.shape:
            raise ValueError(
                f"ray origins shape {origins.shape} does not match directions "
                f"shape {directions.shape}"
            )
        if origins.shape == (self.num_rays, 3):
            origins = np.broadcast_to(origins, (rows, self.num_rays, 3))
            directions = np.broadcast_to(directions, (rows, self.num_rays, 3))
        elif origins.shape == (rows, self.num_rays, 3):
            if not self._ray_capabilities.supports_per_env_rays:
                raise UnsupportedCapabilityError(
                    f"{type(self).__name__} does not support per-environment rays "
                    "(get_ray_capabilities().supports_per_env_rays is False); pass "
                    f"shared rays of shape ({self.num_rays}, 3)"
                )
        else:
            raise ValueError(
                f"ray batches must have shape ({self.num_rays}, 3) or "
                f"({rows}, {self.num_rays}, 3), got {origins.shape}"
            )
        if not np.isfinite(origins).all() or not np.isfinite(directions).all():
            raise ValueError("ray origins and directions must be finite")
        if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("ray directions must be unit vectors")
        return np.array(origins, dtype=np.float64), np.array(directions, dtype=np.float64)

    @staticmethod
    def _resolve_max_distance(max_distance: float) -> float:
        """Validate the trace clipping distance."""
        value = float(max_distance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"max_distance must be a positive finite scalar, got {value!r}")
        return value

    @staticmethod
    def _resolve_min_distance(min_distance: float) -> float:
        """Validate the inclusive lower hit-distance bound."""
        value = float(min_distance)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"min_distance must be a non-negative finite scalar, got {value!r}")
        return value

    def _check_trace_outputs(
        self, outputs: RayTraceOutputs | None, *, tensor: bool = False
    ) -> RayTraceOutputs:
        """Fail closed on undeclared optional outputs and normalize ``None``."""
        if outputs is None:
            return RayTraceOutputs()
        if not isinstance(outputs, RayTraceOutputs):
            raise TypeError("outputs must be a RayTraceOutputs or None")
        require_ray_trace_outputs(self._ray_capabilities, outputs, tensor=tensor)
        return outputs

    def _check_trace_query(
        self,
        max_distance: float,
        min_distance: float,
        ray_filter: RayTraceFilter | None,
    ) -> float:
        """Validate rich-query arguments against declared capabilities."""
        if ray_filter is not None and not isinstance(ray_filter, RayTraceFilter):
            raise TypeError("ray_filter must be a RayTraceFilter or None")
        require_ray_trace_query(self._ray_capabilities, min_distance, ray_filter)
        limit = self._resolve_max_distance(max_distance)
        minimum = self._resolve_min_distance(min_distance)
        if minimum >= limit:
            raise ValueError(
                f"min_distance must be less than max_distance, got {minimum!r} >= {limit!r}"
            )
        return minimum


@dataclass(frozen=True, slots=True)
class RayCasterSpec:
    """Declared package identity of one ray caster plugin."""

    name: str
    package: str
    status: str


RAY_CASTER_SPECS: tuple[RayCasterSpec, ...] = (
    # ``uniray`` is a separately distributed plugin package exposing
    # ``create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster``.
    # The factory resolves it lazily and fails closed with an actionable
    # dependency diagnostic when the package is not installed.
    RayCasterSpec("uniray", "uniray", "available"),
)


def ray_caster_spec(name: str) -> RayCasterSpec:
    """Return one declared ray caster identity or raise a stable ``KeyError``."""
    for spec in RAY_CASTER_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown UniSim ray caster: {name!r}")


__all__ = [
    "RAY_CASTER_SPECS",
    "RayCaster",
    "RayCasterCapabilities",
    "RayCasterSpec",
    "RayCollisionDescription",
    "RayGeomType",
    "RayGeometryState",
    "RayGeometryTensorView",
    "RayMeshDescription",
    "RayPoseTensorView",
    "RaySceneDescription",
    "RaySiteFrameDescription",
    "RayTraceFilter",
    "RayTraceTensorResult",
    "RayTraceOutputs",
    "RayTraceResult",
    "ray_geom_type",
    "ray_caster_spec",
    "require_ray_trace_outputs",
    "require_ray_trace_query",
]
