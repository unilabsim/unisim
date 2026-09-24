"""Backend-neutral ray-query plugin contract.

This module defines the public surface implemented by ray caster plugins such
as ``uni_ray``: a scene description consumed on the cold path, a fixed-shape
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

    def __post_init__(self) -> None:
        if isinstance(self.num_bodies, bool) or not isinstance(self.num_bodies, int):
            raise TypeError("ray scene num_bodies must be an integer")
        if self.num_bodies <= 0:
            raise ValueError("ray scene num_bodies must be positive")
        if isinstance(self.geom_types, (str, bytes)) or not isinstance(self.geom_types, Sequence):
            raise TypeError("ray scene geom_types must be a sequence of geom type names")
        try:
            geom_types = tuple(RayGeomType(value) for value in self.geom_types)
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
class RayCasterCapabilities:
    """Fine-grained ray caster capabilities surfaced through the contract.

    ``supports_pose_sync`` covers pose updates through :meth:`RayCaster.update_pose`
    after materialization. ``supports_per_env_rays`` covers the per-environment
    ray profile of :meth:`RayCaster.trace` (3-D ray batches); every caster
    accepts the shared 2-D profile. ``supports_host_readback`` covers the
    NumPy host result path of ``trace``; ``supports_device_output`` is reserved
    for a future device-resident result path and enables nothing in this
    contract version. The remaining flags declare the optional
    :class:`RayTraceResult` fields a caster can serve.
    """

    supports_pose_sync: bool = False
    supports_per_env_rays: bool = False
    supports_host_readback: bool = False
    supports_device_output: bool = False
    supports_hit_point: bool = False
    supports_normal: bool = False
    supports_geom_id: bool = False
    supports_body_id: bool = False


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
    capabilities: RayCasterCapabilities, outputs: RayTraceOutputs
) -> None:
    """Fail closed when ``outputs`` requests an undeclared optional field."""
    requested = (
        ("hit_point", outputs.hit_point, capabilities.supports_hit_point),
        ("normal", outputs.normal, capabilities.supports_normal),
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

    @abc.abstractmethod
    def trace(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        max_distance: float,
        env_ids: Sequence[int] | np.ndarray | None = None,
        outputs: RayTraceOutputs | None = None,
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
        ``supports_host_readback`` must raise the same error here.
        """

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

    def _check_trace_outputs(self, outputs: RayTraceOutputs | None) -> RayTraceOutputs:
        """Fail closed on undeclared optional outputs and normalize ``None``."""
        if outputs is None:
            return RayTraceOutputs()
        if not isinstance(outputs, RayTraceOutputs):
            raise TypeError("outputs must be a RayTraceOutputs or None")
        require_ray_trace_outputs(self._ray_capabilities, outputs)
        return outputs


@dataclass(frozen=True, slots=True)
class RayCasterSpec:
    """Declared package identity of one ray caster plugin."""

    name: str
    package: str
    status: str


RAY_CASTER_SPECS: tuple[RayCasterSpec, ...] = (
    # ``uni_ray`` is a separately distributed plugin package exposing
    # ``create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster``.
    # The factory resolves it lazily and fails closed with an actionable
    # dependency diagnostic when the package is not installed.
    RayCasterSpec("uni_ray", "uni_ray", "available"),
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
    "RayGeomType",
    "RaySceneDescription",
    "RayTraceOutputs",
    "RayTraceResult",
    "ray_caster_spec",
    "require_ray_trace_outputs",
]
