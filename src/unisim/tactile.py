"""Backend-neutral tactile-sensor plugin contract.

The contract describes detached collision geometry, mounted sensor frames,
pose synchronization, lifecycle, and observation metadata. It deliberately
does not prescribe a solver, device library, or observation layout.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .entity_state import selected_state_rows
from .errors import UnsupportedCapabilityError
from .ray_query import RayCollisionDescription, RayGeometryState, RaySiteFrameDescription


@dataclass(frozen=True)
class TactileSceneDescription:
    """Detached geometry and sensor mounts for one tactile materialization.

    ``collision`` reuses UniSim's public collision descriptor. Object identity
    is selected by body name, with an optional exact geom-name subset. A
    ``geometry_state`` carries an optional per-environment geometry snapshot
    aligned to the collision geom axis. Each sensor mount reuses the public
    site-to-body frame description. No engine model, solver object, device
    pointer, or model-specific configuration crosses this boundary.
    """

    collision: RayCollisionDescription
    object_body_names: tuple[str, ...]
    sensor_frames: tuple[RaySiteFrameDescription, ...]
    object_geom_names: tuple[str, ...] | None = None
    geometry_state: RayGeometryState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.collision, RayCollisionDescription):
            raise TypeError("tactile collision must be a RayCollisionDescription")
        if isinstance(self.object_body_names, (str, bytes)) or not isinstance(
            self.object_body_names, Sequence
        ):
            raise TypeError("object_body_names must be a sequence of body names")
        object_names = tuple(self.object_body_names)
        if any(not isinstance(name, str) or not name for name in object_names):
            raise TypeError("object_body_names must contain non-empty strings")
        if len(set(object_names)) != len(object_names):
            raise ValueError("object_body_names must be unique")
        body_names = set(self.collision.scene.body_names)
        missing_objects = tuple(name for name in object_names if name not in body_names)
        if missing_objects:
            raise ValueError(
                "tactile object bodies are absent from collision geometry: "
                f"{missing_objects}"
            )

        if self.object_geom_names is None:
            object_geom_names = None
        else:
            if isinstance(self.object_geom_names, (str, bytes)) or not isinstance(
                self.object_geom_names, Sequence
            ):
                raise TypeError("object_geom_names must be a sequence of collision geom names")
            object_geom_names = tuple(self.object_geom_names)
            if not object_geom_names:
                raise ValueError("object_geom_names must not be empty when specified")
            if any(not isinstance(name, str) or not name for name in object_geom_names):
                raise TypeError("object_geom_names must contain non-empty strings")
            if len(set(object_geom_names)) != len(object_geom_names):
                raise ValueError("object_geom_names must be unique")
            geom_index = {name: index for index, name in enumerate(self.collision.geom_names)}
            missing_geoms = tuple(name for name in object_geom_names if name not in geom_index)
            if missing_geoms:
                raise ValueError(
                    "tactile object geoms are absent from collision geometry: "
                    f"{missing_geoms}"
                )
            selected_body_ids = {
                self.collision.scene.body_names.index(name) for name in object_names
            }
            wrong_owner = tuple(
                name
                for name in object_geom_names
                if int(self.collision.scene.geom_body_ids[geom_index[name]])
                not in selected_body_ids
            )
            if wrong_owner:
                raise ValueError(
                    "tactile object geoms must belong to object_body_names: "
                    f"{wrong_owner}"
                )

        geometry_state = self.geometry_state
        if geometry_state is not None:
            if not isinstance(geometry_state, RayGeometryState):
                raise TypeError("geometry_state must be a RayGeometryState or None")
            if geometry_state.geom_names != self.collision.geom_names:
                raise ValueError(
                    "tactile geometry_state geom_names must match collision geom_names in order"
                )

        if isinstance(self.sensor_frames, (str, bytes)) or not isinstance(
            self.sensor_frames, Sequence
        ):
            raise TypeError("sensor_frames must be a sequence of RaySiteFrameDescription values")
        frames = tuple(self.sensor_frames)
        if not frames:
            raise ValueError("sensor_frames must contain at least one mounted tactile frame")
        if any(not isinstance(frame, RaySiteFrameDescription) for frame in frames):
            raise TypeError("sensor_frames must contain RaySiteFrameDescription values")
        names = tuple(frame.site_name for frame in frames)
        if len(set(names)) != len(names):
            raise ValueError("sensor frame names must be unique")
        missing_bodies = tuple(
            frame.body_name for frame in frames if frame.body_name not in body_names
        )
        if missing_bodies:
            raise ValueError(
                "tactile sensor bodies are absent from collision geometry: "
                f"{missing_bodies}"
            )
        object.__setattr__(self, "object_body_names", object_names)
        object.__setattr__(self, "object_geom_names", object_geom_names)
        object.__setattr__(self, "sensor_frames", frames)


@dataclass(frozen=True, slots=True)
class TactileFieldSpec:
    """Metadata for one named observation field.

    ``shape`` is the complete field shape, including any environment axis;
    ``None`` marks a dimension whose size may vary between reads. Dimensions
    are descriptive names, not a prescribed tactile grid. A sensor-owned field
    must state when its buffer can be reused; caller-owned data uses
    ``buffer_lifetime="caller"``.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: str
    units: str
    frame: str
    dimensions: tuple[str, ...]
    buffer_owner: Literal["sensor", "caller"]
    buffer_lifetime: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tactile field name must be a non-empty string")
        shape = tuple(self.shape)
        if any(
            size is not None
            and (isinstance(size, bool) or not isinstance(size, int) or size < 0)
            for size in shape
        ):
            raise ValueError("tactile field shape values must be non-negative integers or None")
        dimensions = tuple(self.dimensions)
        if len(dimensions) != len(shape) or any(
            not isinstance(name, str) or not name for name in dimensions
        ):
            raise ValueError("tactile field dimensions must name each shape axis")
        for field_name in ("dtype", "units", "frame", "buffer_lifetime"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"tactile field {field_name} must be a non-empty string")
        if self.buffer_owner not in ("sensor", "caller"):
            raise ValueError("tactile field buffer_owner must be 'sensor' or 'caller'")
        if self.buffer_owner == "caller" and self.buffer_lifetime != "caller":
            raise ValueError("caller-owned tactile fields must use buffer_lifetime='caller'")
        if self.buffer_owner == "sensor" and self.buffer_lifetime == "caller":
            raise ValueError("sensor-owned tactile fields must describe their buffer lifetime")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dimensions", dimensions)


@dataclass(frozen=True, slots=True)
class TactileSensorCapabilities:
    """Declared input/output profiles and lifecycle support for one sensor."""

    pose_input_profiles: tuple[str, ...] = ("numpy",)
    output_profiles: tuple[str, ...] = ("numpy",)
    supports_selected_environments: bool = False
    supports_geometry_rebuild: bool = False
    stream_event_ownership: Literal["none", "caller", "sensor"] = "none"

    def __post_init__(self) -> None:
        allowed = {"numpy", "torch_cpu", "torch_cuda", "cupy_cuda"}
        for name in ("pose_input_profiles", "output_profiles"):
            values = tuple(getattr(self, name))
            if not values or any(value not in allowed for value in values):
                raise ValueError(f"{name} must contain supported array profiles: {sorted(allowed)}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        if self.stream_event_ownership not in ("none", "caller", "sensor"):
            raise ValueError(
                "stream_event_ownership must be 'none', 'caller', or 'sensor'"
            )


class TactileSensor(abc.ABC):
    """Batched tactile computation lifecycle, independent of a physics backend.

    ``update_pose`` only replaces cached world poses. ``advance`` performs one
    model step and advances any history. ``read`` observes only the latest
    completed step and does not advance state. Returned fields may be views
    into reusable sensor-owned buffers as described by ``output_specs``.
    """

    tactile_type: str
    _tactile_capabilities = TactileSensorCapabilities()

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Fixed number of environment rows in this sensor batch."""

    @property
    @abc.abstractmethod
    def output_specs(self) -> tuple[TactileFieldSpec, ...]:
        """Observation field names, shapes, units, frames, and ownership."""

    def get_tactile_capabilities(self) -> TactileSensorCapabilities:
        """Return the immutable input, output, and lifecycle declaration."""
        return self._tactile_capabilities

    @abc.abstractmethod
    def materialize(self, scene: TactileSceneDescription) -> None:
        """Bind static geometry and mounted frames on the cold path.

        Materialization is one-shot. Body poses are identity until the caller
        invokes ``update_pose``. Geometry changes use ``rebuild`` when the
        implementation declares that capability.
        """

    @abc.abstractmethod
    def update_pose(
        self,
        body_pos: Any,
        body_quat: Any,
        env_ids: Sequence[int] | Any | None = None,
    ) -> None:
        """Cache world-space ``(rows, bodies, 3/4)`` positions/quaternions.

        Quaternions use unit ``wxyz`` order. ``rows`` is ``num_envs`` or the
        number of selected ``env_ids``. Accepted array types and devices are
        declared by ``get_tactile_capabilities``. This method does not compute
        observations or advance temporal history.
        """

    @abc.abstractmethod
    def advance(self) -> None:
        """Compute one tactile observation and advance model history once."""

    @abc.abstractmethod
    def read(self) -> Mapping[str, Any]:
        """Return the latest observation without changing model state.

        Before the first ``advance``, an explicit ``reset`` establishes a
        cleared observation that may be read. Each field is described by
        ``output_specs``. Sensor-owned buffers remain valid for the lifetime
        declared by that field.
        """

    @abc.abstractmethod
    def reset(self, env_ids: Sequence[int] | Any | None = None) -> None:
        """Clear temporal state and output for all or selected environments.

        Reset leaves the most recently cached poses unchanged. The next
        ``advance`` seeds fresh history for the selected environments. Before
        the first ``advance``, reset makes the cleared output readable.
        """

    def rebuild(self, scene: TactileSceneDescription) -> None:
        """Replace static geometry when the implementation supports rebuilding.

        A successful rebuild invalidates prior output views and clears model
        history. The caller must synchronize poses again before the next
        ``advance``. Implementations that do not declare rebuild support fail
        closed with :class:`UnsupportedCapabilityError`.
        """
        if not self._tactile_capabilities.supports_geometry_rebuild:
            raise UnsupportedCapabilityError(
                f"{type(self).__name__} does not support geometry rebuilds "
                "(get_tactile_capabilities().supports_geometry_rebuild is False)"
            )
        raise NotImplementedError("sensor declares geometry rebuild but does not implement it")

    @abc.abstractmethod
    def close(self) -> None:
        """Release device buffers and other resources; idempotent."""

    def _resolve_selected_rows(self, env_ids: Sequence[int] | Any | None) -> Any:
        """Validate an environment selection against the fixed sensor batch."""
        return selected_state_rows(env_ids, self.num_envs)

    def __enter__(self) -> TactileSensor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "TactileFieldSpec",
    "TactileSceneDescription",
    "TactileSensor",
    "TactileSensorCapabilities",
]
