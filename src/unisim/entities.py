"""Engine-neutral entity authoring and selected-entity reset requests.

These values carry intent, not native actor handles or asset parsing logic.
Adapters validate source topology and bind public names during materialization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor

AssetFormat = Literal["mjcf", "urdf", "usd", "superdex_bot"]
EntityKind = Literal["articulation", "rigid"]
RootMode = Literal["fixed", "floating", "kinematic"]


def _entity_name(name: str) -> None:
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None:
        raise ValueError("entity name must match [A-Za-z][A-Za-z0-9_-]*")


def _vector(values: tuple[float, ...], width: int, field: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != width:
        raise ValueError(f"{field} must be a tuple of {width} finite numbers")
    array = np.asarray(values)
    if array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite real numbers")
    return tuple(float(value) for value in values)


def _unit_quaternion(values: np.ndarray, field: str) -> None:
    if not np.allclose(np.linalg.norm(values, axis=-1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{field} requires unit wxyz quaternions")


@dataclass(frozen=True)
class EntityInitialState:
    """Root link pose in the environment world frame, without clone offsets.

    Source/keyframe joint defaults remain source-owned. Root velocities start
    at zero; explicit runtime velocity writes use :class:`EntityStatePatch`.
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = _vector(self.quaternion, 4, "quaternion")
        _unit_quaternion(np.asarray(quaternion), "quaternion")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)


@dataclass(frozen=True)
class SceneEntitySpec:
    """One independently materialized physical entity or visual mirror.

    ``mirror_of`` inherits the referenced entity's source and fixed identity,
    never its pose. A mirror has no source of its own, collisions or controls.
    Source format, topology and root-mode compatibility are adapter-audited.
    """

    name: str
    source: ModelSourceDescriptor | None = None
    asset_format: AssetFormat = "mjcf"
    kind: EntityKind = "articulation"
    root_mode: RootMode = "floating"
    initial_state: EntityInitialState = EntityInitialState()
    collision_enabled: bool = True
    mirror_of: str | None = None

    def __post_init__(self) -> None:
        _entity_name(self.name)
        if self.asset_format not in ("mjcf", "urdf", "usd", "superdex_bot"):
            raise ValueError(f"unsupported source format {self.asset_format!r}")
        if self.kind not in ("articulation", "rigid"):
            raise ValueError("entity kind must be articulation or rigid")
        if self.root_mode not in ("fixed", "floating", "kinematic"):
            raise ValueError("root_mode must be fixed, floating or kinematic")
        if not isinstance(self.initial_state, EntityInitialState):
            raise TypeError("initial_state must be EntityInitialState")
        if not isinstance(self.collision_enabled, bool):
            raise TypeError("collision_enabled must be bool")
        if self.mirror_of is None:
            if not isinstance(self.source, ModelSourceDescriptor):
                raise TypeError("physical entity source must be ModelSourceDescriptor")
        else:
            _entity_name(self.mirror_of)
            if self.mirror_of == self.name:
                raise ValueError("an entity cannot mirror itself")
            if self.source is not None:
                raise ValueError("a mirror inherits its source; do not supply another source")
            if self.kind != "rigid" or self.root_mode != "kinematic" or self.collision_enabled:
                raise ValueError("a mirror must be rigid, kinematic and collision-disabled")


@dataclass(frozen=True)
class EntityVariantBinding:
    """Bind one immutable source catalog to one physical entity."""

    target_entity: str
    plan: FixedVariantPlan

    def __post_init__(self) -> None:
        _entity_name(self.target_entity)
        if not isinstance(self.plan, FixedVariantPlan):
            raise TypeError("entity variant plan must be FixedVariantPlan")


def validate_entity_declarations(
    entities: tuple[SceneEntitySpec, ...],
    binding: EntityVariantBinding | None,
    num_envs: int | None = None,
) -> None:
    """Validate authoring relationships without loading an engine or source file."""
    if not isinstance(entities, tuple) or any(
        not isinstance(entity, SceneEntitySpec) for entity in entities
    ):
        raise TypeError("entity_assets must be a tuple of SceneEntitySpec")
    names = {entity.name: entity for entity in entities}
    if len(names) != len(entities):
        raise ValueError("entity names must be unique")
    for entity in entities:
        if entity.mirror_of is None:
            continue
        target = names.get(entity.mirror_of)
        if target is None or target.mirror_of is not None:
            raise ValueError(f"mirror {entity.name!r} must reference a physical entity")
        if entity.asset_format != target.asset_format:
            raise ValueError(f"mirror {entity.name!r} must use its target's asset format")
    if binding is not None:
        if not isinstance(binding, EntityVariantBinding):
            raise TypeError("entity_variant must be EntityVariantBinding or None")
        target = names.get(binding.target_entity)
        if target is None or target.mirror_of is not None:
            raise ValueError("entity variant target must name a physical entity")
        binding.plan.validate(num_envs)


@dataclass(frozen=True)
class _FrozenStateArray:
    values: bytes
    dtype: str
    shape: tuple[int, ...]

    def array(self) -> np.ndarray:
        # Neither this array nor its .base owns any persistent mutable metadata.
        return np.frombuffer(self.values, dtype=self.dtype).reshape(self.shape)


def _state_array(
    value: np.ndarray | None, field: str, width: int | None
) -> _FrozenStateArray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2 or (width is not None and array.shape[1] != width):
        raise ValueError(f"{field} must have shape (selected_envs, {width or 'columns'})")
    if array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite real numbers")
    # Bytes-backed storage cannot be made writable, even through caller aliases.
    return _FrozenStateArray(array.tobytes(), array.dtype.str, array.shape)


@dataclass(frozen=True, eq=False, init=False)
class EntityStatePatch:
    """Selected rows of one entity's root and/or joint state.

    Root pose is link-origin xyz/wxyz. Root velocity is link-origin linear
    velocity followed by angular velocity, both in the environment world frame.
    Joint columns are packed in ``joint_names`` order using the bound joint's
    qpos/qvel widths. Empty names select all non-root joints in entity order.
    Missing fields mean *preserve*, not zero. Adapters validate joint widths,
    spherical-joint quaternions, root permissions and selectors before writing.
    """

    entity: str
    _root_pose: _FrozenStateArray | None = field(repr=False)
    _root_velocity: _FrozenStateArray | None = field(repr=False)
    _joint_positions: _FrozenStateArray | None = field(repr=False)
    _joint_velocities: _FrozenStateArray | None = field(repr=False)
    joint_names: tuple[str, ...] = ()

    def __init__(
        self,
        entity: str,
        root_pose: np.ndarray | None = None,
        root_velocity: np.ndarray | None = None,
        joint_positions: np.ndarray | None = None,
        joint_velocities: np.ndarray | None = None,
        joint_names: tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "entity", entity)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "_root_pose", _state_array(root_pose, "root_pose", 7))
        object.__setattr__(self, "_root_velocity", _state_array(root_velocity, "root_velocity", 6))
        object.__setattr__(
            self, "_joint_positions", _state_array(joint_positions, "joint_positions", None)
        )
        object.__setattr__(
            self, "_joint_velocities", _state_array(joint_velocities, "joint_velocities", None)
        )
        self.__post_init__()

    @property
    def root_pose(self) -> np.ndarray | None:
        return None if self._root_pose is None else self._root_pose.array()

    @property
    def root_velocity(self) -> np.ndarray | None:
        return None if self._root_velocity is None else self._root_velocity.array()

    @property
    def joint_positions(self) -> np.ndarray | None:
        return None if self._joint_positions is None else self._joint_positions.array()

    @property
    def joint_velocities(self) -> np.ndarray | None:
        return None if self._joint_velocities is None else self._joint_velocities.array()

    def __post_init__(self) -> None:
        _entity_name(self.entity)
        if not isinstance(self.joint_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.joint_names
        ):
            raise TypeError("joint_names must be a tuple of non-empty local names")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names cannot contain duplicates")
        rows: set[int] = set()
        for array in (
            self._root_pose,
            self._root_velocity,
            self._joint_positions,
            self._joint_velocities,
        ):
            if array is not None:
                rows.add(array.shape[0])
        if not rows or len(rows) != 1:
            raise ValueError("a patch needs at least one field, with equal row counts")
        if self.root_pose is not None:
            _unit_quaternion(self.root_pose[:, 3:7], "root_pose")
        if self.joint_names and self.joint_positions is None and self.joint_velocities is None:
            raise ValueError("joint_names requires a joint state field")

    def __reduce__(self):
        # Spawn must repeat validation/freezing rather than restore writable arrays.
        return (
            type(self),
            (
                self.entity,
                self.root_pose,
                self.root_velocity,
                self.joint_positions,
                self.joint_velocities,
                self.joint_names,
            ),
        )


@dataclass(frozen=True)
class SceneResetRequest:
    """A prevalidated selected-environment transaction, before adapter binding.

    Native submission must validate every patch before the first write. On a
    partial native failure the adapter becomes faulted unless it can roll back.
    Neither a reset nor a patch can change a fixed variant identity.
    """

    env_ids: tuple[int, ...]
    patches: tuple[EntityStatePatch, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.env_ids, tuple) or not self.env_ids:
            raise ValueError("env_ids must be a non-empty tuple")
        if any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            for i in self.env_ids
        ):
            raise TypeError("env_ids must contain integers")
        if min(self.env_ids) < 0 or len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("env_ids must be unique and nonnegative")
        object.__setattr__(self, "env_ids", tuple(int(i) for i in self.env_ids))
        if (
            not isinstance(self.patches, tuple)
            or not self.patches
            or any(not isinstance(patch, EntityStatePatch) for patch in self.patches)
        ):
            raise TypeError("patches must be a non-empty tuple of EntityStatePatch")
        names = [patch.entity for patch in self.patches]
        if len(set(names)) != len(names):
            raise ValueError("combine writes to one entity into a single patch")
        for patch in self.patches:
            for values in (
                patch.root_pose,
                patch.root_velocity,
                patch.joint_positions,
                patch.joint_velocities,
            ):
                if values is not None and values.shape[0] != len(self.env_ids):
                    raise ValueError(f"patch {patch.entity!r} row count differs from env_ids")

    def validate_env_count(self, num_envs: int) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if max(self.env_ids) >= num_envs:
            raise ValueError("reset env_ids exceed the backend environment count")
