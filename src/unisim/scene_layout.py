"""Frozen public scene addresses and validation before native reset submission.

This module neither loads assets nor knows native actor/view indices. Adapters
bind the public addresses to their own native maps during materialization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from unisim.entities import EntityStatePatch, SceneResetRequest


def _name(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{field} must be a non-empty name")


def _names(values: tuple[str, ...], field: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    for value in values:
        _name(value, field)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique names")


def _integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return int(value)


def _indices(values: tuple[int, ...], field: str, width: int | None = None) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalized = tuple(_integer(value, field) for value in values)
    if width is not None and len(normalized) != width:
        raise ValueError(f"{field} must contain {width} columns")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique indices")
    return normalized


def _wire(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _wire_tuple(value: object, field: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError(f"wire {field} must be a JSON array")
    return tuple(value)


@dataclass(frozen=True)
class JointLayout:
    """One non-root joint's absolute generalized-state columns."""

    name: str
    kind: str
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    body_name: str

    def __post_init__(self) -> None:
        _name(self.name, "joint name")
        _name(self.body_name, "joint body_name")
        if self.kind not in ("hinge", "slide", "ball"):
            raise ValueError("joint kind must be hinge, slide or ball")
        nq, nv = (4, 3) if self.kind == "ball" else (1, 1)
        object.__setattr__(self, "qpos_indices", _indices(self.qpos_indices, "joint qpos", nq))
        object.__setattr__(self, "qvel_indices", _indices(self.qvel_indices, "joint qvel", nv))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "body_name": self.body_name,
            "qpos_indices": list(self.qpos_indices),
            "qvel_indices": list(self.qvel_indices),
        }

    @classmethod
    def from_dict(cls, value: object) -> JointLayout:
        data = _wire(value, {"name", "kind", "body_name", "qpos_indices", "qvel_indices"}, "joint")
        return cls(
            data["name"],
            data["kind"],
            _wire_tuple(data["qpos_indices"], "qpos_indices"),
            _wire_tuple(data["qvel_indices"], "qvel_indices"),
            data["body_name"],
        )


@dataclass(frozen=True)
class EntityLayout:
    """Public names, topology and addresses for one independent physical root.

    Parent names and root/body/joint names are entity-local. Body ordering need
    not be topological; body IDs and state columns need not be contiguous.
    Actuator entries identify scalar controls and their declared target joints;
    adapters separately audit supported transmission/drive semantics.
    """

    name: str
    kind: str
    root_mode: str
    root_body: str
    body_names: tuple[str, ...]
    body_ids: tuple[int, ...]
    body_parent_names: tuple[str | None, ...]
    joints: tuple[JointLayout, ...]
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_indices: tuple[int, ...]
    root_qpos_indices: tuple[int, ...] = ()
    root_qvel_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.name) is None
        ):
            raise ValueError("entity name must match [A-Za-z][A-Za-z0-9_-]*")
        if self.kind not in ("rigid", "articulation"):
            raise ValueError("entity kind must be rigid or articulation")
        if self.root_mode not in ("fixed", "floating", "kinematic"):
            raise ValueError("root_mode must be fixed, floating or kinematic")
        _names(self.body_names, "body_names")
        _name(self.root_body, "root_body")
        object.__setattr__(self, "body_ids", _indices(self.body_ids, "body_ids"))
        if len(self.body_ids) != len(self.body_names) or self.root_body not in self.body_names:
            raise ValueError("body IDs must match body names and include root_body")
        if not isinstance(self.body_parent_names, tuple) or len(self.body_parent_names) != len(
            self.body_names
        ):
            raise ValueError("body_parent_names must be a tuple aligned with body_names")
        parents = dict(zip(self.body_names, self.body_parent_names))
        for body, parent in parents.items():
            if body == self.root_body:
                if parent is not None:
                    raise ValueError("root_body parent must be None")
            elif not isinstance(parent, str) or parent not in parents:
                raise ValueError(f"body {body!r} must have an entity-local parent")
        for body in self.body_names:
            visited: set[str] = set()
            current: str | None = body
            while current is not None:
                if current in visited:
                    raise ValueError("body parent topology contains a cycle")
                visited.add(current)
                current = parents[current]
        if not isinstance(self.joints, tuple) or any(
            not isinstance(joint, JointLayout) for joint in self.joints
        ):
            raise TypeError("joints must be a tuple of JointLayout")
        _names(tuple(joint.name for joint in self.joints), "joint names")
        if self.kind == "rigid" and self.joints:
            raise ValueError("rigid entities cannot contain non-root joints")
        if any(joint.body_name not in parents for joint in self.joints):
            raise ValueError("every joint must reference an entity-local body")
        _names(self.actuator_names, "actuator_names")
        if not isinstance(self.actuator_joint_names, tuple):
            raise TypeError("actuator_joint_names must be a tuple")
        joint_names = {joint.name for joint in self.joints}
        if any(
            not isinstance(name, str) or name not in joint_names
            for name in self.actuator_joint_names
        ):
            raise ValueError("actuator targets must reference declared entity-local joints")
        object.__setattr__(self, "actuator_indices", _indices(self.actuator_indices, "actuators"))
        if not (
            len(self.actuator_names) == len(self.actuator_joint_names) == len(self.actuator_indices)
        ):
            raise ValueError("actuator names, targets and indices must have equal lengths")
        nq, nv = (7, 6) if self.root_mode == "floating" else (0, 0)
        object.__setattr__(
            self, "root_qpos_indices", _indices(self.root_qpos_indices, "root qpos", nq)
        )
        object.__setattr__(
            self, "root_qvel_indices", _indices(self.root_qvel_indices, "root qvel", nv)
        )
        _indices(self.qpos_indices, "entity qpos")
        _indices(self.qvel_indices, "entity qvel")

    @property
    def qpos_indices(self) -> tuple[int, ...]:
        return self.root_qpos_indices + tuple(
            i for joint in self.joints for i in joint.qpos_indices
        )

    @property
    def qvel_indices(self) -> tuple[int, ...]:
        return self.root_qvel_indices + tuple(
            i for joint in self.joints for i in joint.qvel_indices
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "root_mode": self.root_mode,
            "root_body": self.root_body,
            "body_names": list(self.body_names),
            "body_ids": list(self.body_ids),
            "body_parent_names": list(self.body_parent_names),
            "joints": [joint.to_dict() for joint in self.joints],
            "actuator_names": list(self.actuator_names),
            "actuator_joint_names": list(self.actuator_joint_names),
            "actuator_indices": list(self.actuator_indices),
            "root_qpos_indices": list(self.root_qpos_indices),
            "root_qvel_indices": list(self.root_qvel_indices),
        }

    @classmethod
    def from_dict(cls, value: object) -> EntityLayout:
        fields = {
            "name",
            "kind",
            "root_mode",
            "root_body",
            "body_names",
            "body_ids",
            "body_parent_names",
            "joints",
            "actuator_names",
            "actuator_joint_names",
            "actuator_indices",
            "root_qpos_indices",
            "root_qvel_indices",
        }
        data = _wire(value, fields, "entity")
        return cls(
            name=data["name"],
            kind=data["kind"],
            root_mode=data["root_mode"],
            root_body=data["root_body"],
            body_names=_wire_tuple(data["body_names"], "body_names"),
            body_ids=_wire_tuple(data["body_ids"], "body_ids"),
            body_parent_names=_wire_tuple(data["body_parent_names"], "body_parent_names"),
            joints=tuple(JointLayout.from_dict(j) for j in _wire_tuple(data["joints"], "joints")),
            actuator_names=_wire_tuple(data["actuator_names"], "actuator_names"),
            actuator_joint_names=_wire_tuple(data["actuator_joint_names"], "actuator_joint_names"),
            actuator_indices=_wire_tuple(data["actuator_indices"], "actuator_indices"),
            root_qpos_indices=_wire_tuple(data["root_qpos_indices"], "root_qpos_indices"),
            root_qvel_indices=_wire_tuple(data["root_qvel_indices"], "root_qvel_indices"),
        )


@dataclass(frozen=True)
class BoundEntityStatePatch:
    """Validated write values and public columns; contains no native addresses."""

    entity: EntityLayout
    patch: EntityStatePatch
    joints: tuple[JointLayout, ...]
    joint_qpos_indices: tuple[int, ...]
    joint_qvel_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundSceneReset:
    """Complete validation result, returned only after every patch passes."""

    env_ids: tuple[int, ...]
    patches: tuple[BoundEntityStatePatch, ...]


@dataclass(frozen=True)
class CompiledSceneLayout:
    """Engine-neutral public addresses validated once on the cold path.

    ``nbody`` may include unowned engine world bodies. Generalized state and
    control columns must all belong to an entity, with no gaps or overlap.
    Native actor maps, source identity and sensor layouts are separate contracts.
    """

    entities: tuple[EntityLayout, ...]
    nq: int
    nv: int
    nu: int
    nbody: int
    schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.entities, tuple) or any(
            not isinstance(entity, EntityLayout) for entity in self.entities
        ):
            raise TypeError("entities must be a tuple of EntityLayout")
        _names(tuple(entity.name for entity in self.entities), "entity names")
        for field in ("nq", "nv", "nu", "nbody"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        for field, size, complete in (
            ("qpos_indices", self.nq, True),
            ("qvel_indices", self.nv, True),
            ("actuator_indices", self.nu, True),
            ("body_ids", self.nbody, False),
        ):
            columns = tuple(i for entity in self.entities for i in getattr(entity, field))
            _indices(columns, field)
            if any(i >= size for i in columns):
                raise ValueError(f"{field} exceeds its declared dimension {size}")
            if complete and len(columns) != size:
                raise ValueError(f"{field} must completely cover its declared dimension {size}")

    def get_entity(self, name: str) -> EntityLayout:
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise ValueError(f"unknown scene entity {name!r}")

    def _resolve(self, name: str, entity: str | None) -> tuple[EntityLayout, str]:
        _name(name, "lookup name")
        if entity is None:
            prefix, separator, local = name.partition("/")
            if not separator or not local:
                raise ValueError("lookups require entity/local_name or an explicit entity")
            return self.get_entity(prefix), local
        return self.get_entity(entity), name

    def get_body_ids(self, names: Sequence[str], *, entity: str | None = None) -> tuple[int, ...]:
        result = []
        for name in names:
            owner, local = self._resolve(name, entity)
            try:
                result.append(owner.body_ids[owner.body_names.index(local)])
            except ValueError as exc:
                raise ValueError(f"unknown body {owner.name}/{local}") from exc
        return tuple(result)

    def get_joint_layouts(
        self,
        names: Sequence[str],
        *,
        entity: str | None = None,
    ) -> tuple[JointLayout, ...]:
        result = []
        for name in names:
            owner, local = self._resolve(name, entity)
            joint = next((item for item in owner.joints if item.name == local), None)
            if joint is None:
                raise ValueError(f"unknown joint {owner.name}/{local}")
            result.append(joint)
        return tuple(result)

    def get_actuator_ids(
        self, names: Sequence[str], *, entity: str | None = None
    ) -> tuple[int, ...]:
        result = []
        for name in names:
            owner, local = self._resolve(name, entity)
            try:
                result.append(owner.actuator_indices[owner.actuator_names.index(local)])
            except ValueError as exc:
                raise ValueError(f"unknown actuator {owner.name}/{local}") from exc
        return tuple(result)

    def require_same_layout(self, other: CompiledSceneLayout) -> None:
        """Require identical public ordering, topology, semantics and addresses."""
        if not isinstance(other, CompiledSceneLayout) or self != other:
            raise ValueError(
                "scene layouts differ in public names, topology, ordering or addresses"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "nbody": self.nbody,
            "entities": [entity.to_dict() for entity in self.entities],
        }

    @classmethod
    def from_dict(cls, value: object) -> CompiledSceneLayout:
        data = _wire(value, {"schema_version", "entities", "nq", "nv", "nu", "nbody"}, "scene")
        if type(data["schema_version"]) is not int or data["schema_version"] != cls.schema_version:
            raise ValueError(f"unsupported scene layout schema_version {data['schema_version']!r}")
        return cls(
            tuple(EntityLayout.from_dict(e) for e in _wire_tuple(data["entities"], "entities")),
            data["nq"],
            data["nv"],
            data["nu"],
            data["nbody"],
        )

    def validate_reset(self, request: SceneResetRequest, *, num_envs: int) -> BoundSceneReset:
        """Bind all requested writes without executing or exposing a partial commit.

        Root velocities remain world-frame link-origin values. Conversion to
        native frames/COM reference points belongs to the adapter's commit path.
        """
        from unisim.entities import SceneResetRequest

        if not isinstance(request, SceneResetRequest):
            raise TypeError("request must be SceneResetRequest")
        request.validate_env_count(num_envs)
        bound = []
        for patch in request.patches:
            entity = self.get_entity(patch.entity)
            if entity.root_mode == "fixed" and (
                patch.root_pose is not None or patch.root_velocity is not None
            ):
                raise ValueError(f"fixed entity {entity.name!r} does not allow root writes")
            if entity.root_mode == "kinematic" and patch.root_velocity is not None:
                raise ValueError(f"kinematic entity {entity.name!r} does not allow root velocity")
            joints = (
                self.get_joint_layouts(patch.joint_names, entity=entity.name)
                if patch.joint_names
                else entity.joints
            )
            qpos = tuple(i for joint in joints for i in joint.qpos_indices)
            qvel = tuple(i for joint in joints for i in joint.qvel_indices)
            for field, width in (("joint_positions", len(qpos)), ("joint_velocities", len(qvel))):
                values = getattr(patch, field)
                if values is not None and values.shape[1] != width:
                    raise ValueError(f"{entity.name} {field} needs {width} columns")
            positions = patch.joint_positions
            if positions is not None:
                offset = 0
                for joint in joints:
                    if joint.kind == "ball":
                        quaternion = positions[:, offset : offset + 4]
                        if not np.allclose(
                            np.linalg.norm(quaternion, axis=1), 1.0, rtol=0.0, atol=1e-5
                        ):
                            raise ValueError(
                                f"ball joint {entity.name}/{joint.name} needs unit wxyz"
                            )
                    offset += len(joint.qpos_indices)
            bound.append(BoundEntityStatePatch(entity, patch, joints, qpos, qvel))
        return BoundSceneReset(request.env_ids, tuple(bound))
