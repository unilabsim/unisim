"""Immutable compiled model addresses shared by MuJoCo-family adapters.

This cold-path owner inspects structural model metadata, not source assets.
It preserves native names and transmissions without importing an engine SDK
or projecting arbitrary old models onto the narrower public entity schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from unisim.scene_layout import CompiledSceneLayout

_JOINT_WIDTHS = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
_PUBLIC_JOINT_KINDS = {1: "ball", 2: "slide", 3: "hinge"}


def _int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _integers(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be an integer array with shape {shape}")
    # Native structural indices fit int64. Refuse wrapping uint64 metadata.
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{label} exceeds the native index range")
    return array.astype(np.int64, copy=True)


def _names(model: Any, accessor: str, count: int) -> tuple[str, ...]:
    values = tuple(getattr(model, accessor)(index).name for index in range(count))
    if any(not isinstance(name, str) for name in values):
        raise ValueError(f"native {accessor} names must be strings")
    named = tuple(name for name in values if name)
    if len(set(named)) != len(named):
        raise ValueError(f"native {accessor} names must be unique when nonempty")
    return values


def _cover(groups: Sequence[tuple[int, ...]], size: int, label: str) -> None:
    columns = tuple(column for group in groups for column in group)
    if len(columns) != size or set(columns) != set(range(size)):
        raise ValueError(f"{label} columns must cover their dimension exactly once")


@dataclass(frozen=True)
class BodyIndex:
    id: int
    name: str
    parent_id: int
    root_id: int
    joint_ids: tuple[int, ...]
    mocap_id: int


@dataclass(frozen=True)
class JointIndex:
    id: int
    name: str
    kind: int
    body_id: int
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]


@dataclass(frozen=True)
class ActuatorIndex:
    id: int
    name: str
    trntype: int
    trnid: tuple[int, int]
    control_indices: tuple[int, ...]


@dataclass(frozen=True)
class CompiledModelIndex:
    """Frozen native addresses, including physical trees with no public entity.

    Body zero is the engine world. Other bodies belong to their actual
    top-level ancestor's partition. Anonymous bodies retain the empty name
    and are addressed privately by ``bodies[native_id]``; no invented public
    name can collide with a real native name. A top-level hinge/ball body is
    preserved as-is, not relabelled as a fixed or free public root.
    """

    bodies: tuple[BodyIndex, ...]
    joints: tuple[JointIndex, ...]
    actuators: tuple[ActuatorIndex, ...]
    nq: int
    nv: int
    nu: int
    nbody: int
    nmocap: int
    nsite: int
    ntendon: int

    @classmethod
    def from_model(cls, model: Any) -> CompiledModelIndex:
        nq, nv, nu = (_int(getattr(model, name), name) for name in ("nq", "nv", "nu"))
        nbody = _int(model.nbody, "nbody", 1)
        njnt = _int(model.njnt, "njnt")
        nmocap = _int(model.nmocap, "nmocap")
        nsite, ntendon = (_int(getattr(model, name), name) for name in ("nsite", "ntendon"))
        nactuator = _int(getattr(model, "nactuator", nu), "nactuator")
        parent = _integers(model.body_parentid, (nbody,), "body_parentid")
        mocap = _integers(model.body_mocapid, (nbody,), "body_mocapid")
        joint_body = _integers(model.jnt_bodyid, (njnt,), "jnt_bodyid")
        joint_kind = _integers(model.jnt_type, (njnt,), "jnt_type")
        qadr = _integers(model.jnt_qposadr, (njnt,), "jnt_qposadr")
        vadr = _integers(model.jnt_dofadr, (njnt,), "jnt_dofadr")
        body_jadr = _integers(model.body_jntadr, (nbody,), "body_jntadr")
        body_jnum = _integers(model.body_jntnum, (nbody,), "body_jntnum")
        transmissions = _integers(model.actuator_trntype, (nactuator,), "actuator_trntype")
        targets = _integers(model.actuator_trnid, (nactuator, 2), "actuator_trnid")
        cstart = _integers(
            getattr(model, "actuator_ctrladr", np.arange(nactuator)),
            (nactuator,),
            "actuator_ctrladr",
        )
        cwidth = _integers(
            getattr(model, "actuator_ctrlnum", np.ones(nactuator, dtype=int)),
            (nactuator,),
            "actuator_ctrlnum",
        )
        body_names = _names(model, "body", nbody)
        joint_names = _names(model, "joint", njnt)
        actuator_names = _names(model, "actuator", nactuator)
        if parent[0] != 0 or np.any(parent < 0) or np.any(parent >= nbody):
            raise ValueError("body parents must be in range and world must parent itself")
        roots = [0] * nbody
        for body in range(1, nbody):
            current, visited = body, set()
            while parent[current] != 0:
                if current in visited:
                    raise ValueError("body parent topology contains a cycle")
                visited.add(current)
                current = int(parent[current])
            roots[body] = current
        if np.any(joint_body <= 0) or np.any(joint_body >= nbody):
            raise ValueError("joints must belong to non-world bodies")
        if mocap[0] != -1 or np.any(mocap < -1):
            raise ValueError("world cannot be mocap and negative mocap IDs must equal -1")
        _cover(tuple((int(value),) for value in mocap if value >= 0), nmocap, "mocap")
        joints = []
        for index in range(njnt):
            kind = int(joint_kind[index])
            if kind not in _JOINT_WIDTHS:
                raise ValueError(f"unknown native joint type {kind}")
            if qadr[index] < 0 or vadr[index] < 0:
                raise ValueError("joint state addresses must be nonnegative")
            qwidth, vwidth = _JOINT_WIDTHS[kind]
            joints.append(
                JointIndex(
                    index,
                    joint_names[index],
                    kind,
                    int(joint_body[index]),
                    tuple(range(int(qadr[index]), int(qadr[index]) + qwidth)),
                    tuple(range(int(vadr[index]), int(vadr[index]) + vwidth)),
                )
            )
        _cover(tuple(joint.qpos_indices for joint in joints), nq, "qpos")
        _cover(tuple(joint.qvel_indices for joint in joints), nv, "qvel")
        bodies = []
        for index in range(nbody):
            owned = tuple(j.id for j in joints if j.body_id == index)
            count, start = int(body_jnum[index]), int(body_jadr[index])
            if count < 0 or start < -1 or count != len(owned):
                raise ValueError("body joint counts differ from actual joint ownership")
            if count and (start < 0 or tuple(range(start, start + count)) != owned):
                raise ValueError("body joint addresses differ from actual joint ownership")
            if not count and start not in (-1, 0):
                raise ValueError("a body without joints must have an empty joint address")
            if mocap[index] >= 0 and (owned or parent[index] != 0):
                raise ValueError("mocap bodies must be jointless top-level bodies")
            if any(joints[joint].kind == 0 for joint in owned) and (
                len(owned) != 1 or parent[index] != 0
            ):
                raise ValueError("a free joint must be the sole joint on a top-level body")
            bodies.append(
                BodyIndex(
                    index,
                    body_names[index],
                    int(parent[index]),
                    roots[index],
                    owned,
                    int(mocap[index]),
                )
            )
        actuators = []
        for index in range(nactuator):
            kind = int(transmissions[index])
            target = tuple(int(value) for value in targets[index])
            if kind < 0 or kind == 1000 or any(value < -1 for value in target):
                raise ValueError("invalid native actuator transmission")
            first, second = target
            sizes = {0: njnt, 1: njnt, 2: nsite, 3: ntendon, 4: nsite, 5: nbody}
            if kind in sizes and not 0 <= first < sizes[kind]:
                raise ValueError("actuator transmission target is out of range")
            if kind == 2 and not 0 <= second < nsite:
                raise ValueError("slider-crank transmission needs two valid site IDs")
            if kind == 4 and not -1 <= second < nsite:
                raise ValueError("site reference target is out of range")
            if kind in (0, 1, 3, 5) and second != -1:
                raise ValueError("unused actuator transmission target must equal -1")
            start, width = int(cstart[index]), int(cwidth[index])
            if start < 0 or width < 1:
                raise ValueError("actuator control addresses and widths must be positive")
            actuators.append(
                ActuatorIndex(
                    index,
                    actuator_names[index],
                    kind,
                    (first, second),
                    tuple(range(start, start + width)),
                )
            )
        _cover(tuple(actuator.control_indices for actuator in actuators), nu, "control")
        return cls(
            tuple(bodies),
            tuple(joints),
            tuple(actuators),
            nq,
            nv,
            nu,
            nbody,
            nmocap,
            nsite,
            ntendon,
        )

    @property
    def root_ids(self) -> tuple[int, ...]:
        return tuple(body.id for body in self.bodies[1:] if body.parent_id == 0)

    @property
    def partitions(self) -> tuple[tuple[int, ...], ...]:
        """Native body IDs for each root in root_ids order; excludes world body."""
        return tuple(
            tuple(body.id for body in self.bodies if body.root_id == root) for root in self.root_ids
        )

    def body_id(self, name: str) -> int:
        if isinstance(name, str) and name:
            for body in self.bodies:
                if body.name == name:
                    return body.id
        raise ValueError(f"unknown or anonymous native body name {name!r}")

    def joint_id(self, name: str) -> int:
        if isinstance(name, str) and name:
            for joint in self.joints:
                if joint.name == name:
                    return joint.id
        raise ValueError(f"unknown or anonymous native joint name {name!r}")

    def joint_qpos_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(
            column for name in names for column in self.joints[self.joint_id(name)].qpos_indices
        )

    def joint_qvel_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(
            column for name in names for column in self.joints[self.joint_id(name)].qvel_indices
        )

    def free_root_layout(self, body_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        body = self.bodies[self.body_id(body_name)]
        if len(body.joint_ids) != 1 or self.joints[body.joint_ids[0]].kind != 0:
            raise NotImplementedError(f"body {body_name!r} does not own exactly one free joint")
        joint = self.joints[body.joint_ids[0]]
        return joint.qpos_indices, joint.qvel_indices

    def validate_entity_layout(self, layout: CompiledSceneLayout) -> None:
        """Cross-check a declared entity projection against actual compiled trees."""
        from unisim.scene_layout import CompiledSceneLayout

        if not isinstance(layout, CompiledSceneLayout):
            raise TypeError("layout must be CompiledSceneLayout")
        if (layout.nq, layout.nv, layout.nu, layout.nbody) != (
            self.nq,
            self.nv,
            self.nu,
            self.nbody,
        ):
            raise ValueError("public and native model dimensions differ")
        assigned: set[int] = set()
        for entity in layout.entities:
            prefix = entity.name + "/"
            root_id = self.body_id(prefix + entity.root_body)
            expected_bodies = tuple(body.id for body in self.bodies if body.root_id == root_id)
            if root_id == 0 or self.bodies[root_id].parent_id != 0:
                raise ValueError("entity root must be an actual top-level physical root")
            if set(entity.body_ids) != set(expected_bodies):
                raise ValueError("entity body IDs must cover exactly one native physical partition")
            assigned.update(entity.body_ids)
            for name, index, parent_name in zip(
                entity.body_names, entity.body_ids, entity.body_parent_names
            ):
                native_body = self.bodies[index]
                if native_body.name != prefix + name:
                    raise ValueError("entity body name/ID differs from native body")
                parent = 0 if parent_name is None else self.body_id(prefix + parent_name)
                if native_body.parent_id != parent:
                    raise ValueError("entity parent topology differs from native physical tree")
            root = self.bodies[root_id]
            root_joints = tuple(self.joints[index] for index in root.joint_ids)
            if entity.root_mode == "floating":
                actual_q, actual_v = self.free_root_layout(prefix + entity.root_body)
                if actual_q != entity.root_qpos_indices or actual_v != entity.root_qvel_indices:
                    raise ValueError("entity floating root columns differ from native addresses")
            elif root_joints or (entity.root_mode == "kinematic") != (root.mocap_id >= 0):
                raise ValueError("entity fixed/kinematic root mode differs from native root")
            native_joints = tuple(
                joint
                for joint in self.joints
                if joint.body_id in expected_bodies and joint.kind != 0
            )
            if len(native_joints) != len(entity.joints):
                raise ValueError("entity omits native non-root joints")
            declared_joint_ids = []
            for joint in entity.joints:
                native_joint = self.joints[self.joint_id(prefix + joint.name)]
                declared_joint_ids.append(native_joint.id)
                if (
                    native_joint.kind not in _PUBLIC_JOINT_KINDS
                    or _PUBLIC_JOINT_KINDS[native_joint.kind] != joint.kind
                    or native_joint.body_id != self.body_id(prefix + joint.body_name)
                    or native_joint.qpos_indices != joint.qpos_indices
                    or native_joint.qvel_indices != joint.qvel_indices
                ):
                    raise ValueError("entity joint semantics/addresses differ from native joint")
            if set(declared_joint_ids) != {joint.id for joint in native_joints}:
                raise ValueError("entity joints cross native physical partitions")
            for name, target_name, control in zip(
                entity.actuator_names, entity.actuator_joint_names, entity.actuator_indices
            ):
                native_actuator = next(
                    (a for a in self.actuators if control in a.control_indices), None
                )
                if (
                    native_actuator is None
                    or native_actuator.name != prefix + name
                    or native_actuator.trntype != 0
                    or native_actuator.control_indices != (control,)
                    or native_actuator.trnid[0] != self.joint_id(prefix + target_name)
                ):
                    raise ValueError("entity actuator mapping differs from native transmission")
        if assigned != set(range(1, self.nbody)):
            raise ValueError("entity layout leaves native physical bodies unowned")
