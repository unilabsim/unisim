"""Cold-normalized compatibility views over the one native scene executor.

External Python 3.8 workers load this file directly. The synthetic 7/6 root
and D controls are historical wire coordinates, not a claim that the authored
asset declares a free joint or an actuator for every DoF. Native topology and
source policy remain with the cold loader; this module performs no physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LegacyJoint:
    name: str
    qpos_indices: Tuple[int, ...]
    qvel_indices: Tuple[int, ...]


@dataclass(frozen=True)
class LegacyEntity:
    name: str
    root_body: str
    body_names: Tuple[str, ...]
    body_ids: Tuple[int, ...]
    joints: Tuple[LegacyJoint, ...]
    actuator_names: Tuple[str, ...]
    actuator_joint_names: Tuple[str, ...]
    actuator_indices: Tuple[int, ...]
    root_qpos_indices: Tuple[int, ...] = tuple(range(7))
    root_qvel_indices: Tuple[int, ...] = tuple(range(6))
    kind: str = "articulation"
    root_mode: str = "floating"

    @property
    def qpos_indices(self) -> Tuple[int, ...]:
        return self.root_qpos_indices + tuple(
            i for joint in self.joints for i in joint.qpos_indices
        )

    @property
    def qvel_indices(self) -> Tuple[int, ...]:
        return self.root_qvel_indices + tuple(
            i for joint in self.joints for i in joint.qvel_indices
        )


@dataclass(frozen=True, init=False)
class LegacyExecutionLayout:
    """Frozen execution coordinates, intentionally not a public scene layout.

    Do not serialize this as CompiledSceneLayout: parent topology/root kind and
    actuator ownership are not inferred from the old flat wire description.
    """

    entities: Tuple[LegacyEntity, ...]
    nq: int
    nv: int
    nu: int
    nbody: int

    def __init__(
        self,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        root_body_name: Optional[str] = None,
    ) -> None:
        joints, bodies = tuple(joint_names), tuple(body_names)
        for names, label in ((joints, "joint"), (bodies, "body")):
            if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
                raise ValueError("legacy " + label + " names must be nonempty and unique")
        if not bodies:
            raise ValueError("legacy native scene must contain at least one body")
        root = bodies[0] if root_body_name is None else root_body_name
        if root not in bodies:
            raise ValueError("legacy root is absent from native body names")
        count = len(joints)
        entity = LegacyEntity(
            "legacy_model",
            root,
            bodies,
            tuple(range(len(bodies))),
            tuple(LegacyJoint(name, (7 + i,), (6 + i,)) for i, name in enumerate(joints)),
            joints,
            joints,
            tuple(range(count)),
        )
        for key, value in (
            ("entities", (entity,)),
            ("nq", 7 + count),
            ("nv", 6 + count),
            ("nu", count),
            ("nbody", len(bodies)),
        ):
            object.__setattr__(self, key, value)

    def get_entity(self, name: str) -> LegacyEntity:
        if name != "legacy_model":
            raise ValueError("unknown legacy execution entity")
        return self.entities[0]


class LegacySlotProjection:
    """Translate old public buffers into the mapped executor's canonical slots."""

    def __init__(
        self,
        protocol: Any,
        num_envs: int,
        layout: LegacyExecutionLayout,
        *,
        root_com: Optional[np.ndarray] = None,
        body_com: Optional[np.ndarray] = None,
    ) -> None:
        self.protocol, self.num_envs, self.layout = protocol, num_envs, layout
        self.legacy_slots: dict[str, np.ndarray] = {}
        self.slots: dict[str, np.ndarray] = {}
        self.root_com = (
            None
            if root_com is None
            else np.asarray(root_com, dtype=np.float64).reshape(num_envs, 3).copy()
        )
        self.body_com = (
            None
            if body_com is None
            else np.asarray(body_com, dtype=np.float64).reshape(num_envs, layout.nbody, 3).copy()
        )
        if any(
            value is not None and not np.isfinite(value).all()
            for value in (self.root_com, self.body_com)
        ):
            raise ValueError("native legacy COM offsets must be finite")

    def attach(self, legacy_slots: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        expected = self.protocol.slot_shapes(self.num_envs, self.layout.nu, self.layout.nbody)
        if set(legacy_slots) != set(expected):
            raise ValueError("legacy slots do not match the negotiated wire")
        for name, shape in expected.items():
            if legacy_slots[name].shape != shape or legacy_slots[
                name
            ].dtype != self.protocol.slot_dtype(name):
                raise ValueError("invalid legacy slot shape/dtype: " + name)
        self.legacy_slots = legacy_slots
        self.slots = {
            name: np.zeros(shape, dtype=self.protocol.slot_dtype(name))
            for name, shape in self.protocol.scene_slot_shapes(self.num_envs, self.layout).items()
        }
        # Policy input remains a direct shared-memory view; all other public
        # arrays are output projections and cannot mutate canonical addresses.
        self.slots["ctrl"] = legacy_slots["ctrl"]
        self.slots["entity_root_state"][..., 3] = 1
        self.slots["qpos"][:, 3] = 1
        return self.slots

    def publish(self) -> None:
        old, current = self.legacy_slots, self.slots
        roots = current["entity_root_state"][:, 0].copy()
        bodies = current["body_state"].copy()
        if self.root_com is not None:
            roots[:, 7:10] += np.cross(
                roots[:, 10:13], self.protocol.quat_rotate(roots[:, 3:7], self.root_com)
            )
        if self.body_com is not None:
            bodies[..., 7:10] += np.cross(
                bodies[..., 10:13], self.protocol.quat_rotate(bodies[..., 3:7], self.body_com)
            )
        np.copyto(old["root_state"], roots)
        np.copyto(old["dof_state"][..., 0], current["qpos"][:, 7:])
        np.copyto(old["dof_state"][..., 1], current["qvel"][:, 6:])
        np.copyto(old["body_state"], bodies)
        np.copyto(old["contact_force"], current["contact_force"])

    def prepare_reset(self, count: int) -> dict[str, Any]:
        """Prevalidate old full-state input, then populate one mapped transaction.

        Old SET_STATE's root angular velocity is body-frame even though its
        root output slot uses world angular velocity. Gym's old linear channel
        was native COM velocity; its explicit projection preserves that API.
        """
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.num_envs:
            raise ValueError("legacy reset count is invalid")
        old, current = self.legacy_slots, self.slots
        ids = old["reset_env_ids"][:count].copy()
        if len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("legacy reset environment IDs must be distinct and in range")
        qpos, qvel = old["reset_qpos"][:count].copy(), old["reset_qvel"][:count].copy()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("legacy reset state must be finite")
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, rtol=0, atol=1e-5):
            raise ValueError("legacy reset requires unit wxyz quaternions")
        roots = np.empty((count, 1, 13), dtype=np.float32)
        roots[:, 0, :7] = qpos[:, :7]
        roots[:, 0, 10:] = self.protocol.quat_rotate(qpos[:, 3:7], qvel[:, 3:6])
        roots[:, 0, 7:10] = qvel[:, :3]
        if self.root_com is not None:
            roots[:, 0, 7:10] -= np.cross(
                roots[:, 0, 10:], self.protocol.quat_rotate(qpos[:, 3:7], self.root_com[ids])
            )
            qvel[:, :3] = roots[:, 0, 7:10]
        if not np.isfinite(roots).all() or not np.isfinite(qvel).all():
            raise ValueError("legacy root velocity conversion overflowed")
        current["reset_env_ids"][:count] = ids
        current["reset_qpos"][:count] = qpos
        current["reset_qvel"][:count] = qvel
        current["reset_entity_root_state"][:count] = roots
        for name in ("reset_qpos_mask", "reset_qvel_mask", "reset_root_mask"):
            current[name].fill(1)
        return {
            "count": count,
            "entity_names": ["legacy_model"],
            "control_values": current["ctrl"][ids].copy().tolist(),
        }
