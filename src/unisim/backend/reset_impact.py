"""Cold-bound cleanup addresses for MuJoCo-family entity transactions.

These private execution records do not expand the public reset contract.
Runtime selection unions immutable addresses without re-reading model metadata
or reconstructing the body tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from unisim.scene_layout import BoundSceneReset, CompiledSceneLayout


@dataclass(frozen=True)
class ResetImpact:
    bodies: tuple[int, ...] = ()
    dofs: tuple[int, ...] = ()
    controls: tuple[int, ...] = ()
    activations: tuple[int, ...] = ()


def _union(impacts: Sequence[ResetImpact]) -> ResetImpact:
    return ResetImpact(*(tuple(sorted({index for impact in impacts for index in
                                      getattr(impact, field)}))
                         for field in ("bodies", "dofs", "controls", "activations")))


@dataclass(frozen=True)
class ResetImpactIndex:
    roots: Mapping[str, ResetImpact]
    joints: Mapping[tuple[str, str], ResetImpact]

    def select(self, binding: BoundSceneReset) -> ResetImpact:
        selected = []
        for item in binding.patches:
            if item.patch.root_pose is not None or item.patch.root_velocity is not None:
                selected.append(self.roots[item.entity.name])
            else:
                selected.extend(
                    self.joints[(item.entity.name, joint.name)] for joint in item.joints)
        return _union(selected)


def bind_reset_impacts(
    layout: CompiledSceneLayout,
    activation_addresses: Sequence[int],
    activation_counts: Sequence[int],
) -> ResetImpactIndex:
    """Resolve descendant wrench bodies and actuator activation storage once."""
    roots, joints = {}, {}
    for entity in layout.entities:
        parents = dict(zip(entity.body_names, entity.body_parent_names, strict=True))
        by_body: dict[str, set[int]] = {name: set() for name in entity.body_names}
        for name, body in zip(entity.body_names, entity.body_ids, strict=True):
            ancestor: str | None = name
            while ancestor is not None:
                by_body[ancestor].add(body)
                ancestor = parents[ancestor]
        controls_by_joint: dict[str, list[int]] = {joint.name: [] for joint in entity.joints}
        for control, target in zip(
            entity.actuator_indices, entity.actuator_joint_names, strict=True
        ):
            controls_by_joint[target].append(control)

        def activations(controls: Sequence[int]) -> tuple[int, ...]:
            result: list[int] = []
            for control in controls:
                start, count = int(activation_addresses[control]), int(activation_counts[control])
                if count < 0 or start < -1 or (count and start < 0):
                    raise ValueError("invalid compiled actuator activation address")
                result.extend(range(start, start + count))
            return tuple(sorted(result))

        roots[entity.name] = ResetImpact(
            tuple(sorted(entity.body_ids)), tuple(sorted(entity.qvel_indices)),
            tuple(sorted(entity.actuator_indices)), activations(entity.actuator_indices))
        for joint in entity.joints:
            controls = tuple(sorted(controls_by_joint[joint.name]))
            joints[(entity.name, joint.name)] = ResetImpact(
                tuple(sorted(by_body[joint.body_name])), tuple(sorted(joint.qvel_indices)),
                controls, activations(controls))
    return ResetImpactIndex(MappingProxyType(roots), MappingProxyType(joints))
