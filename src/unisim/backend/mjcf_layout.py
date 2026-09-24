"""Stdlib-only MJCF joint-layout extraction for playback validation.

Backends that serve raw native generalized-state rows as physics-state
playback snapshots need the source MJCF joint inventory in MuJoCo
generalized-state order to prove (at construction, fail-closed) that their
native ordering matches the order a MuJoCo playback shell compiles.  The
extractor is deliberately dependency-free (no MuJoCo import) so every
adapter — including subprocess workers — can share it on cold paths.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MjcfJointLayoutEntry:
    """One MJCF joint and its MuJoCo-order generalized-state addresses."""

    name: str
    kind: str
    body_name: str
    qpos_address: int
    qvel_address: int
    num_dof_pos: int
    num_dof_vel: int


_MJCF_JOINT_DOF_WIDTHS = {
    "free": (7, 6),
    "ball": (4, 3),
    "hinge": (1, 1),
    "slide": (1, 1),
}


def _iter_mjcf_children(
    element: ET.Element,
    base_dir: Path,
    include_stack: tuple[Path, ...],
) -> Iterator[tuple[ET.Element, Path, tuple[Path, ...]]]:
    """Yield ``(child, base_dir, include_stack)`` triples with includes inlined.

    MuJoCo replaces an ``<include file="...">`` element with the children of
    the included file's root, so the walker tracks the directory each element
    was parsed from to keep nested relative includes resolvable.
    """
    for child in element:
        if child.tag != "include":
            yield child, base_dir, include_stack
            continue
        include_file = child.get("file")
        if not include_file:
            raise ValueError("MJCF <include> requires a file attribute")
        include_path = (base_dir / include_file).resolve()
        if include_path in include_stack:
            raise ValueError(f"cyclic MJCF <include> of {include_path}")
        included_root = ET.parse(include_path).getroot()
        yield from _iter_mjcf_children(
            included_root, include_path.parent, (*include_stack, include_path)
        )


def extract_mjcf_joint_layout(model_file: str) -> tuple[MjcfJointLayoutEntry, ...]:
    """Return the MJCF joints of ``model_file`` in MuJoCo generalized-state order.

    MuJoCo assigns qpos/qvel addresses depth-first over the worldbody tree:
    every joint of a body (in document order) precedes its child bodies, while
    ``<frame>`` wrappers and ``<include>`` inlining stay transparent.  Backends
    whose physics-state playback snapshots serve raw native generalized-state
    rows (Motrix ``dof_pos``/``dof_vel``, Genesis host qpos/qvel caches) produce
    columns that are only valid while the native ordering matches this source
    ordering; each backend validates that at build time.
    """
    model_path = Path(model_file).resolve()
    root = ET.parse(model_path).getroot()
    # MuJoCo merges every <worldbody> section — including those inlined from
    # <include> files — into one tree, so joints from all of them participate
    # in document order; stopping at the first section would drop included
    # bodies (e.g. a scene that includes both a hand and a free ball).
    worldbodies = [
        (child, base_dir, include_stack)
        for child, base_dir, include_stack in _iter_mjcf_children(
            root, model_path.parent, (model_path,)
        )
        if child.tag == "worldbody"
    ]
    if not worldbodies:
        raise ValueError(f"MJCF {model_path} has no <worldbody>")

    entries: list[MjcfJointLayoutEntry] = []
    seen_names: set[str] = set()
    qpos_address = 0
    qvel_address = 0

    def walk(parent: ET.Element, base_dir: Path, include_stack: tuple[Path, ...]) -> None:
        nonlocal qpos_address, qvel_address
        for body, body_dir, body_stack in _iter_mjcf_children(parent, base_dir, include_stack):
            if body.tag == "frame":
                walk(body, body_dir, body_stack)
                continue
            if body.tag != "body":
                continue
            body_name = body.get("name") or ""
            for joint, _, _ in _iter_mjcf_children(body, body_dir, body_stack):
                if joint.tag not in ("joint", "freejoint"):
                    continue
                kind = "free" if joint.tag == "freejoint" else (joint.get("type") or "hinge")
                if kind not in _MJCF_JOINT_DOF_WIDTHS:
                    raise NotImplementedError(
                        f"MJCF playback joint-order validation does not support MJCF "
                        f"joint type {kind!r} on body {body_name!r}"
                    )
                num_dof_pos, num_dof_vel = _MJCF_JOINT_DOF_WIDTHS[kind]
                name = joint.get("name") or ""
                if name:
                    if name in seen_names:
                        raise ValueError(
                            f"MJCF playback joint-order validation requires unique MJCF "
                            f"joint names; {name!r} is duplicated"
                        )
                    seen_names.add(name)
                entries.append(
                    MjcfJointLayoutEntry(
                        name=name,
                        kind=kind,
                        body_name=body_name,
                        qpos_address=qpos_address,
                        qvel_address=qvel_address,
                        num_dof_pos=num_dof_pos,
                        num_dof_vel=num_dof_vel,
                    )
                )
                qpos_address += num_dof_pos
                qvel_address += num_dof_vel
            walk(body, body_dir, body_stack)

    for worldbody in worldbodies:
        walk(*worldbody)
    return tuple(entries)


__all__ = ["MjcfJointLayoutEntry", "extract_mjcf_joint_layout"]
