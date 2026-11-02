"""Whole-MJCF fixed-variant materialization and public-layout audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unisim.dr.types import FixedVariantLayout
from unisim.scene import SceneCfg

from ..mjcf_layout import extract_mjcf_joint_layout
from .scene import materialize_scene_with_full_sensor_inventory


@dataclass
class MotrixWholeModelRuntime:
    """One whole-model realization and its assigned public environment rows."""

    variant: int
    rows: np.ndarray
    model: Any
    data: Any
    sensor_names: tuple[str, ...]
    sensor_signatures: tuple[tuple[str, str, str], ...]
    frame_quat_sensor_names: frozenset[str]
    found_contact_geom_pairs: dict[str, tuple[int, int]]
    body: Any
    body_link: Any
    position_actuators: tuple[Any, ...]
    default_public_qpos: np.ndarray | None = None
    default_public_qvel: np.ndarray | None = None
    default_native_qpos: np.ndarray | None = None
    default_body_mass: np.ndarray | None = None
    default_body_ipos: np.ndarray | None = None
    default_geom_friction: np.ndarray | None = None
    default_actuator_kp: np.ndarray | None = None
    default_actuator_kd: np.ndarray | None = None
    default_gravity: np.ndarray | None = None
    default_geom_sizes: np.ndarray | None = None
    links_by_id: dict[int, Any] = field(default_factory=dict)
    geoms_by_id: dict[int, Any] = field(default_factory=dict)

    def local_rows(self, public_rows: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.rows, public_rows)


def _layout_signature(model: Any) -> tuple[tuple[str, Any], ...]:
    def object_name(value: Any) -> str:
        return "" if value is None else str(value.name)

    def link_name(geom: Any) -> str:
        return object_name(getattr(geom, "link", None))

    floating_bases_list: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for link in model.links:
        body = model.get_body(str(link.name))
        floating_base = None if body is None else body.floatingbase
        if floating_base is not None:
            floating_bases_list.append(
                (
                    str(link.name),
                    tuple(int(index) for index in floating_base.dof_pos_indices),
                    tuple(int(index) for index in floating_base.dof_vel_indices),
                )
            )
    floating_bases = tuple(floating_bases_list)
    return (
        (
            "dimensions",
            (
                int(model.num_dof_pos),
                int(model.num_dof_vel),
                int(model.num_actuators),
                int(model.num_links),
                int(model.num_geoms),
                int(model.num_sites),
            ),
        ),
        (
            "links",
            tuple(
                (int(link.index), str(link.name), object_name(getattr(link, "parent", None)))
                for link in model.links
            ),
        ),
        (
            "joints",
            tuple(
                (
                    int(joint.index),
                    str(joint.name),
                    int(joint.dof_pos_index),
                    int(joint.dof_vel_index),
                    int(joint.num_dof_pos),
                    int(joint.num_dof_vel),
                )
                for joint in model.joints
            ),
        ),
        (
            "actuators",
            tuple(
                (
                    int(actuator.index),
                    str(actuator.name),
                    str(actuator.typ),
                    str(actuator.target_type),
                    str(actuator.target_name),
                )
                for actuator in model.actuators
            ),
        ),
        (
            "geoms",
            tuple(
                (
                    int(geom.index),
                    str(geom.name),
                    link_name(geom),
                    type(geom).__name__,
                    int(getattr(geom, "collision_group", 0)),
                    int(getattr(geom, "collision_affinity", 0)),
                )
                for geom in model.geoms
            ),
        ),
        (
            "sites",
            tuple(
                (int(site.index), str(site.name), object_name(site.parent_link))
                for site in model.sites
            ),
        ),
        ("floating_bases", floating_bases),
    )


def _compare_array(
    source: str,
    field_name: str,
    actual: Any,
    expected: Any,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape or not np.array_equal(
        actual_array, expected_array
    ):
        raise ValueError(
            f"Motrix fixed variant source {source!r} changes public {field_name}: "
            f"{actual_array!r} != {expected_array!r}"
        )


def materialize_whole_model_variants(
    scene: SceneCfg,
    *,
    num_envs: int,
    sim_dt: float,
    base_name: str,
    add_body_sensors: bool,
    max_iterations: int,
) -> tuple[MotrixWholeModelRuntime, ...]:
    """Load and audit every source, then bind runtimes for the assigned variants."""
    import motrixsim as mtx

    plan = scene.fixed_variant_plan
    if plan is None:
        raise ValueError("whole-MJCF variant materialization requires a fixed variant plan")
    plan.validate(num_envs)
    if plan.layout is not FixedVariantLayout.SAME_LAYOUT:
        raise NotImplementedError(
            f"Motrix whole-MJCF fixed variants support only same_layout; got {plan.layout.value!r}"
        )
    if scene.terrain is not None:
        raise NotImplementedError("Motrix whole-MJCF fixed variants do not support terrain")

    assignment = np.asarray(plan.assignment, dtype=np.int32)
    used = set(int(variant) for variant in np.unique(assignment))
    canonical_signature: tuple[tuple[str, Any], ...] | None = None
    canonical_joint_layout: tuple[Any, ...] | None = None
    canonical_sensors: tuple[tuple[str, str, str], ...] | None = None
    canonical_control_limits: np.ndarray | None = None
    canonical_joint_limits: np.ndarray | None = None
    runtimes: list[MotrixWholeModelRuntime] = []

    for variant, descriptor in enumerate(plan.variants):
        source = str(Path(descriptor.model_file).resolve())
        model, inventory = materialize_scene_with_full_sensor_inventory(
            model_file=source,
            fragment_files=scene.fragment_files,
            add_body_sensors=add_body_sensors,
            base_name=base_name,
        )
        model.options.timestep = float(sim_dt)
        model.options.max_iterations = int(max_iterations)
        signature = _layout_signature(model)
        joint_layout = extract_mjcf_joint_layout(source)
        if canonical_signature is None:
            canonical_signature = signature
            canonical_joint_layout = joint_layout
            canonical_sensors = inventory.signatures
            canonical_control_limits = np.asarray(model.actuator_ctrl_limits).copy()
            canonical_joint_limits = np.asarray(model.joint_limits).copy()
        else:
            actual = dict(signature)
            expected = dict(canonical_signature)
            for field_name in expected:
                if actual[field_name] != expected[field_name]:
                    raise ValueError(
                        f"Motrix fixed variant source {source!r} changes public {field_name} layout"
                    )
            if joint_layout != canonical_joint_layout:
                raise ValueError(
                    f"Motrix fixed variant source {source!r} changes public "
                    "generalized-state layout"
                )
            if inventory.signatures != canonical_sensors:
                raise ValueError(
                    f"Motrix fixed variant source {source!r} changes public sensor identity"
                )
            _compare_array(
                source,
                "actuator control limits",
                model.actuator_ctrl_limits,
                canonical_control_limits,
            )
            _compare_array(source, "joint limits", model.joint_limits, canonical_joint_limits)

        rows = np.flatnonzero(assignment == variant).astype(np.intp, copy=False)
        data = mtx.SceneData(model, batch=[int(rows.size) if variant in used else 1])
        expected_dimensions = {
            name: int(np.asarray(model.get_sensor_value(name, data)).shape[-1])
            for name in inventory.names
        }
        for name, width in expected_dimensions.items():
            values = np.asarray(model.get_sensor_value(name, data))
            if values.ndim != 2 or values.shape != (int(data.dof_pos.shape[0]), width):
                raise RuntimeError(
                    f"Motrix fixed variant source {source!r} sensor {name!r} "
                    f"has invalid public shape {values.shape}"
                )
        if variant not in used:
            continue

        found_pairs: dict[str, tuple[int, int]] = {}
        for identity in inventory.contact_identities:
            if not identity.reports_found:
                continue
            geom1 = model.get_geom_index(identity.geom1)
            geom2 = model.get_geom_index(identity.geom2)
            if geom1 is None or geom2 is None:
                raise RuntimeError(
                    f"Motrix fixed variant source {source!r} contact sensor "
                    f"{identity.name!r} refers to an absent geom"
                )
            found_pairs[identity.name] = (int(geom1), int(geom2))
        body = model.get_body(base_name)
        body_link = model.get_link(base_name)
        if body is None or body_link is None:
            raise ValueError(
                f"Motrix fixed variant source {source!r} is missing base body {base_name!r}"
            )
        runtimes.append(
            MotrixWholeModelRuntime(
                variant=variant,
                rows=rows,
                model=model,
                data=data,
                sensor_names=inventory.names,
                sensor_signatures=inventory.signatures,
                frame_quat_sensor_names=frozenset(
                    identity.name
                    for identity in inventory.frame_identities
                    if identity.sensor_type == mtx.msd.FrameSensorType.FrameQuat
                ),
                found_contact_geom_pairs=found_pairs,
                body=body,
                body_link=body_link,
                position_actuators=tuple(
                    actuator for actuator in model.actuators if actuator.typ == "position"
                ),
                links_by_id={int(link.index): link for link in model.links},
                geoms_by_id={int(geom.index): geom for geom in model.geoms},
            )
        )

    if not runtimes:
        raise RuntimeError("Motrix fixed variant plan materialized no assigned runtime")
    return tuple(runtimes)


__all__ = ["MotrixWholeModelRuntime", "materialize_whole_model_variants"]
