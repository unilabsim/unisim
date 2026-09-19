"""Cold-path authoring plans shared by SuperDex materialization and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from unisim.scene_layout import CompiledSceneLayout


@dataclass(frozen=True)
class NativeActorPlan:
    """One immutable public-to-native actor address mapping."""

    entity_name: str | None
    root_body_id: int
    floating: bool
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    native_qpos_indices: np.ndarray
    native_qvel_indices: np.ndarray
    native_order_qpos_indices: np.ndarray
    native_order_qvel_indices: np.ndarray
    body_ids: np.ndarray
    local_body_link_indices: np.ndarray
    actuator_indices: np.ndarray
    native_actuator_qpos_indices: np.ndarray
    native_actuator_qvel_indices: np.ndarray
    spawn_actor: Callable[[Any], Any]


@dataclass
class NativeSceneActors:
    """All actors created in one native scene, in frozen executor-slot order."""

    actors: tuple[Any, ...]
    actor_links: tuple[tuple[Any, ...], ...]
    actor_dof_counts: tuple[int, ...]
    cleanups: tuple[Callable[[], None], ...]
    body_actor_indices: np.ndarray
    flattened_body_link_indices: np.ndarray


@dataclass(frozen=True)
class SensorPlan:
    name: str
    kind: str
    body_id: int = 0
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint_index: int = -1
    dim: int = 3
    native_link_index: int = -1
    other_actor_name: str | None = None
    contact_distance: float = 0.0
    other_body_id: int = -1
    source_actor_index: int = -1
    source_link_index: int = -1
    other_actor_index: int = -1
    other_link_index: int = -1


@dataclass
class ModelPlan:
    """Canonical arrays and a native scene constructor; no hot-path XML access.

    Body zero is the world. ``body_link_indices`` maps every other canonical
    body to the native articulation's nested link actors. Free roots use an
    identity native reference transform: qpos is world xyz + wxyz followed by
    single-DoF joints; qvel is world origin velocity + body angular velocity
    followed by single-DoF joints. Native free qvel uses world angular velocity.
    """

    source_file: str
    nq: int
    nv: int
    root_body_id: int
    floating: bool
    body_names: tuple[str, ...]
    body_parent_ids: np.ndarray
    body_link_indices: np.ndarray
    body_mass: np.ndarray
    body_ipos: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    joint_names: tuple[str, ...]
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    joint_ranges: np.ndarray
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_qpos_indices: np.ndarray
    actuator_qvel_indices: np.ndarray
    actuator_ctrl_ranges: np.ndarray
    actuator_gear: np.ndarray
    actuator_kp: np.ndarray
    actuator_kd: np.ndarray
    default_qpos: np.ndarray
    keyframes: dict[str, np.ndarray]
    gravity: np.ndarray
    sensors: tuple[SensorPlan, ...]
    # Receives a native scene and returns its articulated actor and an
    # idempotent owner cleanup callback (e.g. RoboticsContext/Bot teardown).
    spawn_actor: Callable[[Any], tuple[Any, Callable[[], None]]]
    cleanup: Callable[[], None]
    # Receives a native scene and returns every executor actor in frozen slot
    # order. Whole-model plans use one slot; portable entity plans use one slot
    # per public entity (including zero-DoF static actors).
    spawn_scene: Callable[[Any], NativeSceneActors] | None = None
    actuator_force_ranges: np.ndarray | None = None
    dof_armature: np.ndarray | None = None
    layout: CompiledSceneLayout | None = None
    actor_plans: tuple[NativeActorPlan, ...] = ()
    actuator_slot_indices: np.ndarray | None = None
