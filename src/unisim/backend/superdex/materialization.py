"""Audited, cold-path native-bot and MJCF materialization for SuperDex."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from unisim.backend.superdex.geometry import primitive_shape, rotation_matrix
from unisim.backend.superdex.plans import ModelPlan, NativeActorPlan, SensorPlan
from unisim.dr.types import FixedVariantLayout
from unisim.scene import SceneCfg


def _noop() -> None:
    pass


def _transform(p: Any, pos: Any, quat: Any = (1, 0, 0, 0)) -> Any:
    return p.TransformRT(translation=pos, rotation=np.asarray(quat)[[1, 2, 3, 0]])


def _effort_ranges(values: Sequence[float], count: int) -> np.ndarray:
    limits = np.asarray(values, dtype=float)
    if limits.shape != (count,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError(f"superdex effort_limits must contain {count} finite positive values")
    return np.column_stack((-limits, limits))


def materialize_model(
    physics: Any,
    robotics: Any,
    scene: SceneCfg,
    *,
    effort_limits: Sequence[float] | None = None,
    allow_contact_approximation: bool = False,
    sim_dt: float = 0.002,
) -> ModelPlan:
    """Resolve all model metadata and shapes before any rollout begins."""
    if scene.terrain is not None:
        raise NotImplementedError("superdex supports authored static planes, not generated terrain")
    if scene.entity_assets:
        if scene.entity_variant is not None:
            return _fixed_variant_plan(
                physics,
                scene,
                effort_limits,
                allow_contact_approximation,
                sim_dt,
            )
        return _portable_plan(
            physics,
            scene,
            effort_limits,
            allow_contact_approximation,
            sim_dt,
        )
    path = Path(scene.model_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".superdex_bot":
        if scene.fragment_files:
            raise NotImplementedError("superdex native bot does not accept MJCF fragments")
        return _native_plan(physics, robotics, path, effort_limits)
    if path.suffix != ".xml":
        raise NotImplementedError("superdex model must be .superdex_bot or audited .xml MJCF")
    return _mjcf_plan(physics, path, scene, effort_limits, allow_contact_approximation)


def _native_plan(p: Any, r: Any, path: Path, efforts: Sequence[float] | None) -> ModelPlan:
    cfg = r.load_bot_prefab_from_file(str(path))
    joints = list(cfg.joints)
    links = list(cfg.links)
    if len(cfg.cycles) or len(cfg.linear_transmissions) or len(cfg.spatial_tendons):
        raise NotImplementedError(
            "superdex native bot transmissions/cycles require a separate audit"
        )
    if any(len(link.sensors) or len(link.actuators) for link in links):
        raise NotImplementedError("superdex native bot sensor/actuator components are unsupported")
    if joints[0].type != p.ArticulatedJointType.HARD:
        raise NotImplementedError("superdex native bot currently requires a fixed HARD root")
    if any(
        j.type
        not in (
            p.ArticulatedJointType.HARD,
            p.ArticulatedJointType.REVOLUTE,
            p.ArticulatedJointType.PRISMATIC,
        )
        for j in joints
    ):
        raise NotImplementedError("superdex native bot supports only fixed/hinge/slide joints")
    active = [i for i, joint in enumerate(joints) if joint.type != p.ArticulatedJointType.HARD]
    n = len(active)
    if efforts is None:
        efforts = [float(joints[i].effort_limit) for i in active]
        if any(v <= 0 or not np.isfinite(v) for v in efforts):
            raise ValueError("superdex native bot requires explicit finite effort_limits")
    ctrl_ranges = _effort_ranges(efforts, n)
    ranges = []
    for i in active:
        joint = joints[i]
        axis = np.asarray(joint.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        ranges.append(
            [
                -np.inf if joint.min_limit is None else np.dot(joint.min_limit, axis),
                np.inf if joint.max_limit is None else np.dot(joint.max_limit, axis),
            ]
        )
    # A temporary cold scene resolves authored/density-derived mass properties.
    temp = p.create_scene("superdex_metadata")
    context = r.create_context()
    bot = None
    try:
        bot = r.create_bot(temp, cfg, context)
        actor = bot.get_articulated_actor()
        native_links = [temp.get_actor(h) for h in actor.get_nested_link_actors()]
        masses = [0.0]
        coms = [np.zeros(3)]
        for link, authored in zip(native_links, links, strict=True):
            # HARD world chains are static and do not expose get_mass().
            if link.is_static() and authored.mass is None and authored.shape_file:
                raise NotImplementedError(
                    "superdex native static links with collision assets require an explicit mass"
                )
            masses.append(float(authored.mass or 0) if link.is_static() else link.get_mass())
            pose = link.get_root_transform()
            rotation = rotation_matrix(np.asarray(pose.rotation)[[3, 0, 1, 2]])
            world_offset = np.asarray(link.get_center_of_mass_transform().translation) - np.asarray(
                pose.translation
            )
            coms.append(rotation.T @ world_offset)
        q0 = np.empty(n, dtype=np.float64 if p.uses_double_precision() else np.float32)
        actor.get_articulated_pose(q0)
    finally:
        if bot is not None:
            r.destroy_bot(temp, bot)
        p.destroy_scene(temp)

    def spawn(native_scene: Any) -> tuple[Any, Any]:
        owner = r.create_context()
        instance = r.create_bot(native_scene, cfg, owner)
        live = True

        def close() -> None:
            nonlocal live, owner
            if live:
                live = False
                r.destroy_bot(native_scene, instance)
                # Keep the explicit context wrapper alive until bot teardown. The
                # SDK owns the process singleton; Python must not destroy it here.
                owner = None

        return instance.get_articulated_actor(), close

    names = tuple(str(joints[i].name) for i in active)
    return ModelPlan(
        source_file=str(path),
        nq=n,
        nv=n,
        root_body_id=1,
        floating=False,
        body_names=("world", *(str(link.name) for link in links)),
        body_parent_ids=np.array([0, *(int(link.parent_link) + 1 for link in links)]),
        body_link_indices=np.array([-1, *range(len(links))]),
        body_mass=np.asarray(masses),
        body_ipos=np.asarray(coms),
        body_pos=np.zeros((len(links) + 1, 3)),
        body_quat=np.tile((1.0, 0.0, 0.0, 0.0), (len(links) + 1, 1)),
        joint_names=names,
        joint_qpos_indices=np.arange(n),
        joint_qvel_indices=np.arange(n),
        joint_ranges=np.asarray(ranges).reshape(n, 2),
        actuator_names=names,
        actuator_joint_names=names,
        actuator_qpos_indices=np.arange(n),
        actuator_qvel_indices=np.arange(n),
        actuator_ctrl_ranges=ctrl_ranges,
        actuator_gear=np.ones(n),
        actuator_kp=np.zeros(n),
        actuator_kd=np.zeros(n),
        default_qpos=q0.astype(float),
        keyframes={},
        gravity=np.array([0, 0, -9.81]),
        sensors=(),
        spawn_actor=spawn,
        cleanup=_noop,
        actuator_force_ranges=ctrl_ranges.copy(),
        dof_armature=np.array([float(joints[i].inertia or 0) for i in active]),
    )


def _load_mjcf(path: Path, scene: SceneCfg) -> tuple[Any, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "superdex MJCF materialization requires the optional mujoco parser"
        ) from exc
    composed = None
    try:
        if scene.fragment_files:
            from unisim.backend.mujoco.xml import materialize_scene_fragments

            composed = materialize_scene_fragments(str(path), fragment_files=scene.fragment_files)
        model = mujoco.MjModel.from_xml_path(composed or str(path))
    finally:
        if composed is not None:
            os.unlink(composed)
    return mujoco, model


def _audit_model(mj: Any, m: Any, *, portable: bool = False) -> None:
    unsupported = {"neq", "ntendon", "nflex", "nhfield", "nplugin"}
    if not portable:
        unsupported.add("nmocap")
    for field in unsupported:
        if int(getattr(m, field, 0)):
            raise NotImplementedError(f"superdex MJCF does not support {field} features")
    if np.any(m.body_jntnum > 1):
        raise NotImplementedError("superdex MJCF supports at most one joint per body")
    root_count = int(np.count_nonzero(np.asarray(m.body_parentid)[1:] == 0))
    if root_count != 1 and not portable:
        raise NotImplementedError("superdex MJCF requires one articulated body tree")
    supported = {
        int(mj.mjtJoint.mjJNT_FREE),
        int(mj.mjtJoint.mjJNT_HINGE),
        int(mj.mjtJoint.mjJNT_SLIDE),
    }
    if any(int(t) not in supported for t in m.jnt_type):
        raise NotImplementedError("superdex MJCF supports free, hinge and slide joints")
    free = np.flatnonzero(m.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    if len(free) > 1 and not portable:
        raise NotImplementedError("superdex MJCF supports only one free joint on the root")
    if portable:
        for joint in free:
            body = int(m.jnt_bodyid[joint])
            if int(m.body_parentid[body]) != 0 or int(m.body_jntnum[body]) != 1:
                raise NotImplementedError("superdex portable free joints must be entity roots")
    if np.any(m.jnt_stiffness != 0) or np.any(m.dof_armature[:6] != 0) and len(free):
        raise NotImplementedError("superdex MJCF joint springs/free-root armature are unsupported")
    for j, jt in enumerate(m.jnt_type):
        if jt != int(mj.mjtJoint.mjJNT_FREE) and m.qpos0[m.jnt_qposadr[j]] != 0:
            raise NotImplementedError("superdex MJCF nonzero joint ref is unsupported")
    if np.any(m.body_gravcomp != 0):
        raise NotImplementedError("superdex MJCF gravity compensation is unsupported")
    if m.opt.disableflags or m.opt.disableactuator or m.opt.density or m.opt.viscosity:
        raise NotImplementedError("superdex MJCF disabled dynamics/fluid options are unsupported")
    if np.any(m.jnt_actfrclimited):
        raise NotImplementedError(
            "superdex MJCF joint-level summed actuator force limits unsupported"
        )
    if np.any(m.geom_gap):
        raise NotImplementedError("superdex MJCF nonzero contact gap is unsupported")


def _portable_plan(
    p: Any,
    scene: SceneCfg,
    efforts: Sequence[float] | None,
    allow_contact_approximation: bool,
    sim_dt: float,
    *,
    composed: Any | None = None,
    model: Any | None = None,
    layout_override: Any | None = None,
) -> ModelPlan:
    """Bind the common portable layout to independent native actor slots."""
    import mujoco

    from unisim.mjcf_compiler import compose_scene

    if not np.isfinite(sim_dt) or sim_dt <= 0:
        raise ValueError("sim_dt must be finite and positive")

    owns_composed = composed is None
    if composed is None:
        composed = compose_scene(scene, 1, sim_dt)
    mj = mujoco
    m = composed.model if model is None else model
    layout = composed.layout if layout_override is None else layout_override
    _audit_model(mj, m, portable=True)
    if any(joint.kind == "ball" for entity in layout.entities for joint in entity.joints):
        raise NotImplementedError("superdex portable MJCF ball joints are unsupported")

    def names(obj: Any, count: int, prefix: str) -> tuple[str, ...]:
        return tuple(mj.mj_id2name(m, obj, i) or f"{prefix}{i}" for i in range(count))

    body_names = names(mj.mjtObj.mjOBJ_BODY, m.nbody, "body")
    joint_names = names(mj.mjtObj.mjOBJ_JOINT, m.njnt, "joint")
    geom_names = names(mj.mjtObj.mjOBJ_GEOM, m.ngeom, "geom")
    active = np.flatnonzero(m.jnt_type != int(mj.mjtJoint.mjJNT_FREE))
    actor_builders: list[tuple[Any, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    actor_plans: list[NativeActorPlan] = []
    has_physical_articulation = False
    static_actor_geoms: dict[int, int] = {}
    global_geom_links: dict[int, int] = {}
    body_local_links = np.full(m.nbody, -1, dtype=np.int32)
    body_actor_indices = np.full(m.nbody, -1, dtype=np.int32)
    kinematic_actor_indices = {
        slot for slot, entity in enumerate(layout.entities) if entity.root_mode == "kinematic"
    }
    flattened_body_links = np.full(m.nbody, -1, dtype=np.int32)
    actor_link_offsets: list[int] = []
    link_offset = 0
    native_dof_offset = 0
    entity_specs = {entity.name: entity for entity in scene.entity_assets}

    for slot, entity in enumerate(layout.entities):
        spec = entity_specs[entity.name]
        if entity.root_mode == "kinematic" and spec.mirror_of is not None:
            if len(entity.body_ids) != 1 or entity.joints:
                raise NotImplementedError(
                    "superdex portable mirrors are limited to one rigid source body"
                )
            body = int(entity.body_ids[0])
            geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body]
            if not geoms or float(m.body_mass[body]) <= 0:
                raise NotImplementedError(
                    "superdex portable mirrors require one inertial rigid source body"
                )
            parts = len(geoms)
            links: list[Any] = []
            joints: list[Any] = []
            inertial_rot = rotation_matrix(m.body_iquat[body])
            inertia = inertial_rot @ np.diag(m.body_inertia[body]) @ inertial_rot.T
            for index, geom in enumerate(geoms):
                kind = {
                    int(mj.mjtGeom.mjGEOM_BOX): "box",
                    int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
                    int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
                    int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
                    int(mj.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
                }.get(int(m.geom_type[geom]))
                if kind is None:
                    raise NotImplementedError(
                        f"superdex unsupported mirror geom {geom_names[geom]!r}"
                    )
                shape = primitive_shape(
                    p, kind, m.geom_size[geom], m.geom_pos[geom], m.geom_quat[geom]
                )
                links.append(
                    p.ArticulatedLinkParams(
                        name=body_names[body].replace("/", "__")
                        if index == 0
                        else f"__mirror_geom_{geom}",
                        parent_link=-1 if index == 0 else len(links) - 1,
                        parent_joint_from_link=p.TransformRT(),
                        shape=shape,
                        collider_type=p.ColliderType.NONE,
                        has_gravity=False,
                        mass=float(m.body_mass[body]) / parts,
                        center_of_mass=np.asarray(m.body_ipos[body]),
                        moment_of_inertia=inertia[np.triu_indices(3)] / parts,
                    )
                )
                joints.append(
                    p.ArticulatedJointParams(
                        name=f"{entity.name}__mirror_root"
                        if index == 0
                        else f"__mirror_geom_joint_{geom}",
                        type=p.ArticulatedJointType.FREE
                        if index == 0
                        else p.ArticulatedJointType.HARD,
                        parent_link_from_joint=p.TransformRT(),
                    )
                )
            native_count = 6
            native_qpos_indices = np.arange(
                native_dof_offset, native_dof_offset + native_count, dtype=np.int32
            )
            body_actor_indices[body] = slot
            flattened_body_links[body] = link_offset
            body_local_links[body] = 0
            actor_link_offsets.append(link_offset)
            link_offset += len(links)

            def spawn_mirror(
                native_scene: Any,
                *,
                entity_name=entity.name,
                params_links=tuple(links),
                params_joints=tuple(joints),
            ) -> Any:
                params = p.ArticulatedActorParams(
                    name=entity_name,
                    joints=params_joints,
                    links=params_links,
                    world_from_root=p.TransformRT(),
                )
                return native_scene.create_articulated_actor(params)

            actor_builders.append(
                (
                    spawn_mirror,
                    entity.name,
                    np.asarray((), dtype=np.int32),
                    native_qpos_indices,
                    np.asarray(entity.body_ids, dtype=np.int32),
                    np.asarray([0], dtype=np.int32),
                )
            )
            actor_plans.append(
                NativeActorPlan(
                    entity_name=entity.name,
                    root_body_id=body,
                    floating=False,
                    kinematic_mirror=True,
                    physical_kinematic=False,
                    qpos_indices=np.asarray((), dtype=np.int32),
                    qvel_indices=np.asarray((), dtype=np.int32),
                    native_qpos_indices=native_qpos_indices,
                    native_qvel_indices=native_qpos_indices.copy(),
                    native_order_qpos_indices=np.asarray((), dtype=np.int32),
                    native_order_qvel_indices=np.asarray((), dtype=np.int32),
                    body_ids=np.asarray(entity.body_ids, dtype=np.int32),
                    local_body_link_indices=np.asarray([0], dtype=np.int32),
                    actuator_indices=np.asarray((), dtype=np.int32),
                    native_actuator_qpos_indices=np.asarray((), dtype=np.int32),
                    native_actuator_qvel_indices=np.asarray((), dtype=np.int32),
                    spawn_actor=spawn_mirror,
                )
            )
            native_dof_offset += native_count
            continue

        if entity.root_mode == "fixed" and entity.kind == "rigid":
            if len(entity.body_ids) != 1 or entity.joints or len(entity.geoms) != 1:
                raise NotImplementedError(
                    "superdex static rigid entities are limited to one named collision geom"
                )
            body = int(entity.body_ids[0])
            owned_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body]
            geom = owned_geoms[0] if owned_geoms else -1
            contact = p.ContactParams()
            if not spec.collision_enabled:
                shape = p.ShapeHandle()
                collider = p.ColliderType.NONE
            else:
                if geom < 0 or m.geom_bodyid[geom] != body:
                    raise RuntimeError(f"superdex entity {entity.name!r} lost its geom binding")
                collisions = [
                    g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
                ]
                friction = _friction_factors(m, collisions)[geom]
                kind = {
                    int(mj.mjtGeom.mjGEOM_BOX): "box",
                    int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
                    int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
                    int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
                    int(mj.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
                }.get(int(m.geom_type[geom]))
                if kind is None:
                    raise NotImplementedError(
                        f"superdex unsupported rigid geom {geom_names[geom]!r}"
                    )
                shape = primitive_shape(
                    p, kind, m.geom_size[geom], m.geom_pos[geom], m.geom_quat[geom]
                )
                collider = p.ColliderType.AUTO
                contact = _contact(p, m, geom, friction)
            root = int(entity.body_ids[0])
            transform = _transform(p, m.body_pos[root], m.body_quat[root])

            def spawn_rigid(
                native_scene: Any,
                *,
                entity_name=entity.name,
                shape=shape,
                collider=collider,
                transform=transform,
                contact=contact if spec.collision_enabled else None,
            ) -> Any:
                params = p.RigidActorParams(
                    name=entity_name,
                    shape=shape,
                    collider_type=collider,
                    is_static=True,
                    world_from_local=transform,
                )
                if contact is not None:
                    params.contact = contact
                return native_scene.create_rigid_actor(params)

            actor_builders.append(
                (
                    spawn_rigid,
                    entity.name,
                    np.asarray((), dtype=np.int32),
                    np.asarray((), dtype=np.int32),
                    np.asarray(entity.body_ids, dtype=np.int32),
                    np.asarray([-1], dtype=np.int32),
                )
            )
            if geom >= 0:
                static_actor_geoms[slot] = geom
            body_actor_indices[body] = slot
            actor_link_offsets.append(link_offset)
            actor_plans.append(
                NativeActorPlan(
                    entity_name=entity.name,
                    root_body_id=body,
                    floating=False,
                    kinematic_mirror=False,
                    physical_kinematic=False,
                    qpos_indices=np.asarray((), dtype=np.int32),
                    qvel_indices=np.asarray((), dtype=np.int32),
                    native_qpos_indices=np.asarray((), dtype=np.int32),
                    native_qvel_indices=np.asarray((), dtype=np.int32),
                    native_order_qpos_indices=np.asarray((), dtype=np.int32),
                    native_order_qvel_indices=np.asarray((), dtype=np.int32),
                    body_ids=np.asarray(entity.body_ids, dtype=np.int32),
                    local_body_link_indices=np.asarray((), dtype=np.int32),
                    actuator_indices=np.asarray((), dtype=np.int32),
                    native_actuator_qpos_indices=np.asarray((), dtype=np.int32),
                    native_actuator_qvel_indices=np.asarray((), dtype=np.int32),
                    spawn_actor=spawn_rigid,
                )
            )
            continue

        floating = entity.root_mode == "floating"
        physical_kinematic = entity.root_mode == "kinematic" and spec.mirror_of is None
        links, joints, geom_links, local_body_links, _ = _build_links(
            p,
            mj,
            m,
            body_names,
            geom_names,
            floating or physical_kinematic,
            allow_contact_approximation,
            entity.body_ids,
        )
        if physical_kinematic:
            # The public entity has no source joints after kinematic compilation.
            # Retain its collision/inertial body tree on a hidden native free-root
            # carrier so boundary conditions can hold its authored world pose.
            joints[0] = p.ArticulatedJointParams(
                name=f"{entity.name}__kinematic_carrier",
                type=p.ArticulatedJointType.FREE,
                parent_link_from_joint=p.TransformRT(),
            )
            for link in links:
                link.has_gravity = False
        public_joint_qpos = tuple(i for joint in entity.joints for i in joint.qpos_indices)
        public_joint_qvel = tuple(i for joint in entity.joints for i in joint.qvel_indices)
        if floating:
            public_qpos_indices = np.asarray(
                entity.root_qpos_indices + public_joint_qpos,
                dtype=np.int32,
            )
            public_qvel_indices = np.asarray(
                entity.root_qvel_indices[:3]
                + entity.root_qvel_indices[3:6]
                + public_joint_qvel,
                dtype=np.int32,
            )
        else:
            public_qpos_indices = np.asarray(public_joint_qpos, dtype=np.int32)
            public_qvel_indices = np.asarray(public_joint_qvel, dtype=np.int32)
        qpos_indices = np.asarray(entity.qpos_indices, dtype=np.int32)
        qvel_indices = np.asarray(entity.qvel_indices, dtype=np.int32)
        native_count = 6 if physical_kinematic else len(public_qvel_indices)
        if not native_count or not len(links):
            raise ValueError(f"superdex entity {entity.name!r} has invalid native DoF layout")
        has_physical_articulation = True
        native_qpos_indices = np.arange(
            native_dof_offset, native_dof_offset + native_count, dtype=np.int32
        )
        native_qvel_indices = native_qpos_indices.copy()
        body_indices = np.asarray(entity.body_ids, dtype=np.int32)
        mapped_links = local_body_links[body_indices]
        actor_link_offsets.append(link_offset)
        for geom, local_link in geom_links.items():
            global_geom_links[int(geom)] = int(local_link) + link_offset
        flattened_body_links[body_indices] = mapped_links + link_offset
        body_local_links[body_indices] = mapped_links
        body_actor_indices[body_indices] = slot
        link_offset += len(links)

        def spawn_articulated(
            native_scene: Any,
            *,
            physics=p,
            model=m,
            entity_layout=entity,
            params_links=links,
            params_joints=joints,
        ) -> Any:
            params = p.ArticulatedActorParams(
                name=entity_layout.name, joints=params_joints, links=params_links
            )
            if entity_layout.root_mode in {"floating", "kinematic"}:
                # Native free-joint coordinates are absolute in this adapter.
                # Keep the actor-root frame identity so reset does not compose
                # the authored pose with the same transform again. For physical
                # kinematic roots, the authored pose is carried only by boundary
                # conditions on the hidden native root.
                params.world_from_root = physics.TransformRT()
            else:
                root = int(entity_layout.body_ids[0])
                params.world_from_root = _transform(
                    physics, model.body_pos[root], model.body_quat[root]
                )
            return native_scene.create_articulated_actor(params)

        actor_builders.append(
            (
                spawn_articulated,
                entity.name,
                qpos_indices,
                native_qpos_indices if physical_kinematic else qvel_indices,
                body_indices,
                mapped_links,
            )
        )
        qpos_sources = public_joint_qpos if floating else public_qpos_indices
        qpos_targets = native_qpos_indices[6:] if floating else native_qpos_indices
        public_to_native_qpos = (
            {}
            if physical_kinematic
            else {
                int(public): int(native)
                for public, native in zip(qpos_sources, qpos_targets, strict=True)
            }
        )
        public_to_native_qvel = (
            {}
            if physical_kinematic
            else {
                int(public): int(native)
                for public, native in zip(public_qvel_indices, native_qvel_indices, strict=True)
            }
        )
        joints_by_name = {joint.name: joint for joint in entity.joints}
        actuator_qpos_sources = np.asarray(
            [
                int(joints_by_name[name].qpos_indices[0])
                for name in entity.actuator_joint_names
            ],
            dtype=np.int32,
        )
        actuator_qvel_sources = np.asarray(
            [
                int(joints_by_name[name].qvel_indices[0])
                for name in entity.actuator_joint_names
            ],
            dtype=np.int32,
        )
        native_actuator_qpos = np.asarray(
            [public_to_native_qpos[int(i)] for i in actuator_qpos_sources], dtype=np.int32
        )
        native_actuator_qvel = np.asarray(
            [public_to_native_qvel[int(i)] for i in actuator_qvel_sources], dtype=np.int32
        )
        actor_plans.append(
            NativeActorPlan(
                entity_name=entity.name,
                root_body_id=int(entity.body_ids[0]),
                floating=floating,
                kinematic_mirror=False,
                physical_kinematic=physical_kinematic,
                qpos_indices=qpos_indices,
                qvel_indices=public_qvel_indices,
                native_qpos_indices=native_qpos_indices,
                native_qvel_indices=native_qvel_indices,
                native_order_qpos_indices=public_qpos_indices.copy(),
                native_order_qvel_indices=public_qvel_indices.copy(),
                body_ids=body_indices,
                local_body_link_indices=mapped_links,
                actuator_indices=np.asarray(entity.actuator_indices, dtype=np.int32),
                native_actuator_qpos_indices=native_actuator_qpos,
                native_actuator_qvel_indices=native_actuator_qvel,
                spawn_actor=spawn_articulated,
            )
        )
        native_dof_offset += native_count

    if not has_physical_articulation:
        raise ValueError(
            "superdex portable scene requires at least one physical articulated actor"
        )
    sensors = _sensors(
        mj,
        m,
        geom_names,
        global_geom_links,
        body_actor_indices=body_actor_indices,
        flattened_body_links=flattened_body_links,
        body_local_links=body_local_links,
    )
    if any(
        sensor.source_actor_index in kinematic_actor_indices
        or sensor.other_actor_index in kinematic_actor_indices
        for sensor in sensors
        if sensor.kind.startswith("contact")
    ):
        raise NotImplementedError(
            "superdex portable contact sensors cannot target kinematic entities"
        )
    actuator = _actuators(mj, m, joint_names, efforts)
    default_ctrl = np.zeros(int(m.nu), dtype=float)
    if scene.default_keyframe_name is not None:
        key_id = int(
            mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, scene.default_keyframe_name)
        )
        if key_id < 0:
            raise ValueError(
                f"superdex default keyframe {scene.default_keyframe_name!r} is missing"
            )
        default_ctrl = np.asarray(m.key_ctrl[key_id], dtype=float)
    if default_ctrl.shape != (int(m.nu),) or not np.isfinite(default_ctrl).all():
        raise ValueError("superdex default controls have an invalid shape or values")
    default_ctrl = np.clip(
        default_ctrl,
        actuator["actuator_ctrl_ranges"][:, 0],
        actuator["actuator_ctrl_ranges"][:, 1],
    )
    joint_ranges = np.array(m.jnt_range[active])
    joint_ranges[~np.asarray(m.jnt_limited[active], dtype=bool)] = [-np.inf, np.inf]

    def spawn_scene(native_scene: Any) -> Any:
        from unisim.backend.superdex.plans import NativeSceneActors

        actors = []
        links: list[tuple[Any, ...]] = []
        dof_counts: list[int] = []
        cleanups = []
        native_geoms: dict[int, Any] = {}
        carriers: list[Any] = []
        for slot, (spawn, _, _, qvel_indices, _, _) in enumerate(actor_builders):
            actor = spawn(native_scene)
            actors.append(actor)
            if qvel_indices.size:
                actor_links = tuple(
                    native_scene.get_actor(handle) for handle in actor.get_nested_link_actors()
                )
                links.append(actor_links)
                dof_counts.append(int(actor.get_num_dofs()))
            else:
                actor_links = ()
                links.append(())
                dof_counts.append(0)
            cleanups.append(_noop)
            if not qvel_indices.size:
                if actor_links:
                    raise RuntimeError("static SuperDex actor unexpectedly owns links")
                if slot in static_actor_geoms:
                    native_geoms[static_actor_geoms[slot]] = actor
                continue
            link_offset = actor_link_offsets[slot]
            for g, flattened_link in global_geom_links.items():
                if body_actor_indices[int(m.geom_bodyid[g])] == slot:
                    native_geoms[g] = native_scene.get_actor(
                        actor.get_nested_link_actors()[flattened_link - link_offset]
                    )
            for body in actor_builders[slot][4].tolist():
                if flattened_body_links[body] >= 0 and all(
                    g not in native_geoms or int(m.geom_bodyid[g]) != body
                    for g in global_geom_links
                ):
                    carriers.append(
                        native_scene.get_actor(
                            actor.get_nested_link_actors()[
                                flattened_body_links[body] - link_offset
                            ]
                        )
                    )
        for carrier in carriers:
            for other in native_geoms.values():
                native_scene.enable_actor_contact_symmetric(
                    carrier.get_handle(), other.get_handle(), False, p.IncludeNestedActors.NO
                )
        items = list(native_geoms.items())
        for index, (g1, first) in enumerate(items):
            for g2, second in items[index + 1 :]:
                native_scene.enable_actor_contact_symmetric(
                    first.get_handle(),
                    second.get_handle(),
                    _geom_pair_allowed(m, g1, g2),
                    p.IncludeNestedActors.NO,
                )
        return NativeSceneActors(
            tuple(actors),
            tuple(links),
            tuple(dof_counts),
            tuple(cleanups),
            body_actor_indices,
            flattened_body_links,
        )

    def spawn_first(native_scene: Any) -> tuple[Any, Callable[[], None]]:
        scene_actors = spawn_scene(native_scene)
        actor = next(
            actor for actor, count in zip(scene_actors.actors, scene_actors.actor_dof_counts)
            if count
        )
        return actor, _noop

    return ModelPlan(
        source_file=composed.model_file,
        nq=int(m.nq),
        nv=int(m.nv),
        root_body_id=int(layout.entities[0].body_ids[0]),
        floating=any(entity.root_mode == "floating" for entity in layout.entities),
        body_names=body_names,
        body_parent_ids=np.array(m.body_parentid),
        body_link_indices=flattened_body_links,
        body_mass=np.array(m.body_mass),
        body_ipos=np.array(m.body_ipos),
        body_pos=np.array(m.body_pos),
        body_quat=np.array(m.body_quat),
        joint_names=tuple(joint_names[i] for i in active),
        joint_qpos_indices=np.array(m.jnt_qposadr[active]),
        joint_qvel_indices=np.array(m.jnt_dofadr[active]),
        joint_ranges=joint_ranges,
        actuator_names=names(mj.mjtObj.mjOBJ_ACTUATOR, m.nu, "actuator"),
        default_qpos=np.array(m.qpos0),
        keyframes={
            mj.mj_id2name(m, mj.mjtObj.mjOBJ_KEY, i) or f"key{i}": np.array(m.key_qpos[i])
            for i in range(m.nkey)
        },
        default_ctrl=default_ctrl,
        gravity=np.array(m.opt.gravity),
        sensors=sensors,
        spawn_actor=spawn_first,
        spawn_scene=spawn_scene,
        cleanup=composed.close if owns_composed else _noop,
        dof_armature=np.array(m.dof_armature),
        layout=layout,
        actor_plans=tuple(actor_plans),
        actuator_slot_indices=np.concatenate(
            [plan.actuator_indices for plan in actor_plans]
        ),
        **actuator,
    )


_FIXED_VARIANT_MODEL_FIELDS = (
    "nq",
    "nv",
    "root_body_id",
    "floating",
    "body_names",
    "joint_names",
    "actuator_names",
    "actuator_joint_names",
    "sensors",
)

_FIXED_VARIANT_ARRAY_FIELDS = (
    "body_parent_ids",
    "body_link_indices",
    "body_pos",
    "body_quat",
    "joint_qpos_indices",
    "joint_qvel_indices",
    "joint_ranges",
    "actuator_qpos_indices",
    "actuator_qvel_indices",
    "actuator_ctrl_ranges",
    "actuator_gear",
    "actuator_kp",
    "actuator_kd",
    "default_ctrl",
    "gravity",
    "actuator_force_ranges",
    "dof_armature",
)

_FIXED_VARIANT_ACTOR_FIELDS = (
    "entity_name",
    "root_body_id",
    "floating",
    "kinematic_mirror",
    "physical_kinematic",
    "qpos_indices",
    "qvel_indices",
    "native_qpos_indices",
    "native_qvel_indices",
    "native_order_qpos_indices",
    "native_order_qvel_indices",
    "body_ids",
    "local_body_link_indices",
    "actuator_indices",
    "native_actuator_qpos_indices",
    "native_actuator_qvel_indices",
)


def _audit_fixed_variant_plans(plans: tuple[ModelPlan, ...]) -> None:
    """Require one public contract while allowing native physical identity to vary."""
    primary, *others = plans
    for plan in others:
        for field in _FIXED_VARIANT_MODEL_FIELDS:
            if getattr(primary, field) != getattr(plan, field):
                raise ValueError(f"superdex fixed variants change public field {field!r}")
        for field in _FIXED_VARIANT_ARRAY_FIELDS:
            if not np.array_equal(getattr(primary, field), getattr(plan, field)):
                raise ValueError(f"superdex fixed variants change public field {field!r}")
        if tuple(primary.keyframes) != tuple(plan.keyframes) or any(
            not np.array_equal(primary.keyframes[name], plan.keyframes[name])
            for name in primary.keyframes
        ):
            raise ValueError("superdex fixed variants change public keyframes")
        if len(primary.actor_plans) != len(plan.actor_plans):
            raise ValueError("superdex fixed variants change actor slot layout")
        for left, right in zip(primary.actor_plans, plan.actor_plans, strict=True):
            for field in _FIXED_VARIANT_ACTOR_FIELDS:
                if not np.array_equal(getattr(left, field), getattr(right, field)):
                    raise ValueError(f"superdex fixed variants change actor field {field!r}")


def _fixed_variant_plan(
    p: Any,
    scene: SceneCfg,
    efforts: Sequence[float] | None,
    allow_contact_approximation: bool,
    sim_dt: float,
) -> ModelPlan:
    """Materialize every assignment-selected realization with one public contract."""
    binding = scene.entity_variant
    if binding is None:
        raise ValueError("superdex fixed variant materialization requires entity_variant")
    if binding.plan.layout is not FixedVariantLayout.SAME_LAYOUT:
        raise NotImplementedError(
            "superdex portable fixed variants require the same_layout contract"
        )
    import mujoco

    from unisim.mjcf_compiler import compose_scene

    composed = compose_scene(scene, binding.plan.assignment.shape[0], sim_dt)
    try:
        if composed.variant_plan is None:
            raise RuntimeError("superdex fixed-variant composition lost its assignment")
        plans: list[ModelPlan] = []
        for index, variant in enumerate(composed.variant_plan.variants):
            model = mujoco.MjModel.from_xml_path(variant.model_file)
            plans.append(
                _portable_plan(
                    p,
                    scene,
                    efforts,
                    allow_contact_approximation,
                    sim_dt,
                    composed=composed,
                    model=model,
                    layout_override=composed.variant_layouts[index],
                )
            )
        materialized = tuple(plans)
        _audit_fixed_variant_plans(materialized)
    except BaseException:
        composed.close()
        raise

    def cleanup_all() -> None:
        composed.close()

    return replace(
        materialized[0],
        cleanup=cleanup_all,
        fixed_variant_plans=materialized,
        fixed_variant_assignment=np.asarray(binding.plan.assignment, dtype=np.int32).copy(),
    )


def _mjcf_plan(
    p: Any,
    path: Path,
    scene: SceneCfg,
    efforts: Sequence[float] | None,
    allow_contact_approximation: bool,
) -> ModelPlan:
    mj, m = _load_mjcf(path, scene)
    _audit_model(mj, m)

    def names(obj: Any, count: int, prefix: str) -> tuple[str, ...]:
        return tuple(mj.mj_id2name(m, obj, i) or f"{prefix}{i}" for i in range(count))

    body_names = names(mj.mjtObj.mjOBJ_BODY, m.nbody, "body")
    joint_names = names(mj.mjtObj.mjOBJ_JOINT, m.njnt, "joint")
    actuator_names = names(mj.mjtObj.mjOBJ_ACTUATOR, m.nu, "actuator")
    geom_names = names(mj.mjtObj.mjOBJ_GEOM, m.ngeom, "geom")
    floating = bool(m.njnt and m.jnt_type[0] == int(mj.mjtJoint.mjJNT_FREE))
    links, joints, geom_links, body_links, plane_params = _build_links(
        p, mj, m, body_names, geom_names, floating, allow_contact_approximation
    )
    sensors = _sensors(mj, m, geom_names, geom_links)
    actuator = _actuators(mj, m, joint_names, efforts)
    active = np.flatnonzero(m.jnt_type != int(mj.mjtJoint.mjJNT_FREE))
    joint_ranges = np.array(m.jnt_range[active])
    joint_ranges[~np.asarray(m.jnt_limited[active], dtype=bool)] = [-np.inf, np.inf]

    def spawn(native_scene: Any) -> tuple[Any, Any]:
        params = p.ArticulatedActorParams(name="robot", joints=joints, links=links)
        actor = native_scene.create_articulated_actor(params)
        handles = list(actor.get_nested_link_actors())
        native_geoms = {g: native_scene.get_actor(handles[li]) for g, li in geom_links.items()}
        for g, param in plane_params:
            native_geoms[g] = native_scene.create_rigid_actor(param)
        # Invisible inertial carrier meshes must not sample contacts either.
        carriers = set(body_links[1:]) - set(geom_links.values())
        for index in carriers:
            for other in native_geoms.values():
                native_scene.enable_actor_contact_symmetric(
                    handles[index], other.get_handle(), False, p.IncludeNestedActors.NO
                )
        # Override native adjacency defaults with the authored MJCF bitmask pairs.
        for g1, a in native_geoms.items():
            for g2, b in native_geoms.items():
                if g2 <= g1:
                    continue
                native_scene.enable_actor_contact_symmetric(
                    a.get_handle(),
                    b.get_handle(),
                    _geom_pair_allowed(m, g1, g2),
                    p.IncludeNestedActors.NO,
                )
        return actor, _noop

    return ModelPlan(
        source_file=str(path),
        nq=int(m.nq),
        nv=int(m.nv),
        root_body_id=1,
        floating=floating,
        body_names=body_names,
        body_parent_ids=np.array(m.body_parentid),
        body_link_indices=body_links,
        body_mass=np.array(m.body_mass),
        body_ipos=np.array(m.body_ipos),
        body_pos=np.array(m.body_pos),
        body_quat=np.array(m.body_quat),
        joint_names=tuple(joint_names[i] for i in active),
        joint_qpos_indices=np.array(m.jnt_qposadr[active]),
        joint_qvel_indices=np.array(m.jnt_dofadr[active]),
        joint_ranges=joint_ranges,
        actuator_names=actuator_names,
        default_qpos=np.array(m.qpos0),
        keyframes={
            mj.mj_id2name(m, mj.mjtObj.mjOBJ_KEY, i) or f"key{i}": np.array(m.key_qpos[i])
            for i in range(m.nkey)
        },
        gravity=np.array(m.opt.gravity),
        sensors=sensors,
        spawn_actor=spawn,
        cleanup=_noop,
        dof_armature=np.array(m.dof_armature),
        **actuator,
    )


def _build_links(
    p: Any,
    mj: Any,
    m: Any,
    body_names: tuple[str, ...],
    geom_names: tuple[str, ...],
    floating: bool,
    allow_contact_approximation: bool,
    bodies: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    links: list[Any] = []
    joints: list[Any] = []
    geom_links: dict[int, int] = {}
    body_links = np.full(m.nbody, -1, dtype=int)
    planes: list[tuple[int, Any]] = []
    collisions = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    friction = _friction_factors(m, collisions)
    for g in collisions:
        if m.geom_bodyid[g] == 0:
            if m.geom_type[g] != int(mj.mjtGeom.mjGEOM_PLANE):
                raise NotImplementedError("superdex supports only static planes in worldbody")
            normal = rotation_matrix(m.geom_quat[g])[:, 2]
            shape = p.create_plane_shape(normal, float(normal @ m.geom_pos[g]))
            planes.append(
                (
                    g,
                    p.RigidActorParams(
                        name=geom_names[g],
                        shape=shape,
                        is_static=True,
                        contact=_contact(p, m, g, friction[g]),
                    ),
                )
            )
    mapping = {
        int(mj.mjtGeom.mjGEOM_BOX): "box",
        int(mj.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mj.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mj.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mj.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    }
    if int(m.npair):
        raise NotImplementedError("superdex explicit MJCF contact pairs are unsupported")
    if any(m.geom_condim[g] > 3 and np.any(m.geom_friction[g, 1:] > 0) for g in collisions):
        if not allow_contact_approximation:
            raise NotImplementedError(
                "superdex cannot preserve MJCF torsional/rolling friction; explicitly set "
                "allow_contact_approximation=True for the experimental sliding-only profile"
            )
        warnings.warn(
            "superdex uses sliding Coulomb contact; MJCF torsional/rolling friction "
            "and solver-specific condim/solref/solimp are not numerically equivalent",
            RuntimeWarning,
            stacklevel=3,
        )
    selected_bodies = tuple(range(1, m.nbody)) if bodies is None else tuple(bodies)
    if not selected_bodies:
        raise ValueError("superdex actor must contain at least one body")
    root_body = selected_bodies[0]
    for body in selected_bodies:
        geoms = [g for g in collisions if m.geom_bodyid[g] == body]
        body_links[body] = len(links)
        parent = int(body_links[m.body_parentid[body]])
        j = int(m.body_jntadr[body]) if m.body_jntnum[body] else -1
        jt = int(m.jnt_type[j]) if j >= 0 else -1
        native_type = {
            int(mj.mjtJoint.mjJNT_FREE): p.ArticulatedJointType.FREE,
            int(mj.mjtJoint.mjJNT_HINGE): p.ArticulatedJointType.REVOLUTE,
            int(mj.mjtJoint.mjJNT_SLIDE): p.ArticulatedJointType.PRISMATIC,
        }.get(jt, p.ArticulatedJointType.HARD)
        pos = np.asarray(m.jnt_pos[j]) if j >= 0 else np.zeros(3)
        root_free = body == root_body and floating
        joint_transform = (
            p.TransformRT()
            if root_free
            else _transform(
                p, m.body_pos[body] + rotation_matrix(m.body_quat[body]) @ pos, m.body_quat[body]
            )
        )
        args: dict[str, Any] = {}
        if j >= 0 and not root_free:
            dof = int(m.jnt_dofadr[j])
            args.update(
                axis=np.asarray(m.jnt_axis[j]),
                inertia=float(m.dof_armature[dof]),
                friction=p.ArticulatedJointFrictionParams(
                    viscous=float(m.dof_damping[dof]), coulomb=float(m.dof_frictionloss[dof])
                ),
            )
            if m.jnt_limited[j]:
                args.update(
                    min_limit=m.jnt_axis[j] * m.jnt_range[j, 0],
                    max_limit=m.jnt_axis[j] * m.jnt_range[j, 1],
                )
        joint_name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) if j >= 0 else None
        main_joint = p.ArticulatedJointParams(
            name=(
                joint_name.replace("/", "__")
                if joint_name is not None
                else f"__joint_{body}"
            ),
            type=native_type,
            parent_link_from_joint=joint_transform,
            **args,
        )
        inertial_rot = rotation_matrix(m.body_iquat[body])
        inertia = inertial_rot @ np.diag(m.body_inertia[body]) @ inertial_rot.T
        parts = max(1, len(geoms))
        for k, g in enumerate(geoms or [-1]):
            if g >= 0:
                kind = mapping.get(int(m.geom_type[g]))
                if kind is None:
                    raise NotImplementedError(
                        f"superdex unsupported collision geom {geom_names[g]!r}"
                    )
                shape = primitive_shape(p, kind, m.geom_size[g], m.geom_pos[g], m.geom_quat[g])
            elif float(m.body_mass[body]) > 0:
                shape = primitive_shape(p, "box", [0.001] * 3, [0, 0, 0], [1, 0, 0, 0])
            else:
                # A massless fixed frame is a native dummy link, not a tiny
                # density-derived solid that changes the articulation mass.
                shape = p.ShapeHandle()
            props: dict[str, Any] = {}
            if float(m.body_mass[body]) > 0:
                props.update(
                    mass=float(m.body_mass[body]) / parts,
                    center_of_mass=np.asarray(m.body_ipos[body]),
                    moment_of_inertia=inertia[np.triu_indices(3)] / parts,
                )
            links.append(
                p.ArticulatedLinkParams(
                    name=body_names[body].replace("/", "__") if k == 0 else f"__geom_{g}",
                    parent_link=parent if k == 0 else int(body_links[body]),
                    parent_joint_from_link=_transform(
                        p, -pos if k == 0 and not root_free else [0, 0, 0]
                    ),
                    shape=shape,
                    contact=_contact(p, m, g, friction[g]) if g >= 0 else p.ContactParams(),
                    collider_type=p.ColliderType.AUTO if g >= 0 else p.ColliderType.NONE,
                    **props,
                )
            )
            joints.append(
                main_joint
                if k == 0
                else p.ArticulatedJointParams(
                    name=f"__geom_joint_{g}", type=p.ArticulatedJointType.HARD
                )
            )
            if g >= 0:
                geom_links[g] = len(links) - 1
    return links, joints, geom_links, body_links, planes


def _contact(p: Any, m: Any, g: int, friction: float) -> Any:
    params = p.ContactParams()
    params.coulomb_friction_coefficient = friction
    params.penalty_threshold_default = float(m.geom_margin[g])
    return params


def _pair_friction(m: Any, g1: int, g2: int) -> float:
    """Preserve MJCF priority/max sliding friction, bypassing native geometric mean."""
    if m.geom_priority[g1] != m.geom_priority[g2]:
        chosen = g1 if m.geom_priority[g1] > m.geom_priority[g2] else g2
        return float(m.geom_friction[chosen, 0]) if m.geom_condim[chosen] >= 3 else 0.0
    if max(m.geom_condim[g1], m.geom_condim[g2]) < 3:
        return 0.0
    return float(max(m.geom_friction[g1, 0], m.geom_friction[g2, 0]))


def _geom_pair_allowed(m: Any, g1: int, g2: int) -> bool:
    """Apply MuJoCo's default collision filters to welded body groups."""
    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
    if w1 == w2:
        return False
    if w1 and w2:
        parent1 = int(m.body_weldid[m.body_parentid[w1]])
        parent2 = int(m.body_weldid[m.body_parentid[w2]])
        if parent1 == w2 or parent2 == w1:
            return False
    signature = (min(b1, b2) << 16) + max(b1, b2)
    if signature in m.exclude_signature:
        return False
    return bool(
        (m.geom_contype[g1] & m.geom_conaffinity[g2])
        or (m.geom_contype[g2] & m.geom_conaffinity[g1])
    )


def _friction_factors(m: Any, geoms: list[int]) -> dict[int, float]:
    """Factor MJCF pair friction into native geometric-mean actor coefficients.

    Published SDK 1.0.0 lacks the newer pair-override API. A floor-only star
    graph always admits an exact factorization. Reject incompatible pair rules.
    """
    pairs = []
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1 :]:
            if _geom_pair_allowed(m, g1, g2):
                pairs.append((g1, g2, _pair_friction(m, g1, g2)))
    positive = [(a, b, mu) for a, b, mu in pairs if mu > 0]
    positive_vertices = {g for a, b, _ in positive for g in (a, b)}
    indices = {g: i for i, g in enumerate(geoms)}
    result = {g: 1.0 for g in geoms}
    if positive:
        matrix = np.zeros((len(positive), len(geoms)))
        target = np.empty(len(positive))
        for row, (a, b, mu) in enumerate(positive):
            matrix[row, indices[a]] = matrix[row, indices[b]] = 1
            target[row] = 2 * np.log(mu)
        solution = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if not np.allclose(matrix @ solution, target, atol=1e-10, rtol=1e-10):
            raise NotImplementedError("superdex cannot factor the authored pair friction rules")
        result.update({g: float(np.exp(solution[i])) for g, i in indices.items()})
    for a, b, mu in pairs:
        if mu == 0:
            if a in positive_vertices and b in positive_vertices:
                raise NotImplementedError(
                    "superdex cannot preserve mixed zero/positive pair friction"
                )
            result[a if a not in positive_vertices else b] = 0.0
    return result


def _actuators(
    mj: Any, m: Any, joint_names: tuple[str, ...], efforts: Sequence[float] | None
) -> dict[str, Any]:
    targets, kp, kd, gears = [], [], [], []
    for a in range(m.nu):
        if (
            m.actuator_trntype[a] != int(mj.mjtTrn.mjTRN_JOINT)
            or m.actuator_dyntype[a] != int(mj.mjtDyn.mjDYN_NONE)
            or m.actuator_gaintype[a] != int(mj.mjtGain.mjGAIN_FIXED)
        ):
            raise NotImplementedError("superdex supports stateless joint motor/position actuators")
        j = int(m.actuator_trnid[a, 0])
        if m.jnt_type[j] == int(mj.mjtJoint.mjJNT_FREE):
            raise NotImplementedError("superdex free-joint actuators are unsupported")
        gear = float(m.actuator_gear[a, 0])
        if gear == 0 or np.any(m.actuator_gear[a, 1:] != 0):
            raise NotImplementedError("superdex requires scalar nonzero joint gear")
        gain = float(m.actuator_gainprm[a, 0])
        bias = np.asarray(m.actuator_biasprm[a])
        if m.actuator_biastype[a] == int(mj.mjtBias.mjBIAS_NONE):
            if gain != 1:
                raise NotImplementedError("superdex motor fixed gain must be one")
            kp.append(0.0)
            kd.append(0.0)
        elif (
            m.actuator_biastype[a] == int(mj.mjtBias.mjBIAS_AFFINE)
            and bias[0] == 0
            and bias[1] == -gain
            and bias[2] <= 0
            and gain > 0
        ):
            kp.append(gain)
            kd.append(float(-bias[2]))
        else:
            raise NotImplementedError("superdex supports only motor or linear position-servo bias")
        targets.append(j)
        gears.append(gear)
    target_indices = np.array(targets, dtype=int)
    ctrl = np.array(m.actuator_ctrlrange)
    ctrl[~np.asarray(m.actuator_ctrllimited, dtype=bool)] = [-np.inf, np.inf]
    force = np.array(m.actuator_forcerange)
    force[~np.asarray(m.actuator_forcelimited, dtype=bool)] = [-np.inf, np.inf]
    if efforts is not None:
        force = _effort_ranges(efforts, m.nu)
    return dict(
        actuator_joint_names=tuple(joint_names[j] for j in target_indices),
        actuator_qpos_indices=np.array(m.jnt_qposadr[target_indices]),
        actuator_qvel_indices=np.array(m.jnt_dofadr[target_indices]),
        actuator_ctrl_ranges=ctrl,
        actuator_force_ranges=force,
        actuator_gear=np.asarray(gears),
        actuator_kp=np.asarray(kp),
        actuator_kd=np.asarray(kd),
    )


def _sensors(
    mj: Any,
    m: Any,
    geom_names: tuple[str, ...],
    geom_links: dict[int, int],
    *,
    body_actor_indices: np.ndarray | None = None,
    flattened_body_links: np.ndarray | None = None,
    body_local_links: np.ndarray | None = None,
) -> tuple[SensorPlan, ...]:
    supported = {
        "GYRO": "gyro",
        "ACCELEROMETER": "accelerometer",
        "VELOCIMETER": "velocimeter",
        "FRAMEPOS": "framepos",
        "FRAMEQUAT": "framequat",
        "FRAMEZAXIS": "framezaxis",
        "FRAMELINVEL": "framelinvel",
        "FRAMEANGVEL": "frameangvel",
        "JOINTPOS": "jointpos",
        "JOINTVEL": "jointvel",
    }
    kinds = {int(getattr(mj.mjtSensor, f"mjSENS_{key}")): val for key, val in supported.items()}
    plans = []
    single_dofs = {
        int(j): i for i, j in enumerate(np.flatnonzero(m.jnt_type != int(mj.mjtJoint.mjJNT_FREE)))
    }
    for i in range(m.nsensor):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_SENSOR, i)
        if not name:
            raise NotImplementedError("superdex requires named sensors")
        if m.sensor_cutoff[i] != 0:
            raise NotImplementedError(f"superdex sensor {name!r} cutoff is unsupported")
        obj = int(m.sensor_objid[i])
        if m.sensor_type[i] == int(mj.mjtSensor.mjSENS_CONTACT):
            if (
                m.sensor_objtype[i] != int(mj.mjtObj.mjOBJ_GEOM)
                or m.sensor_reftype[i] != int(mj.mjtObj.mjOBJ_GEOM)
                or m.sensor_intprm[i, 0] != 1
                or m.sensor_intprm[i, 2] != 1
            ):
                raise NotImplementedError("superdex contact sensors require geom-pair found num=1")
            reference = int(m.sensor_refid[i])
            if int(m.geom_bodyid[obj]) == 0 and int(m.geom_bodyid[reference]) == 0:
                raise NotImplementedError("superdex contact sensors require a native actor")
            if int(m.geom_bodyid[obj]) != 0 or int(m.geom_bodyid[reference]) == 0:
                source, other = obj, reference
            else:
                source, other = reference, obj
            if source not in geom_links:
                raise NotImplementedError(
                    "superdex contact sensors require plane/link or entity pairs"
                )
            source_body = int(m.geom_bodyid[source])
            other_body = int(m.geom_bodyid[other])
            actor_indices = (
                np.full(m.nbody, -1, dtype=np.int32)
                if body_actor_indices is None
                else body_actor_indices
            )
            source_actor = int(actor_indices[source_body])
            other_actor = int(actor_indices[other_body])
            if flattened_body_links is not None and (source_actor < 0 or other_actor < 0):
                raise NotImplementedError(
                    "superdex portable contact sensors require entity-owned geoms"
                )
            if flattened_body_links is None:
                source_link = int(geom_links.get(source, -1))
                other_link = int(geom_links.get(other, -1))
            else:
                assert body_local_links is not None
                source_link = int(body_local_links[source_body])
                other_link = int(body_local_links[other_body])
            plans.append(
                SensorPlan(
                    name=name,
                    kind="contact_found",
                    dim=1,
                    body_id=source_body,
                    native_link_index=geom_links[source],
                    other_actor_name=geom_names[other] if other_body == 0 else None,
                    other_body_id=other_body,
                    source_actor_index=source_actor,
                    source_link_index=source_link,
                    other_actor_index=other_actor,
                    other_link_index=other_link,
                    contact_distance=float(max(m.geom_margin[obj], m.geom_margin[other])),
                )
            )
            continue
        kind = kinds.get(int(m.sensor_type[i]))
        if kind is None or int(m.sensor_refid[i]) >= 0:
            raise NotImplementedError(f"superdex unsupported sensor {name!r} or reference frame")
        if kind.startswith("joint"):
            if obj not in single_dofs:
                raise NotImplementedError("superdex joint sensor requires a single-DoF joint")
            plans.append(SensorPlan(name=name, kind=kind, joint_index=single_dofs[obj], dim=1))
            continue
        typ = int(m.sensor_objtype[i])
        if typ == int(mj.mjtObj.mjOBJ_SITE):
            body, pos, quat = int(m.site_bodyid[obj]), m.site_pos[obj], m.site_quat[obj]
        elif typ == int(mj.mjtObj.mjOBJ_GEOM):
            body, pos, quat = int(m.geom_bodyid[obj]), m.geom_pos[obj], m.geom_quat[obj]
        elif typ in (int(mj.mjtObj.mjOBJ_BODY), int(mj.mjtObj.mjOBJ_XBODY)):
            body = obj
            pos, quat = (
                ([0, 0, 0], [1, 0, 0, 0])
                if typ == int(mj.mjtObj.mjOBJ_XBODY)
                else (m.body_ipos[obj], m.body_iquat[obj])
            )
        else:
            raise NotImplementedError(f"superdex unsupported sensor object for {name!r}")
        plans.append(
            SensorPlan(
                name=name,
                kind=kind,
                body_id=body,
                local_pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                local_quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                dim=int(m.sensor_dim[i]),
            )
        )
    return tuple(plans)
