"""Cold-path MJCF scan and model audit for the Newton adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.backend.materialization_common import TemporarySceneCleanup
from unisim.scene import SceneCfg
from unisim.scene_layout import CompiledSceneLayout


@dataclass(frozen=True, slots=True)
class NewtonSensorPlan:
    """One MJCF sensor reconstructed from public Newton state arrays.

    Site-attached plans populate ``body_id``/``site_pos``/``site_quat``.
    Contact plans (``kind == "contact"``) instead carry the authored geom
    pair; their per-world compiled shape indices are resolved once at
    materialization against ``SolverMuJoCo.mjc_geom_to_newton_shape``.
    """

    name: str
    kind: str
    dim: int
    body_id: int
    site_pos: np.ndarray
    site_quat: np.ndarray
    geom1_name: str = ""
    geom2_name: str = ""
    geom1_id: int = -1
    geom2_id: int = -1


@dataclass(frozen=True, slots=True)
class NewtonModelMetadata:
    """Immutable authoring metadata scanned once with MuJoCo."""

    source_model_file: str
    diagnostic_model_file: str
    cleanup_handle: Any | None
    playback_model: Any
    model_name: str
    nq: int
    nv: int
    nu: int
    nbody: int
    root_qpos_dim: int
    root_qvel_dim: int
    body_names: tuple[str, ...]
    body_parent_ids: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_mass: np.ndarray
    body_ipos: np.ndarray
    body_inertia: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray
    joint_names: tuple[str, ...]
    joint_qpos_adrs: tuple[int, ...]
    joint_dof_adrs: tuple[int, ...]
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_target_kinds: tuple[str, ...]
    actuator_target_qpos_adrs: tuple[int, ...]
    actuator_target_qvel_adrs: tuple[int, ...]
    actuator_ctrl_range: np.ndarray
    actuator_kp: np.ndarray
    actuator_kd: np.ndarray
    keyframes: tuple[tuple[str, np.ndarray], ...]
    default_qpos: np.ndarray
    joint_range: np.ndarray | None
    dof_armature: np.ndarray
    gravity: np.ndarray
    sensor_plans: tuple[NewtonSensorPlan, ...]


@dataclass(frozen=True, slots=True)
class NewtonModelAudit:
    """Successful authored-vs-compiled audit summary."""

    worlds: int
    bodies_per_world: int
    qpos_per_world: int
    qvel_per_world: int


def build_newton_source_builder(
    newton: Any, model_file: str, gravity: np.ndarray | None = None
) -> Any:
    """Import one independently compiled portable full-scene source."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(model_file, ctrl_direct=False)
    if gravity is not None:
        builder.gravity = tuple(float(value) for value in gravity)
    return builder


def validate_newton_variant_sources(
    builders: tuple[Any, ...], metadata: tuple[NewtonModelMetadata, ...]
) -> tuple[tuple[int, ...], ...]:
    """Reject differing shape-type sequences before Newton world assembly.

    Newton's MuJoCo solver accepts same-type heterogeneous dimensions and
    inertials across worlds, but rejects mixed shape types only after model
    finalization/solver construction. Portable profiles get a deterministic
    adapter-owned diagnostic instead of that late SDK failure.
    """
    if not builders or len(builders) != len(metadata):
        raise ValueError("one Newton source builder and metadata record are required per variant")
    sequences: list[tuple[int, ...]] = []
    for variant, builder in enumerate(builders):
        shape_types = builder.shape_type
        if not isinstance(shape_types, list):
            raise RuntimeError(f"Newton variant {variant} did not expose public shape types")
        sequences.append(tuple(int(value) for value in shape_types))
    reference = sequences[0]
    for variant, sequence in enumerate(sequences[1:], start=1):
        if sequence != reference:
            raise ValueError(
                "Newton SolverMuJoCo requires the same shape-type sequence in every world; "
                f"variant 0 has {reference}, variant {variant} has {sequence}"
            )
    return tuple(sequences)


def validate_newton_portable_metadata(
    metadata: tuple[NewtonModelMetadata, ...], layout: CompiledSceneLayout
) -> None:
    """Bind compiled portable MJCF metadata to the frozen public layout."""
    for variant, item in enumerate(metadata):
        if item.nq != layout.nq or item.nv != layout.nv or item.nu != layout.nu:
            raise ValueError(f"Newton variant {variant} differs from the frozen scene layout")
        if item.nbody != layout.nbody:
            raise ValueError(f"Newton variant {variant} has an unexpected body count")
        public_bodies = [""] * (layout.nbody - 1)
        for entity in layout.entities:
            for local_name, body_id in zip(entity.body_names, entity.body_ids, strict=True):
                body_row = body_id - 1
                if body_row < 0 or body_row >= len(public_bodies):
                    raise ValueError(
                        f"Newton public layout body id {body_id} is outside the compiled model"
                    )
                if public_bodies[body_row]:
                    raise ValueError(
                        f"Newton public layout assigns body id {body_id} to multiple entities"
                    )
                public_bodies[body_row] = f"{entity.name}/{local_name}"
        if any(not name for name in public_bodies):
            raise ValueError("Newton public layout does not assign every compiled body")
        if item.body_names[1:] != tuple(public_bodies):
            raise ValueError(f"Newton variant {variant} body order differs from the public layout")
        if any(entity.root_mode == "kinematic" for entity in layout.entities):
            raise NotImplementedError(
                "Newton portable entity composition does not support kinematic mirrors"
            )


def build_newton_assigned_world_builder(
    newton: Any,
    source_builders: tuple[Any, ...],
    metadata: tuple[NewtonModelMetadata, ...],
    assignment: np.ndarray,
) -> Any:
    """Copy immutable full-scene source builders into their assigned worlds."""
    if not source_builders or len(source_builders) != len(metadata):
        raise ValueError("one Newton source builder and metadata record is required per variant")
    assignment_array = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if assignment_array.size == 0:
        raise ValueError("Newton world assignment cannot be empty")
    if np.any(assignment_array < 0) or np.any(assignment_array >= len(source_builders)):
        raise ValueError("Newton world assignment refers to an absent source variant")
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for variant_value in assignment_array.tolist():
        variant = int(variant_value)
        builder.begin_world(gravity=tuple(float(value) for value in metadata[variant].gravity))
        builder.add_builder(source_builders[int(variant)])
        builder.end_world()
    return builder


# MuJoCo compiles the MJCF contact-sensor ``data`` attribute into a bitmask in
# ``sensor_intprm[:, 0]``; bit 0 is ``found``.  ``sensor_intprm[:, 1]`` is the
# ``reduce`` mode and ``sensor_intprm[:, 2]`` is ``num``.
_CONTACT_DATASPEC_FOUND = 1


def compute_contact_found_flags(
    shape_world: np.ndarray,
    contact_shape0: np.ndarray,
    contact_shape1: np.ndarray,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
) -> np.ndarray:
    """Return 1.0 per env whose world produced a contact between the pair.

    ``shape_a``/``shape_b`` hold the compiled Newton shape indices of the
    authored geom pair for each env world, and ``shape_world`` maps every
    Newton shape to its world index (shared/static shapes carry a negative
    index).  Contact shape pairs are unordered.  A contact is attributed to
    the world of its non-static shape; contacts without a consistent env world
    (two shared shapes, or two shapes from different worlds) match nothing.
    """
    shape_a = np.asarray(shape_a, dtype=np.int64).reshape(-1)
    shape_b = np.asarray(shape_b, dtype=np.int64).reshape(-1)
    found = np.zeros((shape_a.shape[0],), dtype=np.float32)
    shape0 = np.asarray(contact_shape0, dtype=np.int64).reshape(-1)
    shape1 = np.asarray(contact_shape1, dtype=np.int64).reshape(-1)
    if not shape0.size:
        return found
    shape_world = np.asarray(shape_world, dtype=np.int64)
    world0 = shape_world[shape0]
    world1 = shape_world[shape1]
    world = np.maximum(world0, world1)
    same_world = (world0 == world1) | (world0 < 0) | (world1 < 0)
    valid = same_world & (world >= 0) & (world < found.shape[0])
    if not np.any(valid):
        return found
    world = world[valid]
    shape0 = shape0[valid]
    shape1 = shape1[valid]
    pair_a = shape_a[world]
    pair_b = shape_b[world]
    matched = ((shape0 == pair_a) & (shape1 == pair_b)) | (
        (shape0 == pair_b) & (shape1 == pair_a)
    )
    found[world[matched]] = 1.0
    return found


def _scan_contact_sensor(mujoco: Any, model: Any, sensor_id: int, name: str) -> NewtonSensorPlan:
    """Scan one ``mjSENS_CONTACT`` sensor restricted to ``found`` geom pairs.

    Only the exact authored shape ``data="found" num=1 geom1=... geom2=...``
    is supported.  Every ``reduce`` mode (none/mindist/netforce) is
    accepted because it only selects which contact fills the single reported
    slot; the binary ``found`` flag is identical for all of them.  Any other
    configuration (other data channels, ``num > 1``, body/subtree/site
    contacts, unnamed geoms) fails closed.
    """
    intprm = np.asarray(model.sensor_intprm[sensor_id], dtype=np.int64)
    dataspec = int(intprm[0])
    num = int(intprm[2])
    if dataspec != _CONTACT_DATASPEC_FOUND:
        raise NotImplementedError(
            f"newton backend supports contact sensor {name!r} only with data=\"found\"; "
            f"compiled dataspec bitmask is {dataspec}"
        )
    if num != 1:
        raise NotImplementedError(
            f"newton backend supports contact sensor {name!r} only with num=1; "
            f"compiled num is {num}"
        )
    if int(model.sensor_dim[sensor_id]) != 1:
        raise NotImplementedError(
            f"newton backend expected contact sensor {name!r} dim 1, "
            f"compiled dim is {int(model.sensor_dim[sensor_id])}"
        )
    geom_obj = int(mujoco.mjtObj.mjOBJ_GEOM)
    if (
        int(model.sensor_objtype[sensor_id]) != geom_obj
        or int(model.sensor_reftype[sensor_id]) != geom_obj
    ):
        raise NotImplementedError(
            f"newton backend supports contact sensor {name!r} only for named geom1/geom2 "
            "pairs; body, subtree, and site contacts are unsupported"
        )
    geom1_id = int(model.sensor_objid[sensor_id])
    geom2_id = int(model.sensor_refid[sensor_id])
    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
    if not geom1_name or not geom2_name:
        raise NotImplementedError(
            f"newton backend requires named geoms on contact sensor {name!r}"
        )
    return NewtonSensorPlan(
        name=str(name),
        kind="contact",
        dim=1,
        body_id=-1,
        site_pos=np.zeros(3, dtype=np.float32),
        site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        geom1_name=str(geom1_name),
        geom2_name=str(geom2_name),
        geom1_id=geom1_id,
        geom2_id=geom2_id,
    )


def _scan_sensors(mujoco: Any, model: Any) -> tuple[NewtonSensorPlan, ...]:
    supported = {
        mujoco.mjtSensor.mjSENS_GYRO: "gyro",
        mujoco.mjtSensor.mjSENS_ACCELEROMETER: "accelerometer",
        mujoco.mjtSensor.mjSENS_VELOCIMETER: "velocimeter",
        mujoco.mjtSensor.mjSENS_FRAMEPOS: "framepos",
        mujoco.mjtSensor.mjSENS_FRAMEQUAT: "framequat",
        mujoco.mjtSensor.mjSENS_FRAMEZAXIS: "framezaxis",
    }
    plans: list[NewtonSensorPlan] = []
    for sensor_id in range(int(model.nsensor)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        sensor_type = mujoco.mjtSensor(int(model.sensor_type[sensor_id]))
        if not name:
            raise NotImplementedError(
                f"newton backend requires named MJCF sensors; sensor id {sensor_id} is unnamed"
            )
        if sensor_type == mujoco.mjtSensor.mjSENS_CONTACT:
            plans.append(_scan_contact_sensor(mujoco, model, sensor_id, str(name)))
            continue
        kind = supported.get(sensor_type)
        if kind is None:
            raise NotImplementedError(
                f"newton backend cannot reconstruct MJCF sensor {name!r} of type "
                f"{sensor_type.name} from public Newton State arrays"
            )
        if int(model.sensor_objtype[sensor_id]) != int(mujoco.mjtObj.mjOBJ_SITE):
            raise NotImplementedError(
                f"newton backend reconstructs {kind} sensor {name!r} from sites only"
            )
        site_id = int(model.sensor_objid[sensor_id])
        body_id = int(model.site_bodyid[site_id])
        plans.append(
            NewtonSensorPlan(
                name=str(name),
                kind=kind,
                dim=int(model.sensor_dim[sensor_id]),
                body_id=body_id,
                site_pos=np.asarray(model.site_pos[site_id], dtype=np.float32).copy(),
                site_quat=np.asarray(model.site_quat[site_id], dtype=np.float32).copy(),
            )
        )
    return tuple(plans)


def _reject_silent_geometry_gaps(mujoco: Any, model: Any) -> None:
    geom_types = np.asarray(model.geom_type, dtype=np.int32)
    cone_value = getattr(mujoco.mjtGeom, "mjGEOM_CONE", None)
    if cone_value is not None and np.any(geom_types == int(cone_value)):
        raise NotImplementedError(
            "newton SolverMuJoCo does not map GeoType.CONE; replace cone geometry "
            "before selecting the newton backend"
        )
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    masks = np.asarray(model.geom_contype, dtype=np.int64) | np.asarray(
        model.geom_conaffinity, dtype=np.int64
    )
    colliding_mesh_ids = np.flatnonzero((geom_types == mesh) & (masks != 0))
    if colliding_mesh_ids.size:
        names = [
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or geom_id)
            for geom_id in colliding_mesh_ids
        ]
        raise NotImplementedError(
            "newton SolverMuJoCo convexifies triangle-mesh collision geometry; "
            f"colliding mesh geoms are rejected: {', '.join(names)}"
        )


def scan_newton_model_metadata(mujoco: Any, scene: SceneCfg) -> NewtonModelMetadata:
    """Compose fragments, reject known silent gaps, and cache MJCF metadata."""
    if scene is None or not scene.model_file:
        raise ValueError("NewtonBackend requires SceneCfg.model_file")
    if scene.terrain is not None:
        raise NotImplementedError(
            "newton backend does not yet support generated terrain or height-field scanners"
        )
    temp_paths: list[str] = []
    source_model_file = str(scene.model_file)
    if scene.fragment_files:
        from unisim.backend.mujoco.xml import materialize_scene_fragments

        source_model_file = materialize_scene_fragments(
            source_model_file, fragment_files=scene.fragment_files
        )
        temp_paths.append(source_model_file)
    model = mujoco.MjModel.from_xml_path(source_model_file)
    _reject_silent_geometry_gaps(mujoco, model)

    free = int(mujoco.mjtJoint.mjJNT_FREE)
    single_dof = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    supported_joints = single_dof | {free}
    unsupported_joints = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) not in supported_joints
    ]
    if unsupported_joints:
        raise NotImplementedError(
            "newton backend supports only free, hinge, and slide joints; unsupported "
            f"MJCF joint ids: {unsupported_joints}"
        )
    root_qpos_dim, root_qvel_dim = (
        (7, 6) if int(model.njnt) and int(model.jnt_type[0]) == free else (0, 0)
    )
    joint_names: list[str] = []
    joint_qpos_adrs: list[int] = []
    joint_dof_adrs: list[int] = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) not in single_dof:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise NotImplementedError(
                f"newton backend requires named single-DoF joints; joint id {joint_id} is unnamed"
            )
        joint_names.append(str(name))
        joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_adrs.append(int(model.jnt_dofadr[joint_id]))

    actuator_names: list[str] = []
    actuator_joint_names: list[str] = []
    actuator_target_kinds: list[str] = []
    actuator_target_qpos_adrs: list[int] = []
    actuator_target_qvel_adrs: list[int] = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if (
            not name
            or int(model.actuator_trntype[actuator_id])
            not in {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
            or joint_id < 0
            or int(model.jnt_type[joint_id]) not in single_dof
        ):
            raise NotImplementedError(
                "newton backend requires named actuators targeting single-DoF joints; "
                f"actuator id {actuator_id} is unsupported"
            )
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        bias = np.asarray(model.actuator_biasprm[actuator_id], dtype=np.float32)
        if abs(float(bias[1])) > 1e-8:
            target_kind = "position"
        elif abs(float(bias[2])) > 1e-8:
            target_kind = "velocity"
        else:
            target_kind = "direct"
        actuator_names.append(str(name))
        actuator_joint_names.append(str(joint_name))
        actuator_target_kinds.append(target_kind)
        actuator_target_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        actuator_target_qvel_adrs.append(int(model.jnt_dofadr[joint_id]))

    keyframes = tuple(
        (
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)),
            np.asarray(model.key_qpos[key_id], dtype=np.float32).copy(),
        )
        for key_id in range(int(model.nkey))
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
    )
    non_free = np.asarray(model.jnt_type, dtype=np.int32) != free
    joint_range = np.asarray(model.jnt_range, dtype=np.float32)[non_free]
    body_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        for body_id in range(int(model.nbody))
    )
    raw_model_names = getattr(model, "names", b"")
    model_name = (
        raw_model_names.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        if isinstance(raw_model_names, bytes)
        else str(raw_model_names)
    )
    return NewtonModelMetadata(
        source_model_file=source_model_file,
        diagnostic_model_file=str(scene.model_file),
        cleanup_handle=TemporarySceneCleanup(*temp_paths) if temp_paths else None,
        playback_model=model,
        model_name=model_name,
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        nbody=int(model.nbody),
        root_qpos_dim=root_qpos_dim,
        root_qvel_dim=root_qvel_dim,
        body_names=body_names,
        body_parent_ids=np.asarray(model.body_parentid, dtype=np.int32).copy(),
        body_pos=np.asarray(model.body_pos, dtype=np.float32).copy(),
        body_quat=np.asarray(model.body_quat, dtype=np.float32).copy(),
        body_mass=np.asarray(model.body_mass, dtype=np.float32).copy(),
        body_ipos=np.asarray(model.body_ipos, dtype=np.float32).copy(),
        body_inertia=np.asarray(model.body_inertia, dtype=np.float32).copy(),
        geom_contype=np.asarray(model.geom_contype, dtype=np.int32).copy(),
        geom_conaffinity=np.asarray(model.geom_conaffinity, dtype=np.int32).copy(),
        joint_names=tuple(joint_names),
        joint_qpos_adrs=tuple(joint_qpos_adrs),
        joint_dof_adrs=tuple(joint_dof_adrs),
        actuator_names=tuple(actuator_names),
        actuator_joint_names=tuple(actuator_joint_names),
        actuator_target_kinds=tuple(actuator_target_kinds),
        actuator_target_qpos_adrs=tuple(actuator_target_qpos_adrs),
        actuator_target_qvel_adrs=tuple(actuator_target_qvel_adrs),
        actuator_ctrl_range=np.asarray(model.actuator_ctrlrange, dtype=np.float32).copy(),
        actuator_kp=np.asarray(model.actuator_gainprm[:, 0], dtype=np.float32).copy(),
        actuator_kd=np.asarray(-model.actuator_biasprm[:, 2], dtype=np.float32).copy(),
        keyframes=keyframes,
        default_qpos=np.asarray(model.qpos0, dtype=np.float32).copy(),
        joint_range=None if not joint_range.size else joint_range.copy(),
        dof_armature=np.asarray(model.dof_armature, dtype=np.float32).copy(),
        gravity=np.asarray(model.opt.gravity, dtype=np.float32).copy(),
        sensor_plans=_scan_sensors(mujoco, model),
    )


def audit_newton_model(
    model: Any, metadata: NewtonModelMetadata, num_envs: int
) -> NewtonModelAudit:
    """Compare authored gravity/mass/layout against the finalized Newton model."""
    expected_bodies = metadata.nbody - 1
    if int(model.world_count) != num_envs:
        raise RuntimeError(
            f"newton compiled world count {model.world_count} != authored request {num_envs}"
        )
    if int(model.joint_coord_count) != metadata.nq * num_envs:
        raise RuntimeError("newton compiled qpos layout differs from the authored MJCF")
    if int(model.joint_dof_count) != metadata.nv * num_envs:
        raise RuntimeError("newton compiled qvel layout differs from the authored MJCF")
    if int(model.body_count) != expected_bodies * num_envs:
        raise RuntimeError("newton compiled body layout differs from the authored MJCF")

    gravity_np = np.asarray(model.gravity.numpy(), dtype=np.float32)
    if gravity_np.shape[0] == num_envs + 1:
        # Newton 1.5.1 appends one gravity entry for the global world -1;
        # only the per-world rows participate in the audit.
        gravity_np = gravity_np[:num_envs]
    gravity = gravity_np.reshape(num_envs, 3)
    if not np.allclose(gravity, metadata.gravity, rtol=1e-6, atol=1e-6):
        raise RuntimeError("newton compiled gravity differs from the authored MJCF")
    mass = np.asarray(model.body_mass.numpy(), dtype=np.float32).reshape(num_envs, expected_bodies)
    if not np.allclose(mass, metadata.body_mass[1:], rtol=1e-5, atol=1e-6):
        raise RuntimeError("newton compiled body masses differ from the authored MJCF")
    return NewtonModelAudit(num_envs, expected_bodies, metadata.nq, metadata.nv)


def audit_newton_variant_model(
    model: Any,
    metadata: tuple[NewtonModelMetadata, ...],
    source_builders: tuple[Any, ...],
    assignment: np.ndarray,
    layout: CompiledSceneLayout,
) -> NewtonModelAudit:
    """Audit effective per-world native identity against each selected source.

    Source provenance is not accepted as effective evidence. The finalized
    Newton model must reproduce each assigned variant's masses, COMs, inertia,
    shape types, and shape dimensions in its actual per-world native rows.
    """
    validate_newton_portable_metadata(metadata, layout)
    validate_newton_variant_sources(source_builders, metadata)
    assignment_array = np.asarray(assignment, dtype=np.int64).reshape(-1)
    num_envs = int(assignment_array.size)
    expected_bodies = layout.nbody - 1
    if int(model.world_count) != num_envs:
        raise RuntimeError(
            f"newton compiled world count {model.world_count} != assigned variants {num_envs}"
        )
    if int(model.joint_coord_count) != layout.nq * num_envs:
        raise RuntimeError("newton compiled qpos layout differs from the portable layout")
    if int(model.joint_dof_count) != layout.nv * num_envs:
        raise RuntimeError("newton compiled qvel layout differs from the portable layout")
    if int(model.body_count) != expected_bodies * num_envs:
        raise RuntimeError("newton compiled body layout differs from the portable layout")
    if int(model.articulation_count) != len(layout.entities) * num_envs:
        raise RuntimeError("newton compiled articulation count differs from the portable layout")

    gravity_np = np.asarray(model.gravity.numpy(), dtype=np.float32)
    if gravity_np.shape[0] == num_envs + 1:
        gravity_np = gravity_np[:num_envs]
    expected_gravity = np.stack([metadata[int(i)].gravity for i in assignment_array])
    if not np.allclose(gravity_np.reshape(num_envs, 3), expected_gravity, rtol=1e-6, atol=1e-6):
        raise RuntimeError("newton compiled gravity differs from the assigned portable variants")

    mass = np.asarray(model.body_mass.numpy(), dtype=np.float32)
    com = np.asarray(model.body_com.numpy(), dtype=np.float32)
    inertia = np.asarray(model.body_inertia.numpy(), dtype=np.float32)
    expected_mass = np.stack(
        [metadata[int(i)].body_mass[1:] for i in assignment_array]
    ).reshape(num_envs, expected_bodies)
    expected_com = np.stack(
        [metadata[int(i)].body_ipos[1:] for i in assignment_array]
    ).reshape(num_envs, expected_bodies, 3)
    variant_inertia: list[np.ndarray] = []
    for item in metadata:
        body_inertia = np.asarray(item.body_inertia, dtype=np.float32)
        if body_inertia.shape != (item.nbody, 3):
            raise RuntimeError("Newton metadata owner did not compile body inertia")
        diagonal = body_inertia[1:]
        full = np.zeros((diagonal.shape[0], 3, 3), dtype=np.float32)
        full[:, 0, 0] = diagonal[:, 0]
        full[:, 1, 1] = diagonal[:, 1]
        full[:, 2, 2] = diagonal[:, 2]
        variant_inertia.append(full)
    selected_inertia = np.stack(variant_inertia)[assignment_array]
    if mass.size != num_envs * expected_bodies:
        raise RuntimeError("newton compiled body-mass layout differs from the portable layout")
    if not np.allclose(
        mass.reshape(num_envs, expected_bodies), expected_mass, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError("newton compiled body masses differ from the assigned variants")
    if not np.allclose(
        com.reshape(num_envs, expected_bodies, 3), expected_com, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError("newton compiled body COMs differ from the assigned variants")
    if not np.allclose(
        inertia.reshape(num_envs, expected_bodies, 3, 3),
        selected_inertia,
        rtol=2e-5,
        atol=2e-6,
    ):
        raise RuntimeError("newton compiled body inertias differ from the assigned variants")

    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int64)
    shape_type = np.asarray(model.shape_type.numpy(), dtype=np.int32)
    shape_scale = np.asarray(model.shape_scale.numpy(), dtype=np.float32)
    for world_id in range(num_envs):
        actual_indices = np.flatnonzero(shape_world == world_id)
        source = source_builders[int(assignment_array[world_id])]
        expected_types = np.asarray(source.shape_type, dtype=np.int32)
        if actual_indices.size != expected_types.size:
            raise RuntimeError(
                f"newton world {world_id} shape count differs from its assigned variant"
            )
        if not np.array_equal(shape_type[actual_indices], expected_types):
            raise RuntimeError(
                f"newton world {world_id} shape types differ from its assigned variant"
            )
        expected_scale = np.asarray(source.shape_scale, dtype=np.float32).reshape(-1, 3)
        if not np.allclose(
            shape_scale[actual_indices], expected_scale, rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError(
                f"newton world {world_id} shape dimensions differ from its assigned variant"
            )
    return NewtonModelAudit(num_envs, expected_bodies, layout.nq, layout.nv)


__all__ = [
    "NewtonModelAudit",
    "NewtonModelMetadata",
    "NewtonSensorPlan",
    "audit_newton_variant_model",
    "audit_newton_model",
    "build_newton_assigned_world_builder",
    "build_newton_source_builder",
    "compute_contact_found_flags",
    "scan_newton_model_metadata",
    "validate_newton_portable_metadata",
    "validate_newton_variant_sources",
]
