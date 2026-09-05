"""Cold-path MJCF scan and model audit for the Newton adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.scene import SceneCfg


class _TemporarySceneCleanup:
    def __init__(self, *paths: str) -> None:
        self._paths = paths
        self._cleaned = False

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        for path in self._paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


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
    nbody: int
    root_qpos_dim: int
    root_qvel_dim: int
    body_names: tuple[str, ...]
    body_parent_ids: np.ndarray
    body_mass: np.ndarray
    body_ipos: np.ndarray
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
        cleanup_handle=_TemporarySceneCleanup(*temp_paths) if temp_paths else None,
        playback_model=model,
        model_name=model_name,
        nq=int(model.nq),
        nv=int(model.nv),
        nbody=int(model.nbody),
        root_qpos_dim=root_qpos_dim,
        root_qvel_dim=root_qvel_dim,
        body_names=body_names,
        body_parent_ids=np.asarray(model.body_parentid, dtype=np.int32).copy(),
        body_mass=np.asarray(model.body_mass, dtype=np.float32).copy(),
        body_ipos=np.asarray(model.body_ipos, dtype=np.float32).copy(),
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


__all__ = [
    "NewtonModelAudit",
    "NewtonModelMetadata",
    "NewtonSensorPlan",
    "audit_newton_model",
    "compute_contact_found_flags",
    "scan_newton_model_metadata",
]
