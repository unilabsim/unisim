"""Cold-path MJCF scan and model audit for the Newton adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from unisim.backend.scene_cleanup import TemporarySceneCleanup
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
    body_iquat: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray
    joint_names: tuple[str, ...]
    joint_qpos_adrs: tuple[int, ...]
    joint_dof_adrs: tuple[int, ...]
    ball_joint_names: tuple[str, ...]
    ball_joint_qpos_adrs: tuple[int, ...]
    ball_joint_qvel_adrs: tuple[int, ...]
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

    @property
    def ball_joint_addresses(self) -> tuple[tuple[int, int], ...]:
        """Return one ``(qpos address, qvel address)`` pair per authored ball joint."""
        return tuple(
            zip(self.ball_joint_qpos_adrs, self.ball_joint_qvel_adrs, strict=True)
        )


@dataclass(frozen=True, slots=True)
class NewtonModelAudit:
    """Successful authored-vs-compiled audit summary."""

    worlds: int
    bodies_per_world: int
    qpos_per_world: int
    qvel_per_world: int


def build_source_builder(
    newton: Any,
    model_file: str,
    gravity: np.ndarray | None = None,
    metadata: NewtonModelMetadata | None = None,
    layout: CompiledSceneLayout | None = None,
) -> Any:
    """Import one independently compiled portable full-scene source.

    Newton's current MJCF importer drops two authored facts on free roots: the
    free-joint armature and physical body mode used by kinematic roots (#444,
    #445). The adapter restores both from the direct MuJoCo compile and frozen
    public layout so its diagnostic and native models cannot silently diverge.
    """
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    # MjSpec round-trips drop geom gap="0" (it equals the built-in default);
    # pin Newton's gap fallback to MuJoCo semantics instead of its 0.1 default.
    builder.rigid_gap = 0.0
    builder.add_mjcf(model_file, ctrl_direct=False)
    if metadata is not None:
        _restore_free_joint_armature(newton, builder, metadata)
    if layout is not None:
        _apply_physical_kinematic_body_modes(newton, builder, layout)
    if gravity is not None:
        builder.gravity = tuple(float(value) for value in gravity)
    return builder


def _restore_free_joint_armature(newton: Any, builder: Any, metadata: NewtonModelMetadata) -> None:
    """Restore authored free-root armature dropped by Newton's importer (#445)."""
    free_type = int(newton.JointType.FREE)
    labels = tuple(str(label) for label in builder.joint_label)
    joint_types = tuple(int(value) for value in builder.joint_type)
    qd_starts = tuple(int(value) for value in builder.joint_qd_start)
    if len(labels) != len(joint_types) or len(labels) != len(qd_starts):
        raise RuntimeError("Newton importer exposed inconsistent joint layouts")
    restored = False
    for joint_type, qd_start in zip(joint_types, qd_starts, strict=True):
        if joint_type != free_type:
            continue
        if qd_start + 6 > len(builder.joint_armature) or qd_start + 6 > metadata.nv:
            raise RuntimeError("Newton importer exposed an invalid free-joint DOF layout")
        authored = np.asarray(metadata.dof_armature[qd_start : qd_start + 6], dtype=np.float32)
        builder.joint_armature[qd_start : qd_start + 6] = authored.tolist()
        restored = True
    if not restored and metadata.root_qvel_dim:
        raise RuntimeError(
            "Newton importer changed the free-joint layout before armature restoration"
        )


def _apply_physical_kinematic_body_modes(
    newton: Any, builder: Any, layout: CompiledSceneLayout
) -> None:
    """Mark physical kinematic roots after Newton MJCF import (#444)."""
    kinematic_flag = int(newton.BodyFlags.KINEMATIC)
    for entity in layout.entities:
        if entity.root_mode != "kinematic" or not entity.root_qpos_indices:
            continue
        root_body = int(entity.body_ids[0]) - 1
        if root_body < 0 or root_body >= len(builder.body_flags):
            raise RuntimeError(
                f"physical kinematic entity {entity.name!r} has an invalid native root body"
            )
        builder.body_flags[root_body] = kinematic_flag


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
        if any(
            entity.root_mode == "kinematic" and not entity.root_qpos_indices
            for entity in layout.entities
        ):
            raise NotImplementedError(
                "Newton portable entity composition does not support kinematic mirrors"
            )


def build_assigned_world_builder(
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
    matched = ((shape0 == pair_a) & (shape1 == pair_b)) | ((shape0 == pair_b) & (shape1 == pair_a))
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
            f'newton backend supports contact sensor {name!r} only with data="found"; '
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
        raise NotImplementedError(f"newton backend requires named geoms on contact sensor {name!r}")
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


# Newton's MJCF importer forwards ``Mesh.MAX_HULL_VERTICES`` (64) to the MuJoCo
# compiler whenever the MJCF mesh asset does not author its own ``maxhullvert``,
# while a reference MuJoCo compile leaves the full hull intact. Colliding meshes
# whose hull exceeds that fallback are rejected below so the compiled collision
# geometry cannot silently diverge from the reference hull.
_NEWTON_FALLBACK_MAXHULLVERT = 64

# Relative volume deviation tolerated between a closed collision mesh and its
# compiled convex hull before the mesh is declared non-convex.
_MESH_CONVEXITY_RTOL = 1e-4

# Absolute deviation floor, scaled by the mesh bounding-box scale cubed. Mesh
# assets are commonly float32-quantized (binary STL), and fan-triangulating
# slightly non-planar hull n-gons adds absolute noise; both grow with scale³
# and dominate the relative deviation of small hulls. The 22 sharpa hand hulls
# (1e-6..2e-4 m³) measure ≤9e-6 × scale³; genuine non-convexity is orders of
# magnitude larger, so the floor cannot mask it.
_MESH_CONVEXITY_ABS_SCALE = 1e-4


def _mesh_name(mujoco: Any, model: Any, mesh_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    return str(name) if name else str(mesh_id)


def _mesh_surface_closed(faces: np.ndarray) -> bool:
    """Return True when every undirected triangle edge is shared by two faces."""
    if not faces.size:
        return False
    edges = np.sort(
        np.concatenate((faces[:, :2], faces[:, 1:3], faces[:, ::2]), axis=0), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _mesh_signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Sum the signed tetrahedron volumes of outward-oriented triangle fans."""
    corners = vertices[triangles]
    return (
        float(
            np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])).sum()
        )
        / 6.0
    )


def _compiled_mesh_hull(model: Any, mesh_id: int) -> tuple[np.ndarray, float]:
    """Read the compiled convex hull polygons of one mesh.

    Returns the distinct mesh-local vertex ids on the hull and the hull volume.
    MuJoCo stores the collision hull as outward-oriented polygons in
    ``mesh_poly*``; ``mesh_polyvert`` indexes the mesh's own vertex block
    (zero-based within the mesh, the same convention as ``mesh_face``), so
    vertex lookups must slice ``mesh_vert`` at ``mesh_vertadr[mesh_id]``.
    Fan-triangulation around each polygon's first vertex sums the hull's
    signed volume.
    """
    vertex_adr = int(model.mesh_vertadr[mesh_id])
    vertex_num = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_adr : vertex_adr + vertex_num], dtype=np.float64
    )
    polyvert = np.asarray(model.mesh_polyvert, dtype=np.int64)
    poly_start = int(model.mesh_polyadr[mesh_id])
    poly_count = int(model.mesh_polynum[mesh_id])
    hull_vertex_ids: set[int] = set()
    volume = 0.0
    for poly_id in range(poly_start, poly_start + poly_count):
        adr = int(model.mesh_polyvertadr[poly_id])
        num = int(model.mesh_polyvertnum[poly_id])
        ids = polyvert[adr : adr + num]
        hull_vertex_ids.update(int(value) for value in ids)
        corners = vertices[ids]
        volume += float(
            np.sum(corners[0] * np.cross(corners[1:-1], corners[2:]))
        )
    local_ids = np.asarray(sorted(hull_vertex_ids), dtype=np.int64)
    return local_ids, abs(volume) / 6.0


def _audit_mesh_convexity(mujoco: Any, model: Any, mesh_id: int) -> str | None:
    """Audit one compiled collision mesh for closedness and convexity.

    MuJoCo-family solvers collide the convex hull of a mesh geom, so a mesh
    that is open or non-convex would silently collide a different shape than
    authored. A closed mesh is convex exactly when its surface volume equals
    its convex-hull volume.
    """
    name = _mesh_name(mujoco, model, mesh_id)
    vertex_adr = int(model.mesh_vertadr[mesh_id])
    vertex_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_adr : vertex_adr + vertex_num], dtype=np.float64
    )
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64)
    if not _mesh_surface_closed(faces):
        return (
            f"collision mesh {name!r} is not closed: every triangle edge must be shared by "
            "exactly two faces, otherwise the compiled convex hull would silently add "
            "surface area"
        )
    hull_vertex_ids, hull_volume = _compiled_mesh_hull(model, mesh_id)
    authored_volume = abs(_mesh_signed_volume(vertices, faces))
    if not hull_vertex_ids.size or hull_volume <= 0.0:
        return (
            f"collision mesh {name!r} produced no compiled convex hull; its convexity "
            "cannot be audited"
        )
    deviation = abs(hull_volume - authored_volume)
    relative = deviation / max(hull_volume, authored_volume)
    scale = float(np.ptp(vertices, axis=0).max())
    if relative > _MESH_CONVEXITY_RTOL and (
        deviation > _MESH_CONVEXITY_ABS_SCALE * scale**3
    ):
        return (
            f"collision mesh {name!r} is not convex: the compiled convex-hull volume "
            f"deviates {relative:.2%} from the authored surface volume, so the mesh would "
            "silently collide a different shape; split the mesh into convex pieces (for "
            "example with CoACD or V-HACD) or mark it visual-only with contype=0 "
            "conaffinity=0"
        )
    return None


def _audit_mesh_hull_budget(
    mujoco: Any, model: Any, source_model_file: str, mesh_id: int
) -> str | None:
    """Reject hulls that newton would decimate differently from a reference compile.

    Newton's importer defaults ``maxhullvert`` to 64 while MuJoCo leaves the
    full hull intact unless the MJCF mesh asset authors a limit. The authored
    value is read from an ``MjSpec`` re-parse of the source file (default
    classes resolved); it is absent from the compiled model.
    """
    hull_vertex_ids, _ = _compiled_mesh_hull(model, mesh_id)
    if hull_vertex_ids.size <= _NEWTON_FALLBACK_MAXHULLVERT:
        return None
    spec_meshes = mujoco.MjSpec.from_file(source_model_file).meshes
    authored = int(spec_meshes[mesh_id].maxhullvert) if mesh_id < len(spec_meshes) else -1
    effective = authored if authored >= 4 else _NEWTON_FALLBACK_MAXHULLVERT
    if hull_vertex_ids.size <= effective:
        return None
    name = _mesh_name(mujoco, model, mesh_id)
    return (
        f"collision mesh {name!r} has a {hull_vertex_ids.size}-vertex convex hull, which "
        f"exceeds the effective maxhullvert of {effective}; newton applies a fallback of "
        f"{_NEWTON_FALLBACK_MAXHULLVERT} while a reference MuJoCo compile keeps the full "
        f"hull, so the compiled collision geometry would silently diverge. Author "
        f"maxhullvert={hull_vertex_ids.size} on the MJCF mesh asset so both engines "
        "compile the identical hull"
    )


def _reject_silent_geometry_gaps(mujoco: Any, model: Any, source_model_file: str) -> None:
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
    mesh_reasons: dict[int, str | None] = {}
    for geom_id in colliding_mesh_ids:
        mesh_id = int(model.geom_dataid[int(geom_id)])
        if mesh_id not in mesh_reasons:
            reason = _audit_mesh_convexity(mujoco, model, mesh_id)
            if reason is None:
                reason = _audit_mesh_hull_budget(mujoco, model, source_model_file, mesh_id)
            mesh_reasons[mesh_id] = reason
    failures: list[str] = []
    for geom_id in colliding_mesh_ids:
        reason = mesh_reasons[int(model.geom_dataid[int(geom_id)])]
        if reason is None:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        failures.append(f"geom {str(geom_name) if geom_name else int(geom_id)!r}: {reason}")
    if failures:
        raise NotImplementedError(
            "newton backend audits colliding mesh geoms for convexity so the compiled "
            "convex hull matches the authored geometry; unsupported meshes were found: "
            + "; ".join(failures)
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
    _reject_silent_geometry_gaps(mujoco, model, source_model_file)

    free = int(mujoco.mjtJoint.mjJNT_FREE)
    ball = int(mujoco.mjtJoint.mjJNT_BALL)
    single_dof = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    supported_joints = single_dof | {free, ball}
    unsupported_joints = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) not in supported_joints
    ]
    if unsupported_joints:
        raise NotImplementedError(
            "newton backend supports only free, ball, hinge, and slide joints; "
            f"unsupported MJCF joint ids: {unsupported_joints}"
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

    # Ball joints are passive multi-DoF joints: like the free root they are
    # not name-addressable through the single-DoF joint API, and the actuator
    # gate below keeps rejecting actuators that target them. MJCF joints have
    # no orientation attribute, so the child anchor rotation is structurally
    # identity and Newton's parent-anchor state layout differs from the
    # public MJCF layout only by the quaternion order and the body-frame
    # angular velocity, which the backend translates exactly.
    ball_joint_names: list[str] = []
    ball_joint_qpos_adrs: list[int] = []
    ball_joint_qvel_adrs: list[int] = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) != ball:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise NotImplementedError(
                f"newton backend requires named ball joints; joint id {joint_id} is unnamed"
            )
        ball_joint_names.append(str(name))
        ball_joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        ball_joint_qvel_adrs.append(int(model.jnt_dofadr[joint_id]))

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
    single_dof_rows = np.isin(
        np.asarray(model.jnt_type, dtype=np.int32), np.asarray(sorted(single_dof), dtype=np.int32)
    )
    joint_range = np.asarray(model.jnt_range, dtype=np.float32)[single_dof_rows]
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
        body_iquat=np.asarray(model.body_iquat, dtype=np.float32).copy(),
        geom_contype=np.asarray(model.geom_contype, dtype=np.int32).copy(),
        geom_conaffinity=np.asarray(model.geom_conaffinity, dtype=np.int32).copy(),
        joint_names=tuple(joint_names),
        joint_qpos_adrs=tuple(joint_qpos_adrs),
        joint_dof_adrs=tuple(joint_dof_adrs),
        ball_joint_names=tuple(ball_joint_names),
        ball_joint_qpos_adrs=tuple(ball_joint_qpos_adrs),
        ball_joint_qvel_adrs=tuple(ball_joint_qvel_adrs),
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
    expected_mass = np.stack([metadata[int(i)].body_mass[1:] for i in assignment_array]).reshape(
        num_envs, expected_bodies
    )
    expected_com = np.stack([metadata[int(i)].body_ipos[1:] for i in assignment_array]).reshape(
        num_envs, expected_bodies, 3
    )
    variant_inertia: list[np.ndarray] = []
    for item in metadata:
        body_inertia = np.asarray(item.body_inertia, dtype=np.float32)
        if body_inertia.shape != (item.nbody, 3):
            raise RuntimeError("Newton metadata owner did not compile body inertia")
        # MuJoCo stores principal-frame eigenvalues while Newton's
        # body_inertia is the full tensor in the body frame; rotate the
        # diagonal into the body frame with body_iquat (wxyz) before the
        # element-wise comparison so rotated principal frames compare equal.
        diagonal = np.asarray(body_inertia[1:], dtype=np.float64)
        iquat = np.asarray(item.body_iquat[1:], dtype=np.float64)
        w, x, y, z = iquat[:, 0], iquat[:, 1], iquat[:, 2], iquat[:, 3]
        rotation = np.stack(
            [
                np.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    axis=-1,
                ),
                np.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    axis=-1,
                ),
                np.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        full = np.einsum("bij,bj,bkj->bik", rotation, diagonal, rotation)
        variant_inertia.append(full.astype(np.float32))
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
        if not np.allclose(shape_scale[actual_indices], expected_scale, rtol=1e-6, atol=1e-7):
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
    "build_assigned_world_builder",
    "build_source_builder",
    "compute_contact_found_flags",
    "scan_newton_model_metadata",
    "validate_newton_portable_metadata",
    "validate_newton_variant_sources",
]
