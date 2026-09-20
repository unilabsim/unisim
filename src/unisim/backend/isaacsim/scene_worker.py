"""Mapped IsaacLab scene execution, loaded by path in the external SDK worker.

Asset parsing, USD edits and native-name discovery are cold-path operations.
Runtime operations use frozen entity/view maps, never actor creation order.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from unisim.backend.isaacsim.physx_solver import (
    PhysxSolverConfig,
    apply_contact_offset,
    build_isaaclab_physx_cfg,
    read_engine_solver_values,
)
from unisim.backend.isaacsim.raw_usd_cache import (
    RAW_USD_ARTIFACT_STAGE,
    ROLE_USD_ARTIFACT_STAGE,
    ROLE_USD_CACHE_SCHEMA_VERSION,
    RawUSDArtifactRequest,
    RawUSDCache,
    RoleUSDCache,
    file_sha256,
    raw_artifact_fingerprint,
)
from unisim.backend.subprocess_ipc.scene_materialization import (
    body_sphere_radii_close,
    validate_body_sphere_radii,
)
from unisim.scene_compiler import (
    SceneContentIdentity,
    derive_scene_artifact_identity,
)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _rotate(q: np.ndarray, v: np.ndarray, inverse: bool = False) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if inverse:
        q[..., 1:] *= -1
    v = np.asarray(v, dtype=np.float64)
    t = 2 * np.cross(q[..., 1:], v)
    return (v + q[..., :1] * t + np.cross(q[..., 1:], t)).astype(np.float32)


def _entity_prim_component(name: str) -> str:
    """Injective USD identifier; public names never become native path syntax."""
    return "entity_" + name.encode("utf-8").hex()


def _native_environment_order(native_paths: list[str], entity_paths: list[str]) -> np.ndarray:
    """Resolve view rows against exact entity subtrees, never string prefixes alone."""
    native_envs = []
    for path in native_paths:
        matches = [
            index
            for index, root in enumerate(entity_paths)
            if path == root or path.startswith(root + "/")
        ]
        if len(matches) != 1:
            raise RuntimeError("native view contains an unowned or ambiguous instance")
        native_envs.append(matches[0])
    if sorted(native_envs) != list(range(len(entity_paths))):
        raise RuntimeError("native view needs exactly one instance per environment")
    return np.asarray(native_envs, dtype=np.int64)


def _validated_assignment(entry: dict[str, Any], count: int) -> np.ndarray:
    """Validate the immutable per-environment variant references before Kit starts."""
    assignment = np.asarray(entry["assignment"])
    if (
        assignment.shape != (count,)
        or assignment.dtype.kind not in "iu"
        or bool(np.issubdtype(assignment.dtype, np.bool_))
    ):
        raise ValueError(
            "entity assignment must be an integer array with one value per environment"
        )
    values = assignment.astype(np.int64, copy=False)
    source_count = len(entry["sources"])
    if values.min(initial=0) < 0 or values.max(initial=-1) >= source_count:
        raise ValueError("entity assignment contains an invalid variant index")
    return values


def _required_package_version(name: str) -> str:
    try:
        return str(version(name))
    except PackageNotFoundError as exc:
        raise RuntimeError(f"IsaacSim worker is missing required package {name!r}") from exc


def _raw_usd_runtime_versions() -> dict[str, str]:
    """Record every importer/runtime input that can change raw USD semantics."""
    import omni.kit.app

    extension_name = "isaacsim.asset.importer.mjcf"
    manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = manager.get_enabled_extension_id(extension_name)
    if not isinstance(extension_id, str) or not extension_id.startswith(
        extension_name + "-"
    ):
        raise RuntimeError(f"IsaacSim extension {extension_name!r} is unavailable")
    extension_version = extension_id[len(extension_name) + 1 :]
    if not isinstance(extension_version, str) or not extension_version:
        raise RuntimeError(f"IsaacSim extension {extension_name!r} has no readable version")
    return {
        "isaacsim": _required_package_version("isaacsim"),
        "isaaclab": _required_package_version("isaaclab"),
        "isaacsim.asset.importer.mjcf": extension_version,
        "python": platform.python_version(),
    }


def _raw_usd_request(
    content_identity: SceneContentIdentity,
    source: str,
    entity: Any,
    variant: int,
    runtime_versions: dict[str, str],
    *,
    self_collision: bool,
) -> RawUSDArtifactRequest:
    source_digest = file_sha256(Path(source))
    parameters = {
        "entity": entity.name,
        "variant": variant,
        "expanded_source_sha256": source_digest,
        "converter": "MjcfConverter",
        "importer": {
            "fix_base": entity.kind == "articulation" and entity.root_mode == "fixed",
            "import_sites": False,
            "import_inertia_tensor": True,
            "link_density": 0.0,
            "make_instanceable": False,
            # Cache identity must track the actual converter input exactly.
            "self_collision": self_collision,
            "force_usd_conversion": True,
            "usd_file": "artifact.usd",
        },
    }
    identity = derive_scene_artifact_identity(
        content_identity,
        RAW_USD_ARTIFACT_STAGE,
        {"parameters": parameters, "runtime_versions": runtime_versions},
    )
    return RawUSDArtifactRequest(identity, source_digest, parameters, runtime_versions)


def _role_usd_request(
    raw_record: Any,
    entity: Any,
    entry: dict[str, Any],
    variant: int,
    *,
    require_bodies: bool,
) -> RawUSDArtifactRequest:
    """Derive one immutable role artifact from its raw identity and bake inputs."""
    parameters: dict[str, Any] = {
        "entity": entity.name,
        "source_entity": entry["mirror_of"] or entity.name,
        "variant": variant,
        "kind": entity.kind,
        "root_mode": entity.root_mode,
        "collision_enabled": bool(entry["collision_enabled"]),
        "mirror": entry["mirror_of"] is not None,
        "mirror_of": entry["mirror_of"],
        "raw_identity": raw_record.identity,
        "raw_artifact_sha256": raw_artifact_fingerprint(raw_record),
        "runtime_versions": dict(raw_record.runtime_versions),
        "bake": {
            "schema_version": ROLE_USD_CACHE_SCHEMA_VERSION,
            "variant_metadata": True,
            "remove_joints": entity.kind == "rigid",
            "articulation_root": "root-prim" if entity.root_mode == "fixed" else "imported",
            "disable_converter_drives": True,
            "disable_gravity": entity.root_mode == "kinematic"
            or entity.kind == "rigid"
            and entity.root_mode == "fixed",
            "require_native_body_paths": require_bodies,
        },
    }
    identity = sha256(
        _canonical_role_json(
            {
                "schema_version": ROLE_USD_CACHE_SCHEMA_VERSION,
                "stage": ROLE_USD_ARTIFACT_STAGE,
                "raw_identity": raw_record.identity,
                "parameters": parameters,
            }
        ).encode("utf-8")
    ).hexdigest()
    return RawUSDArtifactRequest(
        identity, raw_record.source_digest, parameters, raw_record.runtime_versions
    )


def _canonical_role_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    )


def _prototype_spawn_paths(component: str, variant_count: int) -> list[str]:
    """Return one off-stage prototype path per unique converted variant."""
    return [
        f"/World/unisim_prototypes/{component}/{component}_{index}"
        for index in range(variant_count)
    ]


def _assignment_groups(
    assignment: np.ndarray, source_count: int, env_paths: list[str]
) -> tuple[tuple[str, ...], ...]:
    """Group exact environment destinations by prototype without expanding K sources."""
    if assignment.shape != (len(env_paths),) or assignment.min(initial=0) < 0:
        raise ValueError("assignment and environment path counts or values are invalid")
    if assignment.max(initial=-1) >= source_count:
        raise ValueError("assignment contains an unknown prototype index")
    return tuple(
        tuple(env_paths[int(row)] for row in np.flatnonzero(assignment == variant))
        for variant in range(source_count)
    )


def validate_scene_payload(protocol: Any, payload: dict[str, Any]) -> Any:
    """Reject unsupported combinations before launching Kit or converting assets."""
    layout = protocol.load_scene_layout(payload["scene_layout"])
    SceneContentIdentity.from_dict(payload["scene_content_identity"])
    count = payload["num_envs"]
    contact_force_sensors = _validate_contact_force_sensors(payload, layout)
    protocol.scene_slot_shapes(count, layout, len(contact_force_sensors))
    entries = payload["scene_entities"]
    if [entry["name"] for entry in entries] != [entity.name for entity in layout.entities]:
        raise ValueError("scene entity order differs from the frozen layout")
    for entity, entry in zip(layout.entities, entries):
        if entry["asset_format"] != "mjcf":
            raise NotImplementedError("IsaacSim mapped scene currently requires MJCF sources")
        if entity.kind != entry["kind"] or entity.root_mode != entry["root_mode"]:
            raise ValueError("entity declaration differs from compiled layout")
        self_collision = entry.get("self_collision")
        if not isinstance(self_collision, bool):
            raise TypeError("entity self_collision must be bool")
        if self_collision:
            if entity.kind != "articulation":
                raise NotImplementedError("IsaacSim self-collision requires an articulation")
            if entry["mirror_of"] is not None or not entry["collision_enabled"]:
                raise ValueError("self_collision requires a collision-enabled physical entity")
        sources = entry["sources"]
        if not sources or len(sources) != len(entry["variants"]):
            raise ValueError("entity source and variant record counts differ")
        _validated_assignment(entry, count)
        if entity.kind == "rigid" and len(entity.body_names) != 1:
            raise NotImplementedError("IsaacSim rigid entity requires one physical body")
        if entity.root_mode == "kinematic" and entity.kind != "rigid":
            raise NotImplementedError("IsaacSim kinematic articulation is unsupported")
        if any(joint.kind == "ball" for joint in entity.joints):
            raise NotImplementedError("IsaacSim mapped scene supports scalar joints only")
        if len(set(entity.actuator_joint_names)) != len(entity.actuator_joint_names):
            raise NotImplementedError("IsaacSim requires one actuator per controlled joint")
        for record in entry["variants"]:
            if record["joint_names"] != [joint.name for joint in entity.joints]:
                raise ValueError("variant joint names differ from compiled layout")
            if record["body_names"] != list(entity.body_names):
                raise ValueError("variant body names differ from compiled layout")
            validate_body_sphere_radii(record["body_sphere_radii"], len(entity.body_names))
            if record["geom_names"] != [geom.name for geom in entity.geoms]:
                raise ValueError("variant geom names differ from compiled layout")
            if record["geom_body_names"] != [geom.body_name for geom in entity.geoms]:
                raise ValueError("variant geom body ownership differs from compiled layout")
            for field in ("geom_contype", "geom_conaffinity"):
                values = record[field]
                if (
                    not isinstance(values, list)
                    or len(values) != len(entity.geoms)
                    or any(
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer))
                        for value in values
                    )
                ):
                    raise ValueError("invalid variant " + field)
            friction = np.asarray(record["geom_friction"], dtype=np.float32)
            if (
                friction.shape != (len(entity.geoms), 3)
                or not np.isfinite(friction).all()
                or np.any(friction < 0.0)
            ):
                raise ValueError("invalid variant geom_friction")
            if record["actuator_joint_names"] != list(entity.actuator_joint_names):
                raise ValueError("variant actuator targets differ from compiled layout")
            for field in (
                "dof_stiffness",
                "dof_damping",
                "dof_effort",
                "dof_armature",
                "dof_friction",
                "dof_lower",
                "dof_upper",
            ):
                values = np.asarray(record[field])
                valid = (
                    ~np.isnan(values)
                    if field in ("dof_lower", "dof_upper")
                    else np.isfinite(values)
                )
                if values.shape != (len(entity.joints),) or not valid.all():
                    raise ValueError("invalid variant " + field)
            for index, joint in enumerate(entity.joints):
                if joint.name not in entity.actuator_joint_names and record["dof_stiffness"][index]:
                    raise NotImplementedError("passive joint stiffness requires explicit semantics")
        # One view shares one sim-baked effort limit table, and armature and
        # joint friction start from a single ImplicitActuatorCfg. Drive
        # stiffness/damping may differ across variants: initialization rewrites
        # every environment's assigned variant gains through the per-row PhysX
        # view setters, and reset DR owns later per-env drive updates.
        for record in entry["variants"][1:]:
            for field in ("dof_effort", "dof_armature", "dof_friction"):
                if record[field] != entry["variants"][0][field]:
                    raise NotImplementedError(
                        "IsaacSim entity variants require identical effort, armature and "
                        "joint friction; drive stiffness/damping may differ per variant"
                    )
    for field, shape in (
        ("initial_qpos", (count, layout.nq)),
        ("initial_qvel", (count, layout.nv)),
        ("initial_roots", (count, len(layout.entities), 13)),
    ):
        values = np.asarray(payload[field], dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("invalid " + field)
    if "initial_ctrl" in payload:
        values = np.asarray(payload["initial_ctrl"], dtype=np.float32)
        if values.shape != (count, layout.nu) or not np.isfinite(values).all():
            raise ValueError("invalid initial_ctrl")
    return layout


def _self_collision_entity_names(payload: dict[str, Any]) -> set[str]:
    """Entities whose PhysX contacts include self-contacts by request."""
    return {
        entry["name"]
        for entry in payload.get("scene_entities", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and entry.get("self_collision") is True
    }


def _validate_contact_force_sensors(payload: dict[str, Any], layout: Any) -> list[dict[str, str]]:
    records = payload.get("contact_force_sensors", [])
    if not isinstance(records, list):
        raise ValueError("contact_force_sensors must be a list")
    entities = {entity.name: entity for entity in layout.entities}
    self_colliding = _self_collision_entity_names(payload)
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "name", "source_entity", "source_body", "target_entity", "target_body"
        }:
            raise ValueError("malformed contact force sensor declaration")
        if not all(isinstance(record[key], str) and record[key] for key in record):
            raise ValueError("contact force sensor fields must be non-empty strings")
        for role in ("source", "target"):
            entity = entities.get(record[f"{role}_entity"])
            body = record[f"{role}_body"]
            if entity is None or body not in entity.body_names:
                raise ValueError(
                    f"contact force sensor {record['name']!r} references unknown "
                    f"{role} entity/body: {record[f'{role}_entity']}/{body}"
                )
        if (
            record["source_entity"] == record["target_entity"]
            and record["source_entity"] in self_colliding
        ):
            # The filtered pair reporter cannot isolate self-contacts from the
            # requested per-entity self-collision, so the combination fails
            # closed instead of reporting forces the declaration did not mean.
            raise ValueError(
                f"contact force sensor {record['name']!r} measures a same-entity "
                "pair on a self-collision entity"
            )
        if record["name"] in names:
            raise ValueError("duplicate contact force sensor name: " + record["name"])
        names.append(record["name"])
    return records


def _validate_body_net_contact_entities(payload: dict[str, Any], layout: Any) -> list[str]:
    """Validate the host's per-body net contact force coverage request."""
    names = payload.get("body_net_contact_entities", [])
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("body_net_contact_entities must be a list of non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("body_net_contact_entities must not contain duplicates")
    known = {entity.name for entity in layout.entities}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"body_net_contact_entities references unknown entities: {unknown}")
    self_colliding = sorted(
        name for name in names if name in _self_collision_entity_names(payload)
    )
    if self_colliding:
        # PhysX net contact forces of a self-collision entity would include
        # self-contacts, contradicting the wildcard declaration's semantics.
        raise ValueError(
            f"body_net_contact_entities includes self-collision entities: {self_colliding}"
        )
    return names


def _bake(
    usd_path: str,
    entity: Any,
    entry: dict[str, Any],
    variant: int,
    body_paths: dict[str, str],
    require_bodies: bool = False,
) -> str:
    """Author declared root/role semantics and immutable source identity on USD."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError("converted entity USD has no default prim")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    root.CreateAttribute("unisim:variantIndex", Sdf.ValueTypeNames.Int).Set(variant)
    articulation_roots = []
    rigid_bodies = []
    remove_joints = []
    root_path = str(root.GetPath())
    for prim in Usd.PrimRange(root):
        if entity.kind == "rigid" and prim.IsA(UsdPhysics.Joint):
            remove_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if entity.kind == "rigid":
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            elif prim.GetName() == entity.root_body:
                articulation_roots.append(str(prim.GetPath()))
                # The MJCF importer records the converter self_collision input
                # as physxArticulation:enabledSelfCollisions on the articulation
                # root. Authored <contact><exclude> pairs survive separately as
                # UsdPhysics FilteredPairsAPI relationships.
                flag = PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr()
                if bool(flag.Get()) != bool(entry["self_collision"]):
                    raise RuntimeError(
                        f"entity {entity.name} importer self-collision differs from request"
                    )
            else:
                # Importer also tags its synthetic worldBody, outside this entity.
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim)
            body_name = prim.GetName()
            prim_path = str(prim.GetPath())
            if body_name in body_paths and body_paths[body_name] != prim_path[len(root_path):]:
                raise RuntimeError(f"entity {entity.name} has duplicate rigid body {body_name!r}")
            body_paths[body_name] = prim_path[len(root_path):]
            if entity.kind == "rigid":
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(
                    entity.root_mode != "floating"
                )
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(
                entity.root_mode == "kinematic"
                or entity.kind == "rigid"
                and entity.root_mode == "fixed"
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(
                bool(entry["collision_enabled"])
            )
        # Disable converter-authored drives. IsaacLab owns declared control gains.
        for axis in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                drive = UsdPhysics.DriveAPI(prim, axis)
                drive.CreateStiffnessAttr().Set(0.0)
                drive.CreateDampingAttr().Set(0.0)
    if entity.kind == "rigid" and len(rigid_bodies) != 1:
        raise RuntimeError(f"rigid entity {entity.name} has {len(rigid_bodies)} native bodies")
    for path in remove_joints:
        # Converter prims may be authored in referenced layers. An inactive
        # override suppresses them; RemovePrim would merely reveal the reference.
        stage.GetPrimAtPath(path).SetActive(False)
    missing = [name for name in entity.body_names if name not in body_paths]
    if require_bodies and missing:
        raise RuntimeError(
            f"entity {entity.name} converted rigid-body paths are missing bodies: {missing}"
        )
    _author_native_geometry(
        stage, root_path, body_paths, entity, entry["variants"][variant]
    )
    relative = ""
    if entity.kind == "articulation":
        if len(articulation_roots) != 1:
            raise RuntimeError(
                f"entity {entity.name} has ambiguous articulation roots: {articulation_roots}"
            )
        if entity.root_mode == "fixed":
            # A world joint attached to an API-bearing rigid link remains a
            # floating articulation constrained by a maximal-coordinate joint.
            # PhysX recognizes a fixed-base tree when the root API is on the
            # encompassing prim instead (IsaacLab's fix_root_link convention).
            body_prim = stage.GetPrimAtPath(articulation_roots[0])
            body_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if body_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                body_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            UsdPhysics.ArticulationRootAPI.Apply(root)
            # Re-applying the PhysX API on the root prim drops importer-authored
            # attributes, so the verified self-collision flag is authored again.
            PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr().Set(
                bool(entry["self_collision"])
            )
        else:
            relative = articulation_roots[0][len(str(root.GetPath())) :]
    stage.GetRootLayer().Save()
    return relative


def _body_collision_prims(body_prim: Any) -> list[Any]:
    """Return direct collision leaves without crossing a nested rigid body."""
    from pxr import UsdPhysics

    result = []

    def visit(prim: Any, *, root: bool = False) -> None:
        if not root and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            result.append(prim)
            return
        for child in prim.GetChildren():
            visit(child)

    visit(body_prim, root=True)
    return result


def _native_geometry_columns(
    native_body_names: list[str], body_permutation: np.ndarray, entity: Any
) -> np.ndarray:
    """Map public geometry order to PhysX's flattened native shape order."""
    permutation = np.asarray(body_permutation, dtype=np.int64)
    if permutation.shape != (len(native_body_names),):
        raise RuntimeError("native body permutation does not match the rigid-body view")

    public_positions = {name: index for index, name in enumerate(entity.body_names)}
    native_counts = np.zeros(len(native_body_names), dtype=np.int64)
    for geom in entity.geoms:
        public_position = public_positions.get(geom.body_name)
        if public_position is None:
            raise RuntimeError(
                f"entity {entity.name} geometry references unknown body {geom.body_name!r}"
            )
        native_body = int(permutation[public_position])
        native_counts[native_body] += 1

    native_starts = np.zeros(len(native_body_names), dtype=np.int64)
    if len(native_counts) > 1:
        native_starts[1:] = np.cumsum(native_counts[:-1])
    local_offsets = np.zeros(len(native_body_names), dtype=np.int64)
    columns = np.empty(len(entity.geoms), dtype=np.int64)
    for geom_index, geom in enumerate(entity.geoms):
        native_body = int(permutation[public_positions[geom.body_name]])
        columns[geom_index] = native_starts[native_body] + local_offsets[native_body]
        local_offsets[native_body] += 1
    return columns


def _author_native_geometry(
    stage: Any,
    root_path: str,
    body_paths: dict[str, str],
    entity: Any,
    record: dict[str, Any],
) -> None:
    """Author source-indexed collision identity and effective friction materials."""
    from pxr import Sdf, UsdPhysics, UsdShade

    geom_offset = 0
    for body_name in entity.body_names:
        body_prim = stage.GetPrimAtPath(root_path + body_paths[body_name])
        if not body_prim or not body_prim.IsValid():
            raise RuntimeError(f"entity {entity.name} native body prim is missing: {body_name}")
        expected = [
            index
            for index, owner in enumerate(record["geom_body_names"])
            if owner == body_name
        ]
        collisions = _body_collision_prims(body_prim)
        if len(collisions) != len(expected):
            raise RuntimeError(
                f"entity {entity.name} body {body_name} has "
                f"{len(collisions)} native collision geoms, expected {len(expected)}"
            )
        for geom_index, collision in zip(expected, collisions):
            name = record["geom_names"][geom_index]
            collision.CreateAttribute("unisim:geomName", Sdf.ValueTypeNames.String).Set(name)
            collision.CreateAttribute("unisim:geomIndex", Sdf.ValueTypeNames.Int).Set(
                geom_offset
            )
            sliding_friction = float(record["geom_friction"][geom_index][0])
            material_path = f"{root_path}/Looks/unisim_geom_{geom_offset}"
            material = UsdShade.Material.Define(stage, material_path)
            physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics_material.CreateStaticFrictionAttr().Set(sliding_friction)
            physics_material.CreateDynamicFrictionAttr().Set(sliding_friction)
            collision.CreateRelationship("material:binding:physics").SetTargets(
                [Sdf.Path(material_path)]
            )
            geom_offset += 1
    if geom_offset != len(record["geom_names"]):
        raise RuntimeError(f"entity {entity.name} native geometry record is incomplete")


def _native_geometry_record(
    root: Any, entity: Any, record: dict[str, Any], body_paths: dict[str, str]
) -> dict[str, Any]:
    """Read geometry identity/contact/friction from one actual native subtree."""
    from pxr import UsdPhysics

    root_path = str(root.GetPath())
    names: list[str] = []
    body_names: list[str] = []
    masks: list[list[int]] = []
    friction: list[list[float]] = []
    geom_offset = 0
    for body_name in entity.body_names:
        body_prim = root.GetStage().GetPrimAtPath(root_path + body_paths[body_name])
        if not body_prim or not body_prim.IsValid():
            raise RuntimeError(f"entity {entity.name} native body prim is missing: {body_name}")
        expected = [
            index for index, owner in enumerate(record["geom_body_names"]) if owner == body_name
        ]
        collisions = _body_collision_prims(body_prim)
        if len(collisions) != len(expected):
            raise RuntimeError(
                f"entity {entity.name} body {body_name} native geometry differs from source"
            )
        for geom_index, collision in zip(expected, collisions):
            observed_index = collision.GetAttribute("unisim:geomIndex").Get()
            observed_name = collision.GetAttribute("unisim:geomName").Get()
            if observed_index != geom_offset or observed_name != record["geom_names"][geom_index]:
                raise RuntimeError(f"entity {entity.name} native geometry identity differs")
            enabled = UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
            if not isinstance(enabled, bool):
                raise RuntimeError(f"entity {entity.name} native collision state is missing")
            bindings = collision.GetRelationship("material:binding:physics").GetTargets()
            if len(bindings) != 1:
                raise RuntimeError(f"entity {entity.name} native geometry material is missing")
            material = root.GetStage().GetPrimAtPath(bindings[0])
            if not material or not material.IsValid() or not material.HasAPI(
                UsdPhysics.MaterialAPI
            ):
                raise RuntimeError(f"entity {entity.name} native geometry material is invalid")
            physics_material = UsdPhysics.MaterialAPI(material)
            static_friction = physics_material.GetStaticFrictionAttr().Get()
            dynamic_friction = physics_material.GetDynamicFrictionAttr().Get()
            values = [static_friction, dynamic_friction, 0.0]
            if (
                not all(isinstance(value, (int, float)) for value in values[:2])
                or not np.isfinite(values).all()
                or any(value < 0.0 for value in values)
            ):
                raise RuntimeError(f"entity {entity.name} native geometry friction is invalid")
            names.append(str(observed_name))
            body_names.append(body_name)
            masks.append([int(enabled), int(enabled)])
            friction.append([float(value) for value in values])
            geom_offset += 1
    if names != list(record["geom_names"]) or body_names != list(record["geom_body_names"]):
        raise RuntimeError(f"entity {entity.name} native geometry layout differs from source")
    return {
        "geom_names": names,
        "geom_body_names": body_names,
        "geom_contact_masks": masks,
        "geom_friction": friction,
    }


def _inspect_role(
    usd_path: str,
    entity: Any,
    entry: dict[str, Any],
    variant: int,
    require_bodies: bool = False,
) -> tuple[str, dict[str, str]]:
    """Read and validate an immutable role artifact without authoring edits."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError("role entity USD has no default prim")
    observed_variant = root.GetAttribute("unisim:variantIndex").Get()
    if observed_variant != variant:
        raise RuntimeError("role entity USD variant identity differs from request")

    body_paths: dict[str, str] = {}
    articulation_roots: list[str] = []
    active_joints = 0
    rigid_body_count = 0
    root_path = str(root.GetPath())
    expected_collision = bool(entry["collision_enabled"])
    expected_disable_gravity = (
        entity.root_mode == "kinematic" or entity.kind == "rigid" and entity.root_mode == "fixed"
    )
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint) and prim.IsActive():
            active_joints += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_count += 1
            body_name = prim.GetName()
            prim_path = str(prim.GetPath())
            relative_body_path = prim_path[len(root_path) :]
            if body_name in body_paths and body_paths[body_name] != relative_body_path:
                raise RuntimeError(f"entity {entity.name} has duplicate rigid body {body_name!r}")
            body_paths[body_name] = relative_body_path
            if entity.kind == "rigid":
                kinematic = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
                if kinematic is not (entity.root_mode != "floating"):
                    raise RuntimeError(f"rigid entity {entity.name} has the wrong mobility role")
            disable_gravity = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
            if disable_gravity is not expected_disable_gravity:
                raise RuntimeError(f"entity {entity.name} has the wrong gravity role")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if collision is not expected_collision:
                raise RuntimeError(f"entity {entity.name} has the wrong collision role")
        for axis in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                drive = UsdPhysics.DriveAPI(prim, axis)
                if drive.GetStiffnessAttr().Get() != 0.0 or (drive.GetDampingAttr().Get() != 0.0):
                    raise RuntimeError(f"entity {entity.name} retains a converter drive")

    if entity.kind == "rigid":
        if rigid_body_count != 1:
            raise RuntimeError(f"rigid entity {entity.name} has {rigid_body_count} native bodies")
        if active_joints:
            raise RuntimeError(f"rigid entity {entity.name} retains native joints")
        if articulation_roots:
            raise RuntimeError(f"rigid entity {entity.name} retains an articulation root")
        relative = ""
    else:
        if len(articulation_roots) != 1:
            raise RuntimeError(
                f"entity {entity.name} has ambiguous articulation roots: {articulation_roots}"
            )
        articulation_root = articulation_roots[0]
        if entity.root_mode == "fixed":
            if articulation_root != root_path:
                raise RuntimeError(f"fixed entity {entity.name} has no root-prim articulation")
            relative = ""
        else:
            if stage.GetPrimAtPath(articulation_root).GetName() != entity.root_body:
                raise RuntimeError(
                    f"entity {entity.name} articulation root differs from declaration"
                )
            relative = articulation_root[len(root_path) :]
        flag = PhysxSchema.PhysxArticulationAPI(
            stage.GetPrimAtPath(articulation_root)
        ).GetEnabledSelfCollisionsAttr()
        if bool(flag.Get()) != bool(entry["self_collision"]):
            raise RuntimeError(f"entity {entity.name} has the wrong self-collision role")

    missing = [name for name in entity.body_names if name not in body_paths]
    if require_bodies and missing:
        raise RuntimeError(f"entity {entity.name} role USD is missing bodies: {missing}")
    _native_geometry_record(
        root, entity, entry["variants"][variant], body_paths
    )
    return relative, body_paths


class SceneWorkerContext:
    """One independent native asset view per declared entity."""

    def __init__(self, protocol: Any, renderer: Any) -> None:
        self.protocol = protocol
        self.renderer = renderer
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []
        self.assets: list[Any] = []
        self.maps: list[dict[str, Any]] = []
        self.contact_sensors: list[Any] = []
        self.contact_sensor_maps: list[dict[str, Any]] = []
        self.contact_force_sensors: list[dict[str, str]] = []
        self.net_contact_entities: list[str] = []
        self.net_contact_views: list[Any] = []
        self.net_contact_maps: list[dict[str, Any]] = []
        self._contact_reporting = False
        self.faulted = False
        self.legacy_projection: Any = None
        self._legacy_metadata: dict[str, Any] | None = None
        # Per-entity drive/natural damping bookkeeping in public joint order.
        # PhysX exposes one DOF damping quantity per joint; the implicit
        # position drive's kd and MuJoCo-style natural joint damping compose
        # additively (the drive velocity target is always zero), so the worker
        # tracks both channels and writes their sum.
        self._drive_damping: list[np.ndarray | None] = []
        self._natural_damping: list[np.ndarray | None] = []
        self.physx_solver = PhysxSolverConfig()
        self._raw_usd_cache: RawUSDCache | None = None
        self._raw_usd_cache_persistent = False
        self._raw_usd_cache_reports: list[dict[str, Any]] = []
        self._reported_raw_usd_identities: set[str] = set()
        self._reported_raw_usd_source_digests: set[str] = set()
        self._role_usd_cache: RoleUSDCache | None = None
        self._role_usd_cache_reports: list[dict[str, Any]] = []
        self._temporary = tempfile.TemporaryDirectory(prefix="unisim-isaacsim-scene-")

    @staticmethod
    def _native_visual_sphere_radii(prim: Any) -> list[float]:
        """Read sphere dimensions only from the body's actual visual subtree."""
        from pxr import Usd, UsdGeom

        visuals = prim.GetChild("visuals")
        if not visuals or not visuals.IsValid():
            return []
        result: list[float] = []
        for child in Usd.PrimRange(visuals):
            if not child.IsA(UsdGeom.Sphere):
                continue
            radius = UsdGeom.Sphere(child).GetRadiusAttr().Get()
            if radius is None:
                raise RuntimeError(f"native sphere {child.GetPath()} has no radius")
            value = float(radius)
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"native sphere {child.GetPath()} has an invalid radius")
            result.append(value)
        return result

    def _tensor(self, values: np.ndarray) -> Any:
        return self.torch.as_tensor(
            np.ascontiguousarray(values), dtype=self.torch.float32, device=self.device
        )

    def _cpu_tensor(self, values: np.ndarray) -> Any:
        return self.torch.as_tensor(
            np.ascontiguousarray(values), dtype=self.torch.float32, device="cpu"
        )

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "scene_layout" not in payload:
            metadata = self.renderer.init_sim(payload)
            self.adopt_initialized_context(self.renderer, metadata, payload)
            return cast(dict[str, Any], metadata)
        self.layout = validate_scene_payload(self.protocol, payload)
        self.contact_force_sensors = _validate_contact_force_sensors(payload, self.layout)
        self.net_contact_entities = _validate_body_net_contact_entities(payload, self.layout)
        self._contact_reporting = bool(self.contact_force_sensors) or bool(
            self.net_contact_entities
        )
        raw_usd_cache_dir = payload.get("raw_usd_cache_dir")
        if raw_usd_cache_dir is not None and (
            not isinstance(raw_usd_cache_dir, str) or not raw_usd_cache_dir
        ):
            raise ValueError("raw USD cache directory must be a non-empty string or null")
        role_usd_cache_dir = payload.get("role_usd_cache_dir")
        if role_usd_cache_dir is not None and (
            not isinstance(role_usd_cache_dir, str) or not role_usd_cache_dir
        ):
            raise ValueError("role USD cache directory must be a non-empty string or null")
        self._raw_usd_cache_persistent = raw_usd_cache_dir is not None
        self._raw_usd_cache = RawUSDCache(
            Path(raw_usd_cache_dir)
            if raw_usd_cache_dir is not None
            else Path(self._temporary.name) / "raw-usd"
        )
        if role_usd_cache_dir is not None:
            self._role_usd_cache = RoleUSDCache(Path(role_usd_cache_dir))
        self.entity_components = {
            entity.name: _entity_prim_component(entity.name) for entity in self.layout.entities
        }
        self.num_envs = payload["num_envs"]
        self.entries = payload["scene_entities"]
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError("IsaacSim mapped scene requires CUDA")
        self.device = f"cuda:{device_id}"
        render_mode = payload.get("render_mode", "none")
        if render_mode not in ("none", "record", "interactive"):
            raise ValueError("invalid render_mode")
        os.environ["HEADLESS"] = "0" if render_mode == "interactive" else "1"
        os.environ["ENABLE_CAMERAS"] = "1" if render_mode == "record" else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            {
                "headless": render_mode != "interactive",
                "enable_cameras": render_mode == "record",
                "device": self.device,
                "multi_gpu": False,
            }
        ).app
        self.renderer.simulation_app = app
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.prims as prim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
        from isaacsim.core.cloner import Cloner, GridCloner
        from isaacsim.core.utils.extensions import enable_extension

        self.torch = torch
        enable_extension("isaacsim.asset.importer.mjcf")
        self.gravity = np.asarray(payload["gravity"], dtype=np.float64)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("invalid gravity")
        self.physx_solver = PhysxSolverConfig.from_payload(payload.get("physx_solver"))
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.sim_dt,
                device=self.device,
                gravity=tuple(self.gravity.tolist()),
                physx=build_isaaclab_physx_cfg(sim_utils, self.physx_solver),
            )
        )
        if self._contact_reporting:
            # IsaacLab disables PhysX contact processing by default and only
            # ContactSensor construction re-enables it; the body-net path uses
            # raw PhysX contact views, so enable reporting explicitly.
            import carb

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        cloner = GridCloner(spacing=2.0)
        prototype_cloner = Cloner()
        cloner.define_base_env("/World/envs")
        self.env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        self.entity_paths = {
            name: [path + "/" + component for path in self.env_paths]
            for name, component in self.entity_components.items()
        }
        prim_utils.create_prim(self.env_paths[0], "Xform")
        self.origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_paths[0],
                prim_paths=self.env_paths,
                replicate_physics=False,
                copy_from_source=True,
            ),
            dtype=np.float32,
        )
        self.usd_paths = []
        self.entity_body_paths: list[list[dict[str, str]]] = []
        content_identity = SceneContentIdentity.from_dict(payload["scene_content_identity"])
        raw_usd_runtime_versions = _raw_usd_runtime_versions()
        layout_entities = {entity.name: entity for entity in self.layout.entities}
        layout_entries = {entry["name"]: entry for entry in self.entries}
        for entity, entry in zip(self.layout.entities, self.entries):
            component = self.entity_components[entity.name]
            paths, root_paths = [], []
            body_paths_by_variant: list[dict[str, str]] = []
            for index, source in enumerate(entry["sources"]):
                assert self._raw_usd_cache is not None
                raw_cache = self._raw_usd_cache
                raw_entity = (
                    entity if entry["mirror_of"] is None else layout_entities[entry["mirror_of"]]
                )
                # Raw conversion is role-neutral: a mirror shares its source
                # entity's raw USD, including the source's self-collision flag.
                raw_self_collision = bool(layout_entries[raw_entity.name]["self_collision"])
                raw_request = _raw_usd_request(
                    content_identity,
                    source,
                    raw_entity,
                    index,
                    raw_usd_runtime_versions,
                    self_collision=raw_self_collision,
                )

                def convert_raw_usd(artifact_dir: Path, usd_file_name: str) -> Path:
                    raw_converter = MjcfConverter(
                        MjcfConverterCfg(
                            asset_path=source,
                            fix_base=(
                                raw_entity.kind == "articulation"
                                and raw_entity.root_mode == "fixed"
                            ),
                            import_sites=False,
                            import_inertia_tensor=True,
                            make_instanceable=False,
                            self_collision=raw_self_collision,
                            force_usd_conversion=True,
                            usd_dir=str(artifact_dir),
                            usd_file_name=usd_file_name,
                        )
                    )
                    return Path(raw_converter.usd_path)

                cached_raw = raw_cache.materialize(raw_request, convert_raw_usd)
                if cached_raw.record.identity not in self._reported_raw_usd_identities:
                    self._reported_raw_usd_identities.add(cached_raw.record.identity)
                    self._reported_raw_usd_source_digests.add(cached_raw.record.source_digest)
                    self._raw_usd_cache_reports.append(
                        {
                            "identity": cached_raw.record.identity,
                            "entity": raw_entity.name,
                            "variant": index,
                            "hit": cached_raw.hit,
                            "materialize_ms": cached_raw.materialize_ms,
                            "artifact_files": len(cached_raw.record.files),
                            "artifact_bytes": cached_raw.record.size_bytes,
                        }
                    )

                role_request = _role_usd_request(
                    cached_raw.record,
                    entity,
                    entry,
                    index,
                    require_bodies=self._contact_reporting,
                )
                if self._role_usd_cache is None:
                    role_destination = Path(self._temporary.name) / "roles" / component / str(index)
                    usd_path = raw_cache.copy_artifact(cached_raw.record, role_destination)
                    body_paths: dict[str, str] = {}
                    relative = _bake(
                        str(usd_path),
                        entity,
                        entry,
                        index,
                        body_paths,
                        require_bodies=self._contact_reporting,
                    )
                    self._role_usd_cache_reports.append(
                        {
                            "identity": role_request.identity,
                            "entity": entity.name,
                            "variant": index,
                            "hit": False,
                            "materialize_ms": 0.0,
                            "artifact_files": 0,
                            "artifact_bytes": 0,
                        }
                    )
                else:
                    if raw_cache.load(cached_raw.record.identity) != cached_raw.record:
                        raise RuntimeError("raw USD cache entry changed before role baking")
                    baked_body_paths: dict[str, str] | None = None
                    baked_root_path: str | None = None

                    def bake_role_artifact(artifact_dir: Path, _usd_file_name: str) -> Path:
                        nonlocal baked_body_paths, baked_root_path
                        raw_destination = artifact_dir.parent / "raw"
                        raw_cache.copy_artifact(cached_raw.record, raw_destination)
                        artifact_dir.rmdir()
                        os.replace(raw_destination, artifact_dir)
                        copied_usd = artifact_dir / Path(
                            *PurePosixPath(cached_raw.record.usd_relative_path).parts
                        )
                        body_paths: dict[str, str] = {}
                        baked_root_path = _bake(
                            str(copied_usd),
                            entity,
                            entry,
                            index,
                            body_paths,
                            require_bodies=self._contact_reporting,
                        )
                        baked_body_paths = body_paths
                        return copied_usd

                    cached_role = self._role_usd_cache.materialize(role_request, bake_role_artifact)
                    if cached_role.hit:
                        relative, body_paths = _inspect_role(
                            str(cached_role.record.usd_path),
                            entity,
                            entry,
                            index,
                            require_bodies=self._contact_reporting,
                        )
                    else:
                        assert baked_body_paths is not None and baked_root_path is not None
                        body_paths = baked_body_paths
                        relative = baked_root_path
                    usd_path = cached_role.record.usd_path
                    self._role_usd_cache_reports.append(
                        {
                            "identity": cached_role.record.identity,
                            "entity": entity.name,
                            "variant": index,
                            "hit": cached_role.hit,
                            "materialize_ms": cached_role.materialize_ms,
                            "artifact_files": len(cached_role.record.files),
                            "artifact_bytes": cached_role.record.size_bytes,
                        }
                    )
                paths.append(str(usd_path))
                root_paths.append(relative)
                body_paths_by_variant.append(body_paths)
            if len(set(root_paths)) != 1:
                raise RuntimeError("variant articulation root paths differ")
            if self._contact_reporting:
                canonical_body_paths = body_paths_by_variant[0]
                for body_name in entity.body_names:
                    for variant_body_paths in body_paths_by_variant[1:]:
                        if variant_body_paths.get(body_name) != canonical_body_paths[body_name]:
                            raise RuntimeError(
                                f"entity {entity.name} variant rigid-body prim paths differ"
                            )
            self.entity_body_paths.append(body_paths_by_variant)
            self.usd_paths.append(paths)
            prototype_paths = _prototype_spawn_paths(component, len(paths))
            assignment = _validated_assignment(entry, self.num_envs)
            destination_groups = _assignment_groups(
                assignment, len(paths), self.entity_paths[entity.name]
            )
            prim_path = "/World/envs/env_.*/" + component
            prim_utils.create_prim(
                f"/World/unisim_prototypes/{component}", "Scope"
            )
            for index, (prototype_path, prototype_usd_path, destinations) in enumerate(
                zip(prototype_paths, paths, destination_groups)
            ):
                prototype_cfg = sim_utils.UsdFileCfg(usd_path=prototype_usd_path)
                prototype_cfg.activate_contact_sensors = self._contact_reporting
                prototype_cfg.func(
                    prototype_path,
                    prototype_cfg,
                    translation=tuple(entry["initial_pose"][:3]),
                    orientation=tuple(entry["initial_pose"][3:]),
                )
                if destinations:
                    prototype_cloner.clone(
                        source_prim_path=prototype_path,
                        prim_paths=list(destinations),
                        replicate_physics=False,
                        copy_from_source=True,
                    )
                prototype = prim_utils.get_prim_at_path(prototype_path)
                if not prototype or not prototype.IsValid():
                    raise RuntimeError(f"IsaacSim prototype is missing: {prototype_path}")
                prototype.SetActive(False)
            if entity.kind == "articulation":
                names = [joint.name for joint in entity.joints]
                gains = self.renderer._actuator_dicts(entry["variants"][0], names)
                actuators = {}
                if names:
                    actuators["declared"] = ImplicitActuatorCfg(
                        joint_names_expr=names,
                        stiffness=gains["stiffness"],
                        damping=gains["damping"],
                        effort_limit_sim=gains["effort"],
                        armature=gains["armature"],
                        friction=gains["friction"],
                    )
                asset = Articulation(
                    ArticulationCfg(
                        prim_path=prim_path,
                        articulation_root_prim_path=root_paths[0],
                        init_state=ArticulationCfg.InitialStateCfg(
                            pos=tuple(entry["initial_pose"][:3]),
                            rot=tuple(entry["initial_pose"][3:]),
                        ),
                        spawn=None,
                        actuators=actuators,
                    )
                )
            else:
                asset = RigidObject(
                    RigidObjectCfg(
                        prim_path=prim_path,
                        spawn=None,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=tuple(entry["initial_pose"][:3]),
                            rot=tuple(entry["initial_pose"][3:]),
                        ),
                    )
                )
            self.assets.append(asset)
        entity_indexes = {entity.name: index for index, entity in enumerate(self.layout.entities)}
        for record in self.contact_force_sensors:
            source_index = entity_indexes[record["source_entity"]]
            target_index = entity_indexes[record["target_entity"]]
            source_component = self.entity_components[record["source_entity"]]
            target_component = self.entity_components[record["target_entity"]]
            source_path = (
                "/World/envs/env_.*/" + source_component
                + self.entity_body_paths[source_index][0][record["source_body"]]
            )
            target_path = (
                "/World/envs/env_.*/" + target_component
                + self.entity_body_paths[target_index][0][record["target_body"]]
            )
            self.contact_sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=source_path,
                filter_prim_paths_expr=[target_path],
                history_length=0,
            )))
        self._bind_fixed_anchors()
        if self.num_envs > 1:
            cloner.filter_collisions(
                self.sim.cfg.physics_prim_path, "/World/collisions", self.env_paths
            )
        self._setup_renderer(sim_utils, payload)
        # A requested contact offset is authored on every collision shape
        # after all cold-path spawns and before the first physics step.
        if self.physx_solver.contact_offset is not None:
            apply_contact_offset(self.sim.stage, self.physx_solver.contact_offset)
        self.sim.reset()
        for entity, asset in zip(self.layout.entities, self.assets):
            asset.update(self.sim_dt)
        for sensor, record in zip(self.contact_sensors, self.contact_force_sensors):
            sensor.update(self.sim_dt)
            native_paths = list(sensor.body_physx_view.prim_paths)
            native_envs = _native_environment_order(
                native_paths, self.entity_paths[record["source_entity"]]
            )
            self.contact_sensor_maps.append(
                {"envs": np.argsort(native_envs), "public_for_native": np.asarray(native_envs)}
            )
        layout_entities = {entity.name: entity for entity in self.layout.entities}
        # One batched PhysX contact view per requested entity reports the net
        # contact force on every body against any contact object. Explicit
        # per-body patterns are required because the baked USD nests bodies by
        # kinematic depth, which a single-level sensor leaf pattern cannot
        # match.
        if self.net_contact_entities:
            from isaacsim.core.simulation_manager import SimulationManager

            physics_view = SimulationManager.get_physics_sim_view()
            for entity_name in self.net_contact_entities:
                entity = layout_entities[entity_name]
                component = self.entity_components[entity_name]
                body_paths = self.entity_body_paths[entity_indexes[entity_name]][0]
                patterns = [
                    "/World/envs/env_*/" + component + body_paths[body_name]
                    for body_name in entity.body_names
                ]
                body_view = physics_view.create_rigid_body_view(patterns)
                contact_view = physics_view.create_rigid_contact_view(patterns)
                count = self.num_envs * len(entity.body_names)
                prim_paths = list(body_view.prim_paths)
                if len(prim_paths) != count:
                    raise RuntimeError(
                        f"body-net contact reporter for entity {entity_name!r} resolved "
                        f"{len(prim_paths)} bodies; expected {count}"
                    )
                relative_to_body = {body_paths[name]: index for index, name in enumerate(
                    entity.body_names
                )}
                env_rows = np.empty(count, dtype=np.int64)
                body_columns = np.empty(count, dtype=np.int64)
                seen: set[tuple[int, int]] = set()
                for row, path in enumerate(prim_paths):
                    matches = [
                        env
                        for env, root in enumerate(self.entity_paths[entity_name])
                        if path.startswith(root + "/")
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            "body-net contact reporter contains an unowned or ambiguous "
                            f"instance: {path}"
                        )
                    env = matches[0]
                    # entity_paths already include the entity component scope.
                    relative = path[len(self.entity_paths[entity_name][env]):]
                    body = relative_to_body.get(relative)
                    if body is None or (env, body) in seen:
                        raise RuntimeError(
                            f"body-net contact reporter resolved an unexpected prim: {path}"
                        )
                    seen.add((env, body))
                    env_rows[row] = env
                    body_columns[row] = entity.body_ids[body]
                self.net_contact_views.append(contact_view)
                self.net_contact_maps.append(
                    {
                        "env": env_rows,
                        "body": body_columns,
                        "entity": entity_name,
                        "count": count,
                        # Keep the rigid-body view alive alongside the contact view.
                        "body_view": body_view,
                    }
                )
        for entity, entry, asset in zip(self.layout.entities, self.entries, self.assets):
            native_paths = list(asset.root_physx_view.prim_paths)
            native_envs = _native_environment_order(native_paths, self.entity_paths[entity.name])
            env_map = np.argsort(native_envs)
            native_bodies = list(asset.body_names)
            bodies = self.renderer._build_permutation(
                native_bodies, list(entity.body_names), "body"
            )
            native_joints = list(asset.joint_names) if entity.kind == "articulation" else []
            if entity.kind == "articulation" and (
                bool(asset.is_fixed_base) != (entity.root_mode == "fixed")
            ):
                raise RuntimeError("native articulation root mode differs from declaration")
            joints = self.renderer._build_permutation(
                native_joints, [joint.name for joint in entity.joints], "joint"
            )
            control_joints = np.asarray(
                [native_joints.index(name) for name in entity.actuator_joint_names], dtype=np.int64
            )
            geom_columns = _native_geometry_columns(native_bodies, bodies, entity)
            self.maps.append(
                {
                    "bodies": bodies,
                    "geoms": geom_columns,
                    "joints": joints,
                    "controls": control_joints,
                    "envs": env_map,
                    "public_for_native": np.asarray(native_envs),
                }
            )
            self._apply_variant_drives(entity, entry, asset, self.maps[-1])
        ids = np.arange(self.num_envs, dtype=np.int64)
        self._commit(
            ids,
            np.asarray(payload["initial_qpos"], dtype=np.float32),
            np.asarray(payload["initial_qvel"], dtype=np.float32),
            np.asarray(payload["initial_roots"], dtype=np.float32),
            np.ones(self.layout.nq, dtype=np.uint8),
            np.ones(self.layout.nv, dtype=np.uint8),
            np.ones((len(self.assets), 2), dtype=np.uint8),
            initializing=True,
        )
        if "initial_ctrl" in payload:
            self._set_control_targets(np.asarray(payload["initial_ctrl"], dtype=np.float32))
        self.actual = self._audit_instances()
        return self.get_meta()

    def adopt_initialized_context(
        self, renderer: Any, metadata: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        """Project the old wire onto this runtime after its existing cold importer.

        This adopts live views and the already initialized Kit context. The
        synthetic execution layout preserves the historical D-wide control
        surface; it is never published as a new physical entity capability.
        """
        bridge = self.protocol.load_legacy_projection()
        self.renderer = renderer
        self.num_envs, self.sim_dt = renderer.num_envs, renderer.sim_dt
        self.sim, self.torch, self.device = renderer.sim, renderer.torch, renderer.device
        self.origins = np.asarray(renderer.env_origins, dtype=np.float32).copy()
        if self.origins.shape != (self.num_envs, 3) or not np.isfinite(self.origins).all():
            raise RuntimeError("legacy native environment origins are malformed")
        self.layout = bridge.LegacyExecutionLayout(
            renderer.contract_joint_names,
            renderer.contract_body_names,
            root_body_name=payload.get("root_body_name"),
        )
        self.legacy_projection = bridge.LegacySlotProjection(
            self.protocol, self.num_envs, self.layout
        )
        self.assets = [renderer.robot]
        # Resolve instance rows from the existing view rather than assuming
        # GridCloner creation order equals PhysX view order.
        roots = [path + "/Robot" for path in renderer.env_prim_paths]
        native_envs = _native_environment_order(
            list(renderer.robot.root_physx_view.prim_paths), roots
        )
        joints = np.asarray(renderer.native_joint_for_contract, dtype=np.int64).copy()
        bodies = np.asarray(renderer.native_body_for_contract, dtype=np.int64).copy()
        self.maps = [
            {
                "envs": np.argsort(native_envs),
                "public_for_native": native_envs,
                "joints": joints,
                "bodies": bodies,
                "controls": joints.copy(),
            }
        ]
        self._legacy_metadata = metadata.copy()

    def _setup_renderer(self, sim_utils: Any, payload: dict[str, Any]) -> None:
        owner = self.renderer
        owner.sim, owner.torch, owner.device = self.sim, self.torch, self.device
        owner.num_envs, owner.sim_dt = self.num_envs, self.sim_dt
        owner.robot = self.assets[0]
        owner.render_mode = payload.get("render_mode", "none")
        owner.render_width = payload.get("render_width", 1280)
        owner.render_height = payload.get("render_height", 720)
        if owner.render_mode != "none":
            light = sim_utils.DomeLightCfg(intensity=100.0)
            light.func("/World/UniSimLight", light)
        if owner.render_mode == "record":
            from isaaclab.sensors.camera import Camera, CameraCfg

            owner.camera = Camera(
                CameraCfg(
                    prim_path="/World/envs/env_0/UniSimCamera",
                    update_period=0.0,
                    data_types=["rgb"],
                    width=owner.render_width,
                    height=owner.render_height,
                    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 1e5)),
                )
            )

    def _bind_fixed_anchors(self) -> None:
        """Place imported world joints at the actual cloned root world transform."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        for entity, asset in zip(self.layout.entities, self.assets):
            if entity.kind != "articulation" or entity.root_mode != "fixed":
                continue
            for path in self.entity_paths[entity.name]:
                prim = asset.stage.GetPrimAtPath(path)
                fixed = []
                for child in Usd.PrimRange(prim):
                    if not child.IsA(UsdPhysics.FixedJoint):
                        continue
                    joint = UsdPhysics.FixedJoint(child)
                    body0 = joint.GetBody0Rel().GetTargets()
                    body1 = joint.GetBody1Rel().GetTargets()
                    if not body0 and len(body1) == 1:
                        fixed.append((joint, body1[0]))
                if len(fixed) != 1:
                    raise RuntimeError("fixed entity requires one world anchor joint")
                joint, root_path = fixed[0]
                transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                    asset.stage.GetPrimAtPath(root_path)
                )
                position = transform.ExtractTranslation()
                quaternion = transform.ExtractRotationQuat()
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(quaternion))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))

    def _audit_instances(self) -> list[dict[str, Any]]:
        """Read identity and native properties from actual spawned instances."""
        from pxr import PhysxSchema, Usd, UsdPhysics

        result = []
        for entity_index, (entity, entry, asset, mapping) in enumerate(
            zip(self.layout.entities, self.entries, self.assets, self.maps)
        ):
            paths = self.entity_paths[entity.name]
            native_paths = [asset.root_physx_view.prim_paths[i] for i in mapping["envs"]]
            if not np.array_equal(
                _native_environment_order(native_paths, paths), np.arange(self.num_envs)
            ):
                raise RuntimeError("native view row order differs from environment order")
            observed = []
            for path in paths:
                prim = asset.stage.GetPrimAtPath(path)
                value = prim.GetAttribute("unisim:variantIndex").Get()
                if not isinstance(value, int):
                    raise RuntimeError("spawned asset has no observable variant identity")
                observed.append(value)
                if not entry["collision_enabled"]:
                    for child in Usd.PrimRange(prim):
                        if child.HasAPI(UsdPhysics.CollisionAPI) and (
                            UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
                        ):
                            raise RuntimeError("collision-disabled entity has an enabled collider")
                if entity.kind == "articulation":
                    # The bake guarantees exactly one UsdPhysics articulation
                    # root per spawned clone: the importer's synthetic worldBody
                    # keeps an inert PhysxArticulationAPI after its
                    # ArticulationRootAPI is removed, so the flag must be read
                    # from the actual articulation root prim only. Environment
                    # collision filtering must not have clobbered it.
                    roots = [
                        child
                        for child in Usd.PrimRange(prim)
                        if child.HasAPI(UsdPhysics.ArticulationRootAPI)
                    ]
                    if len(roots) != 1:
                        raise RuntimeError(
                            f"entity {entity.name} has ambiguous native articulation roots"
                        )
                    flag = (
                        PhysxSchema.PhysxArticulationAPI(roots[0])
                        .GetEnabledSelfCollisionsAttr()
                        .Get()
                    )
                    if bool(flag) != bool(entry["self_collision"]):
                        raise RuntimeError(
                            f"entity {entity.name} native self-collision differs from request"
                        )
            if observed != entry["assignment"]:
                raise RuntimeError("actual spawned variant assignment differs from requested")
            masses = _numpy(asset.root_physx_view.get_masses()).reshape(self.num_envs, -1)[
                mapping["envs"]
            ]
            masses = masses[:, mapping["bodies"]]
            expected = np.asarray([entry["variants"][index]["body_mass"] for index in observed])
            if not np.allclose(masses, expected, rtol=2e-4, atol=1e-6):
                raise RuntimeError(
                    f"entity {entity.name} native body masses differ: "
                    f"actual={masses.tolist()}, requested={expected.tolist()}"
                )
            coms = _numpy(asset.root_physx_view.get_coms()).reshape(self.num_envs, -1, 7)[
                mapping["envs"]
            ]
            coms = coms[:, mapping["bodies"]]
            expected_coms = np.asarray(
                [entry["variants"][variant]["body_ipos"] for variant in observed]
            )
            if not np.allclose(coms[:, :, :3], expected_coms, rtol=1e-4, atol=1e-6):
                raise RuntimeError(f"entity {entity.name} native COM differs from source")
            inertias = _numpy(asset.root_physx_view.get_inertias()).reshape(
                self.num_envs, -1, 3, 3
            )[mapping["envs"]][:, mapping["bodies"]]
            expected_inertias = []
            for variant in observed:
                record = entry["variants"][variant]
                matrices = []
                for diagonal, quaternion in zip(record["body_inertia"], record["body_iquat"]):
                    # Columns of R are independently rotated unit basis vectors.
                    rotation = _rotate(np.broadcast_to(quaternion, (3, 4)), np.eye(3)).T
                    matrices.append((rotation * np.asarray(diagonal)) @ rotation.T)
                expected_inertias.append(matrices)
            if not np.allclose(inertias, expected_inertias, rtol=2e-4, atol=1e-6):
                raise RuntimeError(f"entity {entity.name} native inertia differs from source")
            sphere_radii = []
            variant_body_paths = self.entity_body_paths[entity_index][0]
            geometry_rows = []
            for path, variant in zip(paths, observed):
                row: list[list[float]] = []
                for body_name in entity.body_names:
                    body_prim = asset.stage.GetPrimAtPath(path + variant_body_paths[body_name])
                    if not body_prim or not body_prim.IsValid():
                        raise RuntimeError(
                            f"entity {entity.name} native body prim is missing: {body_name}"
                        )
                    row.append(self._native_visual_sphere_radii(body_prim))
                expected_radii = entry["variants"][variant]["body_sphere_radii"]
                if not body_sphere_radii_close(
                    row, expected_radii, rtol=2e-6, atol=1e-8
                ):
                    raise RuntimeError(
                        f"entity {entity.name} native sphere radii differ: "
                        f"actual={row}, requested={expected_radii}"
                    )
                sphere_radii.append(row)
                geometry_rows.append(
                    _native_geometry_record(
                        asset.stage.GetPrimAtPath(path),
                        entity,
                        entry["variants"][variant],
                        variant_body_paths,
                    )
                )
            if entity.joints:
                kinds = _numpy(asset.root_physx_view.get_dof_types())[mapping["envs"]][
                    :, mapping["joints"]
                ]
                expected_kinds = [0 if joint.kind == "hinge" else 1 for joint in entity.joints]
                if not np.all(kinds == expected_kinds):
                    raise RuntimeError(f"entity {entity.name} native joint types differ")
                for field, actual in (
                    ("dof_stiffness", asset.root_physx_view.get_dof_stiffnesses()),
                    ("dof_damping", asset.root_physx_view.get_dof_dampings()),
                ):
                    values = _numpy(actual)[mapping["envs"]][:, mapping["joints"]]
                    expected = np.asarray(
                        [entry["variants"][variant][field] for variant in observed]
                    )
                    if not np.allclose(values, expected, rtol=1e-5, atol=1e-6):
                        raise RuntimeError(f"entity {entity.name} native {field} differs")
                for env in range(self.num_envs):
                    metatype = asset.root_physx_view.get_metatype(int(mapping["envs"][env]))
                    parents = dict(zip(metatype.link_names, metatype.link_parents))
                    for body, parent in zip(entity.body_names, entity.body_parent_names):
                        if parent is not None and parents[body] != parent:
                            raise RuntimeError(f"entity {entity.name} native body topology differs")
            result.append(
                {
                    "name": entity.name,
                    "assignment": observed,
                    "prim_paths": native_paths,
                    "body_mass": masses.tolist(),
                    "body_com": coms[:, :, :3].tolist(),
                    "body_inertia": inertias.tolist(),
                    "body_sphere_radii": sphere_radii,
                    "geom_names": [row["geom_names"] for row in geometry_rows],
                    "geom_body_names": [row["geom_body_names"] for row in geometry_rows],
                    "geom_contact_masks": [
                        row["geom_contact_masks"] for row in geometry_rows
                    ],
                    "geom_friction": [row["geom_friction"] for row in geometry_rows],
                }
            )
        return result

    def attach_slots(self, payload: dict[str, Any]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        shapes = (
            self.protocol.slot_shapes(
                self.num_envs, len(self.layout.entities[0].joints), self.layout.nbody
            )
            if self.legacy_projection is not None
            else self.protocol.scene_slot_shapes(
                self.num_envs, self.layout, len(self.contact_force_sensors)
            )
        )
        self.protocol.validate_slot_specs(payload["slots"], shapes)
        attached = {}
        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            self._shm_handles.append(handle)
            attached[name] = np.ndarray(
                tuple(spec["shape"]), dtype=spec["dtype"], buffer=handle.buf
            )
        self.slots = (
            attached if self.legacy_projection is None else self.legacy_projection.attach(attached)
        )
        self.refresh_state_slots()

    def refresh_state_slots(self) -> None:
        for field in ("qpos", "qvel", "entity_root_state", "body_state", "contact_force"):
            self.slots[field].fill(0)
        if "contact_sensor_force" in self.slots:
            self.slots["contact_sensor_force"].fill(0)
        # Unowned engine-world bodies still have a valid identity orientation.
        self.slots["body_state"][:, :, 3] = 1
        for index, (entity, asset, mapping) in enumerate(
            zip(self.layout.entities, self.assets, self.maps)
        ):
            root = _numpy(asset.data.root_link_state_w)[mapping["envs"]].copy()
            root[:, :3] -= self.origins
            bodies = _numpy(asset.data.body_link_state_w)[mapping["envs"]].copy()
            bodies[:, :, :3] -= self.origins[:, None, :]
            self.slots["entity_root_state"][:, index] = root
            self.slots["body_state"][:, entity.body_ids] = bodies[:, mapping["bodies"]]
            if entity.root_mode == "floating":
                self.slots["qpos"][:, entity.root_qpos_indices] = root[:, :7]
                velocity = root[:, 7:13].copy()
                velocity[:, 3:] = _rotate(root[:, 3:7], velocity[:, 3:], inverse=True)
                self.slots["qvel"][:, entity.root_qvel_indices] = velocity
            if entity.joints:
                pos = _numpy(asset.data.joint_pos)[mapping["envs"]][:, mapping["joints"]]
                vel = _numpy(asset.data.joint_vel)[mapping["envs"]][:, mapping["joints"]]
                self.slots["qpos"][:, [j.qpos_indices[0] for j in entity.joints]] = pos
                self.slots["qvel"][:, [j.qvel_indices[0] for j in entity.joints]] = vel
        if self.legacy_projection is not None:
            self.legacy_projection.publish()

    def _refresh_contact_sensor_forces(self) -> None:
        """Publish final-substep filtered normal forces in world coordinates."""
        if not self.contact_sensors:
            return
        if "contact_sensor_force" not in self.slots:
            raise RuntimeError("contact sensors were initialized without their shared-memory slot")
        for index, (sensor, mapping) in enumerate(
            zip(self.contact_sensors, self.contact_sensor_maps)
        ):
            matrix = _numpy(sensor.data.force_matrix_w)
            if matrix.shape != (self.num_envs, 1, 1, 3):
                raise RuntimeError(
                    f"contact sensor {index} returned shape {matrix.shape}; expected "
                    f"{(self.num_envs, 1, 1, 3)}"
                )
            forces = matrix.reshape(self.num_envs, 3)[mapping["envs"]]
            if not np.isfinite(forces).all():
                raise RuntimeError(f"contact sensor {index} returned non-finite force")
            self.slots["contact_sensor_force"][:, index, :] = forces

    def _poll_body_net_contact_forces(self) -> None:
        """Read the raw PhysX views on every physics substep.

        On the PhysX GPU backend a body's net-contact entry is zeroed only on
        the exact step where its contact is lost; a reader that skips that
        step keeps the last in-contact force indefinitely (IsaacLab issue
        7613).  Polling every substep keeps the device buffer current; the
        getter rewrites one cached on-device tensor, so this stays off the
        host-copy path.
        """
        for view in self.net_contact_views:
            view.get_net_contact_forces(dt=self.sim_dt)

    def _refresh_body_net_contact_forces(self) -> None:
        """Publish final-substep per-body net contact forces in world coordinates."""
        if not self.net_contact_views:
            return
        for view, mapping in zip(self.net_contact_views, self.net_contact_maps):
            net = _numpy(view.get_net_contact_forces(dt=self.sim_dt))
            expected = (mapping["count"], 3)
            if net.shape != expected:
                raise RuntimeError(
                    f"body-net contact reporter for entity {mapping['entity']!r} returned "
                    f"shape {net.shape}; expected {expected}"
                )
            if not np.isfinite(net).all():
                raise RuntimeError(
                    f"body-net contact reporter for entity {mapping['entity']!r} returned "
                    "non-finite force"
                )
            self.slots["contact_force"][mapping["env"], mapping["body"]] = net

    def set_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Translate only the old wire; native reset always uses reset_entities."""
        if self.legacy_projection is None:
            raise NotImplementedError("mapped scenes require RESET_ENTITIES")
        count = payload["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.num_envs:
            raise ValueError("invalid reset count")
        if count == 0:
            return {"timing": {}}
        return self.reset_entities(self.legacy_projection.prepare_reset(count))

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = payload["nsteps"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("nsteps must be a positive integer")
        if not np.isfinite(self.slots["ctrl"]).all():
            raise ValueError("control must be finite")
        wrench = None
        if "body_wrench" in payload:
            expected = (self.num_envs, self.layout.nbody, 6)
            encoded = payload["body_wrench"]
            expected_nbytes = int(np.prod(expected, dtype=np.int64)) * np.dtype(np.float32).itemsize
            if not isinstance(encoded, bytes) or len(encoded) != expected_nbytes:
                raise ValueError("body wrench payload must be C-order float32 bytes")
            wrench = np.frombuffer(encoded, dtype=np.float32).reshape(expected)
            if wrench.shape != expected:
                raise ValueError(f"body wrench must have shape {expected}, got {wrench.shape}")
            if not np.isfinite(wrench).all():
                raise ValueError("body wrench contains NaN or Inf")
        started = time.perf_counter()
        try:
            self._set_control_targets(self.slots["ctrl"])
            if wrench is not None:
                self._stage_body_wrench(wrench)
            for _ in range(count):
                for asset in self.assets:
                    asset.write_data_to_sim()
                self.sim.step(render=False)
                for asset in self.assets:
                    asset.update(self.sim_dt)
                for sensor in self.contact_sensors:
                    sensor.update(self.sim_dt)
                self._poll_body_net_contact_forces()
            if wrench is not None:
                self._clear_body_wrench()
            self.refresh_state_slots()
            self._refresh_contact_sensor_forces()
            self._refresh_body_net_contact_forces()
        except Exception:
            self.faulted = True
            if wrench is not None:
                try:
                    self._clear_body_wrench()
                except Exception:
                    pass
            raise
        return {"timing": {"physics_ms": (time.perf_counter() - started) * 1000}}

    def _stage_body_wrench(self, wrench: np.ndarray) -> None:
        """Set one public world-frame wrench table on the native asset views."""
        for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
            native_values = wrench[mapping["public_for_native"]][:, entity.body_ids, :]
            force = self._tensor(native_values[..., 0:3])
            torque = self._tensor(native_values[..., 3:6])
            asset.set_external_force_and_torque(
                force,
                torque,
                body_ids=mapping["bodies"].tolist(),
                is_global=True,
            )

    def _clear_body_wrench(self) -> None:
        """Disable external-wrench buffers after a completed or failed step."""
        for asset in self.assets:
            zero = self.torch.zeros(
                (self.num_envs, asset.num_bodies, 3),
                dtype=self.torch.float32,
                device=self.device,
            )
            asset.set_external_force_and_torque(zero, zero, is_global=True)

    def _set_control_targets(self, control: np.ndarray) -> None:
        """Apply actuator columns; keyframe controls are independent of joint positions."""
        for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
            if entity.actuator_indices:
                asset.set_joint_position_target(
                    self._tensor(control[mapping["public_for_native"]][:, entity.actuator_indices]),
                    joint_ids=mapping["controls"].tolist(),
                )

    def _apply_variant_drives(
        self, entity: Any, entry: dict[str, Any], asset: Any, mapping: dict[str, Any]
    ) -> None:
        """Write every environment's assigned variant drive gains into PhysX.

        ``ImplicitActuatorCfg`` seeds one shared gain table at construction;
        per-variant stiffness/damping then land through the per-row view
        setters so the post-init audit can compare native values against each
        environment's assigned variant.  The tracked drive/natural damping
        split is the composition baseline for reset-time kp/kd/dof_damping DR.
        """
        if entity.kind != "articulation" or not entity.joints:
            self._drive_damping.append(None)
            self._natural_damping.append(None)
            return
        assignment = _validated_assignment(entry, self.num_envs)
        stiffness = np.asarray(
            [entry["variants"][int(variant)]["dof_stiffness"] for variant in assignment],
            dtype=np.float32,
        )
        damping = np.asarray(
            [entry["variants"][int(variant)]["dof_damping"] for variant in assignment],
            dtype=np.float32,
        )
        native_ids = self.torch.as_tensor(
            mapping["envs"], dtype=self.torch.long, device=self.device
        )
        joint_ids = mapping["joints"].tolist()
        asset.write_joint_stiffness_to_sim(
            self._tensor(stiffness), joint_ids=joint_ids, env_ids=native_ids
        )
        asset.write_joint_damping_to_sim(
            self._tensor(damping), joint_ids=joint_ids, env_ids=native_ids
        )
        self._drive_damping.append(damping)
        self._natural_damping.append(np.zeros_like(damping))

    def _commit(
        self,
        ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        roots: np.ndarray,
        pmask: np.ndarray,
        vmask: np.ndarray,
        rmask: np.ndarray,
        *,
        initializing: bool = False,
    ) -> None:
        try:
            for index, (entity, asset, mapping) in enumerate(
                zip(self.layout.entities, self.assets, self.maps)
            ):
                touched = bool(rmask[index].any())
                pcols = [j.qpos_indices[0] for j in entity.joints]
                vcols = [j.qvel_indices[0] for j in entity.joints]
                selected = np.flatnonzero(pmask[pcols] | vmask[vcols])
                if not touched and not selected.size:
                    continue
                native_rows = mapping["envs"][ids]
                native_ids = self.torch.as_tensor(
                    native_rows, dtype=self.torch.long, device=self.device
                )
                if rmask[index, 0] and (
                    entity.root_mode != "fixed" or initializing and entity.kind == "rigid"
                ):
                    pose = roots[:, index, :7].copy()
                    pose[:, :3] += self.origins[ids]
                    asset.write_root_pose_to_sim(self._tensor(pose), env_ids=native_ids)
                if rmask[index, 1] and entity.root_mode == "floating":
                    asset.write_root_link_velocity_to_sim(
                        self._tensor(roots[:, index, 7:]), env_ids=native_ids
                    )
                if entity.joints:
                    if selected.size:
                        touched = True
                        joint_ids = mapping["joints"][selected].tolist()
                        native_joints = self.torch.as_tensor(
                            joint_ids, dtype=self.torch.long, device=self.device
                        )
                        # Gather on-device before crossing the CPU boundary;
                        # a sparse reset must not download every environment.
                        positions = _numpy(
                            asset.data.joint_pos[native_ids[:, None], native_joints]
                        ).copy()
                        velocities = _numpy(
                            asset.data.joint_vel[native_ids[:, None], native_joints]
                        ).copy()
                        for column, public_index in enumerate(selected):
                            if pmask[pcols[public_index]]:
                                positions[:, column] = qpos[:, pcols[public_index]]
                            if vmask[vcols[public_index]]:
                                velocities[:, column] = qvel[:, vcols[public_index]]
                        asset.write_joint_state_to_sim(
                            self._tensor(positions),
                            self._tensor(velocities),
                            joint_ids=joint_ids,
                            env_ids=native_ids,
                        )
                if touched:
                    asset.reset(native_ids)
                    asset.update(self.sim_dt)
        except Exception:
            self.faulted = True
            raise

    def _validated_reset_randomization(
        self, payload: dict[str, Any], count: int
    ) -> dict[str, np.ndarray] | None:
        if "randomization" not in payload:
            return None
        raw = payload["randomization"]
        allowed = {
            "geom_friction",
            "body_mass",
            "body_ipos",
            "body_inertia",
            "kp",
            "kd",
            "dof_damping",
            "dof_armature",
            "dof_frictionloss",
        }
        if not isinstance(raw, dict) or not set(raw) <= allowed:
            raise ValueError("randomization must contain only supported property terms")

        def wire_float_table(value: object) -> np.ndarray:
            if not isinstance(value, (list, np.ndarray)):
                raise ValueError("randomization property tables must be lists or arrays")
            try:
                return np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("randomization property tables must be numeric") from exc

        owned_bodies = np.sort(
            np.concatenate(
                [
                    np.asarray(list(entity.body_ids), dtype=np.int64)
                    for entity in self.layout.entities
                ]
            )
        )

        def body_table(term: str, width: int | None, *, positive: bool) -> np.ndarray:
            values = wire_float_table(raw[term])
            expected = (
                (count, self.layout.nbody)
                if width is None
                else (count, self.layout.nbody, width)
            )
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"randomization {term} must be finite with shape {expected}")
            if positive:
                # Unowned public rows (for example the world body) carry zero
                # canonical placeholders and are never written.
                checked = (
                    values[:, owned_bodies] if width is None else values[:, owned_bodies, :]
                )
                if np.any(checked <= 0.0):
                    raise ValueError(f"randomization {term} must be strictly positive")
            return np.asarray(values, dtype=np.float32).copy()

        result: dict[str, np.ndarray] = {}
        if "geom_friction" in raw:
            values = wire_float_table(raw["geom_friction"])
            expected = (count, self.layout.ngeom, 3)
            if (
                values.shape != expected
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                raise ValueError(
                    "randomization geom_friction must be finite nonnegative (count, ngeom, 3)"
                )
            friction = np.asarray(values, dtype=np.float32).copy()
            if (
                np.any(friction[..., 0] != friction[..., 1])
                or np.any(friction[..., 2] != 0.0)
            ):
                raise ValueError(
                    "randomization geom_friction requires equal static/dynamic and zero torsion"
                )
            result["geom_friction"] = friction
        if "body_mass" in raw:
            result["body_mass"] = body_table("body_mass", None, positive=True)
        if "body_ipos" in raw:
            result["body_ipos"] = body_table("body_ipos", 3, positive=False)
        if "body_inertia" in raw:
            result["body_inertia"] = body_table("body_inertia", 3, positive=True)
        for term in ("kp", "kd"):
            if term in raw:
                values = wire_float_table(raw[term])
                gain_shape = (count, self.layout.nu)
                if (
                    values.shape != gain_shape
                    or not np.isfinite(values).all()
                    or np.any(values < 0.0)
                ):
                    raise ValueError(
                        f"randomization {term} must be finite nonnegative with shape "
                        f"{gain_shape}"
                    )
                result[term] = np.asarray(values, dtype=np.float32).copy()
        root_columns = [
            column for entity in self.layout.entities for column in entity.root_qvel_indices
        ]
        for term in ("dof_damping", "dof_armature", "dof_frictionloss"):
            if term in raw:
                values = wire_float_table(raw[term])
                dof_shape = (count, self.layout.nv)
                if (
                    values.shape != dof_shape
                    or not np.isfinite(values).all()
                    or np.any(values < 0.0)
                ):
                    raise ValueError(
                        f"randomization {term} must be finite nonnegative with shape {dof_shape}"
                    )
                if root_columns and np.any(values[:, root_columns] != 0.0):
                    raise ValueError(
                        f"randomization {term} free-root columns must remain zero; "
                        "PhysX exposes no root DOF damping/armature/friction"
                    )
                result[term] = np.asarray(values, dtype=np.float32).copy()
        return result or None

    def _native_mass_rows(self, asset: Any, mapping: dict[str, Any]) -> np.ndarray:
        masses = _numpy(asset.root_physx_view.get_masses()).reshape(self.num_envs, -1).copy()
        if masses.shape[1] < len(mapping["bodies"]):
            raise RuntimeError("native rigid-body view is narrower than the frozen entity map")
        return masses

    def _native_com_rows(self, asset: Any, mapping: dict[str, Any]) -> np.ndarray:
        coms = _numpy(asset.root_physx_view.get_coms()).reshape(self.num_envs, -1, 7).copy()
        if coms.shape[1] < len(mapping["bodies"]):
            raise RuntimeError("native rigid-body COM view is narrower than the frozen entity map")
        return coms

    def _native_inertia_rows(self, asset: Any, mapping: dict[str, Any]) -> np.ndarray:
        inertias = (
            _numpy(asset.root_physx_view.get_inertias()).reshape(self.num_envs, -1, 9).copy()
        )
        if inertias.shape[1] < len(mapping["bodies"]):
            raise RuntimeError(
                "native rigid-body inertia view is narrower than the frozen entity map"
            )
        return inertias

    def _native_material_rows(
        self, asset: Any, mapping: dict[str, Any], geom_count: int
    ) -> np.ndarray:
        if not geom_count:
            return np.empty((self.num_envs, 0, 3), dtype=np.float32)
        materials = (
            _numpy(asset.root_physx_view.get_material_properties())
            .reshape(self.num_envs, -1, 3)
            .copy()
        )
        if materials.shape[1] != geom_count:
            raise RuntimeError("native material view does not match the frozen geometry map")
        return materials[mapping["envs"]][:, mapping["geoms"]]

    @staticmethod
    def _native_dof_friction(asset: Any) -> Any:
        view = asset.root_physx_view
        properties = getattr(view, "get_dof_friction_properties", None)
        if properties is None:
            return view.get_dof_friction_coefficients()
        # Isaac Sim >= 5.0 writes joint friction through [static, dynamic,
        # viscous] property triplets; reset DR only composes the static term.
        return properties()[..., 0]

    def _apply_reset_randomization(
        self, ids: np.ndarray, randomization: dict[str, np.ndarray]
    ) -> None:
        """Write selected public property rows through public PhysX views."""
        geom_offset = 0
        for index, (entity, asset, mapping) in enumerate(
            zip(self.layout.entities, self.assets, self.maps)
        ):
            native_rows = mapping["envs"][ids]
            native_ids = self.torch.as_tensor(
                native_rows, dtype=self.torch.int32, device="cpu"
            )
            if "body_mass" in randomization:
                masses = self._native_mass_rows(asset, mapping)
                masses[native_rows[:, None], mapping["bodies"]] = randomization["body_mass"][
                    :, list(entity.body_ids)
                ]
                asset.root_physx_view.set_masses(self._cpu_tensor(masses), indices=native_ids)
            if "body_ipos" in randomization:
                coms = self._native_com_rows(asset, mapping)
                coms[native_rows[:, None], mapping["bodies"], 0:3] = randomization["body_ipos"][
                    :, list(entity.body_ids), :
                ]
                asset.root_physx_view.set_coms(self._cpu_tensor(coms), indices=native_ids)
            if "body_inertia" in randomization:
                blocks = self._native_inertia_rows(asset, mapping)
                for row, env in enumerate(ids):
                    matrices = self._expected_inertia_blocks(
                        index, int(env), randomization["body_inertia"][row]
                    )
                    blocks[int(native_rows[row]), mapping["bodies"], :] = matrices.reshape(-1, 9)
                asset.root_physx_view.set_inertias(self._cpu_tensor(blocks), indices=native_ids)
            count = len(entity.geoms)
            if "geom_friction" in randomization and count:
                materials = (
                    _numpy(asset.root_physx_view.get_material_properties())
                    .reshape(self.num_envs, -1, 3)
                    .copy()
                )
                columns = mapping["geoms"]
                materials[native_rows[:, None], columns] = randomization["geom_friction"][
                    :, geom_offset : geom_offset + count
                ]
                asset.root_physx_view.set_material_properties(
                    self._cpu_tensor(materials), indices=native_ids
                )
            self._apply_dof_randomization(entity, asset, mapping, index, ids, randomization)
            geom_offset += len(entity.geoms)

    def _expected_inertia_blocks(
        self, entity_index: int, env: int, public_diagonals: np.ndarray
    ) -> np.ndarray:
        """Compose symmetric COM-frame inertia tensors for one environment.

        Public ``body_inertia`` rows are diagonal and carry no orientation; the
        principal-axes rotation stays the environment's immutable assigned
        variant ``body_iquat`` (reset-time ``body_iquat`` DR fails closed).
        """
        entity = self.layout.entities[entity_index]
        entry = self.entries[entity_index]
        assignment = _validated_assignment(entry, self.num_envs)
        record = entry["variants"][int(assignment[env])]
        diagonals = np.asarray(public_diagonals, dtype=np.float64)[list(entity.body_ids)]
        quaternions = np.asarray(record["body_iquat"], dtype=np.float64)
        matrices = []
        for diagonal, quaternion in zip(diagonals, quaternions):
            # Columns of R are independently rotated unit basis vectors.
            rotation = _rotate(np.broadcast_to(quaternion, (3, 4)), np.eye(3)).T
            matrices.append((rotation * diagonal) @ rotation.T)
        return np.asarray(matrices, dtype=np.float32)

    def _apply_dof_randomization(
        self,
        entity: Any,
        asset: Any,
        mapping: dict[str, Any],
        index: int,
        ids: np.ndarray,
        randomization: dict[str, np.ndarray],
    ) -> None:
        """Write drive/natural joint parameters for one articulation entity."""
        if entity.kind != "articulation" or not entity.joints:
            return
        joint_columns = [joint.qvel_indices[0] for joint in entity.joints]
        actuator_positions = [
            position
            for position, joint in enumerate(entity.joints)
            if joint.name in entity.actuator_joint_names
        ]
        native_ids = self.torch.as_tensor(
            mapping["envs"][ids], dtype=self.torch.long, device=self.device
        )
        kp = randomization.get("kp")
        if kp is not None and entity.actuator_indices:
            asset.write_joint_stiffness_to_sim(
                self._tensor(kp[:, list(entity.actuator_indices)]),
                joint_ids=mapping["controls"].tolist(),
                env_ids=native_ids,
            )
        kd = randomization.get("kd")
        dof_damping = randomization.get("dof_damping")
        if kd is not None or dof_damping is not None:
            drive = self._drive_damping[index]
            natural = self._natural_damping[index]
            assert drive is not None and natural is not None
            if kd is not None and actuator_positions:
                drive[ids[:, None], actuator_positions] = kd[:, list(entity.actuator_indices)]
            if dof_damping is not None:
                natural[ids] = dof_damping[:, joint_columns]
            asset.write_joint_damping_to_sim(
                self._tensor(drive[ids] + natural[ids]),
                joint_ids=mapping["joints"].tolist(),
                env_ids=native_ids,
            )
        armature = randomization.get("dof_armature")
        if armature is not None:
            asset.write_joint_armature_to_sim(
                self._tensor(armature[:, joint_columns]),
                joint_ids=mapping["joints"].tolist(),
                env_ids=native_ids,
            )
        friction = randomization.get("dof_frictionloss")
        if friction is not None:
            asset.write_joint_friction_coefficient_to_sim(
                self._tensor(friction[:, joint_columns]),
                joint_ids=mapping["joints"].tolist(),
                env_ids=native_ids,
            )

    def _readback_reset_properties(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
            masses = self._native_mass_rows(asset, mapping)[mapping["envs"]][:, mapping["bodies"]]
            coms = self._native_com_rows(asset, mapping)[mapping["envs"]][:, mapping["bodies"]]
            inertias = self._native_inertia_rows(asset, mapping).reshape(
                self.num_envs, -1, 3, 3
            )[mapping["envs"]][:, mapping["bodies"]]
            friction = self._native_material_rows(asset, mapping, len(entity.geoms))
            if (
                not np.isfinite(masses).all()
                or not np.isfinite(coms).all()
                or not np.isfinite(inertias).all()
                or not np.isfinite(friction).all()
                or np.any(friction < 0.0)
            ):
                raise RuntimeError(f"entity {entity.name} native property readback is invalid")
            record: dict[str, Any] = {
                "name": entity.name,
                "body_mass": masses.tolist(),
                "body_com": coms[:, :, :3].tolist(),
                "body_inertia": inertias.tolist(),
                "geom_friction": friction.tolist(),
            }
            if entity.kind == "articulation" and entity.joints:
                for field, native in (
                    ("dof_stiffness", asset.root_physx_view.get_dof_stiffnesses()),
                    ("dof_damping", asset.root_physx_view.get_dof_dampings()),
                    ("dof_armature", asset.root_physx_view.get_dof_armatures()),
                    ("dof_friction", self._native_dof_friction(asset)),
                ):
                    table = _numpy(native)[mapping["envs"]][:, mapping["joints"]]
                    if not np.isfinite(table).all():
                        raise RuntimeError(
                            f"entity {entity.name} native {field} readback is invalid"
                        )
                    record[field] = table.tolist()
            records.append(record)
        return records

    def _verify_reset_property_readback(
        self,
        ids: np.ndarray,
        randomization: dict[str, np.ndarray],
        before: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> None:
        selected = np.zeros(self.num_envs, dtype=bool)
        selected[ids] = True

        def check(
            entity: Any,
            field: str,
            previous: dict[str, Any],
            record: dict[str, Any],
            expected: np.ndarray,
        ) -> None:
            actual = np.asarray(record[field], dtype=np.float32).reshape(
                self.num_envs, *expected.shape[1:]
            )
            if not np.allclose(actual[ids], expected, rtol=2e-5, atol=1e-6):
                raise RuntimeError(
                    f"entity {entity.name} native {field} readback differs from reset: "
                    f"expected {expected.tolist()}, got {actual[ids].tolist()}"
                )
            untouched = np.asarray(previous[field], dtype=np.float32).reshape(actual.shape)
            if not np.allclose(actual[~selected], untouched[~selected], rtol=2e-5, atol=1e-6):
                raise RuntimeError(
                    f"entity {entity.name} native {field} write leaked outside selected rows"
                )

        geom_offset = 0
        for index, (entity, previous, record) in enumerate(
            zip(self.layout.entities, before, records)
        ):
            count = len(entity.geoms)
            if "geom_friction" in randomization:
                check(
                    entity,
                    "geom_friction",
                    previous,
                    record,
                    randomization["geom_friction"][:, geom_offset : geom_offset + count, :],
                )
            geom_offset += count
            if "body_mass" in randomization:
                check(
                    entity,
                    "body_mass",
                    previous,
                    record,
                    randomization["body_mass"][:, list(entity.body_ids)],
                )
            if "body_ipos" in randomization:
                check(
                    entity,
                    "body_com",
                    previous,
                    record,
                    randomization["body_ipos"][:, list(entity.body_ids), :],
                )
            if "body_inertia" in randomization:
                expected = np.asarray(
                    [
                        self._expected_inertia_blocks(index, int(env), row)
                        for env, row in zip(ids, randomization["body_inertia"])
                    ]
                )
                check(entity, "body_inertia", previous, record, expected)
            self._verify_dof_readback(
                entity, index, ids, randomization, previous, record, check
            )

    def _verify_dof_readback(
        self,
        entity: Any,
        index: int,
        ids: np.ndarray,
        randomization: dict[str, np.ndarray],
        previous: dict[str, Any],
        record: dict[str, Any],
        check: Any,
    ) -> None:
        if entity.kind != "articulation" or not entity.joints:
            return
        joint_columns = [joint.qvel_indices[0] for joint in entity.joints]
        actuator_positions = [
            position
            for position, joint in enumerate(entity.joints)
            if joint.name in entity.actuator_joint_names
        ]
        kp = randomization.get("kp")
        if kp is not None and actuator_positions:
            expected = np.asarray(previous["dof_stiffness"], dtype=np.float32)[ids].copy()
            expected[:, actuator_positions] = kp[:, list(entity.actuator_indices)]
            check(entity, "dof_stiffness", previous, record, expected)
        if "kd" in randomization or "dof_damping" in randomization:
            drive = self._drive_damping[index]
            natural = self._natural_damping[index]
            assert drive is not None and natural is not None
            check(entity, "dof_damping", previous, record, drive[ids] + natural[ids])
        if "dof_armature" in randomization:
            check(
                entity,
                "dof_armature",
                previous,
                record,
                randomization["dof_armature"][:, joint_columns],
            )
        if "dof_frictionloss" in randomization:
            check(
                entity,
                "dof_friction",
                previous,
                record,
                randomization["dof_frictionloss"][:, joint_columns],
            )

    def reset_entities(self, payload: dict[str, Any]) -> dict[str, Any]:
        count = payload["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= self.num_envs:
            raise ValueError("invalid reset count")
        ids = self.slots["reset_env_ids"][:count].astype(np.int64, copy=True)
        if np.unique(ids).size != count or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("invalid reset environment IDs")
        names = payload["entity_names"]
        if not isinstance(names, list) or len(names) != len(set(names)):
            raise ValueError("entity_names must be unique names")
        if not set(names).issubset({entity.name for entity in self.layout.entities}):
            raise ValueError("unknown reset entity")
        qpos = self.slots["reset_qpos"][:count].copy()
        qvel = self.slots["reset_qvel"][:count].copy()
        roots = self.slots["reset_entity_root_state"][:count].copy()
        pmask = self.slots["reset_qpos_mask"].copy()
        vmask = self.slots["reset_qvel_mask"].copy()
        rmask = self.slots["reset_root_mask"].copy()
        if any(not np.isfinite(values).all() for values in (qpos, qvel, roots)):
            raise ValueError("reset values must be finite")
        if any(np.any((mask != 0) & (mask != 1)) for mask in (pmask, vmask, rmask)):
            raise ValueError("reset masks must contain zero or one")
        control_values = None
        control_columns = tuple(
            column
            for entity in self.layout.entities
            if entity.name in names
            for column in entity.actuator_indices
        )
        if "control_values" in payload:
            raw_control = np.asarray(payload["control_values"])
            if (
                raw_control.shape != (count, self.layout.nu)
                or raw_control.dtype.kind not in "fiu"
                or not np.isfinite(raw_control).all()
                or np.any(np.abs(raw_control.astype(np.float64)) > np.finfo(np.float32).max)
            ):
                raise ValueError("control_values must be finite (count, nu) float32 values")
            control_values = raw_control.astype(np.float32, copy=True)
            unselected = [
                column for column in range(self.layout.nu) if column not in control_columns
            ]
            if not np.array_equal(
                control_values[:, unselected], self.slots["ctrl"][ids][:, unselected]
            ):
                raise ValueError("control_values changes controls of an unselected entity")
        for index, entity in enumerate(self.layout.entities):
            selected = bool(
                pmask[list(entity.qpos_indices)].any()
                or vmask[list(entity.qvel_indices)].any()
                or rmask[index].any()
            )
            if selected and entity.name not in names:
                raise ValueError("reset mask writes undeclared entity")
            if entity.root_mode == "fixed" and rmask[index].any():
                raise ValueError("fixed root cannot be reset")
            if entity.root_mode == "kinematic" and rmask[index, 1]:
                raise ValueError("kinematic root velocity cannot be reset")
            if rmask[index, 0] and not np.allclose(
                np.linalg.norm(roots[:, index, 3:7], axis=1), 1, rtol=0, atol=1e-5
            ):
                raise ValueError("root reset quaternion must be unit wxyz")
            if entity.root_mode == "floating":
                for cols, mask, channel in (
                    (entity.root_qpos_indices, pmask, 0),
                    (entity.root_qvel_indices, vmask, 1),
                ):
                    bits = mask[list(cols)]
                    if np.any(bits) != bool(rmask[index, channel]) or len(set(bits.tolist())) > 1:
                        raise ValueError("root generalized and entity masks disagree")
                if rmask[index, 0] and not np.allclose(
                    qpos[:, entity.root_qpos_indices], roots[:, index, :7], rtol=1e-5, atol=1e-6
                ):
                    raise ValueError("root pose differs between generalized and entity values")
                if rmask[index, 1]:
                    velocity = roots[:, index, 7:].copy()
                    velocity[:, 3:] = _rotate(roots[:, index, 3:7], velocity[:, 3:], inverse=True)
                    if not np.allclose(
                        qvel[:, entity.root_qvel_indices], velocity, rtol=1e-5, atol=1e-6
                    ):
                        raise ValueError("generalized and entity root velocities differ")
        randomization = self._validated_reset_randomization(payload, count)
        self._commit(ids, qpos, qvel, roots, pmask, vmask, rmask)
        try:
            property_records = None
            if randomization is not None:
                before_properties = self._readback_reset_properties()
                self._apply_reset_randomization(ids, randomization)
                property_records = self._readback_reset_properties()
                self._verify_reset_property_readback(
                    ids, randomization, before_properties, property_records
                )
            if control_values is not None:
                for entity, asset, mapping in zip(self.layout.entities, self.assets, self.maps):
                    if entity.name in names and entity.actuator_indices:
                        native_ids = self.torch.as_tensor(
                            mapping["envs"][ids], dtype=self.torch.long, device=self.device
                        )
                        asset.set_joint_position_target(
                            self._tensor(control_values[:, entity.actuator_indices]),
                            joint_ids=mapping["controls"].tolist(),
                            env_ids=native_ids,
                        )
                self.slots["ctrl"][np.ix_(ids, control_columns)] = control_values[
                    :, control_columns
                ]
            self.refresh_state_slots()
        except Exception:
            self.faulted = True
            raise
        if randomization is not None:
            assert property_records is not None
            for entity, record in zip(self.layout.entities, property_records):
                current = self.actual[entity.name] if isinstance(self.actual, dict) else next(
                    item for item in self.actual if item["name"] == entity.name
                )
                for field in (
                    "body_mass",
                    "body_com",
                    "body_inertia",
                    "geom_friction",
                ):
                    if field in record and field in current:
                        current[field] = record[field]
            return {"timing": {}, "native_entity_records": property_records}
        return {"timing": {}}

    def get_meta(self) -> dict[str, Any]:
        if self._legacy_metadata is not None:
            return self._legacy_metadata.copy()
        effective: dict[str, Any] = {
            "dt": float(self.sim.get_physics_dt()),
            "gravity": self.gravity.tolist(),
            "collision_filter": {
                "self_collision": {
                    entry["name"]: bool(entry["self_collision"])
                    for entry in self.entries
                },
                "environment_isolation": True,
                "implicit_ground": False,
            },
        }
        engine_readback = ["dt"]
        solver_fields = self.physx_solver.configured_fields()
        if solver_fields:
            solver_readback = read_engine_solver_values(
                self.sim.stage,
                include_contact_offset=self.physx_solver.contact_offset is not None,
            )
            for field in solver_fields:
                effective[field] = solver_readback[field]
                engine_readback.append(field)
        return {
            "scene_layout": self.layout.to_dict(),
            "scene_entities_actual": self.actual,
            "gravity": self.gravity.tolist(),
            "use_gpu_pipeline": True,
            "env_origins": self.origins.tolist(),
            "collision_filtering_applied": self.num_envs > 1,
            "render_mode": self.renderer.render_mode,
            "render_width": self.renderer.render_width,
            "render_height": self.renderer.render_height,
            "graphics_enabled": self.renderer.render_mode != "none",
            "raw_usd_cache": {
                "enabled": self._raw_usd_cache_persistent,
                "unique_sources": len(self._reported_raw_usd_source_digests),
                "hits": sum(item["hit"] for item in self._raw_usd_cache_reports),
                "conversions": sum(
                    not item["hit"] for item in self._raw_usd_cache_reports
                ),
                "entries": tuple(self._raw_usd_cache_reports),
            },
            "role_usd_cache": {
                "enabled": self._role_usd_cache is not None,
                "hits": sum(item["hit"] for item in self._role_usd_cache_reports),
                "bakes": sum(not item["hit"] for item in self._role_usd_cache_reports),
                "entries": tuple(self._role_usd_cache_reports),
            },
            "configuration_report": {
                "schema_version": 1,
                "effective": effective,
                "engine_readback": engine_readback,
            },
        }

    def shutdown(self) -> None:
        for handle in self._shm_handles:
            handle.close()
        self._shm_handles.clear()
        self.renderer.shutdown()
        self._temporary.cleanup()

    def init_renderer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.init_renderer(payload))

    def render_frame(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.render_frame())

    def capture_frame(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.renderer.capture_frame())
