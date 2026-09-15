"""IsaacSim/IsaacLab Python 3.11 worker for the UniLab subprocess backend.

The worker is intentionally self-contained.  It imports Kit and IsaacLab only
after receiving ``INIT`` and communicates with the host through the canonical
``subprocess_ipc.protocol`` module loaded by path.  Control messages stay on
the pipe; all batched numeric state is copied into shared-memory slots.

This worker implements MJCF-backed articulation physics, masked root/joint
reset, implicit position-target control, and the eval-owned Kit viewer/RGB
camera commands.  Rendering is selected before Kit starts so a training
worker can remain on the inexpensive no-rendering experience.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np


def _load_protocol(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("unisim_subprocess_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_numpy(value: Any) -> np.ndarray:
    """Detach one IsaacLab tensor at the worker/shm boundary."""
    # IsaacLab's ``Articulation.data`` quantities are torch tensors.  Keep the
    # conversion explicit at this one cold/IO boundary: probing arbitrary
    # backend objects with ``hasattr``/``getattr`` in the physics path can hide
    # API drift and violates the backend-isolation contract.  A non-tensor is
    # an implementation error and should fail loudly instead of being
    # silently coerced through NumPy.
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_tensor(torch: Any, value: np.ndarray, device: str) -> Any:
    return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions (used for reset body->world angvel)."""
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return (v + 2.0 * (w * uv + uuv)).astype(np.float32)


def _resolve_articulation_root_prim_path(usd_path: str, root_name: str) -> str:
    """Resolve the imported articulation root relative to the asset prim.

    MJCF conversion does not promise a fixed nesting depth.  G1 currently
    yields ``/<asset>/<pelvis>/<pelvis>`` while other assets may expose the
    articulation root directly under the asset prim.  Discover the prim once
    during materialization and fail closed when the requested root is
    ambiguous or absent; never guess this path on a physics hot path.
    """
    if not root_name:
        raise ValueError("root_name must be a non-empty body name")
    try:
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"IsaacSim could not open converted USD asset {usd_path!r}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"IsaacSim converted USD asset {usd_path!r} has no valid default prim")
    asset_path = str(default_prim.GetPath()).rstrip("/")
    candidates = []
    articulation_roots = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(asset_path + "/"):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)
        if path.rsplit("/", 1)[-1] != root_name:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(path)
    # IsaacSim's URDF converter creates a native ``root_joint`` articulation
    # root when ``fix_base=True``.  That root is the fixed-base articulation
    # contract; it is not the named link prim.  Prefer the explicitly named
    # root only when it is the sole authored root, otherwise accept the sole
    # converter root as the authoritative path.
    if len(candidates) == 1:
        selected = candidates[0]
    elif not candidates and len(articulation_roots) == 1:
        selected = articulation_roots[0]
    else:
        raise RuntimeError(
            "IsaacSim converted USD articulation root lookup for body "
            f"{root_name!r} expected one ArticulationRootAPI prim below "
            f"{asset_path!r}, found named={candidates or '<none>'}, "
            f"all={articulation_roots or '<none>'}"
        )
    relative = selected[len(asset_path) :]
    if not relative.startswith("/"):
        raise RuntimeError(
            f"IsaacSim articulation root {candidates[0]!r} is not below {asset_path!r}"
        )
    return relative


def _patch_urdf_articulation_root(usd_path: str, root_name: str) -> None:
    """Apply ``ArticulationRootAPI`` on the named root link of a converted URDF.

    The URDF converter emits RigidBody prims but no ArticulationRootAPI (the
    original repository applies it in its bake step, scene_utils.py:1461-1539).
    Apply it on the named root link here so articulation root resolution
    succeeds.
    """
    if not root_name:
        raise ValueError(
            "isaacsim URDF INIT requires root_body_name for the articulation-root patch"
        )
    try:
        from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    default_path = str(stage.GetDefaultPrim().GetPath()).rstrip("/")
    matches = [
        prim for prim in stage.Traverse() if str(prim.GetPath()).rsplit("/", 1)[-1] == root_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"isaacsim URDF articulation root patch expected exactly one prim "
            f"named {root_name!r} below {default_path!r}, found "
            f"{[str(prim.GetPath()) for prim in matches] or '<none>'}"
        )
    existing_roots = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(existing_roots) == 1:
        # The URDF converter already authored the correct root (normally
        # ``root_joint`` for a fixed-base import).  Applying another root API
        # to the link would create two articulation roots and makes PhysX
        # report the robot as floating even though ``fix_base=True`` was used.
        stage.GetRootLayer().Save()
        return
    if existing_roots:
        raise RuntimeError(
            "isaacsim URDF articulation root patch found multiple existing roots: "
            f"{[str(prim.GetPath()) for prim in existing_roots]}"
        )
    UsdPhysics.ArticulationRootAPI.Apply(matches[0])
    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------
# USD bake (literal port of the original repository's scene_utils.py
# `_bake_usd` family, lines 1309-1539, edited in place on the converted
# asset directory instead of a copied bake root — every INIT reconverts
# with force_usd_conversion=True, so the copy buys nothing here)
# ---------------------------------------------------------------------

_CONTACT_OFFSET = 0.002  # scene_utils.py:133
_REST_OFFSET = 0.0  # scene_utils.py:134

# Scene-level PhysX and layout declarations arrive on the INIT wire; the
# worker re-validates them at the boundary and applies exactly what the
# scene declared.  Undeclared keys keep the worker's own defaults (Isaac
# Lab's PhysxCfg and the GridCloner 2.0 m spacing) — task-specific tuning
# is owner configuration, never a worker-side guess keyed off scene shape.

_SCENE_PHYSX_INT_FIELDS = (
    "solver_type",
    "min_position_iteration_count",
    "max_position_iteration_count",
    "min_velocity_iteration_count",
    "max_velocity_iteration_count",
    "gpu_max_rigid_contact_count",
    "gpu_max_rigid_patch_count",
)
_SCENE_PHYSX_FLOAT_FIELDS = (
    "bounce_threshold_velocity",
    "friction_offset_threshold",
    "friction_correlation_distance",
)


def parse_scene_physx_declaration(
    payload_entry: Any,
) -> dict[str, Any] | None:
    """Re-validate the INIT ``scene_physx`` entry at the wire boundary.

    Returns the validated PhysxCfg kwargs, or ``None`` when the scene
    declares no scene-level PhysX configuration (the worker keeps Isaac
    Lab's defaults).  A malformed declaration fails closed at INIT rather
    than being partially applied.
    """
    if payload_entry is None:
        return None
    if not isinstance(payload_entry, dict):
        raise TypeError(
            "isaacsim scene_physx declaration must be a dict, got "
            f"{type(payload_entry).__name__}"
        )
    expected = set(_SCENE_PHYSX_INT_FIELDS) | set(_SCENE_PHYSX_FLOAT_FIELDS)
    if set(payload_entry) != expected:
        raise ValueError(
            "isaacsim scene_physx declaration keys must be exactly "
            f"{sorted(expected)}, got {sorted(payload_entry)}"
        )
    result: dict[str, Any] = {}
    for name in _SCENE_PHYSX_INT_FIELDS:
        value = payload_entry[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError(
                f"isaacsim scene_physx {name} must be a non-negative integer, got {value!r}"
            )
        result[name] = int(value)
    if result["solver_type"] not in (0, 1):
        raise ValueError(
            f"isaacsim scene_physx solver_type must be 0 (PGS) or 1 (TGS), "
            f"got {result['solver_type']!r}"
        )
    for name in ("gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count"):
        if result[name] <= 0:
            raise ValueError(
                f"isaacsim scene_physx {name} must be positive, got {result[name]!r}"
            )
    for name in _SCENE_PHYSX_FLOAT_FIELDS:
        value = float(payload_entry[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"isaacsim scene_physx {name} must be a finite non-negative number, "
                f"got {payload_entry[name]!r}"
            )
        result[name] = value
    return result


def parse_env_grid_spacing(payload_entry: Any) -> float | None:
    """Re-validate the INIT ``env_grid_spacing`` entry at the wire boundary.

    Returns the declared spacing in meters, or ``None`` when the scene
    leaves the layout to the worker (the ``GridCloner`` default of 2.0 m).
    """
    if payload_entry is None:
        return None
    if isinstance(payload_entry, bool) or not isinstance(payload_entry, (int, float)):
        raise TypeError(
            f"isaacsim env_grid_spacing must be a number, got {payload_entry!r}"
        )
    spacing = float(payload_entry)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(
            f"isaacsim env_grid_spacing must be a finite positive number, got {payload_entry!r}"
        )
    return spacing


_InitStateParse = tuple[tuple[float, float, float], tuple[float, float, float, float]]


def parse_entity_init_state(payload_entry: Any) -> _InitStateParse | None:
    """Re-validate one entity's INIT ``init_state`` entry at the wire boundary.

    Returns ``(pos, rot_wxyz)``, or ``None`` when the entity declares no
    spawn pose (the worker keeps the articulation's default spawn state).
    """
    if payload_entry is None:
        return None
    if not isinstance(payload_entry, dict) or set(payload_entry) != {"pos", "rot_wxyz"}:
        raise ValueError(
            "isaacsim init_state declaration must be a dict with exactly "
            f"pos/rot_wxyz keys, got {payload_entry!r}"
        )
    pos_raw = payload_entry["pos"]
    rot_raw = payload_entry["rot_wxyz"]
    if (
        isinstance(pos_raw, (str, bytes))
        or not isinstance(pos_raw, (tuple, list))
        or len(pos_raw) != 3
    ):
        raise ValueError(
            f"isaacsim init_state pos must be an xyz triple, got {pos_raw!r}"
        )
    if (
        isinstance(rot_raw, (str, bytes))
        or not isinstance(rot_raw, (tuple, list))
        or len(rot_raw) != 4
    ):
        raise ValueError(
            f"isaacsim init_state rot_wxyz must be a wxyz quaternion, got {rot_raw!r}"
        )
    pos = tuple(float(value) for value in pos_raw)
    rot = tuple(float(value) for value in rot_raw)
    if not all(math.isfinite(value) for value in pos + rot):
        raise ValueError(
            f"isaacsim init_state pos/rot_wxyz must be finite, got {payload_entry!r}"
        )
    if math.sqrt(sum(value * value for value in rot)) <= 0.0:
        raise ValueError(f"isaacsim init_state rot_wxyz must be non-zero, got {rot_raw!r}")
    return pos, rot


def _entity_declared_bool(entry: dict[str, Any], key: str) -> bool | None:
    """Read one declared boolean entity field, failing closed on bad types."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(
            f"isaacsim entity {entry.get('name')!r} {key} must be a boolean, got {value!r}"
        )
    return value


def _entity_capsule_flag(entry: dict[str, Any]) -> bool:
    """Resolve the URDF converter capsule flag for one entity entry.

    A declared ``replace_cylinders_with_capsules`` value is authoritative.
    Otherwise the backend default follows the materialization: floating
    rigid entities convert with capsule replacement (the dynamic object
    contract, scene_utils.py:1701-1706 applies it to every object
    conversion), while articulations and kinematic rigids keep the
    converter default.
    """
    declared = _entity_declared_bool(entry, "replace_cylinders_with_capsules")
    if declared is not None:
        return declared
    return (
        str(entry.get("materialization")) == "rigid"
        and str(entry.get("root_mode")) == "floating"
    )


# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
# scene_utils.py:136-148.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


@dataclass(frozen=True)
class _EntityBakePlan:
    """Per-role bake parameters (scene_utils.py:1701-1739 call sites)."""

    props: dict[str, Any]
    apply_physx_articulation: bool
    collision_enabled: bool | None


def bake_plan_for_entity(
    materialization: str,
    root_mode: str,
    *,
    collision_enabled: bool | None = None,
    is_variant_target: bool = True,
) -> _EntityBakePlan:
    """Return the bake plan for one entity's declared materialization.

    The plan derives from the declared materialization/root_mode plus the
    entity's declared ``collision_enabled`` (the USD-bake collision flag);
    no task semantics are inferred from entity names:

    - robot articulation: scene_utils.py:1730-1739 (gravity off, max
      depenetration velocity, self-collisions on, articulation solver 8/0).
    - floating rigid (object/pool variants): scene_utils.py:1701-1713
      (dynamic, max depenetration velocity, articulation API off).
    - kinematic rigid (table/goalviz): scene_utils.py:1714-1719/1752-1758
      (kinematic, gravity off); the collision flag is the scene's
      declaration — ``None`` keeps the converted-USD collision state
      (the table contract), ``False`` disables it (the goalviz contract).
    """
    if materialization == "articulation":
        return _EntityBakePlan(
            props={
                "disable_gravity": True,
                "max_depenetration_velocity": 1000.0,
                "enabled_self_collisions": True,
                "solver_position_iterations": 8,
                "solver_velocity_iterations": 0,
            },
            apply_physx_articulation=True,
            collision_enabled=None,
        )
    if materialization != "rigid":
        raise ValueError(f"unsupported entity materialization {materialization!r}")
    if root_mode == "kinematic":
        return _EntityBakePlan(
            props={
                "kinematic_enabled": True,
                "disable_gravity": True,
                "articulation_enabled": False,
            },
            apply_physx_articulation=False,
            collision_enabled=collision_enabled,
        )
    if root_mode != "floating":
        raise ValueError(f"unsupported rigid entity root_mode {root_mode!r}")
    if not is_variant_target:
        raise ValueError(
            "unsupported non-dynamic floating rigid entity; every floating "
            "rigid is the dynamic object contract (scene_utils.py:1701-1713)"
        )
    return _EntityBakePlan(
        props={
            "kinematic_enabled": False,
            "disable_gravity": False,
            "max_depenetration_velocity": 1000.0,
            "articulation_enabled": False,
        },
        apply_physx_articulation=False,
        collision_enabled=None,
    )


def _set_usd_attr(prim: Any, name: str, value: Any, value_type: Any) -> None:
    # scene_utils.py:1068-1075.  The URDF converter occasionally emits
    # attributes with malformed type names; in that case remove and recreate
    # so the typed Set lands.
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    (attr or prim.CreateAttribute(name, value_type, False)).Set(value)


def _bake_usd_in_place(usd_path: str, plan: _EntityBakePlan) -> None:
    """Pre-author physics properties on a converted USD (scene_utils.py:1461-1539).

    Differences from the original ``_bake_usd`` are limited to the in-place
    edit (no copy into a bake root); the authored attributes, values, and
    per-prim group dispatch are identical.
    """
    try:
        from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    vtypes = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
        "Int": Sdf.ValueTypeNames.Int,
    }

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD for baking: {usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {usd_path}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if is_art and plan.apply_physx_articulation:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        for key, val in plan.props.items():
            if val is None:
                continue
            group, attr_name, vtype_str = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            _set_usd_attr(prim, attr_name, val, vtypes[vtype_str])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(_REST_OFFSET)
            if plan.collision_enabled is not None:
                ce = UsdPhysics.CollisionAPI(prim)
                (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(
                    plan.collision_enabled
                )

    stage.GetRootLayer().Save()


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules we need here."""
    # scene_utils.py:1088-1093.
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: str, mesh_filename: str) -> str:
    """Resolve a URDF mesh reference against the URDF directory."""
    # scene_utils.py:1096-1100.
    if os.path.isabs(mesh_filename):
        return mesh_filename
    return os.path.normpath(os.path.join(os.path.dirname(urdf_path), mesh_filename))


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class _UrdfSdfCollisionMarker:
    # scene_utils.py:1078-1085.
    mesh_stem: str
    mesh_filename: str
    resolution: int | None = None
    margin: float | None = None
    narrow_band_thickness: float | None = None
    subgrid_resolution: int | None = None


def _parse_urdf_sdf_collision_markers(asset_path: str) -> list[_UrdfSdfCollisionMarker]:
    """Parse ``<sdf>`` collision hints from a URDF (scene_utils.py:1172-1201)."""
    root = ET.parse(asset_path).getroot()
    markers: list[_UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(asset_path, mesh_filename)
        mesh_stem = os.path.splitext(os.path.basename(mesh_path))[0]
        markers.append(
            _UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness") or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution") or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def _apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_asset_path: str,
    markers: list[_UrdfSdfCollisionMarker],
) -> None:
    """Write parsed SDF hints onto USD mesh-collision prims (scene_utils.py:1204-1272)."""
    if not markers:
        return

    from isaaclab.sim.schemas import (  # type: ignore[import-not-found]
        SDFMeshPropertiesCfg,
        define_mesh_collision_properties,
    )
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    stem = os.path.splitext(os.path.basename(usd_path))[0]
    physics_usd_path = os.path.join(
        os.path.dirname(usd_path), "configuration", f"{stem}_physics.usd"
    )
    edit_usd_path = physics_usd_path if os.path.exists(physics_usd_path) else usd_path

    stage = Usd.Stage.Open(edit_usd_path, Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}
    matched: dict[str, int] = {marker.mesh_stem: 0 for marker in markers}

    fallback_matches = []
    collider_matches = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        path_parts = [part for part in path.split("/") if part]
        marker = next((marker_by_stem[part] for part in path_parts if part in marker_by_stem), None)
        if marker is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches.append((prim, marker))
        else:
            fallback_matches.append((prim, marker))

    for prim, marker in collider_matches or fallback_matches:
        define_mesh_collision_properties(
            str(prim.GetPath()),
            SDFMeshPropertiesCfg(
                sdf_margin=marker.margin,
                sdf_narrow_band_thickness=marker.narrow_band_thickness,
                sdf_resolution=marker.resolution,
                sdf_subgrid_resolution=marker.subgrid_resolution,
            ),
            stage=stage,
        )
        matched[marker.mesh_stem] += 1

    stage.GetRootLayer().Save()

    missing = [stem for stem, count in matched.items() if count == 0]
    if missing:
        print(
            f"[worker] warning: URDF SDF markers in {source_asset_path!r} did not match "
            f"USD collision prims for mesh stems {missing}",
            flush=True,
        )


def compute_adjacent_link_pairs(urdf_path: str) -> dict[str, list[str]]:
    """Adjacent robot link pairs whose self-collision must be filtered.

    Reproduces the original repository's authored adjacency map
    (isaacgymenvs/tasks/simtoolreal/adjacent_links.py, loaded and merged by
    scene_utils.py:1275-1306) from URDF structure, so the worker needs no
    task-side data file:

    - fixed joints are merged first (``merge_fixed_joints`` semantics: the
      fixed-joint child is absorbed into its parent, sensors.py:676-687);
    - two merged bodies are adjacent when a movable joint connects them;
    - bodies two movable joints apart are additionally adjacent when the
      intermediate link is a Sharpa virtual spacer (name ends in ``_VL``):
      those spacers exist only to split one multi-axis joint into 1-DoF URDF
      joints (e.g. ``CMC_FE``/``CMC_AA``), so the flanking bodies behave as
      adjacent under Isaac Gym's self-collision masking.

    Verified pair-exact against ``adjacent_links.py`` (LEFT map) on
    ``iiwa14_left_sharpa_adjusted_restricted.urdf``: 35/35 pairs.
    """
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.iter("joint"):
        parent_tag, child_tag = joint.find("parent"), joint.find("child")
        if parent_tag is None or child_tag is None:
            continue
        joints.append(
            (str(joint.get("type")), str(parent_tag.get("link")), str(child_tag.get("link")))
        )
    links = {str(link.get("name")) for link in root.iter("link") if link.get("name")}

    # Connected components over fixed joints = merged rigid bodies.
    fixed_children = {child for jtype, _, child in joints if jtype == "fixed"}
    fixed_adjacency: dict[str, set[str]] = {link: set() for link in links}
    for jtype, parent_link, child_link in joints:
        if jtype == "fixed":
            fixed_adjacency[parent_link].add(child_link)
            fixed_adjacency[child_link].add(parent_link)
    group_of: dict[str, int] = {}
    groups: list[list[str]] = []
    for link in sorted(links):
        if link in group_of:
            continue
        stack, members = [link], []
        while stack:
            current = stack.pop()
            if current in group_of:
                continue
            group_of[current] = len(groups)
            members.append(current)
            stack.extend(fixed_adjacency[current])
        groups.append(members)
    canonical: list[str] = []
    for members in groups:
        roots = [name for name in members if name not in fixed_children]
        if len(roots) != 1:
            raise ValueError(
                f"merged fixed-joint group {sorted(members)} in {urdf_path} has no unique "
                f"non-fixed-child root link: {roots}"
            )
        canonical.append(roots[0])

    merged_adjacency: dict[str, set[str]] = {name: set() for name in canonical}
    for jtype, parent_link, child_link in joints:
        if jtype == "fixed":
            continue
        a = canonical[group_of[parent_link]]
        b = canonical[group_of[child_link]]
        if a != b:
            merged_adjacency[a].add(b)
            merged_adjacency[b].add(a)

    adjacency: dict[str, set[str]] = {name: set(nbs) for name, nbs in merged_adjacency.items()}
    for middle, neighbors in merged_adjacency.items():
        if not middle.endswith("_VL") or len(neighbors) != 2:
            continue
        a, b = sorted(neighbors)
        adjacency[a].add(b)
        adjacency[b].add(a)
    return {name: sorted(neighbors) for name, neighbors in adjacency.items() if neighbors}


def _apply_self_collision_filters(usd_path: str, adjacency: dict[str, list[str]]) -> None:
    """Author USD ``FilteredPairsAPI`` so adjacent link pairs do NOT self-collide.

    Port of scene_utils.py:1309-1364 with the adjacency map supplied by
    :func:`compute_adjacent_link_pairs` instead of the task-side
    ``adjacent_links.py`` load.  Only effective with self-collision enabled
    (bake ``enabled_self_collisions=True`` + converter ``self_collision=True``).
    Links merged away by ``merge_fixed_joints`` have no rigid-body prim and
    are skipped (a merged link shares its parent's body and cannot
    self-collide anyway).
    """
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    stem = os.path.splitext(os.path.basename(usd_path))[0]
    physics_usd_path = os.path.join(
        os.path.dirname(usd_path), "configuration", f"{stem}_physics.usd"
    )
    edit_usd_path = physics_usd_path if os.path.exists(physics_usd_path) else usd_path

    stage = Usd.Stage.Open(edit_usd_path, Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(
            f"Failed to open USD while applying self-collision filters: {edit_usd_path}"
        )
    stage.Load()

    body_by_name: dict[str, Any] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_by_name[prim.GetName()] = prim

    pairs_filtered = 0
    missing: set[str] = set()
    for link, neighbors in adjacency.items():
        a = body_by_name.get(link)
        if a is None:
            missing.add(link)
            continue
        rel = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for nb in neighbors:
            b = body_by_name.get(nb)
            if b is None:
                missing.add(nb)
                continue
            if b.GetPath() not in existing:
                rel.AddTarget(b.GetPath())
                existing.add(b.GetPath())
                pairs_filtered += 1

    stage.GetRootLayer().Save()

    print(
        f"[worker] self-collision: filtered {pairs_filtered} adjacent link pairs "
        f"across {len(body_by_name)} robot bodies in {os.path.basename(edit_usd_path)}"
        + (f"; skipped {len(missing)} merged/absent links" if missing else ""),
        flush=True,
    )


# ---------------------------------------------------------------------
# Runtime PhysX contact materials (port of scene_utils.py:
# apply_physx_material_properties, 1546-1624; the per-env bucketed friction
# DR at 1594-1606/1615-1621 is deliberately not migrated)
# ---------------------------------------------------------------------


def _validate_friction_triple_wire(value: Any, label: str) -> tuple[float, float, float]:
    """Wire-boundary recheck of one (static, dynamic, restitution) triple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a (static, dynamic, restitution) triple, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"{label} must have exactly 3 components, got {value!r}")
    triple: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise TypeError(f"{label} components must be numbers, got {value!r}")
        numeric = float(component)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"{label} components must be finite non-negative numbers, got {value!r}"
            )
        triple.append(numeric)
    return (triple[0], triple[1], triple[2])


def parse_ground_plane_declaration(
    payload_entry: Any,
) -> tuple[tuple[float, float, float], float, float] | None:
    """Re-validate the INIT ``ground_plane`` entry at the wire boundary.

    Returns ``(friction_triple, restitution, size_m)`` or ``None`` when the
    scene declares no ground plane (the render-mode-only Nucleus ground
    applies for rendered scenes).  A
    malformed declaration fails closed at INIT rather than silently
    spawning a default ground the scene never asked for.
    """
    if payload_entry is None:
        return None
    if not isinstance(payload_entry, dict):
        raise TypeError(
            "isaacsim ground_plane declaration must be a dict, got "
            f"{type(payload_entry).__name__}"
        )
    if set(payload_entry) != {"friction", "restitution", "size_m"}:
        raise ValueError(
            "isaacsim ground_plane declaration keys must be exactly "
            f"friction/restitution/size_m, got {sorted(payload_entry)}"
        )
    friction = _validate_friction_triple_wire(
        payload_entry["friction"], "isaacsim ground_plane friction"
    )
    restitution = float(payload_entry["restitution"])
    size_m = float(payload_entry["size_m"])
    if not math.isfinite(restitution) or restitution < 0.0:
        raise ValueError(
            "isaacsim ground_plane restitution must be a finite non-negative number, "
            f"got {payload_entry['restitution']!r}"
        )
    if not math.isfinite(size_m) or size_m <= 0.0:
        raise ValueError(
            f"isaacsim ground_plane size_m must be a finite positive number, "
            f"got {payload_entry['size_m']!r}"
        )
    return friction, restitution, size_m


def parse_entity_friction(
    entry: dict[str, Any],
) -> tuple[tuple[float, float, float], dict[str, tuple[float, float, float]]] | None:
    """Re-validate one entity's INIT friction payload at the wire boundary.

    Returns ``(default, overrides_by_body)`` or ``None`` when the entity
    declares no contact materials (undeclared scenes never carry the keys).
    Per-body overrides without a default fail closed: the writer tiles the
    default across all shapes before applying overrides
    (scene_utils.py:1576-1577).
    """
    name = str(entry.get("name") or "<unnamed>")
    raw_default = entry.get("friction")
    raw_overrides = entry.get("friction_by_body")
    if raw_default is None:
        if raw_overrides:
            raise ValueError(
                f"isaacsim entity {name!r} declares friction_by_body without a friction "
                "default; the default is tiled across all shapes before overrides apply"
            )
        return None
    default = _validate_friction_triple_wire(raw_default, f"isaacsim entity {name!r} friction")
    if raw_overrides is None:
        return default, {}
    if not isinstance(raw_overrides, dict):
        raise TypeError(
            f"isaacsim entity {name!r} friction_by_body must be a dict of body name to "
            f"triple, got {type(raw_overrides).__name__}"
        )
    overrides = {
        str(body): _validate_friction_triple_wire(
            value, f"isaacsim entity {name!r} friction override for body {body!r}"
        )
        for body, value in raw_overrides.items()
    }
    return default, overrides


def build_friction_shape_table(
    link_names: list[str],
    link_shape_counts: list[int],
    default: tuple[float, float, float],
    overrides: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    """Build the (max_shapes, 3) material table for one articulation env.

    Literal translation of scene_utils.py:1576-1592: the default triple is
    tiled across every collision shape, then each overridden link's shape
    slice is overwritten.  ``link_shape_counts`` must parallel ``link_names``
    (the PhysX view's per-link ``max_shapes``); override bodies that the view
    does not list fail closed.  The caller verifies
    ``table.shape[0] == view.max_shapes`` (scene_utils.py:1588-1592).
    """
    names = [str(name) for name in link_names]
    counts: list[int] = []
    for count in link_shape_counts:
        if isinstance(count, bool) or int(count) != count or int(count) < 0:
            raise ValueError(
                f"friction shape counts must be non-negative integers, "
                f"got {list(link_shape_counts)!r}"
            )
        counts.append(int(count))
    if len(names) != len(counts):
        raise ValueError(
            f"friction link_names/link_shape_counts length mismatch: "
            f"{len(names)} names vs {len(counts)} counts"
        )
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(
            f"friction overrides reference bodies not present in the PhysX view: "
            f"{unknown}; view links: {names}"
        )
    default_arr = np.asarray(
        _validate_friction_triple_wire(default, "friction default"), dtype=np.float32
    )
    total = sum(counts)
    table = np.tile(default_arr, (total, 1))
    shape_start = 0
    for name, count in zip(names, counts):
        shape_end = shape_start + count
        override = overrides.get(name)
        if override is not None:
            table[shape_start:shape_end] = np.asarray(
                _validate_friction_triple_wire(override, f"friction override for {name!r}"),
                dtype=np.float32,
            )
        shape_start = shape_end
    return table


def _verify_baked_usd(usd_path: str, plan: _EntityBakePlan) -> dict[str, int]:
    """Re-open one baked USD and verify every authored attribute (fail-closed).

    Readback half of :func:`_bake_usd_in_place`: traverses the same prim set
    and checks the plan's attributes, the PhysX articulation API application,
    contact/rest offsets, and (when the plan sets it) ``collisionEnabled``.
    Returns per-prim-type counts for INIT forensics; any missing or
    mismatching attribute raises, failing INIT.
    """
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - only runs in external worker
        raise RuntimeError("IsaacSim USD bindings are unavailable") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD for bake readback: {usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {usd_path}")

    counts = {"rigid_body_prims": 0, "articulation_prims": 0, "collision_prims": 0}
    collision_enabled_values: list = []
    mismatches: list[str] = []
    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            counts["rigid_body_prims"] += 1
        if is_art:
            counts["articulation_prims"] += 1
            if plan.apply_physx_articulation and not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                mismatches.append(f"{prim.GetPath()}: missing PhysxArticulationAPI")
        for key, expected in plan.props.items():
            if expected is None:
                continue
            group, attr_name, _ = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            attr = prim.GetAttribute(attr_name)
            actual = attr.Get() if attr else None
            if actual != expected:
                mismatches.append(
                    f"{prim.GetPath()}:{attr_name} is {actual!r}, expected {expected!r}"
                )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            counts["collision_prims"] += 1
            ce_observed = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            collision_enabled_values.append(bool(ce_observed.Get()) if ce_observed else None)
            px = PhysxSchema.PhysxCollisionAPI(prim)
            for get_attr, expected in (
                (px.GetContactOffsetAttr, _CONTACT_OFFSET),
                (px.GetRestOffsetAttr, _REST_OFFSET),
            ):
                attr = get_attr()
                actual = attr.Get() if attr else None
                if actual is None or abs(float(actual) - expected) > 1e-6:
                    mismatches.append(
                        f"{prim.GetPath()}:{attr.GetName() if attr else '<missing>'} is "
                        f"{actual!r}, expected {expected!r}"
                    )
            if plan.collision_enabled is not None:
                ce_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                actual = ce_attr.Get() if ce_attr else None
                if actual != plan.collision_enabled:
                    mismatches.append(
                        f"{prim.GetPath()}:physics:collisionEnabled is {actual!r}, "
                        f"expected {plan.collision_enabled!r}"
                    )
    if mismatches:
        raise RuntimeError(
            f"isaacsim bake readback failed for {os.path.basename(usd_path)}: "
            + "; ".join(mismatches[:8])
            + (f" (+{len(mismatches) - 8} more)" if len(mismatches) > 8 else "")
        )
    # Observed collisionEnabled per collision prim: recorded for probe
    # assertions even when the plan does not pin the flag
    # (the table plan leaves collision untouched and must stay enabled).
    return {**counts, "collision_enabled": collision_enabled_values}


def _readback_variant_masses(usd_paths: list[str]) -> list[float]:
    """Measure the physics mass of every baked pool variant USD (fail-closed).

    Backend-authoritative mass source for the fixed-variant channel: each
    pool variant is a single-link rigid tool, so its USD must carry exactly
    one ``UsdPhysics.RigidBodyAPI`` prim, and the ``UsdPhysics.MassAPI``
    mass on that prim must be readable, finite, and strictly positive.
    The INIT payload never carries masses: this measurement of the
    materialized variant USD is the only reported mass source, not a
    payload echo.  The goalviz
    mirror pool is deliberately never measured (kinematic, non-physical).
    """
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    masses: list[float] = []
    for usd_path in usd_paths:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"failed to open variant USD for mass readback: {usd_path}")
        bodies = [
            prim
            for prim in Usd.PrimRange(stage.GetPseudoRoot())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(bodies) != 1:
            raise ValueError(
                f"variant USD {os.path.basename(usd_path)} must carry exactly one "
                f"rigid-body prim for mass readback, found {len(bodies)}"
            )
        attr = UsdPhysics.MassAPI(bodies[0]).GetMassAttr()
        mass = attr.Get() if attr else None
        if mass is None or not math.isfinite(float(mass)) or float(mass) <= 0.0:
            raise ValueError(
                f"variant USD {os.path.basename(usd_path)} rigid body has no readable "
                f"positive physics:mass, got {mass!r}"
            )
        masses.append(float(mass))
    return masses


def _verify_self_collision_filters(
    usd_path: str, adjacency: dict[str, list[str]]
) -> dict[str, int]:
    """Re-open the robot USD and verify the authored FilteredPairs (fail-closed).

    Readback half of :func:`_apply_self_collision_filters`: the expected
    directed target count is the adjacency restricted to links that survive
    ``merge_fixed_joints`` (i.e. have a rigid-body prim), and every expected
    target must be present in the composed relationship.
    """
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    stem = os.path.splitext(os.path.basename(usd_path))[0]
    physics_usd_path = os.path.join(
        os.path.dirname(usd_path), "configuration", f"{stem}_physics.usd"
    )
    edit_usd_path = physics_usd_path if os.path.exists(physics_usd_path) else usd_path

    stage = Usd.Stage.Open(edit_usd_path, Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD for self-collision readback: {edit_usd_path}")
    stage.Load()

    body_by_name: dict[str, Any] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_by_name[prim.GetName()] = prim

    directed_targets = 0
    expected_targets = 0
    missing: set[str] = set()
    mismatches: list[str] = []
    for link, neighbors in adjacency.items():
        a = body_by_name.get(link)
        if a is None:
            missing.add(link)
            continue
        targets = set(UsdPhysics.FilteredPairsAPI(a).GetFilteredPairsRel().GetTargets())
        directed_targets += len(targets)
        for nb in neighbors:
            b = body_by_name.get(nb)
            if b is None:
                missing.add(nb)
                continue
            expected_targets += 1
            if b.GetPath() not in targets:
                mismatches.append(f"{link} -> {nb} not filtered")
    if directed_targets != expected_targets:
        mismatches.append(
            f"filtered target count {directed_targets} != expected {expected_targets}"
        )
    if mismatches:
        raise RuntimeError(
            f"isaacsim self-collision readback failed for "
            f"{os.path.basename(edit_usd_path)}: "
            + "; ".join(mismatches[:8])
            + (f" (+{len(mismatches) - 8} more)" if len(mismatches) > 8 else "")
        )
    return {
        "filtered_pair_targets": directed_targets,
        "expected_pair_targets": expected_targets,
        "bodies": len(body_by_name),
        "merged_or_absent_links": len(missing),
    }


class _WorkerContext:
    # Detail-timing flag reads must tolerate ``__new__``-built contexts (unit
    # tests) that bypass ``__init__``; the env-var assignment below overrides
    # this default for real workers.
    _profile_detail: bool = False

    def __init__(self, protocol: Any) -> None:
        self.protocol = protocol
        self.num_envs = 0
        self.num_dof = 0
        self.num_bodies = 0
        self.sim_dt = 0.0
        self.device = "cuda:0"
        self.sim: Any = None
        self.robot: Any = None
        # Non-articulation scene entities (table/object/goalviz), keyed by
        # declared entity name.  Empty for single-asset scenes; each
        # entry owns an entity_root_state__/entity_reset_state__ slot pair
        # (step 1.3c) attached by the host after INIT.
        self.rigid_objects: dict[str, Any] = {}
        # (target entity name, converted pool USD paths, measured per-variant
        # masses) when a variant pool was materialized; used for the INIT
        # forensics report and the backend-authoritative mass readback
        # (measurement, not payload echo).
        self._variant_pool_usds: tuple[str, list[str], list[float]] | None = None
        # (entity_name, source_pool_target, usd_paths) when a kinematic entity
        # mirrors a declared variant pool (SimToolReal goalviz, 2026-09-14).
        self._goalviz_mirror: tuple[str, str, list[str]] | None = None
        self.simulation_app: Any = None
        self.torch: Any = None
        self.render_mode = "none"
        self.render_width = 1280
        self.render_height = 720
        self.camera: Any = None
        self.camera_distance = 2.0
        self.camera_elevation_deg = 20.0
        self.camera_azimuth_deg = 90.0
        self.native_joint_names: list[str] = []
        self.native_body_names: list[str] = []
        self.contract_joint_names: list[str] = []
        self.contract_body_names: list[str] = []
        self.native_joint_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        self.native_body_for_contract: np.ndarray = np.empty(0, dtype=np.int64)
        # Physical clones are translated apart in the worker so that their
        # collision geometry does not overlap.  UniLab's flat-scene contract
        # exposes per-environment local coordinates (``env_origins`` is zero),
        # therefore these offsets stay private to the worker and are removed
        # at the shared-memory boundary.
        self.env_origins = np.empty((0, 3), dtype=np.float32)
        self.env_prim_paths: list[str] = []
        self.collision_filtering_applied = False
        self.slots: dict[str, np.ndarray] = {}
        self._shm_handles: list[Any] = []
        self._profile_detail = os.environ.get("UNISIM_PROFILE_DETAIL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_refresh_timing_ms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Cold-path materialization
    # ------------------------------------------------------------------

    def init_sim(self, payload: dict[str, Any]) -> dict[str, Any]:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
        self.num_envs = int(payload["num_envs"])
        self.sim_dt = float(payload["sim_dt"])
        device_id = int(payload.get("device_id", 0))
        if device_id < 0:
            raise NotImplementedError(
                "isaacsim requires a CUDA device; CPU IsaacLab physics is outside the "
                "supported subprocess profile"
            )
        self.device = f"cuda:{device_id}"

        raw_render_mode = payload.get("render_mode", "none")
        if not isinstance(raw_render_mode, str):
            raise TypeError(
                "isaacsim worker render_mode must be a string; "
                f"got {type(raw_render_mode).__name__}"
            )
        render_mode = raw_render_mode.strip().lower()
        if render_mode not in {"none", "record", "interactive"}:
            raise ValueError(
                "isaacsim worker render_mode must be one of none, record, interactive; "
                f"got {render_mode!r}"
            )
        self.render_mode = render_mode
        raw_render_width = payload.get("render_width", 1280)
        raw_render_height = payload.get("render_height", 720)
        if (
            isinstance(raw_render_width, bool)
            or not isinstance(raw_render_width, int)
            or isinstance(raw_render_height, bool)
            or not isinstance(raw_render_height, int)
            or raw_render_width <= 0
            or raw_render_height <= 0
        ):
            raise ValueError(
                "isaacsim worker render dimensions must be positive integers; "
                f"got {raw_render_width!r}x{raw_render_height!r}"
            )
        self.render_width = raw_render_width
        self.render_height = raw_render_height
        headless = render_mode != "interactive"
        enable_cameras = render_mode == "record"
        # AppLauncher treats false/default values as "consult the environment".
        # Pin both variables explicitly so a user's shell cannot accidentally
        # turn a training worker into a GUI/camera process.
        os.environ["HEADLESS"] = "1" if headless else "0"
        os.environ["ENABLE_CAMERAS"] = "1" if enable_cameras else "0"
        os.environ["LIVESTREAM"] = "0"
        os.environ["XR"] = "0"

        # Kit must be launched before importing IsaacSim/IsaacLab modules.
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        self.simulation_app = AppLauncher(
            {
                "headless": headless,
                "enable_cameras": enable_cameras,
                "device": self.device,
                "multi_gpu": False,
                "width": self.render_width,
                "height": self.render_height,
                "window_width": self.render_width,
                "window_height": self.render_height,
            }
        ).app

        import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
        import isaacsim.core.utils.prims as prim_utils  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import-not-found]
        from isaaclab.assets import (  # type: ignore[import-not-found]
            Articulation,
            ArticulationCfg,
            RigidObject,
        )
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            MjcfConverter,
            MjcfConverterCfg,
        )
        from isaacsim.core.cloner import GridCloner  # type: ignore[import-not-found]
        from isaacsim.core.utils.extensions import (
            enable_extension,  # type: ignore[import-not-found]
        )

        if render_mode == "record":
            from isaaclab.sensors.camera import Camera, CameraCfg  # type: ignore[import-not-found]

        self.torch = torch
        # The extension is enabled explicitly because IsaacSim 5.1 does not
        # guarantee the MJCF importer is active in a bare headless AppLauncher.
        enable_extension("isaacsim.asset.importer.mjcf")

        model_file = os.fspath(payload["model_file"])
        self._fixed_base = bool(payload.get("fixed_base", False))
        entity_payloads = [dict(entry) for entry in (payload.get("entities") or [])]
        variant_pool = payload.get("variant_pool")
        ground_plane = parse_ground_plane_declaration(payload.get("ground_plane"))
        scene_physx = parse_scene_physx_declaration(payload.get("scene_physx"))
        env_grid_spacing = parse_env_grid_spacing(payload.get("env_grid_spacing"))
        rigid_spawn_cfgs: list[tuple[str, Any]] = []
        if entity_payloads:
            # SimToolReal step-1.3a multi-asset entry: one Articulation (the
            # robot) plus one RigidObject per rigid role on the same stage.
            enable_extension("isaacsim.asset.importer.urdf")
            robot_entry, robot_usd_path, rigid_spawn_cfgs = self._materialize_scene_entities(
                entity_payloads, variant_pool
            )
            contract_source: dict[str, Any] = robot_entry
        elif model_file.lower().endswith(".urdf"):
            # SimToolReal step-0 URDF entry: same converter and flags family as
            # the original repository's scene_utils.py:_convert_urdf_to_usd
            # (zero-gain force position drives so ImplicitActuator owns gains).
            enable_extension("isaacsim.asset.importer.urdf")
            from isaaclab.sim.converters import (  # type: ignore[import-not-found]
                UrdfConverter,
                UrdfConverterCfg,
            )

            converter = UrdfConverter(
                UrdfConverterCfg(
                    asset_path=model_file,
                    fix_base=self._fixed_base,
                    merge_fixed_joints=bool(payload.get("urdf_merge_fixed_joints", True)),
                    self_collision=bool(payload.get("urdf_self_collision", False)),
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        drive_type="force",
                        target_type="position",
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                    ),
                )
            )
            _patch_urdf_articulation_root(
                str(converter.usd_path), str(payload.get("root_body_name") or "")
            )
            robot_usd_path = str(converter.usd_path)
            contract_source = payload
        else:
            converter = MjcfConverter(
                MjcfConverterCfg(
                    asset_path=model_file,
                    fix_base=False,
                    import_sites=True,
                    make_instanceable=True,
                    self_collision=False,
                )
            )
            robot_usd_path = str(converter.usd_path)
            contract_source = payload

        # Build a deterministic environment grid.  The USD importer owns the
        # robot hierarchy; only these Xforms and the articulation wrapper are
        # created here, so no asset/XML parsing occurs on a hot path.  The
        # translations are private worker offsets; state is normalized back to
        # local coordinates before it is published to the host.  The spacing
        # comes from the scene declaration (``None`` keeps the GridCloner
        # default of 2.0 m), and cloning never replicates physics or uses
        # Fabric scene-graph instantiation — every environment is an
        # independent copy under explicit per-env collision filtering.
        cloner = GridCloner(
            spacing=2.0 if env_grid_spacing is None else env_grid_spacing
        )
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # The source Xform must exist before GridCloner.clone.  The returned
        # transforms are the authoritative world origins (a centered grid for
        # two or more environments), so no duplicate hand-written grid math is
        # needed here.
        prim_utils.create_prim(self.env_prim_paths[0], "Xform")
        self.env_origins = np.asarray(
            cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=False,
                copy_from_source=True,
                clone_in_fabric=False,
            ),
            dtype=np.float32,
        )
        expected_origins = (self.num_envs, 3)
        if self.env_origins.shape != expected_origins or not np.isfinite(self.env_origins).all():
            raise RuntimeError(
                "IsaacSim GridCloner returned invalid environment origins: "
                f"shape={self.env_origins.shape}, expected={expected_origins}"
            )

        root_name = str(contract_source.get("root_body_name") or "")
        if not root_name:
            raise ValueError(
                "isaacsim INIT requires root_body_name so articulation_root_prim_path "
                "is explicit and importer discovery cannot choose a wrong root"
            )
        # IsaacLab resolves this path relative to each /Robot instance.  The
        # converter's nesting is asset-dependent, so discover it from the
        # converted USD stage rather than baking in the G1 layout.
        articulation_root = _resolve_articulation_root_prim_path(robot_usd_path, root_name)
        # Entity payloads carry the per-role "joint_names" list; the
        # single-asset contract keeps the "mjcf_joint_names" key.
        joint_names = [
            str(name)
            for name in (
                contract_source.get("joint_names") or contract_source.get("mjcf_joint_names") or []
            )
        ]
        if not joint_names:
            raise ValueError("isaacsim INIT requires the MJCF joint-name contract")
        gains = self._actuator_dicts(contract_source, joint_names)
        robot_prim_path = (
            f"/World/envs/env_.*/{robot_entry['name']}"
            if entity_payloads
            else "/World/envs/env_.*/Robot"
        )
        robot_cfg_kwargs: dict[str, Any] = dict(
            prim_path=robot_prim_path,
            articulation_root_prim_path=articulation_root,
            spawn=sim_utils.UsdFileCfg(usd_path=robot_usd_path),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=gains["stiffness"],
                    damping=gains["damping"],
                    effort_limit_sim=gains["effort"],
                    armature=gains["armature"],
                    friction=gains["friction"],
                )
            },
        )
        # The robot's spawn pose is a declaration, not a path property: the
        # primary articulation entity may declare ``init_state`` (a
        # fixed-base robot's root pose has no other write channel — root
        # writes are reset-event-owned and fixed-base root writes are
        # fail-closed); undeclared scenes keep Isaac Lab's default spawn.
        robot_init_state = parse_entity_init_state(
            contract_source.get("init_state")
            if not entity_payloads
            else robot_entry.get("init_state")
        )
        if robot_init_state is not None:
            robot_cfg_kwargs["init_state"] = ArticulationCfg.InitialStateCfg(
                pos=robot_init_state[0],
                rot=robot_init_state[1],
            )
        robot_cfg = ArticulationCfg(**robot_cfg_kwargs)
        # Scene-level PhysX configuration is equally a declaration: a
        # declared ``scene_physx`` entry rebuilds the solver tuning
        # (iteration clamps, bounce threshold, GPU contact stream buffers),
        # an undeclared scene keeps Isaac Lab's defaults.  Isaac Lab
        # flattens the physx config into carb settings and PhysxSceneAPI
        # attributes at SimulationContext construction
        # (isaaclab/sim/simulation_context.py:261-266, 862-866; PhysicsContext
        # consumers at isaacsim core physics_context.py:155-188).
        if scene_physx is not None:
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.sim_dt,
                device=self.device,
                physx=sim_utils.PhysxCfg(**scene_physx),
            )
        else:
            sim_cfg = sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # Ground plane is declared scene content, not worker policy: the
        # original repository
        # assembles the floor as task-level scene composition (scene_utils.py
        # setup_scene step 5: world-level /World/ground, GroundPlaneCfg
        # defaults, spawned in training too).  A declared scene spawns the
        # offline-safe local collision ground in every runtime mode — the
        # default declaration reproduces the original GroundPlaneCfg physics
        # parameter for parameter — while an undeclared scene keeps the
        # backend's native ground behavior.
        if ground_plane is not None:
            ground_friction, ground_restitution, ground_size_m = ground_plane
            self._spawn_local_ground_plane(
                friction=ground_friction,
                restitution=ground_restitution,
                size_m=ground_size_m,
            )
        elif render_mode != "none":
            # Use IsaacSim's standard grid-world floor for rendered playback.
            # The MJCF floor is retained for the task/physics contract, while
            # this native floor supplies the normal IsaacSim visual ground.
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        # IsaacLab's SimulationContext owns the singleton simulation stage and
        # must be materialized before assets/articulations bind to it.  Keep
        # this ordering explicit so a real Kit worker does not accidentally
        # construct an Articulation against an uninitialized context.
        self.robot = Articulation(robot_cfg)
        # Rigid scene entities (table/object/goalviz) spawn on the same stage
        # and env grid; scenes without rigid entities leave this loop empty.
        for entity_name, rigid_cfg in rigid_spawn_cfgs:
            self.rigid_objects[entity_name] = RigidObject(rigid_cfg)
        if render_mode != "none":
            # MJCF scenes do not necessarily carry a renderer light.  This is
            # a real scene light (not a post-process or synthetic frame), and
            # is created only on the cold rendering path.
            light_cfg = sim_utils.DomeLightCfg(
                # MJCF scenes already provide a world light.  A 2500-lumen
                # dome on top of that light clips the converted materials on
                # RTX cameras (the RGB stream becomes nearly uniform white).
                # Keep a low fill light so the imported scene remains visible
                # without washing out its silver/black contrast.
                intensity=100.0,
                color=(0.75, 0.75, 0.75),
            )
            light_cfg.func("/World/UniLabDomeLight", light_cfg)
            if render_mode == "record":
                camera_cfg = CameraCfg(
                    # Playback emits one video stream, so own one camera in
                    # env 0 rather than allocating an RTX render product for
                    # every policy-eval environment. This mirrors the
                    # IsaacGym capture path and keeps camera cost independent
                    # of ``training.play_env_num``.
                    prim_path="/World/envs/env_0/UniLabCamera",
                    update_period=0.0,
                    data_types=["rgb"],
                    width=self.render_width,
                    height=self.render_height,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 1.0e5),
                    ),
                )
                self.camera = Camera(camera_cfg)
        # Apply IsaacLab's PhysX collision-group filtering before the first
        # reset/step.  Without this stage operation, the translated clones
        # can still collide when a reset puts two local roots at the same pose.
        # Failing closed is important: an unfiltered batch is not equivalent
        # to the SimBackend's independent-environment contract.
        if self.num_envs > 1:
            cloner.filter_collisions(
                self._physics_scene_path(),
                "/World/collisions",
                self.env_prim_paths,
            )
            self.collision_filtering_applied = True
        self.sim.reset()
        self.robot.update(self.sim_dt)
        # Runtime contact materials go through the PhysX views, which exist
        # only after the first sim reset (scene_utils.py:1546-1555: "Must run
        # after DirectRLEnv starts the simulator and root_physx_view exists").
        friction_meta: dict[str, Any] = {}
        if entity_payloads:
            friction_meta = self._apply_friction_writes(entity_payloads)
        if self.camera is not None:
            if not self.camera.is_initialized:
                raise RuntimeError(
                    "IsaacSim RGB camera did not initialize; ensure the Kit experience "
                    "was launched with enable_cameras=True"
                )
            self.camera.reset()
            self.camera.update(self.sim_dt, force_recompute=True)

        self.native_joint_names = [str(name) for name in self.robot.joint_names]
        self.native_body_names = [str(name) for name in self.robot.body_names]
        self.num_dof = int(self.robot.num_joints)
        self.num_bodies = int(self.robot.num_bodies)
        self.contract_joint_names = joint_names
        self.contract_body_names = [
            str(name) for name in (payload.get("mjcf_body_names") or self.native_body_names)
        ]
        self.native_joint_for_contract = self._build_permutation(
            self.native_joint_names, self.contract_joint_names, "joint"
        )
        self.native_body_for_contract = self._build_permutation(
            self.native_body_names, self.contract_body_names, "body"
        )

        keyframe_qpos = payload.get("keyframe_qpos")
        if keyframe_qpos is not None:
            self._apply_keyframe(keyframe_qpos)
        meta: dict[str, Any] = {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            # Runtime fixity diagnostics: the payload flag vs IsaacLab's own
            # view of the articulation root.
            "fixed_base": bool(self._fixed_base),
            "robot_is_fixed_base": bool(getattr(self.robot, "is_fixed_base", False)),
            "robot_articulation_root": articulation_root,
            # Expose UniLab contract order, not the importer/native order.
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "dof_lower": self._joint_limits()[0],
            "dof_upper": self._joint_limits()[1],
            "effort": self._joint_limits()[2],
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": render_mode != "none",
            "render_mode": render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "native_dof_names": list(self.native_joint_names),
            "native_body_names": list(self.native_body_names),
            "usd_path": robot_usd_path,
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
        }
        if entity_payloads:
            # Articulation-root forensics: which prim carries the
            # root API and how the root link is anchored.  A fixed-base
            # conversion anchors the root link to the world through a fixed
            # ``root_joint``; a second root API on the named link would make
            # PhysX report the robot as floating.
            from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

            usd_stage = Usd.Stage.Open(robot_usd_path)
            root_api_prims = [
                str(prim.GetPath())
                for prim in usd_stage.Traverse()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
            anchors = []
            for joint in usd_stage.Traverse():
                if not joint.IsA(UsdPhysics.FixedJoint):
                    continue
                j = UsdPhysics.FixedJoint(joint)
                anchors.append(
                    {
                        "path": str(joint.GetPath()),
                        "body0": str(j.GetBody0Rel().GetTargets() or ["<world>"])[1:-1].strip("'"),
                        "body1": str(j.GetBody1Rel().GetTargets() or ["<world>"])[1:-1].strip("'"),
                    }
                )
            meta["articulation_forensics"] = {
                "articulation_root_prims": root_api_prims,
                "fixed_joints": anchors,
                "default_prim": str(usd_stage.GetDefaultPrim().GetPath()),
            }
        if entity_payloads:
            # Multi-asset forensics for the host/probe: per-entity prim
            # presence, ImplicitActuator gain readback (env 0, contract joint
            # order), and the observed round-robin variant assignment.
            meta["entities"] = [
                {
                    "name": str(entry["name"]),
                    "materialization": str(entry["materialization"]),
                    "root_mode": str(entry["root_mode"]),
                    "usd_path": str(entry.get("_usd_path", "")),
                }
                for entry in entity_payloads
            ]
            meta["entity_prim_counts"] = self._entity_prim_counts(entity_payloads)
            meta["actuator_gains_env0"] = self._readback_actuator_gains()
            # Fail-closed USD readback of the 1.3b bake (per-role physics
            # props, contact/rest offsets, collisionEnabled) and the robot's
            # FilteredPairs, plus the PhysX material write/readback summary.
            meta["bake"] = self._readback_bake(entity_payloads)
            meta["friction"] = friction_meta
            # Scene-level PhysX effective values read back for INIT meta.
            meta["scene_physx"] = self._readback_scene_physx()
            if self._variant_pool_usds is not None:
                target_name, pool_usds, measured_masses = self._variant_pool_usds
                # Authoritative pool echo: the spawner materialized each env
                # from its assigned pool USD by construction (the round-robin
                # prototype selection or the expanded per-env list below), so
                # echoing the assignment is exact rather than observed.  The
                # host compares count, target, and the full assignment
                # against the immutable plan at INIT binding.
                echoed_assignment = [int(value) for value in variant_pool["assignments"]]
                meta["fixed_variant_count"] = len(pool_usds)
                meta["fixed_variant_assignment"] = echoed_assignment
                meta["fixed_variant_target_entity"] = target_name
                meta["variant_assignment"] = {
                    "target_entity": target_name,
                    "expected": echoed_assignment,
                    # Optional stage forensics: None when the prim stacks are
                    # flattened or the env count exceeds the probe regime; the
                    # handshake stands on the authoritative echo above.
                    "observed": self._observe_variant_assignment(target_name, pool_usds),
                    # Backend-authoritative measurement of the baked variant
                    # USDs: the payload never carries masses, so this is the
                    # only mass source; one finite value per pool source file.
                    "masses": [float(value) for value in measured_masses],
                }
            if self._goalviz_mirror is not None:
                mirror_name, mirror_source, mirror_usds = self._goalviz_mirror
                meta["goalviz_mirror"] = {
                    "source_pool_target": mirror_source,
                    "variants": len(mirror_usds),
                    "observed": self._observe_variant_assignment(mirror_name, mirror_usds),
                }
        return meta

    def _readback_scene_physx(self) -> dict[str, Any]:
        """Read the effective scene-level PhysX configuration for INIT meta.

        Isaac Lab applies ``SimulationCfg.physx`` onto the stage's
        PhysxSceneAPI (iteration clamps/thresholds via
        ``SimulationContext._set_physics_engine_settings``; GPU contact stream
        buffers and thresholds via ``PhysicsContext`` setters), so reading the
        prim back proves the original repository's scene-level configuration
        took effect.  ``solver_type`` is
        only authored for PGS, so a missing attribute means the TGS default.
        """
        from pxr import PhysxSchema  # type: ignore[import-not-found]

        scene_prim = self.robot.stage.GetPrimAtPath(self._physics_scene_path())
        api = PhysxSchema.PhysxSceneAPI(scene_prim)
        strict_readers = {
            "min_position_iteration_count": api.GetMinPositionIterationCountAttr,
            "max_position_iteration_count": api.GetMaxPositionIterationCountAttr,
            "min_velocity_iteration_count": api.GetMinVelocityIterationCountAttr,
            "max_velocity_iteration_count": api.GetMaxVelocityIterationCountAttr,
            "bounce_threshold_velocity": api.GetBounceThresholdAttr,
            "friction_offset_threshold": api.GetFrictionOffsetThresholdAttr,
            "friction_correlation_distance": api.GetFrictionCorrelationDistanceAttr,
            "gpu_max_rigid_contact_count": api.GetGpuMaxRigidContactCountAttr,
            "gpu_max_rigid_patch_count": api.GetGpuMaxRigidPatchCountAttr,
        }
        readback: dict[str, Any] = {}
        for name, getter in strict_readers.items():
            attr = getter()
            value = attr.Get() if attr is not None else None
            if value is None:
                raise RuntimeError(
                    f"scene PhysX readback missing {name!r}; the scene-level "
                    "SimulationCfg.physx configuration did not apply"
                )
            readback[name] = value
        solver_attr = api.GetSolverTypeAttr()
        solver_value = solver_attr.Get() if solver_attr is not None else None
        readback["solver_type"] = "default_tgs" if solver_value is None else solver_value
        return readback

    def _spawn_local_ground_plane(
        self,
        prim_path: str = "/World/ground",
        *,
        friction: tuple[float, float, float] = (0.5, 0.5, 0.0),
        restitution: float = 0.0,
        size_m: float = 200.0,
    ) -> None:
        """Author the scene's declared colliding ground without Nucleus.

        ``spawn_ground_plane`` + ``GroundPlaneCfg`` (scene_utils.py:1807)
        creates ``/World/ground`` with a collision plane bound to the default
        rigid-body material, but its default visual grid USD is a Nucleus
        asset that breaks offline cold starts, and this Kit build's UsdPhysics
        exposes neither ``PlaneAPI`` nor a writable ``UsdGeomPlane`` normal.
        This authors the ground as a large thin collision box whose top
        surface is z=0, bound to a rigid-body material — functionally
        equivalent for this task, whose falling objects terminate at
        z < 0.1 long before a bounded extent could matter.  Spawned in
        training (headless) and playback alike, matching the original scene
        contract; the box's own display color provides the visual floor.

        Parameters come from the scene's ``GroundPlaneSceneCfg`` declaration
        (INIT ``ground_plane`` entry): the triple's static and dynamic
        components and ``restitution`` parameterize the material, and
        ``size_m`` is the box's full side length.  With the declaration's
        default values every authored parameter matches the original
        unconditional spawn exactly (static/dynamic friction 0.5,
        restitution 0.0, 200 m extent), so a default-declared scene
        reproduces the original ground prim parameter for parameter.
        """
        from isaaclab.sim.spawners.materials import (
            RigidBodyMaterialCfg,  # type: ignore[import-not-found]
        )
        from isaaclab.sim.utils import bind_physics_material  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        stage = self.sim.stage
        if stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A prim already exists at path: '{prim_path}'.")

        # Flat ground collider as a large thin box whose top surface is z=0.
        # The original uses an infinite UsdPhysics plane; this Kit build's
        # UsdPhysics exposes neither PlaneAPI nor a writable UsdGeomPlane
        # normal, and a bounded box is functionally equivalent here — falling
        # objects rest on the same z=0 surface long before reaching the
        # declared extent (fall terminates at object z < 0.1).
        half = float(size_m) / 2.0
        thickness = 0.1
        box = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
        box.CreateSizeAttr(1.0)
        box_prim = box.GetPrim()
        xform = UsdGeom.XformCommonAPI(box_prim)
        xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -thickness / 2.0))
        xform.SetScale(Gf.Vec3f(half * 2.0, half * 2.0, thickness))
        box.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.34, 0.36)])
        UsdPhysics.CollisionAPI.Apply(box_prim)

        # Isaac Lab's default rigid-body material pattern, parameterized by
        # the declaration: with GroundPlaneSceneCfg defaults this is exactly
        # the 0.5/0.5/0.0 triple GroundPlaneCfg binds to the original ground
        # (from_files_cfg RigidBodyMaterialCfg defaults; scene_utils.py:1807
        # uses cfg defaults).
        material_cfg = RigidBodyMaterialCfg(
            static_friction=float(friction[0]),
            dynamic_friction=float(friction[1]),
            restitution=float(restitution),
        )
        material_path = f"{prim_path}/physicsMaterial"
        material_cfg.func(material_path, material_cfg)
        bind_physics_material(prim_path, material_path)

    def _physics_scene_path(self) -> str:
        """Find the stage's PhysX scene prim on the materialization path."""
        try:
            from pxr import PhysxSchema  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - external worker only
            raise RuntimeError("IsaacSim PhysX USD bindings are unavailable") from exc
        for prim in self.robot.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                return str(prim.GetPath())
        raise RuntimeError(
            "IsaacSim stage has no PhysxSceneAPI; cannot filter environment collisions"
        )

    @staticmethod
    def _build_permutation(native: list[str], contract: list[str], kind: str) -> np.ndarray:
        if len(native) != len(contract) or len(set(native)) != len(native):
            raise RuntimeError(
                f"isaacsim importer returned invalid {kind} names: "
                f"native={native}, contract={contract}"
            )
        native_ids = {name: index for index, name in enumerate(native)}
        missing = [name for name in contract if name not in native_ids]
        extra = [name for name in native if name not in set(contract)]
        if missing or extra or len(set(contract)) != len(contract):
            raise RuntimeError(
                f"isaacsim importer {kind} mapping mismatch: missing={missing}, extra={extra}, "
                f"native={native}, contract={contract}"
            )
        return np.asarray([native_ids[name] for name in contract], dtype=np.int64)

    @staticmethod
    def _actuator_dicts(payload: dict[str, Any], names: list[str]) -> dict[str, dict[str, float]]:
        def values(key: str, default: float = 0.0) -> dict[str, float]:
            raw = list(payload.get(key) or [])
            if len(raw) != len(names):
                raise RuntimeError(
                    f"{key} has {len(raw)} values but the MJCF contract has {len(names)} joints"
                )
            return {name: float(raw[index]) for index, name in enumerate(names)}

        effort = values("dof_effort")
        # PhysX/IsaacLab reject an infinite or excessively large effort in
        # some releases.  The host uses 1e20 as the unlimited sentinel; use
        # the documented finite implicit-actuator ceiling in the worker.
        effort = {
            name: (1.0e9 if value <= 0.0 or value >= 1.0e19 else value)
            for name, value in effort.items()
        }
        return {
            "stiffness": values("dof_stiffness"),
            "damping": values("dof_damping"),
            "effort": effort,
            "armature": values("dof_armature"),
            "friction": values("dof_friction"),
        }

    # ------------------------------------------------------------------
    # Multi-asset materialization (SimToolReal step 1.3a)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entity_name(entry: dict[str, Any]) -> str:
        name = str(entry.get("name") or "")
        if (
            not name
            or not (name[0].isalpha() or name[0] == "_")
            or not all(char.isalnum() or char == "_" for char in name)
        ):
            raise ValueError(f"isaacsim entity name {name!r} is not a valid USD prim identifier")
        return name

    def _convert_entity_urdf(
        self,
        asset_path: str,
        *,
        fix_base: bool,
        self_collision: bool | None,
        with_joint_drive: bool,
        replace_cylinders_with_capsules: bool,
    ) -> str:
        """Convert one entity URDF with the original repository's flag family.

        Literal translation of scene_utils.py:_convert_urdf_to_usd
        (1421-1449): ``force_usd_conversion=True``, ``merge_fixed_joints=True``,
        ``make_instanceable=False``; ``self_collision`` is passed only when not
        ``None`` (scene_utils.py:1441-1442).  The robot's zero-gain force
        position drive comes from scene_utils.py:1453-1458 so the runtime
        ImplicitActuator layer owns the gains (1.3b replaces the zero table).
        """
        from isaaclab.sim.converters import (  # type: ignore[import-not-found]
            UrdfConverter,
            UrdfConverterCfg,
        )

        joint_drive = None
        if with_joint_drive:
            joint_drive = UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            )
        cfg_kwargs: dict[str, Any] = {
            "asset_path": asset_path,
            "force_usd_conversion": True,
            "fix_base": fix_base,
            "merge_fixed_joints": True,
            "make_instanceable": False,
            "replace_cylinders_with_capsules": replace_cylinders_with_capsules,
            "joint_drive": joint_drive,
        }
        if self_collision is not None:
            cfg_kwargs["self_collision"] = self_collision
        usd_path = str(UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path)
        # scene_utils.py:1444-1448: SDF collision markers are part of every
        # conversion call (a no-op for assets without <sdf> tags).
        _apply_urdf_sdf_collision_markers(
            usd_path, asset_path, _parse_urdf_sdf_collision_markers(asset_path)
        )
        return usd_path

    def _validate_variant_pool_payload(
        self, variant_pool: Any, entity_payloads: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Re-validate the host's variant pool at the wire boundary (fail-closed)."""
        if not isinstance(variant_pool, dict):
            raise TypeError(
                f"isaacsim INIT variant_pool must be a dict, got {type(variant_pool).__name__}"
            )
        target = str(variant_pool.get("target_entity") or "")
        source_files = [os.fspath(path) for path in (variant_pool.get("source_files") or [])]
        assignments = [int(value) for value in (variant_pool.get("assignments") or [])]
        raw_masses = variant_pool.get("masses")
        masses = None if raw_masses is None else [float(value) for value in raw_masses]
        entries = {str(entry.get("name")): entry for entry in entity_payloads}
        target_entry = entries.get(target)
        if target_entry is None:
            raise ValueError(
                f"isaacsim INIT variant pool target {target!r} is not one of the declared "
                f"entities: {sorted(entries)}"
            )
        if (
            target_entry.get("materialization") != "rigid"
            or target_entry.get("root_mode") != "floating"
        ):
            raise ValueError(
                f"isaacsim INIT variant pool target {target!r} must be a floating rigid "
                f"entity, got materialization={target_entry.get('materialization')!r} "
                f"root_mode={target_entry.get('root_mode')!r}"
            )
        if not source_files:
            raise ValueError("isaacsim INIT variant pool requires at least one source file")
        for path in source_files:
            if not os.path.isfile(path):
                raise ValueError(f"isaacsim INIT variant source file does not exist: {path}")
        if len(assignments) != self.num_envs:
            raise ValueError(
                f"isaacsim INIT variant pool assignments have {len(assignments)} entries; "
                f"expected num_envs={self.num_envs}"
            )
        if any(value < 0 or value >= len(source_files) for value in assignments):
            raise ValueError(
                f"isaacsim INIT variant pool assignments must be in [0, {len(source_files)}), "
                f"got {assignments}"
            )
        # Only the deterministic round-robin assignment is materializable:
        # the spawner cycles the K unique prototypes (environment i takes
        # source i % K), so an arbitrary assignment cannot be realized
        # without one prototype reference per environment.
        expected_round_robin = [index % len(source_files) for index in range(self.num_envs)]
        if assignments != expected_round_robin:
            raise NotImplementedError(
                "isaacsim INIT variant pool supports round-robin assignments only "
                f"(assignment[i] == i % {len(source_files)}); an arbitrary per-env "
                "assignment would expand to one prototype per environment — the exact "
                "K-prototype spawner is a planned follow-up"
            )
        if masses is not None:
            if (
                len(masses) != len(source_files)
                or not np.isfinite(masses).all()
                or any(value < 0.0 for value in masses)
            ):
                raise ValueError(
                    "isaacsim INIT variant pool masses must contain one finite non-negative "
                    f"value per source file, got {len(masses)} for {len(source_files)} files"
                )
        result = {
            "target_entity": target,
            "source_files": source_files,
            "assignments": assignments,
        }
        if masses is not None:
            result["masses"] = masses
        return result

    def _materialize_scene_entities(
        self, entity_payloads: list[dict[str, Any]], variant_pool: Any
    ) -> tuple[dict[str, Any], str, list[tuple[str, Any]]]:
        """Convert and prepare spawn configs for a declared multi-asset scene.

        Per-role literal translation of the original repository's scene_utils
        calls: robot articulation (conversion 1721-1725, self-collision
        filters 1726-1729, bake 1730-1739), object pool (conversion
        1701-1706, per-variant bake 1707-1713), table (conversion + bake
        1752-1758), goalviz (object-style conversion, bake 1714-1719 with
        ``collision_enabled=False``).  All role physics (kinematic, gravity,
        self-collision, contact/rest offsets) is authored into the USD by the
        bake, so rigid spawns use plain ``UsdFileCfg``/``MultiUsdFileCfg``
        exactly like the original ``build_rigid_object_cfg``
        (scene_utils.py:187-192).  Returns
        ``(robot_entry, robot_usd_path, [(entity_name, RigidObjectCfg), ...])``.
        """
        from isaaclab.assets import RigidObjectCfg  # type: ignore[import-not-found]
        from isaaclab.sim.spawners.from_files import UsdFileCfg  # type: ignore[import-not-found]
        from isaaclab.sim.spawners.wrappers import (  # type: ignore[import-not-found]
            MultiUsdFileCfg,
        )

        pool = None
        if variant_pool is not None:
            pool = self._validate_variant_pool_payload(variant_pool, entity_payloads)

        robot_entry: dict[str, Any] | None = None
        robot_usd_path = ""
        rigid_spawn_cfgs: list[tuple[str, Any]] = []
        for entry in entity_payloads:
            name = self._validate_entity_name(entry)
            materialization = str(entry.get("materialization") or "")
            root_mode = str(entry.get("root_mode") or "")
            # Declared composition fields are validated at the wire boundary
            # before any conversion runs (fail-closed on bad types).
            declared_collision = _entity_declared_bool(entry, "collision_enabled")
            mirrors_pool = _entity_declared_bool(entry, "mirrors_fixed_variant_pool")
            if str(entry.get("asset_format") or "") != "urdf":
                raise NotImplementedError(
                    f"isaacsim multi-asset entity {name!r} requires asset_format='urdf'; "
                    f"got {entry.get('asset_format')!r}"
                )
            if materialization == "articulation":
                if root_mode == "kinematic":
                    raise ValueError(
                        f"isaacsim entity {name!r}: kinematic articulations are unsupported"
                    )
                if robot_entry is not None:
                    raise ValueError(
                        "isaacsim multi-asset scenes must declare exactly one articulation "
                        f"entity (the robot); found both {robot_entry['name']!r} and {name!r}"
                    )
                usd_path = self._convert_entity_urdf(
                    os.fspath(entry["model_file"]),
                    fix_base=root_mode == "fixed",
                    self_collision=True,  # scene_utils.py:1721-1725
                    with_joint_drive=True,  # scene_utils.py:1453-1458
                    replace_cylinders_with_capsules=_entity_capsule_flag(entry),
                )
                _patch_urdf_articulation_root(usd_path, str(entry.get("root_body_name") or ""))
                # scene_utils.py:1726-1729: author FilteredPairsAPI for the
                # adjacent-link pairs before the bake (PhysX additionally
                # auto-filters directly-jointed parent/child links).
                _apply_self_collision_filters(
                    usd_path, compute_adjacent_link_pairs(os.fspath(entry["model_file"]))
                )
                _bake_usd_in_place(
                    usd_path,
                    bake_plan_for_entity(materialization, root_mode),
                )
                entry["_usd_path"] = usd_path
                robot_entry, robot_usd_path = entry, usd_path
                continue
            if materialization != "rigid":
                raise ValueError(
                    f"isaacsim entity {name!r} has unsupported materialization {materialization!r}"
                )
            if root_mode == "fixed":
                raise NotImplementedError(
                    f"isaacsim rigid entity {name!r}: root_mode='fixed' (world-welded "
                    "rigid objects) is not materialized; use a fixed-base articulation"
                )
            if pool is not None and pool["target_entity"] == name:
                # Round-robin tool pool (scene_utils.py:187-192
                # build_rigid_object_cfg semantics with random_choice=False):
                # the wrapper cycles the unique pool deterministically for the
                # task plan; arbitrary external assignments use the expanded
                # per-environment fallback below.  Pool variants convert with
                # the target entity's declared converter profile.
                pool_capsules = _entity_capsule_flag(entry)
                pool_usds = [
                    self._convert_entity_urdf(
                        source,
                        fix_base=False,
                        self_collision=None,
                        with_joint_drive=False,
                        replace_cylinders_with_capsules=pool_capsules,
                    )
                    for source in pool["source_files"]
                ]
                # Per-variant bake, scene_utils.py:1707-1713.
                for pool_usd in pool_usds:
                    _bake_usd_in_place(
                        pool_usd,
                        bake_plan_for_entity(materialization, root_mode),
                    )
                # Backend-authoritative mass measurement of the baked pool:
                # the INIT payload never carries masses, so the worker reads
                # each variant's physics mass back from the materialized
                # USD, failing INIT closed on any ambiguity.  Only this
                # dynamic object pool is measured; the mirror pool below
                # bakes kinematic non-physical copies.
                measured_masses = _readback_variant_masses(pool_usds)
                entry["_usd_path"] = pool_usds
                self._variant_pool_usds = (name, pool_usds, measured_masses)
                # scene_utils.py:187-192: plain MultiUsdFileCfg round-robin.
                # The wire validation above admitted the deterministic
                # round-robin assignment only, and ``random_choice=False``
                # materializes environment i from pool_usds[i % K] by
                # construction — K unique prototypes regardless of the
                # environment count, never one prototype per environment.
                # All physics is authored by the bake above.
                spawn = MultiUsdFileCfg(usd_path=pool_usds, random_choice=False)
            elif root_mode == "kinematic":
                # A kinematic entity may declare itself the pool's visual
                # mirror: every env's copy shows that env's tool shape
                # (the original bakes goalviz copies from the SAME per-tool
                # USD batch, scene_utils.py:1714-1719).  The mirror bakes
                # kinematic and non-physical; its collision flag and
                # converter profile are its own declaration.
                if mirrors_pool:
                    if pool is None:
                        raise ValueError(
                            f"isaacsim entity {name!r} declares "
                            "mirrors_fixed_variant_pool but the INIT payload carries "
                            "no variant pool"
                        )
                    if pool["target_entity"] == name:
                        raise ValueError(
                            f"isaacsim entity {name!r} declares "
                            "mirrors_fixed_variant_pool but is itself the pool target"
                        )
                    mirror_usds = [
                        self._convert_entity_urdf(
                            source,
                            fix_base=False,
                            self_collision=None,
                            with_joint_drive=False,
                            replace_cylinders_with_capsules=_entity_capsule_flag(entry),
                        )
                        for source in pool["source_files"]
                    ]
                    for mirror_usd in mirror_usds:
                        _bake_usd_in_place(
                            mirror_usd,
                            bake_plan_for_entity(
                                materialization,
                                root_mode,
                                collision_enabled=declared_collision,
                            ),
                        )
                    entry["_usd_path"] = mirror_usds
                    self._goalviz_mirror = (name, str(pool["target_entity"]), mirror_usds)
                    # Same by-construction round-robin as the pool target:
                    # the mirror cycles the same K unique sources per env.
                    spawn = MultiUsdFileCfg(usd_path=mirror_usds, random_choice=False)
                else:
                    # Single-file kinematic entity (the table contract):
                    # converter profile and collision flag are declarations,
                    # with the materialization defaults keeping the
                    # converted-USD collision state and the converter's
                    # capsule behavior.
                    usd_path = self._convert_entity_urdf(
                        os.fspath(entry["model_file"]),
                        fix_base=False,
                        self_collision=None,
                        with_joint_drive=False,
                        replace_cylinders_with_capsules=_entity_capsule_flag(entry),
                    )
                    _bake_usd_in_place(
                        usd_path,
                        bake_plan_for_entity(
                            materialization,
                            root_mode,
                            collision_enabled=declared_collision,
                        ),
                    )
                    entry["_usd_path"] = usd_path
                    spawn = UsdFileCfg(usd_path=usd_path)
            else:
                # Floating rigid (object role): author dynamic rigid-body
                # physics. Table and goalviz use the explicit kinematic mode.
                # The object converts with capsule replacement like the pool
                # variants (scene_utils.py:1701-1706 applies to every object
                # conversion, pooled or not).
                usd_path = self._convert_entity_urdf(
                    os.fspath(entry["model_file"]),
                    fix_base=False,
                    self_collision=None,
                    with_joint_drive=False,
                    replace_cylinders_with_capsules=_entity_capsule_flag(entry),
                )
                _bake_usd_in_place(
                    usd_path,
                    bake_plan_for_entity(materialization, root_mode),
                )
                entry["_usd_path"] = usd_path
                spawn = UsdFileCfg(usd_path=usd_path)
            rigid_spawn_cfgs.append(
                (name, RigidObjectCfg(prim_path=f"/World/envs/env_.*/{name}", spawn=spawn))
            )
        if robot_entry is None:
            raise ValueError(
                "isaacsim multi-asset scenes must declare one articulation entity (the robot)"
            )
        if pool is not None and self._variant_pool_usds is None:
            raise ValueError(
                f"isaacsim INIT variant pool target {pool['target_entity']!r} was declared "
                "but never consumed during entity materialization"
            )
        return robot_entry, robot_usd_path, rigid_spawn_cfgs

    def _entity_prim_counts(self, entity_payloads: list[dict[str, Any]]) -> dict[str, int]:
        """Count materialized prims per entity across envs (cold forensics)."""
        stage = self.robot.stage
        counts: dict[str, int] = {}
        for entry in entity_payloads:
            name = str(entry["name"])
            count = 0
            for env_index in range(self.num_envs):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/{name}")
                if prim and prim.IsValid():
                    count += 1
            counts[name] = count
        return counts

    def _observe_variant_assignment(
        self, target_name: str, pool_usd_paths: list[str]
    ) -> list[int] | None:
        """Infer each env's spawned variant from stage prim stacks (probe C).

        Port of probe_multi_asset_spawn.py:_infer_tool_variant.  Returns
        ``None`` when this Kit build flattens the references so deeply that no
        pool layer identifier survives on the prim stack.  Full prim-stack
        inspection is intentionally limited to the small probe regime: at
        production-scale environment counts, walking every cloned USD prim
        here can dominate INIT (and provides no runtime value after the
        immutable assignment has already been validated at the wire boundary).
        """
        from pxr import Usd  # type: ignore[import-not-found]

        if self.num_envs > 4096:
            return None

        stage = self.robot.stage
        basenames = [os.path.basename(path) for path in pool_usd_paths]
        if len(set(basenames)) != len(basenames):
            return None
        observed: list[int] = []
        for env_index in range(self.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/{target_name}")
            if not prim or not prim.IsValid():
                return None
            found: int | None = None
            for sub_prim in Usd.PrimRange(prim):
                for spec in sub_prim.GetPrimStack():
                    layer_base = os.path.basename(spec.layer.identifier)
                    for index, needle in enumerate(basenames):
                        if layer_base == needle:
                            found = index
            if found is None:
                return None
            observed.append(found)
        return observed

    def _readback_actuator_gains(self) -> dict[str, list[float]]:
        """Read the materialized ImplicitActuator gains (env 0, contract order)."""
        actuator = self.robot.actuators["all"]

        def env0(values: Any) -> np.ndarray:
            array = _tensor_numpy(values)
            if array.ndim == 1:
                return array
            if array.ndim == 2 and array.shape[0] == self.num_envs:
                return array[0]
            raise RuntimeError(
                f"IsaacLab actuator gain tensor has shape {array.shape}; expected "
                f"({self.num_envs}, {self.num_dof}) or ({self.num_dof},)"
            )

        stiffness = env0(actuator.stiffness)[self.native_joint_for_contract]
        damping = env0(actuator.damping)[self.native_joint_for_contract]
        report: dict[str, list[float]] = {
            "stiffness": [float(value) for value in stiffness],
            "damping": [float(value) for value in damping],
        }
        # Armature and effort limits are applied through the same override
        # channel; record their effective values so probes can close the
        # readback loop.
        armature = getattr(actuator, "armature", None)
        effort = getattr(actuator, "effort_limit_sim", None)
        for name, values in (("armature", armature), ("effort_limit", effort)):
            if values is None:
                raise RuntimeError(
                    f"IsaacLab actuator does not expose {name!r}; cannot read back gains"
                )
            report[name] = [float(value) for value in env0(values)[self.native_joint_for_contract]]
        # Diagnostic (2026-09-14 lift-latch investigation): also read the LIVE
        # PhysX drive values back so probes can distinguish "cfg tensors look
        # right" from "implicit gains actually landed in the sim".  Pure
        # readback; no behavior change.
        view = getattr(self.robot, "root_physx_view", None)
        if view is not None and hasattr(view, "get_dof_stiffnesses"):

            def _live(getter: str) -> np.ndarray | None:
                if not hasattr(view, getter):
                    return None
                array = _tensor_numpy(getattr(view, getter)())
                if array.ndim == 2:
                    array = array[0]
                return array

            live_stiffness = _live("get_dof_stiffnesses")
            live_damping = _live("get_dof_dampings")
            if live_stiffness is not None:
                report["physx_stiffness_env0_native"] = [float(value) for value in live_stiffness]
                report["physx_stiffness_env0"] = [
                    float(value) for value in live_stiffness[self.native_joint_for_contract]
                ]
            if live_damping is not None:
                report["physx_damping_env0_native"] = [float(value) for value in live_damping]
                report["physx_damping_env0"] = [
                    float(value) for value in live_damping[self.native_joint_for_contract]
                ]
            # Position targets are write-only in this PhysX tensor API; fall
            # back to IsaacLab's own commanded-target buffer (what the
            # articulation last pushed to the sim).
            try:
                live_targets = _tensor_numpy(self.robot.data.joint_pos_target)
                if live_targets.ndim == 2:
                    live_targets = live_targets[0]
                report["physx_targets_env0"] = [
                    float(value) for value in live_targets[self.native_joint_for_contract]
                ]
            except (AttributeError, NotImplementedError):
                pass
        return report

    def _apply_friction_writes(self, entity_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Write INIT-declared contact materials through the PhysX views.

        Port of scene_utils.py:1546-1624 (``apply_physx_material_properties``):
        the default triple is tiled across every shape, per-body overrides
        overwrite their link's shape slice, the write goes through
        ``set_material_properties`` with an all-envs int64 CPU index tensor
        (1566), and the view is read back immediately — any mismatch fails
        INIT.  The per-env bucketed friction DR (1594-1606, 1615-1621) is not
        migrated.  Returns the per-entity forensic summary for INIT meta.
        """
        torch = self.torch
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device="cpu")
        report: dict[str, Any] = {}
        for entry in entity_payloads:
            parsed = parse_entity_friction(entry)
            if parsed is None:
                continue
            default, overrides = parsed
            name = str(entry["name"])
            if str(entry["materialization"]) == "articulation":
                asset = self.robot
            elif name in self.rigid_objects:
                asset = self.rigid_objects[name]
            else:
                raise RuntimeError(
                    f"isaacsim entity {name!r} declares contact friction but no "
                    "materialized asset owns it (articulation robot or rigid object)"
                )
            view = asset.root_physx_view
            if overrides:
                # Per-link shape counts, scene_utils.py:1579-1587 literal.
                link_names = [str(link) for link in view.shared_metatype.link_names]
                link_paths = [str(path) for path in view.link_paths[0]]
                link_shape_counts = [
                    int(asset._physics_sim_view.create_rigid_body_view(path).max_shapes)
                    for path in link_paths
                ]
                table = build_friction_shape_table(
                    link_names, link_shape_counts, default, overrides
                )
                if table.shape[0] != int(view.max_shapes):
                    raise RuntimeError(
                        f"isaacsim entity {name!r} shape count mismatch while assigning "
                        f"materials: computed {table.shape[0]}, view reports "
                        f"{int(view.max_shapes)} (scene_utils.py:1588-1592)"
                    )
            else:
                table = np.tile(np.asarray(default, dtype=np.float32), (int(view.max_shapes), 1))
            materials = view.get_material_properties()
            materials[:] = _to_tensor(torch, table, "cpu")
            view.set_material_properties(materials, env_ids)
            readback = _tensor_numpy(view.get_material_properties())
            expected = np.broadcast_to(table, readback.shape)
            if not np.allclose(readback, expected, rtol=1e-5, atol=1e-6):
                mismatch = int((~np.isclose(readback, expected, rtol=1e-5, atol=1e-6)).sum())
                raise RuntimeError(
                    f"isaacsim entity {name!r} contact material readback mismatch: "
                    f"{mismatch} components differ after set_material_properties; "
                    "failing INIT closed"
                )
            unique = np.unique(readback.reshape(-1, 3), axis=0)
            report[name] = {
                "default": [float(value) for value in default],
                "overrides": {
                    body: [float(value) for value in triple] for body, triple in overrides.items()
                },
                "num_shapes": int(view.max_shapes),
                "unique_values": [[float(component) for component in row] for row in unique],
                "verified": True,
            }
        return report

    def _readback_bake(self, entity_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        """Re-open every baked USD and verify the authored plan (fail-closed).

        Runs the pxr readback inside INIT so probes stay backend-level: each
        entity's bake plan attributes, contact/rest offsets, and (goalviz)
        collisionEnabled are checked on the composed stage; the robot's
        FilteredPairs are checked against the URDF-derived adjacency.
        """
        pool_target = None if self._variant_pool_usds is None else self._variant_pool_usds[0]
        mirror_target = None if self._goalviz_mirror is None else self._goalviz_mirror[0]
        report: dict[str, Any] = {}
        for entry in entity_payloads:
            name = str(entry["name"])
            is_pool_target = name == pool_target
            # Every floating rigid role is dynamic, whether it is backed by a
            # per-env variant pool or by the scene's single bootstrap asset.
            # ``is_variant_target`` is the bake-plan switch retained for the
            # original pool call site, so mirror the materialization branch
            # here to keep readback fail-closed for the non-pool object too.
            is_dynamic_rigid = (
                str(entry["materialization"]) == "rigid" and str(entry["root_mode"]) == "floating"
            )
            plan = bake_plan_for_entity(
                str(entry["materialization"]),
                str(entry["root_mode"]),
                collision_enabled=_entity_declared_bool(entry, "collision_enabled"),
                is_variant_target=is_pool_target or is_dynamic_rigid,
            )
            if is_pool_target:
                usd_paths = sorted(set(self._variant_pool_usds[1]))
            elif name == mirror_target:
                usd_paths = sorted(set(self._goalviz_mirror[2]))
            else:
                usd_paths = [str(entry.get("_usd_path") or "")]
            variants = [{"usd_path": path, **_verify_baked_usd(path, plan)} for path in usd_paths]
            entity_report: dict[str, Any] = (
                variants[0] if len(variants) == 1 else {"variants": variants}
            )
            if str(entry["materialization"]) == "articulation":
                entity_report["self_collision_filter"] = _verify_self_collision_filters(
                    usd_paths[0],
                    compute_adjacent_link_pairs(os.fspath(entry["model_file"])),
                )
            report[name] = entity_report
        return report

    def _joint_limits(self) -> tuple[list[float], list[float], list[float]]:
        limits = _tensor_numpy(self.robot.data.joint_pos_limits)[0]
        efforts = _tensor_numpy(self.robot.data.joint_effort_limits)[0]
        # Reorder native metadata to the public contract order.
        limits = limits[self.native_joint_for_contract]
        efforts = efforts[self.native_joint_for_contract]
        return limits[:, 0].tolist(), limits[:, 1].tolist(), efforts.tolist()

    def _apply_keyframe(self, qpos_values: Any) -> None:
        qpos = np.asarray(qpos_values, dtype=np.float32).reshape(-1)
        if qpos.size != 7 + self.num_dof:
            raise RuntimeError(
                f"keyframe qpos has {qpos.size} entries; expected {7 + self.num_dof}"
            )
        env_ids = self.torch.arange(self.num_envs, dtype=self.torch.long, device=self.device)
        root_pose_np = np.broadcast_to(qpos[:7], (self.num_envs, 7)).copy()
        root_pose_np[:, :3] += self.env_origins
        root_pose = _to_tensor(self.torch, root_pose_np, self.device)
        root_vel = self.torch.zeros(
            (self.num_envs, 6), dtype=self.torch.float32, device=self.device
        )
        native_pos = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        native_pos[:, self.native_joint_for_contract] = qpos[7:][None, :]
        joint_pos = _to_tensor(self.torch, native_pos, self.device)
        joint_vel = self.torch.zeros_like(joint_pos)
        if not self._fixed_base:
            self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            # UniLab's root state is the link-frame state.  IsaacLab's similarly
            # named ``write_root_velocity_to_sim`` targets the COM frame, so use
            # the explicit link writer here.
            self.robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.sim_dt)

    # ------------------------------------------------------------------
    # Shared-memory attachment and state exchange
    # ------------------------------------------------------------------

    def attach_slots(self, payload: dict[str, Any]) -> None:
        from multiprocessing import resource_tracker, shared_memory

        for name, spec in payload["slots"].items():
            handle = shared_memory.SharedMemory(name=spec["shm"], create=False)
            # The host owns unlinking; prevent the worker's resource tracker
            # from unlinking the segment when Kit exits.
            resource_tracker.unregister(handle._name, "shared_memory")  # type: ignore[attr-defined]
            self.slots[name] = np.ndarray(
                tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]), buffer=handle.buf
            )
            self._shm_handles.append(handle)
        # Fail closed when the host's slot layout and the worker's materialized
        # rigid entities disagree (1.3c: one state+reset slot pair per entity).
        expected_entity_slots = {
            slot
            for name in self.rigid_objects
            for slot in (
                self.protocol.entity_root_state_slot(name),
                self.protocol.entity_reset_state_slot(name),
            )
        }
        present_entity_slots = {
            name
            for name in self.slots
            if name.startswith(self.protocol.ENTITY_ROOT_STATE_SLOT_PREFIX)
            or name.startswith(self.protocol.ENTITY_RESET_STATE_SLOT_PREFIX)
        }
        if present_entity_slots != expected_entity_slots:
            raise RuntimeError(
                "isaacsim ATTACH_SLOTS entity slot mismatch: worker materialized rigid "
                f"entities {sorted(self.rigid_objects)} expecting slots "
                f"{sorted(expected_entity_slots)}, host sent {sorted(present_entity_slots)}"
            )
        if self.rigid_objects:
            for slot_name in (
                self.protocol.WRENCH_FORCE_SLOT,
                self.protocol.WRENCH_TORQUE_SLOT,
            ):
                if slot_name not in self.slots:
                    raise RuntimeError(
                        f"isaacsim ATTACH_SLOTS missing required multi-asset wrench slot "
                        f"{slot_name!r}"
                    )
                expected = (self.num_envs, self.num_bodies + len(self.rigid_objects), 3)
                if self.slots[slot_name].shape != expected:
                    raise RuntimeError(
                        f"isaacsim wrench slot {slot_name!r} has shape "
                        f"{self.slots[slot_name].shape}; expected {expected}"
                    )
        for name in sorted(expected_entity_slots):
            slot = self.slots[name]
            if slot.shape != (self.num_envs, 13):
                raise RuntimeError(
                    f"isaacsim entity slot {name!r} has shape {slot.shape}; "
                    f"expected ({self.num_envs}, 13)"
                )
        self.refresh_state_slots()

    def _state_tensors(
        self,
        env_ids: np.ndarray | None = None,
        timing: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.robot.data
        if env_ids is None:
            row_index = None
            row_count = self.num_envs
        else:
            ids = np.asarray(env_ids, dtype=np.int64)
            if ids.ndim != 1:
                raise ValueError(f"state refresh environment ids must be 1-D, got {ids.shape}")
            if np.unique(ids).size != ids.size:
                raise ValueError("state refresh environment ids must not contain duplicates")
            if np.any(ids < 0) or np.any(ids >= self.num_envs):
                raise ValueError("state refresh environment ids are out of range")
            row_index = self.torch.as_tensor(ids, dtype=self.torch.long, device=self.device)
            row_count = int(ids.size)

        def select_rows(value: Any) -> Any:
            if row_index is None:
                return value
            return value.index_select(0, row_index)

        # Select rows while they are still device tensors. The full path stays
        # equivalent; the selected path avoids transferring untouched rows
        # across the IsaacSim/NumPy boundary.
        t0 = time.perf_counter()
        root = _tensor_numpy(select_rows(data.root_link_state_w))
        if timing is not None:
            timing["refresh_root_tensor_to_host_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        dof_pos = _tensor_numpy(select_rows(data.joint_pos))
        if timing is not None:
            timing["refresh_joint_pos_tensor_to_host_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        dof_vel = _tensor_numpy(select_rows(data.joint_vel))
        if timing is not None:
            timing["refresh_joint_vel_tensor_to_host_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        body = _tensor_numpy(select_rows(data.body_link_state_w))
        if timing is not None:
            timing["refresh_body_tensor_to_host_ms"] = (time.perf_counter() - t0) * 1000.0
        if root.shape != (row_count, 13):
            raise RuntimeError(
                f"IsaacLab root state shape is {root.shape}, expected ({row_count}, 13)"
            )
        # Return root, dof(pos/vel), body separately; body is reordered below.
        t0 = time.perf_counter()
        dof = np.stack((dof_pos, dof_vel), axis=-1)
        if timing is not None:
            timing["refresh_joint_stack_ms"] = (time.perf_counter() - t0) * 1000.0
        return root, dof, body

    def refresh_state_slots(self, env_ids: np.ndarray | None = None) -> None:
        detail_timing: dict[str, float] | None = {} if self._profile_detail else None
        refresh_t0 = time.perf_counter()
        root, dof, body = self._state_tensors(env_ids, detail_timing)
        selected = env_ids is not None
        ids = None if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        row_index = (
            None
            if ids is None
            else self.torch.as_tensor(ids, dtype=self.torch.long, device=self.device)
        )
        # IsaacLab reports world-frame positions.  Remove the private clone
        # translation before publishing UniLab's local-frame state.
        t0 = time.perf_counter()
        root = root.copy()
        body = body.copy()
        origins = self.env_origins if ids is None else self.env_origins[ids]
        root[:, :3] -= origins
        body[:, :, :3] -= origins[:, None, :]
        if detail_timing is not None:
            detail_timing["refresh_robot_frame_normalize_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        if selected:
            assert ids is not None
            self.slots["root_state"][ids] = root
            self.slots["dof_state"][ids] = dof[:, self.native_joint_for_contract, :]
            self.slots["body_state"][ids] = body[:, self.native_body_for_contract, :]
        else:
            np.copyto(self.slots["root_state"], root)
            np.copyto(self.slots["dof_state"], dof[:, self.native_joint_for_contract, :])
            np.copyto(self.slots["body_state"], body[:, self.native_body_for_contract, :])
        if detail_timing is not None:
            detail_timing["refresh_robot_slot_copy_ms"] = (time.perf_counter() - t0) * 1000.0
        # IsaacLab's Articulation tensor does not expose a generic net-contact
        # force slot.  Keep the slot deterministic and let the host sensor map
        # fail closed for contact declarations.
        t0 = time.perf_counter()
        if selected:
            assert ids is not None
            self.slots["contact_force"][ids] = 0.0
        else:
            self.slots["contact_force"].fill(0.0)
        if detail_timing is not None:
            detail_timing["refresh_contact_slot_clear_ms"] = (time.perf_counter() - t0) * 1000.0
        # Rigid scene entities (1.3c): publish each root's world state (pos
        # xyz, quat wxyz, lin vel, world ang vel) in local frame.  The loop is
        # empty for scenes without rigid entities.
        for name, rigid in self.rigid_objects.items():
            rigid_t0 = time.perf_counter()
            rigid_state = rigid.data.root_link_state_w
            if row_index is not None:
                rigid_state = rigid_state.index_select(0, row_index)
            state = _tensor_numpy(rigid_state)
            if detail_timing is not None:
                detail_timing[f"refresh_rigid_{name}_tensor_to_host_ms"] = (
                    time.perf_counter() - rigid_t0
                ) * 1000.0
            row_count = self.num_envs if ids is None else int(ids.size)
            expected = (row_count, 13)
            if state.shape != expected:
                raise RuntimeError(
                    f"IsaacLab rigid entity {name!r} root state shape is {state.shape}, "
                    f"expected {expected}"
                )
            rigid_t0 = time.perf_counter()
            state = state.copy()
            state[:, :3] -= origins
            if not np.isfinite(state).all():
                raise RuntimeError(
                    f"IsaacLab rigid entity {name!r} root state contains NaN or Inf; "
                    "refusing to publish non-finite state"
                )
            if selected:
                assert ids is not None
                self.slots[self.protocol.entity_root_state_slot(name)][ids] = state
            else:
                np.copyto(self.slots[self.protocol.entity_root_state_slot(name)], state)
            if detail_timing is not None:
                detail_timing[f"refresh_rigid_{name}_normalize_slot_ms"] = (
                    time.perf_counter() - rigid_t0
                ) * 1000.0
        if detail_timing is not None:
            detail_timing["refresh_total_ms"] = (time.perf_counter() - refresh_t0) * 1000.0
            self._last_refresh_timing_ms = detail_timing

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        worker_step_t0 = time.perf_counter()
        control_t0 = time.perf_counter()
        ctrl = np.asarray(self.slots["ctrl"], dtype=np.float32)
        if ctrl.shape != (self.num_envs, self.num_dof):
            raise ValueError(
                f"ctrl slot has shape {ctrl.shape}; expected {(self.num_envs, self.num_dof)}"
            )
        native_target = np.zeros_like(ctrl)
        native_target[:, self.native_joint_for_contract] = ctrl
        target = _to_tensor(self.torch, native_target, self.device)
        self.robot.set_joint_position_target(target)
        control_prepare_ms = (time.perf_counter() - control_t0) * 1000.0
        nsteps = int(payload["nsteps"])
        if nsteps <= 0:
            raise ValueError(f"nsteps must be positive, got {nsteps}")
        detail_timing: dict[str, float] = {}
        t0 = time.perf_counter()
        if self._profile_detail:
            phase_sums = {
                "physics_wrench_stage_ms": 0.0,
                "physics_rigid_write_ms": 0.0,
                "physics_robot_write_ms": 0.0,
                "physics_sim_step_ms": 0.0,
                "physics_robot_update_ms": 0.0,
                "physics_rigid_update_ms": 0.0,
            }
            for _ in range(nsteps):
                phase_t0 = time.perf_counter()
                self._stage_pending_wrenches()
                phase_sums["physics_wrench_stage_ms"] += (time.perf_counter() - phase_t0) * 1000.0
                phase_t0 = time.perf_counter()
                for rigid in self.rigid_objects.values():
                    rigid.write_data_to_sim()
                phase_sums["physics_rigid_write_ms"] += (time.perf_counter() - phase_t0) * 1000.0
                phase_t0 = time.perf_counter()
                self.robot.write_data_to_sim()
                phase_sums["physics_robot_write_ms"] += (time.perf_counter() - phase_t0) * 1000.0
                phase_t0 = time.perf_counter()
                self.sim.step(render=False)
                phase_sums["physics_sim_step_ms"] += (time.perf_counter() - phase_t0) * 1000.0
                phase_t0 = time.perf_counter()
                self.robot.update(self.sim_dt)
                phase_sums["physics_robot_update_ms"] += (time.perf_counter() - phase_t0) * 1000.0
                phase_t0 = time.perf_counter()
                for rigid in self.rigid_objects.values():
                    rigid.update(self.sim_dt)
                phase_sums["physics_rigid_update_ms"] += (time.perf_counter() - phase_t0) * 1000.0
            detail_timing.update(phase_sums)
        else:
            for _ in range(nsteps):
                self._stage_pending_wrenches()
                for rigid in self.rigid_objects.values():
                    rigid.write_data_to_sim()
                self.robot.write_data_to_sim()
                self.sim.step(render=False)
                self.robot.update(self.sim_dt)
                for rigid in self.rigid_objects.values():
                    rigid.update(self.sim_dt)
        physics_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self.refresh_state_slots()
        refresh_ms = (time.perf_counter() - t0) * 1000.0
        if self._profile_detail:
            detail_timing.update(self._last_refresh_timing_ms)
        for slot_name in (
            self.protocol.WRENCH_FORCE_SLOT,
            self.protocol.WRENCH_TORQUE_SLOT,
        ):
            if slot_name in self.slots:
                self.slots[slot_name].fill(0.0)
        timing = {
            "control_upload_ms": 0.0,
            "physics_ms": physics_ms,
            "state_refresh_ms": refresh_ms,
        }
        if self._profile_detail:
            timing["worker_control_prepare_ms"] = control_prepare_ms
            timing.update(detail_timing)
            timing["worker_step_total_ms"] = (time.perf_counter() - worker_step_t0) * 1000.0
        return {"timing": timing}

    def _stage_pending_wrenches(self) -> None:
        """Copy dense public rigid-root rows into IsaacLab wrench buffers."""
        if not self.rigid_objects:
            return
        force = np.asarray(self.slots[self.protocol.WRENCH_FORCE_SLOT], dtype=np.float32)
        torque = np.asarray(self.slots[self.protocol.WRENCH_TORQUE_SLOT], dtype=np.float32)
        expected = (self.num_envs, self.num_bodies + len(self.rigid_objects), 3)
        if force.shape != expected or torque.shape != expected:
            raise ValueError(f"isaacsim wrench slots must have shape {expected}")
        for index, (_, rigid) in enumerate(self.rigid_objects.items()):
            # SimToolReal rigid assets are single-root bodies. Keep the slot
            # contract on the logical root and let IsaacLab broadcast it to
            # the one-body RigidObject view.
            body_id = self.num_bodies + index
            rigid.set_external_force_and_torque(
                _to_tensor(self.torch, force[:, body_id : body_id + 1, :], self.device),
                _to_tensor(self.torch, torque[:, body_id : body_id + 1, :], self.device),
                is_global=True,
            )

    def set_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        set_state_t0 = time.perf_counter()
        count = int(payload["count"])
        if count < 0 or count > self.num_envs:
            raise ValueError(f"reset count must be in [0, {self.num_envs}], got {count}")
        env_ids_np = np.asarray(self.slots["reset_env_ids"][:count], dtype=np.int64)
        if np.unique(env_ids_np).size != env_ids_np.size:
            raise ValueError("reset environment ids must not contain duplicates")
        if np.any(env_ids_np < 0) or np.any(env_ids_np >= self.num_envs):
            raise ValueError("reset environment ids are out of range")
        # One SET_STATE command is one transaction: the robot's generalized
        # state and any rigid entity roots are written against the same env
        # rows.  ``robot=False`` (multi-asset scenes only) leaves the robot
        # untouched, so a goalviz-only reset cannot perturb object/robot state
        # (DESIGN.md §4).  Payloads carrying neither key write the robot.
        robot_write = bool(payload.get("robot", True))
        entity_names = [str(name) for name in (payload.get("entity_roots") or [])]
        if len(set(entity_names)) != len(entity_names):
            raise ValueError(f"reset entity_roots must not contain duplicates: {entity_names}")
        unknown = sorted(set(entity_names) - set(self.rigid_objects))
        if unknown:
            raise ValueError(
                f"reset entity_roots must name materialized rigid entities "
                f"{sorted(self.rigid_objects)}, got {unknown}"
            )
        if not robot_write and not entity_names:
            raise ValueError("reset transaction writes nothing: robot=False and no entity_roots")
        t0 = time.perf_counter()
        env_ids = self.torch.as_tensor(env_ids_np, dtype=self.torch.long, device=self.device)
        env_id_tensor_ms = (time.perf_counter() - t0) * 1000.0
        robot_write_ms = 0.0
        entity_write_ms = 0.0
        if robot_write:
            qpos = np.asarray(self.slots["reset_qpos"][:count], dtype=np.float32)
            qvel = np.asarray(self.slots["reset_qvel"][:count], dtype=np.float32)
            expected_qpos = (count, 7 + self.num_dof)
            expected_qvel = (count, 6 + self.num_dof)
            if qpos.shape != expected_qpos:
                raise ValueError(f"reset qpos has shape {qpos.shape}; expected {expected_qpos}")
            if qvel.shape != expected_qvel:
                raise ValueError(f"reset qvel has shape {qvel.shape}; expected {expected_qvel}")
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError("reset qpos/qvel must be finite (no NaN or Inf)")
            native_pos = np.zeros((count, self.num_dof), dtype=np.float32)
            native_vel = np.zeros_like(native_pos)
            native_pos[:, self.native_joint_for_contract] = qpos[:, 7 : 7 + self.num_dof]
            native_vel[:, self.native_joint_for_contract] = qvel[:, 6 : 6 + self.num_dof]
            t0 = time.perf_counter()
            if not self._fixed_base:
                root_pose_np = qpos[:, :7].copy()
                root_pose_np[:, :3] += self.env_origins[env_ids_np]
                root_pose = _to_tensor(self.torch, root_pose_np, self.device)
                root_velocity_np = np.empty((count, 6), dtype=np.float32)
                root_velocity_np[:, :3] = qvel[:, :3]
                root_velocity_np[:, 3:] = _quat_rotate_wxyz(qpos[:, 3:7], qvel[:, 3:6])
                self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                self.robot.write_root_link_velocity_to_sim(
                    _to_tensor(self.torch, root_velocity_np, self.device), env_ids=env_ids
                )
            self.robot.write_joint_state_to_sim(
                _to_tensor(self.torch, native_pos, self.device),
                _to_tensor(self.torch, native_vel, self.device),
                env_ids=env_ids,
            )
            self.robot.reset(env_ids)
            self.robot.update(self.sim_dt)
            robot_write_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        for name in entity_names:
            # Slot layout matches the read direction: pos xyz (local frame),
            # quat wxyz, world linear velocity, world angular velocity.
            data = np.asarray(
                self.slots[self.protocol.entity_reset_state_slot(name)][:count],
                dtype=np.float32,
            )
            expected = (count, 13)
            if data.shape != expected:
                raise ValueError(
                    f"reset entity {name!r} root state has shape {data.shape}; expected {expected}"
                )
            if not np.isfinite(data).all():
                raise ValueError(f"reset entity {name!r} root state must be finite (no NaN or Inf)")
            root_pose_np = data[:, :7].copy()
            root_pose_np[:, :3] += self.env_origins[env_ids_np]
            root_velocity_np = np.ascontiguousarray(data[:, 7:13])
            rigid = self.rigid_objects[name]
            rigid.write_root_pose_to_sim(
                _to_tensor(self.torch, root_pose_np, self.device), env_ids=env_ids
            )
            rigid.write_root_link_velocity_to_sim(
                _to_tensor(self.torch, root_velocity_np, self.device), env_ids=env_ids
            )
            rigid.reset(env_ids)
            rigid.update(self.sim_dt)
        entity_write_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        self.refresh_state_slots(env_ids_np)
        refresh_ms = (time.perf_counter() - t0) * 1000.0
        timing = {
            "set_state_reset_upload_ms": 0.0,
            "set_state_host_cache_refresh_ms": refresh_ms,
        }
        if self._profile_detail:
            timing.update(
                {
                    "set_state_reset_upload_ms": robot_write_ms + entity_write_ms,
                    "set_state_env_id_tensor_ms": env_id_tensor_ms,
                    "set_state_robot_write_ms": robot_write_ms,
                    "set_state_entity_write_ms": entity_write_ms,
                    "set_state_refresh_total_ms": refresh_ms,
                    "set_state_worker_total_ms": (time.perf_counter() - set_state_t0) * 1000.0,
                }
            )
            timing.update(
                {
                    f"set_state_{key}": value
                    for key, value in self._last_refresh_timing_ms.items()
                    if key.startswith("refresh_")
                }
            )
        return {"timing": timing}

    def get_meta(self) -> dict[str, Any]:
        return {
            "num_dof": self.num_dof,
            "num_bodies": self.num_bodies,
            "dof_names": list(self.contract_joint_names),
            "body_names": list(self.contract_body_names),
            "gravity": [0.0, 0.0, -9.81],
            "use_gpu_pipeline": True,
            "graphics_enabled": self.render_mode != "none",
            "render_mode": self.render_mode,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "env_origins": self.env_origins.tolist(),
            "collision_filtering_applied": self.collision_filtering_applied,
            # Runtime fixity diagnostics: the payload flag vs IsaacLab's own
            # view of the articulation root.  These must agree and be True for
            # a fixed-base robot.
            "fixed_base": bool(self._fixed_base),
            "robot_is_fixed_base": bool(getattr(self.robot, "is_fixed_base", False)),
        }

    # ------------------------------------------------------------------
    # Native rendering (cold setup + eval/play commands)
    # ------------------------------------------------------------------

    def _require_render_mode(self, expected: str) -> None:
        if self.render_mode != expected:
            raise RuntimeError(
                "isaacsim renderer request is incompatible with the worker startup mode: "
                f"worker={self.render_mode!r}, requested={expected!r}"
            )

    def _camera_view(self) -> tuple[Any, Any]:
        """Return batched eye/target tensors for the spherical tracking view."""
        if self.robot is None:
            raise RuntimeError("isaacsim camera requested before articulation initialization")
        root_pos = self.robot.data.root_pos_w
        if tuple(root_pos.shape) != (self.num_envs, 3):
            raise RuntimeError(
                f"IsaacLab root positions have shape {root_pos.shape}; expected "
                f"({self.num_envs}, 3) for camera tracking"
            )
        elevation = math.radians(self.camera_elevation_deg)
        azimuth = math.radians(self.camera_azimuth_deg)
        offset = self.camera_distance * np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        offset_tensor = _to_tensor(self.torch, offset, self.device)
        targets = root_pos.clone()
        # Aim a little above the pelvis so playback is closer to eye level
        # instead of looking up from below.  Keeping the target above the root
        # also leaves enough vertical margin to keep the feet in frame.
        targets[:, 2] += 0.30
        eyes = targets + offset_tensor[None, :]
        return eyes, targets

    def _set_capture_camera(self) -> None:
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is unavailable; worker was not started in record mode"
            )
        eyes, targets = self._camera_view()
        self.camera.set_world_poses_from_view(eyes[0:1], targets[0:1])

    @staticmethod
    def _app_is_running(app: Any) -> bool:
        """Read the documented SimulationApp lifecycle state."""
        try:
            return bool(app.is_running()) and not bool(app.is_exiting())
        except Exception:
            # A closed Kit app may invalidate the Python proxy before the
            # status methods can be queried. Treat that as a closed window.
            return False

    def init_renderer(self, payload: dict[str, Any]) -> dict[str, Any]:
        headless = bool(payload.get("headless", False))
        capture = bool(payload.get("capture", False))
        requested = "record" if (headless or capture) else "interactive"
        self._require_render_mode(requested)
        width = int(payload.get("width", self.render_width))
        height = int(payload.get("height", self.render_height))
        if width != self.render_width or height != self.render_height:
            raise ValueError(
                "isaacsim renderer dimensions differ from INIT: "
                f"requested={width}x{height}, configured={self.render_width}x{self.render_height}"
            )
        camera = payload.get("camera") or {}
        self.camera_distance = float(camera.get("distance", 2.0))
        self.camera_elevation_deg = float(camera.get("elevation_deg", 20.0))
        self.camera_azimuth_deg = float(camera.get("azimuth_deg", 90.0))
        if (
            not np.isfinite(
                [self.camera_distance, self.camera_elevation_deg, self.camera_azimuth_deg]
            ).all()
            or self.camera_distance <= 0.0
        ):
            raise ValueError(
                "isaacsim camera distance/elevation/azimuth must be finite and distance > 0"
            )

        if requested == "record":
            if not capture:
                raise RuntimeError("isaacsim record renderer requires capture=true")
            self._set_capture_camera()
            # Warm up Hydra/Replicator once on the cold path. Camera buffers
            # are then ready for the first playback frame.
            self.sim.render()
            self.camera.update(self.sim_dt, force_recompute=True)
            return {"viewer": False, "capture": True}

        if headless or capture:
            raise RuntimeError(
                "isaacsim interactive renderer cannot be headless or capture-enabled"
            )
        # Leave the Kit viewport camera under user control.  The interactive
        # viewer must not be re-aimed at the robot during startup or playback.
        self.sim.render()
        return {"viewer": self._app_is_running(self.simulation_app), "capture": False}

    def render_frame(self) -> dict[str, Any]:
        self._require_render_mode("interactive")
        if not self._app_is_running(self.simulation_app):
            return {"closed": True}
        self.sim.render()
        return {"closed": not self._app_is_running(self.simulation_app)}

    def capture_frame(self) -> dict[str, Any]:
        self._require_render_mode("record")
        if self.camera is None:
            raise RuntimeError(
                "isaacsim capture camera is not initialized; call INIT_RENDERER first"
            )
        self._set_capture_camera()
        self.sim.render()
        self.camera.update(self.sim_dt, force_recompute=True)
        output = self.camera.data.output
        if not isinstance(output, dict) or "rgb" not in output:
            raise RuntimeError(
                "IsaacSim camera did not return an rgb output; "
                f"available={list(output) if isinstance(output, dict) else output!r}"
            )
        image = output["rgb"]
        if not isinstance(image, self.torch.Tensor):
            raise RuntimeError("IsaacSim camera rgb output is not a torch tensor")
        frame = np.asarray(image[0].detach().cpu().numpy())
        if frame.ndim != 3 or frame.shape != (self.render_height, self.render_width, 3):
            raise RuntimeError(
                "IsaacSim camera rgb output has invalid shape: "
                f"got {frame.shape}, expected {(self.render_height, self.render_width, 3)}"
            )
        if frame.dtype != np.uint8:
            # IsaacLab's RGB annotator is uint8 by contract. Refuse lossy
            # coercion when an IsaacSim release changes that surface.
            raise RuntimeError(
                f"IsaacSim camera rgb output has dtype {frame.dtype}, expected uint8"
            )
        frame = np.ascontiguousarray(frame)
        if frame.size == 0 or int(np.ptp(frame)) == 0:
            raise RuntimeError("IsaacSim camera returned an empty or uniform RGB frame")
        return {
            "frame": frame,
            "width": self.render_width,
            "height": self.render_height,
        }

    def shutdown(self) -> None:
        self.camera = None
        for handle in self._shm_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._shm_handles = []
        if self.simulation_app is not None:
            try:
                self.simulation_app.close()
            except Exception:
                pass
            self.simulation_app = None


def _dispatch(ctx: _WorkerContext, protocol: Any, cmd: str, payload: Any) -> tuple[str, Any]:
    if cmd == protocol.CMD_INIT:
        return protocol.CMD_META, ctx.init_sim(payload)
    if cmd == protocol.CMD_ATTACH:
        ctx.attach_slots(payload)
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_STEP:
        return protocol.CMD_READY, ctx.step(payload)
    if cmd == protocol.CMD_SET_STATE:
        return protocol.CMD_READY, ctx.set_state(payload)
    if cmd == protocol.CMD_REFRESH:
        ctx.refresh_state_slots()
        return protocol.CMD_READY, None
    if cmd == protocol.CMD_GET_META:
        return protocol.CMD_META, ctx.get_meta()
    if cmd == protocol.CMD_INIT_RENDERER:
        return protocol.CMD_META, ctx.init_renderer(payload or {})
    if cmd == protocol.CMD_RENDER_FRAME:
        return protocol.CMD_META, ctx.render_frame()
    if cmd == protocol.CMD_CAPTURE_FRAME:
        return protocol.CMD_META, ctx.capture_frame()
    raise NotImplementedError(f"isaacsim worker command {cmd!r} is unsupported")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)
    protocol = _load_protocol(args.protocol)
    ctx = _WorkerContext(protocol)

    # Kit and extension startup can write banners to fd 1.  Preserve a private
    # protocol fd and route all incidental output to stderr before INIT.
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    stdin = sys.stdin.buffer
    stdout = protocol_out
    while True:
        try:
            message = protocol.recv_message(stdin)
        except (EOFError, protocol.WorkerDisconnectedError):
            ctx.shutdown()
            return 0
        cmd = message["cmd"]
        if cmd == protocol.CMD_SHUTDOWN:
            try:
                ctx.shutdown()
            finally:
                protocol.send_message(stdout, protocol.CMD_READY)
            return 0
        try:
            reply_cmd, reply_payload = _dispatch(ctx, protocol, cmd, message.get("payload"))
        except Exception as exc:  # noqa: BLE001 - every worker error crosses the wire
            protocol.send_message(stdout, protocol.CMD_ERROR, protocol.serialize_exception(exc))
            continue
        protocol.send_message(stdout, reply_cmd, reply_payload)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
