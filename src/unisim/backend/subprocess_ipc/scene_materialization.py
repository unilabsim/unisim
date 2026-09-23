"""Cold preparation of entity sources for native subprocess adapters.

MuJoCo is an optional source compiler here, never the worker's physics engine.
The worker must independently audit its instances against the compiled intent.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from unisim.entities import EntityInitialState
from unisim.entity_state import entity_state_snapshot
from unisim.scene import SceneCfg
from unisim.scene_layout import CompiledSceneLayout


@dataclass
class PreparedWorkerScene:
    owner: Any
    layout: CompiledSceneLayout
    payload: dict[str, Any]
    qpos: np.ndarray
    qvel: np.ndarray
    roots: np.ndarray
    ctrl_ranges: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    joint_ranges: np.ndarray
    control_lower: np.ndarray
    control_upper: np.ndarray

    def close(self) -> None:
        self.owner.close()


def _actuation(model: Any, sdk: Any) -> dict[str, Any]:
    joints = [
        i for i in range(model.njnt) if int(model.jnt_type[i]) != int(sdk.mjtJoint.mjJNT_FREE)
    ]
    if any(
        int(model.jnt_type[i]) not in (int(sdk.mjtJoint.mjJNT_HINGE), int(sdk.mjtJoint.mjJNT_SLIDE))
        for i in joints
    ):
        raise NotImplementedError("Isaac scene profile supports scalar hinge/slide joints only")
    positions = {joint: i for i, joint in enumerate(joints)}
    if any(float(model.jnt_stiffness[i]) != 0 for i in joints):
        raise NotImplementedError(
            "Isaac entity profile does not yet map passive joint springs; "
            "actuator stiffness is not a substitute for source springref/stiffness"
        )
    if any(float(model.dof_damping[int(model.jnt_dofadr[i])]) != 0 for i in joints):
        raise NotImplementedError(
            "Isaac entity profile does not yet map source passive joint damping; "
            "position actuator kv remains a separate supported drive parameter"
        )
    names = [model.joint(i).name for i in joints]
    n = len(joints)
    stiffness, damping, effort = np.zeros(n), np.zeros(n), np.zeros(n)
    actuator_names, actuator_joints = [], []
    seen = set()
    if model.na:
        raise NotImplementedError("Isaac scene profile does not support actuator activation state")
    for i in range(model.nu):
        joint = int(model.actuator_trnid[i, 0])
        if int(model.actuator_trntype[i]) != int(sdk.mjtTrn.mjTRN_JOINT) or joint not in positions:
            raise NotImplementedError("Isaac scene profile requires scalar joint actuators")
        if joint in seen:
            raise NotImplementedError("Isaac position profile requires one actuator per joint")
        seen.add(joint)
        gain, bias = model.actuator_gainprm[i], model.actuator_biasprm[i]
        if (
            int(model.actuator_gaintype[i]) != int(sdk.mjtGain.mjGAIN_FIXED)
            or int(model.actuator_biastype[i]) != int(sdk.mjtBias.mjBIAS_AFFINE)
            or gain[0] <= 0
            or not np.isclose(bias[1], -gain[0])
            or bias[2] > 0
            or bias[0] != 0
            or np.any(gain[1:] != 0)
            or np.any(bias[3:] != 0)
            or not np.array_equal(model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
        ):
            raise NotImplementedError("Isaac scene profile requires unit-gear position actuators")
        limit = np.asarray(model.actuator_forcerange[i])
        if model.actuator_forcelimited[i] and (
            not np.isclose(limit[0], -limit[1]) or limit[1] <= 0
        ):
            raise NotImplementedError("Isaac position profile requires symmetric positive effort")
        index = positions[joint]
        stiffness[index], damping[index] = gain[0], -bias[2]
        effort[index] = limit[1] if model.actuator_forcelimited[i] else 1e9
        actuator_names.append(model.actuator(i).name)
        actuator_joints.append(model.joint(joint).name)
    lower = [
        float(model.jnt_range[i, 0]) if model.jnt_limited[i] else -float("inf") for i in joints
    ]
    upper = [float(model.jnt_range[i, 1]) if model.jnt_limited[i] else float("inf") for i in joints]
    dof_ids = [int(model.jnt_dofadr[i]) for i in joints]
    body_sphere_radii = _body_sphere_radii(model, sdk)
    body_visual_rgb = _body_visual_rgb(model)
    geom_names, geom_body_names = [], []
    geom_contype, geom_conaffinity, geom_friction = [], [], []
    for body_id in range(1, int(model.nbody)):
        body_name = model.body(body_id).name
        body_offset = 0
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            source_name = model.geom(geom_id).name
            geom_names.append(
                source_name if source_name else f"{body_name}::geom{body_offset}"
            )
            geom_body_names.append(body_name)
            geom_contype.append(int(model.geom_contype[geom_id]))
            geom_conaffinity.append(int(model.geom_conaffinity[geom_id]))
            geom_friction.append(model.geom_friction[geom_id].tolist())
            body_offset += 1
    if len(set(geom_names)) != len(geom_names):
        raise ValueError("entity geometry names are not unique")
    if not np.isfinite(geom_friction).all() or np.any(np.asarray(geom_friction) < 0.0):
        raise ValueError("entity geometry friction is invalid")
    return {
        "joint_names": names,
        "actuator_names": actuator_names,
        "actuator_joint_names": actuator_joints,
        "dof_stiffness": stiffness.tolist(),
        "dof_damping": damping.tolist(),
        "dof_effort": effort.tolist(),
        "dof_lower": lower,
        "dof_upper": upper,
        "dof_armature": model.dof_armature[dof_ids].tolist(),
        "dof_friction": model.dof_frictionloss[dof_ids].tolist(),
        "body_names": [model.body(i).name for i in range(1, model.nbody)],
        "body_mass": model.body_mass[1:].tolist(),
        "body_ipos": model.body_ipos[1:].tolist(),
        "body_inertia": model.body_inertia[1:].tolist(),
        "body_iquat": model.body_iquat[1:].tolist(),
        "body_sphere_radii": body_sphere_radii,
        "body_visual_rgb": body_visual_rgb,
        "geom_names": geom_names,
        "geom_body_names": geom_body_names,
        "geom_contype": geom_contype,
        "geom_conaffinity": geom_conaffinity,
        "geom_friction": geom_friction,
    }


def _body_sphere_radii(model: Any, sdk: Any) -> list[list[float]]:
    """Collect supported geometry dimensions in source geom order."""
    result: list[list[float]] = []
    for body_id in range(1, int(model.nbody)):
        radii: list[float] = []
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(sdk.mjtGeom.mjGEOM_SPHERE):
                continue
            radius = float(model.geom_size[geom_id, 0])
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"body {body_id} has an invalid sphere radius: {radius!r}")
            radii.append(radius)
        result.append(radii)
    return result


def _body_visual_rgb(model: Any) -> list[list[float]]:
    """Collect the first visible source geom color for Gym's per-body viewer."""
    result: list[list[float]] = []
    for body_id in range(1, int(model.nbody)):
        color = [0.5, 0.5, 0.5]
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            rgba = model.geom_rgba[geom_id]
            if float(rgba[3]) > 0.0:
                color = [float(value) for value in rgba[:3]]
                break
        result.append(color)
    return result


def validate_body_sphere_radii(value: Any, body_count: int) -> None:
    """Validate the ragged cold-path sphere record before Kit or worker use."""
    if not isinstance(value, list) or len(value) != body_count:
        raise ValueError("invalid variant body_sphere_radii")
    for radii in value:
        if not isinstance(radii, list):
            raise ValueError("invalid variant body_sphere_radii")
        for radius in radii:
            if (
                isinstance(radius, (bool, np.bool_))
                or not isinstance(radius, (int, float, np.integer, np.floating))
            ):
                raise ValueError("invalid variant body_sphere_radii")
            number = float(radius)
            if not np.isfinite(number) or number <= 0.0:
                raise ValueError("invalid variant body_sphere_radii")


def _materialize_worker_meshes(spec: Any, root: Path, prefix: str) -> dict[str, str]:
    """Copy mesh resources beside an exported source for Gym's path resolution."""
    replacements: dict[str, str] = {}
    for index, mesh in enumerate(spec.meshes):
        if not mesh.file:
            continue
        source = Path(mesh.file)
        if not source.is_file():
            raise ValueError(f"entity mesh source does not exist: {source}")
        filename = f"{prefix}_mesh_{index}{source.suffix}"
        if source.suffix.lower() == ".obj":
            _copy_obj_with_material_libraries(source, root / filename, prefix, index)
        else:
            shutil.copyfile(source, root / filename)
        replacements[str(source)] = filename
    return replacements


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    return "\n" if line.endswith("\n") else ""


def _copy_obj_with_material_libraries(source: Path, destination: Path, prefix: str, index: int):
    """Copy an OBJ and rewrite its local MTL references to unique files."""
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    for line in lines:
        tokens = line.split()
        if tokens and tokens[0] == "mtllib":
            if len(tokens) < 2:
                raise ValueError(f"entity mesh has a malformed mtllib line: {source}")
            copied = []
            for material_index, reference in enumerate(tokens[1:]):
                material = source.parent / reference
                if not material.is_file():
                    raise ValueError(f"entity mesh material source does not exist: {material}")
                material_name = f"{prefix}_mesh_{index}_material_{material_index}.mtl"
                _copy_material_library(material, root=destination.parent, name=material_name)
                copied.append(material_name)
            output.append("mtllib " + " ".join(copied) + _line_ending(line))
        else:
            output.append(line)
    destination.write_text("".join(output), encoding="utf-8")


def _copy_material_library(source: Path, *, root: Path, name: str) -> None:
    """Copy color-only MTL data and its local texture maps under unique names."""
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    for line_index, line in enumerate(lines):
        tokens = line.split()
        is_texture = bool(tokens) and (
            tokens[0].startswith("map_") or tokens[0] in {"bump", "disp", "decal", "refl"}
        )
        if not is_texture:
            output.append(line)
            continue
        if len(tokens) < 2:
            raise ValueError(f"entity mesh material has a malformed texture line: {source}")
        reference = tokens[-1]
        texture = source.parent / reference
        if not texture.is_file():
            raise ValueError(f"entity mesh texture source does not exist: {texture}")
        texture_name = f"{Path(name).stem}_texture_{line_index}{texture.suffix}"
        shutil.copyfile(texture, root / texture_name)
        output.append(line[: line.rfind(reference)] + texture_name + _line_ending(line))
    (root / name).write_text("".join(output), encoding="utf-8")


def body_sphere_radii_close(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    """Compare ragged per-body sphere records without padding empty bodies."""
    try:
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        if len(actual) != len(expected):
            return False
        for actual_body, expected_body in zip(actual, expected):
            if not isinstance(actual_body, list) or not isinstance(expected_body, list):
                return False
            if len(actual_body) != len(expected_body):
                return False
            if not actual_body:
                continue
            actual_values = np.asarray(actual_body, dtype=np.float64)
            expected_values = np.asarray(expected_body, dtype=np.float64)
            if (
                actual_values.shape != (len(actual_body),)
                or expected_values.shape != (len(expected_body),)
                or not np.isfinite(actual_values).all()
                or not np.isfinite(expected_values).all()
                or not np.allclose(actual_values, expected_values, rtol=rtol, atol=atol)
            ):
                return False
        return True
    except (TypeError, ValueError):
        return False


def prepare_worker_scene(scene: SceneCfg, num_envs: int, sim_dt: float) -> PreparedWorkerScene:
    """Compile entity topology/defaults and explicit inertials before spawning workers."""
    import mujoco

    from unisim.mjcf_compiler import (
        compose_scene,
        compute_variant_initial_state,
        load_entity_source,
    )
    from unisim.progress import ProgressBar

    # Isaac workers consume each entity's self_collision flag (IsaacSim through
    # its MJCF converter, IsaacGym through PhysX filter authoring); compilation
    # itself retains authored exclusions.
    owner = compose_scene(scene, num_envs, sim_dt, allow_self_collision=True)
    try:
        root = Path(owner.model_file).parent
        physical = {
            entity.name: entity for entity in scene.entity_assets if entity.mirror_of is None
        }
        binding = scene.entity_variant
        assignment = np.zeros(num_envs, dtype=int) if binding is None else binding.plan.assignment
        qpos: list[np.ndarray | None] = [None] * num_envs
        qvel: list[np.ndarray | None] = [None] * num_envs
        roots: list[np.ndarray | None] = [None] * num_envs
        controls: list[np.ndarray | None] = [None] * num_envs
        lower_controls: list[np.ndarray | None] = [None] * num_envs
        upper_controls: list[np.ndarray | None] = [None] * num_envs
        selected_variants = (0,) if owner.variant_plan is None else np.unique(assignment)
        # Compute initial rows one catalog realization at a time.  The common
        # compiler has already validated the complete catalog fail-closed and
        # normally captured each variant's initial state during composition.
        for selected_variant in selected_variants:
            snapshot = (
                owner.variant_initial_states.get(int(selected_variant))
                if owner.variant_initial_states is not None
                else None
            )
            if snapshot is None:
                model = (
                    owner.model
                    if owner.variant_plan is None
                    else mujoco.MjModel.from_xml_path(
                        owner.variant_plan.variants[int(selected_variant)].model_file
                    )
                )
                snapshot = compute_variant_initial_state(
                    model, owner.layout, scene.default_keyframe_name
                )
                del model
            for env_index in np.flatnonzero(assignment == selected_variant):
                qpos[int(env_index)] = snapshot.qpos.copy()
                qvel[int(env_index)] = snapshot.qvel.copy()
                roots[int(env_index)] = snapshot.entity_rows.copy()
                controls[int(env_index)] = snapshot.ctrl.copy()
                lower_controls[int(env_index)] = snapshot.ctrl_lower.copy()
                upper_controls[int(env_index)] = snapshot.ctrl_upper.copy()
        q = np.asarray(qpos, dtype=np.float32)
        v = np.asarray(qvel, dtype=np.float32)
        root_states = np.asarray(roots, dtype=np.float32)
        entries = []
        source_entity_indexes = {
            entity.name: index
            for index, entity in enumerate(scene.entity_assets)
            if entity.mirror_of is None
        }
        for entity_index, entity in enumerate(scene.entity_assets):
            source_entity = physical[entity.mirror_of] if entity.mirror_of else entity
            assert source_entity.source is not None
            consumes = binding is not None and binding.target_entity == source_entity.name
            sources = (
                binding.plan.variants
                if consumes and binding is not None
                else (source_entity.source,)
            )
            paths, records = [], []
            progress = (
                ProgressBar(
                    f"exporting {len(sources)} worker sources ({entity.name})", len(sources)
                )
                if len(sources) >= 8
                else None
            )
            try:
                for variant, source in enumerate(sources):
                    # Raw conversion is role-neutral.  Physical entities and their
                    # mirrors therefore share the same expanded source and raw USD;
                    # collision, mobility, and visual-role edits belong to copies.
                    spec, _, _ = load_entity_source(
                        replace(source_entity, initial_state=EntityInitialState()),
                        source.model_file,
                        mirror=False,
                    )
                    model = spec.compile()
                    # USD importer uses the MJCF model name as a prim identifier;
                    # MuJoCo's default "MuJoCo Model" contains an invalid space.
                    spec.modelname = f"entity_{source_entity_indexes[source_entity.name]}"
                    # Gym's importer ignores geom mass. Explicit compiler-derived
                    # inertials preserve the actual intended body mass/COM/tensor.
                    for body in spec.bodies[1:]:
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
                        body.mass = float(model.body_mass[bid])
                        body.inertia = model.body_inertia[bid]
                        body.ipos = model.body_ipos[bid]
                        body.iquat = model.body_iquat[bid]
                        # MJCF rejects the mixed full/diagonal spelling: clear a
                        # source fullinertia now that the diagonal form replaces
                        # it (NaN is the spec's "unspecified" sentinel).
                        body.fullinertia = [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]
                        body.explicitinertial = True
                    record = _actuation(model, mujoco)
                    for joint in spec.joints:
                        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
                            joint.limited = int(model.jnt_limited[jid])
                            joint.range = model.jnt_range[jid]
                    # The native MJCF importer does not understand MuJoCo's
                    # canonical <general> spelling emitted by MjSpec. Drive intent
                    # travels in the separately audited table, never through it.
                    for actuator in list(spec.actuators):
                        spec.delete(actuator)
                    for keyframe in list(spec.keys):
                        spec.delete(keyframe)
                    # Mobility is explicit AssetOptions/prim configuration. A mocap
                    # tag has no meaning in the PhysX importer and is never relied on.
                    for body in spec.bodies[1:]:
                        body.mocap = False
                    mesh_files = _materialize_worker_meshes(
                        spec, root, f"entity_{entity_index}_{variant}"
                    )
                    path = root / f"entity_{entity_index}_{variant}.xml"
                    # to_file() may serialize the last compiled model, omitting
                    # edits made after compile(); to_xml() serializes current spec.
                    document = ET.fromstring(spec.to_xml())
                    for mesh in document.findall("./asset/mesh"):
                        filename = mesh_files.get(mesh.get("file", ""))
                        if filename is not None:
                            mesh.set("file", filename)
                    ET.ElementTree(document).write(path, encoding="utf-8")
                    paths.append(str(path))
                    records.append(record)
                    if progress is not None:
                        progress.update(variant + 1)
            finally:
                if progress is not None:
                    progress.close()
            entries.append(
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "root_mode": entity.root_mode,
                    "asset_format": entity.asset_format,
                    "collision_enabled": entity.collision_enabled,
                    "self_collision": entity.self_collision,
                    # None keeps the consuming backend's implicit per-entity
                    # gravity default; the backend host resolves it before INIT.
                    "gravity_disabled": entity.gravity_disabled,
                    "mirror_of": entity.mirror_of,
                    "initial_pose": list(
                        entity.initial_state.position + entity.initial_state.quaternion
                    ),
                    "sources": paths,
                    "assignment": [int(i) for i in assignment] if consumes else [0] * num_envs,
                    "variants": records,
                }
            )
        model = owner.model
        gains = model.actuator_gainprm[:, 0].copy()
        kd = -model.actuator_biasprm[:, 2].copy()
        joint_ids = [
            int(model.joint(entity.name + "/" + joint.name).id)
            for entity in owner.layout.entities
            for joint in entity.joints
        ]
        ranges = np.asarray(
            [model.jnt_range[i] if model.jnt_limited[i] else [-np.inf, np.inf] for i in joint_ids]
        ).reshape(-1, 2)
        payload = {
            "scene_layout": owner.layout.to_dict(),
            "scene_content_identity": owner.content_identity.to_dict(),
            "scene_entities": entries,
            "initial_qpos": q.tolist(),
            "initial_qvel": v.tolist(),
            "initial_roots": root_states.tolist(),
            "initial_ctrl": np.asarray(controls, dtype=np.float32).tolist(),
            "gravity": model.opt.gravity.tolist(),
        }
        return PreparedWorkerScene(
            owner,
            owner.layout,
            payload,
            q,
            v,
            root_states,
            model.actuator_ctrlrange.copy(),
            gains,
            kd,
            ranges,
            np.asarray(lower_controls, dtype=np.float32),
            np.asarray(upper_controls, dtype=np.float32),
        )
    except BaseException:
        owner.close()
        raise


def full_state_reset_patches(layout: CompiledSceneLayout, qpos: np.ndarray, qvel: np.ndarray):
    """Normalize existing full-state writes into the same entity transaction."""
    from unisim.entities import EntityStatePatch

    patches = []
    for entity in layout.entities:
        values = entity_state_snapshot(entity, qpos, qvel, np.zeros((len(qpos), 13)))
        kwargs: dict[str, Any] = {}
        if entity.root_mode == "floating":
            kwargs.update(root_pose=values["root_pose"], root_velocity=values["root_velocity"])
        if entity.joints:
            kwargs.update(
                joint_positions=values["joint_positions"],
                joint_velocities=values["joint_velocities"],
            )
        if kwargs:
            patches.append(EntityStatePatch(entity.name, **kwargs))
    return tuple(patches)
