"""Cold preparation of entity sources for native subprocess adapters.

MuJoCo is an optional source compiler here, never the worker's physics engine.
The worker must independently audit its instances against the compiled intent.
"""

from __future__ import annotations

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
    }


def prepare_worker_scene(scene: SceneCfg, num_envs: int, sim_dt: float) -> PreparedWorkerScene:
    """Compile entity topology/defaults and explicit inertials before spawning workers."""
    import mujoco

    from unisim.mjcf_compiler import compose_scene, load_entity_source

    owner = compose_scene(scene, num_envs, sim_dt)
    try:
        root = Path(owner.model_file).parent
        physical = {
            entity.name: entity for entity in scene.entity_assets if entity.mirror_of is None
        }
        binding = scene.entity_variant
        assignment = np.zeros(num_envs, dtype=int) if binding is None else binding.plan.assignment
        models = (
            [owner.model]
            if owner.variant_plan is None
            else [
                mujoco.MjModel.from_xml_path(source.model_file)
                for source in owner.variant_plan.variants
            ]
        )
        qpos, qvel, roots, controls, lower_controls, upper_controls = [], [], [], [], [], []
        for variant in assignment:
            model = models[int(variant)]
            data = mujoco.MjData(model)
            if scene.default_keyframe_name is not None:
                key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, scene.default_keyframe_name)
                mujoco.mj_resetDataKeyframe(model, data, key)
            mujoco.mj_forward(model, data)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            lower_control = np.where(
                model.actuator_ctrllimited, model.actuator_ctrlrange[:, 0], -np.inf
            )
            upper_control = np.where(
                model.actuator_ctrllimited, model.actuator_ctrlrange[:, 1], np.inf
            )
            lower_controls.append(lower_control)
            upper_controls.append(upper_control)
            controls.append(np.clip(data.ctrl, lower_control, upper_control))
            row = np.zeros((len(owner.layout.entities), 13))
            for index, entity_layout in enumerate(owner.layout.entities):
                bid = entity_layout.body_ids[
                    entity_layout.body_names.index(entity_layout.root_body)
                ]
                row[index, :3], row[index, 3:7] = data.xpos[bid], data.xquat[bid]
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity, 0)
                row[index, 7:10], row[index, 10:] = velocity[3:], velocity[:3]
            roots.append(row)
        q = np.asarray(qpos, dtype=np.float32)
        v = np.asarray(qvel, dtype=np.float32)
        root_states = np.asarray(roots, dtype=np.float32)
        entries = []
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
            for variant, source in enumerate(sources):
                spec, _, _ = load_entity_source(
                    replace(entity, initial_state=EntityInitialState()),
                    source.model_file,
                    mirror=entity.mirror_of is not None,
                )
                model = spec.compile()
                # USD importer uses the MJCF model name as a prim identifier;
                # MuJoCo's default "MuJoCo Model" contains an invalid space.
                spec.modelname = f"entity_{entity_index}"
                # Gym's importer ignores geom mass. Explicit compiler-derived
                # inertials preserve the actual intended body mass/COM/tensor.
                for body in spec.bodies[1:]:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
                    body.mass = float(model.body_mass[bid])
                    body.inertia = model.body_inertia[bid]
                    body.ipos = model.body_ipos[bid]
                    body.iquat = model.body_iquat[bid]
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
                path = root / f"entity_{entity_index}_{variant}.xml"
                # to_file() may serialize the last compiled model, omitting
                # edits made after compile(); to_xml() serializes current spec.
                path.write_text(spec.to_xml(), encoding="utf-8")
                paths.append(str(path))
                records.append(record)
            entries.append(
                {
                    "name": entity.name,
                    "kind": entity.kind,
                    "root_mode": entity.root_mode,
                    "asset_format": entity.asset_format,
                    "collision_enabled": entity.collision_enabled,
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
