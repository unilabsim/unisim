"""MuJoCo structural oracle for the common portable MJCF cold path."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np

from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import SceneEntitySpec
from unisim.inspection import ConfigurationField, ConfigurationProvenance
from unisim.scene import SceneCfg
from unisim.scene_compiler import (
    PORTABLE_MJCF_PROFILE_ID,
    PORTABLE_MJCF_STRUCTURAL_ORACLE,
    SceneCompilerParameters,
    SceneContentIdentity,
    SceneIntentReport,
    SceneResourceProvenance,
    SceneSourceProvenance,
    compute_scene_content_identity,
)
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, GeomLayout, JointLayout

_CONTACT_FORCE_SENSOR_INTPRM = (2, 3, 1)
_CONTACT_FOUND_SENSOR_INTPRM = (1, 0, 1)


@dataclass(frozen=True)
class _CrossEntityContactSensor:
    """One sensor declaration resolved after entity namespace attachment."""

    name: str
    geom1: str
    geom2: str
    intprm: tuple[int, ...]


@dataclass(frozen=True)
class _CrossEntityFrameSensor:
    """One world-referenced body/site frame sensor resolved after attachment."""

    name: str
    sensor_type: int
    object_type: int
    object_name: str


_CrossEntitySensor = _CrossEntityContactSensor | _CrossEntityFrameSensor


@dataclass(frozen=True)
class _SensorFragment:
    """A scene-level sensor-only MJCF fragment and its content identity."""

    path: Path
    digest: str
    sensors: tuple[_CrossEntitySensor, ...]


@dataclass
class ComposedScene:
    """Own generated full-scene sources until their last executor is closed."""

    model_file: str
    variant_plan: FixedVariantPlan | None
    layout: CompiledSceneLayout
    variant_layouts: tuple[CompiledSceneLayout, ...]
    model: mujoco.MjModel
    source_provenance: tuple[SceneSourceProvenance, ...]
    content_identity: SceneContentIdentity
    intent_report: SceneIntentReport
    _directory: tempfile.TemporaryDirectory

    def close(self) -> None:
        self._directory.cleanup()

    def __enter__(self) -> ComposedScene:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _options(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    return {
        name: np.array(getattr(model.opt, name), copy=True)
        for name in dir(model.opt)
        if not name.startswith("_")
        and name != "timestep"
        and not callable(getattr(model.opt, name))
    }


def load_entity_source(
    entity: SceneEntitySpec, path: str, *, mirror: bool
) -> tuple[mujoco.MjSpec, mujoco.MjModel, SceneSourceProvenance]:
    """Return a normalized entity spec and its independent original source model.

    The original model retains source keyframes and compiler-resolved units.
    The mutable spec has no keys, uses absolute asset paths, and applies the
    declared root pose/physics role. Other adapters may reuse this cold-path
    normalization before translating to their native asset representation.
    """
    if entity.asset_format != "mjcf":
        raise NotImplementedError("MuJoCo composition currently accepts MJCF sources only")
    filename = Path(path).resolve()
    if b"<include" in filename.read_bytes().lower():
        raise NotImplementedError(
            f"entity {entity.name!r}: MJCF includes are outside the portable profile"
        )
    spec = mujoco.MjSpec.from_file(str(filename))
    if spec.assets:
        raise NotImplementedError(
            f"entity {entity.name!r}: inline MjSpec assets are outside the portable profile"
        )
    resources: list[SceneResourceProvenance] = []
    for mesh in spec.meshes:
        if mesh.file:
            resources.append(
                SceneResourceProvenance.from_file(
                    "mesh", mesh.file, filename.parent / spec.compiler.meshdir / mesh.file
                )
            )
    for texture in spec.textures:
        if texture.file:
            resources.append(
                SceneResourceProvenance.from_file(
                    "texture",
                    texture.file,
                    filename.parent / spec.compiler.texturedir / texture.file,
                )
            )
    for hfield in spec.hfields:
        if hfield.file:
            resources.append(
                SceneResourceProvenance.from_file(
                    "hfield", hfield.file, filename.parent / hfield.file
                )
            )
    provenance = SceneSourceProvenance(
        entity.name,
        entity.asset_format,
        str(filename),
        SceneResourceProvenance.from_file("source", filename.name, filename).content_digest,
        tuple(resources),
        entity_kind=entity.kind,
        root_mode=entity.root_mode,
        collision_enabled=entity.collision_enabled,
        initial_position=entity.initial_state.position,
        initial_quaternion=entity.initial_state.quaternion,
        mirror_of=entity.mirror_of,
    )
    default_compiler = mujoco.MjSpec().compiler
    for field in (
        "settotalmass",
        "boundmass",
        "boundinertia",
        "balanceinertia",
        "fusestatic",
        "discardvisual",
        "alignfree",
        "inertiafromgeom",
        "inertiagrouprange",
    ):
        if not np.array_equal(getattr(spec.compiler, field), getattr(default_compiler, field)):
            raise NotImplementedError(
                f"entity {entity.name!r}: compiler {field} requires explicit composition support"
            )
    if spec.tendons or spec.equalities:
        raise NotImplementedError("entity composition does not yet support tendons/equalities")
    roots = list(spec.worldbody.bodies)
    if len(roots) != 1 or list(spec.worldbody.geoms):
        raise ValueError(f"entity {entity.name!r} requires exactly one root and no world geoms")
    root = roots[0]
    if any(not body.name for body in spec.bodies[1:]):
        raise ValueError(f"entity {entity.name!r} requires named bodies")
    free = [joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE]
    other = [joint for joint in spec.joints if joint.type != mujoco.mjtJoint.mjJNT_FREE]
    if any(not joint.name for joint in other):
        raise ValueError(f"entity {entity.name!r} requires named non-root joints")
    if free and (len(free) != 1 or free[0] not in list(root.joints)):
        raise ValueError(f"entity {entity.name!r} has a non-root or multiple free joints")
    if not mirror:
        if entity.kind == "rigid" and other:
            raise ValueError(f"rigid entity {entity.name!r} contains non-root joints")
        if entity.root_mode == "floating" and (len(free) != 1 or root.mocap):
            raise ValueError(f"floating entity {entity.name!r} requires one root free joint")
        if entity.root_mode == "fixed" and (free or list(root.joints) or root.mocap):
            raise ValueError(f"fixed entity {entity.name!r} requires a fixed root body")
        if entity.root_mode == "kinematic" and other:
            raise NotImplementedError("kinematic articulation joints are not implemented")
    # Compile before changing root/actuation topology. This resolves source
    # angular units, defaults and keyframe widths independently of the scene.
    original_model = spec.compile()
    if any(not original_model.key(i).name for i in range(original_model.nkey)):
        raise ValueError(f"entity {entity.name!r}: composition requires named source keyframes")
    for key in list(spec.keys):
        spec.delete(key)
    if mirror:
        for collection in (
            spec.actuators,
            spec.sensors,
            spec.equalities,
            spec.tendons,
            spec.pairs,
            spec.excludes,
        ):
            for item in list(collection):
                spec.delete(item)
        for joint in list(spec.joints):
            spec.delete(joint)
    elif entity.root_mode == "kinematic":
        if spec.actuators:
            raise ValueError("kinematic entities cannot retain actuators")
        for joint in free:
            spec.delete(joint)
    if entity.root_mode == "kinematic":
        root.mocap = True
    if not entity.collision_enabled:
        for geom in spec.geoms:
            geom.contype = 0
            geom.conaffinity = 0
        for pair in list(spec.pairs):
            spec.delete(pair)
    root.pos = entity.initial_state.position
    root.quat = entity.initial_state.quaternion
    # MjSpec attachment preserves authored references, but the generated XML
    # lives elsewhere. Resolve external resources before serialization.
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((filename.parent / spec.compiler.meshdir / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file:
            texture.file = str(
                (filename.parent / spec.compiler.texturedir / texture.file).resolve()
            )
    for hfield in spec.hfields:
        if hfield.file:
            hfield.file = str((filename.parent / hfield.file).resolve())
    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""
    return spec, original_model, provenance


def compile_scene_layout(
    model: mujoco.MjModel, entities: tuple[SceneEntitySpec, ...]
) -> CompiledSceneLayout:
    """Bind actual compiled addresses under each entity's attach namespace."""
    layouts = []
    kinds = {
        int(mujoco.mjtJoint.mjJNT_HINGE): ("hinge", 1, 1),
        int(mujoco.mjtJoint.mjJNT_SLIDE): ("slide", 1, 1),
        int(mujoco.mjtJoint.mjJNT_BALL): ("ball", 4, 3),
    }
    for entity in entities:
        prefix = entity.name + "/"
        bodies = tuple(i for i in range(1, model.nbody) if model.body(i).name.startswith(prefix))
        roots = [i for i in bodies if int(model.body_parentid[i]) not in bodies]
        if len(roots) != 1:
            raise ValueError(f"compiled entity {entity.name!r} does not have exactly one root")
        root = roots[0]
        local = lambda name: name[len(prefix) :]  # noqa: E731
        joints = []
        root_q: tuple[int, ...] = ()
        root_v: tuple[int, ...] = ()
        for jid in range(model.njnt):
            body = int(model.jnt_bodyid[jid])
            if body not in bodies:
                continue
            qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                root_q, root_v = tuple(range(qa, qa + 7)), tuple(range(va, va + 6))
                continue
            kind, nq, nv = kinds[int(model.jnt_type[jid])]
            joints.append(
                JointLayout(
                    local(model.joint(jid).name),
                    kind,
                    tuple(range(qa, qa + nq)),
                    tuple(range(va, va + nv)),
                    local(model.body(body).name),
                )
            )
        actuator_ids, actuator_names, actuator_joints = [], [], []
        for aid in range(model.nu):
            name = model.actuator(aid).name
            if not name.startswith(prefix):
                continue
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                raise NotImplementedError("composition supports joint-transmission actuators only")
            joint = model.joint(int(model.actuator_trnid[aid, 0])).name
            if not joint.startswith(prefix):
                raise ValueError("actuator target crosses entity boundary")
            actuator_ids.append(aid)
            actuator_names.append(local(name))
            actuator_joints.append(local(joint))
        geoms = []
        for body in bodies:
            body_name = local(model.body(body).name)
            body_offset = 0
            for geom in range(model.ngeom):
                if int(model.geom_bodyid[geom]) != body:
                    continue
                source_name = model.geom(geom).name
                public_name = (
                    local(source_name)
                    if source_name.startswith(prefix)
                    else f"{body_name}::geom{body_offset}"
                )
                geoms.append(GeomLayout(public_name, body_name))
                body_offset += 1
        layouts.append(
            EntityLayout(
                name=entity.name,
                kind=entity.kind,
                root_mode=entity.root_mode,
                root_body=local(model.body(root).name),
                body_names=tuple(local(model.body(i).name) for i in bodies),
                body_ids=bodies,
                body_parent_names=tuple(
                    None if i == root else local(model.body(int(model.body_parentid[i])).name)
                    for i in bodies
                ),
                joints=tuple(joints),
                actuator_names=tuple(actuator_names),
                actuator_joint_names=tuple(actuator_joints),
                actuator_indices=tuple(actuator_ids),
                root_qpos_indices=root_q,
                root_qvel_indices=root_v,
                geoms=tuple(geoms),
            )
        )
    return CompiledSceneLayout(
        tuple(layouts), model.nq, model.nv, model.nu, model.nbody, model.ngeom
    )


def _sensor_signature(model: mujoco.MjModel) -> tuple:
    return tuple(
        (
            model.sensor(i).name,
            int(model.sensor_type[i]),
            int(model.sensor_dim[i]),
            int(model.sensor_objtype[i]),
            int(model.sensor_objid[i]),
            int(model.sensor_reftype[i]),
            int(model.sensor_refid[i]),
        )
        for i in range(model.nsensor)
    )


def _load_sensor_fragments(scene: SceneCfg) -> tuple[_SensorFragment, ...]:
    """Load the scene-level, sensor-only portable MJCF authoring additions.

    Entity sources remain independently valid MJCF documents.  A fragment may
    introduce ordered collision-pair force sensors, world-referenced body/site
    pose sensors, or world-referenced body motion sensors whose object names are
    in the final ``entity/local-name`` namespace; the compiler resolves them
    after entity attachment.
    """

    fragments: list[_SensorFragment] = []
    names: set[str] = set()
    contact_attributes = {"name", "geom1", "geom2", "data", "reduce", "num"}
    frame_attributes = {"name", "objtype", "objname"}
    for fragment_file in scene.fragment_files:
        path = Path(fragment_file)
        if not path.is_file():
            raise ValueError(f"portable sensor fragment {fragment_file} does not exist")
        path = path.resolve(strict=True)
        root = ET.parse(path).getroot()
        if root.tag != "mujoco":
            raise ValueError(f"portable sensor fragment {path} must have a <mujoco> root")
        sensors: list[_CrossEntitySensor] = []
        frame_sensor_types = {
            "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
            "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
        }
        for section in root:
            if section.tag != "sensor":
                raise ValueError(
                    f"portable sensor fragment {path} may contain only <sensor> sections"
                )
            for item in section:
                if item.tag not in ("contact", *frame_sensor_types):
                    raise ValueError(
                        f"portable sensor fragment {path} supports only contact or "
                        "world-referenced body/site FramePos/FrameQuat and body "
                        "FrameLinVel/FrameAngVel sensors"
                    )
                attributes = set(item.attrib)
                if item.tag == "contact" and not attributes <= contact_attributes:
                    raise ValueError(
                        f"portable sensor fragment {path} contact sensors support only "
                        "name, geom1, geom2, data, reduce and num attributes"
                    )
                if item.tag != "contact" and attributes != frame_attributes:
                    raise ValueError(
                        f"portable sensor fragment {path} {item.tag} sensors support "
                        "only name, objtype and objname attributes"
                    )
                name = item.attrib.get("name", "")
                if not name or name in names:
                    raise ValueError(
                        f"portable sensor fragment {path} requires unique non-empty names"
                    )
                if item.tag != "contact":
                    objtype = item.attrib.get("objtype", "")
                    objname = item.attrib.get("objname", "")
                    object_type = (
                        int(mujoco.mjtObj.mjOBJ_BODY)
                        if objtype == "body"
                        else int(mujoco.mjtObj.mjOBJ_SITE)
                        if objtype == "site"
                        else None
                    )
                    if "/" in name or object_type is None or objname.count("/") != 1:
                        raise ValueError(
                            f"portable sensor fragment {path} {item.tag} sensor "
                            f"{name!r} must be a world-referenced body/site sensor in "
                            "entity/local-name form"
                        )
                    if item.tag in ("framelinvel", "frameangvel") and objtype != "body":
                        raise ValueError(
                            f"portable sensor fragment {path} {item.tag} sensor "
                            f"{name!r} supports only qualified body objects"
                        )
                    names.add(name)
                    sensors.append(
                        _CrossEntityFrameSensor(
                            name,
                            int(frame_sensor_types[item.tag]),
                            object_type,
                            objname,
                        )
                    )
                    continue

                geom1 = item.attrib.get("geom1", "")
                geom2 = item.attrib.get("geom2", "")
                data = item.attrib.get("data")
                intprm: tuple[int, ...]
                force_attributes = {"name", "geom1", "geom2", "data", "reduce"}
                found_attributes = {"name", "geom1", "geom2", "data", "num"}
                if data == "force" and attributes == force_attributes:
                    if item.attrib.get("reduce") != "netforce":
                        raise ValueError(
                            f"portable sensor fragment {path} contact sensor {name!r} "
                            "supports only data='force' reduce='netforce'"
                        )
                    intprm = _CONTACT_FORCE_SENSOR_INTPRM
                elif data == "found" and attributes == found_attributes:
                    if item.attrib.get("num") != "1":
                        raise ValueError(
                            f"portable sensor fragment {path} contact sensor {name!r} "
                            "supports only data='found' num='1'"
                        )
                    intprm = _CONTACT_FOUND_SENSOR_INTPRM
                else:
                    raise ValueError(
                        f"portable sensor fragment {path} contact sensor {name!r} supports "
                        "only data='force' reduce='netforce' or data='found' num='1'"
                    )
                if not geom1 or not geom2 or geom1.count("/") != 1 or geom2.count("/") != 1:
                    raise ValueError(
                        f"portable sensor fragment {path} contact sensor {name!r} requires "
                        "both geoms in entity/local-name form"
                    )
                names.add(name)
                sensors.append(_CrossEntityContactSensor(name, geom1, geom2, intprm))
        if not sensors:
            raise ValueError(f"portable sensor fragment {path} contains no sensors")
        digest = sha256(path.read_bytes()).hexdigest()
        fragments.append(_SensorFragment(path, digest, tuple(sensors)))
    return tuple(fragments)


def _add_sensor_fragments(
    assembled: mujoco.MjSpec, fragments: tuple[_SensorFragment, ...]
) -> None:
    """Add scene-level sensors after all entity geoms have been attached."""

    for fragment in fragments:
        for sensor in fragment.sensors:
            if isinstance(sensor, _CrossEntityContactSensor):
                assembled.add_sensor(
                    name=sensor.name,
                    type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM,
                    objname=sensor.geom1,
                    reftype=mujoco.mjtObj.mjOBJ_GEOM,
                    refname=sensor.geom2,
                    intprm=sensor.intprm,
                )
            else:
                assembled.add_sensor(
                    name=sensor.name,
                    type=sensor.sensor_type,
                    objtype=mujoco.mjtObj(sensor.object_type),
                    objname=sensor.object_name,
                )


def _merge_keys(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    layout: CompiledSceneLayout,
    sources: dict[str, mujoco.MjModel],
    default_name: str | None,
) -> tuple[str, ...]:
    """Merge named keys by native joint/control addresses, never concatenation.

    Source root poses AND root velocities are intentionally overridden by each
    EntityInitialState pose and zero initial velocity. A future import report
    must retain that override provenance. Mirrors retain their independent
    initial mocap pose and contribute no joint, control or activation state.
    """
    names = tuple(
        sorted({source.key(i).name for source in sources.values() for i in range(source.nkey)})
    )
    if default_name is not None and default_name not in names:
        raise ValueError(
            f"default keyframe {default_name!r} is absent from physical entity sources"
        )
    initial = mujoco.MjData(model)
    for name in names:
        qpos = model.qpos0.copy()
        qvel = np.zeros(model.nv)
        ctrl = np.zeros(model.nu)
        act = np.zeros(model.na)
        times = []
        for entity_name, source in sources.items():
            key_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_KEY, name)
            if key_id < 0:
                continue
            times.append(float(source.key_time[key_id]))
            for field in (source.key_qpos, source.key_qvel, source.key_ctrl, source.key_act):
                if not np.isfinite(field[key_id]).all():
                    raise ValueError(f"entity {entity_name!r}: keyframe {name!r} must be finite")
            entity_layout = layout.get_entity(entity_name)
            for joint in entity_layout.joints:
                source_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
                if source_id < 0:
                    raise ValueError(f"keyframe source lost joint {entity_name}/{joint.name}")
                qa, va = int(source.jnt_qposadr[source_id]), int(source.jnt_dofadr[source_id])
                qpos[list(joint.qpos_indices)] = source.key_qpos[
                    key_id, qa : qa + len(joint.qpos_indices)
                ]
                qvel[list(joint.qvel_indices)] = source.key_qvel[
                    key_id, va : va + len(joint.qvel_indices)
                ]
            for local_name, target_id in zip(
                entity_layout.actuator_names, entity_layout.actuator_indices, strict=True
            ):
                source_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_ACTUATOR, local_name)
                if source_id < 0:
                    raise ValueError(f"keyframe source lost actuator {entity_name}/{local_name}")
                ctrl[target_id] = source.key_ctrl[key_id, source_id]
                width = int(model.actuator_actnum[target_id])
                if width != int(source.actuator_actnum[source_id]):
                    raise ValueError("source and scene activation layouts disagree")
                if width:
                    start = int(model.actuator_actadr[target_id])
                    source_start = int(source.actuator_actadr[source_id])
                    act[start : start + width] = source.key_act[
                        key_id, source_start : source_start + width
                    ]
        if not np.isfinite(times).all() or any(time != times[0] for time in times):
            raise ValueError(f"keyframe {name!r} has conflicting or nonfinite source times")
        spec.add_key(
            name=name,
            time=times[0],
            qpos=qpos.tolist(),
            qvel=qvel.tolist(),
            ctrl=ctrl.tolist(),
            act=act.tolist(),
            mpos=initial.mocap_pos.ravel().tolist(),
            mquat=initial.mocap_quat.ravel().tolist(),
        )
    return names


def compose_scene(scene: SceneCfg, num_envs: int, sim_dt: float) -> ComposedScene:
    """Compile independent full-scene realizations with immutable assignments.

    Root declarations override source and keyframe root poses; their root
    velocities are overridden to zero. Joint ref/qpos0 is preserved. Named
    source keys are merged by name; an entity missing a key contributes its
    qpos0 and zero qvel/ctrl/act. Shared key times must agree. Mirrors retain
    independent initial mocap poses. Named default selection must exist in
    every realization; selection itself remains the consuming backend's job.
    Global options must agree across every source and variant, except factory dt.
    """
    scene.validate_composition(num_envs)
    if not scene.entity_assets:
        raise ValueError("compose_scene requires entity_assets")
    sensor_fragments = _load_sensor_fragments(scene)
    if scene.terrain is not None or scene.visual_model_file is not None:
        raise NotImplementedError(
            "entity composition does not yet support terrain/visual override"
        )
    if scene.default_keyframe_name is not None and (
        not isinstance(scene.default_keyframe_name, str) or not scene.default_keyframe_name
    ):
        raise ValueError("default_keyframe_name must be a non-empty string or None")
    if not np.isfinite(sim_dt) or sim_dt <= 0:
        raise ValueError("sim_dt must be finite and positive")
    binding = scene.entity_variant
    if binding is not None and binding.plan.layout is not FixedVariantLayout.SAME_LAYOUT:
        raise NotImplementedError("composition currently supports same_layout variants only")
    entities = scene.entity_assets
    physical = {entity.name: entity for entity in entities if entity.mirror_of is None}
    count = 1 if binding is None else len(binding.plan.variants)
    directory = tempfile.TemporaryDirectory(prefix="unisim-entities-")
    files: list[ModelSourceDescriptor] = []
    layouts: list[CompiledSceneLayout] = []
    source_provenance: list[SceneSourceProvenance] = []
    canonical_model = None
    reference_layout = None
    reference_sensors = None
    reference_keys = None
    reference_activation = None
    reference_options = None
    try:
        # Compile the declared base source too: a catalog must not silently
        # replace the target's advertised public topology with another one.
        for variant in range(-1 if binding is not None else 0, count):
            assembled = mujoco.MjSpec()
            source_models: dict[str, mujoco.MjModel] = {}
            for entity in entities:
                source_entity = physical[entity.mirror_of or entity.name]
                assert source_entity.source is not None
                source = source_entity.source.model_file
                if (
                    binding is not None
                    and variant >= 0
                    and source_entity.name == binding.target_entity
                ):
                    source = binding.plan.variants[variant].model_file
                spec, original_model, provenance = load_entity_source(
                    entity, source, mirror=entity.mirror_of is not None
                )
                record_variant = (
                    variant
                    if binding is not None
                    and variant >= 0
                    and source_entity.name == binding.target_entity
                    else None
                )
                record = replace(provenance, variant=record_variant)
                if record.identity_payload() not in {
                    item.identity_payload() for item in source_provenance
                }:
                    source_provenance.append(record)
                if entity.mirror_of is None:
                    source_models[entity.name] = original_model
                source_model = spec.compile()
                options = _options(source_model)
                if reference_options is None:
                    reference_options = options
                elif any(
                    not np.array_equal(value, reference_options[name])
                    for name, value in options.items()
                ):
                    raise ValueError(f"entity {entity.name!r}: conflicting global physics options")
                for name, value in options.items():
                    setattr(assembled.option, name, value.item() if value.ndim == 0 else value)
                assembled.option.timestep = sim_dt
                assembled.attach(
                    spec, prefix=entity.name + "/", frame=assembled.worldbody.add_frame()
                )
            _add_sensor_fragments(assembled, sensor_fragments)
            model = assembled.compile()
            layout = compile_scene_layout(model, entities)
            keys = _merge_keys(assembled, model, layout, source_models, scene.default_keyframe_name)
            activation = tuple(int(n) for n in model.actuator_actnum)
            if keys:
                model = assembled.compile()
            if reference_layout is not None:
                reference_layout.require_same_layout(layout)
                if _sensor_signature(model) != reference_sensors:
                    raise ValueError("variants change the public sensor layout")
                if keys != reference_keys or activation != reference_activation:
                    raise ValueError("variants change keyframe names or actuator activation widths")
            else:
                reference_layout = layout
                reference_sensors = _sensor_signature(model)
                reference_keys = keys
                reference_activation = activation
            if variant < 0:
                continue
            if canonical_model is None:
                canonical_model = model
            filename = Path(directory.name) / f"scene-{variant}.xml"
            assembled.to_file(str(filename))
            # Serialized XML, not only the in-memory spec, is the executor input.
            loaded = mujoco.MjModel.from_xml_path(str(filename))
            layout.require_same_layout(compile_scene_layout(loaded, entities))
            files.append(ModelSourceDescriptor(str(filename)))
            layouts.append(layout)
        assert canonical_model is not None
        plan = None if binding is None else FixedVariantPlan(binding.plan.assignment, tuple(files))
        parameters = SceneCompilerParameters(
            PORTABLE_MJCF_PROFILE_ID,
            PORTABLE_MJCF_STRUCTURAL_ORACLE,
            str(mujoco.__version__),
            sim_dt,
            scene.default_keyframe_name,
            () if binding is None else tuple(int(i) for i in binding.plan.assignment),
            tuple(fragment.digest for fragment in sensor_fragments),
        )
        ordered_provenance = tuple(source_provenance)
        content_identity = compute_scene_content_identity(ordered_provenance, parameters)
        source_provenance_value = "unisim portable MJCF cold path"
        intent_fields = (
            ConfigurationField(
                "asset.profile",
                PORTABLE_MJCF_PROFILE_ID,
                provenance=(ConfigurationProvenance("source", source_provenance_value),),
                reason="Native effective support is reported after adapter materialization.",
            ),
            ConfigurationField(
                "dt",
                sim_dt,
                provenance=(ConfigurationProvenance("source", source_provenance_value),),
                unit="s",
                reason="Native timestep readback is reported after adapter materialization.",
            ),
            ConfigurationField(
                "scene.entities",
                tuple(entity.name for entity in entities),
                provenance=(ConfigurationProvenance("source", source_provenance_value),),
                reason="Native entity materialization is reported by each adapter.",
            ),
        )
        intent_report = SceneIntentReport(
            PORTABLE_MJCF_PROFILE_ID,
            parameters,
            ordered_provenance,
            content_identity,
            intent_fields,
        )
        return ComposedScene(
            files[0].model_file,
            plan,
            layouts[0],
            tuple(layouts),
            canonical_model,
            ordered_provenance,
            content_identity,
            intent_report,
            directory,
        )
    except BaseException:
        directory.cleanup()
        raise
