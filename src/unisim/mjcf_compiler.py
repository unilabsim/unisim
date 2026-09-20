"""MuJoCo structural oracle for the common portable MJCF cold path."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, overload

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


class _LazyVariantLayouts(Sequence[CompiledSceneLayout]):
    """Rebuild variant layouts on access without retaining one per catalog row."""

    def __init__(
        self,
        files: Sequence[ModelSourceDescriptor],
        entities: tuple[SceneEntitySpec, ...],
        canonical_layout: CompiledSceneLayout,
        canonical_index: int,
        same_layout: bool,
    ) -> None:
        self._files = tuple(files)
        self._entities = entities
        self._canonical_layout = canonical_layout
        self._canonical_index = canonical_index
        self._same_layout = same_layout

    def __len__(self) -> int:
        return len(self._files)

    @overload
    def __getitem__(self, index: int) -> CompiledSceneLayout: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CompiledSceneLayout, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> CompiledSceneLayout | tuple[CompiledSceneLayout, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._same_layout or index == self._canonical_index:
            return self._canonical_layout
        model = mujoco.MjModel.from_xml_path(self._files[index].model_file)
        return compile_scene_layout(model, self._entities)

    def __iter__(self) -> Iterator[CompiledSceneLayout]:
        for index in range(len(self)):
            yield self[index]


@dataclass
class ComposedScene:
    """Own generated full-scene sources until their last executor is closed."""

    model_file: str
    variant_plan: FixedVariantPlan | None
    layout: CompiledSceneLayout
    variant_layouts: Sequence[CompiledSceneLayout]
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


def _layout_without_geoms(layout: CompiledSceneLayout) -> CompiledSceneLayout:
    return replace(
        layout,
        entities=tuple(replace(entity, geoms=()) for entity in layout.entities),
        ngeom=0,
    )


@dataclass(frozen=True)
class UniformVariantLayoutSummary:
    """Variant-delta information needed to validate uniform public geoms."""

    ngeom: int
    geoms: tuple[tuple[str, int, str], ...]


def uniform_variant_body_names(
    model: mujoco.MjModel,
    layout: CompiledSceneLayout,
    affected_entities: tuple[str, ...],
) -> frozenset[str]:
    """Return public body names owned by a variant binding and its mirrors."""

    return frozenset(
        str(model.body(body_id).name)
        for name in affected_entities
        for body_id in layout.get_entity(name).body_ids
    )


def summarize_uniform_variant_layout(
    model: mujoco.MjModel,
    layout: CompiledSceneLayout,
    affected_body_names: frozenset[str],
) -> UniformVariantLayoutSummary:
    """Capture uniform-public validation inputs without retaining an MjModel."""

    return UniformVariantLayoutSummary(
        int(model.ngeom),
        tuple(
            (
                str(model.geom(index).name),
                int(model.geom_type[index]),
                str(model.body(int(model.geom_bodyid[index])).name),
            )
            for index in range(model.ngeom)
            if str(model.body(int(model.geom_bodyid[index])).name) in affected_body_names
        ),
    )


def validate_uniform_entity_variant_layout_summaries(
    bootstrap_summary: UniformVariantLayoutSummary,
    variant_summaries: tuple[UniformVariantLayoutSummary, ...],
) -> int:
    """Validate portable uniform-public semantics from compact summaries."""

    if not variant_summaries:
        raise ValueError("uniform entity variants require compiled source layouts")

    canonical_index = max(
        range(len(variant_summaries)),
        key=lambda index: (variant_summaries[index].ngeom, -index),
    )
    canonical = variant_summaries[canonical_index]
    canonical_names = tuple(item[0] for item in canonical.geoms)
    canonical_ids = {
        name: index for index, (name, _geom_type, _body_name) in enumerate(canonical.geoms)
    }
    if "" in canonical_names or len(canonical_ids) != len(canonical_names):
        raise ValueError("uniform entity variants require unique non-empty canonical geom names")

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    checks: tuple[tuple[str, UniformVariantLayoutSummary], ...] = (
        ("base source", bootstrap_summary),
    )
    checks += tuple(
        (f"variant {index}", summary) for index, summary in enumerate(variant_summaries)
    )
    for label, summary in checks:
        names = tuple(item[0] for item in summary.geoms)
        if "" in names or len(set(names)) != len(names):
            raise ValueError(f"{label} requires unique non-empty variant geom names")
        unknown = set(names) - set(canonical_names)
        if unknown:
            raise ValueError(
                f"{label} has geoms absent from the variant catalog union: {sorted(unknown)}"
            )
        for name, geom_type, body_name in summary.geoms:
            canonical_type = canonical.geoms[canonical_ids[name]][1]
            canonical_body = canonical.geoms[canonical_ids[name]][2]
            if geom_type != canonical_type:
                raise ValueError(f"{label} changes the type of present geom {name!r}")
            if body_name != canonical_body:
                raise ValueError(f"{label} moves geom {name!r} to a different body")
        missing = set(canonical_names) - set(names)
        if any(canonical.geoms[canonical_ids[name]][1] != mesh_type for name in missing):
            raise ValueError(
                "uniform_public_layout only permits optional mesh-geom slots to be absent"
            )
    return canonical_index


def validate_uniform_entity_variant_layouts(
    bootstrap_model: mujoco.MjModel,
    bootstrap_layout: CompiledSceneLayout,
    variant_models: tuple[mujoco.MjModel, ...],
    variant_layouts: tuple[CompiledSceneLayout, ...],
    affected_entities: tuple[str, ...],
) -> int:
    """Validate portable uniform-public semantics and return its canonical index.

    ``bootstrap_model`` is the target entity's declared base source in the
    compiler. A MuJoCo backend can pass its already-canonical source model to
    revalidate independently loaded realizations. In both cases the catalog
    variants, not the bootstrap model, define the executor's canonical layout.
    """
    if not variant_models or len(variant_models) != len(variant_layouts):
        raise ValueError("uniform entity variants require compiled source layouts")

    affected_body_names = uniform_variant_body_names(
        bootstrap_model, bootstrap_layout, affected_entities
    )
    bootstrap_core = _layout_without_geoms(bootstrap_layout)
    layouts = (bootstrap_layout, *variant_layouts)
    labels = ("base source", *(f"variant {index}" for index in range(len(variant_models))))
    for label, layout in zip(labels, layouts, strict=True):
        try:
            bootstrap_core.require_same_layout(_layout_without_geoms(layout))
        except ValueError as exc:
            raise ValueError(f"{label} changes uniform entity public topology") from exc
    return validate_uniform_entity_variant_layout_summaries(
        summarize_uniform_variant_layout(
            bootstrap_model,
            bootstrap_layout,
            affected_body_names,
        ),
        tuple(
            summarize_uniform_variant_layout(model, layout, affected_body_names)
            for model, layout in zip(variant_models, variant_layouts, strict=True)
        ),
    )


def _namespace_uniform_variant_meshes(spec: mujoco.MjSpec, variant: int) -> None:
    """Give catalog mesh assets stable per-variant names for executor pooling.

    Mesh asset names are intentionally not part of the public layout.  Keeping
    the geom slot name fixed while making catalog mesh definitions unique lets
    executors pool different mesh data without treating the authored asset name
    as shared topology.
    """
    for mesh in spec.meshes:
        old_name = mesh.name
        new_name = f"__unisim_variant_{variant}_{old_name}"
        mesh.name = new_name
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname == old_name:
                geom.meshname = new_name


def _unscope_uniform_variant_meshes(assembled: mujoco.MjSpec, entity_name: str) -> None:
    """Remove attachment scopes from catalog mesh names before serialization."""
    prefix = entity_name + "/"
    for mesh in tuple(assembled.meshes):
        old_name = mesh.name
        if not old_name.startswith(prefix):
            continue
        new_name = entity_name + "__" + old_name[len(prefix) :]
        mesh.name = new_name
        for geom in assembled.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname == old_name:
                geom.meshname = new_name


def _copy_mesh_definition(
    target: mujoco.MjSpec, source: mujoco.MjSpec, mesh: mujoco.MjsMesh, name: str
) -> None:
    path = None
    if mesh.file:
        candidate = Path(mesh.file)
        if not candidate.is_absolute():
            candidate = Path(source.modelfiledir or ".") / source.compiler.meshdir / candidate
        path = candidate.resolve()
    copied = target.add_mesh(name=name)
    copied.file = "" if path is None else str(path)
    if mesh.content_type:
        copied.content_type = mesh.content_type
    for field in ("refpos", "refquat", "scale"):
        setattr(copied, field, np.asarray(getattr(mesh, field), copy=True))
    for field in (
        "inertia",
        "smoothnormal",
        "needsdf",
        "maxhullvert",
        "octree_maxdepth",
        "material",
        "plugin",
    ):
        setattr(copied, field, getattr(mesh, field))
    for field in (
        "uservert",
        "usernormal",
        "usertexcoord",
        "userface",
        "userfacenormal",
        "userfacetexcoord",
    ):
        setattr(copied, field, list(getattr(mesh, field)))


def _merge_uniform_variant_mesh_catalog(
    assembled: mujoco.MjSpec,
    variant: int,
    variant_bindings: tuple[tuple[int, str, str], ...],
) -> None:
    """Seed the canonical realization with the complete catalog mesh pool.

    Only the canonical realization needs the complete pool.  Noncanonical
    realizations retain their source-owned definitions, while the canonical
    source provides every mesh identity needed by executors that map per-world
    ``geom_dataid`` rows.
    """
    existing = {mesh.name for mesh in assembled.meshes}
    probes: list[mujoco.MjsGeom] = []
    for other_variant, bound_entity, source_file in variant_bindings:
        if other_variant == variant:
            continue
        source = mujoco.MjSpec.from_file(source_file)
        _namespace_uniform_variant_meshes(source, other_variant)
        for mesh in source.meshes:
            name = bound_entity + "__" + mesh.name
            if name in existing:
                continue
            _copy_mesh_definition(assembled, source, mesh, name)
            existing.add(name)
            probe = assembled.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                name=f"__unisim_mesh_probe_{len(probes)}",
            )
            probe.meshname = name
            probes.append(probe)
        del source
    if probes:
        # Attachment omits unused mesh definitions.  Compile once through
        # disposable world geoms so the final public layout keeps the pool.
        assembled.compile()
        for probe in probes:
            assembled.delete(probe)


def _load_sensor_fragments(scene: SceneCfg) -> tuple[_SensorFragment, ...]:
    """Load the scene-level, sensor-only portable MJCF authoring additions.

    Entity sources remain independently valid MJCF documents.  A fragment may
    introduce ordered collision-pair force sensors, body-net (wildcard, geom2
    omitted) force/found sensors, world-referenced body/site pose sensors, or
    world-referenced body/site motion sensors whose object names are in the
    final ``entity/local-name`` namespace; the compiler resolves them after
    entity attachment.
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
                        "world-referenced body/site FramePos/FrameQuat and "
                        "body/site FrameLinVel/FrameAngVel sensors"
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
                if data == "force" and attributes <= force_attributes:
                    if item.attrib.get("reduce") != "netforce":
                        raise ValueError(
                            f"portable sensor fragment {path} contact sensor {name!r} "
                            "supports only data='force' reduce='netforce'"
                        )
                    intprm = _CONTACT_FORCE_SENSOR_INTPRM
                elif data == "found" and attributes <= found_attributes:
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
                # Omitting geom2 declares the MuJoCo wildcard form: the net
                # force/found reduction over every contact involving geom1.
                if (
                    not geom1
                    or geom1.count("/") != 1
                    or (geom2 and geom2.count("/") != 1)
                ):
                    raise ValueError(
                        f"portable sensor fragment {path} contact sensor {name!r} requires "
                        "geoms in entity/local-name form"
                    )
                names.add(name)
                sensors.append(_CrossEntityContactSensor(name, geom1, geom2, intprm))
        if not sensors:
            raise ValueError(f"portable sensor fragment {path} contains no sensors")
        digest = sha256(path.read_bytes()).hexdigest()
        fragments.append(_SensorFragment(path, digest, tuple(sensors)))
    return tuple(fragments)


def _add_sensor_fragments(assembled: mujoco.MjSpec, fragments: tuple[_SensorFragment, ...]) -> None:
    """Add scene-level sensors after all entity geoms have been attached."""

    for fragment in fragments:
        for sensor in fragment.sensors:
            if isinstance(sensor, _CrossEntityContactSensor):
                contact_args: dict[str, Any] = {}
                if sensor.geom2:
                    contact_args = {
                        "reftype": mujoco.mjtObj.mjOBJ_GEOM,
                        "refname": sensor.geom2,
                    }
                assembled.add_sensor(
                    name=sensor.name,
                    type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM,
                    objname=sensor.geom1,
                    intprm=sensor.intprm,
                    **contact_args,
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
        raise NotImplementedError("entity composition does not yet support terrain/visual override")
    if scene.default_keyframe_name is not None and (
        not isinstance(scene.default_keyframe_name, str) or not scene.default_keyframe_name
    ):
        raise ValueError("default_keyframe_name must be a non-empty string or None")
    if not np.isfinite(sim_dt) or sim_dt <= 0:
        raise ValueError("sim_dt must be finite and positive")
    binding = scene.entity_variant
    uniform = binding is not None and (
        binding.plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
    )
    entities = scene.entity_assets
    physical = {entity.name: entity for entity in entities if entity.mirror_of is None}
    count = 1 if binding is None else len(binding.plan.variants)
    directory = tempfile.TemporaryDirectory(prefix="unisim-entities-")
    files: list[ModelSourceDescriptor] = []
    source_provenance: list[SceneSourceProvenance] = []
    canonical_model = None
    bootstrap_summary: UniformVariantLayoutSummary | None = None
    variant_summaries: list[UniformVariantLayoutSummary] = []
    canonical_index = 0
    canonical_spec: mujoco.MjSpec | None = None
    canonical_layout: CompiledSceneLayout | None = None
    canonical_ngeom = -1
    reference_layout = None
    reference_sensors = None
    reference_keys = None
    reference_activation = None
    reference_options = None
    affected_entities = (
        tuple(
            entity.name
            for entity in entities
            if physical[entity.mirror_of or entity.name].name == binding.target_entity
        )
        if binding is not None
        else ()
    )
    affected_body_names: frozenset[str] = frozenset()
    bootstrap_core: CompiledSceneLayout | None = None
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
                if (
                    binding is not None
                    and binding.plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
                    and variant >= 0
                    and source_entity.name == binding.target_entity
                ):
                    _namespace_uniform_variant_meshes(spec, variant)
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
                if (
                    binding is not None
                    and binding.plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
                    and variant >= 0
                    and source_entity.name == binding.target_entity
                ):
                    _unscope_uniform_variant_meshes(assembled, entity.name)
            _add_sensor_fragments(assembled, sensor_fragments)
            model = assembled.compile()
            layout = compile_scene_layout(model, entities)
            keys = _merge_keys(assembled, model, layout, source_models, scene.default_keyframe_name)
            activation = tuple(int(n) for n in model.actuator_actnum)
            if keys:
                model = assembled.compile()
            if reference_layout is not None:
                if uniform:
                    _layout_without_geoms(reference_layout).require_same_layout(
                        _layout_without_geoms(layout)
                    )
                else:
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
                    if uniform:
                        affected_body_names = uniform_variant_body_names(
                            model, layout, affected_entities
                        )
                        bootstrap_core = _layout_without_geoms(layout)
                        bootstrap_summary = summarize_uniform_variant_layout(
                            model,
                            layout,
                            affected_body_names,
                        )
                    continue

            filename = Path(directory.name) / f"scene-{variant}.xml"
            assembled.to_file(str(filename))
            # Serialized XML, not only the in-memory spec, is the executor input.
            loaded = mujoco.MjModel.from_xml_path(str(filename))
            loaded_layout = compile_scene_layout(loaded, entities)
            layout.require_same_layout(loaded_layout)
            files.append(ModelSourceDescriptor(str(filename)))
            if uniform:
                assert bootstrap_core is not None
                try:
                    bootstrap_core.require_same_layout(_layout_without_geoms(loaded_layout))
                except ValueError as exc:
                    raise ValueError(
                        f"variant {variant} changes uniform entity public topology"
                    ) from exc
                variant_summaries.append(
                    summarize_uniform_variant_layout(
                        loaded,
                        loaded_layout,
                        affected_body_names,
                    )
                )
                if loaded.ngeom > canonical_ngeom:
                    canonical_index = variant
                    canonical_spec = assembled
                    canonical_layout = loaded_layout
                    canonical_ngeom = int(loaded.ngeom)
            elif variant == 0:
                canonical_spec = assembled
                canonical_layout = loaded_layout

        assert files and canonical_spec is not None and canonical_layout is not None
        if uniform:
            assert bootstrap_summary is not None
            assert binding is not None
            validated_index = validate_uniform_entity_variant_layout_summaries(
                bootstrap_summary,
                tuple(variant_summaries),
            )
            if validated_index != canonical_index:
                raise AssertionError(
                    "uniform canonical variant selection changed after serialization"
                )
            canonical_index = validated_index
            variant_bindings = tuple(
                (
                    variant,
                    entity.name,
                    binding.plan.variants[variant].model_file,
                )
                for entity in entities
                if physical[entity.mirror_of or entity.name].name == binding.target_entity
                for variant in range(len(binding.plan.variants))
            )
            _merge_uniform_variant_mesh_catalog(
                canonical_spec,
                canonical_index,
                variant_bindings,
            )
            canonical_spec.to_file(files[canonical_index].model_file)
            canonical_model = mujoco.MjModel.from_xml_path(files[canonical_index].model_file)
            canonical_layout = compile_scene_layout(canonical_model, entities)
        else:
            canonical_model = mujoco.MjModel.from_xml_path(files[canonical_index].model_file)
        plan = (
            None
            if binding is None
            else FixedVariantPlan(binding.plan.assignment, tuple(files), layout=binding.plan.layout)
        )
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
            files[canonical_index].model_file,
            plan,
            canonical_layout,
            _LazyVariantLayouts(
                files,
                entities,
                canonical_layout,
                canonical_index,
                not uniform,
            ),
            canonical_model,
            ordered_provenance,
            content_identity,
            intent_report,
            directory,
        )
    except BaseException:
        directory.cleanup()
        raise
