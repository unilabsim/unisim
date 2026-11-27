from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

# The MJCF joint-layout extractor lives in the shared, stdlib-only
# ``unisim.backend.mjcf_layout`` module so non-Motrix adapters (e.g. the
# IsaacGym playback validation) reuse one implementation; the historical
# motrix.scene import path keeps working through this re-export.
from unisim.backend.mjcf_layout import (  # noqa: F401
    MjcfJointLayoutEntry,
    extract_mjcf_joint_layout,
)
from unisim.backend.scene_cleanup import remove_temporary_file
from unisim.scene import resolve_scene_fragment_path
from unisim.terrain.generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from motrixsim import SceneModel
    from motrixsim.msd import Link, World


@dataclass(frozen=True)
class _MotrixFrameSensorIdentity:
    """Public Motrix frame-sensor identity fields used by portable audits."""

    name: str
    sensor_type: Any
    object_type: Any
    reference_frame: str


@dataclass(frozen=True)
class _MotrixContactSensorIdentity:
    """Public Motrix contact-sensor identity fields used by portable audits."""

    name: str
    match_variant: str  # "geom_pair" | "link_pair" | "subtree_pair"
    value: tuple[str, str]  # geom names for geom_pair, link names otherwise
    reduce_mode: Any
    reports_force: bool
    reports_found: bool


@dataclass(frozen=True)
class _MotrixSensorInventory:
    """Cold-path native sensor names and reviewed sensor identities."""

    names: tuple[str, ...]
    frame_identities: tuple[_MotrixFrameSensorIdentity, ...]
    contact_identities: tuple[_MotrixContactSensorIdentity, ...]
    signatures: tuple[tuple[str, str, str], ...] = ()
    categories: Mapping[str, str] | None = None


def _motrix_sensor_signatures(world: "World") -> tuple[tuple[str, str, str], ...]:
    signatures: list[tuple[str, str, str]] = []
    for sensor in world.sensors.contact:
        if not sensor.name:
            continue
        report = sensor.report
        contact_identity = (
            sensor.match_.variant,
            sensor.match_.value,
            int(sensor.max_num),
            str(sensor.reduce),
            bool(report.found),
            bool(report.force),
            bool(report.torque),
            bool(report.dist),
            bool(report.pos),
            bool(report.normal),
            bool(report.tangent),
        )
        signatures.append((str(sensor.name), "contact", repr(contact_identity)))
    for sensor in world.sensors.frame:
        if sensor.name:
            frame_identity = (sensor.sensor_type, sensor.object_type, sensor.ref_frame)
            signatures.append((str(sensor.name), "frame", repr(frame_identity)))
    for sensor in world.sensors.joint:
        if sensor.name:
            signatures.append((str(sensor.name), "joint", repr((sensor.joint, sensor.type_))))
    for sensor in world.sensors.subtree:
        if sensor.name:
            signatures.append((str(sensor.name), "subtree", repr((sensor.link, sensor.type_))))
    for sensor in world.sensors.touch:
        if sensor.name:
            signatures.append((str(sensor.name), "touch", sensor.site))
    for sensor in world.sensors.actuator:
        if sensor.name:
            signatures.append((str(sensor.name), "actuator", repr(sensor.actuator)))
    return tuple(signatures)


def _motrix_sensor_groups(world: "World") -> tuple[tuple[str, Any], ...]:
    """Return every native named-sensor group in public-inventory order."""

    return (
        ("actuator", world.sensors.actuator),
        ("contact", world.sensors.contact),
        ("frame", world.sensors.frame),
        ("joint", world.sensors.joint),
        ("subtree", world.sensors.subtree),
        ("touch", world.sensors.touch),
    )


def _motrix_sensor_names(world: "World") -> tuple[str, ...]:
    """Collect the names accepted by Motrix's native sensor accessor once."""

    names = tuple(
        str(sensor.name)
        for _, group in _motrix_sensor_groups(world)
        for sensor in group
        if sensor.name
    )
    if len(set(names)) != len(names):
        raise ValueError(f"Motrix scene contains duplicate sensor names: {names}")
    return names


def _motrix_sensor_categories(world: "World") -> Mapping[str, str]:
    """Classify every named native sensor into the public sensor categories.

    Native sensor-type enums are unhashable SDK objects, so the classification
    keys on their stable string forms.
    """

    frame_categories = {
        "FrameSensorType.FramePos": "position",
        "FrameSensorType.FrameQuat": "position",
        "FrameSensorType.XAxis": "position",
        "FrameSensorType.YAxis": "position",
        "FrameSensorType.ZAxis": "position",
        "FrameSensorType.FrameLinVel": "velocity",
        "FrameSensorType.FrameAngVel": "velocity",
        "FrameSensorType.FrameLinAcc": "acceleration",
    }
    subtree_categories = {
        "SubtreeSensorType.Com": "position",
        "SubtreeSensorType.LinVel": "velocity",
        "SubtreeSensorType.AngMom": "other",
    }
    categories: dict[str, str] = {}
    for sensor in world.sensors.frame:
        if sensor.name:
            categories[str(sensor.name)] = frame_categories.get(
                str(sensor.sensor_type), "other"
            )
    for sensor in world.sensors.contact:
        if sensor.name:
            categories[str(sensor.name)] = "contact"
    for sensor in world.sensors.joint:
        if sensor.name:
            joint_type = str(sensor.type_)
            if joint_type == "JointSensorType.JointPos":
                categories[str(sensor.name)] = "position"
            elif joint_type == "JointSensorType.JointVel":
                categories[str(sensor.name)] = "velocity"
            else:
                categories[str(sensor.name)] = "other"
    for sensor in world.sensors.subtree:
        if sensor.name:
            categories[str(sensor.name)] = subtree_categories.get(str(sensor.type_), "other")
    for sensor in world.sensors.touch:
        if sensor.name:
            categories[str(sensor.name)] = "contact"
    for sensor in world.sensors.actuator:
        if sensor.name:
            categories[str(sensor.name)] = "force"
    return categories


def _motrix_frame_quat_sensor_names(world: "World") -> tuple[str, ...]:
    """Collect Motrix FrameQuat sensors that need public ``wxyz`` conversion."""
    import motrixsim.msd as msd

    return tuple(
        str(sensor.name)
        for sensor in world.sensors.frame
        if sensor.name and sensor.sensor_type == msd.FrameSensorType.FrameQuat
    )


def _motrix_frame_sensor_identities(world: "World") -> tuple[_MotrixFrameSensorIdentity, ...]:
    return tuple(
        _MotrixFrameSensorIdentity(
            name=str(sensor.name),
            sensor_type=sensor.sensor_type,
            object_type=sensor.object_type,
            reference_frame=str(sensor.ref_frame),
        )
        for sensor in world.sensors.frame
        if sensor.name
    )


def _motrix_contact_sensor_identities(
    world: "World",
) -> tuple[_MotrixContactSensorIdentity, ...]:
    """Collect the public identity of each native contact sensor.

    The MotrixSim importer maps MuJoCo ``body1/body2`` filters to native
    ``link_pair`` matches and ``subtree1/subtree2`` to ``subtree_pair``; geom
    pairs keep ``geom_pair``.  All three variants are published here so the
    single-MJCF profile accepts the same authored contact sensors as the
    MuJoCo adapters (site filters never survive the importer and fail closed
    in the materializer's authored-sensor audit).
    """

    identities: list[_MotrixContactSensorIdentity] = []
    for sensor in world.sensors.contact:
        if not sensor.name:
            continue
        variant = str(sensor.match_.variant)
        if variant not in ("geom_pair", "link_pair", "subtree_pair"):
            raise ValueError(
                f"Motrix contact sensor {sensor.name!r} has unsupported match {variant!r}"
            )
        first, second = (str(name) for name in sensor.match_.value)
        identities.append(
            _MotrixContactSensorIdentity(
                name=str(sensor.name),
                match_variant=variant,
                value=(first, second),
                reduce_mode=sensor.reduce,
                reports_force=bool(sensor.report.force),
                reports_found=bool(sensor.report.found),
            )
        )
    return tuple(identities)


def materialize_expanded_scene_with_sensor_inventory(
    *,
    model_file: str,
    add_body_sensors: bool,
    base_name: str,
    mesh_variant_sets: Mapping[str, Sequence[str]] | None = None,
    geom_variant_sets: Mapping[str, str] | None = None,
) -> tuple["SceneModel", _MotrixSensorInventory, "World"]:
    """Import an expanded source and retain its cold-path sensor inventory.

    The built ``World`` is returned alongside the model because MotrixSim 1.0.0
    configures the system camera render target on the MSD world before
    ``build()``; the built model is immutable.
    """

    import motrixsim.msd as msd

    world = msd.from_file(str(Path(model_file).resolve()))
    contact_identities = _motrix_contact_sensor_identities(world)
    if mesh_variant_sets or geom_variant_sets:
        _register_motrix_mesh_variant_sets(
            world,
            mesh_variant_sets=mesh_variant_sets or {},
            geom_variant_sets=geom_variant_sets or {},
        )
    if add_body_sensors:
        add_tracking_frame_sensors(world, base_name=base_name)
    frame_identities = _motrix_frame_sensor_identities(world)
    names = _motrix_sensor_names(world)
    sensor_count = sum(1 for _, group in _motrix_sensor_groups(world) for _ in group)
    if sensor_count != len(names):
        raise ValueError("Motrix portable scenes require every native sensor to be named")
    return (
        msd.build(world),
        _MotrixSensorInventory(
            names=names,
            frame_identities=frame_identities,
            contact_identities=contact_identities,
            categories=_motrix_sensor_categories(world),
        ),
        world,
    )


def _register_motrix_mesh_variant_sets(
    world: "World",
    *,
    mesh_variant_sets: Mapping[str, Sequence[str]],
    geom_variant_sets: Mapping[str, str],
) -> None:
    """Bind native per-instance mesh selections before SceneModel compilation."""

    import motrixsim.msd as msd

    if world.mesh_variant_sets:
        raise ValueError("Motrix portable scenes do not accept authored mesh variant sets")
    available_meshes = set(world.assets.meshes)
    registered: dict[str, tuple[str, ...]] = {}
    for name, meshes in mesh_variant_sets.items():
        candidates = tuple(str(mesh) for mesh in meshes)
        if not name or name in registered or len(set(candidates)) != len(candidates):
            raise ValueError("Motrix mesh variant sets require unique names and mesh candidates")
        if not candidates or any(mesh not in available_meshes for mesh in candidates):
            raise ValueError(f"Motrix mesh variant set {name!r} has absent mesh candidates")
        variant_set = msd.GeometryMeshVariantSet()
        variant_set.name = name
        variant_set.meshes = list(candidates)
        world.mesh_variant_sets[name] = variant_set
        registered[name] = candidates

    native_by_name, native_by_link_ordinal = index_world_geoms_by_name_and_ordinal(world)

    def resolve(geom_name: str) -> Any:
        entity_name, _, local_name = geom_name.partition("/")
        return resolve_native_geom_handle(
            local_name,
            entity_name,
            native_by_name=native_by_name,
            native_by_link_ordinal=native_by_link_ordinal,
            native_geom_name=lambda geom: geom.name,
        )

    resolved: dict[str, Any] = {}
    missing = []
    for geom_name in geom_variant_sets:
        try:
            resolved[str(geom_name)] = resolve(str(geom_name))
        except ValueError:
            missing.append(str(geom_name))
    if missing:
        # Uniform-public compilation already froze the public geom-name set.
        raise ValueError(
            f"Motrix native geoms are missing from the uniform variant binding: {missing}"
        )
    for geom_name, set_name in geom_variant_sets.items():
        if set_name not in registered:
            raise ValueError(f"Motrix geom {geom_name!r} refers to absent variant set")
        resolved[str(geom_name)].mesh_variant_set = set_name


def _extract_keyframes(fragment_file: Path) -> list[ET.Element]:
    """Return ``<keyframe>`` child elements declared inside ``fragment_file``."""
    root = ET.parse(fragment_file).getroot()
    return list(root.findall("keyframe"))


def _materialize_robot_with_fragment_keyframes(
    robot_path: Path, fragment_paths: Sequence[Path]
) -> Path:
    """Inject fragment ``<keyframe>`` blocks into a temporary copy of ``robot_path``.

    motrix's ``msd.from_file`` validates ``<keyframe>`` qpos against the loaded
    model. fragment XMLs only carry sensors/contacts (no body), so a fragment
    with its own keyframe fails to parse on its own. Mujoco backend already
    merges fragments into the scene XML before parsing; this helper does the
    equivalent for the keyframe block so motrix can load a robot model that
    owns the keyframe declared in a sibling fragment.

    Returns the original ``robot_path`` when no fragment has a keyframe.
    """
    fragment_keyframes: list[ET.Element] = []
    for fragment_path in fragment_paths:
        fragment_keyframes.extend(_extract_keyframes(fragment_path))
    if not fragment_keyframes:
        return robot_path

    tree = ET.parse(robot_path)
    root = tree.getroot()
    existing = root.find("keyframe")
    if existing is None:
        existing = ET.SubElement(root, "keyframe")
    for keyframe in fragment_keyframes:
        existing.extend(list(keyframe))

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{robot_path.name}",
        dir=str(robot_path.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _materialize_fragment_without_keyframes(fragment_file: Path) -> Path:
    """Strip ``<keyframe>`` from a fragment XML; return original if no change."""
    tree = ET.parse(fragment_file)
    root = tree.getroot()
    keyframes = root.findall("keyframe")
    if not keyframes:
        return fragment_file
    for keyframe in keyframes:
        root.remove(keyframe)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{fragment_file.name}",
        dir=str(fragment_file.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _restore_imported_actuator_forms(model_path: Path) -> Path:
    """Rewrite importer-unsupported actuator forms into a same-directory copy.

    MotrixSim's MJCF importer ignores ``<general input="pos">`` (losing the
    kp/kd override surface) and drops ``<jointactuatorfrc>`` sensors.  The
    canonical equivalents (``<position>``, ``<actuatorfrc>``) round-trip only
    where the shared stdlib rewrite preserves authored semantics exactly;
    anything else keeps the authored form and fails closed downstream.  The
    copy sits next to the source so relative asset paths keep resolving.
    Returns the original path when no rewrite applies.
    """
    from unisim.backend.mjcf_actuators import (
        restore_joint_actuator_force_sensors,
        restore_position_actuator_shorthand,
    )

    xml = model_path.read_text(encoding="utf-8")
    restored = restore_joint_actuator_force_sensors(restore_position_actuator_shorthand(xml))
    if restored == xml:
        return model_path
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_actuators_{model_path.name}",
        dir=str(model_path.parent),
        mode="w",
        delete=False,
    )
    tmp.write(restored)
    tmp.close()
    return Path(tmp.name)


def _authored_contact_sensor_names(paths: Sequence[Path]) -> set[str]:
    """Collect named authored MJCF contact sensors across source files."""

    names: set[str] = set()
    for path in paths:
        root = ET.parse(path).getroot()
        for element in root.iter("contact"):
            name = element.get("name")
            if name:
                names.add(name)
    return names


def _audit_authored_contact_sensors_imported(world: "World", source_paths: Sequence[Path]) -> None:
    """Fail closed when an authored contact sensor did not survive import.

    The MotrixSim importer silently drops contact sensors with site filters
    (and rejects mixed body/geom filters), so an authored/imported name
    mismatch means the scene lost a sensor the author asked for.
    """

    authored = _authored_contact_sensor_names(source_paths)
    if not authored:
        return
    imported = {str(sensor.name) for sensor in world.sensors.contact if sensor.name}
    missing = sorted(authored - imported)
    if missing:
        raise ValueError(
            "Motrix contact sensor(s) did not survive MJCF import: "
            f"{missing}. The MotrixSim importer supports geom-pair, body-pair and "
            "subtree-pair contact filters; site filters and mixed body/geom "
            "filters are unsupported."
        )


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))
    finally:
        remove_temporary_file(sanitized, original=fragment_file)
    world.attach(fragment)


def _iter_motrix_links(link: Link):
    yield link
    for child in link.children:
        yield from _iter_motrix_links(child)


# Public-layout geom names synthesized by ``unisim.mjcf_compiler`` for geoms
# without an entity-prefixed source name: ``<body>::geom<ordinal>`` where the
# ordinal counts every geom of the body in MJCF definition order.  MotrixSim
# preserves that definition order inside ``link.geoms`` (upstream
# ``motrixsim#22``), so unnamed native geoms are addressable by
# (link name, ordinal) without synthesizing SDK-side names.
_SYNTHESIZED_GEOM_NAME = re.compile(r"^(?P<body>.+)::geom(?P<ordinal>\d+)$")


def parse_synthesized_geom_name(local_name: str) -> tuple[str, int] | None:
    """Return ``(body_name, ordinal)`` for a synthesized public geom name."""

    match = _SYNTHESIZED_GEOM_NAME.match(local_name)
    if match is None:
        return None
    return match.group("body"), int(match.group("ordinal"))


def resolve_native_geom_handle(
    local_name: str,
    entity_name: str,
    *,
    native_by_name: Mapping[str, Any],
    native_by_link_ordinal: Mapping[str, Sequence[Any]],
    native_geom_name: Any,
) -> Any:
    """Resolve a public geom name to its native geom handle.

    Named geoms resolve by exact native name first.  Names synthesized by the
    public layout compiler resolve through the owning link's geom list at the
    recorded ordinal, which must address an actually unnamed native geom.
    """

    full_name = f"{entity_name}/{local_name}"
    handle = native_by_name.get(full_name)
    if handle is not None:
        return handle
    parsed = parse_synthesized_geom_name(local_name)
    if parsed is None:
        raise ValueError(f"Motrix native geom {full_name!r} is missing")
    body_name, ordinal = parsed
    link_name = f"{entity_name}/{body_name}"
    handles = native_by_link_ordinal.get(link_name)
    if handles is None or ordinal >= len(handles):
        raise ValueError(
            f"Motrix native link {link_name!r} has no geom at ordinal {ordinal} "
            f"for synthesized geom {full_name!r}"
        )
    handle = handles[ordinal]
    observed_name = native_geom_name(handle)
    if observed_name is not None:
        raise ValueError(
            f"Motrix synthesized geom {full_name!r} resolved to named native geom "
            f"{observed_name!r}; the native geom order differs from the MJCF source"
        )
    return handle


def index_world_geoms_by_name_and_ordinal(
    world: World,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Index an MSD world's geoms by name and by (link name, definition ordinal)."""

    native_by_name: dict[str, Any] = {}
    native_by_link_ordinal: dict[str, list[Any]] = {}
    for body in world.hierarchy.bodies:
        for link in _iter_motrix_links(body.link):
            handles = native_by_link_ordinal.setdefault(str(link.name), [])
            for geom in link.geoms:
                if geom.name is not None:
                    native_by_name.setdefault(str(geom.name), geom)
                handles.append(geom)
    return native_by_name, native_by_link_ordinal


def _motrix_world_link_names(world: World) -> list[str]:
    names: list[str] = []
    for body in world.hierarchy.bodies:
        for link in _iter_motrix_links(body.link):
            if link.name:
                names.append(link.name)
    return names


def add_tracking_frame_sensors(world: World, *, base_name: str) -> None:
    """Add Motrix-native frame sensors matching the model-file tracking sensor contract.

    Only pose sensors are added: body-frame velocities are computed
    analytically from the world-frame link state (see
    ``MotrixBackend.get_body_*_vel_b``), because MotrixSim frame velocity
    sensors report motion relative to the baselink and degenerate to zero for
    the root body.
    """
    import motrixsim.msd as msd

    link_names = _motrix_world_link_names(world)
    if base_name not in link_names:
        raise ValueError(f"Base link '{base_name}' not found in Motrix scene")

    existing = {sensor.name for sensor in world.sensors.frame if sensor.name}
    sensor_specs = (
        ("track_pos_b", msd.FrameSensorType.FramePos),
        ("track_quat_b", msd.FrameSensorType.FrameQuat),
    )
    ref_frame = msd.FrameSensorRef.object(msd.ObjectType.link(base_name))
    for link_name in link_names:
        object_type = msd.ObjectType.link(link_name)
        for prefix, sensor_type in sensor_specs:
            sensor_name = f"{prefix}_{link_name}"
            if sensor_name in existing:
                continue
            sensor = msd.FrameSensor()
            sensor.name = sensor_name
            sensor.sensor_type = sensor_type
            sensor.object_type = object_type
            sensor.ref_frame = ref_frame
            world.sensors.frame.append(sensor)


def materialize_scene_with_full_sensor_inventory(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> tuple["SceneModel", _MotrixSensorInventory, "World"]:
    """Build a model and retain every public sensor identity for variant audits."""
    import motrixsim.msd as msd

    model_path = Path(model_file).resolve()
    fragment_paths = [
        resolve_scene_fragment_path(fragment_file, model_path) for fragment_file in fragment_files
    ]
    robot_path = _materialize_robot_with_fragment_keyframes(model_path, fragment_paths)
    actuator_path = _restore_imported_actuator_forms(robot_path)
    try:
        world = msd.from_file(str(actuator_path))
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_tracking_frame_sensors(world, base_name=base_name)
        _audit_authored_contact_sensors_imported(world, (actuator_path, *fragment_paths))
        model = msd.build(world)
        return model, _MotrixSensorInventory(
            names=_motrix_sensor_names(world),
            frame_identities=_motrix_frame_sensor_identities(world),
            contact_identities=_motrix_contact_sensor_identities(world),
            signatures=_motrix_sensor_signatures(world),
            categories=_motrix_sensor_categories(world),
        ), world
    finally:
        remove_temporary_file(actuator_path, original=robot_path)
        remove_temporary_file(robot_path, original=model_path)


def materialize_scene_with_sensor_inventory(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> tuple["SceneModel", tuple[str, ...], tuple[str, ...], "World"]:
    """Build a Motrix model and return its validated cold-path sensor inventory."""
    import motrixsim.msd as msd

    model, inventory, world = materialize_scene_with_full_sensor_inventory(
        model_file=model_file,
        fragment_files=fragment_files,
        add_body_sensors=add_body_sensors,
        base_name=base_name,
    )
    return (
        model,
        inventory.names,
        tuple(
            identity.name
            for identity in inventory.frame_identities
            if identity.sensor_type == msd.FrameSensorType.FrameQuat
        ),
        world,
    )


def materialize_scene(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> "SceneModel":
    """Build a MotrixSim model through MSD scene composition."""
    return materialize_scene_with_sensor_inventory(
        model_file=model_file,
        fragment_files=fragment_files,
        add_body_sensors=add_body_sensors,
        base_name=base_name,
    )[0]


def materialize_hfield_attached_scene_with_sensor_inventory(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: bool = False,
) -> tuple[SceneModel, np.ndarray, object | None, tuple[str, ...], tuple[str, ...], "World"]:
    """Build a Motrix terrain model and return its cold-path sensor inventory.

    The built ``World`` is returned alongside the model because MotrixSim 1.0.0
    configures the system camera render target on the MSD world before
    ``build()``; the built model is immutable.
    """
    import motrixsim.msd as msd

    from unisim.terrain.generator import TerrainGenerator

    robot_path = Path(model_file).resolve()
    generated = TerrainGenerator(terrain_cfg).generate()

    world = msd.World()
    world.name = "unilab materialized hfield scene"

    hfield = msd.HFieldSource()
    hfield.nrow = int(generated.heights_yx.shape[0])
    hfield.ncol = int(generated.heights_yx.shape[1])
    # MotrixSim's hfield source uses MuJoCo-style X/Y half extents.
    hfield.size = [float(generated.hfield_size[0]), float(generated.hfield_size[1])]
    hfield.height_scale = float(generated.height_extent)
    # MotrixSim buffers use compiled hfield row order: row 0 is the -Y side.
    hfield_data = np.ascontiguousarray(np.flipud(generated.heights_yx).astype(np.float32))
    hfield.source_type = msd.HFieldSourceType.buffer(
        hfield_data.reshape(-1),
        f"{hfield_name}_buffer",
    )
    world.assets.hfields[hfield_name] = hfield

    terrain_geom = msd.Geometry()
    terrain_geom.name = geom_name
    terrain_geom.shape = msd.ShapeType.HField
    terrain_geom.hfield = hfield_name
    terrain_geom.position = np.asarray(generated.geom_pos, dtype=np.float32)
    terrain_geom.collision_mask = msd.CollisionMask.collide_with_all()
    terrain_geom.physics_material.friction = [1.0, 0.005, 0.0001]
    world.hierarchy.geoms.append(terrain_geom)

    fragment_paths = [
        resolve_scene_fragment_path(fragment_file, robot_path) for fragment_file in fragment_files
    ]
    merged_robot_path = _materialize_robot_with_fragment_keyframes(robot_path, fragment_paths)
    actuator_path = _restore_imported_actuator_forms(merged_robot_path)
    try:
        robot_world = msd.from_file(str(actuator_path))
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        remove_temporary_file(actuator_path, original=merged_robot_path)
        remove_temporary_file(merged_robot_path, original=robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
    _audit_authored_contact_sensors_imported(world, (actuator_path, *fragment_paths))
    if add_body_sensors:
        add_tracking_frame_sensors(world, base_name=base_name)

    model = msd.build(world)
    sampler = generated.surface_sampler() if return_surface_sampler else None
    return (
        model,
        generated.terrain_origins,
        sampler,
        _motrix_sensor_names(world),
        _motrix_frame_quat_sensor_names(world),
        world,
    )


@overload
def materialize_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[False] = False,
) -> tuple[SceneModel, np.ndarray]: ...


@overload
def materialize_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[True],
) -> tuple[SceneModel, np.ndarray, object]: ...


def materialize_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: bool = False,
) -> tuple[SceneModel, np.ndarray] | tuple[SceneModel, np.ndarray, object]:
    """Build a MotrixSim model with generated hfield terrain and attached robot."""
    model, origins, sampler, _, _, _ = materialize_hfield_attached_scene_with_sensor_inventory(
        model_file=model_file,
        terrain_cfg=terrain_cfg,
        fragment_files=fragment_files,
        hfield_name=hfield_name,
        geom_name=geom_name,
        add_body_sensors=add_body_sensors,
        base_name=base_name,
        return_surface_sampler=return_surface_sampler,
    )
    if return_surface_sampler:
        if sampler is None:
            raise RuntimeError("Motrix terrain materialization did not produce a surface sampler")
        return model, origins, sampler
    return model, origins
