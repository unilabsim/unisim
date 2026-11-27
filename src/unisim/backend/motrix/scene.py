from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
class _MotrixJointActuatorForceSource:
    """One native actuator-force sensor and its scalar joint gear."""

    sensor_name: str
    gear: float


@dataclass(frozen=True)
class _MotrixSensorInventory:
    """Cold-path native sensor names and reviewed sensor identities."""

    names: tuple[str, ...]
    frame_identities: tuple[_MotrixFrameSensorIdentity, ...]
    contact_identities: tuple[_MotrixContactSensorIdentity, ...]
    signatures: tuple[tuple[str, str, str], ...] = ()
    categories: Mapping[str, str] | None = None
    joint_actuator_force_sensors: Mapping[
        str, tuple[_MotrixJointActuatorForceSource, ...]
    ] = field(default_factory=dict)


def _motrix_sensor_signatures(
    world: "World",
    joint_actuator_force_sensors: Mapping[
        str, tuple[_MotrixJointActuatorForceSource, ...]
    ],
) -> tuple[tuple[str, str, str], ...]:
    internal_names = {
        source.sensor_name
        for sources in joint_actuator_force_sensors.values()
        for source in sources
    }
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
        if sensor.name and sensor.name not in internal_names:
            signatures.append((str(sensor.name), "actuator", repr(sensor.actuator)))
    for name, sources in joint_actuator_force_sensors.items():
        signatures.append((name, "jointactuatorfrc", repr(sources)))
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


def _motrix_sensor_names(
    world: "World",
    joint_actuator_force_sensors: Mapping[
        str, tuple[_MotrixJointActuatorForceSource, ...]
    ]
    | None = None,
) -> tuple[str, ...]:
    """Collect public sensor names, excluding aggregate implementation details."""

    joint_actuator_force_sensors = joint_actuator_force_sensors or {}
    internal_names = {
        source.sensor_name
        for sources in joint_actuator_force_sensors.values()
        for source in sources
    }
    names = tuple(
        str(sensor.name)
        for _, group in _motrix_sensor_groups(world)
        for sensor in group
        if sensor.name and sensor.name not in internal_names
    ) + tuple(joint_actuator_force_sensors)
    if len(set(names)) != len(names):
        raise ValueError(f"Motrix scene contains duplicate sensor names: {names}")
    return names


def _motrix_sensor_categories(
    world: "World",
    joint_actuator_force_sensors: Mapping[
        str, tuple[_MotrixJointActuatorForceSource, ...]
    ]
    | None = None,
) -> Mapping[str, str]:
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
    joint_actuator_force_sensors = joint_actuator_force_sensors or {}
    internal_names = {
        source.sensor_name
        for sources in joint_actuator_force_sensors.values()
        for source in sources
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
        if sensor.name and sensor.name not in internal_names:
            categories[str(sensor.name)] = "force"
    for name in joint_actuator_force_sensors:
        categories[name] = "force"
    return categories


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

    model_path = Path(model_file).resolve()
    world = msd.from_file(str(model_path))
    _prepare_imported_world(world, model_path)
    if mesh_variant_sets or geom_variant_sets:
        _register_motrix_mesh_variant_sets(
            world,
            mesh_variant_sets=mesh_variant_sets or {},
            geom_variant_sets=geom_variant_sets or {},
        )
    if add_body_sensors:
        add_tracking_frame_sensors(world, base_name=base_name)
    inventory = _build_sensor_inventory(world, (model_path,))
    internal_count = sum(
        len(sources) for sources in inventory.joint_actuator_force_sensors.values()
    )
    sensor_count = sum(1 for _, group in _motrix_sensor_groups(world) for _ in group)
    public_native_names = len(inventory.names) - len(inventory.joint_actuator_force_sensors)
    if sensor_count - internal_count != public_native_names:
        raise ValueError("Motrix portable scenes require every native sensor to be named")
    return msd.build(world), inventory, world


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


def _restore_imported_actuator_forms(world: "World") -> None:
    """Restore exact position-actuator semantics on the imported MSD objects."""

    import motrixsim.msd as msd

    for actuator in world.actuators:
        actuator_type = actuator.actuator_type
        if actuator_type.variant != "general":
            continue
        general = actuator_type.value
        if (
            general.gain != msd.GainType.Fixed
            or general.bias != msd.BiasType.Affine
            or general.dyntype != msd.DynType.None_
        ):
            continue
        kp = float(general.gainprm[0])
        bias = tuple(float(value) for value in general.biasprm[:3])
        if bias[0] != 0.0 or bias[1] != -kp:
            continue
        position = msd.PositionParam()
        position.kp = kp
        position.damping_type = msd.DampingType.Kv
        position.damping_value = -bias[2]
        actuator.actuator_type = msd.ActuatorType.position(position)


def _restore_imported_solver_options(world: "World", source_path: Path) -> None:
    """Apply authored solver fields that MotrixSim's MJCF importer ignores."""

    import motrixsim.msd as msd

    option = ET.parse(source_path).getroot().find("option")
    if option is None:
        return
    solver = option.get("solver")
    solver_types = {"PGS": msd.SolverType.PGS, "Newton": msd.SolverType.Newton}
    if solver in solver_types:
        world.simulate_option.solver_type = solver_types[solver]
    tolerance = option.get("tolerance")
    if tolerance is not None:
        world.simulate_option.constraint_solver_tolerance = float(tolerance)


def _prepare_imported_world(world: "World", source_path: Path) -> None:
    _restore_imported_actuator_forms(world)
    _restore_imported_solver_options(world, source_path)


def _authored_sensor_names(paths: Sequence[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        root = ET.parse(path).getroot()
        for sensor_block in root.iter("sensor"):
            for element in sensor_block:
                name = element.get("name")
                if name:
                    names.add(name)
    return names


def _restore_joint_actuator_force_sensors(
    world: "World", source_paths: Sequence[Path]
) -> dict[str, tuple[_MotrixJointActuatorForceSource, ...]]:
    """Materialize native actuator sensors for MuJoCo's per-joint force sum."""

    import motrixsim.msd as msd

    restored: dict[str, tuple[_MotrixJointActuatorForceSource, ...]] = {}
    used_names = {
        str(sensor.name)
        for _, group in _motrix_sensor_groups(world)
        for sensor in group
        if sensor.name
    }
    used_names.update(_authored_sensor_names(source_paths))
    for source_path in source_paths:
        root = ET.parse(source_path).getroot()
        joint_types = {
            element.get("name"): element.get("type", "hinge") for element in root.iter("joint")
        }
        joint_types.update({element.get("name"): "free" for element in root.iter("freejoint")})
        for sensor_block in root.iter("sensor"):
            for element in sensor_block:
                if element.tag != "jointactuatorfrc" or not element.get("name"):
                    continue
                name = str(element.get("name"))
                joint = element.get("joint")
                if name in restored:
                    raise ValueError(f"Motrix scene contains duplicate sensor name {name!r}")
                if joint_types.get(joint) in {"ball", "free"}:
                    raise NotImplementedError(
                        f"Motrix jointactuatorfrc sensor {name!r} requires a single-DoF joint"
                    )
                actuators = [
                    actuator
                    for actuator in world.actuators
                    if actuator.target.variant == "joint"
                    and str(actuator.target.value) == joint
                ]
                if not actuators:
                    raise ValueError(
                        f"Motrix jointactuatorfrc sensor {name!r} joint {joint!r} "
                        "has no joint-targeting actuator"
                    )
                sources: list[_MotrixJointActuatorForceSource] = []
                for actuator in actuators:
                    if not actuator.name:
                        raise ValueError(
                            f"Motrix jointactuatorfrc sensor {name!r} requires named actuators"
                        )
                    internal_name = f"__unisim_jointactuatorfrc_{name}_{len(sources)}"
                    while internal_name in used_names:
                        internal_name = f"_{internal_name}"
                    used_names.add(internal_name)
                    sensor = msd.ActuatorSensor()
                    sensor.name = internal_name
                    sensor.actuator = str(actuator.name)
                    world.sensors.actuator.append(sensor)
                    sources.append(
                        _MotrixJointActuatorForceSource(
                            sensor_name=internal_name,
                            gear=float(actuator.gear[0]),
                        )
                    )
                restored[name] = tuple(sources)
    return restored


def _audit_authored_sensors_imported(
    world: "World",
    source_paths: Sequence[Path],
    joint_actuator_force_sensors: Mapping[
        str, tuple[_MotrixJointActuatorForceSource, ...]
    ],
) -> None:
    """Fail closed by name whenever any authored sensor has no public reader."""

    authored = _authored_sensor_names(source_paths)
    imported = {
        str(sensor.name)
        for _, group in _motrix_sensor_groups(world)
        for sensor in group
        if sensor.name
    }
    missing = sorted(authored - imported - set(joint_actuator_force_sensors))
    if missing:
        raise ValueError(f"Motrix sensor(s) did not survive MJCF import: {missing}")


def _build_sensor_inventory(
    world: "World", source_paths: Sequence[Path], *, signatures: bool = False
) -> _MotrixSensorInventory:
    joint_sensors = _restore_joint_actuator_force_sensors(world, source_paths)
    _audit_authored_sensors_imported(world, source_paths, joint_sensors)
    return _MotrixSensorInventory(
        names=_motrix_sensor_names(world, joint_sensors),
        frame_identities=_motrix_frame_sensor_identities(world),
        contact_identities=_motrix_contact_sensor_identities(world),
        signatures=_motrix_sensor_signatures(world, joint_sensors) if signatures else (),
        categories=_motrix_sensor_categories(world, joint_sensors),
        joint_actuator_force_sensors=joint_sensors,
    )


def _read_joint_actuator_force_sensor(
    model: Any,
    data: Any,
    sources: Sequence[_MotrixJointActuatorForceSource],
) -> np.ndarray:
    values = [
        np.asarray(model.get_sensor_value(source.sensor_name, data), dtype=np.float32)
        * source.gear
        for source in sources
    ]
    return np.asarray(np.sum(values, axis=0, dtype=np.float32), dtype=np.float32)


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))
        _prepare_imported_world(fragment, fragment_file)
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
    try:
        world = msd.from_file(str(robot_path))
        _prepare_imported_world(world, model_path)
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_tracking_frame_sensors(world, base_name=base_name)
        inventory = _build_sensor_inventory(
            world, (model_path, *fragment_paths), signatures=True
        )
        model = msd.build(world)
        return model, inventory, world
    finally:
        remove_temporary_file(robot_path, original=model_path)


def materialize_scene_with_sensor_inventory(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> tuple["SceneModel", _MotrixSensorInventory, "World"]:
    """Build a Motrix model and return its validated cold-path sensor inventory."""

    model, inventory, world = materialize_scene_with_full_sensor_inventory(
        model_file=model_file,
        fragment_files=fragment_files,
        add_body_sensors=add_body_sensors,
        base_name=base_name,
    )
    return model, inventory, world


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
) -> tuple[SceneModel, np.ndarray, object | None, _MotrixSensorInventory, "World"]:
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
    try:
        robot_world = msd.from_file(str(merged_robot_path))
        _prepare_imported_world(robot_world, robot_path)
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        remove_temporary_file(merged_robot_path, original=robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
    if add_body_sensors:
        add_tracking_frame_sensors(world, base_name=base_name)
    inventory = _build_sensor_inventory(world, (robot_path, *fragment_paths))

    model = msd.build(world)
    sampler = generated.surface_sampler() if return_surface_sampler else None
    return (
        model,
        generated.terrain_origins,
        sampler,
        inventory,
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
    model, origins, sampler, _, _ = materialize_hfield_attached_scene_with_sensor_inventory(
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
