from __future__ import annotations

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
    geom1: str
    geom2: str
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
    return tuple(signatures)


def _motrix_sensor_names(world: "World") -> tuple[str, ...]:
    """Collect the names accepted by Motrix's native sensor accessor once."""
    groups = (
        world.sensors.contact,
        world.sensors.frame,
        world.sensors.joint,
        world.sensors.subtree,
        world.sensors.touch,
    )
    names = tuple(str(sensor.name) for group in groups for sensor in group if sensor.name)
    if len(set(names)) != len(names):
        raise ValueError(f"Motrix scene contains duplicate sensor names: {names}")
    return names


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
    """Collect the public identity of each native geom-pair contact sensor."""

    identities: list[_MotrixContactSensorIdentity] = []
    for sensor in world.sensors.contact:
        if not sensor.name:
            continue
        if sensor.match_.variant != "geom_pair":
            raise ValueError(f"Motrix contact sensor {sensor.name!r} is not a geom-pair sensor")
        geom1, geom2 = (str(name) for name in sensor.match_.value)
        identities.append(
            _MotrixContactSensorIdentity(
                name=str(sensor.name),
                geom1=geom1,
                geom2=geom2,
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
) -> tuple["SceneModel", _MotrixSensorInventory]:
    """Import an expanded source and retain its cold-path sensor inventory."""

    import motrixsim.msd as msd

    world = msd.from_file(str(Path(model_file).resolve()))  # pyright: ignore[reportAttributeAccessIssue]
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
    sensor_count = sum(
        1
        for group in (
            world.sensors.contact,
            world.sensors.frame,
            world.sensors.joint,
            world.sensors.subtree,
            world.sensors.touch,
        )
        for _ in group
    )
    if sensor_count != len(names):
        raise ValueError("Motrix portable scenes require every native sensor to be named")
    return (
        msd.build(world),
        _MotrixSensorInventory(
            names=names,
            frame_identities=frame_identities,
            contact_identities=contact_identities,
        ),
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

    native_geoms = {
        str(geom.name): geom
        for body in world.hierarchy.bodies
        for link in _iter_motrix_links(body.link)
        for geom in link.geoms
        if geom.name is not None
    }
    missing = sorted(set(geom_variant_sets) - set(native_geoms))
    if missing:
        # Uniform-public compilation already froze the public geom-name set.
        raise ValueError(
            f"Motrix native geoms are missing from the uniform variant binding: {missing}"
        )
    for geom_name, set_name in geom_variant_sets.items():
        if set_name not in registered:
            raise ValueError(f"Motrix geom {geom_name!r} refers to absent variant set")
        native_geoms[geom_name].mesh_variant_set = set_name


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


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        remove_temporary_file(sanitized, original=fragment_file)
    world.attach(fragment)


def _iter_motrix_links(link: Link):
    yield link
    for child in link.children:
        yield from _iter_motrix_links(child)


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
) -> tuple["SceneModel", _MotrixSensorInventory]:
    """Build a model and retain every public sensor identity for variant audits."""
    import motrixsim.msd as msd

    model_path = Path(model_file).resolve()
    fragment_paths = [
        resolve_scene_fragment_path(fragment_file, model_path) for fragment_file in fragment_files
    ]
    robot_path = _materialize_robot_with_fragment_keyframes(model_path, fragment_paths)
    try:
        world = msd.from_file(str(robot_path))  # pyright: ignore[reportAttributeAccessIssue]
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_tracking_frame_sensors(world, base_name=base_name)
        model = msd.build(world)
        return model, _MotrixSensorInventory(
            names=_motrix_sensor_names(world),
            frame_identities=_motrix_frame_sensor_identities(world),
            contact_identities=_motrix_contact_sensor_identities(world),
            signatures=_motrix_sensor_signatures(world),
        )
    finally:
        remove_temporary_file(robot_path, original=model_path)


def materialize_scene_with_sensor_inventory(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> tuple["SceneModel", tuple[str, ...], tuple[str, ...]]:
    """Build a Motrix model and return its validated cold-path sensor inventory."""
    import motrixsim.msd as msd

    model, inventory = materialize_scene_with_full_sensor_inventory(
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
) -> tuple[SceneModel, np.ndarray, object | None, tuple[str, ...], tuple[str, ...]]:
    """Build a Motrix terrain model and return its cold-path sensor inventory."""
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
        robot_world = msd.from_file(str(merged_robot_path))  # pyright: ignore[reportAttributeAccessIssue]
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        remove_temporary_file(merged_robot_path, original=robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
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
