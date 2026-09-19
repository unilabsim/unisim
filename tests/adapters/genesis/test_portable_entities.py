"""Real Genesis CPU acceptance for portable multi-entity scenes."""

# ruff: noqa: E402
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE
from unisim.dr.types import (
    FixedVariantPlan,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ModelSourceDescriptor,
    ResetRandomizationPayload,
)
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg


def _write(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _quat_mul(left: tuple[float, ...], right: tuple[float, ...]) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quat[0]
    xyz = quat[1:]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    conjugate = quat * np.asarray((1.0, -1.0, -1.0, -1.0))
    return _quat_rotate(conjugate, vector)


def _robot(tmp_path: Path, *, passive: bool = False) -> ModelSourceDescriptor:
    passive_joint = '<joint name="passive" axis="0 0 1"/>' if passive else ""
    drive = "" if passive else '<position name="drive" joint="drive" kp="24" kv="3"/>'
    drive_joint = (
        '<joint name="drive" axis="0 1 0" ref="0.1" damping=".17" '
        'frictionloss=".043" armature=".011"/>'
        if not passive
        else passive_joint
    )
    return _write(
        tmp_path,
        "robot-passive" if passive else "robot",
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
          <geom name="base_geom" type="sphere" size=".08" contype="0" conaffinity="0"/>
          <body name="link" pos="0 0 .2">
            {drive_joint}
            <inertial pos="0 0 0" mass=".2" diaginertia=".03 .03 .03"/>
            <geom name="link_geom" type="sphere" size=".03" contype="0" conaffinity="0"/>
          </body>
        </body></worldbody><actuator>{drive}</actuator></mujoco>
        """,
    )


def _passive(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "passive",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <freejoint name="root"/><inertial pos="0 0 0" mass=".7"
            diaginertia=".1 .1 .1"/>
          <geom name="passive_base_geom" type="sphere" size=".06"
            contype="0" conaffinity="0"/>
          <body name="child" pos="0 0 .15">
            <joint name="passive_hinge" axis="0 1 0" damping=".31"
              frictionloss=".027" armature=".023"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom name="passive_child_geom" type="sphere" size=".02"
              contype="0" conaffinity="0"/>
          </body>
        </body></worldbody></mujoco>
        """,
    )


def _robot_with_home_key(tmp_path: Path) -> ModelSourceDescriptor:
    source = _robot(tmp_path / "robot-source")
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            "</mujoco>",
            '<keyframe><key name="home" qpos="0.4" qvel="0.2" ctrl="0.4"/></keyframe></mujoco>',
        )
    )
    return _write(tmp_path, "robot-home-key", xml)


def _passive_with_ignored_root_key(tmp_path: Path) -> ModelSourceDescriptor:
    source = _passive(tmp_path / "passive-source")
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            "</mujoco>",
            '<keyframe><key name="ignored-root" qpos="9 8 7 1 0 0 0 .25" '
            'qvel="1 2 3 4 5 6 -.3"/></keyframe>'
            "</mujoco>",
        )
    )
    return _write(tmp_path, "passive-root-key", xml)


def _object_with_home_root_key(
    source: ModelSourceDescriptor, tmp_path: Path, name: str
) -> ModelSourceDescriptor:
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            "</mujoco>",
            '<keyframe><key name="home" qpos="9 8 7 .5 .5 .5 .5" '
            'qvel="1 2 3 4 5 6"/></keyframe></mujoco>',
        )
    )
    return _write(tmp_path, name, xml)


def _passive_with_site_sensors(
    tmp_path: Path, *, referenced: bool = False, accelerometer_site: str | None = "child_site"
) -> ModelSourceDescriptor:
    source = _passive(tmp_path)
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            '<geom name="passive_child_geom"',
            '<site name="child_site" pos=".05 0 0"/>'
            '<site name="motion_site" pos=".05 0 0" '
            'quat=".7071067811865476 0 0 .7071067811865476"/>'
            '<geom name="passive_child_geom"',
        )
    )
    reference = ' reftype="site" refname="child_site"' if referenced else ""
    accelerometer = (
        f"<accelerometer name='site_acc' site='{accelerometer_site}'/>"
        if accelerometer_site is not None
        else ""
    )
    xml = xml.replace(
        "</worldbody>",
        "</worldbody><sensor>"
        f"<framepos name='site_pos' objtype='site' objname='child_site'{reference}/>"
        "<framequat name='site_quat' objtype='site' objname='child_site'/>"
        "<gyro name='site_gyro' site='motion_site'/>"
        "<velocimeter name='site_vel' site='motion_site'/>"
        f"{accelerometer}"
        "</sensor>",
    )
    return _write(tmp_path, "passive-site-sensors", xml)


def _passive_with_body_sensor(tmp_path: Path) -> ModelSourceDescriptor:
    source = _passive(tmp_path)
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            "</worldbody>",
            "</worldbody><sensor>"
            "<framepos name='body_pos' objtype='body' objname='child'/>"
            "<framelinvel name='body_linvel' objtype='body' objname='child'/>"
            "<frameangvel name='body_angvel' objtype='body' objname='child'/>"
            "</sensor>",
        )
    )
    return _write(tmp_path, "passive-body-sensor", xml)


def _object(
    tmp_path: Path,
    name: str,
    *,
    radius: float,
    mass: float,
    com_x: float,
    inertia: tuple[float, float, float] = (0.02, 0.03, 0.04),
    com_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        name,
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <freejoint name="root"/>
          <inertial pos="{com_x} 0 0" mass="{mass}"
            quat="{" ".join(str(value) for value in com_quat)}"
            diaginertia="{" ".join(str(value) for value in inertia)}"/>
          <geom name="object_geom" type="sphere" size="{radius}"
            contype="0" conaffinity="0"/>
        </body></worldbody></mujoco>
        """,
    )


def _table(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "table",
        """
        <mujoco><option gravity="0 0 0"/><worldbody><body name="base">
          <inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
          <geom name="table_geom" type="box" size="1 1 .1"
            contype="0" conaffinity="0"/>
        </body></worldbody></mujoco>
        """,
    )


def _object_with_site_sensors(
    tmp_path: Path,
    name: str,
    *,
    radius: float,
    mass: float,
    com_x: float,
    inertia: tuple[float, float, float] = (0.02, 0.03, 0.04),
    com_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ModelSourceDescriptor:
    source = _object(
        tmp_path,
        name,
        radius=radius,
        mass=mass,
        com_x=com_x,
        inertia=inertia,
        com_quat=com_quat,
    )
    xml = (
        Path(source.model_file)
        .read_text(encoding="utf-8")
        .replace(
            '<geom name="object_geom"',
            '<site name="object_site" pos=".05 0 0"/><geom name="object_geom"',
        )
    )
    xml = xml.replace(
        "</worldbody>",
        "</worldbody><sensor>"
        "<framepos name='site_pos' objtype='site' objname='object_site'/>"
        "<framequat name='site_quat' objtype='site' objname='object_site'/>"
        "</sensor>",
    )
    return _write(tmp_path, f"{name}-site-sensors", xml)


def _scene(
    tmp_path: Path,
    *,
    assignment: tuple[int, ...] = (0, 0, 0, 1, 1),
    object_inertias: tuple[tuple[float, float, float], ...] = (
        (0.02, 0.03, 0.04),
        (0.03, 0.04, 0.05),
    ),
    object_com_quats: tuple[
        tuple[float, float, float, float], tuple[float, float, float, float]
    ] = ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    object_site_sensors: bool = False,
) -> SceneCfg:
    object_source = _object_with_site_sensors if object_site_sensors else _object
    object_a = object_source(
        tmp_path,
        "object_a",
        radius=0.1,
        mass=0.5,
        com_x=0.01,
        inertia=object_inertias[0],
        com_quat=object_com_quats[0],
    )
    object_b = object_source(
        tmp_path,
        "object_b",
        radius=0.15,
        mass=1.5,
        com_x=0.03,
        inertia=object_inertias[1],
        com_quat=object_com_quats[1],
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _robot(tmp_path),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "passive",
                _passive(tmp_path),
                initial_state=EntityInitialState((1.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "object",
                object_a,
                kind="rigid",
                initial_state=EntityInitialState((2.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "table",
                _table(tmp_path),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, -3.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(np.asarray(assignment, dtype=np.int32), (object_a, object_b)),
        ),
    )


def _site_sensor_fragment(path: Path) -> Path:
    fragment = path / "site-sensor-fragment.xml"
    fragment.write_text(
        "<mujoco><sensor>"
        "<framepos name='cross_passive_pos' objtype='site' objname='passive/child_site'/>"
        "<framequat name='cross_passive_quat' objtype='site' objname='passive/child_site'/>"
        "<framepos name='cross_object_pos' objtype='site' objname='object/object_site'/>"
        "<framequat name='cross_object_quat' objtype='site' objname='object/object_site'/>"
        "<framelinvel name='cross_passive_linvel' objtype='site' "
        "objname='passive/motion_site'/>"
        "<frameangvel name='cross_passive_angvel' objtype='site' "
        "objname='passive/motion_site'/>"
        "<framelinvel name='cross_object_linvel' objtype='site' "
        "objname='object/object_site'/>"
        "<frameangvel name='cross_object_angvel' objtype='site' "
        "objname='object/object_site'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return fragment


def _contact_sensor_fragment(path: Path) -> Path:
    fragment = path / "contact-sensor-fragment.xml"
    fragment.write_text(
        "<mujoco><sensor>"
        "<contact name='object_table_force' geom1='object/object_geom' "
        "geom2='table/table_geom' data='force' reduce='netforce'/>"
        "<contact name='table_object_force' geom1='table/table_geom' "
        "geom2='object/object_geom' data='force' reduce='netforce'/>"
        "<contact name='object_table_found' geom1='object/object_geom' "
        "geom2='table/table_geom' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return fragment


def _body_sensor_fragment(path: Path) -> Path:
    fragment = path / "body-sensor-fragment.xml"
    fragment.write_text(
        "<mujoco><sensor>"
        "<framepos name='cross_object_body_pos' objtype='body' objname='object/base'/>"
        "<framequat name='cross_object_body_quat' objtype='body' objname='object/base'/>"
        "<framelinvel name='cross_object_body_linvel' objtype='body' objname='object/base'/>"
        "<frameangvel name='cross_object_body_angvel' objtype='body' objname='object/base'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return fragment


def _enable_source_gravity(scene: SceneCfg) -> None:
    sources = [Path(entity.source.model_file) for entity in scene.entity_assets]
    if scene.entity_variant is not None:
        sources.extend(Path(variant.model_file) for variant in scene.entity_variant.plan.variants)
    for source in dict.fromkeys(sources):
        text = source.read_text(encoding="utf-8")
        source.write_text(text.replace('gravity="0 0 0"', 'gravity="0 0 -9.81"'), encoding="utf-8")


def _enable_native_contact_masks(scene: SceneCfg, object_variant_b: tuple[int, int]) -> None:
    masks = {
        "robot": (1, 16),
        "passive": (2, 32),
        "object": (4, 64),
        "table": (8, 128),
    }
    for entity in scene.entity_assets:
        sources = [Path(entity.source.model_file)]
        if scene.entity_variant is not None and (scene.entity_variant.target_entity == entity.name):
            sources.extend(
                Path(variant.model_file) for variant in scene.entity_variant.plan.variants
            )
        for source in sources:
            mask = masks[entity.name]
            if entity.name == "object" and source.stem == "object_b":
                mask = object_variant_b
            text = source.read_text(encoding="utf-8")
            source.write_text(
                text.replace(
                    'contype="0" conaffinity="0"',
                    f'contype="{mask[0]}" conaffinity="{mask[1]}" '
                    'friction=".37 .004 .002" solref=".021 .89" '
                    'solimp=".91 .94 .0013 .53 2.2" condim="6"',
                )
            )


def test_portable_site_sensor_structural_rejections(tmp_path: Path) -> None:
    rotated_scene = _scene(tmp_path / "rotated-accelerometer", assignment=(0, 1))
    rotated_entities = list(rotated_scene.entity_assets)
    rotated_entities[1] = replace(
        rotated_entities[1],
        source=_passive_with_site_sensors(
            tmp_path / "rotated-accelerometer" / "passive-source",
            accelerometer_site="motion_site",
        ),
    )
    rotated_scene.entity_assets = tuple(rotated_entities)
    with pytest.raises(
        NotImplementedError,
        match=r"rotated accelerometer sites are rejected",
    ):
        GenesisBackend(rotated_scene, 2, 0.002)

    referenced_scene = _scene(tmp_path / "referenced", assignment=(0, 1))
    entities = list(referenced_scene.entity_assets)
    entities[1] = replace(
        entities[1],
        source=_passive_with_site_sensors(
            tmp_path / "referenced" / "passive-source", referenced=True
        ),
    )
    referenced_scene.entity_assets = tuple(entities)
    with pytest.raises(
        NotImplementedError,
        match=r"genesis backend maps framepos sensor 'site_pos' only with a world reference",
    ):
        GenesisBackend(referenced_scene, 2, 0.002)

    drifting_scene = _scene(
        tmp_path / "variant-drift",
        assignment=(0, 1),
        object_site_sensors=True,
    )
    variant_b_path = tmp_path / "variant-drift" / "object-b-drifted.xml"
    variant_b_path.write_text(
        Path(drifting_scene.entity_variant.plan.variants[1].model_file)
        .read_text(encoding="utf-8")
        .replace('name="object_site" pos=".05 0 0"', 'name="object_site" pos=".07 0 0"'),
        encoding="utf-8",
    )
    plan = replace(
        drifting_scene.entity_variant.plan,
        variants=(
            drifting_scene.entity_variant.plan.variants[0],
            ModelSourceDescriptor(str(variant_b_path)),
        ),
    )
    drifting_scene.entity_variant = replace(drifting_scene.entity_variant, plan=plan)
    with pytest.raises(NotImplementedError, match="portable sensor identity differs"):
        GenesisBackend(drifting_scene, 2, 0.002)

    unsupported_scene = _scene(tmp_path / "unsupported-sensor", assignment=(0, 1))
    unsupported_entities = list(unsupported_scene.entity_assets)
    unsupported_source = _passive_with_site_sensors(
        tmp_path / "unsupported-sensor" / "passive-source"
    )
    unsupported_path = Path(unsupported_source.model_file)
    unsupported_path.write_text(
        unsupported_path.read_text(encoding="utf-8").replace(
            "<framepos name='site_pos' objtype='site' objname='child_site'/>",
            "<framezaxis name='site_z' objtype='site' objname='child_site'/>",
        ),
        encoding="utf-8",
    )
    unsupported_entities[1] = replace(
        unsupported_entities[1], source=ModelSourceDescriptor(str(unsupported_path))
    )
    unsupported_scene.entity_assets = tuple(unsupported_entities)
    with pytest.raises(
        NotImplementedError,
        match="site FramePos/FrameQuat/Gyro/Velocimeter/Accelerometer sensors",
    ):
        GenesisBackend(unsupported_scene, 2, 0.002)

    source_body_scene = _scene(tmp_path / "source-body-sensor", assignment=(0, 1))
    source_body_entities = list(source_body_scene.entity_assets)
    source_body_entities[1] = replace(
        source_body_entities[1],
        source=_passive_with_body_sensor(tmp_path / "source-body-sensor" / "passive-source"),
    )
    source_body_scene.entity_assets = tuple(source_body_entities)
    with pytest.raises(
        NotImplementedError,
        match="genesis backend maps framepos sensors from MJCF sites only",
    ):
        GenesisBackend(source_body_scene, 2, 0.002)

    orientation_scene = _scene(
        tmp_path / "inertial-orientation",
        assignment=(0, 1),
        object_com_quats=(
            (np.cos(0.11), 0.0, 0.0, np.sin(0.11)),
            (np.cos(0.11), 0.0, 0.0, np.sin(0.11)),
        ),
    )
    orientation_scene.fragment_files = [
        str(_body_sensor_fragment(tmp_path / "inertial-orientation"))
    ]
    orientation_backend = GenesisBackend(orientation_scene, 2, 0.002)
    with pytest.raises(RuntimeError, match="native inertia differs"):
        orientation_backend.materialize()

    contact_scene = _scene(tmp_path / "cross-entity-contact-fragment", assignment=(0, 1))
    contact_fragment = tmp_path / "cross-entity-contact-fragment" / "fragment.xml"
    contact_fragment.write_text(
        "<mujoco><sensor>"
        "<contact name='passive_internal_found' geom1='passive/passive_base_geom' "
        "geom2='passive/passive_child_geom' data='found' num='1'/>"
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    contact_scene.fragment_files = [str(contact_fragment)]
    with pytest.raises(NotImplementedError, match="two distinct public entities"):
        GenesisBackend(contact_scene, 2, 0.002)


def test_portable_site_sensor_fragment_variant_identity_is_frozen(tmp_path: Path) -> None:
    scene = _scene(tmp_path, assignment=(0, 1), object_site_sensors=True)
    entities = list(scene.entity_assets)
    entities[1] = replace(
        entities[1],
        source=_passive_with_site_sensors(tmp_path / "passive-source"),
    )
    scene.entity_assets = tuple(entities)
    scene.fragment_files = [str(_site_sensor_fragment(tmp_path))]
    variant_b_path = tmp_path / "object-b-drifted.xml"
    variant_b_path.write_text(
        Path(scene.entity_variant.plan.variants[1].model_file)
        .read_text(encoding="utf-8")
        .replace('name="object_site" pos=".05 0 0"', 'name="object_site" pos=".07 0 0"'),
        encoding="utf-8",
    )
    plan = replace(
        scene.entity_variant.plan,
        variants=(
            scene.entity_variant.plan.variants[0],
            ModelSourceDescriptor(str(variant_b_path)),
        ),
    )
    scene.entity_variant = replace(scene.entity_variant, plan=plan)
    with pytest.raises(
        NotImplementedError, match="sensor identity differs between composed variants"
    ):
        GenesisBackend(scene, 2, 0.002)


def test_portable_site_sensor_fragments_read_assignment_rows(tmp_path: Path) -> None:
    scene = _scene(tmp_path, object_site_sensors=True)
    entities = list(scene.entity_assets)
    entities[1] = replace(
        entities[1],
        source=_passive_with_site_sensors(tmp_path / "passive-source"),
    )
    scene.entity_assets = tuple(entities)
    scene.fragment_files = [str(_site_sensor_fragment(tmp_path))]
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()
    assert tuple(backend._sensor_slots) == (
        "passive/site_pos",
        "passive/site_quat",
        "passive/site_gyro",
        "passive/site_vel",
        "passive/site_acc",
        "object/site_pos",
        "object/site_quat",
        "cross_passive_pos",
        "cross_passive_quat",
        "cross_object_pos",
        "cross_object_quat",
        "cross_passive_linvel",
        "cross_passive_angvel",
        "cross_object_linvel",
        "cross_object_angvel",
    )
    initial_positions = backend.get_sensor_data("cross_passive_pos").copy()
    initial_quaternions = backend.get_sensor_data("cross_passive_quat").copy()
    initial_object_positions = backend.get_sensor_data("cross_object_pos").copy()
    initial_object_quaternions = backend.get_sensor_data("cross_object_quat").copy()
    np.testing.assert_allclose(initial_positions, np.tile((1.05, 0.0, 1.15), (5, 1)), atol=2e-6)
    np.testing.assert_allclose(
        initial_quaternions, np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)), atol=2e-6
    )
    np.testing.assert_allclose(
        initial_object_positions, np.tile((2.05, 0.0, 1.0), (5, 1)), atol=2e-6
    )
    np.testing.assert_allclose(
        initial_object_quaternions, np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)), atol=2e-6
    )

    rows = np.asarray((1, 4), dtype=np.intp)
    passive_joint = layout.get_entity("passive").joints[0].qpos_indices[0]
    qpos = backend._qpos_cache[1].copy()
    qvel = backend._qvel_cache[1].copy()
    qpos[rows, passive_joint] = (0.35, -0.27)
    backend.set_state(rows, qpos[rows], qvel[rows])

    positions_after = backend.get_sensor_data("cross_passive_pos")
    quaternions_after = backend.get_sensor_data("cross_passive_quat")
    expected_positions = initial_positions.copy()
    expected_quaternions = initial_quaternions.copy()
    expected_positions[1] = (1.0 + 0.05 * np.cos(0.35), 0.0, 1.15 - 0.05 * np.sin(0.35))
    expected_positions[4] = (
        1.0 + 0.05 * np.cos(-0.27),
        0.0,
        1.15 - 0.05 * np.sin(-0.27),
    )
    expected_quaternions[1] = (np.cos(0.175), 0.0, np.sin(0.175), 0.0)
    expected_quaternions[4] = (np.cos(-0.135), 0.0, np.sin(-0.135), 0.0)
    np.testing.assert_allclose(positions_after, expected_positions, atol=2e-6)
    np.testing.assert_allclose(quaternions_after, expected_quaternions, atol=2e-6)
    np.testing.assert_array_equal(
        backend.get_sensor_data("cross_object_pos"), initial_object_positions
    )
    np.testing.assert_array_equal(
        backend.get_sensor_data("cross_object_quat"), initial_object_quaternions
    )

    initial_passive_linear_velocities = backend.get_sensor_data("cross_passive_linvel").copy()
    initial_passive_angular_velocities = backend.get_sensor_data("cross_passive_angvel").copy()
    initial_object_linear_velocities = backend.get_sensor_data("cross_object_linvel").copy()
    initial_object_angular_velocities = backend.get_sensor_data("cross_object_angvel").copy()
    np.testing.assert_allclose(initial_passive_linear_velocities, 0.0, atol=2e-6)
    np.testing.assert_allclose(initial_passive_angular_velocities, 0.0, atol=2e-6)
    np.testing.assert_allclose(initial_object_linear_velocities, 0.0, atol=2e-6)
    np.testing.assert_allclose(initial_object_angular_velocities, 0.0, atol=2e-6)
    untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)

    passive_entity = layout.get_entity("passive")
    passive_root_qpos = np.asarray(passive_entity.root_qpos_indices, dtype=np.intp)
    passive_root_qvel = np.asarray(passive_entity.root_qvel_indices, dtype=np.intp)
    passive_joint_qvel = passive_entity.joints[0].qvel_indices[0]
    object_entity = layout.get_entity("object")
    object_root_qpos = np.asarray(object_entity.root_qpos_indices, dtype=np.intp)
    object_root_qvel = np.asarray(object_entity.root_qvel_indices, dtype=np.intp)
    qpos[rows[:, None], passive_root_qpos[None, 3:]] = np.asarray(
        (
            (np.cos(0.21), np.sin(0.21), 0.0, 0.0),
            (np.cos(-0.18), 0.0, np.sin(-0.18), 0.0),
        ),
        dtype=np.float32,
    )
    qpos[rows[:, None], object_root_qpos[None, 3:]] = np.asarray(
        (
            (np.cos(0.26), 0.0, np.sin(0.26), 0.0),
            (np.cos(-0.23), np.sin(-0.23), 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    qvel[rows[:, None], passive_root_qvel[None, :3]] = np.asarray(
        ((0.31, -0.22, 0.17), (0.12, 0.28, -0.19)), dtype=np.float32
    )
    qvel[rows[:, None], passive_root_qvel[None, 3:]] = np.asarray(
        ((0.09, -0.14, 0.23), (-0.24, 0.17, 0.11)), dtype=np.float32
    )
    qvel[rows, passive_joint_qvel] = (0.41, -0.36)
    qvel[rows[:, None], object_root_qvel[None, :3]] = np.asarray(
        ((0.27, 0.18, -0.12), (-0.16, 0.24, 0.13)), dtype=np.float32
    )
    qvel[rows[:, None], object_root_qvel[None, 3:]] = np.asarray(
        ((-0.21, 0.16, 0.11), (0.19, -0.13, 0.27)), dtype=np.float32
    )
    backend.set_state(rows, qpos[rows], qvel[rows])

    expected_motion = {
        "cross_passive_linvel": initial_passive_linear_velocities.copy(),
        "cross_passive_angvel": initial_passive_angular_velocities.copy(),
        "cross_object_linvel": initial_object_linear_velocities.copy(),
        "cross_object_angvel": initial_object_angular_velocities.copy(),
    }
    sensor_names = tuple(backend._sensor_slots)
    for sensor_name in (
        "cross_passive_linvel",
        "cross_passive_angvel",
        "cross_object_linvel",
        "cross_object_angvel",
    ):
        sensor_index = sensor_names.index(sensor_name)
        binding = backend._sensor_link_bindings[sensor_index]
        native = binding.runtime.entity
        native_link_quat = (
            native.get_links_quat(binding.native_body, relative=True)
            .cpu()
            .numpy()
            .reshape(5, -1, 4)[:, 0]
        )
        native_link_vel = native.get_links_vel(binding.native_body).cpu().numpy().reshape(5, 3)
        native_link_ang = native.get_links_ang(binding.native_body).cpu().numpy().reshape(5, 3)
        site_offset = backend._sensor_constants[sensor_name][1]
        for row in rows:
            offset_world = _quat_rotate(native_link_quat[row], site_offset)
            expected_motion[sensor_name][row] = native_link_ang[row]
            if sensor_name.endswith("linvel"):
                expected_motion[sensor_name][row] = native_link_vel[row] + np.cross(
                    native_link_ang[row], offset_world
                )

    np.testing.assert_allclose(
        backend.get_sensor_data("cross_passive_linvel"),
        expected_motion["cross_passive_linvel"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("cross_passive_angvel"),
        expected_motion["cross_passive_angvel"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("cross_object_linvel"),
        expected_motion["cross_object_linvel"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        backend.get_sensor_data("cross_object_angvel"),
        expected_motion["cross_object_angvel"],
        atol=2e-6,
    )
    for name, expected_values in expected_motion.items():
        np.testing.assert_array_equal(
            backend.get_sensor_data(name)[untouched_rows],
            expected_values[untouched_rows],
        )
    # MuJoCo Frame motion on a site is world-framed, while entity-owned
    # velocimeter/gyro sensors expose the audited nonidentity site frame.
    assert not np.allclose(
        backend.get_sensor_data("passive/site_vel")[rows],
        backend.get_sensor_data("cross_passive_linvel")[rows],
    )
    assert not np.allclose(
        backend.get_sensor_data("passive/site_gyro")[rows],
        backend.get_sensor_data("cross_passive_angvel")[rows],
    )


def test_portable_body_sensor_fragments_read_assignment_rows(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene.fragment_files = [str(_body_sensor_fragment(tmp_path))]
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()
    assert tuple(backend._sensor_slots) == (
        "cross_object_body_pos",
        "cross_object_body_quat",
        "cross_object_body_linvel",
        "cross_object_body_angvel",
    )
    initial_positions = backend.get_sensor_data("cross_object_body_pos").copy()
    initial_quaternions = backend.get_sensor_data("cross_object_body_quat").copy()
    np.testing.assert_allclose(initial_positions[:, 0], (2.01, 2.01, 2.01, 2.03, 2.03), atol=2e-6)
    np.testing.assert_allclose(initial_positions[:, 1], 0.0, atol=2e-6)
    np.testing.assert_allclose(initial_positions[:, 2], 1.0, atol=2e-6)
    np.testing.assert_allclose(
        initial_quaternions, np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)), atol=2e-6
    )
    initial_linear_velocities = backend.get_sensor_data("cross_object_body_linvel").copy()
    initial_angular_velocities = backend.get_sensor_data("cross_object_body_angvel").copy()
    np.testing.assert_allclose(initial_linear_velocities, 0.0, atol=2e-6)
    np.testing.assert_allclose(initial_angular_velocities, 0.0, atol=2e-6)

    rows = np.asarray((1, 4), dtype=np.intp)
    untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)
    entity = layout.get_entity("object")
    root_qpos = np.asarray(entity.root_qpos_indices, dtype=np.intp)
    qpos = backend._qpos_cache[1].copy()
    qvel = backend._qvel_cache[1].copy()
    qpos[rows[:, None], root_qpos[None, :]] = np.asarray(
        (
            (2.2, 0.0, 1.0, np.cos(0.2), 0.0, np.sin(0.2), 0.0),
            (1.8, 0.0, 1.0, np.cos(-0.15), 0.0, np.sin(-0.15), 0.0),
        ),
        dtype=np.float32,
    )
    root_qvel = np.asarray(entity.root_qvel_indices, dtype=np.intp)
    qvel[rows[:, None], root_qvel[None, :3]] = np.asarray(
        ((0.4, -0.3, 0.2), (0.1, 0.3, -0.2)), dtype=np.float32
    )
    qvel[rows[:, None], root_qvel[None, 3:]] = np.asarray(
        ((0.1, 0.2, -0.1), (-0.2, 0.1, 0.3)), dtype=np.float32
    )
    backend.set_state(rows, qpos[rows], qvel[rows])

    positions_after = backend.get_sensor_data("cross_object_body_pos")
    quaternions_after = backend.get_sensor_data("cross_object_body_quat")
    expected_positions = initial_positions.copy()
    expected_quaternions = initial_quaternions.copy()
    expected_positions[1] = (2.2, 0.0, 1.0)
    expected_positions[4] = (1.8, 0.0, 1.0)
    expected_quaternions[1] = (np.cos(0.2), 0.0, np.sin(0.2), 0.0)
    expected_quaternions[4] = (np.cos(-0.15), 0.0, np.sin(-0.15), 0.0)
    np.testing.assert_allclose(positions_after, expected_positions, atol=2e-6)
    np.testing.assert_allclose(quaternions_after, expected_quaternions, atol=2e-6)
    np.testing.assert_array_equal(
        positions_after[untouched_rows], initial_positions[untouched_rows]
    )
    np.testing.assert_array_equal(
        quaternions_after[untouched_rows], initial_quaternions[untouched_rows]
    )
    linear_velocities_after = backend.get_sensor_data("cross_object_body_linvel")
    angular_velocities_after = backend.get_sensor_data("cross_object_body_angvel")
    expected_linear_velocities = initial_linear_velocities.copy()
    expected_angular_velocities = initial_angular_velocities.copy()
    runtime = backend._entity_runtimes["object"]
    binding = backend._sensor_link_bindings[3]
    native_link_quat = (
        runtime.entity.get_links_quat(binding.native_body, relative=True)
        .cpu()
        .numpy()
        .reshape(5, -1, 4)[:, 0]
    )
    native_link_vel = runtime.entity.get_links_vel(binding.native_body).cpu().numpy().reshape(5, 3)
    native_link_ang = runtime.entity.get_links_ang(binding.native_body).cpu().numpy().reshape(5, 3)
    for row in rows:
        variant = int(backend._variant_assignment[row])
        offset_world = _quat_rotate(native_link_quat[row], binding.body_ipos[variant])
        expected_angular_velocities[row] = native_link_ang[row]
        expected_linear_velocities[row] = native_link_vel[row] + np.cross(
            native_link_ang[row], offset_world
        )
    np.testing.assert_allclose(linear_velocities_after, expected_linear_velocities, atol=2e-6)
    np.testing.assert_allclose(angular_velocities_after, expected_angular_velocities, atol=2e-6)
    np.testing.assert_array_equal(
        linear_velocities_after[untouched_rows],
        initial_linear_velocities[untouched_rows],
    )
    np.testing.assert_array_equal(
        angular_velocities_after[untouched_rows],
        initial_angular_velocities[untouched_rows],
    )


def test_portable_contact_sensor_fragments_read_exact_found_and_netforce_rows(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path)
    _enable_native_contact_masks(scene, object_variant_b=(4, 64))
    _enable_source_gravity(scene)
    scene.fragment_files = [str(_contact_sensor_fragment(tmp_path))]
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()

    assert tuple(backend._sensor_slots) == (
        "object_table_force",
        "table_object_force",
        "object_table_found",
    )
    object_binding = backend._sensor_contact_bindings["object_table_force"]
    reversed_binding = backend._sensor_contact_bindings["table_object_force"]
    found_binding = backend._sensor_contact_bindings["object_table_found"]
    assert object_binding.netforce and reversed_binding.netforce
    assert not found_binding.netforce
    assert np.array_equal(object_binding.geom1_ids[:3], np.full((3,), object_binding.geom1_ids[0]))
    assert np.array_equal(object_binding.geom1_ids[3:], np.full((2,), object_binding.geom1_ids[3]))
    assert object_binding.geom1_ids[0] != object_binding.geom1_ids[3]
    assert np.array_equal(object_binding.geom2_ids, np.full((5,), object_binding.geom2_ids[0]))
    assert np.array_equal(reversed_binding.geom1_ids, object_binding.geom2_ids)
    assert np.array_equal(reversed_binding.geom2_ids, object_binding.geom1_ids)

    object_root = layout.get_entity("object").root_qpos_indices
    qpos = backend._qpos_cache[1].copy()
    qvel = backend._qvel_cache[1].copy()
    object_heights = np.asarray([-2.799, -2.799, -2.799, -2.749, -2.749])
    qpos[:, object_root] = np.broadcast_to(
        np.asarray((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)), (5, 7)
    ).copy()
    qpos[:, object_root[2]] = object_heights
    backend.set_state(np.arange(5, dtype=np.intp), qpos, qvel)
    found_after_reset = backend.get_sensor_data("object_table_found")
    object_force_after_reset = backend.get_sensor_data("object_table_force")
    table_force_after_reset = backend.get_sensor_data("table_object_force")
    assert found_after_reset.shape == (5, 1)
    assert object_force_after_reset.shape == (5, 3)
    assert table_force_after_reset.shape == (5, 3)
    np.testing.assert_array_equal(found_after_reset, 0.0)
    np.testing.assert_array_equal(object_force_after_reset, 0.0)
    np.testing.assert_array_equal(table_force_after_reset, 0.0)

    for _ in range(25):
        backend.step(np.zeros((5, 1), dtype=np.float32))
    np.testing.assert_array_equal(backend.get_sensor_data("object_table_found"), 1.0)
    object_force = backend.get_sensor_data("object_table_force")
    table_force = backend.get_sensor_data("table_object_force")
    expected_object_force_z = np.asarray([4.905, 4.905, 4.905, 14.715, 14.715])
    np.testing.assert_allclose(object_force[:, 2], expected_object_force_z, rtol=0.1)
    np.testing.assert_allclose(table_force[:, 2], -expected_object_force_z, rtol=0.1)
    np.testing.assert_allclose(object_force[:, 1], 0.0, atol=0.1)
    np.testing.assert_allclose(table_force[:, 1], 0.0, atol=0.1)
    np.testing.assert_allclose(object_force[:, 0], 0.0, atol=1.5)
    np.testing.assert_allclose(table_force[:, 0], 0.0, atol=1.5)

    qpos[0, object_root] = np.asarray((2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    rows = np.asarray((0,), dtype=np.intp)
    backend.set_state(rows, qpos[rows], qvel[rows])
    found_after_selected_reset = backend.get_sensor_data("object_table_found")
    object_force_after_selected_reset = backend.get_sensor_data("object_table_force")
    table_force_after_selected_reset = backend.get_sensor_data("table_object_force")
    np.testing.assert_array_equal(found_after_selected_reset[0], 0.0)
    np.testing.assert_array_equal(found_after_selected_reset[1:], 1.0)
    np.testing.assert_array_equal(object_force_after_selected_reset[0], 0.0)
    np.testing.assert_array_equal(table_force_after_selected_reset[0], 0.0)
    np.testing.assert_allclose(
        object_force_after_selected_reset[1:], object_force[1:], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        table_force_after_selected_reset[1:], table_force[1:], rtol=1e-6, atol=1e-6
    )

    backend.step(np.zeros((5, 1), dtype=np.float32))
    found_after_step = backend.get_sensor_data("object_table_found")
    object_force_after_step = backend.get_sensor_data("object_table_force")
    table_force_after_step = backend.get_sensor_data("table_object_force")
    np.testing.assert_array_equal(found_after_step[0], 0.0)
    np.testing.assert_array_equal(found_after_step[1:], 1.0)
    np.testing.assert_array_equal(object_force_after_step[0], 0.0)
    np.testing.assert_array_equal(table_force_after_step[0], 0.0)
    np.testing.assert_allclose(
        object_force_after_step[1:, 2], expected_object_force_z[1:], rtol=0.1
    )
    np.testing.assert_allclose(
        table_force_after_step[1:, 2], -expected_object_force_z[1:], rtol=0.1
    )


def test_portable_entities_selected_reset_randomization_mass_and_gains(tmp_path: Path):
    scene = _scene(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002, base_name="passive/base")
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        default_mass = backend.get_body_mass().copy()
        default_reset_mass = backend.get_reset_term_default("body_mass")
        default_reset_base_delta = backend.get_reset_term_default("base_mass_delta")
        default_reset_kp = backend.get_reset_term_default("kp")
        default_reset_kd = backend.get_reset_term_default("kd")
        np.testing.assert_allclose(default_reset_mass, default_mass, rtol=2e-7)
        np.testing.assert_array_equal(default_reset_base_delta, np.zeros((5,)))
        np.testing.assert_allclose(default_reset_kp, np.full((5, 1), 24.0))
        np.testing.assert_allclose(default_reset_kd, np.full((5, 1), 3.0))
        rows = np.asarray((1, 4), dtype=np.intp)
        untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)
        object_body_id = layout.get_entity("object").body_ids[0]
        passive_body_id = layout.get_entity("passive").body_ids[0]
        robot_runtime = backend._entity_runtimes["robot"]
        object_runtime = backend._entity_runtimes["object"]

        requested_mass = default_mass[rows].copy()
        requested_mass[:, object_body_id] = (0.75, 1.9)
        requested_kp = np.asarray(((36.0,), (8.0,)), dtype=np.float32)
        requested_kd = np.asarray(((5.0,), (2.0,)), dtype=np.float32)
        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        robot_joint = layout.get_entity("robot").joints[0].qpos_indices[0]
        qpos[rows, robot_joint] = 0.4
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(
                body_mass=requested_mass,
                kp=requested_kp,
                kd=requested_kd,
            ),
        )

        effective_mass = backend.get_body_mass()
        np.testing.assert_allclose(
            effective_mass[rows, object_body_id],
            (0.75, 1.9),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(effective_mass[untouched_rows], default_mass[untouched_rows])
        native_object_mass = (
            object_runtime.entity.get_links_inertial_mass(
                object_runtime.native_body_indices.tolist()
            )
            .cpu()
            .numpy()
            .reshape(5, -1)[:, 0]
        )
        np.testing.assert_allclose(
            native_object_mass,
            (0.5, 0.75, 0.5, 1.5, 1.9),
            rtol=2e-6,
            atol=1e-7,
        )
        native_kp = (
            robot_runtime.entity.get_dofs_kp()
            .cpu()
            .numpy()
            .reshape(5, -1)[:, robot_runtime.native_actuated_dofs[0]]
        )
        native_kd = (
            robot_runtime.entity.get_dofs_kv()
            .cpu()
            .numpy()
            .reshape(5, -1)[:, robot_runtime.native_actuated_dofs[0]]
        )
        np.testing.assert_allclose(native_kp, (24.0, 36.0, 24.0, 24.0, 8.0), rtol=2e-6)
        np.testing.assert_allclose(native_kd, (3.0, 5.0, 3.0, 3.0, 2.0), rtol=2e-6)

        backend.step(np.zeros((5, 1), dtype=np.float32))
        robot_positions = backend.get_entity_state("robot")["joint_positions"][:, 0]
        correction = 0.4 - robot_positions[rows]
        assert correction[0] > correction[1] > 0.0

        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(
                base_mass_delta=np.asarray((0.25, -0.4), dtype=np.float32)
            ),
        )
        effective_mass = backend.get_body_mass()
        np.testing.assert_allclose(
            effective_mass[rows, passive_body_id],
            (0.95, 0.3),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            effective_mass[rows, object_body_id],
            (0.5, 1.5),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(effective_mass[untouched_rows], default_mass[untouched_rows])
        native_kp = (
            robot_runtime.entity.get_dofs_kp()
            .cpu()
            .numpy()
            .reshape(5, -1)[:, robot_runtime.native_actuated_dofs[0]]
        )
        np.testing.assert_allclose(native_kp[rows], requested_kp[:, 0], rtol=2e-6)
        np.testing.assert_array_equal(
            backend.get_reset_term_default("body_mass"), default_reset_mass
        )
        np.testing.assert_array_equal(
            backend.get_reset_term_default("base_mass_delta"), default_reset_base_delta
        )
        np.testing.assert_array_equal(backend.get_reset_term_default("kp"), default_reset_kp)
        np.testing.assert_array_equal(backend.get_reset_term_default("kd"), default_reset_kd)

        with pytest.raises(
            NotImplementedError,
            match="reset randomization does not support terms",
        ):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(gravity=np.zeros((2, 3))),
            )
    finally:
        # Genesis permits one process-wide session; the final test in this
        # module owns teardown so later native constructions remain valid.
        pass


def test_portable_entities_selected_reset_randomization_com(tmp_path: Path):
    scene = _scene(tmp_path)
    _enable_source_gravity(scene)
    backend = GenesisBackend(scene, 5, 0.002, base_name="passive/base")
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        default_mass = backend.get_body_mass().copy()
        default_ipos = backend.get_body_ipos(np.arange(5)).copy()
        default_reset_ipos = backend.get_reset_term_default("body_ipos")
        default_reset_base_com = backend.get_reset_term_default("base_com_offset")
        np.testing.assert_allclose(default_reset_ipos, default_ipos, rtol=2e-7)
        np.testing.assert_array_equal(default_reset_base_com, np.zeros((5, 3), dtype=np.float32))
        assert not default_reset_ipos.flags.writeable
        assert not default_reset_base_com.flags.writeable

        rows = np.asarray((1, 4), dtype=np.intp)
        untouched_rows = np.asarray((0, 2, 3), dtype=np.intp)
        robot_link_body_id = layout.get_entity("robot").body_ids[1]
        passive_base_body_id = layout.get_entity("passive").body_ids[0]
        object_body_id = layout.get_entity("object").body_ids[0]
        robot_runtime = backend._entity_runtimes["robot"]
        passive_runtime = backend._entity_runtimes["passive"]
        object_runtime = backend._entity_runtimes["object"]
        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()

        with pytest.raises(ValueError, match="body_ipos must have shape"):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(
                    body_ipos=np.zeros((2, layout.nbody, 4), dtype=np.float32)
                ),
            )
        malformed_ipos = default_ipos[rows].copy()
        malformed_ipos[:, robot_link_body_id, 0] = np.nan
        with pytest.raises(ValueError, match="body_ipos must contain only finite values"):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(body_ipos=malformed_ipos),
            )

        requested_mass = default_mass[rows].copy()
        requested_mass[:, object_body_id] = (0.75, 1.9)
        requested_ipos = default_ipos[rows].copy()
        requested_ipos[:, robot_link_body_id, 0] = (0.08, -0.05)
        requested_ipos[:, object_body_id, 0] = (0.07, -0.04)
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(
                body_mass=requested_mass,
                body_ipos=requested_ipos,
            ),
        )

        np.testing.assert_allclose(
            backend.get_body_mass()[rows],
            requested_mass,
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            backend.get_body_ipos(rows),
            requested_ipos,
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            backend.get_body_mass()[untouched_rows], default_mass[untouched_rows]
        )
        np.testing.assert_array_equal(
            backend.get_body_ipos(untouched_rows), default_ipos[untouched_rows]
        )

        solver = backend._scene.sim.rigid_solver
        robot_native_link = int(robot_runtime.entity.get_link("link").idx)
        object_native_link = int(object_runtime.entity.get_link("base").idx)
        native_robot_shift = (
            solver.get_links_COM_shift(links_idx=[robot_native_link], envs_idx=rows.tolist())
            .cpu()
            .numpy()
            .reshape(2, 3)
        )
        native_object_shift = (
            solver.get_links_COM_shift(links_idx=[object_native_link], envs_idx=rows.tolist())
            .cpu()
            .numpy()
            .reshape(2, 3)
        )
        expected_shift = requested_ipos - default_ipos[rows]
        np.testing.assert_allclose(
            native_robot_shift,
            expected_shift[:, robot_link_body_id],
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            native_object_shift,
            expected_shift[:, object_body_id],
            rtol=2e-6,
            atol=1e-7,
        )
        native_object_mass = (
            object_runtime.entity.get_links_inertial_mass(
                object_runtime.native_body_indices.tolist()
            )
            .cpu()
            .numpy()
            .reshape(5, -1)[:, 0]
        )
        np.testing.assert_allclose(
            native_object_mass,
            (0.5, 0.75, 0.5, 1.5, 1.9),
            rtol=2e-6,
            atol=1e-7,
        )

        for _ in range(10):
            backend.step(np.zeros((5, 1), dtype=np.float32))
        robot_joint_velocities = backend.get_entity_state("robot")["joint_velocities"][:, 0]
        assert robot_joint_velocities[1] > 0.01
        assert robot_joint_velocities[4] < -0.005
        np.testing.assert_array_equal(robot_joint_velocities[untouched_rows], 0.0)

        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        base_com_offset = np.asarray(((0.03, 0.0, 0.0), (-0.04, 0.0, 0.0)), dtype=np.float32)
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(base_com_offset=base_com_offset),
        )
        effective_mass = backend.get_body_mass()
        effective_ipos = backend.get_body_ipos(np.arange(5))
        np.testing.assert_allclose(
            effective_mass[rows, object_body_id],
            (0.75, 1.9),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            effective_ipos[rows, passive_base_body_id],
            base_com_offset,
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            effective_ipos[rows, object_body_id, 0],
            ((0.01, 0.03)),
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            effective_ipos[rows, robot_link_body_id, 0],
            (0.0, 0.0),
            atol=1e-7,
        )

        passive_native_base = int(passive_runtime.entity.get_link("base").idx)
        native_base_shift = (
            solver.get_links_COM_shift(links_idx=[passive_native_base], envs_idx=rows.tolist())
            .cpu()
            .numpy()
            .reshape(2, 3)
        )
        native_object_shift = (
            solver.get_links_COM_shift(links_idx=[object_native_link], envs_idx=rows.tolist())
            .cpu()
            .numpy()
            .reshape(2, 3)
        )
        np.testing.assert_allclose(native_base_shift, base_com_offset, rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(native_object_shift, np.zeros((2, 3)))
        np.testing.assert_array_equal(
            backend.get_reset_term_default("body_ipos"), default_reset_ipos
        )
        np.testing.assert_array_equal(
            backend.get_reset_term_default("base_com_offset"), default_reset_base_com
        )
    finally:
        # Genesis permits one process-wide session; the final test in this
        # module owns teardown so later native constructions remain valid.
        pass


def test_portable_entities_selected_reset_randomization_dof_properties(tmp_path: Path):
    scene = _scene(tmp_path)
    _enable_source_gravity(scene)
    backend = GenesisBackend(scene, 5, 0.002, base_name="passive/base")
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        defaults = {
            "dof_damping": backend.get_dof_damping().copy(),
            "dof_frictionloss": backend.get_dof_frictionloss().copy(),
            "dof_armature": backend.get_dof_armature().copy(),
        }
        reset_defaults = {term: backend.get_reset_term_default(term) for term in defaults}
        for term, default in defaults.items():
            assert reset_defaults[term].shape == (5, layout.nv)
            np.testing.assert_allclose(reset_defaults[term], np.tile(default, (5, 1)), rtol=2e-7)
            assert not reset_defaults[term].flags.writeable

        supported_terms = backend.get_dr_capabilities().supported_reset_terms
        assert set(defaults).issubset(supported_terms)
        rows = np.asarray((1, 4), dtype=np.intp)
        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        robot_joint = layout.get_entity("robot").joints[0].qpos_indices[0]
        qpos[rows, robot_joint] = 0.4
        malformed = np.tile(defaults["dof_damping"], (2, 1))[..., None]
        with pytest.raises(ValueError, match="dof_damping must have shape"):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(dof_damping=malformed),
            )
        negative = np.tile(defaults["dof_frictionloss"], (2, 1))
        negative[:, 0] = -0.1
        with pytest.raises(ValueError, match="dof_frictionloss must contain only non-negative"):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(dof_frictionloss=negative),
            )
        nonfinite = np.tile(defaults["dof_armature"], (2, 1))
        nonfinite[:, -1] = np.nan
        with pytest.raises(ValueError, match="dof_armature must contain only finite values"):
            backend.set_state(
                rows,
                qpos[rows],
                qvel[rows],
                randomization=ResetRandomizationPayload(dof_armature=nonfinite),
            )

        requested = {term: np.tile(default, (2, 1)) for term, default in defaults.items()}
        robot_dof = layout.get_entity("robot").qvel_indices[0]
        passive_dof = layout.get_entity("passive").qvel_indices[-1]
        requested["dof_damping"][:, robot_dof] = (0.1, 2.0)
        requested["dof_damping"][:, passive_dof] = (0.03, 3.1)
        requested["dof_frictionloss"][:, robot_dof] = 0.04
        requested["dof_frictionloss"][:, passive_dof] = 0.08
        requested["dof_armature"][:, robot_dof] = 0.02
        requested["dof_armature"][:, passive_dof] = 0.05
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(**requested),
        )

        expected = {term: np.tile(default, (5, 1)) for term, default in defaults.items()}
        for term in expected:
            expected[term][rows] = requested[term]

        getters = {
            "dof_damping": "get_dofs_damping",
            "dof_frictionloss": "get_dofs_frictionloss",
            "dof_armature": "get_dofs_armature",
        }
        for runtime in backend._entity_runtimes.values():
            if not runtime.native_qvel_indices.size:
                continue
            for term, getter_name in getters.items():
                native_values = (
                    getattr(runtime.entity, getter_name)(runtime.native_qvel_indices.tolist())
                    .cpu()
                    .numpy()
                )
                np.testing.assert_allclose(
                    native_values,
                    expected[term][:, runtime.qvel_indices],
                    rtol=2e-6,
                    atol=1e-7,
                )

        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        backend.set_state(
            rows,
            qpos[rows],
            qvel[rows],
            randomization=ResetRandomizationPayload(
                base_com_offset=np.zeros((2, 3), dtype=np.float32)
            ),
        )
        for runtime in backend._entity_runtimes.values():
            if not runtime.native_qvel_indices.size:
                continue
            for term, getter_name in getters.items():
                native_values = (
                    getattr(runtime.entity, getter_name)(runtime.native_qvel_indices.tolist())
                    .cpu()
                    .numpy()
                )
                np.testing.assert_allclose(
                    native_values,
                    expected[term][:, runtime.qvel_indices],
                    rtol=2e-6,
                    atol=1e-7,
                )
        for term, default in defaults.items():
            np.testing.assert_array_equal(
                backend.get_reset_term_default(term), reset_defaults[term]
            )
            backend_getter = getattr(backend, f"get_{term}")
            np.testing.assert_array_equal(backend_getter(), default)

        for _ in range(10):
            backend.step(np.zeros((5, 1), dtype=np.float32))
        robot_positions = backend.get_entity_state("robot")["joint_positions"][:, 0]
        correction = 0.4 - robot_positions
        assert correction[1] > correction[4] > 0.0
    finally:
        # Genesis permits one process-wide session; the final test in this
        # module owns teardown so later native constructions remain valid.
        pass


def test_portable_entities_default_keyframes_and_selected_control_restoration(
    tmp_path: Path,
):
    scene = _scene(tmp_path)
    entities = list(scene.entity_assets)
    entities[0] = replace(entities[0], source=_robot_with_home_key(tmp_path / "keys"))
    entities[1] = replace(entities[1], source=_passive_with_ignored_root_key(tmp_path / "keys"))
    scene.entity_assets = tuple(entities)
    scene.default_keyframe_name = "home"
    binding = scene.entity_variant
    assert binding is not None
    keyed_variants = tuple(
        _object_with_home_root_key(
            source,
            tmp_path / "object-keys",
            f"object-home-root-key-{variant}",
        )
        for variant, source in enumerate(binding.plan.variants)
    )
    scene.entity_variant = replace(
        binding,
        plan=replace(binding.plan, variants=keyed_variants),
    )

    backend = GenesisBackend(scene, 5, 0.002)
    robot_entity = next(entity for entity in backend._scene.entities if str(entity.name) == "robot")
    original_control = robot_entity.control_dofs_position
    control_calls: list[tuple[np.ndarray, list[int], list[int] | None]] = []

    def capture_control(position, dofs_idx_local=None, envs_idx=None):
        control_calls.append(
            (
                position.detach().cpu().numpy().copy(),
                dofs_idx_local,
                envs_idx,
            )
        )
        return original_control(position, dofs_idx_local=dofs_idx_local, envs_idx=envs_idx)

    robot_entity.control_dofs_position = capture_control
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        robot_runtime = backend._entity_runtimes["robot"]
        passive_runtime = backend._entity_runtimes["passive"]

        assert len(control_calls) == 1
        np.testing.assert_allclose(control_calls[0][0], np.full((5, 1), 0.4), atol=1e-7)
        assert control_calls[0][1] == robot_runtime.native_actuated_dofs.tolist()
        assert control_calls[0][2] is None
        np.testing.assert_allclose(robot_runtime.source_metadata[0].default_qpos, [0.4], atol=1e-7)
        np.testing.assert_allclose(robot_runtime.source_metadata[0].default_qvel, [0.2], atol=1e-7)
        np.testing.assert_allclose(
            passive_runtime.source_metadata[0].default_qpos,
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(passive_runtime.source_metadata[0].default_qvel, 0.0, atol=1e-7)

        robot_state = backend.get_entity_state("robot")
        np.testing.assert_allclose(robot_state["root_pose"][:, :3], np.tile((0, 0, 1), (5, 1)))
        np.testing.assert_allclose(robot_state["joint_positions"], 0.4, atol=1e-6)
        np.testing.assert_allclose(robot_state["joint_velocities"], 0.2, atol=1e-6)
        passive_state = backend.get_entity_state("passive")
        np.testing.assert_allclose(
            passive_state["root_pose"][:, :3], np.tile((1, 0, 1), (5, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            passive_state["root_pose"][:, 3:],
            np.tile((1, 0, 0, 0), (5, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(passive_state["root_velocity"], 0.0, atol=1e-6)
        np.testing.assert_allclose(passive_state["joint_positions"], 0.0, atol=1e-6)
        np.testing.assert_allclose(passive_state["joint_velocities"], 0.0, atol=1e-6)

        robot_default = backend.get_entity_default_state("robot", (1, 3))
        for name, values in robot_default.items():
            np.testing.assert_allclose(values, robot_state[name][[1, 3]], atol=1e-6)
        passive_default = backend.get_entity_default_state("passive", (2, 4))
        for name, values in passive_default.items():
            np.testing.assert_allclose(values, passive_state[name][[2, 4]], atol=1e-6)
        object_state = backend.get_entity_state("object")
        object_default = backend.get_entity_default_state("object", (0, 3))
        for name, values in object_default.items():
            np.testing.assert_allclose(values, object_state[name][[0, 3]], atol=1e-6)

        backend.step(np.full((5, 1), 0.1, dtype=np.float32))
        robot_after_step = backend.get_entity_state("robot")
        passive_after_step = backend.get_entity_state("passive")
        control_calls.clear()
        backend.reset_entities(
            SceneResetRequest(
                (2,),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.asarray([[0.8]], np.float32),
                        joint_velocities=np.asarray([[-0.1]], np.float32),
                    ),
                ),
                restore_default_controls=True,
            )
        )
        assert len(control_calls) == 1
        np.testing.assert_allclose(control_calls[0][0], [[0.4]], atol=1e-7)
        assert control_calls[0][2] == [2]
        robot_after_restore = backend.get_entity_state("robot")
        np.testing.assert_allclose(robot_after_restore["joint_positions"][2], 0.8, atol=1e-6)
        np.testing.assert_allclose(robot_after_restore["joint_velocities"][2], -0.1, atol=1e-6)
        np.testing.assert_array_equal(
            robot_after_restore["joint_positions"][[0, 1, 3, 4]],
            robot_after_step["joint_positions"][[0, 1, 3, 4]],
        )
        np.testing.assert_array_equal(
            robot_after_restore["joint_velocities"][[0, 1, 3, 4]],
            robot_after_step["joint_velocities"][[0, 1, 3, 4]],
        )
        for name, values in passive_after_step.items():
            np.testing.assert_array_equal(backend.get_entity_state("passive")[name], values)

        control_calls.clear()
        backend.reset_entities(
            SceneResetRequest(
                (3,),
                (EntityStatePatch("robot", joint_positions=np.asarray([[0.7]], np.float32)),),
            )
        )
        assert len(control_calls) == 1
        np.testing.assert_allclose(control_calls[0][0], [[0.0]], atol=0.0)
        assert control_calls[0][2] == [3]
        assert layout.get_entity("robot").actuator_indices == (0,)
    finally:
        robot_entity.control_dofs_position = original_control


def test_portable_entities_body_force_mapping_and_reset_cancellation(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    backend = GenesisBackend(scene, 5, 0.002)
    backend.materialize()
    layout = backend.get_scene_layout()
    object_body = layout.get_entity("object").body_ids[0]
    passive_base_body = layout.get_entity("passive").body_ids[0]
    object_root = layout.get_entity("object").root_qpos_indices

    capabilities = backend.get_dr_capabilities()
    assert capabilities.supports_interval_body_force
    assert capabilities.supports_interval_term("body_force")
    with pytest.raises(ValueError, match="body force must have shape"):
        backend.apply_body_force(np.asarray((object_body,)), np.zeros((5, 3), np.float32))
    with pytest.raises(ValueError, match="body force contains NaN or Inf"):
        backend.apply_body_force(
            np.asarray((object_body,)), np.full((5, 1, 3), np.nan, np.float32)
        )
    with pytest.raises(ValueError, match="body_ids must be in"):
        backend.apply_body_force(np.asarray((layout.nbody,)), np.zeros((5, 1, 3), np.float32))
    with pytest.raises(ValueError, match="must reference owned physical bodies"):
        backend.apply_body_force(np.asarray((0,)), np.zeros((5, 1, 3), np.float32))
    with pytest.raises(NotImplementedError, match="does not support interval body torque"):
        backend.apply_body_force(
            np.asarray((object_body,)),
            np.zeros((5, 1, 3), np.float32),
            torque=np.zeros((5, 1, 3), np.float32),
        )

    def reject_callback_staging(owner: GenesisBackend, ctrl: np.ndarray) -> np.ndarray:
        owner.apply_body_force(np.asarray((object_body,)), np.zeros((5, 1, 3), np.float32))
        return ctrl

    backend.set_pre_step_control(reject_callback_staging)
    with pytest.raises(RuntimeError, match="must not be called from inside a pre-step control"):
        backend.step(np.zeros((5, 1), dtype=np.float32))
    backend.set_pre_step_control(None)

    qpos = backend._qpos_cache[1].copy()
    qvel = backend._qvel_cache[1].copy()
    qpos[:, object_root] = np.asarray(
        (2.0, 0.0, 1.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32
    )
    backend.set_state(np.arange(5, dtype=np.intp), qpos, qvel)

    force = np.zeros((5, 2, 3), dtype=np.float32)
    force[:, 0, 0] = 0.06
    force[:, 1, 0] = 0.12
    backend.apply_body_force(np.asarray((object_body, passive_base_body)), force)
    force[:, 0, 0] = 0.06
    force[:, 1, :] = 0.0
    backend.apply_body_force(np.asarray((object_body,)), force[:, :1])
    pending = backend._portable_pending_body_forces
    assert pending is not None
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pending[:, passive_base_body, 0], 0.12, rtol=0.0, atol=1e-8)

    backend.reset_entities(
        SceneResetRequest(
            (1,),
            (EntityStatePatch("robot", joint_positions=np.zeros((1, 1), np.float32)),),
        )
    )
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pending[:, passive_base_body, 0], 0.12, rtol=0.0, atol=1e-8)

    backend.reset_entities(
        SceneResetRequest(
            (2, 4),
            (EntityStatePatch("passive", joint_positions=np.zeros((2, 1), np.float32)),),
        )
    )
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        pending[[0, 1, 3], passive_base_body, 0], 0.12, rtol=0.0, atol=1e-8
    )
    np.testing.assert_allclose(
        pending[[2, 4], passive_base_body, :], 0.0, rtol=0.0, atol=1e-8
    )

    backend.step(np.zeros((5, 1), dtype=np.float32))
    object_velocity = backend.get_entity_state("object")["root_velocity"]
    np.testing.assert_allclose(
        object_velocity[:, 0],
        0.12 / np.asarray((0.5, 0.5, 0.5, 1.5, 1.5)) * 0.002,
        rtol=2e-3,
        atol=1e-8,
    )
    np.testing.assert_allclose(object_velocity[:, 1:], 0.0, atol=2e-6)
    passive_velocity = backend.get_entity_state("passive")["root_velocity"]
    assert np.all(passive_velocity[[0, 1, 3], 0] > 0.0)
    np.testing.assert_array_equal(passive_velocity[[2, 4], 0], 0.0)
    np.testing.assert_array_equal(pending, 0.0)

    consumed_velocity = object_velocity.copy()
    backend.step(np.zeros((5, 1), dtype=np.float32))
    np.testing.assert_allclose(
        backend.get_entity_state("object")["root_velocity"], consumed_velocity, atol=1e-7
    )
    np.testing.assert_array_equal(pending, 0.0)
    velocity_before_interval = backend.get_entity_state("object")["root_velocity"].copy()

    stale_force = np.full((5, 1, 3), 0.07, dtype=np.float32)
    backend.apply_body_force(np.asarray((object_body,)), stale_force)
    interval_force = np.full((5, 1, 3), 0.06, dtype=np.float32)
    interval_force[:, :, 1:] = 0.0
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(
                IntervalTermOp(INTERVAL_TERM_BODY_FORCE, interval_force, body_ids=(object_body,)),
                IntervalTermOp(INTERVAL_TERM_BODY_FORCE, interval_force, body_ids=(object_body,)),
            )
        )
    )
    np.testing.assert_allclose(pending[:, object_body, 0], 0.12, rtol=0.0, atol=1e-8)
    backend.step(np.zeros((5, 1), dtype=np.float32))
    np.testing.assert_allclose(
        backend.get_entity_state("object")["root_velocity"],
        velocity_before_interval + consumed_velocity,
        atol=1e-7,
    )
    np.testing.assert_array_equal(pending, 0.0)


def test_portable_entities_layout_variants_selected_state_and_control(tmp_path: Path):
    with pytest.raises(ValueError, match="balanced mapping"):
        GenesisBackend(_scene(tmp_path, assignment=(1, 1, 0, 1, 0)), 5, 0.002)

    scene = _scene(tmp_path, object_site_sensors=True)
    entities = list(scene.entity_assets)
    entities[1] = replace(
        entities[1],
        source=_passive_with_site_sensors(tmp_path / "passive-source"),
    )
    scene.entity_assets = tuple(entities)
    _enable_native_contact_masks(scene, object_variant_b=(4, 64))
    backend = GenesisBackend(scene, 5, 0.002)
    try:
        backend.materialize()
        layout = backend.get_scene_layout()
        assert backend.get_entity_names() == ("robot", "passive", "object", "table")
        assert (layout.nq, layout.nv, layout.nu) == (16, 14, 1)
        assert layout.get_entity("robot").root_mode == "fixed"
        assert layout.get_entity("passive").root_mode == "floating"
        assert layout.get_entity("object").root_mode == "floating"
        assert layout.get_entity("table").root_mode == "fixed"
        assert backend.get_actuator_joint_names() == ("robot/drive",)
        assert backend.get_body_ids(("robot/base", "passive/base"))[0] == 1

        robot_runtime = backend._entity_runtimes["robot"]
        passive_runtime = backend._entity_runtimes["passive"]
        object_runtime = backend._entity_runtimes["object"]
        table_runtime = backend._entity_runtimes["table"]
        assert robot_runtime.native_body_indices.tolist() == [
            int(robot_runtime.entity.get_link(name).idx_local) for name in ("base", "link")
        ]
        assert passive_runtime.native_body_indices.tolist() == [
            int(passive_runtime.entity.get_link(name).idx_local) for name in ("base", "child")
        ]
        assert not passive_runtime.native_actuated_dofs.size
        assert not table_runtime.native_actuated_dofs.size

        native_mass = np.asarray(
            object_runtime.entity.get_links_inertial_mass(
                object_runtime.native_body_indices.tolist()
            )
            .cpu()
            .numpy()
        ).reshape(5, -1)[:, 0]
        np.testing.assert_allclose(native_mass, [0.5, 0.5, 0.5, 1.5, 1.5], rtol=2e-6, atol=1e-7)
        object_body_id = layout.get_entity("object").body_ids[0]
        np.testing.assert_allclose(
            backend.get_body_mass()[:, object_body_id],
            [0.5, 0.5, 0.5, 1.5, 1.5],
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            backend.get_body_ipos(np.arange(5))[:, object_body_id, 0],
            [0.01, 0.01, 0.01, 0.03, 0.03],
            rtol=2e-6,
        )
        assert backend.get_body_ipos().shape == (layout.nbody, 3)
        assert backend.get_geom_names() == (
            "robot/base_geom",
            "robot/link_geom",
            "passive/passive_base_geom",
            "passive/passive_child_geom",
            "object/object_geom",
            "table/table_geom",
        )
        assert backend.get_geom_id("object/object_geom") == 4
        np.testing.assert_array_equal(
            backend.get_geom_body_ids(),
            layout.get_body_ids(
                (
                    "robot/base",
                    "robot/link",
                    "passive/base",
                    "passive/child",
                    "object/base",
                    "table/base",
                )
            ),
        )
        native_contype, native_conaffinity = backend.get_geom_contact_masks()
        np.testing.assert_array_equal(native_contype, [1, 1, 2, 2, 4, 8])
        np.testing.assert_array_equal(native_conaffinity, [16, 16, 32, 32, 64, 128])
        native_friction = backend.get_geom_friction()
        assert native_friction.shape == (6, 3)
        np.testing.assert_allclose(
            native_friction,
            np.tile((0.37, 0.004, 0.002), (6, 1)),
            rtol=2e-6,
            atol=2e-7,
        )
        for entity in layout.entities:
            runtime = backend._entity_runtimes[entity.name]
            native_values = {
                (
                    str(geom.metadata.get("name", "")),
                    str(geom.link.name),
                    float(geom.friction),
                    float(geom.friction_torsional),
                    float(geom.friction_rolling),
                )
                for geom in runtime.entity.geoms
            }
            for geom in entity.geoms:
                assert (
                    geom.name,
                    geom.body_name,
                    0.37,
                    0.004,
                    0.002,
                ) in native_values
        native_solref = backend.get_geom_solref()
        native_solimp = backend.get_geom_solimp()
        assert native_solref.shape == (6, 2)
        assert native_solimp.shape == (6, 5)
        np.testing.assert_allclose(
            native_solref,
            np.tile((0.021, 0.89), (6, 1)),
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            native_solimp,
            np.tile((0.91, 0.94, 0.0013, 0.53, 2.2), (6, 1)),
            rtol=2e-6,
            atol=2e-7,
        )
        for entity in layout.entities:
            runtime = backend._entity_runtimes[entity.name]
            native_values = {
                (
                    str(geom.metadata.get("name", "")),
                    str(geom.link.name),
                ): np.asarray(geom.sol_params.detach().cpu().numpy(), dtype=np.float64)
                for geom in runtime.entity.geoms
            }
            for geom in entity.geoms:
                native_solver_params = native_values[(geom.name, geom.body_name)]
                np.testing.assert_allclose(
                    native_solver_params,
                    (0.021, 0.89, 0.91, 0.94, 0.0013, 0.53, 2.2),
                    rtol=2e-6,
                    atol=2e-7,
                )
        expected_damping = np.zeros((layout.nv,), dtype=np.float64)
        expected_frictionloss = np.zeros((layout.nv,), dtype=np.float64)
        expected_armature = np.zeros((layout.nv,), dtype=np.float64)
        expected_damping[layout.get_entity("robot").qvel_indices] = 0.17
        expected_damping[layout.get_entity("passive").qvel_indices[-1]] = 0.31
        expected_frictionloss[layout.get_entity("robot").qvel_indices] = 0.043
        expected_frictionloss[layout.get_entity("passive").qvel_indices[-1]] = 0.027
        expected_armature[layout.get_entity("robot").qvel_indices] = 0.011
        expected_armature[layout.get_entity("passive").qvel_indices[-1]] = 0.023
        np.testing.assert_allclose(
            backend.get_dof_damping(), expected_damping, rtol=2e-7, atol=2e-8
        )
        np.testing.assert_allclose(
            backend.get_dof_frictionloss(),
            expected_frictionloss,
            rtol=2e-7,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            backend.get_dof_armature(),
            expected_armature,
            rtol=2e-7,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            object_runtime.geom_sizes[:, 0, 0], [0.1, 0.15], rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            backend.get_geom_size("robot/base_geom"), [0.08, 0.0, 0.0], atol=2e-6
        )
        np.testing.assert_allclose(
            backend.get_geom_size("table/table_geom"), [1.0, 1.0, 0.1], atol=2e-6
        )
        with pytest.raises(NotImplementedError, match="non-uniform public geometry sizes"):
            backend.get_geom_size("object/object_geom")
        with pytest.raises(NotImplementedError, match="non-uniform public geometry sizes"):
            backend.get_geom_sizes()
        native_vgeoms = list(object_runtime.entity.vgeoms)
        assert len(native_vgeoms) == 2
        assert native_vgeoms[0].active_envs_idx is not None
        assert native_vgeoms[0].active_envs_idx.tolist() == [0, 1, 2]
        assert native_vgeoms[1].active_envs_idx is not None
        assert native_vgeoms[1].active_envs_idx.tolist() == [3, 4]
        np.testing.assert_allclose(
            np.min(np.asarray(native_vgeoms[0].init_vverts), axis=0), -0.1, atol=2e-6
        )
        np.testing.assert_allclose(
            np.max(np.asarray(native_vgeoms[0].init_vverts), axis=0), 0.1, atol=2e-6
        )
        np.testing.assert_allclose(
            np.min(np.asarray(native_vgeoms[1].init_vverts), axis=0), -0.15, atol=2e-6
        )
        np.testing.assert_allclose(
            np.max(np.asarray(native_vgeoms[1].init_vverts), axis=0), 0.15, atol=2e-6
        )

        table_state = backend.get_entity_state("table")
        np.testing.assert_allclose(
            table_state["root_pose"][:, :3], np.tile((0.0, 0.0, -3.0), (5, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"], 0.1, atol=1e-6
        )
        assert tuple(backend._sensor_slots) == (
            "passive/site_pos",
            "passive/site_quat",
            "passive/site_gyro",
            "passive/site_vel",
            "passive/site_acc",
            "object/site_pos",
            "object/site_quat",
        )
        passive_site_positions = backend.get_sensor_data("passive/site_pos")
        passive_site_quaternions = backend.get_sensor_data("passive/site_quat")
        passive_site_gyros = backend.get_sensor_data("passive/site_gyro")
        passive_site_velocities = backend.get_sensor_data("passive/site_vel")
        passive_site_accelerations = backend.get_sensor_data("passive/site_acc")
        np.testing.assert_allclose(
            passive_site_positions,
            np.tile((1.05, 0.0, 1.15), (5, 1)),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            passive_site_quaternions,
            np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)),
            atol=2e-6,
        )
        np.testing.assert_allclose(passive_site_gyros, 0.0, atol=2e-6)
        np.testing.assert_allclose(passive_site_velocities, 0.0, atol=2e-6)
        np.testing.assert_allclose(passive_site_accelerations, 0.0, atol=2e-6)
        np.testing.assert_allclose(
            backend.get_sensor_data("object/site_pos"),
            np.tile((2.05, 0.0, 1.0), (5, 1)),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            backend.get_sensor_data("object/site_quat"),
            np.tile((1.0, 0.0, 0.0, 0.0), (5, 1)),
            atol=2e-6,
        )

        qpos = backend._qpos_cache[1].copy()
        qvel = backend._qvel_cache[1].copy()
        robot_joint = layout.get_entity("robot").joints[0].qpos_indices[0]
        object_root = layout.get_entity("object").root_qpos_indices
        rows = np.asarray((1, 4), dtype=np.intp)
        qpos[rows, robot_joint] = (0.2, 0.3)
        qpos[rows[:, None], object_root[3:7]] = np.asarray(
            [(0, 1, 0, 0), (0, 0, 1, 0)], dtype=np.float32
        )
        qvel[rows, layout.get_entity("robot").joints[0].qvel_indices[0]] = (0.1, -0.1)
        backend.set_state(rows, qpos[rows], qvel[rows])
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"][rows],
            np.asarray([[0.2], [0.3]], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_velocities"][rows],
            np.asarray([[0.1], [-0.1]], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.get_entity_state("passive")["joint_positions"], 0.0, atol=1e-7
        )

        ctrl = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        robot_before = backend.get_entity_state("robot")["joint_positions"].copy()
        backend.step(ctrl)
        robot_after_step = backend.get_entity_state("robot")["joint_positions"]
        assert np.max(np.abs(robot_after_step - robot_before)) > 1e-4
        assert robot_after_step[1, 0] > 0.2
        # Collision-enabled native mask readback permits a small passive
        # self-contact response; it must remain near the passive default.
        np.testing.assert_allclose(
            backend.get_entity_state("passive")["joint_positions"], 0.0, atol=2e-3
        )

        angle = 0.6
        rate = 0.8
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "passive",
                        joint_positions=np.asarray([[angle]], dtype=np.float32),
                        joint_velocities=np.asarray([[rate]], dtype=np.float32),
                    ),
                ),
            )
        )
        passive_positions_after = backend.get_sensor_data("passive/site_pos")
        passive_quaternions_after = backend.get_sensor_data("passive/site_quat")
        passive_gyros_after = backend.get_sensor_data("passive/site_gyro")
        passive_velocities_after = backend.get_sensor_data("passive/site_vel")
        np.testing.assert_allclose(
            passive_positions_after[1],
            (1.0 + 0.05 * np.cos(angle), 0.0, 1.15 - 0.05 * np.sin(angle)),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            passive_quaternions_after[1],
            (np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0),
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            passive_positions_after[[0, 2, 3, 4]], passive_site_positions[[0, 2, 3, 4]]
        )
        np.testing.assert_array_equal(
            passive_quaternions_after[[0, 2, 3, 4]],
            passive_site_quaternions[[0, 2, 3, 4]],
        )
        link_quat = np.asarray((np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0), dtype=np.float64)
        site_quat = _quat_mul(
            tuple(link_quat),
            (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
        )
        angular_world = np.asarray((0.0, rate, 0.0), dtype=np.float64)
        site_offset_world = _quat_rotate(link_quat, np.asarray((0.05, 0.0, 0.0)))
        velocity_world = np.cross(angular_world, site_offset_world)
        np.testing.assert_allclose(
            passive_gyros_after[1],
            _quat_rotate_inverse(site_quat, angular_world),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            passive_velocities_after[1],
            _quat_rotate_inverse(site_quat, velocity_world),
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            passive_gyros_after[[0, 2, 3, 4]], passive_site_gyros[[0, 2, 3, 4]]
        )
        np.testing.assert_array_equal(
            passive_velocities_after[[0, 2, 3, 4]],
            passive_site_velocities[[0, 2, 3, 4]],
        )

        backend.step(np.zeros((5, 1), dtype=np.float32))
        passive_accelerations_after = backend.get_sensor_data("passive/site_acc")
        native_acceleration = backend._imu_sensors["passive/site_acc"].read().lin_acc.cpu().numpy()
        assert float(np.max(np.abs(passive_accelerations_after[1]))) > 1e-3
        np.testing.assert_allclose(
            passive_accelerations_after,
            native_acceleration,
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            passive_accelerations_after[[0, 2, 3, 4]],
            passive_site_accelerations[[0, 2, 3, 4]],
            atol=2e-6,
        )

        robot_before_reset = backend.get_entity_state("robot")["joint_positions"].copy()
        passive_before = {
            name: values.copy() for name, values in backend.get_entity_state("passive").items()
        }
        object_before = {
            name: values.copy() for name, values in backend.get_entity_state("object").items()
        }
        backend.reset_entities(
            SceneResetRequest(
                (4,),
                (EntityStatePatch("robot", joint_positions=np.asarray([[0.8]], np.float32)),),
            )
        )
        robot_after = backend.get_entity_state("robot")["joint_positions"]
        np.testing.assert_allclose(robot_after[4], 0.8, atol=1e-6)
        np.testing.assert_array_equal(robot_after[:4], robot_before_reset[:4])
        for name, values in passive_before.items():
            np.testing.assert_array_equal(backend.get_entity_state("passive")[name], values)
        for name, values in object_before.items():
            np.testing.assert_array_equal(backend.get_entity_state("object")[name], values)

        variant_inertias = [item.body_inertia[1] for item in object_runtime.source_metadata]
        np.testing.assert_allclose(variant_inertias[0], (0.02, 0.03, 0.04), rtol=2e-6)
        np.testing.assert_allclose(variant_inertias[1], (0.03, 0.04, 0.05), rtol=2e-6)
        object_root_qvel = layout.get_entity("object").root_qvel_indices
        response_qpos = backend._qpos_cache[1].copy()
        response_qpos[:, object_root] = np.asarray(
            (2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
        )
        response_qvel = backend._qvel_cache[1].copy()
        response_qvel[:, object_root_qvel[3:]] = np.asarray((1.4, 0.3, 0.9), dtype=np.float32)
        backend.set_state(np.arange(5, dtype=np.intp), response_qpos, response_qvel)
        for _ in range(20):
            backend.step(np.zeros((5, 1), dtype=np.float32))
        object_quat = backend.get_entity_state("object")["root_pose"][:, 3:7]
        variant_a_distance = float(np.max(np.linalg.norm(object_quat[:3] - object_quat[0], axis=1)))
        variant_b_distance = float(np.max(np.linalg.norm(object_quat[3:] - object_quat[3], axis=1)))
        cross_variant_distance = float(
            np.max(np.linalg.norm(object_quat[:3, None, :] - object_quat[None, 3:, :], axis=-1))
        )
        assert cross_variant_distance > max(variant_a_distance, variant_b_distance) * 3.0
    finally:
        backend.close()

    assert backend._composed_scene is None
    assert backend._portable_sources is None
    assert backend._scene_cleanup_handle is None
