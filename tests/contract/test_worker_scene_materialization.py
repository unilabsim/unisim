"""Real cold-source compilation for native worker payloads, not native physics."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from unisim.backend.subprocess_ipc.scene_materialization import prepare_worker_scene
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg


def scene(tmp_path: Path, *, damping: float = 0) -> SceneCfg:
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base"><geom name="base_collision" size=".1" mass="1"/>'
        f'<body name="tip"><joint name="hinge" damping="{damping}"/>'
        '<geom size=".1" mass="1"/></body></body></worldbody>'
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator></mujoco>'
    )
    objects = []
    for index, (mass, radius) in enumerate(((1, ".1"), (3, ".15"))):
        source = tmp_path / f"object-{index}.xml"
        source.write_text(
            '<mujoco><worldbody><body name="base"><freejoint/>'
            f'<geom type="sphere" size="{radius}" mass="{mass}" '
            f'friction="{0.7 - index * 0.3} {0.2 - index * 0.1} {0.03 - index * 0.01}"/>'
            "</body></worldbody></mujoco>"
        )
        objects.append(ModelSourceDescriptor(str(source)))
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
            SceneEntitySpec(
                "object",
                objects[0],
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                mirror_of="object",
                collision_enabled=False,
                initial_state=EntityInitialState(position=(2.0, 0.0, 1.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), tuple(objects))
        ),
    )


def test_worker_sources_have_explicit_inertia_and_no_unsupported_canonical_actuators(tmp_path):
    prepared = prepare_worker_scene(scene(tmp_path), 5, 0.002)
    directory = Path(prepared.owner.model_file).parent
    try:
        assert (prepared.layout.nq, prepared.layout.nv, prepared.layout.nu) == (8, 7, 1)
        entries = prepared.payload["scene_entities"]
        assert [entry["self_collision"] for entry in entries] == [False, False, False]
        assert [entry["gravity_disabled"] for entry in entries] == [None, None, None]
        assert entries[1]["assignment"] == entries[2]["assignment"] == [1, 1, 0, 1, 0]
        assert entries[1]["variants"][1]["body_mass"] == [3.0]
        assert entries[1]["variants"][0]["body_sphere_radii"] == [[0.1]]
        assert entries[1]["variants"][1]["body_sphere_radii"] == [[0.15]]
        assert entries[0]["variants"][0]["body_sphere_radii"] == [[0.1], [0.1]]
        assert entries[0]["variants"][0]["dof_stiffness"] == [20.0]
        assert entries[0]["variants"][0]["dof_damping"] == [2.0]
        assert entries[0]["variants"][0]["geom_names"] == ["base_collision", "tip::geom0"]
        assert entries[0]["variants"][0]["geom_body_names"] == ["base", "tip"]
        assert entries[0]["variants"][0]["geom_contype"] == [1, 1]
        assert entries[0]["variants"][0]["geom_conaffinity"] == [1, 1]
        np.testing.assert_allclose(
            entries[0]["variants"][0]["geom_friction"], [[1.0, 0.005, 0.0001]] * 2
        )
        assert entries[1]["variants"][0]["geom_names"] == ["base::geom0"]
        assert entries[1]["variants"][1]["geom_names"] == ["base::geom0"]
        np.testing.assert_allclose(
            entries[1]["variants"][0]["geom_friction"], [[0.7, 0.2, 0.03]]
        )
        np.testing.assert_allclose(
            entries[1]["variants"][1]["geom_friction"], [[0.4, 0.1, 0.02]]
        )
        for entry in entries:
            for source in entry["sources"]:
                root = ET.parse(source).getroot()
                assert root.find(".//inertial") is not None
                assert root.find("actuator") is None
                assert not root.findall('.//body[@mocap="true"]')
        np.testing.assert_allclose(prepared.roots[:, 1, :3], np.tile([0, 0, 1], (5, 1)))
        np.testing.assert_allclose(prepared.roots[:, 2, :3], np.tile([2, 0, 1], (5, 1)))
    finally:
        prepared.close()
    assert not directory.exists()


def test_mirror_receives_the_same_role_neutral_expanded_source_as_its_source_entity(tmp_path):
    config = scene(tmp_path)
    config.entity_assets = (
        config.entity_assets[0],
        config.entity_assets[2],
        config.entity_assets[1],
    )
    prepared = prepare_worker_scene(config, 5, 0.002)
    try:
        entries = prepared.payload["scene_entities"]
        mirror_entry, object_entry = entries[1], entries[2]
        assert len(object_entry["sources"]) == len(mirror_entry["sources"]) == 2
        for object_source, mirror_source in zip(object_entry["sources"], mirror_entry["sources"]):
            assert Path(object_source).read_bytes() == Path(mirror_source).read_bytes()
    finally:
        prepared.close()


def test_initial_state_reloads_only_assignment_selected_scene_variants(
    tmp_path, monkeypatch
):
    import mujoco

    config = scene(tmp_path)
    assert config.entity_variant is not None
    old_plan = config.entity_variant.plan
    unused = tmp_path / "object-2.xml"
    unused.write_text(Path(old_plan.variants[0].model_file).read_text(encoding="utf-8"))
    plan = replace(
        old_plan,
        assignment=np.asarray((2, 2, 0, 2, 0), dtype=np.int32),
        variants=(*old_plan.variants, ModelSourceDescriptor(str(unused))),
    )
    config = replace(config, entity_variant=EntityVariantBinding("object", plan))

    original = mujoco.MjModel.from_xml_path
    scene_loads: dict[str, int] = {}

    def tracked(path: str, *args: object, **kwargs: object):
        name = Path(path).name
        if name.startswith("scene-"):
            scene_loads[name] = scene_loads.get(name, 0) + 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", staticmethod(tracked))
    prepared = prepare_worker_scene(config, 5, 0.002)
    try:
        assert scene_loads["scene-1.xml"] == 1
        assert scene_loads["scene-0.xml"] > 1
        assert scene_loads["scene-2.xml"] > 1
        assert len(prepared.owner.variant_plan.variants) == 3
    finally:
        prepared.close()


def test_worker_mesh_sources_are_self_contained(tmp_path):
    obj = tmp_path / "tetrahedron.obj"
    material = tmp_path / "tetrahedron.mtl"
    texture = tmp_path / "checker.png"
    texture.write_bytes(b"fake png bytes")
    material.write_text(
        "newmtl shape\nKd 0.2 0.6 1.0\nKa 0 0 0\nKs 0.3 0.3 0.3\n"
        f"map_Kd {texture.name}\n",
        encoding="utf-8",
    )
    obj.write_text(
        f"mtllib {material.name}\n"
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\n"
        "f 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n",
        encoding="utf-8",
    )
    sources = []
    for name, scale in (("small", ".5 .5 .5"), ("large", "1 1 1")):
        path = tmp_path / f"{name}.xml"
        path.write_text(
            '<mujoco><asset>'
            f'<mesh name="shape" file="{obj.name}" scale="{scale}"/>'
            '</asset><worldbody><body name="base"><freejoint/>'
            '<inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>'
            '<geom name="shape" type="mesh" mesh="shape" rgba="0.2 0.6 1 1"/>'
            "</body></worldbody></mujoco>",
            encoding="utf-8",
        )
        sources.append(ModelSourceDescriptor(str(path)))
    config = SceneCfg(
        entity_assets=(SceneEntitySpec("object", sources[0], kind="rigid"),),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.array([0, 1, 1]),
                tuple(sources),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )
    prepared = prepare_worker_scene(config, 3, 0.002)
    try:
        for source in prepared.payload["scene_entities"][0]["sources"]:
            xml_path = Path(source)
            meshes = ET.parse(xml_path).findall("./asset/mesh")
            assert len(meshes) == 1
            referenced = Path(meshes[0].get("file"))
            assert not referenced.is_absolute()
            assert referenced.parent == Path(".")
            copied_obj = xml_path.parent / referenced
            assert copied_obj.read_text(encoding="utf-8").startswith("mtllib entity_0_")
            copied_mtl = xml_path.parent / copied_obj.read_text(encoding="utf-8").split()[1]
            material_text = copied_mtl.read_text(encoding="utf-8")
            texture_name = next(
                line.split()[-1] for line in material_text.splitlines() if line.startswith("map_Kd")
            )
            copied_texture = xml_path.parent / texture_name
            assert copied_texture.read_bytes() == texture.read_bytes()
            assert texture_name != texture.name
            entry = prepared.payload["scene_entities"][0]
            for variant in entry["variants"]:
                np.testing.assert_allclose(variant["body_visual_rgb"], [[0.2, 0.6, 1.0]])
    finally:
        prepared.close()


def test_source_passive_damping_is_not_silently_replaced_by_drive_damping(tmp_path):
    with pytest.raises(NotImplementedError, match="passive joint damping"):
        prepare_worker_scene(scene(tmp_path, damping=0.1), 5, 0.002)


def test_self_collision_flag_reaches_worker_entity_entries(tmp_path):
    config = scene(tmp_path)
    config.entity_assets = (
        replace(config.entity_assets[0], self_collision=True),
        *config.entity_assets[1:],
    )
    prepared = prepare_worker_scene(config, 5, 0.002)
    try:
        entries = prepared.payload["scene_entities"]
        assert [entry["self_collision"] for entry in entries] == [True, False, False]
    finally:
        prepared.close()


def test_mjcf_compilation_fails_closed_on_self_collision_requests(tmp_path):
    from unisim.mjcf_compiler import compose_scene

    config = scene(tmp_path)
    config.entity_assets = (
        replace(config.entity_assets[0], self_collision=True),
        *config.entity_assets[1:],
    )
    with pytest.raises(NotImplementedError, match="self_collision"):
        compose_scene(config, 5, 0.002)


def test_gravity_disabled_request_reaches_worker_entity_entries(tmp_path):
    config = scene(tmp_path)
    config.entity_assets = (
        replace(config.entity_assets[0], gravity_disabled=True),
        *config.entity_assets[1:],
    )
    prepared = prepare_worker_scene(config, 5, 0.002)
    try:
        entries = prepared.payload["scene_entities"]
        assert [entry["gravity_disabled"] for entry in entries] == [True, None, None]
    finally:
        prepared.close()


@pytest.mark.parametrize(
    ("root_mode", "kind", "expected"),
    [
        ("fixed", "articulation", False),
        ("floating", "articulation", False),
        ("floating", "rigid", False),
        ("fixed", "rigid", True),
        ("kinematic", "rigid", True),
    ],
)
def test_isaacsim_host_resolves_implicit_gravity_default(tmp_path, root_mode, kind, expected):
    from types import SimpleNamespace

    from unisim.backend.isaacsim.backend import IsaacSimBackend

    entry = {
        "name": "entity",
        "kind": kind,
        "root_mode": root_mode,
        "gravity_disabled": None,
    }
    prepared = SimpleNamespace(payload={"scene_entities": [entry]})
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._resolve_worker_entity_gravity(prepared)
    assert entry["gravity_disabled"] is expected
    # An explicit request is never rewritten by host resolution.
    entry["gravity_disabled"] = not expected
    backend._resolve_worker_entity_gravity(prepared)
    assert entry["gravity_disabled"] is (not expected)


def test_isaacgym_host_resolves_implicit_gravity_default_to_enabled():
    from types import SimpleNamespace

    from unisim.backend.isaacgym.backend import IsaacGymBackend

    entries = [
        {"name": "fixed", "gravity_disabled": None},
        {"name": "kinematic", "gravity_disabled": None},
        {"name": "requested", "gravity_disabled": True},
    ]
    prepared = SimpleNamespace(payload={"scene_entities": entries})
    backend = IsaacGymBackend.__new__(IsaacGymBackend)
    backend._resolve_worker_entity_gravity(prepared)
    assert [entry["gravity_disabled"] for entry in entries] == [False, False, True]


@pytest.mark.parametrize("actuated", [True, False])
def test_source_joint_spring_is_not_silently_dropped_from_native_drive_table(tmp_path, actuated):
    config = scene(tmp_path)
    robot = Path(config.entity_assets[0].source.model_file)
    xml = robot.read_text().replace(
        'name="hinge" damping=', 'name="hinge" stiffness="50" springref=".3" damping='
    )
    if not actuated:
        start, end = xml.index("<actuator>"), xml.index("</actuator>") + len("</actuator>")
        xml = xml[:start] + xml[end:]
    robot.write_text(xml)
    with pytest.raises(NotImplementedError, match="passive joint springs"):
        prepare_worker_scene(config, 5, 0.002)


def test_fullinertia_source_is_rewritten_to_the_diagonal_spelling(tmp_path):
    """#278: normalization replaces fullinertia instead of mixing spellings."""
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base">'
        '<inertial pos="0 0 0" mass="2" fullinertia="1.0 1.1 0.9 0.01 0.02 0.03"/>'
        '<geom name="base_collision" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    config = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
        )
    )
    prepared = prepare_worker_scene(config, 1, 0.002)
    try:
        entries = prepared.payload["scene_entities"]
        tensor = np.array(
            [[1.0, 0.01, 0.02], [0.01, 1.1, 0.03], [0.02, 0.03, 0.9]]
        )
        # The compiler's diagonalization stores principal moments in
        # descending order.
        expected = np.linalg.eigvalsh(tensor)[::-1]
        np.testing.assert_allclose(
            entries[0]["variants"][0]["body_inertia"], [expected], rtol=0, atol=1e-12
        )
        for source in entries[0]["sources"]:
            root = ET.parse(source).getroot()
            inertial = root.find(".//inertial")
            assert inertial is not None
            assert inertial.get("fullinertia") is None
            assert inertial.get("diaginertia") is not None
    finally:
        prepared.close()
