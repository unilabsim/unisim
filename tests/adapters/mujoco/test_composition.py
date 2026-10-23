"""Real CPU compilation checks for cold-path entity composition."""

# ruff: noqa: E402

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.mjcf_compiler import compose_scene
from unisim.scene import SceneCfg


def _source(tmp_path, name, *, mass=1, passive=False, fixed=False, option="", key=""):
    joint = "" if fixed else '<freejoint name="free"/>'
    child = (
        (
            '<body name="link" pos="0 0 .3"><joint name="hinge" ref="20"/>'
            '<geom type="sphere" size=".05" mass=".2"/></body>'
        )
        if passive
        else ""
    )
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f'<mujoco>{option}<worldbody><body name="base">{joint}'
        f'<geom name="shape" type="sphere" size=".1" mass="{mass}"/>'
        f"{child}</body></worldbody>{key}</mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def test_compiled_multiroot_passive_and_independent_mirror(tmp_path):
    robot = _source(tmp_path, "robot", passive=True)
    obj = _source(tmp_path, "object", mass=2)
    table = _source(tmp_path, "table", fixed=True)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", robot),
            SceneEntitySpec(
                "object",
                obj,
                kind="rigid",
                initial_state=EntityInitialState(position=(1.0, 2.0, 3.0)),
            ),
            SceneEntitySpec("table", table, kind="rigid", root_mode="fixed"),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
                initial_state=EntityInitialState(position=(4.0, 5.0, 6.0)),
            ),
        )
    )
    with compose_scene(scene, 2, 0.002) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_file)
        assert model.nq == 15 and model.nv == 13 and model.nu == 0
        assert model.nmocap == 1
        robot_layout = composed.layout.get_entity("robot")
        assert robot_layout.joints[0].name == "hinge"
        obj_layout = composed.layout.get_entity("object")
        np.testing.assert_allclose(model.qpos0[list(obj_layout.root_qpos_indices)][:3], [1, 2, 3])
        target_id = model.body("target/base").id
        np.testing.assert_allclose(model.body_pos[target_id], [4, 5, 6])
        assert model.body_mocapid[target_id] == 0
        assert model.geom_contype[model.geom("target/shape").id] == 0
        assert model.geom_conaffinity[model.geom("target/shape").id] == 0
        # Compare source-owned nonzero hinge ref with independent source compile.
        independent = mujoco.MjModel.from_xml_path(robot.model_file)
        np.testing.assert_allclose(model.qpos0[7], independent.qpos0[7], atol=1e-5)
        path = Path(composed.model_file)
    assert not path.exists()


def test_non_round_robin_variants_match_independent_mass_and_mirror_identity(tmp_path):
    a = _source(tmp_path, "a", mass=1)
    b = _source(tmp_path, "b", mass=3)
    assignment = np.array([1, 1, 0, 1, 0])
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", a, kind="rigid"),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
        ),
        entity_variant=EntityVariantBinding("object", FixedVariantPlan(assignment, (a, b))),
    )
    with compose_scene(scene, 5, 0.003) as composed:
        np.testing.assert_array_equal(composed.variant_plan.assignment, assignment)
        for i, descriptor in enumerate(composed.variant_plan.variants):
            actual = mujoco.MjModel.from_xml_path(descriptor.model_file)
            independent = mujoco.MjModel.from_xml_path((a, b)[i].model_file)
            np.testing.assert_allclose(
                actual.body_mass[actual.body("object/base").id], independent.body_mass[1]
            )
            np.testing.assert_allclose(
                actual.body_inertia[actual.body("object/base").id], independent.body_inertia[1]
            )
            assert actual.nq == 7 and actual.nu == 0 and actual.nmocap == 1
            assert actual.opt.timestep == 0.003


def test_global_options_conflict_rejected_in_both_orders(tmp_path):
    a = SceneEntitySpec("a", _source(tmp_path, "a", option='<option gravity="0 0 -3"/>'))
    b = SceneEntitySpec("b", _source(tmp_path, "b"))
    for entities in [(a, b), (b, a)]:
        with pytest.raises(ValueError, match="global physics"):
            compose_scene(SceneCfg(entity_assets=entities), 2, 0.002)


def test_layout_change_same_width_rejected(tmp_path):
    a = _source(tmp_path, "a", passive=True)
    b = _source(tmp_path, "b", passive=True)
    path = Path(b.model_file)
    path.write_text(path.read_text().replace('name="hinge"', 'name="other"'))
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", a),),
        entity_variant=EntityVariantBinding("object", FixedVariantPlan(np.array([0, 1]), (a, b))),
    )
    with pytest.raises(ValueError):
        compose_scene(scene, 2, 0.002)


@pytest.mark.parametrize("change", ["rigid", "fixed", "fragment"])
def test_unimplemented_or_conflicting_source_semantics_fail_closed(tmp_path, change):
    source = _source(
        tmp_path,
        "object",
        passive=True,
    )
    entity = SceneEntitySpec(
        "object",
        source,
        kind="rigid" if change == "rigid" else "articulation",
        root_mode="fixed" if change == "fixed" else "floating",
    )
    scene = SceneCfg(
        entity_assets=(entity,), fragment_files=["extra.xml"] if change == "fragment" else []
    )
    with pytest.raises((ValueError, NotImplementedError)):
        compose_scene(scene, 2, 0.002)


def test_relative_mesh_survives_temp_scene_serialization(tmp_path):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetra.obj").write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    source = tmp_path / "mesh.xml"
    source.write_text(
        '<mujoco><compiler meshdir="meshes"/><asset><mesh name="mesh" file="tetra.obj"/>'
        '</asset><worldbody><body name="base"><freejoint/>'
        '<geom name="shape" type="mesh" mesh="mesh" mass="1"/></body></worldbody></mujoco>'
    )
    scene = SceneCfg(entity_assets=(SceneEntitySpec("object", ModelSourceDescriptor(str(source))),))
    with compose_scene(scene, 1, 0.002) as composed:
        loaded = mujoco.MjModel.from_xml_path(composed.model_file)
        assert loaded.nmesh == 1
        independent = mujoco.MjModel.from_xml_path(str(source))
        np.testing.assert_allclose(loaded.body_inertia[1], independent.body_inertia[1])


def test_authored_mesh_maxhullvert_survives_composition(tmp_path):
    """#439: composed sources keep the authored mesh maxhullvert.

    MuJoCo 3.11's spec writer omits the attribute; newton's convexity audit
    reads it back from an MjSpec re-parse of the composed file to decide the
    effective hull budget, so dropping it breaks the audit's escape hatch.
    """
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetra.obj").write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    source = tmp_path / "mesh.xml"
    source.write_text(
        '<mujoco><compiler meshdir="meshes"/>'
        '<asset><mesh name="mesh" file="tetra.obj" maxhullvert="4"/>'
        '<mesh name="plain" file="tetra.obj"/></asset>'
        '<worldbody><body name="base"><freejoint/>'
        '<geom name="shape" type="mesh" mesh="mesh" mass="1"/>'
        '<geom name="decoration" type="mesh" mesh="plain" contype="0" conaffinity="0"/>'
        "</body></worldbody></mujoco>"
    )
    scene = SceneCfg(entity_assets=(SceneEntitySpec("object", ModelSourceDescriptor(str(source))),))
    with compose_scene(scene, 1, 0.002) as composed:
        xml = Path(composed.model_file).read_text()
        assert 'maxhullvert="4"' in xml
        loaded = mujoco.MjModel.from_xml_path(composed.model_file)
        assert loaded.nmesh == 2
        spec = mujoco.MjSpec.from_file(composed.model_file)
        budgets = {mesh.name: int(mesh.maxhullvert) for mesh in spec.meshes}
        assert budgets["object/mesh"] == 4
        assert budgets["object/plain"] == -1  # unauthored meshes keep the default


def test_articulated_mirror_has_no_hidden_controls_or_dynamics(tmp_path):
    source = _source(tmp_path, "robot", passive=True)
    path = Path(source.model_file)
    path.write_text(
        path.read_text().replace(
            "</mujoco>", '<actuator><motor name="drive" joint="hinge"/></actuator></mujoco>'
        )
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", source),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="robot",
            ),
        )
    )
    with compose_scene(scene, 2, 0.002) as composed:
        assert composed.model.nv == 7 and composed.model.nu == 1
        assert composed.layout.get_entity("target").joints == ()
        assert composed.layout.get_entity("target").actuator_names == ()
        assert composed.model.nmocap == 1


def test_nondefault_mass_compiler_is_not_silently_discarded(tmp_path):
    source = _source(tmp_path, "object")
    path = Path(source.model_file)
    path.write_text(path.read_text().replace("<mujoco>", '<mujoco><compiler settotalmass="7"/>'))
    with pytest.raises(NotImplementedError, match="settotalmass"):
        compose_scene(SceneCfg(entity_assets=(SceneEntitySpec("object", source),)), 1, 0.002)


def test_variant_cannot_change_bootstrap_sensor_layout(tmp_path):
    a = _source(tmp_path, "a")
    b = _source(tmp_path, "b")
    path = Path(b.model_file)
    path.write_text(
        path.read_text().replace(
            "</mujoco>",
            '<sensor><framepos name="position" objtype="body" objname="base"/></sensor></mujoco>',
        )
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", a),),
        entity_variant=EntityVariantBinding("object", FixedVariantPlan(np.array([0]), (b,))),
    )
    with pytest.raises(ValueError, match="sensor layout"):
        compose_scene(scene, 1, 0.002)


def test_failed_composition_cleans_temporary_sources(tmp_path, monkeypatch):
    import unisim.mjcf_compiler as module

    original = module.tempfile.TemporaryDirectory
    directories = []

    def track_directory(*args, **kwargs):
        result = original(*args, **kwargs)
        directories.append(Path(result.name))
        return result

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", track_directory)
    source = _source(tmp_path, "object", passive=True)
    with pytest.raises(ValueError, match="non-root joints"):
        compose_scene(
            SceneCfg(entity_assets=(SceneEntitySpec("object", source, kind="rigid"),)), 1, 0.002
        )
    assert directories and all(not path.exists() for path in directories)


def _keyed_source(
    tmp_path,
    name,
    *,
    position=0.4,
    velocity=0.7,
    ctrl=1.2,
    act=0.8,
    key_name="home",
    time=2.0,
    ball=False,
):
    source = _source(tmp_path, name, passive=True)
    path = Path(source.model_file)
    text = path.read_text()
    if ball:
        text = text.replace('<joint name="hinge" ref="20"/>', '<joint name="hinge" type="ball"/>')
    qpos = f"9 8 7 1 0 0 0 {position}" if not ball else "9 8 7 1 0 0 0 .5 .5 .5 .5"
    qvel = f"1 2 3 4 5 6 {velocity}" if not ball else "1 2 3 4 5 6 .3 .4 .5"
    key = (
        (
            f'<keyframe><key name="{key_name}" time="{time}" qpos="{qpos}" '
            f'qvel="{qvel}" ctrl="{ctrl}" act="{act}"/></keyframe>'
        )
        if key_name
        else ""
    )
    text = text.replace(
        "</mujoco>",
        '<actuator><general name="drive" joint="hinge" dyntype="filter" '
        'dynprm=".1"/></actuator>' + key + "</mujoco>",
    )
    path.write_text(text)
    return source


def test_named_keys_merge_compiled_joint_control_activation_and_independent_roots(tmp_path):
    a = _keyed_source(tmp_path, "a", ball=True)
    b = _keyed_source(tmp_path, "b", position=0.9, velocity=-0.2, ctrl=-2.0, act=0.3)
    absent = _keyed_source(tmp_path, "absent", key_name=None)
    scene = SceneCfg(
        default_keyframe_name="home",
        entity_assets=(
            SceneEntitySpec("b", b, initial_state=EntityInitialState(position=(1.0, 2.0, 3.0))),
            SceneEntitySpec("absent", absent),
            SceneEntitySpec("a", a),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="a",
                initial_state=EntityInitialState(position=(4.0, 5.0, 6.0)),
            ),
        ),
    )
    with compose_scene(scene, 2, 0.002) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_file)
        key_id = model.key("home").id
        assert model.nkey == 1 and model.key_time[key_id] == 2
        assert model.nu == 3 and model.na == 3
        for name, descriptor in [("a", a), ("b", b), ("absent", absent)]:
            source = mujoco.MjModel.from_xml_path(descriptor.model_file)
            entity = composed.layout.get_entity(name)
            joint = entity.joints[0]
            has_key = name != "absent"
            np.testing.assert_allclose(
                model.key_qpos[key_id, list(joint.qpos_indices)],
                source.key_qpos[0, 7:] if has_key else source.qpos0[7:],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                model.key_qvel[key_id, list(joint.qvel_indices)],
                source.key_qvel[0, 6:] if has_key else 0,
            )
            aid = entity.actuator_indices[0]
            np.testing.assert_allclose(
                model.key_ctrl[key_id, aid], source.key_ctrl[0, 0] if has_key else 0
            )
            np.testing.assert_allclose(
                model.key_act[key_id, model.actuator_actadr[aid]],
                source.key_act[0, 0] if has_key else 0,
            )
            np.testing.assert_array_equal(model.key_qvel[key_id, list(entity.root_qvel_indices)], 0)
            np.testing.assert_allclose(
                model.key_qpos[key_id, list(entity.root_qpos_indices)],
                model.qpos0[list(entity.root_qpos_indices)],
            )
        np.testing.assert_allclose(model.key_mpos[key_id].reshape(-1, 3), [[4, 5, 6]])
        np.testing.assert_allclose(model.key_mquat[key_id].reshape(-1, 4), [[1, 0, 0, 0]])
        assert composed.layout.get_entity("target").joints == ()


def test_key_union_missing_entity_uses_its_default(tmp_path):
    a = _keyed_source(tmp_path, "a", key_name="first")
    b = _keyed_source(tmp_path, "b", key_name="second")
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("a", a), SceneEntitySpec("b", b)),
        default_keyframe_name="second",
    )
    with compose_scene(scene, 1, 0.002) as composed:
        assert tuple(composed.model.key(i).name for i in range(composed.model.nkey)) == (
            "first",
            "second",
        )
        target = composed.layout.get_entity("a").joints[0].qpos_indices
        np.testing.assert_allclose(
            composed.model.key_qpos[1, list(target)], composed.model.qpos0[list(target)]
        )


@pytest.mark.parametrize("problem", ["missing_default", "conflicting_time", "unnamed", "width"])
def test_key_merge_rejects_ambiguous_or_invalid_source_values(tmp_path, problem):
    a = _keyed_source(tmp_path, "a")
    b = _keyed_source(tmp_path, "b", time=3 if problem == "conflicting_time" else 2)
    path = Path(b.model_file)
    if problem == "unnamed":
        path.write_text(path.read_text().replace('name="home"', ""))
    if problem == "width":
        path.write_text(path.read_text().replace('act="0.8"', 'act=".8 .9"'))
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("a", a), SceneEntitySpec("b", b)),
        default_keyframe_name="absent" if problem == "missing_default" else "home",
    )
    with pytest.raises(ValueError):
        compose_scene(scene, 2, 0.002)


def test_variant_keys_preserve_per_variant_joint_defaults_and_selected_identity(tmp_path):
    a = _keyed_source(tmp_path, "a", position=0.2)
    b = _keyed_source(tmp_path, "b", position=0.6)
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", a),),
        default_keyframe_name="home",
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0]), (a, b))
        ),
    )
    with compose_scene(scene, 3, 0.002) as composed:
        np.testing.assert_array_equal(composed.variant_plan.assignment, [1, 1, 0])
        for descriptor, expected in zip(composed.variant_plan.variants, [0.2, 0.6], strict=True):
            model = mujoco.MjModel.from_xml_path(descriptor.model_file)
            assert model.key_qpos[model.key("home").id, 7] == expected


TETRAHEDRON = """v 0 0 0
v .1 0 0
v 0 .1 0
v 0 0 .1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _mesh_entity_source(
    tmp_path,
    name,
    *,
    include_head=True,
    head_scale="1 1 1",
    include_handle=True,
    extra_mesh=False,
):
    obj = tmp_path / "tetrahedron.obj"
    obj.write_text(TETRAHEDRON)
    meshes = ""
    geoms = ""
    if include_handle:
        geoms += '<geom name="handle" type="sphere" size=".02" mass="0"/>'
    if include_head:
        meshes += f'<mesh name="head" file="{obj}" scale="{head_scale}"/>'
        geoms += '<geom name="head" type="mesh" mesh="head" mass="0"/>'
    if extra_mesh:
        meshes += f'<mesh name="extra" file="{obj}"/>'
        geoms += '<geom name="extra" type="mesh" mesh="extra" mass="0"/>'
    path = tmp_path / f"{name}.xml"
    path.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/>'
        '<inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>'
        f"{geoms}</body></worldbody><asset>{meshes}</asset></mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def _uniform_scene(tmp_path, **source_kwargs):
    base = _mesh_entity_source(tmp_path, "base", **source_kwargs)
    missing = _mesh_entity_source(tmp_path, "missing", include_head=False)
    present = _mesh_entity_source(tmp_path, "present", head_scale=".5 .5 .5")
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", base, kind="rigid"),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.array([1, 0]),
                (missing, present),
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )


def test_uniform_entity_mesh_variants_compose_canonical_union_layout(tmp_path):
    with compose_scene(_uniform_scene(tmp_path), 2, 0.002) as composed:
        assert composed.variant_plan.layout is FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT
        assert composed.model_file == composed.variant_plan.variants[1].model_file
        assert [geom.name for geom in composed.layout.get_entity("object").geoms] == [
            "handle",
            "head",
        ]
        assert composed.model.ngeom == 4
        missing = mujoco.MjModel.from_xml_path(composed.variant_plan.variants[0].model_file)
        present = mujoco.MjModel.from_xml_path(composed.variant_plan.variants[1].model_file)
        assert mujoco.mj_name2id(missing, mujoco.mjtObj.mjOBJ_GEOM, "object/head") == -1
        assert present.geom("object/head").id >= 0
        assert mujoco.mj_name2id(missing, mujoco.mjtObj.mjOBJ_GEOM, "mirror/head") == -1
        assert present.geom("mirror/head").id >= 0


@pytest.mark.parametrize("problem", ["non_mesh", "base_extra"])
def test_uniform_entity_variant_layouts_fail_closed(tmp_path, problem):
    if problem == "non_mesh":
        base = _mesh_entity_source(tmp_path, "base")
        full = _mesh_entity_source(tmp_path, "full")
        no_handle = _mesh_entity_source(tmp_path, "no-handle", include_handle=False)
        scene = SceneCfg(
            entity_assets=(SceneEntitySpec("object", base, kind="rigid"),),
            entity_variant=EntityVariantBinding(
                "object",
                FixedVariantPlan(
                    np.array([0, 1]),
                    (full, no_handle),
                    layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
                ),
            ),
        )
    else:
        scene = _uniform_scene(tmp_path, extra_mesh=True)
    with pytest.raises(ValueError, match="optional mesh-geom slots|catalog union"):
        compose_scene(scene, 2, 0.002)


def _uniform_scene_many(tmp_path, count):
    base = _mesh_entity_source(tmp_path, "base")
    variants = tuple(
        _mesh_entity_source(
            tmp_path,
            f"variant-{index}",
            include_head=index % 3 != 0,
            head_scale=f"{0.5 + index * 0.01:.2f} .5 .5",
        )
        for index in range(count)
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", base, kind="rigid"),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object",
            FixedVariantPlan(
                np.arange(count),
                variants,
                layout=FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
            ),
        ),
    )


def _compose_fingerprint(composed):
    assert composed.variant_plan is not None
    files = tuple(
        Path(descriptor.model_file).read_bytes() for descriptor in composed.variant_plan.variants
    )
    assert composed.variant_initial_states is not None
    states = tuple(
        (
            state.qpos.tobytes(),
            state.qvel.tobytes(),
            state.entity_rows.tobytes(),
            state.ctrl.tobytes(),
            state.ctrl_lower.tobytes(),
            state.ctrl_upper.tobytes(),
        )
        for _, state in sorted(composed.variant_initial_states.items())
    )
    return (
        Path(composed.model_file).read_bytes(),
        files,
        composed.layout,
        states,
        composed.source_provenance,
        composed.content_identity,
    )


def test_parallel_uniform_compose_matches_sequential(tmp_path, monkeypatch):
    scene = _uniform_scene_many(tmp_path, 16)
    monkeypatch.setenv("UNISIM_COMPOSE_WORKERS", "1")
    with compose_scene(scene, 16, 0.002) as sequential:
        expected = _compose_fingerprint(sequential)
    monkeypatch.setenv("UNISIM_COMPOSE_WORKERS", "4")
    with compose_scene(scene, 16, 0.002) as parallel:
        actual = _compose_fingerprint(parallel)
    assert actual == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_compose_workers_env_must_be_positive_integer(tmp_path, monkeypatch, value):
    monkeypatch.setenv("UNISIM_COMPOSE_WORKERS", value)
    with pytest.raises(ValueError, match="UNISIM_COMPOSE_WORKERS"):
        compose_scene(_uniform_scene_many(tmp_path, 16), 16, 0.002)


def _contact_fixture_sources(tmp_path):
    """Mask-permitted interpenetrating entities on primitive geometry.

    The fixed hand's horizontal finger capsule overlaps the floating support
    cylinder (the load-bearing excludable pair) and carries the floating
    object sphere resting on top (the mask-permitted pair that must keep
    colliding).  All geometry is primitive; no mesh assets are involved.
    """
    hand = tmp_path / "hand.xml"
    hand.write_text(
        '<mujoco><worldbody><body name="base">'
        '<body name="finger" pos="0 0 0.1" quat="0.7071068 0 0.7071068 0">'
        '<geom name="finger_geom" type="capsule" size="0.01 0.05" mass="0.01"/>'
        "</body></body></worldbody></mujoco>"
    )
    support = tmp_path / "support.xml"
    support.write_text(
        '<mujoco><worldbody><body name="base"><freejoint name="free"/>'
        '<geom name="post" type="cylinder" size="0.02 0.1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    obj = tmp_path / "object.xml"
    obj.write_text(
        '<mujoco><worldbody><body name="base"><freejoint name="free"/>'
        '<geom name="ball" type="sphere" size="0.03" mass="0.05"/>'
        "</body></worldbody></mujoco>"
    )
    return (
        SceneEntitySpec("hand", ModelSourceDescriptor(str(hand)), root_mode="fixed"),
        SceneEntitySpec(
            "support",
            ModelSourceDescriptor(str(support)),
            kind="rigid",
            initial_state=EntityInitialState(position=(0.04, 0.0, 0.1)),
        ),
        SceneEntitySpec(
            "object",
            ModelSourceDescriptor(str(obj)),
            kind="rigid",
            initial_state=EntityInitialState(position=(-0.04, 0.0, 0.138)),
        ),
    )


def _exclude_rows(model):
    return {
        frozenset(
            (
                model.body(int(signature) >> 16).name,
                model.body(int(signature) & 0xFFFF).name,
            )
        )
        for signature in model.exclude_signature
    }


def _contact_pair_steps(model, steps=50):
    """Count excluded-vs-permitted geom-pair activations across one rollout."""
    data = mujoco.MjData(model)
    finger = model.geom("hand/finger_geom").id
    post = model.geom("support/post").id
    ball = model.geom("object/ball").id
    excluded_hits = permitted_hits = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        pairs = {
            frozenset((int(data.contact.geom1[index]), int(data.contact.geom2[index])))
            for index in range(data.ncon)
        }
        excluded_hits += frozenset((finger, post)) in pairs
        permitted_hits += frozenset((finger, ball)) in pairs
    return excluded_hits, permitted_hits


def test_scene_contact_excludes_compile_and_gate_contacts(tmp_path):
    entities = _contact_fixture_sources(tmp_path)
    declared = (("hand/finger", "support/base"),)
    with compose_scene(
        SceneCfg(entity_assets=entities, contact_excludes=declared), 1, 0.002
    ) as composed:
        assert _exclude_rows(composed.model) == {frozenset(pair) for pair in declared}
        excluded_hits, permitted_hits = _contact_pair_steps(composed.model)
        # The excluded cross-entity pair never activates across the rollout,
        # while the unlisted mask-permitted pair keeps colliding.
        assert excluded_hits == 0
        assert permitted_hits > 25
    with compose_scene(SceneCfg(entity_assets=entities), 1, 0.002) as control:
        # Masks alone permit the pair: without the scene-level exclusion the
        # same interpenetration collides, so the declared row is load-bearing.
        control_hits, _ = _contact_pair_steps(control.model)
        assert control.model.nexclude == 0
        assert control_hits > 0


def test_scene_contact_excludes_fail_closed_on_unknown_body(tmp_path):
    entities = _contact_fixture_sources(tmp_path)
    scene = SceneCfg(
        entity_assets=entities,
        contact_excludes=(("hand/finger", "support/missing"),),
    )
    with pytest.raises(ValueError, match="contact_excludes.*unknown body 'support/missing'"):
        compose_scene(scene, 1, 0.002)


def test_scene_contact_excludes_apply_to_every_variant_realization(tmp_path):
    hand, support, _ = _contact_fixture_sources(tmp_path)
    variant_a = tmp_path / "support_a.xml"
    variant_b = tmp_path / "support_b.xml"
    template = (
        '<mujoco><worldbody><body name="base"><freejoint name="free"/>'
        '<geom name="post" type="cylinder" size="0.02 0.1" mass="{mass}"/>'
        "</body></worldbody></mujoco>"
    )
    variant_a.write_text(template.format(mass=1))
    variant_b.write_text(template.format(mass=2))
    declared = (("hand/finger", "support/base"),)
    scene = SceneCfg(
        entity_assets=(hand, support),
        entity_variant=EntityVariantBinding(
            "support",
            FixedVariantPlan(
                np.array([0, 1]),
                (
                    ModelSourceDescriptor(str(variant_a)),
                    ModelSourceDescriptor(str(variant_b)),
                ),
            ),
        ),
        contact_excludes=declared,
    )
    with compose_scene(scene, 2, 0.002) as composed:
        assert composed.variant_plan is not None
        for descriptor in composed.variant_plan.variants:
            variant_model = mujoco.MjModel.from_xml_path(descriptor.model_file)
            assert _exclude_rows(variant_model) == {frozenset(declared[0])}


def test_scene_contact_excludes_participate_in_content_identity(tmp_path):
    entities = _contact_fixture_sources(tmp_path)
    declared = (("hand/finger", "support/base"),)
    with compose_scene(SceneCfg(entity_assets=entities), 1, 0.002) as plain:
        plain_identity = plain.content_identity.canonical_identity
    with compose_scene(
        SceneCfg(entity_assets=entities, contact_excludes=declared), 1, 0.002
    ) as excluded:
        assert excluded.content_identity.canonical_identity != plain_identity
        field = next(
            item for item in excluded.intent_report.fields if item.field == "scene.contact_excludes"
        )
        assert field.requested == declared
