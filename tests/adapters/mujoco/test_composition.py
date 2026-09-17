"""Real CPU compilation checks for cold-path entity composition."""

# ruff: noqa: E402

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
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
