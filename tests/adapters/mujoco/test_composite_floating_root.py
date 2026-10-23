"""Composite three-slide-plus-ball floating base composition acceptance (#435)."""

import numpy as np
import pytest

from unisim import create_backend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.entity_state import entity_state_snapshot, prepare_scene_reset
from unisim.mjcf_compiler import compose_scene
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")
mjbatch = pytest.importorskip("mjbatch")


def _composite_file(tmp_path, name="hand"):
    """CHORD-style floating hand: 3 canonical slides + terminal ball, identity frames."""
    path = tmp_path / f"{name}.xml"
    path.write_text(
        "<mujoco><worldbody>"
        '<body name="mount_x">'
        '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
        '<joint name="px" type="slide" axis="1 0 0" armature="0.1" frictionloss="0.2" '
        'damping="0.3" range="-1 1"/>'
        '<body name="mount_y">'
        '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
        '<joint name="py" type="slide" axis="0 1 0" armature="0.11" frictionloss="0.21" '
        'damping="0.31" range="-2 2"/>'
        '<body name="mount_z">'
        '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
        '<joint name="pz" type="slide" axis="0 0 1" armature="0.12" frictionloss="0.22" '
        'damping="0.32" range="-3 3"/>'
        '<body name="palm">'
        '<inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>'
        '<joint name="palm_ball" type="ball" armature="0.13" frictionloss="0.23" '
        'damping="0.33" range="0 0.5"/>'
        '<geom type="sphere" size=".05" mass=".3"/>'
        '<body name="finger" pos="0 0 .1">'
        '<inertial mass=".05" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>'
        '<joint name="finger_hinge" type="hinge" axis="0 1 0" range="-1.5 1.5"/>'
        '<geom type="capsule" size=".01 .05" mass=".05"/>'
        "</body></body></body></body></body></worldbody>"
        "<actuator>"
        '<position name="px_act" joint="px" kp="10"/>'
        '<position name="py_act" joint="py" kp="10"/>'
        '<position name="pz_act" joint="pz" kp="10"/>'
        '<position name="finger_act" joint="finger_hinge" kp="5"/>'
        "</actuator></mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def _free_file(tmp_path, name="hand"):
    """The equivalent single-free-joint writing of the same floating hand."""
    path = tmp_path / f"{name}.xml"
    path.write_text(
        "<mujoco><worldbody>"
        '<body name="palm">'
        '<inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>'
        '<freejoint name="free"/>'
        '<geom type="sphere" size=".05" mass=".3"/>'
        '<body name="finger" pos="0 0 .1">'
        '<inertial mass=".05" pos="0 0 0" diaginertia="1e-5 1e-5 1e-5"/>'
        '<joint name="finger_hinge" type="hinge" axis="0 1 0" range="-1.5 1.5"/>'
        '<geom type="capsule" size=".01 .05" mass=".05"/>'
        "</body></body></worldbody>"
        "<actuator>"
        '<position name="finger_act" joint="finger_hinge" kp="5"/>'
        "</actuator></mujoco>"
    )
    return ModelSourceDescriptor(str(path))


_SLIDE_X = '<joint name="j0" type="slide" axis="1 0 0"/>'
_SLIDE_Y = '<joint name="j1" type="slide" axis="0 1 0"/>'
_SLIDE_Z = '<joint name="j2" type="slide" axis="0 0 1"/>'
_BALL = '<joint name="j3" type="ball"/>'


def _chain_file(tmp_path, name, joints, *, frames=("", "", "", ""), branch=False):
    """Write a four-body base chain with one joint XML fragment per body."""
    mount_inertial = '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
    twig = (
        '<body name="twig"><inertial mass=".01" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
        '<geom type="sphere" size=".01" mass=".01"/></body>'
    )
    path = tmp_path / f"{name}.xml"
    path.write_text(
        "<mujoco><worldbody>"
        f'<body name="mount_x"{frames[0]}>{mount_inertial}{joints[0]}'
        f'<body name="mount_y"{frames[1]}>{mount_inertial}{joints[1]}'
        f'<body name="mount_z"{frames[2]}>{mount_inertial}{joints[2]}'
        f'<body name="palm"{frames[3]}>'
        '<inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>'
        f'{joints[3]}<geom type="sphere" size=".05" mass=".3"/>'
        f"</body>{twig if branch else ''}"
        "</body></body></body></worldbody></mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def _scene(source, **state):
    return SceneCfg(
        entity_assets=(SceneEntitySpec("hand", source, initial_state=EntityInitialState(**state)),)
    )


def _dof_range(model, joint_id):
    start = int(model.jnt_dofadr[joint_id])
    end = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else int(model.nv)
    return start, end


def test_composite_base_composes_and_preserves_source_joint_fields(tmp_path):
    source = _composite_file(tmp_path)
    scene = _scene(source, position=(1.0, 2.0, 3.0))
    with compose_scene(scene, 1, 0.002) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_file)
        layout = composed.layout.get_entity("hand")
        assert tuple(joint.name for joint in layout.root_joints) == ("px", "py", "pz", "palm_ball")
        assert tuple(joint.kind for joint in layout.root_joints) == (
            "slide",
            "slide",
            "slide",
            "ball",
        )
        assert tuple(joint.name for joint in layout.joints) == ("finger_hinge",)
        assert layout.root_qpos_indices == tuple(range(7))
        assert layout.root_qvel_indices == tuple(range(6))
        # Composite slide references stay source-owned: qpos0 carries the
        # source reference configuration, and the declared initial position
        # locates only the mount chain origin frame.
        np.testing.assert_allclose(
            model.qpos0[list(layout.root_qpos_indices)], [0, 0, 0, 1, 0, 0, 0]
        )
        np.testing.assert_allclose(model.body_pos[model.body("hand/mount_x").id], [1, 2, 3])
        # The three slide position actuators still drive the base joints.
        assert layout.actuator_joint_names == ("px", "py", "pz", "finger_hinge")
        driven = {
            model.joint(int(model.actuator_trnid[aid, 0])).name for aid in range(model.nu)
        }
        assert {"hand/px", "hand/py", "hand/pz"} <= driven
        # Every base and articulation joint field matches an independent compile
        # of the unchanged source (float32 XML serialization tolerance).
        independent = mujoco.MjModel.from_xml_path(source.model_file)
        for name in ("px", "py", "pz", "palm_ball", "finger_hinge"):
            composed_joint = model.joint(f"hand/{name}").id
            source_joint = independent.joint(name).id
            c_start, c_end = _dof_range(model, composed_joint)
            s_start, s_end = _dof_range(independent, source_joint)
            assert c_end - c_start == s_end - s_start
            for field in ("dof_armature", "dof_frictionloss", "dof_damping"):
                np.testing.assert_allclose(
                    getattr(model, field)[c_start:c_end],
                    getattr(independent, field)[s_start:s_end],
                )
            np.testing.assert_allclose(
                model.jnt_range[composed_joint], independent.jnt_range[source_joint], atol=1e-6
            )


@pytest.mark.parametrize(
    ("joints", "frames", "branch", "error"),
    [
        # Only two slides ahead of the ball.
        ((_SLIDE_X, _SLIDE_Y, _BALL, ""), None, False, "exactly one slide joint"),
        # A fourth slide where the terminal ball belongs.
        (
            (_SLIDE_X, _SLIDE_Y, _SLIDE_Z, '<joint name="j3" type="slide" axis="1 0 0"/>'),
            None,
            False,
            "exactly one ball joint",
        ),
        # No terminal ball at all.
        ((_SLIDE_X, _SLIDE_Y, _SLIDE_Z, ""), None, False, "exactly one ball joint"),
        # Non-orthogonal slide axis.
        (
            ('<joint name="j0" type="slide" axis="1 1 0"/>', _SLIDE_Y, _SLIDE_Z, _BALL),
            None,
            False,
            "non-canonical axis",
        ),
        # Canonical axes in a non-canonical order.
        (
            (
                '<joint name="j0" type="slide" axis="0 0 1"/>',
                '<joint name="j1" type="slide" axis="0 1 0"/>',
                '<joint name="j2" type="slide" axis="1 0 0"/>',
                _BALL,
            ),
            None,
            False,
            "non-canonical axis",
        ),
        # A hinge inside the base chain.
        (
            (_SLIDE_X, '<joint name="j1" type="hinge" axis="0 1 0"/>', _SLIDE_Z, _BALL),
            None,
            False,
            "exactly one slide joint",
        ),
        # A free joint inside the base chain.
        (
            (_SLIDE_X, '<freejoint name="jf"/>', _SLIDE_Z, _BALL),
            None,
            False,
            "non-root or multiple free joints",
        ),
        # The ball joint is not terminal.
        (
            (_SLIDE_X, _SLIDE_Y, _BALL, '<joint name="j4" type="slide" axis="1 0 0"/>'),
            None,
            False,
            "exactly one slide joint",
        ),
        # A branching base chain.
        ((_SLIDE_X, _SLIDE_Y, _SLIDE_Z, _BALL), None, True, "exactly one child body"),
        # A non-identity intermediate mount frame.
        (
            (_SLIDE_X, _SLIDE_Y, _SLIDE_Z, _BALL),
            ("", ' pos="0 0 .1"', "", ""),
            False,
            "non-identity frame",
        ),
        # A non-identity terminal palm frame.
        (
            (_SLIDE_X, _SLIDE_Y, _SLIDE_Z, _BALL),
            ("", "", "", ' quat="0.7071068 0.7071068 0 0"'),
            False,
            "non-identity frame",
        ),
    ],
)
def test_malformed_composite_bases_fail_closed(tmp_path, joints, frames, branch, error):
    kwargs = {} if frames is None else {"frames": frames}
    source = _chain_file(tmp_path, "hand", joints, branch=branch, **kwargs)
    with pytest.raises(ValueError, match=error):
        compose_scene(_scene(source), 1, 0.002)


def test_composite_base_rejects_a_rotated_initial_quaternion(tmp_path):
    scene = _scene(_composite_file(tmp_path), quaternion=(0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="identity orientation"):
        compose_scene(scene, 1, 0.002)


def test_root_state_layout_matches_the_equivalent_free_joint(tmp_path):
    state = {"position": (1.0, 2.0, 3.0)}
    with compose_scene(_scene(_composite_file(tmp_path, "comp"), **state), 1, 0.002) as composed:
        composite = composed.layout
    with compose_scene(_scene(_free_file(tmp_path, "free"), **state), 1, 0.002) as composed:
        free = composed.layout
    assert composite.nq == free.nq == 8
    assert composite.nv == free.nv == 7
    hand_c = composite.get_entity("hand")
    hand_f = free.get_entity("hand")
    assert hand_c.root_qpos_indices == hand_f.root_qpos_indices == tuple(range(7))
    assert hand_c.root_qvel_indices == hand_f.root_qvel_indices == tuple(range(6))
    assert hand_c.joints[0].qpos_indices == hand_f.joints[0].qpos_indices == (7,)
    # Identical generalized columns produce identical world-frame snapshots.
    qpos = np.array([[0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.7]])
    qvel = np.array([[1.0, 2.0, 3.0, 0.3, -0.2, 0.1, 0.4]])
    snapshot_c = entity_state_snapshot(hand_c, qpos, qvel)
    snapshot_f = entity_state_snapshot(hand_f, qpos, qvel)
    for field in snapshot_c:
        np.testing.assert_array_equal(snapshot_c[field], snapshot_f[field])
    # Identical reset requests produce identical generalized writes.
    pose = np.array([[0.4, -0.5, 0.6, 0.5, -0.5, 0.5, -0.5]])
    velocity = np.array([[0.2, 0.3, 0.4, 0.7, -0.8, 0.9]])
    request = SceneResetRequest(
        (0,),
        (
            EntityStatePatch(
                "hand",
                root_pose=pose,
                root_velocity=velocity,
                joint_positions=np.array([[0.25]]),
            ),
        ),
    )
    roots = np.zeros((1, 1, 13))
    plan_c = prepare_scene_reset(composite, request, qpos.copy(), qvel.copy(), roots.copy())
    plan_f = prepare_scene_reset(free, request, qpos.copy(), qvel.copy(), roots.copy())
    np.testing.assert_array_equal(plan_c.qpos, plan_f.qpos)
    np.testing.assert_array_equal(plan_c.qvel, plan_f.qvel)
    np.testing.assert_array_equal(plan_c.qpos_mask, plan_f.qpos_mask)
    np.testing.assert_array_equal(plan_c.qvel_mask, plan_f.qvel_mask)
    np.testing.assert_array_equal(plan_c.roots, plan_f.roots)


def test_backend_root_state_round_trip_matches_free_joint(tmp_path):
    state = {"position": (1.0, 2.0, 3.0)}
    backends = [
        create_backend(
            "mujoco",
            _scene(source, **state),
            num_envs=2,
            sim_dt=0.002,
            np_dtype=np.float64,
        )
        for source in (_composite_file(tmp_path, "comp"), _free_file(tmp_path, "free"))
    ]
    composite, free = backends
    try:
        for backend in backends:
            backend.materialize()
        layout = composite.get_scene_layout().get_entity("hand")
        assert tuple(joint.name for joint in layout.root_joints) == ("px", "py", "pz", "palm_ball")
        root_layout = composite.get_root_state_layout("hand/mount_x")
        assert root_layout.qpos_indices == tuple(range(7))
        assert root_layout.qvel_indices == tuple(range(6))
        # The composite default root state is source-owned (the slide
        # references), with the declared position locating only the mount
        # chain origin; the free-joint default keeps the declared pose.
        composite_default = np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1))
        free_default = np.tile([1, 2, 3, 1, 0, 0, 0], (2, 1))
        np.testing.assert_allclose(
            composite.get_entity_state("hand")["root_pose"], composite_default
        )
        np.testing.assert_allclose(free.get_entity_state("hand")["root_pose"], free_default)
        # Rotated poses and world-frame velocities round-trip through the
        # slide+ball columns exactly like a free joint.
        pose = np.array(
            [
                [0.4, -0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                [-0.2, 0.1, 0.9, 0.5, -0.5, 0.5, -0.5],
            ]
        )
        velocity = np.array(
            [
                [0.2, 0.3, 0.4, 0.7, -0.8, 0.9],
                [0.1, -0.1, 0.2, -0.3, 0.4, -0.5],
            ]
        )
        request = SceneResetRequest(
            (0, 1), (EntityStatePatch("hand", root_pose=pose, root_velocity=velocity),)
        )
        for backend in backends:
            backend.reset_entities(request)
        actual_c = composite.get_entity_state("hand")
        actual_f = free.get_entity_state("hand")
        for field in ("root_pose", "root_velocity"):
            np.testing.assert_allclose(actual_c[field], pose if field == "root_pose" else velocity)
            np.testing.assert_array_equal(actual_c[field], actual_f[field])
        # The committed generalized columns are bitwise identical to the
        # equivalent free-joint model.
        np.testing.assert_array_equal(composite._qpos_view, free._qpos_view)
        np.testing.assert_array_equal(composite._qvel_view, free._qvel_view)
        # Independent native readback verifies the palm's world velocity.
        model = composite.get_playback_model(0)
        data = mujoco.MjData(model)
        data.qpos[:] = composite._qpos_view[0]
        data.qvel[:] = composite._qvel_view[0]
        mujoco.mj_forward(model, data)
        native = np.zeros(6)
        palm = model.body("hand/palm").id
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, palm, native, 0)
        np.testing.assert_allclose(native[:3], velocity[0, 3:], atol=1e-12)
        np.testing.assert_allclose(native[3:], velocity[0, :3], atol=1e-12)
        # The captured defaults carry the source-owned composite columns and
        # the declared free-joint pose.
        np.testing.assert_allclose(
            composite.get_entity_default_state("hand")["root_pose"], composite_default
        )
        np.testing.assert_allclose(
            free.get_entity_default_state("hand")["root_pose"], free_default
        )
        # The import report treats the base chain as root state: requested and
        # effective articulation joint defaults contain only the finger hinge.
        report = composite.get_import_report()
        defaults_field = next(
            field for field in report.fields if field.field == "entity.initial_defaults"
        )
        assert tuple(defaults_field.requested["joint_positions"]) == (0.0,)
        assert (
            defaults_field.requested["joint_positions"]
            == defaults_field.effective["joint_positions"]
        )
    finally:
        for backend in backends:
            backend.close()


def _keyed_chain_file(tmp_path, key_qpos, key_qvel, *, root="chain", name="hand"):
    """Write a minimal composite-base source with one ``home`` keyframe."""
    path = tmp_path / f"{name}.xml"
    if root == "chain":
        body = (
            '<body name="mount_x">'
            '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
            '<joint name="px" type="slide" axis="1 0 0"/>'
            '<body name="mount_y">'
            '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
            '<joint name="py" type="slide" axis="0 1 0"/>'
            '<body name="mount_z">'
            '<inertial mass="1e-4" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>'
            '<joint name="pz" type="slide" axis="0 0 1"/>'
            '<body name="palm">'
            '<inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>'
            '<joint name="palm_ball" type="ball"/>'
            '<geom type="sphere" size=".05" mass=".3"/>'
            "</body></body></body></body>"
        )
    else:
        body = (
            '<body name="palm">'
            '<inertial mass=".3" pos="0 0 0" diaginertia="1e-4 1e-4 1e-4"/>'
            '<freejoint name="free"/>'
            '<geom type="sphere" size=".05" mass=".3"/>'
            "</body>"
        )
    path.write_text(
        f"<mujoco><worldbody>{body}</worldbody>"
        "<keyframe>"
        f'<key name="home" qpos="{key_qpos}" qvel="{key_qvel}"/>'
        "</keyframe></mujoco>"
    )
    return ModelSourceDescriptor(str(path))


def test_merge_keys_carries_composite_root_columns(tmp_path):
    """Merged keys carry composite root columns bit-exactly; FK keeps source semantics."""
    # Every value is a dyadic fraction so the serialized XML round trip is exact.
    source = _keyed_chain_file(
        tmp_path,
        "-0.25 0.25 0.5 0.5 0.5 0.5 0.5",
        "0.5 -0.25 0.75 0.25 -0.5 1",
    )
    source_model = mujoco.MjModel.from_xml_path(source.model_file)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "hand", source, initial_state=EntityInitialState(position=(1.0, 2.0, 3.0))
            ),
        ),
        default_keyframe_name="home",
    )
    with compose_scene(scene, 1, 0.002) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_file)
        layout = composed.layout.get_entity("hand")
        assert tuple(joint.name for joint in layout.root_joints) == ("px", "py", "pz", "palm_ball")
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert key >= 0
        # Root qpos/qvel columns equal the source key values bit-for-bit.
        np.testing.assert_array_equal(
            model.key_qpos[key][list(layout.root_qpos_indices)], source_model.key_qpos[0]
        )
        np.testing.assert_array_equal(
            model.key_qvel[key][list(layout.root_qvel_indices)], source_model.key_qvel[0]
        )
        # No slide-reference bake: qpos0 stays the source reference
        # configuration and the declared position locates the mount origin.
        np.testing.assert_array_equal(
            model.qpos0[list(layout.root_qpos_indices)], source_model.qpos0
        )
        np.testing.assert_array_equal(model.body_pos[model.body("hand/mount_x").id], [1, 2, 3])
        # FK at the merged keyframe equals the source FK translated by the
        # mount origin, with no residual reference offset.
        source_data = mujoco.MjData(source_model)
        mujoco.mj_resetDataKeyframe(source_model, source_data, 0)
        mujoco.mj_forward(source_model, source_data)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        palm = model.body("hand/palm").id
        expected = source_data.xpos[source_model.body("palm").id] + [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(data.xpos[palm], expected)
        np.testing.assert_array_equal(data.xpos[palm], [0.75, 2.25, 3.5])
        np.testing.assert_array_equal(
            data.xquat[palm], source_data.xquat[source_model.body("palm").id]
        )


def test_merge_keys_keeps_free_root_columns_on_the_declared_pose(tmp_path):
    """Free-joint roots keep the EntityInitialState keyframe override, bit-identically."""
    source = _keyed_chain_file(
        tmp_path, "9 9 9 0.5 0.5 0.5 0.5", "1 1 1 1 1 1", root="free"
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "hand", source, initial_state=EntityInitialState(position=(1.0, 2.0, 3.0))
            ),
        ),
        default_keyframe_name="home",
    )
    with compose_scene(scene, 1, 0.002) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_file)
        layout = composed.layout.get_entity("hand")
        assert not layout.root_joints
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert key >= 0
        # The source key root pose/velocity stay overridden by the declared
        # pose and zero velocity.
        np.testing.assert_array_equal(
            model.key_qpos[key][list(layout.root_qpos_indices)], [1, 2, 3, 1, 0, 0, 0]
        )
        np.testing.assert_array_equal(model.key_qvel[key][list(layout.root_qvel_indices)], 0)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.xpos[model.body("hand/palm").id], [1, 2, 3])
