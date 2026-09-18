"""Real Motrix acceptance for the bounded portable-entity profile."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import FixedVariantLayout, FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg


def _write(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _robot(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "robot",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="0 0 1">
          <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
          <geom name="base_geom" type="sphere" size=".08"/>
          <body name="link" pos="0 0 .2">
            <joint name="drive" axis="0 1 0" ref="0.1"/>
            <inertial pos="0 0 0" mass=".2" diaginertia=".03 .03 .03"/>
            <geom name="link_geom" type="sphere" size=".03"/>
          </body>
        </body></worldbody>
        <actuator><motor name="drive" joint="drive"/></actuator></mujoco>
        """,
    )


def _robot_with_control_default(tmp_path: Path) -> ModelSourceDescriptor:
    source = _robot(tmp_path / "control-default")
    xml = Path(source.model_file).read_text(encoding="utf-8")
    xml = xml.replace(
        '<motor name="drive" joint="drive"/>',
        '<motor name="drive" joint="drive" ctrlrange="-0.2 0.4"/>',
    ).replace(
        "</mujoco>",
        '<keyframe><key name="home" qpos="0.1" ctrl="0.8"/></keyframe></mujoco>',
    )
    return _write(tmp_path / "control-default", "robot-home", xml)


def _passive(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "passive",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="1 0 2">
          <freejoint name="root"/><inertial pos="0 0 0" mass=".7"
            diaginertia=".1 .1 .1"/>
          <geom name="base_geom" type="sphere" size=".06"/>
          <body name="child" pos=".15 0 0">
            <joint name="passive_hinge" axis="0 1 0"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom name="child_geom" type="sphere" size=".02"/>
          </body>
        </body></worldbody></mujoco>
        """,
    )


def _object(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "object",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="2 0 2">
          <freejoint name="root"/><inertial pos=".01 0 0" mass=".5"
            diaginertia=".02 .03 .04"/>
          <geom name="object_geom" type="sphere" size=".1"/>
        </body></worldbody></mujoco>
        """,
    )


def _heavy_passive(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path / "variant",
        "heavy-passive",
        """
        <mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
        <worldbody><body name="base" pos="1 0 2">
          <freejoint name="root"/><inertial pos="-.04 0 .02" mass="1.7"
            diaginertia=".11 .12 .13"/>
          <geom name="base_geom" type="sphere" size=".12"/>
          <body name="child" pos=".2 0 0">
            <joint name="passive_hinge" axis="0 1 0"/>
            <inertial pos="0 0 0" mass=".3" diaginertia=".04 .04 .04"/>
            <geom name="child_geom" type="sphere" size=".04"/>
          </body>
        </body></worldbody></mujoco>
        """,
    )


def _heavy_passive_with_joint_range(tmp_path: Path) -> ModelSourceDescriptor:
    source = _heavy_passive(tmp_path / "limited")
    xml = Path(source.model_file).read_text(encoding="utf-8").replace(
        '<joint name="passive_hinge" axis="0 1 0"/>',
        '<joint name="passive_hinge" axis="0 1 0" limited="true" range="-.2 .2"/>',
    )
    return _write(tmp_path, "heavy-passive-limited", xml)



def _table(tmp_path: Path) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        "table",
        """
        <mujoco><option gravity="0 0 -9.81"/><worldbody>
          <body name="base" pos="0 0 -.1"><inertial pos="0 0 0" mass="10"
            diaginertia="1 1 1"/><geom name="table_geom" type="box"
            size="1 1 .1"/></body>
        </worldbody></mujoco>
        """,
    )


def _scene(tmp_path: Path) -> SceneCfg:
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
                initial_state=EntityInitialState((1.0, 0.0, 2.0)),
            ),
            SceneEntitySpec(
                "object",
                _object(tmp_path),
                kind="rigid",
                initial_state=EntityInitialState((2.0, 0.0, 2.0)),
            ),
            SceneEntitySpec(
                "table",
                _table(tmp_path),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, -0.1)),
            ),
        )
    )


def test_portable_entities_layout_properties_and_selected_reset(tmp_path: Path):
    backend = MotrixBackend(_scene(tmp_path), 5, 0.002, base_name="robot/base")
    composed_path = Path(backend.get_scene_model_file())
    try:
        layout = backend.get_scene_layout()
        assert backend.get_entity_names() == ("robot", "passive", "object", "table")
        assert (layout.nq, layout.nv, layout.nu, layout.nbody, layout.ngeom) == (
            16,
            14,
            1,
            7,
            6,
        )
        assert backend.get_body_ids(
            ("robot/base", "passive/base", "object/base", "table/base")
        ).tolist() == [1, 3, 5, 6]
        assert backend.get_geom_names() == (
            "robot/base_geom",
            "robot/link_geom",
            "passive/base_geom",
            "passive/child_geom",
            "object/object_geom",
            "table/table_geom",
        )
        assert backend.get_actuator_names() == ("robot/drive",)
        assert backend.get_actuator_joint_names() == ("robot/drive",)
        np.testing.assert_array_equal(
            backend._portable_public_to_native_body, [-1, 0, 1, 2, 3, 4, 5]
        )
        np.testing.assert_array_equal(backend._portable_public_to_native_geom, np.arange(6))
        assert backend.get_body_subtree_ids(1).tolist() == [1, 2]
        assert backend.get_body_subtree_ids(2).tolist() == [2]
        np.testing.assert_allclose(
            backend.get_geom_size("robot/base_geom"), [0.08, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.0], atol=1e-7)
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"], 0.1, atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_entity_state("table")["root_pose"][:, :3],
            np.tile((0.0, 0.0, -0.1), (5, 1)),
            atol=1e-6,
        )

        object_body = layout.get_entity("object").body_ids[0]
        passive_joint = layout.get_entity("passive").joints[0].qpos_indices[0]
        np.testing.assert_allclose(backend.get_body_mass()[object_body], 0.5, rtol=2e-6)
        np.testing.assert_allclose(
            backend.get_body_ipos(np.asarray((1, 4)))[:, object_body, 0],
            [0.01, 0.01],
            rtol=2e-6,
        )
        assert backend.get_body_ipos().shape == (layout.nbody, 3)

        controls = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        backend.step(controls, nsteps=2)
        before = {
            name: {
                field: np.asarray(values).copy()
                for field, values in backend.get_entity_state(name).items()
            }
            for name in backend.get_entity_names()
        }
        physics_before = backend.get_physics_state().copy()
        rows = np.asarray((4, 1), dtype=np.intp)
        pose = np.asarray(
            [(2.5, 0.3, 2.4, 0.5, 0.5, 0.5, 0.5), (2.7, -0.2, 2.1, 0, 0, 1, 0)],
            dtype=np.float32,
        )
        velocity = np.asarray(
            [(0.4, -0.2, 0.1, 0.2, -0.1, 0.3), (-0.3, 0.2, 0.2, 0.1, 0.2, -0.4)],
            dtype=np.float32,
        )
        backend.reset_entities(
            SceneResetRequest(
                tuple(rows.tolist()),
                (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),),
            )
        )
        object_state = backend.get_entity_state("object")
        np.testing.assert_allclose(object_state["root_pose"][rows], pose, atol=1e-6)
        np.testing.assert_allclose(object_state["root_velocity"][rows], velocity, atol=1e-5)
        for name in ("robot", "passive", "table"):
            for field, values in before[name].items():
                np.testing.assert_array_equal(
                    np.asarray(backend.get_entity_state(name)[field]), values
                )
        untouched = np.asarray((0, 2, 3), dtype=np.intp)
        np.testing.assert_array_equal(
            backend.get_physics_state()[untouched], physics_before[untouched]
        )
        np.testing.assert_array_equal(backend._data.actuator_ctrls, controls)

        backend.step(controls)
        assert not np.array_equal(
            backend.get_entity_state("object")["root_pose"][rows, :3], pose[:, :3]
        )

        joint_before = backend.get_entity_state("object")["root_pose"].copy()
        object_before = backend.get_entity_state("object")["root_velocity"].copy()
        backend.reset_entities(
            SceneResetRequest(
                (2,),
                (
                    EntityStatePatch(
                        "passive",
                        joint_positions=np.asarray([[0.35]], np.float32),
                        joint_velocities=np.asarray([[1.5]], np.float32),
                    ),
                ),
            )
        )
        passive_state = backend.get_entity_state("passive")
        np.testing.assert_allclose(passive_state["joint_positions"][2], 0.35, atol=1e-6)
        np.testing.assert_allclose(passive_state["joint_velocities"][2], 1.5, atol=1e-6)
        np.testing.assert_array_equal(backend.get_entity_state("object")["root_pose"], joint_before)
        np.testing.assert_array_equal(
            backend.get_entity_state("object")["root_velocity"], object_before
        )
        assert passive_joint == 8

        backend.step(np.zeros((5, 1), np.float32), nsteps=2)
        assert backend.get_entity_state("passive")["joint_positions"][2, 0] != 0.35

        controls_before_restore = backend._portable_current_controls().copy()
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (EntityStatePatch("object", root_pose=pose[:1]),),
                restore_default_controls=True,
            )
        )
        np.testing.assert_array_equal(
            backend._portable_current_controls(), controls_before_restore
        )
    finally:
        backend.close()

    assert not composed_path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_entity_state("object")


def test_fixed_variants_preserve_public_layout_and_native_identity(tmp_path: Path):
    scene = _scene(tmp_path)
    passive_entity = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.array([1, 1, 0, 1, 0], dtype=np.int32),
            (passive_entity.source, _heavy_passive(tmp_path)),
        ),
    )
    backend = MotrixBackend(scene, 5, 0.002, base_name="robot/base")
    composed_path = Path(backend.get_scene_model_file())
    try:
        layout = backend.get_scene_layout()
        assert (layout.nq, layout.nv, layout.nu, layout.nbody, layout.ngeom) == (
            16,
            14,
            1,
            7,
            6,
        )
        assert tuple(backend._portable_variant_assignment) == (1, 1, 0, 1, 0)
        assert backend.get_dr_capabilities().supports_fixed_variants
        assert backend.get_dr_capabilities().supported_fixed_variant_layouts == frozenset(
            {FixedVariantLayout.SAME_LAYOUT}
        )
        with pytest.raises(NotImplementedError, match="portable Motrix site Jacobians"):
            backend.get_site_jacobian_w(0, np.asarray([], dtype=np.intp))
        with pytest.raises(NotImplementedError, match="fixed-variant scenes yet"):
            backend.init_renderer()
        with pytest.raises(NotImplementedError, match="fixed-variant scenes yet"):
            backend.run_playback(env=None, initialize=None, step=None, num_steps=1)

        passive_body = layout.get_entity("passive").body_ids[0]
        passive_geom_id = layout.get_geom_ids(("passive/base_geom",))[0]
        np.testing.assert_allclose(
            backend.get_body_mass()[:, passive_body],
            [1.7, 1.7, 0.7, 1.7, 0.7],
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            backend.get_body_ipos(np.arange(5))[:, passive_body, 0],
            [-0.04, -0.04, 0.0, -0.04, 0.0],
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            backend._portable_variant_geom_sizes[:, passive_geom_id, 0],
            [0.06, 0.12],
            rtol=2e-7,
        )
        controls = np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
        backend.step(controls, nsteps=2)
        before = {
            name: {
                field: np.asarray(values).copy()
                for field, values in backend.get_entity_state(name).items()
            }
            for name in backend.get_entity_names()
        }
        physics_before = backend.get_physics_state().copy()
        rows = np.asarray((0, 3), dtype=np.intp)
        pose = np.asarray(
            [(2.4, 0.2, 2.3, 0.5, 0.5, 0.5, 0.5), (2.6, -0.2, 2.2, 0, 0, 1, 0)],
            dtype=np.float32,
        )
        velocity = np.asarray(
            [(0.3, -0.1, 0.2, 0.1, 0.2, -0.3), (-0.2, 0.1, 0.1, 0.2, -0.1, 0.4)],
            dtype=np.float32,
        )
        backend.reset_entities(
            SceneResetRequest(
                tuple(rows.tolist()),
                (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),),
            )
        )
        object_state = backend.get_entity_state("object")
        np.testing.assert_allclose(object_state["root_pose"][rows], pose, atol=1e-6)
        np.testing.assert_allclose(object_state["root_velocity"][rows], velocity, atol=1e-5)
        for name in ("robot", "passive", "table"):
            for field, values in before[name].items():
                np.testing.assert_array_equal(
                    np.asarray(backend.get_entity_state(name)[field]), values
                )
        untouched = np.asarray((1, 2, 4), dtype=np.intp)
        np.testing.assert_array_equal(
            backend.get_physics_state()[untouched], physics_before[untouched]
        )
        for runtime in backend._portable_runtimes:
            if runtime.variant == 1:
                np.testing.assert_array_equal(runtime.data.actuator_ctrls, controls[runtime.rows])

        passive_child_body = layout.get_entity("passive").body_ids[1]
        joint_velocity_before = backend.get_entity_state("passive")["joint_velocities"][
            :, 0
        ].copy()
        # Motrix does not expose inertia readback. Equal joint torques on the
        # scalar child verify distinct effective native inertia identity.
        for runtime in backend._portable_runtimes:
            passive_link = runtime.binding.links_by_id[
                int(runtime.binding.public_to_native_body[passive_child_body])
            ]
            torque = np.zeros((runtime.rows.size, 3), dtype=np.float32)
            torque[:, 1] = 8.0
            passive_link.add_external_torque(
                runtime.data, np.ascontiguousarray(torque), local=True
            )
        for runtime in backend._portable_runtimes:
            runtime.model.step(runtime.data)
        backend._refresh_link_pose_cache()
        backend._invalidate_link_velocity_cache()
        joint_velocity_after = backend.get_entity_state("passive")["joint_velocities"][:, 0]
        delta = joint_velocity_after - joint_velocity_before
        assert delta[2] > delta[0] * 2
        assert np.all(np.isfinite(delta))
    finally:
        backend.close()

    assert not composed_path.exists()


def test_fixed_variants_map_body_forces_and_reset_scope(tmp_path: Path):
    scene = _scene(tmp_path)
    passive_entity = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.array([1, 1, 0, 1, 0], dtype=np.int32),
            (passive_entity.source, _heavy_passive(tmp_path)),
        ),
    )
    backend = MotrixBackend(scene, 5, 0.002, base_name="robot/base")
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supports_interval_body_force
        assert capabilities.supported_interval_terms == frozenset({"body_force"})

        passive_body = np.asarray(
            [backend.get_scene_layout().get_entity("passive").body_ids[0]], dtype=np.int32
        )
        force = np.zeros((5, 1, 3), dtype=np.float32)
        force[:, 0, 2] = 4.0
        with pytest.raises(ValueError, match="body_ids must be"):
            backend.apply_body_force(np.asarray([-1], dtype=np.int32), force[:, :1])
        with pytest.raises(NotImplementedError, match="interval body torque"):
            backend.apply_body_force(passive_body, force, torque=force)

        backend.apply_body_force(passive_body, force)
        backend.apply_body_force(passive_body, force)
        np.testing.assert_allclose(
            backend._portable_pending_body_forces[int(passive_body[0])][:, 2], 8.0, atol=0
        )

        # Resetting an unrelated entity must not cancel passive's pending force.
        object_state = backend.get_entity_state("object")
        backend.reset_entities(
            SceneResetRequest(
                (0, 3),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=object_state["root_pose"][[0, 3]].copy(),
                        root_velocity=object_state["root_velocity"][[0, 3]].copy(),
                    ),
                ),
            )
        )
        velocity_before = backend.get_entity_state("passive")["root_velocity"][:, 2].copy()
        backend.step(np.zeros((5, 1), dtype=np.float32))
        velocity_with_force = backend.get_entity_state("passive")["root_velocity"][:, 2]
        assert velocity_with_force[2] > 0.0
        assert velocity_with_force[4] > 0.0
        assert np.all(velocity_with_force[[0, 1, 3]] < -0.008)
        assert np.all(
            backend._portable_pending_body_forces[int(passive_body[0])] == 0.0
        )

        backend.apply_body_force(passive_body, force)
        backend.apply_body_force(passive_body, force)
        passive_state = backend.get_entity_state("passive")
        velocity_before = passive_state["root_velocity"][:, 2].copy()
        backend.reset_entities(
            SceneResetRequest(
                (2, 4),
                (
                    EntityStatePatch(
                        "passive",
                        root_pose=passive_state["root_pose"][[2, 4]].copy(),
                        root_velocity=passive_state["root_velocity"][[2, 4]].copy(),
                    ),
                ),
            )
        )
        pending = backend._portable_pending_body_forces[int(passive_body[0])]
        np.testing.assert_allclose(pending[[2, 4], 2], 0.0, atol=0)
        np.testing.assert_allclose(pending[[0, 1, 3], 2], 8.0, atol=0)

        backend.step(np.zeros((5, 1), dtype=np.float32))
        velocity_after_reset = backend.get_entity_state("passive")["root_velocity"][:, 2]
        delta = velocity_after_reset - velocity_before
        assert np.all(delta[[2, 4]] < np.max(delta[[0, 1, 3]]) - 0.002)
    finally:
        backend.close()


def test_no_variant_body_force_is_consumed_by_next_step(tmp_path: Path):
    backend = MotrixBackend(_scene(tmp_path), 2, 0.002, base_name="robot/base")
    try:
        assert backend.get_dr_capabilities().supports_interval_body_force
        object_body = np.asarray(
            [backend.get_scene_layout().get_entity("object").body_ids[0]], dtype=np.int32
        )
        force = np.zeros((2, 1, 3), dtype=np.float32)
        force[:, 0, 2] = 6.0
        velocity_before = backend.get_entity_state("object")["root_velocity"][:, 2].copy()

        backend.apply_body_force(object_body, force)
        backend.step(np.zeros((2, 1), dtype=np.float32))
        velocity_with_force = backend.get_entity_state("object")["root_velocity"][:, 2]
        assert np.all(velocity_with_force > velocity_before)
        assert np.all(backend._portable_pending_body_forces[int(object_body[0])] == 0.0)

        backend.step(np.zeros((2, 1), dtype=np.float32))
        velocity_after = backend.get_entity_state("object")["root_velocity"][:, 2]
        assert np.all(velocity_after < velocity_with_force)
    finally:
        backend.close()


def test_selected_default_controls_restore_only_impacted_native_rows(tmp_path: Path):
    scene = _scene(tmp_path)
    robot = next(entity for entity in scene.entity_assets if entity.name == "robot")
    passive = next(entity for entity in scene.entity_assets if entity.name == "passive")
    robot = replace(robot, source=_robot_with_control_default(tmp_path))
    scene.entity_assets = tuple(
        robot if entity.name == "robot" else entity for entity in scene.entity_assets
    )
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.array([1, 1, 0, 1, 0], dtype=np.int32),
            (passive.source, _heavy_passive(tmp_path)),
        ),
    )
    scene.default_keyframe_name = "home"
    backend = MotrixBackend(scene, 5, 0.002, base_name="robot/base")
    try:
        for runtime in backend._portable_runtimes:
            key = next(key for key in runtime.model.keyframes if str(key.name) == "home")
            np.testing.assert_allclose(key.ctrl, [0.8], atol=0)
        np.testing.assert_allclose(backend._portable_default_controls(), [[0.4]] * 5)

        controls = np.asarray(
            [[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32
        )
        backend.step(controls, nsteps=2)
        backend.reset_entities(
            SceneResetRequest(
                (0, 3),
                (
                    EntityStatePatch(
                        "robot",
                        joint_positions=np.asarray(
                            [[0.4], [0.4]], dtype=np.float32
                        ),
                    ),
                ),
                restore_default_controls=True,
            )
        )
        np.testing.assert_allclose(
            backend._portable_current_controls(),
            [[0.4], [0.2], [0.3], [0.4], [0.5]],
            atol=0,
        )

        pose = np.asarray([(2.4, 0.2, 2.2, 1, 0, 0, 0)], dtype=np.float32)
        velocity = np.asarray([(0.2, 0, 0, 0, 0, 0)], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                (4,),
                (
                    EntityStatePatch("object", root_pose=pose, root_velocity=velocity),
                ),
                restore_default_controls=True,
            )
        )
        np.testing.assert_allclose(
            backend._portable_current_controls(),
            [[0.4], [0.2], [0.3], [0.4], [0.5]],
            atol=0,
        )

        backend.step(backend._portable_current_controls())
        assert backend.get_entity_state("object")["root_pose"][4, :3].tolist() != pose[0].tolist()
    finally:
        backend.close()


def test_unsupported_fixed_variant_layout_fails_closed(tmp_path: Path):
    scene = _scene(tmp_path)
    passive_entity = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "object",
        FixedVariantPlan(
            np.array([0, 1], dtype=np.int32),
            (passive_entity.source, _heavy_passive(tmp_path)),
            FixedVariantLayout.UNIFORM_PUBLIC_LAYOUT,
        ),
    )
    with pytest.raises(NotImplementedError, match="same_layout fixed variants"):
        MotrixBackend(scene, 2, 0.002)


def test_fixed_variant_nonuniform_joint_limits_fail_closed(tmp_path: Path):
    scene = _scene(tmp_path)
    passive_entity = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.array([0, 1], dtype=np.int32),
            (passive_entity.source, _heavy_passive_with_joint_range(tmp_path)),
        ),
    )
    with pytest.raises(NotImplementedError, match="differing public control/joint limits"):
        MotrixBackend(scene, 2, 0.002)


def test_fixed_variant_nonuniform_public_geometry_size_fails_closed(tmp_path: Path):
    scene = _scene(tmp_path)
    passive_entity = next(entity for entity in scene.entity_assets if entity.name == "passive")
    scene.entity_variant = EntityVariantBinding(
        "passive",
        FixedVariantPlan(
            np.array([0, 1], dtype=np.int32),
            (passive_entity.source, _heavy_passive(tmp_path)),
        ),
    )
    backend = MotrixBackend(scene, 2, 0.002, base_name="robot/base")
    try:
        with pytest.raises(NotImplementedError, match="non-uniform public geometry sizes"):
            backend.get_geom_size("passive/base_geom")
    finally:
        backend.close()


def test_unsupported_portable_profiles_fail_closed(tmp_path: Path):
    scene = _scene(tmp_path)
    mirror = SceneEntitySpec(
        "mirror",
        kind="rigid",
        root_mode="kinematic",
        collision_enabled=False,
        mirror_of="object",
    )
    scene.entity_assets = scene.entity_assets + (mirror,)
    with pytest.raises(NotImplementedError, match="kinematic mirrors"):
        MotrixBackend(scene, 2, 0.002)

    scene = _scene(tmp_path / "sensors")
    fragment = tmp_path / "sensors" / "fragment.xml"
    fragment.write_text(
        "<mujoco><sensor><framepos name='sensor' objtype='site'/></sensor></mujoco>"
    )
    scene.fragment_files = (str(fragment),)
    with pytest.raises(NotImplementedError, match="sensor fragments"):
        MotrixBackend(scene, 2, 0.002)
