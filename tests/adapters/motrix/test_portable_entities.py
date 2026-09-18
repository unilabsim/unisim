"""Real Motrix acceptance for the bounded portable-entity profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
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

        with pytest.raises(NotImplementedError, match="restore_default_controls"):
            backend.reset_entities(
                SceneResetRequest(
                    (0,),
                    (EntityStatePatch("object", root_pose=pose[:1]),),
                    restore_default_controls=True,
                )
            )
    finally:
        backend.close()

    assert not composed_path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        backend.get_entity_state("object")


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
