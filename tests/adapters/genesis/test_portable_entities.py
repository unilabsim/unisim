"""Real Genesis CPU acceptance for portable multi-entity scenes."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("genesis")
pytest.importorskip("torch")

from unisim.backend.genesis.backend import GenesisBackend
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
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
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _robot(tmp_path: Path, *, passive: bool = False) -> ModelSourceDescriptor:
    passive_joint = '<joint name="passive" axis="0 0 1"/>' if passive else ""
    drive = "" if passive else '<position name="drive" joint="drive" kp="24" kv="3"/>'
    return _write(
        tmp_path,
        "robot-passive" if passive else "robot",
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <inertial pos="0 0 0" mass="1" diaginertia=".2 .2 .2"/>
          <geom name="base_geom" type="sphere" size=".08" contype="0" conaffinity="0"/>
          <body name="link" pos="0 0 .2">
            {'<joint name="drive" axis="0 1 0" ref="0.1"/>' if not passive else passive_joint}
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
            <joint name="passive_hinge" axis="0 1 0"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom name="passive_child_geom" type="sphere" size=".02"
              contype="0" conaffinity="0"/>
          </body>
        </body></worldbody></mujoco>
        """,
    )


def _object(
    tmp_path: Path, name: str, *, radius: float, mass: float, com_x: float
) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        name,
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <freejoint name="root"/>
          <inertial pos="{com_x} 0 0" mass="{mass}" diaginertia=".02 .03 .04"/>
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


def _scene(
    tmp_path: Path, *, assignment: tuple[int, ...] = (0, 0, 0, 1, 1)
) -> SceneCfg:
    object_a = _object(tmp_path, "object_a", radius=0.1, mass=0.5, com_x=0.01)
    object_b = _object(tmp_path, "object_b", radius=0.15, mass=1.5, com_x=0.03)
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


def test_portable_entities_layout_variants_selected_state_and_control(tmp_path: Path):
    with pytest.raises(ValueError, match="balanced mapping"):
        GenesisBackend(_scene(tmp_path, assignment=(1, 1, 0, 1, 0)), 5, 0.002)

    backend = GenesisBackend(_scene(tmp_path), 5, 0.002)
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
            int(robot_runtime.entity.get_link(name).idx_local)
            for name in ("base", "link")
        ]
        assert passive_runtime.native_body_indices.tolist() == [
            int(passive_runtime.entity.get_link(name).idx_local)
            for name in ("base", "child")
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
        np.testing.assert_allclose(
            native_mass, [0.5, 0.5, 0.5, 1.5, 1.5], rtol=2e-6, atol=1e-7
        )
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

        table_state = backend.get_entity_state("table")
        np.testing.assert_allclose(
            table_state["root_pose"][:, :3], np.tile((0.0, 0.0, -3.0), (5, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            backend.get_entity_state("robot")["joint_positions"], 0.1, atol=1e-6
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
        np.testing.assert_allclose(
            backend.get_entity_state("passive")["joint_positions"], 0.0, atol=1e-7
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
    finally:
        backend.close()

    assert backend._composed_scene is None
    assert backend._portable_sources is None
    assert backend._scene_cleanup_handle is None
