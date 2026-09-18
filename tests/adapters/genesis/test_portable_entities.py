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
    drive_joint = (
        '<joint name="drive" axis="0 1 0" ref="0.1" damping=".17" '
        'frictionloss=".043"/>'
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
            <joint name="passive_hinge" axis="0 1 0" damping=".31" frictionloss=".027"/>
            <inertial pos="0 0 0" mass=".1" diaginertia=".01 .01 .01"/>
            <geom name="passive_child_geom" type="sphere" size=".02"
              contype="0" conaffinity="0"/>
          </body>
        </body></worldbody></mujoco>
        """,
    )


def _object(
    tmp_path: Path,
    name: str,
    *,
    radius: float,
    mass: float,
    com_x: float,
    inertia: tuple[float, float, float] = (0.02, 0.03, 0.04),
) -> ModelSourceDescriptor:
    return _write(
        tmp_path,
        name,
        f"""
        <mujoco><compiler angle="radian"/><option gravity="0 0 0"/>
        <worldbody><body name="base">
          <freejoint name="root"/>
          <inertial pos="{com_x} 0 0" mass="{mass}"
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


def _scene(
    tmp_path: Path,
    *,
    assignment: tuple[int, ...] = (0, 0, 0, 1, 1),
    object_inertias: tuple[tuple[float, float, float], ...] = (
        (0.02, 0.03, 0.04),
        (0.03, 0.04, 0.05),
    ),
) -> SceneCfg:
    object_a = _object(
        tmp_path,
        "object_a",
        radius=0.1,
        mass=0.5,
        com_x=0.01,
        inertia=object_inertias[0],
    )
    object_b = _object(
        tmp_path,
        "object_b",
        radius=0.15,
        mass=1.5,
        com_x=0.03,
        inertia=object_inertias[1],
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


def _enable_native_contact_masks(scene: SceneCfg, object_variant_b: tuple[int, int]) -> None:
    masks = {
        "robot": (1, 16),
        "passive": (2, 32),
        "object": (4, 64),
        "table": (8, 128),
    }
    for entity in scene.entity_assets:
        sources = [Path(entity.source.model_file)]
        if scene.entity_variant is not None and (
            scene.entity_variant.target_entity == entity.name
        ):
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


def test_portable_entities_layout_variants_selected_state_and_control(tmp_path: Path):
    with pytest.raises(ValueError, match="balanced mapping"):
        GenesisBackend(_scene(tmp_path, assignment=(1, 1, 0, 1, 0)), 5, 0.002)

    scene = _scene(tmp_path)
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
        expected_damping[layout.get_entity("robot").qvel_indices] = 0.17
        expected_damping[layout.get_entity("passive").qvel_indices[-1]] = 0.31
        expected_frictionloss[layout.get_entity("robot").qvel_indices] = 0.043
        expected_frictionloss[layout.get_entity("passive").qvel_indices[-1]] = 0.027
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
        variant_a_distance = float(
            np.max(np.linalg.norm(object_quat[:3] - object_quat[0], axis=1))
        )
        variant_b_distance = float(
            np.max(np.linalg.norm(object_quat[3:] - object_quat[3], axis=1))
        )
        cross_variant_distance = float(
            np.max(np.linalg.norm(object_quat[:3, None, :] - object_quat[None, 3:, :], axis=-1))
        )
        assert cross_variant_distance > max(variant_a_distance, variant_b_distance) * 3.0
    finally:
        backend.close()

    assert backend._composed_scene is None
    assert backend._portable_sources is None
    assert backend._scene_cleanup_handle is None
