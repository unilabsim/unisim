"""Newton portable physical-kinematic-root acceptance (#444)."""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
newton = pytest.importorskip("newton")
warp = pytest.importorskip("warp")

from unisim.backend.newton.backend import NewtonBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import (
    EntityInitialState,
    EntityStatePatch,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.mjcf_compiler import compose_scene
from unisim.scene import SceneCfg


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    device = warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton physical-kinematic-root acceptance requires CUDA")


def _free_box(path: Path, name: str, *, half_height: float) -> None:
    path.write_text(
        f"""
        <mujoco>
          <compiler angle="radian"/>
          <option gravity="0 0 -9.81" integrator="Euler"/>
          <worldbody>
            <body name="{name}">
              <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
              <freejoint name="{name}_root"/>
              <geom name="{name}_geom" type="box" size=".1 .1 {half_height}"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )


def test_physical_kinematic_root_keeps_state_and_collides(tmp_path: Path) -> None:
    support = tmp_path / "support.xml"
    dynamic = tmp_path / "dynamic.xml"
    _free_box(support, "support", half_height=0.05)
    _free_box(dynamic, "dynamic", half_height=0.05)
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "support",
                ModelSourceDescriptor(str(support)),
                kind="rigid",
                root_mode="kinematic",
            ),
            SceneEntitySpec(
                "dynamic",
                ModelSourceDescriptor(str(dynamic)),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState((0.0, 0.0, 0.11)),
            ),
        )
    )
    composed = compose_scene(scene, num_envs=1, sim_dt=0.01)
    try:
        support_layout = composed.layout.get_entity("support")
        assert support_layout.has_public_root_state
        assert support_layout.root_qpos_indices == tuple(range(7))
        assert support_layout.root_qvel_indices == tuple(range(6))
    finally:
        composed.close()

    backend = NewtonBackend(
        scene,
        num_envs=1,
        sim_dt=0.01,
        device=str(warp.get_device()),
        nconmax=8,
        njmax=16,
        capacity_check_steps=1,
    )
    try:
        backend.materialize()
        support_entity = backend.get_scene_layout().get_entity("support")
        native_support_body = int(support_entity.body_ids[0]) - 1
        flags = np.asarray(backend._model.body_flags.numpy(), dtype=np.int32)
        assert int(flags[native_support_body]) & int(newton.BodyFlags.KINEMATIC)

        native_armature = np.asarray(
            backend._solver.mjw_model.dof_armature.numpy(), dtype=np.float32
        ).reshape(1, -1)
        np.testing.assert_array_equal(
            native_armature[:, np.asarray(support_entity.root_qvel_indices)],
            np.full((1, 6), 1.0e10, dtype=np.float32),
        )

        before = backend.get_entity_state("support")
        dynamic_before = backend.get_entity_state("dynamic")
        backend._host_step(np.zeros((1, backend.num_actuators), dtype=np.float32), nsteps=10)
        after = backend.get_entity_state("support")
        np.testing.assert_array_equal(
            after["root_pose"],
            before["root_pose"],
        )
        np.testing.assert_array_equal(
            after["root_velocity"],
            np.zeros((1, 6), dtype=np.float32),
        )
        assert not np.array_equal(
            backend.get_entity_state("dynamic")["root_pose"],
            dynamic_before["root_pose"],
        )
        assert int(np.asarray(backend._solver.mjw_data.nacon.numpy()).reshape(-1)[0]) > 0

        reset_pose = np.array(
            [[0.2, 0.0, 0.2, 0.7071068, 0.7071068, 0.0, 0.0]], dtype=np.float32
        )
        backend.reset_entities(
            SceneResetRequest(
                (0,),
                (EntityStatePatch("support", root_pose=reset_pose),),
            )
        )
        np.testing.assert_allclose(
            backend.get_entity_state("support")["root_pose"], reset_pose, atol=1e-6
        )
        backend._host_step(np.zeros((1, backend.num_actuators), dtype=np.float32))
        np.testing.assert_allclose(
            backend.get_entity_state("support")["root_pose"], reset_pose, atol=1e-6
        )
        np.testing.assert_array_equal(
            backend.get_entity_state("support")["root_velocity"],
            np.zeros((1, 6), dtype=np.float32),
        )
    finally:
        backend.cleanup_scene_assets()
