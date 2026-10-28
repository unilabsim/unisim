"""Named scalar joint-range reads on real Newton CUDA backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.scene import SceneCfg

_MODEL = """
<mujoco model="newton-named-joint-ranges">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="base_geom" type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      <body name="slide_body">
        <joint name="slide_joint" type="slide" axis="1 0 0" range="-0.02 0.03"/>
        <geom name="slide_geom" size="0.03" contype="0" conaffinity="0"/>
      </body>
      <body name="hinge_body">
        <joint name="hinge_joint" type="hinge" axis="0 0 1" range="-30 60"/>
        <geom name="hinge_geom" size="0.03" contype="0" conaffinity="0"/>
      </body>
      <body name="unlimited_body">
        <joint name="unlimited_joint" type="hinge" axis="0 1 0" limited="false"/>
        <geom name="unlimited_geom" size="0.03" contype="0" conaffinity="0"/>
      </body>
      <body name="ball_body">
        <joint name="ball_joint" type="ball"/>
        <geom name="ball_geom" size="0.03" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _cuda_backend(tmp_path: Path, xml: str = _MODEL):
    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import (
        NewtonDependencyError,
        load_newton_dependencies,
    )

    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton named joint-range tests require a CUDA Warp device")
    model_file = tmp_path / "joint_ranges.xml"
    model_file.write_text(xml, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


def test_real_cuda_named_joint_range_order_duplicates_errors_and_reset(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        names = ["slide_joint", "hinge_joint", "slide_joint", "unlimited_joint"]
        expected = np.array(
            [
                [-0.02, 0.03],
                [-np.pi / 6, np.pi / 3],
                [-0.02, 0.03],
                [-np.inf, np.inf],
            ],
            dtype=np.float32,
        )
        actual = backend.get_joint_range(names=names)
        assert actual.shape == (4, 2)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

        actual[0, 0] = 99.0
        np.testing.assert_allclose(
            backend.get_joint_range(names=["slide_joint"]),
            [[-0.02, 0.03]],
            rtol=1e-6,
            atol=1e-7,
        )
        with pytest.raises(ValueError, match="missing_joint"):
            backend.get_joint_range(names=["missing_joint"])
        with pytest.raises(ValueError, match="root.*hinge or slide"):
            backend.get_joint_range(names=["root"])
        with pytest.raises(ValueError, match="ball_joint.*hinge or slide"):
            backend.get_joint_range(names=["ball_joint"])

        views = backend.get_state_views()
        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64, device=backend._device),
            views["qpos"][:1].contiguous(),
            views["qvel"][:1].contiguous(),
        )
        np.testing.assert_allclose(
            backend.get_joint_range(names=names), expected, rtol=1e-6, atol=1e-7
        )
        legacy = backend.get_joint_range()
        assert legacy is not None and legacy.shape == (3, 2)
    finally:
        backend.close()


def test_real_cuda_portable_joint_range_qualified_names_and_variant_defaults(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from unisim.entities import EntityStatePatch, SceneResetRequest

    from .test_multi_entity_feasibility import _backend, _scene

    scene = _scene(tmp_path)
    robot_source = scene.entity_assets[0].source
    assert robot_source is not None
    robot_path = Path(robot_source.model_file)
    robot_path.write_text(
        robot_path.read_text(encoding="utf-8").replace(
            'ref=".4"', 'ref=".4" limited="true" range="-0.2 0.3"'
        ),
        encoding="utf-8",
    )
    backend = _backend(scene)
    try:
        backend.model
        names = ["robot/hinge", "robot/hinge"]
        np.testing.assert_allclose(
            backend.get_joint_range(names=names),
            [[-0.2, 0.3], [-0.2, 0.3]],
            rtol=1e-6,
            atol=1e-7,
        )
        with pytest.raises(ValueError, match="hinge.*not found"):
            backend.get_joint_range(names=["hinge"])
        with pytest.raises(ValueError, match="object/root.*hinge or slide"):
            backend.get_joint_range(names=["object/root"])

        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (EntityStatePatch("robot", joint_positions=np.zeros((1, 1), dtype=np.float32)),),
            )
        )
        np.testing.assert_allclose(
            backend.get_joint_range(names=names),
            [[-0.2, 0.3], [-0.2, 0.3]],
            rtol=1e-6,
            atol=1e-7,
        )

        hinge_id = int(
            backend._deps.mujoco.mj_name2id(
                backend._metadata.playback_model,
                backend._deps.mujoco.mjtObj.mjOBJ_JOINT,
                "robot/hinge",
            )
        )
        variant_model = backend._variant_metadata[1].playback_model
        original = float(variant_model.jnt_range[hinge_id, 1])
        variant_model.jnt_range[hinge_id, 1] = original + 0.1
        with pytest.raises(NotImplementedError, match="identical defaults.*robot/hinge"):
            backend.get_joint_range(names=["robot/hinge"])
    finally:
        backend.close()
