from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from unisim.scene import SceneCfg

MODEL = """<mujoco model="named-joint-ranges">
  <compiler angle="degree"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <default>
    <geom type="sphere" size="0.03" mass="0.1" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="robot" pos="0 0 1">
      <freejoint name="robot_root"/>
      <geom/>
      <body name="finger" pos="0.1 0 0">
        <joint name="finger_hinge" type="hinge" axis="0 0 1"
               limited="true" range="-30 60"/>
        <geom/>
      </body>
    </body>
    <body name="object" pos="1 0 1">
      <freejoint name="object_root"/>
      <geom/>
      <body name="slider" pos="0.1 0 0">
        <joint name="object_slide" type="slide" axis="1 0 0"
               limited="true" range="-0.02 0.03"/>
        <geom/>
      </body>
    </body>
    <body name="unlimited-holder" pos="2 0 1">
      <joint name="unlimited_hinge" type="hinge" axis="0 0 1" limited="false"/>
      <geom/>
    </body>
    <body name="ball-holder" pos="3 0 1">
      <joint name="ball_joint" type="ball"/>
      <geom/>
    </body>
  </worldbody>
</mujoco>"""


def _make_mujoco(tmp_path: Path):
    pytest.importorskip("mujoco")
    from unisim import MuJoCoBackend

    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    return MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=1, sim_dt=0.01)


def _make_mjwarp(tmp_path: Path):
    pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")

    from unisim import MjwarpBackend

    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    return MjwarpBackend(SceneCfg(model_file=str(model_path)), num_envs=1, sim_dt=0.01)


@pytest.fixture(params=(_make_mujoco, _make_mjwarp), ids=("mujoco", "mjwarp"))
def make_backend(request) -> Callable[[Path], object]:
    return request.param


def test_get_joint_range_by_name(tmp_path: Path, make_backend: Callable[[Path], object]) -> None:
    backend = make_backend(tmp_path)

    actual = backend.get_joint_range(names=["object_slide", "finger_hinge"])
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(
        actual,
        [[-0.02, 0.03], [-np.pi / 6, np.pi / 3]],
        rtol=1e-6,
        atol=1e-7,
    )

    single = backend.get_joint_range(names=["object_slide"])
    assert single.shape == (1, 2)
    np.testing.assert_allclose(single, [[-0.02, 0.03]], rtol=1e-6, atol=1e-7)

    unlimited = backend.get_joint_range(names=["unlimited_hinge"])
    assert unlimited.shape == (1, 2)
    np.testing.assert_allclose(unlimited, [[-np.inf, np.inf]])

    with pytest.raises(ValueError, match="missing_joint"):
        backend.get_joint_range(names=["missing_joint"])
    with pytest.raises(ValueError, match="robot_root.*hinge or slide"):
        backend.get_joint_range(names=["robot_root"])
    with pytest.raises(ValueError, match="ball_joint.*hinge or slide"):
        backend.get_joint_range(names=["ball_joint"])

    legacy = backend.get_joint_range()
    assert legacy is not None
    assert legacy.shape == (4, 2)
    np.testing.assert_allclose(
        legacy,
        [
            [-np.pi / 6, np.pi / 3],
            [-0.02, 0.03],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        rtol=1e-6,
        atol=1e-7,
    )
