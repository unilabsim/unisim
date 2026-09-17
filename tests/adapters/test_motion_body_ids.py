"""Motion-dataset body-id contract tests (worldbody is id 0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.subprocess_ipc.backend import MjcfSubprocessBackend
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model="unisim-subprocess-test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
      <body name="arm" pos="0 0 0.12">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="hinge_pos" joint="hinge" kp="20" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


def _make_backend(tmp_path: Path) -> MjcfSubprocessBackend:
    model_file = tmp_path / "scene.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    # Construction is lazy: no worker is spawned until materialize(), so the
    # pre-INIT XML metadata path is exercised without an Isaac runtime.
    return MjcfSubprocessBackend(
        SceneCfg(model_file=str(model_file)), num_envs=1, sim_dt=0.005
    )


def test_subprocess_motion_body_ids_follow_mjcf_worldbody_zero_convention(
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    backend = _make_backend(tmp_path)

    names = ["base", "arm"]
    model = mujoco.MjModel.from_xml_string(_MODEL)
    expected = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names],
        dtype=np.int32,
    )
    assert expected.tolist() == [1, 2]  # MJCF body ids, worldbody is id 0
    # The XML body scan excludes worldbody, so backend ids are one behind the
    # motion-dataset columns.
    np.testing.assert_array_equal(backend.get_body_ids(names), expected - 1)
    np.testing.assert_array_equal(backend.get_motion_body_ids(names), expected)


def test_subprocess_motion_body_ids_offset_after_worker_handshake(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    # Post-INIT the worker body table replaces the XML scan; the handshake
    # cross-checks it against the same worldbody-excluded order, so the +1
    # offset must hold there too.
    backend._model_info = object()  # type: ignore[assignment]
    backend._body_id_by_name = {"base": 0, "arm": 1}
    np.testing.assert_array_equal(
        backend.get_motion_body_ids(["base", "arm"]),
        np.asarray([1, 2], dtype=np.int32),
    )
