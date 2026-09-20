"""Motrix fused selected-body state copy (``copy_body_state_w``) contract.

MotrixSim 0.10.1 exposes ``SceneModel.get_link_states`` — one native call
returning pos/wxyz-quat/linvel/angvel for selected links with optional
caller-owned float32 outputs. These tests pin the adapter's parity between
the fused copy path and the cache-based ``get_body_state_w`` getters, for
both whole-MJCF and portable entity scenes, plus the dtype-agnostic
fallback for non-float32 caller buffers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim.backend.motrix.backend import MotrixBackend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
from unisim.scene import SceneCfg

ROBOT = """<mujoco model='unisim-test'>
  <compiler angle="radian"/>
  <option gravity='0 0 -9.81' timestep='0.01'/>
  <worldbody><body name='base' pos='0 0 1'>
    <inertial pos='0 0 0' mass='1' diaginertia='.1 .1 .1'/>
    <geom name='base_geom' type='sphere' size='.1'/>
    <body name='link' pos='0 0 .3'>
      <joint name='drive' type='hinge' axis='0 1 0' ref='0.1'/>
      <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
      <geom name='link_geom' type='sphere' size='.05'/>
    </body>
  </body></worldbody>
  <actuator><motor name='drive' joint='drive' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def _write(tmp_path: Path, name: str, xml: str) -> ModelSourceDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return ModelSourceDescriptor(str(path))


def _alloc(num_envs: int, num_bodies: int, dtype=np.float32):
    return (
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
        np.zeros((num_envs, num_bodies, 4), dtype=dtype),
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
    )


def _assert_copy_parity(backend: MotrixBackend, body_ids: np.ndarray, dtype=np.float32) -> None:
    outs = _alloc(backend.num_envs, len(body_ids), dtype=dtype)
    result = backend.copy_body_state_w(body_ids, *outs)
    assert result[0] is outs[0] and result[1] is outs[1]
    assert result[2] is outs[2] and result[3] is outs[3]
    # A second call through the same buffers pins the repeated-reuse contract:
    # the outputs are overwritten in place, never swapped or reallocated.
    again = backend.copy_body_state_w(body_ids, *outs)
    assert again[0] is outs[0] and again[1] is outs[1]
    assert again[2] is outs[2] and again[3] is outs[3]
    pos, quat, lin_vel, ang_vel = backend.get_body_state_w(body_ids)
    np.testing.assert_allclose(outs[0], pos, atol=1e-5)
    np.testing.assert_allclose(outs[1], quat, atol=1e-5)
    np.testing.assert_allclose(outs[2], lin_vel, atol=1e-4)
    np.testing.assert_allclose(outs[3], ang_vel, atol=1e-4)


def _whole_mjcf_scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(model_file=_write(tmp_path, "robot", ROBOT).model_file)


def _portable_scene(tmp_path: Path) -> SceneCfg:
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "robot",
                _write(tmp_path / "portable", "robot", ROBOT),
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, 1.0)),
            ),
        )
    )


def _drive(backend: MotrixBackend, steps: int = 4) -> None:
    rng = np.random.default_rng(0)
    for _ in range(steps):
        backend.step(rng.uniform(-1, 1, size=(backend.num_envs, 1)).astype(np.float32))


def test_whole_mjcf_copy_parity_across_steps(tmp_path: Path) -> None:
    backend = MotrixBackend(_whole_mjcf_scene(tmp_path), num_envs=3, sim_dt=0.01)
    try:
        ids = backend.get_body_ids(("base", "link"))
        _drive(backend)
        _assert_copy_parity(backend, ids)
        _drive(backend)
        _assert_copy_parity(backend, ids)
        # A permuted selection must reorder columns, not just re-copy them.
        _assert_copy_parity(backend, ids[::-1].copy())
    finally:
        backend.close()


def test_whole_mjcf_copy_supports_float64_buffers(tmp_path: Path) -> None:
    backend = MotrixBackend(_whole_mjcf_scene(tmp_path), num_envs=2, sim_dt=0.01)
    try:
        ids = backend.get_body_ids(("base", "link"))
        _drive(backend)
        _assert_copy_parity(backend, ids, dtype=np.float64)
    finally:
        backend.close()


def test_portable_copy_parity_across_steps(tmp_path: Path) -> None:
    backend = MotrixBackend(_portable_scene(tmp_path), num_envs=4, sim_dt=0.01)
    try:
        ids = backend.get_body_ids(("robot/base", "robot/link"))
        _drive(backend)
        _assert_copy_parity(backend, ids)
        _drive(backend)
        _assert_copy_parity(backend, ids)
        # A permuted selection must reorder columns on portable runtimes too.
        _assert_copy_parity(backend, ids[::-1].copy())
    finally:
        backend.close()


def test_portable_copy_supports_float64_buffers(tmp_path: Path) -> None:
    backend = MotrixBackend(_portable_scene(tmp_path), num_envs=2, sim_dt=0.01)
    try:
        ids = backend.get_body_ids(("robot/base", "robot/link"))
        _drive(backend)
        _assert_copy_parity(backend, ids, dtype=np.float64)
    finally:
        backend.close()
