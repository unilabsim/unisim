"""Physics-state playback contract tests for the IsaacGym subprocess adapter.

Every test runs without the IsaacGym SDK: host-side coverage goes through the
deterministic protocol mock worker, worker-side coverage exercises
``SceneWorker.physics_state`` with a fake context, and the construction-time
scene-shape audit is pure host MJCF parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unisim.backend.isaacgym.backend import (
    IsaacGymBackend,
    IsaacGymModelInfo,
    IsaacGymWorkerError,
)
from unisim.backend.isaacgym.scene_worker import SceneWorker
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.legacy_projection import LegacyExecutionLayout
from unisim.conformance import assert_physics_state_playback_conformance
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
from unisim.factory import create_backend
from unisim.scene import SceneCfg

_MOCK_WORKER = Path(__file__).resolve().parent / "mock_worker.py"
_SIM_DT = 0.005
_NUM_ENVS = 3

_FREE_SCENE = """<mujoco model="playback">
  <worldbody>
    <body name="base" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="link1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="link1_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
      <body name="link2" pos="0.1 0 0">
        <joint name="j2" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="link2_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20" kv="1"/>
    <position name="a2" joint="j2" kp="20" kv="1"/>
  </actuator>
</mujoco>
"""

_FIXED_BASE_SCENE = """<mujoco model="fixed_base">
  <worldbody>
    <body name="base" pos="0 0 0.3">
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="link1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="link1_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20" kv="1"/>
  </actuator>
</mujoco>
"""

_LATE_FREE_SCENE = """<mujoco model="late_free">
  <worldbody>
    <body name="pedestal">
      <geom name="pedestal_geom" type="box" size="0.1 0.1 0.1" mass="5"/>
      <body name="arm" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="sphere" size="0.05" mass="0.2"/>
      </body>
    </body>
    <body name="floater" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="floater_geom" type="sphere" size="0.05" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""

_TWO_FREE_SCENE = """<mujoco model="two_free">
  <worldbody>
    <body name="a" pos="0 0 0.3">
      <freejoint name="root_a"/>
      <geom name="a_geom" type="sphere" size="0.05" mass="0.2"/>
    </body>
    <body name="b" pos="1 0 0.3">
      <freejoint name="root_b"/>
      <geom name="b_geom" type="sphere" size="0.05" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""

_UNNAMED_JOINT_SCENE = _FREE_SCENE.replace('<joint name="j2" ', "<joint ")

_BALL_JOINT_SCENE = _FREE_SCENE.replace(
    '<joint name="j1" type="hinge"', '<joint name="j1" type="ball"'
)


def _scene_file(root: Path, xml: str, name: str = "scene.xml") -> Path:
    path = root / name
    path.write_text(xml, encoding="utf-8")
    return path


def _raw_backend(scene_file: Path, num_envs: int = _NUM_ENVS) -> IsaacGymBackend:
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=str(scene_file)),
        num_envs,
        _SIM_DT,
        base_name="base",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


def test_legacy_scene_playback_contract_end_to_end(tmp_path: Path) -> None:
    scene = _scene_file(tmp_path, _FREE_SCENE)
    backend = _raw_backend(scene)
    try:
        capabilities = backend.get_play_capabilities()
        assert capabilities.supports_physics_state_playback
        assert capabilities.supports_native_interactive_renderer
        assert capabilities.supports_native_video_capture
        assert not capabilities.supports_mocap_playback

        layout = backend.get_physics_state_layout()
        assert (layout.nq, layout.nv, layout.nmocap) == (9, 8, 0)

        backend.materialize()
        snapshot = backend.get_physics_state()
        assert snapshot.shape == (_NUM_ENVS, layout.state_width)
        parts = layout.split_state(snapshot)
        np.testing.assert_allclose(parts.time, 0.0, atol=0.0)
        # The mock worker's initial state carries unit wxyz root quaternions.
        np.testing.assert_allclose(parts.qpos[:, 3], 1.0, rtol=1e-6)

        backend.step(np.zeros((_NUM_ENVS, 2), dtype=np.float32), nsteps=4)
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.time, 4 * _SIM_DT, rtol=1e-5)

        # Selected resets restart the clock on their rows only.
        backend.reset(np.array([1], dtype=np.int32))
        parts = layout.split_state(backend.get_physics_state())
        assert parts.time[1] == 0.0
        np.testing.assert_allclose(parts.time[[0, 2]], 4 * _SIM_DT, rtol=1e-5)

        # Whole-MJCF scenes replay through the construction source.
        assert backend.get_playback_model() == str(scene)
        assert backend.get_playback_model(2) == str(scene)
        with pytest.raises(IndexError):
            backend.get_playback_model(_NUM_ENVS)
        with pytest.raises(TypeError):
            backend.get_playback_model(0.5)  # type: ignore[arg-type]

        # Restore round-trip through the worker state exchange.
        restored = backend.get_physics_state()
        backend.set_physics_state(restored)
        np.testing.assert_allclose(
            backend.get_physics_state(), restored, rtol=1e-5, atol=1e-6
        )

        assert_physics_state_playback_conformance(backend)
    finally:
        backend.close()


def test_set_physics_state_rejects_bad_shape(tmp_path: Path) -> None:
    backend = _raw_backend(_scene_file(tmp_path, _FREE_SCENE))
    try:
        layout = backend.get_physics_state_layout()
        with pytest.raises(ValueError, match="physics snapshot"):
            backend.set_physics_state(np.zeros((_NUM_ENVS, layout.state_width + 1)))
    finally:
        backend.close()


def test_handshake_rejects_native_dof_order_mismatch(tmp_path: Path) -> None:
    backend = _raw_backend(_scene_file(tmp_path, _FREE_SCENE))
    try:
        backend._model_info = IsaacGymModelInfo(
            num_dof=2,
            num_bodies=3,
            dof_names=("j2", "j1"),
            body_names=("base", "link1", "link2"),
            gravity=(0.0, 0.0, -9.81),
            use_gpu_pipeline=False,
        )
        with pytest.raises(IsaacGymWorkerError, match="differs from the MJCF source joint order"):
            backend._validate_legacy_playback_joint_order()
    finally:
        backend.close()


def test_audit_rejects_ball_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="single-DoF"):
        _raw_backend(_scene_file(tmp_path, _BALL_JOINT_SCENE))


def test_audit_rejects_multiple_free_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="at most one MJCF free joint"):
        _raw_backend(_scene_file(tmp_path, _TWO_FREE_SCENE))


def test_audit_rejects_unnamed_joints(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="named MJCF joints"):
        _raw_backend(_scene_file(tmp_path, _UNNAMED_JOINT_SCENE))


def test_fixed_base_scene_disables_playback_capability(tmp_path: Path) -> None:
    backend = _raw_backend(_scene_file(tmp_path, _FIXED_BASE_SCENE))
    try:
        capabilities = backend.get_play_capabilities()
        # Fixed-base physics still constructs and keeps native rendering;
        # only the detached playback contract is disabled fail-closed.
        assert not capabilities.supports_physics_state_playback
        assert capabilities.supports_native_interactive_renderer
        assert capabilities.supports_native_video_capture
        with pytest.raises(NotImplementedError, match="no free joint"):
            backend.get_physics_state_layout()
        with pytest.raises(NotImplementedError, match="no free joint"):
            backend.get_physics_state()
    finally:
        backend.close()


def test_late_free_joint_disables_playback_capability(tmp_path: Path) -> None:
    backend = _raw_backend(_scene_file(tmp_path, _LATE_FREE_SCENE))
    try:
        assert not backend.get_play_capabilities().supports_physics_state_playback
        with pytest.raises(NotImplementedError, match="not the first joint"):
            backend.get_physics_state_layout()
    finally:
        backend.close()


_OBJECT_XML = """<mujoco model="object">
  <worldbody>
    <body name="base">
      <freejoint/>
      <inertial mass="1" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom name="body" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_TABLE_XML = """<mujoco model="table">
  <worldbody>
    <body name="base">
      <inertial mass="10" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom name="top" type="box" size="1 1 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _mapped_scene(root: Path) -> SceneCfg:
    object_path = root / "object.xml"
    object_path.write_text(_OBJECT_XML, encoding="utf-8")
    table_path = root / "table.xml"
    table_path.write_text(_TABLE_XML, encoding="utf-8")
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(object_path)),
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.5)),
            ),
            SceneEntitySpec(
                "mirror",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
            SceneEntitySpec(
                "table",
                ModelSourceDescriptor(str(table_path)),
                kind="rigid",
                root_mode="fixed",
            ),
        ),
    )


def _mapped_backend(scene: SceneCfg, num_envs: int = _NUM_ENVS) -> IsaacGymBackend:
    backend = create_backend(
        "isaacgym",
        scene,
        num_envs,
        _SIM_DT,
        base_name="object",
        worker_command=[sys.executable, str(_MOCK_WORKER)],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


def test_mapped_scene_playback_contract_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    backend = _mapped_backend(_mapped_scene(tmp_path))
    try:
        capabilities = backend.get_play_capabilities()
        assert capabilities.supports_physics_state_playback
        # The kinematic mirror entity compiles as one mocap body in the
        # composed playback MJCF.
        assert capabilities.supports_mocap_playback

        layout = backend.get_physics_state_layout()
        assert (layout.nq, layout.nv, layout.nmocap) == (7, 6, 1)

        backend.materialize()
        snapshot = backend.get_physics_state()
        assert snapshot.shape == (_NUM_ENVS, layout.state_width)
        parts = layout.split_state(snapshot)
        assert parts.mocap_pos is not None and parts.mocap_quat is not None
        assert parts.mocap_pos.shape == (_NUM_ENVS, 1, 3)
        assert parts.mocap_quat.shape == (_NUM_ENVS, 1, 4)
        np.testing.assert_allclose(parts.mocap_quat[..., 0], 1.0, rtol=1e-6)

        mocap_pos, mocap_quat = backend.get_playback_mocap_state(0)
        assert mocap_pos.shape == (1, 3)
        assert mocap_quat.shape == (1, 4)
        with pytest.raises(IndexError):
            backend.get_playback_mocap_state(_NUM_ENVS)

        backend.step(np.zeros((_NUM_ENVS, 0), dtype=np.float32), nsteps=3)
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.time, 3 * _SIM_DT, rtol=1e-5)
        backend.reset(np.array([0, 2], dtype=np.int32))
        parts = layout.split_state(backend.get_physics_state())
        np.testing.assert_allclose(parts.time[[0, 2]], 0.0, atol=0.0)
        np.testing.assert_allclose(parts.time[1], 3 * _SIM_DT, rtol=1e-5)

        # No variant plan: every env replays through the composed scene MJCF.
        import mujoco

        model_file = backend.get_playback_model()
        composed = mujoco.MjModel.from_xml_path(model_file)
        assert (composed.nq, composed.nv, composed.nmocap) == (7, 6, 1)
        assert backend.get_playback_model(1) == model_file

        restored = backend.get_physics_state()
        backend.set_physics_state(restored)
        np.testing.assert_allclose(
            backend.get_physics_state(), restored, rtol=1e-5, atol=1e-6
        )

        assert_physics_state_playback_conformance(backend)
    finally:
        backend.close()


def test_scene_worker_physics_state_block_with_kinematic_entity(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from tests.adapters.isaacgym.scene_fixture import scene_payload

    payload = scene_payload(tmp_path)
    ctx = SimpleNamespace(protocol=protocol)
    worker = SceneWorker(ctx, payload)
    ctx.slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(worker.num_envs, worker.layout).items()
    }
    ctx.slots["qpos"][:] = worker.qpos0
    ctx.slots["qvel"][:] = worker.qvel0
    ctx.slots["entity_root_state"][:] = worker.roots0

    layout = worker.layout
    kinematic = [
        index for index, entity in enumerate(layout.entities) if entity.root_mode == "kinematic"
    ]
    assert [layout.entities[index].name for index in kinematic] == ["target"]

    reply = worker.physics_state()
    width = layout.nq + layout.nv + 7 * len(kinematic)
    assert reply["shape"] == [worker.num_envs, width]
    block = np.frombuffer(reply["state"], dtype=np.float32).reshape(worker.num_envs, width)
    np.testing.assert_allclose(block[:, : layout.nq], worker.qpos0, rtol=1e-6)
    np.testing.assert_allclose(
        block[:, layout.nq : layout.nq + layout.nv], worker.qvel0, rtol=1e-6
    )
    # The mocap tail replays the kinematic target root pose in layout order.
    np.testing.assert_allclose(block[:, -7:-4], worker.roots0[:, 3, :3], rtol=1e-6)
    np.testing.assert_allclose(block[:, -4:], worker.roots0[:, 3, 3:7], rtol=1e-6)


def test_scene_worker_physics_state_block_legacy_layout() -> None:
    layout = LegacyExecutionLayout(("j1", "j2"), ("base", "link1", "link2"))
    num_envs = 2
    slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(num_envs, layout).items()
    }
    slots["qpos"][:] = np.arange(num_envs * layout.nq, dtype=np.float32).reshape(
        num_envs, layout.nq
    )
    slots["qvel"][:] = 0.5
    worker = SceneWorker.__new__(SceneWorker)
    worker.ctx = SimpleNamespace(protocol=protocol, slots=slots)
    worker.layout = layout
    worker.num_envs = num_envs
    worker.faulted = False

    reply = worker.physics_state()
    width = layout.nq + layout.nv
    assert reply["shape"] == [num_envs, width]
    block = np.frombuffer(reply["state"], dtype=np.float32).reshape(num_envs, width)
    np.testing.assert_allclose(block[:, : layout.nq], slots["qpos"], rtol=1e-6)
    np.testing.assert_allclose(block[:, layout.nq :], 0.5, rtol=1e-6)
