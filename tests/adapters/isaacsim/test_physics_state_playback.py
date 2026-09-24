"""IsaacSim physics-state playback contract tests (#312)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from unisim import create_backend
from unisim.backend import mjcf_layout
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.subprocess_ipc import protocol
from unisim.conformance import (
    assert_backend_conformance,
    assert_physics_state_playback_conformance,
)
from unisim.dr.types import FixedVariantPlan, ModelSourceDescriptor
from unisim.entities import EntityInitialState, EntityVariantBinding, SceneEntitySpec
from unisim.scene import SceneCfg

SIM_DT = 0.002
NUM_ENVS = 5


def scene(tmp_path: Path) -> SceneCfg:
    """Author the mapped entity scene (fixed robot, variant object, kinematic mirror)."""
    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base"><geom name="base_collision" size=".1" mass="1"/>'
        '<body name="tip"><joint name="hinge" damping="0"/>'
        '<geom size=".1" mass="1"/></body></body></worldbody>'
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator></mujoco>'
    )
    objects = []
    for index, (mass, radius) in enumerate(((1, ".1"), (3, ".15"))):
        source = tmp_path / f"object-{index}.xml"
        source.write_text(
            '<mujoco><worldbody><body name="base"><freejoint/>'
            f'<geom type="sphere" size="{radius}" mass="{mass}" '
            f'friction="{0.7 - index * 0.3} {0.2 - index * 0.1} {0.03 - index * 0.01}"/>'
            "</body></worldbody></mujoco>"
        )
        objects.append(ModelSourceDescriptor(str(source)))
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
            SceneEntitySpec(
                "object",
                objects[0],
                kind="rigid",
                initial_state=EntityInitialState(position=(0.0, 0.0, 1.0)),
            ),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                mirror_of="object",
                collision_enabled=False,
                initial_state=EntityInitialState(position=(2.0, 0.0, 1.0)),
            ),
        ),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([1, 1, 0, 1, 0]), tuple(objects))
        ),
    )

ROBOT = """<mujoco>
  <worldbody><body name="base"><geom name="base_collision" size=".1" mass="1"/>
    <body name="tip"><joint name="hinge"/>
      <geom size=".1" mass="1"/></body></body></worldbody>
  <actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator>
</mujoco>"""

LEGACY_ROBOT = """<mujoco>
  <worldbody><body name="base" pos="0 0 1"><freejoint name="root"/>
    <geom name="base_collision" size=".1" mass="1"/>
    <body name="tip" pos="0 0 .2"><joint name="hinge" axis="0 1 0"/>
      <geom size=".1" mass="1"/></body></body></worldbody>
  <actuator><position name="drive" joint="hinge" kp="20" kv="2"/></actuator>
</mujoco>"""


def _mapped_backend(tmp_path: Path) -> IsaacSimBackend:
    return IsaacSimBackend(scene(tmp_path), NUM_ENVS, SIM_DT)


def _single_entity_backend(tmp_path: Path) -> IsaacSimBackend:
    robot = tmp_path / "robot.xml"
    robot.write_text(ROBOT, encoding="utf-8")
    config = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
        )
    )
    return IsaacSimBackend(config, 2, SIM_DT)


def _fake_materialized(backend: IsaacSimBackend) -> IsaacSimBackend:
    """Mark a constructed backend as materialized with zeroed slots; no worker."""
    assert backend._entity_scene is not None
    backend._model_info = object()
    backend._worker_dead_error = None
    backend._slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.scene_slot_shapes(
            backend._num_envs, backend._entity_scene.layout
        ).items()
    }
    backend._request = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    return backend


def _legacy_backend(tmp_path: Path, xml: str = LEGACY_ROBOT, num_envs: int = 3) -> IsaacSimBackend:
    model_file = tmp_path / "legacy.xml"
    model_file.write_text(xml, encoding="utf-8")
    return IsaacSimBackend(SceneCfg(model_file=str(model_file)), num_envs, SIM_DT)


def _fake_materialized_legacy(
    backend: IsaacSimBackend, num_dof: int = 1, num_bodies: int = 2
) -> IsaacSimBackend:
    """Mark a constructed legacy backend as materialized with zeroed slots."""
    assert backend._entity_scene is None
    backend._model_info = SimpleNamespace(num_dof=num_dof)
    backend._worker_dead_error = None
    backend._slots = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.slot_shapes(
            backend._num_envs, num_dof, num_bodies
        ).items()
    }
    backend._request = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    return backend


def _zero_ctrl(backend: IsaacSimBackend) -> np.ndarray:
    return np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)


def test_play_capabilities_declare_playback_on_mapped_scene(tmp_path: Path) -> None:
    backend = _mapped_backend(tmp_path)
    try:
        capabilities = backend.get_play_capabilities()
        assert capabilities.supports_physics_state_playback
        # The native Kit viewer and camera paths stay available alongside viser.
        assert capabilities.supports_native_interactive_renderer
        assert capabilities.supports_native_video_capture
        assert not capabilities.supports_mocap_playback
    finally:
        backend.close()


def test_legacy_fixed_base_scene_stays_fail_closed(tmp_path: Path) -> None:
    # A fixed-base MJCF cannot replay the synthetic 7/6-root legacy wire, so
    # only the playback capability fails closed; the scene itself constructs.
    model_file = tmp_path / "model.xml"
    model_file.write_text(ROBOT, encoding="utf-8")
    backend = IsaacSimBackend(SceneCfg(model_file=str(model_file)), 2, SIM_DT)
    try:
        assert not backend.get_play_capabilities().supports_physics_state_playback
        with pytest.raises(NotImplementedError):
            backend.get_physics_state_layout()
        with pytest.raises(NotImplementedError):
            backend.get_physics_state()
        with pytest.raises(NotImplementedError):
            backend.set_physics_state(np.zeros((2, 3), dtype=np.float32))
    finally:
        backend.close()


def test_legacy_floating_scene_declares_playback(tmp_path: Path) -> None:
    backend = _legacy_backend(tmp_path)
    try:
        capabilities = backend.get_play_capabilities()
        assert capabilities.supports_physics_state_playback
        assert capabilities.supports_native_interactive_renderer
        assert capabilities.supports_native_video_capture
        assert not capabilities.supports_mocap_playback
        layout = backend.get_physics_state_layout()
        # 7/6 synthetic floating root plus one scalar hinge.
        assert (layout.nq, layout.nv, layout.nmocap) == (8, 7, 0)
    finally:
        backend.close()


def test_legacy_joint_order_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_extract = mjcf_layout.extract_mjcf_joint_layout

    def dropping_extract(model_file: str):
        return real_extract(model_file)[:-1]

    monkeypatch.setattr(mjcf_layout, "extract_mjcf_joint_layout", dropping_extract)
    backend = _legacy_backend(tmp_path)
    try:
        # The drift disables only playback; the scene still constructs and the
        # contract entry points fail closed through the base class.
        assert not backend.get_play_capabilities().supports_physics_state_playback
        with pytest.raises(NotImplementedError):
            backend.get_physics_state_layout()
    finally:
        backend.close()


def test_legacy_get_playback_model_returns_construction_source(tmp_path: Path) -> None:
    backend = _legacy_backend(tmp_path)
    try:
        model_file = backend.get_playback_model()
        assert isinstance(model_file, str)
        assert model_file == str(tmp_path / "legacy.xml")
        model = mujoco.MjModel.from_xml_path(model_file)
        assert (model.nq, model.nv) == (8, 7)
        # An explicit in-range env_index returns the same single source.
        assert backend.get_playback_model(2) == model_file
        with pytest.raises(IndexError):
            backend.get_playback_model(3)
        with pytest.raises(TypeError):
            backend.get_playback_model(0.5)  # type: ignore[arg-type]
    finally:
        backend.close()


def test_legacy_get_physics_state_assembles_time_qpos_qvel(tmp_path: Path) -> None:
    backend = _fake_materialized_legacy(_legacy_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        root_state = backend._slots["root_state"]
        dof_state = backend._slots["dof_state"]
        # 120-degree rotation about (1, 1, 1): x->y, y->z, z->x.
        root_state[:, 3:7] = (0.5, 0.5, 0.5, 0.5)
        root_state[:, :3] = np.arange(3, dtype=np.float32)
        root_state[:, 7:10] = (0.1, -0.2, 0.3)
        root_state[:, 10:13] = (1.0, 2.0, 3.0)
        dof_state[:, :, 0] = 0.7
        dof_state[:, :, 1] = -0.8
        backend._time_view[:] = np.linspace(0.0, 1.0, backend.num_envs)
        snapshot = backend.get_physics_state()
        assert snapshot.shape == (backend.num_envs, layout.state_width)
        assert snapshot.dtype == np.float32
        parts = layout.split_state(snapshot)
        np.testing.assert_allclose(parts.time, backend._time_view, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(parts.qpos[:, :7], root_state[:, :7])
        np.testing.assert_array_equal(parts.qpos[:, 7:], dof_state[:, :, 0])
        np.testing.assert_array_equal(parts.qvel[:, :3], root_state[:, 7:10])
        # The world-frame angular velocity rotated into the body frame.
        np.testing.assert_allclose(
            parts.qvel[:, 3:6],
            np.tile((2.0, 3.0, 1.0), (backend.num_envs, 1)),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(parts.qvel[:, 6:], dof_state[:, :, 1])
        # The body-frame block rotates back to the published world velocity.
        np.testing.assert_allclose(
            protocol.quat_rotate(root_state[:, 3:7], parts.qvel[:, 3:6]),
            root_state[:, 10:13],
            rtol=0,
            atol=1e-6,
        )
    finally:
        backend.close()


def test_legacy_step_reset_and_set_state_drive_the_clock(tmp_path: Path) -> None:
    backend = _fake_materialized_legacy(_legacy_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        backend.step(np.zeros((backend.num_envs, 1), dtype=np.float32), nsteps=3)
        np.testing.assert_allclose(backend._time_view, 3 * SIM_DT, rtol=0, atol=1e-12)
        backend.reset(np.array([1], dtype=np.int32))
        expected = np.full(backend.num_envs, 3 * SIM_DT)
        expected[1] = 0.0
        np.testing.assert_allclose(backend._time_view, expected, rtol=0, atol=1e-12)
        rows = np.array([0, 2], dtype=np.intp)
        qpos = np.zeros((2, layout.nq), dtype=np.float32)
        qpos[:, 3] = 1.0
        backend.set_state(rows, qpos, np.zeros((2, layout.nv), dtype=np.float32))
        np.testing.assert_array_equal(backend._time_view, np.zeros(backend.num_envs))
        np.testing.assert_array_equal(backend._slots["reset_qpos"][:2], qpos)
    finally:
        backend.close()


def test_legacy_set_physics_state_restores_snapshot_and_clock(tmp_path: Path) -> None:
    backend = _fake_materialized_legacy(_legacy_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        root_state = backend._slots["root_state"]
        root_state[:, 3] = 1.0
        root_state[:, 7:10] = (0.1, 0.2, 0.3)
        root_state[:, 10:13] = (0.4, -0.5, 0.6)
        backend._slots["dof_state"][:, :, 0] = 0.7
        backend._slots["dof_state"][:, :, 1] = -0.8
        backend.step(np.zeros((backend.num_envs, 1), dtype=np.float32), nsteps=5)
        snapshot = backend.get_physics_state()
        # Move the fake state and clock elsewhere, then restore the snapshot.
        root_state[:] = 0.0
        backend._slots["dof_state"][:] = 0.0
        backend._time_view[:] = 0.0
        backend.set_physics_state(snapshot)
        np.testing.assert_allclose(backend._time_view, snapshot[:, 0], rtol=0, atol=1e-12)
        # The reset upload carried the snapshot's qpos/qvel rows verbatim; the
        # worker would apply them and refresh the published slots.
        np.testing.assert_array_equal(
            backend._slots["reset_qpos"][: backend.num_envs], snapshot[:, 1 : 1 + layout.nq]
        )
        np.testing.assert_array_equal(
            backend._slots["reset_qvel"][: backend.num_envs], snapshot[:, 1 + layout.nq :]
        )
        with pytest.raises(ValueError, match="layout with shape"):
            backend.set_physics_state(np.zeros((backend.num_envs, layout.state_width - 1)))
    finally:
        backend.close()


def test_physics_state_layout_matches_scene_layout(tmp_path: Path) -> None:
    backend = _mapped_backend(tmp_path)
    try:
        layout = backend.get_physics_state_layout()
        scene_layout = backend.get_scene_layout()
        assert (layout.nq, layout.nv) == (scene_layout.nq, scene_layout.nv) == (8, 7)
        # Kinematic mocap mirroring remains phase 2, so no mocap tail yet.
        assert layout.nmocap == 0
        assert layout.state_width == 1 + 8 + 7
    finally:
        backend.close()


def test_get_playback_model_fixed_variant_requires_env_index(tmp_path: Path) -> None:
    backend = _mapped_backend(tmp_path)
    try:
        with pytest.raises(ValueError, match="env_index"):
            backend.get_playback_model()
        with pytest.raises(IndexError):
            backend.get_playback_model(NUM_ENVS)
        with pytest.raises(TypeError):
            backend.get_playback_model(1.5)  # type: ignore[arg-type]
        assignment = backend._entity_scene.owner.variant_plan.assignment  # type: ignore[union-attr]
        scene_layout = backend.get_scene_layout()
        for env_index in range(NUM_ENVS):
            model_file = backend.get_playback_model(env_index)
            expected = backend._entity_scene.owner.variant_plan.variants[  # type: ignore[union-attr]
                int(assignment[env_index])
            ].model_file
            assert model_file == expected
            model = mujoco.MjModel.from_xml_path(model_file)
            assert (model.nq, model.nv) == (scene_layout.nq, scene_layout.nv)
    finally:
        backend.close()


def test_get_playback_model_single_source_scene(tmp_path: Path) -> None:
    backend = _single_entity_backend(tmp_path)
    try:
        model_file = backend.get_playback_model()
        assert isinstance(model_file, str)
        model = mujoco.MjModel.from_xml_path(model_file)
        assert (model.nq, model.nv) == (1, 1)
        # An explicit in-range env_index returns the same single source.
        assert backend.get_playback_model(1) == model_file
    finally:
        backend.close()


def test_construction_fail_closed_on_joint_order_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_extract = mjcf_layout.extract_mjcf_joint_layout

    def dropping_extract(model_file: str):
        return real_extract(model_file)[:-1]

    monkeypatch.setattr(mjcf_layout, "extract_mjcf_joint_layout", dropping_extract)
    with pytest.raises(RuntimeError, match="joint inventory"):
        _mapped_backend(tmp_path)


def test_get_physics_state_assembles_time_qpos_qvel(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        backend._slots["qpos"][:] = np.arange(NUM_ENVS * layout.nq, dtype=np.float32).reshape(
            NUM_ENVS, layout.nq
        )
        backend._slots["qvel"][:] = -np.arange(NUM_ENVS * layout.nv, dtype=np.float32).reshape(
            NUM_ENVS, layout.nv
        )
        backend._time_view[:] = np.linspace(0.0, 1.0, NUM_ENVS)
        snapshot = backend.get_physics_state()
        assert snapshot.shape == (NUM_ENVS, layout.state_width)
        assert snapshot.dtype == np.float32
        parts = layout.split_state(snapshot)
        np.testing.assert_allclose(parts.time, backend._time_view, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(parts.qpos, backend._slots["qpos"])
        np.testing.assert_array_equal(parts.qvel, backend._slots["qvel"])
        assert parts.mocap_pos is None and parts.mocap_quat is None
    finally:
        backend.close()


def test_step_advances_playback_clock(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        backend.step(_zero_ctrl(backend), nsteps=3)
        np.testing.assert_allclose(backend._time_view, 3 * SIM_DT, rtol=0, atol=1e-12)
        backend.step(_zero_ctrl(backend))
        np.testing.assert_allclose(backend._time_view, 4 * SIM_DT, rtol=0, atol=1e-12)
        # The snapshot time column serves the same accumulated clock.
        np.testing.assert_allclose(
            backend.get_physics_state()[:, 0], 4 * SIM_DT, rtol=0, atol=1e-7
        )
    finally:
        backend.close()


def test_reset_zeroes_playback_clock_rows(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        backend.step(_zero_ctrl(backend), nsteps=2)
        backend.reset(np.array([1, 3], dtype=np.int32))
        expected = np.full(NUM_ENVS, 2 * SIM_DT)
        expected[[1, 3]] = 0.0
        np.testing.assert_allclose(backend._time_view, expected, rtol=0, atol=1e-12)
        backend.reset()
        np.testing.assert_array_equal(backend._time_view, np.zeros(NUM_ENVS))
    finally:
        backend.close()


def test_set_state_zeroes_playback_clock_rows(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        backend.step(_zero_ctrl(backend), nsteps=2)
        rows = np.array([0, 4], dtype=np.intp)
        qpos = np.zeros((2, layout.nq), dtype=np.float32)
        # The floating "object" root occupies qpos columns 1..7 (pos, quat).
        qpos[:, 4] = 1.0
        backend.set_state(rows, qpos, np.zeros((2, layout.nv), dtype=np.float32))
        expected = np.full(NUM_ENVS, 2 * SIM_DT)
        expected[rows] = 0.0
        np.testing.assert_allclose(backend._time_view, expected, rtol=0, atol=1e-12)
    finally:
        backend.close()


def test_set_physics_state_restores_snapshot_and_clock(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        # The floating "object" root needs a unit quaternion in the fake slots.
        backend._slots["qpos"][:, 4] = 1.0
        backend.step(_zero_ctrl(backend), nsteps=5)
        snapshot = backend.get_physics_state()
        # Move the fake state and clock elsewhere, then restore the snapshot.
        backend._slots["qpos"][:] = 1.0
        backend._slots["qvel"][:] = 2.0
        backend._time_view[:] = 0.0
        backend.set_physics_state(snapshot)
        np.testing.assert_allclose(backend._time_view, snapshot[:, 0], rtol=0, atol=1e-12)
        # The reset upload carried the snapshot's qpos/qvel rows verbatim; the
        # worker would apply them and refresh the public slots.
        np.testing.assert_array_equal(
            backend._slots["reset_qpos"], snapshot[:, 1 : 1 + layout.nq]
        )
        np.testing.assert_array_equal(backend._slots["reset_qvel"], snapshot[:, 1 + layout.nq :])
    finally:
        backend.close()


def test_set_physics_state_rejects_bad_shape(tmp_path: Path) -> None:
    backend = _fake_materialized(_mapped_backend(tmp_path))
    try:
        layout = backend.get_physics_state_layout()
        with pytest.raises(ValueError, match="layout with shape"):
            backend.set_physics_state(np.zeros((NUM_ENVS, layout.state_width - 1)))
        with pytest.raises(ValueError, match="layout with shape"):
            backend.set_physics_state(np.zeros((NUM_ENVS + 1, layout.state_width)))
    finally:
        backend.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_SCENE=1 for IsaacSim vendor acceptance",
)
def test_isaacsim_physics_state_playback_native_acceptance(tmp_path: Path) -> None:
    """Run the conformance playback contract against the real Kit worker."""
    backend = create_backend(
        "isaacsim", scene(tmp_path), num_envs=NUM_ENVS, sim_dt=SIM_DT
    )
    try:
        assert_backend_conformance(backend)
        assert_physics_state_playback_conformance(backend)
        layout = backend.get_physics_state_layout()
        backend.reset()
        backend.step(np.zeros((NUM_ENVS, backend.num_actuators), dtype=np.float32), nsteps=2)
        snapshot = backend.get_physics_state()
        np.testing.assert_allclose(snapshot[:, 0], 2 * SIM_DT, rtol=0, atol=1e-6)
        parts = layout.split_state(snapshot)
        state = backend.get_state()
        np.testing.assert_array_equal(parts.qpos, state["qpos"])
        np.testing.assert_array_equal(parts.qvel, state["qvel"])
        backend.reset()
        np.testing.assert_array_equal(
            backend.get_physics_state()[:, 0], np.zeros(NUM_ENVS, dtype=np.float32)
        )
    finally:
        backend.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_SCENE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_SCENE=1 for IsaacSim vendor acceptance",
)
def test_isaacsim_legacy_physics_state_playback_native_acceptance(tmp_path: Path) -> None:
    """Run the playback contract against the real Kit worker on the legacy path."""
    model_file = tmp_path / "legacy.xml"
    model_file.write_text(LEGACY_ROBOT, encoding="utf-8")
    backend = create_backend(
        "isaacsim",
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=SIM_DT,
        base_name="base",
        worker_timeout_s=300,
    )
    try:
        assert_backend_conformance(backend)
        assert_physics_state_playback_conformance(backend)
        layout = backend.get_physics_state_layout()
        assert (layout.nq, layout.nv) == (8, 7)
        backend.reset()
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float32), nsteps=2)
        snapshot = backend.get_physics_state()
        np.testing.assert_allclose(snapshot[:, 0], 2 * SIM_DT, rtol=0, atol=1e-6)
        # The snapshot feeds straight back through the legacy set_state wire.
        backend.set_physics_state(snapshot)
        np.testing.assert_allclose(
            backend.get_physics_state(), snapshot, rtol=1e-5, atol=1e-6
        )
        backend.reset()
        np.testing.assert_array_equal(
            backend.get_physics_state()[:, 0], np.zeros(2, dtype=np.float32)
        )
    finally:
        backend.close()
