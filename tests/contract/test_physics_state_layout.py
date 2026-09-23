"""Contract tests for the physics-state playback layout and conformance checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim import (
    FakeBackend,
    PhysicsStateLayout,
    assert_backend_conformance,
)
from unisim.backend.base import BackendPlayCapabilities

MODEL_XML = (
    "<mujoco><worldbody><body name='base'><joint name='slide' type='slide'/>"
    "<geom type='sphere' size='.1'/></body></worldbody>"
    "<actuator><motor joint='slide'/></actuator></mujoco>"
)


def test_state_width_without_mocap() -> None:
    layout = PhysicsStateLayout(nq=3, nv=2)
    assert layout.state_width == 1 + 3 + 2


def test_state_width_with_mocap() -> None:
    layout = PhysicsStateLayout(nq=3, nv=2, nmocap=2)
    assert layout.state_width == 1 + 3 + 2 + 14


def test_split_state_batch_without_mocap() -> None:
    layout = PhysicsStateLayout(nq=3, nv=2)
    state = np.arange(2 * layout.state_width, dtype=np.float64).reshape(2, -1)
    parts = layout.split_state(state)
    np.testing.assert_array_equal(parts.time, state[:, 0])
    np.testing.assert_array_equal(parts.qpos, state[:, 1:4])
    np.testing.assert_array_equal(parts.qvel, state[:, 4:6])
    assert parts.mocap_pos is None
    assert parts.mocap_quat is None


def test_split_state_single_row_with_mocap() -> None:
    layout = PhysicsStateLayout(nq=1, nv=1, nmocap=2)
    state = np.arange(layout.state_width, dtype=np.float64)
    parts = layout.split_state(state)
    assert parts.time.shape == ()
    assert parts.qpos.shape == (1,)
    assert parts.qvel.shape == (1,)
    assert parts.mocap_pos is not None and parts.mocap_quat is not None
    np.testing.assert_array_equal(parts.mocap_pos, state[3:9].reshape(2, 3))
    np.testing.assert_array_equal(parts.mocap_quat, state[9:17].reshape(2, 4))


def test_split_state_batch_with_mocap() -> None:
    layout = PhysicsStateLayout(nq=2, nv=2, nmocap=1)
    state = np.arange(4 * layout.state_width, dtype=np.float64).reshape(4, -1)
    parts = layout.split_state(state)
    assert parts.mocap_pos is not None and parts.mocap_quat is not None
    assert parts.mocap_pos.shape == (4, 1, 3)
    assert parts.mocap_quat.shape == (4, 1, 4)
    np.testing.assert_array_equal(parts.mocap_pos[:, 0, :], state[:, 5:8])
    np.testing.assert_array_equal(parts.mocap_quat[:, 0, :], state[:, 8:12])


def test_split_state_rejects_wrong_width() -> None:
    layout = PhysicsStateLayout(nq=3, nv=2)
    with pytest.raises(ValueError, match="physics-state snapshot must use the"):
        layout.split_state(np.zeros((2, layout.state_width + 1)))


class _PlaybackFakeBackend(FakeBackend):
    """Fake backend declaring the physics-state playback contract."""

    _play_capabilities = BackendPlayCapabilities(supports_physics_state_playback=True)

    def __init__(self, model_file: str, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._model_file = model_file
        self._time = np.zeros(self._num_envs, dtype=np.float64)

    def get_physics_state_layout(self) -> PhysicsStateLayout:
        return PhysicsStateLayout(nq=self._num_actuators, nv=self._num_actuators)

    def get_physics_state(self) -> np.ndarray:
        out = np.empty((self._num_envs, 1 + 2 * self._num_actuators))
        out[:, 0] = self._time
        out[:, 1 : 1 + self._num_actuators] = self._qpos
        out[:, 1 + self._num_actuators :] = self._qvel
        return out

    def set_physics_state(self, state: np.ndarray) -> None:
        array = np.asarray(state, dtype=np.float64)
        self._time[...] = array[:, 0]
        self._qpos[...] = array[:, 1 : 1 + self._num_actuators]
        self._qvel[...] = array[
            :, 1 + self._num_actuators : 1 + 2 * self._num_actuators
        ]

    def get_playback_model(self, env_index: int | None = None):
        del env_index
        return self._model_file


class _MocapPlaybackFakeBackend(_PlaybackFakeBackend):
    _play_capabilities = BackendPlayCapabilities(
        supports_physics_state_playback=True,
        supports_mocap_playback=True,
    )

    def get_physics_state_layout(self) -> PhysicsStateLayout:
        return PhysicsStateLayout(nq=self._num_actuators, nv=self._num_actuators, nmocap=1)

    def get_physics_state(self) -> np.ndarray:
        out = np.empty((self._num_envs, self.get_physics_state_layout().state_width))
        out[:, 0] = self._time
        out[:, 1 : 1 + self._num_actuators] = self._qpos
        out[:, 1 + self._num_actuators : 1 + 2 * self._num_actuators] = self._qvel
        base = 1 + 2 * self._num_actuators
        out[:, base : base + 3] = 0.5
        out[:, base + 3 :] = (1.0, 0.0, 0.0, 0.0)
        return out

    def get_playback_mocap_state(self, env_index: int = 0):
        if env_index < 0 or env_index >= self._num_envs:
            raise IndexError("fake playback environment index is out of range")
        return (
            np.full((1, 3), 0.5),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        )


class _UndeclaredMocapFakeBackend(_MocapPlaybackFakeBackend):
    _play_capabilities = BackendPlayCapabilities(supports_physics_state_playback=True)


def _write_model(tmp_path: Path) -> str:
    model = tmp_path / "playback.xml"
    model.write_text(MODEL_XML)
    return str(model)


def test_conformance_accepts_playback_contract(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    backend = _PlaybackFakeBackend(_write_model(tmp_path), num_envs=2, num_actuators=1)
    assert_backend_conformance(backend)


def test_conformance_accepts_mocap_playback_contract(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    backend = _MocapPlaybackFakeBackend(_write_model(tmp_path), num_envs=2, num_actuators=1)
    assert_backend_conformance(backend)


def test_conformance_rejects_snapshot_layout_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    backend = _PlaybackFakeBackend(_write_model(tmp_path), num_envs=2, num_actuators=1)

    def bad_layout() -> PhysicsStateLayout:
        return PhysicsStateLayout(nq=backend.num_actuators + 1, nv=backend.num_actuators)

    backend.get_physics_state_layout = bad_layout  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="physics-state snapshot must have shape"):
        assert_backend_conformance(backend)


def test_conformance_requires_declared_mocap_playback(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    backend = _UndeclaredMocapFakeBackend(_write_model(tmp_path), num_envs=2, num_actuators=1)
    with pytest.raises(AssertionError, match="supports_mocap_playback"):
        assert_backend_conformance(backend)


def test_conformance_rejects_unloadable_playback_model(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    model = tmp_path / "broken.xml"
    model.write_text("<mujoco><worldbody><body></mujoco>")
    backend = _PlaybackFakeBackend(str(model), num_envs=2, num_actuators=1)
    with pytest.raises(ValueError):
        assert_backend_conformance(backend)
