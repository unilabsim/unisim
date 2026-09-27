"""Packed host-bridge tensor lifecycle tests for Drake."""

from __future__ import annotations

import time
import weakref
from typing import Any

import numpy as np
import pytest

from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorProcessTopology,
)
from unisim.backend.drake.backend import (
    DrakeBackend,
    _DrakeRuntimeGroup,
    _DrakeUniModelView,
)

torch = pytest.importorskip("torch")


class _FakeRuntime:
    def __init__(self, count: int) -> None:
        self.count = count
        self.closed = False

    def _full_state(self, rows: np.ndarray) -> np.ndarray:
        state = np.zeros((rows.size, 3), dtype=np.float64)
        state[:, 1] = np.arange(rows.size, dtype=np.float64) + 10.0
        state[:, 2] = np.arange(rows.size, dtype=np.float64) + 20.0
        return state

    def step(self, ctrl: np.ndarray, nsteps: int, forces: Any = None) -> dict[str, Any]:
        del forces
        rows = np.arange(self.count, dtype=np.int32)
        state = self._full_state(rows)
        state[:, 1] += 0.1 * int(nsteps)
        state[:, 2] += 0.2 * int(nsteps)
        return {"state": state, "sensor_data": np.stack((state[:, 1], state[:, 2]), axis=1)}

    def reset(self, rows: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, Any]:
        state = np.zeros((rows.size, 3), dtype=np.float64)
        state[:, 1] = qpos[:, 0]
        state[:, 2] = qvel[:, 0]
        return {
            "env_ids": rows,
            "state": state,
            "sensor_data": np.stack((qpos[:, 0], qvel[:, 0]), axis=1),
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def backend():
    result = DrakeBackend.__new__(DrakeBackend)
    result._entity_closed = False
    result._entity_faulted = False
    result._composed_scene = None
    result._entity_layout = None
    result._host_bridge_plans = weakref.WeakSet()
    result._direct_host_bridge_plan = None
    result._runtime_groups = ()
    result._runtime = None
    result._variant_assignment = None
    result._pre_step_control_fn = None
    result._pending_body_forces = np.zeros((3, 1, 3), dtype=np.float64)

    runtime = _FakeRuntime(3)
    group = _DrakeRuntimeGroup(0, (0, 1, 2), runtime)
    result._runtime_groups = (group,)
    result._runtime = runtime
    result._num_envs = 3
    result._sim_dt = 0.002
    result._model = _DrakeUniModelView(nq=1, nv=1, nu=1)
    result._sensor_names = ("angle", "speed")
    result._sensor_adr = np.asarray((0, 1), dtype=np.int32)
    result._sensor_dim = np.asarray((1, 1), dtype=np.int32)
    result._sensor_views = {}

    result._physics_state = np.zeros((3, 3), dtype=np.float64)
    result._physics_state[:, 1] = (0.1, 0.2, 0.3)
    result._physics_state[:, 2] = (-0.1, -0.2, -0.3)
    result._sensor_data = result._physics_state[:, 1:].copy()
    result._num_bodies = 1
    result._rebuild_sensor_views()
    try:
        yield result
    finally:
        result.close()


def test_tensor_capability_matrix_is_narrow_and_fail_closed(backend):
    capabilities = backend.get_tensor_capabilities()
    assert backend.tensor_execution() is TensorExecution.HOST_BRIDGE
    assert capabilities.execution is TensorExecution.HOST_BRIDGE
    assert capabilities.state_fields == frozenset({"qpos", "qvel"})
    assert capabilities.sensor_views
    assert capabilities.stepping
    assert capabilities.selected_reset
    assert capabilities.packed_host_bridge
    assert capabilities.process_topology is TensorProcessTopology.IN_PROCESS
    assert capabilities.data_plane is TensorDataPlane.HOST_BRIDGE
    assert capabilities.torch_devices == ("cpu", "cuda")
    assert not capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert not capabilities.host_pre_step_control


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_packed_lifecycle_and_semantic_transfer_counts(backend, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos", "qvel"),
            sensor_names=("angle", "speed"),
            device=device,
        )
    )
    assert plan.transfer_stats == {
        "d2h_count": 0,
        "h2d_count": 0,
        "d2h_bytes": 0,
        "h2d_bytes": 0,
        "synchronization_count": 0,
    }

    plan.write_control(torch.zeros((3, 1), dtype=torch.float32, device=device))
    plan.step(2)
    views = plan.read_state_sensors()
    expected_state = backend.get_state_views()
    np.testing.assert_allclose(
        views["qpos"].detach().cpu().numpy(), expected_state["qpos"].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        views["qvel"].detach().cpu().numpy(), expected_state["qvel"].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        views["angle"].detach().cpu().numpy(),
        backend.get_sensor_data("angle"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        views["speed"].detach().cpu().numpy(),
        backend.get_sensor_data("speed"),
        atol=1e-6,
    )
    first_pointers = {name: view.data_ptr() for name, view in views.items()}
    plan.read_state_sensors()
    assert {name: view.data_ptr() for name, view in views.items()} == first_pointers

    state = backend.get_state_views()
    rows = torch.tensor([1, 0], dtype=torch.int64, device=device)
    qpos = state["qpos"][[1, 0]].to(device=device).clone()
    qvel = state["qvel"][[1, 0]].to(device=device).clone()
    qpos[:, 0] = torch.tensor((0.4, -0.2), device=device)
    qvel[:, 0] = torch.tensor((1.5, -0.7), device=device)
    plan.apply_reset(rows, qpos, qvel)
    selected_rows = rows.clone()
    rows.fill_(-1)
    updated = plan.read_selected_state_sensors()
    np.testing.assert_allclose(
        updated["qpos"].cpu().numpy()[selected_rows.cpu()], qpos.cpu(), atol=1e-6
    )
    np.testing.assert_allclose(
        updated["qvel"].cpu().numpy()[selected_rows.cpu()], qvel.cpu(), atol=1e-6
    )
    np.testing.assert_allclose(
        updated["angle"].cpu().numpy()[selected_rows.cpu()], qpos.cpu(), atol=1e-6
    )
    np.testing.assert_allclose(
        updated["speed"].cpu().numpy()[selected_rows.cpu()], qvel.cpu(), atol=1e-6
    )
    assert updated["qpos"].shape == (3, 1)

    stats = plan.transfer_stats
    reset_width = 1 + backend.model.nq + backend.model.nv
    assert stats["d2h_count"] == 2
    assert stats["h2d_count"] == 3
    assert stats["d2h_bytes"] == 3 * 4 + 2 * reset_width * 4
    assert stats["h2d_bytes"] == 3 * 16 * 2 + 2 * 16
    assert stats["synchronization_count"] == (5 if device.type == "cuda" else 0)

    before = dict(plan.transfer_stats)
    result = plan.apply_reset(
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0, backend.model.nq), dtype=torch.float32, device=device),
        torch.empty((0, backend.model.nv), dtype=torch.float32, device=device),
    )
    assert result == {"timing": {}}
    assert plan.transfer_stats == before


def test_packed_compile_and_hot_path_capabilities_fail_closed(backend, monkeypatch):
    with pytest.raises(ValueError, match="exactly qpos and qvel"):
        backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos",), sensor_names=("angle",), device="cpu")
        )
    with pytest.raises(KeyError, match="unknown Drake sensor"):
        backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=("missing",), device="cpu")
        )
    monkeypatch.setattr(backend, "_variant_assignment", np.zeros(backend.num_envs), raising=False)
    with pytest.raises(NotImplementedError, match="fixed variants"):
        backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel"), device="cpu"))
    monkeypatch.setattr(backend, "_variant_assignment", None, raising=False)

    plan = backend.compile_host_bridge_io(
        TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=("angle",), device="cpu")
    )
    monkeypatch.setattr(backend, "_pre_step_control_fn", lambda state, ctrl: ctrl)
    with pytest.raises(NotImplementedError, match="pre-step control"):
        plan.write_control(torch.zeros((3, 1), dtype=torch.float32))
    with pytest.raises(NotImplementedError, match="pre-step control"):
        backend.step_tensor(torch.zeros((3, 1), dtype=torch.float32))
    with pytest.raises(NotImplementedError, match="randomization"):
        plan.apply_reset(
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, backend.model.nq), dtype=torch.float32),
            torch.zeros((1, backend.model.nv), dtype=torch.float32),
            randomization=object(),
        )
    with pytest.raises(NotImplementedError, match="randomization"):
        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, backend.model.nq), dtype=torch.float32),
            torch.zeros((1, backend.model.nv), dtype=torch.float32),
            randomization=object(),
        )
    monkeypatch.setattr(backend, "_pre_step_control_fn", None, raising=False)

    before = backend.get_state_views()
    with pytest.raises(ValueError, match="unique values"):
        plan.apply_reset(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.zeros((2, backend.model.nq), dtype=torch.float32),
            torch.zeros((2, backend.model.nv), dtype=torch.float32),
        )
    after = backend.get_state_views()
    np.testing.assert_allclose(after["qpos"].numpy(), before["qpos"].numpy())

    backend.apply_body_force(np.asarray((0,)), np.ones((3, 1, 3)))
    with pytest.raises(NotImplementedError, match="interval randomization"):
        plan.step(1)
    backend._pending_body_forces.fill(0.0)


def test_packed_plan_close_releases_staging_and_fails_closed(backend):
    plan = backend.compile_host_bridge_io(
        TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=("angle",), device="cpu")
    )
    buffers = plan._buffers()
    host_ref = weakref.ref(buffers.host_packet)
    device_ref = weakref.ref(buffers.device_packet)
    del buffers

    plan.close()
    plan.close()

    assert host_ref() is None
    assert device_ref() is None
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.write_control(torch.zeros((backend.num_envs, backend.num_actuators)))


def test_backend_close_closes_live_host_bridge_plan(backend):
    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel"), device="cpu"))
    runtime = backend._runtime
    backend.close()
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.step(1)
    assert runtime.closed


def test_direct_tensor_apis_match_numpy_state(backend):
    state = backend.get_state_views(("qpos", "qvel"))
    angle = backend.get_sensor_view("angle")
    assert all(isinstance(value, torch.Tensor) for value in (*state.values(), angle))
    expected = {"qpos": backend._state_qpos(), "qvel": backend._state_qvel()}
    for name in state:
        np.testing.assert_allclose(state[name].numpy(), expected[name], atol=1e-6)
    np.testing.assert_allclose(angle.numpy(), backend.get_sensor_data("angle"), atol=1e-6)

    before = {"qpos": backend._state_qpos(), "qvel": backend._state_qvel()}
    qpos = before["qpos"][[1, 2]].copy()
    qvel = before["qvel"][[1, 2]].copy()
    qpos[:, 0] = (0.3, -0.4)
    qvel[:, 0] = (1.2, -0.6)
    backend.set_state_tensor(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor(qpos, dtype=torch.float32),
        torch.tensor(qvel, dtype=torch.float32),
    )
    after = {"qpos": backend._state_qpos(), "qvel": backend._state_qvel()}
    np.testing.assert_allclose(after["qpos"][[1, 2]], qpos, atol=1e-6)
    np.testing.assert_allclose(after["qvel"][[1, 2]], qvel, atol=1e-6)
    np.testing.assert_allclose(after["qpos"][0], before["qpos"][0])

    result = backend.step_tensor(torch.full((3, 1), 0.2, dtype=torch.float32))
    assert "tensor_control_packed_d2h_ms" in result["timing"]
    direct_plan = backend._direct_host_bridge_plan
    backend.step_tensor(torch.full((3, 1), 0.3, dtype=torch.float32))
    assert backend._direct_host_bridge_plan is direct_plan
    empty = backend.set_state_tensor(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0, backend.model.nq), dtype=torch.float32),
        torch.empty((0, backend.model.nv), dtype=torch.float32),
    )
    assert empty == {"timing": {}}
    with pytest.raises(KeyError, match="only qpos and qvel"):
        backend.get_state_views(("qpos", "ctrl"))


def test_packed_reset_validation_is_bounded_timed_and_producer_owned(backend, monkeypatch):
    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
    finite_ctrl = torch.zeros((backend.num_envs, backend.num_actuators), dtype=torch.float32)
    plan.write_control(finite_ctrl)
    before_stats = dict(plan.transfer_stats)
    nonfinite_ctrl = finite_ctrl.clone()
    nonfinite_ctrl[0, 0] = torch.nan
    plan.write_control(nonfinite_ctrl)
    assert plan.transfer_stats["d2h_count"] == before_stats["d2h_count"] + 1
    assert plan.last_timing["tensor_control_packed_d2h_bytes"] == float(nonfinite_ctrl.numel() * 4)
    plan.write_control(finite_ctrl)

    before_state = backend.get_state_views()
    nan_qpos = before_state["qpos"].clone()
    nan_qvel = before_state["qvel"].clone()
    nan_qpos[1, 0] = torch.nan
    result = plan.apply_reset(torch.tensor([1], dtype=torch.int64), nan_qpos[[1]], nan_qvel[[1]])
    assert result is not None
    assert torch.isnan(backend.get_state_views()["qpos"][1, 0])
    after_nan_state = backend.get_state_views()

    before_stats = dict(plan.transfer_stats)
    with pytest.raises(ValueError, match=r"in \[0"):
        plan.apply_reset(
            torch.tensor([-1], dtype=torch.int64),
            torch.zeros((1, backend.model.nq), dtype=torch.float32),
            torch.zeros((1, backend.model.nv), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="unique values"):
        plan.apply_reset(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.zeros((2, backend.model.nq), dtype=torch.float32),
            torch.zeros((2, backend.model.nv), dtype=torch.float32),
        )
    assert plan.transfer_stats == before_stats
    np.testing.assert_array_equal(
        backend.get_state_views()["qpos"].numpy(), after_nan_state["qpos"].numpy()
    )

    original_validate = plan._validate_reset_rows

    def timed_validate(rows):
        time.sleep(0.002)
        original_validate(rows)

    monkeypatch.setattr(plan, "_validate_reset_rows", timed_validate)
    result = plan.apply_reset(
        torch.arange(backend.num_envs, dtype=torch.int64),
        torch.zeros((backend.num_envs, backend.model.nq), dtype=torch.float32),
        torch.zeros((backend.num_envs, backend.model.nv), dtype=torch.float32),
    )
    assert result is not None
    assert result["timing"]["tensor_reset_packed_d2h_ms"] >= 2.0
