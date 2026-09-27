"""Packed host-bridge tensor lifecycle tests for MotrixSim."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend
from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorProcessTopology,
)
from unisim.backend.motrix import tensor as motrix_tensor
from unisim.scene import SceneCfg

torch = pytest.importorskip("torch")

MODEL = """<mujoco model='unisim-motrix-tensor'>
  <option gravity='0 0 0'/>
  <worldbody>
    <body name='base' pos='0 0 1'>
      <freejoint name='root'/>
      <inertial pos='0 0 0' mass='1' diaginertia='.2 .2 .2'/>
      <geom name='base_geom' type='sphere' size='.08'/>
      <body name='link' pos='0 0 .2'>
        <joint name='drive' type='hinge' axis='0 1 0'/>
        <inertial pos='0 0 0' mass='.2' diaginertia='.03 .03 .03'/>
        <geom name='link_geom' type='sphere' size='.03'/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name='drive' joint='drive'/></actuator>
  <sensor>
    <jointpos name='angle' joint='drive'/>
    <jointvel name='speed' joint='drive'/>
  </sensor>
</mujoco>"""


@pytest.fixture
def backend(tmp_path: Path) -> MotrixBackend:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    result = MotrixBackend(SceneCfg(str(model_path)), num_envs=3, sim_dt=0.002)
    try:
        yield result
    finally:
        result.close()


def test_tensor_capability_matrix_is_narrow_and_fail_closed(backend: MotrixBackend) -> None:
    capabilities = backend.get_tensor_capabilities()
    assert backend.tensor_execution() is TensorExecution.HOST_BRIDGE
    assert capabilities.execution is TensorExecution.HOST_BRIDGE
    assert capabilities.state_fields == frozenset({"qpos", "qvel", "ctrl"})
    assert capabilities.sensor_views
    assert capabilities.stepping
    assert capabilities.selected_reset
    assert capabilities.packed_host_bridge
    assert capabilities.process_topology is TensorProcessTopology.IN_PROCESS
    assert capabilities.data_plane is TensorDataPlane.HOST_BRIDGE
    assert capabilities.torch_devices == ("cpu", "cuda")
    assert capabilities.stream_event_ownership
    assert not capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert not capabilities.host_pre_step_control


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_packed_lifecycle_and_semantic_transfer_counts(
    backend: MotrixBackend, device_name: str
) -> None:
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
    expected_state = backend.get_state()
    np.testing.assert_allclose(
        views["qpos"].detach().cpu().numpy(), expected_state["qpos"], atol=1e-6
    )
    np.testing.assert_allclose(
        views["qvel"].detach().cpu().numpy(), expected_state["qvel"], atol=1e-6
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

    state = backend.get_state()
    rows = torch.tensor([1, 0], dtype=torch.int64, device=device)
    qpos = state["qpos"][[1, 0]].copy()
    qvel = state["qvel"][[1, 0]].copy()
    qpos[:, 0] = [0.4, -0.2]
    qpos[:, 7] = [0.3, -0.4]
    qvel[:, 6] = [1.5, -0.7]
    result = plan.apply_reset(
        rows,
        torch.tensor(qpos, dtype=torch.float32, device=device),
        torch.tensor(qvel, dtype=torch.float32, device=device),
    )
    assert result is not None and result["timing"]
    selected_rows = rows.detach().clone()
    rows.fill_(-1)
    updated = plan.read_selected_state_sensors()
    np.testing.assert_allclose(
        updated["qpos"].detach().cpu().numpy()[selected_rows.cpu()], qpos, atol=1e-6
    )
    np.testing.assert_allclose(
        updated["qvel"].detach().cpu().numpy()[selected_rows.cpu()], qvel, atol=1e-6
    )
    np.testing.assert_allclose(
        updated["angle"].detach().cpu().numpy()[selected_rows.cpu()],
        qpos[:, 7:8],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        updated["qpos"].detach().cpu().numpy()[2], state["qpos"][2], atol=1e-6
    )

    expected_sync = 5 if device.type == "cuda" else 0
    reset_width = 1 + backend.model.num_dof_pos + backend.model.num_dof_vel
    assert plan.transfer_stats == {
        "d2h_count": 2,
        "h2d_count": 3,
        "d2h_bytes": 12 + 2 * reset_width * 4,
        "h2d_bytes": 2 * (3 * 17 * 4) + 2 * 17 * 4,
        "synchronization_count": expected_sync,
    }
    plan.close()
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.read_state_sensors()


def test_direct_tensor_apis_use_public_contract(backend: MotrixBackend) -> None:
    state_views = backend.get_state_views(("qpos", "qvel", "ctrl"))
    expected_state = backend.get_state()
    np.testing.assert_allclose(
        state_views["qpos"].detach().numpy(), expected_state["qpos"], atol=1e-6
    )
    np.testing.assert_allclose(
        state_views["qvel"].detach().numpy(), expected_state["qvel"], atol=1e-6
    )
    assert state_views["ctrl"].shape == (3, backend.num_actuators)
    np.testing.assert_allclose(
        backend.get_sensor_view("angle").detach().numpy(),
        backend.get_sensor_data("angle"),
        atol=1e-6,
    )

    step_result = backend.step_tensor(torch.zeros((3, 1), dtype=torch.float32))
    assert step_result is not None and step_result["timing"]["tensor_control_d2h_ms"] >= 0
    before = backend.get_state()
    selected_rows = np.asarray([2, 1], dtype=np.int64)
    qpos = before["qpos"][selected_rows].copy()
    qvel = before["qvel"][selected_rows].copy()
    qpos[:, 0] = [0.3, -0.4]
    qvel[:, 6] = [1.2, -0.6]
    reset_result = backend.set_state_tensor(
        torch.tensor(selected_rows, dtype=torch.int64),
        torch.tensor(qpos, dtype=torch.float32),
        torch.tensor(qvel, dtype=torch.float32),
    )
    assert reset_result is not None and reset_result["timing"]["tensor_reset_d2h_ms"] >= 0
    after = backend.get_state()
    np.testing.assert_allclose(after["qpos"][selected_rows], qpos, atol=1e-6)
    np.testing.assert_allclose(after["qvel"][selected_rows], qvel, atol=1e-6)
    np.testing.assert_allclose(after["qpos"][0], before["qpos"][0], atol=1e-6)
    assert backend.set_state_tensor(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0, backend.model.num_dof_pos), dtype=torch.float32),
        torch.empty((0, backend.model.num_dof_vel), dtype=torch.float32),
    ) == {"timing": {}}


def test_packed_reset_validation_is_bounded_timed_and_producer_owned(
    backend: MotrixBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    before_state = backend.get_state()
    nan_qpos = before_state["qpos"].copy()
    nan_qpos[1, 0] = np.nan
    result = plan.apply_reset(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor(nan_qpos[[1]], dtype=torch.float32),
        torch.tensor(before_state["qvel"][[1]], dtype=torch.float32),
    )
    assert result is not None
    assert np.isnan(backend.get_state()["qpos"][1, 0])
    after_nan_state = backend.get_state()

    before_stats = dict(plan.transfer_stats)
    with pytest.raises(ValueError, match=r"in \[0"):
        plan.apply_reset(
            torch.tensor([-1], dtype=torch.int64),
            torch.tensor(before_state["qpos"][:1], dtype=torch.float32),
            torch.tensor(before_state["qvel"][:1], dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="unique values"):
        plan.apply_reset(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor(before_state["qpos"][:2], dtype=torch.float32),
            torch.tensor(before_state["qvel"][:2], dtype=torch.float32),
        )
    assert plan.transfer_stats == before_stats
    np.testing.assert_array_equal(backend.get_state()["qpos"], after_nan_state["qpos"])

    original_validate = motrix_tensor._validate_reset_rows

    def timed_validate(rows, num_envs, owner):
        time.sleep(0.002)
        original_validate(rows, num_envs, owner)

    monkeypatch.setattr(motrix_tensor, "_validate_reset_rows", timed_validate)
    result = plan.apply_reset(
        torch.arange(backend.num_envs, dtype=torch.int64),
        torch.tensor(before_state["qpos"], dtype=torch.float32),
        torch.tensor(before_state["qvel"], dtype=torch.float32),
    )
    assert result is not None
    assert result["timing"]["tensor_reset_packed_d2h_ms"] >= 2.0


def test_body_state_sensor_aliases_use_public_world_views(backend: MotrixBackend) -> None:
    body_ids = backend.get_body_ids(("base", "link"))
    positions, quaternions, linear_velocities, angular_velocities = backend.get_body_state_w(
        body_ids
    )
    link_index = int(np.flatnonzero(body_ids == backend.get_body_ids(("link",))[0])[0])

    direct_views = {
        "track_pos_w_link": backend.get_sensor_view("track_pos_w_link"),
        "track_quat_w_link": backend.get_sensor_view("track_quat_w_link"),
        "track_linvel_w_link": backend.get_sensor_view("track_linvel_w_link"),
        "track_angvel_w_link": backend.get_sensor_view("track_angvel_w_link"),
    }
    np.testing.assert_allclose(
        direct_views["track_pos_w_link"].detach().numpy(), positions[:, link_index], atol=1e-6
    )
    np.testing.assert_allclose(
        direct_views["track_quat_w_link"].detach().numpy(), quaternions[:, link_index], atol=1e-6
    )
    np.testing.assert_allclose(
        direct_views["track_linvel_w_link"].detach().numpy(),
        linear_velocities[:, link_index],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        direct_views["track_angvel_w_link"].detach().numpy(),
        angular_velocities[:, link_index],
        atol=1e-6,
    )

    plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos", "qvel"),
            sensor_names=(
                "angle",
                "track_pos_w_base",
                "track_pos_w_link",
                "track_quat_w_link",
                "track_linvel_w_link",
                "track_angvel_w_link",
            ),
        )
    )
    views = plan.read_state_sensors()
    positions, quaternions, linear_velocities, angular_velocities = backend.get_body_state_w(
        body_ids
    )
    np.testing.assert_allclose(
        views["track_pos_w_base"].detach().numpy(), positions[:, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        views["track_pos_w_link"].detach().numpy(), positions[:, link_index], atol=1e-6
    )
    np.testing.assert_allclose(
        views["track_quat_w_link"].detach().numpy(), quaternions[:, link_index], atol=1e-6
    )
    np.testing.assert_allclose(
        views["track_linvel_w_link"].detach().numpy(), linear_velocities[:, link_index], atol=1e-6
    )
    np.testing.assert_allclose(
        views["track_angvel_w_link"].detach().numpy(), angular_velocities[:, link_index], atol=1e-6
    )


def test_host_bridge_rejects_unsupported_layouts_and_inputs(backend: MotrixBackend) -> None:
    with pytest.raises(ValueError, match="supports qpos, qvel, and ctrl"):
        backend.compile_host_bridge_io(TensorIOSpec(state_fields=("time",)))
    with pytest.raises(KeyError, match="Unknown Motrix sensor"):
        backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos",), sensor_names=("missing",))
        )

    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
    with pytest.raises(RuntimeError, match="write_control.*before"):
        plan.step()
    with pytest.raises(TypeError, match="contiguous float32"):
        plan.write_control(torch.zeros((3, 1), dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="does not support randomization"):
        plan.apply_reset(
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, backend.model.num_dof_pos), dtype=torch.float32),
            torch.zeros((1, backend.model.num_dof_vel), dtype=torch.float32),
            randomization=object(),
        )
    with pytest.raises(ValueError, match="unique values"):
        plan.apply_reset(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.zeros((2, backend.model.num_dof_pos), dtype=torch.float32),
            torch.zeros((2, backend.model.num_dof_vel), dtype=torch.float32),
        )


def test_tensor_stepping_fails_closed_with_host_callback(backend: MotrixBackend) -> None:
    backend._pre_step_control_fn = lambda ctrl: ctrl
    try:
        with pytest.raises(NotImplementedError, match="host pre-step control"):
            backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
        with pytest.raises(NotImplementedError, match="host pre-step control"):
            backend.step_tensor(torch.zeros((3, 1), dtype=torch.float32))
    finally:
        backend._pre_step_control_fn = None


def test_backend_close_releases_compiled_host_bridge_plan(backend: MotrixBackend) -> None:
    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
    backend.close()
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.read_state_sensors()
