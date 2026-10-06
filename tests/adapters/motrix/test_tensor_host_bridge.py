"""Packed host-bridge tensor lifecycle tests for MotrixSim."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")
pytest.importorskip("mujoco")

from tests.adapters.motrix.test_portable_batch_parity import _fixed_variant_scene
from tests.adapters.motrix.test_portable_entities import _scene
from unisim import MotrixBackend
from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorProcessTopology,
)
from unisim.backend.motrix import tensor as motrix_tensor
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
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


def test_public_state_widths_match_tensor_reset_layout(backend: MotrixBackend) -> None:
    """The public width contract must agree with packed reset validation."""

    widths = backend.get_public_state_widths()
    state = backend.get_state()

    assert widths.nq == state["qpos"].shape[1]
    assert widths.nv == state["qvel"].shape[1]

    with pytest.raises(ValueError, match=r"qpos must have shape"):
        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, widths.nq + 1), dtype=torch.float32),
            torch.zeros((1, widths.nv), dtype=torch.float32),
        )


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


def test_selected_read_before_full_read_preserves_all_rows_or_fails_closed(
    backend: MotrixBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_empty = torch.empty

    with monkeypatch.context() as patch:

        def sentinel_empty(*args, **kwargs):
            return original_empty(*args, **kwargs).fill_(-123.0)

        patch.setattr(torch, "empty", sentinel_empty)
        plan = backend.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=("angle", "speed"),
                device="cpu",
            )
        )

    assert plan.transfer_stats["h2d_count"] == 0
    state = backend.get_state_views(("qpos", "qvel"))
    rows = torch.tensor([1], dtype=torch.int64)
    qpos = state["qpos"][[1]].clone()
    qvel = state["qvel"][[1]].clone()
    qpos[:, 0] = 0.25
    qvel[:, -1] = -0.5
    plan.apply_reset(rows, qpos, qvel)
    updated = plan.read_selected_state_sensors()

    expected_state = backend.get_state_views(("qpos", "qvel"))
    for name in ("qpos", "qvel"):
        np.testing.assert_allclose(
            updated[name].detach().numpy(), expected_state[name].numpy(), atol=1e-6
        )
    for name in ("angle", "speed"):
        np.testing.assert_allclose(
            updated[name].detach().numpy(), backend.get_sensor_data(name), atol=1e-6
        )


def test_selected_read_does_not_compute_unselected_rows(
    backend: MotrixBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selected publication must avoid full-batch body-state allocation."""

    plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos", "qvel"),
            sensor_names=("angle", "speed"),
            device="cpu",
        )
    )
    plan.write_control(torch.zeros((3, 1), dtype=torch.float32))
    plan.step()
    state = backend.get_state()
    rows = np.asarray([2], dtype=np.int64)
    qpos = state["qpos"][[2]].copy()
    qvel = state["qvel"][[2]].copy()
    qpos[:, 0] = 0.3
    qvel[:, -1] = -0.4
    plan.apply_reset(
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(qpos, dtype=torch.float32),
        torch.tensor(qvel, dtype=torch.float32),
    )

    original_state = backend.get_state

    def reject_full_body_state(body_ids):
        raise AssertionError("selected read computed full-batch body state")

    def reject_full_state(fields=None):
        raise AssertionError("selected read computed full-batch public state")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "get_body_state_w", reject_full_body_state)
        patch.setattr(backend, "get_state", reject_full_state)
        selected = plan.read_selected_state_sensors()

    expected = original_state(("qpos", "qvel"))
    for name, values in (("qpos", qpos), ("qvel", qvel)):
        np.testing.assert_allclose(selected[name].detach().numpy()[rows], values, atol=1e-6)
    np.testing.assert_allclose(
        selected[name].detach().numpy()[[0, 1]], expected[name][[0, 1]], atol=1e-6
    )


def test_selected_body_read_matches_full_world_views(backend: MotrixBackend) -> None:
    """Selected body aliases remain numerically identical to full world reads."""

    plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos",),
            sensor_names=(
                "track_pos_w_base",
                "track_pos_w_link",
                "track_quat_w_link",
                "track_linvel_w_link",
                "track_angvel_w_link",
            ),
            device="cpu",
        )
    )
    plan.write_control(torch.zeros((3, 1), dtype=torch.float32))
    plan.step(2)
    full = plan.read_state_sensors()
    body_ids = backend.get_body_ids(("base", "link"))
    positions, quaternions, linear_velocities, angular_velocities = backend.get_body_state_w(
        body_ids
    )
    link_index = 1
    expected = {
        "track_pos_w_base": positions[:, 0],
        "track_pos_w_link": positions[:, link_index],
        "track_quat_w_link": quaternions[:, link_index],
        "track_linvel_w_link": linear_velocities[:, link_index],
        "track_angvel_w_link": angular_velocities[:, link_index],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(full[name].detach().numpy(), values, atol=1e-6)

    state = backend.get_state()
    rows = np.asarray([1, 2], dtype=np.int64)
    qpos = state["qpos"][rows].copy()
    qvel = state["qvel"][rows].copy()
    qpos[:, 0] = [0.2, -0.1]
    plan.apply_reset(
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(qpos, dtype=torch.float32),
        torch.tensor(qvel, dtype=torch.float32),
    )
    selected = plan.read_selected_state_sensors()
    reference = backend.get_body_state_w(body_ids)
    for index, name in enumerate(
        (
            "track_pos_w_base",
            "track_pos_w_link",
            "track_quat_w_link",
            "track_linvel_w_link",
            "track_angvel_w_link",
        )
    ):
        selected_values = selected[name].detach().numpy()
        reference_values = (
            reference[0][:, 0],
            reference[0][:, 1],
            reference[1][:, 1],
            reference[2][:, 1],
            reference[3][:, 1],
        )[index]
        np.testing.assert_allclose(selected_values[rows], reference_values[rows], atol=1e-6)


def test_cuda_packed_hot_path_avoids_hidden_cpu_detours(
    backend: MotrixBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def fail_detour(name: str) -> None:
        raise AssertionError(f"hidden CPU detour through Tensor.{name}")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: fail_detour("cpu"))
        patch.setattr(torch.Tensor, "item", lambda self, *a, **k: fail_detour("item"))
        patch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: fail_detour("tolist"))
        plan = backend.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=("angle", "speed"),
                device="cuda",
            )
        )
        plan.write_control(torch.zeros((3, 1), dtype=torch.float32, device="cuda"))
        plan.step(1)
        state = backend.get_state()
        rows = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        plan.apply_reset(
            rows,
            torch.tensor(state["qpos"][[1, 2]], dtype=torch.float32, device="cuda"),
            torch.tensor(state["qvel"][[1, 2]], dtype=torch.float32, device="cuda"),
        )
        plan.read_selected_state_sensors()
        stats = plan.transfer_stats

    assert stats["d2h_count"] == 2
    assert stats["h2d_count"] == 1
    assert stats["synchronization_count"] == 3


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
    assert step_result is not None
    assert step_result["timing"]["tensor_control_packed_d2h_ms"] >= 0
    direct_plan = backend._direct_host_bridge_plan
    assert direct_plan is not None
    assert backend.step_tensor(torch.full((3, 1), 0.1, dtype=torch.float32)) is not None
    assert backend._direct_host_bridge_plan is direct_plan
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
    assert reset_result is not None
    assert reset_result["timing"]["tensor_reset_packed_d2h_ms"] >= 0
    assert backend._direct_host_bridge_plan is direct_plan
    assert direct_plan.transfer_stats == {
        "d2h_count": 3,
        "h2d_count": 0,
        "d2h_bytes": 2 * (3 * backend.num_actuators * 4)
        + 2 * (1 + backend.model.num_dof_pos + backend.model.num_dof_vel) * 4,
        "h2d_bytes": 0,
        "synchronization_count": 0,
    }
    after = backend.get_state()
    np.testing.assert_allclose(after["qpos"][selected_rows], qpos, atol=1e-6)
    np.testing.assert_allclose(after["qvel"][selected_rows], qvel, atol=1e-6)
    np.testing.assert_allclose(after["qpos"][0], before["qpos"][0], atol=1e-6)
    assert backend.set_state_tensor(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0, backend.model.num_dof_pos), dtype=torch.float32),
        torch.empty((0, backend.model.num_dof_vel), dtype=torch.float32),
    ) == {"timing": {}}


def test_portable_generalized_tensor_parity_and_persistent_counters(
    tmp_path: Path,
) -> None:
    """Match a no-variant portable scene against its NumPy control path."""

    reference = MotrixBackend(_scene(tmp_path / "reference"), 3, 0.002, base_name="robot/base")
    candidate = MotrixBackend(_scene(tmp_path / "candidate"), 3, 0.002, base_name="robot/base")
    try:
        sensor_names = tuple(sorted(candidate._sensor_names))[:2]
        plan = candidate.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=sensor_names,
            )
        )
        initial = motrix_tensor._canonical_states(candidate, ("qpos", "qvel"))
        reference_state = motrix_tensor._canonical_states(reference, ("qpos", "qvel"))
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(initial[name], reference_state[name], atol=1e-6)

        rows = np.asarray([2, 0], dtype=np.int64)
        qpos = initial["qpos"][rows].copy()
        qvel = initial["qvel"][rows].copy()
        qpos[:, 0] = [0.08, -0.06]
        qvel[:, 0] = [0.2, -0.3]
        reference.set_state(rows, qpos.copy(), qvel.copy())
        plan.apply_reset(
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(qpos, dtype=torch.float32),
            torch.tensor(qvel, dtype=torch.float32),
        )
        selected = plan.read_selected_state_sensors()
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                selected[name].detach().numpy()[rows], qpos if name == "qpos" else qvel, atol=1e-6
            )

        ctrl = np.asarray([[0.05], [-0.04], [0.03]], dtype=np.float32)
        plan.write_control(torch.from_numpy(ctrl.copy()))
        plan.step(2)
        reference.step(ctrl, 2)
        views = plan.read_state_sensors()
        expected_state = motrix_tensor._canonical_states(candidate, ("qpos", "qvel"))
        reference_state = motrix_tensor._canonical_states(reference, ("qpos", "qvel"))
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                views[name].detach().numpy(), expected_state[name], atol=1e-6
            )
            np.testing.assert_allclose(expected_state[name], reference_state[name], atol=1e-5)
        for name in sensor_names:
            np.testing.assert_allclose(
                views[name].detach().numpy(), candidate.get_sensor_data(name), atol=1e-6
            )
            np.testing.assert_allclose(
                candidate.get_sensor_data(name), reference.get_sensor_data(name), atol=1e-5
            )

        assert plan.transfer_stats["d2h_count"] == 2
        assert plan.transfer_stats["h2d_count"] == 2
        assert plan.transfer_stats["synchronization_count"] == 0
    finally:
        reference.close()
        candidate.close()


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


def test_direct_tensor_apis_fail_closed_with_fixed_variants(tmp_path: Path) -> None:
    backend = MotrixBackend(
        _fixed_variant_scene(tmp_path, (1, 0)),
        2,
        0.002,
        base_name="robot/base",
    )
    try:
        state = backend.get_state()
        qpos = torch.tensor(state["qpos"][[1]], dtype=torch.float32)
        qvel = torch.tensor(state["qvel"][[1]], dtype=torch.float32)

        with pytest.raises(NotImplementedError, match="does not support fixed variants"):
            backend.get_state_views(("qpos", "qvel"))
        with pytest.raises(NotImplementedError, match="does not support fixed variants"):
            backend.get_sensor_view("source_joint")
        with pytest.raises(NotImplementedError, match="does not support fixed variants"):
            backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
        with pytest.raises(NotImplementedError, match="does not support fixed variants"):
            backend.step_tensor(torch.zeros((2, backend.num_actuators), dtype=torch.float32))
        with pytest.raises(NotImplementedError, match="does not support fixed variants"):
            backend.set_state_tensor(
                torch.tensor([1], dtype=torch.int64),
                qpos,
                qvel,
            )
    finally:
        backend.close()


def test_backend_close_releases_compiled_host_bridge_plan(backend: MotrixBackend) -> None:
    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel")))
    backend.close()
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.read_state_sensors()


def test_portable_zero_actuator_tensor_lifecycle_does_not_submit_empty_control(
    tmp_path: Path,
) -> None:
    """A passive portable body owns no control columns to submit."""
    body = tmp_path / "body.xml"
    floor = tmp_path / "floor.xml"
    body.write_text(
        """<mujoco><option gravity='0 0 -9.81'/><worldbody>
          <body name='base'><freejoint name='root'/>
          <inertial pos='0 0 0' mass='.1' diaginertia='.0000267 .0000267 .0000267'/>
          <geom name='geom' type='box' size='.02 .02 .02'/></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    floor.write_text(
        """<mujoco><option gravity='0 0 -9.81'/><worldbody>
          <body name='base' pos='0 0 -.1'><inertial pos='0 0 0' mass='10'
          diaginertia='1 1 1'/><geom name='floor' type='box' size='5 5 .1'/>
          </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "object",
                ModelSourceDescriptor(str(body)),
                kind="rigid",
                root_mode="floating",
                initial_state=EntityInitialState((0.0, 0.0, 0.02)),
            ),
            SceneEntitySpec(
                "floor",
                ModelSourceDescriptor(str(floor)),
                kind="rigid",
                root_mode="fixed",
                initial_state=EntityInitialState((0.0, 0.0, -0.1)),
            ),
        )
    )
    backend = MotrixBackend(scene, 2, 0.002, base_name="object/base")
    try:
        assert backend.num_actuators == 0
        rows = torch.tensor([0, 1], dtype=torch.int64)
        state = backend.get_state_views(("qpos", "qvel"))
        qvel = state["qvel"].clone()
        qvel[:, 0] = torch.tensor([1.0, -1.0])
        backend.set_state_tensor(rows, state["qpos"].clone(), qvel)
        result = backend.step_tensor(
            torch.empty((2, 0), dtype=torch.float32),
            nsteps=2,
        )
        assert result is not None
        after = backend.get_state_views(("qpos", "qvel"))
        assert backend.tensor_execution() is TensorExecution.HOST_BRIDGE
        assert torch.isfinite(after["qpos"]).all()
        assert torch.isfinite(after["qvel"]).all()
    finally:
        backend.close()
