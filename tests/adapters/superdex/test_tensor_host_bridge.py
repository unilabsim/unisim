"""Packed host-bridge tensor lifecycle tests for SuperDex."""

from __future__ import annotations

import sys
import time
import weakref

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorProcessTopology,
)
from unisim.scene import SceneCfg

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    pytest.skip("SuperDex wheels require Python 3.12 or 3.13", allow_module_level=True)
pytest.importorskip("superdex.physics")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")


def _model(tmp_path):
    path = tmp_path / "tensor.xml"
    path.write_text(
        """<mujoco model="superdex_tensor_test">
          <compiler angle="radian"/>
          <option gravity="0 0 0"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body name="base" pos="0 0 1">
              <inertial mass="2" pos=".03 .02 0" diaginertia=".03 .04 .05"/>
              <geom name="base_geom" type="box" size=".1 .1 .1"/>
              <site name="imu" pos=".01 .02 .03"/>
              <body name="arm" pos="0 0 .2">
                <joint name="hinge" axis="0 1 0" range="-1 1" damping=".1"/>
                <inertial mass="1" pos="0 0 .1" diaginertia=".02 .025 .01"/>
                <geom name="arm_geom" type="box" pos="0 0 .1" size=".025 .025 .1"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor name="motor" joint="hinge" gear="2" ctrlrange="-2 2"/></actuator>
          <sensor>
            <jointpos name="angle" joint="hinge"/>
            <jointvel name="speed" joint="hinge"/>
          </sensor>
        </mujoco>"""
    )
    return path


def _generalized_model(tmp_path):
    path = tmp_path / "tensor-generalized.xml"
    path.write_text(
        """<mujoco model="superdex_tensor_generalized">
          <compiler angle="radian"/>
          <option gravity="0 0 -1"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body name="base" pos="0 0 1">
              <freejoint name="root"/>
              <inertial mass="2" pos=".03 .02 0" diaginertia=".03 .04 .05"/>
              <geom name="base_geom" type="box" size=".1 .1 .1"/>
              <site name="imu" pos=".01 .02 .03"/>
              <body name="arm" pos="0 0 .2">
                <joint name="hinge" axis="0 1 0" range="-1 1" damping=".1"/>
                <inertial mass="1" pos="0 0 .1" diaginertia=".02 .025 .01"/>
                <geom name="arm_geom" type="box" pos="0 0 .1" size=".025 .025 .1"/>
              </body>
            </body>
          </worldbody>
          <actuator><motor name="motor" joint="hinge" gear="2" ctrlrange="-2 2"/></actuator>
          <sensor>
            <framepos name="base_pos" objtype="body" objname="base"/>
            <jointpos name="angle" joint="hinge"/>
            <jointvel name="speed" joint="hinge"/>
          </sensor>
        </mujoco>"""
    )
    return path


@pytest.fixture
def backend(tmp_path):
    result = create_backend("superdex", SceneCfg(str(_model(tmp_path))), 3, 0.002)
    try:
        yield result
    finally:
        result.close()


def test_tensor_capability_matrix_is_narrow_and_fail_closed(backend):
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
    assert not capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert not capabilities.host_pre_step_control


def test_packed_plan_supports_declared_ctrl_state_field(backend):
    plan = backend.compile_host_bridge_io(
        TensorIOSpec(state_fields=("qpos", "qvel", "ctrl"), device="cpu")
    )
    ctrl = torch.full((backend.num_envs, backend.num_actuators), 0.25, dtype=torch.float32)

    plan.write_control(ctrl)
    plan.step()
    views = plan.read_state_sensors()

    np.testing.assert_allclose(views["ctrl"].detach().numpy(), ctrl, atol=0.0)
    assert plan.transfer_stats["d2h_count"] == 1
    assert plan.transfer_stats["h2d_count"] == 1


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
    expected_state = backend.get_state()
    np.testing.assert_allclose(
        views["qpos"].detach().cpu().numpy(), expected_state["qpos"], atol=1e-6
    )
    np.testing.assert_allclose(
        views["qvel"].detach().cpu().numpy(), expected_state["qvel"], atol=1e-6
    )
    np.testing.assert_allclose(
        views["angle"].detach().cpu().numpy(), backend.get_sensor_data("angle"), atol=1e-6
    )
    np.testing.assert_allclose(
        views["speed"].detach().cpu().numpy(), backend.get_sensor_data("speed"), atol=1e-6
    )
    first_pointers = {name: view.data_ptr() for name, view in views.items()}
    plan.read_state_sensors()
    assert {name: view.data_ptr() for name, view in views.items()} == first_pointers

    state = backend.get_state()
    sensors = {name: backend.get_sensor_data(name) for name in ("angle", "speed")}
    rows = torch.tensor([1, 0], dtype=torch.int64, device=device)
    qpos = state["qpos"][[1, 0]].copy()
    qvel = state["qvel"][[1, 0]].copy()
    qpos[:, 0] = [0.4, -0.2]
    qvel[:, 0] = [1.5, -0.7]
    plan.apply_reset(rows, torch.tensor(qpos, device=device), torch.tensor(qvel, device=device))
    selected_rows = rows.clone()
    rows.fill_(-1)
    updated = plan.read_selected_state_sensors()
    np.testing.assert_allclose(updated["qpos"].cpu().numpy()[selected_rows.cpu()], qpos, atol=1e-6)
    np.testing.assert_allclose(updated["qvel"].cpu().numpy()[selected_rows.cpu()], qvel, atol=1e-6)
    np.testing.assert_allclose(
        updated["angle"].cpu().numpy()[selected_rows.cpu()], qpos[:, :1], atol=1e-6
    )
    np.testing.assert_allclose(updated["speed"].cpu().numpy()[selected_rows.cpu()], qvel, atol=1e-6)
    np.testing.assert_allclose(updated["qpos"].cpu().numpy()[2], state["qpos"][2])
    np.testing.assert_allclose(updated["angle"].cpu().numpy()[2], sensors["angle"][2])

    cuda = device.type == "cuda"
    stats = plan.transfer_stats
    reset_width = 1 + backend.model.nq + backend.model.nv
    assert stats["d2h_count"] == 2
    assert stats["h2d_count"] == 3
    assert stats["d2h_bytes"] == 3 * 4 + 2 * reset_width * 4
    assert stats["h2d_bytes"] == 3 * 16 * 2 + 2 * 16
    assert stats["synchronization_count"] == (5 if cuda else 0)

    before = dict(plan.transfer_stats)
    result = plan.apply_reset(
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0, backend.model.nq), dtype=torch.float32, device=device),
        torch.empty((0, backend.model.nv), dtype=torch.float32, device=device),
    )
    assert result == {"timing": {}}
    assert plan.transfer_stats == before


def test_selected_read_before_full_read_preserves_all_rows_or_fails_closed(backend, monkeypatch):
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
    qvel[:, 0] = -0.5
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


def test_cuda_packed_hot_path_avoids_hidden_cpu_detours(backend, monkeypatch):
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


def test_packed_reset_validation_is_bounded_timed_and_producer_owned(backend, monkeypatch):
    plan = backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel"), device="cpu"))
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
    nan_qvel = before_state["qvel"].copy()
    nan_qpos[1, 0] = np.nan
    before_stats = dict(plan.transfer_stats)
    with pytest.raises(Exception, match="must be finite"):
        plan.apply_reset(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor(nan_qpos[[1]], dtype=torch.float32),
            torch.tensor(nan_qvel[[1]], dtype=torch.float32),
        )
    assert plan.transfer_stats == before_stats

    before_stats = dict(plan.transfer_stats)
    invalid_cases = (
        ([-1], "in \\[0"),
        ([0, 0], "unique values"),
    )
    for rows, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            plan.apply_reset(
                torch.tensor(rows, dtype=torch.int64),
                torch.zeros((len(rows), backend.model.nq), dtype=torch.float32),
                torch.zeros((len(rows), backend.model.nv), dtype=torch.float32),
            )
    assert plan.transfer_stats == before_stats
    np.testing.assert_array_equal(backend.get_state()["qpos"], before_state["qpos"])

    original_validate = plan._validate_reset_rows

    def timed_validate(rows):
        time.sleep(0.002)
        original_validate(rows)

    monkeypatch.setattr(plan, "_validate_reset_rows", timed_validate)
    result = plan.apply_reset(
        torch.arange(backend.num_envs, dtype=torch.int64),
        torch.tensor(before_state["qpos"], dtype=torch.float32),
        torch.tensor(before_state["qvel"], dtype=torch.float32),
    )
    assert result is not None
    assert result["timing"]["tensor_reset_packed_d2h_ms"] >= 2.0


def test_generalized_tensor_parity_and_persistent_counters(tmp_path):
    """Match a floating-base scene against its NumPy control path."""

    model = str(_generalized_model(tmp_path))
    reference = create_backend("superdex", SceneCfg(model), 4, 0.002)
    candidate = create_backend("superdex", SceneCfg(model), 4, 0.002)
    try:
        plan = candidate.compile_host_bridge_io(
            TensorIOSpec(
                state_fields=("qpos", "qvel"),
                sensor_names=("base_pos", "angle", "speed"),
            )
        )
        before = candidate.get_state()
        rows = np.asarray([3, 1], dtype=np.int64)
        qpos = before["qpos"][rows].copy()
        qvel = before["qvel"][rows].copy()
        qpos[:, 0] = [0.12, -0.08]
        qvel[:, 0] = [0.4, -0.3]
        reference.set_state(
            rows,
            qpos.copy(),
            qvel.copy(),
        )
        plan.apply_reset(
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(qpos, dtype=torch.float32),
            torch.tensor(qvel, dtype=torch.float32),
        )
        selected = plan.read_selected_state_sensors()
        np.testing.assert_allclose(selected["qpos"].detach().numpy()[rows], qpos, atol=1e-6)
        np.testing.assert_allclose(selected["qvel"].detach().numpy()[rows], qvel, atol=1e-6)

        ctrl = np.asarray([[0.2], [-0.1], [0.0], [0.3]], dtype=np.float32)
        reference.step(ctrl, 2)
        plan.write_control(torch.from_numpy(ctrl.copy()))
        plan.step(2)
        views = plan.read_state_sensors()
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                views[name].detach().numpy(), candidate.get_state()[name], atol=1e-6
            )
            np.testing.assert_allclose(
                candidate.get_state()[name], reference.get_state()[name], atol=1e-5
            )
        for name in plan.spec.sensor_names:
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


def test_packed_tracked_body_sensors_match_public_body_state(tmp_path):
    """Pack deterministic world-frame body views without authored MJCF sensors."""

    candidate = create_backend("superdex", SceneCfg(str(_generalized_model(tmp_path))), 4, 0.002)
    sensor_names = (
        "track_pos_w_base",
        "track_quat_w_base",
        "track_linvel_w_arm",
        "track_angvel_w_arm",
    )
    try:
        body_ids = {name: candidate.get_body_ids((name,))[0] for name in ("base", "arm")}
        plan = candidate.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=sensor_names)
        )
        before = candidate.get_state()
        rows = np.asarray([2, 0], dtype=np.int64)
        qpos = before["qpos"][rows].copy()
        qvel = before["qvel"][rows].copy()
        qpos[:, 0] = [0.31, -0.17]
        qvel[:, 3:6] = [[0.2, -0.1, 0.3], [-0.4, 0.2, -0.1]]
        plan.apply_reset(
            torch.tensor(rows, dtype=torch.int64),
            torch.tensor(qpos, dtype=torch.float32),
            torch.tensor(qvel, dtype=torch.float32),
        )
        selected = plan.read_selected_state_sensors()
        np.testing.assert_allclose(
            selected["track_pos_w_base"].detach().numpy()[rows],
            candidate.get_body_pos_w(body_ids["base"][None])[rows, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            selected["track_quat_w_base"].detach().numpy()[rows],
            candidate.get_body_quat_w(body_ids["base"][None])[rows, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            selected["track_linvel_w_arm"].detach().numpy()[rows],
            candidate.get_body_lin_vel_w(body_ids["arm"][None])[rows, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            selected["track_angvel_w_arm"].detach().numpy()[rows],
            candidate.get_body_ang_vel_w(body_ids["arm"][None])[rows, 0],
            atol=1e-6,
        )

        views = plan.read_state_sensors()
        for name, getter, body_name in (
            ("track_pos_w_base", candidate.get_body_pos_w, "base"),
            ("track_quat_w_base", candidate.get_body_quat_w, "base"),
            ("track_linvel_w_arm", candidate.get_body_lin_vel_w, "arm"),
            ("track_angvel_w_arm", candidate.get_body_ang_vel_w, "arm"),
        ):
            expected = getter(body_ids[body_name][None])[:, 0]
            np.testing.assert_allclose(views[name].detach().numpy(), expected, atol=1e-6)
        assert plan.transfer_stats["h2d_count"] == 2
    finally:
        candidate.close()


def test_accelerometer_sensor_binding_fails_closed(tmp_path):
    source = tmp_path / "accelerometer.xml"
    source.write_text("""<mujoco><worldbody>
      <body name="base" pos="0 0 1">
        <freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="geom" type="sphere" size=".1"/>
        <site name="site" pos=".1 .2 .3"/>
      </body></worldbody>
      <sensor><accelerometer name="acceleration" site="site" cutoff=".2"/></sensor>
    </mujoco>""")
    backend = create_backend("superdex", SceneCfg(str(source)), 1, 0.002)
    try:
        with pytest.raises(NotImplementedError, match="no substitute is published"):
            backend.get_sensor_data("acceleration")
        with pytest.raises(NotImplementedError, match="no substitute is published"):
            backend.compile_host_bridge_io(
                TensorIOSpec(
                    state_fields=("qpos", "qvel"), sensor_names=("acceleration",), device="cpu"
                )
            )
    finally:
        backend.close()


def test_packed_compile_and_hot_path_capabilities_fail_closed(backend, monkeypatch):
    with pytest.raises(ValueError, match="SuperDex packed state I/O supports only"):
        backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("time",), sensor_names=("angle",), device="cpu")
        )
    with pytest.raises(KeyError, match="unknown SuperDex sensor"):
        backend.compile_host_bridge_io(
            TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=("missing",), device="cpu")
        )
    monkeypatch.setattr(backend, "_variant_assignment", np.zeros(backend.num_envs), raising=False)
    with pytest.raises(NotImplementedError, match="fixed variants"):
        backend.compile_host_bridge_io(TensorIOSpec(state_fields=("qpos", "qvel"), device="cpu"))
    monkeypatch.setattr(backend, "_variant_assignment", None)

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
    monkeypatch.setattr(backend, "_pre_step_control_fn", None)


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


def test_backend_close_invalidates_live_packed_plan(backend):
    plan = backend.compile_host_bridge_io(
        TensorIOSpec(state_fields=("qpos", "qvel"), sensor_names=("angle",), device="cpu")
    )
    buffers = plan._buffers()
    host_ref = weakref.ref(buffers.host_packet)
    device_ref = weakref.ref(buffers.device_packet)
    del buffers

    backend.close()

    assert host_ref() is None
    assert device_ref() is None
    assert not backend._host_bridge_plans
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.read_state_sensors()
    with pytest.raises(RuntimeError, match="plan is closed"):
        plan.write_control(torch.zeros((backend.num_envs, backend.num_actuators)))


def test_direct_tensor_apis_match_numpy_state(backend):
    state = backend.get_state_views(("qpos", "qvel", "ctrl"))
    angle = backend.get_sensor_view("angle")
    assert all(isinstance(value, torch.Tensor) for value in (*state.values(), angle))
    expected = backend.get_state(("qpos", "qvel", "ctrl"))
    for name in state:
        np.testing.assert_allclose(state[name].numpy(), expected[name], atol=1e-6)
    np.testing.assert_allclose(angle.numpy(), backend.get_sensor_data("angle"), atol=1e-6)

    before = backend.get_state()
    qpos = before["qpos"][[1, 2]].copy()
    qvel = before["qvel"][[1, 2]].copy()
    qpos[:, 0] = [0.3, -0.4]
    qvel[:, 0] = [1.2, -0.6]
    backend.set_state_tensor(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor(qpos, dtype=torch.float32),
        torch.tensor(qvel, dtype=torch.float32),
    )
    after = backend.get_state()
    np.testing.assert_allclose(after["qpos"][[1, 2]], qpos, atol=1e-6)
    np.testing.assert_allclose(after["qvel"][[1, 2]], qvel, atol=1e-6)
    np.testing.assert_allclose(after["qpos"][0], before["qpos"][0])

    result = backend.step_tensor(torch.full((3, 1), 0.2, dtype=torch.float32))
    assert "tensor_control_packed_d2h_ms" in result["timing"]
    direct_plan = backend._direct_host_bridge_plan
    assert direct_plan is not None
    reset_result = backend.set_state_tensor(
        torch.tensor([0], dtype=torch.int64),
        torch.zeros((1, backend.model.nq), dtype=torch.float32),
        torch.zeros((1, backend.model.nv), dtype=torch.float32),
    )
    assert reset_result is not None
    assert "tensor_reset_packed_d2h_ms" in reset_result["timing"]
    assert backend._direct_host_bridge_plan is direct_plan
    assert direct_plan.transfer_stats["d2h_count"] == 3
    empty = backend.set_state_tensor(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0, backend.model.nq), dtype=torch.float32),
        torch.empty((0, backend.model.nv), dtype=torch.float32),
    )
    assert empty == {"timing": {}}
