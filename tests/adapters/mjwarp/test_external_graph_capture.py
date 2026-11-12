"""External (Torch-owned) CUDA graph capture of MJWarp tensor operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.base import (
    SelectedResetPublication,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.backend.graph_capture import DeviceVerdicts, ExternalGraphCapture, GraphCaptureError

torch = pytest.importorskip("torch")

MODEL = """<mujoco model='unisim-test-mjwarp-capture'>
  <option timestep='0.01'/>
  <worldbody>
    <body name='base'>
      <joint name='slide' type='slide' axis='1 0 0' damping='0.5'/>
      <geom type='box' size='0.05 0.05 0.05' mass='1'/>
    </body>
  </worldbody>
  <sensor><framepos name='base_pos' objtype='body' objname='base'/></sensor>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""

NUM_ENVS = 8


# --------------------------------------------------------------------------- #
# Contract objects (CPU)                                                       #
# --------------------------------------------------------------------------- #


def test_device_verdicts_flush_names_failed_checks_and_clears() -> None:
    verdicts = DeviceVerdicts()
    verdicts.record("a", torch.tensor(True), "a must hold")
    verdicts.record("b", torch.tensor(False), "b must hold")
    verdicts.record("b", torch.tensor(True), "b must hold")  # AND keeps the failure
    assert verdicts.names == ("a", "b")
    assert verdicts.failed() == ("b",)
    with pytest.raises(ValueError, match="b must hold"):
        verdicts.flush()
    assert verdicts.failed() == ()
    verdicts.flush()


def test_device_verdict_flags_keep_a_fixed_address() -> None:
    verdicts = DeviceVerdicts()
    flag = verdicts.ensure("x", device="cpu", message="x")
    verdicts.record("x", torch.tensor(False), "x")
    verdicts.clear()
    verdicts.record("x", torch.tensor(True), "x")
    assert verdicts.device_flags()[0] is flag
    assert bool(flag)


def test_external_capture_replays_host_transitions_in_order() -> None:
    log: list[str] = []
    capture = ExternalGraphCapture(torch_stream=None)
    capture.record_host_transition("first", lambda: log.append("first"))
    capture.record_host_transition("second", lambda: log.append("second"))
    assert log == ["first", "second"]
    capture.close()
    capture.replay_host_transitions()
    assert log == ["first", "second", "first", "second"]
    assert capture.host_transition_names == ("first", "second")
    with pytest.raises(GraphCaptureError, match="after the capture scope closed"):
        capture.record_host_transition("late", lambda: None)


def test_external_capture_guards_fail_closed_at_capture_and_replay() -> None:
    state = {"pending": False}
    capture = ExternalGraphCapture(torch_stream=None)
    capture.record_replay_guard(
        "no-pending", lambda: "work is pending" if state["pending"] else None
    )
    capture.check_replay_guards()
    state["pending"] = True
    with pytest.raises(GraphCaptureError, match="cannot replay: no-pending: work is pending"):
        capture.check_replay_guards()
    other = ExternalGraphCapture(torch_stream=None)
    with pytest.raises(GraphCaptureError, match="cannot capture"):
        other.record_replay_guard("no-pending", lambda: "work is pending")


def _capabilities(**overrides: object) -> TensorLifecycleCapabilities:
    values: dict[str, object] = dict(
        execution=TensorExecution.DEVICE_RESIDENT,
        state_views=True,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        process_topology=TensorProcessTopology.IN_PROCESS,
        data_plane=TensorDataPlane.DIRECT,
        stream_event_ownership="test",
        torch_devices=("cuda",),
        selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
        mask_selected_reset=True,
        external_graph_capture=True,
    )
    values.update(overrides)
    return TensorLifecycleCapabilities(**values)  # type: ignore[arg-type]


def test_external_capture_capability_requires_cuda_mask_reset_and_stepping() -> None:
    assert _capabilities().external_graph_capture
    with pytest.raises(ValueError, match="external CUDA graph capture"):
        _capabilities(torch_devices=("cpu",))
    with pytest.raises(ValueError, match="external CUDA graph capture"):
        _capabilities(stepping=False)
    with pytest.raises(ValueError, match="external CUDA graph capture"):
        _capabilities(mask_selected_reset=False)


# --------------------------------------------------------------------------- #
# MJWarp runtime (CUDA)                                                        #
# --------------------------------------------------------------------------- #


def _backend(tmp_path: Path, name: str, **options: object):
    pytest.importorskip("mujoco_warp")
    warp = pytest.importorskip("warp")
    from unisim import MjwarpBackend
    from unisim.scene import SceneCfg

    warp.init()
    if not torch.cuda.is_available() or not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp capture tests require CUDA")
    path = tmp_path / f"{name}.xml"
    path.write_text(MODEL)
    return MjwarpBackend(
        SceneCfg(model_file=str(path)), num_envs=NUM_ENVS, sim_dt=0.01, **options
    )


class _Inputs:
    """Static graph inputs: refilled before every replay, never reallocated."""

    def __init__(self, backend) -> None:
        views = backend.get_state_views(("qpos", "qvel"))
        self.ctrl = torch.zeros((NUM_ENVS, 1), dtype=torch.float32, device="cuda")
        self.mask = torch.zeros((NUM_ENVS,), dtype=torch.bool, device="cuda")
        self.qpos = torch.zeros_like(views["qpos"])
        self.qvel = torch.zeros_like(views["qvel"])

    def load(self, k: int) -> None:
        rows = torch.arange(NUM_ENVS, device="cuda")
        self.ctrl.copy_(torch.sin(rows.float() + 0.37 * k).reshape(-1, 1) * 3.0)
        self.mask.copy_((rows + k) % 3 == 0)
        self.qpos.fill_(0.01 * k)
        self.qvel.fill_(-0.02 * k)


def _cycle(backend, inputs: _Inputs) -> None:
    backend.step_tensor(inputs.ctrl, nsteps=2)
    backend.set_state_tensor_mask(inputs.mask, inputs.qpos, inputs.qvel)


def _capture_cycle(backend, inputs: _Inputs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with backend.torch_graph_capture_scope() as capture:
            _cycle(backend, inputs)
    return graph, capture


def _state(backend) -> dict[str, torch.Tensor]:
    views = backend.get_state_views(("qpos", "qvel", "time"))
    return {name: value.clone() for name, value in views.items()}


def test_captured_step_and_mask_reset_match_eager(tmp_path: Path) -> None:
    eager = _backend(tmp_path, "eager")
    graphed = _backend(tmp_path, "graphed")
    # The backend-owned graphs are active, so capture must bypass them.
    assert graphed.get_tensor_runtime_diagnostics()["cuda_graph"].enabled
    assert graphed.get_tensor_capabilities().external_graph_capture
    eager_inputs, graph_inputs = _Inputs(eager), _Inputs(graphed)

    # Warm every captured operation eagerly, identically on both backends.
    for backend, inputs in ((eager, eager_inputs), (graphed, graph_inputs)):
        inputs.load(0)
        _cycle(backend, inputs)
    graph_inputs.load(1)
    graph, capture = _capture_cycle(graphed, graph_inputs)
    assert capture.host_transition_names == (
        "mjwarp.step_tensor",
        "mjwarp.set_state_tensor_mask",
    )
    # Capture records without executing; run the captured cycle once.
    graph.replay()
    eager_inputs.load(1)
    _cycle(eager, eager_inputs)

    for k in range(2, 12):
        graph_inputs.load(k)
        capture.check_replay_guards()
        graph.replay()
        capture.replay_host_transitions()
        eager_inputs.load(k)
        _cycle(eager, eager_inputs)
        torch.cuda.synchronize()
        expected, actual = _state(eager), _state(graphed)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    # Host bookkeeping replayed exactly as eager execution produced it.
    np.testing.assert_array_equal(graphed._time_cache, eager._time_cache)
    assert graphed._authored_sensor_state_dirty == eager._authored_sensor_state_dirty
    assert graphed._tracked_body_state_dirty == eager._tracked_body_state_dirty
    assert graphed._host_cache_stale == eager._host_cache_stale


def test_capture_defers_value_checks_into_device_verdicts(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "verdicts")
    inputs = _Inputs(backend)
    inputs.load(0)
    _cycle(backend, inputs)
    assert backend.tensor_validation_verdicts() is None  # eager checks by default

    graph, capture = _capture_cycle(backend, inputs)
    verdicts = backend.tensor_validation_verdicts()
    assert verdicts is not None and {"qpos:finite", "qvel:finite"} <= set(verdicts.names)
    graph.replay()
    backend.flush_tensor_validation()

    inputs.qpos[2, 0] = float("nan")
    graph.replay()
    capture.replay_host_transitions()
    with pytest.raises(ValueError, match="qpos must contain finite values"):
        backend.flush_tensor_validation()
    backend.flush_tensor_validation()  # flags were cleared by the failing flush


def test_deferred_validation_option_keeps_eager_steps_sync_free(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "deferred", defer_tensor_validation=True)
    ctrl = torch.zeros((NUM_ENVS, 1), dtype=torch.float32, device="cuda")
    backend.step_tensor(ctrl)
    ctrl[0, 0] = float("inf")
    backend.step_tensor(ctrl)  # no host read: the verdict is latched
    with pytest.raises(ValueError, match="ctrl must contain finite values"):
        backend.flush_tensor_validation()


def test_capture_scope_rejects_host_sync_nesting_and_pending_wrenches(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "guards")
    inputs = _Inputs(backend)
    inputs.load(0)
    _cycle(backend, inputs)

    with pytest.raises(GraphCaptureError, match="active Torch CUDA graph capture"):
        with backend.torch_graph_capture_scope():
            pass
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with backend.torch_graph_capture_scope():
            with pytest.raises(GraphCaptureError, match="host synchronization"):
                backend._synchronize()
            with pytest.raises(GraphCaptureError, match="cannot nest"):
                with backend.torch_graph_capture_scope():
                    pass
            _cycle(backend, inputs)
    assert backend._external_capture is None

    graph, capture = _capture_cycle(backend, inputs)
    graph.replay()
    backend._device_xfrc_pending = True  # an interval push staged after capture
    with pytest.raises(GraphCaptureError, match="interval wrench or push is staged"):
        capture.check_replay_guards()
    backend._device_xfrc_pending = False
    capture.check_replay_guards()


def test_captured_step_receipt_orders_only_its_own_stream(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "receipt")
    inputs = _Inputs(backend)
    inputs.load(0)
    _cycle(backend, inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with backend.torch_graph_capture_scope() as capture:
            result = backend.step_tensor(inputs.ctrl)
            stream = torch.cuda.current_stream()
    receipt = result["completion"]
    receipt.wait_before_read(stream)
    with pytest.raises(GraphCaptureError, match="capture/replay stream"):
        receipt.wait_before_read(torch.cuda.Stream())
    assert capture.host_transition_names == ("mjwarp.step_tensor",)
    eager = backend.step_tensor(inputs.ctrl)["completion"]
    assert eager.sequence > receipt.sequence


def test_packed_reset_reuses_device_column_selections(tmp_path: Path) -> None:
    backend = _backend(tmp_path, "packed")
    transaction = backend.bind_packed_reset()
    for _ in range(3):
        transaction._column_tensor([0], 1, "qpos")
        transaction._column_tensor(None, 1, "qpos")
    # One upload per distinct host selection; later resets reuse device storage.
    assert len(transaction._column_cache) == 2
