"""SDK-free tests for the experimental IsaacGym CUDA IPC data plane."""

from __future__ import annotations

import gc
import os
import select
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unisim.backend.base import (
    TensorDataPlane,
    TensorExecution,
    TensorProcessTopology,
)
from unisim.backend.isaacgym import tensor as isaacgym_tensor
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.isaacgym.dependencies import (
    build_worker_env,
    resolve_isaacgym_runtime,
)
from unisim.backend.isaacgym.tensor import (
    IsaacGymCudaIpcArenaLayout,
    IsaacGymCudaIpcPlan,
    RawArenaViewToken,
    torch_from_cuda_pointer,
)
from unisim.backend.subprocess_ipc import cuda_ipc, protocol


def _fake_gpu_backend() -> IsaacGymBackend:
    backend = IsaacGymBackend.__new__(IsaacGymBackend)
    backend._model_info = SimpleNamespace(use_gpu_pipeline=True)
    backend._cuda_ipc_plan = None
    return backend


def test_cuda_ipc_arena_layout_is_fixed_and_fails_closed() -> None:
    layout = IsaacGymCudaIpcArenaLayout.create(num_envs=2, nq=9, nv=8, nu=3)
    assert layout.reset_indices_offset == 0
    assert layout.reset_qpos_offset == 256
    assert layout.reset_qvel_offset == 512
    assert layout.qpos_offset == 768
    assert layout.qvel_offset == 1024
    assert layout.ctrl_offset == 1280
    assert layout.body_state_offset == 1536
    assert layout.sensor_state_offset == 1536
    assert layout.size_bytes == 1792
    assert layout.reset_indices_shape == (2,)
    assert layout.reset_qpos_shape == (2, 9)
    assert layout.reset_qvel_shape == (2, 8)
    assert layout.body_state_shape == (2, 0, 13)
    assert layout.sensor_state_shape == (2, 2, 3)
    assert IsaacGymCudaIpcArenaLayout.from_wire(layout.wire()) == layout

    bad = dict(layout.wire())
    bad["qvel_offset"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        IsaacGymCudaIpcArenaLayout.from_wire(bad)
    with pytest.raises(ValueError, match="num_envs > 0"):
        IsaacGymCudaIpcArenaLayout.create(0, 9, 8, 3)


def test_gpu_backend_declares_external_cuda_ipc_capability_matrix() -> None:
    backend = _fake_gpu_backend()
    assert backend.tensor_execution() is TensorExecution.DEVICE_RESIDENT
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.process_topology is TensorProcessTopology.EXTERNAL_WORKER
    assert capabilities.data_plane is TensorDataPlane.CUDA_IPC
    assert capabilities.state_fields == frozenset({"qpos", "qvel"})
    assert capabilities.stepping
    assert capabilities.stream_event_ownership is not None
    assert capabilities.torch_devices == ("cuda",)
    assert capabilities.selected_reset
    assert capabilities.sensor_views
    assert not capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert not capabilities.host_pre_step_control


def test_cpu_backend_removes_cuda_ipc_from_tensor_capabilities() -> None:
    backend = _fake_gpu_backend()
    backend._model_info = SimpleNamespace(use_gpu_pipeline=False)
    assert backend.tensor_execution() is TensorExecution.UNSUPPORTED
    assert backend.get_tensor_capabilities().execution is TensorExecution.UNSUPPORTED


def test_partial_lifecycle_fails_closed() -> None:
    backend = _fake_gpu_backend()
    with pytest.raises(KeyError, match="unknown IsaacGym CUDA IPC tensor sensor"):
        backend.get_sensor_view("base_gyro")
    with pytest.raises(NotImplementedError, match="reset randomization"):
        backend.set_state_tensor(None, None, None, randomization=object())

    result = {"timing": {}}
    backend._cuda_ipc_plan = SimpleNamespace(
        set_state_tensor=lambda rows, qpos, qvel: dict(result, rows=rows, qpos=qpos, qvel=qvel)
    )
    assert backend.set_state_tensor(1, 2, 3) == {
        "timing": {},
        "rows": 1,
        "qpos": 2,
        "qvel": 3,
    }

    plan = SimpleNamespace(get_state_views=lambda: {"qpos": object(), "qvel": object()})
    backend._cuda_ipc_plan = plan
    with pytest.raises(KeyError, match="time"):
        backend.get_state_views(("qpos", "time"))


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        device: str,
        *,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value


class _FakeSelected:
    captured_values: list[Any] = []

    def __init__(self, count: int = 0, forced_count: int | None = None) -> None:
        self.count = count
        self.forced_count = forced_count

    def __setitem__(self, _key: Any, _value: Any) -> None:
        self.captured_values.append(_value)
        self.count = self.forced_count if self.forced_count is not None else self.count + 1

    def zero_(self) -> None:
        self.count = 0

    def sum(self) -> _FakeScalar:
        return _FakeScalar(self.count)


class _FakeRowTensor(_FakeTensor):
    captured_bounds: list[tuple[Any, Any]] = []

    def __init__(self, *, minimum: int, maximum: int, unique_count: int, rows: int = 1) -> None:
        super().__init__((rows,), _FakeTorch.int64, "cuda:0")
        self.minimum = minimum
        self.maximum = maximum
        self.unique_count = unique_count

    def clamp(self, *, min: Any, max: Any) -> "_FakeRowTensor":  # noqa: A002
        self.captured_bounds.append((min, max))
        return self

    def min(self) -> _FakeScalar:
        return _FakeScalar(self.minimum)

    def max(self) -> _FakeScalar:
        return _FakeScalar(self.maximum)


class _FakeTorch:
    float32 = "float32"
    int64 = "int64"
    bool = "bool"
    Tensor = _FakeTensor
    forced_unique_count: int | None = None

    @staticmethod
    def zeros(*_args: Any, **kwargs: Any) -> _FakeSelected:
        return _FakeSelected(kwargs.get("count", 0), forced_count=_FakeTorch.forced_unique_count)

    @staticmethod
    def stack(values: Any) -> Any:
        return SimpleNamespace(
            tolist=lambda: [value.value for value in values],
        )


def test_parent_selected_reset_validation_and_empty_noop() -> None:
    from unisim.backend.isaacgym.tensor import IsaacGymCudaIpcPlan

    plan = IsaacGymCudaIpcPlan.__new__(IsaacGymCudaIpcPlan)
    plan.closed = False
    plan.arena = IsaacGymCudaIpcArenaLayout.create(2, 3, 2, 1)
    plan.device = "cuda:0"
    plan.device_index = 0
    plan._torch = _FakeTorch()
    plan.backend = SimpleNamespace(requests=[])
    plan._reset_row_bounds = [0, plan.arena.num_envs - 1]
    plan._reset_selected = _FakeSelected(forced_count=_FakeTorch.forced_unique_count)
    plan._reset_true = object()
    _FakeSelected.captured_values.clear()
    _FakeRowTensor.captured_bounds.clear()

    empty = plan.set_state_tensor(
        _FakeTensor((0,), _FakeTorch.int64, "cuda:0"),
        _FakeTensor((0, 3), _FakeTorch.float32, "cuda:0"),
        _FakeTensor((0, 2), _FakeTorch.float32, "cuda:0"),
    )
    assert empty == {"timing": {"cuda_ipc_reset_bytes": 0.0}}
    assert plan.backend.requests == []

    with pytest.raises(TypeError, match="reset rows must have dtype int64"):
        plan.set_state_tensor(
            _FakeTensor((1,), _FakeTorch.float32, "cuda:0"),
            _FakeTensor((1, 3), _FakeTorch.float32, "cuda:0"),
            _FakeTensor((1, 2), _FakeTorch.float32, "cuda:0"),
        )
    with pytest.raises(ValueError, match="qpos must have shape"):
        plan.set_state_tensor(
            _FakeTensor((1,), _FakeTorch.int64, "cuda:0"),
            _FakeTensor((1, 2), _FakeTorch.float32, "cuda:0"),
            _FakeTensor((1, 2), _FakeTorch.float32, "cuda:0"),
        )
    with pytest.raises(IndexError, match="out of range"):
        plan.set_state_tensor(
            _FakeRowTensor(minimum=-1, maximum=1, unique_count=1),
            _FakeTensor((1, 3), _FakeTorch.float32, "cuda:0"),
            _FakeTensor((1, 2), _FakeTorch.float32, "cuda:0"),
        )
    with pytest.raises(ValueError, match="unique"):
        _FakeTorch.forced_unique_count = 1
        try:
            plan.set_state_tensor(
                _FakeRowTensor(minimum=0, maximum=1, unique_count=1, rows=2),
                _FakeTensor((2, 3), _FakeTorch.float32, "cuda:0"),
                _FakeTensor((2, 2), _FakeTorch.float32, "cuda:0"),
            )
        finally:
            _FakeTorch.forced_unique_count = None
        _FakeTorch.forced_unique_count = None
    assert _FakeRowTensor.captured_bounds == [(0, plan.arena.num_envs - 1)] * 2
    assert _FakeSelected.captured_values
    assert all(value is plan._reset_true for value in _FakeSelected.captured_values)


def test_existing_plan_state_view_request_validates_exact_device() -> None:
    backend = _fake_gpu_backend()
    requested_fields: list[tuple[str, ...]] = []
    qpos_view = object()

    def get_state_views(fields: tuple[str, ...]) -> dict[str, Any]:
        requested_fields.append(tuple(fields))
        return {"qpos": qpos_view, "qvel": object()}

    backend._cuda_ipc_plan = SimpleNamespace(
        device="cuda:0",
        device_index=0,
        get_state_views=get_state_views,
    )
    assert backend.get_state_views(("qpos",), device="cuda") == {"qpos": qpos_view}

    with pytest.raises(ValueError, match="IsaacGym CUDA IPC state views"):
        backend.get_state_views(device="cuda:1")
    assert requested_fields == [("qpos",)]


def test_state_view_lifecycle_counts_each_caller_held_field() -> None:
    created: list[str] = []

    class _FakeCallerView:
        def __init__(self, token: RawArenaViewToken) -> None:
            self._token = token

        def __del__(self) -> None:
            self._token.release()

    class _FakeCudaDevice:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    class _FakeTorch:
        bool = "bool"
        int64 = "int64"
        cuda = SimpleNamespace(
            device=lambda index: _FakeCudaDevice(index),
            current_stream=lambda index: SimpleNamespace(cuda_stream=100 + index),
        )

        @staticmethod
        def tensor(values: Any, **_kwargs: Any) -> Any:
            return tuple(values)

        @staticmethod
        def zeros(*_args: Any, **_kwargs: Any) -> Any:
            return object()

        @staticmethod
        def ones(*_args: Any, **_kwargs: Any) -> Any:
            return object()

    def fake_tensor_from_pointer(
        _torch: Any,
        _pointer: int,
        _shape: tuple[int, ...],
        _device_index: int,
        *,
        dtype: str = "float32",
        token: RawArenaViewToken | None = None,
    ) -> Any:
        assert token is not None
        created.append(dtype)
        return _FakeCallerView(token)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(isaacgym_tensor, "torch_from_cuda_pointer", fake_tensor_from_pointer)
    try:
        plan = IsaacGymCudaIpcPlan.__new__(IsaacGymCudaIpcPlan)
        plan.closed = False
        plan.arena = IsaacGymCudaIpcArenaLayout.create(2, 3, 2, 1)
        plan.device = "cuda:0"
        plan.device_index = 0
        plan._torch = _FakeTorch()
        plan.memory = SimpleNamespace(pointer=4096)
        plan.state_event = SimpleNamespace(wait_stream=lambda stream: None)
        plan._active_views = {}
        plan._view_serial = 0
        plan._control = None
        plan._qpos = None
        plan._qvel = None
        plan._reset_indices = None
        plan._reset_qpos = None
        plan._reset_qvel = None

        views = plan.get_state_views(("qpos",))
        caller_qpos = views["qpos"]
        assert created == ["float32"]
        assert plan._qvel is None
        assert set(plan._active_views.values()) == {"qpos"}

        plan._drop_views()
        gc.collect()
        assert set(plan._active_views.values()) == {"qpos"}

        try:
            plan.close()
        except RuntimeError:
            pass
        assert not plan.closed

        del caller_qpos
        del views
        gc.collect()
        assert "qpos" not in set(plan._active_views.values())
    finally:
        monkeypatch.undo()


def test_failed_close_preserves_internal_cuda_ipc_views() -> None:
    class _FakeCallerView:
        def __init__(self, token: RawArenaViewToken) -> None:
            self._token = token

        def __del__(self) -> None:
            self._token.release()

    class _FakeCudaDevice:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    class _FakeTorch:
        bool = "bool"
        int64 = "int64"
        cuda = SimpleNamespace(
            device=lambda index: _FakeCudaDevice(index),
            current_stream=lambda index: SimpleNamespace(cuda_stream=100 + index),
        )

        @staticmethod
        def tensor(values: Any, **_kwargs: Any) -> Any:
            return tuple(values)

        @staticmethod
        def zeros(*_args: Any, **_kwargs: Any) -> Any:
            return object()

        @staticmethod
        def ones(*_args: Any, **_kwargs: Any) -> Any:
            return object()

    def fake_tensor_from_pointer(
        _torch: Any,
        _pointer: int,
        _shape: tuple[int, ...],
        _device_index: int,
        *,
        dtype: str = "float32",
        token: RawArenaViewToken | None = None,
    ) -> Any:
        assert token is not None
        return _FakeCallerView(token)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(isaacgym_tensor, "torch_from_cuda_pointer", fake_tensor_from_pointer)
    try:
        plan = IsaacGymCudaIpcPlan.__new__(IsaacGymCudaIpcPlan)
        plan.backend = SimpleNamespace(_request=lambda *args, **kwargs: {})
        plan.closed = False
        plan.arena = IsaacGymCudaIpcArenaLayout.create(2, 3, 2, 1)
        plan.device = "cuda:0"
        plan.device_index = 0
        plan._torch = _FakeTorch()
        plan.memory = SimpleNamespace(pointer=4096, close=lambda: None)
        plan.state_event = SimpleNamespace(wait_stream=lambda stream: None, close=lambda: None)
        plan.control_event = SimpleNamespace(close=lambda: None)
        plan.reset_event = SimpleNamespace(close=lambda: None)
        plan.transport = SimpleNamespace(close=lambda: None)
        plan._protocol = SimpleNamespace(CMD_READY="ready")
        plan._active_views = {}
        plan._view_serial = 0
        plan._control = None
        plan._qpos = None
        plan._qvel = None
        plan._reset_indices = None
        plan._reset_qpos = None
        plan._reset_qvel = None
        plan._reset_sequence = 0
        plan.last_timing = {}
        plan._make_views()
        original_control = plan._control
        views = plan.get_state_views(("qpos",))
        original_qpos = views["qpos"]

        try:
            plan.close()
        except RuntimeError:
            pass

        assert not plan.closed
        assert plan._control is not None
        assert plan._control is not original_control
        replacement_views = plan.get_state_views(("qpos",))
        replacement_qpos = replacement_views["qpos"]
        assert replacement_qpos is not original_qpos

        del original_qpos
        del views
        del original_control
        gc.collect()

        try:
            plan.close()
        except RuntimeError:
            pass
        else:
            pytest.fail("replacement qpos view must keep close fail-closed")

        assert not plan.closed
        del replacement_qpos
        del replacement_views
        gc.collect()
        plan.close()
        assert plan.closed
    finally:
        monkeypatch.undo()


def test_cuda_ipc_plan_fails_closed_for_unsupported_lifecycles() -> None:
    from unisim.backend.isaacgym.tensor import IsaacGymCudaIpcPlan

    fixed = SimpleNamespace(_fixed_variant_plan=object())
    with pytest.raises(NotImplementedError, match="fixed variants"):
        IsaacGymCudaIpcPlan(fixed)

    entity_scene = SimpleNamespace(owner=SimpleNamespace(variant_plan=object()))
    entity_variant = SimpleNamespace(_entity_scene=entity_scene)
    with pytest.raises(NotImplementedError, match="fixed variants"):
        IsaacGymCudaIpcPlan(entity_variant)

    host_callback = SimpleNamespace(_pre_step_control_fn=lambda *_: None)
    with pytest.raises(NotImplementedError, match="host pre-step"):
        IsaacGymCudaIpcPlan(host_callback)


def test_numpy_hot_paths_fail_closed_while_cuda_ipc_plan_is_open() -> None:
    backend = _fake_gpu_backend()
    backend._cuda_ipc_plan = object()
    with pytest.raises(RuntimeError, match="NumPy step"):
        backend.step(None)
    with pytest.raises(RuntimeError, match="NumPy reset"):
        backend.reset()
    with pytest.raises(RuntimeError, match="NumPy state writes"):
        backend.set_state(None, None, None)
    with pytest.raises(RuntimeError, match="NumPy entity reset"):
        backend.reset_entities(None)


def _child_code() -> str:
    return r"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

NUM_ENVS = 2
NUM_DOF = 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package_root = os.path.abspath(sys.argv[1])
protocol_path = os.path.join(
    package_root, "backend", "subprocess_ipc", "protocol.py"
)
worker_path = os.path.join(package_root, "backend", "isaacgym", "worker.py")
protocol = load("unisim_subprocess_protocol", protocol_path)
worker = load("unisim_isaacgym_worker", worker_path)


class Gym:
    def destroy_sim(self, sim):
        pass

    def refresh_actor_root_state_tensor(self, sim):
        pass

    def refresh_dof_state_tensor(self, sim):
        pass

    def refresh_rigid_body_state_tensor(self, sim):
        pass

    def refresh_net_contact_force_tensor(self, sim):
        pass

    def set_dof_position_target_tensor(self, sim, targets):
        return True

    def set_actor_root_state_tensor_indexed(self, sim, states, indices, count):
        return count == indices.shape[0]

    def set_dof_state_tensor_indexed(self, sim, states, indices, count):
        return count == indices.shape[0]

    def simulate(self, sim):
        ctx._dof_state[:, 0] += ctx.targets
        ctx._root_state[:, 7:10] += 0.125

    def fetch_results(self, sim, wait):
        return True


ctx = worker._WorkerContext(protocol)
ctx.torch = torch
ctx.gym = Gym()
ctx.gymtorch = SimpleNamespace(unwrap_tensor=lambda value: value)
ctx.gymapi = SimpleNamespace()
ctx.sim = object()
ctx.num_envs = NUM_ENVS
ctx.num_dof = NUM_DOF
ctx.num_bodies = NUM_DOF + 1
ctx.device = "cuda:0"
ctx.use_gpu_pipeline = True
ctx._root_state = torch.zeros((NUM_ENVS, 13), dtype=torch.float32, device=ctx.device)
ctx._root_state[:, 6] = 1.0
initial_dof = torch.arange(
    NUM_ENVS * NUM_DOF, dtype=torch.float32, device=ctx.device
)
zero_dof = torch.zeros(
    NUM_ENVS * NUM_DOF, dtype=torch.float32, device=ctx.device
)
ctx._dof_state = torch.stack((initial_dof, zero_dof), dim=-1)
ctx.targets = torch.zeros_like(ctx._dof_state[:, 0])
ctx._body_state = torch.zeros(
    (NUM_ENVS * ctx.num_bodies, 13), device=ctx.device
)
ctx._contact_force = torch.zeros(
    (NUM_ENVS * ctx.num_bodies, 3), device=ctx.device
)


def refresh():
    pass


def submit_pending():
    pass


entity = SimpleNamespace(
    root_mode="floating",
    root_qpos_indices=tuple(range(7)),
    root_qvel_indices=tuple(range(6)),
    joints=[
        SimpleNamespace(qpos_indices=(7 + i,), qvel_indices=(6 + i,))
        for i in range(NUM_DOF)
    ],
)
layout = SimpleNamespace(
    nq=7 + NUM_DOF,
    nv=6 + NUM_DOF,
    nu=NUM_DOF,
    nbody=NUM_DOF + 1,
    entities=[entity],
)
records = [
    [{"dof_ids": tuple(range(env * NUM_DOF, (env + 1) * NUM_DOF))}]
    for env in range(NUM_ENVS)
]
body_ids = np.arange(NUM_ENVS, dtype=np.int64).reshape(NUM_ENVS, 1) * (NUM_DOF + 1)
body_com = np.zeros((NUM_ENVS, 1, 3), dtype=np.float32)
body_rows, body_columns = np.nonzero(body_ids >= 0)
ctx.scene_worker = SimpleNamespace(
    layout=layout,
    records=records,
    num_envs=NUM_ENVS,
    targets=ctx.targets,
    actor_ids=np.arange(NUM_ENVS, dtype=np.int64).reshape(NUM_ENVS, 1),
    control_dofs=np.arange(
        NUM_ENVS * NUM_DOF, dtype=np.int64
    ).reshape(NUM_ENVS, NUM_DOF),
    root_com=body_com,
    origins=np.zeros((NUM_ENVS, 3), dtype=np.float32),
    body_ids=body_ids,
    body_com=body_com,
    _body_rows=body_rows,
    _body_columns=body_columns,
    _native_body_ids=body_ids[body_rows, body_columns],
    _body_refresh_com=body_com[body_rows, body_columns],
    pending_roots={},
    pending_dofs={},
    pending_dof_actors=set(),
    faulted=False,
    _refresh_tensors=refresh,
    _submit_pending=submit_pending,
)


def _recv():
    return protocol.recv_message(sys.stdin.buffer)


def _send(cmd, payload=None):
    protocol.send_message(sys.stdout.buffer, cmd, payload)


while True:
    try:
        message = _recv()
    except EOFError:
        break
    command, payload = message["cmd"], message.get("payload")
    if command == protocol.CMD_SHUTDOWN:
        ctx.shutdown()
        _send(protocol.CMD_READY)
        break
    try:
        reply_cmd, reply_payload = worker._dispatch(ctx, protocol, command, payload)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        _send(protocol.CMD_ERROR, protocol.serialize_exception(exc))
        continue
    _send(reply_cmd, reply_payload)
"""


def _request(proc: subprocess.Popen[bytes], command: str, payload: Any = None) -> dict[str, Any]:
    protocol.send_message(proc.stdin, command, payload)
    ready, _, _ = select.select([proc.stdout], [], [], 30)
    if not ready:
        raise TimeoutError("Python 3.8 worker did not answer " + command)
    reply = protocol.recv_message(proc.stdout)
    if reply["cmd"] == protocol.CMD_ERROR:
        raise RuntimeError(str(reply["payload"]))
    return reply


def _cuda_ipc_wire(
    layout: IsaacGymCudaIpcArenaLayout,
    allocation: Any,
    control_event: Any,
    state_event: Any,
    reset_event: Any,
) -> dict[str, Any]:
    memory_handle = allocation.export_handle()
    control_handle = control_event.export_handle()
    state_handle = state_event.export_handle()
    reset_handle = reset_event.export_handle()
    return {
        "arena": layout.wire(),
        "memory": {
            "opaque_handle": memory_handle.opaque_handle,
            "device_uuid": memory_handle.device_uuid,
            "size_bytes": memory_handle.size_bytes,
            "abi_version": memory_handle.abi_version,
            "alignment_bytes": memory_handle.alignment_bytes,
        },
        "control_event": {
            "opaque_handle": control_handle.opaque_handle,
            "device_uuid": control_handle.device_uuid,
            "abi_version": control_handle.abi_version,
            "blocking_sync": control_handle.blocking_sync,
        },
        "state_event": {
            "opaque_handle": state_handle.opaque_handle,
            "device_uuid": state_handle.device_uuid,
            "abi_version": state_handle.abi_version,
            "blocking_sync": state_handle.blocking_sync,
        },
        "reset_event": {
            "opaque_handle": reset_handle.opaque_handle,
            "device_uuid": reset_handle.device_uuid,
            "abi_version": reset_handle.abi_version,
            "blocking_sync": reset_handle.blocking_sync,
        },
        "sensors": [],
    }


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_CUDA_IPC_NATIVE") != "1",
    reason="opt-in native IsaacGym Python 3.8 runtime test",
)
def test_cuda_ipc_control_state_with_native_isaacgym_worker(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not cuda_ipc.cuda_driver_available() or not torch.cuda.is_available():
        pytest.skip("CUDA Torch/driver is unavailable")

    from tests.adapters.isaacgym.scene_client import SceneClient
    from tests.adapters.isaacgym.scene_fixture import scene_payload

    payload = scene_payload(tmp_path / "assets")
    client = SceneClient(payload, tmp_path / "native-worker.log")
    layout = IsaacGymCudaIpcArenaLayout.create(
        payload["num_envs"],
        client.layout.nq,
        client.layout.nv,
        client.layout.nu,
        client.layout.nbody,
    )
    transport = cuda_ipc.CudaIpcTransport(0)
    allocation = transport.allocate(layout.size_bytes)
    control_event = transport.create_event()
    state_event = transport.create_event()
    reset_event = transport.create_event()
    qpos: Any = None
    qvel: Any = None
    control: Any = None
    attached = False
    legacy_slots = {
        name: client.slots[name].copy() for name in ("qpos", "qvel", "ctrl", "entity_root_state")
    }
    body_state = torch_from_cuda_pointer(
        torch,
        allocation.pointer + layout.body_state_offset,
        layout.body_state_shape,
        0,
    )
    sensor_state = torch_from_cuda_pointer(
        torch,
        allocation.pointer + layout.sensor_state_offset,
        layout.sensor_state_shape,
        0,
    )
    body_pointer = body_state.data_ptr()
    sensor_pointer = sensor_state.data_ptr()
    try:
        reset_rows = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_indices_offset,
            layout.reset_indices_shape,
            0,
            dtype="int64",
        )
        reset_qpos = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_qpos_offset,
            layout.reset_qpos_shape,
            0,
        )
        reset_qvel = torch_from_cuda_pointer(
            torch,
            allocation.pointer + layout.reset_qvel_offset,
            layout.reset_qvel_shape,
            0,
        )
        qpos = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.qpos_offset, layout.qpos_shape, 0
        )
        qvel = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.qvel_offset, layout.qvel_shape, 0
        )
        control = torch_from_cuda_pointer(
            torch, allocation.pointer + layout.ctrl_offset, layout.ctrl_shape, 0
        )
        qpos_pointer = qpos.data_ptr()
        qvel_pointer = qvel.data_ptr()

        attach = client.request(
            "ISAACGYM_CUDA_IPC_ATTACH",
            _cuda_ipc_wire(layout, allocation, control_event, state_event, reset_event),
        )
        attached = True
        assert attach["device_uuid"] == transport.identity.uuid
        assert attach["arena"] == layout.wire()
        assert "isaacgym" not in sys.modules
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(
            qpos,
            torch.tensor(payload["initial_qpos"], dtype=torch.float32, device=qpos.device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            qvel,
            torch.tensor(payload["initial_qvel"], dtype=torch.float32, device=qvel.device),
            atol=1e-5,
            rtol=1e-5,
        )

        unselected_qpos = qpos[0].clone()
        unselected_qvel = qvel[0].clone()
        selected_qpos = torch.zeros_like(qpos[1])
        selected_qpos[1:4] = qpos[1, 1:4]
        selected_qpos[4:8] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=qpos.device
        )
        selected_qpos[0] = 0.21
        selected_qpos[8] = 22.0
        selected_qvel = torch.zeros_like(qvel[1])
        selected_qvel[1:7] = torch.tensor(
            [0.25, -0.5, 0.75, -1.0, 0.5, 1.5],
            dtype=torch.float32,
            device=qvel.device,
        )
        selected_qvel[0] = -0.25
        selected_qvel[7] = -0.5
        reset_rows[0] = 1
        reset_qpos[0].copy_(selected_qpos)
        reset_qvel[0].copy_(selected_qvel)
        reset_event.record(torch.cuda.current_stream().cuda_stream)
        client.request("ISAACGYM_CUDA_IPC_SET_STATE", {"count": 1, "sequence": 1})
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(qpos[0], unselected_qpos, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(qvel[0], unselected_qvel, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(qpos[1], selected_qpos, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(qvel[1], selected_qvel, atol=1e-5, rtol=1e-5)
        # IsaacGym body tensors use each env's local publication frame.  In
        # particular, do not subtract the env origin from the kinematic target.
        torch.testing.assert_close(
            body_state[:, 6, 0:3],
            torch.tensor(((0.0, 0.0, 0.5),) * layout.num_envs, device=body_state.device),
            atol=1e-6,
            rtol=1e-6,
        )
        assert qpos.data_ptr() == qpos_pointer
        assert qvel.data_ptr() == qvel_pointer
        assert bool(torch.isfinite(body_state.cpu()).all())
        assert bool(torch.isfinite(sensor_state.cpu()).all())
        assert body_state.data_ptr() == body_pointer
        assert sensor_state.data_ptr() == sensor_pointer
        for name, before in legacy_slots.items():
            np.testing.assert_array_equal(client.slots[name], before)

        # This is intentionally the first SDK simulate after ATTACH.  The
        # selected floating root and controlled joint must survive it, while
        # materialization's initial pending rows must not overwrite the direct
        # CUDA IPC selected reset.
        control.zero_()
        control_event.record(torch.cuda.current_stream().cuda_stream)
        client.request("ISAACGYM_CUDA_IPC_STEP", {"nsteps": 1})
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(qpos[1, 1:8], selected_qpos[1:8], atol=5e-3, rtol=1e-3)
        torch.testing.assert_close(qvel[1, 1:7], selected_qvel[1:7], atol=2e-2, rtol=1e-2)
        assert float(qpos[1, 0]) > 0.19
        torch.testing.assert_close(
            body_state[1, 6, 0:3],
            torch.tensor((0.0, 0.0, 0.5), device=body_state.device),
            atol=1e-6,
            rtol=1e-6,
        )

        # A selected indexed write replaces IsaacGym's pending actor-index set.
        # Exercise the dangerous sequence directly: full-row reset, selected reset,
        # and one simulate.  Row 0 must not lose its authoritative full-row reset.
        full_reset_qpos = qpos.clone()
        full_reset_qvel = qvel.clone()
        full_reset_qpos[:, 0] = 0.2
        full_reset_qvel[:, 0] = 0.0
        reset_rows.copy_(
            torch.arange(layout.num_envs, dtype=torch.int64, device=reset_rows.device)
        )
        reset_qpos.copy_(full_reset_qpos)
        reset_qvel.copy_(full_reset_qvel)
        reset_event.record(torch.cuda.current_stream().cuda_stream)
        client.request(
            "ISAACGYM_CUDA_IPC_SET_STATE", {"count": layout.num_envs, "sequence": 2}
        )
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(qpos, full_reset_qpos, atol=1e-5, rtol=1e-5)

        selected_after_full_qpos = full_reset_qpos[1].clone()
        selected_after_full_qvel = full_reset_qvel[1].clone()
        selected_after_full_qpos[0] = 0.21
        reset_rows[0] = 1
        reset_qpos[0].copy_(selected_after_full_qpos)
        reset_qvel[0].copy_(selected_after_full_qvel)
        reset_event.record(torch.cuda.current_stream().cuda_stream)
        client.request("ISAACGYM_CUDA_IPC_SET_STATE", {"count": 1, "sequence": 3})
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        torch.testing.assert_close(qpos[0], full_reset_qpos[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(qpos[1], selected_after_full_qpos, atol=1e-5, rtol=1e-5)

        control.zero_()
        control_event.record(torch.cuda.current_stream().cuda_stream)
        client.request("ISAACGYM_CUDA_IPC_STEP", {"nsteps": 1})
        state_event.wait_stream(torch.cuda.current_stream().cuda_stream)
        assert float(qpos[0, 0]) > 0.18
        assert float(qpos[1, 0]) > 0.19

        detach = client.request("ISAACGYM_CUDA_IPC_DETACH")
        attached = False
        assert detach["isaacgym_imported"] is True
        assert "isaacgym" not in sys.modules
    finally:
        if attached and client.proc.poll() is None:
            try:
                client.request("ISAACGYM_CUDA_IPC_DETACH")
            except Exception:
                pass
        client.close()
        qpos = None
        qvel = None
        body_state = None
        sensor_state = None
        control = None
        reset_rows = None
        reset_qpos = None
        reset_qvel = None
        gc.collect()
        state_event.close()
        control_event.close()
        reset_event.close()
        allocation.close()
        transport.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_CUDA_IPC_CROSS_PROCESS") != "1",
    reason="opt-in cross-process IsaacGym Python 3.8 test",
)
def test_cuda_ipc_control_state_and_events_cross_python38_process(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not cuda_ipc.cuda_driver_available():
        pytest.skip("CUDA driver is unavailable")
    runtime = resolve_isaacgym_runtime()
    if not torch.cuda.is_available():
        pytest.skip("CUDA Torch is unavailable")

    package_root = Path(__file__).resolve().parents[3] / "src" / "unisim"
    layout = IsaacGymCudaIpcArenaLayout.create(2, 10, 9, 3, nbody=4)
    transport = cuda_ipc.CudaIpcTransport(0)
    allocation = transport.allocate(layout.size_bytes)
    control_event = transport.create_event()
    state_event = transport.create_event()
    reset_event = transport.create_event()
    script = tmp_path / "child.py"
    script.write_text(_child_code(), encoding="utf-8")
    log = tmp_path / "worker.log"
    child_environment = build_worker_env(runtime)
    child_environment.pop("PYTHONPATH", None)
    with log.open("w") as stderr:
        proc = subprocess.Popen(
            [str(runtime.python), str(script), str(package_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=child_environment,
        )
    from unisim.backend.isaacgym.tensor import torch_from_cuda_pointer

    reset_rows = torch_from_cuda_pointer(
        torch,
        allocation.pointer + layout.reset_indices_offset,
        layout.reset_indices_shape,
        0,
        dtype="int64",
    )
    reset_qpos = torch_from_cuda_pointer(
        torch,
        allocation.pointer + layout.reset_qpos_offset,
        layout.reset_qpos_shape,
        0,
    )
    reset_qvel = torch_from_cuda_pointer(
        torch,
        allocation.pointer + layout.reset_qvel_offset,
        layout.reset_qvel_shape,
        0,
    )
    qpos = torch_from_cuda_pointer(
        torch, allocation.pointer + layout.qpos_offset, layout.qpos_shape, 0
    )
    qvel = torch_from_cuda_pointer(
        torch, allocation.pointer + layout.qvel_offset, layout.qvel_shape, 0
    )
    control = torch_from_cuda_pointer(
        torch, allocation.pointer + layout.ctrl_offset, layout.ctrl_shape, 0
    )
    qpos_pointer = qpos.data_ptr()
    qvel_pointer = qvel.data_ptr()
    try:
        attach = _request(
            proc,
            "ISAACGYM_CUDA_IPC_ATTACH",
            _cuda_ipc_wire(layout, allocation, control_event, state_event, reset_event),
        )["payload"]
        assert attach["device_uuid"] == transport.identity.uuid
        assert attach["arena"] == layout.wire()

        control_values = torch.tensor(
            [[0.25, 0.5, 0.75], [1.0, 1.25, 1.5]],
            dtype=torch.float32,
            device="cuda:0",
        )
        control.copy_(control_values)
        stream = torch.cuda.current_stream().cuda_stream
        control_event.record(stream)
        _request(proc, "ISAACGYM_CUDA_IPC_STEP", {"nsteps": 1})
        state_event.synchronize()

        expected_root = torch.zeros((2, 7), device=qpos.device)
        expected_root[:, 3] = 1.0
        torch.testing.assert_close(qpos[:, :7], expected_root)
        expected_dof = (
            torch.arange(6, dtype=torch.float32, device=qpos.device).reshape(2, 3) + control_values
        )
        torch.testing.assert_close(qpos[:, 7:], expected_dof)
        torch.testing.assert_close(
            qvel[:, :3], torch.full((2, 3), 0.125, dtype=torch.float32, device=qvel.device)
        )
        assert qpos.data_ptr() == qpos_pointer
        assert qvel.data_ptr() == qvel_pointer

        unselected_qpos = qpos[0].clone()
        unselected_qvel = qvel[0].clone()
        selected_qpos = torch.zeros_like(qpos[1])
        selected_qpos[3] = 1.0
        selected_qpos[7:] = torch.tensor(
            [31.0, 32.0, 33.0], dtype=torch.float32, device=qpos.device
        )
        selected_qvel = torch.zeros_like(qvel[1])
        selected_qvel[:6] = torch.tensor(
            [0.5, -1.0, 1.5, -2.0, 1.0, 2.0],
            dtype=torch.float32,
            device=qvel.device,
        )
        selected_qvel[6:] = torch.tensor([-0.5, 1.0, -1.5], dtype=torch.float32, device=qvel.device)
        reset_rows[0] = 1
        reset_qpos[0].copy_(selected_qpos)
        reset_qvel[0].copy_(selected_qvel)
        stream = torch.cuda.current_stream().cuda_stream
        reset_event.record(stream)
        with pytest.raises(RuntimeError, match="selected-reset sequence is invalid"):
            _request(proc, "ISAACGYM_CUDA_IPC_SET_STATE", {"count": 1, "sequence": 2})
        _request(proc, "ISAACGYM_CUDA_IPC_SET_STATE", {"count": 1, "sequence": 1})
        state_event.synchronize()
        torch.testing.assert_close(qpos[0], unselected_qpos, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(qvel[0], unselected_qvel, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(qpos[1], selected_qpos, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(qvel[1], selected_qvel, atol=1e-5, rtol=1e-5)
        assert qpos.data_ptr() == qpos_pointer
        assert qvel.data_ptr() == qvel_pointer

        detach = _request(proc, "ISAACGYM_CUDA_IPC_DETACH")["payload"]
        assert detach == {"isaacgym_imported": False}
        _request(proc, protocol.CMD_SHUTDOWN)
        assert proc.wait(timeout=10) == 0
        worker_log = log.read_text(encoding="utf-8")
        assert worker_log.count("Traceback") == 1
        assert "selected-reset sequence is invalid" in worker_log
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        for stream_handle in (proc.stdin, proc.stdout):
            if stream_handle is not None:
                stream_handle.close()
        qpos = None
        qvel = None
        control = None
        reset_rows = None
        reset_qpos = None
        reset_qvel = None
        gc.collect()
        state_event.close()
        control_event.close()
        reset_event.close()
        allocation.close()
        transport.close()
