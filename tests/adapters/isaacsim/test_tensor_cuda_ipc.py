"""SDK-free contracts for the IsaacSim CUDA IPC tensor pilot."""

from __future__ import annotations

import gc
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from unisim.backend.isaacsim import backend as isaac_backend
from unisim.backend.isaacsim import tensor_ipc
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.scene_worker import SceneWorkerContext
from unisim.backend.isaacsim.tensor_ipc import (
    HostCudaIpcArena,
    IsaacSimCudaArenaLayout,
    WorkerCudaIpcArena,
)
from unisim.backend.isaacsim.worker import _dispatch
from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.cuda_ipc import CudaIpcEventHandle, CudaIpcMemHandle


def _arena_payload(
    layout: IsaacSimCudaArenaLayout, *, memory_size_bytes: int | None = None
) -> dict[str, Any]:
    return {
        "schema_version": tensor_ipc.ISAACSIM_TENSOR_SCHEMA_VERSION,
        "device_uuid": "0" * 32,
        "device_index": 0,
        "layout": layout.as_dict(),
        "memory": CudaIpcMemHandle(
            opaque_handle=b"\0" * 64,
            device_uuid="0" * 32,
            size_bytes=(layout.size_bytes if memory_size_bytes is None else memory_size_bytes),
        ),
        "control_event": CudaIpcEventHandle(opaque_handle=b"\0" * 64, device_uuid="0" * 32),
        "state_event": CudaIpcEventHandle(opaque_handle=b"\0" * 64, device_uuid="0" * 32),
        "reset_event": CudaIpcEventHandle(opaque_handle=b"\0" * 64, device_uuid="0" * 32),
    }


def test_cuda_arena_layout_is_aligned_and_rejects_noncanonical_offsets() -> None:
    layout = IsaacSimCudaArenaLayout.create(num_envs=3, nq=7, nv=6, nu=5)
    assert layout.shapes == {
        "qpos": (3, 7),
        "qvel": (3, 6),
        "ctrl": (3, 5),
        "reset_env_indices": (3,),
        "reset_qpos": (3, 7),
        "reset_qvel": (3, 6),
    }
    assert layout.dtypes == {
        "qpos": "float32",
        "qvel": "float32",
        "ctrl": "float32",
        "reset_env_indices": "int64",
        "reset_qpos": "float32",
        "reset_qvel": "float32",
    }
    assert (
        layout.qpos_offset,
        layout.qvel_offset,
        layout.ctrl_offset,
        layout.reset_env_indices_offset,
        layout.reset_qpos_offset,
        layout.reset_qvel_offset,
    ) == (0, 256, 512, 768, 1024, 1280)
    assert layout.size_bytes == 1536

    encoded = layout.as_dict()
    assert IsaacSimCudaArenaLayout.from_dict(encoded) == layout
    encoded["qvel_offset"] += 1
    with pytest.raises(ValueError, match="not canonical"):
        IsaacSimCudaArenaLayout.from_dict(encoded)
    with pytest.raises(ValueError, match="invalid CUDA arena dimensions"):
        IsaacSimCudaArenaLayout.create(0, 1, 1, 1)


def test_cuda_ipc_capabilities_are_opt_in_and_minimal() -> None:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._tensor_cuda_ipc_requested = False
    backend._entity_scene = SimpleNamespace(layout=SimpleNamespace(nq=7, nv=6, nu=5))
    backend.backend_type = "isaacsim"
    assert backend.tensor_execution().value == "unsupported"
    assert backend.get_tensor_capabilities().state_views is False
    with pytest.raises(NotImplementedError):
        backend.set_state_tensor(None, None, None)

    backend._tensor_cuda_ipc_requested = True
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution.value == "device_resident"
    assert capabilities.process_topology.value == "external_worker"
    assert capabilities.data_plane.value == "cuda_ipc"
    assert capabilities.state_views and capabilities.stepping
    assert capabilities.selected_reset
    assert set(capabilities.state_fields) == {"qpos", "qvel"}
    assert not capabilities.sensor_views
    assert not capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert capabilities.torch_devices == ("cuda",)
    backend._cuda_ipc_arena = None
    backend._entity_scene.owner = SimpleNamespace(variant_plan=object())
    backend._require_state = lambda _name: None  # type: ignore[method-assign]
    with pytest.raises(NotImplementedError, match="fixed variants"):
        backend.get_state_views()
    with pytest.raises(NotImplementedError):
        backend.compile_host_bridge_io(SimpleNamespace())  # type: ignore[arg-type]


def _make_backend_and_arena(log: list[str]) -> tuple[Any, Any, Any]:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._tensor_cuda_ipc_requested = True
    backend._entity_scene = SimpleNamespace(layout=SimpleNamespace(nq=7, nv=6, nu=5))
    backend._model_info = SimpleNamespace()
    backend._worker_dead_error = None
    backend._slots = {"qpos": np.zeros((2, 7), dtype=np.float32)}
    backend._num_envs = 2
    requests: list[tuple[str, Any, str]] = []

    class _Arena:
        layout = IsaacSimCudaArenaLayout.create(2, 7, 6, 5)
        device_index = 0
        qpos = SimpleNamespace(device="cuda:0")
        qvel = SimpleNamespace(device="cuda:0")
        ctrl = SimpleNamespace(device="cuda:0")
        reset_env_indices = SimpleNamespace(device="cuda:0")
        reset_qpos = SimpleNamespace(device="cuda:0")
        reset_qvel = SimpleNamespace(device="cuda:0")

        def write_control(self, ctrl: Any) -> None:
            log.append(("write", tuple(ctrl.shape)))

        def record_control(self) -> None:
            log.append(("record-control",))

        def wait_state(self) -> None:
            log.append(("wait-state",))

        def write_reset(self, env_indices: Any, qpos: Any, qvel: Any) -> None:
            log.append(
                ("write-reset", tuple(env_indices.shape), tuple(qpos.shape), tuple(qvel.shape))
            )

        def record_reset(self) -> None:
            log.append(("record-reset",))

    arena = _Arena()
    backend._cuda_ipc_arena = arena
    requests: list[tuple[str, Any, str]] = []

    def request(cmd: str, payload: Any, *, expect: str) -> Any:
        requests.append((cmd, payload, expect))
        return {"timing": {"physics_ms": 1.0}}

    backend._request = request  # type: ignore[method-assign]
    backend._requests = requests
    ctrl = SimpleNamespace(
        is_cuda=True,
        dtype="float32",
        shape=(2, 5),
        device="cuda:0",
        is_contiguous=lambda: True,
    )
    return backend, arena, ctrl


def test_host_step_moves_only_command_metadata_over_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[Any] = []
    backend, _arena, ctrl = _make_backend_and_arena(log)
    monkeypatch.setattr(
        isaac_backend,
        "require_cuda_tensor",
        lambda tensor, *, rank, name: None,
    )
    monkeypatch.setattr(
        isaac_backend,
        "import_torch",
        lambda: pytest.fail("tensor step metadata validation must not synchronize device data"),
    )
    result = backend.step_tensor(ctrl, nsteps=2)
    assert result is not None and result["timing"]["cuda_ipc"] is True
    assert log == [("write", (2, 5)), ("record-control",), ("wait-state",)]
    assert len(backend._requests) == 1  # type: ignore[attr-defined]
    command, payload, expect = backend._requests[0]  # type: ignore[attr-defined]
    assert command == "TENSOR_CUDA_STEP"
    assert payload == {"nsteps": 2}
    assert expect == protocol.CMD_READY


def test_host_selected_reset_moves_only_metadata_over_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[Any] = []
    backend, _arena, _ctrl = _make_backend_and_arena(log)
    backend._cuda_reset_sequence = 0

    class _BoolTensor:
        def __init__(self, value: bool) -> None:
            self.value = value

        def __or__(self, other: "_BoolTensor") -> "_BoolTensor":
            return _BoolTensor(self.value or other.value)

        def __and__(self, other: "_BoolTensor") -> "_BoolTensor":
            return _BoolTensor(self.value and other.value)

        def __invert__(self) -> "_BoolTensor":
            return _BoolTensor(not self.value)

        def any(self) -> bool:
            return self.value

        def sum(self) -> "_BoolTensor":
            return _BoolTensor(self.value)

        def item(self) -> bool:
            return self.value

    class _Rows:
        is_cuda = True
        dtype = "torch.int64"
        shape = (1,)
        device = "cuda:0"

        def __init__(self) -> None:
            self.value = False

        def is_contiguous(self) -> bool:
            return True

        def __lt__(self, value: int) -> _BoolTensor:
            return _BoolTensor(True)

        def __ge__(self, value: int) -> _BoolTensor:
            return _BoolTensor(True)

        def min(self) -> "_BoolTensor":
            return _BoolTensor(True)

        def max(self) -> "_BoolTensor":
            return _BoolTensor(True)

        def clamp(self, *, min: int, max: int) -> "_Rows":  # noqa: A002
            return self

        def __or__(self, other: _BoolTensor) -> _BoolTensor:
            return _BoolTensor(self.value or other.value)  # type: ignore[attr-defined]

    class _State:
        is_cuda = True
        dtype = "torch.float32"
        device = "cuda:0"

        def is_contiguous(self) -> bool:
            return True

    class _Selected(_BoolTensor):
        dtype = "torch.bool"

        def __getitem__(self, _key: Any) -> "_BoolTensor":
            return _BoolTensor(False)

        def __setitem__(self, key: Any, value: Any) -> None:
            self.value = True

    class _Torch:
        bool = "bool"

        @staticmethod
        def zeros(*_args: Any, **_kwargs: Any) -> _Selected:
            return _Selected(False)

        @staticmethod
        def stack(values: tuple[_BoolTensor, ...]) -> Any:
            return SimpleNamespace(tolist=lambda: [value.value for value in values])

    rows, qpos, qvel = _Rows(), _State(), _State()
    qpos.shape = (1, 7)
    qvel.shape = (1, 6)
    with pytest.raises(NotImplementedError, match="randomization"):
        backend.set_state_tensor(rows, qpos, qvel, randomization=SimpleNamespace())  # type: ignore[arg-type]

    empty_rows, empty_qpos, empty_qvel = _Rows(), _State(), _State()
    empty_rows.shape = (0,)
    empty_qpos = _State()
    empty_qvel = _State()
    empty_qpos.shape = (0, 7)
    empty_qvel.shape = (0, 6)
    result = backend.set_state_tensor(empty_rows, empty_qpos, empty_qvel)
    assert result is not None and result["timing"]["cuda_ipc_reset_bytes"] == 0.0
    assert log == []
    assert backend._requests == []  # type: ignore[attr-defined]

    monkeypatch.setattr(isaac_backend, "import_torch", lambda: _Torch)
    result = backend.set_state_tensor(rows, qpos, qvel)
    assert result is not None and result["timing"]["cuda_ipc_reset_bytes"] == 0.0
    assert log == [
        ("write-reset", (1,), (1, 7), (1, 6)),
        ("record-reset",),
        ("wait-state",),
    ]
    assert len(backend._requests) == 1  # type: ignore[attr-defined]
    command, payload, expect = backend._requests[0]  # type: ignore[attr-defined]
    assert command == "TENSOR_CUDA_RESET"
    assert payload == {"count": 1, "sequence": 1}


def test_host_selected_reset_row_closure_uses_one_bounded_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[Any] = []
    backend, _arena, _ctrl = _make_backend_and_arena(log)

    sync_count = 0
    backend._cuda_reset_sequence = 0

    class _Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

    class _Checks:
        def __init__(self, values: tuple[int, ...]) -> None:
            self.values = values

        def tolist(self) -> tuple[int, ...]:
            nonlocal sync_count
            sync_count += 1
            return self.values

    class _Rows:
        is_cuda = True
        dtype = "torch.int64"
        device = "cuda:0"
        ndim = 1

        def __init__(
            self,
            count: int,
            *,
            minimum: int,
            maximum: int,
            unique_count: int,
        ) -> None:
            self.shape = (count,)
            self.minimum = minimum
            self.maximum = maximum
            self.unique_count = unique_count

        def is_contiguous(self) -> bool:
            return True

        def min(self) -> _Scalar:
            return _Scalar(self.minimum)

        def max(self) -> _Scalar:
            return _Scalar(self.maximum)

        def clamp(self, *, min: int, max: int) -> "_Rows":  # noqa: A002
            return self

    class _Selected:
        dtype = "torch.bool"
        unique_count = 0

        def __setitem__(self, _key: Any, _value: Any) -> None:
            return None

        def sum(self) -> _Scalar:
            return _Scalar(self.unique_count)

    class _Torch:
        bool = "bool"

        unique_count = 0

        @staticmethod
        def zeros(*_args: Any, **_kwargs: Any) -> _Selected:
            return _Selected()

        @staticmethod
        def stack(values: tuple[_Scalar, ...]) -> _Checks:
            return _Checks(tuple(value.value for value in values))

    class _State:
        is_cuda = True
        dtype = "torch.float32"
        device = "cuda:0"

        def __init__(self, shape: tuple[int, int]) -> None:
            self.shape = shape

        def is_contiguous(self) -> bool:
            return True

    monkeypatch.setattr(isaac_backend, "import_torch", _Torch)
    invalid_rows = _Rows(1, minimum=-1, maximum=2, unique_count=1)
    with pytest.raises(IndexError, match="out of range"):
        backend.set_state_tensor(invalid_rows, _State((1, 7)), _State((1, 6)))

    _Selected.unique_count = 1
    duplicate_rows = _Rows(2, minimum=0, maximum=1, unique_count=1)
    with pytest.raises(ValueError, match="unique"):
        backend.set_state_tensor(duplicate_rows, _State((2, 7)), _State((2, 6)))

    _Selected.unique_count = 2
    valid_rows = _Rows(2, minimum=0, maximum=1, unique_count=2)
    result = backend.set_state_tensor(valid_rows, _State((2, 7)), _State((2, 6)))

    assert log == [
        ("write-reset", (2,), (2, 7), (2, 6)),
        ("record-reset",),
        ("wait-state",),
    ]
    assert sync_count == 3
    assert result is not None and result["timing"]["cuda_ipc_reset_bytes"] == 0.0


def test_opt_in_tensor_lifecycle_does_not_attach_legacy_cpu_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._tensor_cuda_ipc_requested = True
    backend._entity_scene = SimpleNamespace(layout=SimpleNamespace(nu=3))
    backend._num_envs = 2
    backend._shm_handles = {}
    backend._slots = {}

    def num_actuators(self: IsaacSimBackend) -> int:
        return 3

    monkeypatch.setattr(IsaacSimBackend, "num_actuators", property(num_actuators))
    backend._allocate_slots()
    assert backend._slot_specs() == {}
    assert backend._shm_handles == {}
    assert set(backend._slots) == {"ctrl"}

    monkeypatch.setattr(
        isaac_backend.MjcfSubprocessBackend,
        "_capture_entity_report",
        lambda self, _meta: None,
    )
    backend._capture_entity_report({})
    assert backend._slots == {}


def test_state_views_wait_on_backend_event_before_exposing_storage() -> None:
    log: list[Any] = []
    backend, arena, _ctrl = _make_backend_and_arena(log)
    views = backend.get_state_views()
    assert log == [("wait-state",)]
    assert views["qpos"] is arena.qpos
    assert views["qvel"] is arena.qvel
    with pytest.raises(KeyError):
        backend.get_state_views(("qpos", "body_state"))


def test_control_clamp_uses_preuploaded_device_bounds_without_h2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[Any] = []
    backend, _arena, ctrl = _make_backend_and_arena(log)
    lower = SimpleNamespace(device="cuda:0")
    upper = SimpleNamespace(device="cuda:0")
    backend._cuda_control_bounds = (lower, upper)

    class _Torch:
        @staticmethod
        def clamp(value: Any, *, min: Any, max: Any) -> Any:
            log.append(("clamp", min is lower, max is upper))
            return value

        @staticmethod
        def as_tensor(*_args: Any, **_kwargs: Any) -> Any:
            pytest.fail("step_tensor must reuse pre-uploaded clamp bounds")

    monkeypatch.setattr(isaac_backend, "require_cuda_tensor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(isaac_backend, "import_torch", lambda: _Torch)
    backend.step_tensor(ctrl, nsteps=1)
    backend.step_tensor(ctrl, nsteps=1)
    assert log == [
        ("clamp", True, True),
        ("write", (2, 5)),
        ("record-control",),
        ("wait-state",),
        ("clamp", True, True),
        ("write", (2, 5)),
        ("record-control",),
        ("wait-state",),
    ]


def test_tensor_operand_accepts_current_cuda_family_but_rejects_wrong_index() -> None:
    log: list[Any] = []
    backend, _arena, ctrl = _make_backend_and_arena(log)
    ctrl.device = "cuda"
    backend.step_tensor(ctrl, nsteps=1)

    ctrl.device = "cuda:1"
    with pytest.raises(ValueError, match="IsaacSim CUDA IPC control"):
        backend.step_tensor(ctrl, nsteps=1)


def test_state_visibility_is_a_nonblocking_consumer_stream_wait() -> None:
    arena = HostCudaIpcArena.__new__(HostCudaIpcArena)
    arena.closed = False
    arena._device_index = 0

    class _Stream:
        cuda_stream = 7

    class _Cuda:
        @staticmethod
        def current_stream(_index: int) -> _Stream:
            return _Stream()

    class _Torch:
        cuda = _Cuda

    class _Event:
        waited: list[int] = []

        def wait_stream(self, stream: int) -> None:
            self.waited.append(stream)

        def synchronize(self) -> None:
            pytest.fail("state visibility must not CPU-block in get_state_views")

    arena._torch = _Torch
    arena._state_event = _Event()
    arena.wait_state()
    assert _Event.waited == [7]


class _FakeDevice:
    type = "cuda"
    index = 0

    def __str__(self) -> str:
        return "cuda:0"


class _FakeTensor:
    is_cuda = True

    def __init__(self, values: Any, device: str = "cuda:0", dtype: str = "float32") -> None:
        self.values = np.asarray(values, dtype=dtype)
        self.shape = tuple(self.values.shape)
        self.device = device
        self.dtype = dtype

    def is_contiguous(self) -> bool:
        return True

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self.values.copy(), self.device)

    def index_select(self, axis: int, index: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(
            np.take(self.values, index.values.astype(np.int64), axis=axis), self.device
        )

    def __getitem__(self, key: Any) -> "_FakeTensor":
        return _FakeTensor(self.values[key], self.device, str(self.dtype))

    def __setitem__(self, key: Any, value: Any) -> None:
        native = value.values if isinstance(value, _FakeTensor) else value
        self.values[key] = native

    def __neg__(self) -> "_FakeTensor":
        return _FakeTensor(-self.values, self.device)

    def __add__(self, other: Any) -> "_FakeTensor":
        native = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values + native, self.device)

    def __mul__(self, other: Any) -> "_FakeTensor":
        native = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.values * native, self.device)

    def __rmul__(self, other: Any) -> "_FakeTensor":
        native = other.values if isinstance(other, _FakeTensor) else other
        return _FakeTensor(native * self.values, self.device)

    def contiguous(self) -> "_FakeTensor":
        return self

    def index_copy_(self, axis: int, index: "_FakeTensor", source: "_FakeTensor") -> None:
        selector = tuple([slice(None)] * axis + [index.values.astype(np.int64)])
        self.values[selector] = source.values

    def copy_(self, source: "_FakeTensor", *, non_blocking: bool = False) -> None:
        self.values[...] = source.values

    def sub_(self, source: "_FakeTensor") -> None:
        self.values -= source.values

    def add_(self, source: "_FakeTensor") -> None:
        self.values += source.values

    def numel(self) -> int:
        return int(self.values.size)

    def tolist(self) -> list[Any]:
        return self.values.tolist()

    def isfinite(self) -> "_FakeTensor":
        return _FakeTensor(np.isfinite(self.values), self.device)

    def all(self) -> "_FakeTensor":
        return _FakeTensor(np.all(self.values), self.device)

    def item(self) -> Any:
        return self.values.item()

    def long(self) -> "_FakeTensor":
        return _FakeTensor(self.values.astype(np.int64), self.device, "int64")


class _NoHostSyncTensor(_FakeTensor):
    def tolist(self) -> list[Any]:
        raise AssertionError("static worker joint IDs must not be downloaded during reset")


class _FakeTorch:
    Tensor = _FakeTensor

    @staticmethod
    def device(value: Any) -> Any:
        return _FakeDevice() if value == "cuda:0" else value

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def arange(count: int, *, dtype: Any, device: Any) -> _FakeTensor:
        return _FakeTensor(np.arange(count, dtype=np.int64), device)

    @staticmethod
    def as_tensor(value: Any, *, dtype: Any = None, device: Any = None) -> _FakeTensor:
        return _FakeTensor(np.asarray(value, dtype=dtype), device)

    long = "int64"
    float32 = "float32"

    @staticmethod
    def isfinite(value: _FakeTensor) -> _FakeTensor:
        return value.isfinite()

    @staticmethod
    def cross(left: _FakeTensor, right: _FakeTensor, *, dim: int) -> _FakeTensor:
        return _FakeTensor(np.cross(left.values, right.values, axis=dim), left.device)

    @staticmethod
    def zeros(*args: Any, **kwargs: Any) -> _FakeTensor:
        return _FakeTensor(np.zeros(*args, **kwargs), kwargs.get("device", "cuda:0"))


class _FakeArena:
    layout = IsaacSimCudaArenaLayout.create(2, 1, 1, 1)

    def __init__(self, log: list[str]) -> None:
        self.ctrl = _FakeTensor(np.ones((2, 1)))
        self.qpos = _FakeTensor(np.zeros((2, 1)))
        self.qvel = _FakeTensor(np.zeros((2, 1)))
        self.reset_env_indices = _FakeTensor(np.zeros(2, dtype=np.int64), dtype="int64")
        self.reset_qpos = _FakeTensor(np.zeros((2, 1)))
        self.reset_qvel = _FakeTensor(np.zeros((2, 1)))
        self.log = log

    def wait_control(self) -> None:
        self.log.append("wait-control")

    def record_state(self) -> None:
        self.log.append("record-state")

    def wait_reset(self) -> None:
        self.log.append("wait-reset")

    def synchronize_state(self) -> None:
        self.log.append("synchronize-state")


def _cuda_worker_context() -> tuple[SceneWorkerContext, _FakeArena, list[Any], list[Any]]:
    log: list[Any] = []
    entity = SimpleNamespace(
        root_mode="fixed",
        joints=(SimpleNamespace(name="joint"),),
        actuator_indices=(0,),
    )
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.layout = SimpleNamespace(nu=1, entities=(entity,))
    ctx.num_envs = 2
    ctx.sim_dt = 0.002
    ctx.torch = _FakeTorch
    ctx.device = "cuda:0"
    ctx.origins = np.zeros((2, 3), dtype=np.float32)
    ctx._cuda_origins = _FakeTensor(ctx.origins)
    ctx._contact_reporting = False
    ctx._cuda_ipc = arena = _FakeArena(log)
    ctx._cuda_maps = [
        {
            "rows": _FakeTensor([0, 1]),
            "native_rows": _FakeTensor([1, 0]),
            "public_for_native": _FakeTensor([1, 0]),
            "joints": _FakeTensor([0], dtype="int64"),
            "joint_ids": [0],
            "root_qpos_columns": _FakeTensor([], device="cuda:0"),
            "root_qvel_columns": _FakeTensor([], device="cuda:0"),
            "joint_qpos_columns": _FakeTensor([0]),
            "joint_qvel_columns": _FakeTensor([0]),
            "controls": [7],
        }
    ]

    class _Asset:
        data = SimpleNamespace(
            joint_pos=_FakeTensor([[11.0], [22.0]]),
            joint_vel=_FakeTensor([[1.0], [2.0]]),
        )

        def __init__(self) -> None:
            self.targets: list[Any] = []

        def set_joint_position_target(self, values: Any, *, joint_ids: list[int]) -> None:
            self.targets.append((values, joint_ids))

        def write_data_to_sim(self) -> None:
            log.append("upload")

        def update(self, dt: float) -> None:
            log.append("update")

        def reset(self, env_ids: Any) -> None:
            log.append(("reset", tuple(env_ids.shape)))

        def write_joint_state_to_sim(
            self,
            positions: _FakeTensor,
            velocities: _FakeTensor,
            *,
            joint_ids: list[int],
            env_ids: _FakeTensor,
        ) -> None:
            log.append(
                ("write-joints", positions.values.tolist(), joint_ids, env_ids.values.tolist())
            )
            self.data.joint_pos.values[env_ids.values.astype(np.int64)] = positions.values
            self.data.joint_vel.values[env_ids.values.astype(np.int64)] = velocities.values

    asset = _Asset()
    ctx.assets = [asset]
    ctx.sim = SimpleNamespace(step=lambda render: log.append("physics"))
    ctx.faulted = False
    return ctx, arena, log, asset.targets


def test_worker_control_and_state_projection_stay_on_device_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, arena, log, targets = _cuda_worker_context()
    monkeypatch.setattr(
        _FakeTorch,
        "isfinite",
        staticmethod(lambda _value: pytest.fail("worker step must not synchronize finite checks")),
    )
    result = ctx.step_cuda_ipc({"nsteps": 1})
    assert result is not None and result["timing"]["cuda_ipc"] is True
    assert log == [
        "wait-control",
        "upload",
        "physics",
        "update",
        "record-state",
    ]
    assert len(targets) == 1
    target, joint_ids = targets[0]
    assert isinstance(target, _FakeTensor)
    assert target.device == "cuda:0"
    assert joint_ids == [7]
    # Native row order is env1,env0, so the public arena rows are 22,11 / 2,1.
    np.testing.assert_allclose(arena.qpos.values, [[22.0], [11.0]])
    np.testing.assert_allclose(arena.qvel.values, [[2.0], [1.0]])
    with pytest.raises(NotImplementedError, match="body wrench"):
        ctx.step_cuda_ipc({"nsteps": 1, "body_wrench": b"x"})


def test_worker_selected_reset_projects_prefix_and_republishes_state() -> None:
    ctx, arena, log, _targets = _cuda_worker_context()
    ctx._cuda_reset_sequence = 0
    arena.reset_env_indices.values[:] = [1, 0]
    arena.reset_qpos.values[:] = [[55.0], [66.0]]
    arena.reset_qvel.values[:] = [[5.0], [6.0]]

    result = ctx.reset_cuda_ipc({"count": 1, "sequence": 1})
    assert result is not None and result["timing"]["cuda_ipc"] is True
    assert log == [
        "wait-reset",
        ("write-joints", [[55.0]], [0], [0]),
        ("reset", (1,)),
        "update",
        "record-state",
    ]
    # Native row order is env1,env0.  Reset row env1 publishes into public row 1.
    np.testing.assert_allclose(arena.qpos.values, [[22.0], [55.0]])
    np.testing.assert_allclose(arena.qvel.values, [[2.0], [5.0]])
    assert ctx._cuda_reset_sequence == 1

    with pytest.raises(ValueError, match="sequence"):
        ctx.reset_cuda_ipc({"count": 1, "sequence": 1})


def test_worker_cuda_state_projection_preserves_floating_root_and_joint_columns() -> None:
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.torch = _FakeTorch
    ctx.device = "cuda:0"
    ctx.origins = np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
    ctx._cuda_origins = _FakeTensor(ctx.origins)
    arena = _FakeArena([])
    arena.qpos = _FakeTensor(np.full((2, 8), 9.0, dtype=np.float32))
    arena.qvel = _FakeTensor(np.full((2, 7), 9.0, dtype=np.float32))
    ctx._cuda_ipc = arena
    ctx.layout = SimpleNamespace(
        entities=(SimpleNamespace(root_mode="floating", joints=(SimpleNamespace(name="joint"),)),)
    )
    ctx._cuda_maps = [
        {
            "rows": _FakeTensor([0, 1]),
            "native_rows": _FakeTensor([1, 0]),
            "root_qpos_columns": _FakeTensor(list(range(7))),
            "root_qvel_columns": _FakeTensor(list(range(6))),
            "joint_qpos_columns": _FakeTensor([7]),
            "joint_qvel_columns": _FakeTensor([6]),
            "joints": _FakeTensor([0]),
            "joint_ids": [0],
        }
    ]
    ctx.assets = [
        SimpleNamespace(
            data=SimpleNamespace(
                root_link_state_w=_FakeTensor(
                    [
                        [24.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0],
                        [11.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    ]
                ),
                joint_pos=_FakeTensor([[8.0], [7.0]]),
                joint_vel=_FakeTensor([[0.8], [0.7]]),
            )
        )
    ]

    ctx._publish_cuda_state()

    # Native rows are env1,env0.  The root write must not replace the public
    # joint columns projected immediately afterwards.
    np.testing.assert_allclose(
        arena.qpos.values,
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 7.0],
            [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 8.0],
        ],
    )
    np.testing.assert_allclose(
        arena.qvel.values,
        [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.7], [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.8]],
    )


def test_worker_cuda_selected_reset_projects_floating_root_and_joints() -> None:
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)
    ctx.torch = _FakeTorch
    ctx.device = "cuda:0"
    ctx.num_envs = 2
    ctx.sim_dt = 0.002
    ctx.faulted = False
    ctx.origins = np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
    ctx._cuda_origins = _FakeTensor(ctx.origins)
    ctx._cuda_reset_sequence = 0
    arena = _FakeArena([])
    arena.qpos = _FakeTensor(np.zeros((2, 8), dtype=np.float32))
    arena.qvel = _FakeTensor(np.zeros((2, 7), dtype=np.float32))
    arena.reset_qpos = _FakeTensor(np.zeros((2, 8), dtype=np.float32))
    arena.reset_qvel = _FakeTensor(np.zeros((2, 7), dtype=np.float32))
    arena.reset_env_indices.values[:] = [0, 1]
    arena.reset_qpos.values[:] = [
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    arena.reset_qvel.values[:] = [
        [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    ctx._cuda_ipc = arena
    ctx.layout = SimpleNamespace(
        entities=(SimpleNamespace(root_mode="floating", joints=(SimpleNamespace(name="joint"),)),)
    )
    ctx._cuda_maps = [
        {
            "rows": _FakeTensor([0, 1], dtype="int64"),
            "native_rows": _FakeTensor([1, 0], dtype="int64"),
            "joints": _NoHostSyncTensor([0], device="cuda:0", dtype="int64"),
            "joint_ids": [0],
            "root_qpos_columns": _FakeTensor(list(range(7)), dtype="int64"),
            "root_qvel_columns": _FakeTensor(list(range(6)), dtype="int64"),
            "joint_qpos_columns": _FakeTensor([7], dtype="int64"),
            "joint_qvel_columns": _FakeTensor([6], dtype="int64"),
        }
    ]
    calls: list[Any] = []
    root_state = np.zeros((2, 13), dtype=np.float32)
    root_state[0, 0:3] = [20.0, 0.0, 0.0]
    root_state[0, 3] = 1.0
    joint_pos = np.zeros((2, 1), dtype=np.float32)
    joint_vel = np.zeros((2, 1), dtype=np.float32)

    class _Asset:
        data = SimpleNamespace(
            root_link_state_w=_FakeTensor(root_state),
            joint_pos=_FakeTensor(joint_pos),
            joint_vel=_FakeTensor(joint_vel),
        )

        def write_root_pose_to_sim(self, pose: Any, *, env_ids: Any) -> None:
            calls.append(("pose", pose.values.tolist(), env_ids.values.tolist()))
            root_state[env_ids.values.astype(np.int64), 0:7] = pose.values

        def write_root_link_velocity_to_sim(self, velocity: Any, *, env_ids: Any) -> None:
            calls.append(("velocity", velocity.values.tolist(), env_ids.values.tolist()))
            root_state[env_ids.values.astype(np.int64), 7:] = velocity.values

        def write_joint_state_to_sim(
            self,
            positions: Any,
            velocities: Any,
            *,
            joint_ids: list[int],
            env_ids: Any,
        ) -> None:
            calls.append(
                ("joints", positions.values.tolist(), velocities.values.tolist(), joint_ids)
            )
            rows = env_ids.values.astype(np.int64)
            joint_pos[rows] = positions.values
            joint_vel[rows] = velocities.values

        def reset(self, env_ids: Any) -> None:
            calls.append(("reset", env_ids.values.tolist()))

        def update(self, dt: float) -> None:
            calls.append(("update", dt))

    ctx.assets = [_Asset()]
    result = ctx.reset_cuda_ipc({"count": 1, "sequence": 1})
    assert result is not None and result["timing"]["cuda_ipc"] is True
    assert calls == [
        ("pose", [[11.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], [1]),
        ("velocity", [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0]], [1]),
        ("joints", [[9.0]], [[7.0]], [0]),
        ("reset", [1]),
        ("update", 0.002),
    ]
    np.testing.assert_allclose(
        arena.qpos.values,
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 9.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_allclose(
        arena.qvel.values,
        [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    )


def test_worker_rejects_a_cuda_arena_from_a_different_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = IsaacSimCudaArenaLayout.create(1, 1, 1, 1)
    payload = _arena_payload(layout)

    class _Transport:
        device_index = 1
        identity = SimpleNamespace(uuid="1" * 32)

        def close(self) -> None:
            return None

    requested_indices: list[int] = []

    monkeypatch.setattr(tensor_ipc, "import_torch", lambda: SimpleNamespace())
    monkeypatch.setattr(
        tensor_ipc,
        "CudaIpcTransport",
        lambda index: (requested_indices.append(index), _Transport())[1],
    )
    with pytest.raises(RuntimeError, match="UUID mismatch"):
        WorkerCudaIpcArena(payload, device_index=3)
    assert requested_indices == [3]


def test_worker_rejects_cuda_handles_that_do_not_match_canonical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = IsaacSimCudaArenaLayout.create(1, 1, 1, 1)
    payload = _arena_payload(layout, memory_size_bytes=layout.size_bytes + 256)
    monkeypatch.setattr(tensor_ipc, "import_torch", lambda: SimpleNamespace())
    monkeypatch.setattr(
        tensor_ipc,
        "CudaIpcTransport",
        lambda _index: pytest.fail("unsafe handle metadata must fail before CUDA import"),
    )
    with pytest.raises(ValueError, match="handles do not match"):
        WorkerCudaIpcArena(payload, device_index=0)


def test_worker_dispatch_has_cuda_control_plane_without_protocol_changes() -> None:
    ctx = SimpleNamespace(
        attach_cuda_ipc=lambda payload: {"ok": payload["schema_version"]},
        step_cuda_ipc=lambda payload: {"ok": payload["nsteps"]},
        reset_cuda_ipc=lambda payload: {"ok": payload["count"]},
    )
    assert _dispatch(ctx, protocol, "TENSOR_CUDA_ATTACH", {"schema_version": 1}) == (
        "TENSOR_CUDA_READY",
        {"ok": 1},
    )
    assert _dispatch(ctx, protocol, "TENSOR_CUDA_STEP", {"nsteps": 3}) == (
        protocol.CMD_READY,
        {"ok": 3},
    )
    assert _dispatch(ctx, protocol, "TENSOR_CUDA_RESET", {"count": 1, "sequence": 1}) == (
        protocol.CMD_READY,
        {"ok": 1},
    )
    tensor_ctx = SimpleNamespace(
        _tensor_cuda_ipc=True,
        attach_slots=lambda _payload: pytest.fail("CUDA IPC must not attach CPU slots"),
    )
    assert _dispatch(tensor_ctx, protocol, protocol.CMD_ATTACH, {"slots": {}}) == (
        protocol.CMD_READY,
        None,
    )
    with pytest.raises(ValueError, match="rejects legacy shared-memory slots"):
        _dispatch(tensor_ctx, protocol, protocol.CMD_ATTACH, {"slots": {"ctrl": {}}})


def test_real_raw_cuda_arena_exports_stable_torch_views() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    arena = HostCudaIpcArena(num_envs=2, nq=3, nv=2, nu=1, device="cuda:0")
    try:
        arena.qpos.fill_(7.0)
        arena.qvel.fill_(-2.0)
        arena.ctrl.fill_(0.5)
        arena.reset_env_indices.copy_(torch.tensor([1, 0], device=arena.qpos.device))
        arena.reset_qpos.fill_(8.0)
        arena.reset_qvel.fill_(-3.0)
        torch.cuda.synchronize()
        assert torch.equal(arena.qpos.cpu(), torch.full((2, 3), 7.0))
        assert arena.reset_env_indices.dtype == torch.int64
        assert torch.equal(arena.reset_env_indices.cpu(), torch.tensor([1, 0], dtype=torch.int64))
        assert torch.equal(arena.reset_qpos.cpu(), torch.full((2, 3), 8.0))
        assert torch.equal(arena.reset_qvel.cpu(), torch.full((2, 2), -3.0))
        payload = arena.to_payload()
        assert payload["device_uuid"]
        assert payload["schema_version"] == 2
        assert "reset_event" in payload
        assert payload["layout"] == arena.layout.as_dict()
    finally:
        arena.close()


def test_real_cuda_reset_arena_crosses_raw_ipc_event() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    host = HostCudaIpcArena(num_envs=2, nq=2, nv=1, nu=1, device="cuda:0")
    try:
        device = host.qpos.device
        host.reset_env_indices.copy_(torch.tensor([1, 0], device=device))
        host.reset_qpos.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device))
        host.reset_qvel.copy_(torch.tensor([[5.0], [6.0]], device=device))
        host.record_reset()
        script = (
            "import pickle, sys; "
            "from unisim.backend.isaacsim.tensor_ipc import WorkerCudaIpcArena; "
            "arena = WorkerCudaIpcArena(pickle.load(sys.stdin.buffer), device_index=0); "
            "arena.wait_reset(); "
            "print(arena.reset_env_indices.cpu().tolist(), "
            "arena.reset_qpos.cpu().tolist(), arena.reset_qvel.cpu().tolist()); "
            "arena.close()"
        )
        process = subprocess.run(
            [sys.executable, "-c", script],
            input=pickle.dumps(host.to_payload()),
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        expected = b"[1, 0] [[1.0, 2.0], [3.0, 4.0]] [[5.0], [6.0]]\n"
        assert process.stdout == expected
    finally:
        host.close()


def test_real_cuda_arena_close_fails_closed_while_caller_holds_view() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    arena = HostCudaIpcArena(num_envs=1, nq=1, nv=1, nu=1, device="cuda:0")
    qpos = arena.qpos
    try:
        with pytest.raises(RuntimeError, match="release IsaacSim CUDA"):
            arena.close()
    finally:
        del qpos
        gc.collect()
        arena.close()


def test_real_cuda_arena_close_fails_closed_while_detached_view_remains() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    arena = HostCudaIpcArena(num_envs=1, nq=1, nv=1, nu=1, device="cuda:0")
    base = arena.qpos
    detached = base.detach()
    del base
    gc.collect()
    try:
        with pytest.raises(RuntimeError, match="release IsaacSim CUDA"):
            arena.close()
    finally:
        del detached
        gc.collect()
        arena.close()


def test_cuda_arena_tracks_each_returned_view_independently() -> None:
    class _FakeDLPackTensor:
        is_cuda = True

        def __init__(self, source: Any) -> None:
            self._capsule = source.__dlpack__()
            self.shape = source._shape_tuple

    class _FakeTorch:
        cuda = SimpleNamespace(synchronize=lambda _device_index: None)

        @staticmethod
        def from_dlpack(source: Any) -> Any:
            return _FakeDLPackTensor(source)

    arena = HostCudaIpcArena.__new__(HostCudaIpcArena)
    arena.layout = IsaacSimCudaArenaLayout.create(num_envs=1, nq=1, nv=1, nu=1)
    arena.closed = False
    arena._torch = _FakeTorch()
    arena._device_index = 0
    arena._allocation = SimpleNamespace(pointer=256, close=lambda: None)
    arena._control_event = None
    arena._state_event = None
    arena._reset_event = None
    arena._transport = None
    arena._active_view_names = {}
    arena._view_serial = 0

    first = arena._view("qpos")
    second = arena._view("qpos")
    del first
    gc.collect()
    try:
        with pytest.raises(RuntimeError, match="release IsaacSim CUDA"):
            arena.close()
    finally:
        del second
        gc.collect()
        arena.close()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACSIM_TENSOR_NATIVE") != "1",
    reason="set UNISIM_TEST_ISAACSIM_TENSOR_NATIVE=1 for real IsaacSim tensor acceptance",
)
def test_native_mapped_worker_selected_reset_parity(tmp_path: Path) -> None:
    """Round-trip selected qpos/qvel through a real IsaacLab articulation."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from unisim.dr.types import ModelSourceDescriptor
    from unisim.entities import SceneEntitySpec
    from unisim.scene import SceneCfg

    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="base">'
        '<geom name="base_collision" size=".1" mass="1"/>'
        '<body name="tip"><joint name="hinge" damping="0"/>'
        '<geom size=".1" mass="1"/></body></body></worldbody>'
        '<actuator><position name="drive" joint="hinge" kp="20" kv="2"/>'
        "</actuator></mujoco>",
        encoding="utf-8",
    )
    config = SceneCfg(
        entity_assets=(
            SceneEntitySpec("robot", ModelSourceDescriptor(str(robot)), root_mode="fixed"),
        )
    )
    backend = IsaacSimBackend(config, 2, 0.002, tensor_cuda_ipc=True)
    try:
        assert backend.get_tensor_capabilities().selected_reset
        views = backend.get_state_views()
        rows = torch.tensor([1], dtype=torch.int64, device=views["qpos"].device)
        qpos = views["qpos"].index_select(0, rows).clone()
        qvel = views["qvel"].index_select(0, rows).clone()
        qpos[:, -1] += 0.25
        qvel[:, -1] -= 0.5
        backend.set_state_tensor(rows, qpos, qvel)
        torch.testing.assert_close(backend.get_state_views()["qpos"][rows], qpos)
        torch.testing.assert_close(backend.get_state_views()["qvel"][rows], qvel)
        del views, rows, qpos, qvel
        gc.collect()
    finally:
        backend.close()


def test_worker_shutdown_closes_imported_cuda_arena_before_legacy_resources() -> None:
    log: list[str] = []
    ctx = SceneWorkerContext.__new__(SceneWorkerContext)

    class _Arena:
        def close(self) -> None:
            log.append("cuda")

    class _Handle:
        def close(self) -> None:
            log.append("shm")

    class _Renderer:
        def shutdown(self) -> None:
            log.append("renderer")

    class _Temporary:
        def cleanup(self) -> None:
            log.append("temporary")

    ctx._cuda_ipc = _Arena()
    ctx._shm_handles = [_Handle()]
    ctx.renderer = _Renderer()
    ctx._temporary = _Temporary()
    ctx.shutdown()
    assert log == ["cuda", "shm", "renderer", "temporary"]
