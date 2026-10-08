"""SDK-free CUDA IPC attach and detach cleanup checks for IsaacGym."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

import unisim.backend.subprocess_ipc as subprocess_ipc_package
from unisim.backend.isaacgym.tensor import IsaacGymCudaIpcPlan


class _View:
    is_cuda = True

    def __init__(self, source: Any) -> None:
        self.shape = source.shape
        self.device = f"cuda:{source.device_index}"
        self._token = source.token

    def __del__(self) -> None:
        self._token.release()


class _Torch:
    bool = "bool"
    int64 = "int64"
    cuda = SimpleNamespace(
        device_count=lambda: 1,
        is_available=lambda: True,
        current_device=lambda: 0,
        current_stream=lambda _index=0: SimpleNamespace(cuda_stream=100),
    )

    @staticmethod
    def device(*args: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], str):
            return SimpleNamespace(type="cuda", index=0)
        return SimpleNamespace(type="cuda", index=int(args[1]))

    @staticmethod
    def tensor(values: Any, **_kwargs: Any) -> Any:
        return tuple(values)

    @staticmethod
    def zeros(*_args: Any, **_kwargs: Any) -> Any:
        return object()

    @staticmethod
    def ones(*_args: Any, **_kwargs: Any) -> Any:
        return object()

    @staticmethod
    def from_dlpack(source: Any) -> _View:
        return _View(source)


class _Resource:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def close(self) -> None:
        self.log.append(f"close:{self.name}")


class _Allocation(_Resource):
    pointer = 4096

    def export_handle(self) -> Any:
        return SimpleNamespace(
            opaque_handle=b"memory",
            device_uuid="a" * 32,
            size_bytes=1536,
            abi_version=1,
            alignment_bytes=256,
        )


class _Event(_Resource):
    def export_handle(self) -> Any:
        return SimpleNamespace(
            opaque_handle=b"event",
            device_uuid="a" * 32,
            abi_version=1,
            blocking_sync=False,
        )


class _Transport(_Resource):
    device_index = 0
    identity = SimpleNamespace(uuid="a" * 32, name="fake")

    def __init__(self, log: list[str]) -> None:
        super().__init__("transport", log)

    def event_ipc_supported(self) -> bool:
        return True

    def allocate(self, _size: int) -> _Allocation:
        return _Allocation("memory", self.log)

    def create_event(self) -> _Event:
        return _Event("event", self.log)


class _Backend:
    def __init__(self, *, handshake_uuid: str = "b" * 32, detach_error: bool = False):
        self.handshake_uuid = handshake_uuid
        self.detach_error = detach_error
        self.commands: list[str] = []
        self.num_envs = 2
        self._device_id = 0
        self._model_info = SimpleNamespace(use_gpu_pipeline=True, num_dof=1, num_bodies=0)
        self._entity_scene = None
        self._fixed_variant_plan = None
        self._pre_step_control_fn = None
        self._body_wrench_pending = False

    def _require_state(self, operation: str) -> None:
        del operation

    def _request(self, cmd: str, _payload: Any, *, expect: str) -> Any:
        del expect
        self.commands.append(cmd)
        if cmd == "ISAACGYM_CUDA_IPC_ATTACH":
            return {
                "device_uuid": self.handshake_uuid,
                "arena": _payload["arena"] if self.handshake_uuid == "a" * 32 else None,
            }
        if cmd == "ISAACGYM_CUDA_IPC_DETACH" and self.detach_error:
            raise RuntimeError("detach worker failed")
        return None


def _install_fake_cuda(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    fake_cuda = SimpleNamespace(CudaIpcTransport=lambda _index: _Transport(log))
    monkeypatch.setattr(subprocess_ipc_package, "cuda_ipc", fake_cuda, raising=False)
    monkeypatch.setitem(sys.modules, "unisim.backend.subprocess_ipc.cuda_ipc", fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", _Torch())


def test_failed_cuda_ipc_handshake_detaches_worker_and_closes_host_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    _install_fake_cuda(monkeypatch, log)
    backend = _Backend(handshake_uuid="b" * 32)

    with pytest.raises(RuntimeError, match="same-GPU handshake failed"):
        IsaacGymCudaIpcPlan(backend)

    assert backend.commands == ["ISAACGYM_CUDA_IPC_ATTACH", "ISAACGYM_CUDA_IPC_DETACH"]
    assert log == [
        "close:event",
        "close:event",
        "close:event",
        "close:memory",
        "close:transport",
    ]


def test_failed_detach_still_closes_local_cuda_ipc_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    _install_fake_cuda(monkeypatch, log)
    backend = _Backend(handshake_uuid="a" * 32, detach_error=True)
    plan = IsaacGymCudaIpcPlan(backend)
    assert not plan.closed

    with pytest.raises(RuntimeError, match="detach worker failed"):
        plan.close()

    assert plan.closed
    assert log == [
        "close:event",
        "close:event",
        "close:event",
        "close:memory",
        "close:transport",
    ]
