"""SDK-free IsaacSim CUDA IPC shutdown and attach-failure checks."""

from __future__ import annotations

import gc
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unisim.backend.isaacsim import backend as isaac_backend
from unisim.backend.isaacsim.backend import IsaacSimBackend


class _Arena:
    def __init__(self) -> None:
        self.closed = False
        self.active_views: set[str] = set()

    def close(self) -> None:
        if self.active_views:
            raise RuntimeError("active views: " + ", ".join(sorted(self.active_views)))
        self.closed = True

    def to_payload(self, _sensor_descriptors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"schema_version": 2}

    device_uuid = "a" * 32
    layout = SimpleNamespace(as_dict=lambda: {"num_envs": 1})


def _make_backend(arena: _Arena | None = None) -> IsaacSimBackend:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._cuda_ipc_arena = arena
    backend._proc = None
    backend._shm_handles = {}
    backend._slots = {}
    backend._stderr_file = None
    backend._worker_dead_error = None
    backend._worker_timeout_s = 2.0
    backend._entity_scene = None
    backend._closed = False
    return backend


def test_close_with_active_view_fails_closed_and_retries_with_the_same_arena() -> None:
    arena = _Arena()
    backend = _make_backend(arena)
    arena.active_views.add("qpos")

    with pytest.raises(RuntimeError, match="active views: qpos"):
        backend.close()
    assert backend._cuda_ipc_arena is arena
    assert not arena.closed

    arena.active_views.clear()
    del arena
    gc.collect()
    backend.close()
    assert backend._cuda_ipc_arena is None
    assert backend._closed


def _worker_code() -> str:
    return r"""
import pickle, struct, sys
(size,) = struct.unpack("<Q", sys.stdin.buffer.read(8))
message = pickle.loads(sys.stdin.buffer.read(size))
assert message["cmd"] == "SHUTDOWN"
body = pickle.dumps({"cmd": "READY", "payload": None}, protocol=4)
sys.stdout.buffer.write(struct.pack("<Q", len(body)))
sys.stdout.buffer.write(body)
sys.stdout.buffer.flush()
"""


def test_backend_close_closes_cuda_arena_and_shuts_down_worker() -> None:
    arena = _Arena()
    backend = _make_backend(arena)
    proc = subprocess.Popen(
        [sys.executable, "-c", _worker_code()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    backend._proc = proc
    try:
        backend.close()
        assert arena.closed
        assert backend._cuda_ipc_arena is None
        assert backend._proc is None
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize("failure", ["request", "handshake"])
def test_failed_cuda_ipc_attach_closes_host_arena_and_keeps_backend_unattached(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    arena = _Arena()
    created: list[_Arena] = []

    def create_arena(**_kwargs: Any) -> _Arena:
        created.append(arena)
        return arena

    monkeypatch.setattr(isaac_backend, "HostCudaIpcArena", create_arena)
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._tensor_cuda_ipc_requested = True
    backend._num_envs = 1
    backend._entity_scene = SimpleNamespace(
        owner=SimpleNamespace(variant_plan=None),
        layout=SimpleNamespace(nq=1, nv=1, nu=1, nbody=1),
    )
    backend._cuda_ipc_arena = None
    backend._require_state = lambda _operation: None  # type: ignore[method-assign]

    def request(cmd: str, _payload: Any, *, expect: str) -> Any:
        assert cmd == "TENSOR_CUDA_ATTACH"
        assert expect == "TENSOR_CUDA_READY"
        if failure == "request":
            raise RuntimeError("worker attach failed")
        return {"device_uuid": "wrong", "layout": {}}

    backend._request = request  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="worker attach failed|invalid CUDA IPC handshake"):
        backend._ensure_cuda_ipc_arena()

    assert created == [arena]
    assert arena.closed
    assert backend._cuda_ipc_arena is None
