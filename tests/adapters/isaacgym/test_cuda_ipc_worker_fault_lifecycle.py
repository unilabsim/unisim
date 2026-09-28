"""SDK-free Isaac worker death and reopen lifecycle checks."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.isaacgym.tensor import IsaacGymCudaIpcWorkerRuntime
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.subprocess_ipc.backend import SubprocessWorkerError

_WORKER_CODE = r"""
import os
import pickle
import struct
import sys


def read_exact(size):
    data = b""
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def recv():
    (size,) = struct.unpack("<Q", read_exact(8))
    return pickle.loads(read_exact(size))


def send(cmd, payload=None):
    body = pickle.dumps({"cmd": cmd, "payload": payload}, protocol=4)
    sys.stdout.buffer.write(struct.pack("<Q", len(body)))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


while True:
    try:
        message = recv()
    except EOFError:
        break
    command = message["cmd"]
    if command == "READY":
        send("READY")
    elif command == "EOF":
        break
    elif command == "CRASH":
        sys.stdout.buffer.flush()
        os._exit(87)
    elif command == "SHUTDOWN":
        send("READY")
        break
"""


def _spawn_worker() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", _WORKER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def _make_backend(cls: type[Any], proc: subprocess.Popen[bytes]) -> Any:
    backend = cls.__new__(cls)
    backend._proc = proc
    backend._shm_handles = {}
    backend._slots = {}
    backend._stderr_file = None
    backend._worker_dead_error = None
    backend._worker_timeout_s = 2.0
    backend._entity_scene = None
    backend._closed = False
    return backend


@pytest.mark.parametrize("backend_cls", [IsaacGymBackend, IsaacSimBackend])
@pytest.mark.parametrize("death_command", ["EOF", "CRASH"])
def test_tensor_backend_fail_closes_after_worker_pipe_death_and_reopens(
    backend_cls: type[Any], death_command: str
) -> None:
    proc = _spawn_worker()
    backend = _make_backend(backend_cls, proc)
    replacement: subprocess.Popen[bytes] | None = None
    try:
        assert backend._request("READY", None, expect="READY") is None

        pattern = f"worker closed its pipe during {death_command}"
        with pytest.raises(SubprocessWorkerError, match=pattern):
            backend._request(death_command, None, expect="READY")
        assert proc.wait(timeout=5) == (0 if death_command == "EOF" else 87)
        assert backend._worker_dead_error is not None

        with pytest.raises(SubprocessWorkerError, match="worker is unavailable"):
            backend._request("READY", None, expect="READY")

        # Reopening is a new host/worker pair, not silent use of a dead pipe.
        # It proves the fail-closed state belongs to the dead backend instance.
        replacement = _spawn_worker()
        reopened = _make_backend(backend_cls, replacement)
        assert reopened._request("READY", None, expect="READY") is None
        reopened.close()
        assert replacement.wait(timeout=5) == 0
        assert reopened._proc is None
        replacement = None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if replacement is not None and replacement.poll() is None:
            replacement.kill()
            replacement.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()


def test_cuda_ipc_worker_reset_failure_marks_scene_faulted() -> None:
    torch = pytest.importorskip("torch")
    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.closed = False
    runtime.device_index = 0
    runtime.expected_reset_sequence = 0
    runtime.reset_indices = torch.zeros((1,), dtype=torch.int64)
    runtime.reset_qpos = torch.zeros((1, 1), dtype=torch.float32)
    runtime.reset_qvel = torch.zeros((1, 1), dtype=torch.float32)
    runtime.root_projections = []
    runtime.joint_projections = [
        {
            "dof_ids": torch.zeros((1, 1), dtype=torch.int64),
            "actor_ids": torch.zeros((1, 1), dtype=torch.int32),
            "qpos": torch.zeros((1,), dtype=torch.int64),
            "qvel": torch.zeros((1,), dtype=torch.int64),
        }
    ]
    scene: dict[str, Any] = {
        "faulted": False,
        "pending_roots": {7: object()},
        "pending_dofs": {8: object()},
        "pending_dof_actors": {8},
    }
    runtime.ctx = SimpleNamespace(
        torch=torch,
        device="cpu",
        scene_worker=SimpleNamespace(**scene),
    )
    runtime.reset_event = SimpleNamespace(wait_stream=lambda stream: None)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("native selected DOF-state setter failed")

    runtime._submit_native_dofs = fail
    with pytest.raises(RuntimeError, match="native selected DOF-state setter failed"):
        runtime.set_state({"count": 1, "sequence": 1})
    assert scene["faulted"] is True
    assert scene["pending_roots"] and scene["pending_dofs"]
    with pytest.raises(RuntimeError, match="IsaacGym scene is faulted"):
        runtime.set_state({"count": 1, "sequence": 2})


def test_cuda_ipc_worker_step_failure_marks_scene_faulted() -> None:
    torch = pytest.importorskip("torch")
    scene: dict[str, Any] = {
        "faulted": False,
        "pending_roots": {7: object()},
        "pending_dofs": {8: object()},
        "pending_dof_actors": {8},
        "targets": torch.zeros((1,), dtype=torch.float32),
    }
    runtime = IsaacGymCudaIpcWorkerRuntime.__new__(IsaacGymCudaIpcWorkerRuntime)
    runtime.closed = False
    runtime.arena = SimpleNamespace(nu=1)
    runtime.control_dofs = torch.zeros((1,), dtype=torch.int64)
    runtime.ctrl = torch.zeros((1,), dtype=torch.float32)
    runtime.control_event = SimpleNamespace(wait_stream=lambda stream: None)
    runtime.ctx = SimpleNamespace(
        torch=torch,
        sim=object(),
        gym=SimpleNamespace(set_dof_position_target_tensor=lambda *_args: False),
        gymtorch=SimpleNamespace(unwrap_tensor=lambda value: value),
        scene_worker=SimpleNamespace(**scene),
    )

    with pytest.raises(RuntimeError, match="native DOF position target setter failed"):
        runtime.step({"nsteps": 2})
    assert scene["faulted"] is True
    assert scene["pending_roots"] and scene["pending_dofs"]
    with pytest.raises(RuntimeError, match="IsaacGym scene is faulted"):
        runtime.step({"nsteps": 1})
