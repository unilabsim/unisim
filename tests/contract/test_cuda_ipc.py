from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unisim.backend.subprocess_ipc import cuda_ipc


def test_module_is_sdk_free_and_keeps_torch_lazy() -> None:
    code = (
        "import sys; "
        "from unisim.backend.subprocess_ipc import cuda_ipc; "
        "blocked = {'torch', 'isaacgym', 'isaacsim', 'omni'}; "
        "assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules); "
        "assert cuda_ipc.resolve_device_index(2) == 2"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_opaque_handle_descriptor_is_validated_without_interpreting_it() -> None:
    raw = bytes(range(64))
    handle = cuda_ipc.CudaIpcMemHandle(raw, "0" * 32, 256)

    assert handle.abi_version == 1
    assert handle.opaque_handle == raw
    assert raw.hex() not in repr(handle)

    with pytest.raises(ValueError, match="exactly 64 bytes"):
        cuda_ipc.CudaIpcMemHandle(raw[:-1], "0" * 32, 256)
    with pytest.raises(ValueError, match="unsupported CUDA IPC ABI"):
        cuda_ipc.CudaIpcMemHandle(raw, "0" * 32, 256, abi_version=2)
    with pytest.raises(ValueError, match="positive"):
        cuda_ipc.CudaIpcMemHandle(raw, "0" * 32, 0)
    with pytest.raises(ValueError, match="power of two"):
        cuda_ipc.CudaIpcMemHandle(raw, "0" * 32, 256, alignment_bytes=3)
    with pytest.raises(ValueError, match="32 hexadecimal"):
        cuda_ipc.CudaIpcMemHandle(raw, "not-a-uuid", 256)


def test_opaque_event_handle_descriptor_is_validated_without_interpreting_it() -> None:
    raw = bytes(range(64))
    handle = cuda_ipc.CudaIpcEventHandle(raw, "0" * 32)

    assert handle.abi_version == 1
    assert handle.opaque_handle == raw
    assert handle.blocking_sync is False
    assert raw.hex() not in repr(handle)

    with pytest.raises(ValueError, match="exactly 64 bytes"):
        cuda_ipc.CudaIpcEventHandle(raw[:-1], "0" * 32)
    with pytest.raises(ValueError, match="event ABI"):
        cuda_ipc.CudaIpcEventHandle(raw, "0" * 32, abi_version=2)
    with pytest.raises(TypeError, match="blocking_sync"):
        cuda_ipc.CudaIpcEventHandle(raw, "0" * 32, blocking_sync=1)
    with pytest.raises(ValueError, match="32 hexadecimal"):
        cuda_ipc.CudaIpcEventHandle(raw, "not-a-uuid")


def test_driver_unavailable_fails_closed() -> None:
    def unavailable() -> object:
        raise cuda_ipc.CudaIpcError("driver intentionally unavailable")

    original = cuda_ipc._load_driver
    original_handle = cuda_ipc._driver_handle
    cuda_ipc._load_driver = unavailable  # type: ignore[assignment]
    cuda_ipc._driver_handle = None
    try:
        with pytest.raises(cuda_ipc.CudaIpcError, match="driver intentionally unavailable"):
            cuda_ipc.CudaIpcTransport(0)
    finally:
        cuda_ipc._load_driver = original  # type: ignore[assignment]
        cuda_ipc._driver_handle = original_handle


def test_cuda_transport_lifecycle_and_device_validation() -> None:
    if not cuda_ipc.cuda_driver_available():
        pytest.skip("CUDA driver is unavailable")

    transport = cuda_ipc.CudaIpcTransport(0)
    try:
        identity = cuda_ipc.get_device_identity(0)
        assert transport.identity == identity
        assert cuda_ipc.find_device_by_uuid(identity.uuid) == 0

        allocation = transport.allocate(16)
        try:
            allocation.write(b"0123456789abcdef")
            assert allocation.read() == b"0123456789abcdef"
            handle = allocation.export_handle()
            assert handle.device_uuid == identity.uuid
            assert len(handle.opaque_handle) == 64

            forged_uuid = "f" * 32
            mismatch = cuda_ipc.CudaIpcMemHandle(
                handle.opaque_handle, forged_uuid, handle.size_bytes
            )
            with pytest.raises(cuda_ipc.CudaIpcError, match="device mismatch"):
                transport.import_handle(mismatch)
        finally:
            allocation.close()

        allocation.close()
    finally:
        transport.close()


def test_cuda_event_ipc_lifecycle_and_device_validation() -> None:
    if not cuda_ipc.cuda_driver_available():
        pytest.skip("CUDA driver is unavailable")

    transport = cuda_ipc.CudaIpcTransport(0)
    if not transport.event_ipc_supported():
        transport.close()
        pytest.skip("CUDA device does not support IPC events")

    event = transport.create_event()
    try:
        assert event.device_uuid == transport.identity.uuid
        assert event.query() is True
        event.record()
        event.synchronize()

        handle = event.export_handle()
        assert handle.device_uuid == transport.identity.uuid
        assert len(handle.opaque_handle) == 64

        forged_uuid = "f" * 32
        mismatch = cuda_ipc.CudaIpcEventHandle(
            handle.opaque_handle, forged_uuid, handle.blocking_sync
        )
        with pytest.raises(cuda_ipc.CudaIpcError, match="event device mismatch"):
            transport.import_event_handle(mismatch)

        with pytest.raises(cuda_ipc.CudaIpcError, match="while IPC events remain"):
            transport.close()
    finally:
        event.close()
        transport.close()

    with pytest.raises(cuda_ipc.CudaIpcError, match="closed CUDA IPC event"):
        event.record()


@pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_CUDA_IPC_CROSS_PROCESS") != "1",
    reason="set UNISIM_TEST_CUDA_IPC_CROSS_PROCESS=1 for the real cross-process test",
)
def test_cuda_ipc_memory_and_event_handles_are_shared_across_processes() -> None:
    if not cuda_ipc.cuda_driver_available():
        pytest.skip("CUDA driver is unavailable")

    child = r"""
import importlib.util
import json
import sys
from pathlib import Path

module_name = "unisim_cuda_ipc_child"
spec = importlib.util.spec_from_file_location(module_name, sys.argv[2])
cuda_ipc = importlib.util.module_from_spec(spec)
sys.modules[module_name] = cuda_ipc
spec.loader.exec_module(cuda_ipc)
blocked = {"torch", "isaacgym", "isaacsim", "omni"}
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)

payload = json.loads(sys.argv[1])
memory_handle = cuda_ipc.CudaIpcMemHandle(
    bytes.fromhex(payload["memory_handle"]), payload["uuid"], payload["size_bytes"]
)
event_handle = cuda_ipc.CudaIpcEventHandle(
    bytes.fromhex(payload["event_handle"]),
    payload["uuid"],
    payload["blocking_sync"],
)
transport = cuda_ipc.CudaIpcTransport(0)
memory = transport.import_handle(memory_handle)
event = transport.import_event_handle(event_handle)
event.synchronize()
assert memory.read() == b"0123456789abcdef"
memory.write(b"FEDCBA9876543210")
event.record()
event.synchronize()
memory.close()
event.close()
transport.close()
"""
    transport = cuda_ipc.CudaIpcTransport(0)
    if not transport.event_ipc_supported():
        transport.close()
        pytest.skip("CUDA device does not support IPC events")

    allocation = transport.allocate(16)
    event = transport.create_event()
    try:
        allocation.write(b"0123456789abcdef")
        event.record()
        event.synchronize()
        memory_handle = allocation.export_handle()
        event_handle = event.export_handle()
        payload = json.dumps(
            {
                "memory_handle": memory_handle.opaque_handle.hex(),
                "event_handle": event_handle.opaque_handle.hex(),
                "uuid": memory_handle.device_uuid,
                "size_bytes": memory_handle.size_bytes,
                "blocking_sync": event_handle.blocking_sync,
            }
        )
        child_python = os.environ.get("UNISIM_TEST_CUDA_IPC_CHILD_PYTHON", sys.executable)
        result = subprocess.run(
            [child_python, "-c", child, payload, str(Path(cuda_ipc.__file__).resolve())],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr or result.stdout
        event.synchronize()
        assert allocation.read() == b"FEDCBA9876543210"
    finally:
        event.close()
        allocation.close()
        transport.close()


def test_cuda_ipc_module_has_no_isaac_imports() -> None:
    source = Path(cuda_ipc.__file__).read_text(encoding="utf-8")
    assert "isaacgym" not in source.lower()
    assert "isaacsim" not in source.lower()
