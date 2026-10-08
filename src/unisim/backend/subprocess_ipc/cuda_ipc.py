"""Minimal SDK-free CUDA IPC transport used by subprocess tensor backends.

The transport deliberately talks to CUDA's stable driver ABI through ``ctypes``.
It never imports an Isaac SDK and imports Torch only to parse an optional Torch
device spelling.  The 64-byte IPC handle is opaque on the wire; in particular,
this module does not serialize Torch's private CUDA storage handles.
"""

from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_CUDA_IPC_HANDLE_SIZE = 64
_CUDA_IPC_ABI_VERSION = 1
_CUDA_IPC_EVENT_ABI_VERSION = 1
_CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
_CU_EVENT_BLOCKING_SYNC = 0x1
_CU_EVENT_DISABLE_TIMING = 0x2
_CU_EVENT_INTERPROCESS = 0x4
_CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125
_CUDA_ERROR_NOT_READY = 600
_MAX_DEVICE_NAME_LENGTH = 256
_DEFAULT_DRIVER_NAMES = ("libcuda.so.1", "libcuda.so", "nvcuda.dll")


class CudaIpcError(RuntimeError):
    """Raised when a CUDA IPC operation cannot be completed safely."""


@dataclass(frozen=True)
class CudaDeviceIdentity:
    """The local ordinal and immutable UUID of one visible CUDA device."""

    index: int
    uuid: str
    name: str


@dataclass(frozen=True)
class CudaIpcMemHandle:
    """A serializable, backend-agnostic CUDA IPC handle.

    ``opaque_handle`` must contain exactly the bytes returned by
    ``cuIpcGetMemHandle``.  Consumers must not interpret offsets, reserved
    fields, or pointers from it.
    """

    opaque_handle: bytes = field(repr=False)
    device_uuid: str
    size_bytes: int
    alignment_bytes: int = 256
    abi_version: int = _CUDA_IPC_ABI_VERSION

    def __post_init__(self) -> None:
        if len(self.opaque_handle) != _CUDA_IPC_HANDLE_SIZE:
            raise ValueError(f"CUDA IPC handle must contain exactly {_CUDA_IPC_HANDLE_SIZE} bytes")
        if self.abi_version != _CUDA_IPC_ABI_VERSION:
            raise ValueError(f"unsupported CUDA IPC ABI version {self.abi_version}")
        if self.size_bytes <= 0:
            raise ValueError("CUDA IPC allocation size must be positive")
        if self.alignment_bytes <= 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise ValueError("CUDA IPC alignment must be a positive power of two")
        _validate_uuid(self.device_uuid)


@dataclass(frozen=True)
class CudaIpcEventHandle:
    """A serializable, backend-agnostic CUDA IPC event handle.

    ``opaque_handle`` must contain exactly the bytes returned by
    ``cuIpcGetEventHandle``.  ``device_uuid`` is transport metadata used for a
    physical-device handshake; CUDA deliberately keeps the handle itself
    opaque, so this module never interprets its contents.
    """

    opaque_handle: bytes = field(repr=False)
    device_uuid: str
    blocking_sync: bool = False
    abi_version: int = _CUDA_IPC_EVENT_ABI_VERSION

    def __post_init__(self) -> None:
        if len(self.opaque_handle) != _CUDA_IPC_HANDLE_SIZE:
            raise ValueError(
                f"CUDA IPC event handle must contain exactly {_CUDA_IPC_HANDLE_SIZE} bytes"
            )
        if self.abi_version != _CUDA_IPC_EVENT_ABI_VERSION:
            raise ValueError(f"unsupported CUDA IPC event ABI version {self.abi_version}")
        if not isinstance(self.blocking_sync, bool):
            raise TypeError("CUDA IPC event blocking_sync must be a boolean")
        _validate_uuid(self.device_uuid)


class CudaIpcAllocation:
    """A local ``cuMemAlloc`` allocation owned by this process."""

    def __init__(
        self,
        transport: CudaIpcTransport,
        pointer: int,
        size_bytes: int,
        identity: CudaDeviceIdentity,
    ) -> None:
        self._transport = transport
        self._pointer = pointer
        self.size_bytes = size_bytes
        self.device_uuid = identity.uuid
        self.closed = False

    @property
    def pointer(self) -> int:
        if self.closed:
            raise CudaIpcError("CUDA IPC allocation is closed")
        return self._pointer

    def export_handle(self) -> CudaIpcMemHandle:
        """Export this allocation without exposing a Torch-internal handle."""

        if self.closed:
            raise CudaIpcError("cannot export a closed CUDA IPC allocation")
        with self._transport._device_context():
            raw = _RawIpcHandle()
            result = _driver().cuIpcGetMemHandle(ctypes.byref(raw), self._pointer)
            _check(result, "cuIpcGetMemHandle")
        return CudaIpcMemHandle(
            opaque_handle=bytes(raw),
            device_uuid=self.device_uuid,
            size_bytes=self.size_bytes,
        )

    def write(self, payload: Any, *, offset: int = 0) -> None:
        """Synchronously copy host bytes into device memory for diagnostics."""

        if self.closed:
            raise CudaIpcError("cannot write to a closed CUDA IPC allocation")
        data = memoryview(payload).tobytes() if not isinstance(payload, bytes) else payload
        _check_offset(offset, len(data), self.size_bytes)
        if not data:
            return
        host = (ctypes.c_char * len(data)).from_buffer_copy(data)
        with self._transport._device_context():
            result = _driver().cuMemcpyHtoD(self._pointer + offset, host, len(data))
            _check(result, "cuMemcpyHtoD")

    def read(self, size_bytes: int | None = None, *, offset: int = 0) -> bytes:
        """Synchronously copy device bytes to host memory for diagnostics."""

        if self.closed:
            raise CudaIpcError("cannot read from a closed CUDA IPC allocation")
        count = self.size_bytes if size_bytes is None else size_bytes
        _check_offset(offset, count, self.size_bytes)
        if count == 0:
            return b""
        output = (ctypes.c_char * count)()
        with self._transport._device_context():
            result = _driver().cuMemcpyDtoH(output, self._pointer + offset, count)
            _check(result, "cuMemcpyDtoH")
        return bytes(output)

    def close(self) -> None:
        if self.closed:
            return
        with self._transport._device_context():
            result = _driver().cuMemFree(self._pointer)
            _check(result, "cuMemFree")
        self._transport._forget_allocation(self)
        self.closed = True


class CudaIpcImportedMemory:
    """A device mapping created by ``cuIpcOpenMemHandle``."""

    def __init__(
        self,
        transport: CudaIpcTransport,
        pointer: int,
        handle: CudaIpcMemHandle,
    ) -> None:
        self._transport = transport
        self._pointer = pointer
        self.handle = handle
        self.closed = False

    @property
    def pointer(self) -> int:
        if self.closed:
            raise CudaIpcError("imported CUDA IPC memory is closed")
        return self._pointer

    @property
    def size_bytes(self) -> int:
        return self.handle.size_bytes

    @property
    def device_uuid(self) -> str:
        return self.handle.device_uuid

    def write(self, payload: Any, *, offset: int = 0) -> None:
        if self.closed:
            raise CudaIpcError("cannot write to closed imported CUDA IPC memory")
        data = memoryview(payload).tobytes() if not isinstance(payload, bytes) else payload
        _check_offset(offset, len(data), self.handle.size_bytes)
        if not data:
            return
        host = (ctypes.c_char * len(data)).from_buffer_copy(data)
        with self._transport._device_context():
            result = _driver().cuMemcpyHtoD(self._pointer + offset, host, len(data))
            _check(result, "cuMemcpyHtoD")

    def read(self, size_bytes: int | None = None, *, offset: int = 0) -> bytes:
        if self.closed:
            raise CudaIpcError("cannot read from closed imported CUDA IPC memory")
        count = self.handle.size_bytes if size_bytes is None else size_bytes
        _check_offset(offset, count, self.handle.size_bytes)
        if count == 0:
            return b""
        output = (ctypes.c_char * count)()
        with self._transport._device_context():
            result = _driver().cuMemcpyDtoH(output, self._pointer + offset, count)
            _check(result, "cuMemcpyDtoH")
        return bytes(output)

    def close(self) -> None:
        if self.closed:
            return
        with self._transport._device_context():
            result = _driver().cuIpcCloseMemHandle(self._pointer)
            _check(result, "cuIpcCloseMemHandle")
        self._transport._forget_import(self)
        self.closed = True


class CudaIpcEvent:
    """A local interprocess CUDA event owned by this process."""

    def __init__(
        self,
        transport: CudaIpcTransport,
        pointer: int,
        identity: CudaDeviceIdentity,
        blocking_sync: bool,
    ) -> None:
        self._transport = transport
        self._pointer = pointer
        self.device_uuid = identity.uuid
        self.blocking_sync = blocking_sync
        self.closed = False

    @property
    def pointer(self) -> int:
        if self.closed:
            raise CudaIpcError("CUDA IPC event is closed")
        return self._pointer

    def export_handle(self) -> CudaIpcEventHandle:
        """Export this event without exposing a runtime-private object."""

        if self.closed:
            raise CudaIpcError("cannot export a closed CUDA IPC event")
        self._transport._require_event_ipc_support()
        with self._transport._device_context():
            raw = _RawIpcHandle()
            result = _driver().cuIpcGetEventHandle(
                ctypes.byref(raw), ctypes.c_void_p(self._pointer)
            )
            _check(result, "cuIpcGetEventHandle")
        return CudaIpcEventHandle(
            opaque_handle=bytes(raw),
            device_uuid=self.device_uuid,
            blocking_sync=self.blocking_sync,
        )

    def record(self, stream: int | None = None) -> None:
        """Record the current contents of a raw CUDA stream into this event."""

        if self.closed:
            raise CudaIpcError("cannot record a closed CUDA IPC event")
        _validate_raw_handle(stream, "stream")
        with self._transport._device_context():
            result = _driver().cuEventRecord(
                ctypes.c_void_p(self._pointer), ctypes.c_void_p(stream)
            )
            _check(result, "cuEventRecord")

    def synchronize(self) -> None:
        """Wait on the CPU for the most recent event record."""

        if self.closed:
            raise CudaIpcError("cannot synchronize a closed CUDA IPC event")
        with self._transport._device_context():
            result = _driver().cuEventSynchronize(ctypes.c_void_p(self._pointer))
            _check(result, "cuEventSynchronize")

    def query(self) -> bool:
        """Return whether the most recently captured work has completed."""

        if self.closed:
            raise CudaIpcError("cannot query a closed CUDA IPC event")
        with self._transport._device_context():
            result = _driver().cuEventQuery(ctypes.c_void_p(self._pointer))
        if result == _CUDA_ERROR_NOT_READY:
            return False
        _check(result, "cuEventQuery")
        return True

    def wait_stream(self, stream: int) -> None:
        """Make a raw CUDA stream wait on this event without a CPU round trip."""

        if self.closed:
            raise CudaIpcError("cannot wait on a closed CUDA IPC event")
        _validate_raw_handle(stream, "stream", allow_none=False)
        with self._transport._device_context():
            result = _driver().cuStreamWaitEvent(
                ctypes.c_void_p(stream), ctypes.c_void_p(self._pointer), ctypes.c_uint(0)
            )
            _check(result, "cuStreamWaitEvent")

    def close(self) -> None:
        if self.closed:
            return
        with self._transport._device_context():
            result = _driver().cuEventDestroy(ctypes.c_void_p(self._pointer))
            _check(result, "cuEventDestroy")
        self._transport._forget_event(self)
        self.closed = True


class CudaIpcImportedEvent:
    """An imported CUDA event opened from an opaque IPC handle."""

    def __init__(
        self,
        transport: CudaIpcTransport,
        pointer: int,
        handle: CudaIpcEventHandle,
    ) -> None:
        self._transport = transport
        self._pointer = pointer
        self.handle = handle
        self.closed = False

    @property
    def pointer(self) -> int:
        if self.closed:
            raise CudaIpcError("imported CUDA IPC event is closed")
        return self._pointer

    @property
    def device_uuid(self) -> str:
        return self.handle.device_uuid

    @property
    def blocking_sync(self) -> bool:
        return self.handle.blocking_sync

    def record(self, stream: int | None = None) -> None:
        """Record the current contents of a raw CUDA stream into this event."""

        if self.closed:
            raise CudaIpcError("cannot record a closed imported CUDA IPC event")
        _validate_raw_handle(stream, "stream")
        with self._transport._device_context():
            result = _driver().cuEventRecord(
                ctypes.c_void_p(self._pointer), ctypes.c_void_p(stream)
            )
            _check(result, "cuEventRecord")

    def synchronize(self) -> None:
        """Wait on the CPU for the most recent event record."""

        if self.closed:
            raise CudaIpcError("cannot synchronize a closed imported CUDA IPC event")
        with self._transport._device_context():
            result = _driver().cuEventSynchronize(ctypes.c_void_p(self._pointer))
            _check(result, "cuEventSynchronize")

    def query(self) -> bool:
        """Return whether the most recently captured work has completed."""

        if self.closed:
            raise CudaIpcError("cannot query a closed imported CUDA IPC event")
        with self._transport._device_context():
            result = _driver().cuEventQuery(ctypes.c_void_p(self._pointer))
        if result == _CUDA_ERROR_NOT_READY:
            return False
        _check(result, "cuEventQuery")
        return True

    def wait_stream(self, stream: int) -> None:
        """Make a raw CUDA stream wait on this event without a CPU round trip."""

        if self.closed:
            raise CudaIpcError("cannot wait on a closed imported CUDA IPC event")
        _validate_raw_handle(stream, "stream", allow_none=False)
        with self._transport._device_context():
            result = _driver().cuStreamWaitEvent(
                ctypes.c_void_p(stream), ctypes.c_void_p(self._pointer), ctypes.c_uint(0)
            )
            _check(result, "cuStreamWaitEvent")

    def close(self) -> None:
        if self.closed:
            return
        with self._transport._device_context():
            result = _driver().cuEventDestroy(ctypes.c_void_p(self._pointer))
            _check(result, "cuEventDestroy")
        self._transport._forget_imported_event(self)
        self.closed = True


class CudaIpcTransport:
    """One thread-local primary-context owner for a physical CUDA device.

    Allocation and import operations retain the device's primary context.  The
    transport can coexist with Torch and the Isaac runtimes because all of them
    use the CUDA primary context; it does not create a private context.
    """

    def __init__(self, device_index: int | str | Any) -> None:
        self.device_index = resolve_device_index(device_index)
        self.identity = get_device_identity(self.device_index)
        self.closed = False
        self._lock = threading.RLock()
        self._active_allocations: set[int] = set()
        self._active_imports: set[int] = set()
        self._active_events: set[int] = set()
        self._active_imported_events: set[int] = set()

        driver = _driver()
        _check(driver.cuInit(0), "cuInit")
        context = ctypes.c_void_p()
        result = driver.cuDevicePrimaryCtxRetain(
            ctypes.byref(context), ctypes.c_int(self.device_index)
        )
        _check(result, "cuDevicePrimaryCtxRetain")
        if not context.value:
            raise CudaIpcError("CUDA returned a null primary context")
        self._context = context

    @contextmanager
    def _device_context(self) -> Iterator[None]:
        if self.closed:
            raise CudaIpcError("CUDA IPC transport is closed")
        driver = _driver()
        previous = ctypes.c_void_p()
        _check(driver.cuCtxGetCurrent(ctypes.byref(previous)), "cuCtxGetCurrent")
        _check(driver.cuCtxSetCurrent(self._context), "cuCtxSetCurrent")
        try:
            yield
        finally:
            _check(driver.cuCtxSetCurrent(previous), "cuCtxSetCurrent")

    def allocate(self, size_bytes: int) -> CudaIpcAllocation:
        """Allocate exportable CUDA device memory with ``cuMemAlloc``."""

        size_bytes = _validate_size(size_bytes)
        with self._lock, self._device_context():
            pointer = ctypes.c_ulonglong()
            result = _driver().cuMemAlloc(ctypes.byref(pointer), ctypes.c_size_t(size_bytes))
            _check(result, "cuMemAlloc")
            allocation = CudaIpcAllocation(
                transport=self,
                pointer=int(pointer.value),
                size_bytes=size_bytes,
                identity=self.identity,
            )
            self._active_allocations.add(allocation.pointer)
            return allocation

    def import_handle(
        self, handle: CudaIpcMemHandle, *, device_index: int | str | Any | None = None
    ) -> CudaIpcImportedMemory:
        """Import an opaque handle after validating the physical device UUID."""

        index = self.device_index if device_index is None else resolve_device_index(device_index)
        identity = get_device_identity(index)
        if identity.uuid != handle.device_uuid:
            raise CudaIpcError(
                "CUDA IPC device mismatch: handle belongs to "
                f"{handle.device_uuid}, local device {index} is {identity.uuid}"
            )
        if index != self.device_index:
            raise CudaIpcError(
                f"CUDA IPC transport is bound to device {self.device_index}, got {index}"
            )
        with self._lock, self._device_context():
            raw = _RawIpcHandle.from_buffer_copy(handle.opaque_handle)
            pointer = ctypes.c_ulonglong()
            result = _driver().cuIpcOpenMemHandle(
                ctypes.byref(pointer),
                raw,
                ctypes.c_uint(_CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
            )
            _check(result, "cuIpcOpenMemHandle")
            imported = CudaIpcImportedMemory(
                transport=self,
                pointer=int(pointer.value),
                handle=handle,
            )
            self._active_imports.add(imported.pointer)
            return imported

    def event_ipc_supported(self) -> bool:
        """Return whether the bound device reports CUDA IPC event support."""

        if not getattr(_driver(), "_cuda_ipc_event_api_available", False):
            raise CudaIpcError("CUDA driver does not expose the stable IPC event API")
        with self._lock, self._device_context():
            supported = ctypes.c_int()
            result = _driver().cuDeviceGetAttribute(
                ctypes.byref(supported),
                ctypes.c_int(_CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED),
                ctypes.c_int(self.device_index),
            )
            _check(result, "cuDeviceGetAttribute")
        return supported.value != 0

    def _require_event_ipc_support(self) -> None:
        if not self.event_ipc_supported():
            raise CudaIpcError(f"CUDA device {self.device_index} does not report IPC event support")

    def create_event(self, *, blocking_sync: bool = False) -> CudaIpcEvent:
        """Create an exportable, timing-disabled interprocess CUDA event."""

        if not isinstance(blocking_sync, bool):
            raise TypeError("blocking_sync must be a boolean")
        self._require_event_ipc_support()
        flags = _CU_EVENT_DISABLE_TIMING | _CU_EVENT_INTERPROCESS
        if blocking_sync:
            flags |= _CU_EVENT_BLOCKING_SYNC
        with self._lock, self._device_context():
            event = ctypes.c_void_p()
            result = _driver().cuEventCreate(ctypes.byref(event), ctypes.c_uint(flags))
            _check(result, "cuEventCreate")
            if not event.value:
                raise CudaIpcError("CUDA returned a null interprocess event")
            created = CudaIpcEvent(
                transport=self,
                pointer=int(event.value),
                identity=self.identity,
                blocking_sync=blocking_sync,
            )
            self._active_events.add(created.pointer)
            return created

    def import_event_handle(
        self, handle: CudaIpcEventHandle, *, device_index: int | str | Any | None = None
    ) -> CudaIpcImportedEvent:
        """Import an opaque event handle after validating the device UUID."""

        index = self.device_index if device_index is None else resolve_device_index(device_index)
        identity = get_device_identity(index)
        if identity.uuid != handle.device_uuid:
            raise CudaIpcError(
                "CUDA IPC event device mismatch: handle belongs to "
                f"{handle.device_uuid}, local device {index} is {identity.uuid}"
            )
        if index != self.device_index:
            raise CudaIpcError(
                f"CUDA IPC transport is bound to device {self.device_index}, got {index}"
            )
        self._require_event_ipc_support()
        with self._lock, self._device_context():
            raw = _RawIpcHandle.from_buffer_copy(handle.opaque_handle)
            event = ctypes.c_void_p()
            result = _driver().cuIpcOpenEventHandle(ctypes.byref(event), raw)
            _check(result, "cuIpcOpenEventHandle")
            if not event.value:
                raise CudaIpcError("CUDA returned a null imported IPC event")
            imported = CudaIpcImportedEvent(
                transport=self,
                pointer=int(event.value),
                handle=handle,
            )
            self._active_imported_events.add(imported.pointer)
            return imported

    def _forget_allocation(self, allocation: CudaIpcAllocation) -> None:
        with self._lock:
            self._active_allocations.discard(allocation.pointer)

    def _forget_import(self, imported: CudaIpcImportedMemory) -> None:
        with self._lock:
            self._active_imports.discard(imported.pointer)

    def _forget_event(self, event: CudaIpcEvent) -> None:
        with self._lock:
            self._active_events.discard(event.pointer)

    def _forget_imported_event(self, event: CudaIpcImportedEvent) -> None:
        with self._lock:
            self._active_imported_events.discard(event.pointer)

    def close(self) -> None:
        """Release the retained primary context after all mappings are closed."""

        if self.closed:
            return
        if self._active_allocations or self._active_imports:
            raise CudaIpcError("cannot close CUDA IPC transport while device mappings remain")
        if self._active_events or self._active_imported_events:
            raise CudaIpcError("cannot close CUDA IPC transport while IPC events remain")
        driver = _driver()
        previous = ctypes.c_void_p()
        _check(driver.cuCtxGetCurrent(ctypes.byref(previous)), "cuCtxGetCurrent")
        _check(driver.cuCtxSetCurrent(None), "cuCtxSetCurrent")
        result = driver.cuDevicePrimaryCtxRelease(ctypes.c_int(self.device_index))
        _check(result, "cuDevicePrimaryCtxRelease")
        if previous.value is not None and previous.value != self._context.value:
            _check(driver.cuCtxSetCurrent(previous), "cuCtxSetCurrent")
        self.closed = True


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _RawIpcHandle(ctypes.Structure):
    # Passing the handle as a real C struct is significant: ctypes passes an
    # array by reference, while CUDA's by-value ``CUipcMemHandle`` ABI requires
    # the 64-byte struct.

    _fields_ = [("reserved", ctypes.c_ubyte * _CUDA_IPC_HANDLE_SIZE)]


_driver_lock = threading.Lock()
_driver_handle: Any = None
_driver_failed: str | None = None


def _load_driver() -> Any:
    global _driver_handle, _driver_failed
    with _driver_lock:
        if _driver_handle is not None:
            return _driver_handle
        if _driver_failed is not None:
            raise CudaIpcError(_driver_failed)
        errors: list[str] = []
        for name in _DEFAULT_DRIVER_NAMES:
            try:
                library = ctypes.CDLL(name)
                break
            except OSError as error:
                errors.append(f"{name}: {error}")
        else:
            _driver_failed = "cannot load the CUDA driver library (" + "; ".join(errors) + ")"
            raise CudaIpcError(_driver_failed)
        _configure_driver(library)
        _driver_handle = library
        return library


def _configure_driver(driver: Any) -> None:
    driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorString.restype = ctypes.c_int
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetName.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetName.restype = ctypes.c_int
    driver.cuDeviceGetUuid.argtypes = [ctypes.POINTER(_CUuuid), ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    driver.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    driver.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
    driver.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
    driver.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent.restype = ctypes.c_int
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    driver.cuCtxSetCurrent.restype = ctypes.c_int
    # CUDA headers map these names to their versioned ABI entry points.  Some
    # exported unversioned symbols remain for legacy binaries and require a
    # legacy context, so prefer the same v2 entry point used by current CUDA.
    driver.cuMemAlloc = getattr(driver, "cuMemAlloc_v2", driver.cuMemAlloc)
    driver.cuMemFree = getattr(driver, "cuMemFree_v2", driver.cuMemFree)
    driver.cuMemcpyHtoD = getattr(driver, "cuMemcpyHtoD_v2", driver.cuMemcpyHtoD)
    driver.cuMemcpyDtoH = getattr(driver, "cuMemcpyDtoH_v2", driver.cuMemcpyDtoH)
    driver.cuIpcOpenMemHandle = getattr(driver, "cuIpcOpenMemHandle_v2", driver.cuIpcOpenMemHandle)
    driver.cuMemAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_size_t,
    ]
    driver.cuMemAlloc.restype = ctypes.c_int
    driver.cuMemFree.argtypes = [ctypes.c_ulonglong]
    driver.cuMemFree.restype = ctypes.c_int
    driver.cuMemcpyHtoD.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    driver.cuMemcpyHtoD.restype = ctypes.c_int
    driver.cuMemcpyDtoH.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_size_t,
    ]
    driver.cuMemcpyDtoH.restype = ctypes.c_int
    driver.cuIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_RawIpcHandle),
        ctypes.c_ulonglong,
    ]
    driver.cuIpcGetMemHandle.restype = ctypes.c_int
    driver.cuIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        _RawIpcHandle,
        ctypes.c_uint,
    ]
    driver.cuIpcOpenMemHandle.restype = ctypes.c_int
    driver.cuIpcCloseMemHandle.argtypes = [ctypes.c_ulonglong]
    driver.cuIpcCloseMemHandle.restype = ctypes.c_int
    # CUDA's public driver header maps ``cuEventDestroy`` to its v2 symbol.
    # Current drivers export ``cuEventCreate`` rather than the runtime API's
    # ``cuEventCreateWithFlags`` spelling; both have the same ABI.  Configure
    # the complete event group only when every required symbol is present so a
    # driver without IPC events cannot leave a partially usable transport.
    event_create: Any = getattr(driver, "cuEventCreateWithFlags", None)
    if event_create is None:
        event_create = getattr(driver, "cuEventCreate", None)
    event_record: Any = getattr(driver, "cuEventRecord", None)
    event_synchronize: Any = getattr(driver, "cuEventSynchronize", None)
    event_query: Any = getattr(driver, "cuEventQuery", None)
    event_destroy: Any = getattr(driver, "cuEventDestroy_v2", None)
    if event_destroy is None:
        event_destroy = getattr(driver, "cuEventDestroy", None)
    stream_wait_event: Any = getattr(driver, "cuStreamWaitEvent", None)
    ipc_get_event: Any = getattr(driver, "cuIpcGetEventHandle", None)
    ipc_open_event: Any = getattr(driver, "cuIpcOpenEventHandle", None)
    if all(
        symbol is not None
        for symbol in (
            event_create,
            event_record,
            event_synchronize,
            event_query,
            event_destroy,
            stream_wait_event,
            ipc_get_event,
            ipc_open_event,
        )
    ):
        driver.cuEventCreate = event_create
        driver.cuEventRecord = event_record
        driver.cuEventSynchronize = event_synchronize
        driver.cuEventQuery = event_query
        driver.cuEventDestroy = event_destroy
        driver.cuStreamWaitEvent = stream_wait_event
        driver.cuIpcGetEventHandle = ipc_get_event
        driver.cuIpcOpenEventHandle = ipc_open_event
        driver.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        driver.cuEventCreate.restype = ctypes.c_int
        driver.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        driver.cuEventRecord.restype = ctypes.c_int
        driver.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        driver.cuEventSynchronize.restype = ctypes.c_int
        driver.cuEventQuery.argtypes = [ctypes.c_void_p]
        driver.cuEventQuery.restype = ctypes.c_int
        driver.cuEventDestroy.argtypes = [ctypes.c_void_p]
        driver.cuEventDestroy.restype = ctypes.c_int
        driver.cuStreamWaitEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        driver.cuStreamWaitEvent.restype = ctypes.c_int
        driver.cuIpcGetEventHandle.argtypes = [
            ctypes.POINTER(_RawIpcHandle),
            ctypes.c_void_p,
        ]
        driver.cuIpcGetEventHandle.restype = ctypes.c_int
        driver.cuIpcOpenEventHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _RawIpcHandle,
        ]
        driver.cuIpcOpenEventHandle.restype = ctypes.c_int
        driver._cuda_ipc_event_api_available = True


def _driver() -> Any:
    driver = _load_driver()
    driver.cuInit(0)  # Idempotent; retain errors are reported by callers.
    return driver


def _check(result: int, function: str) -> None:
    if result == 0:
        return
    driver = _driver_handle
    message = ctypes.c_char_p()
    text: str | None = None
    if driver is not None:
        if driver.cuGetErrorString(result, ctypes.byref(message)) == 0 and message.value:
            text = message.value.decode("utf-8", errors="replace")
    detail = f" ({text})" if text else ""
    raise CudaIpcError(f"{function} failed with CUDA error {result}{detail}")


def _validate_uuid(uuid: str) -> None:
    if len(uuid) != 32:
        raise ValueError("CUDA device UUID must contain exactly 32 hexadecimal characters")
    try:
        int(uuid, 16)
    except ValueError as error:
        raise ValueError("CUDA device UUID must be hexadecimal") from error
    if uuid.lower() != uuid:
        raise ValueError("CUDA device UUID must use lowercase hexadecimal")


def _validate_size(size_bytes: int) -> int:
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
        raise TypeError("CUDA IPC allocation size must be an integer")
    if size_bytes <= 0:
        raise ValueError("CUDA IPC allocation size must be positive")
    return size_bytes


def _check_offset(offset: int, count: int, size_bytes: int) -> None:
    if offset < 0 or count < 0 or offset > size_bytes or count > size_bytes - offset:
        raise ValueError("CUDA IPC operation is outside the allocation bounds")


def _validate_raw_handle(handle: int | None, name: str, *, allow_none: bool = True) -> None:
    if handle is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be a non-null raw CUDA handle")
    if not isinstance(handle, int) or isinstance(handle, bool):
        raise TypeError(f"{name} must be an integer raw CUDA handle or None")
    if handle < 0:
        raise ValueError(f"{name} must be a non-negative raw CUDA handle")


def get_device_identity(device_index: int) -> CudaDeviceIdentity:
    """Return the immutable UUID identity of one locally visible device."""

    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ValueError("CUDA device index must be a non-negative integer")
    driver = _driver()
    _check(driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    _check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if device_index >= count.value:
        raise CudaIpcError(f"CUDA device index {device_index} is outside {count.value} devices")
    device = ctypes.c_int()
    _check(driver.cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
    uuid = _CUuuid()
    _check(driver.cuDeviceGetUuid(ctypes.byref(uuid), device), "cuDeviceGetUuid")
    name = ctypes.create_string_buffer(_MAX_DEVICE_NAME_LENGTH)
    _check(driver.cuDeviceGetName(name, _MAX_DEVICE_NAME_LENGTH, device), "cuDeviceGetName")
    return CudaDeviceIdentity(
        index=device_index,
        uuid=bytes(uuid.bytes).hex(),
        name=name.value.decode("utf-8", errors="replace"),
    )


def find_device_by_uuid(uuid: str) -> int:
    """Resolve a UUID in the current process's visible-device namespace."""

    _validate_uuid(uuid)
    driver = _driver()
    _check(driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    _check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    matches = [index for index in range(count.value) if get_device_identity(index).uuid == uuid]
    if not matches:
        raise CudaIpcError(f"CUDA device UUID {uuid} is not visible in this process")
    if len(matches) != 1:
        raise CudaIpcError(f"CUDA device UUID {uuid} is unexpectedly ambiguous")
    return matches[0]


def resolve_device_index(device: int | str | Any) -> int:
    """Resolve an integer or Torch device spelling without a top-level Torch import."""

    if isinstance(device, bool):
        raise ValueError("boolean is not a CUDA device")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be non-negative")
        return device
    try:
        import torch
    except ImportError as error:
        raise CudaIpcError("a Torch device spelling requires Torch to be installed") from error
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be an integer, string, or torch.device")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {torch_device}")
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"Torch returned an invalid CUDA device index: {index!r}")
    return index


def cuda_driver_available() -> bool:
    """Return whether a CUDA driver and at least one device are available."""

    try:
        driver = _load_driver()
        if driver.cuInit(0) != 0:
            return False
        count = ctypes.c_int()
        return driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except CudaIpcError:
        return False


__all__ = [
    "CudaDeviceIdentity",
    "CudaIpcAllocation",
    "CudaIpcError",
    "CudaIpcEvent",
    "CudaIpcEventHandle",
    "CudaIpcImportedEvent",
    "CudaIpcImportedMemory",
    "CudaIpcMemHandle",
    "CudaIpcTransport",
    "cuda_driver_available",
    "find_device_by_uuid",
    "get_device_identity",
    "resolve_device_index",
]
