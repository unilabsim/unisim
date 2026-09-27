"""IsaacSim-owned CUDA IPC tensor arena and stream protocol.

This module is intentionally IsaacSim-specific.  It uses the SDK-free raw CUDA
IPC primitive for bulk storage/events, while control messages and lifecycle
negotiation remain on the existing worker pipe.  It does not alter the legacy
CPU shared-memory protocol.

Event ownership is directional: the host records ``control_event`` after its
D2D control write, records ``reset_event`` after a D2D selected-reset write, and
the worker records ``state_event`` after enqueueing native state projections.
Each host consumer stream waits on ``state_event``.
Worker pipe READY is a process-health acknowledgement, not a CUDA data barrier.
"""

from __future__ import annotations

import ctypes
import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from unisim.backend.subprocess_ipc.cuda_ipc import (
    CudaIpcEventHandle,
    CudaIpcImportedEvent,
    CudaIpcImportedMemory,
    CudaIpcMemHandle,
    CudaIpcTransport,
)

ISAACSIM_TENSOR_SCHEMA_VERSION = 2
ISAACSIM_CUDA_ATTACH = "TENSOR_CUDA_ATTACH"
ISAACSIM_CUDA_STEP = "TENSOR_CUDA_STEP"
ISAACSIM_CUDA_RESET = "TENSOR_CUDA_RESET"
ISAACSIM_CUDA_READY = "TENSOR_CUDA_READY"
_ALIGNMENT_BYTES = 256
_DLPACK_TENSOR_NAME = b"dltensor"
_DLCUDA = 2

_PyMem_RawMalloc = ctypes.pythonapi.PyMem_RawMalloc
_PyMem_RawMalloc.argtypes = [ctypes.c_size_t]
_PyMem_RawMalloc.restype = ctypes.c_void_p
_PyMem_RawFree = ctypes.pythonapi.PyMem_RawFree
_PyMem_RawFree.argtypes = [ctypes.c_void_p]
_PyMem_RawFree.restype = None
_Py_IncRef = ctypes.pythonapi.Py_IncRef
_Py_IncRef.argtypes = [ctypes.py_object]
_Py_IncRef.restype = None
_Py_DecRef = ctypes.pythonapi.Py_DecRef
_Py_DecRef.argtypes = [ctypes.py_object]
_Py_DecRef.restype = None
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_SetName.restype = ctypes.c_int


class RawArenaViewToken:
    """Release callback owned by a public DLPack tensor view."""

    def __init__(self, on_release: Optional[Callable[[], None]] = None) -> None:
        self._on_release = on_release
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        callback, self._on_release = self._on_release, None
        if callback is not None:
            callback()


_DLPackDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


@_DLPackDeleter
def _dlpack_tensor_deleter(managed_ptr: int) -> None:
    managed = _DLManagedTensor.from_address(managed_ptr)
    token = ctypes.cast(managed.manager_ctx, ctypes.py_object).value
    if token is not None:
        token.release()
        _Py_DecRef(token)
    _PyMem_RawFree(ctypes.c_void_p(managed_ptr))


@_DLPackDeleter
def _dlpack_capsule_deleter(capsule_ptr: int) -> None:
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object)
    if _PyCapsule_IsValid(capsule, _DLPACK_TENSOR_NAME):
        managed_ptr = _PyCapsule_GetPointer(capsule, _DLPACK_TENSOR_NAME)
        managed = _DLManagedTensor.from_address(managed_ptr)
        if managed.deleter:
            managed.deleter(managed_ptr)


def _align(value: int) -> int:
    return (int(value) + _ALIGNMENT_BYTES - 1) // _ALIGNMENT_BYTES * _ALIGNMENT_BYTES


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional host dependency
        raise RuntimeError(
            "IsaacSim CUDA IPC tensor lifecycle requires PyTorch with CUDA support"
        ) from exc
    if not getattr(torch, "cuda", None) or not bool(torch.cuda.is_available()):
        raise RuntimeError("IsaacSim CUDA IPC tensor lifecycle requires an available CUDA device")
    return torch


@dataclass(frozen=True)
class IsaacSimCudaArenaLayout:
    """Fixed, 256-byte-aligned public tensor layout."""

    num_envs: int
    nq: int
    nv: int
    nu: int
    qpos_offset: int
    qvel_offset: int
    ctrl_offset: int
    reset_env_indices_offset: int
    reset_qpos_offset: int
    reset_qvel_offset: int
    size_bytes: int

    @classmethod
    def create(cls, num_envs: int, nq: int, nv: int, nu: int) -> "IsaacSimCudaArenaLayout":
        values = (num_envs, nq, nv, nu)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("CUDA arena dimensions must be integers")
        if num_envs <= 0 or min(nq, nv, nu) < 0:
            raise ValueError(
                f"invalid CUDA arena dimensions: num_envs={num_envs}, nq={nq}, nv={nv}, nu={nu}"
            )
        qpos_offset = 0
        qvel_offset = qpos_offset + _align(num_envs * nq * np.dtype(np.float32).itemsize)
        ctrl_offset = qvel_offset + _align(num_envs * nv * np.dtype(np.float32).itemsize)
        reset_env_indices_offset = ctrl_offset + _align(
            num_envs * nu * np.dtype(np.float32).itemsize
        )
        reset_qpos_offset = reset_env_indices_offset + _align(
            num_envs * np.dtype(np.int64).itemsize
        )
        reset_qvel_offset = reset_qpos_offset + _align(
            num_envs * nq * np.dtype(np.float32).itemsize
        )
        size = _align(reset_qvel_offset + num_envs * nv * np.dtype(np.float32).itemsize)
        return cls(
            num_envs,
            nq,
            nv,
            nu,
            qpos_offset,
            qvel_offset,
            ctrl_offset,
            reset_env_indices_offset,
            reset_qpos_offset,
            reset_qvel_offset,
            size,
        )

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "qpos": (self.num_envs, self.nq),
            "qvel": (self.num_envs, self.nv),
            "ctrl": (self.num_envs, self.nu),
            "reset_env_indices": (self.num_envs,),
            "reset_qpos": (self.num_envs, self.nq),
            "reset_qvel": (self.num_envs, self.nv),
        }

    @property
    def dtypes(self) -> dict[str, str]:
        return {
            "qpos": "float32",
            "qvel": "float32",
            "ctrl": "float32",
            "reset_env_indices": "int64",
            "reset_qpos": "float32",
            "reset_qvel": "float32",
        }

    def as_dict(self) -> dict[str, int]:
        return {
            "num_envs": self.num_envs,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "qpos_offset": self.qpos_offset,
            "qvel_offset": self.qvel_offset,
            "ctrl_offset": self.ctrl_offset,
            "reset_env_indices_offset": self.reset_env_indices_offset,
            "reset_qpos_offset": self.reset_qpos_offset,
            "reset_qvel_offset": self.reset_qvel_offset,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IsaacSimCudaArenaLayout":
        if not isinstance(value, dict) or set(value) != {
            "num_envs",
            "nq",
            "nv",
            "nu",
            "qpos_offset",
            "qvel_offset",
            "ctrl_offset",
            "reset_env_indices_offset",
            "reset_qpos_offset",
            "reset_qvel_offset",
            "size_bytes",
        }:
            raise ValueError("malformed IsaacSim CUDA arena layout")
        layout = cls(**{key: value[key] for key in value})
        if layout != cls.create(layout.num_envs, layout.nq, layout.nv, layout.nu):
            raise ValueError("IsaacSim CUDA arena layout offsets are not canonical")
        return layout


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLPackDeleter),
]


class RawCudaTensorView:
    """A stable Torch view created through public DLPack, not private storage."""

    def __init__(
        self,
        *,
        torch: Any,
        pointer: int,
        shape: tuple[int, ...],
        device_index: int,
        dtype: str = "float32",
        token: Optional[RawArenaViewToken] = None,
    ) -> None:
        self.pointer = int(pointer)
        self.device_index = int(device_index)
        self.dtype = str(dtype)
        self.token = token if token is not None else RawArenaViewToken()
        if len(shape) not in (1, 2) or any(isinstance(value, bool) or value < 0 for value in shape):
            raise ValueError(f"IsaacSim CUDA view shape must be rank-1/2, got {shape}")
        dtype_codes = {"float32": (2, 32), "int32": (0, 32), "int64": (0, 64)}
        if self.dtype not in dtype_codes or self.pointer <= 0 or self.device_index < 0:
            raise ValueError("invalid IsaacSim CUDA raw view pointer, device, or dtype")
        self.type_code, self.type_bits = dtype_codes[self.dtype]
        self._shape_tuple = tuple(int(value) for value in shape)
        self._tensor = torch.from_dlpack(self)

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DLCUDA, self.device_index)

    def __dlpack__(self, stream: Any = None) -> Any:
        del stream  # Cross-process ordering is explicit through CUDA IPC events.
        ndim = len(self._shape_tuple)
        managed_size = ctypes.sizeof(_DLManagedTensor)
        shape_size = ndim * ctypes.sizeof(ctypes.c_int64)
        memory = _PyMem_RawMalloc(managed_size + 2 * shape_size)
        if not memory:
            raise MemoryError("cannot allocate an IsaacSim CUDA DLPack descriptor")
        try:
            managed_address = int(memory)
            shape_address = managed_address + managed_size
            strides_address = shape_address + shape_size
            shape = (ctypes.c_int64 * ndim)(*self._shape_tuple)
            strides = (
                (ctypes.c_int64 * ndim)(self._shape_tuple[1], 1)
                if ndim == 2
                else (ctypes.c_int64 * ndim)(
                    1,
                )
            )
            ctypes.memmove(shape_address, shape, shape_size)
            ctypes.memmove(strides_address, strides, shape_size)
            managed = _DLManagedTensor.from_address(managed_address)
            managed.dl_tensor.data = ctypes.c_void_p(self.pointer)
            managed.dl_tensor.device = _DLDevice(_DLCUDA, self.device_index)
            managed.dl_tensor.ndim = ndim
            managed.dl_tensor.dtype = _DLDataType(self.type_code, self.type_bits, 1)
            managed.dl_tensor.shape = ctypes.cast(shape_address, ctypes.POINTER(ctypes.c_int64))
            managed.dl_tensor.strides = ctypes.cast(strides_address, ctypes.POINTER(ctypes.c_int64))
            managed.dl_tensor.byte_offset = 0
            _Py_IncRef(self.token)
            managed.manager_ctx = id(self.token)
            managed.deleter = _dlpack_tensor_deleter
            return _PyCapsule_New(
                ctypes.byref(managed), _DLPACK_TENSOR_NAME, _dlpack_capsule_deleter
            )
        except BaseException:
            _PyMem_RawFree(ctypes.c_void_p(memory))
            raise

    @property
    def tensor(self) -> Any:
        return self._tensor


def require_cuda_tensor(tensor: Any, *, rank: int, name: str) -> None:
    if not hasattr(tensor, "is_cuda") or not bool(tensor.is_cuda):
        raise ValueError(f"{name} must be a CUDA tensor")
    if str(tensor.dtype).removeprefix("torch.") != "float32":
        raise ValueError(f"{name} must have float32 dtype, got {tensor.dtype!r}")
    if len(tuple(tensor.shape)) != rank or not bool(tensor.is_contiguous()):
        raise ValueError(f"{name} must be a contiguous rank-{rank} tensor")


def require_cuda_tensor_of_dtype(tensor: Any, *, rank: int, name: str, dtype: str) -> None:
    if not hasattr(tensor, "is_cuda") or not bool(tensor.is_cuda):
        raise ValueError(f"{name} must be a CUDA tensor")
    if len(tuple(tensor.shape)) != rank or not bool(tensor.is_contiguous()):
        raise ValueError(f"{name} must be a contiguous rank-{rank} tensor")
    if str(tensor.dtype).removeprefix("torch.") != dtype:
        raise ValueError(f"{name} must have {dtype} dtype, got {tensor.dtype!r}")


class HostCudaIpcArena:
    """Producer-owned canonical arena exported to the IsaacSim worker."""

    def __init__(
        self,
        *,
        num_envs: int,
        nq: int,
        nv: int,
        nu: int,
        device: Any = "cuda",
    ) -> None:
        torch = import_torch()
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"IsaacSim CUDA IPC requires a CUDA device, got {resolved}")
        index = resolved.index if resolved.index is not None else int(torch.cuda.current_device())
        self.layout = IsaacSimCudaArenaLayout.create(num_envs, nq, nv, nu)
        self._torch = torch
        self._device_index = index
        self.closed = False
        self._transport: CudaIpcTransport | None = None
        self._allocation: Any = None
        self._control_event: Any = None
        self._state_event: Any = None
        self._reset_event: Any = None
        self._active_view_names: dict[int, str] = {}
        self._view_serial = 0
        try:
            self._transport = CudaIpcTransport(index)
            self._allocation = self._transport.allocate(self.layout.size_bytes)
            self._control_event = self._transport.create_event()
            self._state_event = self._transport.create_event()
            self._reset_event = self._transport.create_event()
        except Exception:
            self.close()
            raise

    @property
    def qpos(self) -> Any:
        return self._view("qpos")

    @property
    def device_uuid(self) -> str:
        if self.closed or self._transport is None:
            raise RuntimeError("IsaacSim CUDA IPC arena is closed")
        return self._transport.identity.uuid

    @property
    def qvel(self) -> Any:
        return self._view("qvel")

    @property
    def device_index(self) -> int:
        if self.closed:
            raise RuntimeError("IsaacSim CUDA IPC arena is closed")
        return self._device_index

    @property
    def ctrl(self) -> Any:
        return self._view("ctrl")

    @property
    def reset_env_indices(self) -> Any:
        return self._view("reset_env_indices")

    @property
    def reset_qpos(self) -> Any:
        return self._view("reset_qpos")

    @property
    def reset_qvel(self) -> Any:
        return self._view("reset_qvel")

    def _view(self, name: str) -> Any:
        if self.closed or self._allocation is None:
            raise RuntimeError("IsaacSim CUDA IPC arena is closed")
        shape = self.layout.shapes[name]
        self._view_serial += 1
        token_id = self._view_serial

        def release_view() -> None:
            self._active_view_names.pop(token_id, None)

        token = RawArenaViewToken(release_view)
        self._active_view_names[token_id] = name
        try:
            view = RawCudaTensorView(
                torch=self._torch,
                pointer=self._allocation.pointer + getattr(self.layout, name + "_offset"),
                shape=shape,
                device_index=self._device_index,
                dtype=self.layout.dtypes[name],
                token=token,
            )
            tensor = view.tensor
            if not bool(tensor.is_cuda) or tuple(tensor.shape) != shape:
                raise RuntimeError("Torch did not import the raw CUDA DLPack view")
            return tensor
        except BaseException:
            token.release()
            raise

    def to_payload(self) -> dict[str, Any]:
        if self.closed or self._allocation is None:
            raise RuntimeError("cannot export a closed IsaacSim CUDA IPC arena")
        assert (
            self._transport is not None
            and self._control_event is not None
            and self._state_event is not None
            and self._reset_event is not None
        )
        return {
            "schema_version": ISAACSIM_TENSOR_SCHEMA_VERSION,
            "device_uuid": self._transport.identity.uuid,
            "device_index": self._device_index,
            "layout": self.layout.as_dict(),
            "memory": self._allocation.export_handle(),
            "control_event": self._control_event.export_handle(),
            "state_event": self._state_event.export_handle(),
            "reset_event": self._reset_event.export_handle(),
        }

    def write_control(self, ctrl: Any) -> None:
        require_cuda_tensor(ctrl, rank=2, name="control")
        expected = self.layout.shapes["ctrl"]
        if tuple(ctrl.shape) != expected:
            raise ValueError(f"control must have shape {expected}, got {tuple(ctrl.shape)}")
        self.ctrl.copy_(ctrl, non_blocking=True)

    def record_control(self) -> None:
        if self.closed or self._control_event is None:
            raise RuntimeError("IsaacSim CUDA IPC control event is closed")
        stream = self._torch.cuda.current_stream(self._device_index).cuda_stream
        self._control_event.record(int(stream))

    def write_reset(self, env_indices: Any, qpos: Any, qvel: Any) -> None:
        require_cuda_tensor_of_dtype(env_indices, rank=1, name="reset env_indices", dtype="int64")
        require_cuda_tensor(qpos, rank=2, name="reset qpos")
        require_cuda_tensor(qvel, rank=2, name="reset qvel")
        capacity = self.layout.num_envs
        count = int(env_indices.shape[0])
        expected_qpos = (count, self.layout.nq)
        expected_qvel = (count, self.layout.nv)
        if count > capacity:
            raise ValueError(f"reset env_indices may contain at most {capacity} rows, got {count}")
        if tuple(qpos.shape) != expected_qpos:
            raise ValueError(f"reset qpos must have shape {expected_qpos}, got {tuple(qpos.shape)}")
        if tuple(qvel.shape) != expected_qvel:
            raise ValueError(f"reset qvel must have shape {expected_qvel}, got {tuple(qvel.shape)}")
        if count == 0:
            return
        self.reset_env_indices[:count].copy_(env_indices, non_blocking=True)
        self.reset_qpos[:count].copy_(qpos, non_blocking=True)
        self.reset_qvel[:count].copy_(qvel, non_blocking=True)

    def record_reset(self) -> None:
        if self.closed or self._reset_event is None:
            raise RuntimeError("IsaacSim CUDA IPC reset event is closed")
        stream = self._torch.cuda.current_stream(self._device_index).cuda_stream
        self._reset_event.record(int(stream))

    def wait_state(self) -> None:
        if self.closed or self._state_event is None:
            raise RuntimeError("IsaacSim CUDA IPC state event is closed")
        stream = int(self._torch.cuda.current_stream(self._device_index).cuda_stream)
        self._state_event.wait_stream(stream)

    def close(self) -> None:
        if self.closed:
            return
        gc.collect()
        active = sorted(set(self._active_view_names.values()))
        if active:
            raise RuntimeError(
                "release IsaacSim CUDA state/control/reset views before closing arena: "
                + ", ".join(sorted(active))
            )
        if (
            self._allocation is not None
            or self._control_event is not None
            or self._state_event is not None
            or self._reset_event is not None
        ):
            self._torch.cuda.synchronize(self._device_index)
        self.closed = True
        self._active_view_names.clear()
        allocation, self._allocation = self._allocation, None
        control, self._control_event = self._control_event, None
        state, self._state_event = self._state_event, None
        reset, self._reset_event = self._reset_event, None
        transport, self._transport = self._transport, None
        if allocation is not None:
            allocation.close()
        if control is not None:
            control.close()
        if state is not None:
            state.close()
        if reset is not None:
            reset.close()
        if transport is not None:
            transport.close()


class WorkerCudaIpcArena:
    """Worker-side imported mapping of the canonical host arena."""

    def __init__(self, payload: dict[str, Any], *, device_index: int) -> None:
        torch = import_torch()
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "device_uuid",
            "device_index",
            "layout",
            "memory",
            "control_event",
            "state_event",
            "reset_event",
        }:
            raise ValueError("malformed IsaacSim CUDA IPC arena descriptor")
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != ISAACSIM_TENSOR_SCHEMA_VERSION
        ):
            raise ValueError("unsupported IsaacSim CUDA IPC schema version")
        layout = IsaacSimCudaArenaLayout.from_dict(payload["layout"])
        payload_device_index = payload["device_index"]
        device_uuid = payload["device_uuid"]
        if (
            isinstance(payload_device_index, bool)
            or not isinstance(payload_device_index, int)
            or payload_device_index < 0
            or isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
            or not isinstance(device_uuid, str)
        ):
            raise ValueError("malformed IsaacSim CUDA IPC device identity")
        memory_handle = _mem_handle(payload["memory"])
        control_handle = _event_handle(payload["control_event"])
        state_handle = _event_handle(payload["state_event"])
        reset_handle = _event_handle(payload["reset_event"])
        if (
            memory_handle.device_uuid != device_uuid
            or memory_handle.size_bytes != layout.size_bytes
            or memory_handle.alignment_bytes < _ALIGNMENT_BYTES
            or any(
                offset % _ALIGNMENT_BYTES != 0
                for offset in (
                    layout.qpos_offset,
                    layout.qvel_offset,
                    layout.ctrl_offset,
                    layout.reset_env_indices_offset,
                    layout.reset_qpos_offset,
                    layout.reset_qvel_offset,
                )
            )
            or control_handle.device_uuid != device_uuid
            or state_handle.device_uuid != device_uuid
            or reset_handle.device_uuid != device_uuid
        ):
            raise ValueError("IsaacSim CUDA IPC handles do not match the canonical arena")
        self.layout = layout
        self._torch = torch
        self._device_index = device_index
        self.closed = False
        self._transport: CudaIpcTransport | None = None
        self._memory: CudaIpcImportedMemory | None = None
        self._control_event: CudaIpcImportedEvent | None = None
        self._state_event: CudaIpcImportedEvent | None = None
        self._reset_event: CudaIpcImportedEvent | None = None
        self._active_view_names: dict[int, str] = {}
        self._view_serial = 0
        try:
            transport = CudaIpcTransport(device_index)
            if transport.identity.uuid != payload["device_uuid"]:
                raise RuntimeError(
                    "IsaacSim CUDA IPC UUID mismatch: host device "
                    f"{payload['device_uuid']}, worker device {transport.identity.uuid}"
                )
            self._transport = transport
            self._memory = transport.import_handle(memory_handle)
            self._control_event = transport.import_event_handle(control_handle)
            self._state_event = transport.import_event_handle(state_handle)
            self._reset_event = transport.import_event_handle(reset_handle)
        except Exception:
            self.close()
            raise

    @property
    def qpos(self) -> Any:
        return self._view("qpos")

    @property
    def qvel(self) -> Any:
        return self._view("qvel")

    @property
    def ctrl(self) -> Any:
        return self._view("ctrl")

    @property
    def reset_env_indices(self) -> Any:
        return self._view("reset_env_indices")

    @property
    def reset_qpos(self) -> Any:
        return self._view("reset_qpos")

    @property
    def reset_qvel(self) -> Any:
        return self._view("reset_qvel")

    def _view(self, name: str) -> Any:
        if self.closed or self._memory is None:
            raise RuntimeError("IsaacSim CUDA IPC arena is closed")
        assert self._transport is not None
        shape = self.layout.shapes[name]
        self._view_serial += 1
        token_id = self._view_serial

        def release_view() -> None:
            self._active_view_names.pop(token_id, None)

        token = RawArenaViewToken(release_view)
        self._active_view_names[token_id] = name
        try:
            view = RawCudaTensorView(
                torch=self._torch,
                pointer=self._memory.pointer + getattr(self.layout, name + "_offset"),
                shape=shape,
                device_index=self._transport.device_index,
                dtype=self.layout.dtypes[name],
                token=token,
            )
            tensor = view.tensor
            if not bool(tensor.is_cuda) or tuple(tensor.shape) != shape:
                raise RuntimeError("Torch did not import the raw CUDA DLPack view")
            return tensor
        except BaseException:
            token.release()
            raise

    @property
    def device_uuid(self) -> str:
        if self.closed or self._transport is None:
            raise RuntimeError("IsaacSim CUDA IPC arena is closed")
        return self._transport.identity.uuid

    def wait_control(self) -> None:
        if self.closed or self._control_event is None:
            raise RuntimeError("IsaacSim CUDA IPC control event is closed")
        stream = int(self._torch.cuda.current_stream().cuda_stream)
        self._control_event.wait_stream(stream)

    def record_state(self) -> None:
        if self.closed or self._state_event is None:
            raise RuntimeError("IsaacSim CUDA IPC state event is closed")
        stream = int(self._torch.cuda.current_stream().cuda_stream)
        self._state_event.record(stream)

    def wait_reset(self) -> None:
        if self.closed or self._reset_event is None:
            raise RuntimeError("IsaacSim CUDA IPC reset event is closed")
        stream = int(self._torch.cuda.current_stream().cuda_stream)
        self._reset_event.wait_stream(stream)

    def close(self) -> None:
        if self.closed:
            return
        gc.collect()
        active = sorted(set(self._active_view_names.values()))
        if active:
            raise RuntimeError(
                "release IsaacSim CUDA state/control/reset views before closing arena: "
                + ", ".join(sorted(active))
            )
        if (
            self._memory is not None
            or self._control_event is not None
            or self._state_event is not None
            or self._reset_event is not None
        ):
            self._torch.cuda.synchronize(self._device_index)
        self.closed = True
        self._active_view_names.clear()
        memory, self._memory = self._memory, None
        control, self._control_event = self._control_event, None
        state, self._state_event = self._state_event, None
        reset, self._reset_event = self._reset_event, None
        transport, self._transport = self._transport, None
        if memory is not None:
            memory.close()
        if control is not None:
            control.close()
        if state is not None:
            state.close()
        if reset is not None:
            reset.close()
        if transport is not None:
            transport.close()


def _mem_handle(value: Any) -> CudaIpcMemHandle:
    if not isinstance(value, CudaIpcMemHandle):
        raise ValueError("IsaacSim CUDA memory handle has the wrong type")
    return value


def _event_handle(value: Any) -> CudaIpcEventHandle:
    if not isinstance(value, CudaIpcEventHandle):
        raise ValueError("IsaacSim CUDA event handle has the wrong type")
    return value
