"""SDK-free CUDA IPC arena helpers for the IsaacGym tensor lifecycle.

This module is intentionally Python 3.8 compatible and importable by file path
in the isolated IsaacGym worker.  It must not import ``unisim``, IsaacGym, or
any other optional runtime at module import time.
"""

from __future__ import annotations

import ctypes
import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

_CUDA_IPC_ARENA_VERSION = 2
_ARENA_ALIGNMENT_BYTES = 256
_FLOAT32_BYTES = 4
_INT64_BYTES = 8
_DLPACK_TENSOR_NAME = b"dltensor"


@dataclass(frozen=True)
class IsaacGymCudaIpcArenaLayout:
    """Fixed offsets for canonical arrays and the selected-reset prefix."""

    num_envs: int
    nq: int
    nv: int
    nu: int
    reset_indices_offset: int
    reset_qpos_offset: int
    reset_qvel_offset: int
    qpos_offset: int
    qvel_offset: int
    ctrl_offset: int
    size_bytes: int

    @classmethod
    def create(cls, num_envs: int, nq: int, nv: int, nu: int) -> "IsaacGymCudaIpcArenaLayout":
        values = (num_envs, nq, nv, nu)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("CUDA IPC arena dimensions must be nonnegative integers")
        if num_envs <= 0:
            raise ValueError("CUDA IPC arena requires num_envs > 0")
        reset_indices_offset = 0
        reset_qpos_offset = _align(reset_indices_offset + num_envs * _INT64_BYTES)
        reset_qvel_offset = _align(reset_qpos_offset + num_envs * nq * _FLOAT32_BYTES)
        qpos_offset = _align(reset_qvel_offset + num_envs * nv * _FLOAT32_BYTES)
        qvel_offset = _align(qpos_offset + num_envs * nq * _FLOAT32_BYTES)
        ctrl_offset = _align(qvel_offset + num_envs * nv * _FLOAT32_BYTES)
        size_bytes = _align(ctrl_offset + num_envs * nu * _FLOAT32_BYTES)
        return cls(
            num_envs,
            nq,
            nv,
            nu,
            reset_indices_offset,
            reset_qpos_offset,
            reset_qvel_offset,
            qpos_offset,
            qvel_offset,
            ctrl_offset,
            size_bytes,
        )

    @property
    def qpos_shape(self) -> Tuple[int, int]:
        return (self.num_envs, self.nq)

    @property
    def qvel_shape(self) -> Tuple[int, int]:
        return (self.num_envs, self.nv)

    @property
    def ctrl_shape(self) -> Tuple[int, int]:
        return (self.num_envs, self.nu)

    @property
    def reset_indices_shape(self) -> Tuple[int]:
        return (self.num_envs,)

    @property
    def reset_qpos_shape(self) -> Tuple[int, int]:
        return (self.num_envs, self.nq)

    @property
    def reset_qvel_shape(self) -> Tuple[int, int]:
        return (self.num_envs, self.nv)

    def wire(self) -> Dict[str, Any]:
        return {
            "version": _CUDA_IPC_ARENA_VERSION,
            "num_envs": self.num_envs,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "reset_indices_offset": self.reset_indices_offset,
            "reset_qpos_offset": self.reset_qpos_offset,
            "reset_qvel_offset": self.reset_qvel_offset,
            "qpos_offset": self.qpos_offset,
            "qvel_offset": self.qvel_offset,
            "ctrl_offset": self.ctrl_offset,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_wire(cls, payload: Dict[str, Any]) -> "IsaacGymCudaIpcArenaLayout":
        required = {
            "num_envs",
            "nq",
            "nv",
            "nu",
            "reset_indices_offset",
            "reset_qpos_offset",
            "reset_qvel_offset",
            "qpos_offset",
            "qvel_offset",
            "ctrl_offset",
            "size_bytes",
        }
        if not isinstance(payload, dict) or set(payload) != required | {"version"}:
            raise ValueError("malformed IsaacGym CUDA IPC arena descriptor")
        if payload["version"] != _CUDA_IPC_ARENA_VERSION:
            raise ValueError("unsupported IsaacGym CUDA IPC arena version")
        layout = cls(
            payload["num_envs"],
            payload["nq"],
            payload["nv"],
            payload["nu"],
            payload["reset_indices_offset"],
            payload["reset_qpos_offset"],
            payload["reset_qvel_offset"],
            payload["qpos_offset"],
            payload["qvel_offset"],
            payload["ctrl_offset"],
            payload["size_bytes"],
        )
        if layout.wire() != payload or layout != cls.create(
            layout.num_envs, layout.nq, layout.nv, layout.nu
        ):
            raise ValueError("inconsistent IsaacGym CUDA IPC arena descriptor")
        return layout


def _align(value: int) -> int:
    return (value + _ARENA_ALIGNMENT_BYTES - 1) // _ARENA_ALIGNMENT_BYTES * _ARENA_ALIGNMENT_BYTES


# --------------------------------------------------------------------------- #
# Stable Torch views over raw CUDA pointers
# --------------------------------------------------------------------------- #


class _DLDataTypeCode(ctypes.c_uint8):
    INT = 0
    FLOAT = 2


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", _DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DLPackDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLPackDeleter),
    ]


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
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _DLPackDeleter]
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
    """One release callback associated with a stable raw-pointer view."""

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


class RawCudaArenaView:
    """A one-shot DLPack producer for a raw CUDA arena allocation."""

    def __init__(
        self,
        pointer: int,
        shape: Sequence[int],
        device_index: int,
        dtype: str = "float32",
        token: Optional[RawArenaViewToken] = None,
    ) -> None:
        self.pointer = int(pointer)
        self.shape = tuple(int(value) for value in shape)
        self.device_index = int(device_index)
        self.dtype = str(dtype)
        if self.dtype not in ("float32", "int64"):
            raise ValueError("raw CUDA arena views support only float32 and int64")
        self.type_code = _DLDataTypeCode.FLOAT if self.dtype == "float32" else _DLDataTypeCode.INT
        self.type_bits = 32 if self.dtype == "float32" else 64
        self.token = token if token is not None else RawArenaViewToken()
        if self.pointer <= 0 or not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("raw CUDA arena views require a positive pointer and shape")

    def __dlpack_device__(self) -> Tuple[int, int]:
        return (_DLCUDA, self.device_index)

    def __dlpack__(self, stream: Any = None) -> Any:
        del stream  # Cross-process ordering is explicit through IPC events.
        ndim = len(self.shape)
        managed_size = ctypes.sizeof(_DLManagedTensor)
        shape_size = ndim * ctypes.sizeof(ctypes.c_int64)
        memory = _PyMem_RawMalloc(managed_size + 2 * shape_size)
        if not memory:
            raise MemoryError("cannot allocate a DLPack descriptor")
        try:
            managed_address = int(memory)
            shape_address = managed_address + managed_size
            strides_address = shape_address + shape_size
            shape = (ctypes.c_int64 * ndim)(*self.shape)
            strides = (ctypes.c_int64 * ndim)()
            running = 1
            for index in range(ndim - 1, -1, -1):
                strides[index] = running
                running *= self.shape[index]
            ctypes.memmove(shape_address, shape, shape_size)
            ctypes.memmove(strides_address, strides, shape_size)
            managed = _DLManagedTensor.from_address(managed_address)
            managed.dl_tensor.data = ctypes.c_void_p(self.pointer)
            managed.dl_tensor.device.device_type = _DLCUDA
            managed.dl_tensor.device.device_id = self.device_index
            managed.dl_tensor.ndim = ndim
            managed.dl_tensor.dtype.type_code = self.type_code
            managed.dl_tensor.dtype.bits = self.type_bits
            managed.dl_tensor.dtype.lanes = 1
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


def torch_from_cuda_pointer(
    torch_module: Any,
    pointer: int,
    shape: Sequence[int],
    device_index: int,
    dtype: str = "float32",
    token: Optional[RawArenaViewToken] = None,
) -> Any:
    """Return a Torch tensor view without Torch private storage IPC."""
    return torch_module.from_dlpack(
        RawCudaArenaView(pointer, shape, device_index, dtype=dtype, token=token)
    )


# --------------------------------------------------------------------------- #
# Worker-side native tensor projection
# --------------------------------------------------------------------------- #


def _quat_rotate(torch: Any, quat_wxyz: Any, vectors: Any) -> Any:
    scalar = quat_wxyz[..., 0:1]
    axis = quat_wxyz[..., 1:]
    first = torch.cross(axis, vectors, dim=-1) * 2.0
    return vectors + scalar * first + torch.cross(axis, first, dim=-1)


def _quat_rotate_inverse(torch: Any, quat_wxyz: Any, vectors: Any) -> Any:
    conjugate = quat_wxyz * torch.tensor(
        [1.0, -1.0, -1.0, -1.0], dtype=quat_wxyz.dtype, device=quat_wxyz.device
    )
    return _quat_rotate(torch, conjugate, vectors)


def _device_tensor(torch: Any, values: Any, dtype: str, device: Any) -> Any:
    array = np.ascontiguousarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError("CUDA IPC projection indices and offsets must be numeric")
    return torch.from_numpy(array).to(device=device, dtype=getattr(torch, dtype))


class IsaacGymCudaIpcWorkerRuntime:
    """Project native IsaacGym tensors into the host-owned CUDA IPC arena."""

    def __init__(
        self,
        context: Any,
        cuda_ipc: Any,
        transport: Any,
        memory: Any,
        control_event: Any,
        state_event: Any,
        reset_event: Any,
        arena: IsaacGymCudaIpcArenaLayout,
    ) -> None:
        self.ctx = context
        self.cuda_ipc = cuda_ipc
        self.transport = transport
        self.memory = memory
        self.control_event = control_event
        self.state_event = state_event
        self.reset_event = reset_event
        self.arena = arena
        self.closed = False
        self.expected_reset_sequence = 0
        torch = context.torch
        if context.device == "cpu" or not context.use_gpu_pipeline:
            raise RuntimeError("IsaacGym CUDA IPC requires the GPU pipeline")
        try:
            ordinal = torch.device(context.device).index
        except (RuntimeError, ValueError):
            ordinal = None
        if ordinal is None:
            ordinal = int(torch.cuda.current_device())
        if ordinal != transport.device_index:
            raise RuntimeError("IsaacGym CUDA IPC worker device differs from its CUDA transport")
        if transport.identity.uuid != memory.handle.device_uuid:
            raise RuntimeError("IsaacGym CUDA IPC memory UUID handshake failed")

        self.device_index = ordinal
        self.qpos: Any = torch_from_cuda_pointer(
            torch, memory.pointer + arena.qpos_offset, arena.qpos_shape, ordinal
        )
        self.qvel: Any = torch_from_cuda_pointer(
            torch, memory.pointer + arena.qvel_offset, arena.qvel_shape, ordinal
        )
        self.ctrl: Any = torch_from_cuda_pointer(
            torch, memory.pointer + arena.ctrl_offset, arena.ctrl_shape, ordinal
        )
        self.reset_indices: Any = torch_from_cuda_pointer(
            torch,
            memory.pointer + arena.reset_indices_offset,
            arena.reset_indices_shape,
            ordinal,
            dtype="int64",
        )
        self.reset_qpos: Any = torch_from_cuda_pointer(
            torch,
            memory.pointer + arena.reset_qpos_offset,
            arena.reset_qpos_shape,
            ordinal,
        )
        self.reset_qvel: Any = torch_from_cuda_pointer(
            torch,
            memory.pointer + arena.reset_qvel_offset,
            arena.reset_qvel_shape,
            ordinal,
        )
        try:
            self._bind_projection()
            # Materialization may leave native indexed writes pending.  Submit them
            # once on this cold attach so the first hot step has no host metadata.
            scene = context.scene_worker
            submit_pending = getattr(scene, "_submit_pending", None)
            if submit_pending is not None:
                submit_pending()
            self._refresh_native()
            self.publish_state(record_event=True)
        except BaseException:
            self._release_views()
            raise

    def _bind_projection(self) -> None:
        torch = self.ctx.torch
        scene = self.ctx.scene_worker
        layout = scene.layout
        if (
            scene.num_envs != self.arena.num_envs
            or layout.nq != self.arena.nq
            or layout.nv != self.arena.nv
            or layout.nu != self.arena.nu
        ):
            raise RuntimeError("IsaacGym native layout differs from the CUDA IPC arena")

        self.actor_ids: Any = _device_tensor(
            torch, np.asarray(scene.actor_ids, dtype=np.int64), "long", self.ctx.device
        )
        self.actor_ids_int32 = self.actor_ids.to(torch.int32).contiguous()
        self.control_dofs: Any = _device_tensor(
            torch, np.asarray(scene.control_dofs, dtype=np.int64), "long", self.ctx.device
        )
        self.root_com = _device_tensor(
            torch, np.asarray(scene.root_com, dtype=np.float32), "float32", self.ctx.device
        )
        self.root_projections = []
        self.joint_projections = []
        for entity_index, entity in enumerate(layout.entities):
            root = {
                "mode": entity.root_mode,
                "qpos": _device_tensor(torch, entity.root_qpos_indices, "long", self.ctx.device),
                "qvel": _device_tensor(torch, entity.root_qvel_indices, "long", self.ctx.device),
                "com": self.root_com[:, entity_index],
                "actor_ids": self.actor_ids_int32[:, entity_index],
            }
            self.root_projections.append(root)
            if not entity.joints:
                continue
            dof_ids = np.asarray(
                [record[entity_index]["dof_ids"] for record in scene.records],
                dtype=np.int64,
            )
            self.joint_projections.append(
                {
                    "dof_ids": _device_tensor(torch, dof_ids, "long", self.ctx.device),
                    "actor_ids": self.actor_ids_int32[:, entity_index],
                    "qpos": _device_tensor(
                        torch,
                        [joint.qpos_indices[0] for joint in entity.joints],
                        "long",
                        self.ctx.device,
                    ),
                    "qvel": _device_tensor(
                        torch,
                        [joint.qvel_indices[0] for joint in entity.joints],
                        "long",
                        self.ctx.device,
                    ),
                }
            )

    def _refresh_native(self) -> None:
        self.ctx._refresh_tensors()

    def _release_views(self) -> None:
        self.qpos = None
        self.qvel = None
        self.ctrl = None
        self.reset_indices = None
        self.reset_qpos = None
        self.reset_qvel = None
        self.actor_ids = None
        self.actor_ids_int32 = None
        self.control_dofs = None
        self.root_com = None
        self.root_projections = []
        self.joint_projections = []
        gc.collect()

    def _public_roots(self, native_roots: Any) -> Any:
        torch = self.ctx.torch
        roots = native_roots.index_select(0, self.actor_ids.reshape(-1))
        roots = roots.reshape(self.arena.num_envs, -1, 13).clone()
        roots[..., 3:7] = roots[..., [6, 3, 4, 5]]
        quat = roots[..., 3:7]
        com_world = _quat_rotate(torch, quat, self.root_com)
        roots[..., 7:10] -= torch.cross(roots[..., 10:13], com_world, dim=-1)
        return roots

    def publish_state(self, *, record_event: bool = True) -> None:
        if self.closed:
            raise RuntimeError("IsaacGym CUDA IPC runtime is closed")
        torch = self.ctx.torch
        scene = self.ctx.scene_worker
        if scene.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        self._refresh_native()
        roots = self._public_roots(self.ctx._root_state)
        self.qpos.zero_()
        self.qvel.zero_()
        for index, projection in enumerate(self.root_projections):
            if projection["mode"] != "floating":
                continue
            entity_root = roots[:, index]
            if projection["qpos"].numel():
                self.qpos.index_copy_(1, projection["qpos"], entity_root[:, :7].contiguous())
            if projection["qvel"].numel():
                angular_body = _quat_rotate_inverse(
                    torch, entity_root[:, 3:7], entity_root[:, 10:13]
                )
                root_velocity = torch.cat((entity_root[:, 7:10], angular_body), dim=-1)
                self.qvel.index_copy_(1, projection["qvel"], root_velocity.contiguous())
        for projection in self.joint_projections:
            dof_ids = projection["dof_ids"]
            flat_dof_ids = dof_ids.reshape(-1)
            joint_count = int(dof_ids.shape[-1])
            dof_positions = self.ctx._dof_state[:, 0].index_select(0, flat_dof_ids)
            dof_velocities = self.ctx._dof_state[:, 1].index_select(0, flat_dof_ids)
            self.qpos.index_copy_(
                1,
                projection["qpos"],
                dof_positions.reshape(self.arena.num_envs, joint_count).contiguous(),
            )
            self.qvel.index_copy_(
                1,
                projection["qvel"],
                dof_velocities.reshape(self.arena.num_envs, joint_count).contiguous(),
            )
        if record_event:
            with torch.cuda.device(self.device_index):
                stream = torch.cuda.current_stream(self.device_index).cuda_stream
                self.state_event.record(stream)

    def _write_native_root(self, actor_ids: Any, values: Any) -> None:
        native_ids = actor_ids.contiguous()
        self.ctx._root_state.index_copy_(0, native_ids.long(), values.contiguous())
        if not self.ctx.gym.set_actor_root_state_tensor_indexed(
            self.ctx.sim,
            self.ctx.gymtorch.unwrap_tensor(self.ctx._root_state),
            self.ctx.gymtorch.unwrap_tensor(native_ids),
            int(native_ids.shape[0]),
        ):
            raise RuntimeError("native selected root-state setter failed")

    def _write_native_dofs(
        self, actor_ids: Any, dof_ids: Any, positions: Any, velocities: Any
    ) -> None:
        flat_dof_ids = dof_ids.reshape(-1).contiguous()
        self.ctx._dof_state[flat_dof_ids, 0] = positions.reshape(-1).contiguous()
        self.ctx._dof_state[flat_dof_ids, 1] = velocities.reshape(-1).contiguous()
        native_ids = actor_ids.contiguous()
        if not self.ctx.gym.set_dof_state_tensor_indexed(
            self.ctx.sim,
            self.ctx.gymtorch.unwrap_tensor(self.ctx._dof_state),
            self.ctx.gymtorch.unwrap_tensor(native_ids),
            int(native_ids.shape[0]),
        ):
            raise RuntimeError("native selected DOF-state setter failed")

    def set_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.closed:
            raise RuntimeError("IsaacGym CUDA IPC runtime is closed")
        count = payload.get("count")
        sequence = payload.get("sequence")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > self.arena.num_envs
        ):
            raise ValueError("CUDA IPC selected-reset count is invalid")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != self.expected_reset_sequence + 1
        ):
            raise ValueError("CUDA IPC selected-reset sequence is invalid")
        if count == 0:
            # Empty resets never cross the device or process boundary in the parent.
            # A stale worker request is still rejected so event reuse remains explicit.
            self.expected_reset_sequence = sequence
            return {"timing": {}}

        torch = self.ctx.torch
        scene = self.ctx.scene_worker
        if scene.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            self.reset_event.wait_stream(stream)

        timing: Dict[str, float] = {}
        started = time.perf_counter()
        rows = self.reset_indices[:count]
        qpos = self.reset_qpos[:count]
        qvel = self.reset_qvel[:count]
        for projection in self.root_projections:
            if projection["mode"] != "floating":
                continue
            native_root = torch.zeros((count, 13), dtype=torch.float32, device=self.ctx.device)
            root_qpos: Any = qpos
            if projection["qpos"].numel():
                root_qpos = qpos.index_select(1, projection["qpos"])
                native_root[:, 0:3] = root_qpos[:, 0:3]
                native_root[:, 3:7] = root_qpos[:, [4, 5, 6, 3]]
            if projection["qvel"].numel():
                root_qvel = qvel.index_select(1, projection["qvel"])
                native_root[:, 7:10] = root_qvel[:, 0:3]
                quat_wxyz = root_qpos[:, 3:7]
                native_root[:, 10:13] = _quat_rotate(torch, quat_wxyz, root_qvel[:, 3:6])
                com = projection["com"][rows]
                com_world = _quat_rotate(torch, quat_wxyz, com)
                native_root[:, 7:10] += torch.cross(native_root[:, 10:13], com_world, dim=-1)
            self._write_native_root(projection["actor_ids"].index_select(0, rows), native_root)
        for projection in self.joint_projections:
            selected_dofs = projection["dof_ids"].index_select(0, rows)
            self._write_native_dofs(
                projection["actor_ids"].index_select(0, rows),
                selected_dofs,
                qpos[:, projection["qpos"]].contiguous(),
                qvel[:, projection["qvel"]].contiguous(),
            )

        timing["reset_apply_ms"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        self.expected_reset_sequence = sequence
        self._refresh_native()
        self.publish_state(record_event=True)
        timing["state_publish_ms"] = (time.perf_counter() - started) * 1000.0
        return {"timing": timing}

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.closed:
            raise RuntimeError("IsaacGym CUDA IPC runtime is closed")
        nsteps = payload.get("nsteps")
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise ValueError("CUDA IPC nsteps must be a positive integer")
        if payload.get("body_wrench") is not None:
            raise NotImplementedError("CUDA IPC body-wrench arena is not implemented")
        torch = self.ctx.torch
        scene = self.ctx.scene_worker
        if scene.faulted:
            raise RuntimeError("IsaacGym scene is faulted")
        stream = torch.cuda.current_stream().cuda_stream
        self.control_event.wait_stream(stream)
        timing: Dict[str, float] = {}
        started = time.perf_counter()
        if self.arena.nu:
            targets = self.ctx.scene_worker.targets
            targets.index_copy_(0, self.control_dofs.reshape(-1), self.ctrl.reshape(-1))
            if not self.ctx.gym.set_dof_position_target_tensor(
                self.ctx.sim, self.ctx.gymtorch.unwrap_tensor(targets)
            ):
                raise RuntimeError("native DOF position target setter failed")
        submit_pending = getattr(scene, "_submit_pending", None)
        if submit_pending is not None:
            submit_pending()
        timing["control_upload_ms"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        for _ in range(nsteps):
            self.ctx.gym.simulate(self.ctx.sim)
            self.ctx.gym.fetch_results(self.ctx.sim, True)
        scene.pending_roots.clear()
        scene.pending_dofs.clear()
        scene.pending_dof_actors.clear()
        timing["physics_ms"] = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        self.publish_state(record_event=True)
        timing["state_publish_ms"] = (time.perf_counter() - started) * 1000.0
        return {"timing": timing}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # Detach is a cold lifecycle boundary: wait for queued projection work
        # before dropping the last worker views and closing the IPC mapping.
        self.state_event.synchronize()
        self._release_views()
        self.memory.close()
        self.control_event.close()
        self.state_event.close()
        self.reset_event.close()
        self.transport.close()


__all__ = [
    "IsaacGymCudaIpcArenaLayout",
    "IsaacGymCudaIpcWorkerRuntime",
    "RawArenaViewToken",
    "RawCudaArenaView",
    "torch_from_cuda_pointer",
]


# --------------------------------------------------------------------------- #
# Host-owned transfer plan
# --------------------------------------------------------------------------- #

CMD_CUDA_IPC_ATTACH = "ISAACGYM_CUDA_IPC_ATTACH"
CMD_CUDA_IPC_STEP = "ISAACGYM_CUDA_IPC_STEP"
CMD_CUDA_IPC_SET_STATE = "ISAACGYM_CUDA_IPC_SET_STATE"
CMD_CUDA_IPC_DETACH = "ISAACGYM_CUDA_IPC_DETACH"


class IsaacGymCudaIpcPlan:
    """Control-only pipe plus stable device arenas for IsaacGym hot tensors."""

    def __init__(self, backend: Any, device: Any = None) -> None:
        self.backend = backend
        self.closed = False
        self._active_views: Dict[int, str] = {}
        self._view_serial = 0
        self._control: Any = None
        self._qpos: Any = None
        self._qvel: Any = None
        self._reset_indices: Any = None
        self._reset_qpos: Any = None
        self._reset_qvel: Any = None
        self._reset_sequence = 0
        self.last_timing: Dict[str, Dict[str, float]] = {}

        fixed_variant_plan = getattr(backend, "_fixed_variant_plan", None)
        entity_scene = getattr(backend, "_entity_scene", None)
        entity_variant_plan = getattr(getattr(entity_scene, "owner", None), "variant_plan", None)
        if fixed_variant_plan is not None or entity_variant_plan is not None:
            raise NotImplementedError("CUDA IPC fixed variants are not implemented")
        if getattr(backend, "_pre_step_control_fn", None) is not None:
            raise NotImplementedError("CUDA IPC host pre-step callbacks are not implemented")
        backend._require_state("IsaacGym CUDA IPC compilation")
        if not bool(getattr(backend._model_info, "use_gpu_pipeline", False)):
            raise RuntimeError("IsaacGym CUDA IPC requires the worker GPU pipeline")
        if getattr(backend, "_body_wrench_pending", False):
            raise NotImplementedError("CUDA IPC body-wrench arena is not implemented")

        import torch

        from unisim.backend.subprocess_ipc import cuda_ipc, protocol

        self._torch = torch
        self._cuda_ipc = cuda_ipc
        self._protocol = protocol
        requested = (
            torch.device(device) if device is not None else torch.device("cuda", backend._device_id)
        )
        if backend._device_id < 0:
            raise RuntimeError("IsaacGym CUDA IPC requires a GPU-bound worker")
        if requested.type != "cuda":
            raise ValueError(f"IsaacGym CUDA IPC tensors must live on cuda, got {requested}")
        index = backend._device_id if requested.index is None else requested.index
        if index != backend._device_id:
            raise ValueError(
                f"IsaacGym worker is bound to cuda:{backend._device_id}, got {requested}"
            )
        if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
            raise ValueError(f"CUDA device cuda:{index} is not available to Torch")
        self.device = torch.device("cuda", index)
        self.device_index = index

        if backend._entity_scene is not None:
            layout = backend._entity_scene.layout
            nq, nv, nu = int(layout.nq), int(layout.nv), int(layout.nu)
        else:
            info = backend._model_info
            nq = 7 + int(info.num_dof)
            nv = 6 + int(info.num_dof)
            nu = int(info.num_dof)
        self.arena = IsaacGymCudaIpcArenaLayout.create(backend.num_envs, nq, nv, nu)

        self.transport = cuda_ipc.CudaIpcTransport(index)
        try:
            if not self.transport.event_ipc_supported():
                raise RuntimeError(f"CUDA device {index} does not support IPC events")
            self.memory = self.transport.allocate(self.arena.size_bytes)
            try:
                self.control_event = self.transport.create_event()
                self.state_event = self.transport.create_event()
                self.reset_event = self.transport.create_event()
                self._make_views()
                payload = {
                    "arena": self.arena.wire(),
                    "memory": self._memory_wire(),
                    "control_event": self._event_wire(self.control_event),
                    "state_event": self._event_wire(self.state_event),
                    "reset_event": self._event_wire(self.reset_event),
                }
                try:
                    response = backend._request(
                        CMD_CUDA_IPC_ATTACH,
                        payload,
                        expect=protocol.CMD_READY,
                    )
                except BaseException:
                    self._detach_worker_best_effort(backend)
                    raise
                if not isinstance(response, dict):
                    self._detach_worker_best_effort(backend)
                    raise RuntimeError("IsaacGym worker returned a malformed CUDA IPC handshake")
                worker_uuid = response.get("device_uuid")
                if worker_uuid != self.transport.identity.uuid:
                    self._detach_worker_best_effort(backend)
                    raise RuntimeError(
                        "IsaacGym CUDA IPC same-GPU handshake failed: "
                        f"worker={worker_uuid!r}, collector={self.transport.identity.uuid!r}"
                    )
                if response.get("arena") != self.arena.wire():
                    self._detach_worker_best_effort(backend)
                    raise RuntimeError("IsaacGym worker changed the CUDA IPC arena layout")
            except BaseException:
                self._drop_views()
                try:
                    if hasattr(self, "state_event"):
                        self.state_event.close()
                    if hasattr(self, "control_event"):
                        self.control_event.close()
                    if hasattr(self, "reset_event"):
                        self.reset_event.close()
                finally:
                    self.memory.close()
                raise
        except BaseException:
            self.transport.close()
            raise

    def _detach_worker_best_effort(self, backend: Any) -> None:
        try:
            backend._request(
                CMD_CUDA_IPC_DETACH,
                None,
                expect=self._protocol.CMD_READY,
            )
        except BaseException:
            pass

    def _memory_wire(self) -> Dict[str, Any]:
        handle = self.memory.export_handle()
        return {
            "opaque_handle": handle.opaque_handle,
            "device_uuid": handle.device_uuid,
            "size_bytes": handle.size_bytes,
            "abi_version": handle.abi_version,
            "alignment_bytes": handle.alignment_bytes,
        }

    @staticmethod
    def _event_wire(event: Any) -> Dict[str, Any]:
        handle = event.export_handle()
        return {
            "opaque_handle": handle.opaque_handle,
            "device_uuid": handle.device_uuid,
            "abi_version": handle.abi_version,
            "blocking_sync": handle.blocking_sync,
        }

    def _make_view(
        self,
        *,
        name: str,
        offset: int,
        shape: Tuple[int, ...],
        dtype: str = "float32",
    ) -> Any:
        torch = self._torch
        self._view_serial += 1
        token_id = self._view_serial
        view = torch_from_cuda_pointer(
            torch,
            self.memory.pointer + offset,
            shape,
            self.device_index,
            dtype=dtype,
            token=RawArenaViewToken(lambda: self._release_view(token_id)),
        )
        self._active_views[token_id] = name
        return view

    def _make_views(self) -> None:
        self._control = self._make_view(
            name="control",
            offset=self.arena.ctrl_offset,
            shape=self.arena.ctrl_shape,
        )
        self._reset_indices = self._make_view(
            name="reset_indices",
            offset=self.arena.reset_indices_offset,
            shape=self.arena.reset_indices_shape,
            dtype="int64",
        )
        self._reset_qpos = self._make_view(
            name="reset_qpos",
            offset=self.arena.reset_qpos_offset,
            shape=self.arena.reset_qpos_shape,
        )
        self._reset_qvel = self._make_view(
            name="reset_qvel",
            offset=self.arena.reset_qvel_offset,
            shape=self.arena.reset_qvel_shape,
        )

    def _release_view(self, token_id: int) -> None:
        self._active_views.pop(token_id, None)

    def get_state_views(self, fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        self._require_open()
        requested = ("qpos", "qvel") if fields is None else tuple(fields)
        unknown = set(requested) - {"qpos", "qvel"}
        if unknown:
            raise KeyError(
                "unknown IsaacGym CUDA IPC tensor state field(s): " + ", ".join(sorted(unknown))
            )
        # READY is only a lifecycle/error barrier.  The worker has already
        # recorded its D2D projection, so enqueue the dependency on the caller's
        # Torch stream instead of synchronizing the parent CPU.
        with self._torch.cuda.device(self.device_index):
            stream = self._torch.cuda.current_stream(self.device_index).cuda_stream
            self.state_event.wait_stream(stream)
        for name in requested:
            if getattr(self, f"_{name}") is None:
                setattr(
                    self,
                    f"_{name}",
                    self._make_view(
                        name=name,
                        offset=getattr(self.arena, f"{name}_offset"),
                        shape=getattr(self.arena, f"{name}_shape"),
                    ),
                )
        return {name: getattr(self, f"_{name}") for name in requested}

    def _require_plan_device(self, tensor: Any, label: str) -> None:
        device = str(tensor.device)
        # A materialized Torch CUDA tensor normally carries an index.  Accept
        # an unindexed label only as the caller's current CUDA device; all
        # other CUDA indices fail closed against the worker-bound device.
        if device not in ("cuda", f"cuda:{self.device_index}"):
            raise ValueError(f"{label} must live on cuda:{self.device_index}, got {tensor.device}")

    def write_control(self, ctrl: Any) -> None:
        self._require_open()
        torch = self._torch
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError("IsaacGym CUDA IPC control must be a torch.Tensor")
        expected = self.arena.ctrl_shape
        if tuple(ctrl.shape) != expected:
            raise ValueError(
                f"IsaacGym CUDA IPC control must have shape {expected}, got {tuple(ctrl.shape)}"
            )
        if ctrl.dtype != torch.float32 or not bool(ctrl.is_contiguous()):
            raise TypeError("IsaacGym CUDA IPC control must be contiguous float32")
        self._require_plan_device(ctrl, "IsaacGym CUDA IPC control")
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            self._control.copy_(ctrl, non_blocking=True)
            self.control_event.record(stream)

    def step(self, nsteps: int = 1) -> Dict[str, Dict[str, float]]:
        self._require_open()
        if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
            raise ValueError("IsaacGym CUDA IPC nsteps must be a positive integer")
        if getattr(self.backend, "_body_wrench_pending", False):
            raise NotImplementedError("CUDA IPC body-wrench arena is not implemented")
        response = self.backend._request(
            CMD_CUDA_IPC_STEP,
            {"nsteps": nsteps},
            expect=self._protocol.CMD_READY,
        )
        if not isinstance(response, dict):
            raise RuntimeError("IsaacGym worker returned a malformed CUDA IPC step reply")
        # Keep the hot path stream-ordered: no default event synchronization or
        # host copy is performed after the control-only worker reply.
        with self._torch.cuda.device(self.device_index):
            stream = self._torch.cuda.current_stream(self.device_index).cuda_stream
            self.state_event.wait_stream(stream)
        timing = dict(response.get("timing", {}))
        timing["cuda_ipc_control_bytes"] = 0.0
        timing["cuda_ipc_state_bytes"] = 0.0
        self.last_timing = {"timing": timing}
        return self.last_timing

    def step_tensor(self, ctrl: Any, nsteps: int = 1) -> Dict[str, Dict[str, float]]:
        self.write_control(ctrl)
        return self.step(nsteps)

    def set_state_tensor(
        self,
        env_indices: Any,
        qpos: Any,
        qvel: Any,
    ) -> Dict[str, Dict[str, float]]:
        self._require_open()
        torch = self._torch
        if not isinstance(env_indices, torch.Tensor):
            raise TypeError("IsaacGym CUDA IPC reset rows must be a torch.Tensor")
        if not isinstance(qpos, torch.Tensor) or not isinstance(qvel, torch.Tensor):
            raise TypeError("IsaacGym CUDA IPC reset states must be torch.Tensors")
        if env_indices.dtype != torch.int64:
            raise TypeError("IsaacGym CUDA IPC reset rows must have dtype int64")
        if qpos.dtype != torch.float32 or qvel.dtype != torch.float32:
            raise TypeError("IsaacGym CUDA IPC reset states must have dtype float32")
        if not bool(env_indices.is_contiguous()) or not bool(qpos.is_contiguous()):
            raise TypeError("IsaacGym CUDA IPC reset rows/qpos must be contiguous")
        if not bool(qvel.is_contiguous()):
            raise TypeError("IsaacGym CUDA IPC reset qvel must be contiguous")
        self._require_plan_device(env_indices, "IsaacGym CUDA IPC reset rows")
        self._require_plan_device(qpos, "IsaacGym CUDA IPC reset qpos")
        self._require_plan_device(qvel, "IsaacGym CUDA IPC reset qvel")
        if env_indices.ndim != 1 or env_indices.shape[0] > self.arena.num_envs:
            raise ValueError("IsaacGym CUDA IPC reset row count is invalid")
        count = int(env_indices.shape[0])
        if tuple(qpos.shape) != (count, self.arena.nq):
            raise ValueError(
                f"IsaacGym CUDA IPC reset qpos must have shape {(count, self.arena.nq)}, "
                f"got {tuple(qpos.shape)}"
            )
        if tuple(qvel.shape) != (count, self.arena.nv):
            raise ValueError(
                f"IsaacGym CUDA IPC reset qvel must have shape {(count, self.arena.nv)}, "
                f"got {tuple(qvel.shape)}"
            )
        if count == 0:
            return {"timing": {"cuda_ipc_reset_bytes": 0.0}}
        # One bounded synchronization combines row-range and uniqueness closure.
        # Finite-value checks deliberately remain producer/task responsibility.
        selected = torch.zeros((self.arena.num_envs,), dtype=torch.bool, device=env_indices.device)
        safe_rows = env_indices.clamp(min=0, max=self.arena.num_envs - 1)
        selected[safe_rows] = True
        checks = torch.stack((env_indices.min(), env_indices.max(), selected.sum()))
        row_min, row_max, unique_count = checks.tolist()
        if row_min < 0 or row_max >= self.arena.num_envs:
            raise IndexError("IsaacGym CUDA IPC reset rows are out of range")
        if unique_count != count:
            raise ValueError("IsaacGym CUDA IPC reset rows must be unique")

        sequence = self._reset_sequence + 1
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            self._reset_indices[:count].copy_(env_indices, non_blocking=True)
            self._reset_qpos[:count].copy_(qpos, non_blocking=True)
            self._reset_qvel[:count].copy_(qvel, non_blocking=True)
            self.reset_event.record(stream)
        response = self.backend._request(
            CMD_CUDA_IPC_SET_STATE,
            {"count": count, "sequence": sequence},
            expect=self._protocol.CMD_READY,
        )
        self._reset_sequence = sequence
        if not isinstance(response, dict):
            raise RuntimeError("IsaacGym worker returned a malformed selected-reset reply")
        with torch.cuda.device(self.device_index):
            stream = torch.cuda.current_stream(self.device_index).cuda_stream
            self.state_event.wait_stream(stream)
        timing = dict(response.get("timing", {}))
        timing["cuda_ipc_reset_bytes"] = 0.0
        self.last_timing = {"timing": timing}
        return self.last_timing

    def _drop_views(self) -> None:
        self._control = None
        self._qpos = None
        self._qvel = None
        self._reset_indices = None
        self._reset_qpos = None
        self._reset_qvel = None
        gc.collect()

    def close(self) -> None:
        if self.closed:
            return
        self._drop_views()
        if self._active_views:
            # A failed close leaves the plan open.  Recreate non-public control
            # and reset arenas so step/reset remain usable; state views are
            # recreated lazily with an independent token generation.
            self._make_views()
            raise RuntimeError(
                "cannot close IsaacGym CUDA IPC plan while caller-held tensor views remain: "
                + ", ".join(sorted(set(self._active_views.values())))
            )
        try:
            self.backend._request(
                CMD_CUDA_IPC_DETACH,
                None,
                expect=self._protocol.CMD_READY,
            )
        finally:
            self.state_event.close()
            self.control_event.close()
            self.reset_event.close()
            self.memory.close()
            self.transport.close()
            self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("IsaacGym CUDA IPC plan is closed")


__all__.extend(
    [
        "CMD_CUDA_IPC_ATTACH",
        "CMD_CUDA_IPC_DETACH",
        "CMD_CUDA_IPC_STEP",
        "IsaacGymCudaIpcPlan",
    ]
)
