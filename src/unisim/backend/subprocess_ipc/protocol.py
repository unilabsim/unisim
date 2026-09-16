"""Canonical pipe and shared-memory protocol for subprocess backends.

The module is loaded both by the host interpreter and by external workers via
an explicit file path.  Keep it compatible with Python 3.8 and import only the
standard library plus NumPy.
"""

from __future__ import annotations

import importlib.util
import pickle
import struct
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO, Dict, Tuple

import numpy as np

CMD_INIT = "INIT"
CMD_ATTACH = "ATTACH_SLOTS"
CMD_STEP = "STEP"
CMD_SET_STATE = "SET_STATE"
CMD_RESET_ENTITIES = "RESET_ENTITIES"
CMD_REFRESH = "REFRESH"
CMD_GET_META = "GET_META"
CMD_INIT_RENDERER = "INIT_RENDERER"
CMD_RENDER_FRAME = "RENDER_FRAME"
CMD_CAPTURE_FRAME = "CAPTURE_FRAME"
CMD_SHUTDOWN = "SHUTDOWN"

CMD_READY = "READY"
CMD_META = "META"
CMD_ERROR = "ERROR"

_PICKLE_PROTOCOL = 4
_HEADER = struct.Struct("<Q")
HEADER_SIZE = _HEADER.size


def pack_message(cmd: str, payload: Any = None) -> bytes:
    return pickle.dumps({"cmd": cmd, "payload": payload}, protocol=_PICKLE_PROTOCOL)


def unpack_header(data: bytes) -> int:
    (size,) = _HEADER.unpack(data)
    return int(size)


def decode_message(body: bytes) -> Dict[str, Any]:
    message = pickle.loads(body)
    if not isinstance(message, dict) or "cmd" not in message:
        raise ValueError(f"malformed worker message: {message!r}")
    return message


class WorkerDisconnectedError(EOFError):
    """Raised when a worker pipe closes before a complete message arrives."""


def send_message(stream: BinaryIO, cmd: str, payload: Any = None) -> None:
    body = pack_message(cmd, payload)
    stream.write(_HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise WorkerDisconnectedError(
                f"pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO) -> Dict[str, Any]:
    size = unpack_header(_read_exactly(stream, _HEADER.size))
    return decode_message(_read_exactly(stream, size))


_SLOT_DTYPES: Dict[str, str] = {
    "ctrl": "float32",
    "root_state": "float32",
    "dof_state": "float32",
    "body_state": "float32",
    "contact_force": "float32",
    "reset_env_ids": "int32",
    "reset_qpos": "float32",
    "reset_qvel": "float32",
}

SLOT_NAMES = tuple(_SLOT_DTYPES)

# Shared scene schema is separate from the M1 configuration-report schema.
SCENE_SCHEMA_VERSION = 1
_SCENE_SLOT_DTYPES: Dict[str, str] = {
    "qpos": "float32",
    "qvel": "float32",
    "entity_root_state": "float32",
    "reset_entity_root_state": "float32",
    "reset_qpos_mask": "uint8",
    "reset_qvel_mask": "uint8",
    "reset_root_mask": "uint8",
}


def load_scene_layout(payload: Dict[str, Any]) -> Any:
    """Use the same strict layout validator in isolated Python 3.8 workers.

    Loading by path avoids importing the host package or an optional engine.
    The dataclass module must be registered before execution for Python 3.8.
    """
    module_name = "unisim_worker_scene_layout"
    module = sys.modules.get(module_name)
    if module is None:
        path = Path(__file__).resolve().parents[2] / "scene_layout.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load shared scene layout validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    return module.CompiledSceneLayout.from_dict(payload)


def scene_slot_shapes(num_envs: int, layout: Any) -> Dict[str, Tuple[int, ...]]:
    """Explicit state/action/root widths for the mapped scene protocol."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    num_entities = len(layout.entities)
    return {
        "ctrl": (num_envs, layout.nu),
        "qpos": (num_envs, layout.nq),
        "qvel": (num_envs, layout.nv),
        "entity_root_state": (num_envs, num_entities, 13),
        "body_state": (num_envs, layout.nbody, 13),
        "contact_force": (num_envs, layout.nbody, 3),
        "reset_env_ids": (num_envs,),
        "reset_qpos": (num_envs, layout.nq),
        "reset_qvel": (num_envs, layout.nv),
        "reset_entity_root_state": (num_envs, num_entities, 13),
        "reset_qpos_mask": (layout.nq,),
        "reset_qvel_mask": (layout.nv,),
        # Position and velocity channels are independently optional.
        "reset_root_mask": (num_entities, 2),
    }


def validate_slot_specs(specs: Dict[str, Any], expected: Dict[str, Tuple[int, ...]]) -> None:
    """Validate all wire descriptors before attaching any shared memory."""
    if not isinstance(specs, dict) or set(specs) != set(expected):
        raise ValueError("shared-memory slot names do not match the negotiated layout")
    for name, shape in expected.items():
        spec = specs[name]
        if not isinstance(spec, dict) or set(spec) != {"shm", "shape", "dtype"}:
            raise ValueError("malformed shared-memory descriptor for " + name)
        if not isinstance(spec["shm"], str) or not spec["shm"]:
            raise ValueError("invalid shared-memory name for " + name)
        actual = spec["shape"]
        if (
            not isinstance(actual, (list, tuple))
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in actual)
            or tuple(actual) != shape
        ):
            raise ValueError("shared-memory shape mismatch for " + name)
        if np.dtype(spec["dtype"]) != slot_dtype(name):
            raise ValueError("shared-memory dtype mismatch for " + name)


def slot_shapes(num_envs: int, num_dof: int, num_bodies: int) -> Dict[str, Tuple[int, ...]]:
    if num_envs <= 0 or num_dof < 0 or num_bodies <= 0:
        raise ValueError(
            "slot shapes require num_envs>0, num_dof>=0, num_bodies>0; "
            f"got {num_envs}, {num_dof}, {num_bodies}"
        )
    return {
        "ctrl": (num_envs, num_dof),
        "root_state": (num_envs, 13),
        "dof_state": (num_envs, num_dof, 2),
        "body_state": (num_envs, num_bodies, 13),
        "contact_force": (num_envs, num_bodies, 3),
        "reset_env_ids": (num_envs,),
        "reset_qpos": (num_envs, 7 + num_dof),
        "reset_qvel": (num_envs, 6 + num_dof),
    }


def slot_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_SLOT_DTYPES[name] if name in _SLOT_DTYPES else _SCENE_SLOT_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unknown shm slot {name!r}; known: {sorted(_SLOT_DTYPES)}") from exc


def slot_nbytes(name: str, shape: Tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(slot_dtype(name).itemsize)


def slot_allocation_nbytes(name: str, shape: Tuple[int, ...]) -> int:
    """SharedMemory needs nonzero storage even when nu or a state width is zero."""
    return max(1, slot_nbytes(name, shape))


def serialize_exception(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def format_worker_error(payload: Dict[str, str], backend: str = "subprocess") -> str:
    return (
        f"{backend} worker raised {payload.get('type', 'Error')}: "
        f"{payload.get('message', '')}\n"
        f"worker traceback:\n{payload.get('traceback', '<unavailable>')}"
    )


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat)[..., [1, 2, 3, 0]]


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()
    q[..., 1:4] = -q[..., 1:4]
    return quat_rotate(q, vec)


__all__ = [
    "CMD_ATTACH",
    "CMD_CAPTURE_FRAME",
    "CMD_ERROR",
    "CMD_GET_META",
    "CMD_INIT",
    "CMD_INIT_RENDERER",
    "CMD_META",
    "CMD_READY",
    "CMD_REFRESH",
    "CMD_RENDER_FRAME",
    "CMD_SET_STATE",
    "CMD_SHUTDOWN",
    "CMD_STEP",
    "HEADER_SIZE",
    "SLOT_NAMES",
    "WorkerDisconnectedError",
    "decode_message",
    "format_worker_error",
    "pack_message",
    "quat_rotate",
    "quat_rotate_inverse",
    "recv_message",
    "send_message",
    "serialize_exception",
    "slot_dtype",
    "slot_nbytes",
    "slot_shapes",
    "unpack_header",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
