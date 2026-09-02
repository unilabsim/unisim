"""Python 3.8-compatible control protocol shared by Isaac workers."""
from __future__ import annotations

import pickle
import struct
import traceback
from typing import Any, BinaryIO

_HEADER = struct.Struct("<Q")
HEADER_SIZE = _HEADER.size
_PICKLE_PROTOCOL = 4
CMD_INIT = "INIT"
CMD_ATTACH = "ATTACH_SLOTS"
CMD_STEP = "STEP"
CMD_SET_STATE = "SET_STATE"
CMD_REFRESH = "REFRESH"
CMD_GET_META = "GET_META"
CMD_SHUTDOWN = "SHUTDOWN"
CMD_READY = "READY"
CMD_META = "META"
CMD_ERROR = "ERROR"
SLOT_NAMES = ("ctrl", "root_state", "dof_state", "body_state", "contact_force")


class WorkerDisconnectedError(EOFError):
    """Raised when a worker closes its pipe before a complete frame arrives."""


def pack_message(cmd: str, payload: Any = None) -> bytes:
    return pickle.dumps({"cmd": cmd, "payload": payload}, protocol=_PICKLE_PROTOCOL)


def unpack_header(data: bytes) -> int:
    return int(_HEADER.unpack(data)[0])


def decode_message(body: bytes) -> dict[str, Any]:
    message = pickle.loads(body)
    if not isinstance(message, dict) or "cmd" not in message:
        raise ValueError(f"malformed worker message: {message!r}")
    return message


def send_message(stream: BinaryIO, cmd: str, payload: Any = None) -> None:
    body = pack_message(cmd, payload)
    stream.write(_HEADER.pack(len(body)))
    stream.write(body)
    stream.flush()


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise WorkerDisconnectedError(
                f"pipe closed while reading {size} bytes (got {size - remaining})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO) -> dict[str, Any]:
    size = unpack_header(_read_exactly(stream, HEADER_SIZE))
    return decode_message(_read_exactly(stream, size))


def serialize_exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def format_worker_error(payload: dict[str, str], backend: str = "subprocess") -> str:
    return (
        f"{backend} worker raised {payload.get('type', 'Error')}: "
        f"{payload.get('message', '')}\nworker traceback:\n"
        f"{payload.get('traceback', '<unavailable>')}"
    )


__all__ = [
    "CMD_ATTACH", "CMD_ERROR", "CMD_GET_META", "CMD_INIT", "CMD_META", "CMD_READY",
    "CMD_REFRESH", "CMD_SET_STATE", "CMD_SHUTDOWN", "CMD_STEP", "HEADER_SIZE", "SLOT_NAMES",
    "WorkerDisconnectedError", "decode_message", "format_worker_error", "pack_message",
    "recv_message", "send_message", "serialize_exception", "unpack_header",
]
