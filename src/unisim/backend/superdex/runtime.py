"""Reference-counted ownership of SuperDex's process-global CPU runtime."""

from __future__ import annotations

import os
import threading
from typing import Any

_LOCK = threading.RLock()
_PID: int | None = None
_USERS = 0


def acquire_runtime(physics: Any) -> None:
    """Initialize the single-threaded SDK once per spawn process."""
    global _PID, _USERS
    with _LOCK:
        if _PID is not None and _PID != os.getpid():
            raise RuntimeError("superdex runtime was inherited by fork; use spawn collectors")
        if not _USERS:
            if physics.is_initialized():
                raise RuntimeError(
                    "SuperDex was initialized outside UniSim; close that runtime before "
                    "constructing a backend so initialization/shutdown ownership is unambiguous"
                )
            physics.initialize(num_worker_threads=0)
            _PID = os.getpid()
        _USERS += 1


def release_runtime(physics: Any) -> None:
    """Shut down only after the last backend has destroyed its native resources."""
    global _PID, _USERS
    with _LOCK:
        if _PID != os.getpid() or not _USERS:
            return
        _USERS -= 1
        if not _USERS:
            physics.shutdown()
            _PID = None
