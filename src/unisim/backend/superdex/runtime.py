"""Reference-counted ownership of SuperDex's process-global CPU runtime."""

from __future__ import annotations

import os
import threading
from typing import Any

_LOCK = threading.RLock()
_PID: int | None = None
_USERS = 0
_THREADS: int | None = None


def acquire_runtime(physics: Any, num_threads: int) -> None:
    """Initialize once per spawn process; reject fork and conflicting thread policies."""
    global _PID, _USERS, _THREADS
    with _LOCK:
        if _PID is not None and _PID != os.getpid():
            raise RuntimeError("superdex runtime was inherited by fork; use spawn collectors")
        if _USERS:
            if num_threads != _THREADS:
                raise ValueError("superdex instances in one process must use the same num_threads")
        else:
            if physics.is_initialized():
                raise RuntimeError(
                    "SuperDex was initialized outside UniSim; close that runtime before "
                    "constructing a backend so initialization/shutdown ownership is unambiguous"
                )
            physics.initialize(num_worker_threads=num_threads)
            _PID, _THREADS = os.getpid(), num_threads
        _USERS += 1


def release_runtime(physics: Any) -> None:
    """Shut down only after the last backend has destroyed its native resources."""
    global _PID, _USERS, _THREADS
    with _LOCK:
        if _PID != os.getpid() or not _USERS:
            return
        _USERS -= 1
        if not _USERS:
            physics.shutdown()
            _PID = _THREADS = None
