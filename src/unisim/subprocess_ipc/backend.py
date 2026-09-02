"""Shared host-side subprocess adapter boundary."""
from __future__ import annotations

from typing import Any

from ..optional import RuntimeBackend


class SubprocessWorkerError(RuntimeError):
    """Normalized worker failure carrying optional traceback diagnostics."""


class SubprocessBackend(RuntimeBackend):
    """Common contract implementation for out-of-process engine workers."""
    backend_type = "subprocess"
    module_names: tuple[str, ...] = ()
    install_hint = "Configure the dedicated vendor worker runtime for this adapter."

    def __init__(self, *, worker_command: list[str] | None = None, **kwargs: Any) -> None:
        if worker_command is not None and (
            not isinstance(worker_command, list)
            or not worker_command
            or not all(isinstance(item, str) for item in worker_command)
        ):
            raise TypeError("worker_command must be a non-empty list of strings")
        self.worker_command = None if worker_command is None else tuple(worker_command)
        super().__init__(**kwargs)


__all__ = ["SubprocessBackend", "SubprocessWorkerError"]
