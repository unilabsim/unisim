"""IsaacGym out-of-process adapter."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .optional import OptionalDependencyError
from .subprocess_ipc.backend import SubprocessBackend, SubprocessWorkerError


class IsaacGymDependencyError(OptionalDependencyError):
    """Raised when the dedicated IsaacGym worker cannot be resolved."""


class IsaacGymBackend(SubprocessBackend):
    backend_type = "isaacgym"
    module_names = ("isaacgym",)
    install_hint = (
        "Configure UNISIM_ISAACGYM_PYTHON or install the dedicated Python 3.8 "
        "worker described in the UniSim IsaacGym support guide."
    )

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        worker_command: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if runtime is None and worker_command is None:
            override = os.environ.get("UNISIM_ISAACGYM_PYTHON")
            if override and Path(override).is_file():
                worker_command = [override]
            else:
                raise IsaacGymDependencyError(self.install_hint)
        super().__init__(runtime=runtime, worker_command=worker_command, **kwargs)


__all__ = ["IsaacGymBackend", "IsaacGymDependencyError", "SubprocessWorkerError"]
