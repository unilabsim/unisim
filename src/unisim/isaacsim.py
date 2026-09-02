"""IsaacSim/IsaacLab out-of-process adapter."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .optional import OptionalDependencyError
from .subprocess_ipc.backend import SubprocessBackend, SubprocessWorkerError


class IsaacSimDependencyError(OptionalDependencyError):
    """Raised when the dedicated IsaacSim worker cannot be resolved."""


class IsaacSimBackend(SubprocessBackend):
    backend_type = "isaacsim"
    module_names = ("isaaclab", "omni.isaac.lab", "omni.isaac.core")
    install_hint = (
        "Configure UNISIM_ISAACSIM_PYTHON or install the dedicated IsaacSim/IsaacLab "
        "worker described in the UniSim IsaacSim support guide."
    )

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        worker_command: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if runtime is None and worker_command is None:
            override = os.environ.get("UNISIM_ISAACSIM_PYTHON")
            if override and Path(override).is_file():
                worker_command = [override]
            else:
                raise IsaacSimDependencyError(self.install_hint)
        super().__init__(runtime=runtime, worker_command=worker_command, **kwargs)


__all__ = ["IsaacSimBackend", "IsaacSimDependencyError", "SubprocessWorkerError"]
