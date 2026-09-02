"""Backend-neutral physics contracts for UniSim.

The distribution is named ``unisim-core`` while the public Python namespace is
``unisim``. Engine adapters are optional and are loaded explicitly by callers.
"""

from .adapters import ADAPTER_SPECS, AdapterSpec, adapter_spec
from .benchmark import BenchmarkCase, BenchmarkResult
from .conformance import assert_backend_conformance
from .contract import BackendCapability, BackendError, SimBackend, UnsupportedCapabilityError
from .factory import create_backend
from .fake import FakeBackend
from .optional import OptionalDependencyError

__all__ = [
    "BackendCapability",
    "BackendError",
    "BenchmarkCase",
    "BenchmarkResult",
    "ADAPTER_SPECS",
    "AdapterSpec",
    "FakeBackend",
    "DrakeBackend",
    "MJWarpBackend",
    "MjwarpBackend",
    "GenesisBackend",
    "IsaacGymBackend",
    "IsaacGymDependencyError",
    "IsaacSimBackend",
    "IsaacSimDependencyError",
    "OptionalDependencyError",
    "SubprocessBackend",
    "SubprocessWorkerError",
    "MuJoCoBackend",
    "MotrixBackend",
    "SimBackend",
    "UnsupportedCapabilityError",
    "assert_backend_conformance",
    "adapter_spec",
    "create_backend",
]


def __getattr__(name: str):
    """Resolve optional adapter exports without importing engine SDKs eagerly."""
    modules = {
        "DrakeBackend": (".backend.drake", "DrakeBackend"),
        "GenesisBackend": (".backend.genesis", "GenesisBackend"),
        "IsaacGymBackend": (".backend.isaacgym", "IsaacGymBackend"),
        "IsaacGymDependencyError": (".backend.isaacgym.dependencies", "IsaacGymDependencyError"),
        "IsaacSimBackend": (".backend.isaacsim", "IsaacSimBackend"),
        "IsaacSimDependencyError": (".backend.isaacsim.dependencies", "IsaacSimDependencyError"),
        "MJWarpBackend": (".backend.mjwarp", "MjwarpBackend"),
        "MjwarpBackend": (".backend.mjwarp", "MjwarpBackend"),
        "MotrixBackend": (".backend.motrix", "MotrixBackend"),
        "MuJoCoBackend": (".backend.mujoco", "MuJoCoBackend"),
        "SubprocessBackend": (".backend.subprocess_ipc.backend", "SubprocessBackend"),
        "SubprocessWorkerError": (".backend.subprocess_ipc.backend", "SubprocessWorkerError"),
    }
    target = modules.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
