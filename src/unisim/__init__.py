"""Backend-neutral physics contracts for UniSim.

The distribution is named ``unisim-core`` while the public Python namespace is
``unisim``. Engine adapters are optional and are loaded explicitly by callers.
"""

from .adapters import ADAPTER_SPECS, AdapterSpec, adapter_spec
from .benchmark import BenchmarkCase, BenchmarkResult
from .conformance import assert_backend_conformance
from .contract import (
    BackendCapability,
    BackendError,
    SimBackend,
    UnsupportedCapabilityError,
)
from .drake import DrakeBackend
from .factory import create_backend
from .fake import FakeBackend
from .genesis import GenesisBackend
from .isaacgym import IsaacGymBackend
from .isaacsim import IsaacSimBackend
from .mjwarp import MJWarpBackend, MjwarpBackend
from .motrix import MotrixBackend
from .mujoco import MuJoCoBackend

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
    "IsaacSimBackend",
    "MuJoCoBackend",
    "MotrixBackend",
    "SimBackend",
    "UnsupportedCapabilityError",
    "assert_backend_conformance",
    "adapter_spec",
    "create_backend",
]
