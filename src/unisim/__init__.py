"""Backend-neutral physics contracts for UniSim.

The distribution is named ``unisim-core`` while the public Python namespace is
``unisim``. Engine adapters are optional and are loaded explicitly by callers.
"""

from .benchmark import BenchmarkCase, BenchmarkResult
from .conformance import assert_backend_conformance
from .contract import (
    BackendCapability,
    BackendError,
    SimBackend,
    UnsupportedCapabilityError,
)
from .factory import create_backend
from .fake import FakeBackend
from .mujoco import MuJoCoBackend

__all__ = [
    "BackendCapability",
    "BackendError",
    "BenchmarkCase",
    "BenchmarkResult",
    "FakeBackend",
    "MuJoCoBackend",
    "SimBackend",
    "UnsupportedCapabilityError",
    "assert_backend_conformance",
    "create_backend",
]
