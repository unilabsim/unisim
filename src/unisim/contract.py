"""Public production backend contract.

The full contract lives in :mod:`unisim.backend.base`; this module preserves
the original public import path while benchmark metadata keeps its coarse
capability labels.
"""

from .backend.base import (
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    PhysicsStateLayout,
    PhysicsStateParts,
    PreStepControlOutput,
    SimBackend,
    TensorExecution,
    TensorLifecycleCapabilities,
    validate_debug_overlays,
)
from .errors import BackendCapability, BackendError, UnsupportedCapabilityError

__all__ = [
    "BackendCapability",
    "BackendError",
    "CameraCfg",
    "DebugOverlayGetter",
    "DebugPrimitive",
    "PhysicsStateLayout",
    "PhysicsStateParts",
    "PreStepControlOutput",
    "SimBackend",
    "TensorExecution",
    "TensorLifecycleCapabilities",
    "UnsupportedCapabilityError",
    "validate_debug_overlays",
]
