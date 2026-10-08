"""Public production backend contract.

The full contract lives in :mod:`unisim.backend.base`; this module preserves
the original public import path while benchmark metadata keeps its coarse
capability labels.
"""

from .backend.base import (
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    HostBridgeTransferPlan,
    PhysicsStateLayout,
    PhysicsStateParts,
    PreStepControlOutput,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorIOSpec,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TensorRuntimeDiagnostic,
    tensor_device_matches,
    validate_debug_overlays,
    validate_tensor_device,
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
    "HostBridgeTransferPlan",
    "TensorExecution",
    "TensorDataPlane",
    "TensorProcessTopology",
    "TensorLifecycleCapabilities",
    "TensorRuntimeDiagnostic",
    "TensorIOSpec",
    "tensor_device_matches",
    "UnsupportedCapabilityError",
    "validate_debug_overlays",
    "validate_tensor_device",
]
