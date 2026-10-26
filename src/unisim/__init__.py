"""Backend-neutral physics contracts for UniSim.

The distribution is named ``unisim-core`` while the public Python namespace is
``unisim``. Engine adapters are optional and are loaded explicitly by callers.
"""

from .adapters import ADAPTER_SPECS, AdapterSpec, adapter_spec
from .benchmark import BenchmarkCase, BenchmarkResult
from .capabilities import (
    CapabilityCondition,
    CapabilityDeclaration,
    CapabilityEvidence,
    CapabilityReport,
    CapabilityScope,
    SupportLevel,
    get_adapter_capabilities,
)
from .conformance import assert_backend_conformance, assert_ray_caster_conformance
from .contract import (
    COMPLETION_CONTRACT_VERSION,
    COMPLETION_OPERATIONS,
    PACKED_RESET_MODEL_FIELDS,
    PACKED_RESET_PERSISTENT_CHANNELS,
    BackendCapability,
    BackendError,
    CameraCfg,
    CompletionToken,
    DebugOverlayGetter,
    DebugPrimitive,
    HostBridgeTransferPlan,
    PackedResetTransaction,
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
    UnsupportedCapabilityError,
    torch_device_matches,
    validate_debug_overlays,
    validate_torch_device,
)
from .entities import (
    EntityInitialState,
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from .factory import create_backend, create_ray_caster
from .fake import FakeBackend
from .fake_ray import FakeRayCaster
from .inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ConfigurationScope,
    ImportReport,
)
from .optional import OptionalDependencyError
from .ray_query import (
    RAY_CASTER_SPECS,
    RayCaster,
    RayCasterCapabilities,
    RayCasterSpec,
    RayCollisionDescription,
    RayGeomType,
    RayMeshDescription,
    RayPoseTensorView,
    RaySceneDescription,
    RayTraceFilter,
    RayTraceOutputs,
    RayTraceResult,
    RayTraceTensorResult,
    ray_caster_spec,
    ray_geom_type,
    require_ray_trace_outputs,
    require_ray_trace_query,
)
from .scene_layout import CompiledSceneLayout, EntityLayout, JointLayout
from .validation import (
    SemanticRequirements,
    SemanticValidationError,
    validate_semantic_requirements,
)

__all__ = [
    "CompiledSceneLayout",
    "EntityLayout",
    "JointLayout",
    "EntityInitialState",
    "EntityStatePatch",
    "EntityVariantBinding",
    "SceneEntitySpec",
    "SceneResetRequest",
    "CapabilityCondition",
    "CapabilityDeclaration",
    "CapabilityEvidence",
    "CapabilityReport",
    "CapabilityScope",
    "SupportLevel",
    "get_adapter_capabilities",
    "SemanticRequirements",
    "SemanticValidationError",
    "validate_semantic_requirements",
    "ConfigurationField",
    "ConfigurationProvenance",
    "ConfigurationScope",
    "ImportReport",
    "BackendCapability",
    "BackendError",
    "BenchmarkCase",
    "BenchmarkResult",
    "ADAPTER_SPECS",
    "AdapterSpec",
    "CameraCfg",
    "DebugOverlayGetter",
    "DebugPrimitive",
    "PhysicsStateLayout",
    "PhysicsStateParts",
    "PreStepControlOutput",
    "FakeBackend",
    "DrakeBackend",
    "MjwarpBackend",
    "NewtonBackend",
    "NewtonDependencyError",
    "SuperDexBackend",
    "SuperDexDependencyError",
    "GenesisBackend",
    "IsaacGymBackend",
    "IsaacGymDependencyError",
    "IsaacSimBackend",
    "IsaacSimDependencyError",
    "OptionalDependencyError",
    "MjcfSubprocessBackend",
    "SubprocessWorkerError",
    "MuJoCoBackend",
    "MotrixBackend",
    "SimBackend",
    "HostBridgeTransferPlan",
    "TensorExecution",
    "TensorDataPlane",
    "TensorProcessTopology",
    "TensorLifecycleCapabilities",
    "TensorRuntimeDiagnostic",
    "COMPLETION_CONTRACT_VERSION",
    "COMPLETION_OPERATIONS",
    "CompletionToken",
    "TensorIOSpec",
    "PACKED_RESET_MODEL_FIELDS",
    "PACKED_RESET_PERSISTENT_CHANNELS",
    "PackedResetTransaction",
    "UnsupportedCapabilityError",
    "torch_device_matches",
    "assert_backend_conformance",
    "assert_ray_caster_conformance",
    "adapter_spec",
    "create_backend",
    "create_ray_caster",
    "validate_debug_overlays",
    "validate_torch_device",
    "RAY_CASTER_SPECS",
    "RayCaster",
    "RayCasterCapabilities",
    "RayCasterSpec",
    "RayCollisionDescription",
    "RayGeomType",
    "RayMeshDescription",
    "RayPoseTensorView",
    "RaySceneDescription",
    "RayTraceFilter",
    "RayTraceTensorResult",
    "RayTraceOutputs",
    "RayTraceResult",
    "ray_geom_type",
    "FakeRayCaster",
    "ray_caster_spec",
    "require_ray_trace_outputs",
    "require_ray_trace_query",
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
        "MjwarpBackend": (".backend.mjwarp", "MjwarpBackend"),
        "NewtonBackend": (".backend.newton", "NewtonBackend"),
        "NewtonDependencyError": (".backend.newton", "NewtonDependencyError"),
        "SuperDexBackend": (".backend.superdex", "SuperDexBackend"),
        "SuperDexDependencyError": (".backend.superdex", "SuperDexDependencyError"),
        "MotrixBackend": (".backend.motrix", "MotrixBackend"),
        "MuJoCoBackend": (".backend.mujoco", "MuJoCoBackend"),
        "MjcfSubprocessBackend": (".backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        "SubprocessWorkerError": (".backend.subprocess_ipc.backend", "SubprocessWorkerError"),
    }
    target = modules.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
