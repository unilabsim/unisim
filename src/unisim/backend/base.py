import abc
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np

from unisim.capabilities import CapabilityReport, backend_capabilities
from unisim.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    IntervalTermOp,
    TensorResetRandomizationPayload,
    _validate_reset_term,
)
from unisim.entities import SceneResetRequest
from unisim.inspection import ImportReport
from unisim.scene_layout import CompiledSceneLayout


@dataclass
class PreStepControlOutput:
    """Per-substep actuator control plus an optional dynamic body wrench.

    ``force`` and ``torque`` are world-frame arrays with shape
    ``(num_envs, len(body_ids), 3)`` in newtons and newton-meters.  The force
    acts at the target body's center of mass and the torque is about that
    center, matching MuJoCo ``xfrc_applied`` semantics.  The wrench is
    recomputed by the callback before every physics substep and replaces the
    previous substep's dynamic wrench; it is never accumulated across substeps
    or across control steps, and it composes additively with wrenches staged
    through the interval randomization path for the same step call.
    """

    ctrl: np.ndarray
    body_ids: np.ndarray | None = None
    force: np.ndarray | None = None
    torque: np.ndarray | None = None


# The explicit TypeAlias keeps the union valid when NumPy resolves to Any
# (mypy runs with no_site_packages, matching unilab-rl).
PreStepControlResult: TypeAlias = np.ndarray | PreStepControlOutput
PreStepControlFn = Callable[[Any, np.ndarray], PreStepControlResult]
TerrainHeightSampleFn = Callable[[np.ndarray], np.ndarray]
SensorReadFn = Callable[[], Any]


DebugPrimitiveKind = Literal["sphere", "box", "frame", "arrow", "ghost_geom", "text"]
DEBUG_PRIMITIVE_KINDS = frozenset({"sphere", "box", "frame", "arrow", "ghost_geom", "text"})


class TensorExecution(Enum):
    """Execution profile of the optional backend tensor lifecycle.

    ``DEVICE_RESIDENT`` keeps the backend hot path and returned state arrays on
    the same accelerator device. ``HOST_BRIDGE`` executes physics on host
    arrays but accepts accelerator control/state tensors at explicit, measured
    host-transfer boundaries. ``UNSUPPORTED`` is the fail-closed default.
    """

    UNSUPPORTED = "unsupported"
    HOST_BRIDGE = "host_bridge"
    DEVICE_RESIDENT = "device_resident"


class TensorProcessTopology(Enum):
    """Process topology of a declared tensor lifecycle."""

    IN_PROCESS = "in_process"
    EXTERNAL_WORKER = "external_worker"


class TensorDataPlane(Enum):
    """Bulk tensor transport used by a declared tensor lifecycle.

    ``NONE`` is the fail-closed default. ``DIRECT`` means an in-process backend
    owns its device storage directly. ``HOST_BRIDGE`` denotes explicit in-process
    accelerator/host boundaries. ``HOST_SHARED_MEMORY`` and ``CUDA_IPC`` describe
    subprocess transports; neither implies that every optional tensor method is
    supported.
    """

    NONE = "none"
    DIRECT = "direct"
    HOST_BRIDGE = "host_bridge"
    HOST_SHARED_MEMORY = "host_shared_memory"
    CUDA_IPC = "cuda_ipc"


class SelectedResetPublication(Enum):
    """Visibility of public tensor views after a selected reset commits.

    ``AUTHORITATIVE_VIEWS`` is a strong postcondition: once
    ``set_state_tensor`` returns, subsequent public state and sensor views are
    authoritative for the committed rows. Adapters may implement immediate
    refresh or lazy refresh at the first public view. Callers must not advance
    physics merely to obtain readiness.
    """

    AUTHORITATIVE_VIEWS = "authoritative_views"


COMPLETION_OPERATIONS: frozenset[str] = frozenset({"step"})
COMPLETION_CONTRACT_VERSION = 3

__all__ = [
    "COMPLETION_CONTRACT_VERSION",
    "COMPLETION_OPERATIONS",
    "DEBUG_PRIMITIVE_KINDS",
    "CameraCfg",
    "CapabilityReport",
    "CompiledSceneLayout",
    "CompletionToken",
    "DebugOverlayGetter",
    "DebugPrimitiveKind",
    "DomainRandomizationCapabilities",
    "HeightScanner",
    "HostBridgeTransferPlan",
    "ImportReport",
    "IntervalRandomizationPlan",
    "IntervalTermOp",
    "MocapPoseBinding",
    "PLAY_RENDER_MODES",
    "PhysicsStateLayout",
    "PhysicsStateParts",
    "PlayCapabilities",
    "PlayRenderPlan",
    "PreStepControlFn",
    "PreStepControlOutput",
    "PreStepControlResult",
    "PublicStateWidths",
    "RenderClosedError",
    "RootStateLayout",
    "SceneResetRequest",
    "SelectedResetPublication",
    "SensorDescriptor",
    "SensorReadFn",
    "SensorView",
    "SimBackend",
    "TensorDataPlane",
    "TensorExecution",
    "TensorIOSpec",
    "TensorLifecycleCapabilities",
    "TensorProcessTopology",
    "TensorResetRandomizationPayload",
    "TensorRuntimeDiagnostic",
    "TerrainHeightSampleFn",
    "TerrainSpawnData",
    "TrackedBodyStateViews",
    "backend_capabilities",
    "log_playback_plan",
    "normalize_play_render_mode",
    "torch_device_matches",
    "unsupported_debug_overlay_error",
    "validate_debug_overlays",
    "validate_torch_device",
]


_TENSOR_DEVICE_LABEL = re.compile(r"^(cpu|cuda)(?::([0-9]+))?$")


def _tensor_device_parts(label: str, *, context: str) -> tuple[str, int | None]:
    if not isinstance(label, str):
        raise ValueError(f"{context} Torch device label must be a string, got {label!r}")
    match = _TENSOR_DEVICE_LABEL.fullmatch(label.strip())
    if match is None:
        raise ValueError(
            f"{context} Torch device label must be 'cpu', 'cuda', or 'cuda:<index>'; got {label!r}"
        )
    family, raw_index = match.groups()
    if family == "cpu" and raw_index is not None:
        raise ValueError(
            f"{context} Torch device label must be 'cpu', 'cuda', or 'cuda:<index>'; got {label!r}"
        )
    return family, int(raw_index) if raw_index is not None else None


def torch_device_matches(
    accepted_devices: Sequence[str],
    requested_device: Any,
    *,
    current_device: int | None = None,
) -> bool:
    """Match declared Torch device families or exact CUDA indices.

    ``cpu`` and ``cuda`` are family declarations: ``cuda`` accepts any valid CUDA
    index and the adapter remains responsible for exact-device validation.
    ``cuda:<index>`` is an exact declaration. An unindexed CUDA request denotes
    the caller's current device, so callers that need exact matching pass its
    index explicitly. Keeping this helper free of Torch imports also makes the
    contract usable by SDK-free capability consumers.
    """

    try:
        requested_family, requested_index = _tensor_device_parts(
            str(requested_device), context="requested"
        )
    except ValueError:
        return False
    if current_device is not None and current_device < 0:
        return False

    resolved_index = requested_index if requested_index is not None else current_device
    for accepted in accepted_devices:
        accepted_family, accepted_index = _tensor_device_parts(accepted, context="accepted")
        if accepted_family != requested_family:
            continue
        if accepted_index is None or accepted_index == resolved_index:
            return True
    return False


def validate_torch_device(
    accepted_devices: Sequence[str],
    requested_device: Any,
    *,
    current_device: int | None = None,
    label: str = "Tensor",
) -> None:
    """Fail closed when a requested Torch device is outside a declaration."""

    if not torch_device_matches(accepted_devices, requested_device, current_device=current_device):
        accepted = ", ".join(repr(device) for device in accepted_devices) or "none"
        raise ValueError(
            f"{label} device {str(requested_device)!r} is not supported; "
            f"accepted Torch devices are {accepted}"
        )


@dataclass(frozen=True)
class TensorLifecycleCapabilities:
    """Machine-readable limits of an adapter's tensor lifecycle.

    The flags describe the optional methods, not whether a particular tensor
    engine is installed. Unsupported operations remain fail-closed even when
    the coarse execution mode is not ``UNSUPPORTED``.

    ``device_reset_randomization`` declares that ``set_state_tensor`` accepts
    :class:`~unisim.dr.types.TensorResetRandomizationPayload` values with
    arrays resident on the backend's declared Torch device. Device-resident
    adapters apply them as selected-row in-place scatters; in-process host
    bridges materialize them inside the same explicit reset boundary.
    """

    execution: TensorExecution
    state_views: bool = False
    state_fields: frozenset[str] = frozenset()
    sensor_views: bool = False
    stepping: bool = False
    generalized_force_injection: bool = False
    selected_reset: bool = False
    reset_randomization: bool = False
    device_reset_randomization: bool = False
    packed_reset: bool = False
    packed_reset_fields: frozenset[str] = frozenset()
    packed_reset_mocap: bool = False
    fixed_variants: bool = False
    host_pre_step_control: bool = False
    packed_host_bridge: bool = False
    process_topology: TensorProcessTopology = TensorProcessTopology.IN_PROCESS
    data_plane: TensorDataPlane = TensorDataPlane.NONE
    stream_event_ownership: str | None = None
    torch_devices: tuple[str, ...] = ()
    selected_reset_publication: SelectedResetPublication | None = None
    requires_post_construction_publication_barrier: bool = False
    tracked_body_views: bool = False
    # Fixed-width mask reset declarations. They require the public
    # ``set_state_tensor_mask`` semantics and authoritative selected-row views.
    mask_selected_reset: bool = False
    mask_selected_reset_full_refresh: bool = False

    def __post_init__(self) -> None:
        if self.execution is TensorExecution.UNSUPPORTED:
            valid = (
                self.process_topology is TensorProcessTopology.IN_PROCESS
                and self.data_plane is TensorDataPlane.NONE
            )
            if not valid or any(
                (
                    self.state_views,
                    self.sensor_views,
                    self.stepping,
                    self.generalized_force_injection,
                    self.selected_reset,
                    self.reset_randomization,
                    self.device_reset_randomization,
                    self.packed_reset,
                    self.packed_reset_mocap,
                    self.fixed_variants,
                    self.host_pre_step_control,
                    self.packed_host_bridge,
                    self.tracked_body_views,
                    self.mask_selected_reset,
                    self.mask_selected_reset_full_refresh,
                )
            ):
                raise ValueError("unsupported tensor lifecycle must remain fail closed")
            if self.state_fields:
                raise ValueError("unsupported tensor lifecycle must not declare state fields")
            if self.packed_reset_fields:
                raise ValueError(
                    "unsupported tensor lifecycle must not declare packed reset fields"
                )
            if self.packed_reset_mocap:
                raise ValueError("unsupported tensor lifecycle must not declare packed reset mocap")
            if self.stream_event_ownership is not None:
                raise ValueError(
                    "unsupported tensor lifecycle must not declare stream/event ownership"
                )
            if self.torch_devices:
                raise ValueError("unsupported tensor lifecycle must not declare Torch devices")
        else:
            valid_topology = (
                (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.process_topology is TensorProcessTopology.IN_PROCESS
                    and self.data_plane is TensorDataPlane.DIRECT
                )
                or (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.process_topology is TensorProcessTopology.EXTERNAL_WORKER
                    and self.data_plane is TensorDataPlane.CUDA_IPC
                )
                or (
                    self.execution is TensorExecution.HOST_BRIDGE
                    and self.process_topology is TensorProcessTopology.IN_PROCESS
                    and self.data_plane is TensorDataPlane.HOST_BRIDGE
                )
                or (
                    self.execution is TensorExecution.HOST_BRIDGE
                    and self.process_topology is TensorProcessTopology.EXTERNAL_WORKER
                    and self.data_plane is TensorDataPlane.HOST_SHARED_MEMORY
                )
            )
            if not valid_topology:
                raise ValueError(
                    "invalid tensor process/data-plane combination: "
                    f"{self.execution.value} requires either in-process direct/bridge storage "
                    "or a matching external-worker IPC plane"
                )
        if not self.stream_event_ownership and self.execution is not TensorExecution.UNSUPPORTED:
            raise ValueError("supported tensor lifecycle must declare stream/event ownership")
        if not self.torch_devices and self.execution is not TensorExecution.UNSUPPORTED:
            raise ValueError("supported tensor lifecycle must declare supported Torch devices")
        if self.state_views and not self.state_fields:
            raise ValueError("tensor state views require at least one declared state field")
        if self.generalized_force_injection and not self.stepping:
            raise ValueError("tensor generalized-force injection requires tensor stepping")
        if self.selected_reset_publication is not None and not self.selected_reset:
            raise ValueError("selected-reset publication requires selected reset")
        if (
            self.requires_post_construction_publication_barrier
            and self.execution is TensorExecution.UNSUPPORTED
        ):
            raise ValueError(
                "unsupported tensor lifecycle cannot require a post-construction barrier"
            )
        if self.mask_selected_reset and not (
            self.selected_reset
            and self.selected_reset_publication is SelectedResetPublication.AUTHORITATIVE_VIEWS
            and {"qpos", "qvel"}.issubset(self.state_fields)
            and (
                (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.data_plane is TensorDataPlane.DIRECT
                )
                or (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.process_topology is TensorProcessTopology.EXTERNAL_WORKER
                    and self.data_plane is TensorDataPlane.CUDA_IPC
                )
                or (
                    self.execution is TensorExecution.HOST_BRIDGE
                    and self.process_topology is TensorProcessTopology.IN_PROCESS
                    and self.data_plane is TensorDataPlane.HOST_BRIDGE
                )
            )
        ):
            raise ValueError(
                "mask-selected reset requires authoritative direct device-resident or "
                "in-process host-bridge reset"
            )
        if self.mask_selected_reset_full_refresh and not self.mask_selected_reset:
            raise ValueError("mask-selected full refresh requires mask-selected reset")
        if self.packed_reset_mocap and not self.packed_reset:
            raise ValueError("packed reset mocap requires packed reset")
        if self.packed_reset and not self.device_reset_randomization:
            raise ValueError("packed reset requires device reset randomization")
        if self.packed_reset and not self.packed_reset_fields:
            raise ValueError("packed reset requires at least one model field")
        if self.packed_reset_fields and not self.packed_reset:
            raise ValueError("packed reset fields require packed reset")
        if self.selected_reset and not {"qpos", "qvel"}.issubset(self.state_fields):
            raise ValueError("tensor selected reset requires qpos and qvel state fields")
        if self.reset_randomization and not self.selected_reset:
            raise ValueError("tensor reset randomization requires selected reset")
        if self.device_reset_randomization and not (
            self.reset_randomization
            and (
                (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.data_plane is TensorDataPlane.DIRECT
                )
                or (
                    self.execution is TensorExecution.DEVICE_RESIDENT
                    and self.process_topology is TensorProcessTopology.EXTERNAL_WORKER
                    and self.data_plane is TensorDataPlane.CUDA_IPC
                )
                or (
                    self.execution is TensorExecution.HOST_BRIDGE
                    and self.data_plane is TensorDataPlane.HOST_BRIDGE
                )
            )
        ):
            raise ValueError(
                "tensor device reset randomization requires reset randomization with either "
                "device-resident direct storage or an in-process host bridge"
            )
        if self.tracked_body_views and not (self.sensor_views and self.selected_reset):
            raise ValueError("tracked-body views require sensor views and selected reset")
        if self.packed_host_bridge and not (
            self.execution is TensorExecution.HOST_BRIDGE
            and self.process_topology is TensorProcessTopology.IN_PROCESS
            and self.data_plane is TensorDataPlane.HOST_BRIDGE
        ):
            raise ValueError(
                "packed host bridge requires in-process HOST_BRIDGE with a HOST_BRIDGE data plane"
            )
        for device in self.torch_devices:
            _tensor_device_parts(device, context="declared")
        if len(set(self.torch_devices)) != len(self.torch_devices):
            raise ValueError("tensor Torch devices must be unique")


@dataclass(frozen=True)
class TensorRuntimeDiagnostic:
    """Machine-readable state of one optional tensor-runtime optimization.

    ``disable_reason`` is always present when an optimization is disabled. It
    records either operator-selected disablement (for example, ``"not
    requested"``) or the backend-owned fallback reason. Adapters expose this
    cold-path metadata through the public backend contract; callers must not
    inspect private implementation fields.
    """

    requested: bool
    enabled: bool
    disable_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.requested, bool):
            raise TypeError(f"requested must be bool, got {type(self.requested).__name__}")
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if self.enabled and not self.requested:
            raise ValueError("an enabled runtime diagnostic must have been requested")
        if self.enabled and self.disable_reason is not None:
            raise ValueError("an enabled runtime diagnostic must not declare a disable reason")
        if not self.enabled:
            if not isinstance(self.disable_reason, str) or not self.disable_reason.strip():
                raise ValueError("a disabled runtime diagnostic must declare a disable reason")


@dataclass(frozen=True)
class CompletionToken:
    """Public receipt for one asynchronous device-resident step operation.

    Receipts exist only for event-backed asynchronous ``step`` work (for
    example an mjwarp CUDA event). Reset operations and synchronous host-bridge
    producers emit no receipt at all. A producer creates the token only after
    enqueueing all step outputs and Torch publications on its execution stream;
    a consumer gates reads through :meth:`wait_before_read` and never inspects
    backend stream or event objects. This is a backend-owned value object, not
    a CUDA event handle.
    """

    contract_version: int
    operation: str
    sequence: int
    producer_id: str

    def __post_init__(self) -> None:
        if self.contract_version != COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"unsupported completion contract {self.contract_version}")
        if self.operation not in COMPLETION_OPERATIONS:
            raise ValueError(f"unsupported completion operation {self.operation!r}")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise TypeError("completion sequence must be a non-negative integer")
        if not isinstance(self.producer_id, str) or not self.producer_id.strip():
            raise TypeError("completion producer_id must be a non-empty string")

    def wait_before_read(self, stream: Any | None = None) -> None:
        """Fail closed: metadata receipts do not own an event dependency.

        A concrete backend returns a backend-owned token carrying an event or
        uses a backend implementation of the protocol. The public base token
        intentionally has no CUDA event to import or leak; callers cannot turn
        receipt metadata into implicit synchronization.
        """
        raise NotImplementedError("this completion receipt does not implement a read gate")


@dataclass(frozen=True)
class PublicStateWidths:
    """Canonical qpos/qvel widths used by tensor reset composition."""

    nq: int
    nv: int

    def __post_init__(self) -> None:
        for name, value in (("nq", self.nq), ("nv", self.nv)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"PublicStateWidths {name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"PublicStateWidths {name} must be positive")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class SensorDescriptor:
    """One public named sensor and its flattened per-row width."""

    name: str
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SensorDescriptor name must be a non-empty string")
        if isinstance(self.width, bool) or not isinstance(self.width, (int, np.integer)):
            raise TypeError("SensorDescriptor width must be an integer")
        if int(self.width) <= 0:
            raise ValueError("SensorDescriptor width must be positive")
        object.__setattr__(self, "width", int(self.width))


@dataclass(frozen=True)
class TrackedBodyStateViews:
    """One public tracked-body read ordered by the caller's request.

    The four fields are backend-owned tensors (or arrays) with leading axes
    ``(num_envs, num_bodies)``.  Device-resident adapters return live or stable
    views; host-bridge adapters return copied views.  The contract intentionally
    does not expose sensor offsets or backend body ids.
    """

    body_names: tuple[str, ...]
    pos_w: Any
    quat_w: Any
    lin_vel_w: Any
    ang_vel_w: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.body_names, (str, bytes))
            or not isinstance(self.body_names, Sequence)
            or not self.body_names
        ):
            raise TypeError("TrackedBodyStateViews body_names must be a non-empty sequence")
        names = tuple(self.body_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("TrackedBodyStateViews body names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"TrackedBodyStateViews body names must be unique: {names}")
        object.__setattr__(self, "body_names", names)


@dataclass(frozen=True)
class TensorIOSpec:
    """Cold-path request for one persistent host-bridge I/O layout.

    ``device`` is intentionally opaque: concrete adapters own the tensor runtime
    and reject devices outside their declared execution profile.
    """

    state_fields: tuple[str, ...]
    sensor_names: tuple[str, ...] = ()
    device: Any | None = None

    def __post_init__(self) -> None:
        if not self.state_fields and not self.sensor_names:
            raise ValueError("tensor I/O request must contain state fields or sensors")
        if len(set(self.state_fields)) != len(self.state_fields):
            raise ValueError("tensor I/O state fields must be unique")
        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("tensor I/O sensor names must be unique")


class HostBridgeTransferPlan(abc.ABC):
    """Public, backend-owned execution plan for explicit host transfers.

    Implementations preallocate staging and destination buffers and expose the
    four semantic boundaries separately: control D2H, physics, state/sensor
    H2D, reset D2H, and (after reset) selected state/sensor H2D. They never
    imply device-resident physics.
    """

    last_timing: dict[str, float]

    @property
    @abc.abstractmethod
    def spec(self) -> TensorIOSpec:
        """Return the immutable layout request used to compile this plan."""

    @property
    @abc.abstractmethod
    def transfer_stats(self) -> dict[str, int]:
        """Return cumulative semantic transfer and synchronization counters."""

    @abc.abstractmethod
    def write_control(self, ctrl: Any) -> None:
        """Stage one complete control tensor on the host."""

    @abc.abstractmethod
    def step(self, nsteps: int = 1) -> dict | None:
        """Run CPU physics with the tensor control staged by ``write_control``."""

    @abc.abstractmethod
    def read_state_sensors(self) -> Mapping[str, Any]:
        """Return persistent tensor views after one packed H2D read."""

    @abc.abstractmethod
    def apply_reset(
        self,
        env_indices: Any,
        qpos: Any,
        qvel: Any,
        randomization: Any | None = None,
    ) -> dict | None:
        """Pack selected reset rows once, D2H them, and commit CPU physics."""

    @abc.abstractmethod
    def read_selected_state_sensors(self) -> Mapping[str, Any]:
        """Return full views after one packed selected-row post-reset H2D."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release transfer staging ownership without closing CPU physics."""


DEFAULT_DEBUG_RGBA = (1.0, 0.2, 0.2, 0.5)

# Expected ``size`` arity per primitive kind; ``ghost_geom`` also accepts an
# empty size (uniform scale defaults to 1.0) and ``text`` carries no size.
_DEBUG_PRIMITIVE_SIZE_ARITY: dict[str, frozenset[int]] = {
    "sphere": frozenset({1}),
    "box": frozenset({3}),
    "frame": frozenset({1}),
    "arrow": frozenset({1}),
    "ghost_geom": frozenset({0, 1}),
    "text": frozenset({0}),
}


@dataclass(frozen=True)
class DebugPrimitive:
    """One task-owned debug primitive overlaid on playback rendering.

    ``pos`` (and the orientation implied by ``quat``) is expressed in the
    environment-local frame; grid offsets are applied by the renderer when
    multiple envs are composed into one frame.  ``size`` semantics depend on
    ``kind``: sphere=(radius,), box=(half_x, half_y, half_z), frame=(axis_length,)
    drawing RGB xyz triads, arrow=(length,) pointing along the local +z axis,
    ghost_geom=(uniform_scale,) defaulting to 1.0, text=().  ``mesh_asset``
    (ghost_geom only) names either a mesh registered in the playback model or
    a mesh asset file (``.obj``/``.stl``) injected into the render model.
    ``text`` (text kind only) is the label anchored at ``pos``; the MuJoCo
    off-screen scene has no text channel, so text primitives are currently
    skipped by the offline renderer (documented no-op, not an error).
    """

    kind: DebugPrimitiveKind
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    size: tuple[float, ...] = ()
    rgba: tuple[float, float, float, float] = DEFAULT_DEBUG_RGBA
    mesh_asset: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DEBUG_PRIMITIVE_KINDS:
            allowed = ", ".join(sorted(DEBUG_PRIMITIVE_KINDS))
            raise ValueError(f"DebugPrimitive kind must be one of: {allowed}; got {self.kind!r}")
        object.__setattr__(self, "pos", _as_float_tuple(self.pos, 3, "DebugPrimitive pos"))
        if self.quat is not None:
            quat = _as_float_tuple(self.quat, 4, "DebugPrimitive quat")
            norm = math.sqrt(sum(component * component for component in quat))
            if not math.isfinite(norm) or norm < 1e-6:
                raise ValueError("DebugPrimitive quat must be a non-zero wxyz quaternion")
            if abs(norm - 1.0) > 1e-3:
                raise ValueError(f"DebugPrimitive quat must be unit-length wxyz (norm {norm:.6f})")
            object.__setattr__(self, "quat", quat)
        size = _as_float_tuple(self.size, None, "DebugPrimitive size")
        if len(size) not in _DEBUG_PRIMITIVE_SIZE_ARITY[self.kind]:
            raise ValueError(
                f"DebugPrimitive kind {self.kind!r} expects size arity "
                f"{sorted(_DEBUG_PRIMITIVE_SIZE_ARITY[self.kind])}; got {len(size)}"
            )
        if any(component <= 0 for component in size):
            raise ValueError("DebugPrimitive size components must be positive")
        object.__setattr__(self, "size", size)
        rgba = _as_float_tuple(self.rgba, 4, "DebugPrimitive rgba")
        if any(component < 0.0 or component > 1.0 for component in rgba):
            raise ValueError("DebugPrimitive rgba components must lie in [0, 1]")
        object.__setattr__(self, "rgba", rgba)
        if self.kind == "ghost_geom":
            if not isinstance(self.mesh_asset, str) or not self.mesh_asset:
                raise ValueError("DebugPrimitive ghost_geom requires a non-empty mesh_asset")
        elif self.mesh_asset is not None:
            raise ValueError("DebugPrimitive mesh_asset is only valid for kind 'ghost_geom'")
        if self.kind == "text":
            if not isinstance(self.text, str):
                raise ValueError("DebugPrimitive text kind requires a text string")
        elif self.text is not None:
            raise ValueError("DebugPrimitive text is only valid for kind 'text'")


def _as_float_tuple(values: Any, arity: int | None, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a numeric sequence")
    if arity is not None and len(values) != arity:
        raise ValueError(f"{label} must have {arity} components; got {len(values)}")
    out = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{label} components must be finite")
    return out


DebugOverlayGetter = Callable[[], "Sequence[Sequence[DebugPrimitive] | None] | None"]


def validate_debug_overlays(
    overlays: Sequence[Sequence[DebugPrimitive] | None] | None,
    num_envs: int,
) -> Sequence[Sequence[DebugPrimitive] | None] | None:
    """Validate one frame of debug overlay primitives against the env batch.

    The outer sequence is indexed by environment and must have length
    ``num_envs``; each entry is the env's primitive sequence (``None`` or
    empty marks an env without overlay).  ``None`` disables overlays for the
    whole frame.  Returns the input unchanged so callers can chain it.
    """
    if overlays is None:
        return None
    if isinstance(overlays, (str, bytes)) or not isinstance(overlays, Sequence):
        raise TypeError(
            "debug overlays must be a per-env sequence of DebugPrimitive sequences or None"
        )
    if len(overlays) != num_envs:
        raise ValueError(
            f"debug overlays must have one entry per env (len == {num_envs}); got {len(overlays)}"
        )
    for env_idx, env_primitives in enumerate(overlays):
        if env_primitives is None:
            continue
        if isinstance(env_primitives, (str, bytes)) or not isinstance(env_primitives, Sequence):
            raise TypeError(f"debug overlays[{env_idx}] must be a sequence of DebugPrimitive")
        for primitive in env_primitives:
            if not isinstance(primitive, DebugPrimitive):
                raise TypeError(
                    f"debug overlays[{env_idx}] entries must be DebugPrimitive; "
                    f"got {type(primitive).__name__}"
                )
    return overlays


_CAMERA_CFG_FIELDS = frozenset(
    {
        "cam_distance",
        "cam_elevation",
        "cam_azimuth",
        "cam_lookat",
        "cam_tracking",
        "cam_tracking_env_idx",
        "cam_tracking_extra_envs",
        "cam_fov",
    }
)


@dataclass(frozen=True)
class CameraCfg:
    """Typed playback/renderer camera configuration.

    Angles are degrees in MuJoCo's free-camera convention (negative elevation
    looks down from above).  ``cam_lookat`` pins the free-camera target;
    ``cam_tracking`` follows one env's root body, showing up to
    ``cam_tracking_extra_envs`` nearest neighbours.  ``cam_fov`` is the
    vertical field of view in degrees where the renderer supports it.
    """

    cam_distance: float = 2.0
    cam_elevation: float = -20.0
    cam_azimuth: float = 90.0
    cam_lookat: tuple[float, float, float] | None = None
    cam_tracking: bool = False
    cam_tracking_env_idx: int = 0
    cam_tracking_extra_envs: int = 2
    cam_fov: float | None = None

    def __post_init__(self) -> None:
        distance = float(self.cam_distance)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"cam_distance must be positive and finite; got {self.cam_distance!r}")
        object.__setattr__(self, "cam_distance", distance)
        elevation = float(self.cam_elevation)
        if not math.isfinite(elevation) or not -90.0 <= elevation <= 90.0:
            raise ValueError(
                f"cam_elevation must lie in [-90, 90] degrees; got {self.cam_elevation!r}"
            )
        object.__setattr__(self, "cam_elevation", elevation)
        azimuth = float(self.cam_azimuth)
        if not math.isfinite(azimuth):
            raise ValueError(f"cam_azimuth must be finite; got {self.cam_azimuth!r}")
        object.__setattr__(self, "cam_azimuth", azimuth)
        if self.cam_lookat is not None:
            lookat = _as_float_tuple(self.cam_lookat, 3, "CameraCfg cam_lookat")
            object.__setattr__(self, "cam_lookat", lookat)
        object.__setattr__(self, "cam_tracking", bool(self.cam_tracking))
        for name in ("cam_tracking_env_idx", "cam_tracking_extra_envs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer; got {value!r}")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative; got {value}")
            object.__setattr__(self, name, int(value))
        if self.cam_fov is not None:
            fov = float(self.cam_fov)
            if not math.isfinite(fov) or not 0.0 < fov < 180.0:
                raise ValueError(f"cam_fov must lie in (0, 180) degrees; got {self.cam_fov!r}")
            object.__setattr__(self, "cam_fov", fov)

    @classmethod
    def from_kwargs(cls, kwargs: "CameraCfg | Mapping[str, Any] | None") -> "CameraCfg":
        """Normalize boundary ``camera_kwargs`` input into a ``CameraCfg``.

        ``None`` yields defaults and an existing ``CameraCfg`` passes through.
        Mapping keys must be a subset of the declared fields; unknown keys
        (including the historical ``distance``/``elevation_deg`` aliases) fail
        closed with an error naming them instead of being silently ignored.
        """
        if kwargs is None:
            return cls()
        if isinstance(kwargs, cls):
            return kwargs
        if not isinstance(kwargs, Mapping):
            raise TypeError(
                f"camera_kwargs must be a CameraCfg, a mapping, or None; "
                f"got {type(kwargs).__name__}"
            )
        unknown = sorted(set(kwargs) - _CAMERA_CFG_FIELDS)
        if unknown:
            allowed = ", ".join(sorted(_CAMERA_CFG_FIELDS))
            raise ValueError(f"unknown camera_kwargs key(s): {unknown}; supported keys: {allowed}")
        return cls(**dict(kwargs))


def unsupported_debug_overlay_error(owner: str) -> NotImplementedError:
    """Build the fail-closed error for backends without debug overlay support."""
    return NotImplementedError(
        f"{owner} does not support debug overlay primitives "
        "(get_play_capabilities().supports_debug_overlay is False); omit "
        "debug_overlay_getter or use a backend on the MuJoCo offline snapshot pipeline"
    )


@dataclass(frozen=True)
class MocapPoseBinding:
    """Cold-bound world-space pose access for one fixed mocap body.

    Poses use xyz + unit wxyz quaternion. Writes affect only selected worlds,
    preserve generalized state and refresh derived state before returning.
    A subsequent selected tensor state reset restores that world's mocap defaults;
    reset owners therefore commit generalized state before their mocap writes.
    """

    backend_type: str
    body_name: str
    num_envs: int
    default_pose: np.ndarray
    _reader: Callable[[], np.ndarray] = field(repr=False, compare=False)
    _writer: Callable[[np.ndarray, np.ndarray], None] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend_type, str) or not self.backend_type:
            raise ValueError("mocap binding backend_type must be a non-empty string")
        if not isinstance(self.body_name, str) or not self.body_name:
            raise ValueError("mocap binding body_name must be a non-empty string")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TypeError("mocap binding num_envs must be an integer")
        if self.num_envs <= 0:
            raise ValueError("mocap binding num_envs must be positive")
        if not callable(self._reader) or not callable(self._writer):
            raise TypeError("mocap binding reader and writer must be callable")
        default = np.array(self.default_pose, copy=True)
        self._validate_poses(default[None, :] if default.ndim == 1 else default, 1)
        if default.shape != (7,):
            raise ValueError("mocap default_pose must have shape (7,)")
        default.setflags(write=False)
        object.__setattr__(self, "default_pose", default)

    @staticmethod
    def _validate_poses(poses: np.ndarray, count: int) -> None:
        if not isinstance(poses, np.ndarray) or not np.issubdtype(poses.dtype, np.floating):
            raise TypeError("mocap poses must be a floating NumPy array")
        if poses.shape != (count, 7):
            raise ValueError(f"mocap poses must have shape {(count, 7)}, got {poses.shape}")
        if not np.isfinite(poses).all():
            raise ValueError("mocap poses must be finite")
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("mocap poses require unit wxyz quaternions")

    def read(self) -> np.ndarray:
        """Return a detached (num_envs, 7) pose snapshot."""
        poses = self._reader()
        self._validate_poses(poses, self.num_envs)
        return poses.copy()

    def write(self, env_ids: np.ndarray, poses: np.ndarray) -> None:
        """Write selected rows without changing any other world's state."""
        if not isinstance(env_ids, np.ndarray) or not np.issubdtype(env_ids.dtype, np.integer):
            raise TypeError("mocap env_ids must be an integer NumPy array")
        if env_ids.ndim != 1 or np.unique(env_ids).size != env_ids.size:
            raise ValueError("mocap env_ids must be one-dimensional and unique")
        if np.any(env_ids < 0) or np.any(env_ids >= self.num_envs):
            raise IndexError("mocap env_ids are outside the backend batch")
        self._validate_poses(poses, env_ids.size)
        if env_ids.size:
            self._writer(env_ids, poses)


class RenderClosedError(RuntimeError):
    """Interface-level signal that the user closed the backend render window.

    Backends with a native renderer translate their private window-closed
    errors into this type at the interface boundary (``render`` /
    ``capture_video_frame``), so play loops can catch it by type instead of
    matching backend-private exception names.
    """


@dataclass(frozen=True)
class TerrainSpawnData:
    """Read-only terrain spawn metadata materialized by a backend.

    ``terrain_origins`` is a detached snapshot with shape
    ``(num_rows, num_cols, 3)``. ``sample_height`` samples world-space XY
    coordinates and returns an array with the same leading shape.
    """

    terrain_origins: np.ndarray
    sample_height: TerrainHeightSampleFn | None = None

    def __post_init__(self) -> None:
        origins = np.array(self.terrain_origins, copy=True)
        if origins.ndim != 3 or origins.shape[2] != 3:
            raise ValueError(
                f"terrain_origins must have shape (num_rows, num_cols, 3); got {origins.shape}"
            )
        origins.setflags(write=False)
        object.__setattr__(self, "terrain_origins", origins)
        if self.sample_height is not None and not callable(self.sample_height):
            raise TypeError("sample_height must be callable")


@dataclass(frozen=True)
class RootStateLayout:
    """Generalized-state columns for one floating root body.

    ``qpos_indices`` address ``[x, y, z, qw, qx, qy, qz]`` in the public
    :meth:`SimBackend.set_state_tensor` qpos representation. ``qvel_indices`` address
    ``[linear_velocity_world, angular_velocity_body]``.  Manager-facing root
    states use world-frame angular velocity, so the base-owned reset
    transaction performs the frame conversion before calling ``set_state_tensor``.
    """

    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values, expected in (
            ("qpos_indices", self.qpos_indices, 7),
            ("qvel_indices", self.qvel_indices, 6),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"RootStateLayout {name} must be a tuple")
            if len(values) != expected:
                raise ValueError(
                    f"RootStateLayout {name} must contain {expected} columns; "
                    f"got {len(values)}"
                )
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError(f"RootStateLayout {name} must contain integer columns")
            normalized = tuple(int(value) for value in values)
            if any(value < 0 for value in normalized):
                raise ValueError(f"RootStateLayout {name} cannot contain negative columns")
            if len(set(normalized)) != expected:
                raise ValueError(f"RootStateLayout {name} must contain unique columns")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class SensorView:
    """Validated batch view over one or more named backend sensors.

    Sensor names and flattened per-sensor widths are resolved while the backend
    is materialized.  Manager terms retain this view and only call ``read`` on
    the hot path; they never inspect backend model objects or resolve XML names.
    The reader is intentionally backend-owned so adapters can use cached
    numeric slots, stable host slices, or an opaque native batch reader without
    changing the manager-facing contract.
    """

    backend_type: str
    names: tuple[str, ...]
    dimensions: tuple[int, ...]
    num_envs: int
    _reader: SensorReadFn = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.backend_type, str) or not self.backend_type:
            raise ValueError("SensorView backend_type must be a non-empty string")
        if not isinstance(self.names, tuple) or not self.names:
            raise ValueError("SensorView names must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("SensorView names must contain non-empty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"SensorView names must be unique: {self.names}")
        if not isinstance(self.dimensions, tuple) or len(self.dimensions) != len(self.names):
            raise ValueError("SensorView dimensions must contain one entry per sensor name")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in self.dimensions
        ):
            raise ValueError("SensorView dimensions must be positive integers")
        if (
            isinstance(self.num_envs, (bool, np.bool_))
            or not isinstance(self.num_envs, (int, np.integer))
            or int(self.num_envs) <= 0
        ):
            raise ValueError("SensorView num_envs must be a positive integer")
        if not callable(self._reader):
            raise TypeError("SensorView reader must be callable")
        object.__setattr__(self, "dimensions", tuple(int(value) for value in self.dimensions))
        object.__setattr__(self, "num_envs", int(self.num_envs))

    @property
    def width(self) -> int:
        """Total flattened sensor width in the configured name order."""
        return int(sum(self.dimensions))

    def read(self) -> np.ndarray:
        """Read the current sensor batch and enforce the stable view contract."""
        try:
            value = np.asarray(self._reader())
        except (KeyError, NotImplementedError, ValueError) as exc:
            raise type(exc)(
                f"Backend '{self.backend_type}' sensor view {self.names} could not be read: {exc}"
            ) from exc
        if value.ndim != 2 or value.shape != (self.num_envs, self.width):
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view {self.names} returned shape "
                f"{value.shape}; expected ({self.num_envs}, {self.width})"
            )
        if not np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_):
            raise TypeError(
                f"Backend '{self.backend_type}' sensor view {self.names} returned non-numeric "
                f"dtype {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view {self.names} returned NaN or Inf"
            )
        return value

    @property
    def data(self) -> np.ndarray:
        """Community-style spelling for a current sensor read."""
        return self.read()


@dataclass(frozen=True)
class PlayCapabilities:
    """Backend-native play/render capabilities surfaced through env contracts.

    ``supports_debug_overlay`` covers the offline/record rendering path;
    ``supports_interactive_debug_overlay`` reports whether the interactive
    rendering path can additionally consume ``debug_overlay_getter``.
    ``supports_mocap_playback`` reports whether the backend exposes recorded
    mocap body poses through ``get_playback_mocap_state``.
    """

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False
    supports_debug_overlay: bool = False
    supports_interactive_debug_overlay: bool = False
    supports_mocap_playback: bool = False


@dataclass(frozen=True)
class PhysicsStateParts:
    """One decoded physics-state snapshot, split by :meth:`PhysicsStateLayout.split_state`.

    Leading dimensions match the input snapshot (``(num_envs, ...)`` for a
    batched snapshot, scalars/1-D for a single row). ``mocap_pos`` and
    ``mocap_quat`` are ``None`` when the layout has no mocap bodies.
    """

    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray | None
    mocap_quat: np.ndarray | None


@dataclass(frozen=True)
class PhysicsStateLayout:
    """Contract-level description of the ``get_physics_state`` snapshot layout.

    Snapshot rows use the ``[time, qpos, qvel]`` layout; models with mocap
    bodies append ``[mocap_pos(nmocap*3), mocap_quat(nmocap*4)]`` so offline
    rendering can replay mocap-driven geometry at its recorded pose.  Render
    frontends must split snapshots through :meth:`split_state` instead of
    hardcoding ``1 + nq + nv`` slices.
    """

    nq: int
    nv: int
    nmocap: int = 0

    @property
    def state_width(self) -> int:
        """Total number of columns in one snapshot row."""
        return 1 + self.nq + self.nv + 7 * self.nmocap

    def split_state(self, state: np.ndarray) -> PhysicsStateParts:
        """Split a snapshot (row or batch) into its contract parts.

        Raises:
            ValueError: If the last dimension does not equal ``state_width``.
        """
        array = np.asarray(state)
        if array.ndim < 1 or array.shape[-1] != self.state_width:
            raise ValueError(
                "physics-state snapshot must use the "
                "[time, qpos, qvel, (mocap_pos, mocap_quat)?] layout with last "
                f"dimension {self.state_width}, got shape {array.shape}."
            )
        base = 1 + self.nq + self.nv
        mocap_pos: np.ndarray | None = None
        mocap_quat: np.ndarray | None = None
        if self.nmocap:
            tail = array[..., base:]
            mocap_pos = tail[..., : 3 * self.nmocap].reshape(*array.shape[:-1], self.nmocap, 3)
            mocap_quat = tail[..., 3 * self.nmocap :].reshape(*array.shape[:-1], self.nmocap, 4)
        return PhysicsStateParts(
            time=array[..., 0],
            qpos=array[..., 1 : 1 + self.nq],
            qvel=array[..., 1 + self.nq : base],
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )


_NATIVE_RENDERER_PLAY_CAPABILITIES = PlayCapabilities(
    supports_native_interactive_renderer=True,
    supports_native_video_capture=True,
)
"""Shared play capabilities of backends with a native interactive renderer and video capture."""


class HeightScanner(abc.ABC):
    """Backend-owned height-field scanner created on the env init path."""

    @abc.abstractmethod
    def scan(self) -> np.ndarray:
        """Return sampled values with shape ``(num_envs, num_points)``."""


PLAY_RENDER_MODES = frozenset({"auto", "interactive", "record", "none"})


@dataclass(frozen=True)
class PlayRenderPlan:
    """Backend-resolved playback rendering behavior.

    ``renderer`` names the concrete renderer the backend selected for the
    plan (for example a native viewer versus an offline snapshot pipeline)
    and is purely diagnostic: consumers must not branch on it.
    """

    mode: str
    headless: bool
    record_video: bool
    num_steps: int | None
    output_video: str | PathLike[str] | None
    renderer: str | None = None


def normalize_play_render_mode(play_render_mode: str | None) -> str:
    mode = "auto" if play_render_mode is None else str(play_render_mode).strip().lower()
    if mode not in PLAY_RENDER_MODES:
        joined = ", ".join(sorted(PLAY_RENDER_MODES))
        raise ValueError(f"play render mode must be one of: {joined}; got {mode!r}.")
    return mode


def log_playback_plan(plan: PlayRenderPlan, *, prefix: str = "") -> None:
    """Print user-facing playback status for a resolved backend plan."""
    if plan.mode == "none":
        print(f"{prefix}Skipping playback because training.play_render_mode=none.")
        return
    via = f" via {plan.renderer}" if plan.renderer else ""
    if plan.record_video:
        print(f"{prefix}Rendering video to {plan.output_video}{via}...")
    elif plan.mode == "interactive":
        print(f"{prefix}Starting interactive visualization{via}...")
        print(f"{prefix}Use the renderer window or browser URL reported by the backend.")
    else:
        print(f"{prefix}Running playback without video recording...")
    print(f"{prefix}Rendering playback frames...")


class SimBackend(abc.ABC):
    """Unified simulation backend contract."""

    _pre_step_control_fn: PreStepControlFn | None = None
    _pre_step_control_active: bool = False
    _scene_cleanup_handle: Any | None
    _play_capabilities = PlayCapabilities()
    backend_type: str

    def get_scene_layout(self) -> CompiledSceneLayout:
        """Return the immutable materialized public addresses, never native handles."""
        raise NotImplementedError(f"{self.backend_type} does not expose a scene layout")

    def get_entity_names(self) -> tuple[str, ...]:
        """Return materialized physical entity names in the frozen public order."""
        raise NotImplementedError(f"{self.backend_type} does not expose physical entities")

    def get_entity_state(self, entity: str) -> Mapping[str, np.ndarray]:
        """Return detached root_pose/root_velocity/joint_positions/joint_velocities.

        Root fields follow EntityStatePatch's link-origin/world-frame contract.
        Non-root joint fields use the frozen entity joint order and native
        generalized widths. Unavailable fields must fail, never return old data
        as if current. State freshness follows the adapter's declared profile.
        """
        raise NotImplementedError(f"{self.backend_type} does not expose entity state")

    def get_entity_default_state(
        self, entity: str, env_ids: Sequence[int] | np.ndarray | None = None
    ) -> Mapping[str, np.ndarray]:
        """Detached construction/keyframe defaults in selected environment order.

        Fields and frames match get_entity_state. Defaults follow immutable
        variant assignment and never reflect current reset-time randomization.
        This query does not reset, step or reparse the scene.
        """
        raise NotImplementedError(f"{self.backend_type} does not expose entity state defaults")

    def reset_entities(self, request: SceneResetRequest) -> None:
        """Validate the complete selected-entity request, then commit it once.

        Validation failure leaves all state unchanged. A native partial commit
        failure must fault the backend unless the adapter can actually roll back.
        Unselected entities/environments and fixed identity remain unchanged.
        """
        raise NotImplementedError(f"{self.backend_type} does not support entity reset")

    def get_import_report(self) -> "ImportReport":
        """Return a detached construction/materialization configuration snapshot.

        This cached report never parses assets, starts workers, or reflects
        reset-time randomization. Use current-property queries for live values.
        """
        report = getattr(self, "_import_report", None)
        if report is None:
            report = ImportReport.unknown(getattr(self, "backend_type", type(self).__name__))
            self._import_report = report
        return report

    @property
    def capability_flags(self):
        """Coarse capability labels for clients that need a cheap feature check.

        The detailed contract is expressed by the methods on this class.  The
        labels remain useful for benchmark/conformance metadata and are derived
        from the mandatory lifecycle methods rather than maintained separately
        by every adapter.
        """
        from unisim.errors import BackendCapability

        return frozenset(
            {
                BackendCapability.RESET,
                BackendCapability.SELECTED_RESET,
                BackendCapability.STATE_READ,
                BackendCapability.STATE_WRITE,
            }
        )

    def get_tensor_execution(self) -> TensorExecution:
        """Declare the optional tensor lifecycle without discovering SDKs."""
        return TensorExecution.UNSUPPORTED

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        """Return fail-closed tensor methods and negotiable state fields."""
        return TensorLifecycleCapabilities(execution=self.get_tensor_execution())

    def get_tensor_runtime_diagnostics(self) -> Mapping[str, TensorRuntimeDiagnostic]:
        """Return runtime-selected diagnostics for optional tensor optimizations.

        Unlike capability negotiation, these values describe actual cold-path
        initialization results and may change when a backend falls back or
        closes. The default empty mapping means the backend declares no
        optional tensor-runtime optimization.
        """
        return {}

    def get_public_state_widths(self) -> PublicStateWidths:
        """Return canonical qpos/qvel widths for packed tensor reset layout."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose public tensor state widths"
        )

    def bind_host_bridge_io(self, spec: TensorIOSpec) -> HostBridgeTransferPlan:
        """Compile a persistent transfer plan for a declared host bridge."""
        raise NotImplementedError(
            f"{self.backend_type} does not support packed host-bridge tensor I/O: "
            f"{self.get_tensor_execution()}"
        )

    def bind_packed_reset(self, *, device: Any | None = None) -> Any:
        """Begin a backend-owned packed reset transaction."""
        raise NotImplementedError(
            f"{self.backend_type} does not support packed reset transactions: "
            f"{self.get_tensor_execution()}"
        )

    def set_mocap_pose_tensor(self, body_name: str, env_indices: Any, poses: Any) -> None:
        """Write selected mocap poses through the declared tensor lifecycle."""
        raise NotImplementedError(
            f"{self.backend_type} does not support tensor mocap pose writes: "
            f"{self.get_tensor_execution()}"
        )

    def get_state_views(
        self, fields: tuple[str, ...] | str | None = None, device: Any | None = None
    ) -> Mapping[str, Any]:
        """Return backend-owned state through the declared tensor lifecycle.

        Tensor views are Torch tensors. Returned values are logically read-only
        and mutation is undefined.

        ``DEVICE_RESIDENT`` adapters return stable live views on the backend's
        exact accelerator device; ``device=None`` selects that device and any
        other device is rejected. ``HOST_BRIDGE`` adapters return explicit
        copies from authoritative host state; ``device=None`` selects host CPU
        and callers pass an explicit accelerator device for H2D copies. Unlike
        the removed 1.x NumPy snapshot API, successful tensor adapters do not
        return NumPy arrays.
        Consume the declared tensor execution mode rather than probing the array
        implementation.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support backend state views: "
            f"{self.get_tensor_execution()}"
        )

    def get_sensor_view(self, name: str, device: Any | None = None) -> Any:
        """Return one named sensor view on the declared tensor lifecycle."""
        raise NotImplementedError(
            f"{self.backend_type} does not support sensor views: {self.get_tensor_execution()}"
        )

    def get_tracked_body_views(
        self,
        body_names: Sequence[str] | None = None,
        device: Any | None = None,
    ) -> TrackedBodyStateViews:
        """Return all tracked-body fields in one public backend read.

        ``body_names`` selects and orders the returned body axis.  ``None``
        requests every tracked body in backend insertion order.  This aggregate
        contract avoids one Python projection boundary per body/field sensor.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support tracked-body views: "
            f"{self.get_tensor_execution()}"
        )

    def step_tensor(self, ctrl: Any, nsteps: int = 1) -> dict | None:
        """Advance physics from a backend-declared accelerator control tensor.

        ``ctrl`` is a contiguous float32 Torch tensor with shape
        ``(num_envs, num_actuators)``, lives on the adapter-required device, and
        must be finite. The method consumes it before synchronizing and returning.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support tensor stepping: {self.get_tensor_execution()}"
        )

    def set_generalized_force_tensor(
        self,
        *,
        joint_f: Any | None = None,
        body_f: Any | None = None,
    ) -> None:
        """Inject persistent generalized forces consumed at every physics substep.

        Declaring adapters (``TensorLifecycleCapabilities.generalized_force_injection``)
        accept contiguous float32 Torch tensors on the adapter-required device:

        ``joint_f`` has shape ``(num_envs, nv)`` and carries generalized forces in
        public qvel column order, matching the engine's ``Control.joint_f``
        semantics: the solver consumes the value at every substep until it is
        overwritten. For free or distance roots, the six root columns are the
        world-frame wrench ``(f_x, f_y, f_z, t_x, t_y, t_z)`` applied at the
        child body's center of mass.

        ``body_f`` has shape ``(num_envs, num_bodies, 6)`` and carries one
        world-frame spatial wrench per body in public body order, matching the
        engine's ``State.body_f`` semantics: the first three components are the
        linear force applied at the body's center of mass and the last three
        are the torque. Engines clear ``State.body_f`` at every substep (the
        adapter calls ``state.clear_forces()`` before each solver step, which
        resets ``body_f`` and never touches ``Control.joint_f``), so the
        adapter owns a persistent device buffer and re-applies the injected
        wrench after every internal clear: one write is consumed by every
        substep of ``step_tensor(nsteps=k)`` and by later steps until
        overwritten.

        Both channels are control inputs, not state: tensor and host resets do
        not clear them, and they compose with actuator controls (they are not
        consumed or altered by ``step_tensor`` control uploads). A ``None``
        channel is left unchanged; write an all-zeros tensor to clear a
        channel. At least one channel must be provided.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support generalized-force injection: "
            f"{self.get_tensor_execution()}"
        )

    def set_state_tensor(
        self,
        env_indices: Any,
        qpos: Any,
        qvel: Any,
        randomization: TensorResetRandomizationPayload | None = None,
    ) -> dict | None:
        """Set selected state through the adapter-declared tensor lifecycle.

        ``env_indices`` is a contiguous one-dimensional int64 Torch tensor with
        unique values in ``[0, num_envs)``; ``qpos`` and ``qvel`` are contiguous
        float32 tensors with shapes ``(len(env_indices), nq)`` and
        ``(len(env_indices), nv)``. All three share one adapter-accepted device
        and must be finite. Adapters may use a bounded synchronization for
        fail-closed row validation. ``DEVICE_RESIDENT`` adapters otherwise avoid
        a host detour; ``HOST_BRIDGE`` adapters make their explicit
        accelerator-to-host boundary measurable before CPU state is updated.
        Inputs are consumed before return.

        ``randomization`` is the unified torch-first
        :class:`TensorResetRandomizationPayload`: every populated field is a
        dense selected-row Torch tensor table. Tensors on ``torch.cpu`` form
        the host plane, checked against
        ``DomainRandomizationCapabilities.supported_reset_terms``; tensors on
        any other device form the device plane, checked against
        ``supported_device_reset_terms`` (empty by default, so undeclared
        device support fails closed naming the term). Non-delta terms hold
        final absolute values applied as selected-row in-place scatters;
        ``base_mass_delta`` and ``base_com_offset`` hold deltas merged with
        the adapter baseline. Host adapters convert tensors at their private
        NumPy boundary through :mod:`unisim.dr._host`.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support tensor state writes: "
            f"{self.get_tensor_execution()}"
        )

    def set_state_tensor_mask(
        self,
        reset_mask: Any,
        qpos: Any,
        qvel: Any,
        randomization: TensorResetRandomizationPayload | None = None,
        *,
        full_refresh: bool = False,
    ) -> dict | None:
        """Set state selected by a fixed-width boolean mask.

        ``reset_mask`` is a contiguous one-dimensional boolean Torch tensor with
        shape ``(num_envs,)``. ``qpos`` and ``qvel`` are full-width contiguous
        float32 tensors with shapes ``(num_envs, nq)`` and ``(num_envs, nv)``.
        All three share one accepted Torch device and contain finite values.
        The adapter changes qpos/qvel/time only for true rows and preserves
        every false row. It publishes authoritative selected state before
        returning; any row materialization required by the engine belongs
        behind this backend boundary.

        ``randomization`` is the unified torch-first
        :class:`TensorResetRandomizationPayload` (full-width tables on the
        mask's plane); the adapter must preserve every false row.
        ``full_refresh=True`` requests an eager equivalent full-batch
        derived-state/publication refresh. It is valid only where
        ``TensorLifecycleCapabilities.mask_selected_reset_full_refresh`` is
        declared.
        """
        raise NotImplementedError(
            f"{self.backend_type} does not support mask-driven tensor state writes: "
            f"{self.get_tensor_execution()}"
        )

    def _host_get_state(
        self, fields: tuple[str, ...] | str | None = None
    ) -> Mapping[str, np.ndarray]:
        """Read generalized state through the documented protected adapter protocol."""
        raise NotImplementedError(f"{self.__class__.__name__} has no host state read")

    def _host_reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset all or selected environments through the protected adapter protocol."""
        raise NotImplementedError(f"{self.__class__.__name__} has no host reset")

    def _host_set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: TensorResetRandomizationPayload | None = None,
    ) -> dict | None:
        """Write selected state through the documented protected adapter protocol.

        ``randomization`` is the unified torch-first
        :class:`TensorResetRandomizationPayload` carrying host-plane
        (``torch.cpu``) tensors; adapters convert fields at their private
        NumPy boundary through :mod:`unisim.dr._host`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} has no host state write")

    def _host_reset_default(self, env_ids: np.ndarray | None = None) -> None:
        """Reset selected environments to construction defaults on the host plane."""
        ids = (
            np.arange(self.num_envs, dtype=np.int32)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int32)
        )
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids must be a one-dimensional in-range index array")
        default_qpos = self.get_default_qpos()
        default_qvel = self.get_init_qvel()
        qpos = np.broadcast_to(default_qpos, (ids.size, default_qpos.size)).copy()
        qvel = np.broadcast_to(default_qvel, (ids.size, default_qvel.size)).copy()
        self._host_set_state(ids, qpos, qvel)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Number of vectorized environments."""

    @property
    @abc.abstractmethod
    def model(self) -> Any:
        """Underlying physics model."""

    # ------------------------------------------------------------------ #
    # Model properties                                                     #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_actuators(self) -> int:
        """Number of actuators."""

    @property
    @abc.abstractmethod
    def num_dof_vel(self) -> int:
        """Number of joint velocity DoFs, excluding the floating base."""

    @abc.abstractmethod
    def get_actuator_ctrl_range(self) -> np.ndarray:
        """Return actuator control ranges.

        Returns:
            Array with shape ``(num_actuators, 2)`` and columns ``[low, high]``.
        """

    def get_actuator_names(self) -> tuple[str, ...]:
        """Return actuator names in control-vector order on the cold path."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose actuator names")

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        """Return each actuator's target single-DoF joint in control-vector order.

        Backends must fail closed when an actuator does not target exactly one
        hinge/slide joint.  Manager action terms use this cold-path metadata to
        map community joint selectors onto the backend control vector without
        inspecting backend-private model objects.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose actuator target joints"
        )

    def get_scene_model_file(self) -> str | None:
        """Return the materialized scene path for diagnostics, when available."""
        return None

    def get_scene_visual_model_file(self) -> str | None:
        """Return the scene visual model file on the cold path, when available.

        Backends without a separate visual scene model return ``None``.
        """
        return None

    def get_terrain_spawn_data(self) -> TerrainSpawnData | None:
        """Return backend-materialized terrain metadata on the cold path.

        Backends without generated terrain support return ``None``. Callers
        should resolve this once during env initialization and cache the
        returned height-sampling callable for reset/reward hot paths.
        """
        return None

    @abc.abstractmethod
    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        """Return the full qpos for a named keyframe, including the floating base.

        Args:
            name: Keyframe name such as ``"stand"`` or ``"home"``.

        Returns:
            Array with shape ``(nq,)``.
        """

    def get_default_qpos(self) -> np.ndarray:
        """Return the backend/model default qpos through a stable contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose default qpos")

    def get_default_dof_pos(self) -> np.ndarray:
        """Return default joint positions in canonical non-root qpos order.

        The returned array is detached, one-dimensional, and excludes floating
        root coordinates.  Backends whose DoF view is actuator-indexed must use
        that same actuator-target order here.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose default DoF positions"
        )

    @abc.abstractmethod
    def get_init_qvel(self) -> np.ndarray:
        """Return a zero-initialized qvel vector compatible with ``set_state_tensor``.

        Returns:
            Zero-filled qvel array.
        """

    def get_root_state_layout(self, root_body_name: str) -> RootStateLayout:
        """Resolve one body's floating-root columns on the cold path.

        Backends must verify that ``root_body_name`` owns a free/floating joint;
        fixed bodies and runtimes without body-to-root metadata fail closed.
        Name/model lookup is forbidden on reset and step hot paths, so callers
        cache either the returned layout or the unsupported result during scene
        materialization.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose root-state layout for "
            f"body {root_body_name!r}"
        )

    @abc.abstractmethod
    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve body/link names to backend integer IDs.

        Args:
            names: Body/link names.

        Returns:
            ``int32`` array with shape ``(len(names),)``.

        Raises:
            ValueError: If any name is not found.
        """

    def get_body_id(self, name: str) -> int:
        """Resolve one body/link name through the backend contract."""
        return int(self.get_body_ids([name])[0])

    def get_geom_id(self, name: str) -> int:
        """Resolve one geom name through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom ids")

    def get_geom_size(self, name: str) -> np.ndarray:
        """Return one geom size vector in backend geom-name order.

        ``get_default_geom_sizes`` returns the complete table of default sizes,
        while this selector resolves one geom by name.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom sizes")

    def get_default_geom_sizes(self) -> np.ndarray:
        """Return default geometry sizes, shape (ngeom, 3)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom size defaults")

    def get_geom_solref(self) -> np.ndarray:
        """Return default contact reference parameters, shape (ngeom, 2)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom solref")

    def get_geom_solimp(self) -> np.ndarray:
        """Return default contact impedance parameters, shape (ngeom, 5)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom solimp")

    def get_dof_damping(self) -> np.ndarray:
        """Return default joint damping, shape (nv,)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof damping")

    def get_dof_frictionloss(self) -> np.ndarray:
        """Return default joint friction loss, shape (nv,)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof frictionloss")

    def bind_mocap_pose(self, body_name: str) -> MocapPoseBinding:
        """Resolve a mocap body once; unavailable capabilities fail at binding."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support mocap pose writes")

    def bind_height_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> HeightScanner:
        """Create a reusable height-field scanner on the init/cold path.

        Backends that support height-field terrain scan must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native height-field scanners"
        )

    def get_body_subtree_ids(self, root_body_id: int) -> np.ndarray:
        """Return body ids in the subtree rooted at ``root_body_id``."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body subtree ids")

    def get_geom_names(self) -> tuple[str, ...]:
        """Return backend geom names in backend id order."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom names")

    def get_geom_body_ids(self) -> np.ndarray:
        """Return the owning body id for each geom."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom body ids")

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-geom contact type and affinity masks."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom contact masks")

    def get_geom_friction(self) -> np.ndarray:
        """Return the backend geom-friction table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom friction")

    def get_gravity(self) -> np.ndarray:
        """Return the backend gravity vector."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose gravity")

    def get_body_mass(self) -> np.ndarray:
        """Return the backend body-mass table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body mass")

    def get_body_ipos(self, env_ids: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        """Return body center-of-mass offsets in each body's local frame, in meters.

        ``body_ipos[b]`` is the position of body ``b``'s center of mass in that
        body's own local coordinate frame; it is neither an inertia tensor nor
        a whole-subtree COM.

        Without ``env_ids`` the canonical model default table is returned with
        shape ``(nbody, 3)`` in backend body-id order.  It never changes with
        reset randomization; fixed-variant backends expose their per-environment
        default baselines through :meth:`get_reset_term_default` instead.

        With ``env_ids`` the current effective per-environment values are
        returned with shape ``(len(env_ids), nbody, 3)``, in ``env_ids`` order.
        The values reflect every reset randomization applied so far, including
        composition with ``base_com_offset``; an environment untouched by a
        partial reset keeps its previous values.  Backends that do not track
        per-environment inertial offsets fail closed with
        ``NotImplementedError`` for this form.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body ipos")

    def _validate_env_ids(self, env_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Coerce and bounds-check per-environment query indices."""
        raw_ids = np.asarray(env_ids)
        if raw_ids.ndim != 1 or (raw_ids.size and raw_ids.dtype.kind not in "iu"):
            raise ValueError("env_ids must be a one-dimensional integer index array")
        if raw_ids.size and (raw_ids.min() < 0 or raw_ids.max() >= self.num_envs):
            raise ValueError(
                f"env_ids entries must lie in [0, {self.num_envs}), got {raw_ids.tolist()}"
            )
        return raw_ids.astype(np.intp)

    def get_dof_armature(self) -> np.ndarray:
        """Return the backend dof-armature table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof armature")

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve body IDs used by motion datasets.

        Motion datasets are generated from MuJoCo, so the returned ids follow
        the MJCF body order with ``worldbody`` as id 0, regardless of the
        backend-native indexing used by ``get_body_ids``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not expose motion body ids")

    def cleanup_scene_assets(self) -> None:
        """Release cold-path scene artifacts owned by the backend."""
        cleanup_handle = getattr(self, "_scene_cleanup_handle", None)
        if cleanup_handle is None:
            return
        cleanup_handle.cleanup()
        self._scene_cleanup_handle = None

    def _reject_named_joint_ranges(self, names: Sequence[str] | None, method_name: str) -> None:
        if names is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support named queries for {method_name}"
            )

    def __del__(self) -> None:
        try:
            self.cleanup_scene_assets()
        except Exception:
            pass

    @abc.abstractmethod
    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        """Return joint position limits, excluding the floating base.

        Args:
            names: Optional joint names in requested return order. Adapters that
                support named scalar joints resolve the model mapping internally.

        Returns:
            Array with shape ``(num_dof, 2)`` and columns ``[low, high]``, or
            ``None`` when the backend does not expose limits.

            When ``names`` is provided by a supporting adapter, the result has
            shape ``(len(names), 2)``. Hinge limits are radians, slide limits
            are meters, and joints without enabled limits return infinities.
        """

    # ------------------------------------------------------------------ #
    # Simulation control                                                   #
    # ------------------------------------------------------------------ #

    def set_pre_step_control(self, fn: PreStepControlFn | None) -> None:
        """Register an env-owned policy-control to physics-control converter.

        The callback receives ``(backend, ctrl)`` so owner code can read the
        backend's freshly-updated sensor contract before every physics substep.
        It must return either backend-native actuator control with the same
        shape, or a :class:`PreStepControlOutput` carrying that control plus an
        optional per-substep body wrench.  A returned wrench is recomposed from
        scratch every substep and cleared when the ``step()`` call finishes.
        Position-actuator envs leave this unset and keep the direct control path.
        """
        self._pre_step_control_fn = fn

    def _reject_wrench_write_inside_pre_step_control(self, operation: str) -> None:
        """Fail closed on staging writes made from inside a substep callback."""
        if self._pre_step_control_active:
            raise RuntimeError(
                f"{operation} must not be called from inside a pre-step control callback; "
                "return a PreStepControlOutput wrench instead so it applies to the current "
                "substep"
            )

    def _convert_pre_step_control(self, ctrl: np.ndarray) -> PreStepControlOutput:
        if self._pre_step_control_fn is None:
            return PreStepControlOutput(ctrl=ctrl)
        result = self._pre_step_control_fn(self, ctrl)
        if isinstance(result, PreStepControlOutput):
            converted = np.asarray(result.ctrl, dtype=ctrl.dtype)
            body_ids = result.body_ids
            force = result.force
            torque = result.torque
        else:
            converted = np.asarray(result, dtype=ctrl.dtype)
            body_ids = None
            force = None
            torque = None
        if converted.shape != ctrl.shape:
            raise ValueError(
                f"pre-step control must return shape {ctrl.shape}, got {converted.shape}"
            )
        if force is None and torque is None:
            if body_ids is not None:
                raise ValueError(
                    "pre-step control wrench requires force and/or torque; body_ids alone "
                    "names no wrench"
                )
            return PreStepControlOutput(ctrl=converted)
        if body_ids is None:
            raise ValueError("pre-step control wrench requires body_ids")
        body_ids_np = np.asarray(body_ids, dtype=np.intp).reshape(-1)
        expected_wrench = (ctrl.shape[0], body_ids_np.size, 3)
        force_np = None if force is None else np.asarray(force, dtype=np.float64)
        torque_np = None if torque is None else np.asarray(torque, dtype=np.float64)
        for name, values in (("force", force_np), ("torque", torque_np)):
            if values is None:
                continue
            if values.shape != expected_wrench:
                raise ValueError(
                    f"pre-step control {name} must have shape {expected_wrench}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"pre-step control {name} contains NaN or Inf")
        if force_np is None and torque_np is None:
            raise ValueError("pre-step control wrench requires force and/or torque")
        return PreStepControlOutput(
            ctrl=converted,
            body_ids=body_ids_np,
            force=force_np,
            torque=torque_np,
        )

    def _apply_pre_step_control(self, ctrl: np.ndarray) -> np.ndarray:
        """Return only the converted actuator control (wrench backends use more).

        Backends that call this wrapper implement the ctrl-only pre-step
        contract.  A callback that returns a wrench must fail closed here
        instead of being silently downgraded to its ``ctrl`` component.
        """
        output = self._convert_pre_step_control(ctrl)
        if output.force is not None or output.torque is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support pre-step control wrenches; return "
                "ctrl only or select a wrench-capable backend"
            )
        return output.ctrl

    def get_capabilities(self, *, profile: str = "default") -> CapabilityReport:
        """Query semantic declarations plus authoritative DR/play/variant support.

        This cold-path query does not discover SDKs or verify runtime behavior.
        Materialized configuration readback is exposed by ``get_import_report``.
        """
        return backend_capabilities(self, profile=profile)

    @abc.abstractmethod
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Return supported domain-randomization capabilities for this backend."""

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return the authoritative default table for a curated reset term.

        Without fixed variants the tail shape is the canonical model table, for
        example ``(nbody,)`` for ``body_mass``. With fixed variants the returned
        table is per-environment, for example ``(num_envs, nbody)``. Callers must
        treat the result as read-only; adapters return detached copies.
        """
        _validate_reset_term(term)
        if not self.get_dr_capabilities().supports_reset_term(term):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support reset term '{term}'"
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose reset term defaults for '{term}'"
        )

    def materialize(self) -> None:
        """Finalize cold-path backend resources before reset/step."""

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Apply a scheduled interval randomization plan.

        Generic dispatch: each op yielded by ``plan.iter_ops()`` is validated
        against the builtin term specs (custom terms pass through) and routed
        to the backend-owned handler table returned by
        :meth:`_interval_term_handlers`.  Op payloads are torch-first: Torch
        tensors or :class:`TensorIntervalPayload` values produced by the
        caller; host adapters convert tensors at their private boundary
        through :mod:`unisim.dr._host`, and a missing producer-sampled payload
        fails closed.  A term without a handler fails
        closed with ``NotImplementedError`` naming the backend class and the
        term.  Backends that need per-plan prologue/epilogue semantics (for
        example clearing staged external forces before the ops accumulate)
        keep a thin override that calls this base implementation.
        """
        if plan.is_empty():
            return
        handlers = self._interval_term_handlers()
        for op in plan.iter_ops():
            op.validate()
            handler = handlers.get(op.term)
            if handler is None:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support interval term '{op.term}'"
                )
            handler(op)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        """Return the backend-owned interval term handler table.

        Backends build this dict once on the cold path (during init or lazily
        cached on first use), keyed by term name; it must not be rebuilt per
        call.  Any op whose term has no handler fails closed in
        :meth:`apply_interval_randomization`.
        """
        return {}

    def apply_body_force(
        self,
        body_ids: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """Apply a world-frame force (and optional torque) to bodies for the upcoming step.

        Args:
            body_ids: Body ids whose external forces should be perturbed.
            force: Force values with shape ``(num_envs, len(body_ids), 3)``.
            torque: Optional world-frame torque values with the same shape.
                Backends without a torque channel must fail closed when this
                is not ``None``.

        Returns:
            None. Backends that support this mutate their pending simulation state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support interval body force perturbation"
        )

    def get_play_capabilities(self) -> PlayCapabilities:
        """Return backend-native play/render capabilities."""
        return self._play_capabilities

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> PlayRenderPlan:
        """Resolve high-level playback mode into backend-owned render parameters."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define playback render mode semantics"
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Callable[[], Any],
        step: Callable[[Any], Any],
        num_steps: int | None,
        output_video: str | PathLike[str] | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Callable[[], np.ndarray] | None = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: DebugOverlayGetter | None = None,
        on_frame: Callable[[int, np.ndarray], np.ndarray | None] | None = None,
    ) -> str | None:
        """Execute backend-owned playback for an env wrapper.

        ``camera_kwargs`` is normalized into :class:`CameraCfg` at this
        boundary; unknown mapping keys fail closed with an error naming them.

        ``debug_overlay_getter`` is an optional per-frame callback returning a
        sequence with one entry per environment (``len == num_envs``); each
        entry is that env's sequence of :class:`DebugPrimitive` (``None`` or
        empty marks an env without overlay) and returning ``None`` disables
        overlays for the frame.  Primitive poses are env-local; the renderer
        applies grid offsets when composing multiple envs.  Backends whose
        ``get_play_capabilities().supports_debug_overlay`` is False fail
        closed with :class:`NotImplementedError` when this is not ``None``.
        On the interactive rendering path only backends whose
        ``supports_interactive_debug_overlay`` is True consume it; the others
        fail closed with :class:`NotImplementedError`.

        ``on_frame`` is an optional per-frame video hook called by offline
        render pipelines before encoding: it receives ``(frame_index, frame)``
        with the frame an ``(H, W, 3)`` uint8 array, and returns a replacement
        frame of the same shape/dtype or ``None`` to keep the original.
        Backends rendering through a native (non-offline) renderer fail closed
        with :class:`NotImplementedError` when this is not ``None``.

        Known boundary: ``env`` is the owning env wrapper, not a physics-layer
        concept. Current playback implementations read env-level configuration
        (e.g. ``cfg.scene``, ``cfg.ctrl_dt``, ``cfg.render_spacing``) and
        env-owned playback helpers (``get_playback_model``,
        ``get_physics_state_snapshot``) that have no backend-native equivalent
        yet. The parameter stays on this contract until playback asset/config
        resolution moves onto backend-owned metadata; backends must only use
        it on the cold playback path.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support playback execution")

    def bind_renderer(
        self,
        spacing: float = 1.0,
        *,
        offset_mode: str = "grid",
        headless: bool = False,
        capture: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a backend-native renderer.

        ``headless`` controls whether a native window is opened. ``capture``
        controls whether ``capture_video_frame`` is valid for the renderer.
        ``camera_kwargs`` is normalized into :class:`CameraCfg` at this
        boundary; unknown mapping keys fail closed with an error naming them.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support native rendering")

    def render(self) -> None:
        """Render one frame through a backend-native interactive renderer.

        Raises:
            RenderClosedError: If the user closed the render window.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native interactive rendering"
        )

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame through a backend-native renderer.

        Raises:
            RenderClosedError: If the user closed the render window.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native video capture"
        )

    def get_physics_state(self) -> np.ndarray:
        """Return a physics snapshot suitable for offline playback/video export.

        Rows use the ``[time, qpos, qvel]`` layout; backends whose model has
        mocap bodies append ``[mocap_pos(nmocap*3), mocap_quat(nmocap*4)]`` so
        offline rendering can replay mocap-driven geometry at its recorded
        pose.  Consumers must decode snapshots through
        ``get_physics_state_layout().split_state`` rather than hardcoding
        column slices.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_physics_state_layout(self) -> PhysicsStateLayout:
        """Return the contract-level layout of ``get_physics_state`` snapshots.

        Backends reporting
        ``get_play_capabilities().supports_physics_state_playback`` must
        implement this so render frontends can split snapshots without
        hardcoding the column layout.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
        )

    def get_playback_mocap_state(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return copied ``(mocap_pos, mocap_quat)`` arrays for detached playback.

        Shapes are ``(nmocap, 3)`` and ``(nmocap, 4)`` for the selected
        environment.  Backends exposing this declare
        ``get_play_capabilities().supports_mocap_playback``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support mocap playback state"
        )

    def set_physics_state(self, state: np.ndarray) -> None:
        """Restore a snapshot produced by ``get_physics_state``.

        Backends implementing this must refresh their host caches so state and
        sensor getters stay consistent with the restored physics state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state restore"
        )

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the playback model for a specific env when variants exist.

        Args:
            env_index: Optional vectorized environment index.

        Returns:
            The backend model object used by playback tooling.
        """
        return self.model

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint (kp, kd) arrays from the backend model."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reading actuator gains"
        )

    # ------------------------------------------------------------------ #
    # Base kinematics                                                      #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Kinematics / Jacobian                                                #
    # ------------------------------------------------------------------ #

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve site names to integer ID arrays.

        Args:
            names: Site names.

        Returns:
            ``int32`` ID array with shape ``(len(names),)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_site_ids")

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to absolute DoF indices in velocity space.

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` addressing the
            complete qvel vector, including any root coordinates.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_joint_dof_indices")

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to DoF indices in position space (qpos).

        Only single-DoF joints are supported; free joints are excluded.

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the joint section of qpos.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_dof_pos_indices"
        )

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to joint-section DoF indices in velocity space.

        Unlike :meth:`get_joint_dof_indices`, the returned indices exclude any
        root/free-joint velocity columns. They address the same joint section
        used by :meth:`get_joint_dof_pos_indices`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_dof_vel_indices"
        )

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve single-DoF joints to full ``set_state_tensor`` qpos columns.

        Unlike :meth:`get_joint_dof_pos_indices`, these indices address the
        complete qpos vector accepted by :meth:`set_state_tensor`, including any root
        coordinates.  Manager reset transactions resolve them on the cold path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_state_qpos_indices"
        )

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve single-DoF joints to full ``set_state_tensor`` qvel columns."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_state_qvel_indices"
        )

    def get_site_jacobian_w(
        self,
        site_id: int,
        dof_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute world-frame Jacobians for one site and selected DoF columns.

        Args:
            site_id: Integer site ID.
            dof_indices: DoF column indices to extract, with shape ``(n_dof,)``.

        Returns:
            ``(jacp, jacr)`` translation/rotation Jacobians, each with shape
            ``(num_envs, 3, n_dof)``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_site_jacobian_w")

    # ------------------------------------------------------------------ #
    # Sensors                                                              #
    # ------------------------------------------------------------------ #

    def get_sensor_names(self) -> tuple[str, ...]:
        """Return the complete public named-sensor namespace."""
        raise NotImplementedError(
            f"Backend '{self.backend_type}' does not expose its named sensor namespace"
        )

    def get_sensor_inventory(self) -> tuple[SensorDescriptor, ...]:
        """Return the complete public named-sensor inventory and widths."""
        names = self.get_sensor_names()
        descriptors: list[SensorDescriptor] = []
        for name in names:
            try:
                shape = tuple(self.get_sensor_view(name).shape)
            except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Backend '{self.backend_type}' cannot inventory sensor '{name}': {exc}"
                ) from exc
            if len(shape) < 2 or shape[0] != self.num_envs:
                raise ValueError(
                    f"Backend '{self.backend_type}' sensor '{name}' returned shape "
                    f"{shape}; expected leading dimension {self.num_envs}"
                )
            width = 1
            for dimension in shape[1:]:
                width *= int(dimension)
            if width <= 0:
                raise ValueError(
                    f"Backend '{self.backend_type}' sensor '{name}' has empty data shape {shape}"
                )
            descriptors.append(SensorDescriptor(name=name, width=width))
        return tuple(descriptors)

    def bind_sensor_data(self, names: Sequence[str]) -> SensorView:
        """Materialize a validated view over named tensor sensors.

        This cold-path binding records the flattened public view width and delegates
        execution to the backend-owned tensor sensor reader.  It never falls back to
        a NumPy named-sensor getter.
        """
        if isinstance(names, (str, bytes)):
            raise TypeError(
                f"Backend '{self.backend_type}' sensor view names must be a sequence of strings, "
                "not one string"
            )
        sensor_names = tuple(names)
        if not sensor_names:
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view requires at least one name"
            )
        if any(not isinstance(name, str) or not name for name in sensor_names):
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view names must be non-empty strings"
            )
        if len(set(sensor_names)) != len(sensor_names):
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view names must be unique: {sensor_names}"
            )

        dimensions: list[int] = []
        for name in sensor_names:
            single_reader = self._bind_sensor_data_reader((name,))
            try:
                batch = np.asarray(single_reader())
            except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Backend '{self.backend_type}' cannot bind sensor '{name}': {exc}"
                ) from exc
            width = 1 if batch.ndim < 2 else int(np.prod(batch.shape[1:], dtype=np.int64))
            if batch.ndim < 2 or batch.shape[0] != self.num_envs or width <= 0:
                raise ValueError(
                    f"Backend '{self.backend_type}' sensor '{name}' returned shape "
                    f"{batch.shape}; expected ({self.num_envs}, width>0)"
                )
            dimensions.append(width)

        reader = self._bind_sensor_data_reader(sensor_names)
        try:
            batch = np.asarray(reader())
        except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
            raise type(exc)(
                f"Backend '{self.backend_type}' cannot bind sensors {sensor_names}: {exc}"
            ) from exc
        expected_width = sum(dimensions)
        if batch.shape != (self.num_envs, expected_width):
            raise ValueError(
                f"Backend '{self.backend_type}' sensor view {sensor_names} returned shape "
                f"{batch.shape}; expected ({self.num_envs}, {expected_width})"
            )

        return SensorView(
            backend_type=self.backend_type,
            names=sensor_names,
            dimensions=tuple(dimensions),
            num_envs=self.num_envs,
            _reader=reader,
        )

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> SensorReadFn:
        """Create a backend-owned tensor reader for a materialized view."""
        views = [self.get_sensor_view(name) for name in names]
        return lambda: tuple(views) if len(views) > 1 else views[0]
