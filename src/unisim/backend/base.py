import abc
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal

import numpy as np

from unisim.dr.types import (
    DomainRandomizationCapabilities,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    IntervalTermOp,
    ResetRandomizationPayload,
)

PreStepControlFn = Callable[[Any, np.ndarray], np.ndarray]
TerrainHeightSampleFn = Callable[[np.ndarray], np.ndarray]
SensorReadFn = Callable[[], np.ndarray]


DebugPrimitiveKind = Literal["sphere", "box", "frame", "arrow", "ghost_geom", "text"]
DEBUG_PRIMITIVE_KINDS = frozenset(
    {"sphere", "box", "frame", "arrow", "ghost_geom", "text"}
)
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
                raise ValueError(
                    f"DebugPrimitive quat must be unit-length wxyz (norm {norm:.6f})"
                )
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
            f"debug overlays must have one entry per env (len == {num_envs}); "
            f"got {len(overlays)}"
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
            raise ValueError(
                f"unknown camera_kwargs key(s): {unknown}; supported keys: {allowed}"
            )
        return cls(**dict(kwargs))


def unsupported_debug_overlay_error(owner: str) -> NotImplementedError:
    """Build the fail-closed error for backends without debug overlay support."""
    return NotImplementedError(
        f"{owner} does not support debug overlay primitives "
        "(get_play_capabilities().supports_debug_overlay is False); omit "
        "debug_overlay_getter or use a backend on the MuJoCo offline snapshot pipeline"
    )


@dataclass(frozen=True)
class BackendMocapPoseBinding:
    """Cold-bound world-space pose access for one fixed mocap body.

    Poses use xyz + unit wxyz quaternion. Writes affect only selected worlds,
    preserve generalized state and refresh derived state before returning.
    A subsequent ``set_state`` resets that world's mocap poses to defaults;
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
class BackendTerrainSpawnData:
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
class BackendRootStateLayout:
    """Generalized-state columns for one floating root body.

    ``qpos_indices`` address ``[x, y, z, qw, qx, qy, qz]`` in the public
    :meth:`SimBackend.set_state` qpos representation. ``qvel_indices`` address
    ``[linear_velocity_world, angular_velocity_body]``.  Manager-facing root
    states use world-frame angular velocity, so the base-owned reset
    transaction performs the frame conversion before calling ``set_state``.
    """

    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values, expected in (
            ("qpos_indices", self.qpos_indices, 7),
            ("qvel_indices", self.qvel_indices, 6),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"BackendRootStateLayout {name} must be a tuple")
            if len(values) != expected:
                raise ValueError(
                    f"BackendRootStateLayout {name} must contain {expected} columns; "
                    f"got {len(values)}"
                )
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in values
            ):
                raise TypeError(f"BackendRootStateLayout {name} must contain integer columns")
            normalized = tuple(int(value) for value in values)
            if any(value < 0 for value in normalized):
                raise ValueError(f"BackendRootStateLayout {name} cannot contain negative columns")
            if len(set(normalized)) != expected:
                raise ValueError(f"BackendRootStateLayout {name} must contain unique columns")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class BackendSensorView:
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
            raise ValueError("BackendSensorView backend_type must be a non-empty string")
        if not isinstance(self.names, tuple) or not self.names:
            raise ValueError("BackendSensorView names must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("BackendSensorView names must contain non-empty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"BackendSensorView names must be unique: {self.names}")
        if not isinstance(self.dimensions, tuple) or len(self.dimensions) != len(self.names):
            raise ValueError("BackendSensorView dimensions must contain one entry per sensor name")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in self.dimensions
        ):
            raise ValueError("BackendSensorView dimensions must be positive integers")
        if (
            isinstance(self.num_envs, (bool, np.bool_))
            or not isinstance(self.num_envs, (int, np.integer))
            or int(self.num_envs) <= 0
        ):
            raise ValueError("BackendSensorView num_envs must be a positive integer")
        if not callable(self._reader):
            raise TypeError("BackendSensorView reader must be callable")
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
class BackendPlayCapabilities:
    """Backend-native play/render capabilities surfaced through env contracts."""

    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False
    supports_debug_overlay: bool = False


class BackendHeightScanner(abc.ABC):
    """Backend-owned height-field scanner created on the env init path."""

    @abc.abstractmethod
    def scan(self) -> np.ndarray:
        """Return sampled values with shape ``(num_envs, num_points)``."""


PLAY_RENDER_MODES = frozenset({"auto", "interactive", "record", "none"})


@dataclass(frozen=True)
class BackendPlayRenderPlan:
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


def log_playback_plan(plan: BackendPlayRenderPlan, *, prefix: str = "") -> None:
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

    _pre_step_control_fn: PreStepControlFn | None
    _scene_cleanup_handle: Any | None
    backend_type: str

    @property
    def capabilities(self):
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

    def get_state(self, fields=None):
        """Return a detached, backend-neutral state snapshot.

        ``qpos`` and ``qvel`` are assembled from the public kinematic getters;
        adapters may override this to expose native fields such as ``ctrl``.
        This convenience API keeps benchmark clients independent from private
        model/data objects while the full reset contract remains ``set_state``.
        """
        requested = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else tuple(fields))
        )
        result = {}
        if "qpos" in requested:
            result["qpos"] = np.concatenate(
                (self.get_base_pos(), self.get_base_quat(), self.get_dof_pos()), axis=1
            )
        if "qvel" in requested:
            result["qvel"] = np.concatenate(
                (self.get_base_lin_vel(), self.get_base_ang_vel(), self.get_dof_vel()), axis=1
            )
        if "ctrl" in requested:
            raise NotImplementedError(f"{self.__class__.__name__} does not expose control state")
        unknown = set(requested) - {"qpos", "qvel", "ctrl"}
        if unknown:
            raise KeyError(f"unknown {self.backend_type} state field(s): {sorted(unknown)}")
        return result

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Reset selected environments to the backend's default state."""
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
        self.set_state(ids, qpos, qvel)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """Number of vectorized environments."""

    @property
    @abc.abstractmethod
    def model(self):
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

    def get_terrain_spawn_data(self) -> BackendTerrainSpawnData | None:
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
        """Return default joint positions in the same column order as ``get_dof_pos``.

        The returned array is detached, one-dimensional, and excludes floating
        root coordinates.  Backends whose DoF view is actuator-indexed must use
        that same actuator-target order here.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose default DoF positions"
        )

    @abc.abstractmethod
    def get_init_qvel(self) -> np.ndarray:
        """Return a zero-initialized qvel vector compatible with ``set_state``.

        Returns:
            Zero-filled qvel array.
        """

    def get_root_state_layout(self, root_body_name: str) -> BackendRootStateLayout:
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
        """Return one geom size vector through the backend contract."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose geom sizes")

    def get_geom_sizes(self) -> np.ndarray:
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

    def bind_mocap_pose(self, body_name: str) -> BackendMocapPoseBinding:
        """Resolve a mocap body once; unavailable capabilities fail at binding."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support mocap pose writes")

    def create_hfield_scanner(
        self,
        *,
        hfield_geom_id: int,
        offsets: np.ndarray,
        frame_body_id: int,
        alignment: str = "yaw",
        output: str = "height",
    ) -> BackendHeightScanner:
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

    def get_body_ipos(self) -> np.ndarray:
        """Return the backend body inertial-position table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose body ipos")

    def get_dof_armature(self) -> np.ndarray:
        """Return the backend dof-armature table."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose dof armature")

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve backend-native body IDs used by motion datasets."""
        raise NotImplementedError(f"{self.__class__.__name__} does not expose motion body ids")

    def cleanup_scene_assets(self) -> None:
        """Release cold-path scene artifacts owned by the backend."""
        cleanup_handle = getattr(self, "_scene_cleanup_handle", None)
        if cleanup_handle is None:
            return
        cleanup_handle.cleanup()
        self._scene_cleanup_handle = None

    def __del__(self) -> None:
        try:
            self.cleanup_scene_assets()
        except Exception:
            pass

    @abc.abstractmethod
    def get_joint_range(self) -> np.ndarray | None:
        """Return joint position limits, excluding the floating base.

        Returns:
            Array with shape ``(num_dof, 2)`` and columns ``[low, high]``, or
            ``None`` when the backend does not expose limits.
        """

    # ------------------------------------------------------------------ #
    # Simulation control                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
        """Advance physics.

        Args:
            ctrl: Control input with shape ``(num_envs, nu)``.
            nsteps: Number of physics substeps.

        Returns:
            Optional dictionary. Backends may include a ``"timing"`` key with
            per-phase timings in milliseconds.
        """

    def set_pre_step_control(self, fn: PreStepControlFn | None) -> None:
        """Register an env-owned policy-control to physics-control converter.

        The callback receives ``(backend, ctrl)`` so owner code can read the
        backend's freshly-updated sensor contract before every physics substep.
        It must return backend-native actuator control with the same shape.
        Position-actuator envs leave this unset and keep the direct control path.
        """
        self._pre_step_control_fn = fn

    def _apply_pre_step_control(self, ctrl: np.ndarray) -> np.ndarray:
        if self._pre_step_control_fn is None:
            return ctrl
        converted = np.asarray(self._pre_step_control_fn(self, ctrl), dtype=ctrl.dtype)
        if converted.shape != ctrl.shape:
            raise ValueError(
                f"pre-step control must return shape {ctrl.shape}, got {converted.shape}"
            )
        return converted

    @abc.abstractmethod
    def set_state(
        self,
        env_indices: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization: ResetRandomizationPayload | None = None,
    ) -> dict | None:
        """Set physics state for selected environments.

        Args:
            env_indices: Environment indices.
            qpos: Position state. Free-root columns exposed by
                :meth:`get_root_state_layout` use world xyz and wxyz quaternion.
            qvel: Velocity state. Free-root columns exposed by
                :meth:`get_root_state_layout` use world linear velocity and
                body-frame angular velocity.
            randomization: Optional backend randomization payload.

        Returns:
            Optional dictionary. Backends MAY include a ``"timing"`` key with
            per-substep timings in milliseconds (e.g. ``set_state_mask_ms``,
            ``set_state_data_slice_ms``, ...). Callers MUST treat ``None`` or
            missing keys as "not reported" — the outer wall-clock measurement in
            ``DomainRandomizationManager.reset`` (``dr_reset_set_state_ms``)
            remains authoritative for total set_state time.
        """

    @abc.abstractmethod
    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Return supported domain-randomization capabilities for this backend."""

    def apply_init_randomization(self, plan: InitRandomizationPlan) -> None:
        """Apply cold-path model/materialization randomization."""
        if plan.is_empty():
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support init-lifecycle randomization"
        )

    def materialize(self) -> None:
        """Finalize cold-path backend resources before reset/step."""

    def apply_interval_randomization(self, plan: IntervalRandomizationPlan) -> None:
        """Apply a scheduled interval randomization plan.

        Generic dispatch: each op yielded by ``plan.iter_ops()`` is validated
        against the builtin term specs (custom terms pass through) and routed
        to the backend-owned handler table returned by
        :meth:`_interval_term_handlers`.  A term without a handler fails
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

    def get_play_capabilities(self) -> BackendPlayCapabilities:
        """Return backend-native play/render capabilities."""
        return BackendPlayCapabilities()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: str | PathLike[str] | None,
    ) -> BackendPlayRenderPlan:
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

    def init_renderer(
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
        """Return a physics snapshot suitable for offline playback/video export."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support physics-state playback"
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

    @abc.abstractmethod
    def get_base_pos(self) -> np.ndarray:
        """Return base position in the world frame.

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_quat(self) -> np.ndarray:
        """Return base quaternion in the world frame as ``wxyz``.

        Returns:
            (num_envs, 4)
        """

    @abc.abstractmethod
    def get_base_lin_vel(self) -> np.ndarray:
        """Return base linear velocity in the world frame.

        This is the first three dimensions of generalized velocity ``qvel``,
        expressed in world coordinates.

        Returns:
            (num_envs, 3)
        """

    @abc.abstractmethod
    def get_base_ang_vel(self) -> np.ndarray:
        """Return base angular velocity in the world frame.

        This is dimensions 3-5 of generalized velocity ``qvel``, expressed in
        world coordinates. It differs from gyro readings: gyro sensors report
        angular velocity components in the body/sensor local frame, while this
        contract returns world-frame values. Use the matching sensor contract
        when body-frame angular velocity is required.

        Returns:
            (num_envs, 3)
        """

    # ------------------------------------------------------------------ #
    # DOF state                                                            #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_dof_pos(self) -> np.ndarray:
        """Return joint positions, excluding the base.

        Returns:
            (num_envs, num_dof)
        """

    @abc.abstractmethod
    def get_dof_vel(self) -> np.ndarray:
        """Return joint velocities, excluding the base.

        Returns:
            (num_envs, num_dof)
        """

    # ------------------------------------------------------------------ #
    # Body kinematics — world frame                                        #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body positions in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body quaternions in the world frame as ``wxyz``.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 4)
        """

    def get_body_pose_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return selected body positions and quaternions in the world frame."""
        return self.get_body_pos_w(body_ids), self.get_body_quat_w(body_ids)

    @abc.abstractmethod
    def get_body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body linear velocities in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    def get_body_vel_w(self, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return selected body linear and angular velocities in the world frame."""
        return self.get_body_lin_vel_w(body_ids), self.get_body_ang_vel_w(body_ids)

    @abc.abstractmethod
    def get_body_ang_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body angular velocities in the world frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    def get_body_state_w(
        self, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get selected body position, quaternion, linear velocity, and angular velocity."""
        return (
            self.get_body_pos_w(body_ids),
            self.get_body_quat_w(body_ids),
            self.get_body_lin_vel_w(body_ids),
            self.get_body_ang_vel_w(body_ids),
        )

    def copy_body_state_w(
        self,
        body_ids: np.ndarray,
        out_pos: np.ndarray,
        out_quat: np.ndarray,
        out_lin_vel: np.ndarray,
        out_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Copy selected world-frame body state into caller-owned buffers."""
        pos, quat, lin_vel, ang_vel = self.get_body_state_w(body_ids)
        out_pos[...] = pos
        out_quat[...] = quat
        out_lin_vel[...] = lin_vel
        out_ang_vel[...] = ang_vel
        return out_pos, out_quat, out_lin_vel, out_ang_vel

    def get_body_pose_w_rows(
        self, env_ids: np.ndarray, body_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get selected env rows of world-frame body position and quaternion."""
        rows = np.asarray(env_ids, dtype=np.intp)
        return self.get_body_pos_w(body_ids)[rows], self.get_body_quat_w(body_ids)[rows]

    def get_body_lin_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        """Get selected env rows of world-frame body linear velocity."""
        rows = np.asarray(env_ids, dtype=np.intp)
        return self.get_body_lin_vel_w(body_ids)[rows]

    def get_body_ang_vel_w_rows(self, env_ids: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
        """Get selected env rows of world-frame body angular velocity."""
        rows = np.asarray(env_ids, dtype=np.intp)
        return self.get_body_ang_vel_w(body_ids)[rows]

    # ------------------------------------------------------------------ #
    # Body kinematics — baselink frame                                     #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_body_pos_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body positions in the baselink frame.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_quat_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body quaternions in the baselink frame as ``wxyz``.

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 4)
        """

    @abc.abstractmethod
    def get_body_lin_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body linear velocities expressed in each body's own frame.

        The value is the body's world-frame velocity rotated by the inverse of
        the body's world-frame orientation, i.e.
        ``quat_apply_inverse(quat_w, lin_vel_w)`` (mjlab/Isaac-style analytical
        definition). It is well-defined for every body — including the root
        body — and must NOT be implemented as the motion relative to the
        baselink frame (which degenerates to zero for the root body).

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

    @abc.abstractmethod
    def get_body_ang_vel_b(self, body_ids: np.ndarray) -> np.ndarray:
        """Return selected body angular velocities expressed in each body's own frame.

        The value is the body's world-frame angular velocity rotated by the
        inverse of the body's world-frame orientation, i.e.
        ``quat_apply_inverse(quat_w, ang_vel_w)`` (mjlab/Isaac-style analytical
        definition). It is well-defined for every body — including the root
        body — and must NOT be implemented as the motion relative to the
        baselink frame (which degenerates to zero for the root body).

        Args:
            body_ids: Body ID array.

        Returns:
            (num_envs, len(body_ids), 3)
        """

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
        """Resolve joint names to DoF indices in velocity space (qvel).

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the qvel start.
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
        """Resolve joint names to DoF indices in velocity space (qvel).

        Args:
            names: Joint names.

        Returns:
            ``int32`` index array with shape ``(len(names),)`` relative to
            the joint section start.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_dof_vel_indices"
        )

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve single-DoF joints to full ``set_state`` qpos columns.

        Unlike :meth:`get_joint_dof_pos_indices`, these indices address the
        complete qpos vector accepted by :meth:`set_state`, including any root
        coordinates.  Manager reset transactions resolve them on the cold path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_joint_state_qpos_indices"
        )

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve single-DoF joints to full ``set_state`` qvel columns."""
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

    @abc.abstractmethod
    def get_sensor_data(self, name: str) -> np.ndarray:
        """Return sensor data.

        Args:
            name: Sensor name.

        Returns:
            Sensor data array.
        """

    def get_sensor_data_rows(self, name: str, env_ids: np.ndarray) -> np.ndarray:
        """Get selected env rows of a sensor array."""
        return self.get_sensor_data(name)[np.asarray(env_ids, dtype=np.intp)]

    def get_sensor_data_batch(self, names: Sequence[str]) -> np.ndarray:
        """Fetch multiple sensors and concatenate their flattened values.

        Args:
            names: Sensor names in output order.

        Returns:
            Array with shape ``(num_envs, total_sensor_values)``.
        """
        sensor_names = tuple(names)
        if not sensor_names:
            return np.empty((self.num_envs, 0), dtype=np.float64)
        values = [np.asarray(self.get_sensor_data(name)) for name in sensor_names]
        flat_values = [value.reshape(value.shape[0], -1) for value in values]
        return np.concatenate(flat_values, axis=1)

    def bind_sensor_data(self, names: Sequence[str]) -> BackendSensorView:
        """Materialize a validated view over named sensors on the cold path.

        The existing sensor getters remain the sole backend adapter surface.  This
        method validates each requested sensor once, records its flattened width,
        and returns a stable view for manager terms.  Backends override the
        protected reader hook when numeric slots or stable cache slices are
        available; callers do not depend on that implementation detail.
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
            try:
                value = np.asarray(self.get_sensor_data(name))
            except (KeyError, NotImplementedError, ValueError) as exc:
                raise type(exc)(
                    f"Backend '{self.backend_type}' cannot bind sensor '{name}': {exc}"
                ) from exc
            if value.ndim < 1 or value.shape[0] != self.num_envs:
                raise ValueError(
                    f"Backend '{self.backend_type}' sensor '{name}' returned shape "
                    f"{value.shape}; expected leading dimension {self.num_envs}"
                )
            width = int(np.prod(value.shape[1:], dtype=np.int64)) if value.ndim > 1 else 1
            if width <= 0:
                raise ValueError(
                    f"Backend '{self.backend_type}' sensor '{name}' has empty data shape "
                    f"{value.shape}"
                )
            dimensions.append(width)

        view = BackendSensorView(
            backend_type=self.backend_type,
            names=sensor_names,
            dimensions=tuple(dimensions),
            num_envs=self.num_envs,
            _reader=self._bind_sensor_data_reader(sensor_names),
        )
        # Validate the batch implementation at materialization as well.  This
        # catches adapters whose individual and batch sensor contracts disagree.
        view.read()
        return view

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> SensorReadFn:
        """Create the backend-owned reader retained by a materialized sensor view.

        The default keeps the existing batch getter as the compatibility path
        for lightweight adapters and test doubles.  Concrete backends that
        expose stable numeric slots or host-cache slices override this hook so
        manager hot paths never resolve model metadata.
        """
        batch_reader = self.get_sensor_data_batch
        return lambda: batch_reader(names)
