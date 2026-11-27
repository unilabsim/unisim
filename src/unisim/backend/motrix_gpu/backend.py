"""In-process CUDA-resident MotrixSim backend for single-articulation training.

This adapter is the Motrix counterpart of the MJWarp device-resident lifecycle:
``motrixsim.gpu`` builds a Warp ``SceneModel`` and a batched ``SceneData`` whose
``dof_pos``/``dof_vel``/``actuator_ctrls`` arrays are exposed as zero-copy Torch
views, physics steps on the GPU, and selected reset writes directly into the
device state.  Portable scene fragments are composed through the public MSD
world API before the GPU model is built.  Selected resets also publish link
kinematics, tracked-body views and the kinematics-dependent sensors, and the
device-resident packed reset transaction scatters per-world actuator
``kp``/``kd``, link mass / center of mass, and geom friction final values in
place.  It is intentionally narrow: portable entities, fixed variants, and
terrain remain unsupported and fail closed.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from unisim.dr.interval import IntervalTermOp, require_op_body_ids
from unisim.dr.types import (
    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    TensorResetRandomizationPayload,
    _validate_reset_term,
    validate_tensor_reset_payload,
)
from unisim.scene import (
    SceneCfg,
    require_scene_composition_support,
    resolve_scene_fragment_path,
)

from ..base import (
    CameraCfg,
    PhysicsStateLayout,
    PlayCapabilities,
    PlayRenderPlan,
    PublicStateWidths,
    RootStateLayout,
    SelectedResetPublication,
    SensorCategory,
    SensorDescriptor,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
    TrackedBodyStateViews,
    normalize_play_render_mode,
    unsupported_debug_overlay_error,
)
from ..graph_capture import (
    DeviceVerdicts,
    ExternalGraphCapture,
    GraphCaptureError,
    closing_capture,
)
from ..motrix.playback import run_motrix_playback
from .dependencies import load_motrix_gpu_dependencies
from .playback import MotrixGpuPlaybackRenderer

_SUPPORTED_STATE_FIELDS = frozenset({"qpos", "qvel", "ctrl"})
# Link and collider term support depends on the materialized scene.
_BASE_SUPPORTED_RESET_TERMS = frozenset({RESET_TERM_KP, RESET_TERM_KD})
_RBI_STRIDE = 13  # mass + h (vec3) + inertia (mat33), matching the engine layout.
_SPATIAL_INERTIA_STRIDE = 36  # 6x6 row-major float matrix.
# Mirror the CPU adapter's reviewed solver budget.
_DEFAULT_MAX_ITERATIONS = 3
# Public tracked-body sensor contract shared with the other device-resident
# adapters: ``track_<field>_w_<body>`` names published from one stable block per
# field.  MotrixSim stores link quaternions as xyzw, so the public ``quat_w``
# block is permuted to the MuJoCo-derived ``wxyz`` layout.
_TRACKED_SENSOR_LAYOUT: tuple[tuple[str, str, int, bool], ...] = (
    ("track_pos_w_", "pos_w", 3, False),
    ("track_quat_w_", "quat_w", 4, True),
    ("track_linvel_w_", "lin_vel_w", 3, False),
    ("track_angvel_w_", "ang_vel_w", 3, False),
)
_TRACKED_GETTERS: dict[str, str] = {
    "pos_w": "get_position",
    "quat_w": "get_rotation",
    "lin_vel_w": "get_linear_velocity",
    "ang_vel_w": "get_angular_velocity",
}


def _collect_sensor_declarations(
    model_file: str,
    fragment_files: Sequence[str] = (),
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, str]],
    tuple[str, ...],
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str, str]],
]:
    """Collect named ``<sensor>`` declarations from the composed scene.

    The GPU model exposes no sensor-name list, so the adapter rebuilds it from
    the source files (following ``<include>`` and portable fragments) on the
    cold path.  The result is validated against the runtime's addressable name
    table before use.  Geom-pair ``<contact data="found">`` declarations are
    returned separately because the runtime exposes them through the row-local
    contact query rather than ``get_sensor_value``.  ``<actuatorfrc>`` and
    ``<jointactuatorfrc>`` declarations are returned as ``name -> (kind,
    target)``; the runtime has no actuator-force sensor stage, so the adapter
    republishes them from the affine servo law (see
    ``_initialize_actuator_force_sensors``).  ``<contact data="force"
    reduce="netforce">`` declarations on a geom or body pair are returned as
    ``name -> (filter kind, first, second)`` and republished from the solver's
    normal impulses (see ``_initialize_contact_force_sensors``).
    """
    names: list[str] = []
    declared_names: set[str] = set()
    contacts: dict[str, tuple[str, str]] = {}
    actuator_forces: dict[str, tuple[str, str]] = {}
    netforces: dict[str, tuple[str, str, str]] = {}
    frame_quats: list[str] = []
    visited: set[Path] = set()

    def walk(path: Path, element: ET.Element) -> None:
        for child in element:
            if child.tag == "include":
                target = (path.parent / str(child.get("file"))).resolve()
                if target in visited:
                    raise ValueError(f"cyclic MJCF include in sensor scan: {target}")
                visited.add(target)
                walk(target, ET.parse(target).getroot())
            elif child.tag == "sensor":
                for declaration in child:
                    name = declaration.get("name")
                    if not name:
                        continue
                    if name in declared_names:
                        raise ValueError(
                            f"composed Motrix scene contains duplicate sensor name {name!r}"
                        )
                    declared_names.add(name)
                    names.append(name)
                    if declaration.tag == "framequat":
                        frame_quats.append(name)
                    if declaration.tag == "contact" and declaration.get("data") == "found":
                        geom1 = declaration.get("geom1")
                        geom2 = declaration.get("geom2")
                        if geom1 and geom2:
                            contacts[name] = (geom1, geom2)
                    if (
                        declaration.tag == "contact"
                        and declaration.get("data", "force") == "force"
                        and declaration.get("reduce") == "netforce"
                    ):
                        force_geom1 = declaration.get("geom1")
                        force_geom2 = declaration.get("geom2")
                        body1 = declaration.get("body1")
                        body2 = declaration.get("body2")
                        if force_geom1 and force_geom2:
                            netforces[name] = ("geom", force_geom1, force_geom2)
                        elif body1 and body2:
                            netforces[name] = ("body", body1, body2)
                    if declaration.tag in ("actuatorfrc", "jointactuatorfrc"):
                        target_name = declaration.get("actuator") or declaration.get("joint")
                        if target_name:
                            actuator_forces[name] = (declaration.tag, target_name)
            else:
                walk(path, child)

    root = Path(model_file).resolve()
    visited.add(root)
    walk(root, ET.parse(root).getroot())
    for fragment_file in fragment_files:
        fragment = resolve_scene_fragment_path(fragment_file, root)
        if fragment in visited:
            continue
        visited.add(fragment)
        walk(fragment, ET.parse(fragment).getroot())
    return tuple(names), contacts, tuple(frame_quats), actuator_forces, netforces


def _collect_sensor_names(
    model_file: str,
    fragment_files: Sequence[str] = (),
) -> tuple[str, ...]:
    """Collect only the named ``<sensor>`` declarations in document order."""
    return _collect_sensor_declarations(model_file, fragment_files)[0]


def _interval_value_to_numpy(value: Any) -> np.ndarray:
    """Convert a torch-first interval payload value to a host NumPy array."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        return np.asarray(value)
    raise NotImplementedError(
        f"motrix_gpu interval handlers require a Torch tensor payload; got {type(value).__name__}"
    )


class MotrixGpuBackend(SimBackend):
    """CUDA-resident MotrixSim backend with a narrow device tensor lifecycle."""

    backend_type = "motrix_gpu"

    def __init__(
        self,
        scene: SceneCfg,
        num_envs: int = 1,
        sim_dt: float = 0.01,
        *,
        base_name: str | None = None,
        max_iterations: int | None = None,
        max_contacts: int | None = None,
        max_contact_constraints: int | None = None,
        max_convexhull_pairs: int | None = None,
        device: str | None = None,
    ) -> None:
        if not isinstance(scene, SceneCfg):
            raise TypeError("motrix_gpu requires a SceneCfg")
        require_scene_composition_support(scene, self.backend_type)
        if not scene.model_file:
            raise ValueError("motrix_gpu requires SceneCfg(model_file=...)")
        if scene.entity_assets or scene.fixed_variant_plan is not None or scene.terrain is not None:
            raise NotImplementedError(
                "motrix_gpu does not yet support portable entities, fixed variants, or terrain"
            )
        if isinstance(num_envs, bool) or int(num_envs) < 1:
            raise ValueError("num_envs must be a positive integer")
        if not float(sim_dt) > 0.0:
            raise ValueError("sim_dt must be positive")
        if max_iterations is not None and (
            isinstance(max_iterations, bool) or int(max_iterations) <= 0
        ):
            raise ValueError(
                f"max_iterations must be a positive integer or None, got {max_iterations!r}"
            )
        if max_convexhull_pairs is not None and (
            isinstance(max_convexhull_pairs, bool)
            or not isinstance(max_convexhull_pairs, int)
            or max_convexhull_pairs < 0
        ):
            raise ValueError(
                "max_convexhull_pairs must be a non-negative integer or None, "
                f"got {max_convexhull_pairs!r}"
            )

        self._num_envs = int(num_envs)
        self._sim_dt = float(sim_dt)
        self._closed = False
        self._state_torch_views: dict[str, Any] = {}
        self._struct_flat_views: dict[str, Any] = {}
        self._pre_step_control_fn = None
        self._pre_step_control_active = False
        self._qpos_public: Any = None
        self._renderer: MotrixGpuPlaybackRenderer | None = None
        self._external_capture: ExternalGraphCapture | None = None
        self._supported_reset_terms: set[str] = set(_BASE_SUPPORTED_RESET_TERMS)
        self._interval_handler_cache: dict[str, Callable[[IntervalTermOp], None]] | None = None

        options = {}
        for name, value in (
            ("max_contacts", max_contacts),
            ("max_contact_constraints", max_contact_constraints),
        ):
            if value is not None:
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
                options[name] = value

        self._deps = load_motrix_gpu_dependencies()
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("motrix_gpu requires NVIDIA CUDA and a CUDA-enabled PyTorch build")
        selected = torch.device("cuda" if device is None else device)
        if selected.type != "cuda":
            raise ValueError(f"motrix_gpu requires a CUDA device, got {selected}")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        self._device = selected
        self._warp_device = self._deps.warp.get_device(str(selected))
        self._warp_stream = self._deps.warp.Stream(self._warp_device)
        self._torch_stream = torch.cuda.ExternalStream(self._warp_stream.cuda_stream, selected)
        self._caller_ready = torch.cuda.Event()
        self._engine_ready = torch.cuda.Event()
        with torch.cuda.device(selected), self._warp_execution():
            self._time = torch.zeros(self._num_envs, dtype=torch.float32, device=selected)
            self._initialize(scene, base_name, max_iterations, max_convexhull_pairs, options)

    def _initialize(
        self,
        scene: SceneCfg,
        base_name: str | None,
        max_iterations: int | None,
        max_convexhull_pairs: int | None,
        options: dict[str, int],
    ) -> None:
        import torch

        self._scene_world: Any = None
        # Every scene goes through the MSD materializer so the GPU model gets
        # the same imported actuator forms, solver options, joint force sensors
        # and collision-cooking repairs as the CPU ``motrix`` adapter.
        from ..motrix.scene import materialize_scene_with_full_sensor_inventory

        cmodel, _, self._scene_world = materialize_scene_with_full_sensor_inventory(
            model_file=scene.model_file,
            fragment_files=scene.fragment_files,
        )
        self._model: Any = self._deps.mgpu.SceneModel.from_cmodel(cmodel, **options)
        if max_convexhull_pairs is not None:
            self._model.options.max_convexhull_pairs = max_convexhull_pairs
        self._nq = int(self._model.num_dof_pos)
        self._nv = int(self._model.num_dof_vel)
        self._nu = int(self._model.num_actuators)
        # Match the MJWarp adapter: the public ``sim_dt`` is authoritative and
        # overrides the MJCF ``<option timestep>`` (Warp reads ``options.timestep``).
        self._model.options.timestep = self._sim_dt
        self._model.options.max_iterations = int(
            _DEFAULT_MAX_ITERATIONS if max_iterations is None else max_iterations
        )
        self._data: Any = self._deps.mgpu.SceneData(self._model, batch=self._num_envs)
        # Publish the compiled default kinematics so cold-path tracked-body
        # reads report the default pose instead of an all-zero buffer.  Named
        # sensors are produced by the physics step and stay zero until the
        # first ``step_tensor`` call.
        self._refresh_high_level()
        self._ctrl_range = self._read_ctrl_range()
        self._gravity = np.asarray(self._model.options.gravity, dtype=np.float64).reshape(3)
        self._scene_model_file = scene.model_file
        # The GPU engine reports unnamed entities as ``None``; the backend-neutral
        # namespace uses the empty string (matching the CPU ``motrix`` adapter).
        self._link_names = tuple(
            "" if name is None else str(name) for name in getattr(self._model, "link_names", ())
        )
        self._geom_names = tuple(
            "" if name is None else str(name) for name in self._model.geom_names
        )
        # The shared Manager contract always passes the scene entity's root body
        # name. MotrixSim GPU derives its root layout from the leading free
        # joint, so the value is validated as an entity-contract sanity check
        # and otherwise unused.
        if base_name is not None and base_name not in self._link_names:
            raise KeyError(f"motrix_gpu base_name {base_name!r} is not a link in the scene")
        self._base_name = base_name
        self._actuator_names = tuple(
            "" if name is None else str(name) for name in self._model.actuator_names
        )
        joint_pos_nums = np.asarray(
            getattr(self._model, "joint_dof_pos_nums", ()), dtype=np.intp
        ).reshape(-1)
        joint_vel_nums = np.asarray(
            getattr(self._model, "joint_dof_vel_nums", ()), dtype=np.intp
        ).reshape(-1)
        # MotrixSim lists only the articulations' scalar joints; floating bases
        # live in ``floating_bases`` and may sit anywhere in the state vector
        # (a fixed-base robot followed by a free object puts them last).
        self._joint_section_pos_offsets = np.concatenate(([0], np.cumsum(joint_pos_nums)))
        self._joint_section_vel_offsets = np.concatenate(([0], np.cumsum(joint_vel_nums)))
        self._floating_roots: dict[str, Any] = {}
        for body in getattr(self._model, "bodies", ()):
            # Fixed bodies carry an unbound FloatingBase whose starts are None.
            base = getattr(body, "floatingbase", None)
            base_link = getattr(body, "base_link", None)
            if (
                base is not None
                and base_link is not None
                and getattr(base, "dof_pos_start", None) is not None
            ):
                self._floating_roots[str(base_link.name)] = base
        (
            self._sensor_names,
            contact_declarations,
            frame_quat_sensor_names,
            actuator_force_declarations,
            netforce_declarations,
        ) = _collect_sensor_declarations(scene.model_file, scene.fragment_files)
        self._frame_quat_sensor_names = frozenset(frame_quat_sensor_names)
        self._frame_quat_public_views: dict[str, Any] = {}
        self._frame_quat_permutations: dict[Any, Any] = {}
        self._contact_sensor_names = tuple(contact_declarations)
        self._contact_sensor_slots = {
            name: slot for slot, name in enumerate(self._contact_sensor_names)
        }
        self._contact_pairs_np = self._resolve_contact_pairs(contact_declarations)
        self._contact_pairs: Any = None
        self._contact_results: Any = None
        self._sensor_torch_views: dict[str, Any] = {}
        self._tracked_body_names = self._link_names
        self._tracked_body_slots = {
            name: index for index, name in enumerate(self._tracked_body_names) if name
        }
        self._tracked_sensor_slots = {
            f"{prefix}{name}": (field, int(width), index)
            for prefix, field, width, _wxyz in _TRACKED_SENSOR_LAYOUT
            for index, name in enumerate(self._tracked_body_names)
            if name
        }
        self._tracked_sensor_names = tuple(self._tracked_sensor_slots)
        self._published_sensor_names = self._sensor_names + self._tracked_sensor_names
        self._tracked_buffers: dict[str, Any] = {}
        self._tracked_field_views: dict[str, dict[int, Any]] = {}
        # Batched tracked-sensor republish caches (resolved on first use).
        self._tracked_packed: dict[str, Any] | None = None
        self._tracked_column_indices: Any = None
        self._tracked_quat_stage: Any = None
        self._tracked_wxyz_perm: Any = None
        self._public_column_index: Any = None
        self._public_source_index: Any = None
        self._engine_source_index: Any = None
        # Hot-path caches: the engine's addressable sensor-name table and the
        # inventory verification are immutable after model build, and
        # tracked-body column tensors are requested with a fixed name set per
        # read plan. Rebuilding any of these per ``get_sensor_view`` call
        # dominated the collector's host time (py-spy: ~30% of samples).
        self._native_sensor_name_cache: frozenset[str] | None = None
        self._reset_verdicts = DeviceVerdicts()
        self._reset_verdicts.ensure(
            "state", device=self._device, message="motrix_gpu reset state must be finite"
        )
        self._link_inertia_indices: tuple[Any, ...] | None = None
        self._sensor_inventory_verified = False
        self._tracked_body_column_cache: dict[tuple[str, ...], Any] = {}
        self._joint_actuator_slots = self._read_joint_actuator_slots()
        self._initialize_gain_views()
        self._initialize_actuator_force_sensors(actuator_force_declarations)
        (
            self._quat_columns,
            self._quat_public_sources,
            self._quat_engine_sources,
        ) = self._read_root_quaternion_slices()
        self._capture_import_report()
        self._initialize_override_support()
        self._initialize_contact_force_sensors(netforce_declarations)
        self._ctrl_public = torch.zeros(
            (self._num_envs, self._nu), dtype=torch.float32, device=self._device
        )
        self._initialize_gravity_compensation()
        qpos_device = self._field_view("qpos").device
        for term in sorted(self._supported_reset_terms):
            self._reset_verdicts.ensure(
                term,
                device=qpos_device,
                message=f"motrix_gpu device reset term {term!r} must be finite",
            )
        if RESET_TERM_BODY_IPOS in self._supported_reset_terms:
            import torch

            self._prepare_link_inertia_indices(qpos_device, torch)
        self._publish_public_qpos()
        self._refresh_tracked_sensors()
        self._refresh_contact_sensors()
        self._refresh_actuator_forces()

    # ------------------------------------------------------------------ #
    # Cold metadata                                                       #
    # ------------------------------------------------------------------ #

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def model(self) -> Any:
        """Return the backend-owned Warp model, never a CPU adapter model."""
        return self._model

    @property
    def num_actuators(self) -> int:
        return self._nu

    @property
    def num_dof_vel(self) -> int:
        return self._nv

    def get_actuator_ctrl_range(self) -> np.ndarray:
        return self._ctrl_range.copy()

    def get_init_qvel(self) -> np.ndarray:
        return np.zeros((self._nv,), dtype=np.float32)

    def get_keyframe_qpos(self, name: str) -> np.ndarray:
        keyframe = self._find_keyframe(name)
        qpos = np.array(np.asarray(keyframe.dof_pos, dtype=np.float32), copy=True)
        permutation = self._root_quaternion_permutation()
        if permutation is not None:
            columns, public_sources, _ = permutation
            qpos[list(columns)] = qpos[list(public_sources)]
        return qpos

    def get_body_ids(self, names: Sequence[str]) -> np.ndarray:
        ids = []
        for name in names:
            # MotrixSim's ``body`` space holds only the articulation root; the
            # backend-neutral body/link namespace is the link space, matching
            # the CPU ``motrix`` adapter.
            index = self._model.get_link_index(name)
            if index is None or int(index) < 0:
                raise ValueError(f"Body {name!r} not found in motrix_gpu model")
            ids.append(int(index))
        return np.asarray(ids, dtype=np.intp)

    def get_joint_range(self, *, names: Sequence[str] | None = None) -> np.ndarray | None:
        """Return ``(num_dof, 2)`` joint position limits in joint order.

        MotrixSim stores the limits as ``(2, num_joints)`` ``[low, high]`` rows
        and excludes the floating base from its joint list, which is exactly the
        backend-neutral contract. Single-DoF joints keep one row per joint, so
        the joint index and the joint-section DoF index agree.
        """
        self._require_open()
        limits = np.asarray(self._model.joint_limits, dtype=np.float32)
        if limits.ndim != 2 or limits.shape[0] != 2:
            raise RuntimeError(
                f"motrix_gpu joint limits must be a (2, num_joints) table, got shape {limits.shape}"
            )
        pairs = np.stack((limits[0], limits[1]), axis=1)
        if names is None:
            return pairs
        indices = [int(self._resolve_single_dof_joint(name).index) for name in names]
        return pairs[indices]

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        """Advertise the reset families and interval kicks this engine can realize."""
        interval_terms: frozenset[str] = frozenset()
        if self._root_body_id is not None and self._root_qvel_start is not None:
            interval_terms = frozenset(
                {
                    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA,
                    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA,
                }
            )
        return DomainRandomizationCapabilities(
            # unisim >= 2.8 declares term support independent of payload device;
            # this adapter realizes every declared term with device-resident
            # payload tensors (host payloads, if unilab hands them over, are
            # converted at the payload boundary in _resolve_reset_randomization).
            supported_reset_terms=frozenset(self._supported_reset_terms),
            supported_interval_terms=interval_terms,
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        """Return a detached baseline including any committed per-world overrides."""
        _validate_reset_term(term)
        if term not in self._supported_reset_terms:
            raise NotImplementedError(
                f"MotrixGpuBackend does not support reset term {term!r}: the installed "
                "motrixsim.gpu build does not expose its per-world override buffer"
            )
        self._require_open()
        if term in (RESET_TERM_KP, RESET_TERM_KD):
            values = np.zeros((self._num_envs, self._nu), dtype=np.float32)
            if self._gain_columns_np.size:
                values[:, self._gain_columns_np] = (
                    self._gain_views[term]
                    .detach()
                    .cpu()
                    .numpy()[:, self._joint_actuator_slots[self._gain_columns_np]]
                )
        elif term == RESET_TERM_BODY_MASS:
            values = self._override_views["apparent_link_mass"].detach().cpu().numpy()
        elif term == RESET_TERM_BODY_IPOS:
            values = self._override_views["apparent_link_com"].detach().cpu().numpy()
        elif term == RESET_TERM_GEOM_FRICTION:
            friction = self._override_views["collider_friction"].detach().cpu().numpy()
            values = np.zeros((self._num_envs, self._num_geoms, 3), dtype=np.float32)
            for geom, colliders in enumerate(self._geom_collider_indices):
                if colliders:
                    values[:, geom, :] = friction[:, colliders[0], :]
        else:
            raise NotImplementedError(f"MotrixGpuBackend does not expose reset term {term!r}")
        if np.array_equal(values, np.broadcast_to(values[0], values.shape)):
            values = values[0]
        result = np.array(values, dtype=np.float32, copy=True)
        result.setflags(write=False)
        return result

    def _read_joint_actuator_slots(self) -> np.ndarray:
        """Map every control index to its joint-actuator override column.

        MotrixSim stores the flat ``kp``/``kd`` overrides per joint actuator, so
        tendon- and body-targeted actuators have no column (sentinel ``-1``).
        """
        slots = np.full((self._nu,), -1, dtype=np.intp)
        for index, actuator in enumerate(self._model.actuators):
            slot = getattr(actuator, "_joint_slot", None)
            if slot is not None:
                slots[index] = int(slot)
        return slots

    def _initialize_gain_views(self) -> None:
        import torch

        slots = self._joint_actuator_slots
        self._gain_columns_np = np.flatnonzero(slots >= 0)
        self._gain_columns = torch.as_tensor(self._gain_columns_np, device=self._device)
        self._gain_slots = torch.as_tensor(slots[self._gain_columns_np], device=self._device)
        self._non_joint_columns = torch.as_tensor(np.flatnonzero(slots < 0), device=self._device)
        to_torch = self._deps.warp.to_torch
        self._gain_views = {
            RESET_TERM_KP: to_torch(self._data.actuator_kp_overrides),
            RESET_TERM_KD: to_torch(self._data.actuator_damping_overrides),
        }
        self._override_views: dict[str, Any] = {}

    def _initialize_actuator_force_sensors(
        self, declarations: Mapping[str, tuple[str, str]]
    ) -> None:
        """Bind ``<actuatorfrc>``/``<jointactuatorfrc>`` sensors to actuator columns.

        The GPU runtime has no actuator-force sensor stage. MuJoCo's
        ``actuatorfrc`` is the position servo's affine force on the published
        state, ``kp * (ctrl - q) - kd * qd``, so the adapter republishes exactly
        that from the live state and per-world gain overrides. A
        ``<jointactuatorfrc>`` aggregate sums the unit-gear forces of every
        actuator driving its single-DoF joint. Every other form fails closed.
        """
        import torch

        self._actuator_force_sensor_kinds: dict[str, str] = {}
        self._actuator_force_sensor_slots: dict[str, int] = {}
        self._actuator_force_storage: Any = None
        self._actuator_force_aggregate: Any = None
        self._actuator_force_columns: Any = None
        self._actuator_qpos_columns: Any = None
        self._actuator_qvel_columns: Any = None
        if not declarations:
            return
        actuator_joints = list(self.get_actuator_joint_names())
        columns = self._gain_columns_np
        members_by_sensor: list[tuple[int, ...]] = []
        for name, (kind, target) in declarations.items():
            if kind == "actuatorfrc":
                members = [
                    index
                    for index, actuator_name in enumerate(self._actuator_names)
                    if actuator_name == target
                ]
                if len(members) != 1:
                    raise NotImplementedError(
                        f"motrix_gpu actuatorfrc sensor {name!r} requires one named actuator, "
                        f"got {target!r}"
                    )
            else:
                self._resolve_single_dof_joint(target)
                members = [index for index, joint in enumerate(actuator_joints) if joint == target]
                if not members:
                    raise NotImplementedError(
                        f"motrix_gpu jointactuatorfrc sensor {name!r} requires actuators on "
                        f"joint {target!r}"
                    )
            if any(self._joint_actuator_slots[index] < 0 for index in members):
                raise NotImplementedError(
                    f"motrix_gpu {kind} sensor {name!r} requires joint-transmission actuators"
                )
            members_by_sensor.append(tuple(members))
            self._actuator_force_sensor_kinds[name] = kind
            self._actuator_force_sensor_slots[name] = len(members_by_sensor) - 1
        # Column i of the aggregation matrix belongs to ``columns[i]``.
        aggregate = np.zeros((len(columns), len(members_by_sensor)), dtype=np.float32)
        for sensor, member_indices in enumerate(members_by_sensor):
            for index in member_indices:
                aggregate[int(np.flatnonzero(columns == index)[0]), sensor] = 1.0
        joint_targets = [
            self._model.get_actuator(self._actuator_names[index]).target_name for index in columns
        ]
        self._actuator_force_columns = self._gain_columns
        self._actuator_qpos_columns = torch.as_tensor(
            [int(self._resolve_single_dof_joint(joint).dof_pos_index) for joint in joint_targets],
            dtype=torch.long,
            device=self._device,
        )
        self._actuator_qvel_columns = torch.as_tensor(
            [int(self._resolve_single_dof_joint(joint).dof_vel_index) for joint in joint_targets],
            dtype=torch.long,
            device=self._device,
        )
        self._actuator_force_aggregate = torch.as_tensor(aggregate, device=self._device)
        self._actuator_force_storage = torch.zeros(
            (self._num_envs, len(members_by_sensor)), dtype=torch.float32, device=self._device
        )

    def _refresh_actuator_forces(self) -> None:
        """Republish the affine servo forces from the current state and gains."""
        import torch

        if self._actuator_force_storage is None:
            return
        columns = self._actuator_force_columns
        error = self._ctrl_public.index_select(1, columns) - self._field_view("qpos").index_select(
            1, self._actuator_qpos_columns
        )
        velocity = self._field_view("qvel").index_select(1, self._actuator_qvel_columns)
        force = (
            self._gain_views[RESET_TERM_KP].index_select(1, self._gain_slots) * error
            - self._gain_views[RESET_TERM_KD].index_select(1, self._gain_slots) * velocity
        )
        torch.matmul(force, self._actuator_force_aggregate, out=self._actuator_force_storage)

    def _initialize_contact_force_sensors(
        self, declarations: Mapping[str, tuple[str, str, str]]
    ) -> None:
        """Bind ``<contact data="force" reduce="netforce">`` sensors to collider sets.

        The GPU runtime has no contact-force sensor stage, but its solver keeps
        one normal impulse per contact point. The adapter republishes each
        sensor as the normal contact force transmitted to the filter's second
        side, in newtons and in the world frame. The runtime publishes no
        tangential impulses, so friction components are not included.
        """
        import torch

        self._contact_force_slots: dict[str, int] = {}
        self._contact_force_storage: Any = None
        self._contact_force_views: dict[str, Any] = {}
        self._contact_force_layout_verified = False
        if not declarations:
            return
        to_torch = self._deps.warp.to_torch
        warp = self._deps.warp
        link_colliders: dict[str, list[int]] = {}
        geom_colliders: dict[str, list[int]] = {}
        for geom, colliders in zip(self._geoms, self._geom_collider_indices, strict=True):
            link = getattr(geom, "link", None)
            name = "" if geom.name is None else str(geom.name)
            link_name = "" if link is None else str(link.name)
            link_colliders.setdefault(link_name, []).extend(colliders)
            geom_colliders.setdefault(name, []).extend(colliders)
        num_colliders = int(self._model.low.colliders.masks.shape[0])
        first = torch.zeros(
            (len(declarations), num_colliders), dtype=torch.bool, device=self._device
        )
        second = torch.zeros_like(first)
        for slot, (name, (kind, left, right)) in enumerate(declarations.items()):
            table = geom_colliders if kind == "geom" else link_colliders
            if left not in table or right not in table:
                raise NotImplementedError(
                    f"motrix_gpu netforce sensor {name!r} refers to unknown {kind} "
                    f"{left!r}/{right!r}"
                )
            first[slot, table[left]] = True
            second[slot, table[right]] = True
            self._contact_force_slots[name] = slot
        self._contact_force_first = first.t().contiguous()
        self._contact_force_second = second.t().contiguous()
        domains = self._data.low.domains
        rigid = domains.get("rigid_collision")
        solver = domains.get("solver")
        contacts = rigid.contacts
        rows = solver.persistent_lambdas.rows
        self._contact_capacity = int(contacts.shape[1])
        self._row_capacity = int(rows.shape[1])
        self._unilateral_rows = int(solver.constraint_lambdas.unilateral.shape[1])
        if self._contact_capacity > self._row_capacity:
            raise NotImplementedError(
                "motrix_gpu netforce sensors need one constraint row per contact slot"
            )
        # The engine's NumPy view of its arrays carries the struct field layout.
        contact_dtype = contacts.numpy().dtype
        row_dtype = rows.numpy().dtype
        self._contact_words = contact_dtype.itemsize // 4
        self._row_words = row_dtype.itemsize // 4
        self._pair_word = contact_dtype.fields["pair_index"][1] // 4
        self._normal_word = contact_dtype.fields["normal"][1] // 4
        self._line_word = row_dtype.fields["lineno"][1] // 4
        self._axis_word = row_dtype.fields["axis"][1] // 4
        line_bit = (row_dtype.fields["lineno"][1] % 4) * 8
        axis_bit = (row_dtype.fields["axis"][1] % 4) * 8
        self._line_shift, self._axis_shift = line_bit, axis_bit
        self._contact_count = to_torch(rigid.contact_count)
        self._row_counter = to_torch(solver.persistent_lambdas.row_counter)
        self._unilateral = to_torch(solver.constraint_lambdas.unilateral)
        contact_numel = self._num_envs * self._contact_capacity * self._contact_words
        self._contacts_int = self._flat_torch_view(
            "contact_force_contacts_int", contacts, contact_numel, torch, dtype=warp.int32
        )
        self._contacts_float = self._flat_torch_view(
            "contact_force_contacts_float", contacts, contact_numel, torch
        )
        self._constraint_rows = self._flat_torch_view(
            "contact_force_rows",
            rows,
            self._num_envs * self._row_capacity * self._row_words,
            torch,
            dtype=warp.int32,
        )
        self._contact_ordinal = torch.arange(self._contact_capacity, device=self._device)
        self._contact_force_storage = torch.zeros(
            (self._num_envs, len(declarations), 3), dtype=torch.float32, device=self._device
        )

    def _verify_contact_force_layout(self) -> None:
        """Check once, on the host, how the runtime numbers contact constraint rows.

        ``persistent_lambdas.rows`` lists one row per contact axis and its
        ``lineno`` indexes ``constraint_lambdas.unilateral``. The numbering
        changes with the constraint capacity (contact-major with the engine's
        large-capacity kernels, axis-major below), so the adapter only relies on
        what both share: the first two listed rows are the normal rows of
        contacts 0 and 1, normal rows are numbered with a constant stride, and
        every contact has the same axis count.
        """
        rows = self._constraint_rows.view(self._num_envs, self._row_capacity, self._row_words)
        counts = self._contact_count.cpu().numpy()
        live = self._row_counter.cpu().numpy()
        verified = False
        for world in range(self._num_envs):
            count = int(counts[world])
            if count < 2:
                continue
            verified = True
            live_rows = rows[world, : int(live[world])].cpu()
            axis = ((live_rows[:, self._axis_word] >> self._axis_shift) & 0xFF).numpy()
            line = ((live_rows[:, self._line_word] >> self._line_shift) & 0xFFFF).numpy()
            axes, remainder = divmod(int(live_rows.shape[0]), count)
            normals = np.sort(line[axis == 0])
            stride = int(line[1]) - int(line[0])
            if (
                remainder
                or axes < 1
                or bool(axis[:2].any())
                or normals.shape[0] != count
                or not np.array_equal(normals, int(line[0]) + stride * np.arange(count))
                or int(line.max()) >= self._unilateral_rows
            ):
                raise NotImplementedError(
                    "motrix_gpu netforce sensors require one uniform axis block per "
                    f"contact: contacts={count}, rows={int(live_rows.shape[0])}, "
                    f"axes={np.bincount(axis, minlength=8).tolist()}, "
                    f"lines={np.sort(line).tolist()}"
                )
        if not verified:
            return
        self._contact_force_layout_verified = True

    def _refresh_contact_forces(self) -> None:
        """Republish netforce contact sensors from the solver's normal impulses."""
        import torch

        if self._contact_force_storage is None:
            return
        if not self._contact_force_layout_verified:
            self._verify_contact_force_layout()
        worlds = self._num_envs
        capacity = self._contact_capacity
        contacts_int = self._contacts_int.view(worlds, capacity, self._contact_words)
        contacts_float = self._contacts_float.view(worlds, capacity, self._contact_words)
        rows = self._constraint_rows.view(worlds, self._row_capacity, self._row_words)
        axis = (rows[:, :, self._axis_word] >> self._axis_shift) & 0xFF
        line = (rows[:, :, self._line_word] >> self._line_shift) & 0xFFFF
        count = self._contact_count.long()
        # The first two listed rows are the normal rows of contacts 0 and 1 in
        # both numberings the runtime uses, so their distance is the stride
        # between consecutive normal rows; see _verify_contact_force_layout.
        stride = (line[:, 1] - line[:, 0]).clamp(min=1)
        ordinal = self._contact_ordinal
        line_of_contact = line[:, 0:1] + stride[:, None] * ordinal[None, :]
        impulse = self._unilateral.gather(1, line_of_contact.clamp(max=self._unilateral_rows - 1))
        impulse = torch.where(
            (ordinal[None, :] < count[:, None]) & (axis[:, 0:1] == 0) & (axis[:, 1:2] == 0),
            impulse,
            0.0,
        )

        def members(table: Any, ids: Any) -> Any:
            return table[ids.clamp(0, table.shape[0] - 1).long()]

        first_pair = contacts_int[:, :, self._pair_word]
        second_pair = contacts_int[:, :, self._pair_word + 1]
        first, second = self._contact_force_first, self._contact_force_second
        # The force transmitted to the filter's second side: the contact normal
        # points from the first listed collider toward the second.
        sign = (members(first, first_pair) & members(second, second_pair)).float() - (
            members(second, first_pair) & members(first, second_pair)
        ).float()
        normal = contacts_float[:, :, self._normal_word : self._normal_word + 3]
        force = torch.einsum("wcs,wcf->wsf", sign * impulse[..., None], normal)
        self._contact_force_storage.copy_(force / self._sim_dt)

    def _initialize_gravity_compensation(self) -> None:
        """Cancel the weight of MuJoCo ``gravcomp`` links through their position servos.

        The GPU runtime has no gravity compensation: the weight of every link
        enters ``bias_dof_force``. For each single-DoF joint below ``gravcomp``
        links the adapter shifts the position servo's target by ``-tau / kp``,
        where ``tau`` is the gravity generalized force of the compensated
        subtree. An affine servo turns that shift into ``-tau`` of joint force,
        which is what MuJoCo's compensation applies. A joint that carries
        compensated weight without a position servo fails closed.
        """
        import torch

        self._gravcomp: dict[str, Any] | None = None
        weight: dict[str, float] = {}
        parent_of: dict[str, str | None] = {}

        def walk(link: Any, parent: Any) -> None:
            name = "" if link.name is None else str(link.name)
            parent_of[name] = None if parent is None else str(parent.name)
            weight[name] = float(link.gravcomp)
            for child in link.children:
                walk(child, link)

        for body in self._scene_world.hierarchy.bodies:
            walk(body.link, None)
        index = {name: position for position, name in enumerate(self._link_names)}

        def parent_index(name: str) -> int:
            parent_name = parent_of.get(name)
            return index[parent_name] if parent_name is not None and parent_name in index else -1

        parent = np.asarray(
            [parent_index(name) for name in self._link_names],
            dtype=np.int64,
        )
        weights = np.zeros(len(self._link_names), dtype=np.float32)
        for name, value in weight.items():
            if name in index:
                weights[index[name]] = value
        nominal_mass = self._override_views["apparent_link_mass"][0].detach().cpu().numpy()
        compensated = (weights != 0.0) & (nominal_mass > 0.0)
        if not compensated.any():
            return
        joints = self._model.low.joints
        axes = joints.all_scalar_joints.numpy()
        kinds = joints.joint_types.numpy()
        addresses = joints.dof_addrs.numpy()
        # Joint kinds: 0 revolute, 1 prismatic, 2 spherical, 3 translational (3-DoF).
        scalar = {
            int(address["vel"]): (axis["axis"].astype(np.float32), int(kind) == 0)
            for axis, kind, address in zip(axes, kinds, addresses, strict=True)
        }
        dof_axes: list[np.ndarray] = []
        dof_revolute: list[bool] = []
        dof_links: list[int] = []
        dof_slots: list[int] = []
        columns: list[int] = []
        compensated_dofs: set[int] = set()
        for column, actuator_index in enumerate(self._gain_columns_np):
            actuator = self._model.get_actuator(self._actuator_names[actuator_index])
            joint = self._model.get_joint(str(actuator.target_name))
            dof_vel = None if joint is None else int(joint.dof_vel_index)
            if dof_vel not in scalar or int(joint.num_dof_vel) != 1:
                continue
            axis, revolute = scalar[dof_vel]
            dof_axes.append(axis)
            dof_revolute.append(revolute)
            dof_links.append(index[str(joint.link.name)])
            dof_slots.append(int(self._joint_actuator_slots[actuator_index]))
            columns.append(column)
            compensated_dofs.add(dof_vel)
        for link_index in np.flatnonzero(compensated):
            node = int(link_index)
            while node >= 0:
                link = self._model.get_link(self._link_names[node])
                for joint_index in link.joint_indices:
                    joint = self._model.joints[joint_index]
                    if int(joint.dof_vel_index) not in compensated_dofs:
                        raise NotImplementedError(
                            "motrix_gpu cannot cancel the weight of gravcomp links below "
                            f"joint {joint.name!r}: it needs a single-DoF position servo"
                        )
                node = int(parent[node])
        depth = np.zeros(len(self._link_names), dtype=np.int64)
        for node in range(len(depth)):
            walk_node = int(parent[node])
            while walk_node >= 0:
                depth[node] += 1
                walk_node = int(parent[walk_node])
        levels = []
        for level in range(int(depth.max()), 0, -1):
            children = np.flatnonzero(depth == level)
            levels.append(
                (
                    torch.as_tensor(children, dtype=torch.long, device=self._device),
                    torch.as_tensor(parent[children], dtype=torch.long, device=self._device),
                )
            )
        column_index = torch.as_tensor(columns, dtype=torch.long, device=self._device)
        self._gravcomp = {
            "weight": torch.as_tensor(weights, device=self._device),
            "levels": levels,
            "axis": torch.as_tensor(np.stack(dof_axes), device=self._device),
            "revolute": torch.as_tensor(dof_revolute, device=self._device),
            "link": torch.as_tensor(dof_links, dtype=torch.long, device=self._device),
            "slots": torch.as_tensor(dof_slots, dtype=torch.long, device=self._device),
            "columns": self._gain_columns.index_select(0, column_index),
            "gravity": torch.as_tensor(self._gravity, dtype=torch.float32, device=self._device),
            "shift": torch.zeros(
                (self._num_envs, self._nu), dtype=torch.float32, device=self._device
            ),
        }

    def _apply_gravity_compensation(self, ctrl_view: Any) -> None:
        """Write the servo targets that cancel the compensated weight into the engine."""
        import torch

        state = self._gravcomp
        if state is None:
            ctrl_view.copy_(self._ctrl_public)
            return
        packed = self._tracked_packed_views()
        if packed is None:
            raise RuntimeError("motrix_gpu gravity compensation needs the packed link poses")
        mass = self._override_views["apparent_link_mass"] * state["weight"]
        poses = packed["poses"]
        position = poses[..., :3]
        # The engine stores link quaternions as xyzw.
        x, y, z, w = (poses[..., column : column + 1] for column in (3, 4, 5, 6))
        rows = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        rotation = torch.cat([torch.cat(row, -1).unsqueeze(-2) for row in rows], -2)
        com = self._override_views["apparent_link_com"]
        total = mass
        first = mass[..., None] * (position + (rotation @ com[..., None]).squeeze(-1))
        for children, parents in state["levels"]:
            total = total.index_add(1, parents, total[:, children])
            first = first.index_add(1, parents, first[:, children])
        link = state["link"]
        axis = (rotation[:, link] @ state["axis"][None, :, :, None]).squeeze(-1)
        revolute = state["revolute"][None, :, None]
        omega = torch.where(revolute, axis, torch.zeros_like(axis))
        velocity = torch.where(revolute, torch.cross(position[:, link], omega, dim=-1), axis)
        torque = (
            state["gravity"]
            * (torch.cross(omega, first[:, link], dim=-1) + total[:, link, None] * velocity)
        ).sum(-1)
        kp = self._gain_views[RESET_TERM_KP].index_select(1, state["slots"])
        shift = state["shift"]
        shift.zero_()
        shift.index_copy_(1, state["columns"], -torque / kp)
        torch.add(self._ctrl_public, shift, out=ctrl_view)

    def _initialize_override_support(self) -> None:
        """Detect the engine's per-world override buffers and seed defaults.

        The adapter must not infer support from the Python API alone: the
        per-world buffers are what make selected-row writes possible.  Each
        family therefore stays fail-closed unless its materialized buffer and
        engine write path are both present in the installed build.
        """
        overrides = self._low_overrides()
        self._links = list(self._model.links)
        self._geoms = list(self._model.geoms)
        self._num_links = len(self._links)
        self._num_geoms = len(self._geoms)

        mass_buffer = getattr(overrides, "apparent_link_mass", None)
        com_buffer = getattr(overrides, "apparent_link_com", None)
        rbi_buffer = getattr(overrides, "link_rbi", None)
        spatial_buffer = getattr(overrides, "link_spatial_inertia", None)
        supports_link_inertia = bool(
            self._num_links
            and mass_buffer is not None
            and com_buffer is not None
            and rbi_buffer is not None
            and spatial_buffer is not None
            and callable(getattr(self._links[0], "set_mass_override", None))
            and callable(getattr(self._links[0], "set_center_of_mass_override", None))
        )
        friction_buffer = getattr(overrides, "collider_friction", None)
        supports_geom_friction = bool(
            self._num_geoms
            and friction_buffer is not None
            and callable(getattr(self._geoms[0], "set_friction_override", None))
        )
        if supports_link_inertia:
            self._supported_reset_terms.update({RESET_TERM_BODY_MASS, RESET_TERM_BODY_IPOS})
        if supports_geom_friction:
            self._supported_reset_terms.add(RESET_TERM_GEOM_FRICTION)

        self._geom_collider_indices = tuple(
            tuple(int(index) for index in geom._collider_indices()) for geom in self._geoms
        )

        for name, buffer in (
            ("apparent_link_mass", mass_buffer),
            ("apparent_link_com", com_buffer),
            ("collider_friction", friction_buffer),
        ):
            if buffer is not None:
                self._override_views[name] = self._deps.warp.to_torch(buffer)

        self._link_low_indices = np.full((self._num_links,), -1, dtype=np.intp)
        self._link_low_rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32), (self._num_links, 3, 3)
        ).copy()
        self._link_low_translations = np.zeros((self._num_links, 3), dtype=np.float32)
        for link_index in range(self._num_links):
            low_index = self._model._low_link_index_for_high_link(link_index)
            if low_index is None:
                continue
            self._link_low_indices[link_index] = int(low_index)
            rotation, translation = self._model._apparent_to_low_transform(link_index)
            self._link_low_rotations[link_index] = np.asarray(rotation, dtype=np.float32)
            self._link_low_translations[link_index] = np.asarray(translation, dtype=np.float32)

        # Interval velocity terms are advertised only when the configured base
        # owns a floating root; a fixed base has no root velocity to perturb.
        self._root_qpos_start: int | None = None
        self._root_qvel_start: int | None = None
        self._root_body_id: int | None = None
        root_name = self._base_name
        if root_name is None and len(self._floating_roots) == 1:
            root_name = next(iter(self._floating_roots))
        root_base = self._floating_roots.get(root_name) if root_name is not None else None
        if root_base is not None and root_name in self._link_names:
            self._root_qpos_start = int(root_base.dof_pos_start)
            self._root_qvel_start = int(root_base.dof_vel_start)
            self._root_body_id = self._link_names.index(root_name)

    def _low_overrides(self) -> Any:
        """Return override buffers across legacy and domain-owned LowData APIs."""
        overrides = getattr(self._data.low, "overrides", None)
        if overrides is not None:
            return overrides
        from motrixsim.gpu.overrides.data import OverridesData

        return self._data.low.domains.require("overrides", OverridesData)

    def _flat_torch_view(
        self, name: str, array: Any, numel: int, torch: Any, dtype: Any = None
    ) -> Any:
        """Reinterpret a struct-typed Warp array as a flat Torch tensor.

        DLPack cannot describe Warp user structs, so the adapter mirrors the
        engine's own ``_flat_float_view``: a zero-copy ``wp.array`` of ``dtype``
        (float32 by default) aliases the original pointer and is kept alive in
        ``_struct_flat_views``.
        """
        flat = self._struct_flat_views.get(name)
        if flat is None:
            warp = self._deps.warp
            flat = warp.array(
                ptr=array.ptr,
                shape=(numel,),
                dtype=warp.float32 if dtype is None else dtype,
                device=array.device,
            )
            flat._ref = array  # noqa: SLF001 - same parent-ref trick wp.array.view() uses
            self._struct_flat_views[name] = flat
        return torch.from_dlpack(flat)  # pyright: ignore[reportPrivateImportUsage]

    def _read_root_quaternion_slices(
        self,
    ) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None, tuple[int, ...] | None]:
        """Resolve the public/engine root-quaternion column permutation.

        MotrixSim stores a floating base's orientation as ``[x, y, z, w]``
        (``FloatingBase.set_rotation`` documents ``[i, j, k, w]``), while the
        public MuJoCo-derived state layout used by every other adapter is
        ``[w, x, y, z]``.  Only those four columns differ; the remaining root
        coordinates are identical, so the adapter keeps a stable public device
        mirror and permutes in place at the two state boundaries.
        """
        bases = list(getattr(self._model, "floating_bases", ()))
        if not bases:
            return None, None, None
        # Every floating base permutes its own four columns; the column sets of
        # distinct bases are disjoint, so the tables concatenate.
        columns: list[int] = []
        public_sources: list[int] = []
        engine_sources: list[int] = []
        for base in bases:
            start = int(base.dof_pos_start)
            columns.extend((start + 3, start + 4, start + 5, start + 6))
            public_sources.extend((start + 6, start + 3, start + 4, start + 5))
            engine_sources.extend((start + 4, start + 5, start + 6, start + 3))
        return tuple(columns), tuple(public_sources), tuple(engine_sources)

    def _root_quaternion_permutation(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        """Return the coupled root-quaternion tables, or ``None`` without a root."""
        columns = self._quat_columns
        if columns is None:
            return None
        public_sources = self._quat_public_sources
        engine_sources = self._quat_engine_sources
        assert public_sources is not None and engine_sources is not None
        return columns, public_sources, engine_sources

    def _public_qpos_view(self) -> Any:
        """Return the stable public (``wxyz`` root) qpos device mirror."""
        import torch

        if self._qpos_public is None:
            self._qpos_public = torch.empty_like(  # pyright: ignore[reportPrivateImportUsage]
                self._field_view("qpos")
            )
        return self._qpos_public

    def _publish_public_qpos(self) -> None:
        """Refresh the public qpos mirror from the engine's current state."""
        mirror = self._public_qpos_view()
        engine = self._field_view("qpos")
        mirror.copy_(engine)
        permutation = self._root_quaternion_permutation()
        if permutation is not None:
            self._prepare_qpos_indices(mirror.device)
            mirror.index_copy_(
                1,
                self._public_column_index,
                engine.index_select(1, self._public_source_index),
            )

    def _prepare_qpos_indices(self, device: Any) -> None:
        import torch

        permutation = self._root_quaternion_permutation()
        if permutation is None or (
            self._public_column_index is not None and self._public_column_index.device == device
        ):
            return
        columns, public_sources, engine_sources = permutation
        self._public_column_index = torch.as_tensor(columns, dtype=torch.long, device=device)
        self._public_source_index = torch.as_tensor(public_sources, dtype=torch.long, device=device)
        self._engine_source_index = torch.as_tensor(engine_sources, dtype=torch.long, device=device)

    def _engine_qpos_from_public(self, values: Any) -> Any:
        """Return engine-order (``xyzw`` root) qpos for a public-order input."""
        self._prepare_qpos_indices(values.device)
        if self._engine_source_index is None:
            return values
        staged = values.clone()
        staged.index_copy_(
            1, self._public_column_index, values.index_select(1, self._engine_source_index)
        )
        return staged

    def _public_qpos(self) -> np.ndarray:
        """Copy the current public qpos to host NumPy order."""
        engine = np.array(self._data.dof_pos.numpy(), dtype=np.float32, copy=True)
        permutation = self._root_quaternion_permutation()
        if permutation is not None:
            columns, public_sources, _ = permutation
            engine[:, list(columns)] = engine[:, list(public_sources)]
        return engine

    def _read_ctrl_range(self) -> np.ndarray:
        limits = getattr(self._model, "actuator_ctrl_limits", None)
        if limits is None:
            return np.tile(np.asarray([[-np.inf], [np.inf]], dtype=np.float32), (1, self._nu))
        # The engine stores the bounds as a ``(2, num_actuators)``
        # ``[low, high]`` table; the public contract is ``(num_actuators, 2)``.
        array = np.asarray(limits, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] != self._nu:
            raise RuntimeError(
                "motrix_gpu actuator control limits must be a (2, num_actuators) table, "
                f"got shape {array.shape}"
            )
        return np.ascontiguousarray(array.T)

    def _capture_import_report(self) -> None:
        """Record the cold-path effective configuration (dt/gravity readback)."""
        from unisim.inspection import (
            REPORT_FIELDS,
            ConfigurationField,
            ConfigurationProvenance,
            ImportReport,
        )

        readback = ConfigurationProvenance(
            kind="engine_readback", source="motrixsim.gpu Warp SceneModel"
        )
        fields = [
            ConfigurationField(
                field="dt",
                # The engine stores the timestep as float32; compare in that
                # representation so the adapter-setting vs readback is exact.
                requested=float(np.float32(self._sim_dt)),
                effective=float(self._model.timestep),
                difference="exact",
                provenance=(readback,),
                unit="s",
            ),
            ConfigurationField(
                field="gravity",
                requested=tuple(self._gravity.tolist()),
                effective=tuple(self._gravity.tolist()),
                difference="exact",
                provenance=(readback,),
                unit="m/s^2",
                frame="world",
            ),
            *(
                ConfigurationField(
                    field=name,
                    reason="Adapter has not provided this effective configuration.",
                )
                for name in REPORT_FIELDS
                if name not in {"dt", "gravity"}
            ),
        ]
        self._import_report = ImportReport(backend=self.backend_type, fields=tuple(fields))

    def _find_keyframe(self, name: str) -> Any:
        finder = getattr(self._model, "find_keyframe", None)
        if finder is None:
            raise NotImplementedError("motrix_gpu model does not expose keyframes")
        keyframe = finder(name)
        if keyframe is None:
            raise ValueError(f"Keyframe {name!r} not found")
        return keyframe

    def get_scene_model_file(self) -> str | None:
        return self._scene_model_file

    def get_geom_names(self) -> tuple[str, ...]:
        return self._geom_names

    def get_geom_body_ids(self) -> np.ndarray:
        """Return the owning link id for each geom; -1 for world-owned geoms."""
        return np.asarray(
            [
                -1 if getattr(geom, "link", None) is None else int(geom.link.index)
                for geom in self._geoms
            ],
            dtype=np.int32,
        )

    def get_geom_contact_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-geom collision group and affinity from the engine colliders.

        MJCF ``contype``/``conaffinity`` import as the collider ``group`` and
        ``collide_with`` masks. Visual-only geoms own no collider and report 0.
        """
        masks = self._model.low.colliders.masks.numpy()
        contype = np.zeros((self._num_geoms,), dtype=np.int32)
        conaffinity = np.zeros((self._num_geoms,), dtype=np.int32)
        for geom_id, colliders in enumerate(self._geom_collider_indices):
            if colliders:
                contype[geom_id] = int(masks[colliders[0]]["group"])
                conaffinity[geom_id] = int(masks[colliders[0]]["collide_with"])
        return contype, conaffinity

    def get_gravity(self) -> np.ndarray:
        return self._gravity.copy()

    def get_actuator_names(self) -> tuple[str, ...]:
        return self._actuator_names

    def get_actuator_joint_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for actuator_name in self._actuator_names:
            actuator = self._model.get_actuator(actuator_name)
            target = getattr(actuator, "target_name", None)
            if not target:
                raise NotImplementedError(
                    f"motrix_gpu actuator {actuator_name!r} has no named joint target"
                )
            names.append(str(target))
        return tuple(names)

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-actuator ``(kp, kd)`` read from the model/override tables."""
        kp = np.empty((self._nu,), dtype=np.float32)
        kd = np.empty((self._nu,), dtype=np.float32)
        for index, actuator_name in enumerate(self._actuator_names):
            actuator = self._model.get_actuator(actuator_name)
            # Implicit drive gains only exist for position actuators; torque
            # `motor` actuators have none and report zero gain.
            try:
                kp[index] = float(np.asarray(actuator.get_kp_override(self._data))[0])
                kd[index] = float(np.asarray(actuator.get_damping_override(self._data))[0])
            except (AttributeError, ValueError):
                kp[index] = 0.0
                kd[index] = 0.0
        return kp, kd

    def get_joint_state_qpos_indices(self, names: Sequence[str]) -> np.ndarray:
        # MotrixSim's ``dof_pos_index`` already addresses the full qpos vector
        # (a free root occupies the leading columns), so no offset is applied.
        indices = [int(self._resolve_single_dof_joint(name).dof_pos_index) for name in names]
        return np.asarray(indices, dtype=np.intp)

    def get_joint_state_qvel_indices(self, names: Sequence[str]) -> np.ndarray:
        indices = [int(self._resolve_single_dof_joint(name).dof_vel_index) for name in names]
        return np.asarray(indices, dtype=np.intp)

    def get_joint_dof_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to their qvel columns."""
        indices = [int(self._resolve_single_dof_joint(name).dof_vel_index) for name in names]
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_pos_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to joint-section qpos offsets (floating bases excluded)."""
        indices = [
            int(self._joint_section_pos_offsets[int(self._resolve_single_dof_joint(name).index)])
            for name in names
        ]
        return np.asarray(indices, dtype=np.int32)

    def get_joint_dof_vel_indices(self, names: Sequence[str]) -> np.ndarray:
        """Resolve joint names to joint-section qvel offsets (floating bases excluded)."""
        indices = [
            int(self._joint_section_vel_offsets[int(self._resolve_single_dof_joint(name).index)])
            for name in names
        ]
        return np.asarray(indices, dtype=np.int32)

    def _resolve_single_dof_joint(self, name: str) -> Any:
        joint = self._model.get_joint(name)
        if joint is None:
            raise KeyError(f"unknown motrix_gpu joint: {name!r}")
        if int(joint.num_dof_pos) != 1 or int(joint.num_dof_vel) != 1:
            raise ValueError(f"motrix_gpu joint {name!r} is not a single-DoF joint")
        return joint

    def get_site_ids(self, names: Sequence[str]) -> np.ndarray:
        ids = []
        for name in names:
            index = self._model.get_site_index(name)
            if index is None or int(index) < 0:
                raise KeyError(f"unknown motrix_gpu site: {name!r}")
            ids.append(int(index))
        return np.asarray(ids, dtype=np.int32)

    def get_motion_body_ids(self, names: Sequence[str]) -> np.ndarray:
        """Resolve motion-dataset body ids (MuJoCo order, ``worldbody`` is 0)."""
        ids = []
        for name in names:
            index = self._model.get_link_index(name)
            if index is None or int(index) < 0:
                raise KeyError(f"Motion body {name!r} not found in motrix_gpu model")
            ids.append(int(index) + 1)
        return np.asarray(ids, dtype=np.int32)

    def get_default_qpos(self) -> np.ndarray:
        """Return the model's initial qpos in the public ``wxyz`` root order."""
        self._require_open()
        qpos = np.array(np.asarray(self._model.compute_init_dof_pos(), dtype=np.float32), copy=True)
        permutation = self._root_quaternion_permutation()
        if permutation is not None:
            columns, public_sources, _ = permutation
            qpos[list(columns)] = qpos[list(public_sources)]
        return qpos

    def get_default_dof_pos(self) -> np.ndarray:
        """Return default joint positions in canonical non-root qpos order."""
        qpos = self.get_default_qpos()
        values = [
            qpos[int(joint.dof_pos_index) : int(joint.dof_pos_index) + int(joint.num_dof_pos)]
            for joint in self._model.joints
        ]
        return np.concatenate(values).astype(np.float32) if values else np.zeros(0, np.float32)

    def get_root_state_layout(self, root_body_name: str) -> RootStateLayout:
        if root_body_name not in self._link_names:
            raise KeyError(f"unknown motrix_gpu root body: {root_body_name!r}")
        base = self._floating_roots.get(root_body_name)
        if base is None:
            raise NotImplementedError(
                f"motrix_gpu root state layout requires a floating root; "
                f"body {root_body_name!r} is fixed"
            )
        start = int(base.dof_pos_start)
        vel_start = int(base.dof_vel_start)
        return RootStateLayout(
            qpos_indices=tuple(range(start, start + 7)),
            qvel_indices=tuple(range(vel_start, vel_start + 6)),
        )

    def get_physics_state_layout(self) -> PhysicsStateLayout:
        return PhysicsStateLayout(nq=self._nq, nv=self._nv)

    def get_physics_state(self) -> np.ndarray:
        """Assemble a contract ``[time, qpos, qvel]`` snapshot (host copy)."""
        self._require_open()
        layout = self.get_physics_state_layout()
        out = np.empty((self._num_envs, layout.state_width), dtype=np.float32)
        out[:, 0] = self._time.cpu().numpy()
        out[:, 1 : 1 + layout.nq] = self._public_qpos()
        out[:, 1 + layout.nq :] = np.asarray(self._data.dof_vel.numpy(), dtype=np.float32)
        return out

    def get_playback_model(self, env_index: int | None = None) -> Any:
        """Return the MJCF construction source of this single-source scene."""
        del env_index  # one canonical source serves every environment
        return self._scene_model_file

    def get_play_capabilities(self) -> Any:
        # Playback mirrors the GPU link poses into MotrixSim's native renderer,
        # so the interactive viewer and the headless video-capture path share
        # one pose source while physics stays on the device.
        return PlayCapabilities(
            supports_native_interactive_renderer=True,
            supports_physics_state_playback=True,
            supports_native_video_capture=True,
        )

    def _playback_renderer(self) -> MotrixGpuPlaybackRenderer:
        self._require_open()
        if self._renderer is None:
            self._renderer = MotrixGpuPlaybackRenderer(
                model_file=str(self._scene_model_file),
                scene_world=self._scene_world,
                num_envs=self._num_envs,
                link_names=self._tracked_body_names,
                root_link_index=self._tracked_body_slots.get(str(self._base_name), 0),
                pose_getter=self._playback_link_poses,
            )
        return self._renderer

    def _playback_link_poses(self) -> np.ndarray:
        """Return ``(num_envs, num_links, 7)`` engine-order link poses.

        The renderer synchronizes ``[x, y, z, qx, qy, qz, qw]`` rows, which is
        exactly the engine's own link pose convention.
        """
        self._require_open()
        import torch

        position = self._tracked_buffers["pos_w"]
        rotation = self._tracked_buffers["quat_w"][..., [1, 2, 3, 0]]
        return torch.cat((position, rotation), dim=-1).detach().cpu().numpy()

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
        """Initialize the native renderer, optionally arming video capture."""
        self._playback_renderer().init(
            spacing=spacing,
            offset_mode=offset_mode,
            headless=headless,
            capture=capture,
            width=width,
            height=height,
            camera_kwargs=camera_kwargs,
        )

    def render(self) -> None:
        """Render the current state in the interactive viewer."""
        self._playback_renderer().render()

    def capture_video_frame(self) -> np.ndarray:
        """Capture one RGB frame of the mirrored GPU state."""
        return self._playback_renderer().capture_frame()

    def resolve_play_render_plan(
        self,
        *,
        play_render_mode: str | None,
        play_steps: int | None,
        output_video: Any | None,
    ) -> PlayRenderPlan:
        mode = normalize_play_render_mode(play_render_mode)
        effective_mode = "interactive" if mode == "auto" else mode
        renderer = "motrixsim.gpu pose-mirror renderer"
        if effective_mode == "none":
            return PlayRenderPlan(
                mode=effective_mode,
                headless=True,
                record_video=False,
                num_steps=None,
                output_video=None,
                renderer=renderer,
            )
        if effective_mode == "interactive":
            return PlayRenderPlan(
                mode=effective_mode,
                headless=False,
                record_video=False,
                num_steps=None,
                output_video=None,
                renderer=renderer,
            )
        assert effective_mode == "record"
        if play_steps is None:
            raise ValueError(
                "motrix_gpu record playback requires a finite training.play_steps value."
            )
        if output_video is None:
            raise ValueError("motrix_gpu record playback requires an output video path.")
        return PlayRenderPlan(
            mode=effective_mode,
            headless=True,
            record_video=True,
            num_steps=int(play_steps),
            output_video=output_video,
            renderer=renderer,
        )

    def run_playback(
        self,
        *,
        env: Any,
        initialize: Any,
        step: Any,
        num_steps: int | None,
        output_video: Any | None = None,
        render_spacing: float | None = None,
        render_offset_mode: str | None = None,
        headless: bool | None = None,
        record_video: bool | None = None,
        frame_state_getter: Any | None = None,
        camera_kwargs: CameraCfg | Mapping[str, Any] | None = None,
        debug_overlay_getter: Any | None = None,
        on_frame: Any | None = None,
    ) -> str | None:
        del frame_state_getter
        if debug_overlay_getter is not None:
            raise unsupported_debug_overlay_error(self.__class__.__name__)
        if on_frame is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} renders through a native renderer and "
                "does not support on_frame callbacks"
            )
        return run_motrix_playback(
            backend=self,
            env=env,
            initialize=initialize,
            step=step,
            num_steps=num_steps,
            output_video=output_video,
            render_spacing=render_spacing,
            render_offset_mode=render_offset_mode,
            headless=bool(headless),
            record_video=bool(record_video),
            camera_kwargs=camera_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Named sensors and tracked bodies                                    #
    # ------------------------------------------------------------------ #

    def get_sensor_names(self) -> tuple[str, ...]:
        self._require_sensor_inventory()
        return self._published_sensor_names

    def get_sensor_inventory(self) -> tuple[SensorDescriptor, ...]:
        self._require_sensor_inventory()
        scene_entries = [
            SensorDescriptor(
                name=name,
                width=self._sensor_width(name),
                category=self._scene_sensor_category(name),
            )
            for name in self._sensor_names
        ]
        tracked_entries = [
            SensorDescriptor(name=name, width=self._tracked_sensor_slots[name][1])
            for name in self._tracked_sensor_names
        ]
        return tuple(scene_entries + tracked_entries)

    def _scene_sensor_category(self, name: str) -> SensorCategory:
        """Classify one declared scene sensor by its public value category."""
        if name in self._contact_sensor_slots or name in self._contact_force_slots:
            return "contact"
        if name in self._actuator_force_sensor_slots:
            return "force"
        return "unknown"

    def _public_native_sensor_view(self, name: str, view: Any) -> Any:
        """Publish MotrixSim ``framequat`` rows in the public ``wxyz`` order.

        MotrixSim stores quaternions as ``xyzw``; MuJoCo ``framequat`` sensors
        (and every other UniSim public quaternion) are ``wxyz``. Only the
        declared FrameQuat sensors are permuted, matching the CPU ``motrix``
        adapter's 1.8.1 fix, so G1 foot-roll/pitch penalties are not inflated
        by a component-order mismatch.
        """
        if name not in self._frame_quat_sensor_names:
            return view
        import torch

        if tuple(view.shape)[-1:] != (4,):
            raise NotImplementedError(
                f"motrix_gpu FrameQuat sensor {name!r} has trailing width "
                f"{tuple(view.shape)[-1:]}, expected 4"
            )
        public = self._frame_quat_public_views.get(name)
        if (
            public is None
            or public.shape != view.shape
            or public.device != view.device
            or public.dtype != view.dtype
        ):
            public = torch.empty_like(view)
            self._frame_quat_public_views[name] = public
        permutation = self._frame_quat_permutations.get(view.device)
        if permutation is None:
            permutation = torch.tensor((3, 0, 1, 2), dtype=torch.long, device=view.device)
            self._frame_quat_permutations[view.device] = permutation
        # Write the permuted block straight into the stable public view; a
        # plain ``public.copy_(view.index_select(...))`` allocates a fresh
        # gather output on every read-phase boundary crossing.
        torch.index_select(view, -1, permutation, out=public)
        return public

    def get_sensor_view(self, name: str, device: Any | None = None) -> Any:
        """Return one named sensor block.

        Sensor values are produced by the physics step, so they report the last
        completed ``step_tensor`` call (a selected reset does not re-run the
        sensor stage).
        """
        import torch

        self._require_open()
        self._require_sensor_inventory()
        tracked = self._tracked_sensor_slots.get(name)
        if tracked is not None:
            field, _width, body_slot = tracked
            view = self._tracked_buffers[field][:, body_slot, :]
            self._require_sensor_device(name, view, device)
            return view
        slot = self._contact_force_slots.get(name)
        if slot is not None:
            view = self._contact_force_storage[:, slot, :]
            self._require_sensor_device(name, view, device)
            return view
        slot = self._contact_sensor_slots.get(name)
        if slot is not None:
            view = self._contact_results[:, slot : slot + 1]
            self._require_sensor_device(name, view, device)
            return view
        slot = self._actuator_force_sensor_slots.get(name)
        if slot is not None:
            view = self._actuator_force_storage[:, slot : slot + 1]
            self._require_sensor_device(name, view, device)
            return view
        if name not in self._native_sensor_names:
            available = ", ".join(self._sensor_names)
            raise ValueError(f"Sensor {name!r} not found; available: {available}")
        view = self._sensor_torch_views.get(name)
        if view is None:
            view = torch.from_dlpack(  # pyright: ignore[reportPrivateImportUsage]
                self._model.get_sensor_value(name, self._data)
            )
            self._sensor_torch_views[name] = view
        self._require_sensor_device(name, view, device)
        return self._public_native_sensor_view(name, view)

    @staticmethod
    def _require_sensor_device(name: str, view: Any, device: Any | None) -> None:
        """Fail unless ``device`` is the device-resident sensor view's device."""
        import torch

        if device is None:
            return
        target = torch.device(device)
        if target.type != view.device.type or target.index not in (None, view.device.index):
            raise ValueError(
                f"motrix_gpu sensor {name!r} must live on {view.device}, requested {target}"
            )

    def _bind_sensor_data_reader(self, names: tuple[str, ...]) -> Callable[[], np.ndarray]:
        """Return a host NumPy reader over named device sensor views.

        The shared cold-path binding probes this reader for the public width
        before any tensor read plan exists; device-resident callers read the
        device views directly through ``get_sensor_view``.
        """

        def read() -> np.ndarray:
            self._require_open()
            self._require_sensor_inventory()
            columns = [
                np.asarray(view.detach().cpu(), dtype=np.float32).reshape(self._num_envs, -1)
                for view in (self.get_sensor_view(name) for name in names)
            ]
            return columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)

        return read

    def get_tracked_body_views(
        self, body_names: Sequence[str] | None = None, device: Any | None = None
    ) -> TrackedBodyStateViews:
        import torch

        self._require_open()
        names = tuple(self._tracked_body_slots) if body_names is None else tuple(body_names)
        if not names:
            raise ValueError("motrix_gpu tracked-body views require at least one body")
        if len(set(names)) != len(names):
            raise ValueError(f"tracked-body view names must be unique: {names}")
        missing = [name for name in names if name not in self._tracked_body_slots]
        if missing:
            raise ValueError(
                "motrix_gpu tracked-body views requested unknown links "
                f"{missing}; available={list(self._tracked_body_slots)}"
            )

        columns = self._tracked_body_column_cache.get(names)
        if columns is None or columns.device != self._field_view("qpos").device:
            columns = torch.as_tensor(
                [self._tracked_body_slots[name] for name in names],
                dtype=torch.int64,
                device=self._field_view("qpos").device,
            )
            self._tracked_body_column_cache[names] = columns

        def block(field: str) -> Any:
            # The per-body blocks are published once per step into stable
            # buffers shared with the ``track_*_w_<body>`` sensor names.
            return self._tracked_buffers[field].index_select(1, columns)

        views = TrackedBodyStateViews(
            body_names=names,
            pos_w=block("pos_w"),
            quat_w=block("quat_w"),
            lin_vel_w=block("lin_vel_w"),
            ang_vel_w=block("ang_vel_w"),
        )
        reference = views.pos_w
        if device is not None:
            target = torch.device(device)
            if target.type != reference.device.type or target.index not in (
                None,
                reference.device.index,
            ):
                raise ValueError(
                    f"motrix_gpu tracked-body views must live on {reference.device}, "
                    f"requested {target}"
                )
        return views

    def _require_sensor_inventory(self) -> None:
        if self._sensor_inventory_verified:
            return
        # The importer counts ``<actuatorfrc>`` declarations but drops
        # ``<jointactuatorfrc>``; the MSD materializer replaces the latter with
        # internal actuator sensors. Either way the engine must not own fewer
        # sensors than the adapter resolved from the source.
        engine_counted = len(self._sensor_names) - sum(
            kind == "jointactuatorfrc" for kind in self._actuator_force_sensor_kinds.values()
        )
        if int(self._model.num_sensors) < engine_counted:
            raise NotImplementedError(
                "motrix_gpu could not rebuild the sensor namespace from the source MJCF: "
                f"parsed {len(self._sensor_names)} names but the engine reports "
                f"{int(self._model.num_sensors)} sensors; sensor views fail closed"
            )
        native = self._native_sensor_names
        unresolved = [
            name
            for name in self._sensor_names
            if name not in self._contact_sensor_slots
            and name not in self._actuator_force_sensor_slots
            and name not in self._contact_force_slots
            and name not in native
        ]
        if unresolved:
            raise NotImplementedError(
                "motrix_gpu cannot resolve sensor(s) "
                f"{unresolved} on the MotrixSim GPU runtime: only native "
                "frame/site/subtree/joint sensors, geom-pair <contact data='found'>, "
                "<contact data='force' reduce='netforce'> and "
                "<actuatorfrc>/<jointactuatorfrc> sensors are addressable; "
                "other sensor forms fail closed"
            )
        self._sensor_inventory_verified = True

    def _sensor_width(self, name: str) -> int:
        if name in self._contact_force_slots:
            return 3
        if name in self._contact_sensor_slots or name in self._actuator_force_sensor_slots:
            return 1
        value = self._model.get_sensor_value(name, self._data)
        return int(np.asarray(value.shape)[-1])

    def _resolve_contact_pairs(self, declarations: dict[str, tuple[str, str]]) -> np.ndarray:
        """Resolve declared contact-sensor geom names to runtime geom ids."""
        pairs: list[list[int]] = []
        for geom1, geom2 in declarations.values():
            resolved = [self._model.get_geom_index(name) for name in (geom1, geom2)]
            if resolved[0] is None or resolved[1] is None:
                raise NotImplementedError(
                    f"motrix_gpu contact sensor references unknown geoms {geom1!r}/{geom2!r}"
                )
            pairs.append([int(resolved[0]), int(resolved[1])])
        return np.asarray(pairs, dtype=np.int32).reshape(len(pairs), 2)

    @property
    def _native_sensor_names(self) -> frozenset[str]:
        """Names the GPU runtime's ``get_sensor_value`` can address directly.

        The engine's name table is immutable after model build, so the
        frozenset is resolved once and cached; rebuilding it per
        ``get_sensor_view`` call was a top collector-side hotspot.
        """
        cached = self._native_sensor_name_cache
        if cached is None:
            table = getattr(self._model.low.sensors, "sensor_name_to_value_range", {})
            cached = frozenset(str(name) for name in table)
            self._native_sensor_name_cache = cached
        return cached

    def _refresh_contact_sensors(self) -> None:
        """Publish geom-pair ``found`` contact sensors from the row-local query."""
        if not self._contact_sensor_names:
            return
        import torch

        if self._contact_pairs is None:
            self._contact_pairs = self._deps.warp.array(
                self._contact_pairs_np,
                dtype=self._deps.warp.int32,
                device=self._deps.warp.get_device(),
            )
        query = self._model.get_contact_query(self._data)
        results = torch.from_dlpack(  # pyright: ignore[reportPrivateImportUsage]
            query.is_colliding(self._contact_pairs)
        )
        if self._contact_results is None:
            self._contact_results = torch.empty(
                (self._num_envs, len(self._contact_sensor_names)),
                dtype=torch.float32,
                device=results.device,
            )
        self._contact_results.copy_(results)

    def _refresh_tracked_sensors(self) -> None:
        """Publish one stable ``(num_envs, nbody, width)`` block per field.

        The blocks feed both ``get_tracked_body_views()`` and the public
        ``track_<field>_w_<body>`` sensor names that device-resident Manager
        read plans request. The republish batches over the engine's packed
        per-link blocks (``high_level_poses`` and the per-link velocity
        storages): one ``index_select`` per field (plus one component reorder
        for the quaternion) instead of one small copy per tracked body. When
        the packed blocks or link indices are unavailable, the per-body
        getter path is kept as the fallback.
        """
        import torch

        packed = self._tracked_packed_views()
        indices = self._tracked_link_indices()
        if packed is not None and indices is not None:
            poses = packed["poses"]
            lin_vels = packed["lin_vels"]
            ang_vels = packed["ang_vels"]
            pos_buffer = self._tracked_buffer("pos_w", 3)
            quat_buffer = self._tracked_buffer("quat_w", 4)
            lin_buffer = self._tracked_buffer("lin_vel_w", 3)
            ang_buffer = self._tracked_buffer("ang_vel_w", 3)
            if (
                self._tracked_wxyz_perm is None
                or self._tracked_wxyz_perm.device != pos_buffer.device
            ):
                # The engine stores link quaternions as xyzw.
                self._tracked_wxyz_perm = torch.tensor(
                    (3, 0, 1, 2), dtype=torch.long, device=pos_buffer.device
                )
            if (
                self._tracked_quat_stage is None
                or self._tracked_quat_stage.shape != quat_buffer.shape
                or self._tracked_quat_stage.device != quat_buffer.device
            ):
                self._tracked_quat_stage = torch.empty_like(quat_buffer)
            torch.index_select(poses[..., 0:3], 1, indices, out=pos_buffer)
            torch.index_select(poses[..., 3:7], 1, indices, out=self._tracked_quat_stage)
            torch.index_select(
                self._tracked_quat_stage, 2, self._tracked_wxyz_perm, out=quat_buffer
            )
            torch.index_select(lin_vels, 1, indices, out=lin_buffer)
            torch.index_select(ang_vels, 1, indices, out=ang_buffer)
            return

        for _prefix, field, width, wxyz in _TRACKED_SENSOR_LAYOUT:
            buffer = self._tracked_buffer(field, width)
            cache = self._tracked_field_views.setdefault(field, {})
            for index, name in enumerate(self._tracked_body_names):
                value = cache.get(index)
                if value is None:
                    # The engine returns a slice of a persistent device buffer,
                    # so the DLPack wrap is resolved once and reused.
                    link = self._model.get_link(name)
                    value = torch.from_dlpack(  # pyright: ignore[reportPrivateImportUsage]
                        getattr(link, _TRACKED_GETTERS[field])(self._data)
                    ).reshape(self._num_envs, width)
                    cache[index] = value
                if wxyz:
                    # The engine stores link quaternions as xyzw.
                    value = value[:, [3, 0, 1, 2]]
                buffer[:, index, :] = value

    def _tracked_buffer(self, field: str, width: int) -> Any:
        """Return the stable ``(num_envs, nbody, width)`` block for one field."""
        import torch

        buffer = self._tracked_buffers.get(field)
        if buffer is None:
            buffer = torch.empty(
                (self._num_envs, len(self._tracked_body_names), width),
                dtype=torch.float32,
                device=self._field_view("qpos").device,
            )
            self._tracked_buffers[field] = buffer
        return buffer

    def _tracked_packed_views(self) -> dict[str, Any] | None:
        """Wrap the engine's packed per-link state blocks as Torch views.

        The per-link getters are column slices of these persistent blocks, so
        one DLPack wrap per block is enough to batch every tracked-body field
        into a single gather. Returns ``None`` when an engine build without
        the packed blocks is in use.
        """
        import torch

        if self._tracked_packed is not None:
            return self._tracked_packed
        poses = getattr(self._data, "high_level_poses", None)
        lin_vels = getattr(self._data, "_high_link_linear_vels", None)
        ang_vels = getattr(self._data, "_high_link_angular_vels", None)
        if poses is None or lin_vels is None or ang_vels is None:
            return None
        try:
            packed = {
                "poses": torch.from_dlpack(poses),  # pyright: ignore[reportPrivateImportUsage]
                "lin_vels": torch.from_dlpack(lin_vels),  # pyright: ignore[reportPrivateImportUsage]
                "ang_vels": torch.from_dlpack(ang_vels),  # pyright: ignore[reportPrivateImportUsage]
            }
        except (TypeError, ValueError, RuntimeError):
            return None
        if packed["poses"].ndim != 3 or packed["poses"].shape[0] != self._num_envs:
            return None
        self._tracked_packed = packed
        return packed

    def _tracked_link_indices(self) -> Any:
        """Resolve tracked bodies' columns in the packed per-link blocks."""
        import torch

        if self._tracked_column_indices is not None:
            return self._tracked_column_indices
        columns: list[int] = []
        for name in self._tracked_body_names:
            if not name:
                return None
            link = self._model.get_link(name)
            inner = getattr(link, "_inner", None)
            if inner is None:
                base = getattr(link, "base_link", None)
                inner = getattr(base, "_inner", None) if base is not None else None
            index = getattr(inner, "index", None)
            if not isinstance(index, int) or isinstance(index, bool):
                return None
            columns.append(index)
        packed = self._tracked_packed_views()
        if packed is None:
            return None
        link_count = int(packed["poses"].shape[1])
        if not columns or max(columns) >= link_count:
            return None
        self._tracked_column_indices = torch.tensor(
            columns, dtype=torch.long, device=packed["poses"].device
        )
        return self._tracked_column_indices

    # ------------------------------------------------------------------ #
    # Tensor lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def get_tensor_execution(self) -> TensorExecution:
        return TensorExecution.DEVICE_RESIDENT

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        # A selected reset republishes link kinematics, the high-level link
        # poses/velocities and the kinematics-dependent sensors (site pose/vel,
        # subtree, joint pos/vel) before returning, so every published view is
        # authoritative immediately.  Acceleration-only sensor rows still read
        # the last solved step, matching MJWarp's ``sensor_pos``/``sensor_vel``
        # refresh boundary.
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel", "ctrl"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            device_reset_randomization=True,
            packed_reset=True,
            packed_reset_fields=frozenset(self._supported_reset_terms),
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="The caller's Torch stream orders step, reset and publication",
            torch_devices=("cuda",),
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            tracked_body_views=True,
            mask_selected_reset=True,
            mask_selected_reset_full_refresh=True,
            external_graph_capture=True,
        )

    def bind_packed_reset(self, *, device: Any | None = None) -> Any:
        """Begin a device-resident packed reset with a single scatter commit."""
        from ..packed_reset import PackedResetTransaction

        return PackedResetTransaction(self, device=device)

    def get_public_state_widths(self) -> PublicStateWidths:
        return PublicStateWidths(nq=self._nq, nv=self._nv)

    def get_state_views(
        self, fields: tuple[str, ...] | str | None = None, device: Any | None = None
    ) -> Mapping[str, Any]:
        import torch

        self._require_open()
        names = (
            ("qpos", "qvel")
            if fields is None
            else ((fields,) if isinstance(fields, str) else tuple(fields))
        )
        unknown = set(names) - _SUPPORTED_STATE_FIELDS
        if unknown:
            raise KeyError(f"unknown motrix_gpu state field(s): {sorted(unknown)}")
        views = {
            name: self._public_qpos_view()
            if name == "qpos"
            else self._ctrl_public
            if name == "ctrl"
            else self._field_view(name)
            for name in names
        }
        if device is not None and views:
            target = torch.device(device)
            reference = next(iter(views.values()))
            if target.type != reference.device.type or (
                target.index is not None and target.index != reference.device.index
            ):
                raise ValueError(
                    "motrix_gpu state views are device-resident on "
                    f"{reference.device}; requested {target}"
                )
        return views

    def step_tensor(self, ctrl: Any, nsteps: int = 1) -> dict | None:
        import torch

        self._require_open()
        if self._pre_step_control_fn is not None:
            raise NotImplementedError(
                "motrix_gpu tensor stepping does not support host pre-step control callbacks"
            )
        if type(nsteps) is not int or nsteps < 1:
            raise ValueError("nsteps must be a positive integer")
        ctrl_tensor = self._validate_ctrl(ctrl, torch)

        ctrl_view = self._field_view("ctrl")
        if ctrl_tensor.device != ctrl_view.device:
            raise ValueError(
                f"motrix_gpu ctrl must live on {ctrl_view.device}, got {ctrl_tensor.device}"
            )

        t0 = time.perf_counter()
        self._ctrl_public.copy_(ctrl_tensor, non_blocking=True)
        control_upload_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with self._warp_execution() as stream:
            if self._external_capture is None:
                for _ in range(nsteps):
                    self._apply_gravity_compensation(ctrl_view)
                    self._model.step(self._data, stream=stream)
            else:
                for _ in range(nsteps):
                    self._apply_gravity_compensation(ctrl_view)
                    self._model.step_no_capture(self._data)
            self._refresh_contact_sensors()
            self._refresh_contact_forces()
        # The engine advanced ``dof_pos``; republish the public root-quaternion
        # mirror so the state views stay authoritative after the step.
        self._publish_public_qpos()
        self._refresh_tracked_sensors()
        self._refresh_actuator_forces()
        self._time.add_(nsteps * self._sim_dt)
        physics_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timing": {
                "control_upload_ms": control_upload_ms,
                "physics_ms": physics_ms,
                "host_cache_refresh_ms": 0.0,
            }
        }

    @contextmanager
    def _warp_execution(self) -> Iterator[Any]:
        """Order eager engine work by events and record directly on an external stream."""
        import torch

        if self._external_capture is not None:
            yield self._external_capture.backend_stream
            return
        caller = torch.cuda.current_stream(self._device)
        with self._deps.warp.ScopedStream(self._warp_stream, sync_enter=False, sync_exit=False):
            if caller.cuda_stream == self._torch_stream.cuda_stream:
                yield self._warp_stream
                return
            self._caller_ready.record(caller)
            self._torch_stream.wait_event(self._caller_ready)
            try:
                with torch.cuda.stream(self._torch_stream):
                    yield self._warp_stream
            finally:
                self._engine_ready.record(self._torch_stream)
                caller.wait_event(self._engine_ready)

    def set_state_tensor(
        self,
        env_indices: Any,
        qpos: Any,
        qvel: Any,
        randomization: TensorResetRandomizationPayload | None = None,
    ) -> dict | None:
        import torch

        self._require_open()
        if self._external_capture is not None:
            raise GraphCaptureError("motrix_gpu external capture requires a fixed-width reset mask")
        rows = self._validate_rows(env_indices, torch)
        qpos_tensor = self._validate_state("qpos", qpos, rows.shape[0], self._nq, torch)
        qvel_tensor = self._validate_state("qvel", qvel, rows.shape[0], self._nv, torch)
        pos_view = self._field_view("qpos")
        vel_view = self._field_view("qvel")
        device = pos_view.device
        if qpos_tensor.device != device or qvel_tensor.device != device or rows.device != device:
            raise ValueError("motrix_gpu reset tensors must live on the backend device")
        if not bool(torch.isfinite(qpos_tensor).all() & torch.isfinite(qvel_tensor).all()):
            raise ValueError("motrix_gpu reset state must be finite")
        payload = self._resolve_reset_randomization(randomization, rows.numel(), torch)

        started = time.perf_counter()
        if rows.shape[0]:
            pos_view[rows] = self._engine_qpos_from_public(qpos_tensor)
            vel_view[rows] = qvel_tensor
            self._time.index_fill_(0, rows, 0)
        mask_ms = (time.perf_counter() - started) * 1000.0

        model_update_ms = 0.0
        if payload is not None:
            update_started = time.perf_counter()
            self._apply_tensor_reset_randomization(rows, payload, torch)
            model_update_ms = (time.perf_counter() - update_started) * 1000.0

        forward_ms = 0.0
        if rows.shape[0] or payload is not None:
            forward_started = time.perf_counter()
            # Republish link kinematics, link poses/velocities and the
            # kinematics-dependent sensors for every published view.
            self._refresh_high_level()
            self._publish_public_qpos()
            self._refresh_tracked_sensors()
            self._refresh_actuator_forces()
            forward_ms = (time.perf_counter() - forward_started) * 1000.0

        return {
            "timing": {
                "set_state_tensor_mask_ms": mask_ms,
                "set_state_tensor_commit_forward_ms": forward_ms,
                "set_state_tensor_host_cache_refresh_ms": 0.0,
                "set_state_tensor_model_update_ms": model_update_ms,
            }
        }

    def set_state_tensor_mask(
        self,
        reset_mask: Any,
        qpos: Any,
        qvel: Any,
        randomization: TensorResetRandomizationPayload | None = None,
        *,
        full_refresh: bool = False,
    ) -> dict | None:
        """Commit full-width state selected by a fixed-width bool mask.

        Only true rows move; every false row keeps its authoritative value.
        Model-field randomization is device-resident only, matching the mask
        boundary's explicit device contract.

        The state commit uses full-width ``torch.where`` masked writes instead
        of ``nonzero()`` + row indexing: the write shape is fixed regardless
        of how many rows reset, no host reads the mask, and the operation is
        CUDA-graph capturable (required by ``autoreset_cuda_graph``). The
        per-row cost of writing the full width is negligible compared to the
        forward-kinematics refresh that follows.
        """
        import torch

        self._require_open()
        mask = self._validate_mask(reset_mask, torch)
        qpos_tensor = self._validate_state("qpos", qpos, self._num_envs, self._nq, torch)
        qvel_tensor = self._validate_state("qvel", qvel, self._num_envs, self._nv, torch)
        pos_view = self._field_view("qpos")
        vel_view = self._field_view("qvel")
        device = pos_view.device
        if qpos_tensor.device != device or qvel_tensor.device != device or mask.device != device:
            raise ValueError("motrix_gpu mask reset tensors must live on the backend device")
        payload = self._resolve_reset_randomization(randomization, self._num_envs, torch)

        in_capture = self._external_capture is not None
        started = time.perf_counter()
        finite_rows = torch.isfinite(qpos_tensor).all(dim=1) & torch.isfinite(qvel_tensor).all(
            dim=1
        )
        if in_capture:
            self._reset_verdicts.record(
                "state", finite_rows.all(), "motrix_gpu reset state must be finite"
            )
            mask = mask & finite_rows
        elif not bool(finite_rows.all()):
            raise ValueError("motrix_gpu reset state must be finite")
        rows = None if in_capture else mask.nonzero(as_tuple=False).reshape(-1)

        if in_capture:
            # Graph-capture mode: fixed-shape full-width masked writes (the
            # CUDA graph replays these unconditionally; the mask content
            # selects which rows take effect at replay time).
            mask_col = mask.unsqueeze(-1)
            engine_qpos = self._engine_qpos_from_public(qpos_tensor)
            pos_view.copy_(torch.where(mask_col, engine_qpos, pos_view))
            vel_view.copy_(torch.where(mask_col, qvel_tensor, vel_view))
            self._time.masked_fill_(mask, 0)
        elif rows is not None and rows.numel():
            # Eager mode: sparse row-based writes (only the reset rows move;
            # with ~13 resets per 4096-env step this is far cheaper than a
            # full-width torch.where pass).
            pos_view[rows] = self._engine_qpos_from_public(qpos_tensor.index_select(0, rows))
            vel_view[rows] = qvel_tensor.index_select(0, rows)
            self._time.index_fill_(0, rows, 0)
        mask_ms = (time.perf_counter() - started) * 1000.0

        model_update_ms = 0.0
        if payload is not None:
            update_started = time.perf_counter()
            if in_capture:
                self._apply_tensor_reset_randomization(mask, payload, torch)
            elif rows is not None and rows.numel():
                self._apply_tensor_reset_randomization(
                    rows, self._select_payload_rows(payload, rows), torch
                )
            model_update_ms = (time.perf_counter() - update_started) * 1000.0

        forward_ms = 0.0
        if in_capture or full_refresh or (rows is not None and rows.numel()):
            forward_started = time.perf_counter()
            self._refresh_high_level()
            self._publish_public_qpos()
            self._refresh_tracked_sensors()
            self._refresh_actuator_forces()
            forward_ms = (time.perf_counter() - forward_started) * 1000.0

        return {
            "timing": {
                "set_state_tensor_mask_ms": mask_ms,
                "set_state_tensor_commit_forward_ms": forward_ms,
                "set_state_tensor_host_cache_refresh_ms": 0.0,
                "set_state_tensor_model_update_ms": model_update_ms,
            }
        }

    def _select_payload_rows(
        self, payload: TensorResetRandomizationPayload, rows: Any
    ) -> TensorResetRandomizationPayload:
        return TensorResetRandomizationPayload(
            **{
                term: getattr(payload, term).index_select(0, rows)
                for term in payload.requested_terms()
            }
        )

    def _reset_term_shape(self, term: str, num_rows: int) -> tuple[int, ...]:
        """Return the canonical selected-row payload shape for one reset term."""
        if term in (RESET_TERM_KP, RESET_TERM_KD):
            return (num_rows, self._nu)
        if term == RESET_TERM_BODY_MASS:
            return (num_rows, self._num_links)
        if term == RESET_TERM_BODY_IPOS:
            return (num_rows, self._num_links, 3)
        if term == RESET_TERM_GEOM_FRICTION:
            return (num_rows, self._num_geoms, 3)
        raise KeyError(f"unknown motrix_gpu reset term {term!r}")

    def _resolve_reset_randomization(
        self, randomization: TensorResetRandomizationPayload | None, num_rows: int, torch: Any
    ) -> TensorResetRandomizationPayload | None:
        """Validate all terms before committing state or overrides."""
        if randomization is None:
            return None
        validate_tensor_reset_payload(
            randomization,
            device=self._device,
            allowed_terms=self._supported_reset_terms,
        )
        requested = randomization.requested_terms()
        if not requested:
            return None
        for term in sorted(requested):
            value = getattr(randomization, term)
            expected = self._reset_term_shape(term, num_rows)
            if value.dtype != torch.float32 or tuple(value.shape) != expected:
                raise ValueError(
                    f"motrix_gpu reset term {term!r} requires float32 and shape {expected}, "
                    f"got {value.dtype} and {tuple(value.shape)}"
                )
            valid = torch.isfinite(value).all()
            message = f"motrix_gpu reset term {term!r} must be finite"
            if term in self._gain_views and self._non_joint_columns.numel():
                valid = valid & (value.index_select(1, self._non_joint_columns) == 0).all()
                message += " with zero values for non-joint actuators"
            if self._external_capture is not None:
                self._reset_verdicts.record(term, valid, message)
            elif not bool(valid):
                raise ValueError(message)
        return randomization

    def _apply_tensor_reset_randomization(
        self, rows: Any, payload: TensorResetRandomizationPayload, torch: Any
    ) -> None:
        """Scatter through either sparse row indices or a graph-safe full-width mask."""
        for term, buffer in self._gain_views.items():
            values = getattr(payload, term)
            if values is None or not self._gain_slots.numel():
                continue
            values = values.index_select(1, self._gain_columns)
            if rows.dtype == torch.bool:
                current = buffer.index_select(1, self._gain_slots)
                values = torch.where(rows[:, None] & torch.isfinite(values), values, current)
                buffer.index_copy_(1, self._gain_slots, values)
            else:
                buffer[rows[:, None], self._gain_slots[None, :]] = values

        if payload.geom_friction is not None:
            self._apply_geom_friction_override(rows, payload.geom_friction, torch)
        if payload.body_mass is not None or payload.body_ipos is not None:
            self._apply_link_inertia_override(rows, payload.body_mass, payload.body_ipos, torch)

    def tensor_validation_verdicts(self) -> DeviceVerdicts | None:
        return self._reset_verdicts if len(self._reset_verdicts) else None

    def _apply_geom_friction_override(self, rows: Any, values: Any, torch: Any) -> None:
        """Write selected-row three-axis geom friction into collider buffers.

        The DR contract carries all three axes.  MotrixSim currently consumes
        only the sliding component at contact generation, but the materialized
        per-world buffer is three-wide, so the adapter preserves the full table
        and keeps spin/roll at their baseline values.

        ``rows`` may be an int64 index tensor or a bool mask; the bool mask
        path uses full-width ``torch.where`` per collider for graph
        capturability.
        """
        if not self._num_geoms:
            return

        # Bool-mask path: fixed-shape per-collider masked writes.
        if isinstance(rows, torch.Tensor) and rows.dtype == torch.bool:
            buffer = self._override_views["collider_friction"]
            # values may be (num_envs, num_geoms) or (num_envs, num_geoms, 3);
            # normalize to a 2-D per-env value before the masked write.
            for geom_index, indices in enumerate(self._geom_collider_indices):
                if not indices:
                    continue
                new_val = values[:, geom_index]
                if new_val.ndim == 1:
                    new_val = new_val.unsqueeze(-1)  # (num_envs, 1)
                # new_val is now (num_envs, 1) or (num_envs, 3); both broadcast
                # against current (num_envs, 3).
                for collider_index in indices:
                    current = buffer[:, collider_index, :]
                    applied = torch.where(new_val >= 0.0, new_val, current)
                    buffer[:, collider_index, :] = torch.where(rows.unsqueeze(-1), applied, current)
            return

        if int(rows.numel()) == 0:
            return
        collider_ids: list[int] = []
        geom_columns: list[int] = []
        for geom_index, indices in enumerate(self._geom_collider_indices):
            for collider_index in indices:
                collider_ids.append(collider_index)
                geom_columns.append(geom_index)
        if not collider_ids:
            return
        buffer = self._override_views["collider_friction"]
        collider_tensor = torch.as_tensor(collider_ids, dtype=torch.int64, device=values.device)
        column_tensor = torch.as_tensor(geom_columns, dtype=torch.int64, device=values.device)
        current = buffer[rows[:, None], collider_tensor[None, :], :]
        selected = values.index_select(1, column_tensor)
        # Match the engine setter: negative values are ignored per element.
        selected = torch.where(selected >= 0.0, selected, current)
        buffer[rows[:, None], collider_tensor[None, :], :] = selected

    def _prepare_link_inertia_indices(self, device: Any, torch: Any) -> None:
        if self._link_inertia_indices is not None:
            return
        physical = np.flatnonzero(self._link_low_indices >= 0)
        low = self._link_low_indices[physical]
        unique_low, first_link = np.unique(low, return_index=True)
        self._link_inertia_indices = (
            torch.as_tensor(physical, dtype=torch.int64, device=device),
            torch.as_tensor(low, dtype=torch.int64, device=device),
            torch.as_tensor(unique_low, dtype=torch.int64, device=device),
            torch.as_tensor(first_link, dtype=torch.int64, device=device),
            torch.as_tensor(self._link_low_rotations[physical], device=device),
            torch.as_tensor(self._link_low_translations[physical], device=device),
            torch.arange(self._num_envs, dtype=torch.int64, device=device),
        )

    def _apply_link_inertia_override(
        self, rows: Any, mass_values: Any, com_values: Any, torch: Any
    ) -> None:
        """Apply link mass / CoM for index rows or a fixed-width mask.

        MotrixSim keys the simulated inertia by low-level body while the public
        override is keyed by apparent link.  This mirrors the engine's
        ``ApparentLink::override_mass_properties``: store the link's own final
        values and scatter the difference of the corresponding spatial inertias
        into the shared low-level body, so welded children compose exactly like
        the CPU engine.
        """
        if int(rows.numel()) == 0:
            return
        overrides = self._low_overrides()
        mass_buffer = self._override_views["apparent_link_mass"]
        com_buffer = self._override_views["apparent_link_com"]
        device = mass_buffer.device
        self._prepare_link_inertia_indices(device, torch)
        assert self._link_inertia_indices is not None
        physical_columns, low_indices, unique_low, first_link, rotation, translation, all_rows = (
            self._link_inertia_indices
        )
        if not physical_columns.numel():
            return
        mask = rows if rows.dtype == torch.bool else None
        if mask is not None:
            rows = all_rows
        old_mass = mass_buffer.index_select(0, rows).index_select(1, physical_columns)
        old_com = com_buffer.index_select(0, rows).index_select(1, physical_columns)
        new_mass = old_mass.clone()
        new_com = old_com.clone()
        if mass_values is not None:
            candidate = mass_values.index_select(1, physical_columns)
            new_mass = torch.where(
                torch.isfinite(candidate) & (candidate > 0.0), candidate, new_mass
            )
        if com_values is not None:
            candidate = com_values.index_select(1, physical_columns)
            new_com = torch.where(
                torch.isfinite(candidate).all(dim=-1, keepdim=True), candidate, new_com
            )

        if mask is not None:
            new_mass = torch.where(mask[:, None], new_mass, old_mass)
            new_com = torch.where(mask[:, None, None], new_com, old_com)

        delta_mass = new_mass - old_mass
        delta_h = new_mass.unsqueeze(-1) * new_com - old_mass.unsqueeze(-1) * old_com

        def skew(vector: Any) -> Any:
            zero = torch.zeros_like(vector[..., 0])
            return torch.stack(
                (
                    torch.stack((zero, -vector[..., 2], vector[..., 1]), dim=-1),
                    torch.stack((vector[..., 2], zero, -vector[..., 0]), dim=-1),
                    torch.stack((-vector[..., 1], vector[..., 0], zero), dim=-1),
                ),
                dim=-2,
            )

        def parallel_axis(center: Any) -> Any:
            eye = torch.eye(3, dtype=center.dtype, device=center.device)
            squared = (center * center).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
            return squared * eye - center.unsqueeze(-1) * center.unsqueeze(-2)

        delta_inertia = new_mass.unsqueeze(-1).unsqueeze(-1) * parallel_axis(
            new_com
        ) - old_mass.unsqueeze(-1).unsqueeze(-1) * parallel_axis(old_com)

        h_minus_mr = delta_h - delta_mass.unsqueeze(-1) * translation
        skew_translation = skew(translation).unsqueeze(0)
        inner = (
            delta_inertia + skew_translation @ skew(delta_h) + skew(h_minus_mr) @ skew_translation
        )
        low_delta_h = torch.einsum("hij,rhj->rhi", rotation, h_minus_mr)
        low_delta_inertia = rotation.unsqueeze(0) @ inner @ rotation.transpose(-1, -2).unsqueeze(0)
        delta = torch.cat(
            (
                delta_mass.unsqueeze(-1),
                low_delta_h,
                low_delta_inertia.reshape(delta_mass.shape[0], physical_columns.numel(), 9),
            ),
            dim=-1,
        )

        num_low_links = int(overrides.link_rbi.shape[1])
        rbi = self._flat_torch_view(
            "link_rbi",
            overrides.link_rbi,
            self._num_envs * num_low_links * _RBI_STRIDE,
            torch,
        ).view(self._num_envs * num_low_links, _RBI_STRIDE)
        flat_indices = rows.unsqueeze(1) * num_low_links + low_indices.unsqueeze(0)
        rbi.index_add_(0, flat_indices.reshape(-1), delta.reshape(-1, _RBI_STRIDE))

        mass_buffer[rows.unsqueeze(1), physical_columns.unsqueeze(0)] = new_mass
        com_buffer[rows.unsqueeze(1), physical_columns.unsqueeze(0)] = new_com

        updated = rbi.index_select(0, flat_indices.reshape(-1)).view(
            int(rows.shape[0]), physical_columns.numel(), _RBI_STRIDE
        )
        mass = updated[..., 0]
        first_moment = updated[..., 1:4]
        inertia = updated[..., 4:13].view(int(rows.shape[0]), physical_columns.numel(), 3, 3)
        matrices = torch.zeros(
            (int(rows.shape[0]), physical_columns.numel(), 6, 6),
            dtype=updated.dtype,
            device=device,
        )
        matrices[..., :3, :3] = inertia
        matrices[..., :3, 3:] = skew(first_moment)
        matrices[..., 3:, :3] = skew(first_moment).transpose(-1, -2)
        matrices[..., 3:, 3:] = mass.unsqueeze(-1).unsqueeze(-1) * torch.eye(
            3, dtype=updated.dtype, device=device
        )

        spatial = self._flat_torch_view(
            "link_spatial_inertia",
            overrides.link_spatial_inertia,
            self._num_envs * num_low_links * _SPATIAL_INERTIA_STRIDE,
            torch,
        ).view(self._num_envs * num_low_links, _SPATIAL_INERTIA_STRIDE)
        chosen = matrices.index_select(1, first_link).flatten(start_dim=2)
        spatial_indices = rows.unsqueeze(1) * num_low_links + unique_low.unsqueeze(0)
        if mask is not None:
            chosen = torch.where(mask[:, None, None], chosen, spatial[spatial_indices])
        spatial[spatial_indices.reshape(-1)] = chosen.reshape(-1, _SPATIAL_INERTIA_STRIDE)

    def _interval_term_handlers(self) -> dict[str, Callable[[IntervalTermOp], None]]:
        """Return the interval terms this adapter can apply without a reset."""
        if self._interval_handler_cache is None:
            handlers: dict[str, Callable[[IntervalTermOp], None]] = {}
            if self._root_body_id is not None and self._root_qvel_start is not None:
                handlers = {
                    INTERVAL_TERM_BODY_LINEAR_VELOCITY_DELTA: (
                        self._apply_root_linear_velocity_delta
                    ),
                    INTERVAL_TERM_BODY_ANGULAR_VELOCITY_DELTA: (
                        self._apply_root_angular_velocity_delta
                    ),
                }
            self._interval_handler_cache = handlers
        return self._interval_handler_cache

    def _validate_root_velocity_op(self, op: IntervalTermOp) -> np.ndarray:
        """Validate a root velocity-delta op and return its ``(W, 3)`` payload."""
        if self._root_body_id is None:
            raise NotImplementedError(
                "motrix_gpu interval velocity perturbation requires a leading free root"
            )
        body_ids = _interval_value_to_numpy(require_op_body_ids(op))
        if (
            body_ids.ndim != 1
            or not np.issubdtype(body_ids.dtype, np.integer)
            or np.issubdtype(body_ids.dtype, np.bool_)
        ):
            raise TypeError(
                "motrix_gpu interval velocity body_ids must be a 1-D integer array, "
                f"got shape={body_ids.shape}, dtype={body_ids.dtype}"
            )
        if body_ids.size != 1 or int(body_ids[0]) != self._root_body_id:
            raise NotImplementedError(
                "motrix_gpu interval velocity perturbation only supports the configured "
                f"root body (id={self._root_body_id}); received body_ids="
                f"{body_ids.tolist()}"
            )
        payload = _interval_value_to_numpy(op.payload)
        expected = (self._num_envs, 1, 3)
        if payload.shape != expected:
            raise ValueError(
                f"motrix_gpu interval velocity payload must have shape {expected}, "
                f"got {payload.shape}"
            )
        if not np.issubdtype(payload.dtype, np.floating) or not np.isfinite(payload).all():
            raise ValueError("motrix_gpu interval velocity payload must be finite floating data")
        return payload[:, 0, :]

    def _apply_root_linear_velocity_delta(self, op: IntervalTermOp) -> None:
        """Add a world-frame linear velocity delta to the root qvel slice."""
        payload = self._validate_root_velocity_op(op)
        assert self._root_qvel_start is not None
        import torch

        qvel = self._field_view("qvel")
        delta = torch.as_tensor(payload, dtype=qvel.dtype, device=qvel.device)
        qvel[:, self._root_qvel_start : self._root_qvel_start + 3] += delta
        self._refresh_after_interval_velocity_write()

    def _apply_root_angular_velocity_delta(self, op: IntervalTermOp) -> None:
        """Rotate a world-frame angular delta into the root's local qvel frame."""
        payload = self._validate_root_velocity_op(op)
        assert self._root_qvel_start is not None and self._root_qpos_start is not None
        import torch

        qpos = self._field_view("qpos")
        qvel = self._field_view("qvel")
        quaternion = qpos[:, self._root_qpos_start + 3 : self._root_qpos_start + 7]
        vector_part = quaternion[:, :3]
        scalar_part = quaternion[:, 3]
        world_delta = torch.as_tensor(payload, dtype=qvel.dtype, device=qvel.device)
        # MuJoCo free-joint angular velocity is local-frame; the inverse rotation
        # is v - w * (2 q_v x v) + q_v x (2 q_v x v).
        cross = 2.0 * torch.linalg.cross(vector_part, world_delta)
        local_delta = (
            world_delta - scalar_part.unsqueeze(-1) * cross + torch.linalg.cross(vector_part, cross)
        )
        qvel[:, self._root_qvel_start + 3 : self._root_qvel_start + 6] += local_delta
        self._refresh_after_interval_velocity_write()

    def _refresh_after_interval_velocity_write(self) -> None:
        """Order the Torch qvel write before Warp kinematics and republish views."""
        self._refresh_high_level()
        self._refresh_tracked_sensors()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def materialize(self) -> None:
        self._require_open()

    def close(self) -> None:
        if self._closed:
            return
        import torch

        torch.cuda.synchronize(self._device)
        self._data.close()
        self._closed = True
        self._state_torch_views.clear()
        self._struct_flat_views.clear()
        self._interval_handler_cache = None
        self._tracked_field_views.clear()
        self._tracked_buffers.clear()
        self._sensor_torch_views.clear()
        self._frame_quat_public_views.clear()
        self._gain_views.clear()
        self._override_views.clear()
        self._tracked_packed = None
        self._tracked_column_indices = None
        self._tracked_quat_stage = None
        self._tracked_wxyz_perm = None
        self._public_column_index = None
        self._public_source_index = None
        self._engine_source_index = None
        self._link_inertia_indices = None
        self._qpos_public = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._data = None
        self._model = None

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("motrix_gpu backend is closed")

    @contextmanager
    def torch_graph_capture_scope(
        self,
        torch_stream: Any | None = None,
        *,
        capture_error_mode: str = "global",
    ) -> Iterator[ExternalGraphCapture]:
        """Record motrix_gpu tensor stepping and mask reset into a Torch CUDA graph.

        The active Torch capture stream is bridged to Warp via
        ``wp.stream_from_torch``, made current with ``wp.ScopedStream``, and
        registered as an *external* capture so the engine's temporary
        allocations become graph memory nodes. The engine's internal step
        graph is bypassed during external capture so its Warp kernels are
        recorded directly into the owner's graph.
        """
        import torch

        wp = self._deps.warp
        modes = {
            "global": wp.CaptureMode.GLOBAL,
            "thread_local": wp.CaptureMode.THREAD_LOCAL,
            "relaxed": wp.CaptureMode.RELAXED,
        }
        if capture_error_mode not in modes:
            raise ValueError(
                f"capture_error_mode must be one of {tuple(modes)}, got {capture_error_mode!r}"
            )
        self._require_open()
        if self._external_capture is not None:
            raise GraphCaptureError("motrix_gpu already has an active external capture")
        device = self._field_view("qpos").device
        stream = torch.cuda.current_stream(device) if torch_stream is None else torch_stream
        if not isinstance(stream, torch.cuda.Stream) or stream.device != device:
            raise ValueError(f"motrix_gpu capture stream must be a torch.cuda.Stream on {device}")
        warp_stream = wp.stream_from_torch(stream)
        if not warp_stream.is_capturing:
            raise GraphCaptureError(
                "motrix_gpu torch_graph_capture_scope must run inside an active "
                "Torch CUDA graph capture on its stream"
            )
        capture = ExternalGraphCapture(torch_stream=stream, backend_stream=warp_stream)
        warp_capture = wp.ScopedCapture(
            stream=warp_stream,
            external=True,
            capture_mode=modes[capture_error_mode],
        )
        self._external_capture = capture
        try:
            with (
                closing_capture(capture),
                wp.ScopedStream(warp_stream, sync_enter=False, sync_exit=False),
                warp_capture,
            ):
                yield capture
            capture.backend_graph = warp_capture.graph
        finally:
            self._external_capture = None

    def _refresh_high_level(self) -> None:
        """Publish link kinematics and kinematics sensors without stepping dynamics."""
        with self._warp_execution():
            self._model.forward_kinematic(self._data)

    def _device_arrays(self) -> dict[str, Any]:
        return {
            "qpos": self._data.dof_pos,
            "qvel": self._data.dof_vel,
            "ctrl": self._data.actuator_ctrls,
        }

    def _field_view(self, name: str) -> Any:
        import torch

        view = self._state_torch_views.get(name)
        if view is None:
            try:
                view = torch.from_dlpack(self._device_arrays()[name])  # pyright: ignore[reportPrivateImportUsage]
            except Exception as exc:  # pragma: no cover - CUDA Torch is required
                raise RuntimeError(
                    "motrix_gpu device views require a CUDA Torch build that can wrap "
                    "the MotrixSim GPU arrays through DLPack"
                ) from exc
            self._state_torch_views[name] = view
        return view

    def _validate_ctrl(self, ctrl: Any, torch: Any) -> Any:
        if not isinstance(ctrl, torch.Tensor):
            raise TypeError("motrix_gpu tensor ctrl must be a torch.Tensor")
        if ctrl.dtype != torch.float32:
            raise TypeError(f"motrix_gpu tensor ctrl must be float32, got {ctrl.dtype}")
        if tuple(ctrl.shape) != (self._num_envs, self._nu):
            raise ValueError(
                f"motrix_gpu tensor ctrl must have shape {(self._num_envs, self._nu)}, "
                f"got {tuple(ctrl.shape)}"
            )
        if not ctrl.is_contiguous():
            raise ValueError("motrix_gpu tensor ctrl must be contiguous")
        return ctrl

    def _validate_rows(self, rows: Any, torch: Any) -> Any:
        if not isinstance(rows, torch.Tensor):
            raise TypeError("motrix_gpu env_indices must be a torch.Tensor")
        if rows.dtype != torch.int64 or rows.ndim != 1 or not rows.is_contiguous():
            raise TypeError("motrix_gpu env_indices must be a contiguous 1-D int64 tensor")
        if rows.numel():
            values = rows.detach().cpu()
            if int(values.min()) < 0 or int(values.max()) >= self._num_envs:
                raise ValueError("motrix_gpu env_indices must be in [0, num_envs)")
            if int(torch.unique(values).numel()) != int(values.numel()):
                raise ValueError("motrix_gpu env_indices must be unique")
        return rows

    def _validate_mask(self, mask: Any, torch: Any) -> Any:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("motrix_gpu reset_mask must be a torch.Tensor")
        if mask.dtype != torch.bool or mask.ndim != 1 or not mask.is_contiguous():
            raise TypeError("motrix_gpu reset_mask must be a contiguous 1-D bool tensor")
        if int(mask.shape[0]) != self._num_envs:
            raise ValueError(
                f"motrix_gpu reset_mask must have shape {(self._num_envs,)}, "
                f"got {tuple(mask.shape)}"
            )
        return mask

    def _validate_state(self, name: str, value: Any, rows: int, width: int, torch: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"motrix_gpu {name} must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"motrix_gpu {name} must be float32, got {value.dtype}")
        if tuple(value.shape) != (rows, width):
            raise ValueError(
                f"motrix_gpu {name} must have shape {(rows, width)}, got {tuple(value.shape)}"
            )
        if not value.is_contiguous():
            raise ValueError(f"motrix_gpu {name} must be contiguous")
        return value


__all__ = ["MotrixGpuBackend"]
