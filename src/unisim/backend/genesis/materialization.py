"""Cold-path scene materialization for the independent ``genesis`` backend.

Genesis 1.3.3 drops three MJCF features UniLab relies on, as measured in
the Genesis 1.3.3 feasibility audit (#1372 §3): ``<keyframe>``,
the global ``<option>`` block, and the whole ``<sensor>`` block.  This module
compensates on the cold path exactly as the report prescribes: keyframe and
sensor metadata are scanned once with the ``mujoco`` package (the isaacgym
parent-process metadata-scan pattern) and cached — hot paths never parse XML;
global options arrive as explicit owner fields; and ``gs.init`` host-side
effects (torch default device/dtype and RNG) are contained by
:func:`preserve_torch_globals` plus the one-session-per-process guard.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from unisim.backend.materialization_common import TemporarySceneCleanup
from unisim.scene import SceneCfg

# Supported spellings for the explicit owner-YAML global options.  Genesis
# defaults apply when the owner leaves them unset (REPORT #1372 §5.2).
_INTEGRATOR_ENUMS = {
    "euler": "Euler",
    "implicitfast": "implicitfast",
    "approximate_implicitfast": "approximate_implicitfast",
}
_CONSTRAINT_SOLVER_ENUMS = {
    "newton": "Newton",
    "cg": "CG",
}
_FRICTION_CONE_ENUMS = {
    "pyramidal": "pyramidal",
    "elliptic": "elliptic",
}

# Contact ``data="found"`` equivalent: per-link net contact force magnitude
# threshold in newtons (REPORT #1372 §3.4: standing G1 foot reads ~138 N).
CONTACT_FOUND_FORCE_THRESHOLD_N = 1.0


@dataclass(frozen=True)
class GenesisSensorPlan:
    """One MJCF sensor mapped onto a Genesis-readable equivalent.

    ``body_name`` is the owning link for site sensors and the robot-side geom
    body for contact sensors.  ``site_pos``/``site_quat`` (wxyz) are the local
    site frame in the body frame; both are ``None`` for contact sensors.
    """

    name: str
    kind: str
    dim: int
    body_name: str
    object_name: str
    reference_type: int
    reference_id: int
    site_pos: tuple[float, ...] | None
    site_quat: tuple[float, ...] | None


@dataclass(frozen=True)
class GenesisModelMetadata:
    """MJCF metadata scanned once on the cold path (never on hot paths)."""

    source_model_file: str
    cleanup_handle: Any | None
    nq: int
    nv: int
    nbody: int
    root_qpos_dim: int
    root_qvel_dim: int
    joint_names: tuple[str, ...]
    joint_qpos_adrs: tuple[int, ...]
    joint_dof_adrs: tuple[int, ...]
    body_names: tuple[str, ...]
    geom_names: tuple[str, ...]
    geom_body_names: tuple[str, ...]
    geom_types: tuple[int, ...]
    geom_sizes: np.ndarray
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_ctrl_range: np.ndarray
    actuator_kp: np.ndarray
    actuator_kv: np.ndarray
    keyframe_qpos: tuple[tuple[str, np.ndarray], ...]
    default_qpos: np.ndarray
    joint_range: np.ndarray | None
    dof_armature: np.ndarray
    gravity: np.ndarray
    body_mass: np.ndarray
    body_ipos: np.ndarray
    body_inertia: np.ndarray
    body_iquat: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    sensor_plans: tuple[GenesisSensorPlan, ...]


@dataclass(frozen=True)
class GenesisPortableEntitySource:
    """Independent normalized MJCF sources for one public entity."""

    name: str
    model_files: tuple[str, ...]
    metadata: tuple[GenesisModelMetadata, ...]


@dataclass(frozen=True)
class GenesisPortableSources:
    """Cold-path Genesis inputs derived from the common portable compiler."""

    entities: tuple[GenesisPortableEntitySource, ...]
    cleanup_handle: Any


def genesis_balanced_variant_mapping(num_variants: int, num_envs: int) -> np.ndarray:
    """Return Genesis 1.3.3's documented heterogeneous env/variant mapping.

    Genesis dispatches variants in contiguous balanced blocks (the first
    remainder-sized block gets one extra environment). This local owner-side
    implementation deliberately does not call Genesis' private solver helper;
    construction accepts an assignment only when it is exactly this mapping.
    """

    if num_variants < 1:
        raise ValueError("genesis heterogeneous variants require at least one source")
    if num_envs < 1:
        raise ValueError("genesis heterogeneous variants require at least one environment")
    if num_envs >= num_variants:
        base, extra = divmod(num_envs, num_variants)
        return np.repeat(
            np.arange(num_variants, dtype=np.int32),
            np.r_[np.full(extra, base + 1), np.full(num_variants - extra, base)],
        )
    return np.arange(num_envs, dtype=np.int32)


def validate_genesis_variant_assignment(
    assignment: np.ndarray, num_variants: int
) -> np.ndarray:
    """Require the exact mapping Genesis will materialize natively."""

    values = np.asarray(assignment, dtype=np.int32)
    if values.ndim != 1:
        raise ValueError("genesis variant assignment must be one-dimensional")
    expected = genesis_balanced_variant_mapping(num_variants, values.size)
    if not np.array_equal(values, expected):
        raise ValueError(
            "genesis heterogeneous variant assignment must exactly equal the native "
            f"balanced mapping {expected.tolist()}; got {values.tolist()}. Reorder the "
 "assignment until Genesis exposes a public owner-controlled mapping."
        )
    return values.copy()


def prepare_genesis_portable_sources(
    mujoco: Any, scene: SceneCfg, layout: Any
) -> GenesisPortableSources:
    """Normalize common portable sources into independent Genesis MJCF files.

    The common compiler remains the sole layout, identity, and source
    relationship authority. This owner adapter only serializes each already
    normalized ``MjSpec`` and scans the resulting source for native binding.
    """

    from unisim.mjcf_compiler import load_entity_source

    if scene.fragment_files:
        raise NotImplementedError(
            "genesis portable entity sensors from cross-entity fragments are not yet mapped"
        )
    if any(entity.mirror_of is not None for entity in scene.entity_assets):
        raise NotImplementedError("genesis portable kinematic mirrors are not yet supported")
    if any(entity.root_mode == "kinematic" for entity in scene.entity_assets):
        raise NotImplementedError("genesis portable kinematic entities are not yet supported")

    binding = scene.entity_variant
    directory = tempfile.TemporaryDirectory(prefix="unisim-genesis-entities-")
    entities: list[GenesisPortableEntitySource] = []
    try:
        for entity_spec in scene.entity_assets:
            owner = layout.get_entity(entity_spec.name)
            paths: list[str]
            if binding is not None and entity_spec.name == binding.target_entity:
                if owner.kind != "rigid" or len(owner.body_names) != 1:
                    raise NotImplementedError(
                        "genesis heterogeneous variants are native only for single-link "
                        f"rigid entities; entity {owner.name!r} is {owner.kind} with "
                        f"{len(owner.body_names)} bodies"
                    )
                paths = [str(item.model_file) for item in binding.plan.variants]
            else:
                assert entity_spec.source is not None
                paths = [str(entity_spec.source.model_file)]

            files: list[str] = []
            metadata: list[GenesisModelMetadata] = []
            for variant, source_path in enumerate(paths):
                spec, _, _ = load_entity_source(entity_spec, source_path, mirror=False)
                model_file = str(Path(directory.name) / f"{entity_spec.name}-{variant}.xml")
                spec.to_file(model_file)
                item = scan_genesis_model_metadata(
                    mujoco, SceneCfg(model_file=model_file)
                )
                _validate_portable_source(owner, item)
                files.append(model_file)
                metadata.append(item)
            entities.append(
                GenesisPortableEntitySource(entity_spec.name, tuple(files), tuple(metadata))
            )
    except BaseException:
        directory.cleanup()
        raise
    return GenesisPortableSources(tuple(entities), directory)


def _validate_portable_source(owner: Any, metadata: GenesisModelMetadata) -> None:
    """Compare one normalized source with the frozen public entity layout."""

    expected_joints = tuple(joint.name for joint in owner.joints)
    if metadata.joint_names != expected_joints:
        raise RuntimeError(
            f"genesis entity {owner.name!r} joint names/order {metadata.joint_names} "
            f"differ from the public layout {expected_joints}"
        )
    if metadata.nq != len(owner.qpos_indices) or metadata.nv != len(owner.qvel_indices):
        raise RuntimeError(
            f"genesis entity {owner.name!r} state dimensions "
            f"{metadata.nq}/{metadata.nv} differ from the public layout "
            f"{len(owner.qpos_indices)}/{len(owner.qvel_indices)}"
        )
    expected_root = (7, 6) if owner.root_mode == "floating" else (0, 0)
    if (metadata.root_qpos_dim, metadata.root_qvel_dim) != expected_root:
        raise RuntimeError(
            f"genesis entity {owner.name!r} root dimensions differ from the public layout"
        )
    if metadata.actuator_names != tuple(owner.actuator_names):
        raise RuntimeError(
            f"genesis entity {owner.name!r} actuator names/order differ from the public layout"
        )
    if metadata.actuator_joint_names != tuple(owner.actuator_joint_names):
        raise RuntimeError(
            f"genesis entity {owner.name!r} actuator targets differ from the public layout"
        )
    if metadata.geom_names != tuple(geom.name for geom in owner.geoms) or (
        metadata.geom_body_names != tuple(geom.body_name for geom in owner.geoms)
    ):
        raise RuntimeError(
            f"genesis entity {owner.name!r} geom names/ownership differ from "
            "the public layout"
        )


def validate_genesis_portable_sensor_plans(
    mujoco: Any, sources: GenesisPortableSources, layout: Any
) -> tuple[GenesisSensorPlan, ...]:
    """Bind the bounded portable source-sensor subset to public names.

    Genesis does not import MJCF sensors. Portable support is deliberately
    limited to entity-owned, world-referenced site ``FramePos``/``FrameQuat``
    declarations whose complete semantic identity is identical in every source
    variant. The returned plans use qualified public sensor/body names; the
    backend computes their values from audited native link state.
    """

    source_by_entity = {source.name: source for source in sources.entities}
    public_plans: list[GenesisSensorPlan] = []
    public_names: set[str] = set()
    for owner in layout.entities:
        source = source_by_entity[owner.name]
        reference = source.metadata[0].sensor_plans
        for variant, metadata in enumerate(source.metadata[1:], start=1):
            if metadata.sensor_plans != reference:
                raise NotImplementedError(
                    f"genesis entity {owner.name!r} portable sensor identity differs "
                    f"between variants 0 and {variant}"
                )
        for plan in reference:
            if plan.kind not in ("framepos", "framequat"):
                raise NotImplementedError(
                    "genesis portable entity source sensors support only "
                    "world-referenced site FramePos/FrameQuat sensors"
                )
            expected_dim = 3 if plan.kind == "framepos" else 4
            if plan.dim != expected_dim:
                raise RuntimeError("genesis portable site sensor dimension disagrees with its type")
            if plan.reference_type != int(mujoco.mjtObj.mjOBJ_UNKNOWN) or (
                plan.reference_id != -1
            ):
                raise NotImplementedError(
                    "genesis portable entity source sensors support only "
                    "world-referenced site FramePos/FrameQuat sensors"
                )
            if plan.body_name not in owner.body_names or not plan.object_name:
                raise NotImplementedError(
                    f"genesis portable site sensor {plan.name!r} must reference a named "
                    f"site owned by entity {owner.name!r}"
                )
            if plan.site_pos is None or plan.site_quat is None:
                raise RuntimeError("genesis portable site sensor has malformed site identity")
            public_name = f"{owner.name}/{plan.name}"
            if public_name in public_names:
                raise RuntimeError(
                    f"genesis portable site sensor name {public_name!r} is not unique"
                )
            public_names.add(public_name)
            public_plans.append(
                replace(
                    plan,
                    name=public_name,
                    body_name=f"{owner.name}/{plan.body_name}",
                )
            )
    return tuple(public_plans)


@contextmanager
def preserve_torch_globals(torch: Any) -> Iterator[None]:
    """Snapshot torch global defaults and restore them after ``gs.init``.

    ``gs.init`` mutates the host process: it forces
    ``torch.set_default_device("cuda:0")`` on the GPU lane, and passing
    ``seed=`` would reseed the global RNG (REPORT #1372 §3.5 [10]).  The
    adapter never passes a seed and restores default device/dtype plus the
    torch RNG state so the training process observes no pollution
    (REPORT #1372 §5.8).
    """
    default_device = torch.get_default_device()
    default_dtype = torch.get_default_dtype()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_default_device(default_device)
        torch.set_default_dtype(default_dtype)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


_SESSION_ACTIVE = False
_SESSION_DESTROYED = False
_SESSION_DEVICE_ID: int | None = None


def _pin_cuda_visible_devices(torch: Any, device_id: int) -> int:
    """Pin ``CUDA_VISIBLE_DEVICES`` to the requested device; return the in-process index.

    Quadrants (the Genesis 1.3.x compute runtime) binds its CUDA runtime to
    the first entry of ``CUDA_VISIBLE_DEVICES`` regardless of torch's current
    device — ``torch.cuda.set_device(N)`` alone leaves the engine on physical
    GPU 0 and crashes later with cross-device illegal-memory errors (verified
    on genesis_world 1.3.3 / Quadrants 1.3.0, UniLab issue #1508).  The only
    owner-layer way to place a session on another physical GPU is to shrink
    visibility to that GPU before the first CUDA context exists; afterwards
    the in-process device index is 0.

    ``device_id`` addresses the *current* visibility namespace: without the
    variable set it is the host index, otherwise it indexes into the existing
    entries (physical indices or UUID strings).  Returns 0 on success so the
    session records the in-process index actually used.
    """

    if bool(torch.cuda.is_initialized()):
        raise RuntimeError(
            f"genesis device_id={device_id} requires pinning CUDA_VISIBLE_DEVICES "
            "before any CUDA context is created in this process, but torch CUDA "
            "is already initialized; select the device earlier (entrypoint cold "
            "path) or set CUDA_VISIBLE_DEVICES before launching the process"
        )
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()] if raw else None
    if entries is None:
        device_count = int(torch.cuda.device_count())
        if device_id >= device_count:
            raise ValueError(
                f"genesis device_id={device_id} is out of range; "
                f"torch.cuda.device_count()={device_count}"
            )
        target = str(device_id)
    else:
        if device_id >= len(entries):
            raise ValueError(
                f"genesis device_id={device_id} is out of range for "
                f"CUDA_VISIBLE_DEVICES={raw!r} ({len(entries)} entr(ies))"
            )
        target = entries[device_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = target
    # ``device_count`` is lru-cached; drop any pre-pin host-wide count so
    # later callers observe the pinned single-device namespace.
    cache_clear = getattr(torch.cuda.device_count, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    return 0


def _resolve_genesis_device_id(torch: Any, device_id: int | None) -> int | None:
    """Validate and select the CUDA index used by a Genesis session."""

    if device_id is not None:
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError(
                f"genesis device_id must be a non-negative integer or None, got {device_id!r}"
            )
        if device_id > 0:
            # The engine only honors the first visible device; pin visibility
            # and continue with the in-process index 0.  The pin must run
            # before any torch CUDA query (even ``is_available`` latches
            # CUDA_VISIBLE_DEVICES in the runtime), so it precedes the
            # availability check below.
            device_id = _pin_cuda_visible_devices(torch, device_id)
        if not bool(torch.cuda.is_available()):
            raise ValueError(
                f"genesis device_id={device_id} requires CUDA, but CUDA is unavailable"
            )
        # Genesis' ``gs.init`` consults the process-wide current CUDA device.
        # Bind it before entering the initialization routine; doing this after
        # ``gs.init`` is too late because Genesis has already allocated its
        # global backend on device zero.
        torch.cuda.set_device(device_id)
        return int(device_id)
    if bool(torch.cuda.is_available()):
        return int(torch.cuda.current_device())
    return None


def init_genesis_session(
    deps: Any,
    *,
    device_id: int | None = None,
    logging_level: str = "warning",
) -> None:
    """Initialize the process-wide Genesis session exactly once.

    Repeated ``init -> destroy`` cycles are functional but leak 200-450 MB of
    host RSS per cycle (REPORT #1372 §3.5 [9a]), so a long-lived training
    process gets exactly one session.  Multiple GenesisBackend instances share
    the live session (multi-scene coexistence is measured OK, REPORT [9b]);
    after :func:`destroy_genesis_session` any further construction fails
    closed with a clear error.
    """
    global _SESSION_ACTIVE, _SESSION_DESTROYED, _SESSION_DEVICE_ID
    if _SESSION_DESTROYED:
        raise RuntimeError(
            "genesis backend supports exactly one gs.init per process and the session "
            "was already destroyed; start a fresh process instead of re-initializing "
            "(init/destroy cycles leak host RSS, REPORT #1372 §3.5 [9a])."
        )
    selected_device_id = _resolve_genesis_device_id(deps.torch, device_id)
    if _SESSION_ACTIVE:
        if (
            selected_device_id is not None
            and _SESSION_DEVICE_ID is not None
            and selected_device_id != _SESSION_DEVICE_ID
        ):
            raise RuntimeError(
                "genesis session is already initialized on CUDA device "
                f"{_SESSION_DEVICE_ID}, cannot reuse it on device {selected_device_id}"
            )
        return
    gs = deps.genesis
    cuda_available = bool(deps.torch.cuda.is_available())
    backend_kind = gs.gpu if cuda_available else gs.cpu
    try:
        with preserve_torch_globals(deps.torch):
            gs.init(backend=backend_kind, logging_level=logging_level)
    except Exception as exc:
        raise RuntimeError(
            "genesis backend failed to initialize the Genesis runtime "
            f"(backend={'gpu' if cuda_available else 'cpu'}; only the gs.gpu lane is "
            f"validated by REPORT #1372): {type(exc).__name__}: {exc}"
        ) from exc
    _SESSION_ACTIVE = True
    _SESSION_DEVICE_ID = selected_device_id


def destroy_genesis_session(deps: Any) -> None:
    """Tear down the process-wide session; re-initialization stays forbidden."""
    global _SESSION_ACTIVE, _SESSION_DESTROYED, _SESSION_DEVICE_ID
    if not _SESSION_ACTIVE:
        return
    deps.genesis.destroy()
    _SESSION_ACTIVE = False
    _SESSION_DESTROYED = True
    _SESSION_DEVICE_ID = None


def _reset_session_state_for_tests() -> None:
    """Reset the lifecycle guard; test-only seam for the fake runtime lane."""
    global _SESSION_ACTIVE, _SESSION_DESTROYED, _SESSION_DEVICE_ID
    _SESSION_ACTIVE = False
    _SESSION_DESTROYED = False
    _SESSION_DEVICE_ID = None


def _map_global_option(name: str, value: str, table: dict[str, str], enum_ns: Any) -> Any:
    try:
        attr = table[value]
    except KeyError as exc:
        supported = ", ".join(sorted(table))
        raise ValueError(
            f"{name} must be one of: {supported}; got {value!r}. Genesis drops the MJCF "
            "global <option> block, so the owner YAML must declare a supported value."
        ) from exc
    return getattr(enum_ns, attr)


def build_genesis_scene(
    deps: Any,
    *,
    sim_dt: float,
    gravity: np.ndarray,
    integrator: str | None,
    constraint_solver: str | None,
    friction_cone: str | None,
    solver_iterations: int | None,
) -> Any:
    """Construct the unbuilt Genesis scene with explicit global options."""
    gs = deps.genesis
    rigid_kwargs: dict[str, Any] = {
        # Per-env DR setters (mass/frictionloss/kp/kv) are materialize-time
        # decisions and require batched link/dof info (REPORT #1372 §5.7).
        "batch_links_info": True,
        "batch_dofs_info": True,
    }
    if integrator is not None:
        rigid_kwargs["integrator"] = _map_global_option(
            "genesis_integrator", integrator, _INTEGRATOR_ENUMS, gs.integrator
        )
    if constraint_solver is not None:
        rigid_kwargs["constraint_solver"] = _map_global_option(
            "genesis_constraint_solver",
            constraint_solver,
            _CONSTRAINT_SOLVER_ENUMS,
            gs.constraint_solver,
        )
    if friction_cone is not None:
        rigid_kwargs["friction_cone"] = _map_global_option(
            "genesis_friction_cone", friction_cone, _FRICTION_CONE_ENUMS, gs.friction_cone
        )
    if solver_iterations is not None:
        rigid_kwargs["iterations"] = int(solver_iterations)
    gravity_tuple = tuple(float(component) for component in np.asarray(gravity, dtype=np.float64))
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=float(sim_dt), gravity=gravity_tuple),
        rigid_options=gs.options.RigidOptions(**rigid_kwargs),
        show_viewer=False,
    )


def _scan_sensor_plans(mujoco: Any, model: Any) -> tuple[GenesisSensorPlan, ...]:
    """Map the MJCF sensor table onto Genesis equivalents (REPORT §5.3).

    Genesis 1.3.3 does not import ``<sensor>`` at all.  Supported mappings:
    gyro/accelerometer/velocimeter -> IMU-class site sensors computed from
    link state; framepos/framequat/framezaxis -> link state plus site-frame
    math; contact ``data="found"`` -> per-link net contact force threshold.
    Anything else fails closed here, at the nearest cold path.
    """
    sensor_obj = mujoco.mjtObj.mjOBJ_SENSOR
    plans: list[GenesisSensorPlan] = []
    for sensor_id in range(int(model.nsensor)):
        name = mujoco.mj_id2name(model, sensor_obj, sensor_id)
        if not name:
            raise NotImplementedError(
                f"genesis backend requires named MJCF sensors; sensor id {sensor_id} is unnamed"
            )
        sensor_type = mujoco.mjtSensor(int(model.sensor_type[sensor_id]))
        dim = int(model.sensor_dim[sensor_id])
        if sensor_type == mujoco.mjtSensor.mjSENS_CONTACT:
            plans.append(_scan_contact_sensor(mujoco, model, sensor_id, str(name), dim))
            continue
        site_kinds = {
            mujoco.mjtSensor.mjSENS_GYRO: "gyro",
            mujoco.mjtSensor.mjSENS_ACCELEROMETER: "accelerometer",
            mujoco.mjtSensor.mjSENS_VELOCIMETER: "velocimeter",
            mujoco.mjtSensor.mjSENS_FRAMEPOS: "framepos",
            mujoco.mjtSensor.mjSENS_FRAMEQUAT: "framequat",
            mujoco.mjtSensor.mjSENS_FRAMEZAXIS: "framezaxis",
        }
        kind = site_kinds.get(sensor_type)
        if kind is None:
            raise NotImplementedError(
                f"genesis backend cannot map MJCF sensor {name!r} of type "
                f"{sensor_type.name}; supported types: contact(found), gyro, accelerometer, "
                "velocimeter, framepos, framequat, framezaxis (REPORT #1372 §3.4)."
            )
        if int(model.sensor_objtype[sensor_id]) != int(mujoco.mjtObj.mjOBJ_SITE):
            raise NotImplementedError(
                f"genesis backend maps {kind} sensors from MJCF sites only; sensor {name!r} "
                f"uses objtype {int(model.sensor_objtype[sensor_id])}."
            )
        site_id = int(model.sensor_objid[sensor_id])
        body_id = int(model.site_bodyid[site_id])
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if not site_name:
            raise NotImplementedError(
                f"genesis site sensor {name!r} references unnamed site id {site_id}"
            )
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        site_pos = tuple(float(v) for v in np.asarray(model.site_pos[site_id], dtype=np.float64))
        site_quat = tuple(float(v) for v in np.asarray(model.site_quat[site_id], dtype=np.float64))
        reference_type = int(model.sensor_reftype[sensor_id])
        reference_id = int(model.sensor_refid[sensor_id])
        if kind in ("framepos", "framequat", "framezaxis") and (
            reference_type != int(mujoco.mjtObj.mjOBJ_UNKNOWN) or reference_id != -1
        ):
            raise NotImplementedError(
                f"genesis backend maps {kind} sensor {name!r} only with a world reference"
            )
        if kind == "accelerometer" and not np.allclose(site_quat, (1.0, 0.0, 0.0, 0.0)):
            raise NotImplementedError(
                f"genesis backend maps accelerometer {name!r} onto an IMUSensor, whose "
                "euler_offset path is not a validated support lane; rotated accelerometer "
                "sites are rejected (gyro/velocimeter/frame sensors support rotation)."
            )
        plans.append(
            GenesisSensorPlan(
                name=str(name),
                kind=kind,
                dim=dim,
                body_name=str(body_name),
                object_name=str(site_name),
                reference_type=reference_type,
                reference_id=reference_id,
                site_pos=site_pos,
                site_quat=site_quat,
            )
        )
    return tuple(plans)


def _scan_contact_sensor(
    mujoco: Any, model: Any, sensor_id: int, name: str, dim: int
) -> GenesisSensorPlan:
    if dim != 1:
        raise NotImplementedError(
            f'genesis backend maps contact sensors with data="found" only (dim 1); '
            f"sensor {name!r} has dim {dim}."
        )
    if int(model.sensor_objtype[sensor_id]) != int(mujoco.mjtObj.mjOBJ_GEOM) or int(
        model.sensor_reftype[sensor_id]
    ) != int(mujoco.mjtObj.mjOBJ_GEOM):
        raise NotImplementedError(
            f"genesis backend maps contact sensors over geom pairs only; sensor {name!r} "
            "uses a non-geom object type."
        )
    geom1_id = int(model.sensor_objid[sensor_id])
    geom2_id = int(model.sensor_refid[sensor_id])
    body1_id = int(model.geom_bodyid[geom1_id])
    body2_id = int(model.geom_bodyid[geom2_id])
    if (body1_id > 0) == (body2_id > 0):
        raise NotImplementedError(
            f"genesis backend maps contact sensor {name!r} onto the robot-side link's net "
            "contact force; exactly one geom must belong to the world body "
            f"(got body ids {body1_id}/{body2_id})."
        )
    body_id = body1_id if body1_id > 0 else body2_id
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
    if not geom1_name:
        raise NotImplementedError(
            f"genesis contact sensor {name!r} references unnamed geom id {geom1_id}"
        )
    return GenesisSensorPlan(
        name=name,
        kind="contact",
        dim=dim,
        body_name=str(body_name),
        object_name=str(geom1_name),
        reference_type=int(model.sensor_reftype[sensor_id]),
        reference_id=geom2_id,
        site_pos=None,
        site_quat=None,
    )


def scan_genesis_model_metadata(mujoco: Any, scene: SceneCfg) -> GenesisModelMetadata:
    """Resolve the scene and scan MJCF metadata with the ``mujoco`` package."""
    if scene is None or not scene.model_file:
        raise ValueError("GenesisBackend requires SceneCfg.model_file")
    if scene.terrain is not None:
        raise NotImplementedError(
            "genesis backend does not support generated terrain or height-field scanners; "
            "select a flat owner YAML or a backend with terrain support."
        )
    temp_paths: list[str] = []
    if not scene.fragment_files:
        source_model_file = str(scene.model_file)
    else:
        # Cold-path-only helper, shared with the MuJoCo/mjwarp backends.
        from unisim.backend.mujoco.xml import materialize_scene_fragments

        source_model_file = materialize_scene_fragments(
            str(scene.model_file),
            fragment_files=scene.fragment_files,
        )
        temp_paths.append(source_model_file)

    model = mujoco.MjModel.from_xml_path(source_model_file)
    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    single_dof_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    root_qpos_dim, root_qvel_dim = (
        (7, 6) if int(model.njnt) > 0 and int(model.jnt_type[0]) == free_joint else (0, 0)
    )

    joint_names: list[str] = []
    joint_qpos_adrs: list[int] = []
    joint_dof_adrs: list[int] = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) not in single_dof_types:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise NotImplementedError(
                f"genesis backend requires named single-DoF joints; joint id {joint_id} is unnamed"
            )
        joint_names.append(str(name))
        joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_adrs.append(int(model.jnt_dofadr[joint_id]))

    actuator_names: list[str] = []
    actuator_joint_names: list[str] = []
    supported_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        transmission = int(model.actuator_trntype[actuator_id])
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if not name or transmission not in supported_transmissions or joint_id < 0:
            raise NotImplementedError(
                "genesis backend requires position actuators with a joint transmission; "
                f"actuator id {actuator_id} (name {name!r}, transmission {transmission}) "
                "is unsupported."
            )
        if int(model.jnt_type[joint_id]) not in single_dof_types:
            raise NotImplementedError(
                f"genesis backend requires actuators on single-DoF joints; actuator {name!r} "
                f"targets joint id {joint_id}."
            )
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_names.append(str(name))
        actuator_joint_names.append(str(joint_name))

    keyframes: list[tuple[str, np.ndarray]] = []
    for key_id in range(int(model.nkey)):
        key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
        if key_name is not None:
            keyframes.append(
                (str(key_name), np.asarray(model.key_qpos[key_id], dtype=np.float32).copy())
            )

    non_free_mask = np.asarray(model.jnt_type, dtype=np.int32) != free_joint
    joint_range = np.asarray(model.jnt_range, dtype=np.float32)[non_free_mask]
    body_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        for body_id in range(int(model.nbody))
    )
    geom_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        for geom_id in range(int(model.ngeom))
    )
    geom_body_names = tuple(
        str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            or ""
        )
        for geom_id in range(int(model.ngeom))
    )

    return GenesisModelMetadata(
        source_model_file=source_model_file,
        cleanup_handle=TemporarySceneCleanup(*temp_paths) if temp_paths else None,
        nq=int(model.nq),
        nv=int(model.nv),
        nbody=int(model.nbody),
        root_qpos_dim=root_qpos_dim,
        root_qvel_dim=root_qvel_dim,
        joint_names=tuple(joint_names),
        joint_qpos_adrs=tuple(joint_qpos_adrs),
        joint_dof_adrs=tuple(joint_dof_adrs),
        body_names=body_names,
        geom_names=geom_names,
        geom_body_names=geom_body_names,
        geom_types=tuple(int(value) for value in np.asarray(model.geom_type)),
        geom_sizes=np.asarray(model.geom_size, dtype=np.float32).copy(),
        actuator_names=tuple(actuator_names),
        actuator_joint_names=tuple(actuator_joint_names),
        actuator_ctrl_range=np.asarray(model.actuator_ctrlrange, dtype=np.float32).copy(),
        actuator_kp=np.asarray(model.actuator_gainprm[:, 0], dtype=np.float32).copy(),
        actuator_kv=np.asarray(-model.actuator_biasprm[:, 2], dtype=np.float32).copy(),
        keyframe_qpos=tuple(keyframes),
        default_qpos=np.asarray(model.qpos0, dtype=np.float32).copy(),
        joint_range=None if joint_range.size == 0 else joint_range.copy(),
        dof_armature=np.asarray(model.dof_armature, dtype=np.float32).copy(),
        gravity=np.asarray(model.opt.gravity, dtype=np.float32).copy(),
        body_mass=np.asarray(model.body_mass, dtype=np.float32).copy(),
        body_ipos=np.asarray(model.body_ipos, dtype=np.float32).copy(),
        body_inertia=np.asarray(model.body_inertia, dtype=np.float32).copy(),
        body_iquat=np.asarray(model.body_iquat, dtype=np.float32).copy(),
        body_pos=np.asarray(model.body_pos, dtype=np.float32).copy(),
        body_quat=np.asarray(model.body_quat, dtype=np.float32).copy(),
        sensor_plans=_scan_sensor_plans(mujoco, model),
    )
