"""SDK-free, source-reviewed semantic inventory for the declared adapters.

This module does not discover runtimes or confer runtime verification. DR, play,
body-wrench and fixed-variant support are resolved by their existing instance APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from unisim.adapters import ADAPTER_SPECS, adapter_spec
from unisim.backend.base import TensorDataPlane, TensorExecution, TensorProcessTopology
from unisim.capabilities import (
    CapabilityCondition,
    CapabilityDeclaration,
    CapabilityEvidence,
    CapabilityReport,
    CapabilityScope,
    SupportLevel,
)

# Pin source evidence to the implementation reviewed for the initial M1 inventory.
SOURCE_REVISION = "6d62d9ba40a1f637fb04da190fd38f64d02b05d1"
FEATURES = (
    "asset.mjcf",
    "asset.urdf",
    "entity.single_articulation",
    "entity.multiple",
    "entity.gravity_disable",
    "entity.joint_velocity_limit",
    "root.free",
    "root.fixed",
    "joint.hinge",
    "joint.slide",
    "joint.ball",
    "actuator.motor",
    "actuator.position",
    "collision.rigid",
    "collision.self",
    "contact.query",
    "terrain.heightfield",
    "sensor.imu",
    "sensor.gyro",
    "reset.state",
    "dr.interval.body_force",
    "state.final_refresh",
    "state.callback_refresh",
    "variant.same_layout",
    "tensor.execution",
    "tensor.state_views",
    "tensor.state_fields",
    "tensor.sensor_views",
    "tensor.stepping",
    "tensor.selected_reset",
    "tensor.reset_randomization",
    "tensor.fixed_variants",
    "tensor.host_pre_step_control",
    "tensor.packed_host_bridge",
    "tensor.process_topology",
    "tensor.data_plane",
    "tensor.stream_event_ownership",
    "tensor.torch_devices",
)


@dataclass(frozen=True)
class TensorPlatformProfile:
    """SDK-free platform view of the reviewed default tensor lifecycle."""

    adapter: str
    execution: TensorExecution
    process_topology: TensorProcessTopology
    data_plane: TensorDataPlane
    torch_devices: tuple[str, ...]
    cuda_runtime: str
    linux_cuda: str
    macos_tensor_profile: str
    rocm_tensor_profile: str
    worker_requirement: str
    reset_randomization: SupportLevel
    fixed_variants: SupportLevel
    host_pre_step_control: SupportLevel
    packed_host_bridge: SupportLevel


_TENSOR_WORKER_REQUIREMENTS = {
    "mujoco": "In-process; no external Python worker",
    "motrix": "In-process; no external Python worker",
    "drake": "In-process; no external Python worker",
    "mjwarp": "In-process; no external Python worker",
    "newton": "In-process; no external Python worker",
    "superdex": "In-process; no external Python worker",
    "genesis": "In-process; no external Python worker",
    "isaacgym": "Dedicated external Python 3.8 worker; host Python paths are not inherited",
    "isaacsim": "Dedicated external Python 3.11 worker; host Python paths are not inherited",
}


def get_adapter_capabilities(name: str, profile: str = "default") -> CapabilityReport:
    """Return conservative declarations without loading or discovering an SDK.

    ``exact`` describes only the named semantic subset, never arbitrary importer
    parity. Unknown profiles have unknown declarations. Evidence is source-only;
    installed distribution metadata is not proof of native runtime availability.
    """
    adapter_spec(name)
    try:
        installed_version = version("unisim-core")
    except PackageNotFoundError:
        installed_version = None
    scope = CapabilityScope(
        adapter=name,
        profile=profile,
        unisim_version=installed_version,
        adapter_version=SOURCE_REVISION,
    )
    source_path = f"src/unisim/backend/{name}/backend.py"
    if name in {"isaacgym", "isaacsim"}:
        source_path = f"src/unisim/backend/{name}/worker.py"
    evidence = CapabilityEvidence(
        kind="source",
        source=f"https://github.com/unilabsim/unisim/blob/{SOURCE_REVISION}/{source_path}",
        scope=scope,
    )
    values: dict[str, tuple[SupportLevel, str, tuple[CapabilityCondition, ...]]] = {}

    def declare(
        feature: str,
        level: SupportLevel,
        reason: str,
        conditions: tuple[CapabilityCondition, ...] = (),
    ) -> None:
        values[feature] = (level, reason, conditions)

    exact, approximate, unsupported = (
        SupportLevel.EXACT,
        SupportLevel.APPROXIMATE,
        SupportLevel.UNSUPPORTED,
    )
    tensor_features = tuple(feature for feature in FEATURES if feature.startswith("tensor."))
    known_profile = profile == "default"
    if not known_profile:
        pass
    elif name == "mjwarp":
        declare("tensor.execution", exact, "DEVICE_RESIDENT, in-process direct storage")
        declare("tensor.state_views", exact, "MJWarp public tensor state views")
        declare("tensor.state_fields", exact, "qpos, qvel, ctrl, sensordata, and time")
        declare("tensor.sensor_views", exact, "MJWarp public tensor sensor views")
        declare("tensor.stepping", exact, "MJWarp tensor stepping")
        declare("tensor.selected_reset", exact, "MJWarp selected-row tensor reset")
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Minimal tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Minimal tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", unsupported, "MJWarp is not a host bridge")
        declare("tensor.process_topology", exact, "in_process")
        declare("tensor.data_plane", exact, "direct")
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Backend completes step and refresh; caller owns the Torch stream",
        )
        declare("tensor.torch_devices", exact, "CUDA")
    elif name == "mujoco":
        declare(
            "tensor.execution",
            exact,
            "HOST_BRIDGE, in-process packed accelerator/host boundaries",
        )
        declare("tensor.state_views", exact, "MuJoCo public tensor state views")
        declare("tensor.state_fields", exact, "qpos, qvel, and ctrl")
        declare("tensor.sensor_views", exact, "MuJoCo public tensor sensor views")
        declare("tensor.stepping", exact, "MuJoCo tensor stepping")
        declare("tensor.selected_reset", exact, "MuJoCo selected-row tensor reset")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", exact, "Persistent packed host-bridge plan")
        declare("tensor.process_topology", exact, "in_process")
        declare("tensor.data_plane", exact, "host_bridge")
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Caller Torch stream with per-packed-boundary synchronization",
        )
        declare("tensor.torch_devices", exact, "CPU and CUDA")
    elif name == "superdex":
        declare(
            "tensor.execution",
            exact,
            "HOST_BRIDGE, in-process packed accelerator/host boundaries",
        )
        declare("tensor.state_views", exact, "SuperDex public tensor state views")
        declare("tensor.state_fields", exact, "qpos, qvel, and ctrl")
        declare("tensor.sensor_views", exact, "SuperDex public named/tracked tensor sensor views")
        declare("tensor.stepping", exact, "SuperDex tensor stepping")
        declare("tensor.selected_reset", exact, "SuperDex selected-row tensor reset")
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Packed tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Packed tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", exact, "Persistent packed host-bridge plan")
        declare("tensor.process_topology", exact, "in_process")
        declare("tensor.data_plane", exact, "host_bridge")
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Caller Torch stream with per-packed-boundary synchronization",
        )
        declare("tensor.torch_devices", exact, "CPU and CUDA")
    elif name == "motrix":
        declare(
            "tensor.execution",
            exact,
            "HOST_BRIDGE, in-process packed accelerator/host boundaries",
        )
        declare("tensor.state_views", exact, "MotrixSim public tensor state views")
        declare("tensor.state_fields", exact, "qpos, qvel, and ctrl")
        declare("tensor.sensor_views", exact, "MotrixSim public named/tracked tensor sensor views")
        declare("tensor.stepping", exact, "MotrixSim tensor stepping")
        declare("tensor.selected_reset", exact, "MotrixSim selected-row tensor reset")
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Packed tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Packed tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", exact, "Persistent packed host-bridge plan")
        declare("tensor.process_topology", exact, "in_process")
        declare("tensor.data_plane", exact, "host_bridge")
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Caller Torch stream with per-packed-boundary synchronization",
        )
        declare("tensor.torch_devices", exact, "CPU and CUDA")
    elif name == "drake":
        declare(
            "tensor.execution",
            exact,
            "HOST_BRIDGE, in-process packed accelerator/host boundaries",
        )
        declare("tensor.state_views", exact, "Drake public tensor state views")
        declare("tensor.state_fields", exact, "qpos, qvel, and ctrl")
        declare("tensor.sensor_views", exact, "Drake public named/tracked tensor sensor views")
        declare("tensor.stepping", exact, "Drake tensor stepping")
        declare("tensor.selected_reset", exact, "Drake selected-row tensor reset")
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Packed tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Packed tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", exact, "Persistent packed host-bridge plan")
        declare("tensor.process_topology", exact, "in_process")
        declare("tensor.data_plane", exact, "host_bridge")
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Caller Torch stream with per-packed-boundary synchronization",
        )
        declare("tensor.torch_devices", exact, "CPU and CUDA")
    elif name == "newton":
        newton_tensor: tuple[CapabilityCondition, ...] = ()
        declare(
            "tensor.execution",
            exact,
            "DEVICE_RESIDENT, in-process direct storage for a single articulation",
            newton_tensor,
        )
        declare(
            "tensor.state_views",
            exact,
            "Newton public qpos/qvel tensor state views",
            newton_tensor,
        )
        declare(
            "tensor.state_fields",
            exact,
            "qpos and qvel",
            newton_tensor,
        )
        declare(
            "tensor.sensor_views",
            exact,
            "Negotiated Newton scalar and tracked-body tensor sensor views",
            newton_tensor,
        )
        declare(
            "tensor.stepping",
            exact,
            "Newton tensor stepping with eager or reviewed CUDA-graph execution",
            newton_tensor,
        )
        declare(
            "tensor.selected_reset",
            exact,
            "Non-portable single-articulation selected-row tensor reset",
            newton_tensor,
        )
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Newton tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Newton tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", unsupported, "Newton is not a host bridge")
        declare(
            "tensor.process_topology",
            exact,
            "in_process",
            newton_tensor,
        )
        declare(
            "tensor.data_plane",
            exact,
            "direct",
            newton_tensor,
        )
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Newton synchronizes its device stream before return; caller owns subsequent "
            "Torch ordering",
            newton_tensor,
        )
        declare(
            "tensor.torch_devices",
            exact,
            "CUDA",
            newton_tensor,
        )
    elif name == "genesis":
        genesis_tensor: tuple[CapabilityCondition, ...] = ()
        declare(
            "tensor.execution",
            exact,
            "DEVICE_RESIDENT, in-process direct storage for a single articulation",
            genesis_tensor,
        )
        declare(
            "tensor.state_views",
            exact,
            "Genesis public qpos/qvel tensor state views",
            genesis_tensor,
        )
        declare(
            "tensor.state_fields",
            exact,
            "qpos and qvel",
            genesis_tensor,
        )
        declare(
            "tensor.sensor_views",
            exact,
            "Negotiated Genesis scalar and tracked-body tensor sensor views",
            genesis_tensor,
        )
        declare(
            "tensor.stepping",
            exact,
            "Genesis tensor stepping on the exact CUDA backend with zero-copy enabled",
            genesis_tensor,
        )
        declare(
            "tensor.selected_reset",
            exact,
            "Non-portable single-articulation selected-row tensor reset",
            genesis_tensor,
        )
        declare(
            "tensor.reset_randomization",
            unsupported,
            "Genesis tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Genesis tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", unsupported, "Genesis is not a host bridge")
        declare(
            "tensor.process_topology",
            exact,
            "in_process",
            genesis_tensor,
        )
        declare(
            "tensor.data_plane",
            exact,
            "direct",
            genesis_tensor,
        )
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Genesis completes control stepping and state publication; caller owns "
            "subsequent Torch ordering",
            genesis_tensor,
        )
        declare(
            "tensor.torch_devices",
            exact,
            "CUDA",
            genesis_tensor,
        )
    elif name == "isaacsim":
        isaacsim_tensor: tuple[CapabilityCondition, ...] = ()
        declare(
            "tensor.execution",
            exact,
            "DEVICE_RESIDENT, external-worker CUDA IPC for opt-in mapped entity scenes",
            isaacsim_tensor,
        )
        declare(
            "tensor.state_views",
            exact,
            "IsaacSim public qpos/qvel tensor state views",
            isaacsim_tensor,
        )
        declare(
            "tensor.state_fields",
            exact,
            "qpos and qvel",
            isaacsim_tensor,
        )
        declare(
            "tensor.sensor_views",
            exact,
            "Negotiated IsaacSim scalar and tracked-body tensor sensor views",
            isaacsim_tensor,
        )
        declare(
            "tensor.stepping",
            exact,
            "IsaacSim tensor stepping through metadata-only worker commands and CUDA IPC",
            isaacsim_tensor,
        )
        declare(
            "tensor.selected_reset",
            exact,
            "Mapped-scene selected-row tensor reset",
            isaacsim_tensor,
        )
        declare(
            "tensor.reset_randomization",
            unsupported,
            "IsaacSim CUDA IPC tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "Mapped tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", unsupported, "IsaacSim is not a host bridge")
        declare(
            "tensor.process_topology",
            exact,
            "external_worker",
            isaacsim_tensor,
        )
        declare(
            "tensor.data_plane",
            exact,
            "cuda_ipc",
            isaacsim_tensor,
        )
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Host records control/reset readiness; worker records state readiness; host "
            "consumer waits on the state event",
            isaacsim_tensor,
        )
        declare(
            "tensor.torch_devices",
            exact,
            "CUDA",
            isaacsim_tensor,
        )
    elif name == "isaacgym":
        # The static inventory describes only the reviewed GPU-pipeline profile;
        # runtime construction and `get_tensor_capabilities()` remain fail-closed.
        isaacgym_tensor: tuple[CapabilityCondition, ...] = ()
        declare(
            "tensor.execution",
            exact,
            "DEVICE_RESIDENT, external-worker CUDA IPC for the reviewed GPU-pipeline profile",
            isaacgym_tensor,
        )
        declare(
            "tensor.state_views",
            exact,
            "IsaacGym public qpos/qvel tensor state views",
            isaacgym_tensor,
        )
        declare("tensor.state_fields", exact, "qpos and qvel", isaacgym_tensor)
        declare(
            "tensor.sensor_views",
            exact,
            "Negotiated IsaacGym scalar and tracked-body views after a tensor step",
            isaacgym_tensor,
        )
        declare(
            "tensor.stepping",
            exact,
            "Callback-free IsaacGym tensor stepping through metadata-only worker commands",
            isaacgym_tensor,
        )
        declare(
            "tensor.selected_reset",
            exact,
            "Selected-row tensor reset that re-submits the authoritative IsaacGym actor union",
            isaacgym_tensor,
        )
        declare(
            "tensor.reset_randomization",
            unsupported,
            "IsaacGym CUDA IPC tensor reset has no randomization",
        )
        declare("tensor.fixed_variants", unsupported, "CUDA IPC tensor reset has no fixed variants")
        declare("tensor.host_pre_step_control", unsupported, "Host callbacks remain NumPy-only")
        declare("tensor.packed_host_bridge", unsupported, "IsaacGym is not a host bridge")
        declare(
            "tensor.process_topology",
            exact,
            "external_worker",
            isaacgym_tensor,
        )
        declare("tensor.data_plane", exact, "cuda_ipc", isaacgym_tensor)
        declare(
            "tensor.stream_event_ownership",
            exact,
            "Host records control/reset readiness; worker records state readiness; host "
            "consumer waits on the state event",
            isaacgym_tensor,
        )
        declare("tensor.torch_devices", exact, "CUDA", isaacgym_tensor)
    else:
        for feature in tensor_features:
            declare(
                feature,
                unsupported,
                "Public SimBackend tensor lifecycle is not declared; fail closed.",
            )
    if known_profile:
        declare("asset.mjcf", exact, "MJCF entry point exists; importer-specific subsets apply.")
        declare("entity.single_articulation", exact, "One primary articulation is supported.")
        declare("reset.state", exact, "Generalized-state reset exists; layout is adapter-owned.")
        for feature in (
            "root.free",
            "root.fixed",
            "joint.hinge",
            "joint.slide",
            "actuator.motor",
            "collision.rigid",
        ):
            declare(feature, exact, "Implemented for the adapter's accepted MJCF subset.")
        if name in {"mujoco", "mjwarp"}:
            if name in {"mujoco", "mjwarp"}:
                declare(
                    "entity.multiple",
                    exact,
                    "MJCF entity composition with one same-layout fixed variant consumer; "
                    "unsupported source/compiler/global-option combinations fail closed.",
                    (CapabilityCondition("entity.asset_format", "mjcf"),),
                )
            declare("actuator.position", exact, "Native compiled position actuators are retained.")
            declare("joint.ball", exact, "Native MuJoCo joint layout is preserved.")
            declare(
                "collision.self",
                exact,
                "Compiled geom masks and exclusions are retained exactly as authored; "
                "the per-entity self_collision toggle cannot be applied and fails closed.",
                (CapabilityCondition("entity.self_collision", "authored"),),
            )
            declare(
                "entity.gravity_disable",
                unsupported,
                "MJCF compilation applies scene gravity to every body; per-entity "
                "gravity_disabled requests fail closed at composition negotiation.",
                (CapabilityCondition("entity.gravity_disabled", "explicit"),),
            )
            declare(
                "entity.joint_velocity_limit",
                unsupported,
                "MJCF has no joint-velocity-limit concept and the compiled model "
                "applies none; per-entity joint_velocity_limits requests fail "
                "closed at composition negotiation.",
                (CapabilityCondition("entity.joint_velocity_limits", "explicit"),),
            )
            declare(
                "contact.query",
                exact,
                "Named MJCF contact sensordata only; no general contact-pair query API.",
            )
            declare("sensor.imu", exact, "Named native gyro and accelerometer sensors are exposed.")
            declare("terrain.heightfield", exact, "MJCF heightfield collision is supported.")
            declare(
                "state.final_refresh",
                exact,
                "Tracked body state is final-state fresh; force/contact stays substep-solved.",
                (CapabilityCondition("add_body_sensors", "true"),) if name == "mujoco" else (),
            )
            declare(
                "state.callback_refresh",
                exact,
                "Tracked body state is refreshed at the beginning of each callback substep.",
                (
                    CapabilityCondition("refresh_pre_step_body_state", "true"),
                    CapabilityCondition("add_body_sensors", "true"),
                )
                if name == "mujoco"
                else (),
            )
        elif name == "drake":
            declare(
                "entity.multiple",
                exact,
                "No-variant scenes use one common expanded portable MJCF model; assigned "
                "same-layout fixed variants use one DrakeUni runtime per used variant with "
                "explicit public-row scatter/gather and native property identity audit. "
                "Fixed/floating physical entities and passive joints are supported, while "
                "kinematic mirrors fail closed. Native support is bounded to this "
                "reviewed profile.",
                (
                    CapabilityCondition("entity.asset_format", "mjcf"),
                    CapabilityCondition("entity.kinematic", "none"),
                ),
            )
            declare(
                "actuator.motor",
                SupportLevel.UNKNOWN,
                "External importer motor mapping has not been reviewed for this profile.",
            )
        elif name == "motrix":
            declare(
                "entity.multiple",
                exact,
                "Common expanded portable MJCF models support fixed/floating "
                "physical entities, passive scalar joints, immutable "
                "same-layout fixed variants and collision-disabled kinematic "
                "mirrors. Native link/root/joint/actuator "
                "names, state addresses and variant identity are audited "
                "against the frozen public layout; selected entity resets "
                "preserve unrelated state and controls, while selected control "
                "restoration and full default reset use cold-captured native "
                "construction/default-keyframe controls and selected keyframe "
                "qpos/qvel. World-frame body-force and portable body-torque "
                "submissions map through audited public body IDs, accumulate "
                "for the upcoming native step, and reset cancellation is "
                "scoped to impacted entity bodies. Mirror roots bind public "
                "native mocap objects, selected-row pose writes use public "
                "Mocap.set_pose, imported mirror collision masks are audited "
                "as disabled, and fixed variants retain mirror identity. "
                "Generated body-frame position/quaternion tracking sensors are "
                "materialized for every public body and gathered by immutable "
                "variant assignment, and world-referenced authored body "
                "FramePos/FrameQuat sensors are cold-audited against native "
                "identity. Scene-level fragment world-referenced qualified-body "
                "FramePos/FrameQuat/FrameLinVel/FrameAngVel sensors audit native "
                "type/body/world-reference identity and dimensions; motion rows "
                "gather by assignment, with FrameLinVel reporting world velocity "
                "at the inertial body-frame origin and FrameAngVel reporting "
                "world angular velocity. Scene-level fragment "
                "world-referenced qualified-site FrameLinVel/FrameAngVel sensors "
                "audit native type/site/world-reference identity, dimensions and "
                "the complete parent/local-pose site identity; their rows gather "
                "by assignment, with site FrameLinVel reporting world-frame "
                "site-point velocity and site FrameAngVel reporting world angular "
                "velocity. Scene-level geom-pair "
                "netforce and found contact fragments are likewise audited "
                "against native geom-pair/reduction/report identity and read "
                "from native sensor storage. Qualified named-site world Jacobians "
                "along with entity-owned and scene-level fragment "
                "world-referenced site pose sensors are gathered from native "
                "variant contexts by assignment. Entity-owned site "
                "velocimeter/gyro sensors require native local-frame motion "
                "identity and gather through the same audited variant contexts. "
                "Portable selected-row reset randomization supports "
                "body_mass/base_mass_delta, body_ipos/base_com_offset and "
                "scalar-joint dof_armature/dof_frictionloss by prevalidating "
                "public columns and applying public Link mass/COM and Joint "
                "armature/friction-loss overrides through owning variant data "
                "slices; free-root DOF columns remain defaults and reject "
                "mutation. Motrix 0.8.2 has no public runtime joint-damping "
                "override, so dof_damping fails closed. "
                "Non-uniform public control parameters, absent native wrench "
                "APIs, actuator activation state, physical kinematic entities, entity-owned "
                "frame motion, other source sensors, other site-sensor forms, "
                "terrain and other reset randomization fail closed.",
                (
                    CapabilityCondition("entity.asset_format", "mjcf"),
                    CapabilityCondition("entity.kinematic", "none"),
                ),
            )
            declare(
                "actuator.motor",
                SupportLevel.UNKNOWN,
                "Motor mapping has not been reviewed for this profile.",
            )
            declare(
                "actuator.position", exact, "Native position actuators expose joint target gains."
            )
            declare(
                "terrain.heightfield",
                exact,
                "Scene materializer attaches the configured heightfield on the cold path.",
            )
        elif name == "newton":
            declare(
                "entity.multiple",
                exact,
                "Portable MJCF entity scenes with independent Newton articulation "
                "views, assigned same-layout variants and selected state reset; "
                "selected controls clear while unrelated controls persist. Named "
                "contact found sensors attribute only to their own world. Restore-default "
                "controls, mixed shape-type variants and unsupported profiles fail closed.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
            declare(
                "root.fixed",
                exact,
                "Portable MJCF fixed-root articulations and static rigid entities are "
                "retained as fixed Newton articulations.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
            declare(
                "contact.query",
                exact,
                "Only named geom-pair contact found sensors with num=1 are mapped.",
                (CapabilityCondition("contact.kind", "geom_pair_found"),),
            )
            declare(
                "sensor.imu",
                approximate,
                "Site sensor signals are reconstructed from public Newton state arrays.",
            )
        elif name == "genesis":
            declare("actuator.motor", unsupported, "Materializer requires position actuators.")
            declare("actuator.position", exact, "Joint position targets use native PD control.")
            declare(
                "root.fixed",
                exact,
                "Portable MJCF fixed-root articulations and static rigid entities are "
                "retained as independent Genesis entities.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
            declare(
                "entity.multiple",
                exact,
                "Portable MJCF entity scenes use independent Genesis entities, public-layout "
                "name binding, selected state reset, and heterogeneous single-link rigid "
                "variants only when the assignment exactly equals Genesis' balanced mapping. "
                "Collision-disabled mirror declarations following a physical source "
                "materialize as public Genesis Kinematic entities: topology and fixed-variant "
                "visual identity remain audited, mirrors expose zero qpos/DoFs and collision "
                "masks, selected world-root pose writes use public set_pos/set_quat, and "
                "full reset restores the independent mirror default. "
                "Construction supports selected scalar hinge/slide default-keyframe qpos/qvel "
                "and actuator controls through public per-entity APIs; absent keys retain "
                "normalized scalar qpos and zero qvel/controls, raw keyframe root pose/"
                "velocity are ignored, and declared portable root placement with zero root "
                "velocity is retained. Selected entity resets clear selected controls when "
                "restore_default_controls is false or restore assignment-aware selected "
                "rows from the default control table when true, while unrelated rows and "
                "entities persist; no persistent public control-target getter is claimed. "
                "Portable geometry exposes audited names, IDs, body ownership, uniform "
                "Genesis-native sphere/box sizes, contact masks, friction coefficients, "
                "and solver parameters for complete uniform collision identity. "
                "Portable DOF damping, friction loss and armature expose "
                "cold-captured native values through audited public qvel addresses "
                "and require uniform active-row/fixed-variant values. "
                "Entity-owned unreferenced site FramePos/FrameQuat/Gyro/"
                "Velocimeter/Accelerometer sensors and scene-level fragment "
                "world-referenced qualified-site FramePos/FrameQuat/FrameLinVel/"
                "FrameAngVel sensors are computed from audited public link/site "
                "identity and require identical complete sensor identity across "
                "fixed variants; site FrameLinVel is world-frame site-point "
                "velocity, site FrameAngVel is world angular velocity, site "
                "quaternions remain public wxyz, while accelerometers use clean "
                "public native IMUs and require identity site orientation. "
                "Scene-level fragment world-referenced qualified-body "
                "FramePos/FrameQuat/FrameLinVel/FrameAngVel sensors compose "
                "audited public native link-origin pose/velocity and assignment-"
                "selected source inertial identity; body FrameLinVel adds the "
                "world-angular cross product with the source body_ipos offset, "
                "while body FrameAngVel returns public native world angular "
                "velocity. "
                "Scene-level cross-entity geom-pair found and netforce fragments "
                "bind exact native collision identities by name, owner and active "
                "rows, gather Genesis' public contact geom IDs/valid mask by "
                "assignment, and expose completed-step flags or three-vector "
                "forces on authored geom1; netforce values sum force_a/force_b "
                "over exact valid slots, and selected reset rows stay cleared "
                "until the next step. "
                "Portable selected-row reset randomization supports body_mass, "
                "base_mass_delta, body_ipos, base_com_offset, DOF damping/friction "
                "loss/armature and actuator kp/kd "
                "by prevalidating public columns and submitting them through "
                "audited owning entities. "
                "Portable world-frame body-force and body-torque submissions map "
                "audited public owned-body IDs to Genesis solver links through "
                "the public solver API at each link COM; repeated submissions "
                "and ops within one interval plan accumulate independently for "
                "the upcoming native step, a later interval plan replaces prior "
                "pending staging, selected state/entity resets cancel matching "
                "rows while unrelated pending wrenches persist, and callback-time "
                "staging fails closed. "
                "Non-uniform variant sizes/masks/friction/solver parameters, physical "
                "kinematic entities, mirror mass/inertia/DR mutation, mirror contact "
                "fragments, source contact sensors, same-entity pairs, other contact forms, "
                "source body sensors, inertial orientation mismatches, other body "
                "fragment forms, referenced forms, other site fragment forms, "
                "other reset randomization, activation state, arbitrary keyframe "
                "semantics and arbitrary force application points fail closed.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
            declare(
                "contact.query",
                approximate,
                "World/robot contact found is approximated by robot-link net force threshold.",
                (CapabilityCondition("contact.kind", "world_link_found"),),
            )
            declare(
                "sensor.imu",
                approximate,
                "Site signals use native IMU/rigid state; rotated accelerometers are rejected.",
            )
        elif name == "superdex":
            declare(
                "entity.multiple",
                exact,
                "Portable MJCF entity scenes map each physical entity to one audited "
                "native actor slot. Scenes without physical kinematic roots use "
                "SceneBatchExecutorV2; physical roots use SceneBatchExecutorV3 ABI 3 "
                "selective boundary-condition writes. The reviewed profile supports "
                "fixed/floating physical entities, scalar joints, immutable same-layout "
                "fixed variants with assignment-selected native realizations, and "
                "one-body collision-disabled mirrors. Physical kinematic roots retain "
                "source-declared collision on a hidden six-DoF free-root carrier with no "
                "public state/control, gravity, or body-wrench ownership. Mirrors likewise "
                "use a hidden carrier with no collision or physical ownership. Both support "
                "row-local world-pose writes and independent full-reset defaults, and "
                "mirror contact sensors, physical-root contact sensors, and world-body "
                "portable contact sensors remain fail-closed; selected entity/reset-"
                "impacted control semantics are "
                "preserved.",
                (
                    CapabilityCondition("entity.asset_format", "mjcf"),
                    CapabilityCondition("entity.kinematic", "none_or_physical"),
                ),
            )
            declare("joint.ball", unsupported, "Only free, hinge and slide MJCF joints are mapped.")
            declare("terrain.heightfield", unsupported, "MJCF nhfield is rejected by materializer.")
            declare(
                "sensor.imu",
                approximate,
                "Named site gyro/accelerometer signals are reconstructed from native state.",
            )
            declare(
                "contact.query",
                approximate,
                "Named plane/link found sensors use geometric distance, not solver contacts.",
                (CapabilityCondition("contact.kind", "plane_link_found"),),
            )
            declare(
                "collision.rigid",
                approximate,
                "Sliding-only Coulomb contact omits authored torsional/rolling friction.",
                (CapabilityCondition("superdex_allow_contact_approximation", "true"),),
            )
        elif name in {"isaacgym", "isaacsim"}:
            multiple_reason = (
                "Mapped MJCF scalar-joint entity scenes; worker audits native layout, "
                "inertials and identity. Variants use immutable construction-time "
                "assignments; IsaacSim materializes each unique assignment as a K-prototype "
                "catalog. Unsupported source/root/geometry profiles fail closed."
            )
            if name == "isaacsim":
                multiple_reason += (
                    " Per-environment reset domain randomization covers geometry friction, "
                    "body mass/COM/inertia, drive kp/kd and joint damping/armature/friction; "
                    "per-variant drive gains are written at spawn and audited per environment."
                )
            declare(
                "entity.multiple",
                exact,
                multiple_reason,
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
            if name == "isaacgym":
                declare(
                    "root.fixed",
                    exact,
                    "Mapped and legacy workers import fixed-base MJCF roots with "
                    "fix_base_link and audit native layout, inertials and identity; "
                    "fixed-root reset writes fail closed by contract.",
                    (CapabilityCondition("entity.asset_format", "mjcf"),),
                )
            elif name == "isaacsim":
                declare(
                    "root.fixed",
                    exact,
                    "Fixed roots use the root-prim articulation convention with fixed-anchor "
                    "world-pose rebinding; the worker audits is_fixed_base and rejects "
                    "fixed-root state/velocity writes.",
                    (CapabilityCondition("scene.profile", "mapped_entities"),),
                )
            else:
                declare(
                    "root.fixed",
                    SupportLevel.UNKNOWN,
                    "Current worker/host public root layout is only established for free roots.",
                )
            declare("actuator.motor", unsupported, "Worker control accepts position targets only.")
            declare(
                "actuator.position", exact, "MJCF position actuators map to worker joint drives."
            )
            declare("asset.urdf", unsupported, "The current worker path imports MJCF only.")
            if name == "isaacsim":
                declare(
                    "collision.self",
                    exact,
                    "Mapped entity scenes apply each collision-enabled articulation's "
                    "self_collision request through the MJCF converter, re-author the "
                    "imported PhysX articulation flag when fixed roots move the "
                    "articulation API to the root prim, audit the flag on every spawned "
                    "instance and report it per entity. Source <contact><exclude> pairs "
                    "stay excluded as USD filtered pairs; rigid entities and mirrors "
                    "cannot request self-collision. Legacy model-file scenes keep "
                    "self-collision disabled.",
                    (
                        CapabilityCondition("entity.self_collision", "true"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            elif name == "isaacgym":
                declare(
                    "collision.self",
                    exact,
                    "Mapped entity scenes honor each collision-enabled articulation's "
                    "self_collision request through PhysX filter authoring: the actor "
                    "receives a zero collision filter and every collision-shaped body "
                    "a distinct per-shape filter bit (verified by native shape-property "
                    "readback), so intra-actor body pairs collide while other physical "
                    "entities keep the per-entity bit scheme and collision-disabled "
                    "entities carry every allocated bit and stay excluded. Authored "
                    "<contact><exclude> pairs on a self-collision entity cannot be "
                    "expressed without per-pair bit coloring and fail closed, as do "
                    "scenes whose entity plus body bits exceed the 30-bit filter "
                    "budget; rigid entities and mirrors cannot request self-collision. "
                    "Legacy model-file scenes keep self-collision disabled.",
                    (
                        CapabilityCondition("entity.self_collision", "true"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            else:
                declare("collision.self", unsupported, "Worker explicitly disables self-collision.")
            if name == "isaacsim":
                declare(
                    "entity.gravity_disable",
                    exact,
                    "Mapped entity scenes honor each entity's explicit gravity_disabled "
                    "request exactly: role baking authors physxRigidBody:disableGravity "
                    "from the resolved value, every baked or cached role re-validates "
                    "it, and the worker reports the resolved flag per entity for strict "
                    "host comparison. gravity_disabled=None keeps the implicit default "
                    "(kinematic entities and fixed rigid bodies feel no gravity). "
                    "Legacy model-file scenes keep their authored gravity behavior.",
                    (
                        CapabilityCondition("entity.gravity_disabled", "explicit"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            else:
                declare(
                    "entity.gravity_disable",
                    exact,
                    "Mapped entity scenes honor each entity's explicit gravity_disabled "
                    "request exactly through per-entity AssetOptions.disable_gravity; "
                    "IsaacGym exposes no per-actor gravity readback, so honoring is "
                    "enforced at asset authoring and the resolved flag is reported per "
                    "entity for strict host comparison. gravity_disabled=None keeps "
                    "gravity enabled on every entity asset. Legacy model-file scenes "
                    "keep their authored gravity behavior.",
                    (
                        CapabilityCondition("entity.gravity_disabled", "explicit"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            if name == "isaacsim":
                declare(
                    "entity.joint_velocity_limit",
                    exact,
                    "Mapped entity scenes honor each articulation's explicit "
                    "joint_velocity_limits table exactly: the host resolves the "
                    "declaration against the compiled variant joint names "
                    "fail-closed, the worker authors it through IsaacLab's "
                    "ImplicitActuatorCfg velocity_limit_sim (PhysX brakes joints "
                    "at the cap), verifies the PhysX view readback at INIT, and "
                    "reports the per-entity table for strict host comparison. "
                    "joint_velocity_limits=None keeps the uncapped default. "
                    "Legacy model-file scenes keep their uncapped behavior.",
                    (
                        CapabilityCondition("entity.joint_velocity_limits", "explicit"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            else:
                declare(
                    "entity.joint_velocity_limit",
                    unsupported,
                    "IsaacGym mapped scenes do not author per-joint velocity "
                    "limits; explicit joint_velocity_limits requests fail closed "
                    "at composition negotiation.",
                    (
                        CapabilityCondition("entity.joint_velocity_limits", "explicit"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
                )
            declare(
                "sensor.imu",
                unsupported,
                "Accelerometers are rejected; use the narrower gyro declaration.",
            )
            declare(
                "sensor.gyro",
                approximate,
                "Host maps worker rigid-body angular velocity into the local sensor frame.",
            )
            if name == "isaacsim":
                declare(
                    "contact.query",
                    approximate,
                    "Mapped explicit geom-pair declarations are approximated as "
                    "ordered rigid-body pair reporters: IsaacSim returns the "
                    "world-frame net force on geom1's source rigid body, "
                    "aggregated across collision shapes and patches between that "
                    "body and geom2's target body. Duplicate declarations that "
                    "collapse to one ordered body pair fail closed. Mapped scenes "
                    "also report per-body net normal force (geom2 omitted, any "
                    "contact object) and body-net found flags through one batched "
                    "per-entity PhysX contact view. Legacy scenes reject all "
                    "contact declarations.",
                    (CapabilityCondition("scene.profile", "mapped_entities"),),
                )
                declare(
                    "state.callback_refresh",
                    exact,
                    "Mapped host control splits one public step into worker substeps and "
                    "refreshes shared state before every owner callback.",
                    (CapabilityCondition("scene.profile", "mapped_entities"),),
                )
            else:
                declare(
                    "contact.query",
                    approximate,
                    "Body-net found and wildcard (geom2-omitted) netforce queries "
                    "read the per-body net contact force tensor rather than geom "
                    "pairs; geom-pair force queries fail closed.",
                    (CapabilityCondition("contact.kind", "body_net_force"),),
                )
    declarations = []
    for feature in FEATURES:
        entry = values.get(feature)
        if entry is None:
            reason = "No reviewed declaration for this feature/profile; fail closed."
            if feature in {"dr.interval.body_force", "variant.same_layout"}:
                reason = "Query the instance's authoritative DR/fixed-variant capabilities."
            declarations.append(
                CapabilityDeclaration(
                    feature=feature,
                    support=SupportLevel.UNKNOWN,
                    reason=reason,
                )
            )
        else:
            level, reason, conditions = entry
            feature_evidence = evidence
            if (
                name in {"mujoco", "mjwarp", "isaacgym", "isaacsim", "drake", "motrix"}
                and feature == "entity.multiple"
            ):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source=(
                        "https://github.com/unilabsim/unisim/issues/122"
                        if name == "drake"
                        else "https://github.com/unilabsim/unisim/issues/121"
                        if name == "motrix"
                        else "https://github.com/unilabsim/unisim/issues/108"
                    ),
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version=(
                            "drake-portable-entities-v2"
                            if name == "drake"
                            else "motrix-portable-entities-v1"
                            if name == "motrix"
                            else "m2-entity-composition"
                        ),
                    ),
                )
            elif name == "newton" and feature in {"entity.multiple", "root.fixed"}:
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/unilabsim/unisim/issues/123",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="newton-portable-entities-v1",
                    ),
                )
            elif name == "genesis" and feature == "entity.multiple":
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/unilabsim/unisim/issues/120",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="genesis-portable-entities-v1",
                    ),
                )
            elif name == "motrix" and feature.startswith("tensor."):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/Motphys/UniLab/issues/1680",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="m9-packed-host-bridge-v1",
                    ),
                )
            elif name == "drake" and feature.startswith("tensor."):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/Motphys/UniLab/issues/1679",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="m9-packed-host-bridge-v1",
                    ),
                )
            elif name == "newton" and feature.startswith("tensor."):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/Motphys/UniLab/issues/1677",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="m9-direct-device-v1",
                    ),
                )
            elif name == "genesis" and feature.startswith("tensor."):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/Motphys/UniLab/issues/1676",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="m9-direct-device-v1",
                    ),
                )
            elif name == "isaacsim" and feature.startswith("tensor."):
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/Motphys/UniLab/issues/1675",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="m9-external-cuda-ipc-v1",
                    ),
                )
            elif name == "superdex" and feature in {"entity.multiple", *tensor_features}:
                tensor_evidence = feature.startswith("tensor.")
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source=(
                        "https://github.com/Motphys/UniLab/issues/1678"
                        if tensor_evidence
                        else "https://github.com/unilabsim/unisim/issues/124"
                    ),
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version=(
                            "m9-packed-host-bridge-v1"
                            if tensor_evidence
                            else "superdex-portable-entities-v4"
                        ),
                    ),
                )
            declarations.append(
                CapabilityDeclaration(
                    feature=feature,
                    support=level,
                    reason=reason,
                    conditions=conditions,
                    evidence=(feature_evidence,),
                )
            )
    return CapabilityReport(scope=scope, declarations=tuple(declarations))


def _tensor_platform_profile(name: str) -> TensorPlatformProfile:
    report = get_adapter_capabilities(name)

    def reviewed_reason(feature: str) -> str:
        declaration = report.get(feature)
        if declaration.support is not SupportLevel.EXACT or declaration.conditions:
            raise ValueError(
                f"{name} {feature} must be an unconditional exact declaration "
                "in the reviewed default profile"
            )
        return declaration.reason

    execution_reason = reviewed_reason("tensor.execution").upper()
    if "DEVICE_RESIDENT" in execution_reason:
        execution = TensorExecution.DEVICE_RESIDENT
    elif "HOST_BRIDGE" in execution_reason:
        execution = TensorExecution.HOST_BRIDGE
    else:
        raise ValueError(f"Unsupported reviewed tensor execution for {name}: {execution_reason}")

    topology_reason = reviewed_reason("tensor.process_topology")
    topology = {
        "in_process": TensorProcessTopology.IN_PROCESS,
        "external_worker": TensorProcessTopology.EXTERNAL_WORKER,
    }.get(topology_reason)
    if topology is None:
        raise ValueError(f"Unsupported tensor process topology for {name}: {topology_reason}")

    data_plane_reason = reviewed_reason("tensor.data_plane")
    data_plane = {
        "direct": TensorDataPlane.DIRECT,
        "cuda_ipc": TensorDataPlane.CUDA_IPC,
        "host_bridge": TensorDataPlane.HOST_BRIDGE,
    }.get(data_plane_reason)
    if data_plane is None:
        raise ValueError(f"Unsupported tensor data plane for {name}: {data_plane_reason}")

    device_reason = reviewed_reason("tensor.torch_devices").upper()
    device_families = set(device_reason.replace(",", " ").split())
    torch_devices: tuple[str, ...]
    if {"CPU", "CUDA"} <= device_families:
        torch_devices = ("cpu", "cuda")
    elif "CUDA" in device_families:
        torch_devices = ("cuda",)
    else:
        raise ValueError(f"Unsupported reviewed Torch device family for {name}: {device_reason}")

    def support(feature: str) -> SupportLevel:
        return report.get(feature).support

    if execution is TensorExecution.DEVICE_RESIDENT:
        cuda_runtime = "Required for the entire tensor lifecycle"
        linux_cuda = "Supported: Linux CUDA only"
        other_platform = "Unsupported; no CPU, MPS, or ROCm fallback"
        macos_tensor_profile = other_platform
        rocm_tensor_profile = other_platform
    else:
        cuda_runtime = "Required only when the learner requests CUDA state/control buffers"
        linux_cuda = "Supported: CPU-authoritative physics with optional CUDA Torch buffers"
        macos_tensor_profile = "CPU-authoritative host bridge only; no CUDA physics claim"
        rocm_tensor_profile = "CPU-authoritative host bridge only; no ROCm CUDA-only fallback"

    return TensorPlatformProfile(
        adapter=name,
        execution=execution,
        process_topology=topology,
        data_plane=data_plane,
        torch_devices=torch_devices,
        cuda_runtime=cuda_runtime,
        linux_cuda=linux_cuda,
        macos_tensor_profile=macos_tensor_profile,
        rocm_tensor_profile=rocm_tensor_profile,
        worker_requirement=_TENSOR_WORKER_REQUIREMENTS[name],
        reset_randomization=support("tensor.reset_randomization"),
        fixed_variants=support("tensor.fixed_variants"),
        host_pre_step_control=support("tensor.host_pre_step_control"),
        packed_host_bridge=support("tensor.packed_host_bridge"),
    )


def get_tensor_platform_profiles() -> dict[str, TensorPlatformProfile]:
    """Return the reviewed default platform matrix without SDK discovery.

    The execution/process/data-plane/capability fields are derived from
    ``get_adapter_capabilities``. Platform and worker fields are explicit
    source-reviewed boundaries; they never imply that an optional SDK is
    installed or that every task owner is supported. An ``unknown`` lifecycle
    field remains fail-closed and must be resolved by a backend instance; it is
    never promoted to support.
    """

    return {spec.name: _tensor_platform_profile(spec.name) for spec in ADAPTER_SPECS}


__all__ = [
    "FEATURES",
    "SOURCE_REVISION",
    "TensorPlatformProfile",
    "get_adapter_capabilities",
    "get_tensor_platform_profiles",
]
