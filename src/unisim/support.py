"""SDK-free, source-reviewed semantic inventory for the declared adapters.

This module does not discover runtimes or confer runtime verification. DR, play,
body-wrench and fixed-variant support are resolved by their existing instance APIs.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from unisim.adapters import adapter_spec
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
)


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
    known_profile = profile == "default"
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
            declare("collision.self", exact, "Compiled geom masks and exclusions are retained.")
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
                "One common expanded portable MJCF model instance supports fixed/floating "
                "physical entities and passive joints; fixed variants and kinematic "
                "mirrors fail closed. Native support is bounded to this reviewed profile.",
                (
                    CapabilityCondition("entity.asset_format", "mjcf"),
                    CapabilityCondition("entity.variant", "none"),
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
                "physical entities, passive scalar joints and immutable "
                "same-layout fixed variants. Native link/root/joint/actuator "
                "names, state addresses and variant identity are audited "
                "against the frozen public layout; selected entity resets "
                "preserve unrelated state and controls, while selected control "
                "restoration and full default reset use cold-captured native "
                "construction/default-keyframe controls and selected keyframe "
                "qpos/qvel. World-frame body-force and portable body-torque "
                "submissions map through audited public body IDs, accumulate "
                "for the upcoming native step, and reset cancellation is "
                "scoped to impacted entity bodies. "
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
                "APIs, actuator activation state, kinematic mirrors, entity-owned "
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
                "Non-uniform variant sizes/masks/friction/solver parameters, mirrors, "
                "source contact sensors, same-entity pairs, other contact forms, "
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
                unsupported,
                "Portable entity scenes fail closed until the native SceneBatchExecutor "
                "publishes a versioned multi-actor/state contract with explicit actor "
                "offsets and failure semantics.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
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
            declare(
                "entity.multiple",
                exact,
                "Mapped MJCF scalar-joint entity scenes; worker audits native layout, "
                "inertials and identity. Variants use immutable construction-time "
                "assignments; IsaacSim materializes each unique assignment as a K-prototype "
                "catalog. Unsupported source/root/geometry profiles fail closed.",
                (CapabilityCondition("entity.asset_format", "mjcf"),),
            )
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
            declare("collision.self", unsupported, "Worker explicitly disables self-collision.")
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
                    "Mapped geom-pair net normal force comes from the IsaacLab PhysX "
                    "contact reporter; body-net found queries are rejected.",
                    (
                        CapabilityCondition("contact.kind", "geom_pair_netforce"),
                        CapabilityCondition("scene.profile", "mapped_entities"),
                    ),
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
                    "Contact found mapping uses per-body net force rather than geom pairs.",
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
                            "drake-portable-entities-v1"
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
            elif name == "superdex" and feature == "entity.multiple":
                feature_evidence = CapabilityEvidence(
                    kind="source",
                    source="https://github.com/unilabsim/unisim/issues/124",
                    scope=CapabilityScope(
                        adapter=name,
                        profile=profile,
                        unisim_version=installed_version,
                        adapter_version="superdex-single-actor-executor-v1",
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


__all__ = ["FEATURES", "SOURCE_REVISION", "get_adapter_capabilities"]
