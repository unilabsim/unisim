# Mapped entity host for Isaac workers

[English](m2-worker-host.md) | [中文](../zh/m2-worker-host.md)

The shared subprocess host consumes `SceneCfg.entity_assets` and `entity_variant` through the public entity/layout/reset contracts. Both IsaacGym and IsaacSim retain their dedicated interpreters and native execution. MuJoCo compiles source intent on the host cold path; it is not substituted for PhysX simulation.

## Source preparation and identity

The common MJCF composer validates sources, defaults, names and same-layout variants. Standalone worker assets receive explicit compiler-derived body inertials and joint limits. Actuators are removed from exported XML after their unit-gear position-drive intent has been validated and copied into a separate table; the native MJCF importers cannot safely consume MuJoCo's canonical general-actuator spelling. Source passive joint damping, activation state, unsupported transmissions and non-scalar joints fail closed in this profile.

Generated filenames use safe internal USD identifiers; they do not define public entity identity. Post-compilation edits are serialized from the current spec, avoiding stale last-compiled XML. The host retains the generated full scenes and standalone sources until worker shutdown. Worker-returned entity names, complete assignment and actual instance masses are checked against compiled intent. Worker audits additionally establish native topology, drive and inertia adoption. Echo alone is not an independent asset identity proof.

The host publishes separate nq/nv/nu and entity root layouts. Public root state uses link-origin position/linear velocity, wxyz quaternion and world angular velocity with clone offsets removed exactly once by the adapter. Full generalized qvel retains body-frame angular velocity. Passive joints contribute state but no implicit action columns.

## Reset and control

All entity patches are validated before materializing or writing native state. Prepared rows and explicit masks pass through one versioned reset command. Existing full qpos/qvel writes on the new entity entry point normalize into the same patch submission. Full `reset()` restores each selected environment's variant defaults and independent keyframe control; controls need not equal joint positions. Other environments and unselected entities retain their state and targets.

IsaacGym accumulates indexed root/DoF submissions until the next physics step, because a second indexed setter can otherwise overwrite an earlier reset. Native COM velocity is converted at both directions of the public link-origin boundary. Root and joint state is available immediately; articulation descendant body/sensor state after initialization or affected resets is explicitly unavailable until the next step. The host refuses those reads instead of presenting stale values as current.

Unrecoverable native commit failures set the worker fault marker and the host refuses further state or step use. Validation failures before native submission preserve the session. Shared-memory slot shapes/dtypes are checked before attachment, including zero-width action or state layouts.

## Inspection and playback

Versioned worker configuration reports are required and checked before being accepted. Entity assignment and body masses have scoped records distinguishing source intent and instance readback; source values never stand in for missing runtime fields. Complete selected-environment scene sources are returned for playback, retaining robot, object, table and mirror. Native rendering remains worker-owned; mapped-worker physics snapshot export is not yet exposed by this slice.

IsaacSim currently supports same-drive round-robin variant assignment and one-body rigid views, with explicit refusal of other combinations. Its fixed-root native root mode and environment view-row mapping are independently audited. The common host only enables the documented MJCF scalar-joint profiles, not URDF or all PhysX asset features.

## Validation and remaining work

`tests/contract/test_worker_scene_native.py` enables real factory-to-worker acceptance with `UNISIM_TEST_ISAACGYM_SCENE=1` or `UNISIM_TEST_ISAACSIM_SCENE=1`. Tests include non-round-robin Gym identity, the supported IsaacSim round-robin profile, nq/nv/nu, partial reset isolation and persistence, keyframe control distinct from qpos, complete-scene playback and a versioned import report. Worker-specific suites add independent native mass/COM/inertia, topology, mirror and passive-articulation checks.

The existing `model_file` entry point still uses its previous worker dispatch in this implementation slice; normalizing that entry point and removing the transitional split remain #109 work. Native renderer acceptance, final four-backend evidence and the downstream shared task remain #113 requirements. These limitations must not be mistaken for completion of #108.
