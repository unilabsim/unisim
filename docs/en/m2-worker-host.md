# Mapped entity host for Isaac workers

[English](m2-worker-host.md) | [中文](../zh/m2-worker-host.md)

The shared subprocess host consumes `SceneCfg.entity_assets` and `entity_variant` through the public entity/layout/reset contracts. Both IsaacGym and IsaacSim retain their dedicated interpreters and native execution. MuJoCo compiles source intent on the host cold path; it is not substituted for PhysX simulation.

Legacy compiled MuJoCo/MJWarp scenes use the same cold `CompiledModelIndex` audit internally. It records native body partitions, roots, joint qpos/qvel addresses, mocap addresses and actuator transmission/control columns without renaming anonymous objects or pretending tendon/site/root transmissions are scalar joint actuators. Old whole-model APIs retain their source semantics; the restricted entity layout is only exposed when its partition cross-check passes.

## Source preparation and identity

The common MJCF composer validates sources, defaults, names and same-layout variants. Standalone worker assets receive explicit compiler-derived body inertials and joint limits. Actuators are removed from exported XML after their unit-gear position-drive intent has been validated and copied into a separate table; the native MJCF importers cannot safely consume MuJoCo's canonical general-actuator spelling. Source passive joint damping/springs, activation state, unsupported transmissions and non-scalar joints fail closed in this profile. Compiled per-environment actuator control limits apply to step targets and initial/full-reset controls; unlimited controls are not clamped to a stored zero range.

Generated filenames use safe internal USD identifiers; they do not define public entity identity. Post-compilation edits are serialized from the current spec, avoiding stale last-compiled XML. The host retains the generated full scenes and standalone sources until worker shutdown. Worker-returned entity names, complete assignment and actual instance masses are checked against compiled intent. Worker audits additionally establish native topology, drive and inertia adoption. Echo alone is not an independent asset identity proof.

The host publishes separate nq/nv/nu and entity root layouts. Public root state uses link-origin position/linear velocity, wxyz quaternion and world angular velocity with clone offsets removed exactly once by the adapter. Full generalized qvel retains body-frame angular velocity. Passive joints contribute state but no implicit action columns.

## Reset and control

All entity patches are validated before materializing or writing native state. Prepared rows and explicit masks pass through one versioned reset command. Existing full qpos/qvel writes on the new entity entry point normalize into the same patch submission. Full `reset()` restores each selected environment's variant defaults and independent keyframe control; controls need not equal joint positions. Other environments and unselected entities retain their state and targets.

IsaacGym accumulates indexed root/DoF submissions until the next physics step, because a second indexed setter can otherwise overwrite an earlier reset. Native COM velocity is converted at both directions of the public link-origin boundary. Root and joint state is available immediately; articulation descendant body/sensor state after initialization or affected resets is explicitly unavailable until the next step. The host refuses those reads instead of presenting stale values as current.

Unrecoverable native commit failures set the worker fault marker and the host refuses further state or step use. Validation failures before native submission preserve the session. Shared-memory slot shapes/dtypes are checked before attachment, including zero-width action or state layouts.

## Inspection and playback

Versioned worker configuration reports are required and checked before being accepted. Entity assignment and body masses have scoped records distinguishing source intent and instance readback; source values never stand in for missing runtime fields. Complete selected-environment scene sources are returned for playback, retaining robot, object, table and mirror. Native rendering remains worker-owned; mapped-worker physics snapshot export is not yet exposed by this slice.

The current native camera profile captures the first entity in environment 0 with its existing tracking behavior. Only `cam_distance`, `cam_elevation` and `cam_azimuth` configure capture. Nondefault `cam_lookat`, `cam_tracking`, `cam_tracking_env_idx`, `cam_tracking_extra_envs` or `cam_fov` raise `NotImplementedError` before worker access, including repeated renderer initialization. Default `CameraCfg` values retain the existing native view; they do not select the MuJoCo grid camera. Interactive viewers also reject custom spherical offsets, because their view is controlled by the native viewer. Returning the complete source for any selected environment does not imply that native cameras can select that environment.

IsaacSim currently supports same-drive round-robin variant assignment and one-body rigid views, with explicit refusal of other combinations. Its fixed-root native root mode and environment view-row mapping are independently audited. The common host only enables the documented MJCF scalar-joint profiles, not URDF or all PhysX asset features.

## Validation and remaining work

`tests/contract/test_worker_scene_native.py` enables real factory-to-worker acceptance with `UNISIM_TEST_ISAACGYM_SCENE=1` or `UNISIM_TEST_ISAACSIM_SCENE=1`. Tests include non-round-robin Gym identity, the supported IsaacSim round-robin profile, nq/nv/nu, partial reset isolation and persistence, keyframe control distinct from qpos, complete-scene playback and a versioned import report. Worker-specific suites add independent native mass/COM/inertia, topology, mirror and passive-articulation checks.

The existing `model_file` entry point retains its cold importer and source settings, then adopts the initialized native objects into the same scene executor as explicit entities. `LegacySlotProjection` preserves historical root/state/control buffer shapes and names; it contains no physics loop. Both workers now have one step, reset and refresh implementation. Old D-wide actions (including passive columns) and synthetic 7/6 root coordinates remain an explicit compatibility projection, not a claim about authored free joints or actuator ownership. Gym's historical COM linear-velocity outputs and world angular-velocity root slot are translated separately from canonical link/body-frame coordinates. Existing ground/importer policies are retained on the cold source path; no new SDK dependency is added to the legacy Isaac host.

Native renderer acceptance, final four-backend evidence and released downstream dependency validation remain #113 requirements. These remaining gates must not be mistaken for completion of #108.
