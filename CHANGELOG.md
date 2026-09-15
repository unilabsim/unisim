# Changelog

## Unreleased

- **Declarative scene composition: declarations are consumed or rejected, never ignored.**  `SceneCfg.entity_assets`, `SceneCfg.ground_plane`, the new `SceneCfg.physx`, and the new `SceneCfg.env_grid_spacing` are composition declarations, not hints: every adapter states what it consumes through the shared `validate_scene_composition_support` gate (the subprocess family routes it through `_supports_entity_assets`/`_supports_ground_plane`/`_supports_scene_physx`/`_supports_env_grid_spacing` hooks, overridden by IsaacSim), and a backend that cannot materialize a declaration raises `NotImplementedError` at construction instead of silently degrading a declared multi-asset scene to single-asset behavior.  See `docs/en/scene-composition.md` for the decision record behind the contract.

- **Declarative scene-level PhysX and environment spacing (`SceneCfg.physx`, `SceneCfg.env_grid_spacing`).**  `ScenePhysxCfg` mirrors IsaacLab's `PhysxCfg` fields (solver type 0/PGS-1/TGS, iteration clamps, bounce threshold, friction offset/correlation, GPU contact stream buffers; defaults equal IsaacLab's so a partial declaration overrides exactly the authored fields) and `env_grid_spacing` declares the environment clone grid spacing in meters.  The IsaacSim worker re-validates both at the INIT wire boundary and applies exactly what the scene declared; undeclared scenes keep the backend's own defaults (IsaacLab `PhysxCfg`, the 2.0 m `GridCloner` grid) — solver tuning and layout no longer switch on scene shape, and cloning runs with explicit `replicate_physics=False`/`clone_in_fabric=False`.

- **Per-entity composition declarations on `SceneEntitySpec`.**  `init_state` (`EntityInitStateCfg`, articulation-only) declares the spawn pose — a fixed-base robot's root pose has no other write channel; `collision_enabled` is the USD-bake collision flag (`None` keeps the converted-USD state, `False` disables collision for non-physical visual twins); `replace_cylinders_with_capsules` is the URDF converter flag (`None` keeps the materialization-based default: floating rigid entities convert with capsule replacement, everything else keeps the converter default); `mirrors_fixed_variant_pool` binds a kinematic rigid visual twin to the scene's variant pool (mutually exclusive with `consumes_fixed_variant_pool`).  The worker derives bake plans from these declarations and the materialization/root mode — task semantics are never inferred from entity names, and per-entity entries carry the keys only when declared.

- **Declarative world-level ground plane (`SceneCfg.ground_plane`).**  Scenes declare `GroundPlaneSceneCfg` (friction triple `(0.5, 0.5, 0.0)`, `restitution=0.0`, `size_m=200.0` defaults mirroring IsaacLab `GroundPlaneCfg`'s physics material); the host serializes the declaration into the INIT `ground_plane` entry, and the IsaacSim worker consumes it as task-level scene composition: the offline-safe local world-level collision ground (`/World/ground` box, top surface z=0) spawns in training and playback alike, with the declaration defaults reproducing the original unconditional spawn parameter for parameter.  Undeclared scenes keep the backend's native ground behavior (render modes get IsaacSim's Nucleus `GroundPlaneCfg` floor at `/World/defaultGroundPlane`, headless gets none); backends that do not consume a declarative ground reject the scene at construction.

- **Construction-time fixed variant pools for multi-asset scenes.**  Whole-file model identity is carried solely by `SceneCfg.fixed_variant_plan`, bound to exactly one scene entity through `SceneEntitySpec.consumes_fixed_variant_pool`.  `IsaacSimBackend.materialize()` stages the pool through `build_init_variant_pool_payload()` before any worker process is spawned, and staging fails closed symmetrically: a plan without exactly one declared consumer (or mirror declarer), a consumer without a plan, a consumer whose entity shape cannot host a rigid-object pool, a `UNIFORM_PUBLIC_LAYOUT` plan, or a non-round-robin assignment (an arbitrary assignment would expand to one prototype per environment — O(num_envs) stage authoring — and the exact K-prototype spawner is a planned follow-up) never spawns a worker.  Host-side source validation scans every variant URDF before the worker starts: parseable single-root documents, no unsupported joint types, no movable joints (each variant must be one rigid body), a public root/body layout identical across the catalog and the target's bootstrap asset, and the `.urdf` extension enforced so a non-URDF file fails on the host instead of inside the Kit worker.  The validated pool rides INIT as `variant_pool`; the worker converts each source URDF once, bakes the physics, measures each variant's mass from the baked USD (the payload never carries masses), and materializes the pool through `MultiUsdFileCfg(random_choice=False)` with the K unique prototypes — environment `i` takes source `i % K` by construction, so reset never recompiles or reassigns.

- **Authoritative pool handshake and per-environment playback.**  The worker echoes the pool it materialized at INIT (`fixed_variant_count`, `fixed_variant_assignment`, `fixed_variant_target_entity`), and the host compares count, target, and the full assignment against the immutable plan; the stage forensics (`variant_assignment.observed`) are optional diagnostics (`None` is acceptable when prim stacks cannot be walked) but a computed observation that contradicts the echo fails closed.  Pooled subprocess scenes declare `supports_per_env_playback` and `get_playback_model(env_index)` resolves each environment to its assigned variant source (an explicit index is required); `get_entity_variant_metadata()` expands the worker-measured mass table per environment.  The legacy init-randomization channel (`apply_init_randomization`, `_SUPPORTS_INIT_MODEL_VARIANTS`) is removed; the channel has no post-construction entry point.

- **Typed multi-asset URDF scene contract.**  `SceneCfg` carries typed `entity_assets` (`SceneEntitySpec`: role name, `model_file`, `asset_format` urdf/mjcf tag, `materialization` articulation/rigid, `root_mode` fixed/floating/kinematic, and per-joint `ActuatorGainOverride` tables); the `entities` passthrough stays owner-level and unchanged.  The subprocess host scans every declared entity on the cold path (`scan_scene_entities`), deriving `fixed_base` per role — URDF roots take the declared converter flag (floating roots report the URDF root link as `freejoint_body_name`), MJCF declarations are cross-checked against the scanned free joint — and validates owner gain tables against scanned joint names fail-closed.  INIT carries an `entities` list with per-role asset/fixed_base/actuation payloads, and the top-level `dof_stiffness`/`dof_damping`/`dof_armature`/`dof_friction` arrays apply the primary entity's gain overrides, so URDF actuators honor owner gain tables.  Single-asset MJCF/URDF behavior is unchanged.

- **IsaacSim URDF scene entry.**  `scene.model_file` may point to a `.urdf`: the worker dispatches to Isaac Lab's `UrdfConverter` (payload-controlled `fix_base`, `urdf_self_collision`, `urdf_merge_fixed_joints`; zero-gain force position drives so the runtime ImplicitActuator layer owns gains) and patches the converted USD with `ArticulationRootAPI` on the named root link (the URDF converter emits only RigidBody prims).  The host metadata scan has a URDF branch reporting links/movable joints/limits and synthesizing zero-gain position actuators, replicating `merge_fixed_joints` semantics so the host/worker name handshake holds.  INIT carries a `fixed_base` flag; the worker skips root pose/velocity writes for fixed-base articulations.  MJCF behaviour is unchanged.

- **Manager reset routing for rigid entity roots.**  The public backend contract exposes materialized independent rigid-root names, and the UniLab reset transaction routes table/object/goalviz root writes through the corresponding `entity_root_states` slots instead of overwriting the articulation qpos/qvel root.  This preserves per-root reset isolation (no false per-step fall terminations) and the goalviz-only reset isolation promised by the multi-root IPC contract.

- **Multi-root IPC slots for rigid scene entities.**  `subprocess_ipc.protocol.slot_shapes` takes an optional `rigid_root_entities` parameter: each declared rigid entity (`materialization="rigid"`) owns one read slot `entity_root_state__<name>` ((num_envs, 13): pos xyz, quat wxyz, world linear and angular velocity, batch-first, host local frame) and one write slot `entity_reset_state__<name>` (same layout) consumed by `SET_STATE`; both families are allocated only when the scene declares rigid entities, and single-articulation scenes keep the original `SLOT_NAMES` layout and `{"count": count}` SET_STATE payload.  At INIT-metadata binding the host cross-checks the worker's rigid entity list against the declared specs fail-closed and maps each entity's scanned root body name to an extended body id (`num_bodies + declaration index`), so `get_body_ids` / `get_body_pos_w/quat_w/lin_vel_w/ang_vel_w` / `get_body_state_w` route rigid roots to the new slots while robot bodies keep the `body_state` slot; name collisions, worker/host entity mismatches, and out-of-range ids all fail closed.  `set_state` accepts keyword-only `entity_root_states` and allows `qpos`/`qvel` to be omitted as a pair: one SET_STATE command is one transaction over the same env rows, and a goalviz-only reset (`robot=False`) cannot perturb object or robot state.  Host and worker both validate shapes and finiteness of qpos/qvel/entity states; the worker publishes rigid root states on every refresh with a finite check, verifies the attached entity slot set against its materialized rigid objects at ATTACH, and writes entity roots through `write_root_pose_to_sim` / `write_root_link_velocity_to_sim` (world-frame velocities, env-origin translations added back worker-side).  Fixed-base robot root reads stay fail-closed.

- **Kit-verified rigid-root wrench path.**  Floating rigid bootstrap objects use the dynamic bake plan even without a variant pool; rigid entity roots expose the public 7+6 reset layout; and the dense wrench slots are applied to each IsaacLab `RigidObject` on every physics substep and cleared after the control step.  The D2 probe verifies root isolation, kinematic goalviz stability, and a world-frame force displacement in IsaacSim.

- **Subprocess interval wrench on the upstream dispatch contract.**  Interval `body_force`/`body_torque` terms route through the backend-owned `_interval_term_handlers()` table into the public `apply_body_force()` staging entry (world-frame force and optional torque on rigid entity roots), and every other interval term fails closed in the base dispatch with `NotImplementedError` naming the backend class and the term.  Staging accumulates within one interval plan: the thin `apply_interval_randomization` prologue clears the wrench slots per plan, so submissions from separate plans replace rather than add to each other — the contract expects at most one wrench term per plan, because a later plan's prologue would clear an earlier plan's staged rows.  Dense `wrench_force`/`wrench_torque` shm slots exist only for scenes declaring rigid entities, and `set_state` zeroes the selected rows so a freshly reset row never receives a pre-reset impulse.

- **Pre-step control fails closed as a declared gap.**  `set_pre_step_control()` on the subprocess family rejects callback registration with `NotImplementedError` (physics substeps are integrated inside the worker process, so a stored host callback would be silently dropped; declared gap); unregistering with `None` keeps the base contract.

- **IsaacSim worker USD bake, self-collision filters, and runtime contact materials.**  All role physics is authored into the converted USD before spawn (`_bake_usd_in_place` + `bake_plan_for_entity`, a literal port of the original repository's `_bake_usd` family: robot gravity-off/self-collisions-on/articulation solver 8/0, dynamic tool-pool variants, kinematic gravity-off with the declared collision flag, plus PhysX contact/rest offsets on every collision prim), so rigid spawns use plain `UsdFileCfg`/`MultiUsdFileCfg` exactly like the original `build_rigid_object_cfg`.  The robot USD carries `FilteredPairsAPI` for adjacent link pairs derived from URDF structure (`compute_adjacent_link_pairs`: fixed-joint merge graph, distance-2 pairs through `_VL` spacers; pair-exact with the original `adjacent_links.py` LEFT map on the Sharpa hand), replacing the task-side data file.  Contact friction is an opt-in cold-path channel: `SceneEntitySpec.contact_friction` (PhysX static/dynamic/restitution triple) and articulation-only `contact_friction_by_body` (`BodyFrictionOverride`) are validated at construction (finite non-negative triples, no duplicates, overrides require a default) and cross-checked against scanned body names in `scan_scene_entities`; the host serializes them into the INIT entity payload only when declared.  After the first `sim.reset()` the worker writes materials through each entity's `root_physx_view` (default tiled across all shapes, per-body overrides on their link's shape slice with the original per-link shape-count consistency check), reads the view back, and fails INIT on any mismatch.  INIT meta carries a fail-closed `bake` readback (re-opens every baked USD and verifies the plan attributes, contact/rest offsets, `collisionEnabled`, and the robot's FilteredPairs against the URDF-derived adjacency) and a `friction` summary for probes.

- **IsaacSim adapter tests grouped under `tests/adapters/isaacsim/`.**  The SimToolReal-scoped coverage (scene-entity contract, contact-friction channel, rigid-root slots and set_state transactions, fixed-variant pool staging/handshake/playback, ground-plane declaration, scene-composition fail-closed gates) lives in the adapter-owned subtree, matching the repository's core/contract/factory/adapters test layout; the Kit-level PhysX/USD assertions are exercised by the probe suite instead of pytest.

- **Bilingual scene-composition contract docs.**  `docs/en/scene-composition.md` and `docs/zh/scene-composition.md` record the decisions behind the contract expansion (single-entity pool binding, the backend-neutral-but-capability-gated `entity_assets`, construction-time fail-closed, measured `FixedVariantMetadata.mass`, the rigid-root state/wrench semantics, and ground-plane consumption) together with the field reference, the handshake, and the declared deferrals; the architecture docs reference it from both language trees.

## 1.4.1 - 2026-09-15

- Implement construction-time fixed model variants in the IsaacGym adapter (unilabsim/unisim#77). The worker loads each complete MJCF source once, validates identical public dof/body counts and name order, and creates every environment's actor from the immutable assignment row. Per-variant actuator properties and task-initial keyframes are mapped by joint name, the handshake echoes the assignment, playback resolves the assigned source, and layout drift fails closed with the variant filename. Reset-time model-field randomization remains undeclared on this adapter.


## 1.4.0 - 2026-09-14

- Fixed MuJoCo playback model resolution for scenes without fixed variants: the backend no longer advertises per-env playback merely because `VariantPack` is installed, and direct playback-model consumers now compile the renderable scene source. Visual-only geoms and meshes are therefore preserved in offline videos and interactive viewers while the physics executor continues to use `discardvisual`.
- Reorganized documentation into strictly parallel `docs/en/` and `docs/zh/` trees, removed the backend-specific README section in favor of the shared support matrix, normalized Markdown paragraph breaks, grouped tests under core/contract/factory/adapter subtrees, and documented the maintainer-only purpose of the reorganized `scripts/benchmarks/` and `scripts/diagnostics/` directories.
- Added per-environment gravity reset to the MJWarp adapter (unilabsim/unisim#71). The per-world `opt.gravity` vector is tiled during the cold-path model-field expansion, reset rows take effect for that world's post-reset forward, other worlds keep their gravity and state, and unsupported terms still fail closed through the capability negotiation.
- Completed the cross-backend body wrench contract (unilabsim/unisim#71). MJWarp `apply_body_force()` now accepts the world-frame torque channel, declares and handles the interval `body_torque` term, starts each non-empty interval plan from cleared staging, and both force and torque act at the target body's center of mass with MuJoCo `xfrc_applied` semantics: submissions accumulate within a control step, apply to every substep of the next `step()`, and are consumed afterwards.
- Extended `set_pre_step_control()` with per-substep dynamic wrenches (unilabsim/unisim#71). A callback may return the new public `PreStepControlOutput` (ctrl plus optional body_ids/force/torque); MuJoCo and MJWarp recompose the wrench from scratch every substep, compose it additively with the staged interval wrench, clear both channels when the step call finishes, and refresh tracked-body world state to the substep-start state inside the callback path. Calling `apply_body_force()`, `push_robots()`, or `apply_interval_randomization()` from inside a callback now fails closed instead of being silently dropped, and ctrl-only adapters reject wrench results with `NotImplementedError` rather than silently downgrading them.
- Require `mjbatch-uni~=0.2.1` and consume its split-substep sensor copyout (unilabsim/mjbatch_uni#28) as the sole MuJoCo pre-step-control body-state source: the tracked world-frame sensor views are memcpy-refreshed at every substep boundary including substep 0 (about 14x faster than a host-kinematics recompute at 4096 envs, with bit-identical trajectories). The host-side recompute fallback is removed rather than kept for older executors.

## 1.3.0 - 2026-09-13

- Implement fixed model variants in the MuJoCo CPU adapter. Construction-time `SceneCfg.fixed_variant_plan` is independently compiled for oracle/default extraction, layout-validated, merged through mjbatch `VariantPack`, and realized with per-world expanded model fields. Same-layout variants and uniform-public-layout optional mesh slots are supported; heterogeneous public topology fails closed. The adapter now exposes canonical/per-world reset defaults, additional curated reset terms (geometry solver fields, DoF damping/friction, and per-variant actuator tables), per-env compiler defaults on reset, and per-env playback without retaining one full compiled model per variant.
- Add the backend-neutral fixed-variant contract needed for per-env model identity. `FixedVariantPlan` carries a final read-only assignment, complete materialized `ModelSourceDescriptor` entries, and a same-layout/uniform-public layout declaration; it uses only stdlib and NumPy data and preserves its read-only assignment across pickle. `DomainRandomizationCapabilities` now advertises fixed-variant layouts and per-env playback, while `SceneCfg.fixed_variant_plan` is the sole construction-time lifecycle input. `SimBackend.get_reset_term_default()` defines authoritative canonical or per-world default exposure. The legacy `InitRandomizationPlan`, `ModelVariantSpec`, and `GeomSizeOverride` init-lifecycle API is removed. Contract tests cover negotiation and fail-closed behavior without exposing mjbatch, MuJoCo, or Warp objects.
- Implement construction-time fixed variants in the MJWarp adapter. Each complete MJCF source is compiled independently as the correctness oracle, validated against `same_layout` or `uniform_public_layout`, and merged into one canonical asset pool with stable named geom slots. After `put_model` and model-field expansion but before the first forward and CUDA-graph capture, the adapter installs per-world `geom_dataid`, `geom_matid`, and the eleven mesh-dependent model fields. Host reset mirrors and reset-term defaults use the assigned variant rows, and playback resolves each world to its source model. Variant identity is accepted only at construction because replacing an initialized Warp model would invalidate captured pointers.
- Consume the published `mjbatch-uni~=0.2.0` executor API; the integration-only git dependency is removed.

## 1.2.1 - 2026-09-13

- **Breaking (mujoco executor):** the MuJoCo adapter's native batch executor is now the unilabsim `mjbatch` fork (`mjbatch.Batch`, published on PyPI as `mjbatch-uni~=0.1.0` and pulled in by the `mujoco` extra), replacing the `mujoco-uni-runtime` `BatchEnvPool`. Canonical state storage is the batch's bound per-field views (`time`/`qpos`/`qvel`/`act`/`ctrl` bound at the configured numpy dtype, `xfrc_applied`/`qacc_warmstart` native float64, `sensordata` at the configured dtype); the adapter no longer ships full state rows to the pool or maintains a host FULLPHYSICS array. Behavioral contract (each with dedicated tests in `tests/adapters/mujoco/test_batch.py`): `xfrc_applied` is written absolutely before every dispatch (an idle step writes zeros, so staged wrenches cannot persist in the now-batch-persistent channel); warmstart is structurally zeroed on `set_state` (`Batch.reset` runs `mj_resetData` before overlaying per-field writes) and explicitly on the interval velocity-delta path; state layout offsets are derived from `mujoco.mj_stateSize` per component instead of hardcoded FULLPHYSICS offsets. The pre-step control hook is driven by mjbatch's native per-substep callback (`fn(k, state_view, ctrl_view)`; sensordata stays one substep behind qpos/qvel, matching `post_step_forward_sensor=False`, the only mode the previous executor's default served).
- **Breaking (mujoco):** per-env model variants are no longer supported (`apply_init_randomization` model-variant plans now fail closed via the base class); field-level reset randomization uses mjbatch `expand` views + `set_const` (lazy first expansion allocates one model copy per worker thread, so DR tasks pay `nthread x model` memory instead of `num_envs x model`). `get_physics_state` snapshots are exactly `[time, qpos, qvel]` per row (the old rows carried a FULLPHYSICS tail), which also fixes the previous length mismatch for `na > 0` models in the offline render workers. Height scanning and site Jacobians run as mjbatch query ops on the live state; query ops skip the bound-field CopyOut, so the bound views are untouched by the calls. The height scanner is `output="height"`-only on this backend and passes `alignment` through to mjbatch (`"world"`/`"yaw"`).
- **Breaking (mujoco):** `post_step_forward_sensor` is removed end to end (its only `True` behavior is unreachable on the new executor); the chunk tuner is deleted entirely (`chunk_size`/`adaptive_chunk_size` are warn-and-ignore `DeprecationWarning` shims at the factory, and `bench_nsteps` is accepted and ignored by the factory). Models with `sleep` enabled now fail fast at `Batch` construction.
- Playback model resolution no longer maps per-env variant geom sizes: one visual model file (or one saved mjb) serves every rendered env, and `materialize_visual_playback_model` is removed from the mujoco package exports.

## 1.2.0 - 2026-09-10

- Promote the current contract and adapter surface to the `1.2.x` line. No functional changes since 1.1.6; the public import boundary (`SimBackend`, `create_backend`, `ADAPTER_SPECS`, adapter classes, and `unisim.backend.subprocess_ipc`) is unchanged.

## 1.1.6 - 2026-09-10

- **Fix:** SuperDex serial-mode native interactive playback now frames the scene (`viewer.frame_scene()`) immediately after `set_scene`, before the first `frame_tick`. Polyscope's camera view matrix is uninitialized (NaN) until the first explicit camera placement, and the viewer's navigation gizmo reads it while building the first ImGui frame, so on-screen interactive playback crashed on the first frame with `ValueError: cannot convert float NaN to integer` (UniLab `eval --sim superdex --render-mode interactive`).

## 1.1.5 - 2026-09-10

- **Breaking (snapshot layout):** `mjwarp` `get_physics_state` snapshots now append `[mocap_pos(nmocap*3), mocap_quat(nmocap*4)]` after `[time, qpos, qvel]` when the model has mocap bodies, and `run_playback_mode` declares the extended `snapshot_shape`. The offline render workers (`render_many`) replay the recorded mocap pose instead of resetting mocap bodies to the model defaults, fixing record-mode videos where mocap-driven geometry (e.g. the Wuji mocap palm, whose wrist pitch is randomized at reset) rendered misaligned with — and interpenetrating — the free-joint objects. Legacy `[time, qpos, qvel]` snapshots keep the previous defaults-plus-grid-offset fallback. `validate_offline_visual_model` now also requires `nmocap` parity between the physics and visual models.

- **Fix:** `ghost_geom` debug overlays now inherit the material (with mesh UV texturing) of the model geom that renders the same mesh, in both the offline render workers and the mjwarp interactive viewer, matching the source task's textured goal indicator. `append_debug_primitives` gains an optional `mesh_materials` mapping; assets without a textured model geom keep the flat primitive rgba.

- **Fix:** multi-env grid recording without an explicit `cam_lookat` widens `cam_distance` so every grid cell fits the frame (`render_many` `_grid_fit_distance`, from the grid span, fovy, and frame aspect ratio). An explicit `cam_lookat` still pins the camera to a single env.
- Add a SuperDex `execution_mode` option (`superdex_execution_mode` factory kwarg, `"batch"` default or `"serial"`). Serial mode never constructs the `SceneBatchExecutor` and steps every scene on the environment thread so the native SuperDex debugger can attach without violating the scene's thread-affine `DebugDraw`. Batch mode now fails closed with an actionable `RuntimeError` naming the serial mode when a debugger client is connected at construction or attaches before a later step (unilabsim/unisim#55). Serial mode also enables native interactive playback: `run_playback` in the `interactive` render mode drives the upstream Polyscope viewer on the single environment scene, failing closed unless the backend is serial with `num_envs=1`.

- Switch the SuperDex adapter's optional runtime to the temporary unilabsim `superdex-physics-uni` / `superdex-robotics-uni` 1.0.0 wheels, which carry the native batch executor ahead of the upstream project_superdex release, and extend the supported interpreter range to CPython 3.12 and 3.13. The adapter still rejects other Python versions with a targeted diagnostic. Switch the distribution names back to upstream once the upstream PR merges. Map unlimited actuator force ranges to the dtype's finite bounds so the native `step_control` validation accepts MJCF motors without a `forcerange`.

- **Fix:** multi-env grid rendering in `render_many.render_frame_job` now translates mocap bodies with the environment. Worker `MjData` is reused across frames, so `init_worker` caches cold-path `mocap_pos` defaults and `set_state` resets from them before adding the grid offset; mocap bodies are excluded from the legacy `geom_xpos`/`site_xpos` post-shift so the two mechanisms cannot double the offset. Previously, models whose first body has a free joint (e.g. the Wuji in-hand scene: free-joint cube plus mocap palm) rendered every env's mocap-driven geometry stacked at env 0, misaligned with both the free-joint objects and the debug overlay primitives, which already receive the offset exactly once (unilabsim/wuji_unilab#21).

- **Breaking:** replace `run_playback(..., extra_data_getter=...)` with `debug_overlay_getter`. The new callback returns per-frame, per-env sequences of typed `DebugPrimitive` values (`sphere`, `box`, `frame`, `arrow`, `ghost_geom`, `text`) with env-local poses instead of a single `(num_envs, 3)` marker-position array; grid offsets are applied by the renderer. `BackendPlayCapabilities` gains `supports_debug_overlay`; the MuJoCo-family offline snapshot pipeline (mujoco, mjwarp, drake, newton, superdex) advertises it, while other backends fail closed with `NotImplementedError` when a getter is supplied. `ghost_geom` primitives resolve `mesh_asset` against playback-model mesh names or mesh asset files injected into the render model; `text` primitives are a documented no-op on the MuJoCo off-screen path. Newton record playback with overlays routes to the offline MuJoCo snapshot renderer (the native ViewerGL path cannot inject user geoms).
- mjwarp interactive playback now consumes `debug_overlay_getter`: each frame injects the tracked world's primitives into the passive viewer's `user_scn` before `sync()`, resolving `ghost_geom` meshes against the playback model (fail-closed when unregistered). `BackendPlayCapabilities` gains `supports_interactive_debug_overlay` (default False; mjwarp reports True) so callers can tell whether the interactive path consumes the getter. The interactive `on_frame` hook remains fail-closed (unilabsim/wuji_unilab#21).
- **Breaking:** `camera_kwargs` is normalized into the typed frozen `CameraCfg` at the `run_playback`/`init_renderer` boundary. Unknown keys — including the historical `distance`/`elevation_deg`/`azimuth_deg` aliases — now raise an error naming them instead of being silently ignored; the optional `cam_fov` key is supported by the MuJoCo offline renderer and Genesis. `DebugPrimitive`, `DebugOverlayGetter`, `CameraCfg`, and `validate_debug_overlays` are exported from `unisim` and `unisim.contract` (unilabsim/wuji_unilab#21).

- Add `unisim.visualization.render_many.append_debug_primitives`, the public primitive-injection entry shared by the offline render workers and interactive viewers (single-env `viewer.user_scn` callers pass `overlays=[primitives]`, `offsets=None`); it returns the injected geom count. Interactive `ghost_geom` meshes must already be registered in the loaded model (resolved via `mesh_ids`), failing closed otherwise — the interactive path cannot recompile the model (unilabsim/wuji_unilab#21).

- Add `run_playback(..., on_frame=...)`: the offline MuJoCo pipeline calls `on_frame(frame_index, frame)` with each `(H, W, 3)` uint8 frame before video encoding; returning a replacement array (same shape/dtype, validated fail-closed) substitutes it and `None` keeps the original. Backends on native renderers (motrix, genesis, subprocess IPC; newton/mjwarp interactive paths) fail closed with `NotImplementedError`; newton record playback with `on_frame` routes to the offline snapshot renderer (unilabsim/wuji_unilab#21).


- Add a local-source SuperDex `SceneBatchExecutor` integration: a persistent C++ CPU barrier batches independent-scene generalized force writes, stepping, and articulated state reads. `superdex_num_workers=0` resolves an affinity-aware outer worker count while SDK-internal and outer workers remain mutually exclusive.

- Batch SuperDex body and sensor frame transforms over selected environments, removing repeated small-array work while preserving native stepping order, controls, sensor precision and reset isolation.

- Add the optional SuperDex 1.0.0 CPU adapter with native fixed-base bot and audited MJCF articulation materialization, NumPy state/control translation, independent scene resets, named state/contact sensors and process-owned cleanup. Python 3.12 is required by the upstream wheels. See `docs/en/superdex.md` for the experimental contact profile and explicit limits.

## 1.1.4 - 2026-09-08

- Add cold-bound selected-world mocap pose reads/writes and reset ordering to `SimBackend`, with an explicit unsupported default and a MJWarp implementation.
- Add MJWarp reset randomization for primitive geometry size (including derived bounds), contact solref/solimp, joint damping and joint friction loss. New payload tables validate before mutation, preserve unselected worlds, and expose cold-path defaults through the public backend contract. See [the owner contract](docs/en/mocap-reset-contract.md) and issue #40.
- Preserve position actuator gain signs in MJWarp domain randomization.

## 1.1.3 - 2026-09-06

- Ensure Newton ViewerGL playback shows authored static planes and supplies a default visual-only floor when a scene does not define one.

## 1.1.2 - 2026-09-06

- Merge Newton's native ViewerGL dependencies (`pyglet` and `imgui-bundle`) into the single `newton` extra. Newton playback now uses its native renderer by default after `uv sync --extra newton`; the separate `newton-render` extra is removed.

## 1.1.1 - 2026-09-06

- Align all MuJoCo-related extras on the 3.11 line (unilabsim/UniLab#1515, unilabsim/unisim#34): the `mujoco` extra now requires `mujoco~=3.11.0` with `mujoco-uni-runtime==0.5.0` (exact pin — one runtime release carries one prebuilt MuJoCo binding, so a lock bump is gated on wheel availability); the `mjwarp` extra moves from `mujoco-warp==3.10.0.3` to `mujoco-warp~=3.11.0` with `warp-lang==1.16.0`; and the `newton` extra keeps its exact pins. The extras are now jointly resolvable, so the `[tool.uv]` extra conflicts are removed and the MJWarp runtime check accepts the whole `mujoco-warp` 3.11 line instead of one exact version.
- Add snapshot playback support to the Newton adapter: `get_physics_state` / `set_physics_state` ([time, qpos, qvel] host-cache layout), record/none `resolve_play_render_plan` semantics, and `run_playback` through the shared offline MuJoCo snapshot renderer now in `unisim.backend.playback_common` (mjwarp behavior and messages unchanged). Add a fail-closed `SimBackend.set_physics_state` default.
- Add MJCF contact-sensor (`mjSENS_CONTACT`) support to the Newton adapter for the exact `data="found" num=1` named-geom-pair shape: per-env binary flags are resolved once against `SolverMuJoCo.mjc_geom_to_newton_shape` at materialization and refreshed through `SolverMuJoCo.update_contacts`; all other contact-sensor configurations remain fail-closed.
- Rework the README around the project overview, UniLab relationship, installation, and quick start, and add the Chinese `README_zh.md`.
- Add the Newton 1.5.1 runtime extra and a metadata/import probe on the MuJoCo-Warp 3.11 line (the 3.11 alignment above later lifted the initial mutual exclusion with the `mjwarp` extra).
- Add fail-closed Newton nconmax/njmax capacity sampling and overflow diagnostics for the forthcoming adapter.
- Add the Newton `SimBackend` adapter with explicit CUDA placement, cold-path MJCF materialization/audits, host NumPy state caches, and fail-closed sensor and geometry coverage.
- Fixed Genesis device selection on multi-GPU hosts: the engine only honors the first entry of `CUDA_VISIBLE_DEVICES`, so the adapter now pins `CUDA_VISIBLE_DEVICES` to the requested physical device before any CUDA query (including `torch.cuda.is_available()`, which itself latches the visible-device set) and remaps the process-local device index to `cuda:0`. Out-of-range requests fail closed with a clear error.

## 1.1.0 - 2026-09-05

- Added the declarative interval domain-randomization term contract in `unisim.dr.interval`: builtin term specs (`INTERVAL_TERM_SPECS`, `interval_term_spec`), the pickle-safe `IntervalTermOp` descriptor with builtin-contract validation, and the `ops` field on `IntervalRandomizationPlan` (`iter_ops()` translates the legacy fields).
- Added the `supported_interval_terms` capability set on `DomainRandomizationCapabilities` with `supports_interval_term()` / `get_unsupported_interval_terms()`, falling back to the legacy bools so old constructor call sites keep their meaning.
- Replaced the abstract per-backend `apply_interval_randomization` implementations with generic `SimBackend` dispatch over the backend-owned `_interval_term_handlers()` table; terms without a handler fail closed with `NotImplementedError` naming the backend class and the term.
- Deprecated the five legacy `IntervalRandomizationPlan` fields and the five `supports_interval_*` capability bools; they remain functional and will be removed in the next major release.
- Fixed the mjwarp and genesis backends silently dropping unsupported interval body-torque and body-angular-velocity randomization; both now fail closed through the base dispatch.

## 1.0.0 - 2026-09-04

- Promote the contract and seven-adapter manifest to the stable `1.0.x` line; the public import boundary (`SimBackend`, `create_backend`, `ADAPTER_SPECS`, adapter classes, and `unisim.backend.subprocess_ipc`) is now stable.
- Support `SimBackend.set_pre_step_control` on the `mjwarp` backend: a registered converter now runs on the host before every physics substep with the qpos/qvel cache refreshed to the substep-start state (matching the MuJoCo backend's substep boundary and `callback_sensordata=False` sensor semantics), and `None` unregisters it.  The callback path uses eager kernel launches instead of captured step graphs.
- Restore the missing 0.1.10 changelog entry and the 0.1.4/0.1.5 ordering, and correct the `unisim.backend.subprocess_ipc` path and Isaac extras spelling in the migration and support-matrix documentation.

## 0.1.14 - 2026-09-02

- Update the trusted-publishing action to support the source distribution's current Python Core Metadata version.
- Require successful cross-platform tests and pre-release sdist verification before a version tag can publish to PyPI.

## 0.1.13 - 2026-09-02

- Add GitHub Actions CI and tag-triggered PyPI trusted publishing.
- Publish only the source distribution so releases do not select a Python version, operating system, or wheel platform.
- Document repository development, compatibility, and release conventions.

## 0.1.12 - 2026-09-02

- Fix the root public export surface so wildcard imports resolve `MjcfSubprocessBackend` and its historical `SubprocessBackend` alias.
- Use package-owned `UNISIM_*` worker/cache environment variables and `~/.cache/unisim` defaults, with read-only fallback to legacy `UNILAB_*` overrides.
- Expand adapter/factory/import-boundary tests for the complete seven-backend manifest and package isolation.

## 0.1.11

- Corrected the Drake adapter to consume the external `drake-uni` distribution through its `drake_uni` import namespace.
- Added fail-closed import diagnostics and support-matrix documentation for the standalone Drake runtime boundary.

## 0.1.10

- Replaced the `drake` PyPI dependency with the external `drake-uni==0.1.0` distribution and aligned the Drake adapter with its batch runtime API.

## 0.1.9

- Added the MuJoCo batch runtime to the `mujoco` optional extra so the production adapter is installable from a clean UniSim environment.
- Made conformance checks exercise adapters through their cold-path `materialize()` lifecycle before stepping.
- Expanded standalone MuJoCo and Motrix adapter tests to cover full state shapes and identity-quaternion reset semantics.

## 0.1.8

- Kept playback video I/O monkeypatchable while preserving lazy optional ``imageio`` loading for import isolation.

## 0.1.7

- Corrected factory option translation for the extracted backend adapters.
- Preserved backend-specific validation and fail-closed diagnostics when callers pass options from the UniLab owner layer.

## 0.1.6

- Migrated the complete production backend implementations and shared subprocess IPC into `unisim-core`.
- Removed the test-only runtime bridge from the public factory so every named backend resolves to its concrete adapter and fails closed when unavailable.
- Added support-matrix, migration, and package-boundary documentation for all seven adapters.

## 0.1.5

- Exported adapter-specific dependency diagnostics and the shared subprocess backend types from the public `unisim` namespace.

## 0.1.4

- Added public Drake, MJWarp, Genesis, IsaacGym and IsaacSim adapter boundaries.
- Added shared subprocess IPC framing used by Isaac worker integrations.
- Promoted all seven UniLab backend identities to the adapter manifest; SDK availability remains lazy and fail-closed.

## 0.1.3

- Add the staged adapter identity manifest for all roadmap backends.

## 0.1.2

- Add the lazy Motrix adapter and shared contract smoke coverage.

## 0.1.1

- Add the lazy MuJoCo adapter and backend factory.
- Add MuJoCo contract smoke coverage and adapter documentation.

## 0.1.0

- Bootstrap the `unisim` namespace and `unisim-core` distribution.
- Add the backend-neutral `SimBackend` contract, fake backend, and conformance helper.
- Reserve benchmark case/result interfaces without implementing workloads or measurements.
