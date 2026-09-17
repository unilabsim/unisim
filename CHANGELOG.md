# Changelog

## Unreleased

- Reject unsupported Isaac camera overrides before worker access instead of silently dropping explicit look-at, tracking/environment/neighbor and field-of-view settings. Native capture retains its existing environment-0 target and spherical offset; interactive viewers reject custom spherical options that they do not apply (#113).

- Normalize legacy Isaac model-file entry points onto the mapped scene executors (#109). Cold importers retain existing source policies; an SDK-free Python 3.8 compatibility projection preserves historical names, synthetic root buffers, D-wide controls and velocity conventions. Repeated step/reset/refresh loops are removed from the legacy workers; native maps and selected-reset submission are shared with explicit entity scenes. Legacy control snapshots are exposed as detached arrays.

- Unify MJWarp whole-state, entity-patch and default resets behind one prepared StateCommitPlan/native submitter (#109). State/model DR validation completes before host-cache or device mutation; default controls and variant-aware main-data routing are preserved, while homogeneous scratch forward remains an internal execution strategy. Native failures now consistently fault all reset intents, and unrelated state/control/force/sensor channels retain their lifecycle semantics.

- Allow selected entity resets to explicitly restore affected keyframe control/activation defaults in the same transaction, preserving unrelated entities and rows. MuJoCo and MJWarp now expose detached control-state snapshots alongside the Isaac mapped profiles so downstream Manager-Based reset can synchronize controls without backend-private access (#113).

- Unify MuJoCo full-state, entity-patch and default reset execution through one prepared StateCommitPlan (#109). Complete shape/finite/selected-row and physical-domain checks precede native writes; full-reset intent explicitly clears dirty bound force/warmstart and pending staging so stale episode forces cannot be uploaded again. Local entity intent preserves other entities, and legacy valid topologies/transmissions retain their existing executor path.

- Expose detached per-entity construction/keyframe state defaults in selected environment order on all four M2 adapters. The public query preserves each environment's fixed variant identity, performs no reset or source parsing, and lets downstream reset transactions avoid environment-zero broadcasts or private adapter state (#113).

- Add a cold compiled-model index shared by MuJoCo-family adapters for legacy root/body/joint/actuator partition auditing. It preserves anonymous names and complex native transmission records while cross-checking restricted entity layouts; it does not rewrite old models or claim generic tendon/ball/jointed-root entity support.

- Connect IsaacGym/IsaacSim entity workers through the public factory, complete state/action layouts, selected reset transaction, independent default controls and scoped native import reports (#109). Host source export preserves explicit compiled inertials/limits and avoids unsafe canonical actuator tags and USD filenames. Mapped root/body freshness and native failure poisoning are explicit; existing whole-model dispatch normalization and renderer acceptance remain tracked roadmap work.

- Implement MJWarp composed entities, immutable entity variants, world-frame entity state, selected resets and complete mocap playback on the existing main Model/Data runtime (#112). Reset preserves unselected persistent/control/force/sensor channels across its documented full-forward barrier; native failures fault state consumers. Real CUDA tests cover independent model/rollout references and entity/environment isolation. Fix the Genesis device-test environment cleanup so it cannot hide GPU 0 and falsely skip subsequent CUDA acceptance.

- Implement MuJoCo entity composition and one entity-bound fixed variant catalog on the existing mjbatch executor (#112). Cold-path namespacing and independent source compilation preserve multiple roots, passive joints, keyframes and variant inertials; kinematic visual mirrors retain independent pose without controls or collisions. Selected-entity resets prevalidate all writes, preserve other entities' control/force/activation state, and fault on native submission failure. Full playback snapshots include mocap pose. The supported MJCF subset is declared through M1; unsupported compiler/global-option or source combinations fail closed.

- Add validated entity/root/joint/actuator layouts with separate nq/nv/nu, strict scene wire schema, selected-reset prevalidation, and array-only root frame conversion. Shared-memory descriptors are validated before worker attachment and zero-width slots have safe backing allocation. These are #109 mapping/IPC foundations; native multi-entity execution remains gated until adapter integration.

- Add the roadmap #108 / issue #84 entity authoring and selected-entity reset value contracts: physical sources, one entity-bound immutable variant catalog, collision-free visual mirrors with independent poses, and explicit root/joint patches. Until an adapter implements composition, both factory and direct construction reject these declarations rather than discard them. Fixed variant assignments are detached from caller arrays and remain immutable across spawn.

## 1.4.3 - 2026-09-16

- Audited M1 report paths with reproducible A/B measurements: removed repeated snapshot freezing/serialization, avoided unused MJWarp device-row readback, released temporary source tables, narrowed actuator-only snapshots, and combined subprocess report normalization. Semantic condition validation now uses adapter-owned public reports consistently in both the factory and standalone validator; duplicate wrench/refresh feature names are consolidated. Existing Manager-Based startup/materialization ordering is unchanged. Bilingual ablation notes record measured cold-path costs and their limits.

- Added issue #91 M1 semantic capability declarations, version/profile-scoped evidence, immutable construction/materialization import reports, and opt-in fail-closed semantic requirements. Existing DR/play/fixed-variant APIs remain authoritative. Nine-adapter source inventory and bilingual ADR/migration documentation are generated/checked from public declarations; tiny CPU/CUDA/vendor-worker diagnostics distinguish actual runtime checks, approximations, rejections and unverified fields. Strict factory construction materializes before returning; legacy callers retain their lifecycle.

- Audited the M0 paths from issue #91: MuJoCo now refreshes only requested dirty body-state rows, preserves unread rows across partial resets/velocity updates, reuses cold-allocated scratch data, merges adjacent sensor copies, and avoids duplicate host kinematics inside native callbacks. Callback failures clear staged and dynamic wrenches on MuJoCo; MJWarp publishes completed-substep state/time even on interruption. MJWarp tracking uses one preallocated packed device-to-host transfer instead of four allocating strided transfers. All advertised MJWarp reset defaults now use immutable canonical/variant tables, with shared and K-variant storage instead of redundant N-environment default copies. Per-env COM queries reject malformed indices and avoid duplicate snapshot copies; the factory's MuJoCo-only refresh-option validation is consolidated.
- Optimized the MJWarp post-step tracked-body synchronization path. Final-state kinematics now run lazily on the first body-state getter (so ctrl-only callbacks and unread body state pay no refresh), and the private duplicate full-`sensordata` pinned cache is replaced by narrow per-block scratch buffers for only the four tracked sensor families. Force/contact and unrelated authored sensors retain their completed-substep values. The MuJoCo-only `refresh_pre_step_body_state` factory option is now rejected by other backends instead of being silently ignored, and an interrupted MJWarp callback leaves body state marked for lazy realignment.
- Extended `get_joint_range()` with an optional `names` argument (issue #86). MuJoCo and MJWarp resolve hinge and slide joints by name inside the adapter, preserve request order, convert hinge limits to radians, report unlimited scalar joints as `[-inf, inf]`, and reject unknown or non-scalar names with `ValueError`. The legacy no-argument table remains unchanged.
- Added the MuJoCo `refresh_pre_step_body_state` option (issue #89). The default `True` preserves substep-fresh tracked-body state in pre-step callbacks; `False` keeps body sensors and body-state getters enabled while omitting the Euler-only mjbatch split-substep sensor copyout, so generalized-state dynamic-wrench controllers work with `implicitfast`.
- Fixed `body_ipos` default stability and clarified the current-value query (unilabsim/unisim#87). `SimBackend.get_body_ipos()` now has two explicit forms: without arguments it returns the canonical model default table of shape `(nbody, 3)` in every mode (MJWarp fixed-variant mode previously returned the drifting per-env current table instead), and the new `env_ids` parameter returns the current effective per-environment values with shape `(len(env_ids), nbody, 3)` in `env_ids` order, reflecting all applied reset randomization including composition with `base_com_offset`, with partial resets leaving untouched environments unchanged. MJWarp `get_reset_term_default("body_ipos")` now returns the immutable default rows (each env's assigned variant row under fixed variants) instead of env 0's drifted current values. The MJWarp delta payload terms `base_mass_delta`/`base_com_offset` now compose against per-env immutable default rows, fixing a crash when they were used without an explicit `body_mass`/`body_ipos` payload outside fixed-variant mode; the MuJoCo adapter no longer crashes when a `body_ipos` payload is combined with `base_com_offset` (the flat coerced field was indexed with the shaped tail). The MuJoCo and MJWarp adapters implement the per-env query; other adapters fail closed with `NotImplementedError`.
- Fixed MJWarp public body-state timing after `step()` (unilabsim/unisim#85). Tracked body pose and velocity now correspond to the final qpos/qvel on both callback and callback-free paths, including multi-substep calls and consecutive control cycles, and lazy callback refreshes use the live per-world model. Contact and force sensors intentionally retain their completed-substep solver values instead of being recomputed by the kinematics-only body refresh.
- Fixed MuJoCo tracked-body state timing after `step()` (unilabsim/unisim#90). The first body-state getter after a control step now recomputes injected position/velocity tracking sensors from that step's final `qpos`/`qvel`, covering both direct and pre-step-callback paths, single and multiple substeps, and consecutive control steps. Authored acceleration-stage sensors such as contact forces keep the last physical substep's solved values instead of being replaced by a final-state full forward.
- Fixed generalized-state snapshot layout on the MuJoCo and MJWarp adapters (unilabsim/unisim#88). `get_state(("qpos", "qvel"))` now returns detached copies of the complete `nq`/`nv` layouts accepted by `set_state()` and addressed by named-state and root-layout indices. Fixed-base models no longer contain synthetic root columns, a free joint after another joint remains at its native column position, and MJWarp no longer routes these snapshots through legacy getters that require a first free joint.

## 1.4.2 - 2026-09-15

- Fixed the remaining MJWarp pre-step-control lifecycle issues from issue #71: tracked body state is now refreshed for the first callback of every control step (not only after reset), and an exception raised after one or more dynamic-wrench substeps still clears and synchronizes the device wrench channel so force cannot leak into a later step.
- Removed dead and redundant code surfaced by a repo-wide audit (all removals verified to have zero consumers in unisim and UniLab, with the full suite green before and after): the `unisim.backend.isaacgym.playback` alias module; the `FakeBackend.capabilities`, MuJoCo/MJWarp `_reject_wrench_write_inside_pre_step_control`, and MJWarp `set_pre_step_control` overrides that byte-identically duplicated the `SimBackend` base implementations; the `FakeBackend.set_state` historical Mapping-spelling migration branch; the unused `BackendCapability.MUTATION` enum member; the write-only `IntervalTermSpec.doc` field; the drake factory's silent `base_name`/`push_body_name` kwarg swallowing; write-only attributes on the subprocess/IsaacSim/Newton backends (Newton's fail-closed `audit_newton_model`/`calibrate_capacity` calls are kept, only their unread stored results are gone); and the unconsumed package-level `GENESIS_AVAILABLE`/`NEWTON_AVAILABLE`/`MJWARP_AVAILABLE`/`DRAKE_AVAILABLE`/`DRAKE_BATCH_AVAILABLE` flags (`MOTRIX_AVAILABLE` remains, as the factory and UniLab consume it).
- Consolidated triplicated backend helpers by concern: `TemporarySceneCleanup` moved to the new `unisim.backend.materialization_common`, `display_available` to `unisim.backend.playback_common`, and the Warp process-device binding core to `unisim.backend.process_device` (all standard-library only); sunk the three identical `get_play_capabilities` overrides into a shared `SimBackend._play_capabilities` class attribute; and made the IsaacGym worker-environment builder delegate to the shared `subprocess_ipc` `WorkerRuntime` (byte-identical env construction; the public `IsaacGymRuntime`/`build_worker_env` names are preserved). MJWarp and Genesis dependency errors now derive from the unified `OptionalDependencyError` like every other adapter; `except ImportError` handling is unaffected. No behavior changes.
- Fixed the remaining `mjbatch` → `mjbatch_uni` rename leftovers in the README BibTeX URLs (both languages) and a `pyproject.toml` comment.
- Added a static type-checking gate: `pyproject.toml` now configures mypy (`uv run mypy src/unisim`) and pinned pyright 1.1.408 (`uv run pyright`), with the Python targets (3.10), the pyright pin, and mypy's `no_site_packages` + `ignore_missing_imports` mechanism aligned with the sister unilab-rl repository (pyright additionally sets `useLibraryCodeForTypes = false` because unisim imports mujoco and other SDKs that ship no `py.typed` marker), `make typecheck` runs both (and `make check` is now lint + typecheck + test), and CI gained an Ubuntu `typecheck` job (Python 3.11, like unilab-rl) that the pre-release package job also waits on. Real issues surfaced by the checkers were fixed in code: the interval body-term handlers now fail closed through `require_op_body_ids` instead of relying on an unchecked `body_ids` invariant, `create_backend` types its `scene` parameter as `SceneCfg | None`, `SimBackend.get_state`/`model` gained the annotations the overrides already assumed, Drake's `init_renderer` signature matches the base contract, the MJWarp interactive ghost-mesh cache and DR mirror declarations are type-visible, and the Newton lazy-materialization attributes are annotated honestly. No behavior changes.

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
