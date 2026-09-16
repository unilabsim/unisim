# MuJoCo entity composition and selected reset

[English](m2-mujoco.md) | [中文](../zh/m2-mujoco.md)

## Scope and ownership

This is the MuJoCo CPU implementation slice of [#108](https://github.com/unilabsim/unisim/issues/108), following the [M2 entity decision](adr-m2-entities.md). It composes MJCF entity sources and executes the complete scene through the existing `mjbatch.Batch`/`VariantPack` runtime. It does not introduce separate simulators for individual entities or establish support for other engines.

`backend/mujoco/composition.py` owns cold-path source normalization, namespaced attachment, compiled layouts and temporary full-scene artifacts. `backend/mujoco/backend.py` owns batch state, native mappings, reset and playback. UniLab continues to own registered assets, task selectors, policy I/O and checkpoint compatibility. The source directory must remain available until construction finishes; the backend retains generated artifacts until `close()`/scene cleanup.

## Supported declaration slice

| Declaration | MuJoCo CPU behavior |
| --- | --- |
| Physical MJCF entity | One named root body, named non-root joints, no world-level geoms; multiple entities are attached under `<entity>/`. |
| Floating articulation or rigid object | The source root has one free joint; a rigid source has no non-root joints. |
| Fixed articulation or static rigid object | The source root has no root joint or mocap flag; passive child joints remain state and do not add actions. |
| Kinematic rigid object | Uses a mocap root, without retained free-joint dynamics or actuators. |
| Visual mirror | Inherits the target's source and fixed identity, has independent pose, uses mocap and removes joints, actuators, sensors and collision participation. |
| Entity variants | One physical consumer with `same_layout`; arbitrary validated assignments are preserved. Each catalog entry produces an independently compiled full scene. |
| Named source keyframes | Names are merged across physical sources; each realization preserves its own joint/control/activation values. |

Sources use the same global physics options; conflicts are rejected across entities and variants, independent of declaration order. The factory `sim_dt` explicitly overrides source timesteps. Relative mesh/texture references are resolved before generated XML is written. The declared base source and every catalog entry must agree on compiled public layout; sensor layout, key names and activation widths are also checked.

The current adapter rejects non-MJCF entity sources, `uniform_public_layout`, source tendons/equalities, unnamed keyframes, fragments, terrain composition and `visual_model_file` with the new entity entry point. It also rejects nondefault compiler settings that attachment would not preserve reliably, including `settotalmass`, mass/inertia bounds, static fusion and visual discarding. Existing whole-model entry points retain their existing support. Rejection does not imply that MuJoCo itself lacks those features; these combinations need explicit adapter implementation and evidence.

## Initial state and keyframes

Root pose always comes from `EntityInitialState`, using link-origin xyz and unit wxyz in the environment world frame. Initial root velocities are zero. These rules override both the original source root pose and every source keyframe's root pose/velocity. Mirrors retain their own declared mocap poses rather than following a target's pose.

The scene key set is the union of named keys from physical sources. For each key, the adapter copies compiled non-root joint qpos/qvel, actuator ctrl and activation values using actual joint widths and activation addresses. It does not concatenate guessed scalar joint arrays. An entity missing a key contributes its source qpos0 and zero qvel/ctrl/act. Same-name keys must have equal finite times. `default_keyframe_name`, when set, must exist in every scene realization. Without it, the source qpos0 path remains the default; the first key is not silently selected.

The backend stages the selected default separately for each variant/environment before binding the batch. A full `reset(env_ids)` restores those per-environment defaults, including controls, activations and mocap poses. `reset_entities()` instead applies only the explicit patch fields; omitted fields retain their current values, and neither reset changes variant identity.

## Public state and reset

`get_scene_layout()` returns the frozen compiled public addresses; `get_entity_names()` and `get_entity_state(name)` expose named root/joint state. Returned arrays are detached. Floating-root pose and velocity are obtained from current generalized state, converting MuJoCo's body-frame angular velocity into world coordinates. Fixed roots use their compiled pose, and kinematic roots use the batch's mocap state. Root linear velocity refers to the link origin, not the center of mass.

`reset_entities(SceneResetRequest(...))` validates the complete request and prepares scratch rows before the first write. It scatters only the requested columns and mocap entries, clears controls/activations and applied-force/warmstart channels associated with the changed state, and forwards only selected environments. Other entities' channels and unselected environments are preserved. The request's row order is retained for scatter; the native `forward(ids)` call receives sorted IDs as required by mjbatch. A pose-only write also converts the preserved world angular velocity to the new body's frame.

Selected-entity reset does not call whole-environment `Batch.reset()`. A native submission failure can leave partially committed state, so the backend becomes faulted and rejects further stepping or state consumption; reconstruction is required. Validation errors before submission leave state unchanged.

The existing full-state `set_state(env_ids, qpos, qvel, ...)` retains whole-environment reset semantics. It and `reset_entities()` now prepare different intents for one adapter-owned `StateCommitPlan` submitter. Full resets clear time, control, activation, pending/applied forces and warmstart in the selected worlds; entity patches preserve unrelated channels. Model writes and state shape/finite/range checks complete before the first native write. Invalid negative mass/inertia, armature, damping/frictionloss, geometry size/friction or nonunit inertial quaternions reject during preparation; signed solref/solimp conventions are retained. Callers requiring preservation of other entities still use `reset_entities()`; shared execution does not make whole-world and local-reset semantics identical.

## Playback and lifetime

`get_playback_model(env_index)` returns the selected environment's complete independently compiled scene, including robot, object, static geometry and mirror. `get_physics_state()` appends `[mocap_pos, mocap_quat]` when the composed scene has mocap bodies. The existing renderer consumes that tail, preserving independently moved visual targets. Scenes without that tail retain their existing snapshot representation.

The backend owns generated XML for its lifetime and cleans it on close or failed construction. Source parsing, key merging and identity/layout validation are construction work; step, selected reset and report access do not reparse assets.

## Construction report and provenance

The existing [M1 report schema](adr-m1-capabilities.md) gains adapter-owned fields without a new schema or capability registry. Every record is scoped to entity, catalog variant and the environments assigned to that variant; an unused variant has an empty environment set.

| Field | Requested/effective values and provenance |
| --- | --- |
| `entity.source_root_pose` | Original independently compiled source root pose → assembled compiled default pose. Provenance distinguishes source from compiled model readback. |
| `entity.keyframe_roots` | Original named key root poses/world velocities → assembled named key values after declared-pose/zero-velocity overrides. These are compiled key values, not current simulation state. |
| `entity.initial_defaults` | Selected source key or source defaults → adapter-staged initial/reset root/joint/ctrl/act values. Effective provenance is `adapter_setting`, not native batch readback. |

Values that differ are marked `overridden`; matching values are `exact`. A mirror's record makes removal of inherited joints/control/activation and its independent pose visible. The report is an immutable cached construction snapshot. It neither changes after reset/step nor proves that a later live state equals its default. Native batch acceptance is separate evidence; these fields do not automatically establish runtime-verified capabilities.

## Validation and remaining acceptance

Focused commands, from the UniSim repository root:

```bash
uv sync --locked --extra mujoco
uv run --no-sync pytest -q tests/adapters/mujoco/test_composition.py tests/adapters/mujoco/test_entity_runtime.py
uv run --no-sync pytest -q tests/adapters/mujoco
```

The fixtures exercise real MuJoCo/mjbatch CPU execution: N=2/K=2 and N=5/K=2 with assignment `[1,1,0,1,0]`; fixed/floating robots and passive objects; native mass/inertia and independent link-velocity readback with COM offset; partial resets preserving controls, activations, forces, warmstart and untouched rows; pose-only angular-velocity preservation; named defaults; complete playback; mirror/no-mirror rollout equality; independent native single-world rollout; physical contact and environment isolation; and validation/native-failure behavior. Composition tests cover relative meshes, global-option conflicts, key widths/times and cleanup.

The merge gate must run the repository checks on the final head and retain exact SHA, engine/runtime versions, hardware, commands, tolerances and unverified scope. Working-tree test runs are useful implementation evidence, not final-head acceptance or cross-engine equivalence. MJWarp and Isaac adapters require their own implementation and real runtime results under #108; this CPU slice does not close the entire milestone.
