# MJWarp entity composition and selected reset

[English](m2-mjwarp.md) | [中文](../zh/m2-mjwarp.md)

## Scope and construction

This CUDA slice of [#108](https://github.com/unilabsim/unisim/issues/108) implements the [M2 entity contract](adr-m2-entities.md) through one existing MJWarp model/data runtime. It reuses the [MuJoCo cold-path composition owner](m2-mujoco.md) to compile namespaced MJCF entities, validate actual public layouts and generate complete scene variants. It does not instantiate separate runtimes per entity.

Supported declarations include fixed/floating articulations, passive joints without extra actions, static rigid objects, kinematic rigid objects and collision-disabled mocap mirrors. One physical entity may consume a `same_layout` catalog with an explicit arbitrary assignment. Source formats, global-option/compiler conflicts and unsupported composition combinations follow the cold-path composition rules; the MJWarp variant validator additionally requires shared native fields to agree unless it implements a per-world representation.

Every entity/root/joint/actuator is bound from the compiled model. Fixed variant fields are installed through the existing MJWarp realization path, including mass, inertia, COM, geometry bounds and compiler-derived constant refresh. Generated sources remain owned until close so per-environment playback can load the complete selected scene.

The pinned `FixedVariantRealization` requires every variant geometry to have a unique non-empty compiled name. The adapter fails closed with that diagnostic and cleans generated sources; it does not silently invent names or infer geometry identity from declaration order. Named source geoms are therefore part of this current MJWarp profile. Anonymous-geom normalization is a follow-up composition improvement, not a runtime claim.

## Defaults, identity and state

The backend initializes each environment from its assigned full-scene source. The named default key, when selected, contributes joint qpos/qvel, controls and activations; otherwise compiled qpos0 and zero velocity/control/activation apply. Root and mirror poses come from `EntityInitialState`, with zero initial root velocity, overriding source/keyframe root values. Mocap poses, device time and selected key values are uploaded into the main Data before exposing the initial state.

`get_scene_layout()`, `get_entity_names()` and `get_entity_state(name)` use the same public layout as CPU composition. State snapshots are detached. Floating-root linear velocity refers to the root link origin; angular velocity is converted from MuJoCo generalized body-frame coordinates into world coordinates. Kinematic poses come from the current mocap cache. Neither selected nor full reset changes fixed variant identity.

The cached construction `ImportReport` adds `entity.initial_defaults` records scoped to variant and assigned environment IDs. Requested values have adapter-setting provenance from the compiled scene/key defaults. Effective values are read from actual main device Data after initialization uploads and forward. This does not relabel the composed defaults as original source values, or claim later live states remain equal to defaults. The reason records the root/keyframe override policy; current state queries remain authoritative after stepping or reset.

## Selected reset and state freshness

`reset_entities()` first calls the shared `prepare_scene_reset()` validator on coherent host snapshots. It resolves every patch before mutable submission, preserves input row order and uses frozen qpos/qvel/mocap mappings. Pose-only patches preserve world angular velocity. Only the selected state and its associated actuator, activation, force and warmstart channels are cleared; other entity/environment channels and staged wrenches are preserved. Full `reset(env_ids)` restores per-environment compiled/key defaults. The older generalized `set_state()` retains its existing whole-environment reset semantics; it is not silently redefined as a selected-entity transaction.

All three reset entry points now prepare an adapter-owned `StateCommitPlan` and use one `_commit_state()` submission path. A whole-world intent explicitly invokes `reset_data()` and restores the appropriate time, controls, activation and mocap defaults; a selected-entity intent preserves unmentioned state and honors `restore_default_controls`. Legacy models use their actual generalized-state widths without being converted into `SceneEntitySpec`. Model randomization prepares all updates, including final physical-domain validation, before changing any host cache or device field. Invalid shapes, nonfinite/overflowing values, negative mass/inertia/armature/friction and nonunit inertia quaternions fail before submission. Legal signed solref values retain their existing rules; a negative mass delta remains valid when the resulting mass is nonnegative.

The pinned `mujoco_warp.forward(model, data)` has no selected-world argument. Selected-entity commits therefore upload prepared values in place and forward the **existing main Data**, recomputing derived workspaces across the batch. They do not use `reset_data()`. The existing bounded scratch-forward optimization remains available inside the common submitter only for homogeneous legacy whole-world resets without model updates or mocap. Per-world variants always use the main Data so scratch-local world IDs cannot select incorrect model rows. This is a correctness rule, not a new selective-forward performance claim.

Explicit reset-barrier transfers snapshot persistent ctrl/act/qfrc/xfrc/warmstart channels. Unchanged values are retained and warmstart is restored after forward so recomputation cannot clear another entity's integration history. Unselected environments' host sensor snapshots remain unchanged. Authored sensor snapshots belonging to untouched entities in selected environments also retain their previous values; sensors belonging to changed entities refresh with the reset. Derived device kinematics/contact workspaces may be recomputed because contacts couple entities. Such recomputation is not a new physics step and is not exposed as a new force measurement for an untouched entity. Existing tracked-body freshness rules continue to apply when body queries request current kinematics.

Validation failures leave state unchanged. Native submission/forward failures mark the backend faulted; subsequent step and state/body/sensor/playback reads fail until reconstruction. No GPU rollback is promised.

## Playback and validation

Playback resolves the complete scene for the selected environment. Physics snapshots include the existing mocap position/quaternion tail, so independently positioned targets replay correctly. Closing the backend releases owned generated sources.

Run real CUDA acceptance on an available GPU, without overlapping another vendor-engine acceptance run:

```bash
uv sync --locked --extra mujoco --extra mjwarp
uv run --no-sync pytest -q tests/adapters/mjwarp/test_entities.py
uv run --no-sync pytest -q tests/adapters/mjwarp
```

Tests cover N=2/K=2 and N=5/K=2 with assignment `[1,1,0,1,0]`, native per-world parameter readback, fixed/floating robots and passive objects, COM-offset/link-world velocity checks against independent native MuJoCo data, reordered reset rows, control/activation/force/warmstart isolation, consecutive reset and step, named defaults, independent mirror poses and full-scene playback, fault handling, mirror/no-mirror trajectories and a short independent CPU oracle rollout. CPU/CUDA comparison uses stated numerical tolerances rather than exact cross-engine trajectories.

Final acceptance must retain the final SHA, engine/Warp version, GPU/driver, precise command, tolerances and unsupported/unverified scope. SDK availability, contract tests and constructor success alone are not runtime verification. Large-batch reset optimization, broad contact semantics, remaining backend implementations and the complete M2 audit remain separate work under #108.
