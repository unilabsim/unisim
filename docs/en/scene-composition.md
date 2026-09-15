# Declarative Scene Composition and Fixed Variant Pools

[English](scene-composition.md) | [中文](../zh/scene-composition.md)

Status: accepted contract for multi-asset URDF scenes on the IsaacSim subprocess adapter. This document is both the decision record for the contract expansion and the reference for its fields, fail-closed behavior, and handshake.

## Decision record

Six questions govern how far the public surface grows. The decisions:

1. **May a fixed variant plan bind to a single scene entity?** Yes, through `SceneEntitySpec.consumes_fixed_variant_pool`. Exactly one entity per scene may declare it, and the binding is validated symmetrically on the host cold path: a plan without exactly one declared consumer, a consumer (or a mirror declarer) without a plan, or a consumer whose entity shape cannot host a rigid-object pool all fail before any worker is spawned. Model-level realization (the IsaacGym/MJCF channel) and entity-bound realization (the IsaacSim/URDF pool) share `SceneCfg.fixed_variant_plan` as the single construction-time input.
2. **Is `entity_assets` backend-neutral or IsaacSim-private?** Backend-neutral by intent, single-consumer by capability. The typed declaration lives on `SceneCfg` because scene composition is owner configuration, not adapter internals; today only the IsaacSim worker materializes it, and every other backend rejects a declared scene at construction rather than ignoring the field. A second consumer would adopt the same declaration instead of a new one.
3. **How do unsupported backends fail closed?** At construction, through `validate_scene_composition_support`. Scene composition fields (`entity_assets`, `ground_plane`, `physx`, `env_grid_spacing`) are declarations, not hints: a backend that cannot materialize them raises `NotImplementedError` when the backend is constructed. This keeps the public surface unchanged (no new capability object) while removing every silent-degradation path. The subprocess family routes the flags through `_supports_entity_assets`/`_supports_ground_plane`/`_supports_scene_physx`/ `_supports_env_grid_spacing` hooks so the IsaacSim specialization owns its declaration.
4. **Is `FixedVariantMetadata.mass` a general public need?** Yes, as a measured quantity. URDF assets carry mass in their inertial blocks, but the USD conversion bakes it, so the backend is the only authority for what actually simulated. `SimBackend.get_entity_variant_metadata(entity)` reports the worker-measured per-environment mass table; assignment and scale stay out of the public type (they are construction-time inputs the owner already holds).
5. **What is the public semantics of rigid entity root state and wrench?** Each declared rigid entity owns one 13-wide root-state row layout (world xyz, wxyz quaternion, world linear and angular velocity) exposed through per-entity shared-memory slots and `set_state(entity_root_states=...)`. One `set_state` command is one transaction over the same environment rows; a rigid-only reset cannot perturb the articulation. World-frame force/torque wrenches stage through `apply_body_force` on the same roots, accumulate within one interval plan, apply for every substep of the next control step, and are consumed afterwards; a reset cancels staged rows for the reset environments.
6. **How is ground-plane consumption declared?** By presence: `SceneCfg.ground_plane` declares a world-level ground plane (friction/restitution/extent) that the IsaacSim worker authors as an offline-safe collision ground. An undeclared scene keeps the backend's native ground behavior. Backends whose scene models carry their own floor (the MuJoCo family) reject the declaration at construction — the MJCF file is the ground channel there, so a declaration would be silently dropped content.

## Scene declarations

| Field | Type | Default | Consumed by |
| --- | --- | --- | --- |
| `entity_assets` | `tuple[SceneEntitySpec, ...]` | `()` | IsaacSim |
| `ground_plane` | `GroundPlaneSceneCfg \| None` | `None` | IsaacSim |
| `physx` | `ScenePhysxCfg \| None` | `None` | IsaacSim |
| `env_grid_spacing` | `float \| None` | `None` | IsaacSim |
| `fixed_variant_plan` | `FixedVariantPlan \| None` | `None` | IsaacSim pool, IsaacGym model-level |

`ScenePhysxCfg` mirrors IsaacLab's `PhysxCfg` fields (solver type, iteration clamps, bounce threshold, friction offset/correlation, GPU contact stream buffers); declaring it is what opts a scene into explicit scene-level tuning, and undeclared scenes keep the backend's own defaults. `env_grid_spacing` declares the environment clone grid spacing in meters (undeclared keeps the native 2.0 m layout). Neither is a hint: backends that cannot consume them reject the scene at construction.

## Entity declarations

`SceneEntitySpec` declares one logical asset role: source file, format tag (`urdf`/`mjcf`), materialization (`articulation`/`rigid`), root mode (`fixed`/`floating`/`kinematic`), and the opt-in composition fields:

- `actuator_gain_overrides`: per-joint PD/dynamics table (URDF assets carry no gains); unknown joint names fail at scan time.
- `contact_friction` and `contact_friction_by_body`: PhysX contact material default and per-body overrides, written through the runtime PhysX views and read back fail-closed at INIT.
- `init_state` (`EntityInitStateCfg`): articulation spawn pose. A fixed-base robot's root pose has no other write channel, so this is the spawn channel; undeclared keeps the backend's default spawn pose. Articulation entities only.
- `collision_enabled`: the USD-bake collision flag. `None` keeps the converted-USD collision state; `False` disables collision for non-physical visual twins.
- `replace_cylinders_with_capsules`: the URDF converter flag. `None` keeps the materialization-based default (floating rigid entities convert with capsule replacement — the dynamic object contract; everything else keeps the converter default).
- `consumes_fixed_variant_pool`: binds this entity's asset to the scene's fixed variant plan (exactly one entity per scene).
- `mirrors_fixed_variant_pool`: a kinematic rigid visual twin that mirrors the pool target's per-environment variants. Mutually exclusive with consuming the pool and only valid on rigid kinematic entities.

No task semantics are inferred from entity names: the worker derives the bake plan from the declared materialization/root mode and the declared collision flag, never from the role name.

## The variant pool channel and its handshake

The IsaacSim realization of `fixed_variant_plan` is an entity-bound rigid object pool staged before any worker process is spawned. Host-side staging (`build_init_variant_pool_payload`) validates, fail-closed: exactly one consumer binding, a rigid floating URDF target, the `SAME_LAYOUT` plan layout, round-robin assignments only, and — through the host URDF scan — parseable single-root sources with no movable joints and a root/body layout identical across the catalog and the target's bootstrap asset. The validated pool rides the INIT payload; the worker converts each source once, bakes the physics, measures each variant's mass from the baked USD, and materializes the pool with `K` unique prototypes (`MultiUsdFileCfg`, `random_choice=False`): environment `i` takes source `i % K` by construction.

The INIT handshake is authoritative: the worker echoes `fixed_variant_count`, `fixed_variant_assignment`, and `fixed_variant_target_entity`, and the host compares count, target, and the full assignment against the immutable plan. Stage forensics (`variant_assignment.observed`) remain optional diagnostics — `None` is acceptable when prim stacks cannot be walked, but a computed observation that contradicts the echo fails closed. Identity is construction-time: resets never reassign variants.

Capability negotiation advertises `supports_fixed_variants` with the `SAME_LAYOUT` layout and `supports_per_env_playback` for pooled scenes; `get_playback_model(env_index)` resolves each environment to its assigned variant source and requires an explicit environment index. `get_entity_variant_metadata(entity)` expands the worker-measured mass table per environment.

## Evidence and limits

Host-side coverage lives in `tests/adapters/isaacsim/` (pool staging, the handshake guard including the tamper scenario, source pre-validation, layout and round-robin gates, playback resolution, mass readback, composition fail-closed) and `tests/contract/test_scene_composition_fail_closed.py` (cross-backend construction gates). Kit-level behavior (USD bake readback, scene PhysX application, pool materialization, wrench semantics) is exercised by the workspace probe suite rather than pytest.

Known deferrals, recorded as follow-ups rather than silent gaps: content- addressed URDF→USD conversion caching (every INIT reconverts), convert-once copy-per-role for pool mirrors (the mirror batch converts the sources a second time), and the exact K-prototype spawner for arbitrary (non-round-robin) assignments, which today fails closed with an actionable error. Only round-robin assignments and single-rigid-body URDF variants are supported; articulation, fixed-root, or kinematic pool targets and `UNIFORM_PUBLIC_LAYOUT` plans are rejected.
