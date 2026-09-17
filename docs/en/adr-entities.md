# Design decision: entities, fixed identity and selected reset

[English](adr-entities.md) | [中文](../zh/adr-entities.md)

## Status, scope and owners

**Status: Accepted.** This decision defines the public entity, immutable identity and selected-reset contracts. Adapter-specific profiles may support only a subset and must reject unsupported combinations explicitly.

UniSim owns physical entity declarations, materialization, native mappings, IPC, adapter lifecycle and conformance. UniLab owns asset registration/materialization before submission, logical selectors, task configuration, Manager-Based scheduling, policy I/O and checkpoint compatibility. No SDK objects or UniLab dependency enter the public values. Capability and evidence rules follow the [capability decision](adr-capabilities.md).

## Context

A robot, a movable object, a static table and a visual target require separate identity and state even when they share one environment. An object may have passive joints that contribute state without increasing the action dimension. Selecting a tool variant must not require callers to duplicate a complete robot/table scene for each tool. Actor creation order and a single root at the beginning of qpos/qvel cannot express these requirements reliably.

The design separates source intent, the compiled public layout and native execution. This first slice implements source/reset request values and rejection of unsupported composition. Frozen native layouts, execution and their evidence remain implementation work.

## Entity and variant declarations

The public values live in `unisim.entities`; `SceneCfg` carries their scene-level relationships.

| Value | Decision |
| --- | --- |
| `EntityInitialState` | Root-link position and unit `wxyz` quaternion; source/keyframe owns joint defaults, and initial root velocities are zero. |
| `SceneEntitySpec` | Stable name, `ModelSourceDescriptor`, explicit format, `articulation`/`rigid` kind, `fixed`/`floating`/`kinematic` root mode, initial pose, collision flag and optional `mirror_of`. |
| `EntityVariantBinding` | One `target_entity` and one existing immutable `FixedVariantPlan`; this is the only variant-consumer binding. |
| `SceneCfg.entity_assets` | Tuple of physical source or mirror declarations, distinct from `SceneCfg.entities`, which remains the logical selector mapping. |
| `SceneCfg.entity_variant` | Optional binding; the first version allows at most one physical variant consumer and any number of explicitly declared mirrors. |

Entity names match `[A-Za-z][A-Za-z0-9_-]*` and are unique within the scene. Physical entities require a source; descriptors contain file paths, not live engine objects. Recognized format names (`mjcf`, `urdf`, `usd`, `superdex_bot`) are authoring vocabulary, not promises that every adapter imports them. Source topology, root-mode compatibility and supported format combinations require adapter validation during materialization.

A mirror references an existing physical entity directly: self references, chains and missing targets are rejected. It inherits source and fixed variant identity, **not pose**. It has its own initial pose, is rigid and kinematic, disables collisions and introduces no controls or physical influence. It cannot declare a competing source or be the variant consumer. Its declared format must match its target. A visual target may therefore show the selected tool at a different position without changing the simulated tool.

`FixedVariantPlan.assignment` remains the final immutable environment-to-catalog mapping, with its length checked against `num_envs` when available. In an entity binding, catalog sources describe the target entity; in the existing `fixed_variant_plan` entry point they retain whole-model meaning. Reset never resamples identity. The first implementation must not silently substitute round-robin for an explicit assignment. `same_layout` requires a semantic layout match, not merely equal array lengths; `uniform_public_layout` needs adapter-specific remapping, padding and real verification before support can be declared. Multiple independent consumers, changing topology on reset and cross-entity transmissions/constraints require separate decisions.

## Coordinate and selected-reset contract

Root pose refers to the **root link origin**, not its center of mass. Position uses the environment world frame with any backend clone translation removed; orientation is a unit quaternion in `wxyz` order. Root velocity contains link-origin linear velocity followed by angular velocity, both expressed in that world frame. Native COM velocity or body-frame angular velocity must be converted by the adapter. Initial poses use the same convention.

`EntityStatePatch` addresses one entity and carries any nonempty combination of these fields:

| Field | Columns and meaning |
| --- | --- |
| `root_pose` | Seven columns: link-origin xyz, then unit wxyz. |
| `root_velocity` | Six columns: link-origin world linear velocity, then world angular velocity. |
| `joint_positions` | Packed qpos columns in `joint_names` order, using each selected joint's bound qpos width. |
| `joint_velocities` | Packed qvel columns in `joint_names` order, using each selected joint's bound qvel width. |
| `joint_names` | Unique entity-local names; an empty tuple selects all non-root joints in the bound entity order. |

Joint widths are not assumed to be one: a spherical joint has different position and velocity widths. The adapter checks names, widths, spherical-joint quaternion validity and legal root operations against the frozen layout before writing. A fixed root does not acquire a writable pose merely because a patch can represent one. Missing fields mean preserve, not zero. Numeric arrays are copied into immutable storage, must be finite and two-dimensional, and supplied fields must have equal row counts. Root-pose quaternions are validated without silently normalizing caller input.

`SceneResetRequest(env_ids, patches)` preserves the caller's selected-row order. Environment IDs are a nonempty tuple of distinct nonnegative integers; boolean IDs are rejected. Each patch has exactly that many rows. There is one patch per entity, so callers combine that entity's writes instead of submitting conflicting duplicate patches. The runtime additionally checks environment upper bounds and entity/layout permissions.

`restore_default_controls=True` explicitly restores the affected actuator and activation defaults in that same native transaction. It does not reset another entity or environment, and omitted state fields still preserve their values. The default false retains ordinary manual-patch clear/hold behavior. This lets a Manager-Based reset restore keyframe control independently of joint position without a second private write. All four mapped backends also expose detached `get_state("ctrl")` snapshots so the consumer can synchronize its action buffer with the effective backend controls.

The current request shape uses one shared `env_ids` tuple for all entity patches; a downstream transaction therefore requires the same selected environment rows for every entity in one commit. Different per-entity row sets require a future row-mask extension and must fail or be split by the owner rather than silently rewriting extra rows. The first consumer fixture supports scalar hinge/slide joints and rejects other widths.

The runtime transaction has four required phases:

1. Validate **every** selector, patch, shape, frame and value before the first native write. Validation failure guarantees zero state mutation.
2. Translate through frozen mappings and submit only the selected environments, entities and fields.
3. Apply the M0 control-target, pending-wrench and cache lifecycle for the affected state, then refresh it. Preserve unrelated entity/environment state and fixed variant identity.
4. If native submission fails partway through, either perform a proven rollback or mark the backend faulted and require reconstruction. Do not continue stepping or publish a partially written state as successful.

The request types provide validation of values that can be checked without an engine. They do not implement GPU atomicity, rollback, a native transaction or a backend fault state by themselves. Adapter execution must deliver and test those guarantees.

`SimBackend` declares `get_entity_names()`, `get_entity_state(entity)` and `reset_entities(request)`. Their base implementations explicitly raise `NotImplementedError`. Implemented state reads must return detached arrays under the root/joint field names above, in frozen entity joint order, with freshness determined by the declared profile; unavailable state must fail rather than masquerade as current data. These methods establish the future adapter interface without asserting that its execution is implemented.

## Compiled layout and migration

`SimBackend.get_entity_default_state(entity, env_ids=None)` supplies the same root/joint fields as the live entity query, using construction/keyframe defaults for each environment's immutable variant identity. Returned arrays are detached and preserve the requested environment order; unknown entities and invalid/duplicate/out-of-range IDs fail. Querying defaults never steps, resets or reparses sources, and subsequent DR or state writes cannot change the result. This public boundary lets the UniLab reset owner preserve entity-local defaults without reading private adapter arrays or broadcasting environment zero.

Cold-path materialization must freeze entity-qualified public names and public-to-native root/body/joint/actuator mappings. State includes passive joints; actions include only declared actuators. Native actor/prim/body/DoF indices are learned from the actual scene, never inferred from environment IDs or creation order. Internal padding must not leak as additional public controls. Step, reset and queries use bound arrays and handles without parsing source assets.

`model_file` and nonempty `entity_assets` are mutually exclusive. Whole-model `fixed_variant_plan` cannot accompany `entity_assets`; `entity_variant` requires them. Existing `model_file`/whole-model variant callers retain their meaning. **One model file is not necessarily one articulation:** it may contain several independent roots. A later normalization layer must inspect the actual compiled partition and preserve existing supported behavior instead of assuming one root per file. Single-entity and multi-entity execution should converge on one mapped runtime; this decision does not introduce a second permanent compatibility runtime.

The declaration gate validates again at consumption because `SceneCfg` remains mutable. Adapter capability declarations determine whether an entity profile can materialize. The public vocabulary alone does not mark `entity.multiple` or related variant/reset operations as implemented or runtime verified.

Adapter implementation evidence must connect declaration identity to the actual native instance/view and its parameters or physical response. Reusing the same incorrect mapping for both write and read is not a valid independent oracle. Relevant coverage includes reordered environment IDs, creation-order permutations, passive joints without extra actions, nonidentity orientation and COM offsets, reset preservation, mirror/environment isolation, immutable assignment and compiled-parameter readback. Contract tests, mocks, skipped tests and SDK availability do not verify a composition runtime.

## Alternatives and consequences

| Alternative | Decision and reason |
| --- | --- |
| Duplicate a complete scene for each tool variant | Keep entity-scoped authoring; backend-internal realizations may still reuse existing executors without burdening callers with duplicated robot/table assets. |
| A global plan plus consumer flags on entities | Use one explicit binding so the target and plan cannot diverge across parallel declaration fields. |
| Accept arbitrary heterogeneous topology immediately | Start with validated public-layout guarantees; variable action/state shapes and native restrictions need separately tested support. |
| Copy one prototype implementation wholesale as the public contract | Reuse bounded adapter mechanisms; avoid making PhysX/importer details and task-parity settings universal entity semantics. |
| Treat reset as several independent writes | Validate the complete request before commit and fault on unrecoverable partial native failure, making preservation and failure behavior reviewable. |
| Declare all backends supported once they reject invalid input | Keep declaration acceptance, actual implementation and runtime evidence distinct. |

This adds a public declaration surface and obligations for each implementing adapter. Unsupported combinations remain unavailable until materialization, mapping, execution and evidence are implemented and tested. The contract does not add a generic asset IR, dynamic topology replacement, engine dependencies, DR provider lifecycle or routine nine-engine GPU CI.
