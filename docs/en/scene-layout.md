# Scene layout and reset mapping

[English](scene-layout.md) | [中文](../zh/scene-layout.md)

The [entity decision](adr-entities.md) defines the authoring and coordinate contract. Layout types and protocol helpers do not themselves enable an adapter's multi-entity runtime.

## Public and native addresses

`CompiledSceneLayout` contains `EntityLayout` records with entity-local body/parent names, absolute public body IDs, root qpos/qvel columns, non-root `JointLayout` records, actuator names/targets/control columns, and entity-local geometry records. Floating roots occupy seven position and six velocity columns; fixed/kinematic roots occupy none. Ball joints have four position and three velocity columns. Passive joints have state columns but no implicit actuator entry.

Generalized position, velocity and control columns must each be covered exactly once across the scene, without gaps, duplicates or out-of-range values. Body IDs are unique and bounded; unowned native world bodies may remain outside the entities. Native actor handles, tensor offsets and asset identities are kept separately by the adapter. Entity body order need not be topological and addresses need not be contiguous.

Qualified body, joint, actuator, and geometry lookup requires `entity/local_name` or an explicit `entity` argument. `require_same_layout` checks complete names, parent topology, joint kinds, root modes, actuator targets, geometry ownership, ordering and addresses rather than only dimensions. Adapter-specific remapping may first normalize native data into this public order; it must not pretend two different public signatures are equivalent.

## Reset preparation

`layout.validate_reset(request, num_envs=...)` validates all patches and returns a complete `BoundSceneReset`. It never yields a partial plan or writes state. Root-mode permissions, joint selection, packed widths and ball quaternions are checked before any caller can submit a native write. Fixed roots reject pose/velocity writes; kinematic roots allow pose only.

`prepare_scene_reset` consumes coherent generalized/root snapshots and builds owned selected rows and explicit write masks. Missing fields preserve current values. Floating root angular velocity is converted between world-frame public values and body-frame generalized values; a pose-only update preserves world angular velocity by adjusting its generalized representation. Kinematic poses remain separate from generalized state. No input snapshot is mutated, including on validation failure.

The adapter still owns native index mapping, selected control/wrench cleanup, refresh and fault handling. These helpers do not implement native rollback. `SimBackend.get_scene_layout()` fails explicitly until an adapter publishes a materialized layout.

## Worker boundary

Scene wire schema version 2 is independent of the configuration-report version. `to_dict`/`from_dict` enforce an exact field set at each nesting level, version and full layout validity. The schema freezes geometry name/body ownership and the total geometry count; it does not freeze primitive dimensions or contact parameters. The same module loads by file path in a Python 3.8 worker using only standard-library and NumPy dependencies; host reset request types are imported only when the host validates a request.

Mapped slots separate `(N, nq)`, `(N, nv)`, `(N, nu)` and `(N, E, 13)` entity roots. Reset masks identify position/velocity/root channels independently. Workers validate every slot name, shape and dtype before attaching any memory. A zero-width state/action slot keeps zero public elements while allocating the minimal nonzero shared-memory backing required by the operating system. Existing worker slots retain their current wire shape until their execution paths are migrated; this does not create a second permanent scene runtime.

## Verification

Contract tests exercise independent roots, passive joints, non-contiguous indices, topology and wire tampering, selected reset validation and detached snapshots. Rotation expectations use explicit nonidentity poses and independent numeric vectors. The actual IsaacGym Python 3.8 interpreter also loads the shared validator without importing UniSim. Native scene execution, instance identity, reset isolation and physical behavior still require adapter-specific native acceptance; these tests cannot substitute for them.
