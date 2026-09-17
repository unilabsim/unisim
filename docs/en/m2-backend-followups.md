# M2 assessment of the remaining adapters

[English](m2-backend-followups.md) | [中文](../zh/m2-backend-followups.md)

This records work package E's source assessment under #108/#112. The roadmap requires real implementations on MuJoCo, MJWarp, IsaacGym and IsaacSim; it requires a concrete assessment and follow-up for the other five adapters. The proposals below do not expand that implementation commitment or count explicit rejection as delivered support. No native engine test was run for this assessment.

## Evidence and next actions

| Adapter / follow-up | Source evidence | First implementation to validate | Unresolved boundary |
| --- | --- | --- | --- |
| Genesis 1.3.3 — [#120](https://github.com/unilabsim/unisim/issues/120) | Public `Scene.add_entity(morph=[...])` exists; heterogeneous loader checks joint names/types/DoFs and variant inertials. UniSim currently binds one entity. | Independent entity maps plus native same-layout heterogeneous targets; preserve passive joints without action columns. | Native assignment uses balanced blocks, e.g. N5/K2 `[0,0,0,1,1]`; arbitrary assignment needs an upstream public API. Multi-link variant support requires pinned runtime evidence despite broader loader code. |
| Motrix 0.8.2 — [#121](https://github.com/unilabsim/unisim/issues/121) | Shipped stubs expose `World.attach`, batched SceneData and per-data Link mass/COM overrides. UniSim selected reset currently resets the whole chosen world. | Compose independent named roots, freeze all quaternion/state mappings, and implement masked entity writes without whole-world reset. | Per-env geometry and coherent inertia variation remain unverified; mass/COM overrides alone are not geometric variants. Coordinate with the SDK owner. |
| Drake / drake-uni — [#122](https://github.com/unilabsim/unisim/issues/122) | Runtime accepts one model path; inspected C++ requires one parser model instance but already stores multiple free-joint mappings. | First test one composed MJCF model with multiple independent roots and passive joints, then bind UniSim entity addresses. | One model instance does not imply one root. Geometry variants and multiple model instances may need an explicit upstream runtime contract, not private plant access. |
| Newton 1.5.1 — [#123](https://github.com/unilabsim/unisim/issues/123) | `ModelBuilder.add_builder` and world construction can compose entities; current adapter replicates one template and assumes a leading free root. | Compose per-world sub-builders, audit body/root/joint/actuator maps, support fixed and multiple free roots before declaring variants. | SolverMuJoCo conversion, ArticulationView and differing per-world geometry require real runtime evidence. Replication alone does not establish heterogeneous support. |
| SuperDex — [#124](https://github.com/unilabsim/unisim/issues/124) | Current cold ModelPlan and batch executor bind one articulated actor per world; MJCF audit rejects multiple free roots. | Extend audited plans and public executor metadata for multiple actors; use serial native reference before batch parity. | Installed `superdex-*-uni` extended batch ABI must be tied to an exact source/build. Public project main alone does not prove that distributed executor's multi-actor capabilities. |

## Source provenance

The UniSim adapter assessment inspected development baseline `0189060b7eb9cf441bf43a818102856bbb3013c9`, with main at `dd8984c28a5042d267600c95cb403334f112f193`. Subsequent core adapter integration does not establish support in these five owner modules. Each linked issue includes specific UniSim paths, upstream owner interfaces and proposed acceptance.

Genesis upstream v1.3.3 was pinned to `76f8f5b3457e7c6d6a078de2244066f9a8694c45`; the inspected heterogeneous loader and tests are source evidence, not a run result. Drake runtime source was inspected at `4cdc9ba4c9b1a7542755631afe0d57dbb54cdb63`; parity with the installed release still needs verification. Motrix evidence comes from installed 0.8.2 public stubs. SuperDex public source was inspected at `b717b1ccf8a9312ac63e709bd0bead32f37fdd6f`; the extended executor distribution provenance remains an explicit dependency.

## Ownership and acceptance

Each follow-up belongs to the UniSim adapter owner, with missing public runtime interfaces owned upstream. UniLab consumes public state/scene contracts and must not wire engine-private actors. Before implementation, confirm the PR base and declared support combinations; new upstream protocols or durable support/CI commitments require an explicit issue/ADR decision.

Acceptance must establish actual instance identity, source/native inertia and geometry, state/action layout, passive articulation, selected-row/entity preservation, reset persistence across a later step, mirror isolation, and complete playback. Use exact runtime/version/device/command evidence. A same-layout or constrained-assignment profile may refuse other combinations honestly; no profile becomes supported solely because a constructor guard or mock test passes.

Genesis native probes should use isolated processes with device visibility fixed before engine initialization. The repository's stub device tests now restore their original environment, so they cannot silently hide GPU 0 and turn later CUDA tests into skips. This test-isolation repair does not itself validate Genesis multi-entity execution.
