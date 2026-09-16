# M2 IsaacSim mapped scene implementation

[English](m2-isaacsim.md) | [中文](../zh/m2-isaacsim.md)

This page records the worker implementation and bounded native evidence for [M2 #108](https://github.com/unilabsim/unisim/issues/108), work package [D #111](https://github.com/unilabsim/unisim/issues/111). It is not a claim that D or the final integration audit F is complete. The public host scene path has not yet been connected and accepted with this worker. The mapped renderer reuses the existing renderer helpers but has not received native acceptance.

## Implemented profile

The implementation in `backend/isaacsim/scene_worker.py` targets the dedicated Isaac Sim 5.1.0 / IsaacLab 0.47.2 Python 3.11 CUDA runtime. Each named entity owns an IsaacLab `Articulation` or `RigidObject` view. Supported requests use standalone MJCF sources, scalar hinge/slide joints, the frozen public scene layout, and position actuators with one actuator per controlled joint. Passive joints retain state columns without adding control columns. Fixed and floating articulations, one-body floating rigid objects, fixed one-body tables, and collision-disabled kinematic one-body mirrors are implemented. No implicit ground is added.

Entity variants use K converted USD prototypes and `MultiUsdFileCfg` with deterministic round-robin assignment. Each entity is spawned independently; mirrors share source identity while retaining independent poses. The present profile requires the same public layout and identical drive properties across an entity's variants. Distinct source masses and inertias are preserved and checked against the native runtime.

Preflight rejects URDF/USD/SuperDex source declarations, non-round-robin assignment, ball joints, kinematic articulations, multiple controls for one joint, passive stiffness, variant-dependent drive settings, and rigid entities containing multiple physical bodies. In particular, an articulated target's mirror is not supported when its frozen geometry still has multiple bodies. These are explicit profile limits, not evidence of support for every entity or asset type. Source features such as passive joint damping also require the host's source-semantic validation; this worker does not infer missing wire semantics. Contact-force slots remain deterministic zeros, without a contact sensor support claim.

## Cold construction and native identity

INIT validates the complete layout, declarations, scalar joint records, assignment, dimensions and initial arrays before launching Kit. Conversion uses explicit source inertials and a private temporary USD directory. USD edits set declared collision and root behavior and clear importer-created drives before applying the declared IsaacLab drive configuration. Optional `initial_ctrl` has actuator width `(N, nu)` and initializes native targets after the initial state write. Keyframe controls are independent of joint positions and are not reconstructed from them.

Fixed roots need two distinct repairs. Imported world joints must anchor to the actual cloned root world transform, including the environment offset and initial pose. Also, `ArticulationRootAPI` must move from the rigid root link to the encompassing asset prim. Otherwise PhysX can report a floating articulation constrained by an external joint even while its pose appears fixed. The worker checks native `is_fixed_base` rather than inferring this from pose stability. For a rigid entity, unwanted importer world joints are made inactive. Removing a prim from only the edit layer can reveal the same joint from a referenced layer and is insufficient.

Native view prim paths establish an explicit public-environment-to-view-row mapping. Native body/joint names establish separate column mappings. Reads, controls, resets and metadata all use these maps; they do not assume creation order or lexicographic prim order equals public environment order.

Each converted source authors an immutable variant marker. The worker reads that marker from each actual spawned prim and compares the observed assignment with the request. It separately reads mass, COM and inertia tensors from the actual PhysX view, reorders them by the native maps, and compares them with independently compiled source records. Articulation root mode, joint types, parent topology and drive gains are also checked. Returning the requested assignment alone is not treated as native identity evidence.

## State and reset behavior

The worker publishes `(N, nq)`, `(N, nv)`, `(N, nu)` and `(N, E, 13)` scene buffers. Floating generalized angular velocity is body-frame; entity root velocity is world-frame at the root link origin. Native clone offsets are removed on reads and added on writes. Explicit IsaacLab link-velocity writers handle the COM offset; a COM velocity API is not substituted for them.

`RESET_ENTITIES` copies selected rows and masks before validation. It rejects duplicate/out-of-range environments, undeclared entity writes, invalid masks, nonfinite values, invalid root quaternions, root-mode violations, and inconsistent generalized/root values. All validation completes before native writes begin. Joint reset preserves each unselected position/velocity channel using current native state. Only touched entities and selected native environment rows are reset. Failure after native submission begins, including refresh failure, marks the worker faulted; the shared host must refuse reuse. No native rollback is claimed.

## Reproducible bounded verification

The normal CPU check imports no Isaac SDK and exercises profile rejection, mask/selector validation, independent ninety-degree rotation expectations, reordered native environment rows, preservation of omitted joint channels, and faulting after a failed post-write refresh:

```bash
uv run --no-sync pytest -q tests/adapters/isaacsim/test_mapped_scene.py
```

The opt-in native test starts the production worker with the real shared-memory protocol. Provision the dedicated SDK using the existing runtime instructions and install the normal MuJoCo development extra for the independent source oracle. No SDK download occurs in the test:

```bash
UNISIM_TEST_ISAACSIM_SCENE=1 uv run --no-sync pytest -q \
  tests/adapters/isaacsim/test_scene_native.py \
  --basetemp=/tmp/unisim-isaacsim-scene-acceptance
```

This runs three sequential N=2 scenes: floating controlled robot, fixed passive articulation, and floating passive articulation. Each includes a two-mass rigid variant pool, fixed table and visual mirror. Assertions cover actual instance identity and source properties, `nu` excluding the passive joint, gravity, selected-entity/environment reset isolation, persistence after a subsequent step, nonidentity orientation with offset COM, controlled joint response, and equal object trajectories with the mirror far away versus co-located. INIT has a 240-second deadline, commands a 60-second deadline, and teardown terminates an unresponsive worker. Each case records `command.json`, `init.json`, `meta.json`, `result.json`, and `stderr.log` in its pytest temporary directory. The test skips by default and is not an ordinary CI GPU requirement.

During implementation on 2026-09-17, equivalent direct-worker probes passed on Isaac Sim 5.1.0 / IsaacLab 0.47.2 with the local CUDA runtime. The last probe included the native environment-row mapper and floating passive articulation; earlier probes covered fixed passive and floating controlled articulations. The repository test is the maintained reproduction of those probes and still needs execution on the integrated final head. These checks construct an independent test INIT payload: they are worker acceptance, not evidence for public host integration, renderer support, arbitrary assignment, every source format, all rigid topologies, or completion of D/F.
