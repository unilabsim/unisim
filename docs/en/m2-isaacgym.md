# M2 IsaacGym mapped scene worker

[中文](../zh/m2-isaacgym.md)

This documents the native worker of [#108](https://github.com/unilabsim/unisim/issues/108), work package C. The [shared host](m2-worker-host.md) connects the public entity factory to this runtime; old model-file input adopts its native instances into the same executor through a compatibility projection. Worker and public-factory tests provide separate evidence; the complete roadmap still requires its final integration gates.

## Supported profile and boundaries

- One self-contained MJCF source per entity or fixed variant. The native importer sees an explicit root mode and importer-safe XML; materialization remains a host cold-path responsibility.
- Fixed/floating articulations with hinge/slide joints, floating rigid bodies, fixed objects and kinematic rigid visual mirrors. No ball-joint support is advertised by this profile.
- Immutable arbitrary assignment, including `[1, 1, 0, 1, 0]`, chooses one of K loaded assets per consuming entity. Mirrors must use the same assignment.
- Position drives have one declared actuator per controlled joint. Passive joints have `DOF_MODE_NONE`, zero drive gains/effort and no action columns. Nonzero source joint passive damping is rejected until separately validated.
- Up to 30 entities use distinct physical collision-filter bits. Cross-entity collisions remain enabled; visual actors share all physical filter bits and cannot collide with them. Self-collision within each physical entity is disabled, an explicit profile approximation rather than general MJCF collision parity.
- No implicit ground is added. The scene must declare its physical ground/table. General contact-pair queries, runtime DR and wrench APIs are outside this slice.
- Articulated visual sources require a host-produced rigid visual bake; the worker does not remove joints or infer mirror geometry itself. The host retains full-scene playback sources, and old-wire compatibility shares the same native execution path.

MuJoCo canonical XML can contain `<general>` actuators even when the source used `<position>`. This Gym importer can loop indefinitely on unsupported actuator tags. The worker rejects those tags before loading the SDK asset. Host staging should remove imported actuator elements and supply the explicit drive records. Limited joints must carry explicit `limited="true"` and compiler-resolved ranges; native `hasLimits/lower/upper` readback is audited. Native mass, COM and inertia must match the staged compiler records; raw `geom mass` alone was observed to be ignored by the importer in a preliminary probe.

## Wire contract and native binding

INIT carries `scene_layout` schema 1, ordered `scene_entities`, complete initial `qpos/qvel/entity_root_state`, and gravity. Each entity supplies its name, kind, root mode, format, collision/mirror declaration, sources, assignment and per-source body/drive records. The worker uses the common `scene_layout.py` validator without importing the host package. Shared-memory descriptors must exactly match `protocol.scene_slot_shapes()` before any segment is attached.

The worker queries actual actor/body/DoF indices. A source's native name sets and joint types are checked, then remapped into the public layout. Actor asset handles are queried again after creation to derive observed variant identity. Native mass, COM, inertia and drive properties are read back and audited; META returns `scene_entities_actual` with observed assignments, source paths, actor IDs and physical evidence. Matching a requested assignment alone is insufficient.

Public root/body poses describe the link origin, use `wxyz`, and omit environment clone offsets. The tested Gym tensor API already returns environment-local positions. Native linear velocity refers to the COM, so readback applies `v_link = v_com - omega × R(q) * com_local`; writes apply the inverse relation. Root angular velocity in entity state is world-frame; generalized root qvel uses body-frame angular velocity. Source COM offsets are included in both conversions.

`RESET_ENTITIES` sends selected row count/entity names plus explicit root/qpos/qvel write masks. Validation finishes before native submission. Actor root setters and DoF setters use queried global actor IDs, not environment IDs. Writes pending since the last physics step are unioned: repeated Gym indexed setter calls can otherwise discard earlier disjoint resets. The union is cleared only after a successful simulation step. A native submission failure faults the worker.

Root/joint state is fresh after reset. Articulation descendant-body kinematics can remain at the previous solved state until STEP; the host must advertise and enforce that boundary. A body-cache refresh is not a kinematics-only forward.

## Validation evidence

The following bounded native run completed during development on **2026-09-17**:

```sh
UNISIM_TEST_ISAACGYM_SCENE=1 uv run --extra mujoco pytest -q \
  tests/adapters/isaacgym/test_scene_native.py \
  --basetemp=/tmp/unisim-m2-gym-native-finalpass -x
```

All three acceptance tests passed in **6.35 s**, using four sequential worker processes. Runtime: IsaacGym Preview 4 `gym_38`, Python 3.8.20, Torch 2.4.1+cu121, PhysX GPU pipeline, NVIDIA GeForce RTX 4090, driver 595.84. The working tree was based on `df31d60bc7210fca551349eefeccd26ee65ff337`; this is development-tree evidence, not a claim of final-PR-head validation.

| Check | Observed result |
| --- | --- |
| N=5, K=2, assignment `[1,1,0,1,0]` | Native object root masses `[3,3,1,3,1]` kg |
| Fixed robot + passive floating articulation + table + mirror | One action column; passive joint moved after reset |
| Root pose/velocity and nonzero COM | 90° orientation and independent position finite-difference oracle passed; velocity tolerance 0.02 m/s |
| Three disjoint resets before STEP | Both object resets and mirror reset persisted after physics; other rows unchanged before STEP |
| Mirror intersecting fall path versus moved aside | 480 steps; maximum pose trajectory difference `0.0` |
| Kinematic rigid-only scene, nq=nv=nu=0 | Selected pose reset and STEP passed |

The repository includes the opt-in native tests and CPU regression tests. The native tests write `evidence.json` and worker logs under the chosen pytest temporary directory. Those `/tmp` files are session artifacts, not durable public evidence; the table above preserves the concise observations. Final integration must rerun the exact final head, archive its versioned evidence, cover the public host path, and follow the [#108](https://github.com/unilabsim/unisim/issues/108) completion gates. Missing vendor runtimes must remain explicit skips, never equivalent to native acceptance.
