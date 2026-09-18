# Adapter Support Matrix

[English](support-matrix.md) | [中文](../zh/support-matrix.md)

| Backend | Public class | Install and runtime boundary | Status |
| --- | --- | --- | --- |
| MuJoCo | `unisim.MuJoCoBackend` | `uv sync --extra mujoco` (mjbatch runtime) | available |
| Motrix | `unisim.MotrixBackend` | `uv sync --extra motrix` | available |
| Drake | `unisim.DrakeBackend` | `uv sync --extra drake` (`drake-uni`) plus its native batch extension | available |
| MJWarp | `unisim.MJWarpBackend` | `uv sync --extra mjwarp`, CUDA | available |
| Genesis | `unisim.GenesisBackend` | `uv sync --extra genesis` (`genesis-world==1.3.3`) | available (native CPU evidence) |
| Newton | `unisim.NewtonBackend` | `uv sync --extra newton`, Newton 1.5.1 and MuJoCo-Warp 3.11.0 | available (CUDA) |
| SuperDex | `unisim.SuperDexBackend` | `uv sync --extra superdex`, CPython 3.12 or 3.13, SuperDex 1.0.0 | experimental CPU; see the [profile](superdex.md) |
| IsaacGym | `unisim.IsaacGymBackend` | `uv sync --extra isaacgym` (empty extra) plus a dedicated Python 3.8 worker | available |
| IsaacSim | `unisim.IsaacSimBackend` | `uv sync --extra isaacsim` (empty extra) plus a dedicated IsaacSim or IsaacLab worker | available |

The base wheel imports none of these SDKs. Construction performs cold-path runtime discovery and raises an adapter-specific, actionable error when a runtime is unavailable. This matrix is an adapter and API support statement, not a claim that every host has every vendor SDK or GPU capability.

The SDK-free portable MJCF compiler contract is available in the base import, but actual cold-path compilation lazily requires `unisim-core[scene-compiler]` (`mujoco~=3.11.0`, without the mjbatch executor). The compiler's source/intent report and content identity do not themselves declare native adapter support; every adapter still needs its own materialization and readback evidence. The governing boundary is the [portable MJCF ADR](adr-portable-mjcf.md).

IsaacSim's raw- and role-derived-USD caches are cold materialization optimizations only. They do not cache native scenes, views, parameters, or effective reports; hits still materialize and read back from cached USD. See [entity scene execution](entity-scenes.md) for cache roots, environment overrides, identity inputs, role validation, and atomic publication behavior.

Mapped IsaacSim scenes expose frozen public geometry names, owning body IDs, normalized native collider masks, and per-environment current PhysX friction materials. Selected mapped `set_state()` rows support positive `body_mass` and Coulomb `geom_friction` writes with native current readback; gravity, COM/inertia, joint/actuator parameters, geometry dimensions and other contact parameters fail closed. Legacy model-file scenes fail closed. See [entity scene execution](entity-scenes.md) for the source-intent/materialization/current-value boundary.

The MuJoCo adapter's native executor is [mjbatch](https://github.com/unilabsim/mjbatch_uni), a maintained fork of `kevinzakka/mjbatch` with prebuilt wheels for Linux x86_64 and aarch64 plus macOS (CPython 3.10 through 3.14t) and an exact `mujoco==3.11.0` pin. Windows and musllinux are unsupported for the native executor, so the Windows CI job runs only the core and import-boundary subset. Numerical results before and after the switch from the previous executor are not guaranteed identical; drift is characterized by a recorded baseline rather than gated bit-exactly. The adapter supports construction-time `FixedVariantPlan` catalogs with `same_layout` and `uniform_public_layout` guarantees. Same-layout variants and optional named mesh-geom slots are merged into one canonical mjbatch executor through `VariantPack`; heterogeneous public topology fails closed. Reset model-field writes and per-world compiler defaults use mjbatch `expand` and `set_const`, and playback exposes a per-env independently compiled visual oracle. `chunk_size` and `adaptive_chunk_size` are deprecated warn-and-ignore knobs; the chunk scheduler was removed, and mjbatch's work-stealing thread pool is the tuning mechanism.

The MuJoCo-related extras share one version line (MuJoCo 3.11, MuJoCo-Warp 3.11, and warp-lang 1.16.0) and are jointly installable. `mjwarp` tracks the line with `mujoco-warp~=3.11.0`, while `newton` keeps exact upstream-coupled pins (`newton==1.5.1`, `mujoco-warp==3.11.0`, `mujoco==3.11.0`, and `warp-lang==1.16.0`). After installation, run `uv run scripts/diagnostics/check_newton_runtime.py` for a metadata-only probe; add `--import` when the native runtime should be imported explicitly. Newton's cold-path calibration samples solver counts and raises an explicit capacity error when `nconmax` or `njmax` is too small; it never accepts silent constraint truncation.

Newton's portable entity profile is bounded: it materializes independent same-layout variant builders into explicit worlds and binds one public articulation view per physical entity. It covers fixed and floating roots, passive/static entities, same-shape-type heterogeneous identity and force response, selected state reset, named found contacts with per-world attribution, and per-variant playback. Selected controls are cleared while unrelated controls are preserved; `restore_default_controls` and keyframe control restoration fail closed. Kinematic mirrors and mixed shape-type assignments fail closed. See [entity scene execution](entity-scenes.md) for the audit and native validation boundary.

Genesis' portable entity profile is a bounded MJCF subset: independent fixed/floating/passive/static entities bind by audited public names, inertia and addresses, and only single-link rigid heterogeneous assignments equal to Genesis' native balanced mapping are accepted. Public geometry exposes cold-audited names, IDs and body ownership plus audited uniform Genesis-native sphere/box sizes, collision masks, friction coefficients and solver parameters in frozen public order; public DOF damping/friction-loss/armature readback uses cold-captured native values, audited qvel addresses and uniform active-row/fixed-variant values. Entity-owned unreferenced site pose and motion sensors plus scene-level world-referenced qualified-site pose fragments require identical complete sensor identity across fixed variants and return wxyz site quaternions; identity-orientation accelerometers use clean public native IMUs and return completed-step proper linear acceleration. Scene-level cross-entity geom-pair found/netforce fragments route exact native collision identities by assignment, expose completed-step public-contact flags or three-vector forces on authored geom1, and clear selected reset rows until the next step. Selected state/reset rows preserve unrelated entities and environments; selected-row body-mass/base-mass-delta and actuator kp/kd randomization is prevalidated in public columns and submitted through audited owning entities. Absent or ambiguous collision identity, source contact sensors, same-entity pairs, non-uniform variant sizes/masks/friction/solver/DOF values, mirrors, kinematic entities, rotated accelerometers, other source sensor forms, body fragments, other contact forms, other reset randomization, body-force mapping and control restoration fail closed. Native CPU evidence uses Genesis 1.3.3, Torch 2.14.0+cpu and Quadrants 1.3.0; no GPU capability is claimed. See [entity scene execution](entity-scenes.md) for the exact variant and validation boundary.

Motrix's portable entity profile is a bounded MJCF subset covering fixed/floating/passive/static entities and immutable same-layout fixed variants. Each used variant owns one native Motrix model/data context; public state and controls are explicitly scattered and gathered across those contexts. Native layout, per-row mass/COM, actual geometry sizes and variant control identity are audited, while effective inertia is validated through native response because Motrix does not expose inertia readback. World-frame body-force/torque submissions map through audited public body IDs and public native Link APIs, accumulate for the upcoming step, and reset cancellation is scoped to impacted bodies. Selected entity resets preserve unrelated state and controls; selected control restoration and full default reset use cold-captured native construction/default-keyframe controls plus the selected keyframe qpos/qvel. Entity-owned plus scene-level fragment world-referenced site pose sensors and qualified named-site world Jacobians gather through audited variant contexts by assignment. Non-uniform public control parameters or geometry-size reads, missing native wrench APIs, actuator activation state, kinematic mirrors, other site-sensor forms, terrain, reset randomization and fixed-variant native playback fail closed. Native evidence uses MotrixSim Core 0.8.2 with MuJoCo 3.11.0; no claim beyond the documented CPU-profile acceptance is made. See [entity scene execution](entity-scenes.md).

The Motrix profile also accepts entity-owned, unreferenced site `velocimeter` and `gyro` declarations. Their native `FrameLinVel(local)` and `FrameAngVel(local)` values are gathered through the audited variant contexts; accelerometers, body motion targets, referenced site motion and scene-level motion fragments fail closed.

Newton supports opt-in CUDA graphs with `NewtonBackend(..., use_cuda_graph=True)` or `create_backend(..., newton_use_cuda_graph=True)`. Graphs are captured only after cold-path capacity calibration rebuilds the final fixed-address state, using one graph for each Newton input/output state parity. Capture requires a CUDA device, driver 12.4 or newer, and an enabled CUDA mempool; otherwise Newton emits a `RuntimeWarning` with the reason and keeps eager execution. Capture failure also falls back eagerly. State reset and registered pre-step control callbacks remain eager; callback-free physics steps replay the parity-selected graph.

Newton playback renders natively through `ViewerGL` (`pyglet>=2.1.6,<3` and `imgui-bundle>=1.92.0`) when installed with the single `newton` extra: `record` renders offscreen, `interactive` opens the windowed viewer, and `auto` chooses based on display availability. If the runtime is incomplete, `record` falls back to the offline MuJoCo snapshot pipeline and `interactive` fails closed with an actionable error. Headless offscreen GL needs EGL (`PYOPENGL_PLATFORM=egl`) or GLX under Wayland.

## Semantic inventory

The following table is generated from `get_adapter_capabilities()` in `src/unisim/support.py`; run `uv run scripts/diagnostics/check_support.py --check-docs` to check it or `--write-docs` to regenerate both languages. These are source-reviewed declarations, not runtime verification. `exact` applies only to the documented subset; `approximate` requires specific consent, `unsupported` rejects the named request, and `unknown` has no support guarantee. A `*` requires the declaration's configuration conditions; query the public report for the reason, conditions and pinned source evidence. The default profile is the only declared profile; unknown profiles stay unknown.

<!-- semantic-inventory:start -->
| Feature | mujoco | motrix | drake | mjwarp | newton | superdex | genesis | isaacgym | isaacsim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `asset.mjcf` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `asset.urdf` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unsupported | unsupported |
| `entity.single_articulation` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `entity.multiple` | exact* | exact* | exact* | exact* | exact* | unsupported* | exact* | exact* | exact* |
| `root.free` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `root.fixed` | exact | exact | exact | exact | exact* | exact | exact* | unknown | unknown |
| `joint.hinge` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.slide` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `joint.ball` | exact | unknown | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `actuator.motor` | exact | unknown | unknown | exact | exact | exact | unsupported | unsupported | unsupported |
| `actuator.position` | exact | exact | unknown | exact | unknown | unknown | exact | exact | exact |
| `collision.rigid` | exact | exact | exact | exact | exact | approximate* | exact | exact | exact |
| `collision.self` | exact | unknown | unknown | exact | unknown | unknown | unknown | unsupported | unsupported |
| `contact.query` | exact | unknown | unknown | exact | exact* | approximate* | approximate* | approximate* | approximate* |
| `terrain.heightfield` | exact | exact | unknown | exact | unknown | unsupported | unknown | unknown | unknown |
| `sensor.imu` | exact | unknown | unknown | exact | approximate | approximate | approximate | unsupported | unsupported |
| `sensor.gyro` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | approximate | approximate |
| `reset.state` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `dr.interval.body_force` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| `state.final_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | unknown |
| `state.callback_refresh` | exact* | unknown | unknown | exact | unknown | unknown | unknown | unknown | exact* |
| `variant.same_layout` | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
<!-- semantic-inventory:end -->

DR, playback, body-wrench and fixed-variant capability sources remain their existing instance APIs. The static inventory intentionally leaves those dependent entries unknown; `backend.get_capabilities()` aggregates authoritative instance declarations. Multiple logical entity partitions do not imply arbitrary multi-articulation composition. URDF investigations and unmerged branches are not current support. IsaacSim's legacy reserved zero contact buffer never means a valid contact query or absence of physical contact; only mapped `contact data="force" reduce="netforce"` declarations use the dedicated PhysX pair-force slot. Isaac worker sensors support gyro reconstruction but reject accelerometers.

The [capability design decision](adr-capabilities.md) defines evidence matching and snapshot lifetime. `available` in the installation table above is never a task compatibility decision.
