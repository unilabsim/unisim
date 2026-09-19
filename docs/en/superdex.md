# SuperDex CPU Development Profile

[English](superdex.md) | [中文](../zh/superdex.md)

The `superdex` adapter runs SuperDex Physics and Robotics 1.3.0 directly behind `SimBackend`. Its development profile remains adapter-owned; the package version is unchanged and no additional UniSim release is required for local integration.

## Installation and ownership

Use CPython 3.12 or 3.13, as covered by the `superdex-uni` wheels. From the UniSim checkout, synchronize with both optional extras:

```sh
uv sync --python 3.12 --extra superdex --extra mujoco
```

`superdex-physics-uni==1.3.0` and `superdex-robotics-uni==1.3.0` are optional. They are a temporary unilabsim build of the SuperDex facades, published from [unilabsim/superdex-uni](https://github.com/unilabsim/superdex-uni) tag `v1.3.0` (packaging commit `f4fd2abdcdf71a316a0368f2a13aca95bbd9de66`) until upstream SuperDex publishes equivalent official wheels. That tag pins the public `project_superdex` executor contract at `0ed957041092a30bb60d804e31591871741cf7e4`; wheel metadata records the same source provenance. They install into the same `superdex/` namespace as the upstream packages and must not be co-installed with them. The extra also supplies MuJoCo 3.11 as a cold MJCF parser; SuperDex executes every physics step, and native `.superdex_bot` loading does not use that parser. Importing `unisim` or its `SuperDexBackend` class loads neither engine. SuperDex Lab, Gymnasium, and a learner are not adapter dependencies.

For a sibling UniLab checkout, keep both versions unchanged and install the local editable projects together, for example `uv pip install -e './[superdex,mujoco]' -e ../UniLab`. Use `uv run --no-sync` (or `UV_NO_SYNC=1 make check`) while testing editable overrides so normal project synchronization does not replace them with index distributions. UniLab's local-provenance test profile uses `UNILAB_LOCAL_UNISIM` pointing at the exact UniSim checkout. The UniLab backend guide describes its task and registered-asset setup.

The verified platform is Linux x86_64 with CPU FP32. Upstream also provides Windows x86_64 and macOS ARM wheels, but this integration has not established those platforms. The default x86 build requires AVX2 and related instructions. The upstream source exposes optional CUDA linear solvers, but the tested wheel reports that they were not built with CUDA; this adapter does not enable GPU solvers. FP64 upstream packages require a process-wide precision choice before import, and the integration's numerical validation currently targets FP32.

Each environment owns an independent native scene. The adapter reference-counts the process-global engine, so closing one instance leaves other instances alive. The source-built SuperDex `SceneBatchExecutor` batches force writes, stepping, articulated state, link state, contact sensors, and solver status in persistent C++ workers. `superdex_num_workers=0` uses the physical cores visible to the process (Linux topology or macOS `sysctl`) and disables SDK-internal workers. Runtime initialization belongs to UniSim, and live backends cannot be transferred between processes. Call the public `cleanup_scene_assets()` hook or `close()` before interpreter shutdown; UniLab's `env.close()` calls that public hook.

Portable entity scenes without physical kinematic roots use `SceneBatchExecutorV2`, supplied by the `superdex-uni` 1.3.0 wheels. Its public constructor exposes actor-slot DoF, link, and actuator offsets; selective state writes; and closed-worker failure semantics. A scene with a physical kinematic root additionally requires `SceneBatchExecutorV3`, the wheel's ABI marker 3, and its selective `write_boundary_conditions` method; absent class, ABI, or method evidence fails closed. UniSim binds exactly one native actor slot per public physical entity in the same per-environment scene. Whole-MJCF and native-bot paths continue to use the existing V1 single-actor executor.

## Native debugger and serial execution

A SuperDex scene's `DebugDraw` object is thread-affine. When the native SuperDex debugger is connected, its sync callbacks gather debug-draw data from the scene's step thread, so stepping scenes on `SceneBatchExecutor` workers with an attached debugger violates that affinity and traps natively. The default `batch` execution mode therefore fails closed: constructing or stepping the backend while a debugger client is connected raises an actionable `RuntimeError`.

Attach the debugger only in serial execution mode, which never constructs the executor and steps every scene on the environment thread:

```sh
create_backend("superdex", scene, num_envs, sim_dt, superdex_execution_mode="serial")
```

In UniLab, pass `env.superdex_execution_mode=serial` on the Hydra command line. `superdex_num_workers` has no effect in serial mode. The mode is a debugging profile, not a performance configuration; prefer `batch` for training.

Serial mode also unlocks the native Polyscope viewer (`superdex.physics.viewer`) for `run_playback` in `interactive` render mode. The viewer shares the scene's stepping thread, so interactive playback requires serial mode and exactly one environment, and fails closed with an actionable error otherwise. `record` and `auto` playback still use the shared MuJoCo offline renderer and work in both modes. UniLab's interactive SuperDex evaluation injects both settings (`serial` plus `training.play_env_num=1`).

## Native fixed-base robot

Preprocessed SuperDex assets stay outside the code repositories. The FR3 example uses the upstream `assets/bots/arms/fr3_v2` directory, including its HDF5 collision and GLB render files, and preserves its `LICENSE` and `NOTICE`. The native bot must have a hard root and fixed, hinge, or slide joints; components, cycles, tendons, and transmissions outside this profile fail closed.

```python
import numpy as np
from unisim import create_backend
from unisim.scene import SceneCfg

backend = create_backend(
    "superdex",
    SceneCfg("/path/to/project_superdex/assets/bots/arms/fr3_v2/fr3_v2.superdex_bot"),
    num_envs=2,
    sim_dt=0.002,
    base_name="fr3_link0",
    superdex_num_workers=0,
    superdex_effort_limits=[20, 20, 20, 20, 5, 5, 5],
)
try:
    backend.step(np.zeros((2, 7)), nsteps=5)
    state = backend.get_state()
finally:
    backend.cleanup_scene_assets()
```

The native control-vector names and ordering follow the single-DoF joint names. Positive finite effort limits must be present in the asset or supplied explicitly. The example values define a research control profile, not verified FR3 hardware ratings. Fixed-base `get_state()` contains only joint coordinates; requesting a floating-root layout for a fixed body is rejected. A named keyframe must actually exist in the scene; the adapter does not invent `home` for bots.

## Audited MJCF profile

The cold importer accepts one articulation tree, one optional free root, hinge and slide joints, scalar stateless motor or linear position actuators, and authored static planes. Existing scene fragments and named keyframes are materialized before stepping. Joint and actuator ordering remain distinct. Mass, inertial frame and center of mass, joint frames and axes, armature, joint friction, and control and force limits are mapped explicitly.

Portable `entity_assets` extend this audited MJCF profile to no-variant or same-layout fixed-variant physical fixed/floating entities, physical kinematic roots, zero-DoF native rigid actors for fixed static entities, and bounded one-body collision-disabled mirrors of rigid sources. Each entity's public qpos/qvel, body, actuator, and contact identities map to frozen native actor layouts. Immutable fixed-variant assignments select independently materialized native realizations; native mass, inertia, and collision geometry may differ while public topology and addressing stay frozen. A physical kinematic root retains its source-declared collision and inertial identity on a hidden native six-DoF free-root carrier; gravity is disabled on the carrier links, all six native DoFs are boundary conditions, and no public state, control, or body-wrench columns are exposed. A mirror remains a public kinematic entity with zero joints, DoFs, and actuators; its hidden native free-root carrier supplies native state writes and assignment-selected source identity but has no collision, gravity, actuator, public state, or physical-wrench ownership. World-pose writes are row-local for mirrors and physical roots, full reset restores each declared pose, and contact sensors targeting either fail closed. Entity-owned geom-pair contact sensors and selected reset-impacted control restoration are supported; variant changes to public topology, joint limits, actuator contracts, keyframes, or sensors, and portable world-body plane contact sensors fail closed.

Dynamic primitive collision geometry is triangulated and baked to SDF once during materialization. Separate welded geometry links retain authored geom-pair contact-sensor identity, and their mass and inertia parts sum to the original body's inertial properties. Mesh collision, arbitrary multiple joints per body, multiple articulations, equality, tendon, flex, source-authored mocap, hfield, and plugin features, and unsupported actuator or sensor semantics are rejected; the generated mocap representation of a mirror is adapter-private and never public state. Visual mesh files must still be present for the source MJCF parser even though this adapter is headless. No model parsing or SDF baking occurs during reset, step, or getters.

SuperDex contact and its implicit integration are not numerically equivalent to MuJoCo. Primitive SDFs approximate analytic surfaces, and solver settings have different meanings. Torsional and rolling friction require the explicit `superdex_allow_contact_approximation=True` experimental profile, which warns that only the sliding Coulomb component is preserved. The default rejects that loss of semantics. Go2's task owner opts into this profile; a finite rollout is not evidence of locomotion quality or equivalent contacts.

The 1.3.0 wheel exposes a broader per-pair friction-override API. This reviewed adapter slice continues to use its established factorization of authored sliding-friction pairs into native actor coefficients, whose geometric-mean mixing reproduces the selected MuJoCo pair coefficient. Incompatible friction graphs are rejected; no private engine API or silently changed mixing rule is used.

## State, controls, and sensors

Public free-root qpos is world xyz plus a wxyz quaternion, followed by single-DoF joints. Public reset qvel is world body-origin linear velocity plus body-frame angular velocity, followed by joint velocity. Native SuperDex free qpos stores a rotation vector, but its free rotational velocity is not the ordinary derivative of that vector. With an identity native reference transform, native free qvel uses world-origin linear velocity and world angular velocity. The adapter rotates the angular component at the state barrier and verifies body-origin and center-of-mass velocity against authored MuJoCo kinematics at nontrivial poses.

The pre-step control callback runs once per physics substep. Motor and position controls respect authored order, gains, gear, and limits. Pending body forces are accumulated as generalized forces and submitted together with control; one native external-force write cannot erase a separate control contribution.

Named joint position and velocity, frame pose, axis, and velocity, gyro, and velocimeter signals are reconstructed from native state into NumPy caches. Supported plane and geom `contact data="found" num="1"` signals use native contact points and the actual actor pair, not a nonzero-force proxy. Contacts represent the last completed physics solve. A reset clears solved contact state; `step(0)` does not rebuild the contact manifold after teleportation, so the first positive physics step supplies fresh contact results. Do not use reset-time contact flags as a geometric-overlap test.

Authored accelerometers are recognized but unavailable: requesting or binding one raises `NotImplementedError` because the public runtime does not supply instantaneous point acceleration. An unused accelerometer does not prevent loading an otherwise supported asset, and no zero or finite-difference substitute is presented as the authored sensor. Native bot sensor components, cameras, arbitrary force and touch sensors, and site Jacobians are outside this profile.

Full reset restores a private initial dynamic snapshot, writes selected qpos and qvel, clears controls and external forces, and refreshes kinematic caches. Other rows are unchanged. Portable selected-entity reset preserves unrelated environments, entities, and controls. It clears only controls targeting reset root/joint fields; `restore_default_controls=True` restores exactly those columns from zero construction defaults or, when a named default keyframe is selected, that merged keyframe's clipped control values. A mirror or physical kinematic root exposes world-pose readback, selected-row world-pose writes, zero public velocity, and its declared full-reset pose; body wrenches and contact sensors targeting it are rejected. Fixed-variant construction defaults and inertial readback remain assignment-selected. Snapshot bytes are not exposed as portable checkpoints. Model domain randomization, rendering and video, ROM, soft, and tactile state, and GPU batched physics are unsupported and must not be advertised by callers. Playback uses the shared offline MuJoCo renderer when a visual MJCF model is available.

## Validation

`scripts/benchmarks/superdex_scene_step.py` is a maintainer-only native physics-barrier measurement for direct scene stepping and the batch executor. It excludes model loading, actions, observations, rewards, resets, collectors, and learners, so it is not an RL throughput benchmark.

```sh
uv run --no-sync pytest -q tests/adapters/superdex/test_contract.py tests/adapters/superdex/test_backend.py tests/adapters/superdex/test_materialization.py tests/adapters/superdex/test_portable_scene.py
UV_NO_SYNC=1 make check
uv lock --check
make package
```

Set `SUPERDEX_ASSETS_PATH` to the upstream `assets` directory to include the external native FR3 fixture. Other numerical tests use small authored models and require the optional Python 3.12 runtime; contract and import tests also run without it. UniLab owns task-rollout, training-checkpoint, and sim2sim policy-I/O validation, and those outcomes are tracked in the roadmap's integration child.
