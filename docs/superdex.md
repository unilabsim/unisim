# SuperDex CPU development profile

The `superdex` adapter runs SuperDex Physics/Robotics 1.0.0 directly behind
`SimBackend`. UniLab roadmap [#1533](https://github.com/Motphys/UniLab/issues/1533)
tracks this development profile. Changes remain on roadmap branches; the
version is unchanged and no PyPI release is required for local integration.

## Installation and ownership

Use CPython 3.12, as required by the upstream wheels. From the UniSim checkout:

```sh
uv sync --python 3.12 --extra superdex --extra mujoco
```

`superdex-physics==1.0.0` and `superdex-robotics==1.0.0` are optional. The extra
also supplies MuJoCo 3.11 as a **cold MJCF parser**; SuperDex executes every
physics step. Native `.superdex_bot` loading does not use that parser.
Importing `unisim` or its `SuperDexBackend` class does not load either engine.
SuperDex Lab, Gymnasium and a learner are not adapter dependencies.

For a sibling UniLab checkout, keep both versions unchanged and install local
editable projects together, for example `uv pip install -e './[superdex,mujoco]'
-e ../UniLab`. Use `uv run --no-sync` (or `UV_NO_SYNC=1 make check`) while
testing editable overrides so normal project synchronization does not replace
them with index distributions. UniLab's local provenance test profile uses
`UNILAB_LOCAL_UNISIM` pointing at the exact UniSim checkout. The UniLab backend
guide describes its task and registered asset setup.

The verified platform is Linux x86_64, CPU FP32. Upstream also provides
Windows x86_64 and macOS ARM wheels, but this integration has not established
those platforms. The default x86 build requires AVX2 and related instructions.
The upstream source exposes optional CUDA linear solvers, but the tested wheel
rejects them as not built with CUDA. This adapter does not enable GPU solvers.
FP64 upstream packages require a process-wide precision choice before import;
the integration's numerical validation currently targets FP32.

Each environment owns an independent native scene. The adapter reference
counts the process-global engine: closing one instance leaves other instances
alive. The source-built SuperDex `SceneBatchExecutor` batches force writes,
stepping, articulated state, link state, contact sensors and solver status in
persistent C++ workers. `superdex_num_workers=0` uses the process CPU affinity;
SDK-internal workers are disabled. Runtime initialization must belong to UniSim,
and live backends cannot be transferred between processes. Call the public
`cleanup_scene_assets()` hook or `close()` before interpreter shutdown. UniLab's
`env.close()` calls that public hook.

## Native fixed-base robot

Preprocessed SuperDex assets stay outside the code repositories. The FR3 example
uses the upstream `assets/bots/arms/fr3_v2` directory including its HDF5 collision
and GLB render files. Preserve its LICENSE/NOTICE. The native bot must have a
HARD root and fixed/hinge/slide joints; components, cycles, tendons and
transmissions outside this profile fail closed.

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

The native control vector names and ordering follow the single-DoF joint names.
Positive finite effort limits must be present in the asset or supplied
explicitly. The example values define a research control profile, not verified
FR3 hardware ratings. Fixed-base `get_state()` contains only joint coordinates;
requesting a floating-root layout for a fixed body is rejected. A named keyframe
must actually exist in the scene; the adapter does not invent `home` for bots.

## Audited MJCF profile

The cold importer accepts one articulation tree, one optional free root,
hinge/slide joints, scalar stateless motor or linear position actuators and
authored static planes. Existing scene fragments and named keyframes are
materialized before stepping. Joint and actuator ordering remain distinct.
Mass, inertial frame/COM, joint frames/axes, armature, joint friction and
control/force limits are mapped explicitly.

Dynamic primitive collision geometry is triangulated and baked to SDF once
during materialization. Separate welded geometry links retain authored
geom-pair contact sensor identity; their mass/inertia parts sum to the original
body's inertial properties. Mesh collision, arbitrary multiple joints per body,
multiple articulations, equality/tendon/flex/mocap/hfield/plugin features and
unsupported actuator/sensor semantics are rejected. Visual mesh files still
need to be present for the source MJCF parser even though this adapter is
headless. No model parsing or SDF baking occurs during reset, step or getters.

SuperDex contact and its implicit integration are not numerically equivalent to
MuJoCo. Primitive SDFs approximate analytic surfaces, and solver settings have
different meanings. Torsional/rolling friction requires the explicit
`superdex_allow_contact_approximation=True` experimental profile, which warns
that only the sliding Coulomb component is preserved. The default rejects that
loss of semantics. Go2's task owner opts into this profile; a finite rollout is
not evidence of locomotion quality or equivalent contacts.

The 1.0.0 wheel lacks the newer source tree's per-pair friction override API.
The importer therefore factors authored sliding-friction pairs into native
actor coefficients so their geometric-mean mixing reproduces the selected
MuJoCo pair coefficient. Incompatible friction graphs are rejected; no private
engine API or silently changed mixing rule is used.

## State, controls and sensors

Free-root public qpos is world xyz + **wxyz**, followed by single-DoF joints.
Public reset qvel is world body-origin linear velocity + **body-frame angular
velocity**, followed by joint velocity. Native SuperDex free qpos stores a
rotation vector, but its free rotational velocity is **not** the ordinary
derivative of that vector. With an identity native reference transform, native
free qvel uses world origin linear velocity and world angular velocity. The
adapter rotates the angular component at the state barrier and verifies body
origin/COM velocity against authored MuJoCo kinematics at nontrivial poses.

The pre-step control callback runs once per physics substep. Motor/position
controls respect authored order, gains, gear and limits. Pending body forces
are accumulated as generalized forces and submitted together with control;
one native external-force write cannot erase a separate control contribution.

Named joint position/velocity, frame pose/axis/velocity, gyro and velocimeter
signals are reconstructed from native state into NumPy caches. Supported
plane/geom `contact data="found" num="1"` signals use native contact points and
the actual actor pair, not a nonzero-force proxy. Contacts represent the last
completed physics solve. A reset clears the solved contact state; `step(0)` does
not rebuild the contact manifold after teleportation, so the first positive
physics step supplies fresh contact results. Do not use reset-time contact
flags as a geometric-overlap test.

Authored accelerometers are recognized but unavailable: requesting/binding one
raises `NotImplementedError`, because the public runtime does not supply
instantaneous point acceleration. Unused accelerometers do not prevent loading
an otherwise supported asset; no zero or finite-difference substitute is
presented as the authored sensor. Native bot sensor components, cameras,
arbitrary force/touch sensors and site Jacobians are outside this profile.

Reset restores a private initial dynamic snapshot, writes selected qpos/qvel,
clears controls/external forces and refreshes kinematic caches. Other rows are
unchanged. Snapshot bytes are not exposed as portable checkpoints. Model DR,
rendering/video, ROM/soft/tactile state and GPU batched physics are unsupported
and must not be advertised by callers.

## Validation

```sh
uv run --no-sync pytest -q tests/test_superdex_contract.py tests/test_superdex.py tests/test_superdex_materialization.py
UV_NO_SYNC=1 make check
uv lock --check
make package
```

Set `SUPERDEX_ASSETS_PATH` to the upstream `assets` directory to include the
external native FR3 fixture. Other numerical tests use small authored models
and require the optional Python 3.12 runtime. Contract/import tests also run
without it. UniLab owns task rollouts, training checkpoints and sim2sim policy
I/O validation; those outcomes are tracked in the roadmap's integration child.
