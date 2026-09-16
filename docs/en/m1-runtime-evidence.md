# M1 runtime evidence and remaining acceptance

[English](m1-runtime-evidence.md) | [中文](../zh/m1-runtime-evidence.md)

## Scope and reproducibility

The initial runs on 2026-09-16 used the implementation worktree based on `6d62d9ba40a1f637fb04da190fd38f64d02b05d1` (dirty, not a claim about the unmodified base). [Machine-readable records](../evidence/m1-runtime.json) retain revision, dirty state, exact command, host and worker versions, fixture hash, device, tolerances, individual checks and report samples. Re-run against the committed implementation before promoting verification evidence. No mock or skipped test is counted as a real runtime pass.

```bash
uv run --no-sync python scripts/diagnostics/check_support.py --runtime mujoco --output /tmp/m1-mujoco.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime mjwarp --output /tmp/m1-mjwarp.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime isaacgym --output /tmp/m1-isaacgym.json
uv run --no-sync python scripts/diagnostics/check_support.py --runtime isaacsim --output /tmp/m1-isaacsim.json
uv run --no-sync python scripts/diagnostics/check_support.py --check-docs
```

The independent assets are `tests/contract/fixtures/m1_semantics.xml` (motor) and `m1_position.xml` (position drive). Each has a floating root, one hinge, authored mass/inertia, a self-collision exclusion, gyro/accelerometer, Newton/Euler source options and gravity. Source dt is 0.004 s and requested factory dt is 0.002 s, explicitly exercising an override. The diagnostic uses two environments and actual reset/step calls; it checks semantic fields and units rather than cross-engine trajectory equality.

## Recorded runs

| Runtime | Environment | Result | Actual checks and remaining limits |
| --- | --- | --- | --- |
| MuJoCo CPU | Python 3.13.14; MuJoCo 3.11.0; mjbatch-uni 0.2.1 | passed | Newton/Euler, dt, gravity, motor gear, exclusion, mass/inertia and sensor map; actual finite gyro/accel after stepping |
| MJWarp CUDA | Python 3.13.14; MuJoCo-Warp 3.11.0; Warp 1.16.0; RTX 4090, driver 595.84 | passed | Same fixture checks, effective device timestep/gravity and real CUDA stepping; no broad solver equivalence claim |
| IsaacGym worker | Python 3.8.20; IsaacGym 1.0rc4; Torch 2.4.1; RTX 4090 | passed | Worker dt/gravity, position gains, disabled self-collision, per-env mass/inertia; real gyro; motor/accelerometer refusal; integrator unknown |
| IsaacSim worker | Python 3.11.16; IsaacSim 5.1.0.0; IsaacLab 0.47.2; Torch 2.7.0+cu128; RTX 4090 | passed | Worker dt/gravity, position gains, disabled self-collision, per-env mass/inertia; real gyro; motor/accelerometer refusal; solver/integrator unknown |

Tolerances are 1e-9 s absolute for dt, 1e-5 m/s² for gravity and 1e-6 absolute for authored mass/inertia. Sensor checks require finite `(2, 3)` arrays, not a calibrated accuracy claim. A cached snapshot must remain identical after reset/step. Worker position-drive and collision entries retain adapter-setting provenance where no engine readback exists; numerical checks do not relabel that provenance. Unsupported sensor rejection is evidence of refusal, not sensor implementation. Gym float32 gravity readback can be marked approximate despite being within the diagnostic tolerance; strict gravity requirements still require explicit consent. Heterogeneous solver labels remain unknown rather than being silently accepted as an override.

## Dependencies, owners and open work

| Work item | Dependency | Owner boundary | Acceptance status |
| --- | --- | --- | --- |
| #101 declarations | M0 semantics | capability/contract | SDK-free contract and instance aggregation implemented |
| #102 evidence | #101 schema | capability/adapter profile | Exact identity matching, withdrawal and serialization implemented; source-only inventory stays unverified |
| #103 import reports | #101 and #102 | materialization/worker IPC | MuJoCo, MJWarp and both workers instrumented; unavailable effective fields remain unknown |
| #104 strict validation | #101–#103 | factory/binding/adapters | Explicit strict requests implemented; existing callers keep their lifecycle and audits |
| #105 inventory and acceptance | #101–#104 | conformance/runtime owners | Nine source entries and four tiny real runtime runs; broader profiles and unknown fields remain open |

Declared base is `main`; the integration branch is `dev/issue-91-trusted-multibackend`. The table tracks implementation boundaries, not permission to close issues automatically. CPU, CUDA and vendor worker evidence stays separate. Motrix, Drake, Newton, Genesis and SuperDex were source-reviewed here but not exercised by this diagnostic. Their real-profile validation belongs to their adapter/runtime owners and remains unverified.

The materialization/worker owners must extend readback for unresolved Isaac solver/integrator semantics before consumers can strictly require them. Adapter/runtime owners must add broader collision/contact behavior and sensor accuracy evidence; the current snapshot validates configured masks/settings, not all physical contact cases. Multi-entity composition and general contact APIs remain #84/#72 work. No missing SDK is silently passed: an unavailable runtime exits failed with its diagnostic, and its acceptance remains incomplete until the responsible runtime owner reruns successfully.
