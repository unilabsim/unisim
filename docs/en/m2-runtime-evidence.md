# M2 runtime evidence and outstanding gates

[English](m2-runtime-evidence.md) | [中文](../zh/m2-runtime-evidence.md)

The [machine-readable record](../evidence/m2-runtime.json) captures clean implementation commit `2e571e31c835592becb844fb368150a427e3e34b`. Integration merge `e30ee8379a0a16abcf1cf4e444e1efe0ad8f74ed` has an identical tree (`git diff` is empty). This evidence addition changes tests/documentation only and does not rewrite that provenance. [#108](https://github.com/unilabsim/unisim/issues/108) remains open: successful physics tests do not imply that every release or rendering gate is complete.

## Physics and protocol gate

```bash
uv sync --locked --extra mujoco --extra mjwarp
UNISIM_TEST_ISAACGYM_SCENE=1 UNISIM_TEST_ISAACSIM_SCENE=1 make check
```

Result: **853 passed, 11 skipped**, 106.63 seconds. Ruff, mypy and Pyright passed. This includes actual MuJoCo CPU, MJWarp CUDA, both isolated vendor workers, public factory-to-worker scenes and old IsaacGym model-file inputs. Optional unsupported-runtime skips are not counted as native passes. The initial fresh audit worktree lacked the MuJoCo extra and failed collection; installing the explicit extras above and rerunning the full gate resolved that environment error.

| Runtime | Exercised behavior | Boundary |
| --- | --- | --- |
| MuJoCo 3.11 / mjbatch 0.2.1 | Multiple roots/passive joints, N2/K2 and N5/K2 identities, compiled inertial oracle, selected reset, independent native rollout and complete mocap playback | Declared MJCF composition subset; whole-model compatibility remains distinct from new public entity declarations |
| MuJoCo-Warp 3.11 / Warp 1.16 | Per-world identity/defaults, GPU state/frame oracle, reset-channel preservation, faulting, main-world/scratch routing and independent one-world rollout | Selected reset uses the documented forward barrier, not a new selective-forward performance claim |
| IsaacGym Preview 4 / Python 3.8 | Native asset identity, mass/COM/inertia, actuator-only entity controls, non-round-robin variants, repeated selected reset, zero-DoF scenes and legacy wire compatibility | Entity queries use link-origin velocity; old wire preserves native COM velocity conventions through an explicit projection |
| IsaacSim 5.1 / IsaacLab 0.47.2 / Python 3.11 | Actual prim/view identity, fixed/floating controlled and passive articulation, scoped reset, mirror isolation and public factory/default controls | Supported same-drive round-robin profile; unimplemented formats/layouts reject; native camera remains unverified |

Fixture source hashes, host package versions, GPU/driver and commands are recorded in JSON. Numerical expectations are defined in the linked test fixtures: independent compiled/native values and same-backend rollouts use their documented tolerances; cross-engine complex contact trajectories are not compared bitwise. The separate legacy IsaacSim diagnostic also passed actual initialization, reset and step through the unified runtime.

## Native rendering

IsaacGym produced nonuniform 320×240 RGB images. Capture preserved qpos/qvel/ctrl and assigned full-scene identity. None/record trajectories had maximum absolute difference zero, including an object outside the table at z=0.03: it fell to approximately 0.0217596 in both runs, checking that recording did not introduce a floor. The reproducible repository regression is `tests/contract/test_scene_rendering.py`, enabled with `UNISIM_TEST_ISAACGYM_RENDER=1`.

IsaacSim camera initialization fails before scene creation on this host. A stock IsaacLab AppLauncher with `enable_cameras=True`, importing no UniSim and loading no assets, exits with SIGSEGV (-11) in RTX/Hydra scene initialization. Disabling user configuration loading/persistence, Fabric scene delegation and sampled direct lighting individually did not resolve it. [#133](https://github.com/unilabsim/unisim/issues/133) records the independent reproducer. The corresponding renderer test is enabled only with `UNISIM_TEST_ISAACSIM_RENDER=1`; its absence from the passing physics gate is explicit, not a native camera support claim.

## Downstream and release boundary

[UniLab #1599](https://github.com/Motphys/UniLab/issues/1599) and [draft PR #1600](https://github.com/Motphys/UniLab/pull/1600) consume the public entity contracts with a registered pickleable EnvFactory. Development validation passed 1477 tests, 70% coverage, required 34/34 module and 35/35 script benchmark import checks, and two actual IsaacSim consumer scenes. Those runs used an explicitly identified editable M2 UniSim checkout. The released dependency lock still needs the agreed upstream version and final installation/CI verification; the editable result is not substituted for that gate.

The remaining-five-adapter assessment and follow-up issues are in [the owner matrix](m2-backend-followups.md). They do not count as five additional implementations. Before closing #108, retain final-head package/CI evidence, resolve or explicitly decide the native renderer acceptance boundary, and finish the downstream released-dependency gate. No missing evidence is promoted to success by this record.
