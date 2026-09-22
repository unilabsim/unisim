# Repository scripts

This directory contains maintainer entry points that are deliberately outside `src/unisim` and are therefore not part of the installed package or public API.

- `benchmarks/superdex_scene_step.py` measures the raw SuperDex scene-step and native batch-executor barrier; it is not an RL throughput benchmark.
- `benchmarks/m1_report_ablation.py` compares report construction/serialization against a local Git revision, verifies identical outputs, and optionally measures full versus selected CUDA readback (`--cuda`); it does not change production behavior.
- `benchmarks/m2_path_ablation.py`, `m2_entity_query_ablation.py` and `m2_sim_reset_ablation.py` compare host mapping/query paths and sparse reset transfer volume against Git baselines. They reuse test-owned fixtures, assert output/call equality and do not measure overall native throughput.
- `benchmarks/issue141_fk_path_ablation.py` compares the legacy IsaacGym metadata/FK scan, reset staging and refresh publication against the named fix commit using NumPy-backed worker doubles.
- `benchmarks/isaacsim_contact_throughput.py` measures mapped IsaacSim scene-step throughput and worker GPU memory with no contact reporting, per-body net contact views, pair contact sensors, or both at configurable environment/body counts; it is a maintainer scaling probe, not an RL throughput benchmark.
- `benchmarks/isaacsim_kprototype_scale.py` sweeps variants x envs for the mapped IsaacSim K-prototype path, recording materialization time, VRAM delta and host-worker step throughput; it requires the optional IsaacSim worker runtime and an external MJCF asset pool.
- `diagnostics/isaacsim_scene_parity.py` dumps and fail-closed diffs the mapped IsaacSim scene materialization parity surface (worker INIT metadata including the init telemetry and native audit records, plus a deterministic zero-control rollout); it requires the optional IsaacSim worker runtime, a CUDA GPU, and an external MJCF asset pool, and is the archived reproduction of the #283 equivalence evidence.
- `diagnostics/check_newton_runtime.py` checks the pinned Newton distribution metadata and can optionally import the native stack.
- `diagnostics/check_support.py` generates/checks the bilingual semantic inventory (`--write-docs`/`--check-docs`) and explicitly runs one real runtime with the small `tests/contract/fixtures/m1_*.xml` assets (`--runtime mujoco`, `mjwarp`, `isaacgym`, or `isaacsim`; `--output` saves JSON evidence).

Put reusable runtime code in `src/unisim`, regression coverage in the matching `tests/` subtree, and add a script here only when it needs to be a standalone maintainer command. Do not use this directory as an unversioned scratch area.
