# M2 path ablation and contract audit

[English](m2-ablation.md) | [中文](../zh/m2-ablation.md)

## Scope and reproduction

The [machine-readable record](../evidence/m2-ablation.json) compares UniSim baseline `e952419` with clean implementation `fcbf8c7`, and UniLab baseline `044a11ff` with clean implementation `71c11431`. Each A/B compares identical inputs and checks output equality. Host tensor doubles isolate mapping or transfer volume; they do not establish native simulation or training throughput. Required native regression gates are recorded separately in the PR.

```bash
uv run --no-sync python scripts/benchmarks/m2_path_ablation.py --output /tmp/gym-ab.json
uv run --no-sync python scripts/benchmarks/m2_entity_query_ablation.py --output /tmp/query-ab.json
uv run --no-sync python scripts/benchmarks/m2_sim_reset_ablation.py --output /tmp/sim-ab.json
uv run --no-sync python scripts/benchmarks/issue141_fk_path_ablation.py --output /tmp/issue141-fk-ab.json
# In the UniLab consumer checkout:
uv run python scripts/benchmark/physics/m2_reset_ablation.py --output /tmp/reset-ab.json
```

The scripts load the old implementation from Git, not a rewritten approximation. Query A also loads the old shared snapshot helper. UniLab includes a third path changing only row indexing, so the indexing and unused-snapshot effects can be separated. There is no production A/B switch or CI performance threshold.

## Findings and changes

- Gym rebuilt public state using Python loops over every environment and joint, and recomputed body masks every refresh. Cold-bound gathers now use actual native addresses in bulk. Pending indexed resets, unowned body slots, COM conversion and legacy projections remain intact.
- MuJoCo/MJWarp entity queries previously constructed all entity roots, copied unrelated joints and then took the requested snapshot. They now gather one entity. Advanced-index copies no longer receive another redundant copy; results remain detached.
- Subprocess getters now use cold-bound entity/name/column maps. They no longer scan the layout or copy the same gathered array twice.
- IsaacSim sparse joint resets downloaded every environment's joint position/velocity before selection and uploaded IDs for untouched entities. Selection now happens on-device before download; untouched entities are skipped. Native setter values/order, actuator reset/update and post-commit refresh remain unchanged.
- Both MuJoCo-family reset adapters reuse the prepared request binding. Descendant-body, DoF, actuator and activation cleanup addresses are compiled once by the internal reset-impact owner instead of traversing static topology and reading model activation metadata on reset.
- UniLab reset staging replaces quadratic row lookup with a row map and stores only requested fields. Current state is read only when merging different joint-field selections needs missing columns. A logical root naming a descendant body is rejected at cold binding, preventing root read/default/write disagreement.
- The #141 follow-up parses each legacy MJCF source once for metadata and FK tables, prepares FK arrays once in the worker, and uses NumPy row indices for fixed-variant grouping. The sparse per-environment overlay remains: an always-allocated dense buffer was tested and rejected because it consumed full-batch memory without a reliable refresh gain.

## Measurements and limits

| Isolated operation | A | B | Evidence boundary |
| --- | --- | --- | --- |
| Gym refresh, N=4096, 32 joints | 142.329 ms | 1.387 ms | All shared slots exactly equal; excludes GPU/IPC |
| MuJoCo entity query, N=4096 | 714.07 µs | 128.90 µs | Equal root/joint arrays; host cache only |
| MJWarp entity query, N=4096 | 679.83 µs | 124.55 µs | Equal root/joint arrays; no device transfer |
| IsaacSim one-joint reset, N=1024, one selected row | 262,144 bytes downloaded | 8 bytes downloaded | Executed tensor-double byte accounting; native calls equal |
| UniLab pose staging, N=R=4096 | 23.456 ms | Row map only: 1.697 ms; sparse fields: 1.407 ms | Identical single reset request; no engine/IPC |
| UniLab defaults staging, N=R=4096 | 23.876 ms | Row map only: 3.243 ms; sparse fields: 1.877 ms | Current snapshot eliminated: 557,056 → 0 bytes |
| #141 cold metadata+FK scan, 128 bodies | 1.002 ms / 313,765 peak bytes | 0.817 ms / 208,935 peak bytes | Equal metadata/FK tables; one local NumPy-file median |
| #141 stage+refresh, N=512, 128 bodies, 4 variants | Stage 38.559 ms; refresh 6.343 ms | Stage 37.826 ms; refresh 6.348 ms | All published slots equal; excludes SDK/GPU/IPC |

Gym bulk gathers trade temporary memory for speed: peak traced Python/NumPy allocations at N=4096 increased from 2.25 MB to 3.54 MB. The cold native-index cache also scales with N and joint/body count. This does not measure native/GPU memory. Query peak traced allocations fall from 2.72 MB (CPU) / 2.08 MB (Warp) to 0.95 MB. Timings are local medians and can vary with load; they are not speed guarantees.

MJWarp still has a full-batch state-preservation barrier on partial reset. A separate real CUDA diagnostic with one free body and no contacts measured N=512, one selected row at about 0.334 ms, uploading 90,624 bytes and reading 49,152 persistent-channel bytes; selecting all rows took about 0.364 ms with the same volume. Its exact probe source is included in JSON. This is a remaining scaling cost, not removed by the host optimizations. Replacing it requires selected device scatter plus warmstart/control/wrench/sensor isolation evidence; removing the preservation barrier would violate the reset contract. IsaacSim post-commit full refresh similarly remains an explicit cost.

## Contract audit and retained paths

The parent `AGENTS.md` requires backend behavior behind `SimBackend`, asset metadata on cold paths, Manager-Based events as the DR owner, and pickleable environment factories. This audit removes static topology interpretation from reset and fixes physical-root binding. UniLab continues to consume only public state/layout/reset methods; no engine-private calls, hot XML parsing, task-name dispatch, second DR lifecycle, learner dependency or new `utils` owner is introduced. Dict observations, action dimensions and existing sim2sim policy-I/O remain unchanged.

Whole-world reset and selected-entity patch remain distinct intents submitted through one native owner. Compatibility projection preserves old wire shapes; independent native/source identity audits remain necessary. Scratch/full-forward strategies, native failure poisoning and sensor/wrench lifecycle are retained because deleting them would change semantics rather than remove redundancy.

The maintainer explicitly deferred IsaacSim recording acceptance to [#133](https://github.com/unilabsim/unisim/issues/133) on 2026-09-17. Its camera profile remains unverified; the deferral does not turn startup failure into a rendering pass. M2 completion still requires the final integration, package and downstream dependency gates recorded in #108/#113.
