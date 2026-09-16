# M1 path ablation and ownership audit

[English](m1-ablation.md) | [中文](../zh/m1-ablation.md)

## Scope and reproduction

This audit compares the initial M1 implementation at `b0f77abb9866c2e72b082f5897871b0e1a8b3052` with the optimized reporting paths. [Machine-readable evidence](../evidence/m1-ablation.json) records the evaluated revision, worktree status, Python version, device, shapes and timings. These measurements isolate cold reporting work; they do not establish rollout throughput or broaden the [runtime support evidence](m1-runtime-evidence.md).

```bash
uv run --no-sync python scripts/benchmarks/m1_report_ablation.py --baseline b0f77ab --cuda --mujoco --output /tmp/unisim-m1-ablation.json
```

The benchmark reads historical modules from Git for the A/B comparison and requires the MuJoCo development extra plus a working Warp CUDA installation for the optional runtime paths. It asserts identical serialized report values and capability values. The MuJoCo experiment separately runs reporting disabled, reporting enabled and strict factory construction with 32 environments and 128 joints, then checks bit-identical final qpos/qvel after stepping. Reporting is disabled only by a temporary benchmark patch; there is no production bypass switch.

## Changes justified by the audit

- Configuration comparison previously built frozen trees solely for equality, then built them again for storage. Equal JSON values now avoid discarded trees, while validation and detachment remain enforced. A shared requested/effective input is frozen once. Capability serialization no longer recursively serializes evidence twice.
- MJWarp report capture avoids copying unused environment rows from the device and releases temporary requested-value tables after publishing the immutable snapshot. MuJoCo actuator-only overrides read only actuator tables instead of collecting a complete model report.
- Subprocess reports group environment assignments once and combine provenance/semantic normalization into one field replacement. Worker per-environment evidence remains intact; source XML never substitutes for runtime readback.
- Generic binding now validates configuration conditions through the public report. MuJoCo and SuperDex record their consumed options in their own adapter reports; the factory no longer probes MuJoCo private state or accepts forwarded kwargs as proof. The standalone validator now rejects conditions contradicted by materialized values, including mixed variant scopes and unauthorized approximations.
- The duplicate `wrench.body_force` key is replaced by the authoritative `dr.interval.body_force` key. The ambiguous `state.refresh` key is removed in favor of `state.final_refresh` and `state.callback_refresh`. Existing DR/play/variant APIs remain the source of support declarations.

## Recorded measurements

Python 3.13.14 and an NVIDIA GeForce RTX 4090 were used. Microbenchmarks report medians of nine repetitions; MuJoCo paths report seven measured repetitions after warmup. The JSON retains full precision.

| Isolated operation | Before / A | After / B | Interpretation |
| --- | --- | --- | --- |
| Configuration comparison, separate equal `(32, 512, 3)` values | 31.461 ms | 16.143 ms | Same serialized report; separate snapshots retained |
| Configuration comparison, shared input object | 31.522 ms; 2,376,912 peak Python bytes | 7.936 ms; 1,191,512 peak Python bytes | Detached immutable snapshot; storage shared only for identical input |
| Capability serialization | 0.077 ms | 0.031 ms | Same serialized declaration values |
| CUDA `(4096, 128, 3)` readback, consume one row | 0.403 ms, copy all rows | 0.021 ms, copy selected row | Isolated transfer mechanism; not full initialization latency |
| MuJoCo construct + materialize | Report off: 7.862 ms | Report on: 9.864 ms; strict: 10.351 ms | Reporting has measurable cold-path cost |
| MuJoCo ten substeps | Report off: 0.285 ms | Report on: 0.281 ms; strict: 0.285 ms | Identical states; timing differences are not a throughput claim |

Peak Python bytes come from `tracemalloc`; they exclude native and GPU allocations. Report shape and representative-row count affect transfer savings. These bounded measurements do not establish high-variant-count performance or Isaac worker startup speed.

## Ownership and lifecycle boundary

The parent `AGENTS.md` requires backend-specific behavior to remain behind `SimBackend`, DR to remain owned by Manager-Based event terms, and asset parsing to stay on cold paths. The changes keep option interpretation in adapters and generic condition checks in the existing validator. They add no environment/training policy, second DR registry or engine dependency to the base package.

UniLab's Manager-Based environment applies startup events before calling `materialize()`. Strict factory construction intentionally completes materialization early and therefore is not a universal drop-in for that lifecycle. Consumers with startup work should keep ordinary construction, perform their startup events and existing materialization, then call `validate_semantic_requirements(backend.get_capabilities(), requirements, backend.get_import_report())` before stepping. The report still describes initial adopted configuration; current values after DR must use current-property queries. No materialization idempotence shim or downstream lifecycle change is introduced.

Focused regressions cover independent validator use, approximation consent, configuration scope, detached snapshots and selected device rows. The repository's `make check` and `make package` remain the completion gates; microbenchmark timings are diagnostic evidence, not CI performance thresholds.
