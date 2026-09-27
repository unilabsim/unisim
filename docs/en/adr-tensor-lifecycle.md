# ADR: Optional Backend Tensor Lifecycle

[English](adr-tensor-lifecycle.md) | [中文](../zh/adr-tensor-lifecycle.md)

## Status

Accepted (2026-09-27)

## Context

UniLab's Manager-Based runtime and collectors are NumPy-oriented, while GPU physics engines expose stable batched device arrays. Replacing every NumPy boundary with an engine-specific tensor API would break existing managers, replay, and playback consumers. CPU physics engines also need to remain useful when manager arithmetic is accelerated.

## Decision

`SimBackend` exposes an optional, fail-closed tensor lifecycle rather than a global environment replacement. `TensorExecution` distinguishes `DEVICE_RESIDENT`, `HOST_BRIDGE`, and the default `UNSUPPORTED` profile. `get_tensor_capabilities()` exposes the partial method and reset-feature limits. Declaring adapters may implement `get_state_views()`, `get_sensor_view()`, `step_tensor()`, and `set_state_tensor()`.

The base package still does not depend on Torch. Adapters lazy-import their tensor runtime and own stream, layout, and transfer semantics.

Tensor inputs are contiguous Torch tensors. Controls are float32 `(num_envs, num_actuators)`; reset row IDs are unique int64 values in `[0, num_envs)`, and reset qpos/qvel are float32 `(rows, nq/nv)` tensors on one accepted device. Producers own finiteness so adapters do not force hot-path scalar synchronization. Row validation may perform one bounded synchronization. A completed step or reset consumes its inputs and synchronizes backend work.

For `DEVICE_RESIDENT`, `device=None` means the backend's exact CUDA device and returned live views are logically read-only. For `HOST_BRIDGE`, `device=None` means CPU; an explicit device requests an H2D copy and returned values are detached snapshots. `TensorLifecycleCapabilities.state_fields` is the machine-readable generalized-field set (`qpos`, `qvel`, `ctrl`, `time`, and where available `sensordata`). Body pose and velocity consumers use named sensor views rather than engine-private body-array conventions.

- MJWarp declares `DEVICE_RESIDENT`: Torch controls and selected reset rows are copied or scattered into stable MJWarp device storage, physics runs on CUDA, and DLPack exposes live state views. `step_tensor()` completes physics and leaves tracked-sensor refresh pending; the first tracked tensor sensor or `sensordata` read refreshes only device-resident state, and legacy NumPy generalized/body host caches lazily refresh only when mixed back in.
- MJWarp tensor stepping does not support host pre-step callbacks. Its minimal selected-row reset does not support model randomization, fixed variants, pending interval wrenches, or models with mocap bodies. Mixed legacy writes first refresh their host mirror so unselected device rows cannot regress.
- MuJoCo/MJBatch declares `HOST_BRIDGE`: accelerator controls and reset rows cross explicit host boundaries, CPU physics remains authoritative, and requested state or sensors are copied to the selected Torch device. Its tensor reset delegates the existing reset-randomization and fixed-variant host paths, while tensor stepping does not support host pre-step callbacks. Transfer optimization remains adapter work; there is no hidden device execution claim.
- All other adapters remain `UNSUPPORTED` and raise `NotImplementedError` rather than silently converting through NumPy.

## Related decisions

- [Capability and evidence ADR](adr-capabilities.md) defines fail-closed declarations and runtime evidence.
- [Entities and state ADR](adr-entities.md) defines the selected-row reset and public state-layout boundary.
- [Benchmark API reservation](benchmark-api.md) defines schema, synchronization, digest, and provenance expectations for stable benchmark contracts.

## Consequences

The contract keeps the existing NumPy lifecycle compatible and makes host-device boundaries measurable. It is not yet a complete GPU collector/replay protocol: `uni_rl`, IPC, replay ingress, and Manager-Based task dispatch require separate versioned migration. Backend callers must validate producer tensors where hot-path value checks would otherwise force scalar device synchronization, while adapters continue to validate metadata and fail closed for unsupported lifecycle features.
