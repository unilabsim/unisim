# ADR: Optional Backend Tensor Lifecycle

[English](adr-tensor-lifecycle.md) | [中文](../zh/adr-tensor-lifecycle.md)

## Status

Accepted (2026-09-27)

## Context

UniLab's Manager-Based runtime and collectors are NumPy-oriented, while GPU physics engines expose stable batched device arrays. Replacing every NumPy boundary with an engine-specific tensor API would break existing managers, replay, and playback consumers. CPU physics engines also need to remain useful when manager arithmetic is accelerated.

## Decision

`SimBackend` exposes an optional, fail-closed tensor lifecycle rather than a global environment replacement. `TensorExecution` distinguishes `DEVICE_RESIDENT`, `HOST_BRIDGE`, and the default `UNSUPPORTED` profile. `get_tensor_capabilities()` exposes the partial method and reset-feature limits plus the process topology, bulk data plane, stream/event ownership, and accepted Torch devices. An unsupported lifecycle must use the default in-process topology and no data plane. Valid supported combinations are in-process direct device storage, in-process host-bridge storage, external-worker CUDA IPC for device-resident physics, or external-worker host shared memory for host-bridged physics; no coarse classification implies that every optional tensor method is supported.

Capability construction also rejects contradictory declarations: unsupported matrices carry no feature, field, ownership, or device metadata; state views require fields; selected reset requires `qpos` and `qvel`; reset randomization requires selected reset; and packed plans require the in-process host-bridge matrix.

Declaring adapters may implement `get_state_views()`, `get_sensor_view()`, `step_tensor()`, and `set_state_tensor()`. A host-bridge adapter may additionally advertise `packed_host_bridge` and compile a `TensorIOSpec` into a `HostBridgeTransferPlan`.

The base package still does not depend on Torch. Adapters lazy-import their tensor runtime and own stream, layout, and transfer semantics.

An external GPU worker keeps its SDK interpreter and dependencies isolated from the host process. Its pipe carries control messages and errors only; bulk tensors cross through stable CUDA IPC arenas. The shared SDK-free transport exports opaque memory and event driver handles with ABI versions, a physical-device UUID, size, and alignment. It never serializes Torch-private storage, imports Isaac SDKs in the host process, or treats an engine-native buffer as a public stable pointer. The transport primitive does not by itself make IsaacGym or IsaacSim tensor-capable, and their existing CPU shared-memory subprocess path is not a tensor support claim.

Tensor inputs are contiguous Torch tensors. Controls are float32 `(num_envs, num_actuators)`; reset row IDs are unique int64 values in `[0, num_envs)`, and reset qpos/qvel are float32 `(rows, nq/nv)` tensors on one accepted device. Producers own finiteness so adapters do not force hot-path scalar synchronization. Row validation may perform one bounded synchronization. A completed step or reset consumes its inputs and synchronizes backend work.

For `DEVICE_RESIDENT`, `device=None` means the backend's exact CUDA device and returned live views are logically read-only. For `HOST_BRIDGE`, `device=None` means CPU; an explicit device requests an H2D copy and returned values are detached snapshots. `TensorLifecycleCapabilities.state_fields` is the machine-readable generalized-field set (`qpos`, `qvel`, `ctrl`, `time`, and where available `sensordata`). Body pose and velocity consumers use named sensor views rather than engine-private body-array conventions.

A packed host-bridge plan is compiled on the cold path after backend materialization. It freezes requested state and sensor shapes, offsets, dtypes, row IDs, and body IDs; preallocates pinned host staging and persistent accelerator destinations; and exposes the four semantic boundaries separately: one packed control D2H before CPU physics, one packed full state/sensor H2D after physics, one selected-row reset D2H, and one packed selected-row post-reset H2D. Empty reset sets perform no reset transfer. Selected transfers copy a contiguous prefix before scattering rows on the accelerator. Operations carry per-boundary timing plus cumulative direction, byte, and synchronization counters. Closing the backend invalidates its plans; plans are process-local and are not serialized with an environment factory.

- MJWarp declares `DEVICE_RESIDENT` with in-process direct storage: Torch controls and selected reset rows are copied or scattered into stable MJWarp device storage, physics runs on CUDA, and DLPack exposes live state views. `step_tensor()` completes physics and leaves tracked-sensor refresh pending; the first tracked tensor sensor or `sensordata` read refreshes only device-resident state, and legacy NumPy generalized/body host caches lazily refresh only when mixed back in.
- MJWarp tensor stepping does not support host pre-step callbacks. Its minimal selected-row reset does not support model randomization, fixed variants, pending interval wrenches, or models with mocap bodies. Mixed legacy writes first refresh their host mirror so unselected device rows cannot regress.
- MuJoCo/MJBatch declares an in-process `HOST_BRIDGE` and implements the packed plan: accelerator controls and reset rows cross explicit host boundaries, CPU physics remains authoritative, and requested state and sensors are packed into one stable H2D layout on the selected Torch device. Its tensor reset delegates the existing NumPy reset-randomization payload to the host adapter, while packed stepping does not support host pre-step callbacks. This is a transfer-layout optimization, not a device-resident physics claim.
- SuperDex has a backend-owned packed `HOST_BRIDGE` implementation with the same four semantic boundaries. It accepts CPU/CUDA tensors, but fixed variants, reset randomization, and host pre-step callbacks fail closed. The static support matrix does not promote it to a supported tensor backend until generalized-task parity and runtime benchmark evidence are recorded.
- MotrixSim and Drake also have backend-owned packed `HOST_BRIDGE` candidates with the same one-transfer-per-boundary contract. They accept CPU/CUDA tensors, keep CPU physics authoritative, and fail closed for fixed variants, reset randomization, host pre-step callbacks, and backend-specific unsupported reset work. Drake additionally fails closed while interval body forces are pending. These implemented instance candidates are not static support promotions; Drake still needs evidence against a native batch runtime.
- Newton has a partial in-process direct `DEVICE_RESIDENT` lifecycle for `qpos`/`qvel`, tensor stepping, and the current non-portable selected-reset profile. Sensors and portable multi-entity selected reset fail closed. It is not promoted until sensor/parity/benchmark evidence is complete.
- Genesis has a narrowly gated partial in-process direct `DEVICE_RESIDENT` candidate for a non-portable single articulation using Genesis' exact CUDA backend, zero-copy session, and public Torch getters. Stable backend-owned CUDA mirrors absorb Genesis 1.3.3's copying/non-contiguous public getter results without a bulk host detour; producers own finiteness, and selected reset packs row range and uniqueness checks into one bounded scalar read. CPU/ROCm, portable multi-entity, zero-copy-disabled, sensor, randomization, fixed-variant, and callback paths fail closed. This candidate is not a portable Genesis support promotion and still requires sensor, parity, hidden-sync, benchmark, and multi-GPU evidence.
- IsaacGym has a narrow experimental external-worker CUDA IPC instance candidate for `qpos`/`qvel` control, stepping, and selected reset. The SDK stays in its Python 3.8 worker, the host process never imports IsaacGym, the worker pipe carries only control metadata and errors, and same-GPU raw memory/event handles publish stable device arenas. Sensors, body/contact views, randomization, fixed variants, callbacks, full collector lifecycle parity, and benchmarks remain unsupported until their evidence is complete.
- IsaacSim has a narrow external-worker CUDA IPC candidate for device-resident `qpos`/`qvel` control, stepping, and selected reset. Its Kit/IsaacLab SDK stays in the worker and same-GPU raw arenas/events replace the legacy CPU shared-memory bridge on the opt-in tensor path. A minimal real mapped-worker selected-reset parity test passes, but sensors, randomization, callbacks, generalized-task/G1 parity, fault/shutdown coverage, benchmark evidence, and promotion review remain incomplete.
- Other adapters remain tensor `UNSUPPORTED` in the public support matrix and raise `NotImplementedError` rather than silently converting through NumPy. In particular, IsaacGym and IsaacSim's existing CPU shared-memory subprocess path is not a tensor support claim.

## Related decisions

- [Capability and evidence ADR](adr-capabilities.md) defines fail-closed declarations and runtime evidence.
- [Entities and state ADR](adr-entities.md) defines the selected-row reset and public state-layout boundary.
- [Benchmark API reservation](benchmark-api.md) defines schema, synchronization, digest, and provenance expectations for stable benchmark contracts.

## Consequences

The contract keeps the existing NumPy lifecycle compatible and makes host-device boundaries measurable. It is not yet a complete GPU collector/replay protocol: `uni_rl`, IPC, replay ingress, and Manager-Based task dispatch require separate versioned migration. Backend callers must validate producer tensors where hot-path value checks would otherwise force scalar device synchronization, while adapters continue to validate metadata and fail closed for unsupported lifecycle features.
