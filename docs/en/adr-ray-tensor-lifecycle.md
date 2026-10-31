# ADR: Ray Tensor Lifecycle

[English](adr-ray-tensor-lifecycle.md) | [中文](../zh/adr-ray-tensor-lifecycle.md)

## Status

Accepted for the 2.x tensor ray contract (#462; initial implementation slice). Supplemented for perception geometry, filtering, minimum distance, and optional tensor normals by [#472](adr-ray-perception-queries.md).

## Context

Phase-1 `uniray` established correctness through a host NumPy pose bridge: device poses were downloaded, converted, uploaded, traced, read back, and then often uploaded again by an observation. That data plane conflicts with UniSim's tensor lifecycle. `DEVICE_RESIDENT` backends already expose authoritative tracked-body Torch views; `HOST_BRIDGE` backends already publish one packed state/sensor packet to a Torch collector; UniLab observation code is Torch-native and can select either a CPU or CUDA carrier.

A ray query that crosses the host merely to cross back is also an ordering hazard: producer kernels, Warp kernels, and consumer Torch kernels can run on different streams, while a caster-owned reusable output can be overwritten before a caller reads it. Making pointers, Warp arrays, engine data objects, CUDA streams, or events public would only move that complexity into task code and couple UniLab to one caster implementation.

## Decision

The ray-query plugin contract is two-profile and remains backend-private at its boundary. The existing NumPy profile is a tool, debug, and conformance surface. Training collectors use the tensor profile:

```text
bind_pose_layout(body_names)
  → update_pose_tensor(RayPoseTensorView)
  → trace_tensor(...)
  → RayTraceTensorResult
```

The public tensor types are `RayPoseTensorView`, `RayTraceTensorResult`, and the tensor fields of `RayCasterCapabilities`. Torch remains a lazy runtime dependency: importing `unisim` or a caster package must not eager-import Torch, Warp, or an engine SDK. No `MjModel`, `MjData`, `mjwarp.Data`, `SceneData`, native model, Warp array, pointer, stream, or event crosses the boundary.

Pose sources are caller-owned. A caller passes full-batch float32 Torch views named by `body_names`, with positions shaped `(E, B, 3)` and unit quaternions shaped `(E, B, 4)` in public `wxyz` order. Fully contiguous storage is valid, and packed packet column views are equally valid when their innermost stride is one and their strides are positive and non-overlapping. The caster aliases that storage for the duration of one update only; it does not copy, reorder, transpose, silently call `.contiguous()`, extend source lifetime, or retain the tensors as its next source. Bind-time validation resolves the immutable name layout once. A second bind, missing body, wrong order, wrong device, wrong dtype, or unsupported storage pattern fails closed.

`DEVICE_RESIDENT` adapters (MJWarp in this slice) feed the caster from their public tracked-body Torch views. The ray hot path performs no pose D2H or H2D and no result D2D-to-host detour. `HOST_BRIDGE` adapters (MuJoCo and MotrixSim) add requested `body_pos_w` and `body_quat_w` as contiguous column blocks in the existing packed packet. The packet remains the sole public H2D: state, sensors, and ray pose blocks transfer together, and there is no ray-specific transfer or NumPy conversion. A CPU collector uses the corresponding CPU packet without creating a host-device crossing.

The Torch/Warp boundary is a zero-copy storage alias made by the caster. Public quaternion storage remains `wxyz`; a Warp kernel performs any component conversion required by caster-private storage (for example to `xyzw`) while scattering into the caster's own buffer. Vec-type pointer aliases must not own or free caller storage. Unsupported dtype, shape, device, or stride combinations fail closed rather than being normalized by a hidden copy.

Stream and event ownership is `caller`. Kernels launch on the caller's current stream, the caster creates no public synchronization primitive, and it does not synchronize the device on the tensor path. The caller's subsequent operations preserve stream order before reading persistent outputs. Explicit host validation is permitted only on a documented cold path; hot-path value checks that would force scalar device reads are deferred to a device-side reduction or to an explicit caller-selected debug read gate.

Outputs are caster-owned persistent buffers. `trace_tensor` reuses tensors keyed by batch shape and requested output profile, computes optional hit points and body IDs device-side, and returns views. A later trace may overwrite them. A caller that needs to retain values across calls writes them into its own carrier. Rebuild or scene replacement is cold: close the caster, obtain a fresh `RaySceneDescription`, materialize a new one, and bind a new pose layout. Existing plan or caster views are invalid after close; selected-row publication writes only the selected packet rows.

### Backend scene projection

Backends expose collision geometry through `SimBackend.get_ray_scene_description(body_names=None)`. The detached descriptor preserves public geom order, remaps geom body IDs to the selected public body axis, and names anonymous MuJoCo-family bodies with stable `body_<i>` aliases without rewriting native model names. Unknown, duplicate, empty, or reordered-but-unresolvable body names fail closed. This projection is metadata and is never parsed or refreshed on a ray hot path.

## Consequences

MJWarp can remain GPU-resident end to end, while MuJoCo and MotrixSim reuse one measurable packed H2D boundary. The contract makes transfer, synchronization, allocation, and stream-order behavior auditable rather than relying on implicit copies.

The caster must implement strict metadata validation and caller-owned storage discipline, and callers must copy reusable outputs when retaining them. Device-side errors need either a device reduction or an explicit debug read gate before Python sees a diagnostic. The initial tensor result deliberately excludes normals, and mesh support remains a caster capability decision rather than a universal backend promise. These restrictions buy a hot path with no hidden host round trip and no backend-private leakage.

The #462 exclusion of normals and public mesh topology was an initial-slice boundary, not a permanent contract decision. #472 adds the detached collision companion and a separately negotiated tensor-normal output while preserving every ownership and stream rule in this ADR.

## Related decisions

- [Tensor lifecycle ADR](adr-tensor-lifecycle.md) defines backend tensor execution and packed host-bridge plans.
- [Packed reset lifecycle ADR](adr-packed-reset-lifecycle.md) defines selected-row publication and plan transaction ownership.
- [Ray-query contract](ray-query.md) documents the concrete public API and capability matrix.
- [Perception-ready ray geometry ADR](adr-ray-perception-queries.md) defines the public collision companion, static filtering, minimum distance, and optional normals.
