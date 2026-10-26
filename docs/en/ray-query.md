# Ray-Query Plugin Contract

[English](ray-query.md) | [中文](../zh/ray-query.md)

UniSim defines a backend-neutral ray-query plugin contract so ray caster implementations (such as the separately distributed `uni_ray` package) can serve batched ray queries to task code without exposing engine-private types. The contract has two transports: a NumPy profile for tools, debugging, and conformance, and a Torch-native profile that joins the training tensor lifecycle without a host detour.

## Public surface

All names are exported lazily-safe from the `unisim` package root and live in `unisim.ray_query`:

- `RayCaster` — the abstract plugin lifecycle: `materialize(scene)` → pose update → trace → `close()`.
- `RaySceneDescription` — the backend-neutral structure-of-arrays scene consumed on the cold path.
- `RayCasterCapabilities` — the fine-grained capability declaration.
- `RayTraceOutputs` / `RayTraceResult` — the per-call output request and result containers.
- `RayPoseTensorView` / `RayTraceTensorResult` — caller-owned Torch pose inputs and caster-owned reusable Torch result views.
- `RayGeomType` — the primitive geometry kinds (`plane`, `sphere`, `box`, `cylinder`, `capsule`, `ellipsoid`, plus reserved `mesh`).
- `ray_geom_type(...)` — the fail-closed native MuJoCo geom-code/name mapping helper.
- `require_ray_trace_outputs(...)` — the fail-closed output negotiation helper.
- `RAY_CASTER_SPECS` / `ray_caster_spec(...)` — the declared plugin manifest.
- `create_ray_caster(...)` — the lazy factory dispatch.
- `FakeRayCaster` — the NumPy analytic reference implementation with a CPU Torch transport profile.
- `assert_ray_caster_conformance(...)` — the reusable conformance check for plugin authors.

## Type boundary

The host profile is NumPy-only; the training profile accepts Torch tensors without changing package import behavior. `mujoco.MjModel`/`mujoco.MjData`, Warp kernels or arrays, CUDA pointers, streams, events, and every other backend-private type must never appear in signatures or results. Real adapters translate their native scene and pose state into public descriptors and tensors on their own side of the boundary. Importing `unisim` never imports an engine SDK or Torch; plugin packages are imported lazily by `create_ray_caster`, tensor methods import Torch lazily, and a missing package or entry point fails closed with `OptionalDependencyError`.

## Scene description

`RaySceneDescription` carries one structure-of-arrays record per geom: `geom_types`, `geom_sizes` `(n, 3)`, `geom_local_pos` `(n, 3)`, `geom_local_quat` `(n, 4)` unit `wxyz`, and `geom_body_ids` `(n,)`, plus the explicit `num_bodies`. `body_names` is the public body axis; anonymous MuJoCo-family bodies receive stable `body_<i>` aliases. Optional body-name selection chooses and reorders that axis and remaps geom body IDs; missing, duplicate, or empty names fail closed. Geoms are expressed in their owning body's local frame. Sizes follow MuJoCo-style primitive conventions: sphere `(radius, 0, 0)`; box half extents `(x, y, z)`; cylinder and capsule `(radius, half_length, 0)` along local `+z`; ellipsoid semi-axes `(x, y, z)`; the plane is the infinite local `z = 0` plane with normal `+z` and ignores its sizes. All arrays are validated, detached, and write-protected at construction; invalid input fails closed.

MuJoCo-family backends expose this projection through `SimBackend.get_ray_scene_description(body_names=None)` and map MuJoCo geom codes through `ray_geom_type`. The call is cold-path metadata; a ray hot path never parses a model or refreshes the descriptor. The Motrix projection is limited to an audited source whose compiled geometry exactly matches the native runtime; generated hfield terrain, attached fragments, per-row mesh variants, and heterogeneous fixed-variant sources fail closed rather than silently omitting or misrepresenting runtime geometry.

## Lifecycle

1. `create_ray_caster(name, num_envs=..., num_rays=...)` fixes the batch shape `(num_envs, num_rays)` before any plugin import.
2. `materialize(scene)` binds the immutable scene geometry once on the cold path; every body pose starts as the identity. Materializing twice fails.
3. `update_pose(body_pos, body_quat, env_ids=None)` writes world-space body poses `(rows, num_bodies, 3)` / `(rows, num_bodies, 4)` for all rows or a validated selection, and requires `supports_pose_sync`.
4. `trace(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None)` casts the fixed ray batch and returns a `RayTraceResult`.
5. `close()` releases resources idempotently; later queries fail.

The tensor profile inserts an explicit pose layout bind:

1. `bind_pose_layout(body_names)` resolves the immutable body order exactly once and must match `scene.body_names`.
2. `update_pose_tensor(RayPoseTensorView)` passes caller-owned float32 Torch `(E, B, 3)` positions and `(E, B, 4)` `wxyz` quaternions on a declared device. Pose sources may be packed-strided views (for example columns of a host-bridge packet) when their innermost stride is one and their strides are positive and non-overlapping; unsupported layouts fail closed.
3. `trace_tensor(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None)` returns persistent Torch views.

The caster aliases pose storage rather than copying it, keeps public quaternion storage in `wxyz` and performs any component reorder inside its kernel, and replaces the alias at the next tensor pose update. Scene replacement is a rebuild, not a mutation: close, materialize a new descriptor, and bind again.

## Trace semantics

Rays are world-space with unit directions. Two input profiles exist: the shared profile passes `(num_rays, 3)` arrays broadcast over every selected row; the per-environment profile passes `(rows, num_rays, 3)` arrays and requires `supports_per_env_rays`. `max_distance` is a positive finite scalar clipping distance. The minimal result is `distance` (clipped to `max_distance`; missed rays report exactly `max_distance`) and `hit`. `env_ids` selects a validated subset of environment rows; result rows follow the selection order.

Result arrays may be views into implementation-owned buffers reused by the next `trace` call; callers retaining results across calls must copy them. Implementations must not require unbounded per-call allocation.

The tensor path accepts the same shared `(num_rays, 3)` and per-environment `(rows, num_rays, 3)` profiles, contiguous float32 Torch inputs on the caster's declared device, and a contiguous one-dimensional int64 `env_ids` tensor. Its minimal result is also clipped `distance` plus `hit`; `hit_point`, `geom_id`, and `body_id` are optional device-side outputs when declared. Caster-owned result views are reused by the next `trace_tensor`; a consumer must copy them into a term-owned carrier before the next trace. Finiteness and output-id diagnostics are deferred to device-side validation or an explicit debug read rather than forcing a hot-path synchronization.

## Capabilities and fail-closed negotiation

`RayCasterCapabilities` declares:

- `supports_pose_sync` — `update_pose` after materialization.
- `supports_per_env_rays` — the 3-D per-environment ray profile.
- `supports_host_readback` — the NumPy host result path of `trace` (required by the current conformance helper).
- `supports_tensor_pose_sync` / `supports_tensor_rays` — the Torch-native pose and trace lifecycle.
- `supports_device_output` — persistent caster-owned Torch result buffers.
- `torch_devices` — accepted Torch device families or exact CUDA indices, using backend tensor-contract spelling.
- `stream_event_ownership` — ordering policy; the current profile is `caller` only.
- `supports_hit_point`, `supports_normal`, `supports_geom_id`, `supports_body_id` — the optional `RayTraceResult` fields.

Optional outputs are requested per call through `RayTraceOutputs`. Requesting an undeclared output, using the per-environment profile without `supports_per_env_rays`, or calling `update_pose` without `supports_pose_sync` raises `UnsupportedCapabilityError`; nothing is silently ignored or downgraded. Hit points and normals are world-space, normals are oriented against the ray direction, and `geom_id`/`body_id` index the scene description order (`-1` on missed rays). Optional fields are only meaningful where `hit` is true.

Capability construction is internally cross-validated: tensor pose sync implies pose sync, tensor rays imply per-environment rays, device output implies the complete tensor profile and a non-empty unique Torch-device list, and device output requires caller stream ownership. A caster declaring `cpu` only must reject CUDA tensors, and vice versa, without copying them to a supported device.

| Capability | Fake | `uni_ray` host revision | `uni_ray` tensor revision |
| --- | --- | --- | --- |
| Host NumPy pose/trace | yes | yes | retained for tools, debugging, and conformance |
| Tensor pose sync | CPU | no | CPU/CUDA as declared by the instance |
| Tensor rays | CPU | no | shared and per-environment |
| Persistent Torch output | CPU | no | caster-owned reusable views |
| Torch devices | `cpu` | undeclared | exact device accepted by the caster |
| Stream ownership | caller | n/a | caller |
| Hit point / geom ID / body ID | host and tensor | host | tensor |
| Normal | host only | host only | outside the initial tensor result |
| Mesh / hfield | unsupported / explicit rejection | profile-specific | supported by the current `uni_ray` profile (hfields are cold-path triangulated meshes) |

## Reference implementation and conformance

`FakeRayCaster` implements analytic ray-vs-primitive intersection for the six primitive types in NumPy and exposes a CPU Torch transport profile; mesh geoms are rejected fail closed at materialization. It runs the core contract tests without MuJoCo, Warp, or any engine SDK. `assert_ray_caster_conformance(caster)` exercises lifecycle ordering, a canonical ground-plane scene with known analytic distances, capability-declared outputs, both ray profiles, selected rows, and fail-closed rejection — plugin authors should run it against their implementation.

## Plugin discovery

`RAY_CASTER_SPECS` declares plugin identities (`uni_ray` today). A plugin package must expose `create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster` at its top level; the UniSim factory imports it lazily, validates the returned object is a `RayCaster`, and reports an actionable `OptionalDependencyError` when the package or entry point is missing. Unknown caster names raise `ValueError`.
