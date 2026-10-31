# Ray-Query Plugin Contract

[English](ray-query.md) | [中文](../zh/ray-query.md)

UniSim defines a backend-neutral ray-query plugin contract so ray caster implementations (such as the separately distributed `uniray` package) can serve batched ray queries to task code without exposing engine-private types. The contract has two transports: a NumPy profile for tools, debugging, and conformance, and a Torch-native profile that joins the training tensor lifecycle without a host detour.

## Public surface

All names are exported lazily-safe from the `unisim` package root and live in `unisim.ray_query`:

- `RayCaster` — the abstract plugin lifecycle: `materialize(scene)` → pose update → trace → `close()`.
- `RaySceneDescription` — the backend-neutral structure-of-arrays scene consumed on the cold path.
- `RayCollisionDescription` / `RayMeshDescription` — the detached collision companion carrying mesh topology, geom/mesh identity, semantic tags, and body ancestry.
- `RayCasterCapabilities` — the fine-grained capability declaration.
- `RayTraceOutputs` / `RayTraceResult` — the per-call output request and result containers.
- `RayTraceFilter` — the immutable static geom/body/tag filter request.
- `RayPoseTensorView` / `RayTraceTensorResult` — caller-owned Torch pose inputs and caster-owned reusable Torch result views.
- `RayGeomType` — the primitive geometry kinds (`plane`, `sphere`, `box`, `cylinder`, `capsule`, `ellipsoid`, plus reserved `mesh`).
- `ray_geom_type(...)` — the fail-closed native MuJoCo geom-code/name mapping helper.
- `require_ray_trace_outputs(...)` — the fail-closed output negotiation helper.
- `require_ray_trace_query(...)` — the fail-closed static-filter and minimum-distance negotiation helper.
- `RAY_CASTER_SPECS` / `ray_caster_spec(...)` — the declared plugin manifest.
- `create_ray_caster(...)` — the lazy factory dispatch.
- `FakeRayCaster` — the NumPy analytic reference implementation with a CPU Torch transport profile.
- `assert_ray_caster_conformance(...)` — the reusable conformance check for plugin authors.

## Type boundary

The host profile is NumPy-only; the training profile accepts Torch tensors without changing package import behavior. `mujoco.MjModel`/`mujoco.MjData`, Warp kernels or arrays, CUDA pointers, streams, events, and every other backend-private type must never appear in signatures or results. Real adapters translate their native scene and pose state into public descriptors and tensors on their own side of the boundary. Importing `unisim` never imports an engine SDK or Torch; plugin packages are imported lazily by `create_ray_caster`, tensor methods import Torch lazily, and a missing package or entry point fails closed with `OptionalDependencyError`.

## Scene description

`RaySceneDescription` carries one structure-of-arrays record per geom: `geom_types`, `geom_sizes` `(n, 3)`, `geom_local_pos` `(n, 3)`, `geom_local_quat` `(n, 4)` unit `wxyz`, and `geom_body_ids` `(n,)`, plus the explicit `num_bodies`. `body_names` is the public body axis; anonymous MuJoCo-family bodies receive stable `body_<i>` aliases. Optional body-name selection chooses and reorders that axis and remaps geom body IDs; missing, duplicate, or empty names fail closed. Geoms are expressed in their owning body's local frame. Sizes follow MuJoCo-style primitive conventions: sphere `(radius, 0, 0)`; box half extents `(x, y, z)`; cylinder and capsule `(radius, half_length, 0)` along local `+z`; ellipsoid semi-axes `(x, y, z)`; the plane is the infinite local `z = 0` plane with normal `+z` and ignores its sizes. All arrays are validated, detached, and write-protected at construction; invalid input fails closed.

MuJoCo-family backends expose this projection through `SimBackend.get_ray_scene_description(body_names=None)` and map MuJoCo geom codes through `ray_geom_type`. Only geoms with non-zero `contype` or `conaffinity` are projected; visual-only geoms with both masks set to zero are excluded. The call is cold-path metadata; a ray hot path never parses a model or refreshes the descriptor. The Motrix projection is limited to an audited source whose compiled geometry exactly matches the native runtime; generated hfield terrain, attached fragments, per-row mesh variants, and heterogeneous fixed-variant sources fail closed rather than silently omitting or misrepresenting runtime geometry.

The companion covers collision geometry and ray queries only. Public materialized terrain projection, hfield grid metadata, and subterrain identity are tracked separately in #474.

Motrix collision descriptors are primitive-only in this phase: source hfield/mesh geometry fails closed because MotrixSim normalization and native mesh introspection differ from the detached MuJoCo source projection.

`SimBackend.get_ray_collision_description(body_names=None)` returns the companion `RayCollisionDescription`. It carries `geom_names`, opaque source-qualified `geom_tags`, deduplicated `RayMeshDescription` assets, `geom_mesh_ids`, and optional acyclic `body_parent_ids`. Parent IDs describe ancestry within the selected public body axis; an omitted native ancestor becomes a selected-axis root, so callers that need complete ancestor exclusion should select the relevant ancestor chain. A geom instance identity is its public index and name; a mesh asset identity is separate and is shared when multiple geoms reference one asset. Both are stable only within one descriptor/materialization lifecycle. A rebuild produces a new descriptor and invalidates the previous identities. Tags have no universal vocabulary: the MuJoCo-family adapter currently emits `mujoco:geom_group:<n>` as an opaque, source-qualified tag, while callers select tags through the public descriptor rather than through native fields. All arrays are detached and write-protected; native models, native meshes, Warp arrays, pointers, and USD prims never cross the boundary.

Portable mesh/hfield callers pass this companion to the caster factory as `collision=...` and materialize `collision.scene`; no backend-specific builder is required.

## Lifecycle

1. `create_ray_caster(name, num_envs=..., num_rays=...)` fixes the batch shape `(num_envs, num_rays)` before any plugin import.
2. `create_ray_caster(..., collision=RayCollisionDescription)` binds the detached companion on the cold path, then `materialize(collision.scene)` binds the immutable scene geometry. Every body pose starts as the identity. Materializing twice fails.
3. `update_pose(body_pos, body_quat, env_ids=None)` writes world-space body poses `(rows, num_bodies, 3)` / `(rows, num_bodies, 4)` for all rows or a validated selection, and requires `supports_pose_sync`.
4. `trace(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None, min_distance=0.0, ray_filter=None)` casts the fixed ray batch and returns a `RayTraceResult`.
5. `close()` releases resources idempotently; later queries fail.

The tensor profile inserts an explicit pose layout bind:

1. `bind_pose_layout(body_names)` resolves the immutable body order exactly once and must match `scene.body_names`.
2. `update_pose_tensor(RayPoseTensorView)` passes caller-owned float32 Torch `(E, B, 3)` positions and `(E, B, 4)` `wxyz` quaternions on a declared device. Pose sources may be packed-strided views (for example columns of a host-bridge packet) when their innermost stride is one and their strides are positive and non-overlapping; unsupported layouts fail closed.
3. `trace_tensor(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None, min_distance=0.0, ray_filter=None)` returns persistent Torch views.

The caster aliases pose storage rather than copying it, keeps public quaternion storage in `wxyz` and performs any component reorder inside its kernel, and replaces the alias at the next tensor pose update. Scene replacement is a rebuild, not a mutation: close, materialize a new descriptor, and bind again.

## Trace semantics

Rays are world-space with unit directions. Two input profiles exist: the shared profile passes `(num_rays, 3)` arrays broadcast over every selected row; the per-environment profile passes `(rows, num_rays, 3)` arrays and requires `supports_per_env_rays`. `max_distance` is a positive finite scalar clipping distance. The minimal result is `distance` (clipped to `max_distance`; missed rays report exactly `max_distance`) and `hit`. `env_ids` selects a validated subset of environment rows; result rows follow the selection order.

Result arrays may be views into implementation-owned buffers reused by the next `trace` call; callers retaining results across calls must copy them. Implementations must not require unbounded per-call allocation.

The tensor path accepts the same shared `(num_rays, 3)` and per-environment `(rows, num_rays, 3)` profiles, contiguous float32 Torch inputs on the caster's declared device, and a contiguous one-dimensional int64 `env_ids` tensor. Its minimal result is also clipped `distance` plus `hit`; `hit_point`, `normal`, `geom_id`, and `body_id` are optional device-side outputs when declared. Caster-owned result views are reused by the next `trace_tensor`; a consumer must copy them into a term-owned carrier before the next trace. Finiteness and output-id diagnostics are deferred to device-side validation or an explicit debug read rather than forcing a hot-path synchronization.

## Rich query semantics

`min_distance` is an inclusive lower bound and `max_distance` is an exclusive upper bound: an accepted hit satisfies `min_distance <= distance < max_distance`. The lower bound must be strictly less than the upper bound. Both distances, and every returned distance, are measured from the original ray origin. A zero-distance hit is valid when `min_distance == 0`. A miss, including a ray whose only candidates are filtered out or closer than `min_distance`, reports `hit == False` and exactly `max_distance`. Equal-distance hits use the smaller public geom index as the deterministic tie-break.

`RayTraceFilter` is immutable and identical for every environment and ray in one call. Its constraints combine with AND semantics: `allowed_geom_ids` and `allowed_tags` are allowlists, `excluded_body_ids` removes geoms directly owned by those bodies, and `excluded_ancestor_body_ids` removes a geom when its owning body or any selected ancestor is excluded. Ancestor exclusion requires `body_parent_ids`; unknown identities or tags fail closed. The current contract deliberately has no literal bounded-continuation request: the default and only semantics are one nearest-valid-hit search. A future hop-for-hop compatibility capability, if needed, must be added as a separate capability and contract rather than overloading `min_distance`. An empty `RayTraceFilter()` is an explicit no-op and does not require `supports_static_filter`; any non-`None` constraint field makes the request a static filter.

Normals are optional on each transport. Mesh hits use the geometric face normal (no smooth/interpolated normal is implied), primitive hits use the corresponding side/cap normal, and the caster transforms the normal to world space and flips it when its dot product with the ray direction is positive. If the dot product is exactly zero, the finite geometric normal is returned without a flip. Degenerate triangles are rejected by the public mesh descriptor. `supports_host_normal` and `supports_tensor_normal` are separate capabilities so a plugin cannot accidentally promise an unimplemented tensor output. At an exact shared mesh edge, equal-distance faces can have different geometric normals; the geom identity and distance remain deterministic, but normal parity tests must exclude those measure-zero edge rays unless a caster declares a stronger face-index tie-break.

## Capabilities and fail-closed negotiation

`RayCasterCapabilities` declares:

- `supports_pose_sync` — `update_pose` after materialization.
- `supports_per_env_rays` — the 3-D per-environment ray profile.
- `supports_host_readback` — the NumPy host result path of `trace` (required by the current conformance helper).
- `supports_tensor_pose_sync` / `supports_tensor_rays` — the Torch-native pose and trace lifecycle.
- `supports_device_output` — persistent caster-owned Torch result buffers.
- `torch_devices` — accepted Torch device families or exact CUDA indices, using backend tensor-contract spelling.
- `stream_event_ownership` — ordering policy; the current profile is `caller` only.
- `supports_hit_point`, `supports_host_normal`, `supports_tensor_normal`, `supports_geom_id`, `supports_body_id` — the optional result fields, with normal split by transport.
- `supports_static_filter` / `supports_min_distance` — static geom/body/tag filtering and the inclusive lower distance bound.

Optional outputs are requested per call through `RayTraceOutputs`. Requesting an undeclared output, using the per-environment profile without `supports_per_env_rays`, or calling `update_pose` without `supports_pose_sync` raises `UnsupportedCapabilityError`; nothing is silently ignored or downgraded. Hit points and normals are world-space, normals are oriented against the ray direction, and `geom_id`/`body_id` index the scene description order (`-1` on missed rays). Optional fields are only meaningful where `hit` is true.

Capability construction is internally cross-validated: tensor pose sync implies pose sync, tensor rays imply per-environment rays, device output implies the complete tensor profile and a non-empty unique Torch-device list, and device output requires caller stream ownership. A caster declaring `cpu` only must reject CUDA tensors, and vice versa, without copying them to a supported device.

| Capability | Fake | `uniray` host revision | `uniray` tensor revision |
| --- | --- | --- | --- |
| Host NumPy pose/trace | yes | yes | retained for tools, debugging, and conformance |
| Tensor pose sync | CPU | no | CPU/CUDA as declared by the instance |
| Tensor rays | CPU | no | shared and per-environment |
| Persistent Torch output | CPU | no | caster-owned reusable views |
| Torch devices | `cpu` | undeclared | exact device accepted by the caster |
| Stream ownership | caller | n/a | caller |
| Hit point / geom ID / body ID | host and tensor | host | tensor |
| Normal | host and tensor | host | tensor |
| Static filter / minimum distance | host and tensor | host and tensor | host and tensor |
| Mesh / hfield | public descriptor + reference intersection | profile-specific | supported by the current `uniray` profile (hfields are cold-path triangulated meshes) |

## Reference implementation and conformance

`FakeRayCaster` implements analytic ray-vs-primitive and triangle-mesh intersection in NumPy and exposes a CPU Torch transport profile. Mesh geoms require the public `RayCollisionDescription` companion. It runs the core contract tests without MuJoCo, Warp, or any engine SDK. `assert_ray_caster_conformance(caster)` exercises lifecycle ordering, a canonical ground-plane scene with known analytic distances, capability-declared outputs, rich queries when declared, both ray profiles, selected rows, and fail-closed rejection — plugin authors should run it against their implementation.

## Plugin discovery

`RAY_CASTER_SPECS` declares plugin identities (`uniray` today). A plugin package must expose `create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster` at its top level; the UniSim factory imports it lazily, validates the returned object is a `RayCaster`, and reports an actionable `OptionalDependencyError` when the package or entry point is missing. Unknown caster names raise `ValueError`.
