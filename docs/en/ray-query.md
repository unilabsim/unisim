# Ray-Query Plugin Contract

[English](ray-query.md) | [中文](../zh/ray-query.md)

UniSim defines a backend-neutral ray-query plugin contract so ray caster implementations (such as the separately distributed `uni_ray` package) can serve batched ray queries to task code without exposing engine-private types. The first revision covers the minimal closed loop only: lifecycle, fixed-shape batched tracing, capability declaration, and fail-closed output negotiation.

## Public surface

All names are exported lazily-safe from the `unisim` package root and live in `unisim.ray_query`:

- `RayCaster` — the abstract plugin lifecycle: `materialize(scene)` → `update_pose(...)` → `trace(...)` → `close()`.
- `RaySceneDescription` — the backend-neutral structure-of-arrays scene consumed on the cold path.
- `RayCasterCapabilities` — the fine-grained capability declaration.
- `RayTraceOutputs` / `RayTraceResult` — the per-call output request and result containers.
- `RayGeomType` — the primitive geometry kinds (`plane`, `sphere`, `box`, `cylinder`, `capsule`, `ellipsoid`, plus reserved `mesh`).
- `require_ray_trace_outputs(...)` — the fail-closed output negotiation helper.
- `RAY_CASTER_SPECS` / `ray_caster_spec(...)` — the declared plugin manifest.
- `create_ray_caster(...)` — the lazy factory dispatch.
- `FakeRayCaster` — the pure-NumPy analytic reference implementation.
- `assert_ray_caster_conformance(...)` — the reusable conformance check for plugin authors.

## Type boundary

The public interface is NumPy-only. `mujoco.MjModel`/`mujoco.MjData`, Warp kernels or arrays, CUDA pointers, and every other backend-private type must never appear in signatures or results. Real adapters translate their native scene and pose state into `RaySceneDescription` and pose arrays on their own side of the boundary. Importing `unisim` never imports an engine SDK; plugin packages are imported lazily by `create_ray_caster` and a missing package or entry point fails closed with `OptionalDependencyError`.

## Scene description

`RaySceneDescription` carries one structure-of-arrays record per geom: `geom_types`, `geom_sizes` `(n, 3)`, `geom_local_pos` `(n, 3)`, `geom_local_quat` `(n, 4)` unit `wxyz`, and `geom_body_ids` `(n,)`, plus the explicit `num_bodies`. Geoms are expressed in their owning body's local frame. Sizes follow MuJoCo-style primitive conventions: sphere `(radius, 0, 0)`; box half extents `(x, y, z)`; cylinder and capsule `(radius, half_length, 0)` along local `+z`; ellipsoid semi-axes `(x, y, z)`; the plane is the infinite local `z = 0` plane with normal `+z` and ignores its sizes. All arrays are validated, detached, and write-protected at construction; invalid input fails closed.

## Lifecycle

1. `create_ray_caster(name, num_envs=..., num_rays=...)` fixes the batch shape `(num_envs, num_rays)` before any plugin import.
2. `materialize(scene)` binds the immutable scene geometry once on the cold path; every body pose starts as the identity. Materializing twice fails.
3. `update_pose(body_pos, body_quat, env_ids=None)` writes world-space body poses `(rows, num_bodies, 3)` / `(rows, num_bodies, 4)` for all rows or a validated selection, and requires `supports_pose_sync`.
4. `trace(ray_origins, ray_directions, max_distance, env_ids=None, outputs=None)` casts the fixed ray batch and returns a `RayTraceResult`.
5. `close()` releases resources idempotently; later queries fail.

## Trace semantics

Rays are world-space with unit directions. Two input profiles exist: the shared profile passes `(num_rays, 3)` arrays broadcast over every selected row; the per-environment profile passes `(rows, num_rays, 3)` arrays and requires `supports_per_env_rays`. `max_distance` is a positive finite scalar clipping distance. The minimal result is `distance` (clipped to `max_distance`; missed rays report exactly `max_distance`) and `hit`. `env_ids` selects a validated subset of environment rows; result rows follow the selection order.

Result arrays may be views into implementation-owned buffers reused by the next `trace` call; callers retaining results across calls must copy them. Implementations must not require unbounded per-call allocation.

## Capabilities and fail-closed negotiation

`RayCasterCapabilities` declares:

- `supports_pose_sync` — `update_pose` after materialization.
- `supports_per_env_rays` — the 3-D per-environment ray profile.
- `supports_host_readback` — the NumPy host result path of `trace` (required by the current conformance helper).
- `supports_device_output` — reserved for a future device-resident result path; enables nothing in this revision.
- `supports_hit_point`, `supports_normal`, `supports_geom_id`, `supports_body_id` — the optional `RayTraceResult` fields.

Optional outputs are requested per call through `RayTraceOutputs`. Requesting an undeclared output, using the per-environment profile without `supports_per_env_rays`, or calling `update_pose` without `supports_pose_sync` raises `UnsupportedCapabilityError`; nothing is silently ignored or downgraded. Hit points and normals are world-space, normals are oriented against the ray direction, and `geom_id`/`body_id` index the scene description order (`-1` on missed rays). Optional fields are only meaningful where `hit` is true.

## Reference implementation and conformance

`FakeRayCaster` implements the contract in pure NumPy with analytic ray-vs-primitive intersection for the six primitive types; mesh geoms are rejected fail closed at materialization. It runs the contract tests without MuJoCo, Warp, or any engine SDK. `assert_ray_caster_conformance(caster)` exercises lifecycle ordering, a canonical ground-plane scene with known analytic distances, capability-declared outputs, both ray profiles, selected rows, and fail-closed rejection — plugin authors should run it against their implementation.

## Plugin discovery

`RAY_CASTER_SPECS` declares plugin identities (`uni_ray` today). A plugin package must expose `create_ray_caster(num_envs=..., num_rays=..., **kwargs) -> RayCaster` at its top level; the UniSim factory imports it lazily, validates the returned object is a `RayCaster`, and reports an actionable `OptionalDependencyError` when the package or entry point is missing. Unknown caster names raise `ValueError`.
