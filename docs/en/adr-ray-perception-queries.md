# ADR: Perception-Ready Ray Geometry and Rich Queries

[English](adr-ray-perception-queries.md) | [中文](../zh/adr-ray-perception-queries.md)

## Status

Accepted for the 2.x ray-query contract (#472).

## Context

The tensor-native ray lifecycle from #462 removed host detours, but portable perception callers still could not obtain mesh topology or express a nearest valid hit. Mesh points and faces lived in `uni_ray`'s private `MjBatchCollision`; callers therefore retained a native model or called a backend-specific builder. Real sensors also need initialization-time geom/body/tag exclusions, a minimum hit distance to reject self-hits, and optional camera normals. Literal hop-by-hop continuation exists only as an application compatibility behavior and is not required by a nearest-valid-hit query.

## Decision

Keep `RaySceneDescription` as the compact structure-of-arrays pose/geom contract and add a detached `RayCollisionDescription` companion. The companion carries stable geom instance names, opaque semantic tags, deduplicated `RayMeshDescription` assets, geom-to-mesh mapping, and optional acyclic body ancestry. Geom instance identity and mesh asset identity are separate. Both are stable only within one descriptor/materialization lifecycle; rebuild creates a new descriptor and invalidates the old identities. No native model, native mesh, Warp array, pointer, stream, event, or USD prim appears in the public type.

Tags are opaque strings with no universal vocabulary. Adapters may emit source-qualified tags (the MuJoCo-family adapter emits `mujoco:geom_group:<n>`), but the contract does not define a universal tag set and callers discover tags through the public descriptor. `RayTraceFilter` consumes public geom indices, tags, direct owning-body IDs, and ancestor body IDs. Its constraints are call-static, shared by all rows and rays, and combined with AND semantics. `excluded_body_ids` means direct geom ownership; `excluded_ancestor_body_ids` means the owning body or any selected ancestor. Parent IDs describe ancestry within the selected public body axis; an omitted native ancestor becomes a selected-axis root. Missing ancestry, unknown identities, and unknown tags fail closed.

`trace` and `trace_tensor` accept `min_distance` and `ray_filter`. A valid hit satisfies the inclusive lower bound and exclusive upper bound:

```text
min_distance <= distance < max_distance
```

The lower bound must be strictly less than the upper bound. All distances are relative to the original ray origin. Zero-distance hits are valid when the minimum is zero, misses report exactly `max_distance`, and equal-distance hits use the smaller geom index. The default implementation is one nearest-valid-hit traversal; this contract does **not** add literal bounded continuation, epsilon advancement, or hop exhaustion. If exact hop-for-hop compatibility is needed later, it must be introduced as a separate capability and ADR rather than overloading `min_distance`.

Normals remain optional and are split by transport: `supports_host_normal` and `supports_tensor_normal`. Mesh normals are finite geometric face normals, not smooth normals. Primitive normals identify the intersected side or cap. The caster returns world-space normals and flips a normal when its dot product with the ray direction is positive; a zero dot product leaves the finite geometric normal unchanged. Public mesh descriptors reject zero-area triangles so a normal is always defined. Exact shared mesh edges can expose either adjacent face normal; distance and geom identity remain deterministic, but normal parity excludes those measure-zero rays unless a future contract adds a face-index tie-break.

Sensor attachment remains body-only in this phase. A task applies its local sensor offset in application code and supplies body poses through the existing public tensor pose contract. Site and geom frame views remain out of scope. Public terrain projection is tracked separately in #474: this ADR covers collision geometry and ray queries only.

Host and tensor implementations must preserve the same hit, distance, identity, filter, and normal semantics within their documented floating-point tolerance. The tensor path keeps persistent caster-owned outputs, caller stream ownership, no implicit host readback, and no unbounded steady-state allocation.

## Consequences

Portable callers can materialize a caster from the public descriptor without `MjBatchCollision`, a native model, or a backend-specific builder. Adapters own the cold-path projection and fail closed when runtime geometry, identity, or ancestry cannot be represented. Applications retain grouped ray patterns, camera intrinsics, noise, history, scheduling, and other perception logic. A future bounded-continuation contract can be added without changing the nearest-valid-hit default.

MuJoCo and MJWarp fixed-variant scenes are not projected by this first collision descriptor because a variant may change per-environment geometry; those adapters fail closed. Motrix reuses its audited ray-source selection and continues to reject runtime-only geometry, attached fragments, per-row mesh variants, heterogeneous fixed-variant sources, and source hfield/mesh geometry until native introspection or the terrain follow-up defines an exact projection.

## Related decisions

- [Ray tensor lifecycle ADR](adr-ray-tensor-lifecycle.md) defines pose ownership, caller streams, and persistent tensor outputs.
- [Tensor lifecycle ADR](adr-tensor-lifecycle.md) defines backend tensor execution and packed host-bridge plans.
- [Terrain projection roadmap](https://github.com/unilabsim/unisim/issues/474) tracks public hfield/terrain metadata separately.
