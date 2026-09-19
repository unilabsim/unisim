# Design decision: portable MJCF scene compilation

[English](adr-portable-mjcf.md) | [中文](../zh/adr-portable-mjcf.md)

## Status, scope and owners

**Status: Accepted for profile v1.** This decision extends, and is bounded by, the [entity and immutable-identity decision](adr-entities.md) and the [capability and evidence decision](adr-capabilities.md). It covers cold-path authoring, structural compilation, canonical identity and source intent; native materialization, effective reports and execution remain adapter-owned.

UniSim owns the portable profile, source/resource resolution, canonical expanded MJCF, `CompiledSceneLayout`, assignment semantics and content identity. MuJoCo `MjSpec` is only the structural oracle that resolves the restricted profile. Adapters consume the common result and translate it through public engine APIs; they do not create a second composer or identity scheme. UniLab continues to own asset registration, task configuration, Manager scheduling, policy I/O and training.

## Context

M2 made entities, mirrors, immutable variant assignment and selected reset public, while the working composer lived in the MuJoCo adapter. Isaac workers already reuse that composer, but its temporary paths and adapter placement are not a stable cross-backend contract. Roadmap [#154](https://github.com/unilabsim/unisim/issues/154) requires one trusted authoring path before cache and native backend extensions.

## Decision

Portable profile v1 uses restricted MJCF as the only physical authoring source. `SceneCfg.entity_assets`, `mirror_of`, `entity_variant` and `FixedVariantPlan.assignment` define all entity, mirror and variant relations. Source-file organization, body order and appearance never infer ownership. `SceneCfg.fragment_files` may add only scene-level, sensor-only MJCF fragments for cross-entity collision-pair force declarations; entity sources themselves remain independently valid MJCF.

The common cold path:

1. parses each entity source with MuJoCo `MjSpec`;
2. rejects semantics outside profile v1 before attachment;
3. namespaces entities, merges keyframes by compiled public addresses and requires compatible global options;
4. compiles every unique variant independently;
5. reads body, joint, geom, site, actuator, qpos and qvel addresses from the final compiled model into `CompiledSceneLayout`;
6. emits serialized expanded MJCF for adapter materialization; and
7. emits source provenance, a source/intent report and a versioned content identity.

The structural oracle is lazy and uses the `scene-compiler` extra (`mujoco~=3.11.0`). It does not require the `mjbatch` executor used by the MuJoCo adapter. Importing UniSim and the SDK-free contract module loads no engine. The legacy whole-model `model_file` entry point and explicit native profiles remain adapter paths; they are not portable-profile claims.

### Restricted profile v1

Each entity has one named root body and no world-body geometry or geoms. Bodies must be named except for the source root. Root mobility is one root free joint for floating entities, no joint for fixed entities, and a compiler-generated mocap root for kinematic mirrors or kinematic rigid entities. Non-root hinge, slide and ball joints must be named. Rigid entities cannot contain non-root joints; kinematic articulations are unsupported.

Body geometry, explicit inertials, meshes, textures, hfields, named keyframes, contact declarations, source-local sensors and joint-transmission actuators are eligible for the profile when the structural oracle accepts their combination. A scene-level sensor fragment may contain only ordered `contact data="force" reduce="netforce"` or `contact data="found" num="1"` declarations whose geom references use the final `entity/local-name` namespace; world-referenced `framepos`/`framequat` declarations whose body/site object references use that namespace; or world-referenced `framelinvel`/`frameangvel` declarations whose object is a qualified body. The compiler resolves all of them after every entity is attached and before each variant is compiled. Tendons, equalities, non-joint transmissions, cross-entity constraints, source `<include>` documents, inline asset overrides, non-default compiler transforms, differing global options, ambiguous keyframe names or times and variant topology or sensor changes fail closed. Unsupported or unverified native semantics are never silently dropped; an adapter must reject them or record each explicit approximation in its effective report.

Mirrors are collision-free and control-free visual roles. They inherit source and selected variant identity, never the target pose or physical influence.

### Identity and reports

`SceneContentIdentity` schema version 1 hashes every source's entity role, kind, root/collision role, initial pose, format, mirror relation, variant role and source bytes; referenced mesh, texture and hfield logical paths and bytes; every scene-level sensor fragment's bytes in declaration order; and profile identity, structural-oracle identity and version, timestep, keyframe selection and immutable assignment.

It deliberately excludes absolute checkout paths and adapter/runtime settings. Absolute locations remain in provenance. USD cache owners extend the canonical identity with importer parameters and Isaac/importer/runtime versions; they never replace it. Role baking likewise derives a separate identity from the raw artifact identity plus collision, visual, mirror and bake parameters.

`SceneIntentReport` is serializable and contains only source intent and provenance. It cannot carry native effective values. After materialization each adapter supplies `ImportReport` effective fields from actual native readback, with backend, runtime, profile, configuration and lifecycle scope. Cross-engine numerical equality is not an acceptance criterion.

## Alternatives considered

- **A backend-specific composer per adapter** was rejected because it duplicates identity and namespace rules and makes cache correctness backend-dependent.
- **URDF as portable source** was rejected for v1: URDF remains authoring vocabulary or an explicit translated/native compatibility profile, while MJCF carries the physics semantics this roadmap must compare.
- **USD as truth** was rejected: USD is a materialization and cache artifact, not the physical authoring contract.
- **A universal asset IR** was rejected as unnecessary scope; versioned expanded MJCF plus the frozen public layout and reports is the smallest complete bridge.

## Verification boundary

Focused tests cover SDK-free report and identity schemas, relocation invariance, resource, sensor-fragment and compiler invalidation, downstream artifact-identity extension, missing-compiler diagnostics, source `<include>` rejection, and the golden robot, passive object, table and mirror scene with N5 assignment `[1,1,0,1,0]`. They do not claim native support for a backend. Each adapter must materialize the common result, read back effective identity and configuration, and pass its own native tests before extending its support matrix.

`SceneCfg.fragment_files` adds only scene-level, sensor-only MJCF fragments for cross-entity collision-pair force declarations or world-referenced body/site pose declarations using final qualified names. Entity sources remain independently valid, fragment bytes participate in canonical identity in declaration order, and all other fragment authoring fails closed.
