# Design decision: semantic capabilities and import evidence

[English](adr-capabilities.md) | [中文](../zh/adr-capabilities.md)

## Context and decision

Adapter installation, supported semantics, verification and actual adopted settings are separate concerns. A backend being importable does not prove that a semantic feature is supported, and a supported feature does not determine the effective settings of a constructed backend.

`CapabilityReport` holds a `CapabilityScope` and immutable `CapabilityDeclaration` records. Each declaration has a dotted feature key, `SupportLevel` (`exact`, `approximate`, `unsupported`, `unknown`), a reason, optional `CapabilityCondition` constraints and independent `CapabilityEvidence`. `get_adapter_capabilities(name, profile="default")` reads source declarations without importing or discovering SDKs. Missing features and unsatisfied conditions return unknown. `SimBackend.get_capabilities()` adds existing DR/play/fixed-variant declarations from their authoritative APIs instead of maintaining another registry.

## Evidence and version matching

Evidence records are source reviews or runtime results. A runtime result only verifies an exact, complete match of adapter, profile, UniSim version, adapter version/commit, engine version, platform and device with runtime availability explicitly true. Unknown values are `None`; missing SDKs, missing identity fields, failed/skipped runs, different versions/profiles and source-only evidence never count as runtime verified. An approximation remains an approximation after passing a runtime check. Constructor success does not create broad verification evidence.

To add evidence, record the precise command, small asset, revision (and dirty state), runtime/SDK, device, numerical tolerance and result, then attach a `CapabilityEvidence(kind="runtime", ...)` only to exercised features in that exact scope. To update it, add a new scope-bound record; do not widen its version range or silently reuse old evidence. To withdraw evidence, retain its source and set `revoked=True`. The initial inventory retains only pinned source reviews; diagnostic JSON is reviewable evidence and is not automatically promoted into declarations.

## Configuration reports and lifecycle

`SimBackend.get_import_report()` returns an immutable `ImportReport` cached on construction/materialization. Its `ConfigurationField` records requested/effective values, difference (`exact`, `overridden`, `approximate`, `unknown`, `not_applicable`), provenance, units, frame and `ConfigurationScope` (entity, environment IDs and variant). Source declarations, engine readback, adapter settings and unverified values remain distinct. Missing effective values are never filled with source values. Solver/integrator names retain their engine meaning; shared names do not promise numerical equivalence. `to_dict()`/`from_dict()` round trips contain no SDK objects.

Reports cover solver, integrator, timestep, gravity, actuator mapping, collision filters, body mass/inertia and sensors. MuJoCo reads compiled model settings; MJWarp distinguishes host source from effective device options. Isaac workers return a versioned configuration envelope with their own provenance; host XML is not worker readback. Unavailable readback remains unknown. Canonical/variant rows are scoped rather than generalized from environment zero. Reports are initial snapshots: reset-time DR/current property getters remain authoritative for current per-environment values. Reading a report never reparses assets or steps physics.

## Validation and compatibility

`SemanticRequirements` selects feature keys and report setting keys, profile, conditions and individually authorized approximation keys. `validate_semantic_requirements()` checks declarations/reports without loading an SDK. `create_backend(..., semantic_requirements=...)` runs declaration preflight, completes materialization, binds conditions to actual configuration and checks requested report fields before returning. Unknown/unavailable semantics and unauthorized approximations fail closed with backend/profile/field diagnostics; initialized resources close on a failed validation. Overrides such as an explicit factory timestep remain visible in the report. Fixed worker gravity conflicts require approximation consent; incomparable native solver/integrator labels remain unknown and cannot pass a strict setting requirement. `require_runtime_verified=True` additionally requires matching attached runtime evidence, which source-only inventory cannot satisfy.

Existing callers that omit semantic requirements retain their existing lifecycle and adapter audits. The strict opt-in path is a staged migration, not a claim that all older import paths are completely audited. Strict construction already materializes the backend; do not call `materialize()` again. Existing SuperDex approximation opt-in and IsaacSim contact refusal remain enforced. No engine fallback, solver substitution, new importer IR, runtime SDK dependency or hot-path XML parsing is introduced.

Configuration conditions supplied with an import report are checked against every applicable effective field by the standalone validator as well as the factory. Adapter options enter through adapter-owned report fields; generic validation never probes private backend state or treats caller kwargs as adopted values. Consumers that apply Manager-Based startup events before materialization should retain that ordering and validate after their ordinary materialization, rather than opt into early strict construction.
