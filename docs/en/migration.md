# UniLab Migration

[English](migration.md) | [中文](../zh/migration.md)

The migration is staged by backend. Each adapter child moves implementation and documentation together, adds optional dependency diagnostics and conformance coverage, and updates the UniLab consumer boundary. The former `unilab.base.backend` re-export shim has been removed; there is one production implementation owned by `unisim-core`.

MuJoCo is the first in-process adapter. It accepts a package-neutral `SceneCfg`, materializes XML on construction, and exposes cached numeric state through `unisim.SimBackend`; task-owned scene composition remains in UniLab. Its native batch executor is mjbatch (`unilabsim/mjbatch_uni`, a maintained fork of `kevinzakka/mjbatch`); heterogeneous model variants are unsupported, and field-level domain randomization goes through mjbatch `expand` and `set_const`.

## Portable MJCF scenes

Explicit `SceneCfg.entity_assets`, mirrors and entity-bound variants use the cold portable MJCF profile described in the [ADR](adr-portable-mjcf.md). UniSim expands and compiles those declarations once, freezes the public entity layout, and records source provenance, source intent and content identity. Install `unisim-core[scene-compiler]` to use the compiler; it does not require the MuJoCo adapter's mjbatch executor. Legacy `model_file` and explicit native profiles remain supported as adapter paths and are not portable-profile claims.

For entity scenes, `fragment_files` is limited to sensor-only MJCF fragments. The portable compiler accepts ordered cross-entity `contact data="force" reduce="netforce"` and `contact data="found" num="1"` declarations with final `entity/local-name` geom addresses, includes their bytes in content identity, and rejects all other fragment authoring.

Motrix is the second in-process adapter. It uses Motrix's batched `SceneData` and masked data slices behind the same public state, control, and reset contract.

The remaining UniLab identities are represented in UniSim as first-class adapters: Drake, MJWarp, Genesis, Newton, SuperDex, IsaacGym, and IsaacSim. The latter two reuse `unisim.backend.subprocess_ipc` and resolve their vendor workers without importing Kit or Python 3.8 modules into the host process. Missing SDKs are reported at construction time; no backend is silently downgraded to another engine.

Runtime-owned caches and worker installations use `UNISIM_*` environment variables and `~/.cache/unisim` defaults. The previous `UNILAB_*` names are accepted only as migration fallbacks so existing installations can move without losing cached state.

## Semantic requirements

Existing construction keeps its original lifecycle and adapter audits. To migrate a task, first inspect `get_adapter_capabilities("mujoco")`, then explicitly request the semantic features and adopted configuration fields it needs. Strict construction completes `materialize()` before returning, so remove the separate materialization call on this path. Every requested unknown or unsupported feature fails closed; approximation consent names specific keys and does not suppress other adapter checks.

```python
from unisim import SemanticRequirements, create_backend, get_adapter_capabilities
from unisim.scene import SceneCfg

static = get_adapter_capabilities("mujoco")  # No SDK discovery or import.
backend = create_backend(
    "mujoco", SceneCfg(model_file="robot.xml"),
    semantic_requirements=SemanticRequirements(
        features=("asset.mjcf", "actuator.motor"),
        settings=("dt", "gravity", "body_mass"),
    ),
)
initial_configuration = backend.get_import_report().to_dict()
```

Use `backend.get_capabilities()` to aggregate existing DR/play/variant authorities. Read the import report as an initial configuration snapshot, not current values after reset randomization. `require_runtime_verified=True` refuses source-only evidence; SDK presence or construction success never automatically verifies every feature. Configuration conditions must match effective report values, including options recorded by the adapter; invented context cannot authorize another profile. See the [ADR](adr-capabilities.md) for exact limits.

For a Manager-Based consumer that applies startup events before materialization, keep ordinary construction and the existing startup/materialization order. After materialization, call `validate_semantic_requirements(backend.get_capabilities(), requirements, backend.get_import_report())` before stepping. This public validator applies the same configuration-condition checks as strict factory construction without moving startup events across the materialization boundary. Initial reports do not replace current-property queries after DR.
