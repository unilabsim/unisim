# ADR: Remove The Legacy NumPy SimBackend Lifecycle

[English](adr-legacy-numpy-removal.md) | [中文](../zh/adr-legacy-numpy-removal.md)

## Status

Accepted for `unisim-core` 2.0.0 (UniLab issue #2280).

## Context

The 1.x `SimBackend` interface grew from the pre-tensor UniLab runtime. It exposed NumPy stepping, whole-batch reset, generalized-state snapshots, base/DoF/body getters, and named sensor getters. UniLab's Manager runtime no longer uses that surface, and the accepted tensor lifecycle now provides public state, sensor, body, stepping, selected-reset, packed host-bridge, and packed reset boundaries.

Keeping both surfaces makes unsupported adapters look like host fallbacks and forces task code to preserve two authoritative carriers. It also conflicts with ADR-0015 in UniLab, which allows NumPy only behind explicit adapter, cold metadata, serialization, or playback leaves.

## Decision

Version 2.0.0 removes the legacy public training lifecycle without aliases. Version 1.7.15 is the final release that contains it. Private implementation names describe their owner instead of using `legacy`:

- `_host_*` names adapter-private CPU-authoritative state, control, reset, sensor, and body operations.
- `_wire_*` names a versioned external-worker protocol projection.
- `_model_file_*` names the still-supported `SceneCfg(model_file=...)` profile and its metadata.
- `_playback_*` names explicit playback snapshot layout and serialization boundaries.

Test-only private hooks and compatibility aliases are deleted rather than renamed. The repository blocks the `_legacy_*` prefix in runtime and test source.

| Removed 1.x surface | 2.0.0 replacement |
| --- | --- |
| `step(np.ndarray)` | `step_tensor(torch.Tensor, nsteps=...)` |
| `set_state(...)`, NumPy reset composition | `set_state_tensor(...)`, or `begin_packed_reset(...)` for staging, selected-column merges, baselines, model fields, and mocap |
| `get_state()`, `get_base_*`, `get_dof_pos/vel` | `get_state_views(...)` |
| `get_sensor_data*` | `get_sensor_view(...)`, `bind_sensor_data(...)`, or a negotiated sensor packet |
| Body getters and body-state copy helpers | `get_tracked_body_views(...)` or public tensor sensor projections |
| Scattered CPU host exchange | `compile_host_bridge_io(...)` and its plan boundaries |

NumPy remains valid behind explicitly scoped public leaves: cold model/default/layout metadata and index tables, declarative interval-DR inputs and pre-step callback formats, mapped entity transactions, playback/rendering APIs, physics-state snapshots, and serialization. These leaves do not restore a NumPy training lifecycle and must not be used as hidden fallbacks from tensor-capable Manager code.

The repository gates both the base contract and concrete backend classes against reappearance of the removed method names. It also gates runtime and test source against reintroducing the `_legacy_*` prefix.

## Consequences

- Consumers linked against the 1.x methods fail at attribute lookup rather than silently converting carriers.
- Tensor support is negotiated with `TensorLifecycleCapabilities`; unsupported lifecycles fail closed.
- CPU-authoritative adapters translate at explicit, measurable host-bridge or reset boundaries.
- Playback and rendering remain independently supportable even when the training lifecycle is tensor-only.

## Related Documents

- [Tensor lifecycle](adr-tensor-lifecycle.md)
- [Packed reset lifecycle](adr-packed-reset-lifecycle.md)
- UniLab [ADR-0015: No Numba runtime and explicit NumPy leaf boundaries](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md)
