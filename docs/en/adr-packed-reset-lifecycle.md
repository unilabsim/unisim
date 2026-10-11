# ADR: Packed Reset Lifecycle Transaction

[English](adr-packed-reset-lifecycle.md) | [中文](../zh/adr-packed-reset-lifecycle.md)

## Status

Accepted for the 1.x contract line (initial slice, #358).

## Context

UniLab's Torch-only Manager closeout left `ResetStateTransaction` as the last structural NumPy reset composer. It maintained host masks and buffers, merged selected-row state and disjoint-column model-field writes, recorded committed baselines, and finally called the lower-level `SimBackend.set_state_tensor()` boundary. That composition belongs to UniSim, not to task or Manager code.

The existing tensor contract already provides the correct commit primitives: selected-row qpos/qvel, device-resident final-value model-field scatters, persistent-channel clearing, time reset, and `AUTHORITATIVE_VIEWS` publication. The missing owner is the transaction that composes final values and owns committed baselines without a host mirror.

## Decision

UniSim owns a backend-neutral, staged packed reset transaction:

- `SimBackend.begin_packed_reset(device=None)` returns a backend-owned `TorchPackedResetComposer`, or fails closed.
- `TensorLifecycleCapabilities` adds `packed_reset`, `packed_reset_fields`, and `packed_reset_mocap`. Unsupported lifecycles must not declare them.
- The composer owns full-width, device-resident committed baselines seeded from the backend's authoritative (fixed-variant aware) `get_reset_term_default()`.
- `write_model_field(field, rows, values, columns=None)` merges final absolute selected-row values, including disjoint selected-column merges for terms such as PD `kp`/`kd`.
- `commit_state(rows, qpos, qvel)` performs exactly one public reset boundary by calling `set_state_tensor()` with the staged selected-row tables. Baselines are promoted only for the committed rows and only after that boundary returns successfully; staged writes for other rows remain until `discard()`.
- `discard(field=None)` explicitly drops staged writes before commit.
- The transaction also owns full-width device state staging: `set_state_default()` installs authoritative one-row defaults, `begin_state(rows)` restores them for the selected rows, `write_state(...)` merges selected columns, and `commit(rows)` submits exactly the staged rows. This preserves selected-reset exactness (`reset rows == written rows`) without a host mask.
- Persistent channels (`ctrl`, `act`, `qfrc_applied`, `xfrc_applied`, `qacc_warmstart`) and `time` keep the existing reset semantics of the underlying tensor boundary.
- `write_mocap_pose(body_name, rows, poses)` stages selected-row xyz/unit-wxyz poses. A combined commit applies generalized state first and then mocap poses; `commit_mocap_poses()` provides the existing mocap-only boundary without resetting generalized state.

### Adapter behavior

- MuJoCo (`HOST_BRIDGE`) accepts `TensorResetRandomizationPayload` inside the same explicit tensor-to-host reset boundary, synchronizing once before materializing the host payload. The persistent `HostBridgeTransferPlan.apply_reset()` remains the single-packet fast path.
- MJWarp (`DEVICE_RESIDENT`) composes on device and commits through existing device-side scatters with `AUTHORITATIVE_VIEWS` publication. `geom_size` stays excluded because its derived bounds are host-computed today.
- MuJoCo declares packed reset mocap support for kinematic entity roots; the selected Torch rows and poses cross its explicit host boundary only at commit. MJWarp keeps mocap writes fail-closed in this slice.
- Every other adapter remains fail-closed; there is no backend-private probing.

### Explicit exclusions

- `base_mass_delta` and `base_com_offset` remain host-only payload fields, but their reset semantics do **not** require a device delta representation: owners compose the authoritative baseline plus the sampled delta into final absolute `body_mass` / `body_ipos` device tables before staging them in the transaction. Passing the delta fields themselves through a device payload fails closed; no hidden host fallback or adapter-private derivation is introduced.
- Owner-subset dynamic-row APIs (for example motion-wrap mutation) remain separate owner APIs rather than overloading the reset row set.

## Consequences

- Downstream Managers can delete their NumPy mask/buffer and committed-baseline arrays for model-field reset composition.
- A failed commit cannot pollute later baselines because promotion happens only after success.
- Model-field mutation retains existing fixed-graph invalidation semantics; state and sensor views remain authoritative immediately after the public commit returns.
- The whole transaction reports `packed_reset_compose_ms`, `packed_reset_commit_ms`, and `packed_reset_total_ms` in addition to adapter-specific reset timings.

## Future Work

- Extend final-value device representations for base mass/COM.
- Compose the transaction with multi-step/CUDA-graph execution while keeping graph-unsafe field mutation fail-closed under graph invalidation.
