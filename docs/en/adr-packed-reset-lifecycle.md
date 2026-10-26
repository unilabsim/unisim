# ADR: Packed Reset Lifecycle Transaction

[English](adr-packed-reset-lifecycle.md) | [中文](../zh/adr-packed-reset-lifecycle.md)

## Status

Accepted for the 1.x contract line (initial slice, #358).

## Context

UniLab's Torch-only Manager closeout left `ResetStateTransaction` as the last structural NumPy reset composer. It maintained host masks and buffers, merged selected-row state and disjoint-column model-field writes, recorded committed baselines, and finally called the lower-level `SimBackend.set_state_tensor()` boundary. That composition belongs to UniSim, not to task or Manager code.

The existing tensor contract already provides the correct commit primitives: selected-row qpos/qvel, device-resident final-value model-field scatters, persistent-channel clearing, time reset, and `AUTHORITATIVE_VIEWS` publication. The missing owner is the transaction that composes final values and owns committed baselines without a host mirror.

## Decision

UniSim owns a backend-neutral, staged packed reset transaction:

- `SimBackend.bind_packed_reset(device=None)` returns a backend-owned `PackedResetTransaction`, or fails closed.
- `TensorLifecycleCapabilities` adds `packed_reset`, `packed_reset_fields`, and `packed_reset_mocap`. Unsupported lifecycles must not declare them.
- The transaction owns full-width, device-resident committed baselines seeded from the backend's authoritative (fixed-variant aware) `get_reset_term_default()`.
- The fixed-width variant keeps the same staging width end to end: `begin_state_mask(mask)`, `write_state_mask(mask, ...)`, and `commit_state_mask(mask, ...)` consume a full-width bool mask, preserve false rows, and delegate selected-row semantics to the public backend mask boundary. This avoids owner-side `nonzero()` compaction without changing the staged-row contract above.
- `write_model_field(field, rows, values, columns=None)` merges final absolute selected-row values, including disjoint selected-column merges for terms such as PD `kp`/`kd`.
- `commit_state(rows, qpos, qvel)` performs exactly one public reset boundary by calling `set_state_tensor()` with the staged selected-row tables. Baselines are promoted only for the committed rows and only after that boundary returns successfully; staged writes for other rows remain until `discard()`.
- `discard(field=None)` explicitly drops staged writes before commit.
- The transaction also owns full-width device state staging: `set_state_default()` installs authoritative one-row defaults, `begin_state(rows)` restores them for the selected rows, `write_state(...)` merges selected columns, and `commit(rows)` submits exactly the staged rows. This preserves selected-reset exactness (`reset rows == written rows`) without a host mask.
- Persistent channels (`ctrl`, `act`, `qfrc_applied`, `xfrc_applied`, `qacc_warmstart`) and `time` keep the existing reset semantics of the underlying tensor boundary.
- `write_mocap_pose(body_name, rows, poses)` stages selected-row xyz/unit-wxyz poses. A combined commit applies generalized state first and then mocap poses; `commit_mocap_poses()` provides the existing mocap-only boundary without resetting generalized state.

### Adapter behavior

- MuJoCo (`HOST_BRIDGE`) accepts selected-row and full-width mask `TensorResetRandomizationPayload` forms inside the same explicit tensor-to-host reset boundary, synchronizing once before converting them into adapter-private host arrays. The persistent `HostBridgeTransferPlan.apply_reset()` remains the single-packet fast path for selected-row IDs.
- MotrixSim (`HOST_BRIDGE`) accepts the same selected-row and full-width mask forms for its runtime-negotiated field subset, converting the payload inside the host reset boundary and publishing selected rows authoritatively without a separate full-refresh phase.
- MJWarp (`DEVICE_RESIDENT`) composes on device and commits through existing device-side scatters with `AUTHORITATIVE_VIEWS` publication. `geom_size` stays excluded on the device plane because its derived bounds are host-computed today.
- Genesis and Newton (`DEVICE_RESIDENT`) commit selected-row state with narrow final-value field subsets: Genesis exposes `body_mass`, `kp`, and `kd` on both whole-MJCF and portable scenes under the [unified Genesis DR lifecycle](adr-genesis-dr-lifecycle.md); Newton exposes `kp` and `kd` through MuJoCo-family actuator arrays.
- IsaacGym and IsaacSim (`EXTERNAL_WORKER` / `CUDA_IPC`) stage selected-row `kp`/`kd` in device arenas. Their workers own one explicitly named reset-only property bridge for CPU-native engine setters; step/state/control/sensor hot paths never enter that bridge. The backend-owned mask boundary may compact true rows behind the public `set_state_tensor_mask()` contract.
- MuJoCo declares packed reset mocap support for kinematic entity roots; the selected Torch rows and poses cross its explicit host boundary only at commit. MJWarp keeps mocap writes fail-closed in this slice.
- Unsupported fields, fixed variants, model-file tensor lanes, and every other adapter remain fail-closed; there is no backend-private probing.

### Explicit exclusions

- Owner-subset dynamic-row APIs (for example motion-wrap mutation) remain separate owner APIs rather than overloading the reset row set.

### Delta terms

- `base_mass_delta` and `base_com_offset` are first-class fields of the single torch payload family since UniLab issue #2410; the former host-only restriction and `PACKED_RESET_HOST_ONLY_FIELDS` are removed. The transaction stages the sampled deltas and merges them with the authoritative baseline — seeded from the cold NumPy metadata leaf `get_reset_term_default()` — into final absolute `body_mass` / `body_ipos` tables in torch at commit: `final = baseline + delta`. Adapter-internal delta merges may remain as adapter-private implementations, but the public payload no longer restricts the delta fields.

## Consequences

- Downstream Managers can delete their NumPy mask/buffer and committed-baseline arrays for model-field reset composition.
- A failed commit cannot pollute later baselines because promotion happens only after success.
- Model-field mutation retains existing fixed-graph invalidation semantics; state and sensor views remain authoritative immediately after the public commit returns.
- The whole transaction reports `packed_reset_compose_ms`, `packed_reset_commit_ms`, and `packed_reset_total_ms` in addition to adapter-specific reset timings.

## Future Work

- Compose the transaction with multi-step/CUDA-graph execution while keeping graph-unsafe field mutation fail-closed under graph invalidation.
