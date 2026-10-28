# ADR: Torch-First Domain-Randomization Payloads

[English](adr-dr-torch-first-payloads.md) | [中文](../zh/adr-dr-torch-first-payloads.md)

## Status

Accepted (2026-10-19, UniLab issue #2410).

## Context

The full-stack DR audit (UniLab `DR_AUDIT.md`, 2026-10-19) found that the public domain-randomization contract carried two parallel payload carriers. A NumPy `ResetRandomizationPayload` served the host plane while `TensorResetRandomizationPayload` served device-resident adapters, and `require_op_host_payload` gave interval ops a public host-payload contract. The same `push` interval term had two payload semantics — a sampling range on host adapters that sampled through backend-internal global RNG (`np.random`), and a producer-sampled final force table on the MJWarp device path — so Manager seeds did not govern host pushes. `base_mass_delta` and `base_com_offset` were host-only fields excluded from device payloads (`PACKED_RESET_HOST_ONLY_FIELDS`), forcing owners to pre-compose deltas into absolute tables. `DomainRandomizationCapabilities` advertised a single term set per adapter, hiding that host and device planes support different subsets (MJWarp `geom_size`, Isaac host-mapped versus CUDA IPC lanes). Backend conformance never asserted that a declared reset term actually changed a model field. On the UniLab side, CUDA-sampled reset values paid a D2H→H2D round trip through CPU-tensor enforcement before commit.

## Decision

The public DR contract is torch-first and torch-only:

- `ResetRandomizationPayload` (NumPy) and `require_op_host_payload` are removed from the public contract. `TensorResetRandomizationPayload` is the sole public reset payload; it keeps its class name and gains the `base_mass_delta` / `base_com_offset` fields, covering all 16 reset terms. `IntervalTermOp.payload` and `TensorIntervalPayload` are the torch-first interval carriers; `body_ids` are torch-first as well.
- Payload fields stay `Any`-typed with lazy torch validation (`validate_tensor_reset_payload`, `validate_interval_op`) so `unisim-core` keeps its NumPy-only dependency footprint: the base package never imports torch at module top level.
- Payload tensors remain on the caller-selected Torch device. The public contract does not expose host/device execution-plane negotiation or a configuration switch.
- Following UniLab issue #2550, `DomainRandomizationCapabilities` has one `supported_reset_terms` set and one `supported_interval_terms` set. The sets describe all terms the backend can implement across its private host and device paths. `DataPlaneName`, device-specific term sets, and plane parameters and helpers are removed. Undeclared terms fail closed by name; aggregate capabilities use only `dr.reset.*` and `dr.interval.*`.
- Delta terms are no longer host-only. The packed reset transaction stages `base_mass_delta` / `base_com_offset` and merges them with the authoritative baseline — seeded from the cold NumPy metadata leaf `get_reset_term_default()` — into final absolute `body_mass` / `body_ipos` tables in torch at commit (`final = baseline + delta`). `PACKED_RESET_HOST_ONLY_FIELDS` is deleted.
- Push is producer-sampled on every backend, following the MJWarp device pattern. Backend-internal global-RNG push sampling (MuJoCo, Motrix) is deleted; a host adapter that receives no producer-sampled push payload fails closed.
- NumPy survives only behind explicit boundaries: the adapter-private `unisim.dr._host` conversion helpers used by host-boundary adapters (MuJoCo, Motrix, IsaacGym, IsaacSim, subprocess IPC), cold metadata such as `get_reset_term_default()`, and serialization/playback leaves.
- Backend conformance gains teeth: every adapter that declares reset randomization must demonstrably round-trip at least one declared reset term — write a non-baseline value, commit, and read the baseline back in a new transaction with an allclose assertion.

## Consequences

- Consumers constructing NumPy reset or interval payloads must build torch tensors instead. They query only unified term support and do not select a capability plane; adapters still validate the actual payload device at their private execution boundaries.
- CUDA-resident reset DR no longer pays a D2H→H2D round trip on the UniLab side; event terms submit device tensors directly.
- Device-specific implementation differences no longer fragment public capability declarations. Host bridges convert internally, and device-resident adapters may use a private host fallback for a declared term, as MJWarp does for `geom_size` and its derived bounds.
- Adapter-private NumPy conversion is centralized in `unisim.dr._host` and remains private: it is excluded from `__all__` and is not a public fallback for Manager code.
- Genesis is not a host-boundary adapter and does not import `unisim.dr._host`; its unified runtime DR lifecycle consumes Torch tensors directly on the backend-bound CUDA device and continues to reject CPU payloads.
- Declared-but-ineffective reset terms are caught by conformance instead of by downstream observation drift.

## Related decisions

- [Tensor lifecycle ADR](adr-tensor-lifecycle.md) defines the optional fail-closed tensor lifecycle this payload family plugs into.
- [Packed reset lifecycle ADR](adr-packed-reset-lifecycle.md) defines the transaction that composes and commits the torch payloads.
- [Unified Genesis DR lifecycle ADR](adr-genesis-dr-lifecycle.md) applies the device-resident contract uniformly to Genesis whole-MJCF and portable scenes.
- [Legacy NumPy removal ADR](adr-legacy-numpy-removal.md) defines the remaining public NumPy leaves.
- UniLab [ADR-0015: No Numba runtime and explicit NumPy leaf boundaries](https://github.com/Motphys/UniLab/blob/main/docs/sphinx/source/adr/ADR-0015-no-numba-and-explicit-numpy-leaf-boundaries.md) constrains NumPy to explicit leaf owners on the consumer side.
