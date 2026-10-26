# ADR: Unified Genesis Device-Resident DR Lifecycle

[English](adr-genesis-dr-lifecycle.md) | [中文](../zh/adr-genesis-dr-lifecycle.md)

## Status

Accepted for the 1.x contract line (UniLab issue #2520).

## Context

Genesis previously treated domain randomization differently by scene materializer. The reviewed whole-MJCF CUDA lane exposed device-resident reset fields, while portable `entity_assets` scenes carried host reset setters and a portable-only pending wrench path. That split advertised a broad portable reset set (`base_mass_delta`, COM, DOF, and actuator fields), converted hot operands through host arrays, and allowed reset/interval work to depend on the scene mode. It also made CPU sessions appear to support runtime DR even though those host setters had no audited lifecycle, and aggregate capability reports did not project the device-only term sets.

The portable materializer already owns the immutable public-to-native mappings needed for a device path: entity body IDs, native link indices, actuator columns, and variant-assigned cold defaults. Genesis 1.3.3 exposes public Torch setters for selected-row mass/gains and public solver wrench submission APIs on its zero-copy CUDA/AMDGPU lane.

## Decision

Genesis has one runtime DR lifecycle, and it is device-resident:

- Both whole-MJCF and portable scenes use the exact `gs.cuda` + non-HIP Torch CUDA or exact `gs.amdgpu` + HIP/ROCm Torch pairing, Genesis zero-copy, and the public Torch lane. CPU, Metal, runtime-family mismatch, and zero-copy-disabled sessions declare no runtime DR.
- Reset randomization is the selected-row final-value subset `body_mass`, actuator `kp`, and actuator `kd`. Validation and commit stay on Torch tensors on the accepted CUDA device. Portable rows use cached device-resident public and native index tables and are scattered through audited owning native entities. Host/CPU reset payloads fail closed naming the term and plane.
- Interval randomization is the world-frame `body_force` and `body_torque` subset. Torch int64 audited public body IDs select cached native solver links; float32 CUDA force/torque operands are validated for shape, dtype, device, finiteness, and physical ownership, then submitted at each link COM for the upcoming native step. Undeclared device-plane interval terms fail closed by name.
- A new interval plan cancels unconsumed wrenches before its ops accumulate. Direct submissions and multiple ops in one plan accumulate independently. A selected tensor or entity reset cancels matching rows/bodies before Genesis' native state setter clears solver staging; unrelated rows and bodies persist. Successful stepping consumes and clears the pending mirror.
- The cold boundary is explicit: startup, scene materialization, variant materialization, immutable public/native index construction, and cold default tables. Hot reset and interval execution performs no host payload detour.
- Legacy portable host reset/interval paths and their capability advertising are removed without aliases or time-based compatibility names. Portable host-only setters are not runtime DR and are not advertised. NumPy remains only at cold metadata/serialization leaves.
- `get_dr_capabilities()` is authoritative. Without the device lifecycle, and therefore on the host plane in every session, its term sets are empty. An active session populates only `supported_device_reset_terms` and `supported_device_interval_terms`; the host sets remain empty. Aggregate `get_capabilities()` projects these planes independently as `dr.device.reset.*` and `dr.device.interval.*`, while the pre-existing `dr.reset.*` and `dr.interval.*` keys continue to describe only the host plane.
- The declared portable reset set is narrowed accordingly: COM, base delta, DOF friction/damping/armature, geometry, gravity, and other reset terms are unsupported on both planes until a reviewed device lifecycle and acceptance evidence exist.

## Evidence

- Real CUDA acceptance covers whole-MJCF and portable scenes in `tests/adapters/genesis/test_tensor_runtime.py`. The portable test materializes a real multi-entity/variant scene, checks the empty host and populated device capability sets plus their separate aggregate key families, mutates only selected rows through selected-row and mask resets, verifies direct and plan-driven wrench staging plus step consumption/clearing, and exercises shape, finiteness, row/body range, ownership, dtype, and non-Torch validation failures on the accepted device.
- A separately masked CPU process (`CUDA_VISIBLE_DEVICES=""`) verifies that portable capability and tensor-DR declarations are empty and CPU/host requests fail closed. Skipped tests are not acceptance evidence.

## Consequences

- Consumers cannot move portable Genesis DR to CPU or torch.cpu operands. Owners must sample on the accepted CUDA device and use the torch-first public payloads.
- Portable scenes have a smaller but uniform reset contract. The removed host setters provide no compatibility path; migration is to the device lifecycle or another adapter.
- Aggregate support is derived from instance APIs rather than the static scene-mode narrative, so CPU sessions no longer inherit portable DR declarations.
- Pending wrench cancellation is deterministic at reset and plan replacement, while ordinary stepping remains the consumption boundary. Partial native submission/cancellation failure faults the backend.

## Related decisions

- [Tensor lifecycle ADR](adr-tensor-lifecycle.md) defines the exact CUDA/AMDGPU execution lane.
- [Torch-first DR payload ADR](adr-dr-torch-first-payloads.md) defines the payload family and per-plane capability semantics.
- [Packed reset lifecycle ADR](adr-packed-reset-lifecycle.md) defines the transaction that composes reset fields.
- [Entities and state ADR](adr-entities.md) defines selected-row public state semantics.
