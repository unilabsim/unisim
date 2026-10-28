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
- Reset randomization is the selected-row final-value subset `body_mass`, `body_ipos`, the base aliases `base_mass_delta` / `base_com_offset`, actuator `kp`, and actuator `kd`. Validation and commit stay on Torch tensors on the accepted CUDA device. Portable rows use cached device-resident public and native index tables and are scattered through audited owning native entities. Host/CPU reset payloads fail closed naming the term and plane.
- `body_mass` is mass-only and never rescales inertia. Absolute `body_ipos` is committed as `set_COM_shift(final ipos - immutable construction default)`, so repeated resets replay the same shift instead of accumulating it. A payload rejects the redundant combination of an absolute body table and its base-delta alias; delta payloads merge with the authoritative packed baseline.
- Interval randomization is the world-frame `body_force` and `body_torque` subset plus six-axis root velocity deltas. `body_linear_velocity_delta` stays world-frame. `body_angular_velocity_delta` is producer-sampled world-frame input and is converted to the free-root local frame with the current root quaternion before submission. Only non-zero selected rows are submitted through `set_dofs_velocity`; qpos, controls, time, unrelated solver state, and unrelated rows/columns remain intact. Undeclared interval terms fail closed by name.
- A new interval plan that contains a wrench term cancels unconsumed wrenches before its ops accumulate. Velocity-only plans preserve pending wrenches. Direct submissions and multiple ops in one plan accumulate independently. A selected tensor or entity reset cancels matching rows/bodies before Genesis' native state setter clears solver staging; unrelated rows and bodies persist. Successful stepping consumes and clears the pending mirror.
- The cold boundary is explicit: startup, scene materialization, variant materialization, immutable public/native index construction, and cold default tables. Hot reset and interval execution performs no host payload detour.
- Legacy portable host reset/interval paths and their capability advertising are removed without aliases or time-based compatibility names. Portable host-only setters are not runtime DR and are not advertised. NumPy remains only at cold metadata/serialization leaves.
- `get_dr_capabilities()` is authoritative and uses the unified `supported_reset_terms` and `supported_interval_terms` sets. Without the device lifecycle both sets are empty. An active session declares the terms implemented by that lifecycle; aggregate `get_capabilities()` projects them only as `dr.reset.*` and `dr.interval.*`, with no `dr.device.*` family. Private CPU-payload guards remain unchanged and do not imply host DR support.
- The declared reset set remains deliberately bounded. `geom_friction` remains unsupported because Genesis has no equivalent per-environment absolute-friction public semantic; DOF friction/damping/armature, geometry, gravity, and other reset terms stay unsupported until a reviewed device lifecycle and acceptance evidence exist.

## Evidence

- Real CUDA acceptance covers whole-MJCF and portable scenes in `tests/adapters/genesis/test_tensor_runtime.py`. The portable test materializes a real multi-entity/variant scene, checks the unified capability sets and aggregate keys, mutates only selected rows through selected-row and mask resets, verifies direct and plan-driven wrench staging plus step consumption/clearing, and exercises shape, finiteness, row/body range, ownership, dtype, and non-Torch validation failures on the accepted device.
- A separately masked CPU process (`CUDA_VISIBLE_DEVICES=""`) verifies that portable capability and tensor-DR declarations are empty and CPU/host requests fail closed. Skipped tests are not acceptance evidence.
- Fake-runtime CUDA coverage extends the device set and packed fields, proves selected-row mass/COM/PD isolation, replays a non-accumulating COM baseline, submits linear and angular root kicks on selected rows with native column masks, and verifies that velocity-only interval plans do not cancel pending wrenches. Real exact-CUDA acceptance covers the capability/report surface; available exact-AMDGPU/ROCm hardware repeats the same vendor-neutral public lane without a ROCm-specific branch.

## Consequences

- Consumers cannot move portable Genesis DR to CPU or torch.cpu operands. Owners must sample on the accepted CUDA device and use the torch-first public payloads.
- Portable scenes have a smaller but uniform reset contract. The removed host setters provide no compatibility path; migration is to the device lifecycle or another adapter.
- Aggregate support is derived from instance APIs rather than the static scene-mode narrative, so CPU sessions no longer inherit portable DR declarations.
- Pending wrench cancellation is deterministic at reset and wrench-plan replacement; velocity-only plans do not participate. Ordinary stepping remains the consumption boundary. Partial native submission/cancellation failure faults the backend.

## Related decisions

- [Tensor lifecycle ADR](adr-tensor-lifecycle.md) defines the exact CUDA/AMDGPU execution lane.
- [Torch-first DR payload ADR](adr-dr-torch-first-payloads.md) defines the payload family and unified capability semantics.
- [Packed reset lifecycle ADR](adr-packed-reset-lifecycle.md) defines the transaction that composes reset fields.
- [Entities and state ADR](adr-entities.md) defines selected-row public state semantics.
