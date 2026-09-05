# Architecture

`unisim-core` owns the public physics contract, backend capabilities, adapter
factory boundary, engine-native resources, conformance checks, and the reserved
benchmark case/result schemas. It has no dependency on UniLab, Hydra, Torch,
Gymnasium, learner, runner, or task code.

UniLab owns task/env/manager lifecycle, Hydra owner YAML, robot assets,
training, checkpoint, and sim2sim policy I/O. UniLab translates task-owned
scene and randomization inputs into the UniSim contract.

The MuJoCo adapter is constructed with a package-neutral `SceneCfg` and an
optional vectorized environment count. XML is parsed only during construction;
state/control arrays are copied through the public contract and engine objects
never escape the adapter. Drake, MJWarp and Genesis expose the same boundary
through concrete engine adapters. IsaacGym and IsaacSim share the subprocess IPC
framing and keep their Python 3.8/Kit workers outside the core wheel.

`unisim.ADAPTER_SPECS` is the single migration manifest for all seven UniLab
backend identities. ``available`` means a public adapter and diagnostics exist;
it does not claim that a proprietary SDK or GPU runtime is installed on every
host. Runtime support is established by the adapter's optional-extra and
worker smoke tests.

All asset and model metadata resolution is a cold-path concern. Hot-path
`step`/`reset` code receives validated arrays and cached identifiers; adapters
must not probe private engine attributes dynamically.

Interval domain randomization is declarative: UniLab's manager builds
`IntervalRandomizationPlan.ops` from `IntervalTermOp` descriptors defined in
`unisim.dr.interval` (term name, NumPy payload, optional body ids; stdlib +
NumPy only so plans stay pickle-safe across spawn-based collector processes).
Each backend owns its capability declaration (`supported_interval_terms`) and
a cold-path-built `_interval_term_handlers()` table; the generic
`SimBackend.apply_interval_randomization` dispatch validates each op against
the builtin term specs, routes it to the matching handler, and fails closed
with `NotImplementedError` for any term a backend does not declare. Custom
terms are free-form strings owned by the registering backend and validated
only against its capability set.
