# Architecture

`unisim-core` owns the public physics contract, backend capabilities, adapter
factory boundary, engine-native resources, conformance checks, and future
benchmark result schema. It has no dependency on UniLab, Hydra, Torch,
Gymnasium, learner, runner, or task code.

UniLab owns task/env/manager lifecycle, Hydra owner YAML, robot assets,
training, checkpoint, and sim2sim policy I/O. UniLab translates task-owned
scene and randomization inputs into the UniSim contract.

The MuJoCo adapter is constructed with a `model_path` and optional vectorized
environment count. XML is parsed only during construction; state/control arrays
are copied through the public contract and engine objects never escape the
adapter.

All asset and model metadata resolution is a cold-path concern. Hot-path
`step`/`reset` code receives validated arrays and cached identifiers; adapters
must not probe private engine attributes dynamically.
