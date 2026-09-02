# UniSim

UniSim provides backend-neutral physics contracts and optional engine
adapters. The PyPI distribution is `unisim-core`; the Python import namespace
is `unisim`.

This repository is the extraction target for UniLab's unified physics backend.
The initial `0.1.x` slice contains the public contract, a deterministic fake
backend, a lightweight conformance helper, and benchmark API/result-schema
reservations. It does not run benchmark workloads or include engine SDKs in the
base install.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

Engine adapters are optional extras and are loaded lazily. The first real
adapter is available as `unisim.MuJoCoBackend` after installing the `mujoco`
extra (or through `unisim.create_backend("mujoco", model_path=...)`). Each adapter must
document its supported Python/platform/runtime matrix and pass the conformance
helper before it is published.

Motrix is available through `unisim.MotrixBackend` after installing the
`motrix` extra. Both adapters expose the same state/control/reset contract;
engine-native model and data objects remain private.

## Relationship to UniLab

UniLab retains Hydra configuration, task/env/manager lifecycle, robot assets,
RL training, checkpoint, and sim2sim ownership. UniSim must not import UniLab.
During migration UniLab may use a short-lived re-export shim; the final state
has one implementation owned by this repository.
