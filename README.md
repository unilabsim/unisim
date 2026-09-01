# UniSim

UniSim provides backend-neutral physics contracts and optional engine
adapters. The PyPI distribution is `unisim-core`; the Python import namespace
is `unisim`.

This repository is the extraction target for UniLab's unified physics backend.
The initial `0.1.0` slice contains the public contract, a deterministic fake
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

Engine adapters are optional extras and are loaded lazily. Each adapter must
document its supported Python/platform/runtime matrix and pass the conformance
helper before it is published.

## Relationship to UniLab

UniLab retains Hydra configuration, task/env/manager lifecycle, robot assets,
RL training, checkpoint, and sim2sim ownership. UniSim must not import UniLab.
During migration UniLab may use a short-lived re-export shim; the final state
has one implementation owned by this repository.

