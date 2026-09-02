# UniSim

UniSim provides backend-neutral physics contracts and optional engine
adapters. The PyPI distribution is `unisim-core`; the Python import namespace
is `unisim`.

This repository is the extraction target for UniLab's unified physics backend.
The `0.1.x` line contains the public contract, seven production backend
adapters, a deterministic fake backend, a lightweight conformance helper, and
benchmark API/result-schema reservations. It does not run benchmark workloads
or include engine SDKs in the base install.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

Engine adapters are optional extras and are loaded lazily. The MuJoCo adapter
is available as `unisim.MuJoCoBackend` after installing the `mujoco` extra (or
through `unisim.create_backend("mujoco", scene=...)`). Each adapter must
document its supported Python/platform/runtime matrix and pass the conformance
helper before it is published.

Motrix is available through `unisim.MotrixBackend` after installing the
`motrix` extra. Drake, MJWarp and Genesis use the same lazy optional boundary;
IsaacGym and IsaacSim are external-worker adapters because their vendor SDKs
are not redistributable PyPI dependencies. All seven identities are exposed by
the factory and fail closed with actionable diagnostics when their runtime is
absent. Every adapter exposes the same state/control/reset contract;
engine-native model and data objects remain private.

External worker roots can be configured with `UNISIM_ISAACGYM_HOME`,
`UNISIM_ISAACGYM_PYTHON`, `UNISIM_ISAACSIM_HOME`, and
`UNISIM_ISAACSIM_PYTHON`. The package also accepts the former `UNILAB_*`
spellings as a migration fallback.

## Relationship to UniLab

UniLab retains Hydra configuration, task/env/manager lifecycle, robot assets,
RL training, checkpoint, and sim2sim ownership. UniSim must not import UniLab.
The final state has one implementation owned by this repository; UniLab only
assembles task-owned scene/configuration inputs.
