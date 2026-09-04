# UniSim

[English](README.md) | [中文](README_zh.md)

UniSim provides backend-neutral physics contracts and optional engine
adapters for robot learning and simulation. The PyPI distribution is
`unisim-core`; the Python import namespace is `unisim`.

A single `SimBackend` contract covers state access, control, reset, and
domain-randomization boundaries, so the same task code runs on MuJoCo,
Motrix, Drake, MJWarp, Genesis, IsaacGym, or IsaacSim without engine-specific
branches. The base install depends only on NumPy; every engine SDK is an
optional extra loaded lazily, and importing `unisim` never imports an engine.

## Relationship to UniLab

UniSim is the extracted, backend-neutral physics layer used by UniLab.
UniLab retains Hydra configuration,
task/env/manager lifecycle, robot assets, RL training, checkpoints, and
sim2sim policy I/O; UniSim owns the physics contract, adapter lifecycle and
state translation, optional-runtime diagnostics, and the shared subprocess
IPC layer. There is exactly one production implementation of each backend,
owned by this repository — UniLab only assembles task-owned scene and
configuration inputs and consumes the public contract. UniSim never imports
UniLab.

## Installation

```bash
pip install unisim-core                # base: contract, factory, fake backend
pip install "unisim-core[mujoco]"      # plus an engine extra when needed
```

Available extras: `mujoco`, `motrix`, `drake`, `mjwarp`, `genesis`,
`isaacgym`, `isaacsim`. The Isaac extras are empty spellings because those
vendor SDKs are not redistributable; their adapters discover dedicated worker
installations at construction time. See
[`docs/support-matrix.md`](docs/support-matrix.md) for the full adapter
support matrix.

## Quick start

The public boundary is deliberately lazy and safe to import anywhere:

```python
from unisim import SimBackend, create_backend
```

Construct a backend through the factory with a package-neutral `SceneCfg`:

```python
backend = create_backend("mujoco", scene=scene_cfg, num_envs=64, sim_dt=0.01)
backend.materialize()      # cold path: parse XML, build engine objects
backend.reset()
state = backend.get_state()
backend.step(ctrl)         # hot path: validated arrays, cached handles
```

Each adapter fails closed with an actionable, backend-specific diagnostic
when its optional runtime is missing — no backend is silently downgraded to
another engine. Engine-native model and data objects never escape the
adapter; state and control flow through validated NumPy arrays.

A deterministic `FakeBackend` and the `assert_backend_conformance` helper let
consumers test task code without any engine installed. `BenchmarkCase` and
`BenchmarkResult` are reserved schema extension points for a future benchmark
package; no workload runner is implemented here.

External worker roots can be configured with `UNISIM_ISAACGYM_HOME`,
`UNISIM_ISAACGYM_PYTHON`, `UNISIM_ISAACSIM_HOME`, and
`UNISIM_ISAACSIM_PYTHON`. The package also accepts the former `UNILAB_*`
spellings as a migration fallback.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — ownership boundary between
  UniSim and UniLab
- [`docs/support-matrix.md`](docs/support-matrix.md) — adapter install and
  runtime requirements
- [`docs/migration.md`](docs/migration.md) — migrating from the historical
  UniLab backend layer
- [`docs/benchmark-api.md`](docs/benchmark-api.md) — reserved benchmark
  schemas
- [`docs/release.md`](docs/release.md) — TestPyPI and automated production
  release procedure

## Development

```bash
make sync       # locked environment plus the MuJoCo test extra
make check      # Ruff + pytest
make package    # source distribution and wheel for local inspection
```

Every adapter must document its supported Python/platform/runtime matrix and
pass the conformance helper before it is published.

## License

Apache-2.0, see [LICENSE](LICENSE).
