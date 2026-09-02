# AGENTS.md

Guidance for coding agents and maintainers working in this repository.

## Project overview

UniSim is the extracted, backend-neutral physics contract used by UniLab.  The
PyPI distribution is `unisim-core`; the import namespace is `unisim`.  The
current `0.1.x` line contains the public `SimBackend` contract, the adapter
factory and manifest, seven optional engine boundaries, a deterministic fake
backend, conformance helpers, and benchmark result schemas.

UniSim owns contracts, adapter lifecycle/state translation, optional-runtime
diagnostics, and the shared subprocess IPC layer.  UniLab remains the owner of
Hydra/task configuration, robot assets, rewards and observations, rollouts,
training, checkpoints, and sim2sim policy I/O.  Do not add a dependency on
UniLab or eagerly import an engine SDK into the base package.

The public import boundary is deliberately lazy:

```python
from unisim import SimBackend, create_backend
```

`create_backend()` resolves one of the declared adapters (`mujoco`, `motrix`,
`drake`, `mjwarp`, `genesis`, `isaacgym`, or `isaacsim`) and reports a
backend-specific diagnostic when its optional runtime is unavailable.  The
base wheel must remain importable with none of those SDKs installed.

## Repository layout

```text
src/unisim/
  contract.py, factory.py, adapters.py   # stable contract and dispatch boundary
  fake.py, conformance.py                # deterministic test backend and checks
  backend/<engine>/                       # lazy concrete adapters
  backend/subprocess_ipc/                 # Isaac worker protocol and host runtime
tests/                                    # contract, import-boundary, and adapter tests
docs/                                     # architecture, migration, support, release docs
pyproject.toml                            # package metadata and optional extras
uv.lock                                   # locked development dependencies
Makefile                                  # common local commands
.github/workflows/ci.yml                 # PR/main checks
.github/workflows/release.yml            # tag-based build, verification, and publish
```

Build output (`dist/`, `build/`, `*.egg-info/`, caches, and virtual
environments) is ignored and must not be committed.

## Development commands

Use `uv` for all project-managed commands.  The normal loop is:

```bash
make sync                 # uv sync --locked --extra mujoco
make lint                 # uv run ruff check .
make test                 # uv run pytest -q
make check                # lint + test
make package              # uv build --out-dir dist
```

Equivalent raw commands are `uv sync --locked --extra mujoco`,
`uv run ruff check .`,
`uv run pytest -q`, and `uv build --out-dir dist`.  Run
`uv lock --check` when changing dependency metadata.  Install an optional
adapter only when needed beyond the MuJoCo development-test baseline;
vendor-managed Isaac SDKs are intentionally not dependencies of this
repository.  The published base distribution still depends only on NumPy.

`make test-no-sync` (or `uv run --no-sync pytest -q`) is useful after changing
an external runtime while preserving the active environment.  Do not put
engine installation or downloads in the base test path.

## Testing and style

- `pytest` is configured to collect from `tests/`; engine-specific numerical
  tests use `pytest.importorskip`.  The adapter-manifest tests resolve the
  MuJoCo public class, so the normal `make sync` MuJoCo extra is required for
  the complete suite.
- Import-boundary tests must continue to show that importing `unisim` does not
  load `unilab`, MuJoCo, Drake, Motrix, Warp, Genesis, IsaacGym, or IsaacSim.
- Every public contract or factory change needs focused tests, documentation,
  and a `CHANGELOG.md` entry.
- Ruff is configured for line length 100, Python 3.10 syntax, and rules
  `E`, `F`, `I`, `N`, `W`.  Use four-space indentation and
  `from __future__ import annotations` in new modules.  Keep comments and
  docstrings in English; update `README_zh.md` only when one is added later.
- Keep model/XML parsing and SDK discovery on cold paths.  `step`, `reset`, and
  state access should operate on validated arrays and cached handles.

Before a change is complete, run `make check` and, when packaging code changed,
`make package`.  The published base package does not require a native or vendor
runtime; the complete repository suite installs the pinned MuJoCo extra as
described above.  Do not weaken tests merely to make an unavailable optional
SDK pass.

## Scope and compatibility boundaries

Keep these names and paths stable for UniLab consumers: `unisim.SimBackend`,
`unisim.create_backend`, `unisim.ADAPTER_SPECS`, the adapter classes exported by
`unisim`, and `unisim.backend.subprocess_ipc`.  Preserve the historical
`SubprocessBackend` alias while `MjcfSubprocessBackend` is the concrete name.

In scope: backend-neutral contracts, adapter validation/materialization,
optional dependency diagnostics, subprocess framing, conformance coverage, and
package/release tooling.  Out of scope: engine source/solver changes, task YAML,
reward or rollout policy, training orchestration, distributed execution, and
private SDK redistribution.

## CI and automated release

`.github/workflows/ci.yml` runs Ruff, pytest (including the pinned MuJoCo
adapter extra), and a package build on pull requests, pushes to `main`, and
manual dispatch.  The check matrix covers Ubuntu, macOS, and Windows with
Python 3.10 and 3.13.

`.github/workflows/release.yml` is the production release path:

1. Update `[project].version` in `pyproject.toml`, `CHANGELOG.md`, and any
   affected documentation.  The version is single-sourced from `pyproject.toml`.
2. Run `make check`, `make package`, and (optionally) `uvx --from twine twine
   check dist/*` locally.
3. Push an annotated tag whose name is exactly `v<project.version>` (for
   example, `v0.1.13`).
4. GitHub Actions builds both an sdist and a wheel, installs and tests the fresh
   sdist and wheel on the supported matrix, and retains the canonical Linux
   artifacts.
5. After all matrix jobs pass, the `publish` job uploads those artifacts to
   PyPI with `pypa/gh-action-pypi-publish` and GitHub trusted publishing (OIDC).
   It has no API token or credential checked into the repository and uses
   `skip-existing: true` for safe reruns.

Manual workflow dispatch performs build/test verification only; publishing is
gated to `refs/tags/v*`.  A failed release should be fixed and rerun with the
same tag only when the artifacts are unchanged, or with a new version/tag after
changing code.  Never rewrite a published version.  TestPyPI instructions and
artifact-install smoke checks live in `docs/release.md`.

## GitHub CLI

Remote: `github.com/unilabsim/unisim`.

```bash
gh pr list
gh pr view <number>
gh run list --workflow=ci.yml
gh run list --workflow=release.yml
gh run view <run-id>
```

Before reporting a pull request or release complete, inspect the workflow for
the final head SHA and confirm `git status --short --branch` is clean.
