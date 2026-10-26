# AGENTS.md

Guidance for coding agents and maintainers working in this repository.

## Project overview

UniSim is the extracted, backend-neutral physics contract used by UniLab.  The PyPI distribution is `unisim-core`; the import namespace is `unisim`.  The current `1.x` line contains the public `SimBackend` contract, the adapter factory and manifest, optional engine boundaries, a deterministic fake backend, conformance helpers, and benchmark result schemas.

UniSim owns contracts, adapter lifecycle/state translation, optional-runtime diagnostics, and the shared subprocess IPC layer.  The NumPy-only ray-query plugin contract (`RayCaster`, `create_ray_caster`, `RAY_CASTER_SPECS` in `ray_query.py`/`factory.py`, with the pure-NumPy `FakeRayCaster` reference) follows the same lazy manifest-plus-dispatch boundary as engine adapters: plugin packages such as `uni_ray` are imported on demand and fail closed when unavailable.  UniLab remains the owner of Hydra/task configuration, robot assets, rewards and observations, rollouts, training, checkpoints, and sim2sim policy I/O.  Do not add a dependency on UniLab or eagerly import an engine SDK into the base package.

The public import boundary is deliberately lazy:

```python
from unisim import SimBackend, create_backend
```

`create_backend()` resolves one of the declared adapters (`mujoco`, `motrix`, `drake`, `mjwarp`, `newton`, `genesis`, `superdex`, `isaacgym`, or `isaacsim`) and reports a backend-specific diagnostic when its optional runtime is unavailable.  The base wheel must remain importable with none of those SDKs installed.

## Repository layout

```text
src/unisim/
  contract.py, factory.py, adapters.py   # stable contract and dispatch boundary
  fake.py, conformance.py                # deterministic test backend and checks
  backend/<engine>/                       # lazy concrete adapters
  backend/subprocess_ipc/                 # Isaac worker protocol and host runtime
tests/                                    # contract, import-boundary, and adapter tests
  core/                                   # import and package-boundary invariants
  contract/                               # engine-neutral contract coverage
  factory/                                # manifest and factory dispatch coverage
  adapters/<engine>/                      # adapter-specific tests
docs/en/                                  # English architecture and operations docs
docs/zh/                                  # Chinese counterparts of docs/en
scripts/benchmarks/                       # maintainer-only benchmark entry points
scripts/diagnostics/                      # maintainer-only runtime probes
pyproject.toml                            # package metadata and optional extras
uv.lock                                   # locked development dependencies
Makefile                                  # common local commands
.github/workflows/ci.yml                 # PR/main checks
.github/workflows/release.yml            # tag-based build, verification, and publish
```

Build output (`dist/`, `build/`, `*.egg-info/`, caches, and virtual environments) is ignored and must not be committed. `scripts/` intentionally contains only standalone maintainer commands; reusable runtime code belongs in `src/unisim`, and regression coverage belongs in the most specific `tests/` subtree.

## Development commands

Use `uv` for all project-managed commands.  The normal loop is:

```bash
make sync                 # uv sync --locked --extra mujoco
make lint                 # uv run ruff check .
make typecheck            # uv run mypy src/unisim && uv run pyright
make test                 # uv run pytest -q
make check                # lint + typecheck + test
make package              # uv build --out-dir dist
```

Equivalent raw commands are `uv sync --locked --extra mujoco`, `uv run ruff check .`, `uv run mypy src/unisim`, `uv run pyright`, `uv run pytest -q`, and `uv build --out-dir dist`.  Run `uv lock --check` when changing dependency metadata.  Install an optional adapter only when needed beyond the MuJoCo development-test baseline; vendor-managed Isaac SDKs are intentionally not dependencies of this repository.  The published base distribution still depends only on NumPy.

`make test-no-sync` (or `uv run --no-sync pytest -q`) is useful after changing an external runtime while preserving the active environment.  Do not put engine installation or downloads in the base test path.

## Testing and style

- `pytest` is configured to collect from `tests/`; engine-specific numerical tests use `pytest.importorskip`.  The adapter-manifest tests resolve the MuJoCo public class, so the normal `make sync` MuJoCo extra is required for the complete suite.
- Import-boundary tests must continue to show that importing `unisim` does not load `unilab`, MuJoCo, Drake, Motrix, Warp, Genesis, IsaacGym, or IsaacSim.
- Every public contract or factory change needs focused tests, documentation, and a `CHANGELOG.md` entry.
- Ruff is configured for line length 100, Python 3.10 syntax, and rules `E`, `F`, `I`, `N`, `W`.  Use four-space indentation and `from __future__ import annotations` in new modules.  Keep comments and docstrings in English.
- mypy and pyright gate `src/unisim` (configuration in `pyproject.toml`; both the pyright pin and the mypy/pyright Python targets match the sister unilab-rl repository).  mypy runs with `no_site_packages` + `ignore_missing_imports`, so site-package stubs (NumPy included) and the stub-less optional engine SDKs resolve as Any for mypy while pyright still checks NumPy fully; first-party code is otherwise fully checked, so fix real type errors in code rather than widening the exemptions.
- Documentation is bilingual by filename under `docs/en/` and `docs/zh/`; update both language trees together and preserve their heading, list, table, and code-block structure. Keep `README_zh.md` in sync with `README.md`.
- Keep model/XML parsing and SDK discovery on cold paths.  `step`, `reset`, and state access should operate on validated arrays and cached handles.

Before a change is complete, run `make check` and, when packaging code changed, `make package`.  The published base package does not require a native or vendor runtime; the complete repository suite installs the pinned MuJoCo extra as described above.  Do not weaken tests merely to make an unavailable optional SDK pass.

## Scope and compatibility boundaries

Keep these names and paths stable for UniLab consumers: `unisim.SimBackend`, `unisim.create_backend`, `unisim.ADAPTER_SPECS`, the adapter classes exported by `unisim`, and `unisim.backend.subprocess_ipc`.  `MjcfSubprocessBackend` is the concrete shared-host name; do not restore historical aliases.

Beyond that public boundary, UniLab already consumes the following adapter-internal modules and names, which are likewise stable for UniLab and pinned by `tests/core/test_unilab_integration_surface.py`: the optional-runtime diagnostics modules `unisim.backend.genesis.dependencies`, `unisim.backend.isaacsim.dependencies`, `unisim.backend.isaacgym.dependencies`, and `unisim.backend.superdex.dependencies`; `unisim.backend.drake.backend.ensure_drake_batch_available`; `unisim.backend.mujoco.xml.materialize_scene_visual_override`; the process device binders `unisim.backend.mjwarp.runtime.bind_mjwarp_process_device` and `unisim.backend.newton.runtime.bind_newton_process_device`/`get_bound_newton_process_device`; and the playback helpers `unisim.backend.playback.write_playback_video` and `unisim.backend.mujoco.playback.resolve_render_play_model_files`.  Every listed module stays importable with no engine SDK installed, and each name follows the removal policy for stable public names below.

In scope: backend-neutral contracts, adapter validation/materialization, optional dependency diagnostics, subprocess framing, conformance coverage, and package/release tooling.  Out of scope: engine source/solver changes, task YAML, reward or rollout policy, training orchestration, distributed execution, and private SDK redistribution.

## Deprecation and removal policy

Breaking removals are hard cuts at release boundaries: minor releases may cut internal surfaces, while the stable public names listed above are removed only at a major release.  A published release is never retroactively edited.  Two bounded compatibility windows are allowed:

- Warn-and-ignore shims (for example removed factory options that still parse but warn) live until the next major release; the removal milestone is named in `CHANGELOG.md` when the shim is introduced.
- Reader compatibility for persisted data (for example legacy pickle state keys that cross process boundaries) is documented in code with the producing version window and a named removal boundary — by default the next major release.

An announced "removed next major" must name the version, and the removal ships no earlier than its named milestone.

## CI and automated release

`.github/workflows/ci.yml` runs Ruff and pytest (including the pinned MuJoCo adapter extra on Unix) on pull requests, pushes to `main`, and manual dispatch. It has exactly three test jobs: Python 3.10 on Ubuntu, macOS, and Windows. Windows runs the core/import-boundary subset because the pinned mjbatch runtime, like the MuJoCoUni runtime before it, skips Windows (mujoco.dll ships no import library); this optional runtime limitation must not block the pure-Python core. A fourth, lint-class `typecheck` job (Ubuntu, Python 3.11, matching unilab-rl) runs mypy and pyright against `src/unisim`; it is not a test job, so the three-test-job description above is unchanged. A separate Ubuntu package job waits for the three test jobs plus the typecheck job, builds one source distribution and one universal pure-Python wheel without selecting a Python version, checks both with Twine, installs the wheel with `--only-binary :all:` and runs the fake-backend smoke test, and uploads both as pre-release CI artifacts. The runner only executes the build; the artifacts carry no Python ABI or OS platform tag.

`.github/workflows/release.yml` is the production release path.  It publishes the source distribution plus one universal pure-Python wheel (`unisim_core-<version>-py3-none-any.whl`); the base package is pure Python and NumPy-only, so no Python version, OS, ABI-specific wheel, or native SDK is part of the release gate.  The universal base wheel does not change the per-platform support matrix of the optional engine extras:

1. Update `[project].version` in `pyproject.toml`, `CHANGELOG.md`, and any affected documentation.  The version is single-sourced from `pyproject.toml`.
2. Run `make check`, `make package`, and (optionally) `uvx --from twine twine check dist/*` locally.
3. Wait for the three cross-platform test jobs, the typecheck job, and the pre-release package job to pass for the commit you intend to release.
4. Push an annotated tag whose name is exactly `v<project.version>` (for example, `v0.1.13`). GitHub Actions verifies the successful CI run, then builds one sdist and one universal wheel on `ubuntu-latest`, checks both with Twine, verifies the wheel is `py3-none-any` with no native binaries and NumPy-only base dependencies, installs each, and runs a core import/fake-backend smoke test on each (the wheel with `uv pip install --only-binary :all:` so it cannot fall back to the sdist). The runner is only the machine executing the build; it does not constrain the artifacts. Both artifacts are retained and published, and downstream users choose their own Python and OS.
5. After the distribution builds and smoke tests pass, the `publish` job uploads both artifacts to PyPI with `pypa/gh-action-pypi-publish` and GitHub trusted publishing (OIDC). It has no API token or credential checked into the repository and uses `skip-existing: true` for safe reruns.  A final `github-release` job then creates the GitHub Release for the tag with the sdist and wheel attached and the matching `CHANGELOG.md` section as its notes.

Manual workflow dispatch performs build/test verification only; publishing is gated to a `push` event for `refs/tags/v*`.  A failed release should be fixed and rerun with the same tag only when the artifacts are unchanged, or with a new version/tag after changing code.  Never rewrite a published version. TestPyPI instructions and artifact-install smoke checks live in `docs/en/release.md` and `docs/zh/release.md`.

## GitHub CLI

Remote: `github.com/unilabsim/unisim`.

```bash
gh pr list
gh pr view <number>
gh run list --workflow=ci.yml
gh run list --workflow=release.yml
gh run view <run-id>
```

Before reporting a pull request or release complete, inspect the workflow for the final head SHA and confirm `git status --short --branch` is clean.
