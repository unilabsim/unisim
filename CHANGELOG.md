# Changelog

## Unreleased

- Rework the README around the project overview, UniLab relationship,
  installation, and quick start, and add the Chinese `README_zh.md`.

## 1.1.0 - 2026-09-05

- Added the declarative interval domain-randomization term contract in
  `unisim.dr.interval`: builtin term specs (`INTERVAL_TERM_SPECS`,
  `interval_term_spec`), the pickle-safe `IntervalTermOp` descriptor with
  builtin-contract validation, and the `ops` field on
  `IntervalRandomizationPlan` (`iter_ops()` translates the legacy fields).
- Added the `supported_interval_terms` capability set on
  `DomainRandomizationCapabilities` with `supports_interval_term()` /
  `get_unsupported_interval_terms()`, falling back to the legacy bools so old
  constructor call sites keep their meaning.
- Replaced the abstract per-backend `apply_interval_randomization`
  implementations with generic `SimBackend` dispatch over the backend-owned
  `_interval_term_handlers()` table; terms without a handler fail closed with
  `NotImplementedError` naming the backend class and the term.
- Deprecated the five legacy `IntervalRandomizationPlan` fields and the five
  `supports_interval_*` capability bools; they remain functional and will be
  removed in the next major release.
- Fixed the mjwarp and genesis backends silently dropping unsupported
  interval body-torque and body-angular-velocity randomization; both now fail
  closed through the base dispatch.

## 1.0.0 - 2026-09-04

- Promote the contract and seven-adapter manifest to the stable `1.0.x` line;
  the public import boundary (`SimBackend`, `create_backend`, `ADAPTER_SPECS`,
  adapter classes, and `unisim.backend.subprocess_ipc`) is now stable.
- Support `SimBackend.set_pre_step_control` on the `mjwarp` backend: a
  registered converter now runs on the host before every physics substep with
  the qpos/qvel cache refreshed to the substep-start state (matching the
  MuJoCo backend's substep boundary and `callback_sensordata=False` sensor
  semantics), and `None` unregisters it.  The callback path uses eager kernel
  launches instead of captured step graphs.
- Restore the missing 0.1.10 changelog entry and the 0.1.4/0.1.5 ordering, and
  correct the `unisim.backend.subprocess_ipc` path and Isaac extras spelling in
  the migration and support-matrix documentation.

## 0.1.14 - 2026-09-02

- Update the trusted-publishing action to support the source distribution's
  current Python Core Metadata version.
- Require successful cross-platform tests and pre-release sdist verification
  before a version tag can publish to PyPI.

## 0.1.13 - 2026-09-02

- Add GitHub Actions CI and tag-triggered PyPI trusted publishing.
- Publish only the source distribution so releases do not select a Python
  version, operating system, or wheel platform.
- Document repository development, compatibility, and release conventions.

## 0.1.12 - 2026-09-02

- Fix the root public export surface so wildcard imports resolve
  `MjcfSubprocessBackend` and its historical `SubprocessBackend` alias.
- Use package-owned `UNISIM_*` worker/cache environment variables and
  `~/.cache/unisim` defaults, with read-only fallback to legacy `UNILAB_*`
  overrides.
- Expand adapter/factory/import-boundary tests for the complete seven-backend
  manifest and package isolation.

## 0.1.11

- Corrected the Drake adapter to consume the external `drake-uni` distribution
  through its `drake_uni` import namespace.
- Added fail-closed import diagnostics and support-matrix documentation for the
  standalone Drake runtime boundary.

## 0.1.10

- Replaced the `drake` PyPI dependency with the external `drake-uni==0.1.0`
  distribution and aligned the Drake adapter with its batch runtime API.

## 0.1.9

- Added the MuJoCo batch runtime to the `mujoco` optional extra so the
  production adapter is installable from a clean UniSim environment.
- Made conformance checks exercise adapters through their cold-path
  `materialize()` lifecycle before stepping.
- Expanded standalone MuJoCo and Motrix adapter tests to cover full state
  shapes and identity-quaternion reset semantics.

## 0.1.8

- Kept playback video I/O monkeypatchable while preserving lazy optional
  ``imageio`` loading for import isolation.

## 0.1.7

- Corrected factory option translation for the extracted backend adapters.
- Preserved backend-specific validation and fail-closed diagnostics when
  callers pass options from the UniLab owner layer.

## 0.1.6

- Migrated the complete production backend implementations and shared
  subprocess IPC into `unisim-core`.
- Removed the test-only runtime bridge from the public factory so every named
  backend resolves to its concrete adapter and fails closed when unavailable.
- Added support-matrix, migration, and package-boundary documentation for all
  seven adapters.

## 0.1.5

- Exported adapter-specific dependency diagnostics and the shared subprocess
  backend types from the public `unisim` namespace.

## 0.1.4

- Added public Drake, MJWarp, Genesis, IsaacGym and IsaacSim adapter boundaries.
- Added shared subprocess IPC framing used by Isaac worker integrations.
- Promoted all seven UniLab backend identities to the adapter manifest; SDK
  availability remains lazy and fail-closed.

## 0.1.3

- Add the staged adapter identity manifest for all roadmap backends.

## 0.1.2

- Add the lazy Motrix adapter and shared contract smoke coverage.

## 0.1.1

- Add the lazy MuJoCo adapter and backend factory.
- Add MuJoCo contract smoke coverage and adapter documentation.

## 0.1.0

- Bootstrap the `unisim` namespace and `unisim-core` distribution.
- Add the backend-neutral `SimBackend` contract, fake backend, and conformance helper.
- Reserve benchmark case/result interfaces without implementing workloads or measurements.
