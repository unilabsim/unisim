# Changelog

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
## 0.1.4

- Added public Drake, MJWarp, Genesis, IsaacGym and IsaacSim adapter boundaries.
- Added shared subprocess IPC framing used by Isaac worker integrations.
- Promoted all seven UniLab backend identities to the adapter manifest; SDK
  availability remains lazy and fail-closed.
## 0.1.5

- Exported adapter-specific dependency diagnostics and the shared subprocess
  backend types from the public `unisim` namespace.
