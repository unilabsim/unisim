# UniLab Migration

[English](migration.md) | [中文](../zh/migration.md)

The migration is staged by backend. Each adapter child moves implementation and documentation together, adds optional dependency diagnostics and conformance coverage, and updates the UniLab consumer boundary. The former `unilab.base.backend` re-export shim has been removed; there is one production implementation owned by `unisim-core`.

MuJoCo is the first in-process adapter. It accepts a package-neutral `SceneCfg`, materializes XML on construction, and exposes cached numeric state through `unisim.SimBackend`; task-owned scene composition remains in UniLab. Its native batch executor is mjbatch (`unilabsim/mjbatch_uni`, a maintained fork of `kevinzakka/mjbatch`); heterogeneous model variants are unsupported, and field-level domain randomization goes through mjbatch `expand` and `set_const`.

Motrix is the second in-process adapter. It uses Motrix's batched `SceneData` and masked data slices behind the same public state, control, and reset contract.

The remaining UniLab identities are represented in UniSim as first-class adapters: Drake, MJWarp, Genesis, Newton, SuperDex, IsaacGym, and IsaacSim. The latter two reuse `unisim.backend.subprocess_ipc` and resolve their vendor workers without importing Kit or Python 3.8 modules into the host process. Missing SDKs are reported at construction time; no backend is silently downgraded to another engine.

Runtime-owned caches and worker installations use `UNISIM_*` environment variables and `~/.cache/unisim` defaults. The previous `UNILAB_*` names are accepted only as migration fallbacks so existing installations can move without losing cached state.
