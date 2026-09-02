# UniLab Migration

The migration is staged by backend. Each adapter child moves implementation and
its documentation together, adds optional dependency diagnostics and
conformance coverage, and updates the UniLab consumer boundary. The former
`unilab.base.backend` re-export shim has been removed; there is one production
implementation owned by `unisim-core`.

The first real adapter is MuJoCo. It accepts a package-neutral `SceneCfg`,
materializes the XML on construction, and exposes cached numeric state through
`unisim.SimBackend`; task-owned scene composition remains in UniLab.

Motrix is the second in-process adapter. It uses Motrix's batched `SceneData`
and masked data slices behind the same public state/control/reset contract.

The remaining UniLab identities are represented in UniSim as first-class
adapters: Drake, MJWarp, Genesis, IsaacGym and IsaacSim. The latter two reuse
`unisim.subprocess_ipc` and resolve their vendor workers without importing Kit
or Python 3.8 modules into the host process. Missing SDKs are reported at
construction time; no backend is silently downgraded to another engine.
