# UniLab Migration

The migration is staged by backend. Each adapter child moves implementation and
its documentation together, adds optional dependency diagnostics and
conformance coverage, and updates the UniLab consumer boundary. The temporary
`unilab.base.backend` re-export shim is not a second implementation and is
removed after all current backends use the released `unisim-core` package.

