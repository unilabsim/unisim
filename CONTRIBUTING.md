# Contributing

Use `make sync` for the locked development environment, then run `make check`
and `make package` before opening a pull request. Keep the core package free of
UniLab and engine-SDK imports; optional adapters must remain lazy and declare
their dependencies explicitly. Every public contract change needs focused
tests, documentation, and a changelog entry. Release tags and the OIDC PyPI
workflow are documented in [`docs/release.md`](docs/release.md).
