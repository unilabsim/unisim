# Release Runbook

This package is published as `unisim-core` and imported as `unisim`.
Development and roadmap validation use TestPyPI only. Production PyPI remains
a manual maintainer action after the UniLab roadmap handoff is accepted.

## TestPyPI

1. Update `project.version`, `CHANGELOG.md`, and the migration/support docs.
2. Run `uv run pytest -q`, `uv run ruff check src tests`, and `uv build`.
3. Run `uv run --with twine twine check dist/unisim_core-<version>*`.
4. Upload the wheel and sdist to TestPyPI using credentials from `~/.pypirc`.
   Never print, copy, or commit that file.
5. Install the exact version in an isolated environment with TestPyPI as the
   package index and verify `import unisim` without loading `unilab` or engine
   SDK modules.
6. Record the published URLs, artifact hashes, gate results, and rollback
   decision in the release PR and the UniLab roadmap issue.

Use the package-owned distribution and import names in all commands. Do not
publish production PyPI while roadmap #1428 is open.

## Production PyPI

After the maintainer merges the final UniLab handoff PR and closes roadmap
#1428, the maintainer may repeat the same artifact checks and publish the exact
version to PyPI. Production publishing is intentionally not automated by this
repository and is not performed by roadmap implementation agents.

