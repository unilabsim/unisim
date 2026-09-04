# Release Runbook

This package is published as `unisim-core` and imported as `unisim`. Production
PyPI publishing is automated by `.github/workflows/release.yml` after a
matching version tag is pushed and the tagged commit has a successful `ci.yml`
run. The workflow publishes only the source distribution using GitHub trusted
publishing (OIDC); no PyPI token is stored in the repository. It does not select
a Python-version or OS matrix and never publishes a wheel.

The repository administrator must configure a PyPI trusted publisher once with
owner `unilabsim`, repository `unisim`, workflow `release.yml`, and environment
`pypi`. Keep the GitHub environment protected with the desired reviewer rules.

## TestPyPI

1. Update `[project].version` in `pyproject.toml`, `CHANGELOG.md`, and affected
   migration/support docs.
2. Run `make check` and `make package`.
3. Run `uvx --from twine twine check dist/unisim_core-<version>.tar.gz`.
4. Upload the sdist to TestPyPI using credentials from `~/.pypirc`.
   Never print, copy, or commit that file.
5. Install the exact version in an isolated environment with TestPyPI as the
   package index and verify `import unisim` without loading `unilab` or engine
   SDK modules.
6. Record the published URLs, artifact hashes, gate results, and rollback
   decision in the release PR or release tracking issue.

Use the package-owned distribution and import names in all commands. Never print,
copy, or commit `~/.pypirc`.

## Production PyPI

1. Confirm the working tree is clean, `make check` passes, and the changelog
   contains the release entry.
2. Wait for the three cross-platform `ci.yml` test jobs and the pre-release
   sdist package job to pass for the commit you will tag.
3. Create and push an annotated tag exactly matching the package version, for
   example, `git tag -a v0.1.13 -m "release: unisim-core 0.1.13"` followed by
   `git push origin v0.1.13`. The release workflow verifies the tag and the
   successful CI run, builds and smoke-tests one sdist on `ubuntu-latest`, then
   publishes that sdist after the checks succeed. The runner only executes the
   build and does not constrain the source artifact. There is no release-time
   Python/OS matrix and no wheel publication; manual dispatch runs verification
   only and cannot publish.
4. Inspect the workflow and PyPI artifact metadata. A failed run may be
   re-run; never overwrite an already published version. Fix the source and
   release a new patch version when an artifact is wrong.
