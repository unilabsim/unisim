"""Runtime discovery for the out-of-process IsaacGym (Preview 4) worker.

IsaacGym only supports Python 3.6-3.8 and can never be installed into the
main UniLab environment (Python >= 3.10).  Physics therefore runs in a
dedicated conda environment under
``$UNISIM_ISAACGYM_HOME`` (default ``~/.cache/unisim/isaacgym``):

- ``miniconda3/envs/hsgym/bin/python3.8`` — the worker interpreter,
- ``miniconda3/envs/hsgym/lib`` — prepended to the worker's
  ``LD_LIBRARY_PATH`` (conda ``libstdcxx-ng`` fixes the GLIBCXX mismatch),
- ``miniconda3/envs/hsgym/bin`` — prepended to the worker's ``PATH`` so the
  pip-installed ``ninja`` is reachable for the one-time gymtorch JIT compile,
- ``isaacgym/python`` — appended to the worker's ``sys.path``.

``UNISIM_ISAACGYM_PYTHON`` overrides the interpreter path explicitly.

The former ``UNILAB_*`` names remain accepted as a migration fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from unisim.backend.subprocess_ipc.runtime import WorkerRuntime, build_worker_environment
from unisim.optional import OptionalDependencyError

ENV_PYTHON = "UNISIM_ISAACGYM_PYTHON"
ENV_HOME = "UNISIM_ISAACGYM_HOME"
_LEGACY_ENV_PYTHON = "UNILAB_ISAACGYM_PYTHON"
_LEGACY_ENV_HOME = "UNILAB_ISAACGYM_HOME"

_CONDA_ENV_REL = Path("miniconda3") / "envs" / "hsgym"
_ISAACGYM_PYTHON_REL = Path("isaacgym") / "python"

_SETUP_HINT = (
    "Create the dedicated IsaacGym worker environment under UNISIM_ISAACGYM_HOME "
    "(see docs/en/support-matrix.md)."
)


class IsaacGymDependencyError(OptionalDependencyError):
    """Raised with an actionable setup hint when the worker runtime is absent."""


@dataclass(frozen=True)
class IsaacGymRuntime:
    """Resolved on-disk locations for the Python 3.8 IsaacGym worker."""

    python: Path
    isaacgym_python: Path
    lib_path: Path


def default_isaacgym_home() -> Path:
    """Return the default IsaacGym install root (including legacy override)."""
    value = os.environ.get(ENV_HOME) or os.environ.get(_LEGACY_ENV_HOME)
    return Path(value or "~/.cache/unisim/isaacgym").expanduser()


def _candidate_paths(home: Path) -> IsaacGymRuntime:
    env_root = home / _CONDA_ENV_REL
    return IsaacGymRuntime(
        python=env_root / "bin" / "python3.8",
        isaacgym_python=home / _ISAACGYM_PYTHON_REL,
        lib_path=env_root / "lib",
    )


def resolve_isaacgym_runtime(python_override: str | None = None) -> IsaacGymRuntime:
    """Resolve the worker interpreter and IsaacGym package locations.

    Raises:
        IsaacGymDependencyError: If any required component is missing.
    """
    home = default_isaacgym_home()
    runtime = _candidate_paths(home)

    if python_override:
        python = Path(python_override).expanduser()
    elif os.environ.get(ENV_PYTHON) or os.environ.get(_LEGACY_ENV_PYTHON):
        python = Path(os.environ.get(ENV_PYTHON) or os.environ[_LEGACY_ENV_PYTHON]).expanduser()
    else:
        python = runtime.python
    if not python.is_file():
        raise IsaacGymDependencyError(
            f"isaacgym backend could not find the Python 3.8 worker interpreter at "
            f"{python}. {_SETUP_HINT}"
        )
    if not runtime.isaacgym_python.is_dir():
        raise IsaacGymDependencyError(
            f"isaacgym backend could not find the IsaacGym package directory at "
            f"{runtime.isaacgym_python}. {_SETUP_HINT}"
        )
    if not runtime.lib_path.is_dir():
        raise IsaacGymDependencyError(
            f"isaacgym backend could not find the worker conda env lib directory at "
            f"{runtime.lib_path}. {_SETUP_HINT}"
        )
    return IsaacGymRuntime(
        python=python,
        isaacgym_python=runtime.isaacgym_python,
        lib_path=runtime.lib_path,
    )


def isaacgym_runtime_available() -> bool:
    """Return True when the worker runtime resolves without raising."""
    try:
        resolve_isaacgym_runtime()
    except IsaacGymDependencyError:
        return False
    return True


def build_worker_env(runtime: IsaacGymRuntime) -> dict[str, str]:
    """Build the worker process environment.

    Prepends the conda env's ``lib`` to ``LD_LIBRARY_PATH`` (the
    ``libstdcxx-ng`` GLIBCXX fix and ``libpython3.8``) and its ``bin`` to
    ``PATH`` (the pip-installed ``ninja`` must be reachable for the one-time
    gymtorch JIT compile on a fresh machine).
    """
    return build_worker_environment(
        WorkerRuntime(
            python=runtime.python,
            lib_path=runtime.lib_path,
            bin_path=runtime.lib_path.parent / "bin",
        )
    )


__all__ = [
    "ENV_HOME",
    "ENV_PYTHON",
    "IsaacGymDependencyError",
    "IsaacGymRuntime",
    "build_worker_env",
    "default_isaacgym_home",
    "isaacgym_runtime_available",
    "resolve_isaacgym_runtime",
]
