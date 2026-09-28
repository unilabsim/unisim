from __future__ import annotations

from pathlib import Path

import pytest

from unisim.backend.isaacgym.dependencies import IsaacGymRuntime, build_worker_env


def test_worker_environment_does_not_inherit_host_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    runtime = IsaacGymRuntime(
        python=Path("/worker/bin/python3.8"),
        isaacgym_python=Path("/worker/isaacgym/python"),
        lib_path=Path("/worker/lib"),
    )

    environment = build_worker_env(runtime)

    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["LD_LIBRARY_PATH"].startswith("/worker/lib")
    assert environment["PATH"].startswith("/worker/bin")
