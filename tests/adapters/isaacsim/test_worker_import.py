"""SDK-free checks for the external IsaacSim worker import boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from unisim.backend.isaacsim.dependencies import IsaacSimRuntime, build_worker_env


def test_worker_owner_import_is_limited_to_unisim_modules() -> None:
    worker_path = Path(__file__).resolve().parents[3] / "src" / "unisim"
    worker_path = worker_path / "backend" / "isaacsim" / "worker.py"
    spec = importlib.util.spec_from_file_location("unisim_isaacsim_worker_test", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_meta_path = list(sys.meta_path)
    try:
        spec.loader.exec_module(module)
        finder = module._HostUniSimFinder(module._HOST_PACKAGE_ROOT)
        package_path = [str(module._HOST_PACKAGE_ROOT / "unisim")]
        assert finder.find_spec("unisim.scene_compiler", package_path) is not None
        assert finder.find_spec("numpy") is None
    finally:
        sys.meta_path[:] = original_meta_path


def test_worker_environment_does_not_shadow_runtime_dependencies(tmp_path: Path) -> None:
    runtime = IsaacSimRuntime(
        python=tmp_path / "venv" / "bin" / "python",
        isaaclab_source=None,
    )
    environment = build_worker_env(runtime)
    assert environment.get("PYTHONPATH", "") == ""
