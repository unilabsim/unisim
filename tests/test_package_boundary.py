from __future__ import annotations

from pathlib import Path

from unisim.backend import process_device
from unisim.backend.isaacgym import dependencies as isaacgym_dependencies
from unisim.backend.isaacsim import dependencies as isaacsim_dependencies


def test_package_owned_worker_overrides_take_precedence(monkeypatch, tmp_path: Path) -> None:
    package_home = tmp_path / "package-home"
    legacy_home = tmp_path / "legacy-home"
    monkeypatch.setenv("UNISIM_ISAACGYM_HOME", str(package_home))
    monkeypatch.setenv("UNILAB_ISAACGYM_HOME", str(legacy_home))
    assert isaacgym_dependencies.default_isaacgym_home() == package_home

    monkeypatch.setenv("UNISIM_ISAACSIM_HOME", str(package_home))
    monkeypatch.setenv("UNILAB_ISAACSIM_HOME", str(legacy_home))
    assert isaacsim_dependencies.default_isaacsim_home() == package_home


def test_backend_process_device_module_has_no_unilab_imports() -> None:
    source = Path(process_device.__file__).read_text(encoding="utf-8")
    assert "unilab" not in source.lower()
