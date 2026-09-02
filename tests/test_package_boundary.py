from __future__ import annotations

from pathlib import Path

from unisim.backend import process_device
from unisim.backend.isaacgym import dependencies as isaacgym_dependencies
from unisim.backend.isaacsim import dependencies as isaacsim_dependencies
from unisim.backend.mujoco import chunk_tuner


def test_chunk_cache_uses_unisim_namespace_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("UNISIM_CHUNK_SIZE_CACHE", raising=False)
    monkeypatch.delenv("UNILAB_CHUNK_SIZE_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert chunk_tuner.cache_path() == tmp_path / "unisim" / "chunk_size.json"


def test_chunk_cache_accepts_legacy_override(monkeypatch, tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    monkeypatch.delenv("UNISIM_CHUNK_SIZE_CACHE", raising=False)
    monkeypatch.setenv("UNILAB_CHUNK_SIZE_CACHE", str(legacy))

    assert chunk_tuner.cache_path() == legacy


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
