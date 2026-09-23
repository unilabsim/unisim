"""SDK-free contracts for IsaacSim role-derived USD caching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.physx_solver import PhysxSolverConfig
from unisim.backend.isaacsim.raw_usd_cache import (
    ENV_RAW_USD_CACHE,
    ENV_ROLE_USD_CACHE,
    RawUSDCacheFile,
    RoleUSDCache,
    resolve_role_usd_cache_root,
)
from unisim.backend.isaacsim.scene_worker import _raw_usd_request, _role_usd_request
from unisim.scene_compiler import SceneContentIdentity


def _identity() -> SceneContentIdentity:
    return SceneContentIdentity("portable-mjcf-v1", 1, "a" * 64, "b" * 64, "c" * 64)


def _runtime() -> dict[str, str]:
    return {
        "isaacsim": "5.1.0.0",
        "isaaclab": "0.47.2",
        "isaacsim.asset.importer.mjcf": "2.5.13",
        "python": "3.11.16",
    }


def _entity(name: str = "target", *, root_mode: str = "kinematic") -> SimpleNamespace:
    return SimpleNamespace(name=name, kind="rigid", root_mode=root_mode)


def _entry(
    *, collision_enabled: bool = False, mirror_of: str | None = "object",
    gravity_disabled: bool = True,
) -> dict[str, object]:
    return {
        "collision_enabled": collision_enabled,
        "mirror_of": mirror_of,
        "gravity_disabled": gravity_disabled,
    }


def test_mirror_and_physical_roles_share_role_neutral_raw_identity(tmp_path: Path) -> None:
    source = tmp_path / "object.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    physical = _entity("object", root_mode="floating")
    runtime = _runtime()
    physical_request = _raw_usd_request(
        _identity(), str(source), physical, 0, runtime, self_collision=False
    )
    source_request = _raw_usd_request(
        _identity(), str(source), physical, 0, runtime, self_collision=False
    )
    mirror_request = _raw_usd_request(
        _identity(), str(source), physical, 0, runtime, self_collision=False
    )
    assert source_request.identity == physical_request.identity
    assert mirror_request.identity == physical_request.identity
    assert mirror_request.parameters["entity"] == "object"


def test_role_identity_extends_raw_and_distinguishes_role_inputs() -> None:
    raw_record = SimpleNamespace(
        identity="d" * 64,
        source_digest="e" * 64,
        usd_relative_path="artifact.usd",
        files=(RawUSDCacheFile("artifact.usd", 8, "f" * 64),),
        runtime_versions=_runtime(),
    )
    request = _role_usd_request(
        raw_record, _entity(), _entry(collision_enabled=False), 2, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
    )
    assert request.identity != raw_record.identity
    assert request.source_digest == raw_record.source_digest
    assert request.runtime_versions == raw_record.runtime_versions
    changed_inputs = (
        _role_usd_request(
            raw_record,
            _entity("target", root_mode="floating"),
            _entry(collision_enabled=False),
            2,
            require_bodies=False,
            contact_offset=None,
            rest_offset=None,
        ),
        _role_usd_request(
            raw_record, _entity(), _entry(collision_enabled=True), 2, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
        ),
        _role_usd_request(
            raw_record, _entity(), _entry(collision_enabled=False), 3, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
        ),
        _role_usd_request(
            raw_record,
            _entity("target"),
            _entry(collision_enabled=False, mirror_of=None),
            2,
            require_bodies=False,
            contact_offset=None,
            rest_offset=None,
        ),
        _role_usd_request(
            raw_record, _entity(), _entry(collision_enabled=False), 2, require_bodies=True,
        contact_offset=None,
        rest_offset=None,
        ),
        _role_usd_request(
            raw_record,
            _entity(),
            _entry(collision_enabled=False, gravity_disabled=False),
            2,
            require_bodies=False,
            contact_offset=None,
            rest_offset=None,
        ),
    )
    assert all(item.identity != request.identity for item in changed_inputs)


def test_role_cache_bakes_once_and_keeps_role_artifacts_immutable(
    tmp_path: Path,
) -> None:
    role_root = tmp_path / "roles"
    record = SimpleNamespace(
        identity="d" * 64,
        source_digest="e" * 64,
        usd_relative_path="artifact.usd",
        files=(RawUSDCacheFile("artifact.usd", 8, "f" * 64),),
        runtime_versions=_runtime(),
    )
    role_cache = RoleUSDCache(role_root)
    calls = 0

    def bake(artifact_dir: Path, usd_file_name: str) -> Path:
        nonlocal calls
        calls += 1
        usd_path = artifact_dir / usd_file_name
        usd_path.write_text(f"role bake {calls}\n", encoding="utf-8")
        return usd_path

    request = _role_usd_request(
        record, _entity(), _entry(collision_enabled=False), 0, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
    )
    cold = role_cache.materialize(request, bake)
    cold_hashes = {item.path: item.sha256 for item in cold.record.files}
    warm = role_cache.materialize(request, bake)

    assert calls == 1
    assert cold.record == warm.record
    assert warm.hit and not cold.hit
    assert cold.record.usd_path.read_text(encoding="utf-8") == "role bake 1\n"
    assert warm.record.files == cold.record.files
    assert {item.path: item.sha256 for item in warm.record.files} == cold_hashes

    warm.record.usd_path.write_text("corrupted role bake\n", encoding="utf-8")
    recovered = role_cache.materialize(request, bake)
    assert calls == 2 and not recovered.hit
    assert recovered.record.usd_path.read_text(encoding="utf-8") == "role bake 2\n"

    changed = _role_usd_request(
        record, _entity(), _entry(collision_enabled=True), 0, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
    )
    collision_role = role_cache.materialize(changed, bake)
    assert not collision_role.hit and calls == 3
    assert collision_role.record.identity != cold.record.identity


def test_role_cache_environment_and_worker_payload_are_independent(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_ROLE_USD_CACHE, str(tmp_path / "role-cache"))
    assert resolve_role_usd_cache_root() == tmp_path / "role-cache"
    monkeypatch.setenv(ENV_ROLE_USD_CACHE, "disabled")
    assert resolve_role_usd_cache_root() is None

    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._requested_render_mode = None
    backend._render_width = 320
    backend._render_height = 240
    backend._entity_scene = None
    backend._physx_solver = PhysxSolverConfig()
    monkeypatch.setenv(ENV_RAW_USD_CACHE, str(tmp_path / "raw-cache"))
    monkeypatch.setenv(ENV_ROLE_USD_CACHE, str(tmp_path / "role-cache"))
    payload = backend._worker_init_payload()
    assert payload["raw_usd_cache_dir"] == str(tmp_path / "raw-cache")
    assert payload["role_usd_cache_dir"] == str(tmp_path / "role-cache")


def test_role_identity_tracks_baked_solver_offsets() -> None:
    raw_record = SimpleNamespace(
        identity="d" * 64,
        source_digest="e" * 64,
        usd_relative_path="artifact.usd",
        files=(RawUSDCacheFile("artifact.usd", 8, "f" * 64),),
        runtime_versions=_runtime(),
    )
    base = _role_usd_request(
        raw_record, _entity(), _entry(collision_enabled=True), 2, require_bodies=False,
        contact_offset=None,
        rest_offset=None,
    )
    contact = _role_usd_request(
        raw_record, _entity(), _entry(collision_enabled=True), 2, require_bodies=False,
        contact_offset=0.002,
        rest_offset=None,
    )
    both = _role_usd_request(
        raw_record, _entity(), _entry(collision_enabled=True), 2, require_bodies=False,
        contact_offset=0.002,
        rest_offset=0.001,
    )
    same = _role_usd_request(
        raw_record, _entity(), _entry(collision_enabled=True), 2, require_bodies=False,
        contact_offset=0.002,
        rest_offset=0.001,
    )
    assert len({base.identity, contact.identity, both.identity}) == 3
    assert same.identity == both.identity
    assert both.parameters["bake"]["contact_offset"] == 0.002
    assert both.parameters["bake"]["rest_offset"] == 0.001
