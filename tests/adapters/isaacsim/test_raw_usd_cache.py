"""SDK-free contracts for the immutable IsaacSim raw-USD cache."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unisim.backend.isaacsim import dependencies as isaacsim_dependencies
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.dependencies import IsaacSimRuntime, build_worker_env
from unisim.backend.isaacsim.raw_usd_cache import (
    ENV_RAW_USD_CACHE,
    RAW_USD_ARTIFACT_STAGE,
    RawUSDArtifactRequest,
    RawUSDCache,
    file_sha256,
    resolve_raw_usd_cache_root,
)
from unisim.backend.isaacsim.scene_worker import _raw_usd_request
from unisim.scene_compiler import SceneContentIdentity, derive_scene_artifact_identity


def _identity(canonical: str = "c" * 64) -> SceneContentIdentity:
    return SceneContentIdentity("portable-mjcf-v1", 1, "a" * 64, "b" * 64, canonical)


def _parameters(*, fix_base: bool = False) -> dict[str, object]:
    return {
        "entity": "robot",
        "variant": 0,
        "expanded_source_sha256": "d" * 64,
        "converter": "MjcfConverter",
        "importer": {
            "fix_base": fix_base,
            "import_sites": False,
            "import_inertia_tensor": True,
            "link_density": 0.0,
            "make_instanceable": False,
            "self_collision": False,
            "force_usd_conversion": True,
            "usd_file": "artifact.usd",
        },
    }


def _runtime(importer: str = "2.5.13") -> dict[str, str]:
    return {
        "isaacsim": "5.1.0.0",
        "isaaclab": "0.47.2",
        "isaacsim.asset.importer.mjcf": importer,
        "python": "3.11.13",
    }


def _request(
    *,
    content_identity: SceneContentIdentity | None = None,
    source_digest: str = "d" * 64,
    parameters: dict[str, object] | None = None,
    runtime_versions: dict[str, str] | None = None,
) -> RawUSDArtifactRequest:
    parameters = _parameters() if parameters is None else parameters
    if parameters.get("expanded_source_sha256") != source_digest:
        parameters = copy.deepcopy(parameters)
        parameters["expanded_source_sha256"] = source_digest
    runtime_versions = _runtime() if runtime_versions is None else runtime_versions
    content_identity = _identity() if content_identity is None else content_identity
    identity = derive_scene_artifact_identity(
        content_identity,
        RAW_USD_ARTIFACT_STAGE,
        {"parameters": parameters, "runtime_versions": runtime_versions},
    )
    return RawUSDArtifactRequest(identity, source_digest, parameters, runtime_versions)


class Converter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, artifact_dir: Path, usd_file_name: str) -> Path:
        self.calls += 1
        usd_path = artifact_dir / usd_file_name
        usd_path.write_text(f"raw usd {self.calls}\n", encoding="utf-8")
        texture = artifact_dir / "textures" / "material.png"
        texture.parent.mkdir(parents=True)
        texture.write_bytes(b"raw texture\n")
        return usd_path


def _manifest(cache: RawUSDCache, request: RawUSDArtifactRequest) -> dict[str, object]:
    return json.loads(
        (cache.entries_root / request.identity / "manifest.json").read_text(encoding="utf-8")
    )


def test_cold_miss_publishes_a_complete_manifest_and_warm_hit_skips_conversion(
    tmp_path: Path,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    converter = Converter()
    request = _request()

    cold = cache.materialize(request, converter)
    warm = cache.materialize(request, converter)

    assert converter.calls == 1
    assert not cold.hit and warm.hit
    assert cold.record == warm.record
    assert cold.record.usd_relative_path == "artifact.usd"
    assert {item.path for item in cold.record.files} == {
        "artifact.usd",
        "textures/material.png",
    }
    manifest = _manifest(cache, request)
    assert manifest["identity"] == request.identity
    assert manifest["source_digest"] == request.source_digest
    assert manifest["parameters"] == request.parameters
    assert manifest["runtime_versions"] == request.runtime_versions
    assert not any(cache.staging_root.iterdir())


def test_source_importer_runtime_and_canonical_changes_create_new_entries(
    tmp_path: Path,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    converter = Converter()
    base_request = _request()
    assert cache.materialize(base_request, converter).hit is False

    changed_parameters = copy.deepcopy(_parameters())
    changed_parameters["importer"]["fix_base"] = True  # type: ignore[index]
    changed_runtime = _runtime(importer="2.5.14")
    requests = (
        _request(source_digest="e" * 64),
        _request(parameters=changed_parameters),
        _request(runtime_versions=changed_runtime),
        _request(content_identity=_identity(canonical="f" * 64)),
    )
    identities = {base_request.identity}
    for request in requests:
        assert request.identity not in identities
        assert cache.materialize(request, converter).hit is False
        identities.add(request.identity)
    assert converter.calls == len(requests) + 1


@pytest.mark.parametrize(
    "corruption",
    ["artifact", "missing", "extra", "symlink", "traversal", "manifest", "duplicate-json"],
)
def test_corrupt_entries_are_misses_and_are_replaced_by_complete_entries(
    tmp_path: Path,
    corruption: str,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    converter = Converter()
    request = _request()
    cache.materialize(request, converter)
    entry = cache.entries_root / request.identity
    artifact = entry / "artifact"
    manifest_path = entry / "manifest.json"

    if corruption == "artifact":
        (artifact / "artifact.usd").write_text("partial write", encoding="utf-8")
    elif corruption == "missing":
        (artifact / "textures" / "material.png").unlink()
    elif corruption == "extra":
        (artifact / "unexpected.usda").write_text("unexpected", encoding="utf-8")
    elif corruption == "symlink":
        texture = artifact / "textures" / "material.png"
        target = tmp_path / "external.png"
        target.write_bytes(b"external")
        texture.unlink()
        texture.symlink_to(target)
    elif corruption == "traversal":
        manifest = _manifest(cache, request)
        manifest["files"][0]["path"] = "../escaped.usd"  # type: ignore[index]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "manifest":
        manifest_path.write_text("{ interrupted", encoding="utf-8")
    else:
        manifest_path.write_text(
            json.dumps(_manifest(cache, request))[:-1]
            + ', "identity": "'
            + request.identity
            + '"}',
            encoding="utf-8",
        )

    assert cache.load(request.identity) is None
    repaired = cache.materialize(request, converter)
    assert converter.calls == 2 and repaired.hit is False
    assert cache.load(request.identity) == repaired.record
    assert not any(cache.staging_root.iterdir())


def test_a_valid_manifest_with_mismatched_request_metadata_does_not_count_as_a_hit(
    tmp_path: Path,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    converter = Converter()
    request = _request()
    cache.materialize(request, converter)
    manifest_path = cache.entries_root / request.identity / "manifest.json"
    manifest = _manifest(cache, request)
    manifest["source_digest"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = cache.materialize(request, converter)

    assert converter.calls == 2 and result.hit is False
    assert cache.load(request.identity) == result.record
    assert result.record.source_digest == request.source_digest


def test_converter_output_must_be_a_new_regular_file_inside_the_stage(tmp_path: Path) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    request = _request()
    outside = tmp_path / "outside.usd"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside the raw cache stage"):
        cache.materialize(request, lambda artifact, name: outside)
    assert not any(cache.entries_root.iterdir())
    assert not any(cache.staging_root.iterdir())

    def symlink_converter(artifact: Path, name: str) -> Path:
        target = artifact / "target.usd"
        target.write_text("target", encoding="utf-8")
        link = artifact / name
        link.symlink_to(target)
        return link

    with pytest.raises(ValueError, match="symlinks"):
        cache.materialize(request, symlink_converter)
    assert cache.load(request.identity) is None
    assert not any(cache.staging_root.iterdir())


def test_delayed_conversion_is_not_visible_until_atomic_publication(tmp_path: Path) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    request = _request()
    entered = threading.Event()
    release = threading.Event()

    def blocked_converter(artifact: Path, name: str) -> Path:
        entered.set()
        release.wait(timeout=5.0)
        path = artifact / name
        path.write_text("complete", encoding="utf-8")
        return path

    materializing = threading.Thread(
        target=lambda: cache.materialize(request, blocked_converter)
    )
    materializing.start()
    assert entered.wait(timeout=5.0)
    assert cache.load(request.identity) is None
    assert not (cache.entries_root / request.identity).exists()
    release.set()
    materializing.join(timeout=5.0)
    assert not materializing.is_alive()
    assert cache.load(request.identity) is not None


def test_concurrent_publishers_expose_one_valid_entry_and_never_partial_stages(
    tmp_path: Path,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    request = _request()
    barrier = threading.Barrier(2, timeout=5.0)

    def converter(artifact: Path, name: str) -> Path:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        path = artifact / name
        path.write_text(f"raw usd {threading.get_ident()}", encoding="utf-8")
        return path

    results: list[object] = []
    results_lock = threading.Lock()

    def materialize() -> None:
        result = cache.materialize(request, converter)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=materialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert len(results) == 2
    first, second = results
    assert first.record == second.record  # type: ignore[attr-defined]
    assert len(list((cache.entries_root).iterdir())) == 1
    assert not any(cache.staging_root.iterdir())


def test_role_copy_contains_all_files_and_mutation_does_not_change_the_raw_cache(
    tmp_path: Path,
) -> None:
    cache = RawUSDCache(tmp_path / "cache")
    converter = Converter()
    request = _request()
    result = cache.materialize(request, converter)
    cached_hashes = {
        item.path: item.sha256
        for item in cache.load(request.identity).files  # type: ignore[union-attr]
    }

    copied_usd = cache.copy_artifact(result.record, tmp_path / "roles" / "robot-0")
    assert copied_usd == tmp_path / "roles" / "robot-0" / "artifact.usd"
    copied_usd.write_text("role bake mutation", encoding="utf-8")

    current = cache.load(request.identity)
    assert current is not None
    assert {
        item.path: item.sha256 for item in current.files
    } == cached_hashes
    assert file_sha256(current.usd_path) == cached_hashes["artifact.usd"]


@pytest.mark.parametrize("value", ["", " ", "0", "false", "NO", "off", "none", "disabled"])
def test_cache_environment_can_be_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV_RAW_USD_CACHE, value)
    assert resolve_raw_usd_cache_root() is None


def test_preferred_and_legacy_environment_values_are_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(ENV_RAW_USD_CACHE, raising=False)
    monkeypatch.setenv("UNILAB_ISAACSIM_RAW_USD_CACHE", str(tmp_path / "legacy"))
    assert resolve_raw_usd_cache_root() == tmp_path / "legacy"
    monkeypatch.setenv(ENV_RAW_USD_CACHE, str(tmp_path / "preferred"))
    assert resolve_raw_usd_cache_root() == tmp_path / "preferred"


def test_backend_worker_payload_carries_resolved_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = IsaacSimBackend.__new__(IsaacSimBackend)
    backend._requested_render_mode = None
    backend._render_width = 320
    backend._render_height = 240
    backend._entity_scene = None
    monkeypatch.setenv(ENV_RAW_USD_CACHE, str(tmp_path / "cache"))
    assert backend._worker_init_payload()["raw_usd_cache_dir"] == str(tmp_path / "cache")
    monkeypatch.setenv(ENV_RAW_USD_CACHE, "disabled")
    assert backend._worker_init_payload()["raw_usd_cache_dir"] is None


def test_sdk_worker_environment_isolates_host_site_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    runtime = IsaacSimRuntime(
        python=tmp_path / "venv" / "bin" / "python",
        isaaclab_source=tmp_path / "IsaacLab" / "source",
    )
    package_root = Path(isaacsim_dependencies.__file__).resolve().parents[3]
    python_path = build_worker_env(runtime)["PYTHONPATH"].split(os.pathsep)
    assert str(package_root) not in python_path
    assert python_path[0] == str(runtime.isaaclab_source)


def test_worker_raw_identity_includes_expanded_source_importer_and_runtime_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "expanded.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    source_digest = file_sha256(source)
    entity = SimpleNamespace(name="robot", kind="articulation", root_mode="floating")
    runtime = _runtime()
    request = _raw_usd_request(_identity(), str(source), entity, 2, runtime, self_collision=False)
    expected_parameters = _parameters()
    expected_parameters["entity"] = "robot"
    expected_parameters["variant"] = 2
    expected_parameters["expanded_source_sha256"] = source_digest
    expected_identity = derive_scene_artifact_identity(
        _identity(),
        RAW_USD_ARTIFACT_STAGE,
        {"parameters": expected_parameters, "runtime_versions": runtime},
    )
    assert request.source_digest == source_digest
    assert request.parameters == expected_parameters
    assert request.runtime_versions == runtime
    assert request.identity == expected_identity

    fixed_entity = SimpleNamespace(
        name=entity.name, kind=entity.kind, root_mode="fixed"
    )
    changed_content = _identity(canonical="f" * 64)
    changed_runtime = _runtime(importer="2.5.14")
    changed_requests = (
        _raw_usd_request(
            _identity(), str(source), fixed_entity, 0, runtime, self_collision=False
        ),
        _raw_usd_request(changed_content, str(source), entity, 0, runtime, self_collision=False),
        _raw_usd_request(_identity(), str(source), entity, 0, changed_runtime,
                         self_collision=False),
        _raw_usd_request(_identity(), str(source), entity, 2, runtime, self_collision=True),
    )
    assert all(item.identity != request.identity for item in changed_requests)


def test_self_collision_is_a_raw_conversion_parameter_and_identity_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "expanded.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    entity = SimpleNamespace(name="robot", kind="articulation", root_mode="floating")
    runtime = _runtime()
    disabled = _raw_usd_request(_identity(), str(source), entity, 0, runtime,
                                self_collision=False)
    enabled = _raw_usd_request(_identity(), str(source), entity, 0, runtime,
                               self_collision=True)
    assert disabled.parameters["importer"]["self_collision"] is False
    assert enabled.parameters["importer"]["self_collision"] is True
    assert disabled.identity != enabled.identity


def test_changed_expanded_source_changes_worker_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    first.write_text("<mujoco><compiler angle='radian'/></mujoco>", encoding="utf-8")
    second.write_text("<mujoco><compiler angle='degree'/></mujoco>", encoding="utf-8")
    entity = SimpleNamespace(name="robot", kind="articulation", root_mode="floating")
    runtime = _runtime()
    first_request = _raw_usd_request(_identity(), str(first), entity, 0, runtime,
                                     self_collision=False)
    second_request = _raw_usd_request(_identity(), str(second), entity, 0, runtime,
                                      self_collision=False)
    assert first_request.source_digest != second_request.source_digest
    assert first_request.identity != second_request.identity
