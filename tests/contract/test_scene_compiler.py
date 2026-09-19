"""SDK-free contracts for portable scene identity and source-intent reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unisim import scene_compiler
from unisim.inspection import ConfigurationField, ConfigurationProvenance
from unisim.scene_compiler import (
    PORTABLE_MJCF_PROFILE,
    PORTABLE_MJCF_PROFILE_ID,
    SceneCompilerParameters,
    SceneContentIdentity,
    SceneIntentReport,
    SceneResourceProvenance,
    SceneSourceProvenance,
    compute_scene_content_identity,
    derive_scene_artifact_identity,
)


def _resource(logical_path: str, digest: str) -> SceneResourceProvenance:
    return SceneResourceProvenance("mesh", logical_path, "/absolute/ignored.png", digest)


def _source(*, digest: str, resource_digest: str = "b" * 64) -> SceneSourceProvenance:
    return SceneSourceProvenance(
        "robot", "mjcf", "/absolute/ignored.xml", digest, (_resource("mesh.obj", resource_digest),)
    )


def _parameters(sim_dt: float = 0.002) -> SceneCompilerParameters:
    return SceneCompilerParameters(
        PORTABLE_MJCF_PROFILE_ID,
        PORTABLE_MJCF_PROFILE.structural_oracle,
        "3.11.0",
        sim_dt,
    )


def test_content_identity_hashes_content_not_absolute_locations() -> None:
    source = _source(digest="a" * 64)
    identity = compute_scene_content_identity((source,), _parameters())
    relocated = SceneSourceProvenance(
        source.entity,
        source.format,
        "/another/location.xml",
        source.source_digest,
        (_resource("mesh.obj", "b" * 64),),
    )
    assert compute_scene_content_identity((relocated,), _parameters()) == identity

    changed_resource = _source(digest="a" * 64, resource_digest="c" * 64)
    changed_compiler = _parameters(sim_dt=0.003)
    changed_pose = SceneSourceProvenance(
        source.entity,
        source.format,
        source.source_path,
        source.source_digest,
        source.resources,
        initial_position=(1.0, 2.0, 3.0),
    )
    assert compute_scene_content_identity((changed_resource,), _parameters()) != identity
    assert compute_scene_content_identity((source,), changed_compiler) != identity
    assert compute_scene_content_identity((changed_pose,), _parameters()) != identity


def test_artifact_identity_extends_but_never_replaces_canonical_identity() -> None:
    identity = compute_scene_content_identity((_source(digest="a" * 64),), _parameters())
    raw = derive_scene_artifact_identity(
        identity, "isaacsim.raw-usd", {"importer": "MjcfConverter", "runtime": "5.1.0"}
    )
    role = derive_scene_artifact_identity(
        identity, "isaacsim.role-usd", {"role": "visual", "collision": False}
    )
    assert raw != identity.canonical_identity
    assert role != raw
    assert derive_scene_artifact_identity(
        identity, "isaacsim.raw-usd", {"importer": "MjcfConverter", "runtime": "5.1.0"}
    ) == raw


def test_sensor_fragment_digests_validate_and_round_trip() -> None:
    digest = "d" * 64
    parameters = SceneCompilerParameters(
        PORTABLE_MJCF_PROFILE_ID,
        PORTABLE_MJCF_PROFILE.structural_oracle,
        "3.11.0",
        0.002,
        sensor_fragment_digests=(digest,),
    )
    decoded = SceneCompilerParameters.from_dict(parameters.to_dict())
    assert decoded == parameters
    assert decoded.identity_payload() == parameters.identity_payload()

    with pytest.raises(ValueError, match="sensor fragment digests"):
        SceneCompilerParameters(
            PORTABLE_MJCF_PROFILE_ID,
            PORTABLE_MJCF_PROFILE.structural_oracle,
            "3.11.0",
            0.002,
            sensor_fragment_digests=("not-a-digest",),
        )


def test_intent_report_serializes_source_provenance_without_effective_claims() -> None:
    source = _source(digest="a" * 64)
    parameters = _parameters()
    identity = compute_scene_content_identity((source,), parameters)
    report = SceneIntentReport(
        PORTABLE_MJCF_PROFILE_ID,
        parameters,
        (source,),
        identity,
        (
            ConfigurationField(
                "dt",
                0.002,
                provenance=(ConfigurationProvenance("source", "portable compiler"),),
                unit="s",
            ),
        ),
    )
    decoded = SceneIntentReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert decoded == report
    assert report.fields[0].effective is None

    with pytest.raises(ValueError, match="native effective"):
        SceneIntentReport(
            PORTABLE_MJCF_PROFILE_ID,
            parameters,
            (source,),
            identity,
            (
                ConfigurationField(
                    "dt",
                    0.002,
                    0.002,
                    "exact",
                    (ConfigurationProvenance("engine_readback", "fixture"),),
                ),
            ),
        )


def test_compiler_parameters_and_identity_records_reject_non_json_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="sim_dt"):
        SceneCompilerParameters(PORTABLE_MJCF_PROFILE_ID, "oracle", "1", float("nan"))
    with pytest.raises(ValueError, match="unsupported portable"):
        SceneCompilerParameters("other-profile", "oracle", "1", 0.1)
    with pytest.raises(ValueError, match="assignment"):
        SceneCompilerParameters(
            PORTABLE_MJCF_PROFILE_ID, "oracle", "1", 0.1, assignment=(True,)
        )
    with pytest.raises(ValueError, match="digest"):
        _source(digest="not-a-digest")
    with pytest.raises(ValueError, match="schema version"):
        SceneContentIdentity(PORTABLE_MJCF_PROFILE_ID, 2, "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(TypeError, match="finite JSON"):
        derive_scene_artifact_identity(
            compute_scene_content_identity((_source(digest="a" * 64),), _parameters()),
            "stage",
            {"bad": object()},
        )


def test_missing_compiler_dependency_has_actionable_lazy_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingImports:
        @staticmethod
        def import_module(name: str) -> object:
            raise ModuleNotFoundError(name=name)

    def missing_version(name: str) -> str:
        raise scene_compiler.PackageNotFoundError(name)

    monkeypatch.setattr(scene_compiler, "importlib", MissingImports)
    monkeypatch.setattr(scene_compiler, "version", missing_version)
    with pytest.raises(ImportError, match=r"unisim-core\[scene-compiler\]"):
        scene_compiler.load_portable_mjcf_compiler()


def test_compiler_contract_module_never_imports_an_sdk(tmp_path: Path) -> None:
    source = Path(scene_compiler.__file__).read_text(encoding="utf-8")
    assert "import mujoco" not in source
    assert "scene-compiler" in source
