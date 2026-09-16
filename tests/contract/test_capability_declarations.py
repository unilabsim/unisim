"""Semantic declarations remain independent from runtime verification."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from unisim import (
    CapabilityCondition,
    CapabilityDeclaration,
    CapabilityEvidence,
    CapabilityReport,
    CapabilityScope,
    FakeBackend,
    SupportLevel,
)
from unisim.backend.base import BackendPlayCapabilities
from unisim.dr.types import DomainRandomizationCapabilities, FixedVariantLayout


def runtime_scope() -> CapabilityScope:
    return CapabilityScope("fixture", "cpu", "1.0", "abc123", "2.0", "linux-x86_64", "cpu", True)


@pytest.mark.parametrize("support", list(SupportLevel))
def test_support_is_orthogonal_to_runtime_verification(support: SupportLevel) -> None:
    scope = runtime_scope()
    source = CapabilityEvidence("source", "src/fixture.py:20", scope)
    declaration = CapabilityDeclaration(
        "contact.query", support, "Fixture semantics", evidence=(source,)
    )
    assert not declaration.runtime_verified(scope)
    declaration = replace(
        declaration, evidence=(CapabilityEvidence("runtime", "run/fixture/test_contact", scope),)
    )
    assert declaration.runtime_verified(scope) == (
        support in (SupportLevel.EXACT, SupportLevel.APPROXIMATE)
    )
    assert declaration.support is support


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter", "another"),
        ("profile", "gpu"),
        ("unisim_version", "1.1"),
        ("adapter_version", "other-commit"),
        ("engine_version", "2.1"),
        ("platform", "darwin-arm64"),
        ("device", "cuda:0"),
        ("runtime_available", False),
        ("runtime_available", None),
        ("engine_version", None),
        ("unisim_version", None),
        ("adapter_version", None),
        ("platform", None),
        ("device", None),
    ],
)
def test_evidence_never_inherits_verification_across_scope(field: str, value: object) -> None:
    scope = runtime_scope()
    declaration = CapabilityDeclaration(
        "asset.mjcf",
        SupportLevel.EXACT,
        "Fixture",
        evidence=(CapabilityEvidence("runtime", "run/1", scope),),
    )
    assert not declaration.runtime_verified(replace(scope, **{field: value}))
    unknown_evidence = replace(declaration.evidence[0], scope=replace(scope, **{field: value}))
    assert not replace(declaration, evidence=(unknown_evidence,)).runtime_verified(scope)


@pytest.mark.parametrize(
    "result,revoked", [("failed", False), ("skipped", False), ("passed", True)]
)
def test_unsuccessful_or_revoked_evidence_is_not_verification(result: str, revoked: bool) -> None:
    evidence = CapabilityEvidence(
        "runtime", "run/1", runtime_scope(), result=result, revoked=revoked
    )
    assert not evidence.verifies(runtime_scope())


def test_conditions_unknown_defaults_and_json_round_trip() -> None:
    scope = runtime_scope()
    declarations = tuple(
        CapabilityDeclaration(
            feature,
            SupportLevel.EXACT,
            "Fixture conditional declaration",
            conditions=(
                CapabilityCondition("profile", "cpu"),
                CapabilityCondition("solver", "newton"),
            ),
            evidence=(CapabilityEvidence("source", "fixture/source", scope),),
        )
        for feature in (
            "asset.mjcf",
            "entity.articulation",
            "root.free",
            "joint.hinge",
            "actuator.motor",
            "collision.rigid",
            "contact.query",
            "terrain.heightfield",
            "sensor.imu",
            "reset.state",
            "wrench.body_force",
            "state.refresh",
            "variant.same_layout",
        )
    )
    report = CapabilityReport(scope, declarations)
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    restored = CapabilityReport.from_dict(json.loads(serialized))
    assert restored == report
    assert [item.feature for item in report.declarations] == sorted(
        item.feature for item in declarations
    )
    assert json.dumps(restored.to_dict(), sort_keys=True) == serialized
    for declaration in declarations:
        assert report.get(declaration.feature).support is SupportLevel.UNKNOWN
        assert (
            report.get(declaration.feature, configuration={"solver": "newton"}).support
            is SupportLevel.EXACT
        )
        assert (
            report.get(declaration.feature, configuration={"solver": "pgs"}).support
            is SupportLevel.UNKNOWN
        )
    assert report.get("not.recorded").support is SupportLevel.UNKNOWN
    assert report.get("not.recorded").evidence == ()
    changed = replace(report, scope=replace(scope, profile="gpu"))
    assert (
        changed.get("asset.mjcf", configuration={"solver": "newton", "profile": "cpu"}).support
        is SupportLevel.UNKNOWN
    )


def test_invalid_empty_identifiers_duplicates_and_unknown_schema_fail_closed() -> None:
    with pytest.raises(ValueError, match="engine_version"):
        CapabilityScope("fake", engine_version="")
    with pytest.raises(ValueError, match="source"):
        CapabilityEvidence("source", " ", CapabilityScope("fake"))
    declaration = CapabilityDeclaration("asset.mjcf", SupportLevel.UNKNOWN, "Not audited")
    with pytest.raises(ValueError, match="unique"):
        CapabilityReport(CapabilityScope("fake"), (declaration, declaration))
    with pytest.raises(ValueError, match="schema_version"):
        CapabilityReport(CapabilityScope("fake"), schema_version=2)
    with pytest.raises(TypeError, match="conditions"):
        CapabilityDeclaration("asset.mjcf", SupportLevel.EXACT, "Fixture", conditions=[object()])


class MutableDeclarationsBackend(FakeBackend):
    dr = DomainRandomizationCapabilities()

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return self.dr


def assert_aggregate_matches_sources(backend: FakeBackend) -> None:
    report = backend.get_capabilities()
    dr = backend.get_dr_capabilities()
    for term in ("body_mass", "gravity"):
        assert (
            report.get("dr.reset." + term).support is SupportLevel.EXACT
        ) == dr.supports_reset_term(term)
    for term in ("push", "body_force", "body_torque"):
        assert (
            report.get("dr.interval." + term).support is SupportLevel.EXACT
        ) == dr.supports_interval_term(term)
    for layout in FixedVariantLayout:
        assert (report.get("variant." + layout.value).support is SupportLevel.EXACT) == (
            dr.supports_fixed_variants and layout in dr.supported_fixed_variant_layouts
        )
    play = backend.get_play_capabilities()
    for field in fields(play):
        assert (
            report.get("play." + field.name.removeprefix("supports_")).support is SupportLevel.EXACT
        ) == getattr(play, field.name)
    assert all(not item.runtime_verified(report.scope) for item in report.declarations)


def test_aggregation_tracks_authoritative_sources_and_legacy_interval_fallback() -> None:
    backend = MutableDeclarationsBackend()
    assert_aggregate_matches_sources(backend)
    before = backend.get_capabilities()
    backend.dr = DomainRandomizationCapabilities(
        supported_reset_terms=frozenset({"gravity"}),
        supports_interval_body_force=True,
        supported_interval_terms=frozenset({"body_torque"}),
        supports_fixed_variants=True,
        supported_fixed_variant_layouts=frozenset({FixedVariantLayout.SAME_LAYOUT}),
    )
    backend._play_capabilities = BackendPlayCapabilities(supports_debug_overlay=True)
    assert_aggregate_matches_sources(backend)
    assert before.get("dr.reset.gravity").support is SupportLevel.UNSUPPORTED
    assert backend.get_capabilities().get("dr.reset.gravity").support is SupportLevel.EXACT
    assert backend.get_capabilities().get("play.debug_overlay").support is SupportLevel.EXACT


def test_real_mujoco_adapter_reuses_existing_capabilities(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    from unisim import MuJoCoBackend
    from unisim.scene import SceneCfg

    model = tmp_path / "capabilities.xml"
    model.write_text(
        "<mujoco><worldbody><body name='base'><joint name='slide' type='slide'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<actuator><motor joint='slide'/></actuator></mujoco>"
    )
    backend = MuJoCoBackend(SceneCfg(model_file=str(model)), num_envs=1, sim_dt=0.01)
    assert_aggregate_matches_sources(backend)


def test_instance_does_not_claim_existing_capabilities_for_another_profile() -> None:
    backend = MutableDeclarationsBackend()
    backend.dr = DomainRandomizationCapabilities(supported_reset_terms=frozenset({"gravity"}))
    assert backend.get_capabilities().get("dr.reset.gravity").support is SupportLevel.EXACT
    changed = backend.get_capabilities(profile="unrecorded-profile")
    assert changed.get("dr.reset.gravity").support is SupportLevel.UNKNOWN


@pytest.mark.parametrize("tracking,callback", [(True, True), (True, False), (False, True)])
def test_mujoco_state_refresh_declarations_follow_actual_flags(tmp_path, tracking, callback):
    pytest.importorskip("mujoco")
    from unisim import MuJoCoBackend
    from unisim.scene import SceneCfg

    model = tmp_path / "refresh.xml"
    model.write_text(
        "<mujoco><worldbody><body name='base'><joint name='slide' type='slide'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<actuator><motor joint='slide'/></actuator></mujoco>"
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model)),
        num_envs=1,
        sim_dt=0.01,
        add_body_sensors=tracking,
        refresh_pre_step_body_state=callback,
        base_name="base",
    )
    report = backend.get_capabilities()
    assert (report.get("state.final_refresh").support is SupportLevel.EXACT) == tracking
    assert (report.get("state.callback_refresh").support is SupportLevel.EXACT) == (
        tracking and callback
    )
    if not callback:
        assert (
            report.get(
                "state.callback_refresh",
                configuration={
                    "refresh_pre_step_body_state": "true",
                    "add_body_sensors": "true",
                },
            ).support
            is SupportLevel.UNSUPPORTED
        )
    assert (
        backend.get_capabilities(profile="other").get("state.callback_refresh").support
        is SupportLevel.UNKNOWN
    )
