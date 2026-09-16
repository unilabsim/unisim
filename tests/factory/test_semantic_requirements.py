from __future__ import annotations

import pytest

import unisim.factory as factory
from unisim import FakeBackend
from unisim.dr.types import DomainRandomizationCapabilities
from unisim.validation import SemanticRequirements, SemanticValidationError


def test_unknown_feature_rejected_before_sdk_dispatch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("strict unsupported request must fail before SDK construction")

    monkeypatch.setattr(factory, "_create_backend", forbidden)
    with pytest.raises(SemanticValidationError, match="not.a.feature"):
        factory.create_backend(
            "isaacsim",
            semantic_requirements=SemanticRequirements(
                features=("not.a.feature",),
            ),
        )


def test_failed_post_materialization_check_closes_resources(monkeypatch):
    backend = FakeBackend()
    cleaned = []
    monkeypatch.setattr(backend, "cleanup_scene_assets", lambda: cleaned.append(True))
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    with pytest.raises(SemanticValidationError, match="effective value is unknown"):
        factory.create_backend("fake", semantic_requirements=SemanticRequirements(settings=("dt",)))
    assert cleaned == [True]


def test_authoritative_instance_capability_checked_after_construction(monkeypatch):
    monkeypatch.setattr(
        FakeBackend,
        "get_dr_capabilities",
        lambda self: DomainRandomizationCapabilities(supported_reset_terms=frozenset({"gravity"})),
    )
    backend = factory.create_backend(
        "fake",
        semantic_requirements=SemanticRequirements(
            features=("dr.reset.gravity",),
        ),
    )
    assert isinstance(backend, FakeBackend)


def test_invalid_requirements_rejected_without_dispatch():
    with pytest.raises(TypeError, match="SemanticRequirements"):
        factory.create_backend("fake", semantic_requirements={"features": ["x"]})


def test_strict_path_materializes_before_readback(monkeypatch):
    from unisim.inspection import ConfigurationField, ConfigurationProvenance, ImportReport

    backend = FakeBackend()

    def report():
        assert backend._materialized
        return ImportReport(
            "fake",
            (
                ConfigurationField(
                    "dt",
                    0.01,
                    0.01,
                    "exact",
                    (ConfigurationProvenance("engine_readback", "fixture handshake"),),
                ),
            ),
            lifecycle="materialization",
        )

    monkeypatch.setattr(backend, "get_import_report", report)
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    assert (
        factory.create_backend(
            "fake",
            semantic_requirements=SemanticRequirements(
                settings=("dt",),
            ),
        )
        is backend
    )


def test_default_path_preserves_lazy_materialization():
    backend = factory.create_backend("fake")
    assert not backend._materialized


def test_materialization_failure_cleans_resources_and_preserves_original_error(monkeypatch):
    backend = FakeBackend()

    def materialize():
        raise ValueError("handshake failed")

    def cleanup():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(backend, "materialize", materialize)
    monkeypatch.setattr(backend, "cleanup_scene_assets", cleanup)
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    with pytest.raises(ValueError, match="handshake failed") as exc:
        factory.create_backend("fake", semantic_requirements=SemanticRequirements())
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_factory_rejects_self_reported_configuration_without_readback(monkeypatch):
    from unisim import CapabilityCondition

    backend = FakeBackend()
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    with pytest.raises(SemanticValidationError, match="not established"):
        factory.create_backend(
            "fake",
            mode="native",
            semantic_requirements=SemanticRequirements(
                configuration=(CapabilityCondition("mode", "native"),),
            ),
        )


def test_factory_accepts_condition_matching_effective_scalar_readback(monkeypatch):
    from unisim import CapabilityCondition
    from unisim.inspection import ConfigurationField, ConfigurationProvenance, ImportReport

    backend = FakeBackend()
    backend._import_report = ImportReport(
        "fake",
        (
            ConfigurationField(
                "solver",
                "newton",
                "newton",
                "exact",
                (ConfigurationProvenance("engine_readback", "fixture"),),
            ),
        ),
    )
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    assert (
        factory.create_backend(
            "fake",
            semantic_requirements=SemanticRequirements(
                configuration=(CapabilityCondition("solver", "newton"),),
            ),
        )
        is backend
    )


def test_runtime_verified_request_does_not_accept_source_only_instance(monkeypatch):
    monkeypatch.setattr(
        FakeBackend,
        "get_dr_capabilities",
        lambda self: DomainRandomizationCapabilities(supported_reset_terms=frozenset({"gravity"})),
    )
    with pytest.raises(SemanticValidationError, match="no passing runtime evidence"):
        factory.create_backend(
            "fake",
            semantic_requirements=SemanticRequirements(
                features=("dr.reset.gravity",),
                require_runtime_verified=True,
            ),
        )


def test_empty_requirements_do_not_allow_wrong_actual_profile():
    with pytest.raises(SemanticValidationError, match="actual backend/profile"):
        factory.create_backend("fake", semantic_requirements=SemanticRequirements(profile="other"))


@pytest.mark.parametrize("actual", [True, False, None])
def test_approximation_condition_cannot_spoof_actual_constructor_flag(monkeypatch, actual):
    from unisim import CapabilityCondition

    backend = FakeBackend()
    backend.backend_type = "superdex"
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    kwargs = {} if actual is None else {"superdex_allow_contact_approximation": actual}
    with pytest.raises(SemanticValidationError, match="not established"):
        factory.create_backend(
            "superdex",
            semantic_requirements=SemanticRequirements(
                features=("collision.rigid",),
                approximations=("collision.rigid",),
                configuration=(
                    CapabilityCondition("superdex_allow_contact_approximation", "true"),
                ),
            ),
            **kwargs,
        )


def test_real_mujoco_strict_settings_use_engine_readback(tmp_path):
    pytest.importorskip("mujoco")
    from unisim import CapabilityCondition
    from unisim.scene import SceneCfg

    model = tmp_path / "strict.xml"
    model.write_text(
        "<mujoco><worldbody><body name='base'><joint name='slide' type='slide'/>"
        "<geom type='sphere' size='.1'/></body></worldbody>"
        "<actuator><motor joint='slide'/></actuator></mujoco>"
    )
    backend = factory.create_backend(
        "mujoco",
        SceneCfg(model_file=str(model)),
        sim_dt=0.005,
        semantic_requirements=SemanticRequirements(
            features=("asset.mjcf",),
            settings=("dt", "solver"),
            configuration=(CapabilityCondition("dt", "0.005"),),
        ),
    )
    try:
        assert backend._pool is not None
        assert (
            next(
                item for item in backend.get_import_report().fields if item.field == "dt"
            ).effective
            == 0.005
        )
    finally:
        backend.cleanup_scene_assets()


def test_condition_cannot_bypass_setting_approximation_consent(monkeypatch):
    from dataclasses import replace

    from unisim import CapabilityCondition
    from unisim.inspection import ConfigurationField, ConfigurationProvenance, ImportReport

    backend = FakeBackend()
    backend._import_report = ImportReport(
        "fake",
        (
            ConfigurationField(
                "collision_filter",
                "pairs",
                "bodies",
                "approximate",
                (ConfigurationProvenance("adapter_setting", "fixture mapping"),),
            ),
        ),
    )
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    requirements = SemanticRequirements(
        configuration=(CapabilityCondition("collision_filter", "bodies"),),
    )
    with pytest.raises(SemanticValidationError, match="relies on an approximation"):
        factory.create_backend("fake", semantic_requirements=requirements)
    assert (
        factory.create_backend(
            "fake",
            semantic_requirements=replace(
                requirements,
                settings=("collision_filter",),
                approximations=("collision_filter",),
            ),
        )
        is backend
    )


def test_failed_strict_check_closes_native_runtime_then_scene_assets(monkeypatch):
    backend = FakeBackend()
    calls = []
    monkeypatch.setattr(backend, "close", lambda: calls.append("close"), raising=False)
    monkeypatch.setattr(backend, "cleanup_scene_assets", lambda: calls.append("cleanup"))
    monkeypatch.setattr(factory, "_create_backend", lambda *args, **kwargs: backend)
    with pytest.raises(SemanticValidationError):
        factory.create_backend("fake", semantic_requirements=SemanticRequirements(settings=("dt",)))
    assert calls == ["close", "cleanup"]
