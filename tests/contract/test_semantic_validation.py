from __future__ import annotations

import json
from dataclasses import replace

import pytest

from unisim.capabilities import (
    CapabilityCondition,
    CapabilityDeclaration,
    CapabilityReport,
    CapabilityScope,
    SupportLevel,
)
from unisim.inspection import (
    ConfigurationField,
    ConfigurationProvenance,
    ConfigurationScope,
    ImportReport,
)
from unisim.validation import (
    SemanticRequirements,
    SemanticValidationError,
    validate_semantic_requirements,
)


def _capabilities(support=SupportLevel.EXACT, *, profile="default"):
    return CapabilityReport(
        CapabilityScope("fixture", profile),
        (CapabilityDeclaration("collision.contact", support, "fixture contact semantics"),),
    )


def _setting(*, difference="exact", variant=None):
    return ConfigurationField(
        "collision_filter",
        requested="pair",
        effective="pair" if difference == "exact" else "body",
        difference=difference,
        provenance=(ConfigurationProvenance("adapter_setting", "fixture materialization"),),
        scope=ConfigurationScope(variant=variant),
        reason="fixture mapping",
    )


def test_exact_and_approved_approximation_pass():
    requirements = SemanticRequirements(features=("collision.contact",))
    validate_semantic_requirements(_capabilities(), requirements)
    with pytest.raises(SemanticValidationError, match="approximation requires explicit consent"):
        validate_semantic_requirements(_capabilities(SupportLevel.APPROXIMATE), requirements)
    validate_semantic_requirements(
        _capabilities(SupportLevel.APPROXIMATE),
        replace(requirements, approximations=("collision.contact",)),
    )


@pytest.mark.parametrize("support", [SupportLevel.UNKNOWN, SupportLevel.UNSUPPORTED])
def test_unknown_and_unsupported_never_authorized_by_approximation(support):
    with pytest.raises(SemanticValidationError, match="fixture/default.*collision.contact"):
        validate_semantic_requirements(
            _capabilities(support),
            SemanticRequirements(
                features=("collision.contact",),
                approximations=("collision.contact",),
            ),
        )


def test_profile_and_conditions_cannot_be_bypassed():
    declaration = replace(
        _capabilities().declarations[0], conditions=(CapabilityCondition("mode", "native"),)
    )
    report = replace(_capabilities(), declarations=(declaration,))
    requirements = SemanticRequirements(features=("collision.contact",))
    with pytest.raises(SemanticValidationError, match="conditions"):
        validate_semantic_requirements(report, requirements)
    validate_semantic_requirements(
        report, replace(requirements, configuration=(CapabilityCondition("mode", "native"),))
    )
    with pytest.raises(SemanticValidationError, match="profile"):
        validate_semantic_requirements(report, replace(requirements, profile="approximation"))


def test_source_only_support_is_not_runtime_verified():
    with pytest.raises(SemanticValidationError, match="no passing runtime evidence"):
        validate_semantic_requirements(
            _capabilities(),
            SemanticRequirements(
                features=("collision.contact",),
                require_runtime_verified=True,
            ),
        )


def test_every_variant_setting_checked_and_field_consent_scoped():
    requirements = SemanticRequirements(settings=("collision_filter",))
    report = ImportReport(
        "fixture",
        (
            _setting(variant="first"),
            _setting(
                difference="approximate",
                variant="second",
            ),
        ),
    )
    with pytest.raises(SemanticValidationError, match="materialized approximation"):
        validate_semantic_requirements(_capabilities(), requirements, report)
    validate_semantic_requirements(
        _capabilities(),
        replace(
            requirements,
            approximations=("collision_filter",),
        ),
        report,
    )
    unknown = replace(report, fields=report.fields + (ConfigurationField("collision_filter"),))
    with pytest.raises(SemanticValidationError, match="unknown"):
        validate_semantic_requirements(
            _capabilities(),
            replace(
                requirements,
                approximations=("collision_filter",),
            ),
            unknown,
        )


def test_missing_or_wrong_report_is_rejected():
    requirements = SemanticRequirements(settings=("dt",))
    for report in (None, ImportReport("fixture"), ImportReport("other")):
        with pytest.raises(SemanticValidationError):
            validate_semantic_requirements(_capabilities(), requirements, report)


def test_requirements_json_round_trip_and_bad_consent():
    requirements = SemanticRequirements(
        features=("collision.contact",),
        configuration=(CapabilityCondition("mode", "native"),),
        approximations=("collision.contact",),
    )
    payload = json.loads(json.dumps(requirements.to_dict()))
    assert SemanticRequirements.from_dict(payload) == requirements
    with pytest.raises(ValueError, match="explicitly required"):
        SemanticRequirements(approximations=("everything",))
    with pytest.raises(ValueError, match="profile field"):
        SemanticRequirements(configuration=(CapabilityCondition("profile", "other"),))


def test_variant_rejection_identifies_entity_environment_and_source():
    report = ImportReport("fixture", (_setting(difference="approximate", variant="second"),))
    with pytest.raises(SemanticValidationError) as exc:
        validate_semantic_requirements(
            _capabilities(),
            SemanticRequirements(
                settings=("collision_filter",),
            ),
            report,
        )
    message = str(exc.value)
    assert "entity='scene'" in message
    assert "env_ids=None" in message
    assert "variant='second'" in message
    assert "fixture materialization" in message


@pytest.mark.parametrize("effective", ["other", None, ["native"]])
def test_materialized_conditions_reject_unproven_or_mismatching_values(effective):
    requirements = SemanticRequirements(configuration=(CapabilityCondition("mode", "native"),))
    field = ConfigurationField(
        "mode",
        requested="native",
        effective=effective,
        difference="unknown" if effective is None else "overridden",
        provenance=(ConfigurationProvenance("adapter_setting", "fixture mode"),),
    )
    with pytest.raises(SemanticValidationError, match="not established"):
        validate_semantic_requirements(
            _capabilities(), requirements, ImportReport("fixture", (field,))
        )


def test_materialized_conditions_validate_all_scopes_and_approximation_consent():
    requirements = SemanticRequirements(
        configuration=(CapabilityCondition("collision_filter", "body"),),
    )
    report = ImportReport("fixture", (_setting(difference="approximate", variant="second"),))
    with pytest.raises(SemanticValidationError, match="relies on an approximation"):
        validate_semantic_requirements(_capabilities(), requirements, report)
    approved = replace(
        requirements, settings=("collision_filter",), approximations=("collision_filter",)
    )
    validate_semantic_requirements(_capabilities(), approved, report)
    report = replace(report, fields=report.fields + (_setting(variant="first"),))
    with pytest.raises(SemanticValidationError, match="variant='first'"):
        validate_semantic_requirements(_capabilities(), approved, report)


def test_materialized_conditions_require_a_report_field():
    requirements = SemanticRequirements(configuration=(CapabilityCondition("mode", "native"),))
    with pytest.raises(SemanticValidationError, match="not established"):
        validate_semantic_requirements(_capabilities(), requirements, ImportReport("fixture"))
