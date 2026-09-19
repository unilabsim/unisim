"""SuperDex semantic support declarations."""

from __future__ import annotations

from unisim.capabilities import SupportLevel, get_adapter_capabilities


def test_superdex_multiple_entities_are_supported_in_bounded_profile() -> None:
    for variant in ("none", "fixed"):
        for kinematic in ("none", "none_or_physical"):
            declaration = get_adapter_capabilities("superdex").get(
                "entity.multiple",
                configuration={
                    "entity.asset_format": "mjcf",
                    "entity.variant": variant,
                    "entity.kinematic": kinematic,
                },
            )
            assert declaration.support is SupportLevel.EXACT
            assert "same-layout fixed variants" in declaration.reason
    assert any(
        evidence.source == "https://github.com/unilabsim/unisim/issues/124"
        and evidence.scope.adapter_version == "superdex-portable-entities-v4"
        for evidence in declaration.evidence
    )
    assert "SceneBatchExecutorV2" in declaration.reason
    assert "SceneBatchExecutorV3 ABI 3" in declaration.reason
    assert "Physical kinematic roots retain source-declared collision" in declaration.reason
    assert "hidden six-DoF free-root carrier" in declaration.reason
    assert "no public state/control" in declaration.reason
    assert "mirror contact sensors" in declaration.reason
