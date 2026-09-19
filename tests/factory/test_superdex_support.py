"""SuperDex semantic support declarations."""

from __future__ import annotations

from unisim.capabilities import SupportLevel, get_adapter_capabilities


def test_superdex_multiple_entities_are_supported_in_bounded_profile() -> None:
    for variant in ("none", "fixed"):
        declaration = get_adapter_capabilities("superdex").get(
            "entity.multiple",
            configuration={
                "entity.asset_format": "mjcf",
                "entity.variant": variant,
                "entity.kinematic": "none",
            },
        )
        assert declaration.support is SupportLevel.EXACT
        assert "SceneBatchExecutorV2" in declaration.reason
        assert "same-layout fixed variants" in declaration.reason
    assert any(
        evidence.source == "https://github.com/unilabsim/unisim/issues/124"
        and evidence.scope.adapter_version == "superdex-portable-entities-v3"
        for evidence in declaration.evidence
    )
    assert "one-body collision-disabled mirrors" in declaration.reason
    assert "hidden native free-root carrier" in declaration.reason
    assert "mirror contact sensors" in declaration.reason
