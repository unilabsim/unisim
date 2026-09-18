"""SuperDex semantic support declarations."""

from __future__ import annotations

from unisim.capabilities import SupportLevel, get_adapter_capabilities


def test_superdex_multiple_entities_are_explicitly_unsupported() -> None:
    declaration = get_adapter_capabilities("superdex").get(
        "entity.multiple", configuration={"entity.asset_format": "mjcf"}
    )
    assert declaration.support is SupportLevel.UNSUPPORTED
    assert "SceneBatchExecutor" in declaration.reason
    assert any(
        evidence.source == "https://github.com/unilabsim/unisim/issues/124"
        and evidence.scope.adapter_version == "superdex-single-actor-executor-v1"
        for evidence in declaration.evidence
    )
