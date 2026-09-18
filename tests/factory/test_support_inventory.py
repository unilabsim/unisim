"""The documented inventory is generated from the public declarations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from unisim import ADAPTER_SPECS, SupportLevel, get_adapter_capabilities
from unisim.support import FEATURES

ROOT = Path(__file__).resolve().parents[2]


def test_every_adapter_has_complete_source_scoped_inventory() -> None:
    for adapter in ADAPTER_SPECS:
        report = get_adapter_capabilities(adapter.name)
        assert {item.feature for item in report.declarations} == set(FEATURES)
        assert report.scope.adapter == adapter.name
        for item in report.declarations:
            assert not item.runtime_verified(report.scope)
            if item.support != SupportLevel.UNKNOWN:
                assert item.evidence
                assert all(evidence.kind == "source" for evidence in item.evidence)
        unknown = get_adapter_capabilities(adapter.name, profile="undeclared-profile")
        assert all(item.support == SupportLevel.UNKNOWN for item in unknown.declarations)


def test_isaacsim_entity_multiple_reports_exact_k_prototype_assignments() -> None:
    report = get_adapter_capabilities("isaacsim")
    declaration = report.get(
        "entity.multiple", configuration={"entity.asset_format": "mjcf"}
    )
    assert declaration.support is SupportLevel.EXACT
    assert "immutable construction-time assignments" in declaration.reason
    assert "K-prototype" in declaration.reason
    assert "round-robin" not in declaration.reason


def test_isaacsim_callback_refresh_is_mapped_scene_conditional() -> None:
    report = get_adapter_capabilities("isaacsim")
    declaration = report.get(
        "state.callback_refresh", configuration={"scene.profile": "mapped_entities"}
    )
    assert declaration.support is SupportLevel.EXACT
    assert "one public step into worker substeps" in declaration.reason
    assert report.get("state.callback_refresh").support is SupportLevel.UNKNOWN


def test_bilingual_inventory_matches_public_declarations() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnostics/check_support.py"), "--check-docs"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
