"""The documented inventory is generated from the public declarations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
    declaration = report.get("entity.multiple", configuration={"entity.asset_format": "mjcf"})
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


def test_drake_entity_multiple_is_bounded_to_no_variant_mjcf_without_mirrors() -> None:
    report = get_adapter_capabilities("drake")
    supported = {
        "entity.asset_format": "mjcf",
        "entity.variant": "none",
        "entity.kinematic": "none",
    }
    declaration = report.get("entity.multiple", configuration=supported)
    assert declaration.support is SupportLevel.EXACT
    assert "fixed/floating physical entities and passive joints" in declaration.reason
    assert "fixed variants and kinematic mirrors fail closed" in declaration.reason
    assert declaration.evidence
    assert declaration.evidence[0].source.endswith("/issues/122")
    assert declaration.evidence[0].scope.adapter_version == "drake-portable-entities-v1"


def test_motrix_entity_multiple_supports_no_variant_and_same_layout_variants() -> None:
    supported = {
        "entity.asset_format": "mjcf",
        "entity.kinematic": "none",
    }
    declaration = get_adapter_capabilities("motrix").get("entity.multiple", configuration=supported)
    assert declaration.support is SupportLevel.EXACT
    assert "fixed/floating physical entities, passive scalar joints" in declaration.reason
    assert "immutable same-layout fixed variants" in declaration.reason
    assert "selected keyframe qpos/qvel" in declaration.reason
    assert "actuator activation state" in declaration.reason
    assert "Generated body-frame position/quaternion tracking sensors" in declaration.reason
    assert "world-referenced authored body FramePos/FrameQuat sensors" in declaration.reason
    assert (
        "Scene-level fragment world-referenced qualified-body "
        "FramePos/FrameQuat/FrameLinVel/FrameAngVel sensors" in declaration.reason
    )
    assert "FrameLinVel reporting world velocity at the inertial body-frame origin" in (
        declaration.reason
    )
    assert "FrameAngVel reporting world angular velocity" in declaration.reason
    assert "entity-owned and scene-level fragment world-referenced site pose sensors" in (
        declaration.reason
    )
    assert (
        "Scene-level fragment world-referenced qualified-site "
        "FrameLinVel/FrameAngVel sensors" in declaration.reason
    )
    assert "complete parent/local-pose site identity" in declaration.reason
    assert "site FrameLinVel reporting world-frame site-point velocity" in declaration.reason
    assert "site FrameAngVel reporting world angular velocity" in declaration.reason
    assert "geom-pair netforce and found contact fragments" in declaration.reason
    assert "Portable selected-row reset randomization supports body_mass" in declaration.reason
    assert "body_ipos/base_com_offset" in declaration.reason
    assert "scalar-joint dof_armature/dof_frictionloss" in declaration.reason
    assert "free-root DOF columns remain defaults" in declaration.reason
    assert "no public runtime joint-damping override" in declaration.reason
    assert "entity-owned frame motion" in declaration.reason
    assert "other source sensors, other site-sensor forms" in declaration.reason
    assert "other reset randomization" in declaration.reason
    assert declaration.evidence
    assert declaration.evidence[0].source.endswith("/issues/121")
    assert declaration.evidence[0].scope.adapter_version == "motrix-portable-entities-v1"


def test_genesis_entity_multiple_supports_bounded_site_accelerometers() -> None:
    supported = {
        "entity.asset_format": "mjcf",
        "entity.kinematic": "none",
    }
    declaration = get_adapter_capabilities("genesis").get(
        "entity.multiple", configuration=supported
    )
    assert declaration.support is SupportLevel.EXACT
    assert "Portable MJCF entity scenes use independent Genesis entities" in declaration.reason
    assert "selected scalar hinge/slide default-keyframe qpos/qvel" in declaration.reason
    assert "actuator controls through public per-entity APIs" in declaration.reason
    assert "raw keyframe root pose/velocity are ignored" in declaration.reason
    assert "declared portable root placement with zero root velocity is retained" in (
        declaration.reason
    )
    assert "restore assignment-aware selected rows from the default control table" in (
        declaration.reason
    )
    assert "no persistent public control-target getter is claimed" in declaration.reason
    assert (
        "Entity-owned unreferenced site FramePos/FrameQuat/Gyro/Velocimeter/"
        "Accelerometer sensors" in declaration.reason
    )
    assert (
        "scene-level fragment world-referenced qualified-site "
        "FramePos/FrameQuat/FrameLinVel/FrameAngVel sensors" in declaration.reason
    )
    assert "site FrameLinVel is world-frame site-point velocity" in declaration.reason
    assert "site FrameAngVel is world angular velocity" in declaration.reason
    assert (
        "Scene-level fragment world-referenced qualified-body "
        "FramePos/FrameQuat/FrameLinVel/FrameAngVel sensors" in declaration.reason
    )
    assert "compose audited public native link-origin pose/velocity" in declaration.reason
    assert "world-angular cross product with the source body_ipos offset" in declaration.reason
    assert "Scene-level cross-entity geom-pair found and netforce fragments" in declaration.reason
    assert "three-vector forces on authored geom1" in declaration.reason
    assert "Portable selected-row reset randomization supports body_mass" in declaration.reason
    assert "body_ipos, base_com_offset, DOF damping/friction loss/armature" in declaration.reason
    assert "and actuator kp/kd" in declaration.reason
    assert "identical complete sensor identity across fixed variants" in declaration.reason
    assert "site quaternions remain public wxyz" in declaration.reason
    assert "source contact sensors, same-entity pairs, other contact forms" in declaration.reason
    assert "source body sensors, inertial orientation mismatches" in declaration.reason
    assert "other body fragment forms" in declaration.reason
    assert "referenced forms, other site fragment forms" in declaration.reason
    assert "other reset randomization" in declaration.reason
    assert "arbitrary keyframe semantics" in declaration.reason
    assert declaration.evidence
    assert declaration.evidence[0].source.endswith("/issues/120")
    assert declaration.evidence[0].scope.adapter_version == "genesis-portable-entities-v1"


@pytest.mark.parametrize(
    "adapter,configuration",
    [
        ("drake", {}),
        ("drake", {"entity.asset_format": "mjcf"}),
        (
            "drake",
            {"entity.asset_format": "urdf", "entity.variant": "none", "entity.kinematic": "none"},
        ),
        (
            "drake",
            {"entity.asset_format": "mjcf", "entity.variant": "fixed", "entity.kinematic": "none"},
        ),
        (
            "drake",
            {
                "entity.asset_format": "mjcf",
                "entity.variant": "none",
                "entity.kinematic": "present",
            },
        ),
        ("motrix", {}),
        ("motrix", {"entity.asset_format": "mjcf"}),
        (
            "motrix",
            {"entity.asset_format": "urdf", "entity.variant": "none", "entity.kinematic": "none"},
        ),
        (
            "motrix",
            {
                "entity.asset_format": "mjcf",
                "entity.variant": "none",
                "entity.kinematic": "present",
            },
        ),
    ],
)
def test_bounded_entity_multiple_fails_closed_outside_reviewed_profile(
    adapter: str, configuration: dict[str, str]
) -> None:
    declaration = get_adapter_capabilities(adapter).get(
        "entity.multiple", configuration=configuration
    )
    assert declaration.support is SupportLevel.UNKNOWN
    assert "Configuration conditions are not satisfied" in declaration.reason


def test_bilingual_inventory_matches_public_declarations() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnostics/check_support.py"), "--check-docs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
