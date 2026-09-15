"""Validation tests for the typed scene-entity contract (SimToolReal step 1)."""

from __future__ import annotations

import pytest

from unisim.scene import ActuatorGainOverride, SceneCfg, SceneEntitySpec


def _spec(**overrides) -> SceneEntitySpec:
    kwargs = {
        "name": "robot",
        "model_file": "robot.urdf",
        "asset_format": "urdf",
        "materialization": "articulation",
        "root_mode": "fixed",
    }
    kwargs.update(overrides)
    return SceneEntitySpec(**kwargs)


def test_valid_entity_spec_and_fixed_base_property():
    assert _spec().fixed_base is True
    assert _spec(root_mode="floating").fixed_base is False
    assert _spec(root_mode="kinematic", materialization="rigid").fixed_base is False


def test_unknown_asset_format_fails_closed():
    with pytest.raises(ValueError, match="asset_format"):
        _spec(asset_format="usd")


def test_unknown_materialization_fails_closed():
    with pytest.raises(ValueError, match="materialization"):
        _spec(materialization="deformable")


def test_unknown_root_mode_fails_closed():
    with pytest.raises(ValueError, match="root_mode"):
        _spec(root_mode="wheeled")


def test_empty_name_and_model_file_fail_closed():
    with pytest.raises(ValueError, match="name"):
        _spec(name="")
    with pytest.raises(ValueError, match="model_file"):
        _spec(model_file="")


def test_duplicate_gain_override_joints_fail_closed():
    overrides = (
        ActuatorGainOverride(joint_name="j1", stiffness=1.0, damping=0.1),
        ActuatorGainOverride(joint_name="j1", stiffness=2.0, damping=0.2),
    )
    with pytest.raises(ValueError, match="duplicate"):
        _spec(actuator_gain_overrides=overrides)


def test_gain_override_values_must_be_finite_non_negative():
    with pytest.raises(ValueError, match="stiffness"):
        ActuatorGainOverride(joint_name="j1", stiffness=-1.0, damping=0.1)
    with pytest.raises(ValueError, match="damping"):
        ActuatorGainOverride(joint_name="j1", stiffness=1.0, damping=float("nan"))
    with pytest.raises(ValueError, match="armature"):
        ActuatorGainOverride(joint_name="j1", stiffness=1.0, damping=0.1, armature=-0.5)
    with pytest.raises(ValueError, match="frictionloss"):
        ActuatorGainOverride(joint_name="j1", stiffness=1.0, damping=0.1, frictionloss=-1.0)
    with pytest.raises(ValueError, match="joint_name"):
        ActuatorGainOverride(joint_name="", stiffness=1.0, damping=0.1)


def test_gain_override_none_armature_keeps_scanned_value():
    override = ActuatorGainOverride(joint_name="j1", stiffness=10.0, damping=1.0)
    assert override.armature is None
    assert override.frictionloss is None


def test_scene_cfg_entity_assets_default_and_declaration():
    assert SceneCfg(model_file="scene.xml").entity_assets == ()
    scene = SceneCfg(model_file="robot.urdf", entity_assets=(_spec(),))
    assert scene.entity_assets[0].name == "robot"
