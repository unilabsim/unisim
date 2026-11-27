"""Unit coverage for the shared MJCF actuator rewrites (stdlib-only)."""

from unisim.backend.mjcf_actuators import (
    restore_joint_actuator_force_sensors,
    restore_position_actuator_shorthand,
)


def test_position_shorthand_restored_when_it_round_trips() -> None:
    xml = """
<mujoco><worldbody><body><joint name='j1'/><geom type='sphere' size='0.1'/></body></worldbody>
<actuator>
  <general name='g1' joint='j1' input='pos' biastype='affine' ctrlrange='-1 1'
           gainprm='12.5' biasprm='0 -12.5 -0.5'/>
</actuator></mujoco>
"""
    restored = restore_position_actuator_shorthand(xml)
    assert "<position" in restored
    assert 'kp="12.5"' in restored
    assert 'kv="0.5"' in restored
    assert "<general" not in restored


def test_position_shorthand_keeps_non_round_tripping_forms() -> None:
    # A non-matching bias position is authored semantics the rewrite must
    # never touch; the velocity-form general stays untouched as well.
    xml = """
<mujoco><worldbody><body><joint name='j1'/><geom type='sphere' size='0.1'/></body></worldbody>
<actuator>
  <general name='g1' joint='j1' input='pos' biastype='affine'
           gainprm='12.5' biasprm='0 -9 -0.5'/>
  <general name='g2' joint='j1' input='vel' gainprm='12.5' biasprm='0 0 -0.5'/>
</actuator></mujoco>
"""
    restored = restore_position_actuator_shorthand(xml)
    assert restored.count("<general") == 2


def test_joint_actuator_force_sensor_restored_for_single_actuator_joint() -> None:
    xml = """
<mujoco><worldbody><body><joint name='j1'/><geom type='sphere' size='0.1'/></body></worldbody>
<actuator><motor name='m1' joint='j1'/></actuator>
<sensor><jointactuatorfrc name='f1' joint='j1'/></sensor></mujoco>
"""
    restored = restore_joint_actuator_force_sensors(xml)
    assert '<actuatorfrc name="f1" actuator="m1" />' in restored
    assert "jointactuatorfrc" not in restored


def test_joint_actuator_force_sensor_kept_for_multi_actuator_joint() -> None:
    xml = """
<mujoco><worldbody><body><joint name='j1'/><geom type='sphere' size='0.1'/></body></worldbody>
<actuator><motor name='m1' joint='j1'/><motor name='m2' joint='j1'/></actuator>
<sensor><jointactuatorfrc name='f1' joint='j1'/></sensor></mujoco>
"""
    restored = restore_joint_actuator_force_sensors(xml)
    assert "jointactuatorfrc" in restored
    assert "<actuatorfrc" not in restored


def test_unrelated_xml_is_unchanged() -> None:
    xml = "<mujoco><worldbody><body><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
    assert restore_position_actuator_shorthand(xml) == xml
    assert restore_joint_actuator_force_sensors(xml) == xml
