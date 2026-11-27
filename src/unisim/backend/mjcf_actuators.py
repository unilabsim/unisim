"""Stdlib-only MJCF actuator rewrites shared by engine adapters.

MotrixSim's MJCF importer (as of 1.0.0) ignores the ``input`` attribute of
``<general>`` actuators and has no ``jointactuatorfrc`` sensor support, while
MuJoCo's spec writer serializes position actuators only in the general form.
The helpers here restore the canonical forms on a cold path so the imported
model keeps the kp/kd override surface and the actuator-force sensor namespace
without ever importing an engine SDK or MuJoCo into this module.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# Attributes carried by every MuJoCo actuator element that targets a joint.
_JOINT_ACTUATOR_TAGS = ("motor", "position", "velocity", "general", "cylinder", "intvelocity")


def restore_position_actuator_shorthand(xml: str) -> str:
    """Rewrite ``<general input="pos">`` rows back to the ``<position>`` form.

    MuJoCo's spec writer serializes position actuators only as the equivalent
    general form ``<general input="pos" biastype="affine" gainprm="kp"
    biasprm="0 -kp -kv"/>`` (plain specs and ``mjcf`` attachments alike).
    MuJoCo compiles both forms identically, but MotrixSim's importer ignores
    the ``input`` attribute and yields a ``GeneralActuator`` without the
    per-actuator kp/kd override surface.  Sources therefore keep the
    shorthand wherever the canonical form round-trips exactly; anything else
    (dynamics, non-affine bias, non-matching gains) stays in the general form
    so no authored semantics are ever rewritten.
    """
    root = ET.fromstring(xml)
    changed = False
    for actuator_block in root.iter("actuator"):
        for element in list(actuator_block):
            if element.tag != "general" or element.get("input") != "pos":
                continue
            if element.get("gaintype") not in (None, "fixed"):
                continue
            gain = (element.get("gainprm") or "").split()
            bias = (element.get("biasprm") or "").split()
            if len(gain) != 1 or len(bias) != 3:
                continue
            try:
                kp = float(gain[0])
                bias_constant, bias_position, bias_velocity = (float(value) for value in bias)
            except ValueError:
                continue
            if (
                element.get("biastype") != "affine"
                or bias_constant != 0.0
                or bias_position != -kp
            ):
                continue
            attributes = {
                key: value
                for key, value in element.attrib.items()
                if key not in ("input", "gaintype", "biastype", "gainprm", "biasprm")
            }
            attributes["kp"] = format(kp, "g")
            attributes["kv"] = format(-bias_velocity, "g")
            replacement = ET.Element("position", attributes)
            index = list(actuator_block).index(element)
            actuator_block.remove(element)
            actuator_block.insert(index, replacement)
            changed = True
    if not changed:
        return xml
    return ET.tostring(root, encoding="unicode")


def restore_joint_actuator_force_sensors(xml: str) -> str:
    """Rewrite ``<jointactuatorfrc joint="j">`` to ``<actuatorfrc actuator="a">``.

    MotrixSim's importer drops ``jointactuatorfrc`` sensors (upstream accepts
    only the per-actuator form), so the per-joint form is restored when the
    joint is driven by exactly one joint-targeting actuator and the sensor
    names no actuator explicitly.  Multiply-actuated joints keep the authored
    form; the adapter's authored-sensor audit fails closed on the dropped
    sensor instead of silently losing it.
    """
    root = ET.fromstring(xml)
    joint_actuators: dict[str, list[str]] = {}
    for actuator_block in root.iter("actuator"):
        for element in actuator_block:
            if element.tag not in _JOINT_ACTUATOR_TAGS:
                continue
            joint = element.get("joint")
            name = element.get("name")
            if joint and name:
                joint_actuators.setdefault(joint, []).append(name)
    changed = False
    for sensor_block in root.iter("sensor"):
        for element in list(sensor_block):
            if element.tag != "jointactuatorfrc" or element.get("actuator"):
                continue
            joint = element.get("joint")
            actuators = joint_actuators.get(joint or "", [])
            if joint is None or len(actuators) != 1:
                continue
            attributes = {key: value for key, value in element.attrib.items() if key != "joint"}
            attributes["actuator"] = actuators[0]
            replacement = ET.Element("actuatorfrc", attributes)
            index = list(sensor_block).index(element)
            sensor_block.remove(element)
            sensor_block.insert(index, replacement)
            changed = True
    if not changed:
        return xml
    return ET.tostring(root, encoding="unicode")


__all__ = [
    "restore_joint_actuator_force_sensors",
    "restore_position_actuator_shorthand",
]
