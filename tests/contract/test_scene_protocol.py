"""Mapped-scene wire widths and descriptor validation without engine SDKs."""

from __future__ import annotations

import copy
from multiprocessing import shared_memory

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def _layout() -> CompiledSceneLayout:
    return CompiledSceneLayout(
        entities=(
            EntityLayout(
                name="robot", kind="articulation", root_mode="fixed", root_body="base",
                body_names=("base", "tip"), body_ids=(1, 2), body_parent_names=(None, "base"),
                joints=(JointLayout("hinge", "hinge", (0,), (0,), "tip"),),
                actuator_names=("motor",), actuator_joint_names=("hinge",), actuator_indices=(0,),
            ),
            EntityLayout(
                name="object", kind="rigid", root_mode="floating", root_body="base",
                body_names=("base",), body_ids=(3,), body_parent_names=(None,), joints=(),
                actuator_names=(), actuator_joint_names=(), actuator_indices=(),
                root_qpos_indices=tuple(range(1, 8)), root_qvel_indices=tuple(range(1, 7)),
            ),
        ), nq=8, nv=7, nu=1, nbody=4,
    )


def _specs():
    return {
        name: {"shm": "test_" + name, "shape": list(shape), "dtype": str(protocol.slot_dtype(name))}
        for name, shape in protocol.scene_slot_shapes(5, _layout()).items()
    }


def test_scene_slots_separate_state_actuator_and_entity_dimensions() -> None:
    layout = protocol.load_scene_layout(_layout().to_dict())
    shapes = protocol.scene_slot_shapes(5, layout)
    assert shapes["ctrl"] == (5, 1)
    assert shapes["qpos"] == (5, 8)
    assert shapes["qvel"] == (5, 7)
    assert shapes["entity_root_state"] == (5, 2, 13)
    assert shapes["reset_root_mask"] == (2, 2)
    assert shapes["reset_qpos_mask"] == (8,)
    protocol.validate_slot_specs(_specs(), shapes)


def test_collision_pair_force_slot_is_negotiated_only_when_declared() -> None:
    layout = protocol.load_scene_layout(_layout().to_dict())
    base = protocol.scene_slot_shapes(5, layout)
    assert "contact_sensor_force" not in base

    shapes = protocol.scene_slot_shapes(5, layout, 2)
    assert shapes["contact_sensor_force"] == (5, 2, 3)
    specs = {
        name: {"shm": "pair_" + name, "shape": list(shape), "dtype": str(protocol.slot_dtype(name))}
        for name, shape in shapes.items()
    }
    protocol.validate_slot_specs(specs, shapes)
    with pytest.raises(ValueError, match="num_contact_force_sensors"):
        protocol.scene_slot_shapes(5, layout, True)


@pytest.mark.parametrize("corruption", ["missing", "extra", "dtype", "shape", "boolean", "shm"])
def test_slot_descriptors_reject_tampering_before_any_attachment(corruption) -> None:
    specs = _specs()
    if corruption == "missing":
        del specs["qvel"]
    elif corruption == "extra":
        specs["unused"] = copy.deepcopy(specs["ctrl"])
    elif corruption == "dtype":
        specs["ctrl"]["dtype"] = "float64"
    elif corruption == "shape":
        specs["ctrl"]["shape"] = [5, 7]
    elif corruption == "boolean":
        specs["ctrl"]["shape"] = [5, True]
    else:
        specs["ctrl"]["shm"] = ""
    with pytest.raises(ValueError):
        protocol.validate_slot_specs(specs, protocol.scene_slot_shapes(5, _layout()))


def test_zero_actuator_shared_memory_has_storage_but_zero_public_elements() -> None:
    shape = (2, 0)
    assert protocol.slot_nbytes("ctrl", shape) == 0
    memory = shared_memory.SharedMemory(
        create=True, size=protocol.slot_allocation_nbytes("ctrl", shape)
    )
    try:
        array = np.ndarray(shape, dtype=protocol.slot_dtype("ctrl"), buffer=memory.buf)
        assert array.shape == (2, 0)
        assert array.size == 0
    finally:
        memory.close()
        memory.unlink()


def test_legacy_slots_keep_the_existing_wire_shape_contract() -> None:
    shapes = protocol.slot_shapes(2, 3, 4)
    specs = {name: {"shm": name, "shape": list(shape), "dtype": str(protocol.slot_dtype(name))}
             for name, shape in shapes.items()}
    protocol.validate_slot_specs(specs, shapes)
    assert tuple(shapes) == protocol.SLOT_NAMES
    assert shapes["reset_qpos"] == (2, 10)
