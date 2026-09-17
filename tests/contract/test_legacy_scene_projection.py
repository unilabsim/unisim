"""Historical buffers are projections, not a second native execution path."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unisim.backend.subprocess_ipc import protocol
from unisim.backend.subprocess_ipc.legacy_projection import (
    LegacyExecutionLayout,
    LegacySlotProjection,
)


def projection(*, com=False, ndof=2):
    layout = LegacyExecutionLayout(tuple(f"j{i}" for i in range(ndof)), ("base", "tip"))
    p = LegacySlotProjection(
        protocol,
        3,
        layout,
        root_com=np.tile([0.1, 0, 0], (3, 1)) if com else None,
        body_com=np.tile([0.1, 0, 0], (3, 2, 1)) if com else None,
    )
    old = {
        name: np.zeros(shape, dtype=protocol.slot_dtype(name))
        for name, shape in protocol.slot_shapes(3, ndof, 2).items()
    }
    p.attach(old)
    return p, old


def test_legacy_synthetic_coordinates_keep_names_and_passive_control_width():
    layout = LegacyExecutionLayout(("old_joint", "passive"), ("base:old", "unnested/path"))
    assert (layout.nq, layout.nv, layout.nu) == (9, 8, 2)
    assert layout.entities[0].body_names == ("base:old", "unnested/path")
    assert layout.entities[0].actuator_names == ("old_joint", "passive")
    assert not hasattr(layout, "to_dict")  # Never masquerade as a public physical declaration.
    with pytest.raises(ValueError):
        layout.get_entity("robot")


def test_projection_round_trip_has_explicit_world_and_body_angular_velocity():
    p, old = projection(com=True)
    q = np.sqrt(0.5)
    old["reset_env_ids"][:2] = [2, 0]
    old["reset_qpos"][:2, :7] = [1, 2, 3, q, 0, 0, q]
    old["reset_qpos"][:2, 7:] = [[0.3, 0.4], [0.5, 0.6]]
    old["reset_qvel"][:2, :6] = [0, 0, 0, 0, 0, 2]
    old["ctrl"][:] = [[1, 2], [3, 4], [5, 6]]
    payload = p.prepare_reset(2)
    assert payload["control_values"] == [[5, 6], [1, 2]]
    # Rz90 COM=(0,.1,0): world omega cross COM=(-.2,0,0).
    # Legacy native COM velocity zero corresponds to +.2 link-origin x velocity.
    np.testing.assert_allclose(
        p.slots["reset_entity_root_state"][:2, 0, 7:10], [[0.2, 0, 0]] * 2, atol=1e-7
    )
    np.testing.assert_allclose(p.slots["reset_qvel"][:2, :3], [[0.2, 0, 0]] * 2, atol=1e-7)
    p.slots["entity_root_state"][[2, 0]] = p.slots["reset_entity_root_state"][:2]
    p.slots["qpos"][[2, 0]] = p.slots["reset_qpos"][:2]
    p.slots["qvel"][[2, 0]] = p.slots["reset_qvel"][:2]
    p.slots["body_state"][..., 3] = 1
    p.publish()
    np.testing.assert_allclose(old["root_state"][[2, 0], 7:10], 0, atol=1e-7)
    np.testing.assert_allclose(old["root_state"][[2, 0], 10:13], [[0, 0, 2]] * 2)
    np.testing.assert_allclose(old["dof_state"][[2, 0], :, 0], [[0.3, 0.4], [0.5, 0.6]])
    assert p.slots["ctrl"] is old["ctrl"]


@pytest.mark.parametrize("bad", ["count", "rows", "nan", "quat"])
def test_invalid_legacy_reset_does_not_write_canonical_buffers(bad):
    p, old = projection()
    old["reset_env_ids"][:2] = [2, 0]
    old["reset_qpos"][:2, 3] = 1
    count = 2
    if bad == "count":
        count = True
    elif bad == "rows":
        old["reset_env_ids"][1] = 2
    elif bad == "nan":
        old["reset_qvel"][0, 0] = np.nan
    else:
        old["reset_qpos"][0, 3] = 0
    before = {key: value.copy() for key, value in p.slots.items()}
    with pytest.raises(ValueError):
        p.prepare_reset(count)
    for key, value in before.items():
        np.testing.assert_array_equal(p.slots[key], value)


def test_zero_dof_projection_is_well_shaped():
    p, old = projection(ndof=0)
    assert p.slots["ctrl"].shape == (3, 0)
    p.slots["body_state"][..., 3] = 1
    p.publish()
    assert old["dof_state"].shape == (3, 0, 2)


def test_worker_projection_is_py38_syntax_and_loads_without_host_import():
    path = Path(protocol.__file__).with_name("legacy_projection.py")
    ast.parse(path.read_text(), feature_version=(3, 8))
    code = """import importlib.util,sys
spec=importlib.util.spec_from_file_location('wire',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.load_legacy_projection();l=p.LegacyExecutionLayout(['hinge'],['base','tip'])
assert l.nq==8 and l.nu==1
assert 'unisim' not in sys.modules
assert not {'torch','mujoco','warp'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(Path(protocol.__file__))], check=True)
