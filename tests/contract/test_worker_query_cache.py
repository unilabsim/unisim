"""Mapped host queries consume frozen indices and keep detached return values."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.contract.test_worker_scene_host import backend
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout


def test_repeated_queries_do_not_walk_layout_or_alias_shared_memory(tmp_path, monkeypatch):
    owner = backend(tmp_path, monkeypatch)
    try:
        expected_pos = owner.get_dof_pos()
        expected_vel = owner.get_dof_vel()
        expected_state = owner.get_entity_state("object")
        expected_default = owner.get_entity_default_state("object")
        expected_names = owner.get_entity_names()
        primary = owner._primary_entity_index()

        class UnavailableLayout:
            @property
            def entities(self):
                pytest.fail("a query walked the cold entity layout")

            def get_entity(self, name):
                pytest.fail("a query resolved an entity through the layout")

        owner._entity_scene.layout = UnavailableLayout()
        for _ in range(4):
            pos, vel = owner.get_dof_pos(), owner.get_dof_vel()
            np.testing.assert_array_equal(pos, expected_pos)
            np.testing.assert_array_equal(vel, expected_vel)
            assert not np.shares_memory(pos, owner._slots["qpos"])
            assert not np.shares_memory(vel, owner._slots["qvel"])
            pos[:] = 99
            vel[:] = 99
            assert owner._primary_entity_index() == primary
            assert owner.get_entity_names() == expected_names
            for field, values in owner.get_entity_state("object").items():
                np.testing.assert_array_equal(values, expected_state[field])
                values[:] = 99
            for field, values in owner.get_entity_default_state("object").items():
                np.testing.assert_array_equal(values, expected_default[field])
                values[:] = 99
        with pytest.raises(ValueError, match="unknown scene entity"):
            owner.get_entity_state("missing")
    finally:
        owner.close()


@pytest.mark.parametrize("base", [None, "robot", "robot/base"])
def test_primary_and_packed_columns_are_bound_once_with_generalized_widths(base):
    robot = EntityLayout(
        "robot",
        "articulation",
        "fixed",
        "base",
        ("base",),
        (0,),
        (None,),
        (JointLayout("ball", "ball", (0, 1, 2, 3), (0, 1, 2), "base"),),
        ("drive",),
        ("ball",),
        (0,),
    )
    object_entity = EntityLayout(
        "object",
        "rigid",
        "floating",
        "base",
        ("base",),
        (1,),
        (None,),
        (),
        (),
        (),
        (),
        tuple(range(4, 11)),
        tuple(range(3, 9)),
    )
    layout = CompiledSceneLayout((object_entity, robot), 11, 9, 1, 2)
    owner = IsaacGymBackend.__new__(IsaacGymBackend)
    owner._entity_scene = SimpleNamespace(layout=layout)
    owner._base_name = base
    owner._bind_entity_query_maps()
    np.testing.assert_array_equal(owner._entity_dof_qpos_columns, [0, 1, 2, 3])
    np.testing.assert_array_equal(owner._entity_dof_qvel_columns, [0, 1, 2])
    assert owner._primary_entity_index() == 1
    # No physical/runtime claims: this checks the pure compiled metadata boundary.
    owner._entity_scene = None


def test_missing_primary_is_rejected_at_cold_binding(tmp_path, monkeypatch):
    owner = backend(tmp_path, monkeypatch)
    try:
        owner._base_name = "missing"
        with pytest.raises(ValueError, match="does not name an entity root"):
            owner._bind_entity_query_maps()
    finally:
        owner.close()
