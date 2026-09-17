"""Cleanup scope is bound once from topology and unioned without metadata reads."""

from dataclasses import replace

import numpy as np
import pytest

from tests.contract.test_entity_state import _scene
from unisim.backend.reset_impact import ResetImpact, bind_reset_impacts
from unisim.entities import EntityStatePatch, SceneResetRequest


def test_cold_impacts_cover_root_passive_joint_and_mirror_without_reloading_metadata():
    layout = _scene()
    address, count = np.array([2]), np.array([3])
    index = bind_reset_impacts(layout, address, count)
    assert index.roots["robot"] == ResetImpact((1, 2), (0, 2, 4, 6, 8, 10, 12), (0,), (2, 3, 4))
    assert index.joints[("robot", "hinge")] == ResetImpact((2,), (12,), (0,), (2, 3, 4))
    assert index.joints[("object", "ball")] == ResetImpact((4,), (14, 15, 16), (), ())
    address[:] = 999
    count[:] = 999
    request = SceneResetRequest((4, 1), (
        EntityStatePatch("robot", joint_positions=np.ones((2, 1))),
        EntityStatePatch("object", joint_names=("ball",), joint_velocities=np.ones((2, 3))),
    ))
    assert index.select(layout.validate_reset(request, num_envs=5)) == ResetImpact(
        (2, 4), (12, 14, 15, 16), (0,), (2, 3, 4))
    pose = np.array([[1., 2, 3, 1, 0, 0, 0]])
    root = SceneResetRequest((0,), (EntityStatePatch("robot", root_pose=pose),))
    assert index.select(layout.validate_reset(root, num_envs=5)) == index.roots["robot"]
    mirror = SceneResetRequest((0,), (EntityStatePatch("target", root_pose=pose),))
    assert index.select(layout.validate_reset(mirror, num_envs=5)) == ResetImpact((5,), (), (), ())
    with pytest.raises(TypeError):
        index.roots["robot"] = ResetImpact()


def test_descendant_wrenches_include_fixed_children_but_not_other_branches():
    layout = _scene()
    robot = replace(layout.entities[0], body_names=("base", "finger", "fixed", "sibling"),
                    body_ids=(1, 2, 6, 7), body_parent_names=(None, "base", "finger", "base"))
    layout = replace(layout, entities=(robot, *layout.entities[1:]), nbody=8)
    index = bind_reset_impacts(layout, np.array([-1]), np.array([0]))
    assert index.joints[("robot", "hinge")].bodies == (2, 6)
    assert index.roots["robot"].bodies == (1, 2, 6, 7)
    assert index.joints[("robot", "hinge")].activations == ()
