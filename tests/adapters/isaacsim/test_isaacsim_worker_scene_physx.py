"""Tests for the IsaacSim worker's declared scene configuration parsers.

Pure-Python coverage only: the wire-boundary parsers behind the declarative
scene composition contract (scene-level PhysX, environment grid spacing, and
per-entity spawn poses).  The Kit-side application is exercised by the INIT
meta ``scene_physx`` readback consumed by ``probes/probe_b2_full_scene_readback.py``.
"""

from __future__ import annotations

import pytest

from unisim.backend.isaacsim.worker import (
    parse_entity_init_state,
    parse_env_grid_spacing,
    parse_scene_physx_declaration,
)
from unisim.scene import ScenePhysxCfg


def _declaration() -> dict:
    # The values the SimToolReal owner YAML declares (a literal port of the
    # original repository's `_default_sim_cfg` PhysxCfg,
    # simtoolreal_env_cfg.py:515-535 — the friction offset/correlation
    # distances equal Isaac Lab's defaults and stay explicit).
    return ScenePhysxCfg(
        solver_type=1,
        min_position_iteration_count=8,
        max_position_iteration_count=8,
        min_velocity_iteration_count=0,
        max_velocity_iteration_count=0,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
        gpu_max_rigid_contact_count=2**24,
        gpu_max_rigid_patch_count=2**23,
    ).as_kwargs()


def test_wire_parser_accepts_the_owner_declaration():
    parsed = parse_scene_physx_declaration(_declaration())
    assert parsed == {
        "solver_type": 1,  # 1 = TGS (matches legacy)
        "min_position_iteration_count": 8,
        "max_position_iteration_count": 8,
        "min_velocity_iteration_count": 0,
        "max_velocity_iteration_count": 0,
        "bounce_threshold_velocity": 0.2,
        "friction_offset_threshold": 0.04,
        "friction_correlation_distance": 0.025,
        "gpu_max_rigid_contact_count": 2**24,  # 16777216
        "gpu_max_rigid_patch_count": 2**23,  # 8388608
    }


def test_wire_parser_none_keeps_worker_defaults():
    # Undeclared: the worker keeps Isaac Lab's own PhysX defaults and the
    # GridCloner's native 2.0 m layout — there is no undeclared fallback to
    # task-specific tuning.
    assert parse_scene_physx_declaration(None) is None
    assert parse_env_grid_spacing(None) is None


def test_wire_parser_rejects_malformed_physx():
    with pytest.raises(TypeError, match="dict"):
        parse_scene_physx_declaration([1, 2])
    bad_keys = dict(_declaration())
    bad_keys.pop("solver_type")
    with pytest.raises(ValueError, match="keys must be exactly"):
        parse_scene_physx_declaration(bad_keys)
    bad_solver = dict(_declaration(), solver_type=2)
    with pytest.raises(ValueError, match="PGS"):
        parse_scene_physx_declaration(bad_solver)
    bad_int = dict(_declaration(), min_position_iteration_count=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_scene_physx_declaration(bad_int)
    bad_float = dict(_declaration(), bounce_threshold_velocity=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        parse_scene_physx_declaration(bad_float)


def test_wire_parser_rejects_malformed_spacing():
    with pytest.raises(TypeError, match="number"):
        parse_env_grid_spacing("1.2")
    with pytest.raises(ValueError, match="finite positive"):
        parse_env_grid_spacing(0.0)
    with pytest.raises(ValueError, match="finite positive"):
        parse_env_grid_spacing(float("inf"))
    assert parse_env_grid_spacing(1.2) == 1.2
    assert parse_env_grid_spacing(2) == 2.0


def test_wire_parser_accepts_declared_spawn_pose():
    pose = parse_entity_init_state({"pos": [0.0, 0.8, 0.0], "rot_wxyz": [1.0, 0, 0, 0]})
    assert pose == ((0.0, 0.8, 0.0), (1.0, 0.0, 0.0, 0.0))
    # The original repository's robot spawn (scene_utils.py:1811-1842): the
    # iiwa stands 0.8 m behind the table with identity orientation.
    assert parse_entity_init_state(None) is None


def test_wire_parser_rejects_malformed_spawn_pose():
    with pytest.raises(ValueError, match="pos/rot_wxyz"):
        parse_entity_init_state({"pos": [0.0, 0.8, 0.0]})
    with pytest.raises(ValueError, match="xyz triple"):
        parse_entity_init_state({"pos": [0.0, 0.8], "rot_wxyz": [1, 0, 0, 0]})
    with pytest.raises(ValueError, match="wxyz quaternion"):
        parse_entity_init_state({"pos": [0, 0, 0], "rot_wxyz": [1, 0, 0]})
    with pytest.raises(ValueError, match="non-zero"):
        parse_entity_init_state({"pos": [0, 0, 0], "rot_wxyz": [0, 0, 0, 0]})
    with pytest.raises(ValueError, match="finite"):
        parse_entity_init_state({"pos": [0, 0, float("nan")], "rot_wxyz": [1, 0, 0, 0]})


def test_scene_physx_cfg_defaults_mirror_isaac_lab():
    # Field defaults mirror IsaacLab's PhysxCfg so a partial declaration
    # overrides exactly the authored fields; the class-level validation is
    # the host-side twin of the worker wire parser.
    defaults = ScenePhysxCfg()
    assert defaults.solver_type == 1
    assert defaults.min_position_iteration_count == 1
    assert defaults.max_position_iteration_count == 255
    assert defaults.min_velocity_iteration_count == 0
    assert defaults.max_velocity_iteration_count == 255
    assert defaults.bounce_threshold_velocity == 0.5
    assert defaults.friction_offset_threshold == 0.04
    assert defaults.friction_correlation_distance == 0.025
    assert defaults.gpu_max_rigid_contact_count == 2**23
    assert defaults.gpu_max_rigid_patch_count == 5 * 2**15
    with pytest.raises(ValueError, match="solver_type"):
        ScenePhysxCfg(solver_type=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        ScenePhysxCfg(max_velocity_iteration_count=-1)
    with pytest.raises(ValueError, match="positive"):
        ScenePhysxCfg(gpu_max_rigid_patch_count=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        ScenePhysxCfg(bounce_threshold_velocity=-0.1)
