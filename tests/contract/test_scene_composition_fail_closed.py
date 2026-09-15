"""Scene composition declarations fail closed on non-consuming backends.

``SceneCfg.entity_assets`` and ``SceneCfg.ground_plane`` are composition
declarations, not hints: a backend that cannot materialize them must reject
the scene at construction (fail closed) instead of silently dropping content
and degrading to single-asset behavior.  These tests pin that contract for
the subprocess family default and representative adapters; the IsaacSim
specialization opts in through the ``_supports_entity_assets`` /
``_supports_ground_plane`` hooks.
"""

from __future__ import annotations

import pytest

from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.scene import (
    GroundPlaneSceneCfg,
    SceneCfg,
    SceneEntitySpec,
    validate_scene_composition_support,
)

_ENTITY = SceneEntitySpec(
    name="object",
    model_file="object.urdf",
    asset_format="urdf",
    materialization="rigid",
    root_mode="floating",
)


def test_helper_passes_undeclared_scenes_through():
    # Nothing declared: the helper must not constrain any backend.
    validate_scene_composition_support(SceneCfg(model_file="scene.xml"), "any")


def test_helper_rejects_declared_content_without_support():
    scene = SceneCfg(model_file="scene.xml", entity_assets=(_ENTITY,))
    with pytest.raises(NotImplementedError, match="entity_assets"):
        validate_scene_composition_support(scene, "example")
    scene = SceneCfg(model_file="scene.xml", ground_plane=GroundPlaneSceneCfg())
    with pytest.raises(NotImplementedError, match="ground plane"):
        validate_scene_composition_support(scene, "example")


def test_subprocess_family_default_rejects_entity_assets(tmp_path):
    # IsaacGym keeps the family defaults (no entity-asset materialization),
    # so a declared multi-asset scene fails at construction, before any
    # worker runtime is resolved.
    scene = SceneCfg(
        model_file=str(tmp_path / "scene.xml"), entity_assets=(_ENTITY,)
    )
    with pytest.raises(NotImplementedError, match="isaacgym.*entity_assets"):
        IsaacGymBackend(scene, num_envs=2, sim_dt=0.01)


def test_subprocess_family_default_rejects_ground_plane(tmp_path):
    scene = SceneCfg(
        model_file=str(tmp_path / "scene.xml"), ground_plane=GroundPlaneSceneCfg()
    )
    with pytest.raises(NotImplementedError, match="isaacgym.*ground plane"):
        IsaacGymBackend(scene, num_envs=2, sim_dt=0.01)


def test_isaacsim_accepts_the_declarations(tmp_path):
    # The specialization opts in; construction must not raise even before
    # materialize() runs (the scene files do not exist yet, proving the gate
    # is a capability check, not an asset scan).
    scene = SceneCfg(
        model_file=str(tmp_path / "robot.urdf"),
        entity_assets=(_ENTITY,),
        ground_plane=GroundPlaneSceneCfg(),
    )
    backend = IsaacSimBackend(scene, num_envs=2, sim_dt=0.01)
    try:
        assert backend._supports_entity_assets() is True
        assert backend._supports_ground_plane() is True
    finally:
        backend.close()


def test_mujoco_backend_rejects_both_declarations(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    assert mujoco is not None
    from unisim.backend.mujoco.backend import MuJoCoBackend

    # A deliberately missing model file proves the composition gate fires
    # before the scene context builds.
    scene = SceneCfg(
        model_file=str(tmp_path / "absent.xml"),
        entity_assets=(_ENTITY,),
        ground_plane=GroundPlaneSceneCfg(),
    )
    with pytest.raises(NotImplementedError, match="mujoco.*entity_assets"):
        MuJoCoBackend(scene, num_envs=2, sim_dt=0.01)
    scene = SceneCfg(
        model_file=str(tmp_path / "absent.xml"), ground_plane=GroundPlaneSceneCfg()
    )
    with pytest.raises(NotImplementedError, match="mujoco.*ground plane"):
        MuJoCoBackend(scene, num_envs=2, sim_dt=0.01)
