"""Factory dispatch coverage for the ray caster plugin boundary."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import unisim
from unisim import FakeRayCaster, RayCaster, RayGeomType, RaySceneDescription
from unisim.errors import BackendError
from unisim.optional import OptionalDependencyError


def test_manifest_covers_declared_ray_casters():
    names = {spec.name for spec in unisim.RAY_CASTER_SPECS}
    assert names == {"uni_ray"}
    assert all(spec.status == "available" for spec in unisim.RAY_CASTER_SPECS)
    assert unisim.ray_caster_spec("uni_ray").package == "uni_ray"
    with pytest.raises(KeyError, match="unknown UniSim ray caster"):
        unisim.ray_caster_spec("nope")


def test_factory_builds_fake_caster():
    caster = unisim.create_ray_caster("fake", num_envs=2, num_rays=3)
    assert isinstance(caster, FakeRayCaster)
    assert caster.num_envs == 2
    assert caster.num_rays == 3


def test_factory_rejects_unknown_caster():
    with pytest.raises(ValueError, match="unknown UniSim ray caster"):
        unisim.create_ray_caster("nope")


def test_factory_validates_batch_shape_before_plugin_import():
    with pytest.raises(TypeError, match="num_envs must be an integer"):
        unisim.create_ray_caster("uni_ray", num_envs=True)
    with pytest.raises(ValueError, match="num_rays must be positive"):
        unisim.create_ray_caster("uni_ray", num_rays=0)


def test_uni_ray_fails_closed_without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "uni_ray", None)
    with pytest.raises(OptionalDependencyError, match="uni_ray"):
        unisim.create_ray_caster("uni_ray", num_envs=1, num_rays=1)


def test_uni_ray_fails_closed_without_entry_point(monkeypatch):
    module = types.ModuleType("uni_ray")
    monkeypatch.setitem(sys.modules, "uni_ray", module)
    with pytest.raises(OptionalDependencyError, match="create_ray_caster"):
        unisim.create_ray_caster("uni_ray")


def test_uni_ray_dispatch_uses_plugin_factory(monkeypatch):
    module = types.ModuleType("uni_ray")
    seen = {}

    def plugin_factory(num_envs, num_rays, **kwargs):
        seen.update(num_envs=num_envs, num_rays=num_rays, **kwargs)
        return FakeRayCaster(num_envs=num_envs, num_rays=num_rays)

    module.create_ray_caster = plugin_factory
    monkeypatch.setitem(sys.modules, "uni_ray", module)
    caster = unisim.create_ray_caster("uni_ray", num_envs=4, num_rays=2, device="cpu")
    assert isinstance(caster, RayCaster)
    assert seen == {"num_envs": 4, "num_rays": 2, "device": "cpu"}


def test_uni_ray_factory_must_return_a_ray_caster(monkeypatch):
    module = types.ModuleType("uni_ray")
    module.create_ray_caster = lambda num_envs, num_rays, **kwargs: object()
    monkeypatch.setitem(sys.modules, "uni_ray", module)
    with pytest.raises(BackendError, match="not a unisim.RayCaster"):
        unisim.create_ray_caster("uni_ray")


def test_plugin_boundary_stays_numpy_only(monkeypatch):
    """A plugin built on blocked engine modules still crosses only NumPy arrays."""
    code = """
import importlib.abc
import sys

blocked = {'mujoco', 'mjbatch', 'warp', 'motrixsim', 'torch'}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoSDKs())
import numpy as np
import unisim

caster = unisim.create_ray_caster('fake', num_envs=2, num_rays=1)
scene = unisim.RaySceneDescription(
    num_bodies=1,
    geom_types=(unisim.RayGeomType.PLANE,),
    geom_sizes=np.zeros((1, 3)),
    geom_local_pos=np.zeros((1, 3)),
    geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
    geom_body_ids=np.zeros(1, dtype=np.intp),
)
caster.materialize(scene)
result = caster.trace(
    np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]), 10.0
)
assert result.hit.all() and result.distance[0, 0] == 1.0
caster.close()
assert not blocked.intersection(sys.modules)
"""
    import subprocess

    outcome = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert outcome.returncode == 0, outcome.stderr or outcome.stdout


def test_ray_caster_scene_descriptor_avoids_engine_types():
    """The scene descriptor round-trips through plain NumPy containers only."""
    scene = RaySceneDescription(
        num_bodies=1,
        geom_types=(RayGeomType.SPHERE,),
        geom_sizes=np.array([[1.0, 0.0, 0.0]]),
        geom_local_pos=np.zeros((1, 3)),
        geom_local_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        geom_body_ids=np.zeros(1, dtype=np.intp),
    )
    annotations = {
        name: field.type for name, field in RaySceneDescription.__dataclass_fields__.items()
    }
    assert set(annotations) == {
        "num_bodies",
        "geom_types",
        "geom_sizes",
        "geom_local_pos",
        "geom_local_quat",
        "geom_body_ids",
    }
    assert scene.num_geoms == 1
