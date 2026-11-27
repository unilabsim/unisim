import subprocess
import sys

import pytest

import unisim


def test_import_does_not_pull_unilab_or_engine_modules():
    code = (
        "import sys, unisim; "
        "blocked = [name for name in sys.modules if name == 'unilab' or "
        "name.startswith(('hydra', 'torch', 'gymnasium', 'mujoco', 'mjbatch', 'motrixsim', "
        "'newton', 'warp', 'superdex'))]; "
        "assert not blocked, blocked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_utils_math_stays_numpy_only_when_torch_is_unavailable():
    code = """
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('torch blocked for this test')
        return None

sys.meta_path.insert(0, NoTorch())
import unisim.utils
import unisim.utils.math as m
assert 'torch' not in sys.modules
import numpy as np
q = m.np_yaw_to_quat(np.array([0.25]))
assert m.np_quat_mul(q, m.np_quat_conjugate(q)).shape == (1, 4)
assert 'torch' not in sys.modules
try:
    m.quat_mul
except ImportError:
    pass
else:
    raise AssertionError('expected ImportError for a Torch primitive without Torch')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_public_exports_are_resolvable_and_wildcard_import_is_safe():
    namespace: dict[str, object] = {}
    exec("from unisim import *", {}, namespace)

    assert set(unisim.__all__).issubset(namespace)
    assert namespace["MjcfSubprocessBackend"] is unisim.MjcfSubprocessBackend


def test_capability_inventory_does_not_import_or_discover_optional_sdks():
    code = """
import importlib.abc
import sys

blocked = {'unilab', 'mujoco', 'mjbatch', 'motrixsim', 'pydrake', 'warp',
           'newton', 'genesis', 'superdex', 'isaacgym', 'isaacsim', 'torch', 'omni'}

class NoSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoSDKs())
from unisim import ADAPTER_SPECS, CapabilityReport, FakeBackend, get_adapter_capabilities
from unisim.support import get_tensor_platform_profiles
for spec in ADAPTER_SPECS:
    for profile in ('default', 'unrecorded-profile'):
        report = get_adapter_capabilities(spec.name, profile=profile)
        assert isinstance(report, CapabilityReport)
        assert all(not item.runtime_verified(report.scope) for item in report.declarations)
        assert report == CapabilityReport.from_dict(report.to_dict())
assert FakeBackend().get_capabilities().scope.adapter == 'fake'
profiles = get_tensor_platform_profiles()
assert set(profiles) == {spec.name for spec in ADAPTER_SPECS}
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_visualization_playback_primitives_load_without_viser_or_engine_sdks():
    code = """
import importlib.abc
import sys

blocked = {'unilab', 'viser', 'mujoco', 'mjbatch', 'motrixsim', 'pydrake', 'warp',
           'newton', 'genesis', 'superdex', 'isaacgym', 'isaacsim', 'torch', 'omni'}

class NoOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked, fullname
        return None

sys.meta_path.insert(0, NoOptionalRuntime())
import unisim.visualization
assert 'unisim.visualization.render_many' not in sys.modules
from unisim.visualization import playback_clock, viser_playback_controls, viser_utils
clock = playback_clock.PlaybackClock([0.0, 0.1, 0.35])
clock.set_playing(True, now=10.0)
assert clock.advance(now=10.2) == 1
assert viser_playback_controls.PlaybackTimelineGui is not None
assert viser_utils.nominal_fps([0.0, 0.02, 0.04]) == 50.0
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_visualization_resolves_render_many_on_first_access():
    pytest.importorskip("mujoco")
    import importlib

    import unisim.visualization as visualization

    assert "render_many" in visualization.__all__
    renderer = visualization.render_many
    assert renderer is importlib.import_module("unisim.visualization.render_many")
    assert renderer is visualization.render_many
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        visualization.missing
