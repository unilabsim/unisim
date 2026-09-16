import subprocess
import sys

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


def test_public_exports_are_resolvable_and_wildcard_import_is_safe():
    namespace: dict[str, object] = {}
    exec("from unisim import *", {}, namespace)

    assert set(unisim.__all__).issubset(namespace)
    assert namespace["SubprocessBackend"] is namespace["MjcfSubprocessBackend"]


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
for spec in ADAPTER_SPECS:
    for profile in ('default', 'unrecorded-profile'):
        report = get_adapter_capabilities(spec.name, profile=profile)
        assert isinstance(report, CapabilityReport)
        assert all(not item.runtime_verified(report.scope) for item in report.declarations)
        assert report == CapabilityReport.from_dict(report.to_dict())
assert FakeBackend().get_capabilities().scope.adapter == 'fake'
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
