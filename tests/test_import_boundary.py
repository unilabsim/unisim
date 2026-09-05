import subprocess
import sys

import unisim


def test_import_does_not_pull_unilab_or_engine_modules():
    code = (
        "import sys, unisim; "
        "blocked = [name for name in sys.modules if name == 'unilab' or "
        "name.startswith(('hydra', 'torch', 'gymnasium', 'mujoco', 'motrixsim', "
        "'newton', 'warp'))]; "
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
