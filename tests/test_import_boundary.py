import subprocess
import sys


def test_import_does_not_pull_unilab_or_engine_modules():
    code = (
        "import sys, unisim; "
        "blocked = [name for name in sys.modules if name == 'unilab' or "
        "name.startswith(('hydra', 'torch', 'gymnasium', 'mujoco', 'motrixsim'))]; "
        "assert not blocked, blocked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or result.stdout
