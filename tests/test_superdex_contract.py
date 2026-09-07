"""SuperDex optional boundary checks that run without the engine installed."""

import subprocess
import sys
from unittest.mock import patch

import pytest

from unisim import SuperDexBackend, create_backend
from unisim.backend.superdex.dependencies import (
    SuperDexDependencyError,
    load_superdex_dependencies,
)
from unisim.scene import SceneCfg


def test_superdex_class_is_concrete_and_does_not_import_runtime():
    assert not SuperDexBackend.__abstractmethods__
    code = (
        "import sys; from unisim import SuperDexBackend; "
        "assert not [n for n in sys.modules if n.startswith(('superdex', 'mujoco', 'torch'))]"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unsupported_python_has_actionable_diagnostic():
    with patch("unisim.backend.superdex.dependencies.sys.version_info", (3, 11, 0)):
        with pytest.raises(SuperDexDependencyError, match="Python 3.12"):
            load_superdex_dependencies()


def test_factory_routes_only_superdex_options(monkeypatch):
    seen = {}

    def construct(scene, num_envs, sim_dt, **kwargs):
        seen.update(kwargs)
        return "backend"

    monkeypatch.setattr("unisim.backend.superdex.SuperDexBackend", construct)
    result = create_backend(
        "superdex",
        SceneCfg("robot.superdex_bot"),
        superdex_num_threads=2,
        superdex_effort_limits=[3.0],
        superdex_allow_contact_approximation=True,
        newton_device="cuda:0",
        body_state_required=True,
    )
    assert result == "backend"
    assert seen == {"num_threads": 2, "effort_limits": [3.0], "allow_contact_approximation": True}


@pytest.mark.parametrize("num_envs", [True, 0, -1, 1.5])
def test_invalid_batch_is_rejected_before_loading_engine(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        SuperDexBackend(SceneCfg("unused"), num_envs, 0.01)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), 0.0, -0.01])
def test_invalid_step_size_is_rejected_before_loading_engine(dt):
    with pytest.raises(ValueError, match="sim_dt"):
        SuperDexBackend(SceneCfg("unused"), 1, dt)
