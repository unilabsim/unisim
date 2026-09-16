"""SuperDex optional boundary checks that run without the engine installed."""

import subprocess
import sys
from types import SimpleNamespace
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
        with pytest.raises(SuperDexDependencyError, match="Python 3.12 or 3.13"):
            load_superdex_dependencies()


def test_render_capability_uses_mujoco_offline_renderer():
    plan = SuperDexBackend.resolve_play_render_plan(
        play_render_mode="none",
        play_steps=10,
        output_video=None,
    )
    assert plan.mode == "none" and not plan.record_video
    for mode in ("auto", "record"):
        plan = SuperDexBackend.resolve_play_render_plan(
            play_render_mode=mode, play_steps=10, output_video="play.mp4"
        )
        assert plan.mode == "record" and plan.record_video
    plan = SuperDexBackend.resolve_play_render_plan(
        play_render_mode="interactive", play_steps=None, output_video=None
    )
    assert plan.mode == "interactive" and not plan.headless and not plan.record_video


def test_factory_routes_only_superdex_options(monkeypatch):
    seen = {}

    def construct(scene, num_envs, sim_dt, **kwargs):
        seen.update(kwargs)
        return "backend"

    monkeypatch.setattr("unisim.backend.superdex.SuperDexBackend", construct)
    result = create_backend(
        "superdex",
        SceneCfg("robot.superdex_bot"),
        superdex_num_workers=1,
        superdex_execution_mode="serial",
        superdex_effort_limits=[3.0],
        superdex_allow_contact_approximation=True,
        newton_device="cuda:0",
        body_state_required=True,
    )
    assert result == "backend"
    assert seen == {
        "num_workers": 1,
        "execution_mode": "serial",
        "effort_limits": [3.0],
        "allow_contact_approximation": True,
    }


@pytest.mark.parametrize("num_envs", [True, 0, -1, 1.5])
def test_invalid_batch_is_rejected_before_loading_engine(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        SuperDexBackend(SceneCfg("unused"), num_envs, 0.01)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), 0.0, -0.01])
def test_invalid_step_size_is_rejected_before_loading_engine(dt):
    with pytest.raises(ValueError, match="sim_dt"):
        SuperDexBackend(SceneCfg("unused"), 1, dt)


def test_invalid_execution_mode_is_rejected_before_loading_engine():
    with pytest.raises(ValueError, match="execution_mode"):
        SuperDexBackend(SceneCfg("unused"), 1, 0.01, execution_mode="threaded")
    with pytest.raises(TypeError, match="execution_mode"):
        SuperDexBackend(SceneCfg("unused"), 1, 0.01, execution_mode=True)
    with pytest.raises(ValueError, match="serial"):
        SuperDexBackend(SceneCfg("unused"), 1, 0.01, execution_mode="serial", num_workers=2)


@pytest.mark.parametrize("enabled", [False, True])
def test_contact_approximation_report_records_materializer_option(monkeypatch, enabled):
    from unisim.backend.superdex import backend as module
    from unisim.backend.superdex import materialization

    options = []
    physics = SimpleNamespace(uses_double_precision=lambda: False)
    plan = SimpleNamespace(
        body_names=("base",), joint_names=(), root_body_id=0, cleanup=lambda: None
    )

    def materialize_model(*args, **kwargs):
        options.append(kwargs["allow_contact_approximation"])
        return plan

    monkeypatch.setattr(module, "load_superdex_dependencies", lambda: (physics, None))
    monkeypatch.setattr(module, "acquire_runtime", lambda p: None)
    monkeypatch.setattr(module, "release_runtime", lambda p: None)
    monkeypatch.setattr(materialization, "materialize_model", materialize_model)
    monkeypatch.setattr(SuperDexBackend, "_allocate_caches", lambda self: None)
    monkeypatch.setattr(SuperDexBackend, "materialize", lambda self: None)
    backend = SuperDexBackend(
        SceneCfg("unused"), 1, 0.01, execution_mode="serial", allow_contact_approximation=enabled
    )
    try:
        field = next(
            item for item in backend.get_import_report().fields
            if item.field == "superdex_allow_contact_approximation"
        )
        assert field.effective is options[0] is enabled
        assert field.provenance[0].kind == "adapter_setting"
        assert all(
            item.difference == "unknown" for item in backend.get_import_report().fields
            if item.field != field.field
        )
    finally:
        backend.close()
