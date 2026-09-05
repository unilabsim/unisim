"""Static and optional-runtime boundary tests for the Newton adapter."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import unisim
from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencyError,
    load_newton_dependencies,
    newton_dependencies_available,
)
from unisim.backend.newton.materialization import compute_contact_found_flags
from unisim.conformance import assert_backend_conformance
from unisim.scene import SceneCfg

_MODEL = """
<mujoco model="unisim-newton-test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="base_geom" type="sphere" size="0.08" mass="1"/>
      <body name="arm" pos="0 0 0.12">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge_motor" joint="hinge" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


def test_contact_found_flags_match_unordered_pairs_per_world() -> None:
    # Two env worlds; per-world pairs are (0, 1) and (2, 3).
    shape_world = np.array([0, 0, 1, 1], dtype=np.int64)
    shape_a = np.array([0, 2], dtype=np.int64)
    shape_b = np.array([1, 3], dtype=np.int64)
    # One contact in world 0 with the pair reversed, one in world 1 in order.
    shape0 = np.array([1, 2], dtype=np.int64)
    shape1 = np.array([0, 3], dtype=np.int64)
    flags = compute_contact_found_flags(shape_world, shape0, shape1, shape_a, shape_b)
    assert flags.dtype == np.float32
    assert flags.tolist() == [1.0, 1.0]


def test_contact_found_flags_attribute_only_the_contacting_world() -> None:
    shape_world = np.array([0, 0, 1, 1], dtype=np.int64)
    shape_a = np.array([0, 2], dtype=np.int64)
    shape_b = np.array([1, 3], dtype=np.int64)
    flags = compute_contact_found_flags(
        shape_world,
        np.array([3], dtype=np.int64),
        np.array([2], dtype=np.int64),
        shape_a,
        shape_b,
    )
    assert flags.tolist() == [0.0, 1.0]


def test_contact_found_flags_follow_non_static_shape_for_shared_worlds() -> None:
    # Shape 0 is a shared/static shape (world -1); each env owns one box shape.
    shape_world = np.array([-1, 0, 1], dtype=np.int64)
    shape_a = np.array([0, 0], dtype=np.int64)
    shape_b = np.array([1, 2], dtype=np.int64)
    flags = compute_contact_found_flags(
        shape_world,
        np.array([0, 2], dtype=np.int64),
        np.array([1, 0], dtype=np.int64),
        shape_a,
        shape_b,
    )
    assert flags.tolist() == [1.0, 1.0]


def test_contact_found_flags_reject_contacts_without_an_env_world() -> None:
    shape_world = np.array([-1, 0, 1], dtype=np.int64)
    shape_a = np.array([0, 0], dtype=np.int64)
    shape_b = np.array([1, 2], dtype=np.int64)
    # A contact between two shared shapes matches no env even when the shared
    # shape index collides with a resolved pair entry.
    flags = compute_contact_found_flags(
        shape_world,
        np.array([0, 0], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        shape_a,
        shape_b,
    )
    assert flags.tolist() == [0.0, 0.0]


def test_contact_found_flags_reject_cross_world_pairs() -> None:
    shape_world = np.array([0, 0, 1, 1], dtype=np.int64)
    shape_a = np.array([0, 2], dtype=np.int64)
    shape_b = np.array([1, 3], dtype=np.int64)
    # Shapes 1 (world 0) and 2 (world 1) can never legitimately touch; even
    # such a malformed contact must not raise a flag.
    flags = compute_contact_found_flags(
        shape_world,
        np.array([1], dtype=np.int64),
        np.array([2], dtype=np.int64),
        shape_a,
        shape_b,
    )
    assert flags.tolist() == [0.0, 0.0]


def test_contact_found_flags_empty_contacts_are_zero() -> None:
    shape_world = np.array([0, 0], dtype=np.int64)
    flags = compute_contact_found_flags(
        shape_world,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    assert flags.tolist() == [0.0]


def _patch_render_probes(monkeypatch: pytest.MonkeyPatch, *, native: bool, display: bool) -> None:
    monkeypatch.setattr(
        "unisim.backend.newton.backend.newton_render_dependencies_available",
        lambda: native,
    )
    monkeypatch.setattr(
        "unisim.backend.newton.backend.display_available",
        lambda: display,
    )


def test_newton_play_render_plan_record_native_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_render_probes(monkeypatch, native=True, display=False)
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="record", play_steps=24, output_video="play.mp4"
    )
    assert plan.mode == "record"
    assert plan.headless
    assert plan.record_video
    assert plan.num_steps == 24
    assert plan.output_video == "play.mp4"
    assert plan.renderer == "newton-viewer-gl"


def test_newton_play_render_plan_record_snapshot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_probes(monkeypatch, native=False, display=False)
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="record", play_steps=24, output_video="play.mp4"
    )
    assert plan.mode == "record"
    assert plan.headless
    assert plan.record_video
    assert plan.renderer == "mujoco-snapshot"


def test_newton_play_render_plan_none_is_inert() -> None:
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="none", play_steps=None, output_video=None
    )
    assert plan.mode == "none"
    assert plan.headless
    assert not plan.record_video
    assert plan.num_steps is None
    assert plan.output_video is None
    assert plan.renderer is None


def test_newton_play_render_plan_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_render_probes(monkeypatch, native=True, display=True)
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="interactive", play_steps=None, output_video=None
    )
    assert plan.mode == "interactive"
    assert not plan.headless
    assert not plan.record_video
    assert plan.num_steps is None
    assert plan.output_video is None
    assert plan.renderer == "newton-viewer-gl"


def test_newton_play_render_plan_interactive_requires_render_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_probes(monkeypatch, native=False, display=True)
    with pytest.raises(NotImplementedError, match="newton-render"):
        NewtonBackend.resolve_play_render_plan(
            play_render_mode="interactive", play_steps=None, output_video=None
        )


def test_newton_play_render_plan_interactive_requires_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_probes(monkeypatch, native=True, display=False)
    with pytest.raises(NotImplementedError, match="DISPLAY"):
        NewtonBackend.resolve_play_render_plan(
            play_render_mode="interactive", play_steps=None, output_video=None
        )


def test_newton_play_render_plan_auto_with_display(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_render_probes(monkeypatch, native=True, display=True)
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="auto", play_steps=None, output_video=None
    )
    assert plan.mode == "interactive"
    assert plan.renderer == "newton-viewer-gl"


def test_newton_play_render_plan_auto_without_display_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_probes(monkeypatch, native=False, display=False)
    plan = NewtonBackend.resolve_play_render_plan(
        play_render_mode="auto", play_steps=12, output_video="play.mp4"
    )
    assert plan.mode == "record"
    assert plan.headless
    assert plan.record_video
    assert plan.renderer == "mujoco-snapshot"


@pytest.mark.parametrize("play_steps", [None, 0, -3])
def test_newton_play_render_plan_requires_positive_steps(
    monkeypatch: pytest.MonkeyPatch, play_steps: int | None
) -> None:
    _patch_render_probes(monkeypatch, native=False, display=False)
    with pytest.raises(ValueError, match="play_steps"):
        NewtonBackend.resolve_play_render_plan(
            play_render_mode="record", play_steps=play_steps, output_video="play.mp4"
        )


def test_newton_play_render_plan_requires_output_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_render_probes(monkeypatch, native=False, display=False)
    with pytest.raises(ValueError, match="output video"):
        NewtonBackend.resolve_play_render_plan(
            play_render_mode="record", play_steps=10, output_video=None
        )


def test_newton_play_capabilities_follow_render_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NewtonBackend.__new__(NewtonBackend)
    monkeypatch.setattr(
        "unisim.backend.newton.backend.newton_render_dependencies_available",
        lambda: True,
    )
    capabilities = backend.get_play_capabilities()
    assert capabilities.supports_physics_state_playback
    assert capabilities.supports_native_interactive_renderer
    assert capabilities.supports_native_video_capture
    monkeypatch.setattr(
        "unisim.backend.newton.backend.newton_render_dependencies_available",
        lambda: False,
    )
    capabilities = backend.get_play_capabilities()
    assert capabilities.supports_physics_state_playback
    assert not capabilities.supports_native_interactive_renderer
    assert not capabilities.supports_native_video_capture


def test_newton_log_playback_plan_reports_renderer(capsys: pytest.CaptureFixture) -> None:
    from unisim.backend.base import BackendPlayRenderPlan, log_playback_plan

    plan = BackendPlayRenderPlan(
        mode="record",
        headless=True,
        record_video=True,
        num_steps=5,
        output_video="play.mp4",
        renderer="newton-viewer-gl",
    )
    log_playback_plan(plan)
    out = capsys.readouterr().out
    assert "newton-viewer-gl" in out


def test_newton_render_dependency_probe_is_fail_closed_when_extra_absent() -> None:
    from unisim.backend.newton.dependencies import (
        newton_render_dependencies_available,
        require_newton_render_dependencies,
    )

    if newton_render_dependencies_available():
        pytest.skip("newton render extra is installed in this environment")
    with pytest.raises(NewtonDependencyError, match="newton-render"):
        require_newton_render_dependencies()


def test_newton_init_renderer_fails_closed_without_render_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise NewtonDependencyError("missing viewer deps")

    monkeypatch.setattr(
        "unisim.backend.newton.backend.require_newton_render_dependencies", _raise
    )
    backend = NewtonBackend.__new__(NewtonBackend)
    backend._viewer = None
    backend._render_config = None
    with pytest.raises(NewtonDependencyError, match="missing viewer deps"):
        backend.init_renderer(headless=True, capture=True)


def test_newton_native_playback_validates_before_renderer_init() -> None:
    from unisim.backend.newton.playback import run_newton_native_playback

    backend = object()  # validation must run before any backend method is touched
    with pytest.raises(ValueError, match="headless=true"):
        run_newton_native_playback(
            backend=backend,
            env=None,
            initialize=lambda: None,
            step=lambda obs: obs,
            num_steps=5,
            output_video="play.mp4",
            render_spacing=None,
            headless=False,
            record_video=True,
            camera_kwargs=None,
        )
    with pytest.raises(ValueError, match="num_steps"):
        run_newton_native_playback(
            backend=backend,
            env=None,
            initialize=lambda: None,
            step=lambda obs: obs,
            num_steps=None,
            output_video="play.mp4",
            render_spacing=None,
            headless=True,
            record_video=True,
            camera_kwargs=None,
        )
    with pytest.raises(ValueError, match="output_video"):
        run_newton_native_playback(
            backend=backend,
            env=None,
            initialize=lambda: None,
            step=lambda obs: obs,
            num_steps=5,
            output_video=None,
            render_spacing=None,
            headless=True,
            record_video=True,
            camera_kwargs=None,
        )


def test_base_set_physics_state_fails_closed_by_default() -> None:
    from unisim.fake import FakeBackend

    with pytest.raises(NotImplementedError, match="physics-state restore"):
        FakeBackend().set_physics_state(np.zeros((2, 3), dtype=np.float32))


def test_newton_getters_do_not_materialize_warp_arrays() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(NewtonBackend)))
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    getter_nodes = [
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_")
    ]
    offenders = [
        getter.name
        for getter in getter_nodes
        for node in ast.walk(getter)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "numpy"
    ]
    assert offenders == []


def test_newton_import_boundary_does_not_load_optional_modules() -> None:
    code = (
        "import sys, unisim; "
        "assert not [name for name in sys.modules if name == 'newton' or "
        "name == 'warp' or name.startswith('mujoco_warp')]"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


def test_newton_manifest_and_lazy_exports() -> None:
    assert unisim.adapter_spec("newton").extra == "newton"
    assert unisim.NewtonBackend is NewtonBackend
    assert unisim.NewtonDependencyError is NewtonDependencyError


def test_newton_dependency_probe_is_fail_closed_when_extra_is_absent() -> None:
    if newton_dependencies_available():
        pytest.skip("Newton optional runtime is installed in this environment")
    with pytest.raises(NewtonDependencyError, match="newton backend requires"):
        unisim.create_backend("newton", scene=object())


def test_newton_conformance_when_cuda_runtime_is_available(tmp_path: Path) -> None:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton conformance requires a CUDA Warp device")
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    assert_backend_conformance(backend)
    backend.close()


def test_newton_physics_state_roundtrip_when_cuda_runtime_is_available(
    tmp_path: Path,
) -> None:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton physics-state roundtrip requires a CUDA Warp device")
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    ctrl = np.zeros((2, backend.num_actuators), dtype=np.float32)
    backend.step(ctrl, nsteps=3)
    snapshot = backend.get_physics_state()
    assert snapshot.dtype == np.float32
    assert snapshot.shape == (2, 1 + 8 + 7)
    assert np.isfinite(snapshot).all()

    backend.step(ctrl, nsteps=3)
    backend.set_physics_state(snapshot)
    restored = backend.get_physics_state()
    np.testing.assert_allclose(restored, snapshot, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="physics snapshot"):
        backend.set_physics_state(snapshot[:, :-1])
    backend.close()
