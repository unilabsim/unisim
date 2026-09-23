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
from unisim.backend.newton.backend import (
    NewtonBackend,
    _add_newton_render_floor,
    _cuda_graph_eligibility,
    _prepare_newton_render_floor,
)
from unisim.backend.newton.dependencies import (
    NewtonDependencies,
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


class _RenderFloorCfg:
    def __init__(self) -> None:
        self.is_visible = False
        self.has_shape_collision = True
        self.has_particle_collision = True

    def copy(self) -> "_RenderFloorCfg":
        copied = _RenderFloorCfg()
        copied.is_visible = self.is_visible
        copied.has_shape_collision = self.has_shape_collision
        copied.has_particle_collision = self.has_particle_collision
        return copied


class _RenderFloorBuilder:
    def __init__(
        self, shape_type: list[int], shape_body: list[int], shape_flags: list[int]
    ) -> None:
        self.shape_type = shape_type
        self.shape_body = shape_body
        self.shape_flags = shape_flags
        self.default_shape_cfg = _RenderFloorCfg()
        self.floor_cfg = None

    def add_ground_plane(self, *, cfg) -> None:
        self.floor_cfg = cfg


class _RenderFloorNewton:
    class GeoType:
        PLANE = 7

    class ShapeFlags:
        VISIBLE = 1


def test_newton_render_floor_makes_authored_static_plane_visible() -> None:
    builder = _RenderFloorBuilder([7, 2], [-1, 0], [2, 3])
    assert _prepare_newton_render_floor(builder, _RenderFloorNewton)
    assert builder.shape_flags == [3, 3]


def test_newton_render_floor_fallback_is_visual_only() -> None:
    builder = _RenderFloorBuilder([], [], [])
    _add_newton_render_floor(builder)
    assert builder.floor_cfg is not None
    assert builder.floor_cfg.is_visible
    assert not builder.floor_cfg.has_shape_collision
    assert not builder.floor_cfg.has_particle_collision


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
    with pytest.raises(NotImplementedError, match="uv sync --extra newton"):
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


def test_newton_render_dependency_probe_is_fail_closed_when_dependencies_absent() -> None:
    from unisim.backend.newton.dependencies import (
        newton_render_dependencies_available,
        require_newton_render_dependencies,
    )

    if newton_render_dependencies_available():
        pytest.skip("newton render dependencies are installed in this environment")
    with pytest.raises(NewtonDependencyError, match="uv sync --extra newton"):
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


class _StubCudaDevice:
    is_cuda = True

    def __str__(self) -> str:
        return "cuda:0"


class _StubWarp:
    @staticmethod
    def set_device(device: str) -> None:
        del device

    @staticmethod
    def get_device() -> _StubCudaDevice:
        return _StubCudaDevice()


class _EligibilityDevice:
    def __init__(self, *, is_cuda: bool = True) -> None:
        self.is_cuda = is_cuda


class _EligibilityWarp:
    def __init__(
        self,
        *,
        driver_version: tuple[int, int] | None | Exception = (12, 4),
        mempool_enabled: bool | Exception = True,
    ) -> None:
        self.driver_version = driver_version
        self.mempool_enabled = mempool_enabled

    def get_cuda_driver_version(self) -> tuple[int, int] | None:
        if isinstance(self.driver_version, Exception):
            raise self.driver_version
        return self.driver_version

    def is_mempool_enabled(self, device: _EligibilityDevice) -> bool:
        del device
        if isinstance(self.mempool_enabled, Exception):
            raise self.mempool_enabled
        return self.mempool_enabled


@pytest.mark.parametrize(
    ("warp", "device", "expected_reason"),
    [
        (
            _EligibilityWarp(),
            _EligibilityDevice(is_cuda=False),
            "active Warp device is not CUDA",
        ),
        (
            _EligibilityWarp(driver_version=RuntimeError("driver unavailable")),
            _EligibilityDevice(),
            "CUDA driver query failed: RuntimeError: driver unavailable",
        ),
        (
            _EligibilityWarp(driver_version=None),
            _EligibilityDevice(),
            "CUDA driver version is unavailable",
        ),
        (
            _EligibilityWarp(mempool_enabled=RuntimeError("mempool unavailable")),
            _EligibilityDevice(),
            "CUDA mempool query failed: RuntimeError: mempool unavailable",
        ),
        (
            _EligibilityWarp(driver_version=(12, 3), mempool_enabled=False),
            _EligibilityDevice(),
            "CUDA driver 12.3 is older than 12.4; CUDA mempool is disabled",
        ),
    ],
)
def test_newton_cuda_graph_eligibility_reports_fallback_reason(
    warp: _EligibilityWarp, device: _EligibilityDevice, expected_reason: str
) -> None:
    assert _cuda_graph_eligibility(warp, device) == (False, expected_reason)


def test_newton_cuda_graph_eligibility_accepts_cuda_12_4_and_mempool() -> None:
    assert _cuda_graph_eligibility(_EligibilityWarp(), _EligibilityDevice()) == (True, None)


def test_newton_constructor_validates_cuda_graph_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(
        "unisim.backend.newton.backend.load_newton_dependencies",
        lambda: NewtonDependencies(
            newton=None, warp=_StubWarp, mujoco=mujoco, mujoco_warp=None
        ),
    )
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=1,
        sim_dt=0.005,
        device="cuda:0",
        use_cuda_graph=False,
    )
    assert backend._use_cuda_graph is False
    assert backend._cuda_graph_enabled is False
    backend.close()

    with pytest.raises(TypeError, match="use_cuda_graph must be bool"):
        NewtonBackend(
            SceneCfg(model_file=str(model_file)),
            num_envs=1,
            sim_dt=0.005,
            device="cuda:0",
            use_cuda_graph=1,
        )


def test_factory_routes_newton_cuda_graph_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unisim.backend.newton import backend as newton_backend

    calls: list[dict[str, object]] = []

    class _RoutedBackend:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(newton_backend, "NewtonBackend", _RoutedBackend)
    scene = SceneCfg(model_file=str(tmp_path / "newton.xml"))
    unisim.create_backend(
        "newton",
        scene=scene,
        num_envs=2,
        sim_dt=0.005,
        newton_device="cuda:0",
        newton_use_cuda_graph=True,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["device"] == "cuda:0"
    assert calls[0]["kwargs"]["use_cuda_graph"] is True


def test_newton_motion_body_ids_follow_mjcf_worldbody_zero_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(
        "unisim.backend.newton.backend.load_newton_dependencies",
        lambda: NewtonDependencies(
            newton=None, warp=_StubWarp, mujoco=mujoco, mujoco_warp=None
        ),
    )
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=1,
        sim_dt=0.005,
        device="cuda:0",
    )

    model = mujoco.MjModel.from_xml_path(str(model_file))
    names = ["base", "arm"]
    expected = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names],
        dtype=np.int32,
    )
    assert expected.tolist() == [1, 2]  # MJCF body ids, worldbody is id 0
    np.testing.assert_array_equal(backend.get_motion_body_ids(names), expected)
    np.testing.assert_array_equal(
        backend.get_motion_body_ids(names), backend.get_body_ids(names) + 1
    )


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


def _newton_backend_for_graph_tests(
    model_file: Path, *, use_cuda_graph: bool
) -> NewtonBackend:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton CUDA graph tests require a CUDA Warp device")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
        use_cuda_graph=use_cuda_graph,
    )
    backend.materialize()
    return backend


def test_newton_cuda_graphs_match_eager_for_odd_even_and_repeated_steps(
    tmp_path: Path,
) -> None:
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    eager = _newton_backend_for_graph_tests(model_file, use_cuda_graph=False)
    graph = _newton_backend_for_graph_tests(model_file, use_cuda_graph=True)
    try:
        if not graph._cuda_graph_enabled:
            pytest.skip(graph._cuda_graph_disable_reason)
        ctrl = np.full((2, eager.num_actuators), 0.25, dtype=np.float32)
        for nsteps in (1, 2, 3):
            eager.step(ctrl, nsteps=nsteps)
            graph.step(ctrl, nsteps=nsteps)
            np.testing.assert_allclose(
                graph.get_physics_state(),
                eager.get_physics_state(),
                rtol=2e-4,
                atol=2e-5,
            )

        graph.step(ctrl, nsteps=2)
        eager.step(ctrl, nsteps=2)
        np.testing.assert_allclose(
            graph.get_physics_state(),
            eager.get_physics_state(),
            rtol=2e-4,
            atol=2e-5,
        )
    finally:
        eager.close()
        graph.close()


def test_newton_cuda_graph_replay_survives_set_state(tmp_path: Path) -> None:
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    eager = _newton_backend_for_graph_tests(model_file, use_cuda_graph=False)
    graph = _newton_backend_for_graph_tests(model_file, use_cuda_graph=True)
    try:
        if not graph._cuda_graph_enabled:
            pytest.skip(graph._cuda_graph_disable_reason)
        snapshot = eager.get_physics_state().copy()
        snapshot[:, 1:4] = np.array([0.1, -0.1, 0.6], dtype=np.float32)
        snapshot[:, 5] = 0.0
        snapshot[:, 1 + eager._metadata.nq :] = 0.0
        graph.set_physics_state(snapshot)
        eager.set_physics_state(snapshot)
        ctrl = np.zeros((2, graph.num_actuators), dtype=np.float32)
        graph.step(ctrl, nsteps=3)
        eager.step(ctrl, nsteps=3)
        assert graph._cuda_graph_enabled
        np.testing.assert_allclose(
            graph.get_physics_state(),
            eager.get_physics_state(),
            rtol=2e-4,
            atol=2e-5,
        )
    finally:
        eager.close()
        graph.close()


def test_newton_pre_step_control_callback_uses_eager_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    graph = _newton_backend_for_graph_tests(model_file, use_cuda_graph=True)
    try:
        if not graph._cuda_graph_enabled:
            pytest.skip(graph._cuda_graph_disable_reason)

        def _reject_replay() -> None:
            raise AssertionError("pre-step control path must remain eager")

        monkeypatch.setattr(graph, "_replay_cuda_graph_substep", _reject_replay)
        graph.set_pre_step_control(lambda backend, ctrl: ctrl)
        ctrl = np.zeros((2, graph.num_actuators), dtype=np.float32)
        graph.step(ctrl, nsteps=2)
        assert graph._cuda_graph_enabled
    finally:
        graph.close()


def test_newton_cuda_graph_ineligibility_warns_and_records_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "unisim.backend.newton.backend._cuda_graph_eligibility",
        lambda warp, device: (False, "forced ineligibility"),
    )
    model_file = tmp_path / "newton.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="forced ineligibility"):
        backend = _newton_backend_for_graph_tests(model_file, use_cuda_graph=True)
    try:
        assert not backend._cuda_graph_enabled
        assert backend._cuda_graph_disable_reason == "forced ineligibility"
    finally:
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

    layout = backend.get_physics_state_layout()
    assert (layout.nq, layout.nv, layout.nmocap) == (8, 7, 0)
    assert layout.state_width == snapshot.shape[1]
    parts = layout.split_state(snapshot)
    np.testing.assert_array_equal(parts.qpos, snapshot[:, 1:9])
    assert parts.mocap_pos is None and parts.mocap_quat is None

    backend.step(ctrl, nsteps=3)
    backend.set_physics_state(snapshot)
    restored = backend.get_physics_state()
    np.testing.assert_allclose(restored, snapshot, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="physics snapshot"):
        backend.set_physics_state(snapshot[:, :-1])
    backend.close()
