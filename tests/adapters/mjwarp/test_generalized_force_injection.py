"""Real-CUDA MJWarp generalized-force injection tests (#500).

The scenes are deliberately small and primitive-only.  They cover public
qvel/body ordering, ball-joint torque coordinates, persistent and clearing
lifecycle, reset independence, asynchronous rewrites without host publication,
eager/CUDA-graph parity, and a short trajectory comparison with Newton.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# ruff: noqa: E402
from unisim import MjwarpBackend
from unisim.dr.interval import TensorIntervalPayload
from unisim.scene import SceneCfg

pytest.importorskip("mujoco_warp")
torch = pytest.importorskip("torch")
warp = pytest.importorskip("warp")

_NUM_ENVS = 2
_DT = 0.005

_HINGE_MODEL = """
<mujoco model="unisim-mjwarp-force">
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
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

_BALL_MODEL = _HINGE_MODEL.replace(
    '<joint name="hinge" type="hinge" axis="0 1 0"/>',
    '<joint name="wrist_ball" type="ball"/>',
).replace('joint="hinge" ctrlrange="-1 1"', 'joint="wrist_ball" ctrlrange="-1 1"')

_SLIDE_ROOT_MODEL = """
<mujoco model="unisim-mjwarp-distance-root">
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="root_ball" type="ball"/>
      <!-- A tiny off-center mass keeps the root physically valid without
           coupling the first slide to the ball torque columns. -->
      <body name="tail" pos="0 0.001 0">
        <geom name="tail_geom" type="sphere" size="0.001" mass="0"/>
      </body>
      <geom name="body" type="sphere" size="0.07" mass="1.3"/>
    </body>
  </worldbody>
</mujoco>
"""

_ROOT_QVEL = slice(0, 6)
_HINGE_QVEL = 6
_HINGE_QPOS = 7
_BALL_QVEL = slice(6, 9)
_BALL_QPOS = slice(7, 11)


def _write_model(tmp_path: Path, name: str, xml: str) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return str(path)


def _mjwarp(
    tmp_path: Path,
    xml: str = _HINGE_MODEL,
    name: str = "model.xml",
    *,
    base_name: str | None = None,
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp generalized-force injection requires real CUDA")
    return MjwarpBackend(
        SceneCfg(model_file=_write_model(tmp_path, name, xml)),
        num_envs=_NUM_ENVS,
        sim_dt=_DT,
        base_name=base_name,
        nconmax=32,
        njmax=32,
    )


def _newton(tmp_path: Path, xml: str = _HINGE_MODEL) -> Any:
    from unisim.backend.newton.backend import NewtonBackend
    from unisim.backend.newton.dependencies import NewtonDependencyError

    try:
        from unisim.backend.newton.dependencies import load_newton_dependencies

        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    if not bool(deps.warp.get_device().is_cuda):
        pytest.skip("Newton trajectory comparison requires a CUDA Warp device")
    backend = NewtonBackend(
        SceneCfg(model_file=_write_model(tmp_path, "newton.xml", xml)),
        num_envs=_NUM_ENVS,
        sim_dt=_DT,
        device=str(deps.warp.get_device()),
        nconmax=32,
        njmax=32,
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


def _ctrl(backend: Any, device: Any, value: float = 0.0) -> Any:
    return torch.full((_NUM_ENVS, backend.num_actuators), value, device=device)


def test_joint_and_body_channels_use_public_coordinates_and_compose_with_ctrl(
    tmp_path: Path,
) -> None:
    backend = _mjwarp(tmp_path)
    try:
        assert backend.get_tensor_capabilities().generalized_force_injection
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        joint_f[:, _HINGE_QVEL] = 0.11
        body_f = torch.zeros((_NUM_ENVS, 3, 6), device=device)
        # The child body force is off-axis, so it must load hinge columns too.
        body_f[:, 2, 0] = 0.2
        backend.set_generalized_force_tensor(joint_f=joint_f, body_f=body_f)
        backend.step_tensor(_ctrl(backend, device, value=0.07), nsteps=3)
        views = backend.get_state_views()
        assert views["qvel"][:, _HINGE_QVEL].abs().max().item() > 0.01
        assert views["qvel"][:, :3].abs().max().item() > 0.0

        # Writing zero clears both control channels.  Reset the state so the
        # follow-up velocity assertions are independent of accumulated motion.
        backend.set_generalized_force_tensor(
            joint_f=torch.zeros_like(joint_f), body_f=torch.zeros_like(body_f)
        )
        backend.set_state_tensor(
            torch.arange(_NUM_ENVS, dtype=torch.int64, device=device),
            views["qpos"].clone(),
            torch.zeros_like(views["qvel"]),
        )
        backend.step_tensor(_ctrl(backend, device), nsteps=2)
        views = backend.get_state_views()
        assert views["qvel"][:, _HINGE_QVEL].abs().max().item() < 1e-5
        assert views["qvel"][:, :3].abs().max().item() < 1e-5
    finally:
        backend.close()


def test_distance_root_and_ball_joint_coordinates(tmp_path: Path) -> None:
    root = _mjwarp(tmp_path, _SLIDE_ROOT_MODEL, "root.xml")
    ball = _mjwarp(tmp_path / "ball", _BALL_MODEL, "ball.xml")
    try:
        views = root.get_state_views()
        nv = root.get_public_state_widths().nv
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=views["qpos"].device)
        joint_f[:, 0] = 0.3
        joint_f[:, 1] = -0.2
        root.set_generalized_force_tensor(joint_f=joint_f)
        root.step_tensor(_ctrl(root, views["qpos"].device), nsteps=4)
        views = root.get_state_views()
        np.testing.assert_allclose(
            views["qvel"][:, [0, 1]].cpu(),
            [[0.0046153846, -0.0030769231]] * _NUM_ENVS,
            rtol=2e-5,
            atol=2e-6,
        )

        # MJWarp and public MuJoCo qvel carry the same body-frame ball torque.
        views = ball.get_state_views()
        device = views["qpos"].device
        nv = ball.get_public_state_widths().nv
        expected = np.array([[0.3, -0.2, 0.5], [-0.4, 0.15, -0.35]], dtype=np.float32)
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        joint_f[:, _BALL_QVEL] = torch.as_tensor(expected, device=device)
        snapshot = joint_f.clone()
        ball.set_generalized_force_tensor(joint_f=joint_f)
        native = ball._persistent_joint_f.detach().cpu().numpy()
        np.testing.assert_allclose(native[:, _BALL_QVEL], expected, atol=0)
        # Other columns remain zero and the caller tensor is never converted.
        np.testing.assert_array_equal(native[:, : _BALL_QVEL.start], 0.0)
        assert torch.equal(joint_f, snapshot)
    finally:
        root.close()
        ball.close()


def test_injection_persists_across_nsteps_reset_and_clears_on_zero(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        joint_f[:, _HINGE_QVEL] = 0.08
        backend.set_generalized_force_tensor(joint_f=joint_f)
        before = views["qpos"].clone()
        rows = torch.arange(_NUM_ENVS, dtype=torch.int64, device=device)
        backend.set_state_tensor(rows, before, torch.zeros_like(views["qvel"]))
        backend.step_tensor(_ctrl(backend, device), nsteps=4)
        backend.step_tensor(_ctrl(backend, device), nsteps=4)
        views = backend.get_state_views()
        assert views["qvel"][:, _HINGE_QVEL].abs().max().item() > 0.02

        backend.set_generalized_force_tensor(joint_f=torch.zeros_like(joint_f))
        before_force = backend._persistent_joint_f[:, _HINGE_QVEL].clone()
        backend.step_tensor(_ctrl(backend, device), nsteps=2)
        views = backend.get_state_views()
        torch.testing.assert_close(backend._persistent_joint_f[:, _HINGE_QVEL], before_force)
        # The zero control applies no new force; nonzero velocity is momentum
        # left over from the driven interval.
        assert views["qvel"][:, _HINGE_QVEL].abs().max().item() > 0.01
    finally:
        backend.close()


def test_asynchronous_rewrite_is_ordered_and_needs_no_host_transfer(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        zero = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        weak = zero.clone()
        weak[:, _HINGE_QVEL] = 0.01
        strong = zero.clone()
        strong[:, _HINGE_QVEL] = 0.12

        backend.set_generalized_force_tensor(joint_f=weak)
        backend.step_tensor(_ctrl(backend, device), nsteps=2)
        # The rewrite is deliberately issued before waiting: same-stream CUDA
        # ordering means the first step reads weak, and the next reads strong.
        backend.set_generalized_force_tensor(joint_f=strong)
        second = backend.step_tensor(_ctrl(backend, device), nsteps=2)["completion"]
        second.wait_before_read()
        views = backend.get_state_views()
        velocity = views["qvel"][:, _HINGE_QVEL].detach().cpu().numpy()
        assert 0.01 < velocity.max() < 2.0

        # Another async rewrite while host cache remains stale.
        assert backend._host_cache_stale
        with (
            patch.object(backend, "_refresh_host_cache", side_effect=AssertionError("D2H leaked")),
            patch.object(backend, "_sync_host_cache", side_effect=AssertionError("sync leaked")),
        ):
            backend.set_generalized_force_tensor(joint_f=zero)
        assert backend._host_cache_stale
    finally:
        backend.close()


def test_eager_and_cuda_graph_execution_match_and_one_shot_wrench_is_unchanged(
    tmp_path: Path,
) -> None:
    graph = _mjwarp(tmp_path / "graph", base_name="base")
    eager = _mjwarp(tmp_path / "eager")
    try:
        if not graph._cuda_graph_enabled:
            pytest.skip(f"CUDA graph capture unavailable: {graph._cuda_graph_disable_reason}")
        eager._cuda_graph_enabled = False
        trajectories = []
        for backend in (graph, eager):
            views = backend.get_state_views()
            device = views["qpos"].device
            nv = backend.get_public_state_widths().nv
            joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
            joint_f[:, _HINGE_QVEL] = 0.05
            backend.set_generalized_force_tensor(joint_f=joint_f)
            backend.step_tensor(_ctrl(backend, device), nsteps=3)
            backend.set_state_tensor(
                torch.arange(_NUM_ENVS, dtype=torch.int64, device=device),
                views["qpos"].clone(),
                torch.zeros_like(views["qvel"]),
            )
            backend.step_tensor(_ctrl(backend, device), nsteps=5)
            trajectories.append(backend.get_state_views()["qpos"].detach().cpu().numpy())
        np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=2e-5, atol=2e-5)

        # One-shot interval force still applies exactly once and then clears;
        # this path deliberately uses the graph backend.
        views = graph.get_state_views()
        device = views["qpos"].device
        graph.step_tensor(_ctrl(graph, device), nsteps=1)["completion"].wait_before_read()
        # The legacy one-shot interval guard is still host-freshness based;
        # only clear it after awaiting the completed event.
        graph._host_cache_stale = False
        push = torch.ones((_NUM_ENVS, 3), dtype=torch.float32, device=device) * torch.tensor(
            [[1.0], [-1.0]], device=device
        )
        graph.push_robots(TensorIntervalPayload(push_force=push))
        baseline = graph.step_tensor(_ctrl(graph, device), nsteps=2)["completion"]
        baseline.wait_before_read()
        before = graph.get_state_views()["qvel"][:, :3].clone()
        graph.step_tensor(_ctrl(graph, device), nsteps=2)
        after = graph.get_state_views()["qvel"][:, :3]
        # Gravity continues after the push, but the native one-shot channel is
        # zero and does not replay the interval impulse.
        assert (after - before).abs().max().item() < 0.1
        torch.testing.assert_close(
            graph._persistent_body_f, torch.zeros_like(graph._persistent_body_f)
        )
    finally:
        graph.close()
        eager.close()


def test_short_hinge_trajectory_matches_newton_and_ctrl_composition(tmp_path: Path) -> None:
    mjwarp = _mjwarp(tmp_path / "mjwarp")
    newton = _newton(tmp_path / "newton")
    try:
        trajectories = []
        for backend in (mjwarp, newton):
            views = backend.get_state_views()
            device = views["qpos"].device
            nv = backend.get_public_state_widths().nv
            joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
            joint_f[:, _HINGE_QVEL] = 0.035
            # MJWarp includes the native world row; Newton's public table
            # starts at base. Translate the same physical wrench in zero gravity.
            nbody = backend._nbody if isinstance(backend, MjwarpBackend) else 2
            body_f = torch.zeros((_NUM_ENVS, nbody, 6), device=device)
            body_f[:, nbody - 2, 2] = 0.13
            body_f[:, nbody - 1, 1] = -0.015
            backend.set_generalized_force_tensor(joint_f=joint_f, body_f=body_f)
            for _ in range(12):
                backend.step_tensor(_ctrl(backend, device, value=0.03), nsteps=2)
            views = backend.get_state_views()
            trajectories.append(
                {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in views.items()
                    if name in ("qpos", "qvel")
                }
            )
        np.testing.assert_allclose(
            trajectories[0]["qpos"], trajectories[1]["qpos"], rtol=1e-3, atol=1e-7
        )
        np.testing.assert_allclose(
            trajectories[0]["qvel"], trajectories[1]["qvel"], rtol=1e-3, atol=1e-7
        )
    finally:
        mjwarp.close()
        newton.close()


def test_validation_fails_closed_without_partial_writes(tmp_path: Path) -> None:
    backend = _mjwarp(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        joint_f = torch.ones((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        body_f = torch.ones((_NUM_ENVS, backend._nbody, 6), device=device)

        with pytest.raises(ValueError, match="requires joint_f or body_f"):
            backend.set_generalized_force_tensor()
        with pytest.raises(ValueError, match=r"joint_f must have shape"):
            backend.set_generalized_force_tensor(joint_f=joint_f[:, :1])
        with pytest.raises(TypeError, match="float32"):
            backend.set_generalized_force_tensor(joint_f=joint_f.double())
        with pytest.raises(ValueError, match="contiguous"):
            backend.set_generalized_force_tensor(
                body_f=torch.empty_strided(
                    body_f.shape, (body_f.stride(0), body_f.stride(1) * 2, 1), device=device
                )
            )
        with pytest.raises(ValueError, match="finite"):
            bad = joint_f.clone()
            bad[0, 0] = float("nan")
            backend.set_generalized_force_tensor(joint_f=bad)
        with pytest.raises(ValueError, match="cuda"):
            backend.set_generalized_force_tensor(joint_f=joint_f.cpu())

        torch.testing.assert_close(backend._persistent_joint_f, torch.zeros_like(joint_f))
        torch.testing.assert_close(backend._persistent_body_f, torch.zeros_like(body_f))
    finally:
        backend.close()
