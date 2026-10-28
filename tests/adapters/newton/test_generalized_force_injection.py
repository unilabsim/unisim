"""Newton generalized-force injection acceptance tests (issue #433).

Covers the public ``set_generalized_force_tensor`` contract: per-substep
``body_f`` gravity compensation on a free body (the VOC smoke case), a
hinge-only ``joint_f`` PD servo, ball-joint torque frame conversion (public
body-frame columns rotate into Newton's parent anchor frame with the ball's
current orientation at injection time), per-substep semantic parity with
native Newton, the clearing/ownership contract, validation boundaries, and
CUDA graph replay. Scenes use primitive geoms and ball/hinge/free joints
only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencyError,
    load_newton_dependencies,
)
from unisim.scene import SceneCfg
from unisim.utils.rotation import np_quat_apply_batched

_FREE_BODY_MODEL = """
<mujoco model="unisim-newton-voc">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="object" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="object_geom" type="sphere" size="0.08" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_HINGE_MODEL = """
<mujoco model="unisim-newton-hinge">
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

_NUM_ENVS = 2
_SIM_DT = 0.005
_HINGE_QD_INDEX = 6
_HINGE_QPOS_INDEX = 7
_BODY_MASSES = (1.0, 0.2)

_BALL_MODEL = """
<mujoco model="unisim-newton-ball-force">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1"/>
    <body name="hand" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="palm" type="sphere" size="0.03" mass="0.5"/>
      <body name="finger" pos="0 0 0.05">
        <joint name="wrist_ball" type="ball"/>
        <geom name="tip" type="sphere" size="0.02" mass="0.1"/>
        <body name="nail" pos="0 0 0.03">
          <joint name="flex" type="hinge" axis="0 1 0"/>
          <geom name="nail_geom" type="sphere" size="0.01" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="flex_motor" joint="flex" ctrlrange="-1 1"/></actuator>
</mujoco>
"""

_BALL_QD_SLICE = slice(6, 9)
_BALL_QPOS_SLICE = slice(7, 11)
_BALL_HINGE_QD_INDEX = 9


def _axis_angle_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * unit))


# One non-identity ball orientation per env (public wxyz layout).
_BALL_QUATS = np.stack(
    (
        _axis_angle_wxyz((1.0, 2.0, 3.0), 0.4),
        _axis_angle_wxyz((0.0, 1.0, 0.0), -0.7),
    )
)
_BALL_TORQUES_BODY = np.array(
    [[0.3, -0.2, 0.5], [-0.4, 0.15, -0.35]], dtype=np.float32
)


def _cuda_backend(
    tmp_path: Path,
    xml: str = _HINGE_MODEL,
    **kwargs: Any,
) -> NewtonBackend:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton generalized-force injection requires a CUDA Warp device")
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "newton_force.xml"
    model_file.write_text(xml, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=_NUM_ENVS,
        sim_dt=_SIM_DT,
        device=str(device),
        capacity_check_steps=1,
        **kwargs,
    )
    backend.materialize()
    return backend


def _gravity_compensation(device: Any) -> Any:
    import torch

    wrench = torch.zeros(
        (_NUM_ENVS, len(_BODY_MASSES), 6), dtype=torch.float32, device=device
    )
    masses = torch.tensor(_BODY_MASSES, dtype=torch.float32, device=device)
    wrench[:, :, 2] = masses * 9.81
    return wrench


def test_body_f_gravity_compensation_holds_free_body_stationary(tmp_path: Path) -> None:
    """One injected constant wrench is consumed at every substep (VOC smoke)."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _FREE_BODY_MODEL)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        wrench = torch.zeros((_NUM_ENVS, 1, 6), dtype=torch.float32, device=device)
        wrench[:, :, 2] = 9.81
        backend.set_generalized_force_tensor(body_f=wrench)
        initial_qpos = views["qpos"].clone()
        initial_qvel = views["qvel"].clone()
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        # Two multi-substep steps prove consumption at every substep of
        # step_tensor(nsteps=k) and persistence across public calls.
        backend.step_tensor(ctrl, nsteps=8)
        backend.step_tensor(ctrl, nsteps=8)
        views = backend.get_state_views()
        assert (views["qpos"] - initial_qpos).abs().max().item() <= 1e-6
        assert (views["qvel"] - initial_qvel).abs().max().item() <= 1e-5
    finally:
        backend.close()


def test_body_f_zero_write_clears_injection(tmp_path: Path) -> None:
    """The clearing contract: an all-zeros write disarms the injected wrench."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _FREE_BODY_MODEL)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        wrench = torch.zeros((_NUM_ENVS, 1, 6), dtype=torch.float32, device=device)
        wrench[:, :, 2] = 9.81
        backend.set_generalized_force_tensor(body_f=wrench)
        backend.step_tensor(ctrl, nsteps=4)
        assert views["qvel"].abs().max().item() <= 1e-5
        backend.set_generalized_force_tensor(body_f=torch.zeros_like(wrench))
        backend.step_tensor(ctrl, nsteps=4)
        views = backend.get_state_views()
        falling = views["qvel"][:, 2]
        assert bool((falling < -0.1).all())
    finally:
        backend.close()


def test_joint_f_pd_servo_tracks_hinge_target(tmp_path: Path) -> None:
    """A hinge driven only by injected joint_f PD tracks its target angle."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        assert backend.get_tensor_capabilities().generalized_force_injection
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        target = 0.6
        kp, kd = 8.0, 0.3
        for _ in range(600):
            views = backend.get_state_views()
            joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
            joint_f[:, _HINGE_QD_INDEX] = (
                kp * (target - views["qpos"][:, _HINGE_QPOS_INDEX])
                - kd * views["qvel"][:, _HINGE_QD_INDEX]
            )
            backend.set_generalized_force_tensor(joint_f=joint_f)
            backend.step_tensor(ctrl)
        views = backend.get_state_views()
        # The unsupported chain keeps a small gravity-torque steady-state
        # error; the PD must still settle on the target neighborhood.
        assert (views["qpos"][:, _HINGE_QPOS_INDEX] - target).abs().max().item() < 0.015
        assert views["qvel"][:, _HINGE_QD_INDEX].abs().max().item() < 0.02
    finally:
        backend.close()


def test_injected_forces_survive_tensor_reset(tmp_path: Path) -> None:
    """Injected forces are control inputs: tensor resets do not clear them."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _FREE_BODY_MODEL)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        wrench = torch.zeros((_NUM_ENVS, 1, 6), dtype=torch.float32, device=device)
        wrench[:, :, 2] = 9.81
        backend.set_generalized_force_tensor(body_f=wrench)
        # Perturb, then reset all rows to the default state.
        backend.set_generalized_force_tensor(body_f=torch.zeros_like(wrench))
        backend.step_tensor(ctrl, nsteps=4)
        backend.set_generalized_force_tensor(body_f=wrench)
        env_indices = torch.arange(_NUM_ENVS, dtype=torch.int64, device=device)
        reset_qpos = views["qpos"].clone()
        backend.set_state_tensor(
            env_indices,
            reset_qpos,
            torch.zeros_like(views["qvel"]),
        )
        backend.step_tensor(ctrl, nsteps=8)
        views = backend.get_state_views()
        assert (views["qpos"] - reset_qpos).abs().max().item() <= 1e-6
    finally:
        backend.close()


def test_set_generalized_force_tensor_validation_fails_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        nv = backend.get_public_state_widths().nv
        nbody = 2
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        body_f = torch.zeros((_NUM_ENVS, nbody, 6), dtype=torch.float32, device=device)

        with pytest.raises(ValueError, match="requires joint_f or body_f"):
            backend.set_generalized_force_tensor()
        with pytest.raises(TypeError, match="joint_f must be a torch.Tensor"):
            backend.set_generalized_force_tensor(joint_f=np.zeros((_NUM_ENVS, nv)))
        with pytest.raises(ValueError, match="joint_f must have shape"):
            backend.set_generalized_force_tensor(joint_f=joint_f[:, :1])
        with pytest.raises(TypeError, match="body_f must be contiguous float32"):
            backend.set_generalized_force_tensor(body_f=body_f.double())
        non_contiguous = torch.zeros((_NUM_ENVS, nbody, 12), dtype=torch.float32, device=device)[
            :, :, :6
        ]
        assert not non_contiguous.is_contiguous()
        with pytest.raises(TypeError, match="body_f must be contiguous float32"):
            backend.set_generalized_force_tensor(body_f=non_contiguous)
        with pytest.raises(ValueError, match="Newton tensor device must be"):
            backend.set_generalized_force_tensor(joint_f=joint_f.cpu())
        # Valid single-channel writes succeed and leave the other channel alone.
        backend.set_generalized_force_tensor(joint_f=joint_f)
        backend.set_generalized_force_tensor(body_f=body_f)
    finally:
        backend.close()


def test_generalized_force_injection_portable_profile_is_declared() -> None:
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._portable_mode = True
    backend._device = "cuda:0"
    backend._entity_layout = None
    backend._entity_runtimes = {"entity_0": object()}
    assert backend.get_tensor_capabilities().generalized_force_injection

    multi = cast(Any, NewtonBackend.__new__(NewtonBackend))
    multi._portable_mode = True
    multi._device = "cuda:0"
    multi._entity_layout = None
    multi._entity_runtimes = {"entity_0": object(), "entity_1": object()}
    assert multi.get_tensor_capabilities().generalized_force_injection


def test_injected_forces_match_native_newton_per_substep_semantics(tmp_path: Path) -> None:
    """Public-API injection reproduces the native per-substep force sequence."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    rounds, kp, kd, target = 200, 3.0, 0.3, 0.6
    public_qpos: list[np.ndarray] = []
    public_hinge_qd: list[np.ndarray] = []
    try:
        views = backend.get_state_views()
        device = views["qpos"].device
        device_name = backend._device
        nv = backend.get_public_state_widths().nv
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        backend.set_generalized_force_tensor(
            body_f=_gravity_compensation(device)
        )
        for _ in range(rounds):
            views = backend.get_state_views()
            joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
            joint_f[:, _HINGE_QD_INDEX] = (
                kp * (target - views["qpos"][:, _HINGE_QPOS_INDEX])
                - kd * views["qvel"][:, _HINGE_QD_INDEX]
            )
            backend.set_generalized_force_tensor(joint_f=joint_f)
            backend.step_tensor(ctrl)
            views = backend.get_state_views()
            public_qpos.append(views["qpos"].cpu().numpy().copy())
            public_hinge_qd.append(
                views["qvel"][:, _HINGE_QD_INDEX].cpu().numpy().copy()
            )
    finally:
        backend.close()

    deps = load_newton_dependencies()
    warp, newton = deps.warp, deps.newton
    model_file = tmp_path / "newton_force.xml"
    previous_layout = bool(newton.use_coord_layout_targets)
    newton.use_coord_layout_targets = True
    try:
        template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.add_mjcf(str(model_file), ctrl_direct=False)
        builder = newton.ModelBuilder()
        builder.replicate(template, _NUM_ENVS)
        model = builder.finalize(device=device_name)
    finally:
        newton.use_coord_layout_targets = previous_layout
    solver = newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=True,
        nconmax=512,
        njmax=512,
        solver="newton",
        integrator="implicitfast",
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        update_data_interval=1,
    )
    state = model.state()
    state_out = model.state()
    control = model.control()
    contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=device_name)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    wrench = np.zeros((int(model.body_count), 6), dtype=np.float32)
    wrench[:, 2] = np.tile(np.asarray(_BODY_MASSES, dtype=np.float32) * 9.81, _NUM_ENVS)
    injected = warp.array(wrench, dtype=warp.spatial_vector, device=device_name)
    native_qpos: list[np.ndarray] = []
    native_hinge_qd: list[np.ndarray] = []
    for _ in range(rounds):
        joint_q = state.joint_q.numpy().reshape(_NUM_ENVS, -1)
        joint_qd = state.joint_qd.numpy().reshape(_NUM_ENVS, -1)
        joint_f = np.zeros_like(joint_qd)
        joint_f[:, _HINGE_QD_INDEX] = (
            kp * (target - joint_q[:, _HINGE_QPOS_INDEX]) - kd * joint_qd[:, _HINGE_QD_INDEX]
        )
        control.joint_f.assign(joint_f.reshape(-1))
        state.clear_forces()
        warp.copy(state.body_f, injected)
        solver.step(state, state_out, control, contacts, _SIM_DT)
        state, state_out = state_out, state
        native_qpos.append(state.joint_q.numpy().reshape(_NUM_ENVS, -1).copy())
        native_hinge_qd.append(
            state.joint_qd.numpy().reshape(_NUM_ENVS, -1)[:, _HINGE_QD_INDEX].copy()
        )

    public = np.asarray(public_qpos)
    native = np.asarray(native_qpos)
    # Public qpos reorders the root quaternion xyzw -> wxyz.
    public_xyzw_layout = public[..., [0, 1, 2, 4, 5, 6, 3, 7]]
    assert np.abs(public_xyzw_layout - native).max() <= 1e-5
    assert np.abs(np.asarray(public_hinge_qd) - np.asarray(native_hinge_qd)).max() <= 1e-5


def test_body_f_injection_applies_under_cuda_graph_replay(tmp_path: Path) -> None:
    """Graph replay consumes the persistent body_f buffer on every substep."""
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _FREE_BODY_MODEL, use_cuda_graph=True)
    try:
        if not backend._cuda_graph_enabled:
            pytest.skip(
                f"CUDA graph capture unavailable: {backend._cuda_graph_disable_reason}"
            )
        views = backend.get_state_views()
        device = views["qpos"].device
        wrench = torch.zeros((_NUM_ENVS, 1, 6), dtype=torch.float32, device=device)
        wrench[:, :, 2] = 9.81
        backend.set_generalized_force_tensor(body_f=wrench)
        initial_qpos = views["qpos"].clone()
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        backend.step_tensor(ctrl, nsteps=8)
        views = backend.get_state_views()
        assert (views["qpos"] - initial_qpos).abs().max().item() <= 1e-6
    finally:
        backend.close()


def test_joint_f_injection_matches_eager_under_cuda_graph_replay(tmp_path: Path) -> None:
    """A constant joint_f torque drives graph and eager steps identically."""
    torch = pytest.importorskip("torch")
    graph = _cuda_backend(tmp_path / "graph", use_cuda_graph=True)
    eager = _cuda_backend(tmp_path / "eager")
    try:
        if not graph._cuda_graph_enabled:
            pytest.skip(f"CUDA graph capture unavailable: {graph._cuda_graph_disable_reason}")
        trajectories: list[np.ndarray] = []
        for backend in (graph, eager):
            views = backend.get_state_views()
            device = views["qpos"].device
            nv = backend.get_public_state_widths().nv
            joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
            joint_f[:, _HINGE_QD_INDEX] = 0.05
            backend.set_generalized_force_tensor(joint_f=joint_f)
            ctrl = torch.zeros(
                (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
            )
            backend.step_tensor(ctrl, nsteps=8)
            views = backend.get_state_views()
            trajectories.append(views["qpos"].cpu().numpy().copy())
        assert np.abs(trajectories[0] - trajectories[1]).max() <= 1e-5
    finally:
        graph.close()
        eager.close()


def _posed_ball_backend(tmp_path: Path) -> NewtonBackend:
    """Materialize the ball model at a non-identity wrist pose on every env."""
    backend = _cuda_backend(tmp_path, _BALL_MODEL)
    nv = backend.get_public_state_widths().nv
    qpos = np.tile(backend.get_default_qpos(), (_NUM_ENVS, 1))
    qpos[:, _BALL_QPOS_SLICE] = _BALL_QUATS.astype(np.float32)
    backend._host_set_state(
        np.arange(_NUM_ENVS, dtype=np.intp),
        qpos,
        np.zeros((_NUM_ENVS, nv), dtype=np.float32),
    )
    return backend


def test_joint_f_ball_torque_injection_converts_body_to_anchor_frame(tmp_path: Path) -> None:
    """Public body-frame ball torques land in native joint_f in the anchor frame."""
    torch = pytest.importorskip("torch")
    backend = _posed_ball_backend(tmp_path)
    try:
        nv = backend.get_public_state_widths().nv
        views = backend.get_state_views()
        device = views["qpos"].device
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        joint_f[:, _BALL_QD_SLICE] = torch.as_tensor(
            _BALL_TORQUES_BODY, dtype=torch.float32, device=device
        )
        joint_f[:, _BALL_HINGE_QD_INDEX] = torch.tensor([0.7, -0.6], device=device)
        snapshot = joint_f.clone()
        backend.set_generalized_force_tensor(joint_f=joint_f)
        native = np.asarray(backend._control.joint_f.numpy(), dtype=np.float32).reshape(
            _NUM_ENVS, nv
        )
        expected = np_quat_apply_batched(_BALL_QUATS, _BALL_TORQUES_BODY)
        np.testing.assert_allclose(native[:, _BALL_QD_SLICE], expected, rtol=1e-6, atol=1e-7)
        # Hinge and root columns pass through unconverted.
        np.testing.assert_allclose(native[:, _BALL_HINGE_QD_INDEX], [0.7, -0.6], atol=1e-7)
        np.testing.assert_allclose(native[:, :6], 0.0, atol=1e-7)
        # The conversion never mutates the caller's tensor.
        assert torch.equal(joint_f, snapshot)
    finally:
        backend.close()


def test_joint_f_ball_torque_injection_matches_native_newton(tmp_path: Path) -> None:
    """A resident ball torque drives the adapter exactly like native Newton."""
    torch = pytest.importorskip("torch")
    backend = _posed_ball_backend(tmp_path)
    substeps = 16
    try:
        nv = backend.get_public_state_widths().nv
        views = backend.get_state_views()
        device = views["qpos"].device
        device_name = backend._device
        joint_f = torch.zeros((_NUM_ENVS, nv), dtype=torch.float32, device=device)
        joint_f[:, _BALL_QD_SLICE] = torch.as_tensor(
            _BALL_TORQUES_BODY, dtype=torch.float32, device=device
        )
        backend.set_generalized_force_tensor(joint_f=joint_f)
        # The resident torque is converted once, at injection time; snapshot
        # the raw engine state the conversion consumed for the native side.
        raw_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32).copy()
        raw_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32).copy()
        ctrl = torch.zeros(
            (_NUM_ENVS, backend.num_actuators), dtype=torch.float32, device=device
        )
        backend.step_tensor(ctrl, nsteps=substeps)
        final_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32).copy()
        final_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32).copy()
    finally:
        backend.close()

    deps = load_newton_dependencies()
    warp, newton = deps.warp, deps.newton
    model_file = tmp_path / "newton_force.xml"
    previous_layout = bool(newton.use_coord_layout_targets)
    newton.use_coord_layout_targets = True
    try:
        template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.add_mjcf(str(model_file), ctrl_direct=False)
        builder = newton.ModelBuilder()
        builder.replicate(template, _NUM_ENVS)
        model = builder.finalize(device=device_name)
    finally:
        newton.use_coord_layout_targets = previous_layout
    solver = newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=True,
        nconmax=512,
        njmax=512,
        solver="newton",
        integrator="implicitfast",
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
        update_data_interval=1,
    )
    state = model.state()
    state_out = model.state()
    control = model.control()
    contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=device_name)
    state.joint_q.assign(raw_q)
    state.joint_qd.assign(raw_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    mask = warp.array(
        np.concatenate((np.ones(_NUM_ENVS, dtype=np.bool_), (False,))),
        dtype=warp.bool,
        device=device_name,
    )
    solver.reset(state, mask, flags=0)
    # Native joint_f consumes parent-anchor-frame torques: rotate the public
    # body-frame payload once with the injection-time pose (native xyzw
    # becomes wxyz first), then leave the resident torque untouched.
    ball_wxyz = raw_q.reshape(_NUM_ENVS, -1)[:, _BALL_QPOS_SLICE][:, [3, 0, 1, 2]]
    native_joint_f = np.zeros((_NUM_ENVS, nv), dtype=np.float32)
    native_joint_f[:, _BALL_QD_SLICE] = np_quat_apply_batched(ball_wxyz, _BALL_TORQUES_BODY)
    control.joint_f.assign(native_joint_f.reshape(-1))
    for _ in range(substeps):
        state.clear_forces()
        solver.step(state, state_out, control, contacts, _SIM_DT)
        state, state_out = state_out, state
    np.testing.assert_allclose(
        np.asarray(state.joint_q.numpy(), dtype=np.float32), final_q, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(state.joint_qd.numpy(), dtype=np.float32), final_qd, rtol=1e-5, atol=1e-6
    )
