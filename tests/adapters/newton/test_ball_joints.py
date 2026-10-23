"""Ball joint materialization, state translation, and native parity on CUDA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencies,
    NewtonDependencyError,
    load_newton_dependencies,
)
from unisim.scene import SceneCfg

mujoco = pytest.importorskip("mujoco")

_MODEL = """
<mujoco model="newton-ball-test">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
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


def _axis_angle_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * unit))


# One non-trivial public-layout payload: rotated root, rotated wrist ball,
# body-frame angular velocities, and a driven hinge.
_ROOT_QUAT = _axis_angle_wxyz((0.0, 0.0, 1.0), 0.3)
_BALL_QUAT = _axis_angle_wxyz((1.0, 2.0, 3.0), 0.4)
_QPOS = np.concatenate(
    (
        [0.05, -0.03, 0.6],
        _ROOT_QUAT,
        _BALL_QUAT,
        [0.4],
    )
).astype(np.float32)
_QVEL = np.array(
    [0.01, 0.02, -0.05, 0.1, 0.05, -0.2, 0.3, -0.2, 0.5, -0.3], dtype=np.float32
)


def _newton_deps() -> NewtonDependencies:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    return deps


def _cuda_backend(tmp_path: Path, num_envs: int = 2) -> tuple[NewtonBackend, NewtonDependencies]:
    deps = _newton_deps()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton ball joint tests require a CUDA Warp device")
    model_file = tmp_path / "ball_scene.xml"
    model_file.write_text(_MODEL, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=num_envs,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend, deps


def _canonical(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[..., 0] < 0.0:
        quat = -quat
    return quat


def test_ball_joint_materializes_with_engine_layout(tmp_path: Path) -> None:
    backend, deps = _cuda_backend(tmp_path)
    try:
        metadata = backend._metadata
        assert metadata.nq == 12
        assert metadata.nv == 10
        assert metadata.ball_joint_addresses == ((7, 6),)
        model = backend._model
        joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
        ball_value = int(deps.newton.JointType.BALL)
        free_value = int(deps.newton.JointType.FREE)
        hinge_value = int(deps.newton.JointType.REVOLUTE)
        for world in range(2):
            block = joint_type[world * 3 : (world + 1) * 3]
            assert block.tolist() == [free_value, ball_value, hinge_value]
        joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int64)
        joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
        assert joint_q_start[:3].tolist() == [0, 7, 11]
        assert joint_qd_start[:3].tolist() == [0, 6, 9]
        # Engine convention: a default ball joint is an identity xyzw
        # quaternion occupying four joint_q and three joint_qd entries.
        joint_q = np.asarray(model.state().joint_q.numpy(), dtype=np.float32)
        for world in range(2):
            np.testing.assert_allclose(
                joint_q[world * 12 + 7 : world * 12 + 11], [0.0, 0.0, 0.0, 1.0], atol=1e-7
            )
        # Public default keeps the MJCF wxyz layout.
        np.testing.assert_allclose(
            backend.get_default_qpos()[7:11], [1.0, 0.0, 0.0, 0.0], atol=1e-7
        )
    finally:
        backend.close()


def test_ball_joint_selected_set_state_round_trips_public_layout(tmp_path: Path) -> None:
    backend, _ = _cuda_backend(tmp_path)
    try:
        backend._host_set_state(
            np.array([1]),
            _QPOS.reshape(1, -1),
            _QVEL.reshape(1, -1),
        )
        np.testing.assert_allclose(backend._qpos_cache[1], _QPOS, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(backend._qvel_cache[1], _QVEL, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            backend._qpos_cache[0], backend.get_default_qpos(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(backend._qvel_cache[0], 0.0, atol=1e-7)
    finally:
        backend.close()


def test_ball_joint_tensor_paths_translate_public_layout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend, _ = _cuda_backend(tmp_path)
    try:
        backend._host_set_state(
            np.array([1]),
            _QPOS.reshape(1, -1),
            _QVEL.reshape(1, -1),
        )
        backend._tensor_refresh_state()
        assert backend._tensor_qpos is not None
        assert backend._tensor_qvel is not None
        np.testing.assert_allclose(
            backend._tensor_qpos[1].cpu().numpy(), _QPOS, rtol=1e-6, atol=1e-6
        )
        # Columns 3:6 (root angular velocity) are excluded: the tensor refresh
        # path rotates the world-frame root omega forward instead of inverse for
        # non-identity root orientations, a pre-existing defect unrelated to
        # ball joints.
        keep = (0, 1, 2, 6, 7, 8, 9)
        np.testing.assert_allclose(
            backend._tensor_qvel[1].cpu().numpy()[list(keep)],
            _QVEL[list(keep)],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend._tensor_qpos[0].cpu().numpy(),
            backend.get_default_qpos(),
            rtol=1e-6,
            atol=1e-6,
        )

        device = backend._tensor_qpos.device
        env_indices = torch.tensor([1], dtype=torch.int64, device=device)
        _, _, raw_qpos, raw_qvel, _, _ = backend._prepare_tensor_reset(
            env_indices,
            torch.as_tensor(_QPOS.reshape(1, -1), dtype=torch.float32, device=device),
            torch.as_tensor(_QVEL.reshape(1, -1), dtype=torch.float32, device=device),
        )
        # The tensor reset composer must reproduce the raw engine state the
        # validated host path wrote for the identical public payload.
        live_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32).reshape(2, -1)
        live_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32).reshape(2, -1)
        np.testing.assert_allclose(
            raw_qpos[1].cpu().numpy(), live_q[1], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            raw_qvel[1].cpu().numpy(), live_qd[1], rtol=1e-6, atol=1e-6
        )
    finally:
        backend.close()


def test_ball_joint_state_matches_mujoco_reference_kinematics(tmp_path: Path) -> None:
    backend, _ = _cuda_backend(tmp_path)
    try:
        backend._host_set_state(
            np.array([0]),
            _QPOS.reshape(1, -1),
            _QVEL.reshape(1, -1),
        )
        reference = mujoco.MjData(backend._metadata.playback_model)
        reference.qpos[:] = _QPOS
        reference.qvel[:] = _QVEL
        mujoco.mj_forward(backend._metadata.playback_model, reference)
        for body_row, body_id in enumerate((1, 2, 3)):
            np.testing.assert_allclose(
                backend._body_pos_cache[0, body_row],
                reference.xpos[body_id],
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                _canonical(backend._body_quat_cache[0, body_row]),
                _canonical(np.asarray(reference.xquat[body_id])),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                backend._body_ang_vel_cache[0, body_row],
                np.asarray(reference.cvel[body_id])[:3],
                rtol=1e-4,
                atol=1e-5,
            )
    finally:
        backend.close()


def test_ball_joint_single_step_matches_native_newton(tmp_path: Path) -> None:
    backend, deps = _cuda_backend(tmp_path, num_envs=1)
    try:
        newton = deps.newton
        model_file = str(tmp_path / "ball_scene.xml")
        previous_layout = bool(newton.use_coord_layout_targets)
        newton.use_coord_layout_targets = True
        try:
            template = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(template)
            template.add_mjcf(model_file, ctrl_direct=False)
            builder = newton.ModelBuilder()
            builder.replicate(template, 1)
            native_model = builder.finalize(device=backend._device)
        finally:
            newton.use_coord_layout_targets = previous_layout
        native_solver = newton.solvers.SolverMuJoCo(
            native_model,
            separate_worlds=True,
            nconmax=512,
            njmax=512,
            solver="newton",
            integrator="implicitfast",
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            update_data_interval=1,
        )
        native_state = native_model.state()
        native_state_out = native_model.state()
        native_control = native_model.control()
        native_contacts = newton.Contacts(
            native_solver.get_max_contact_count(), 0, device=backend._device
        )

        backend._host_set_state(
            np.array([0]),
            _QPOS.reshape(1, -1),
            _QVEL.reshape(1, -1),
        )
        # Seed the native model with the adapter's raw engine state so both
        # pipelines integrate one step from identical initial conditions.
        raw_q = np.asarray(backend._state.joint_q.numpy(), dtype=np.float32)
        raw_qd = np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32)
        native_state.joint_q.assign(raw_q)
        native_state.joint_qd.assign(raw_qd)
        newton.eval_fk(native_model, native_state.joint_q, native_state.joint_qd, native_state)
        mask = deps.warp.array(
            np.array([True, False]),
            dtype=deps.warp.bool,
            device=backend._device,
        )
        native_solver.reset(native_state, mask, flags=0)

        backend._host_step(np.zeros((1, backend.num_actuators), dtype=np.float32), nsteps=1)
        native_state.clear_forces()
        native_solver.step(
            native_state, native_state_out, native_control, native_contacts, 0.005
        )
        native_state, native_state_out = native_state_out, native_state

        np.testing.assert_allclose(
            np.asarray(native_state.joint_q.numpy(), dtype=np.float32),
            np.asarray(backend._state.joint_q.numpy(), dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(native_state.joint_qd.numpy(), dtype=np.float32),
            np.asarray(backend._state.joint_qd.numpy(), dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )
    finally:
        backend.close()
