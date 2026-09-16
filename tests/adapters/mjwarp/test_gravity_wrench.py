"""Real-CUDA tests for mjwarp per-env gravity and body-wrench control (#71)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp

from unisim import MjwarpBackend, PreStepControlOutput
from unisim.dr.interval import INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import IntervalRandomizationPlan, IntervalTermOp, ResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """<mujoco>
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="object" pos="0 0 0.5"><freejoint/>
      <geom name="ball" type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="slider" pos="1 0 0.5">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="slide_geom" type="box" size="0.05 0.06 0.07" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slide" ctrlrange="-10 10"/></actuator>
</mujoco>"""


DT = 0.005


def _make_backend(
    tmp_path: Path, name: str = "scene.xml", *, num_envs: int = 2, add_body_sensors: bool = False
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / name
    model_path.write_text(MODEL)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs,
        0.005,
        base_name="object",
        add_body_sensors=add_body_sensors,
    )


def _reset(backend: MjwarpBackend, rows=None, gravity=None) -> None:
    if rows is None:
        rows = np.arange(backend.num_envs, dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (len(rows), 1))
    qvel = np.tile(backend.get_init_qvel(), (len(rows), 1))
    payload = None if gravity is None else ResetRandomizationPayload(gravity=gravity)
    backend.set_state(rows, qpos, qvel, randomization=payload)


def _zero_ctrl(backend: MjwarpBackend) -> np.ndarray:
    return np.zeros((backend.num_envs, backend.num_actuators), dtype=np.float32)


# --------------------------------------------------------------------- #
# Per-environment gravity reset                                         #
# --------------------------------------------------------------------- #


def test_gravity_reset_sets_selected_worlds_only(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(
        backend,
        gravity=np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -20.0]], dtype=np.float32),
    )
    dt = DT
    nsteps = 4
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    # Free-joint qvel layout: [vx, vy, vz, wx, wy, wz]; index 2 is world z.
    np.testing.assert_allclose(
        backend.get_state(("qvel",))["qvel"][:, 2],
        [-10.0 * dt * nsteps, -20.0 * dt * nsteps],
        atol=1e-5,
    )

    # A partial reset of env 0 back to weightlessness must not disturb env 1:
    # env 0 restarts at rest and stays at rest, while env 1 keeps accelerating.
    env1_velocity = backend.get_state(("qvel",))["qvel"][1].copy()
    _reset(backend, np.array([0], dtype=np.int32), gravity=np.zeros((1, 3), dtype=np.float32))
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][0, 2], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        backend.get_state(("qvel",))["qvel"][1, 2],
        env1_velocity[2] - 20.0 * dt * nsteps,
        atol=1e-5,
    )


def test_gravity_reset_capability_default_and_validation(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    assert backend.get_dr_capabilities().supports_reset_term("gravity")
    default = backend.get_reset_term_default("gravity")
    assert default.shape == (3,)
    np.testing.assert_allclose(default, np.zeros(3), atol=0.0)

    with pytest.raises(ValueError, match="gravity must have shape"):
        _reset(backend, gravity=np.zeros((backend.num_envs, 2), dtype=np.float32))


def test_gravity_reset_persists_until_next_update(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend, gravity=np.array([[0.0, 0.0, -9.81]] * 2, dtype=np.float32))
    for _ in range(3):
        backend.step(_zero_ctrl(backend), nsteps=2)
    dt = DT
    np.testing.assert_allclose(
        backend.get_state(("qvel",))["qvel"][:, 2], -9.81 * dt * 6, atol=1e-4
    )


# --------------------------------------------------------------------- #
# Body wrench: direct call and interval torque                          #
# --------------------------------------------------------------------- #


def _object_body_id(backend: MjwarpBackend) -> np.ndarray:
    return backend.get_body_ids(["object"])


def _torque_wrench(backend: MjwarpBackend, torque_z: float) -> tuple[np.ndarray, np.ndarray]:
    bodies = _object_body_id(backend)
    zero_force = np.zeros((backend.num_envs, bodies.size, 3), dtype=np.float32)
    torque = np.zeros_like(zero_force)
    torque[..., 2] = torque_z
    return zero_force, torque


def test_apply_body_force_torque_only_and_consumption(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    # Solid sphere, m=1, r=0.05: I_zz = 2/5 * m * r^2 = 1e-3.
    zero_force, torque = _torque_wrench(backend, 1e-3)
    backend.apply_body_force(bodies, zero_force, torque=torque)
    backend.step(_zero_ctrl(backend), nsteps=2)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 5], 2 * DT, atol=1e-5)

    # The staged wrench is consumed by the step and must not affect the next one.
    omega = backend.get_state(("qvel",))["qvel"][:, 5].copy()
    backend.step(_zero_ctrl(backend), nsteps=2)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 5], omega, atol=1e-6)


def test_apply_body_force_accumulates_within_control_step(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    zero_force, torque = _torque_wrench(backend, 5e-4)
    backend.apply_body_force(bodies, zero_force, torque=torque)
    backend.apply_body_force(bodies, zero_force, torque=torque)
    backend.step(_zero_ctrl(backend), nsteps=1)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 5], DT, atol=1e-5)


def test_interval_body_torque_term_matches_direct_call(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    assert backend.get_dr_capabilities().supports_interval_term("body_torque")
    bodies = _object_body_id(backend)
    _, torque = _torque_wrench(backend, 1e-3)
    plan = IntervalRandomizationPlan(
        ops=(IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, torque, body_ids=bodies),)
    )
    backend.apply_interval_randomization(plan)
    backend.step(_zero_ctrl(backend), nsteps=2)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 5], 2 * DT, atol=1e-5)


def test_interval_plan_replaces_previous_unconsumed_wrench(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    zero_force, _ = _torque_wrench(backend, 0.0)
    _, first = _torque_wrench(backend, 1e-3)
    _, second = _torque_wrench(backend, 2e-3)
    backend.apply_body_force(bodies, zero_force, torque=first)
    backend.apply_interval_randomization(
        IntervalRandomizationPlan(
            ops=(IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, second, body_ids=bodies),)
        )
    )
    backend.step(_zero_ctrl(backend), nsteps=1)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 5], 2 * DT, atol=1e-5)


# --------------------------------------------------------------------- #
# Per-substep dynamic wrench through the pre-step control callback       #
# --------------------------------------------------------------------- #


def test_pre_step_wrench_recomputed_per_substep_and_cleared(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    nsteps = 4
    calls = {"k": 0}

    def controller(owner: MjwarpBackend, ctrl: np.ndarray) -> PreStepControlOutput:
        k = calls["k"]
        calls["k"] += 1
        force = np.zeros((owner.num_envs, bodies.size, 3), dtype=np.float64)
        force[..., 0] = float(k)
        return PreStepControlOutput(ctrl=ctrl, body_ids=bodies, force=force)

    backend.set_pre_step_control(controller)
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    assert calls["k"] == nsteps
    # Unit mass, zero gravity: dv = sum_k F_k * dt.  The per-substep forces
    # 0,1,2,3 N must not accumulate (persisting the last value would give
    # 4*3 instead of 0+1+2+3).
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 0], 6 * DT, atol=1e-5)

    # A later substep with zero dynamic wrench must not see the previous
    # substep's value persist; the channel is cleared between control steps.
    velocity = backend.get_state(("qvel",))["qvel"][:, 0].copy()

    def idle(owner: MjwarpBackend, ctrl: np.ndarray) -> PreStepControlOutput:
        zero = np.zeros((owner.num_envs, bodies.size, 3), dtype=np.float64)
        return PreStepControlOutput(ctrl=ctrl, body_ids=bodies, force=zero)

    backend.set_pre_step_control(idle)
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 0], velocity, atol=1e-6)

    # The strongest leak probe: unregister the callback entirely.  A leftover
    # dynamic wrench on the device channel would keep accelerating the body
    # through the direct control path, which does not rewrite xfrc.
    backend.set_pre_step_control(None)
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 0], velocity, atol=1e-6)


def test_pre_step_wrench_composes_with_fixed_interval_disturbance(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    nsteps = 4
    fixed_force = np.zeros((backend.num_envs, bodies.size, 3), dtype=np.float32)
    fixed_force[..., 0] = 1.0
    backend.apply_body_force(bodies, fixed_force)
    calls = {"k": 0}

    def controller(owner: MjwarpBackend, ctrl: np.ndarray) -> PreStepControlOutput:
        k = calls["k"]
        calls["k"] += 1
        dynamic = np.zeros((owner.num_envs, bodies.size, 3), dtype=np.float64)
        if k % 2 == 1:
            dynamic[..., 0] = 0.5
        return PreStepControlOutput(ctrl=ctrl, body_ids=bodies, force=dynamic)

    backend.set_pre_step_control(controller)
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    # Fixed 1 N on all four substeps plus 0.5 N on substeps 1 and 3.
    np.testing.assert_allclose(
        backend.get_state(("qvel",))["qvel"][:, 0], (4 * 1.0 + 2 * 0.5) * DT, atol=1e-5
    )

    # The fixed interval wrench is consumed with the step call, and no dynamic
    # wrench remains once the callback is unregistered.
    backend.set_pre_step_control(None)
    velocity = backend.get_state(("qvel",))["qvel"][:, 0].copy()
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    np.testing.assert_allclose(backend.get_state(("qvel",))["qvel"][:, 0], velocity, atol=1e-6)


def test_apply_body_force_inside_callback_fails_closed(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    _reset(backend)
    bodies = _object_body_id(backend)
    zero = np.zeros((backend.num_envs, bodies.size, 3), dtype=np.float32)

    def bad(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        owner.apply_body_force(bodies, zero)
        return ctrl

    backend.set_pre_step_control(bad)
    with pytest.raises(RuntimeError, match="must not be called from inside a pre-step control"):
        backend.step(_zero_ctrl(backend), nsteps=1)
    # The backend remains usable after the fail-closed callback error.
    backend.set_pre_step_control(None)
    backend.step(_zero_ctrl(backend), nsteps=1)


def test_pre_step_callback_sees_fresh_body_state(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, add_body_sensors=True)
    _reset(backend, gravity=np.array([[0.0, 0.0, -10.0]] * backend.num_envs, dtype=np.float32))
    bodies = _object_body_id(backend)
    observed: list[np.ndarray] = []

    def recorder(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        observed.append(owner.get_body_pos_w(bodies)[:, 0, :].copy())
        return ctrl

    backend.set_pre_step_control(recorder)
    nsteps = 4
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    # Exercise the control-cycle boundary, not only the first step after reset.
    # The host qpos/qvel cache was refreshed at the previous step boundary, but
    # tracked-body views must be recomputed from that state before substep 0.
    observed.clear()
    backend.step(_zero_ctrl(backend), nsteps=nsteps)
    assert len(observed) == nsteps
    # Semi-implicit Euler free fall: z_k = z_0 - g*dt^2*k*(k+1)/2; exact
    # equality proves the body-position getter returned substep-start state
    # rather than the previous control step's cache.
    dt = DT
    for k, body_pos in enumerate(observed):
        expected = (
            backend.get_default_qpos()[2]
            - 10.0 * dt * dt * (nsteps + k) * (nsteps + k + 1) / 2.0
        )
        np.testing.assert_allclose(body_pos[:, 2], expected, atol=1e-5)


@pytest.mark.parametrize("with_callback", [False, True])
def test_step_returns_body_state_aligned_with_final_state(
    tmp_path: Path, with_callback: bool
) -> None:
    backend = _make_backend(tmp_path, add_body_sensors=True)
    rows = np.arange(backend.num_envs, dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (backend.num_envs, 1))
    qvel = np.zeros((backend.num_envs, backend.get_init_qvel().size), dtype=np.float32)
    qvel[:, 0] = 1.0
    qvel[:, 5] = 0.5
    backend.set_state(rows, qpos, qvel)
    if with_callback:
        backend.set_pre_step_control(lambda owner, ctrl: ctrl)
    bodies = _object_body_id(backend)
    position_columns = np.asarray(backend.get_root_state_layout("object").qpos_indices[:3])
    nsteps = 4 if with_callback else 1

    for _ in range(2):
        state = backend.get_state(("qpos", "qvel"))
        backend.step(_zero_ctrl(backend), nsteps=nsteps)
        final_state = backend.get_state(("qpos", "qvel"))
        np.testing.assert_allclose(
            backend.get_body_pos_w(bodies)[:, 0, :],
            final_state["qpos"][:, position_columns],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.get_body_quat_w(bodies)[:, 0, :],
            final_state["qpos"][:, 3:7],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            backend.get_body_lin_vel_w(bodies)[:, 0, :],
            final_state["qvel"][:, 0:3],
            atol=1e-5,
        )
        assert not np.allclose(
            final_state["qpos"][:, position_columns], state["qpos"][:, position_columns]
        )

    backend.set_pre_step_control(None)


def test_body_state_refresh_preserves_authored_force_sensor(tmp_path: Path) -> None:
    xml = """<mujoco>
      <option timestep="0.005" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="root" pos="0 0 0.2">
          <freejoint/>
          <geom name="ball" type="sphere" size="0.1" mass="1"/>
          <site name="root_site"/>
        </body>
      </worldbody>
      <sensor><force name="root_force" site="root_site"/></sensor>
    </mujoco>"""
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / "force-scene.xml"
    model_path.write_text(xml)
    backend = MjwarpBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=1,
        sim_dt=DT,
        base_name="root",
        add_body_sensors=True,
    )
    rows = np.array([0], dtype=np.int32)
    qpos = backend.get_default_qpos()[None]
    qvel = np.zeros((1, backend.get_init_qvel().size), dtype=np.float32)
    qvel[:, 2] = -2.0
    backend.set_state(rows, qpos, qvel)
    backend.step(_zero_ctrl(backend), nsteps=30)

    completed_substep_force = backend.get_sensor_data("root_force").copy()
    assert np.linalg.norm(completed_substep_force) > 0.0
    bodies = backend.get_body_ids(["root"])
    backend.get_body_state_w(bodies)
    np.testing.assert_array_equal(
        backend.get_sensor_data("root_force"), completed_substep_force
    )


def test_ctrl_only_callback_skips_body_kinematics_refresh(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, add_body_sensors=True)
    _reset(backend)
    bodies = _object_body_id(backend)
    calls = {"n": 0}
    original = backend._refresh_tracked_body_state_device

    def counting() -> None:
        calls["n"] += 1
        original()

    backend._refresh_tracked_body_state_device = counting
    backend.set_pre_step_control(lambda owner, c: owner.get_dof_pos() * 0.0)
    backend.step(_zero_ctrl(backend), nsteps=4)
    # Neither the ctrl-only callback nor an unread post-step body view pays
    # the kinematics-refresh cost.
    assert calls["n"] == 0
    calls["n"] = 0

    # A callback that reads body state refreshes at most once per substep,
    # including the first substep after a previous control-step barrier.
    def reader(owner, c):
        owner.get_body_pos_w(bodies)
        return c

    backend.set_pre_step_control(reader)
    backend.step(_zero_ctrl(backend), nsteps=4)
    assert calls["n"] == 4


@pytest.mark.parametrize("num_envs", [2, 1024])
@pytest.mark.parametrize("read_before_reset", [False, True])
def test_partial_reset_preserves_current_complement_body_state(
    tmp_path: Path, num_envs: int, read_before_reset: bool
) -> None:
    backend = _make_backend(tmp_path, num_envs=num_envs, add_body_sensors=True)
    rows = np.arange(num_envs, dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (num_envs, 1))
    qvel = np.tile(backend.get_init_qvel(), (num_envs, 1))
    qvel[:, 0] = 1.0
    backend.set_state(rows, qpos, qvel)
    backend.step(_zero_ctrl(backend), nsteps=2)
    bodies = _object_body_id(backend)
    if read_before_reset:
        backend.get_body_pos_w(bodies)
    backend.set_state(rows[:1], qpos[:1], np.zeros_like(qvel[:1]))
    np.testing.assert_allclose(
        backend.get_body_pos_w(bodies)[:, 0, 0],
        backend.get_state("qpos")["qpos"][:, 0],
        atol=1e-6,
    )


@pytest.mark.parametrize("completed_steps", [0, 1, 3])
def test_pre_step_wrench_cleared_after_midstep_callback_exception(
    tmp_path: Path, completed_steps: int
) -> None:
    backend = _make_backend(tmp_path, add_body_sensors=True)
    _reset(backend, gravity=np.zeros((backend.num_envs, 3), dtype=np.float32))
    bodies = _object_body_id(backend)
    calls = {"k": 0}

    def interrupted(owner: MjwarpBackend, ctrl: np.ndarray) -> PreStepControlOutput:
        k = calls["k"]
        calls["k"] += 1
        if k == completed_steps:
            raise RuntimeError("intentional callback interruption")
        force = np.zeros((owner.num_envs, bodies.size, 3), dtype=np.float32)
        force[..., 0] = 1.0
        return PreStepControlOutput(ctrl=ctrl, body_ids=bodies, force=force)

    backend.set_pre_step_control(interrupted)
    with pytest.raises(RuntimeError, match="intentional callback interruption"):
        backend.step(_zero_ctrl(backend), nsteps=4)

    # Completed impulses remain visible through public state, without a
    # private synchronization call or a subsequent step to repair the cache.
    velocity_before = backend.get_state(("qvel",))["qvel"][:, 0].copy()
    np.testing.assert_allclose(velocity_before, completed_steps * DT, atol=1e-6)
    np.testing.assert_allclose(backend.get_physics_state()[:, 0], completed_steps * DT)
    expected_x = DT * DT * completed_steps * (completed_steps + 1) / 2
    np.testing.assert_allclose(
        backend.get_state("qpos")["qpos"][:, 0], expected_x, atol=1e-6
    )
    np.testing.assert_allclose(backend.get_body_pos_w(bodies)[:, 0, 0], expected_x, atol=1e-6)

    # No staged or residual device wrench may continue into a later direct step.
    backend.set_pre_step_control(None)
    backend.step(_zero_ctrl(backend), nsteps=1)
    velocity_after = backend.get_state(("qvel",))["qvel"][:, 0]
    np.testing.assert_allclose(velocity_after, velocity_before, atol=1e-6)
