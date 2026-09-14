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
    assert len(observed) == nsteps
    np.testing.assert_allclose(observed[0][:, 2], backend.get_default_qpos()[2], atol=1e-6)
    # Semi-implicit Euler free fall: z_k = z_0 - g*dt^2*k*(k+1)/2; exact
    # equality proves the body-position getter returned substep-start state
    # rather than the previous control step's cache.
    dt = DT
    for k, body_pos in enumerate(observed):
        expected = backend.get_default_qpos()[2] - 10.0 * dt * dt * k * (k + 1) / 2.0
        np.testing.assert_allclose(body_pos[:, 2], expected, atol=1e-5)
