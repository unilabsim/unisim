"""Runtime tests for the CUDA ``mjwarp`` backend pre-step control contract."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp

from unisim import MjwarpBackend
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""


def _make_backend(tmp_path: Path, model_name: str = "model.xml") -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / model_name
    model_path.write_text(MODEL)
    return MjwarpBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)


def test_mjwarp_pre_step_control_per_substep(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    nsteps = 4
    step_calls = 8
    target = 0.2
    kp = 20.0
    observed_qpos: list[np.ndarray] = []
    observed_ctrl: list[np.ndarray] = []

    def p_controller(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        observed_qpos.append(owner.get_dof_pos().copy())
        observed_ctrl.append(ctrl.copy())
        return (target - owner.get_dof_pos()) * kp

    ctrl = np.zeros((2, 1), dtype=np.float32)
    backend.set_pre_step_control(p_controller)
    result = backend.step(ctrl, nsteps=nsteps)
    assert set(result["timing"]) == {"control_upload_ms", "physics_ms", "host_cache_refresh_ms"}
    for _ in range(step_calls - 1):
        backend.step(ctrl, nsteps=nsteps)

    # (a) The converter ran exactly once per physics substep and always
    # received the policy-level ctrl, not a previously converted value.
    assert len(observed_qpos) == step_calls * nsteps
    for received in observed_ctrl:
        np.testing.assert_array_equal(received, ctrl)
    # The callback saw fresh substep-start state: later observations reflect
    # the motion driven by earlier substep controls.
    assert not np.allclose(observed_qpos[0], observed_qpos[-1])

    # (b) The converted control drove the joint toward the P-law target.
    final_qpos = backend.get_dof_pos().copy()
    assert np.all(np.abs(final_qpos - target) < 0.05)

    # (c) Unregistering restores the direct control path: no further
    # callbacks, and zero ctrl applies no force (this model has no damping or
    # friction, so the joint coasts at constant velocity instead of seeking
    # the P-law target).
    backend.set_pre_step_control(None)
    coast_qvel = backend.get_dof_vel().copy()
    backend.step(np.zeros((2, 1), dtype=np.float32), nsteps=nsteps)
    assert len(observed_qpos) == step_calls * nsteps
    np.testing.assert_allclose(backend.get_dof_vel(), coast_qvel, atol=1e-5)


def test_mjwarp_pre_step_control_changes_trajectory(tmp_path: Path) -> None:
    baseline = _make_backend(tmp_path, "baseline.xml")
    driven = _make_backend(tmp_path, "driven.xml")
    nsteps = 4
    ctrl = np.zeros((2, 1), dtype=np.float32)
    for _ in range(8):
        baseline.step(ctrl, nsteps=nsteps)
    driven.set_pre_step_control(lambda owner, c: (0.2 - owner.get_dof_pos()) * 20.0)
    for _ in range(8):
        driven.step(ctrl, nsteps=nsteps)
    np.testing.assert_allclose(baseline.get_dof_pos(), 0.0, atol=1e-6)
    assert np.all(np.abs(driven.get_dof_pos() - baseline.get_dof_pos()) > 1e-2)


def test_mjwarp_pre_step_control_validates_return_shape(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    ctrl = np.zeros((2, 1), dtype=np.float32)
    backend.set_pre_step_control(lambda owner, c: np.zeros((2, 2), dtype=c.dtype))
    with pytest.raises(ValueError, match="pre-step control must return shape"):
        backend.step(ctrl, nsteps=1)
    backend.set_pre_step_control(None)
    backend.step(ctrl, nsteps=1)
