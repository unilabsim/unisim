from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("motrixsim")

from unisim import MotrixBackend, assert_backend_conformance
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def test_motrix_backend_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    assert_backend_conformance(backend)
    assert not backend.get_tensor_capabilities().packed_reset
    backend._host_step(np.zeros((2, 1)))
    state = backend._host_get_state()
    widths = backend.get_public_state_widths()
    assert state["qpos"].shape == (2, 7 + backend.num_dof_vel)
    assert state["qpos"].shape == (2, widths.nq)
    assert state["qvel"].shape == (2, widths.nv)


def test_non_portable_body_torque_supported(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    try:
        capabilities = backend.get_dr_capabilities()
        assert "body_torque" in capabilities.supported_interval_terms
        zero = np.zeros((2, 1, 3), dtype=np.float32)
        torque = np.zeros((2, 1, 3), dtype=np.float32)
        torque[:, 0, 1] = (0.5, -0.25)
        backend.apply_body_force(np.asarray([0]), zero, torque=torque)
        # Re-applying the identical torque is a zero native delta and stays accepted.
        backend.apply_body_force(np.asarray([0]), zero, torque=torque)
        with pytest.raises(ValueError, match="body torque must have shape"):
            backend.apply_body_force(np.asarray([0]), zero, torque=np.zeros((2, 2, 3)))
        backend._host_reset()
        # After a reset the applied-torque baseline is cleared; a zero torque write
        # cancels the pending native torque through the delta path without faulting.
        backend.apply_body_force(np.asarray([0]), zero, torque=zero)
    finally:
        backend.close()
