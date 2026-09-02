from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from unisim import MuJoCoBackend, assert_backend_conformance
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def test_mujoco_backend_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    assert_backend_conformance(backend)
    backend.step(np.ones((2, 1)), nsteps=2)
    assert backend.get_state(("qpos",))["qpos"].shape == (2, 7 + backend.get_dof_pos().shape[1])
    backend.reset(np.asarray([1], dtype=np.intp))
    reset_qpos = backend.get_state(("qpos",))["qpos"][1]
    np.testing.assert_allclose(reset_qpos[:3], 0.0)
    np.testing.assert_allclose(reset_qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(reset_qpos[7:], 0.0)
