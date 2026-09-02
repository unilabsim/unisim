from pathlib import Path

import numpy as np
import pytest

from unisim import MotrixBackend, assert_backend_conformance

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""


def test_motrix_backend_contract(tmp_path: Path) -> None:
    pytest.importorskip("motrixsim")
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MotrixBackend(model_path, num_envs=2)
    assert_backend_conformance(backend)
    backend.step(np.zeros((2, 1)))
    assert backend.get_state(("qpos",))["qpos"].shape == (2, 1)

