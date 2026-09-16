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
    assert backend.get_state(("qpos",))["qpos"].shape == (2, 1)
    backend.reset(np.asarray([1], dtype=np.intp))
    reset_qpos = backend.get_state(("qpos",))["qpos"][1]
    np.testing.assert_allclose(reset_qpos, 0.0)


@pytest.mark.parametrize("with_object_free_joint", [False, True])
def test_mujoco_state_snapshot_matches_set_state_layout(
    tmp_path: Path, with_object_free_joint: bool
) -> None:
    extra_body = (
        "<body name='object' pos='1 0 1'><freejoint name='object_free'/>"
        "<geom type='sphere' size='0.1' mass='1' contype='0' conaffinity='0'/></body>"
        if with_object_free_joint
        else ""
    )
    model_path = tmp_path / "model.xml"
    model_path.write_text(
        "<mujoco><option timestep='0.01' gravity='0 0 0'/>"
        "<worldbody><body name='base'><body name='arm' pos='0 0 1'>"
        "<joint name='hinge' axis='0 0 1'/>"
        "<geom type='box' size='0.1 0.1 0.1' mass='1' contype='0' conaffinity='0'/>"
        f"</body></body>{extra_body}</worldbody>"
        "<actuator><motor joint='hinge' ctrlrange='-1 1'/></actuator></mujoco>"
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=2,
        sim_dt=0.01,
        base_name="object" if with_object_free_joint else "base",
    )
    backend.materialize()
    ids = np.arange(2, dtype=np.int32)
    qpos = np.zeros((2, backend.nq))
    qvel = np.zeros((2, backend.nv))
    qpos[:, 0] = [0.2, 0.4]
    qvel[:, 0] = [0.3, 0.6]
    object_pose = np.tile([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], (2, 1))
    if with_object_free_joint:
        qpos[:, 1:8] = object_pose

    backend.set_state(ids, qpos, qvel)
    state = backend.get_state(("qpos", "qvel"))
    assert state["qpos"].shape == qpos.shape
    assert state["qvel"].shape == qvel.shape
    np.testing.assert_allclose(state["qpos"], qpos, atol=1e-12)
    np.testing.assert_allclose(state["qvel"], qvel, atol=1e-12)
    if with_object_free_joint:
        layout = backend.get_root_state_layout("object")
        np.testing.assert_allclose(state["qpos"][:, layout.qpos_indices], object_pose)

    detached = state["qpos"].copy()
    state["qpos"][:] += 10.0
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], detached)

    backend.step(np.zeros((2, 1)), nsteps=1)
    after_step = backend.get_state(("qpos", "qvel"))
    assert after_step["qpos"].shape == qpos.shape
    assert after_step["qvel"].shape == qvel.shape
    backend.set_state(ids, after_step["qpos"], after_step["qvel"])
