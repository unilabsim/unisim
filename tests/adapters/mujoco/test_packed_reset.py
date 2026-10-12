from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")

from unisim import MuJoCoBackend  # noqa: E402
from unisim.dr.types import RESET_TERM_KP  # noqa: E402
from unisim.entities import ModelSourceDescriptor, SceneEntitySpec  # noqa: E402
from unisim.scene import SceneCfg  # noqa: E402

MODEL = """<mujoco model='packed-reset'>
  <option timestep='0.01' gravity='0 0 -9.81'/>
  <worldbody>
    <body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
      <geom type='box' size='0.05 0.05 0.05' mass='1'/>
      <body name='arm' pos='0 0 0.1'><joint name='hinge' axis='0 0 1'/>
        <geom type='box' size='0.02 0.02 0.02' mass='0.2'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name='slide_drive' joint='slide' kp='20' kv='2'/>
    <position name='arm_drive' joint='hinge' kp='30' kv='3'/>
  </actuator>
</mujoco>"""


def _backend(tmp_path: Path, *, num_envs: int = 3) -> MuJoCoBackend:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=num_envs,
        sim_dt=0.01,
        base_name="base",
    )
    backend.materialize()
    return backend


def _mocap_backend(tmp_path: Path, *, num_envs: int = 3) -> MuJoCoBackend:
    object_path = tmp_path / "object.xml"
    object_path.write_text(
        "<mujoco><worldbody><body name='base'><freejoint name='free'/>"
        "<geom name='shape' type='sphere' size='0.1' mass='1'/></body></worldbody></mujoco>"
    )
    scene = SceneCfg(
        entity_assets=(
            SceneEntitySpec("object", ModelSourceDescriptor(str(object_path)), kind="rigid"),
            SceneEntitySpec(
                "target",
                kind="rigid",
                root_mode="kinematic",
                collision_enabled=False,
                mirror_of="object",
            ),
        )
    )
    backend = MuJoCoBackend(
        scene,
        num_envs=num_envs,
        sim_dt=0.01,
        base_name="object/base",
    )
    backend.materialize()
    return backend


def test_mujoco_packed_reset_merges_pd_columns_and_promotes_baseline(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.packed_reset
    assert capabilities.device_reset_randomization
    assert {RESET_TERM_KP, "kd", "body_mass"} <= capabilities.packed_reset_fields

    transaction = backend.begin_packed_reset(device="cpu")
    baseline = transaction.baseline(RESET_TERM_KP).clone()
    np.testing.assert_allclose(baseline.numpy(), [[20.0, 30.0]] * 3, rtol=0.0)

    rows = torch.tensor([1], dtype=torch.int64)
    transaction.write_model_field(
        RESET_TERM_KP,
        rows,
        torch.tensor([[55.0]], dtype=torch.float32),
        columns=torch.tensor([1], dtype=torch.int64),
    )
    qpos = torch.zeros((1, backend.nq), dtype=torch.float32)
    qvel = torch.zeros((1, backend.nv), dtype=torch.float32)
    result = transaction.commit_state(rows, qpos, qvel)
    assert result["timing"]["packed_reset_total_ms"] >= 0.0

    gains = np.asarray(backend._pool.expand("actuator_gainprm"))[:, :, 0]
    np.testing.assert_allclose(gains[1], [20.0, 55.0], rtol=0.0)
    np.testing.assert_allclose(gains[[0, 2]], [[20.0, 30.0]] * 2, rtol=0.0)
    np.testing.assert_allclose(
        transaction.baseline(RESET_TERM_KP).numpy(),
        [[20.0, 30.0], [20.0, 55.0], [20.0, 30.0]],
        rtol=0.0,
    )

    # A later reset without a new PD write consumes the promoted baseline.
    transaction.commit_state(rows, qpos, qvel)
    gains = np.asarray(backend._pool.expand("actuator_gainprm"))[:, :, 0]
    np.testing.assert_allclose(gains[1], [20.0, 55.0], rtol=0.0)


def test_mujoco_packed_reset_selected_state_and_persistent_channels(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend._host_step(np.full((3, 2), 0.5), nsteps=2)
    transaction = backend.begin_packed_reset(device="cpu")
    rows = torch.tensor([2], dtype=torch.int64)
    qpos = torch.tensor([[0.25, -0.1]], dtype=torch.float32)
    qvel = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    before = backend.get_state_views(("qpos", "qvel"), device="cpu")
    transaction.commit_state(rows, qpos, qvel)
    after = backend.get_state_views(("qpos", "qvel"), device="cpu")
    torch.testing.assert_close(after["qpos"][rows], qpos)
    torch.testing.assert_close(after["qvel"][rows], qvel)
    torch.testing.assert_close(after["qpos"][0], before["qpos"][0])
    np.testing.assert_allclose(backend._ctrl_view[2], 0.0)
    np.testing.assert_allclose(backend._time_view[2], 0.0)


def test_mujoco_packed_reset_commits_selected_mocap_poses(tmp_path: Path) -> None:
    backend = _mocap_backend(tmp_path)
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.packed_reset_mocap

    transaction = backend.begin_packed_reset(device="cpu")
    state = backend._host_get_state()
    transaction.set_state_default(
        torch.as_tensor(state["qpos"][0], dtype=torch.float32),
        torch.as_tensor(state["qvel"][0], dtype=torch.float32),
    )
    rows = torch.tensor([1])
    transaction.begin_state(rows)
    pose = torch.tensor([[2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0]])
    transaction.write_mocap_pose("target/base", rows, pose)
    result = transaction.commit(rows)
    assert result["timing"]["packed_reset_total_ms"] >= 0.0

    binding = backend.bind_mocap_pose("target/base")
    np.testing.assert_allclose(binding.read()[rows.detach().numpy()], pose.numpy(), rtol=0.0)

    qpos_before = backend.get_state_views(("qpos",), device="cpu")["qpos"].clone()
    second_rows = torch.tensor([2])
    second_pose = torch.tensor([[5.0, 6.0, 7.0, 1.0, 0.0, 0.0, 0.0]])
    transaction.write_mocap_pose("target/base", second_rows, second_pose)
    transaction.commit_mocap_poses()
    torch.testing.assert_close(
        backend.get_state_views(("qpos",), device="cpu")["qpos"], qpos_before
    )
    np.testing.assert_allclose(
        binding.read()[second_rows.detach().numpy()], second_pose.numpy(), rtol=0.0
    )
