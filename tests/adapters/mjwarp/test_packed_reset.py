"""Device-resident packed reset transactions for the CUDA ``mjwarp`` backend."""

# ruff: noqa: E402
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
warp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")

from unisim import MjwarpBackend
from unisim.dr.types import RESET_TERM_BODY_MASS, RESET_TERM_GEOM_SIZE, RESET_TERM_KP
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp-packed-reset'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><position name='drive' joint='slide' kp='20' kv='2'/></actuator>
</mujoco>"""


def _make_backend(tmp_path: Path) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    return MjwarpBackend(SceneCfg(model_file=str(model_path)), num_envs=3, sim_dt=0.01)


def test_mjwarp_packed_reset_capabilities_and_geom_size_fail_closed(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.packed_reset
    assert {RESET_TERM_BODY_MASS, RESET_TERM_KP} <= capabilities.packed_reset_fields
    assert RESET_TERM_GEOM_SIZE not in capabilities.packed_reset_fields

    transaction = backend.begin_packed_reset()
    with pytest.raises(NotImplementedError, match="does not support packed reset field"):
        transaction.write_model_field(
            RESET_TERM_GEOM_SIZE,
            torch.empty(0, dtype=torch.int64, device="cuda"),
            torch.empty(0, 0, 3, dtype=torch.float32, device="cuda"),
        )
    backend.close()


def test_mjwarp_packed_reset_commits_device_resident_merge(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    transaction = backend.begin_packed_reset()
    baseline = transaction.baseline(RESET_TERM_BODY_MASS).clone()
    rows = torch.tensor([1], dtype=torch.int64, device="cuda")
    transaction.write_model_field(
        RESET_TERM_BODY_MASS,
        rows,
        torch.tensor([[2.0, 3.0]], dtype=torch.float32, device="cuda"),
    )
    qpos = torch.zeros((1, backend._nq), dtype=torch.float32, device="cuda")
    qvel = torch.zeros((1, backend._nv), dtype=torch.float32, device="cuda")
    result = transaction.commit_state(rows, qpos, qvel)
    assert result["timing"]["packed_reset_total_ms"] >= 0.0

    mass = backend._device_model.body_mass.numpy()
    np.testing.assert_allclose(mass[1], [2.0, 3.0])
    np.testing.assert_allclose(mass[[0, 2]], baseline[[0, 2]].cpu().numpy())
    np.testing.assert_allclose(
        transaction.baseline(RESET_TERM_BODY_MASS)[1].cpu().numpy(), [2.0, 3.0]
    )
    backend.close()
