"""Real CUDA checks for the Motrix GPU tensor and capture lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from unisim import create_backend
from unisim.backend.motrix_gpu.dependencies import motrix_gpu_available
from unisim.conformance import assert_backend_conformance
from unisim.dr.types import TensorResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """
<mujoco model="motrix_gpu_test">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size=".15 .13 .12" mass="1"/>
      <body name="arm" pos=".2 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size=".1 .05 .04" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="drive" joint="hinge" kp="4" kv=".3"/></actuator>
  <sensor><framepos name="base_pos" objtype="body" objname="base"/></sensor>
</mujoco>
"""


@pytest.fixture
def cuda_backend(tmp_path):
    torch = pytest.importorskip("torch")
    if not motrix_gpu_available() or not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("Motrix GPU requires its optional NVIDIA CUDA runtime")
    model = tmp_path / "robot.xml"
    model.write_text(MODEL)
    backend = create_backend(
        "motrix_gpu", scene=SceneCfg(model_file=str(model)), num_envs=4, device="cuda"
    )
    try:
        yield backend, torch
    finally:
        backend.close()


def test_default_stream_tensor_lifecycle_and_reset_baseline(cuda_backend):
    backend, torch = cuda_backend
    assert_backend_conformance(backend)
    state = backend.get_state_views(device="cuda")
    ctrl = torch.zeros((4, 1), device="cuda")
    backend.step_tensor(ctrl, 2)
    before = backend.get_physics_state().copy()
    assert np.isfinite(before).all()
    mask = torch.tensor([True, False, False, True], device="cuda")
    qpos = state["qpos"].clone()
    qpos[mask, 2] += 1.0
    backend.set_state_tensor_mask(mask, qpos, state["qvel"].clone())
    after = backend.get_physics_state()
    np.testing.assert_array_equal(after[[1, 2]], before[[1, 2]])
    np.testing.assert_array_equal(after[[0, 3], 0], np.zeros(2))
    torch.testing.assert_close(backend.get_sensor_view("base_pos")[mask, 2], qpos[mask, 2])


def test_external_capture_replays_changing_masks_and_gain_overrides(cuda_backend):
    backend, torch = cuda_backend
    state = backend.get_state_views(device="cuda")
    ctrl = torch.zeros((4, 1), device="cuda")
    qpos = state["qpos"].clone()
    qvel = torch.zeros_like(state["qvel"])
    mask = torch.zeros(4, dtype=torch.bool, device="cuda")
    kp = torch.full((4, 1), 4.0, device="cuda")
    payload = TensorResetRandomizationPayload(kp=kp)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.step_tensor(ctrl, 2)
        backend.set_state_tensor_mask(mask, qpos, qvel, payload)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), backend.torch_graph_capture_scope():
        backend.step_tensor(ctrl, 2)
        backend.set_state_tensor_mask(mask, qpos, qvel, payload)

    for row, gain in ((1, 5.0), (3, 6.0)):
        mask.zero_()
        mask[row] = True
        kp[row] = gain
        graph.replay()
        backend.tensor_validation_verdicts().flush()
        torch.testing.assert_close(state["qpos"][row], qpos[row])
        assert torch.all(state["qpos"][~mask, 2] < qpos[~mask, 2])
    np.testing.assert_allclose(backend.get_reset_term_default("kp")[:, 0], [4, 5, 4, 6])
    del graph
