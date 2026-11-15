"""Real CUDA oracles for substep samples, contact frames and stream ordering."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from unisim import MjwarpBackend
from unisim.scene import SceneCfg

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
mjw = pytest.importorskip("mujoco_warp")

MODEL = """<mujoco model='physics-tensor-oracle'>
<option timestep='0.005'/>
<worldbody><geom type='plane' size='2 2 .1'/>
<body name='base' pos='0 0 .049'><joint name='slide' type='slide' axis='0 0 1'/>
<geom type='sphere' size='.05' mass='1'/></body></worldbody>
<actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""


def _contact_oracle(backend):
    """Accumulate official native contact forces with independent NumPy indexing."""
    data = backend._device_data
    capacity = data.contact.geom.shape[0]
    ids = wp.array(np.arange(capacity, dtype=np.int32), dtype=int, device=data.qvel.device)
    forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=data.qvel.device)
    mjw.contact_force(backend._device_model, data, ids, False, forces)
    wp.synchronize()
    count = int(data.nacon.numpy()[0])
    worlds, geoms = data.contact.worldid.numpy(), data.contact.geom.numpy()
    frames, values = data.contact.frame.numpy(), forces.numpy()
    body_id = backend._body_ids["base"]
    result = np.zeros((2, 3), dtype=np.float32)
    for contact in range(count):
        for side in range(2):
            geom = int(geoms[contact, side])
            if geom >= 0 and backend._cpu_model.geom_bodyid[geom] == body_id:
                result[worlds[contact]] += (2 * side - 1) * values[contact, 0] * frames[contact, 0]
    return result


def _step_and_wait(backend, ctrl, nsteps):
    consumer = torch.cuda.current_stream(ctrl.device)
    # Exercise backend-owned Warp stream ordering, including an alternate stream.
    with wp.ScopedStream(wp.stream_from_torch(consumer)):
        receipt = backend.step_tensor(ctrl, nsteps=nsteps)["completion"]
    receipt.wait_before_read(consumer)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("alternate_stream", [False, True])
@pytest.mark.parametrize("reset_kind", ["selected", "mask"])
def test_substeps_contact_and_selected_reset(tmp_path, graph, alternate_stream, reset_kind):
    if not torch.cuda.is_available():
        pytest.skip("requires a explicitly allocated CUDA device")
    wp.init()
    xml = tmp_path / "model.xml"
    xml.write_text(MODEL)
    backends = [
        MjwarpBackend(SceneCfg(model_file=str(xml)), num_envs=2, sim_dt=0.005) for _ in range(2)
    ]
    try:
        for backend in backends:
            if graph and not backend._cuda_graph_enabled:
                pytest.fail(f"required graph unavailable: {backend._cuda_graph_disable_reason}")
            backend._cuda_graph_enabled = graph
        views = [
            b.bind_physics_step_tensor_data(
                joint_names=("slide",),
                body_names=("base",),
                history_length=3,
            )
            for b in backends
        ]
        ptrs = [
            (v.joint_acceleration.data_ptr(), v.contact_force_history.data_ptr()) for v in views
        ]
        stream = torch.cuda.Stream() if alternate_stream else None
        ctrl = torch.zeros((2, 1), device="cuda", dtype=torch.float32)
        for cycle in range(3):
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                ctrl.fill_(-0.2 * cycle)
                _step_and_wait(backends[0], ctrl, nsteps=4)
                for _ in range(4):
                    before = backends[1].get_state_views(("qvel",))["qvel"].clone()
                    previous = views[1].contact_force_history.clone()
                    _step_and_wait(backends[1], ctrl, nsteps=1)
                    after = backends[1].get_state_views(("qvel",))["qvel"]
                    torch.testing.assert_close(
                        views[1].joint_acceleration, (after - before) / 0.005
                    )
                    torch.testing.assert_close(
                        views[1].contact_force_history[:, 1:], previous[:, :-1]
                    )
            torch.cuda.synchronize()
            torch.testing.assert_close(views[0].joint_acceleration, views[1].joint_acceleration)
            torch.testing.assert_close(
                views[0].contact_force_history, views[1].contact_force_history
            )
            oracle = _contact_oracle(backends[1])
            np.testing.assert_allclose(
                views[1].contact_force_history[:, 0, 0].cpu(), oracle, rtol=2e-5, atol=1e-5
            )
            assert np.max(np.linalg.norm(oracle, axis=-1)) > 1
        keep = views[0].contact_force_history[1].clone()
        if reset_kind == "selected":
            backends[0].set_state_tensor(
                torch.tensor([0], device="cuda"),
                torch.zeros((1, 1), device="cuda"),
                torch.zeros((1, 1), device="cuda"),
            )
        else:
            backends[0].set_state_tensor_mask(
                torch.tensor([True, False], device="cuda"),
                torch.zeros((2, 1), device="cuda"),
                torch.zeros((2, 1), device="cuda"),
            )
        assert not bool(views[0].joint_acceleration[0].any())
        assert not bool(views[0].contact_force_history[0].any())
        torch.testing.assert_close(views[0].contact_force_history[1], keep)
        assert ptrs == [
            (v.joint_acceleration.data_ptr(), v.contact_force_history.data_ptr()) for v in views
        ]
    finally:
        for backend in backends:
            backend.cleanup_scene_assets()
