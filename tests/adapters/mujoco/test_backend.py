from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from unisim import MuJoCoBackend, TensorExecution, assert_backend_conformance
from unisim.backend.base import TensorIOSpec
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <sensor><framepos name='base_pos' objtype='body' objname='base'/></sensor>
  <actuator><motor joint='slide' ctrlrange='-1 1'/></actuator>
</mujoco>"""
_TRACKED_SENSOR_ORDER_FALLBACK_PREFIXES = (
    "track_pos_w",
    "track_quat_w",
    "track_linvel_w",
    "track_angvel_w",
)


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


def test_mujoco_packed_host_bridge_plan_matches_legacy_views(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=3,
        sim_dt=0.01,
        base_name="base",
        add_body_sensors=True,
        tracked_body_names=("base",),
    )
    backend.materialize()

    spec = TensorIOSpec(
        state_fields=("qpos", "qvel"),
        sensor_names=("base_pos", "track_pos_w_base", "track_linvel_w_base"),
        device="cpu",
    )
    assert backend.get_tensor_capabilities().packed_host_bridge
    plan = backend.compile_host_bridge_io(spec)
    plan.write_control(torch.full((3, 1), 2.0, dtype=torch.float32))
    result = plan.step(nsteps=2)
    assert result is not None
    assert result["timing"]["tensor_control_packed_d2h_count"] == 1.0

    views = plan.read_state_sensors()
    expected_state = backend.get_state_views(("qpos", "qvel"), device="cpu")
    expected_pos = backend.get_sensor_view("base_pos", device="cpu")
    expected_track_pos = backend.get_sensor_view("track_pos_w_base", device="cpu")
    expected_track_vel = backend.get_sensor_view("track_linvel_w_base", device="cpu")
    torch.testing.assert_close(views["qpos"], expected_state["qpos"])
    torch.testing.assert_close(views["qvel"], expected_state["qvel"])
    torch.testing.assert_close(views["base_pos"], expected_pos)
    torch.testing.assert_close(views["track_pos_w_base"], expected_track_pos)
    torch.testing.assert_close(views["track_linvel_w_base"], expected_track_vel)
    assert plan.transfer_stats["h2d_count"] == 1

    rows = torch.tensor([1, 2], dtype=torch.int64)
    qpos = torch.zeros((2, backend.nq), dtype=torch.float32)
    qvel = torch.zeros((2, backend.nv), dtype=torch.float32)
    qpos[:, 0] = torch.tensor([0.25, 0.5])
    reset_result = plan.apply_reset(rows, qpos, qvel)
    assert reset_result is not None
    assert reset_result["timing"]["tensor_reset_packed_d2h_count"] == 1.0
    assert "tensor_state_packed_h2d_count" not in reset_result["timing"]
    selected_views = plan.read_selected_state_sensors()
    assert "tensor_reset_packed_d2h_count" not in plan.last_timing
    torch.testing.assert_close(selected_views["qpos"][rows], qpos)
    np.testing.assert_allclose(
        selected_views["qpos"][0].numpy(),
        views["qpos"][0].numpy(),
        atol=1e-12,
    )
    assert plan.transfer_stats == {
        "d2h_count": 2,
        "h2d_count": 2,
        "d2h_bytes": 3 * 4 + 2 * (2 + backend.nq + backend.nv) * 4,
        "h2d_bytes": (3 + 2) * (backend.nq + backend.nv + 9) * 4,
        "synchronization_count": 0,
    }

    invalid = torch.tensor([0, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="unique"):
        plan.apply_reset(
            invalid,
            torch.zeros((2, backend.nq), dtype=torch.float32),
            torch.zeros((2, backend.nv), dtype=torch.float32),
        )


def test_mujoco_packed_host_bridge_sensor_order_falls_back_correctly(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model_path = tmp_path / "two-body.xml"
    model_path.write_text(
        "<mujoco><option timestep='0.01' gravity='0 0 0'/>"
        "<worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>"
        "<geom type='box' size='0.05 0.05 0.05'/><body name='arm' pos='0 0 0.1'>"
        "<joint name='hinge' axis='0 0 1'/><geom type='box' size='0.02 0.02 0.02'/>"
        "</body></body></worldbody>"
        "<actuator><motor joint='slide'/><motor joint='hinge'/></actuator></mujoco>"
    )
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(model_path)),
        num_envs=2,
        sim_dt=0.01,
        base_name="base",
        add_body_sensors=True,
        tracked_body_names=("base", "arm"),
    )
    backend.materialize()
    sensor_names = tuple(
        f"{prefix}_{body}"
        for prefix in _TRACKED_SENSOR_ORDER_FALLBACK_PREFIXES
        for body in ("arm", "base")
    )
    plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos", "qvel"),
            sensor_names=sensor_names,
            device="cpu",
        )
    )
    assert plan._tracked_sensor_groups == ()

    plan.write_control(torch.zeros((2, backend.num_actuators), dtype=torch.float32))
    plan.step(nsteps=1)
    views = plan.read_state_sensors()
    for name in sensor_names:
        torch.testing.assert_close(views[name], backend.get_sensor_view(name, device="cpu"))

    rows = torch.tensor([1], dtype=torch.int64)
    qpos = torch.zeros((1, backend.nq), dtype=torch.float32)
    qvel = torch.zeros((1, backend.nv), dtype=torch.float32)
    qpos[:, :] = torch.tensor([[0.2, 0.4]])
    qvel[:, :] = torch.tensor([[0.3, 0.6]])
    plan.apply_reset(rows, qpos, qvel)
    selected_views = plan.read_selected_state_sensors()
    torch.testing.assert_close(selected_views["qpos"][rows], qpos)
    torch.testing.assert_close(selected_views["qvel"][rows], qvel)
    for name in sensor_names:
        expected = backend.get_sensor_data(name)[1]
        torch.testing.assert_close(
            selected_views[name][rows], torch.from_numpy(expected.copy())[None, :]
        )


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


def test_mujoco_host_bridge_tensor_lifecycle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("host-bridge tensor test requires CUDA")
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL)
    backend = MuJoCoBackend(SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01)
    backend.materialize()
    assert backend.tensor_execution().value == "host_bridge"
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution is TensorExecution.HOST_BRIDGE
    assert capabilities.state_views
    assert capabilities.state_fields == {"qpos", "qvel", "ctrl"}
    assert capabilities.sensor_views
    assert capabilities.stepping
    assert capabilities.selected_reset
    assert capabilities.packed_host_bridge
    assert not capabilities.host_pre_step_control
    packed_plan = backend.compile_host_bridge_io(
        TensorIOSpec(
            state_fields=("qpos", "qvel"),
            sensor_names=("base_pos",),
            device="cuda",
        )
    )
    assert packed_plan._host_packet.is_pinned()
    assert packed_plan._selected_host.is_pinned()
    assert packed_plan._host_ctrl.is_pinned()
    assert packed_plan._reset_host.is_pinned()
    assert backend.get_state_views(("qpos",))["qpos"].is_cpu
    cpu_sensor = backend.get_sensor_view("base_pos").clone()
    initial_sensor = cpu_sensor.clone()
    backend.step(np.ones((2, 1), dtype=np.float32), nsteps=2)
    np.testing.assert_array_equal(cpu_sensor.numpy(), initial_sensor.numpy())

    ctrl = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    result = backend.step_tensor(ctrl, nsteps=2)
    assert result is not None
    assert result["timing"]["tensor_control_d2h_ms"] >= 0.0
    state = backend.get_state_views(("qpos", "qvel"), device="cuda")
    assert state["qpos"].is_cuda
    sensor = backend.get_sensor_view("base_pos", device="cuda")
    assert sensor.is_cuda and sensor.shape == (2, 3)
    np.testing.assert_allclose(
        sensor.detach().cpu().numpy(),
        backend.get_sensor_data("base_pos"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        state["qpos"].detach().cpu().numpy(),
        backend.get_state(("qpos",))["qpos"],
        atol=1e-6,
    )
    before_nonfinite = state["qpos"].detach().clone()
    with pytest.raises(ValueError, match="ctrl.*finite"):
        backend.step_tensor(torch.full_like(ctrl, torch.nan), nsteps=1)
    torch.testing.assert_close(
        backend.get_state_views(("qpos",), device="cuda")["qpos"], before_nonfinite
    )

    rows = torch.tensor([1], dtype=torch.int64, device=ctrl.device)
    qpos = torch.zeros((1, backend.nq), dtype=torch.float32, device=ctrl.device)
    qvel = torch.zeros((1, backend.nv), dtype=torch.float32, device=ctrl.device)
    backend.set_state_tensor(rows, qpos, qvel)
    np.testing.assert_allclose(
        backend.get_state_views(("qpos",), device="cuda")["qpos"][1].detach().cpu().numpy(),
        qpos.cpu().numpy()[0],
        atol=1e-6,
    )

    invalid_rows = (
        torch.tensor([2], dtype=torch.int64, device="cuda"),
        torch.tensor([0, 0], dtype=torch.int64, device="cuda"),
    )
    before = backend.get_state_views(("qpos",), device="cuda")["qpos"].clone()
    for rows in invalid_rows:
        invalid_qpos = torch.zeros((rows.numel(), backend.nq), device="cuda")
        invalid_qvel = torch.zeros((rows.numel(), backend.nv), device="cuda")
        with pytest.raises(ValueError, match="env_indices"):
            backend.set_state_tensor(rows, invalid_qpos, invalid_qvel)
    np.testing.assert_allclose(
        backend.get_state_views(("qpos",), device="cuda")["qpos"].detach().cpu().numpy(),
        before.detach().cpu().numpy(),
        atol=1e-12,
    )
    valid_rows = torch.tensor([1], dtype=torch.int64, device="cuda")
    invalid_qpos = torch.zeros((1, backend.nq), dtype=torch.float32, device="cuda")
    invalid_qpos.fill_(torch.nan)
    with pytest.raises(ValueError, match="qpos.*finite"):
        backend.set_state_tensor(valid_rows, invalid_qpos, torch.zeros_like(invalid_qpos))
    np.testing.assert_allclose(
        backend.get_state_views(("qpos",), device="cuda")["qpos"].detach().cpu().numpy(),
        before.detach().cpu().numpy(),
        atol=1e-12,
    )

    backend.set_pre_step_control(lambda owner, control: control)
    try:
        with pytest.raises(NotImplementedError, match="pre-step control"):
            backend.step_tensor(ctrl)
    finally:
        backend.set_pre_step_control(None)
