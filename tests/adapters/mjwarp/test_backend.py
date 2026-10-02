"""Runtime tests for the CUDA ``mjwarp`` backend pre-step control contract."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp

from unisim import MjwarpBackend, TensorExecution
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <sensor><framepos name='base_pos' objtype='body' objname='base'/></sensor>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""

NONZERO_CTRL_KEYFRAME_MODEL = MODEL.replace(
    "</mujoco>", "<keyframe><key name='stand' ctrl='0.25'/></keyframe></mujoco>"
)


def _make_backend(
    tmp_path: Path,
    model_name: str = "model.xml",
    xml: str = MODEL,
    *,
    base_name: str | None = None,
    default_keyframe_name: str | None = None,
    add_body_sensors: bool = False,
    num_envs: int = 2,
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / model_name
    model_path.write_text(xml)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_path), default_keyframe_name=default_keyframe_name),
        num_envs=num_envs,
        sim_dt=0.01,
        base_name=base_name,
        add_body_sensors=add_body_sensors,
    )


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


def test_mjwarp_pre_step_control_replays_captured_step_graph(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires an mjwarp CUDA step graph; graphs disabled: {reason}")

    original_module = backend._mujoco_warp
    observed_qpos: list[np.ndarray] = []

    class RejectEagerStep:
        def __getattr__(self, name: str):
            return getattr(original_module, name)

        def step(self, device_model, device_data) -> None:
            raise AssertionError("pre-step callback path must replay the captured step graph")

    def controller(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        observed_qpos.append(owner.get_dof_pos().copy())
        return ctrl

    backend._mujoco_warp = RejectEagerStep()
    backend.set_pre_step_control(controller)
    try:
        backend.step(np.ones((2, 1), dtype=np.float32), nsteps=2)
    finally:
        backend.set_pre_step_control(None)
        backend._mujoco_warp = original_module

    assert len(observed_qpos) == 2
    assert not np.allclose(observed_qpos[0], observed_qpos[-1])


def test_mjwarp_pre_step_control_falls_back_to_eager_steps(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires an mjwarp CUDA step graph to force eager fallback; {reason}")

    original_module = backend._mujoco_warp
    eager_calls = 0

    class CountingStep:
        def __getattr__(self, name: str):
            return getattr(original_module, name)

        def step(self, device_model, device_data) -> None:
            nonlocal eager_calls
            eager_calls += 1
            original_module.step(device_model, device_data)

    backend._cuda_graph_enabled = False
    backend._mujoco_warp = CountingStep()
    backend.set_pre_step_control(lambda owner, ctrl: ctrl)
    try:
        backend.step(np.ones((2, 1), dtype=np.float32), nsteps=3)
    finally:
        backend.set_pre_step_control(None)
        backend._mujoco_warp = original_module
        backend._cuda_graph_enabled = True

    assert eager_calls == 3


def test_mjwarp_pre_step_control_graph_matches_eager_short_horizon(tmp_path: Path) -> None:
    graph_backend = _make_backend(tmp_path, "graph.xml")
    eager_backend = _make_backend(tmp_path, "eager.xml")
    if not graph_backend._cuda_graph_enabled:
        reason = graph_backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires an mjwarp CUDA step graph; graphs disabled: {reason}")

    eager_backend._cuda_graph_enabled = False
    rows = np.arange(2, dtype=np.int32)
    qpos = np.array([[0.1], [-0.1]], dtype=np.float32)
    qvel = np.array([[0.2], [-0.2]], dtype=np.float32)
    graph_backend.set_state(rows, qpos, qvel)
    eager_backend.set_state(rows, qpos, qvel)

    graph_observed: list[np.ndarray] = []
    eager_observed: list[np.ndarray] = []

    def graph_controller(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        graph_observed.append(owner.get_dof_pos().copy())
        return 0.2 - owner.get_dof_pos()

    def eager_controller(owner: MjwarpBackend, ctrl: np.ndarray) -> np.ndarray:
        eager_observed.append(owner.get_dof_pos().copy())
        return 0.2 - owner.get_dof_pos()

    graph_backend.set_pre_step_control(graph_controller)
    eager_backend.set_pre_step_control(eager_controller)
    ctrl = np.zeros((2, 1), dtype=np.float32)
    for _ in range(2):
        graph_backend.step(ctrl, nsteps=4)
        eager_backend.step(ctrl, nsteps=4)

    np.testing.assert_array_equal(graph_observed[0], eager_observed[0])
    np.testing.assert_allclose(graph_observed, eager_observed, atol=2e-6)
    np.testing.assert_allclose(
        graph_backend.get_state(("qpos", "qvel"))["qpos"],
        eager_backend.get_state(("qpos", "qvel"))["qpos"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        graph_backend.get_state(("qpos", "qvel"))["qvel"],
        eager_backend.get_state(("qpos", "qvel"))["qvel"],
        atol=2e-5,
    )


def test_mjwarp_pre_step_control_validates_return_shape(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    ctrl = np.zeros((2, 1), dtype=np.float32)
    backend.set_pre_step_control(lambda owner, c: np.zeros((2, 2), dtype=c.dtype))
    with pytest.raises(ValueError, match="pre-step control must return shape"):
        backend.step(ctrl, nsteps=1)
    backend.set_pre_step_control(None)
    backend.step(ctrl, nsteps=1)


@pytest.mark.parametrize("with_object_free_joint", [False, True])
def test_mjwarp_state_snapshot_matches_set_state_layout(
    tmp_path: Path, with_object_free_joint: bool
) -> None:
    extra_body = (
        "<body name='object' pos='1 0 1'><freejoint name='object_free'/>"
        "<geom type='sphere' size='0.1' mass='1' contype='0' conaffinity='0'/></body>"
        if with_object_free_joint
        else ""
    )
    xml = (
        "<mujoco><option timestep='0.01' gravity='0 0 0'/>"
        "<worldbody><body name='base'><body name='arm' pos='0 0 1'>"
        "<joint name='hinge' axis='0 0 1'/>"
        "<geom type='box' size='0.1 0.1 0.1' mass='1' contype='0' conaffinity='0'/>"
        f"</body></body>{extra_body}</worldbody>"
        "<actuator><motor joint='hinge' ctrlrange='-1 1'/></actuator></mujoco>"
    )
    backend = _make_backend(tmp_path, xml=xml)
    ids = np.arange(2, dtype=np.int32)
    qpos = np.zeros((2, backend.get_default_qpos().size), dtype=np.float32)
    qvel = np.zeros((2, backend.get_init_qvel().size), dtype=np.float32)
    qpos[:, 0] = [0.2, 0.4]
    qvel[:, 0] = [0.3, 0.6]
    object_pose = np.tile(
        np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (2, 1),
    )
    if with_object_free_joint:
        qpos[:, 1:8] = object_pose

    backend.set_state(ids, qpos, qvel)
    state = backend.get_state(("qpos", "qvel"))
    assert state["qpos"].shape == qpos.shape
    assert state["qvel"].shape == qvel.shape
    np.testing.assert_allclose(state["qpos"], qpos, atol=1e-6)
    np.testing.assert_allclose(state["qvel"], qvel, atol=1e-6)
    if with_object_free_joint:
        layout = backend.get_root_state_layout("object")
        np.testing.assert_allclose(state["qpos"][:, layout.qpos_indices], object_pose, atol=1e-6)

    detached = state["qpos"].copy()
    state["qpos"][:] += 10.0
    np.testing.assert_array_equal(backend.get_state(("qpos",))["qpos"], detached)

    backend.step(np.zeros((2, 1), dtype=np.float32), nsteps=1)
    after_step = backend.get_state(("qpos", "qvel"))
    assert after_step["qpos"].shape == qpos.shape
    assert after_step["qvel"].shape == qvel.shape
    backend.set_state(ids, after_step["qpos"], after_step["qvel"])


MOCAP_MODEL = """<mujoco model='unisim-test-mjwarp-mocap'>
  <option timestep='0.01'/>
  <worldbody>
    <body name='base'>
      <joint name='slide' type='slide' axis='1 0 0'/>
      <geom type='box' size='0.05 0.05 0.05'/>
    </body>
    <body name='palm' mocap='true' pos='0 0 0.5'>
      <geom type='box' size='0.02 0.02 0.02'/>
    </body>
  </worldbody>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""


def test_mjwarp_snapshot_carries_mocap_state(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path, "mocap.xml", xml=MOCAP_MODEL)

    snapshot = backend.get_physics_state()

    # Layout: [time, qpos, qvel, mocap_pos(nmocap*3), mocap_quat(nmocap*4)].
    assert snapshot.shape == (2, 1 + 1 + 1 + 7)
    np.testing.assert_allclose(snapshot[:, 3:6], [[0.0, 0.0, 0.5]] * 2, atol=1e-6)
    np.testing.assert_allclose(snapshot[:, 6:10], [[1.0, 0.0, 0.0, 0.0]] * 2, atol=1e-6)

    layout = backend.get_physics_state_layout()
    assert (layout.nq, layout.nv, layout.nmocap) == (1, 1, 1)
    assert layout.state_width == snapshot.shape[1]
    parts = layout.split_state(snapshot)
    assert parts.mocap_pos is not None and parts.mocap_quat is not None
    np.testing.assert_allclose(parts.mocap_pos[:, 0, :], snapshot[:, 3:6], atol=1e-6)
    np.testing.assert_allclose(parts.mocap_quat[:, 0, :], snapshot[:, 6:10], atol=1e-6)
    assert backend.get_play_capabilities().supports_mocap_playback
    mocap_pos, mocap_quat = backend.get_playback_mocap_state(1)
    np.testing.assert_allclose(mocap_pos, [[0.0, 0.0, 0.5]], atol=1e-6)
    np.testing.assert_allclose(mocap_quat, [[1.0, 0.0, 0.0, 0.0]], atol=1e-6)

    binding = backend.bind_mocap_pose("palm")
    poses = np.array(
        [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    binding.write(np.arange(2, dtype=np.int32), poses)

    snapshot = backend.get_physics_state()
    np.testing.assert_allclose(snapshot[:, 3:6], poses[:, :3], atol=1e-6)
    np.testing.assert_allclose(snapshot[:, 6:10], poses[:, 3:], atol=1e-6)
    mocap_pos, _ = backend.get_playback_mocap_state(0)
    np.testing.assert_allclose(mocap_pos, poses[:1, :3], atol=1e-6)


def test_mjwarp_body_ipos_default_stability_and_per_env_current_query(
    tmp_path: Path,
) -> None:
    backend = _make_backend(tmp_path, base_name="base")
    nbody = int(backend._cpu_model.nbody)
    base_id = int(backend.get_body_ids(["base"])[0])
    canonical = backend.get_body_ipos()
    assert canonical.shape == (nbody, 3)
    default_before = backend.get_reset_term_default("body_ipos")
    assert default_before.shape == (nbody, 3)

    rows = np.array([0, 1], dtype=np.int32)
    qpos = np.tile(backend.get_default_qpos(), (2, 1))
    qvel = np.tile(backend.get_init_qvel(), (2, 1))

    ipos = np.tile(canonical, (2, 1, 1))
    ipos[:, base_id, 0] += np.array([0.1, -0.2], dtype=np.float32)
    backend.set_state(rows, qpos, qvel, randomization=ResetRandomizationPayload(body_ipos=ipos))

    # Default-facing queries never drift with reset randomization (issue #87).
    np.testing.assert_array_equal(backend.get_body_ipos(), canonical)
    np.testing.assert_array_equal(backend.get_reset_term_default("body_ipos"), default_before)
    current = backend.get_body_ipos(env_ids=rows)
    assert current.shape == (2, nbody, 3)
    np.testing.assert_allclose(
        current[:, base_id, 0], canonical[base_id, 0] + [0.1, -0.2], rtol=1e-6
    )

    # A partial reset of env 1 (body_ipos composed with base_com_offset)
    # leaves env 0 untouched.
    backend.set_state(
        np.array([1], dtype=np.int32),
        qpos[[1]],
        qvel[[1]],
        randomization=ResetRandomizationPayload(
            body_ipos=np.tile(canonical, (1, 1, 1)),
            base_com_offset=np.array([[0.3, 0.0, 0.0]], dtype=np.float32),
        ),
    )
    after = backend.get_body_ipos(env_ids=rows)
    np.testing.assert_allclose(after[0, base_id, 0], canonical[base_id, 0] + 0.1, rtol=1e-6)
    np.testing.assert_allclose(after[1, base_id, 0], canonical[base_id, 0] + 0.3, rtol=1e-6)
    np.testing.assert_array_equal(backend.get_reset_term_default("body_ipos"), default_before)

    # base_com_offset alone composes on top of the immutable defaults.
    backend.set_state(
        np.array([0], dtype=np.int32),
        qpos[[0]],
        qvel[[0]],
        randomization=ResetRandomizationPayload(
            base_com_offset=np.array([[0.0, 0.05, 0.0]], dtype=np.float32)
        ),
    )
    final = backend.get_body_ipos(env_ids=rows)
    np.testing.assert_allclose(final[0, base_id, 0], canonical[base_id, 0], rtol=1e-6)
    np.testing.assert_allclose(final[0, base_id, 1], canonical[base_id, 1] + 0.05, rtol=1e-6)

    with pytest.raises(ValueError, match="env_ids"):
        backend.get_body_ipos(env_ids=[backend.num_envs])


def test_mjwarp_device_tensor_lifecycle_matches_host_path(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    host = _make_backend(
        tmp_path,
        "host.xml",
        xml=NONZERO_CTRL_KEYFRAME_MODEL,
        default_keyframe_name="stand",
    )
    device = _make_backend(
        tmp_path,
        "device.xml",
        xml=NONZERO_CTRL_KEYFRAME_MODEL,
        default_keyframe_name="stand",
    )
    assert device.tensor_execution().value == "device_resident"
    capabilities = device.get_tensor_capabilities()
    assert capabilities.execution is TensorExecution.DEVICE_RESIDENT
    assert capabilities.state_views
    assert capabilities.state_fields == {"qpos", "qvel", "ctrl", "sensordata", "time"}
    assert capabilities.sensor_views
    assert capabilities.stepping
    assert capabilities.selected_reset
    assert capabilities.reset_randomization
    assert not capabilities.fixed_variants
    assert not capabilities.host_pre_step_control

    ctrl = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError, match="nsteps must be a positive integer"):
        device.step_tensor(ctrl, nsteps=1.5)
    with pytest.raises(TypeError, match="nsteps must be a positive integer"):
        host.step(ctrl.detach().cpu().numpy(), nsteps=1.5)
    host.step(ctrl.detach().cpu().numpy(), nsteps=2)
    result = device.step_tensor(ctrl, nsteps=2)
    assert result is not None
    assert result["timing"]["host_cache_refresh_ms"] == 0.0
    views = device.get_state_views(("qpos", "qvel", "ctrl"))
    assert views["qpos"].is_cuda and views["qpos"].dtype == torch.float32
    sensor_view = device.get_sensor_view("base_pos")
    assert sensor_view.is_cuda and sensor_view.shape == (2, 3)
    assert device.get_state_views(("qpos",))["qpos"].data_ptr() == views["qpos"].data_ptr()
    assert device.get_sensor_view("base_pos").data_ptr() == sensor_view.data_ptr()
    np.testing.assert_allclose(
        sensor_view.detach().cpu().numpy(),
        host.get_sensor_data("base_pos"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        views["qpos"].detach().cpu().numpy(),
        host.get_state(("qpos",))["qpos"],
        atol=1e-6,
    )

    before_reject = views["qpos"].detach().clone()
    invalid_ctrl = torch.full((2, 1), torch.nan, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="ctrl.*finite"):
        device.step_tensor(invalid_ctrl, nsteps=1)
    torch.testing.assert_close(views["qpos"], before_reject)
    np.testing.assert_allclose(
        views["qvel"].detach().cpu().numpy(),
        host.get_state(("qvel",))["qvel"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        device.get_state(("qpos",))["qpos"],
        host.get_state(("qpos",))["qpos"],
        atol=1e-6,
    )

    rows = torch.tensor([1], dtype=torch.int64, device=ctrl.device)
    qpos = torch.full((1, 1), 0.5, dtype=torch.float32, device=ctrl.device)
    qvel = torch.zeros_like(qpos)
    host_rows = rows.detach().cpu().numpy()
    host.set_state(host_rows, qpos.cpu().numpy(), qvel.cpu().numpy())
    reset_result = device.set_state_tensor(rows, qpos, qvel)
    assert reset_result is not None
    assert reset_result["timing"]["set_state_tensor_host_cache_refresh_ms"] == 0.0
    np.testing.assert_allclose(
        device.get_state_views(("qpos",))["qpos"].detach().cpu().numpy(),
        host.get_state(("qpos",))["qpos"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        device.get_state(("qpos",))["qpos"],
        host.get_state(("qpos",))["qpos"],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        device.get_state(("ctrl",))["ctrl"],
        host.get_state(("ctrl",))["ctrl"],
    )


def test_mjwarp_tensor_step_then_host_reset_preserves_unselected_rows(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    device = _make_backend(tmp_path)
    ctrl = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    device.step_tensor(ctrl, nsteps=2)
    before_host_reset = device.get_state_views(("qpos",))["qpos"].detach().cpu().numpy().copy()

    device.set_state(
        np.array([0], dtype=np.int64),
        np.array([[0.25]], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    )
    after_host_reset = device.get_state(("qpos",))["qpos"]
    np.testing.assert_allclose(after_host_reset[0], 0.25, atol=1e-7)
    np.testing.assert_allclose(after_host_reset[1], before_host_reset[1], atol=1e-6)


def test_mjwarp_tensor_sensor_after_reset_avoids_host_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _make_backend(tmp_path, add_body_sensors=True)
    backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device="cuda"), nsteps=2)
    rows = torch.tensor([1], dtype=torch.int64, device="cuda")
    qpos = torch.full((1, 1), 0.5, dtype=torch.float32, device="cuda")
    backend.set_state_tensor(rows, qpos, torch.zeros_like(qpos))

    def fail_device_refresh() -> None:
        raise AssertionError("invalid sensor negotiation must not refresh tracked state")

    monkeypatch.setattr(backend, "_refresh_tracked_body_state_device_only", fail_device_refresh)
    with pytest.raises(ValueError, match="Sensor 'missing' not found"):
        backend.get_sensor_view("missing")
    monkeypatch.undo()

    def fail_host_publication() -> None:
        raise AssertionError("device sensor views must not publish host caches")

    monkeypatch.setattr(backend, "_refresh_tracked_body_state_device", fail_host_publication)
    tensor_sensor = backend.get_sensor_view("track_pos_w_base")
    assert backend._tracked_body_state_dirty is False
    host_sensor = backend.get_sensor_data("track_pos_w_base")
    assert tensor_sensor.is_cuda
    np.testing.assert_allclose(tensor_sensor.detach().cpu().numpy(), host_sensor, atol=1e-6)


def test_mjwarp_tracked_body_views_are_ordered_device_blocks(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    two_body_model = MODEL.replace(
        "</worldbody>",
        "<body name='arm'><joint name='hinge' type='hinge' axis='0 1 0'/>"
        "<geom type='sphere' size='0.02'/></body></worldbody>",
    )
    backend = _make_backend(
        tmp_path,
        "tracked_bodies.xml",
        xml=two_body_model,
        add_body_sensors=True,
    )

    views = backend.get_tracked_body_views()

    assert views.body_names == ("base", "arm")
    assert views.pos_w.shape == (2, 2, 3)
    assert views.quat_w.shape == (2, 2, 4)
    assert views.lin_vel_w.shape == (2, 2, 3)
    assert views.ang_vel_w.shape == (2, 2, 3)
    for value in (views.pos_w, views.quat_w, views.lin_vel_w, views.ang_vel_w):
        assert value.is_cuda
    expected_pos = backend.get_sensor_view("track_pos_w_base")
    torch.testing.assert_close(views.pos_w[:, 0], expected_pos)

    ordered = backend.get_tracked_body_views(("arm", "base"))
    assert ordered.body_names == ("arm", "base")
    torch.testing.assert_close(ordered.pos_w[:, 0], views.pos_w[:, 1])
    torch.testing.assert_close(ordered.pos_w[:, 1], views.pos_w[:, 0])

    with pytest.raises(ValueError, match="missing from the tracked namespace"):
        backend.get_tracked_body_views(("missing",))
    with pytest.raises(ValueError, match="unique"):
        backend.get_tracked_body_views(("base", "base"))
    with pytest.raises(TypeError, match="sequence of strings"):
        backend.get_tracked_body_views("base")  # pyright: ignore[reportArgumentType]


def test_mjwarp_tracked_body_views_after_selected_reset_are_authoritative(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _make_backend(tmp_path, add_body_sensors=True)
    backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device="cuda"), nsteps=1)
    rows = torch.tensor([1], dtype=torch.int64, device="cuda")
    qpos = torch.full((1, 1), 0.5, dtype=torch.float32, device="cuda")
    backend.set_state_tensor(rows, qpos, torch.zeros_like(qpos))

    views = backend.get_tracked_body_views()
    expected = backend.get_sensor_view("track_pos_w_base")

    torch.testing.assert_close(views.pos_w[:, 0], expected)
    assert views.pos_w.device == expected.device


def test_mjwarp_selected_tensor_reset_uses_scratch_publication_when_bounded(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _make_backend(
        tmp_path,
        "selected_scratch.xml",
        add_body_sensors=True,
        num_envs=1024,
    )
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires mjwarp CUDA reset graphs; graphs disabled: {reason}")
    assert backend._reset_scratch_capacity >= 1

    backend.step_tensor(torch.zeros((1024, 1), dtype=torch.float32, device="cuda"), nsteps=1)
    rows = torch.tensor([3, 17, 999], dtype=torch.int64, device="cuda")
    qpos = torch.linspace(0.2, 0.8, rows.numel(), dtype=torch.float32, device="cuda").reshape(-1, 1)
    qvel = torch.zeros_like(qpos)
    backend.set_state_tensor(rows, qpos, qvel)
    assert backend._tracked_body_state_dirty is False

    selected = backend.get_tracked_body_views()
    expected_qpos = backend.get_state_views(("qpos",))["qpos"]
    torch.testing.assert_close(expected_qpos[rows], qpos)
    assert selected.pos_w.shape == (1024, 1, 3)

    # Force the full-width refresh and prove the selected-row publication is
    # numerically authoritative.  Untouched rows remain live stable views.
    backend._tracked_body_state_dirty = True
    backend._refresh_tracked_body_state_device_only()
    full = backend.get_tracked_body_views()
    torch.testing.assert_close(selected.pos_w, full.pos_w, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(selected.quat_w, full.quat_w, rtol=2e-6, atol=2e-6)


def test_mjwarp_selected_tensor_reset_scratch_overflow_uses_full_refresh(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _make_backend(
        tmp_path,
        "selected_overflow.xml",
        add_body_sensors=True,
        num_envs=1024,
    )
    if not backend._cuda_graph_enabled:
        reason = backend._cuda_graph_disable_reason or "unknown reason"
        pytest.skip(f"test requires mjwarp CUDA reset graphs; graphs disabled: {reason}")
    capacity = backend._reset_scratch_capacity
    assert capacity >= 1
    backend.step_tensor(torch.zeros((1024, 1), dtype=torch.float32, device="cuda"), nsteps=1)

    rows = torch.arange(capacity + 1, dtype=torch.int64, device="cuda")
    qpos = torch.linspace(0.1, 0.9, rows.numel(), dtype=torch.float32, device="cuda").reshape(-1, 1)
    backend.set_state_tensor(rows, qpos, torch.zeros_like(qpos))
    assert backend._tracked_body_state_dirty is True

    views = backend.get_tracked_body_views()
    assert views.pos_w.shape == (1024, 1, 3)
    assert backend._tracked_body_state_dirty is False
    expected_qpos = backend.get_state_views(("qpos",))["qpos"]
    torch.testing.assert_close(expected_qpos[rows], qpos)


def test_mjwarp_tensor_reset_validates_rows(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    device = _make_backend(tmp_path)
    device.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device="cuda"))
    before = device.get_state_views(("qpos",))["qpos"].detach().cpu().numpy().copy()

    invalid_rows = (
        torch.tensor([2], dtype=torch.int64, device="cuda"),
        torch.tensor([0, 0], dtype=torch.int64, device="cuda"),
    )
    for rows in invalid_rows:
        qpos = torch.zeros((rows.numel(), 1), dtype=torch.float32, device="cuda")
        qvel = torch.zeros_like(qpos)
        with pytest.raises(ValueError, match="env_indices"):
            device.set_state_tensor(rows, qpos, qvel)
    np.testing.assert_array_equal(device.get_state(("qpos",))["qpos"], before)


def test_mjwarp_tensor_reset_rejects_pending_wrench_and_mocap(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pushed = _make_backend(tmp_path, "pushed.xml", base_name="base")
    pushed.push_robots(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    rows = torch.tensor([0], dtype=torch.int64, device="cuda")
    qpos = torch.zeros((1, 1), dtype=torch.float32, device="cuda")
    qvel = torch.zeros_like(qpos)
    with pytest.raises(NotImplementedError, match="pending interval body wrenches"):
        pushed.set_state_tensor(rows, qpos, qvel)

    mocap_model = MODEL.replace(
        "</worldbody>",
        "<body name='marker' mocap='true'><geom type='sphere' size='0.01'/></body></worldbody>",
    )
    mocap = _make_backend(tmp_path, "mocap.xml", xml=mocap_model)
    with pytest.raises(NotImplementedError, match="mocap"):
        mocap.set_state_tensor(rows, qpos, qvel)


def test_mjwarp_host_step_tensor_sensor_view_refreshes_tracking(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    backend = _make_backend(tmp_path, add_body_sensors=True)
    backend.step(np.ones((2, 1), dtype=np.float32), nsteps=2)
    tensor_sensor = backend.get_sensor_view("track_pos_w_base")
    body_id = backend.get_body_ids(("base",))[0]
    host_sensor = backend.get_body_pos_w(np.array([body_id], dtype=np.intp))[:, 0, :]
    assert tensor_sensor.shape == (2, 3)
    np.testing.assert_allclose(
        tensor_sensor.detach().cpu().numpy(),
        host_sensor,
        atol=1e-6,
    )
