"""Runtime tests for the CUDA ``mjwarp`` backend pre-step control contract."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import warp

from unisim import MjwarpBackend
from unisim.dr.types import ResetRandomizationPayload
from unisim.scene import SceneCfg

MODEL = """<mujoco model='unisim-test-mjwarp'>
  <option timestep='0.01'/>
  <worldbody><body name='base'><joint name='slide' type='slide' axis='1 0 0'/>
    <geom type='box' size='0.05 0.05 0.05'/></body></worldbody>
  <actuator><motor joint='slide' ctrlrange='-10 10'/></actuator>
</mujoco>"""


def _make_backend(
    tmp_path: Path,
    model_name: str = "model.xml",
    xml: str = MODEL,
    *,
    base_name: str | None = None,
) -> MjwarpBackend:
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.skip("mjwarp runtime tests require an active CUDA Warp device")
    model_path = tmp_path / model_name
    model_path.write_text(xml)
    return MjwarpBackend(
        SceneCfg(model_file=str(model_path)), num_envs=2, sim_dt=0.01, base_name=base_name
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
