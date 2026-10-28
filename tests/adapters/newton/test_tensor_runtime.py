"""CUDA-resident Newton tensor lifecycle contract tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unisim.backend.newton.backend import NewtonBackend
from unisim.backend.newton.dependencies import (
    NewtonDependencyError,
    load_newton_dependencies,
)
from unisim.scene import SceneCfg
from unisim.support import _assert_tensor_platform_capabilities

from .test_contract import _MODEL

_SENSOR_MODEL = (
    _MODEL.replace(
        '<geom name="base_geom" type="sphere" size="0.08" mass="1"/>',
        '<geom name="base_geom" type="sphere" size="0.08" mass="1"/>'
        '<site name="imu_in_pelvis" pos="0.01 0.02 0.03"/>',
    )
    .replace(
        '<geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>',
        '<geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" mass="0.2"/>'
        '<site name="imu_in_torso" pos="-0.01 0.02 0.04"/>',
    )
    .replace(
        "<actuator>",
        "<sensor>"
        '<velocimeter site="imu_in_pelvis" name="pelvis_local_linvel"/>'
        '<gyro site="imu_in_torso" name="torso_gyro"/>'
        '<gyro site="imu_in_pelvis" name="pelvis_gyro"/>'
        '<framezaxis name="torso_upvector" objtype="site" objname="imu_in_torso"/>'
        "</sensor><actuator>",
    )
)


_REORDERED_ACTUATOR_MODEL = """
<mujoco model="newton-reordered-actuator-reset">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="base_geom" type="sphere" size=".08" mass="1"/>
      <body name="selected_link" pos=".1 0 0">
        <joint name="selected" type="hinge" axis="0 1 0"/>
        <geom name="selected_geom" type="capsule" fromto="0 0 0 .1 0 0" size=".03" mass=".2"/>
      </body>
      <body name="other_link" pos="-.1 0 0">
        <joint name="other" type="hinge" axis="0 1 0"/>
        <geom name="other_geom" type="capsule" fromto="0 0 0 .1 0 0" size=".03" mass=".2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="other_motor" joint="other"/>
    <position name="selected_pos" joint="selected" kp="20"/>
    <velocity name="selected_vel" joint="selected" kv="2"/>
  </actuator>
</mujoco>
"""


def _portable_fake(entity_count: int) -> NewtonBackend:
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._portable_mode = True
    backend._device = "cuda:0"
    backend._entity_layout = None
    backend._entity_runtimes = {f"entity_{index}": object() for index in range(entity_count)}
    return backend


def _cuda_backend(tmp_path: Path, xml: str = _MODEL) -> NewtonBackend:
    try:
        deps = load_newton_dependencies()
    except NewtonDependencyError as exc:
        pytest.skip(str(exc))
    deps.warp.init()
    device = deps.warp.get_device()
    if not bool(device.is_cuda):
        pytest.skip("Newton tensor lifecycle requires a CUDA Warp device")
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "newton.xml"
    model_file.write_text(xml, encoding="utf-8")
    backend = NewtonBackend(
        SceneCfg(model_file=str(model_file)),
        num_envs=2,
        sim_dt=0.005,
        device=str(device),
        capacity_check_steps=1,
    )
    backend.materialize()
    return backend


@pytest.mark.parametrize(
    ("qpos_indices", "qvel_indices", "expected_q", "expected_v"),
    (
        ([0, 1, 2], [0, 1], True, True),
        ([0, 2], [0, 1], False, True),
        ([0, 1, 2], [0, 0], True, False),
        ([0, 1, 1], [0, 1], False, True),
    ),
    ids=["complete", "missing-q", "duplicate-v", "duplicate-q"],
)
def test_portable_projection_ownership_requires_unique_complete_columns(
    qpos_indices: list[int],
    qvel_indices: list[int],
    expected_q: bool,
    expected_v: bool,
) -> None:
    pytest.importorskip("torch")
    backend = cast(NewtonBackend, object.__new__(NewtonBackend))
    backend._device = "cpu"
    backend._metadata = SimpleNamespace(nq=3, nv=2)
    backend._tensor_portable_indices = None
    backend._tensor_portable_q_complete = False
    backend._tensor_portable_v_complete = False
    backend._entity_runtimes = {
        "entity": SimpleNamespace(view_body_indices=np.array([0], dtype=np.int64))
    }
    backend._entity_layout = SimpleNamespace(
        entities=(
            SimpleNamespace(
                name="entity",
                qpos_indices=qpos_indices,
                qvel_indices=qvel_indices,
                body_ids=[1],
                joints=(),
                root_joints=(),
                has_public_root_state=False,
            ),
        )
    )

    layout = backend._portable_tensor_layout()

    assert set(layout) == {"entity"}
    assert backend._tensor_portable_q_complete is expected_q
    assert backend._tensor_portable_v_complete is expected_v


def test_portable_projection_zeroes_only_unowned_state() -> None:
    torch = pytest.importorskip("torch")
    backend = cast(NewtonBackend, object.__new__(NewtonBackend))
    backend._device = "cpu"
    backend._metadata = SimpleNamespace(nq=2, nv=1)
    backend._entity_layout = SimpleNamespace(entities=())
    backend._entity_runtimes = {}
    backend._tensor_portable_indices = None
    backend._tensor_refresh_body_state = lambda: None
    backend._tensor_qpos = torch.full((2, 2), -7.0, dtype=torch.float32)
    backend._tensor_qvel = torch.full((2, 1), -9.0, dtype=torch.float32)

    backend._tensor_refresh_portable_state()

    assert torch.count_nonzero(backend._tensor_qpos).item() == 0
    assert torch.count_nonzero(backend._tensor_qvel).item() == 0

    backend._tensor_portable_indices = {}
    backend._tensor_portable_q_complete = True
    backend._tensor_portable_v_complete = True
    backend._tensor_qpos.fill_(-7.0)
    backend._tensor_qvel.fill_(-9.0)
    backend._tensor_refresh_portable_state()

    torch.testing.assert_close(backend._tensor_qpos, torch.full_like(backend._tensor_qpos, -7.0))
    torch.testing.assert_close(backend._tensor_qvel, torch.full_like(backend._tensor_qvel, -9.0))


def test_newton_declares_partial_device_resident_tensor_lifecycle(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        assert backend.get_tensor_execution().value == "device_resident"
        capabilities = backend.get_tensor_capabilities()
        _assert_tensor_platform_capabilities("newton", capabilities)
        assert capabilities.process_topology.value == "in_process"
        assert capabilities.data_plane.value == "direct"
        assert capabilities.torch_devices == ("cuda",)
        assert capabilities.state_fields == frozenset({"qpos", "qvel", "ctrl"})
        assert capabilities.stepping
        assert capabilities.deferred_contact_publication
        assert capabilities.generalized_force_injection
        assert capabilities.selected_reset
        assert capabilities.sensor_views
        assert capabilities.reset_randomization
        assert capabilities.device_reset_randomization
        from unisim.dr.types import RESET_TERM_KD, RESET_TERM_KP

        assert capabilities.packed_reset
        assert capabilities.packed_reset_fields == frozenset({RESET_TERM_KP, RESET_TERM_KD})
        assert capabilities.mask_selected_reset

        with pytest.raises(NotImplementedError, match="unsupported: 'unused'"):
            backend.get_sensor_view("unused", device=torch.device(backend._device))
    finally:
        backend.close()


def test_newton_ctrl_view_is_current_public_order_and_survives_reset(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _REORDERED_ACTUATOR_MODEL)
    try:
        views = backend.get_state_views()
        assert set(views) == {"qpos", "qvel", "ctrl"}
        ctrl_view = views["ctrl"]
        assert ctrl_view.shape == (2, 3)
        assert ctrl_view.dtype == torch.float32
        assert ctrl_view.is_contiguous()
        assert backend.get_state_views(("ctrl",))["ctrl"] is ctrl_view
        control_pointer = ctrl_view.data_ptr()

        host_control = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        backend._host_step(host_control)
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), host_control)
        assert ctrl_view.data_ptr() == control_pointer

        tensor_control = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], dtype=np.float32)
        backend.step_tensor(torch.as_tensor(tensor_control, device=backend._device))
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), tensor_control)
        assert backend._tensor_control_q_staging is not None
        assert backend._tensor_control_qd_staging is not None
        staging_pointers = (
            backend._tensor_control_q_staging.data_ptr(),
            backend._tensor_control_qd_staging.data_ptr(),
        )
        second_tensor_control = tensor_control * 0.5
        backend.step_tensor(torch.as_tensor(second_tensor_control, device=backend._device))
        assert staging_pointers == (
            backend._tensor_control_q_staging.data_ptr(),
            backend._tensor_control_qd_staging.data_ptr(),
        )
        warp = backend._deps.warp
        target_q = warp.to_torch(backend._control.joint_target_q).reshape(backend.num_envs, -1)
        target_qd = warp.to_torch(backend._control.joint_target_qd).reshape(backend.num_envs, -1)
        expected_q = torch.zeros_like(backend._tensor_qpos)
        expected_qd = torch.zeros_like(backend._tensor_qvel)
        q_rows, q_columns = backend._tensor_control_q_indices
        qd_rows, qd_columns = backend._tensor_control_qd_indices
        second_control_tensor = torch.as_tensor(second_tensor_control, device=backend._device)
        expected_q[:, q_columns] = second_control_tensor[:, q_rows]
        expected_qd[:, qd_columns] = second_control_tensor[:, qd_rows]
        torch.testing.assert_close(target_q, expected_q)
        torch.testing.assert_close(target_qd, expected_qd)

        backend.set_actuator_gain_scale((1, 2), 0.0, env_ids=(0,))
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), second_tensor_control)
        backend.set_actuator_gain_scale((1, 2), 1.0, env_ids=(0,))
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), second_tensor_control)

        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64, device=backend._device),
            views["qpos"][:1].contiguous(),
            views["qvel"][:1].contiguous(),
        )
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), second_tensor_control)

        backend._host_set_state(
            np.array([1], dtype=np.intp),
            backend._qpos_cache[1:2].copy(),
            backend._qvel_cache[1:2].copy(),
        )
        np.testing.assert_array_equal(ctrl_view.cpu().numpy(), second_tensor_control)
        assert ctrl_view.data_ptr() == control_pointer
    finally:
        backend.close()


def test_newton_dr_planes_split_and_host_payload_fails_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from unisim.dr.types import TensorResetRandomizationPayload

    backend = _cuda_backend(tmp_path)
    try:
        capabilities = backend.get_dr_capabilities()
        assert capabilities.supported_reset_terms == frozenset()
        assert capabilities.supported_device_reset_terms == frozenset({"kp", "kd"})
        assert capabilities.get_unsupported_reset_terms(["kp"], plane="host") == frozenset({"kp"})
        assert capabilities.get_unsupported_reset_terms(["kp"], plane="device") == frozenset()

        views = backend.get_state_views()
        device = views["qpos"].device
        # A torch.cpu (host-plane) payload on the device reset path fails
        # closed naming the offending term.
        payload = TensorResetRandomizationPayload(
            kp=torch.ones((1, backend.num_actuators), dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="kp"):
            backend.set_state_tensor(
                torch.tensor([1], dtype=torch.int64, device=device),
                views["qpos"][1:2].contiguous(),
                views["qvel"][1:2].contiguous(),
                payload,
            )
    finally:
        backend.close()


def test_newton_portable_multi_entity_declares_tensor_lifecycle() -> None:
    backend = _portable_fake(2)

    assert backend.get_tensor_execution().value == "device_resident"
    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution.value == "device_resident"
    assert capabilities.state_views
    assert capabilities.stepping
    assert capabilities.generalized_force_injection
    assert capabilities.selected_reset
    assert capabilities.mask_selected_reset
    assert capabilities.reset_randomization
    assert capabilities.device_reset_randomization
    assert capabilities.packed_reset
    assert capabilities.packed_reset_fields == frozenset({"kp", "kd"})


def test_newton_portable_single_entity_declares_narrow_tensor_lifecycle() -> None:
    backend = _portable_fake(1)

    capabilities = backend.get_tensor_capabilities()
    assert capabilities.execution.value == "device_resident"
    assert capabilities.state_views
    assert capabilities.stepping
    # One articulation covers the whole model, so the whole-model tensor paths
    # serve single-entity portable scenes unchanged.
    assert capabilities.generalized_force_injection
    assert capabilities.selected_reset
    assert capabilities.mask_selected_reset
    assert capabilities.reset_randomization
    assert capabilities.device_reset_randomization
    assert capabilities.packed_reset
    assert capabilities.packed_reset_fields == frozenset({"kp", "kd"})


def test_newton_tensor_sensor_routing_is_device_only_without_sdk() -> None:
    torch = pytest.importorskip("torch")
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._portable_mode = False
    backend._device = "cpu"
    backend._entity_layout = None
    backend._entity_runtimes = {}
    backend._body_names = ("base", "arm")
    backend._body_ids = {"base": 0, "arm": 1}
    backend._metadata = SimpleNamespace(
        sensor_plans=(
            SimpleNamespace(
                name="pelvis_local_linvel",
                kind="velocimeter",
                dim=3,
                body_id=1,
                site_pos=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                site_quat=np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
            ),
            SimpleNamespace(
                name="torso_gyro",
                kind="gyro",
                dim=3,
                body_id=2,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            SimpleNamespace(
                name="pelvis_gyro",
                kind="gyro",
                dim=3,
                body_id=1,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            SimpleNamespace(
                name="torso_upvector",
                kind="framezaxis",
                dim=3,
                body_id=2,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32),
            ),
            SimpleNamespace(
                name="frame_upvector",
                kind="framezaxis",
                dim=3,
                body_id=2,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            SimpleNamespace(
                name="left_foot_pos",
                kind="framepos",
                dim=3,
                body_id=2,
                site_pos=np.array((0.25, 0.0, 0.0), dtype=np.float32),
                site_quat=np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            ),
            SimpleNamespace(
                name="left_foot_quat",
                kind="framequat",
                dim=4,
                body_id=2,
                site_pos=np.array((0.25, 0.0, 0.0), dtype=np.float32),
                site_quat=np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            ),
            SimpleNamespace(
                name="accelerometer",
                kind="accelerometer",
                dim=3,
                body_id=2,
                site_pos=np.zeros(3, dtype=np.float32),
                site_quat=np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            ),
        )
    )
    backend._tensor_body_state_stale = False
    backend._tensor_body_pos = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_pos[:, 1] = torch.tensor((9.0, 8.0, 7.0))
    backend._tensor_body_quat = torch.zeros((2, 2, 4), dtype=torch.float32)
    backend._tensor_body_quat[..., 0] = 1.0
    original_arm_quat = torch.tensor((1.0, 0.0, 0.0, 0.0))
    rotated_arm_quat = torch.tensor((0.5, 0.5, 0.5, 0.5))
    backend._tensor_body_quat[:, 1] = original_arm_quat
    backend._tensor_body_lin_vel = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_lin_vel[:, 0] = torch.tensor((1.0, 2.0, 3.0))
    backend._tensor_body_lin_vel[:, 1] = torch.tensor((4.0, 5.0, 6.0))
    backend._tensor_body_ang_vel = torch.zeros((2, 2, 3), dtype=torch.float32)
    backend._tensor_body_ang_vel[:, 0] = torch.tensor((0.0, 0.0, 1.0))
    backend._tensor_body_ang_vel[:, 1] = torch.tensor((0.1, 0.2, 0.3))
    backend._tensor_sensor_views = {}
    backend._tensor_sensor_constants = {}
    backend._tensor_device = lambda requested=None: torch.device("cpu")
    backend._tensor_ensure_state = lambda: None

    linvel = backend.get_sensor_view("pelvis_local_linvel")
    gyro = backend.get_sensor_view("torso_gyro")
    assert linvel.shape == (2, 3)
    assert gyro.shape == (2, 3)
    torch.testing.assert_close(
        linvel, torch.tensor(((3.0, -1.0, 3.0), (3.0, -1.0, 3.0)), dtype=torch.float32)
    )
    torch.testing.assert_close(
        gyro, torch.tensor(((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)), dtype=torch.float32)
    )
    upvector = backend.get_sensor_view("torso_upvector")
    assert upvector.shape == (2, 3)
    torch.testing.assert_close(upvector, torch.tensor(((0.0, 0.0, 1.0),) * 2, dtype=torch.float32))
    backend._tensor_sensor_views.clear()
    backend._tensor_body_quat[:, 1] = rotated_arm_quat
    torch.testing.assert_close(
        backend.get_sensor_view("torso_upvector"), torch.tensor(((1.0, 0.0, 0.0),) * 2)
    )
    backend._tensor_body_quat[:, 1] = original_arm_quat
    backend._tensor_sensor_views.clear()
    assert torch.equal(backend.get_sensor_view("track_pos_w_arm"), backend._tensor_body_pos[:, 1])
    assert torch.equal(
        backend.get_sensor_view("track_quat_w_base"), backend._tensor_body_quat[:, 0]
    )
    assert torch.equal(
        backend.get_sensor_view("track_linvel_w_base"), backend._tensor_body_lin_vel[:, 0]
    )
    assert torch.equal(
        backend.get_sensor_view("track_angvel_w_arm"), backend._tensor_body_ang_vel[:, 1]
    )
    frame_pos = backend.get_sensor_view("left_foot_pos")
    frame_quat = backend.get_sensor_view("left_foot_quat")
    assert frame_pos.shape == (2, 3)
    assert frame_quat.shape == (2, 4)
    torch.testing.assert_close(
        frame_pos,
        backend._tensor_body_pos[:, 1] + torch.tensor(((0.25, 0.0, 0.0),), dtype=torch.float32),
    )
    torch.testing.assert_close(frame_quat, backend._tensor_body_quat[:, 1])
    with pytest.raises(NotImplementedError, match="unsupported kind 'accelerometer'"):
        backend.get_sensor_view("accelerometer")
    backend._metadata.sensor_plans = tuple(
        plan for plan in backend._metadata.sensor_plans if plan.name != "torso_upvector"
    )
    backend._tensor_sensor_views.clear()
    with pytest.raises(NotImplementedError, match="unsupported: 'torso_upvector'"):
        backend.get_sensor_view("torso_upvector")
    with pytest.raises(KeyError, match="unknown Newton tensor tracked body"):
        backend.get_sensor_view("track_pos_w_missing")


def test_newton_tensor_selected_reset_preserves_unselected_updated_rows(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    reference = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        ctrl = torch.zeros(
            (backend.num_envs, backend.num_actuators),
            dtype=torch.float32,
            device=views["qpos"].device,
        )
        backend.step_tensor(ctrl)
        reference._host_step(ctrl.cpu().numpy())

        legacy_views = reference.get_state_views()
        legacy_qpos = legacy_views["qpos"].clone()
        legacy_qvel = legacy_views["qvel"].clone()
        legacy_qpos[0, 2] += 0.2
        legacy_qvel[0, 0] = 0.3
        reference._host_set_state(
            np.array([0]),
            legacy_qpos[:1].cpu().numpy(),
            legacy_qvel[:1].cpu().numpy(),
        )
        backend._host_set_state(
            np.array([0]),
            legacy_qpos[:1].cpu().numpy(),
            legacy_qvel[:1].cpu().numpy(),
        )

        refreshed = backend.get_state_views()
        torch.testing.assert_close(refreshed["qpos"][0], legacy_qpos[0])
        reset_qpos = refreshed["qpos"].clone()
        reset_qvel = refreshed["qvel"].clone()
        reset_qpos[1, 2] += 0.1
        reset_qvel[1, 0] = 0.25
        backend.set_state_tensor(
            torch.tensor([1], device=reset_qpos.device),
            reset_qpos[1:2],
            reset_qvel[1:2],
        )
        reference._host_set_state(
            np.array([1]),
            reset_qpos[1:2].cpu().numpy(),
            reset_qvel[1:2].cpu().numpy(),
        )

        result = backend.get_state_views()
        expected = reference.get_state_views()
        torch.testing.assert_close(result["qpos"], expected["qpos"])
        torch.testing.assert_close(result["qvel"], expected["qvel"])
        torch.testing.assert_close(result["qpos"][0], legacy_qpos[0])
        torch.testing.assert_close(result["qvel"][0], legacy_qvel[0])
        scratch = (
            backend._tensor_reset_full_qpos,
            backend._tensor_reset_full_qvel,
            backend._tensor_reset_raw_qpos,
            backend._tensor_reset_raw_qvel,
            backend._tensor_reset_mask,
            backend._tensor_reset_solver_mask,
        )
        pointers = [tensor.data_ptr() for tensor in scratch]
        second_qpos = result["qpos"].clone()
        second_qvel = result["qvel"].clone()
        second_qpos[0, 2] += 0.3
        second_qvel[0, 0] = -0.4
        backend.set_state_tensor(
            torch.tensor([0], device=second_qpos.device),
            second_qpos[:1].contiguous(),
            second_qvel[:1].contiguous(),
        )
        assert pointers == [tensor.data_ptr() for tensor in scratch]
        assert backend._tensor_reset_mask.tolist() == [False, False]
        assert backend._tensor_reset_solver_mask.tolist() == [False, False, False]
        torch.testing.assert_close(backend.get_state_views()["qpos"][1], result["qpos"][1])
        torch.testing.assert_close(backend.get_state_views()["qpos"][0, 2], second_qpos[0, 2])
    finally:
        backend.close()
        reference.close()


def test_newton_tensor_root_angular_velocity_matches_body_frame_host_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        qpos = views["qpos"].clone()
        qvel = views["qvel"].clone()
        qpos[0, 3:7] = torch.tensor([np.cos(0.15), 0.0, 0.0, np.sin(0.15)], device=qpos.device)
        qvel[0, :6] = torch.tensor([0.01, -0.02, 0.03, 0.1, 0.05, -0.2], device=qvel.device)
        backend._host_set_state(np.array([0]), qpos[:1].cpu().numpy(), qvel[:1].cpu().numpy())

        backend._refresh_host_cache()
        np.testing.assert_allclose(backend._qvel_cache[0], qvel[0].cpu().numpy(), rtol=0, atol=1e-6)
        refreshed = backend.get_state_views()
        torch.testing.assert_close(refreshed["qvel"][0], qvel[0], rtol=0, atol=1e-6)
    finally:
        backend.close()


def test_newton_tensor_hot_path_has_bounded_scalar_synchronization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        ctrl = torch.zeros(
            (backend.num_envs, backend.num_actuators),
            dtype=torch.float32,
            device=views["qpos"].device,
        )

        scalar_reads: list[str] = []
        original_item = torch.Tensor.item
        original_tolist = torch.Tensor.tolist

        def counted_item(self: Any) -> Any:
            scalar_reads.append("item")
            # Torch 2.14 leaves a DeviceContext on the Torch-function stack
            # after ``torch.set_default_device`` cycles. Genesis' cold session
            # restores the default but cannot clear that SDK-owned mode, so it
            # re-enters this monkeypatch through its dispatcher. Disable the
            # dispatcher while delegating so this counts the semantic call once.
            with torch.DisableTorchFunction():
                return original_item(self)

        def counted_tolist(self: Any) -> Any:
            scalar_reads.append("tolist")
            return original_tolist(self)

        monkeypatch.setattr(torch.Tensor, "item", counted_item)
        monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
        backend.step_tensor(ctrl)
        assert scalar_reads == []

        rows = torch.tensor([1], dtype=torch.int64, device=views["qpos"].device)
        qpos = views["qpos"][1:2].clone()
        qvel = views["qvel"][1:2].clone()
        backend.set_state_tensor(rows, qpos, qvel)
        assert scalar_reads == ["item"]
    finally:
        backend.close()


def test_newton_actuator_mapping_rejects_ambiguous_rows() -> None:
    backend = cast(Any, NewtonBackend.__new__(NewtonBackend))
    backend._public_to_native_actuator_rows = None
    backend._metadata = SimpleNamespace(
        actuator_names=("motor", "position", "velocity"),
        actuator_target_kinds=("direct", "position", "velocity"),
        actuator_target_qvel_adrs=np.zeros(3, dtype=np.int32),
    )
    backend._solver = SimpleNamespace(
        mjc_actuator_ctrl_source=SimpleNamespace(numpy=lambda: np.array([0, 0, 1], dtype=np.int32)),
        mjc_actuator_to_newton_idx=SimpleNamespace(
            numpy=lambda: np.array([0, 0, 0], dtype=np.int32)
        ),
    )

    with pytest.raises(RuntimeError, match="public position actuator row 1.*exactly one"):
        backend._native_actuator_rows()


def test_newton_direct_reset_randomization_maps_reordered_native_actuators(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.dr.types import TensorResetRandomizationPayload

    backend = _cuda_backend(tmp_path, _REORDERED_ACTUATOR_MODEL)
    try:
        assert backend.get_actuator_names() == ("other_motor", "selected_pos", "selected_vel")
        source = np.asarray(backend._solver.mjc_actuator_ctrl_source.numpy(), dtype=np.int32)
        mapping = np.asarray(backend._solver.mjc_actuator_to_newton_idx.numpy(), dtype=np.int32)
        native_rows: list[int] = []
        for public_row, kind in enumerate(backend._metadata.actuator_target_kinds):
            if kind == "direct":
                matches = np.flatnonzero((source != 0) & (mapping == public_row))
            else:
                dof = int(backend._metadata.actuator_target_qvel_adrs[public_row])
                encoded = dof if kind == "position" else -(dof + 2)
                matches = np.flatnonzero((source == 0) & (mapping == encoded))
            assert matches.size == 1
            native_rows.append(int(matches[0]))
        assert native_rows != list(range(backend.num_actuators))

        warp = backend._deps.warp
        gainprm = warp.to_torch(backend._solver.mjw_model.actuator_gainprm)
        biasprm = warp.to_torch(backend._solver.mjw_model.actuator_biasprm)
        baseline_gain = gainprm.clone()
        baseline_bias = biasprm.clone()
        views = backend.get_state_views()
        device = views["qpos"].device
        kp = torch.tensor([[11.0, 22.0, 33.0]], dtype=torch.float32, device=device)
        kd = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device=device),
            views["qpos"][1:2].clone(),
            views["qvel"][1:2].clone(),
            TensorResetRandomizationPayload(kp=kp, kd=kd),
        )

        expected_gain = baseline_gain.clone()
        expected_bias = baseline_bias.clone()
        expected_gain[1, native_rows, 0] = kp[0]
        expected_bias[1, native_rows, 1] = -kp[0]
        expected_bias[1, native_rows, 2] = -kd[0]
        torch.testing.assert_close(gainprm, expected_gain)
        torch.testing.assert_close(biasprm, expected_bias)
        torch.testing.assert_close(gainprm[0], baseline_gain[0])
        torch.testing.assert_close(biasprm[0], baseline_bias[0])
    finally:
        backend.close()


def test_newton_packed_mask_reset_preserves_false_rows_and_gains(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.dr.types import RESET_TERM_KD, RESET_TERM_KP

    backend = _cuda_backend(tmp_path)
    try:
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.packed_reset
        transaction = backend.bind_packed_reset()
        default_qpos = torch.asarray(backend.get_default_qpos(), device=backend._device)
        default_qvel = torch.asarray(backend.get_init_qvel(), device=backend._device)
        transaction.set_state_default(default_qpos, default_qvel)
        mask = torch.asarray((False, True), dtype=torch.bool, device=backend._device)
        before = backend.get_state_views()
        inertia_before = backend.get_reset_term_default("body_inertia").copy()
        tracked = backend.get_tracked_body_views()
        transaction.begin_state_mask(mask)
        staged_qpos, staged_qvel = transaction.state_views()
        staged_qpos[1, 2] += 0.2
        kp_default = (
            torch.asarray(backend.get_reset_term_default(RESET_TERM_KP), device=backend._device)
            .reshape(1, -1)
            .repeat(2, 1)
        )
        kd_default = (
            torch.asarray(backend.get_reset_term_default(RESET_TERM_KD), device=backend._device)
            .reshape(1, -1)
            .repeat(2, 1)
        )
        transaction.write_model_field_mask(
            RESET_TERM_KP,
            mask,
            kp_default * 1.5,
        )
        transaction.write_model_field_mask(
            RESET_TERM_KD,
            mask,
            kd_default * 2.0,
        )
        result = transaction.commit_mask(mask)

        assert result is not None
        assert result["timing"]["packed_reset_total_ms"] >= 0.0
        after = backend.get_state_views()
        torch.testing.assert_close(after["qpos"][0], before["qpos"][0])
        torch.testing.assert_close(after["qvel"][0], before["qvel"][0])
        torch.testing.assert_close(after["qpos"][1], staged_qpos[1])
        torch.testing.assert_close(after["qvel"][1], staged_qvel[1])
        torch.testing.assert_close(after["ctrl"], before["ctrl"])
        np.testing.assert_array_equal(
            backend.get_reset_term_default("body_inertia"), inertia_before
        )
        torch.testing.assert_close(tracked.pos_w, backend.get_tracked_body_views().pos_w)
        torch.testing.assert_close(tracked.pos_w[:, 0], backend.get_sensor_view("track_pos_w_base"))

        warp = backend._deps.warp
        gainprm = warp.to_torch(backend._solver.mjw_model.actuator_gainprm)
        biasprm = warp.to_torch(backend._solver.mjw_model.actuator_biasprm)
        torch.testing.assert_close(gainprm[0, :, 0], kp_default[0])
        torch.testing.assert_close(gainprm[1, :, 0], kp_default[1] * 1.5)
        torch.testing.assert_close(biasprm[0, :, 1], torch.zeros_like(kp_default[0]))
        torch.testing.assert_close(biasprm[1, :, 1], -kp_default[1] * 1.5)
        torch.testing.assert_close(biasprm[0, :, 2], torch.zeros_like(kd_default[0]))
        torch.testing.assert_close(biasprm[1, :, 2], -kd_default[1] * 2.0)
    finally:
        backend.close()


def test_newton_selected_reset_avoids_scalar_index_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views()
        rows = torch.tensor([1], dtype=torch.int64, device=views["qpos"].device)
        scalar_writes: list[object] = []
        original_setitem = torch.Tensor.__setitem__

        def counting_setitem(self: Any, key: Any, value: Any) -> None:
            if isinstance(value, (bool, int, float)):
                scalar_writes.append(value)
            return original_setitem(self, key, value)

        monkeypatch.setattr(torch.Tensor, "__setitem__", counting_setitem)
        backend.set_state_tensor(rows, views["qpos"][1:2], views["qvel"][1:2])

        assert scalar_writes == []
    finally:
        backend.close()


def test_newton_tensor_sensor_views_match_host_and_selected_reset(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _SENSOR_MODEL)
    host = _cuda_backend(tmp_path / "host", _SENSOR_MODEL)
    try:
        device = backend.get_state_views()["qpos"].device
        ctrl = torch.full(
            (backend.num_envs, backend.num_actuators),
            0.2,
            dtype=torch.float32,
            device=device,
        )
        backend.step_tensor(ctrl, nsteps=2)
        host._host_step(ctrl.detach().cpu().numpy(), nsteps=2)

        def assert_sensor_parity() -> None:
            for name in ("pelvis_local_linvel", "torso_gyro", "torso_upvector"):
                actual = backend.get_sensor_view(name, device=device)
                expected = host._host_read_sensor(name)
                np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, atol=2e-5)
            body_ids = host.get_body_ids(("base", "arm"))
            expected = (
                host._host_get_body_pos_w(body_ids),
                host._host_get_body_quat_w(body_ids),
                host._host_get_body_lin_vel_w(body_ids),
                host._host_get_body_ang_vel_w(body_ids),
            )
            for body_name, body_id in zip(("base", "arm"), body_ids.tolist(), strict=True):
                actual = (
                    backend.get_sensor_view(f"track_pos_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_quat_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_linvel_w_{body_name}", device=device),
                    backend.get_sensor_view(f"track_angvel_w_{body_name}", device=device),
                )
                expected_body = tuple(value[:, body_id] for value in expected)
                for index, (tensor_value, host_value) in enumerate(
                    zip(actual, expected_body, strict=True)
                ):
                    np.testing.assert_allclose(
                        tensor_value.detach().cpu().numpy(), host_value, atol=2e-5
                    )
                    del index

        assert_sensor_parity()

        views = backend.get_state_views()
        rows = torch.tensor([1], dtype=torch.int64, device=device)
        reset_qpos = views["qpos"].clone()
        reset_qvel = views["qvel"].clone()
        reset_qpos[1, 2] += 0.15
        reset_qpos[1, 7] = 0.4
        reset_qvel[1, 3] = -0.2
        backend.set_state_tensor(rows, reset_qpos[1:2], reset_qvel[1:2])
        host._host_set_state(
            rows.detach().cpu().numpy(),
            reset_qpos[1:2].detach().cpu().numpy(),
            reset_qvel[1:2].detach().cpu().numpy(),
        )
        assert_sensor_parity()
    finally:
        backend.close()
        host.close()


def test_newton_tensor_step_and_reset_avoid_host_cache_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    host = _cuda_backend(tmp_path)
    try:
        monkeypatch.setattr(
            backend,
            "_refresh_host_cache",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("hidden host cache refresh")),
        )
        views = backend.get_state_views(device=torch.device(backend._device))
        ctrl = torch.full(
            (2, backend.num_actuators),
            0.2,
            dtype=torch.float32,
            device=views["qpos"].device,
        )
        result = backend.step_tensor(ctrl, nsteps=2)
        assert result is not None
        assert result["timing"]["tensor_host_cache_refresh_ms"] == 0.0

        host._host_step(np.full((2, host.num_actuators), 0.2, dtype=np.float32), nsteps=2)
        host_state = host.get_physics_state()
        host_qpos = host_state[:, 1 : 1 + host._metadata.nq]
        host_qvel = host_state[:, 1 + host._metadata.nq :]
        torch.testing.assert_close(views["qpos"], torch.tensor(host_qpos, device=ctrl.device))
        torch.testing.assert_close(views["qvel"], torch.tensor(host_qvel, device=ctrl.device))

        rows = torch.tensor([1], dtype=torch.int64, device=ctrl.device)
        reset_qpos = views["qpos"].clone()
        reset_qvel = views["qvel"].clone()
        reset_qpos[0, 2] += 0.1
        reset_qvel[0, 0] = 0.25
        backend.set_state_tensor(rows, reset_qpos[:1], reset_qvel[:1])
        assert views["qpos"][1, 2].item() == pytest.approx(reset_qpos[0, 2].item(), abs=1e-6)
        assert views["qvel"][1, 0].item() == pytest.approx(0.25, abs=1e-6)
        host._host_set_state(
            rows.cpu().numpy(), reset_qpos[:1].cpu().numpy(), reset_qvel[:1].cpu().numpy()
        )
        host_state = host.get_physics_state()
        torch.testing.assert_close(
            views["qpos"],
            torch.tensor(host_state[:, 1 : 1 + host._metadata.nq], device=ctrl.device),
        )
        torch.testing.assert_close(
            views["qvel"],
            torch.tensor(host_state[:, 1 + host._metadata.nq :], device=ctrl.device),
        )
    finally:
        backend.close()
        host.close()


def test_newton_tensor_physics_snapshot_tracks_current_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        views = backend.get_state_views(device=torch.device(backend._device))
        before = backend.get_physics_state().copy()
        ctrl = torch.full(
            (2, backend.num_actuators),
            0.2,
            dtype=torch.float32,
            device=views["qpos"].device,
        )
        backend.step_tensor(ctrl, nsteps=2)
        after = backend.get_physics_state()
        nq = backend._metadata.nq
        np.testing.assert_allclose(
            after[:, 0], before[:, 0] + 2 * backend._sim_dt, rtol=0.0, atol=1.0e-7
        )
        torch.testing.assert_close(
            views["qpos"], torch.tensor(after[:, 1 : 1 + nq], device=ctrl.device)
        )
        torch.testing.assert_close(
            views["qvel"], torch.tensor(after[:, 1 + nq :], device=ctrl.device)
        )
        assert not np.array_equal(after[:, 1:], before[:, 1:])
    finally:
        backend.close()


def test_newton_tensor_inputs_fail_closed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path)
    try:
        device = torch.device(backend._device)
        views = backend.get_state_views(device=device)
        with pytest.raises(ValueError, match="shape"):
            backend.step_tensor(torch.zeros((1, backend.num_actuators), device=device))
        with pytest.raises(ValueError, match="device"):
            backend.get_state_views(device="cpu")
        with pytest.raises(ValueError, match="unique"):
            backend.set_state_tensor(
                torch.tensor([0, 0], device=device),
                views["qpos"][:2],
                views["qvel"][:2],
            )
    finally:
        backend.close()
