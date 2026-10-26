"""Persistent multi-body tracked views on real Newton CUDA backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.utils.rotation import np_quat_apply_batched

from .test_contract import _MODEL
from .test_tensor_runtime import _cuda_backend

_OFFSET_INERTIA_MODEL = _MODEL.replace(
    '<joint name="root" type="free"/>',
    '<joint name="root" type="free"/>'
    '<inertial pos=".01 .02 .03" mass="1" diaginertia=".01 .01 .01"/>',
).replace(
    '<joint name="hinge" type="hinge" axis="0 1 0"/>',
    '<joint name="hinge" type="hinge" axis="0 1 0"/>'
    '<inertial pos=".02 -.01 .04" mass=".2" diaginertia=".001 .001 .001"/>',
)


def _assert_matches_track_sensors(backend, views) -> None:
    for index, body_name in enumerate(views.body_names):
        for field, prefix in (
            ("pos_w", "track_pos_w_"),
            ("quat_w", "track_quat_w_"),
            ("lin_vel_w", "track_linvel_w_"),
            ("ang_vel_w", "track_angvel_w_"),
        ):
            np.testing.assert_allclose(
                getattr(views, field)[:, index].cpu().numpy(),
                backend.get_sensor_view(prefix + body_name).cpu().numpy(),
                rtol=2e-5,
                atol=2e-5,
            )


def test_real_cuda_whole_tracked_views_order_semantics_and_live_reset(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    backend = _cuda_backend(tmp_path, _OFFSET_INERTIA_MODEL)
    try:
        assert backend.get_tensor_capabilities().tracked_body_views
        views = backend.get_tracked_body_views(("arm", "base"), device=backend._device)
        assert views.body_names == ("arm", "base")
        for field in ("pos_w", "lin_vel_w", "ang_vel_w"):
            value = getattr(views, field)
            assert value.shape == (2, 2, 3)
            assert value.dtype == torch.float32 and value.is_cuda
        assert views.quat_w.shape == (2, 2, 4)
        _assert_matches_track_sensors(backend, views)
        assert backend.get_tracked_body_views(("arm", "base")) is views

        all_views = backend.get_tracked_body_views()
        assert all_views.body_names == ("base", "arm")
        np.testing.assert_allclose(
            all_views.pos_w[:, 0].cpu().numpy(), views.pos_w[:, 1].cpu().numpy()
        )
        with pytest.raises(ValueError, match="unknown bodies"):
            backend.get_tracked_body_views(("missing",))
        with pytest.raises(ValueError, match="unique"):
            backend.get_tracked_body_views(("base", "base"))
        with pytest.raises(TypeError, match="sequence of strings"):
            backend.get_tracked_body_views("base")  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="non-empty"):
            backend.get_tracked_body_views(())

        state = backend.get_state_views()
        qpos = state["qpos"].clone()
        qvel = state["qvel"].clone()
        qpos[0, :7] = torch.tensor(
            [0.2, -0.1, 0.6, 0.9238795, 0.0, 0.38268343, 0.0], device=backend._device
        )
        qpos[1, :7] = torch.tensor(
            [-0.3, 0.4, 0.8, 0.98078528, 0.19509032, 0.0, 0.0], device=backend._device
        )
        qvel[0, :6] = torch.tensor([0.1, 0.2, 0.3, 0.2, -0.1, 0.3], device=backend._device)
        qvel[1, :6] = torch.tensor([-0.2, 0.1, -0.3, -0.1, 0.4, 0.2], device=backend._device)
        rows = torch.tensor([0, 1], dtype=torch.int64, device=backend._device)
        backend.set_state_tensor(rows, qpos, qvel)

        # The retained aggregate is authoritative immediately after reset,
        # without a fresh getter call to trigger lazy publication.
        expected_qpos = qpos.cpu().numpy()
        expected_qvel = qvel.cpu().numpy()
        np.testing.assert_allclose(views.pos_w[:, 1].cpu().numpy(), expected_qpos[:, :3])
        np.testing.assert_allclose(views.quat_w[:, 1].cpu().numpy(), expected_qpos[:, 3:7])
        np.testing.assert_allclose(views.lin_vel_w[:, 1].cpu().numpy(), expected_qvel[:, :3])
        np.testing.assert_allclose(
            views.ang_vel_w[:, 1].cpu().numpy(),
            np_quat_apply_batched(expected_qpos[:, 3:7], expected_qvel[:, 3:6]),
            rtol=2e-5,
            atol=2e-5,
        )
        _assert_matches_track_sensors(backend, views)

        backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device=backend._device))
        _assert_matches_track_sensors(backend, views)
        backend._host_step(np.zeros((2, 1), dtype=np.float32))
        _assert_matches_track_sensors(backend, views)
    finally:
        backend.close()


def test_real_cuda_portable_tracked_views_qualified_names_and_entity_reset(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from unisim.entities import EntityStatePatch, SceneResetRequest

    from .test_multi_entity_feasibility import _backend, _scene

    backend = _backend(_scene(tmp_path, assignment=(1, 1)))
    try:
        backend.model
        views = backend.get_tracked_body_views(
            ("robot/link", "object/base", "robot/base"), device=backend._device
        )
        assert views.body_names == ("robot/link", "object/base", "robot/base")
        assert views.pos_w.shape == views.lin_vel_w.shape == views.ang_vel_w.shape == (2, 3, 3)
        assert views.quat_w.shape == (2, 3, 4)
        _assert_matches_track_sensors(backend, views)
        before = views.pos_w.cpu().numpy().copy()

        pose = np.array([[0.3, -0.2, 0.7, 0.9238795, 0.0, 0.38268343, 0.0]], dtype=np.float32)
        velocity = np.array([[0.2, -0.1, 0.4, 0.3, -0.2, 0.1]], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (EntityStatePatch("object", root_pose=pose, root_velocity=velocity),),
            )
        )
        np.testing.assert_allclose(views.pos_w[1, 1].cpu().numpy(), pose[0, :3])
        np.testing.assert_allclose(views.quat_w[1, 1].cpu().numpy(), pose[0, 3:7])
        np.testing.assert_allclose(views.lin_vel_w[1, 1].cpu().numpy(), velocity[0, :3])
        np.testing.assert_allclose(
            views.ang_vel_w[1, 1].cpu().numpy(), velocity[0, 3:6], atol=1e-7
        )
        np.testing.assert_allclose(views.pos_w[0].cpu().numpy(), before[0])
        _assert_matches_track_sensors(backend, views)

        backend.step_tensor(torch.zeros((2, 1), dtype=torch.float32, device=backend._device))
        _assert_matches_track_sensors(backend, views)
    finally:
        backend.close()
