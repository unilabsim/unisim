"""Portable packed reset composition on real Newton CUDA backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unisim.dr.types import RESET_TERM_KD, RESET_TERM_KP, FixedVariantPlan
from unisim.entities import (
    EntityStatePatch,
    EntityVariantBinding,
    SceneEntitySpec,
    SceneResetRequest,
)
from unisim.scene import SceneCfg

from .test_multi_entity_feasibility import _backend, _object, _scene


def _actuated_variants(tmp_path: Path):
    common = {
        "radius": 0.1,
        "mass": 0.5,
        "com": (0.01, 0.0, 0.0),
        "actuated": True,
    }
    object_a = _object(tmp_path, "object_a", **common)
    object_b = _object(tmp_path, "object_b", **common)
    for source, kp, kv in ((object_a, 10.0, 1.0), (object_b, 20.0, 2.0)):
        path = Path(source.model_file)
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            '<actuator><motor name="drive" joint="drive"/></actuator>',
            '<actuator><position name="drive_pos" joint="drive" '
            f'kp="{kp}"/><velocity name="drive_vel" joint="drive" kv="{kv}"/></actuator>',
        )
        path.write_text(text, encoding="utf-8")
    return object_a, object_b


def test_real_cuda_portable_packed_reset_partial_rows_and_entity_patch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    object_a, object_b = _actuated_variants(tmp_path)
    backend = _backend(_scene(tmp_path, object_a=object_a, object_b=object_b, assignment=(0, 1)))
    try:
        backend.model
        capabilities = backend.get_tensor_capabilities()
        assert capabilities.packed_reset
        assert capabilities.device_reset_randomization
        assert capabilities.packed_reset_fields == frozenset({RESET_TERM_KP, RESET_TERM_KD})
        assert backend.get_dr_capabilities().supported_reset_terms == frozenset({"kp", "kd"})
        np.testing.assert_allclose(
            backend.get_reset_term_default(RESET_TERM_KP),
            [[1.0, 10.0, 1.0], [1.0, 20.0, 2.0]],
        )
        np.testing.assert_allclose(
            backend.get_reset_term_default(RESET_TERM_KD),
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        )
        inertia_before = backend.get_reset_term_default("body_inertia").copy()

        controls = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        backend._host_step(controls)
        before = backend.get_state_views()
        tracked = backend.get_tracked_body_views(("object/base",))
        layout = backend.get_scene_layout()
        object_entity = layout.get_entity("object")
        backend.set_actuator_gain_scale((1, 2), 0.0, env_ids=(0,))
        target_ke = backend._deps.warp.to_torch(backend._model.joint_target_ke)
        target_kd = backend._deps.warp.to_torch(backend._model.joint_target_kd)
        unselected_ke = target_ke[0].clone()
        unselected_kd = target_kd[0].clone()

        transaction = backend.bind_packed_reset(device=backend._device)
        transaction.set_state_default(
            torch.asarray(backend.get_default_qpos(), device=backend._device),
            torch.asarray(backend.get_init_qvel(), device=backend._device),
        )
        mask = torch.tensor([False, True], dtype=torch.bool, device=backend._device)
        transaction.begin_state_mask(mask)
        staged_qpos, staged_qvel = transaction.state_views()
        staged_qpos[1] = before["qpos"][1]
        staged_qvel[1] = 0.0
        staged_qpos[1, object_entity.root_qpos_indices[2]] += 0.2
        staged_qpos[1, object_entity.joints[0].qpos_indices[0]] += 0.1
        transaction.write_model_field_mask(
            RESET_TERM_KP,
            mask,
            torch.tensor([[99.0, 99.0, 99.0], [30.0, 30.0, 30.0]], device=backend._device),
        )
        transaction.write_model_field_mask(
            RESET_TERM_KD,
            mask,
            torch.tensor([[99.0, 99.0, 99.0], [3.0, 3.0, 3.0]], device=backend._device),
        )
        result = transaction.commit_mask(mask)
        assert result["timing"]["packed_reset_total_ms"] >= 0.0

        after = backend.get_state_views()
        torch.testing.assert_close(after["qpos"][0], before["qpos"][0])
        torch.testing.assert_close(after["qvel"][0], before["qvel"][0])
        torch.testing.assert_close(after["qpos"][1], staged_qpos[1])
        torch.testing.assert_close(after["qvel"][1], staged_qvel[1])
        torch.testing.assert_close(after["ctrl"], before["ctrl"])
        native_rows = backend._native_actuator_rows()
        warp = backend._deps.warp
        gainprm = warp.to_torch(backend._solver.mjw_model.actuator_gainprm)
        biasprm = warp.to_torch(backend._solver.mjw_model.actuator_biasprm)
        torch.testing.assert_close(
            gainprm[1, native_rows, 0], torch.full((3,), 30.0, device=backend._device)
        )
        torch.testing.assert_close(
            biasprm[1, native_rows, 1], torch.full((3,), -30.0, device=backend._device)
        )
        torch.testing.assert_close(
            biasprm[1, native_rows, 2], torch.full((3,), -3.0, device=backend._device)
        )
        torch.testing.assert_close(
            gainprm[0, native_rows, 0], torch.tensor([1.0, 0.0, 0.0], device=backend._device)
        )
        np.testing.assert_array_equal(
            backend.get_reset_term_default("body_inertia"), inertia_before
        )
        torch.testing.assert_close(
            tracked.pos_w[:, 0], backend.get_sensor_view("track_pos_w_object/base")
        )
        selected_dof = int(object_entity.joints[0].qvel_indices[0])
        torch.testing.assert_close(
            target_ke[1 * backend._metadata.nv + selected_dof],
            torch.tensor(30.0, device=backend._device),
        )
        torch.testing.assert_close(
            target_kd[1 * backend._metadata.nv + selected_dof],
            torch.tensor(3.0, device=backend._device),
        )
        torch.testing.assert_close(target_ke[0], unselected_ke)
        torch.testing.assert_close(target_kd[0], unselected_kd)
        selected_qpos = int(object_entity.joints[0].qpos_indices[0])
        qpos_before_step = float(after["qpos"][1, selected_qpos])
        backend.step_tensor(after["ctrl"])
        assert abs(float(after["qvel"][1, selected_dof])) > 1e-3
        assert abs(float(after["qpos"][1, selected_qpos]) - qpos_before_step) > 1e-6

        pose = np.array([[0.4, -0.3, 0.9, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        backend.reset_entities(
            SceneResetRequest(
                (1,),
                (
                    EntityStatePatch(
                        "object",
                        root_pose=pose,
                        root_velocity=np.zeros((1, 6), dtype=np.float32),
                        joint_positions=np.array([[0.25]], dtype=np.float32),
                    ),
                ),
            )
        )
        torch.testing.assert_close(
            after["ctrl"],
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.0]], device=backend._device),
        )
        np.testing.assert_allclose(tracked.pos_w[1, 0].cpu().numpy(), pose[0, :3])
        torch.testing.assert_close(
            gainprm[1, native_rows, 0], torch.full((3,), 30.0, device=backend._device)
        )
        torch.testing.assert_close(
            biasprm[1, native_rows, 2], torch.full((3,), -3.0, device=backend._device)
        )
    finally:
        backend.close()


def test_real_cuda_single_entity_variant_com_drives_root_velocity_roundtrip(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    object_a = _object(
        tmp_path, "single_a", radius=0.1, mass=0.5, com=(0.01, 0.0, 0.0), actuated=True
    )
    object_b = _object(
        tmp_path,
        "single_b",
        radius=0.1,
        mass=0.5,
        com=(0.23, 0.07, -0.03),
        actuated=True,
    )
    scene = SceneCfg(
        entity_assets=(SceneEntitySpec("object", object_a),),
        entity_variant=EntityVariantBinding(
            "object", FixedVariantPlan(np.array([0, 1], dtype=np.int32), (object_a, object_b))
        ),
    )
    backend = _backend(scene)
    try:
        backend.model
        views = backend.get_state_views()
        tracked = backend.get_tracked_body_views(("object/base",))
        root_row = backend.get_scene_layout().get_entity("object").body_ids[0] - 1
        body_ipos = backend.get_body_ipos(np.arange(2))[:, root_row]
        assert not np.allclose(body_ipos[0], body_ipos[1])

        qpos = views["qpos"].clone()
        half = np.sqrt(0.5)
        qpos[:, 3:7] = torch.tensor(
            [half, 0.0, half, 0.0], dtype=torch.float32, device=backend._device
        )
        qvel = torch.zeros_like(views["qvel"])
        qvel[:, :6] = torch.tensor(
            [1.5, -2.0, 0.5, 0.2, -0.3, 0.4],
            dtype=torch.float32,
            device=backend._device,
        )
        backend.set_state_tensor(
            torch.tensor([0, 1], dtype=torch.int64, device=backend._device), qpos, qvel
        )
        backend._tensor_refresh_state()
        torch.testing.assert_close(views["qvel"], qvel, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(
            tracked.lin_vel_w[:, 0], qvel[:, :3], rtol=1e-5, atol=2e-6
        )

        qvel[1, :6] *= -0.5
        backend.set_state_tensor(
            torch.tensor([1], dtype=torch.int64, device=backend._device),
            qpos[1:2].contiguous(),
            qvel[1:2].contiguous(),
        )
        backend._tensor_refresh_state()
        torch.testing.assert_close(views["qvel"][1], qvel[1], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(
            tracked.lin_vel_w[1, 0], qvel[1, :3], rtol=1e-5, atol=2e-6
        )
    finally:
        backend.close()
