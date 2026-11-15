"""Real-CUDA MJWarp acceptance for selective composed body-state views."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
mujoco = pytest.importorskip("mujoco")
warp = pytest.importorskip("warp")
pytest.importorskip("torch")

import torch

from unisim import MjwarpBackend, create_backend
from unisim.dr.types import ModelSourceDescriptor
from unisim.entities import EntityInitialState, SceneEntitySpec
from unisim.scene import SceneCfg


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    warp.init()
    if not bool(warp.get_device().is_cuda):
        pytest.fail("MJWarp tracked-body acceptance requires an active CUDA Warp device")


def _source(tmp_path: Path, name: str) -> ModelSourceDescriptor:
    path = tmp_path / f"{name}.xml"
    path.write_text(
        '<mujoco><option gravity="0 0 0" integrator="Euler"/>'
        '<worldbody><body name="base"><freejoint name="root"/>'
        '<inertial pos=".03 .01 .02" mass="1" diaginertia=".1 .2 .3"/>'
        '<geom name="base_geom" size=".05" contype="0" conaffinity="0"/>'
        '<body name="hand" pos="0 0 .3"><joint name="hinge" axis="0 1 0"/>'
        '<geom name="hand_geom" size=".03" mass=".2" contype="0" conaffinity="0"/>'
        "</body></body></worldbody>"
        '<actuator><general name="drive" joint="hinge" dyntype="filter" dynprm=".1"/>'
        "</actuator></mujoco>",
        encoding="utf-8",
    )
    return ModelSourceDescriptor(str(path))


def _scene(tmp_path: Path) -> SceneCfg:
    source = _source(tmp_path, "hand")
    fragment = tmp_path / "explicit-tracking.xml"
    fragment.write_text(
        "<mujoco><sensor>"
        '<framepos name="ref_pos_left_hand" objtype="body" objname="left/hand"/>'
        '<framequat name="ref_quat_left_hand" objtype="body" objname="left/hand"/>'
        '<framelinvel name="ref_linvel_left_hand" objtype="body" objname="left/hand"/>'
        '<frameangvel name="ref_angvel_left_hand" objtype="body" objname="left/hand"/>'
        '<framepos name="ref_pos_right_base" objtype="body" objname="right/base"/>'
        '<framequat name="ref_quat_right_base" objtype="body" objname="right/base"/>'
        '<framelinvel name="ref_linvel_right_base" objtype="body" objname="right/base"/>'
        '<frameangvel name="ref_angvel_right_base" objtype="body" objname="right/base"/>'
        "</sensor></mujoco>",
        encoding="utf-8",
    )
    return SceneCfg(
        entity_assets=(
            SceneEntitySpec(
                "left",
                source,
                initial_state=EntityInitialState(position=(0.0, 0.0, 0.0)),
            ),
            SceneEntitySpec(
                "right",
                source,
                initial_state=EntityInitialState(position=(1.0, 0.0, 0.0)),
            ),
        ),
        fragment_files=(str(fragment),),
    )


def test_selective_composed_tracking_matches_explicit_fragments(tmp_path: Path) -> None:
    selected = ("right/base", "left/hand")
    backend = create_backend(
        "mjwarp",
        _scene(tmp_path),
        num_envs=2,
        sim_dt=0.002,
        body_state_required=True,
        tracked_body_names=selected,
        mjwarp_nconmax=32,
        mjwarp_njmax=64,
    )
    try:
        assert backend._tracked_body_names == selected
        injected = {
            sensor_name
            for sensor_name in backend._tracked_sensor_names
            if "left/base" in sensor_name or "right/hand" in sensor_name
        }
        assert injected == set()
        assert backend.get_body_ids(selected).tolist() == [3, 2]

        single = backend.get_sensor_view("track_pos_w_left/hand")
        assert single.shape == (2, 3)
        initial = backend.get_tracked_body_views(device="cuda")
        for field, prefix in (
            ("pos_w", "ref_pos"),
            ("quat_w", "ref_quat"),
            ("lin_vel_w", "ref_linvel"),
            ("ang_vel_w", "ref_angvel"),
        ):
            for slot, body_name in enumerate(selected):
                if body_name == "right/base":
                    # The public fragment spelling is BODY (COM); the aggregate
                    # tracking contract is XBODY (link origin). CPU FK below
                    # covers that distinction for this offset-COM root.
                    continue
                entity, local = body_name.split("/", 1)
                explicit = backend.get_sensor_view(f"{prefix}_{entity}_{local}")
                torch.testing.assert_close(
                    getattr(initial, field)[:, slot], explicit, rtol=2e-5, atol=2e-6
                )

        ctrl = torch.zeros((2, backend.num_actuators), dtype=torch.float32, device="cuda")
        ctrl[:] = 0.4
        backend.step_tensor(ctrl, nsteps=8)

        auto = backend.get_tracked_body_views(device="cuda")
        state = backend.get_state_views(("qpos", "qvel"))
        data = mujoco.MjData(backend._cpu_model)
        data.qpos[:] = state["qpos"][0].detach().cpu().numpy()
        data.qvel[:] = state["qvel"][0].detach().cpu().numpy()
        mujoco.mj_forward(backend._cpu_model, data)
        for slot, body_name in enumerate(selected):
            body_id = backend.get_body_ids((body_name,))[0]
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                backend._cpu_model,
                data,
                mujoco.mjtObj.mjOBJ_XBODY,
                int(body_id),
                velocity,
                0,
            )
            np.testing.assert_allclose(
                auto.pos_w[:, slot].detach().cpu().numpy(),
                np.broadcast_to(data.xpos[body_id], (2, 3)),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                auto.quat_w[:, slot].detach().cpu().numpy(),
                np.broadcast_to(data.xquat[body_id], (2, 4)),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                auto.lin_vel_w[:, slot].detach().cpu().numpy(),
                np.broadcast_to(velocity[3:], (2, 3)),
                rtol=2e-5,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                auto.ang_vel_w[:, slot].detach().cpu().numpy(),
                np.broadcast_to(velocity[:3], (2, 3)),
                rtol=2e-5,
                atol=2e-6,
            )

        reordered = backend.get_tracked_body_views(("left/hand", "right/base"))
        assert reordered.body_names == ("left/hand", "right/base")
        torch.testing.assert_close(
            reordered.pos_w[:, 0], auto.pos_w[:, 1], rtol=2e-5, atol=2e-6
        )
        with pytest.raises(ValueError, match="missing from the tracked namespace"):
            backend.get_tracked_body_views(("left/base",))
    finally:
        backend.close()


def test_composed_tracking_rejects_duplicate_and_unknown_selection(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    with pytest.raises(ValueError, match="tracked_body_names must be unique"):
        MjwarpBackend(
            scene,
            num_envs=2,
            sim_dt=0.002,
            add_body_sensors=True,
            tracked_body_names=("left/hand", "left/hand"),
        )
    with pytest.raises(ValueError, match="missing from the model"):
        MjwarpBackend(
            scene,
            num_envs=2,
            sim_dt=0.002,
            add_body_sensors=True,
            tracked_body_names=("left/missing",),
        )


def test_selective_tracking_requires_sensor_injection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires add_body_sensors=True"):
        MjwarpBackend(
            _scene(tmp_path),
            num_envs=2,
            sim_dt=0.002,
            tracked_body_names=("left/hand",),
        )


def test_reordered_tracked_body_views_do_not_sync_after_first_read(tmp_path: Path) -> None:
    """A repeated reordered read reuses its device column index (no host upload)."""
    backend = create_backend(
        "mjwarp",
        _scene(tmp_path),
        num_envs=2,
        sim_dt=0.002,
        body_state_required=True,
        tracked_body_names=("right/base", "left/hand"),
        mjwarp_nconmax=32,
        mjwarp_njmax=64,
    )
    try:
        order = ("left/hand", "right/base")
        first = backend.get_tracked_body_views(order, device="cuda")
        expected = first.pos_w.clone()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            again = backend.get_tracked_body_views(order, device="cuda")
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.testing.assert_close(again.pos_w, expected, rtol=0, atol=0)
        full = backend.get_tracked_body_views(device="cuda")
        torch.testing.assert_close(again.pos_w[:, 0], full.pos_w[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(again.pos_w[:, 1], full.pos_w[:, 0], rtol=0, atol=0)
    finally:
        backend.close()
