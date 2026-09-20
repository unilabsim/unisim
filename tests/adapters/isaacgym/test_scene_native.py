"""Opt-in bounded GPU evidence for the isolated multi-actor worker.

Run with UNISIM_TEST_ISAACGYM_SCENE=1 in a machine with the dedicated SDK.
This is not a claim of routine vendor-runtime CI availability.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from tests.adapters.isaacgym.scene_client import SceneClient
from tests.adapters.isaacgym.scene_fixture import scene_payload
from unisim.backend.subprocess_ipc import protocol
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout

pytestmark = pytest.mark.skipif(
    os.environ.get("UNISIM_TEST_ISAACGYM_SCENE") != "1",
    reason="requires explicit opt-in and the native IsaacGym GPU runtime",
)


def test_native_identity_controls_com_frames_and_consecutive_partial_resets(tmp_path) -> None:
    payload = scene_payload(tmp_path / "assets", env_spacing=1.5)
    client = SceneClient(payload, tmp_path / "worker.log")
    try:
        actual = {row["name"]: row for row in client.meta["scene_entities_actual"]}
        expected_origins = np.array(
            [
                [index % 3, index // 3, 0.0]
                for index in range(payload["num_envs"])
            ],
            dtype=float,
        ) * 1.5
        assert client.meta["env_spacing"] == 1.5
        np.testing.assert_allclose(
            np.asarray(client.meta["env_origins"]), expected_origins, rtol=0, atol=1e-6
        )
        assert actual["object"]["assignment"] == [1, 1, 0, 1, 0]
        assert actual["object"]["actor_ids"] == [1, 5, 9, 13, 17]
        np.testing.assert_allclose(np.asarray(actual["object"]["body_mass"])[:, 0], [3, 3, 1, 3, 1])
        assert actual["object"]["body_sphere_radii"] == [[[], []]] * 5
        assert actual["robot"]["body_sphere_radii"] == [[[0.1], [0.05]]] * 5
        assert len(actual["object"]["body_visual_rgb"]) == 5
        assert all(len(row) == 2 for row in actual["object"]["body_visual_rgb"])
        assert all(row == [0] for row in actual["object"]["drive_modes"])
        assert client.slots["ctrl"].shape == (5, 1)
        np.testing.assert_allclose(client.slots["qpos"], payload["initial_qpos"], atol=1e-6)
        np.testing.assert_allclose(client.slots["qvel"], payload["initial_qvel"], atol=1e-6)
        before = {key: client.slots[key].copy() for key in ("qpos", "qvel", "entity_root_state")}
        quaternion = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        poses = np.array([[0, 0, 1.4, *quaternion], [0, 0, 1.7, *quaternion]])
        velocities = np.array([[0.2, 0.3, 0.4, 0, 0, 1], [0.5, 0.6, 0.7, 0, 0, 2]])
        client.reset(
            SceneResetRequest(
                (4, 1), (EntityStatePatch("object", root_pose=poses, root_velocity=velocities),)
            )
        )
        np.testing.assert_allclose(
            client.slots["entity_root_state"][[4, 1], 1, 7:], velocities, atol=1e-6
        )
        for key in before:
            np.testing.assert_array_equal(client.slots[key][[0, 2, 3]], before[key][[0, 2, 3]])
        mirror = np.array([[2, 3, 4, 1, 0, 0, 0.0]])
        client.reset(SceneResetRequest((3,), (EntityStatePatch("target", root_pose=mirror),)))
        client.reset(
            SceneResetRequest(
                (2,),
                (
                    EntityStatePatch(
                        "object",
                        joint_positions=np.array([[0.4]]),
                        joint_velocities=np.array([[1.0]]),
                    ),
                ),
            )
        )
        client.slots["ctrl"][:] = 0.1
        client.request(protocol.CMD_STEP, {"nsteps": 1})
        assert client.slots["entity_root_state"][4, 1, 2] > 1.39
        assert client.slots["entity_root_state"][1, 1, 2] > 1.69
        np.testing.assert_allclose(
            client.slots["entity_root_state"][3, 3, :7], mirror[0], atol=1e-6
        )
        assert client.slots["qpos"][2, 8] > 0.4001
        displacement = client.slots["entity_root_state"][[4, 1], 1, :3] - poses[:, :3]
        np.testing.assert_allclose(displacement / payload["sim_dt"], velocities[:, :3], atol=0.02)
        result = {
            "meta": client.meta,
            "root_after_step": client.slots["entity_root_state"].tolist(),
            "qpos_after_step": client.slots["qpos"].tolist(),
            "com_velocity_oracle_passed": True,
            "multiple_reset_persistence_passed": True,
        }
        (tmp_path / "evidence.json").write_text(json.dumps(result, indent=2))
    finally:
        client.close()


def test_native_visual_mirror_does_not_change_falling_object_trajectory(tmp_path) -> None:
    trajectories = []
    for overlap in (False, True):
        folder = tmp_path / str(overlap)
        payload = scene_payload(
            folder / "assets", mirror_overlap=overlap, gravity=(0.0, 0.0, -9.81)
        )
        client = SceneClient(payload, folder / "worker.log")
        try:
            trajectory = []
            for _ in range(120):
                client.request(protocol.CMD_STEP, {"nsteps": 4})
                trajectory.append(client.slots["entity_root_state"][:, 1, :7].copy())
            trajectories.append(np.array(trajectory))
            assert np.all(client.slots["entity_root_state"][:, 1, 2] > 0.15)
            assert np.all(client.slots["entity_root_state"][:, 1, 2] < 0.4)
        finally:
            client.close()
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=1e-5)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "num_envs": 5,
                "assignment": [1, 1, 0, 1, 0],
                "steps": 480,
                "mirror_trajectory_max_delta": float(
                    np.max(np.abs(trajectories[0] - trajectories[1]))
                ),
            },
            indent=2,
        )
    )


def test_native_zero_joint_zero_action_scene_can_reset_and_step(tmp_path) -> None:
    payload = scene_payload(tmp_path / "assets")
    original = CompiledSceneLayout.from_dict(payload["scene_layout"])
    entity = replace(original.entities[3], body_ids=(0,))
    layout = CompiledSceneLayout((entity,), nq=0, nv=0, nu=0, nbody=1)
    spec = dict(payload["scene_entities"][3])
    spec["mirror_of"] = None
    payload.update(
        scene_layout=layout.to_dict(),
        scene_entities=[spec],
        initial_qpos=[[] for _ in range(5)],
        initial_qvel=[[] for _ in range(5)],
        initial_ctrl=[[] for _ in range(5)],
        initial_roots=np.asarray(payload["initial_roots"])[:, 3:4].tolist(),
    )
    client = SceneClient(payload, tmp_path / "worker.log")
    try:
        assert client.slots["ctrl"].shape == (5, 0)
        assert client.slots["qpos"].shape == client.slots["qvel"].shape == (5, 0)
        pose = np.array([[3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        client.reset(SceneResetRequest((4,), (EntityStatePatch("target", root_pose=pose),)))
        client.request(protocol.CMD_STEP, {"nsteps": 1})
        np.testing.assert_allclose(client.slots["entity_root_state"][4, 0, :7], pose[0], atol=1e-6)
    finally:
        client.close()
