"""Host mapping uses negotiated entity layouts without claiming native execution."""

from pathlib import Path

import numpy as np
import pytest

from tests.contract.test_worker_scene_materialization import scene
from unisim.backend.isaacgym.backend import IsaacGymBackend
from unisim.backend.subprocess_ipc import protocol
from unisim.entities import EntityStatePatch, SceneResetRequest


def backend(tmp_path: Path, monkeypatch) -> IsaacGymBackend:
    import unisim.backend.subprocess_ipc.backend as host

    # Construction is gated until the separately developed native worker lands.
    monkeypatch.setattr(host, "require_scene_composition_support", lambda *args: None)
    owner = IsaacGymBackend(scene(tmp_path), 5, 0.002)
    prepared = owner._entity_scene
    assert prepared is not None
    records = []
    for entry in prepared.payload["scene_entities"]:
        records.append(
            {
                "name": entry["name"],
                "assignment": entry["assignment"],
                "body_mass": [entry["variants"][i]["body_mass"] for i in entry["assignment"]],
            }
        )
    owner._bind_scene_metadata(
        {
            "scene_layout": prepared.layout.to_dict(),
            "scene_entities_actual": records,
            "gravity": [0, 0, -9.81],
        }
    )
    owner._allocate_slots()
    owner._slots["qpos"][:] = prepared.qpos
    owner._slots["qvel"][:] = prepared.qvel
    owner._slots["entity_root_state"][:] = prepared.roots
    return owner


def test_host_controls_and_complete_state_follow_distinct_public_layouts(tmp_path, monkeypatch):
    owner = backend(tmp_path, monkeypatch)
    try:
        assert owner.num_actuators == 1
        assert owner.get_actuator_names() == ("robot/drive",)
        assert owner.get_actuator_joint_names() == ("robot/hinge",)
        assert owner.get_state()["qpos"].shape == (5, 8)
        assert owner.get_state()["qvel"].shape == (5, 7)
        np.testing.assert_array_equal(owner.get_joint_state_qpos_indices(["robot/hinge"]), [0])
        before = owner.get_state()
        owner.get_entity_state("object")["root_pose"][:] = 100
        np.testing.assert_array_equal(owner.get_state()["qpos"], before["qpos"])
        model = owner.get_playback_model(0)
        import mujoco

        compiled = mujoco.MjModel.from_xml_path(model)
        assert compiled.body("robot/base").id > 0
        assert compiled.body("object/base").mass == 3
        assert compiled.body("target/base").id > 0
    finally:
        owner.close()


def test_host_prepares_one_transaction_and_bad_late_patch_never_uploads(tmp_path, monkeypatch):
    owner = backend(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(
        owner, "_request", lambda cmd, payload, **kw: commands.append((cmd, payload))
    )
    try:
        pose = np.array([[2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0]])
        owner.reset_entities(SceneResetRequest((3,), (EntityStatePatch("object", root_pose=pose),)))
        assert commands == [(protocol.CMD_RESET_ENTITIES, {"count": 1, "entity_names": ["object"]})]
        np.testing.assert_array_equal(owner._slots["reset_env_ids"][:1], [3])
        np.testing.assert_array_equal(owner._slots["reset_root_mask"][:, 0], [0, 1, 0])
        before = {name: values.copy() for name, values in owner._slots.items()}
        bad = SceneResetRequest(
            (1,),
            (
                EntityStatePatch("object", root_pose=pose),
                EntityStatePatch("robot", joint_positions=np.ones((1, 2))),
            ),
        )
        with pytest.raises(ValueError, match="columns"):
            owner.reset_entities(bad)
        assert len(commands) == 1
        for name, values in before.items():
            np.testing.assert_array_equal(owner._slots[name], values)
    finally:
        owner.close()


def test_host_rejects_native_mass_identity_mismatch(tmp_path, monkeypatch):
    owner = backend(tmp_path, monkeypatch)
    try:
        prepared = owner._entity_scene
        records = []
        for entry in prepared.payload["scene_entities"]:
            masses = [entry["variants"][i]["body_mass"] for i in entry["assignment"]]
            if entry["name"] == "object":
                masses = [[1.0] for _ in masses]
            records.append(
                {"name": entry["name"], "assignment": entry["assignment"], "body_mass": masses}
            )
        with pytest.raises(RuntimeError, match="native entity body masses"):
            owner._bind_scene_metadata(
                {
                    "scene_layout": prepared.layout.to_dict(),
                    "scene_entities_actual": records,
                    "gravity": [0, 0, -9.81],
                }
            )
    finally:
        owner.close()
